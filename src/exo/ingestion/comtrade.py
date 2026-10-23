"""UN Comtrade AI supply-chain ingestor — monthly.

Reads the keyless Comtrade preview API and emits three structural signals per
tracked country for the AI exposure overlay (all in [0, 1], higher = more
exposed):

  hw_contested_share     — share of computer, telecom and chip imports
                           (HS 8471, 8517, 8542) from China, Hong Kong and
                           Taiwan, shrunk toward the tracked-country mean for
                           small baskets (weight = value / (value + $100m)).
                           A country with no report in the last 3 years
                           (Russia stopped after 2021) is rebuilt from major
                           exporters' exports to it (mirror data).

  cn_mineral_dependence  — dependence on Chinese supply of the materials on
                           China's export-control lists, at the fixed
                           pre-controls baseline year AI_MINERAL_BASELINE_YEAR.
                           Per material group, the share from China + Hong Kong
                           is shrunk toward the world share ($5m prior), then
                           groups are averaged weighted by China's world share
                           of each (near-monopolies count most). No record
                           below $10m of basket imports.

  supplier_chokepoint    — the country's largest share, as an origin, of the
                           12 large importers' imports of a critical mineral,
                           lithium, refined copper, nickel or chips (8542).
                           A 50% share scores 1.0.

Values are stored with their raw shares in metadata so the scorer can be
re-weighted without re-ingesting. About 75 calls a run, paced to one per 2.5 s.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

BASE_URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"

# ISO 3166-1 alpha-2 → Comtrade reporter code (M49, with Comtrade's own codes
# for France 251 and India 699). Taiwan reports as "Other Asia, nes" (490).
ISO2_TO_COMTRADE: dict[str, int] = {
    "US": 842, "RU": 643, "CN": 156, "UA": 804, "IL": 376,
    "IR": 364, "IN": 699, "PK": 586, "KP": 408,
    "HT": 332, "BR": 76, "MX": 484, "NG": 566, "KE": 404,
    "ZA": 710, "FR": 251, "GB": 826, "MY": 458, "CL": 152,
    "KR": 410, "JP": 392, "DE": 276, "NL": 528, "AE": 784,
    "SG": 702, "VN": 704, "ID": 360, "CD": 180, "GN": 324,
    "TW": 490,
}
# As a partner Taiwan is mostly 490, occasionally 158.
TAIWAN_PARTNER_CODES = (490, 158)
PARTNER_TO_ISO2: dict[int, str] = {
    **{code: iso2 for iso2, code in ISO2_TO_COMTRADE.items()},
    840: "US", 250: "FR", 158: "TW",
}
TRACKED = list(ISO2_TO_COMTRADE)

HW_CODES = ("8471", "8517", "8542")
CONTESTED_ORIGINS = {156, 344, *TAIWAN_PARTNER_CODES}   # China, Hong Kong, Taiwan
HW_PRIOR_USD = 1e8

# Major chip and electronics exporters whose exports rebuild the hardware basket
# of countries that don't report it.
MIRROR_EXPORTERS = (156, 276, 344, 392, 410, 458, 490, 528, 608, 702, 704, 764, 842)

# Large importers whose imports by origin define supplier concentration.
CHOKEPOINT_IMPORTERS = (56, 124, 156, 246, 276, 392, 410, 458, 528, 578, 699, 842)
# Critical minerals, lithium, refined copper and nickel. HS 2805 mixes alkali
# metals with rare earths (280530 is used instead); 8103 (tantalum metal) and
# 282200 (cobalt oxides) are processed goods that make processors look like
# suppliers, so they are left out.
CHOKEPOINT_CODES = (
    "2504", "2602", "2603", "2605", "2606", "2611", "2615", "2822", "2846",
    "7110", "8105", "8112", "280530",
    "253090", "282520", "283691",    # lithium
    "740311",                        # refined copper cathodes
    "2604", "7501", "7502",          # nickel
)
CHOKEPOINT_MIN_TOTAL_USD = 1e8
CHOKEPOINT_SATURATION = 0.5

# Materials on China's export-control lists, by group.
CN_MINERAL_GROUPS: dict[str, tuple[str, ...]] = {
    "Ga/Ge": ("811292", "282560"),
    "rare earths": ("280530", "2846"),
    "magnets": ("850511",),
    "graphite": ("250410", "380110"),
    "antimony": ("8110",),
    "tungsten": ("8101",),
    "bismuth": ("8106",),
    "molybdenum": ("8102",),
    "polysilicon": ("280461",),
}
CN_SUPPLIERS = (156, 344)
CN_MINERAL_PRIOR_USD = 5e6
CN_MINERAL_FLOOR_USD = 1e7

_PACE_SECONDS = 2.5
_ROW_CAP = 500   # the preview API truncates silently at 500 rows


def _rows(body: dict, totals: bool = False) -> list[dict]:
    """Plain customs-total rows: partner rows, or the World (0) total rows."""
    return [
        x for x in body.get("data") or []
        if x.get("customsCode") == "C00" and x.get("motCode") == 0
        and x.get("partner2Code", 0) == 0
        and (x["partnerCode"] == 0) == totals
    ]


class ComtradeIngestor(BaseIngestor):
    """AI hardware, mineral and supplier-concentration signals from UN Comtrade."""

    source = "comtrade"

    def __init__(self, *args, pace: float = _PACE_SECONDS, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pace = pace

    async def _get(self, client: httpx.AsyncClient, **params) -> dict:
        """One paced preview-API call; retries the API's rate-limit replies."""
        query = {k: ",".join(map(str, v)) if isinstance(v, (list, tuple)) else v
                 for k, v in params.items()}
        for attempt in range(4):
            await asyncio.sleep(self._pace if attempt == 0 else 8.0)
            try:
                resp = await client.get(BASE_URL, params=query)
                if resp.status_code == 429:
                    continue
                resp.raise_for_status()
                body = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.debug("Comtrade call failed (%s): %s", query, exc)
                continue
            if "data" in body:
                return body
        logger.warning("Comtrade call gave up after 4 attempts: %s", query)
        return {}

    # ------------------------------------------------------------------
    # Hardware: share of HS 8471/8517/8542 imports from China, HK, Taiwan
    # ------------------------------------------------------------------

    async def _hardware(self, client, latest_year: int):
        """{iso2: (contested_usd, total_usd, year, basis)} plus 8542 imports by origin."""
        out: dict[str, tuple[float, float, int, str]] = {}
        chip_origin: dict[int, float] = defaultdict(float)
        for iso2, code in ISO2_TO_COMTRADE.items():
            for year in range(latest_year, latest_year - 3, -1):
                rows = _rows(await self._get(
                    client, reporterCode=code, period=year, cmdCode=HW_CODES, flowCode="M",
                ))
                if rows:
                    total = sum(x["primaryValue"] for x in rows)
                    contested = sum(x["primaryValue"] for x in rows
                                    if x["partnerCode"] in CONTESTED_ORIGINS)
                    out[iso2] = (contested, total, year, "own")
                    for x in rows:
                        if x["cmdCode"] == "8542":
                            chip_origin[x["partnerCode"]] += x["primaryValue"]
                    break

        # Countries with no recent report: rebuild from exporters' side.
        missing = [iso2 for iso2 in TRACKED if iso2 not in out]
        if missing:
            partner_codes = [c for iso2 in missing
                             for c in (TAIWAN_PARTNER_CODES if iso2 == "TW" else (ISO2_TO_COMTRADE[iso2],))]
            year = latest_year - 1
            mirror: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
            for exporter in MIRROR_EXPORTERS:
                body = await self._get(client, reporterCode=exporter, period=year,
                                       cmdCode=HW_CODES, flowCode="X", partnerCode=partner_codes)
                for x in _rows(body):
                    iso2 = PARTNER_TO_ISO2.get(x["partnerCode"])
                    if iso2 in missing:
                        mirror[iso2][exporter] += x["primaryValue"]
            for iso2, by_exporter in mirror.items():
                total = sum(by_exporter.values())
                if total > 0:
                    contested = sum(v for e, v in by_exporter.items() if e in CONTESTED_ORIGINS)
                    out[iso2] = (contested, total, year, "mirror")
        return out, chip_origin

    # ------------------------------------------------------------------
    # Supplier chokepoint: largest share of large importers' imports
    # ------------------------------------------------------------------

    async def _importer_rows(self, client, importers: tuple[int, ...], year: int, cmd: str) -> list[dict]:
        """Import rows for several reporters, halving the reporter list until
        each call fits under the preview API's silent 500-row cap."""
        body = await self._get(client, reporterCode=importers, period=year, cmdCode=cmd, flowCode="M")
        if body.get("count", 0) < _ROW_CAP:
            return _rows(body)
        if len(importers) == 1:
            logger.warning("Comtrade: %s for reporter %s exceeds %d rows; truncated",
                           cmd, importers[0], _ROW_CAP)
            return _rows(body)
        half = len(importers) // 2
        return (await self._importer_rows(client, importers[:half], year, cmd)
                + await self._importer_rows(client, importers[half:], year, cmd))

    async def _chokepoint(self, client, year: int, chip_origin: dict[int, float]):
        """{iso2: (share, hs_code)} — each country's largest share as an origin."""
        by_code: dict[str, dict[str, float]] = {}
        for cmd in CHOKEPOINT_CODES:
            origins: dict[str, float] = defaultdict(float)
            for x in await self._importer_rows(client, CHOKEPOINT_IMPORTERS, year, cmd):
                origins[PARTNER_TO_ISO2.get(x["partnerCode"], str(x["partnerCode"]))] += x["primaryValue"]
            by_code[cmd] = origins
        by_code["8542"] = defaultdict(float)
        for code, v in chip_origin.items():
            by_code["8542"][PARTNER_TO_ISO2.get(code, str(code))] += v

        best: dict[str, tuple[float, str]] = {}
        for cmd, origins in by_code.items():
            total = sum(origins.values())
            if total < CHOKEPOINT_MIN_TOTAL_USD:
                continue
            for iso2 in TRACKED:
                share = origins.get(iso2, 0.0) / total
                if share > best.get(iso2, (0.0, ""))[0]:
                    best[iso2] = (share, cmd)
        return best

    # ------------------------------------------------------------------
    # Dependence on Chinese export-controlled materials (fixed baseline)
    # ------------------------------------------------------------------

    async def _cn_minerals(self, client, year: int):
        """{iso2: {group: (cn_usd, total_usd)}} and each group's China world share."""
        code_group = {c: g for g, codes in CN_MINERAL_GROUPS.items() for c in codes}
        total: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        from_cn: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        world_total: dict[str, float] = defaultdict(float)
        world_cn: dict[str, float] = defaultdict(float)
        for cmd, group in code_group.items():
            body = await self._get(client, period=year, cmdCode=cmd, flowCode="M",
                                   partnerCode=(0, *CN_SUPPLIERS))
            for x in _rows(body, totals=True) + _rows(body):
                if x["reporterCode"] in CN_SUPPLIERS:
                    continue
                if x["partnerCode"] == 0:
                    world_total[group] += x["primaryValue"]
                else:
                    world_cn[group] += x["primaryValue"]
                iso2 = PARTNER_TO_ISO2.get(x["reporterCode"])
                if iso2:
                    bucket = total if x["partnerCode"] == 0 else from_cn
                    bucket[iso2][group] += x["primaryValue"]

        # Non-reporters (Taiwan, Russia after 2021, North Korea): exporters' side.
        missing = [iso2 for iso2 in TRACKED if not sum(total[iso2].values())]
        partner_codes = [c for iso2 in missing
                         for c in (TAIWAN_PARTNER_CODES if iso2 == "TW" else (ISO2_TO_COMTRADE[iso2],))]
        if partner_codes:
            for cmd, group in code_group.items():
                body = await self._get(client, period=year, cmdCode=cmd, flowCode="X",
                                       partnerCode=partner_codes)
                for x in _rows(body):
                    iso2 = PARTNER_TO_ISO2.get(x["partnerCode"])
                    if iso2 in missing:
                        total[iso2][group] += x["primaryValue"]
                        if x["reporterCode"] in CN_SUPPLIERS:
                            from_cn[iso2][group] += x["primaryValue"]

        cn_world_share = {g: (world_cn[g] / world_total[g]) if world_total[g] else 0.0
                          for g in CN_MINERAL_GROUPS}
        groups = {iso2: {g: (from_cn[iso2][g], total[iso2][g]) for g in CN_MINERAL_GROUPS
                         if total[iso2][g] > 0}
                  for iso2 in TRACKED}
        return groups, cn_world_share

    async def fetch(self) -> list[RawRecord]:
        now = self.utcnow()
        async with httpx.AsyncClient(timeout=90.0) as client:
            hardware, chip_origin = await self._hardware(client, now.year - 1)
            chokepoint = await self._chokepoint(client, now.year - 2, chip_origin)
            minerals, cn_world_share = await self._cn_minerals(
                client, config.AI_MINERAL_BASELINE_YEAR,
            )

        if not hardware and not chokepoint and not any(minerals.values()):
            logger.warning("Comtrade returned no usable data this cycle")
            return []

        own_shares = [c / t for c, t, _, basis in hardware.values() if t > 0 and basis == "own"]
        hw_mean = sum(own_shares) / len(own_shares) if own_shares else 0.0
        logger.info("Comtrade: hardware for %d, chokepoint origins %d, minerals %d of %d countries",
                    len(hardware), len(chokepoint), sum(1 for g in minerals.values() if g), len(TRACKED))

        return [
            RawRecord(
                source=self.source,
                entity=iso2,
                raw={
                    "hardware": hardware.get(iso2),
                    "hw_mean_share": hw_mean,
                    # None when the chokepoint calls all failed, so a fetch
                    # failure isn't written as "supplies nothing".
                    "chokepoint": chokepoint.get(iso2, (0.0, "")) if chokepoint else None,
                    "chokepoint_year": now.year - 2,
                    "minerals": minerals.get(iso2, {}),
                    "cn_world_share": cn_world_share,
                    "mineral_year": config.AI_MINERAL_BASELINE_YEAR,
                },
                fetched_at=now,
            )
            for iso2 in TRACKED
        ]

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        d = raw.raw
        records: list[FeatureRecord] = []

        def record(signal_type: str, value: float, metadata: dict) -> None:
            records.append(FeatureRecord(
                source=self.source, entity=raw.entity, signal_type=signal_type,
                value=round(max(0.0, min(1.0, value)), 4), metadata=metadata,
                as_of_ts=raw.fetched_at,
            ))

        if d.get("hardware"):
            contested, total, year, basis = d["hardware"]
            share = contested / total if total else 0.0
            shrunk = (share * total + d["hw_mean_share"] * HW_PRIOR_USD) / (total + HW_PRIOR_USD)
            record("hw_contested_share", shrunk, {
                "raw_share": round(share, 4), "imports_usd": total,
                "year": year, "basis": basis,
            })

        if d.get("chokepoint") is not None:
            share, cmd = d["chokepoint"]
            record("supplier_chokepoint", share / CHOKEPOINT_SATURATION, {
                "top_share": round(share, 4), "top_hs_code": cmd, "year": d.get("chokepoint_year"),
            })

        groups = d.get("minerals") or {}
        basket = sum(t for _, t in groups.values())
        if raw.entity == "CN":
            # China's dependence on itself is not a supply risk.
            record("cn_mineral_dependence", 0.0, {"baseline_year": d.get("mineral_year"), "self": True})
        elif basket >= CN_MINERAL_FLOOR_USD:
            weights = d.get("cn_world_share") or {}
            num = den = 0.0
            for g, (cn_usd, total) in groups.items():
                w = weights.get(g, 0.0)
                shrunk = (cn_usd + w * CN_MINERAL_PRIOR_USD) / (total + CN_MINERAL_PRIOR_USD)
                num += w * shrunk
                den += w
            if den:
                record("cn_mineral_dependence", num / den, {
                    "baseline_year": d.get("mineral_year"), "basket_usd": basket,
                    "groups": {g: round(c / t, 4) for g, (c, t) in groups.items()},
                })
        return records
