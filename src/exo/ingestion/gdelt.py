"""GDELT ingestor — runs every 15 minutes via APScheduler.

Reads GDELT 2.0's raw 15-minute event exports (data.gdeltproject.org) rather
than the DOC API. The DOC API throttles at one request per 5 seconds and, as
of 2026-10, refuses every request from this deployment's IP even after a month
of silence, while the export files are static downloads with no rate limit.
One export file also carries everything both signals need, so a single feed
replaces what used to be two DOC API queries per country.

Emits per country, every cycle:

- ``news_sentiment`` — article-weighted mean event tone over the trailing 24h,
  mapped to [0, 1] with negative tone → high risk (same mapping as the DOC API
  era, so the risk index's calibration carries over).
- ``news_magnitude`` — trailing-24h article volume relative to the country's
  own 6-day baseline, mapped to [0, 1] with 0.5 meaning "normal coverage". A
  fixed article cap can't do this: the US out-covers Kenya by two orders of
  magnitude every day, so a cap that suits one country pins the other.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

_BASE_URL = "https://data.gdeltproject.org/gdeltv2"
_LASTUPDATE_URL = f"{_BASE_URL}/lastupdate.txt"

_SLOT = timedelta(minutes=15)
_WINDOW = timedelta(hours=24)
# The magnitude baseline is the 6 days before the current 24h window.
_HISTORY = timedelta(days=7)

# Export files are ~70 KB; a cold start pulls a week of them (672 files) once.
_DOWNLOAD_CONCURRENCY = 8

# Minimum fraction of a window's slots that must have been read before it is
# trusted. GDELT does occasionally skip a slot, and volumes are scaled up for
# the missing share, but a window mostly made of gaps would turn a download
# outage into a fake coverage collapse.
_MIN_COVERAGE = 0.5

# Below this many articles in the 24h window, tone is too noisy to report.
_MIN_ARTICLES_FOR_TONE = 20

# GDELT 2.0 event export columns (tab-separated, no header).
_COL_GOLDSTEIN = 30
_COL_NUM_ARTICLES = 33
_COL_AVG_TONE = 34
_COL_ACTION_COUNTRY = 53
_NUM_COLUMNS = 61

# ISO 3166 alpha-2 (what the rest of exo keys countries by) → the FIPS 10-4
# codes GDELT geocodes events with. They disagree for most countries (Russia is
# RS, China CH, Nigeria NI, Chile CI), so the export cannot be read by ISO
# code directly. Israel includes Gaza and the West Bank: the DOC API era
# searched the keyword "Israel", which caught that coverage, and the conflict
# dimension this feeds would otherwise miss most of Israel's conflict news.
COUNTRY_FIPS_MAP: dict[str, tuple[str, ...]] = {
    "US": ("US",), "RU": ("RS",), "CN": ("CH",), "UA": ("UP",),
    "IL": ("IS", "GZ", "WE"), "IR": ("IR",), "KP": ("KN",), "TW": ("TW",),
    "IN": ("IN",), "PK": ("PK",), "HT": ("HA",), "BR": ("BR",),
    "MX": ("MX",), "NG": ("NI",), "KE": ("KE",), "ZA": ("SF",),
    "FR": ("FR",), "GB": ("UK",), "MY": ("MY",), "CL": ("CI",),
    "KR": ("KS",), "JP": ("JA",), "DE": ("GM",), "NL": ("NL",),
    "AE": ("AE",), "SG": ("SN",), "VN": ("VM",), "ID": ("ID",),
    "CD": ("CG",), "GN": ("GV",),
}


@dataclass
class _Tally:
    """Per-country aggregate of one 15-minute export file."""

    articles: int = 0
    events: int = 0
    tone_x_articles: float = 0.0
    goldstein_x_articles: float = 0.0

    def add(self, other: "_Tally") -> None:
        self.articles += other.articles
        self.events += other.events
        self.tone_x_articles += other.tone_x_articles
        self.goldstein_x_articles += other.goldstein_x_articles


def _slot_url(slot: datetime) -> str:
    return f"{_BASE_URL}/{slot:%Y%m%d%H%M%S}.export.CSV.zip"


def _parse_export(blob: bytes, fips_to_iso: dict[str, str]) -> dict[str, _Tally]:
    """Aggregate one zipped export file into per-ISO-country tallies."""
    tallies: dict[str, _Tally] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            for row in csv.reader(text, delimiter="\t"):
                if len(row) < _NUM_COLUMNS:
                    continue
                iso = fips_to_iso.get(row[_COL_ACTION_COUNTRY])
                if iso is None:
                    continue
                try:
                    articles = int(row[_COL_NUM_ARTICLES] or 0)
                    tone = float(row[_COL_AVG_TONE] or 0.0)
                    goldstein = float(row[_COL_GOLDSTEIN] or 0.0)
                except ValueError:
                    continue
                t = tallies.setdefault(iso, _Tally())
                t.articles += articles
                t.events += 1
                t.tone_x_articles += tone * articles
                t.goldstein_x_articles += goldstein * articles
    return tallies


def _tone_to_risk(avg_tone: float) -> float:
    """Map GDELT tone (roughly -10..+10 in practice) to a [0, 1] risk score."""
    tone_score = max(-1.0, min(1.0, avg_tone / 10.0))
    return (1.0 - tone_score) / 2.0


def _ratio_to_score(ratio: float) -> float:
    """Map volume/baseline to [0, 1]: 1× → 0.5, 3× → 0.75, 9× → 0.9."""
    return ratio / (1.0 + ratio)


class GDELTIngestor(BaseIngestor):
    """Ingest GDELT 15-minute event exports into news_sentiment/news_magnitude."""

    source = "gdelt"

    def __init__(self, countries: dict[str, tuple[str, ...]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.countries = countries or COUNTRY_FIPS_MAP
        self._fips_to_iso = {
            fips: iso for iso, codes in self.countries.items() for fips in codes
        }
        # slot start → that file's tallies, or None for a slot GDELT never
        # published (404). Remembering 404s keeps a known gap from being
        # re-requested every cycle for the week it stays in the window.
        self._slots: dict[datetime, dict[str, _Tally] | None] = {}

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    async def _latest_slot(self, client: httpx.AsyncClient) -> datetime:
        resp = await client.get(_LASTUPDATE_URL)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            url = line.split()[-1] if line.strip() else ""
            if url.endswith(".export.CSV.zip"):
                stamp = url.rsplit("/", 1)[-1].split(".", 1)[0]
                return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        raise ValueError(f"no export file listed in lastupdate.txt: {resp.text[:200]!r}")

    async def _load_slot(
        self, client: httpx.AsyncClient, sem: asyncio.Semaphore, slot: datetime
    ) -> None:
        async with sem:
            try:
                resp = await client.get(_slot_url(slot))
            except httpx.HTTPError as exc:
                # Left unrecorded so the next cycle retries it.
                logger.warning("GDELT export %s failed: %s", f"{slot:%Y%m%d%H%M%S}", exc)
                return
        if resp.status_code == 404:
            self._slots[slot] = None
            return
        if resp.status_code != 200:
            logger.warning(
                "GDELT export %s returned HTTP %d", f"{slot:%Y%m%d%H%M%S}", resp.status_code
            )
            return
        try:
            self._slots[slot] = await asyncio.to_thread(
                _parse_export, resp.content, self._fips_to_iso
            )
        except (zipfile.BadZipFile, IndexError) as exc:
            logger.warning("GDELT export %s unreadable: %s", f"{slot:%Y%m%d%H%M%S}", exc)

    async def _sync_slots(self, latest: datetime) -> None:
        """Download every slot in the history window not yet held, drop older."""
        oldest = latest - _HISTORY + _SLOT
        self._slots = {s: v for s, v in self._slots.items() if s >= oldest}

        wanted = []
        slot = latest
        while slot >= oldest:
            if slot not in self._slots:
                wanted.append(slot)
            slot -= _SLOT
        if not wanted:
            return

        if len(wanted) > 4:
            logger.info("GDELT backfilling %d export files", len(wanted))
        sem = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            await asyncio.gather(*(self._load_slot(client, sem, s) for s in wanted))

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def _window(self, start: datetime, end: datetime) -> tuple[dict[str, _Tally], int, int]:
        """Sum tallies over slots in [start, end]; return (totals, read, expected).

        A 404 slot counts as read: GDELT published nothing for it, so it is a
        true zero rather than missing data.
        """
        totals: dict[str, _Tally] = {}
        expected = read = 0
        slot = end
        while slot >= start:
            expected += 1
            if slot in self._slots:
                read += 1
                for iso, t in (self._slots[slot] or {}).items():
                    totals.setdefault(iso, _Tally()).add(t)
            slot -= _SLOT
        return totals, read, expected

    async def fetch(self) -> list[RawRecord]:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            latest = await self._latest_slot(client)
        await self._sync_slots(latest)

        win_start = latest - _WINDOW + _SLOT
        current, cur_read, cur_expected = self._window(win_start, latest)
        baseline, base_read, base_expected = self._window(
            latest - _HISTORY + _SLOT, win_start - _SLOT
        )

        cur_cov = cur_read / cur_expected
        if cur_cov < _MIN_COVERAGE:
            logger.error(
                "GDELT 24h window only %d/%d slots readable; skipping this cycle",
                cur_read, cur_expected,
            )
            return []
        base_cov = base_read / base_expected if base_expected else 0.0

        now = self.utcnow()
        raws: list[RawRecord] = []
        for iso in self.countries:
            cur = current.get(iso, _Tally())
            base = baseline.get(iso, _Tally())
            # Scale each window up for its unread share so a few missed slots
            # read as missing data, not as a drop in coverage.
            articles_24h = cur.articles / cur_cov
            baseline_daily = (
                base.articles / base_cov / (base_expected * _SLOT / timedelta(days=1))
                if base_cov >= _MIN_COVERAGE else None
            )
            raws.append(
                RawRecord(
                    source=self.source,
                    entity=iso,
                    raw={
                        "articles_24h": articles_24h,
                        "events_24h": cur.events,
                        "avg_tone": cur.tone_x_articles / cur.articles if cur.articles else None,
                        "avg_goldstein": cur.goldstein_x_articles / cur.articles if cur.articles else None,
                        "baseline_daily_articles": baseline_daily,
                        "window_end": latest.isoformat(),
                        "coverage_24h": round(cur_cov, 3),
                        "coverage_baseline": round(base_cov, 3),
                    },
                    fetched_at=now,
                )
            )
        return raws

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        now = raw.fetched_at
        d = raw.raw
        common = {
            "window_end": d.get("window_end"),
            "articles_24h": round(d["articles_24h"], 1),
            "events_24h": d.get("events_24h"),
            "coverage_24h": d.get("coverage_24h"),
        }
        records: list[FeatureRecord] = []

        avg_tone = d.get("avg_tone")
        if avg_tone is not None and d["articles_24h"] >= _MIN_ARTICLES_FOR_TONE:
            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=raw.entity,
                    signal_type="news_sentiment",
                    value=_tone_to_risk(avg_tone),
                    metadata={
                        **common,
                        "raw_tone": round(avg_tone, 4),
                        "avg_goldstein": round(d["avg_goldstein"], 4),
                    },
                    as_of_ts=now,
                )
            )

        baseline = d.get("baseline_daily_articles")
        if baseline:
            ratio = d["articles_24h"] / baseline
            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=raw.entity,
                    signal_type="news_magnitude",
                    value=_ratio_to_score(ratio),
                    metadata={
                        **common,
                        "baseline_daily_articles": round(baseline, 1),
                        "ratio": round(ratio, 3),
                        "coverage_baseline": d.get("coverage_baseline"),
                    },
                    as_of_ts=now,
                )
            )

        return records
