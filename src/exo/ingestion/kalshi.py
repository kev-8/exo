"""Kalshi ingestor — RSA-PSS auth, REST metadata.

REST polls market metadata every 15 minutes, scoped to a category-filtered
set of series (see _refresh_focused_series / _refresh_active_series below).

Each poll emits per-snapshot observations (market_price / market_volume /
market_spread) and, only when they are new or have changed, a `contract_terms`
record carrying the contract's static settlement criteria — see
_terms_fingerprint for why those are a separate signal rather than extra
metadata on every price row.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.models import FeatureRecord, KalshiMarket, RawRecord

logger = logging.getLogger(__name__)

# How often to re-pull the full series catalog (rarely changes — new series
# are added occasionally, not continuously).
_SERIES_REFRESH_INTERVAL = timedelta(hours=24)

# How often to re-probe *every* focused series for open markets, to catch
# series that just became active. Between refreshes, only the already-active
# subset is polled (cheap — most focused series have nothing open at any
# given moment).
_ACTIVE_REFRESH_INTERVAL = timedelta(hours=1)

# Max concurrent /markets requests during a series probe. Kept modest to
# stay well under Kalshi's rate limit even when probing all ~6k series.
_PROBE_CONCURRENCY = 20

# Fields of the /markets payload that describe how a contract settles, as
# opposed to how it is currently priced. Captured because misreading the
# resolution criteria is the dominant source of fake edge: `title` alone is
# systematically ambiguous ("Fed cuts rates" says nothing about which meeting,
# which measure, or the rounding convention), and a forecaster reasoning from
# the title instead of the rules will manufacture divergence from the market
# that reflects its own misunderstanding rather than genuine mispricing.
_TERMS_FIELDS = (
    "rules_primary",
    "rules_secondary",
    "can_close_early",
    "strike_type",
    "floor_strike",
    "cap_strike",
    "yes_sub_title",
    "no_sub_title",
    "subtitle",
    "market_type",
    "settlement_timer_seconds",
)

_CRYPTO_AVAILABLE = False
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    _CRYPTO_AVAILABLE = True
except ImportError:
    logger.warning("cryptography not installed; Kalshi auth disabled")


def _load_private_key():
    """Load RSA private key from KALSHI_PRIVATE_KEY environment variable (PEM content)."""
    if not _CRYPTO_AVAILABLE:
        return None
    pem = config.KALSHI_PRIVATE_KEY
    if not pem:
        logger.warning("KALSHI_PRIVATE_KEY env var not set; Kalshi auth disabled")
        return None
    return serialization.load_pem_private_key(pem.encode(), password=None)


def _build_auth_headers(method: str, path: str) -> dict[str, str]:
    """Build RSA-PSS signed auth headers for Kalshi API."""
    if not _CRYPTO_AVAILABLE:
        return {}
    private_key = _load_private_key()
    if private_key is None:
        return {}

    ts_ms = str(int(time.time() * 1000))
    msg = ts_ms + method.upper() + path
    signature = private_key.sign(
        msg.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(signature).decode()
    return {
        "KALSHI-ACCESS-KEY": config.KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
    }


class KalshiIngestor(BaseIngestor):
    """Ingest Kalshi market data via REST."""

    source = "kalshi"

    def __init__(
        self,
        tickers: list[str] | None = None,
        categories: set[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tickers = tickers or []
        self.categories = categories or config.FOCUSED_CATEGORIES

        # series_ticker -> category, for every series in self.categories.
        self._series_category: dict[str, str] = {}
        self._series_refreshed_at: datetime | None = None

        # Subset of _series_category currently believed to have open markets.
        self._active_series: set[str] = set()
        self._active_refreshed_at: datetime | None = None

        # ticker -> fingerprint of the settlement terms last written for it.
        # Seeded lazily from the store so a restart doesn't re-emit terms for
        # every contract already on disk. None means "not loaded yet".
        self._terms_seen: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Series discovery
    # ------------------------------------------------------------------

    async def _refresh_focused_series(self, client: httpx.AsyncClient) -> None:
        """Rebuild the series -> category allowlist from /series.

        A single unpaginated call returns the entire series catalog (~14k
        entries); filtering client-side to self.categories is far cheaper
        than discovering relevant series by paging through /markets or
        /events, both of which are dominated by high-frequency synthetic
        markets unrelated to these categories.
        """
        path = "/series"
        headers = _build_auth_headers("GET", "/trade-api/v2" + path)
        try:
            resp = await client.get(config.KALSHI_BASE_URL + path, headers=headers)
            resp.raise_for_status()
            all_series = resp.json().get("series", [])
        except Exception as exc:
            logger.warning("Kalshi /series refresh failed: %s", exc)
            return

        self._series_category = {
            s["ticker"]: s["category"]
            for s in all_series
            if s.get("ticker") and s.get("category") in self.categories
        }
        self._series_refreshed_at = self.utcnow()
        logger.info(
            "Kalshi focused-series allowlist refreshed: %d series across %d categories",
            len(self._series_category), len(self.categories),
        )

    async def _probe_series(
        self, client: httpx.AsyncClient, series_ticker: str, sem: asyncio.Semaphore
    ) -> list[dict]:
        """Return currently open markets for one series (empty if none/error)."""
        async with sem:
            path = "/markets"
            headers = _build_auth_headers("GET", "/trade-api/v2" + path)
            try:
                resp = await client.get(
                    config.KALSHI_BASE_URL + path,
                    headers=headers,
                    params={"series_ticker": series_ticker, "status": "open", "limit": 200},
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.debug("Kalshi rate limited on series=%s; sleeping %ds", series_ticker, retry_after)
                    await asyncio.sleep(retry_after)
                    return []
                resp.raise_for_status()
                return resp.json().get("markets", [])
            except Exception as exc:
                logger.debug("Kalshi series probe failed for %s: %s", series_ticker, exc)
                return []

    async def _refresh_active_series(self, client: httpx.AsyncClient) -> dict[str, list[dict]]:
        """Probe every focused series for open markets.

        This call itself carries live pricing, so its result is used
        directly as this cycle's market data rather than being fetched
        again — it's the expensive (~len(series_category) requests) sweep,
        run at most every _ACTIVE_REFRESH_INTERVAL.
        """
        sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
        tickers = list(self._series_category.keys())
        results = await asyncio.gather(*(self._probe_series(client, t, sem) for t in tickers))
        by_series = {t: markets for t, markets in zip(tickers, results) if markets}
        self._active_series = set(by_series.keys())
        self._active_refreshed_at = self.utcnow()
        logger.info(
            "Kalshi active-series refresh: %d/%d focused series have open markets",
            len(self._active_series), len(tickers),
        )
        return by_series

    # ------------------------------------------------------------------
    # Settlement terms
    # ------------------------------------------------------------------

    SOURCE_TERMS_SIGNAL = "contract_terms"

    @staticmethod
    def _terms(m: dict) -> dict[str, Any]:
        """Settlement-criteria fields present in *m*, absent keys omitted."""
        return {k: m[k] for k in _TERMS_FIELDS if m.get(k) not in (None, "")}

    @staticmethod
    def _terms_fingerprint(terms: dict[str, Any]) -> str:
        """Stable short hash of *terms*.

        Terms are a property of the contract, not an observation of it: they
        are identical across all 96 daily snapshots of a ticker. Writing them
        onto every market_price row would add roughly 210 MB/day of byte-
        identical text to a store that currently grows 458 MB/day — so they go
        out as their own signal, emitted once per contract and again only when
        this fingerprint changes (Kalshi does occasionally amend rules, and a
        forecast made under the old wording must stay attributable to it).
        """
        blob = json.dumps(terms, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _ensure_terms_seen(self) -> dict[str, str]:
        """Return the ticker -> fingerprint map, loading it from the store once.

        Propagates a read failure instead of degrading to an empty map: empty
        means "emit terms for everything", which on a populated store would
        re-write tens of thousands of unchanged rows.
        """
        if self._terms_seen is None:
            self._terms_seen = self.store.latest_metadata_by_entity(
                self.source, self.SOURCE_TERMS_SIGNAL, "terms_hash"
            )
            logger.info(
                "Kalshi contract-terms cache seeded with %d tickers",
                len(self._terms_seen),
            )
        return self._terms_seen

    # ------------------------------------------------------------------
    # REST fetch
    # ------------------------------------------------------------------

    async def fetch(self) -> list[RawRecord]:
        raws: list[RawRecord] = []
        now = self.utcnow()

        async with httpx.AsyncClient(timeout=30.0) as client:
            if self._series_refreshed_at is None or now - self._series_refreshed_at > _SERIES_REFRESH_INTERVAL:
                await self._refresh_focused_series(client)

            if not self._series_category:
                logger.warning("Kalshi focused-series allowlist is empty; skipping poll cycle")
                return raws

            if self._active_refreshed_at is None or now - self._active_refreshed_at > _ACTIVE_REFRESH_INTERVAL:
                by_series = await self._refresh_active_series(client)
            else:
                sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
                tickers = list(self._active_series)
                results = await asyncio.gather(*(self._probe_series(client, t, sem) for t in tickers))
                by_series = dict(zip(tickers, results))
                # Markets close between polls — drop series with nothing open
                # this cycle so the next fast poll doesn't keep probing them.
                self._active_series = {t for t, markets in by_series.items() if markets}

            terms_seen = self._ensure_terms_seen()
            for series_ticker, markets in by_series.items():
                category = self._series_category.get(series_ticker, "")
                for m in markets:
                    # Decided here rather than in normalise() so the terms row
                    # is emitted at most once per cycle per ticker even if the
                    # same market appears under two series.
                    terms = self._terms(m)
                    fp = self._terms_fingerprint(terms) if terms else ""
                    emit_terms = bool(terms) and terms_seen.get(m.get("ticker", "")) != fp

                    raw = {
                        **m,
                        "category": category,
                        "series_ticker": series_ticker,
                        "_emit_terms": emit_terms,
                        "_terms_hash": fp,
                    }
                    raws.append(
                        RawRecord(
                            source=self.source,
                            entity=m.get("ticker", ""),
                            raw=raw,
                            fetched_at=now,
                        )
                    )

        self._warn_if_prices_look_dead(raws)
        return raws

    @staticmethod
    def _warn_if_prices_look_dead(raws: list[RawRecord]) -> None:
        """Loudly flag the case where every market prices at zero.

        """
        if len(raws) < 50:
            return
        priced = sum(
            1 for r in raws
            if KalshiIngestor._price(r.raw, "yes_ask") > 0
            or KalshiIngestor._price(r.raw, "yes_bid") > 0
        )
        if priced == 0:
            logger.error(
                "Kalshi: all %d markets priced at 0.0 — likely an API field "
                "rename; check the price keys in the /markets response",
                len(raws),
            )

    @staticmethod
    def _price(m: dict, side: str) -> float:
        """Return a price in probability units (0.0-1.0) for e.g. side='yes_ask'.
        """
        val = m.get(f"{side}_dollars")
        if val is not None:
            return float(val)
        legacy = m.get(side)
        return float(legacy or 0) / 100

    @staticmethod
    def _num(m: dict, *keys: str) -> float:
        """First present key parsed as float (Kalshi returns these as strings)."""
        for k in keys:
            v = m.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        m = raw.raw
        ticker = m.get("ticker", raw.entity)
        now = raw.fetched_at

        yes_ask = self._price(m, "yes_ask")
        yes_bid = self._price(m, "yes_bid")
        no_ask = self._price(m, "no_ask")
        no_bid = self._price(m, "no_bid")
        # volume_24h_fp matches the downstream `kalshi_volume_24h` feature and
        # the opportunity filter's liquidity intent (is this market traded
        # *now*), where lifetime `volume_fp` would let a dormant market pass.
        volume = float(m.get("volume_24h_fp") or m.get("volume") or 0)

        # Captured for market-quality filtering
        # rejects exactly the markets that are responding to information.
        #   volume_lifetime — has this contract *ever* drawn real participation
        #   yes_bid_size / yes_ask_size — resting size at top of book, i.e. how
        #   much can be filled before the quote moves
        volume_lifetime = self._num(m, "volume_fp", "volume")
        yes_bid_size = self._num(m, "yes_bid_size_fp")
        yes_ask_size = self._num(m, "yes_ask_size_fp")

        spread = round(yes_ask - yes_bid, 4)
        mid = round((yes_ask + yes_bid) / 2, 4) if yes_ask and yes_bid else yes_ask

        close_str = m.get("close_time") or m.get("expected_expiration_time")
        try:
            close_time = datetime.fromisoformat(str(close_str).rstrip("Z")).replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            close_time = None

        records = [
            FeatureRecord(
                source=self.source,
                entity=ticker,
                signal_type="market_price",
                value=mid,
                metadata={
                    "yes_ask": yes_ask,
                    "yes_bid": yes_bid,
                    "no_ask": no_ask,
                    "no_bid": no_bid,
                    "spread": spread,
                    "status": m.get("status", ""),
                    "title": m.get("title", ""),
                    "close_time": close_str,
                    "category": m.get("category", ""),
                    "event_ticker": m.get("event_ticker", ""),
                    # Top-of-book depth, alongside the prices it applies to.
                    "yes_bid_size": yes_bid_size,
                    "yes_ask_size": yes_ask_size,
                },
                ticker=ticker,
                as_of_ts=now,
            ),
            FeatureRecord(
                source=self.source,
                entity=ticker,
                signal_type="market_volume",
                value=volume,
                metadata={
                    "open_interest": self._num(m, "open_interest_fp", "open_interest"),
                    "volume_lifetime": volume_lifetime,
                },
                ticker=ticker,
                as_of_ts=now,
            ),
            FeatureRecord(
                source=self.source,
                entity=ticker,
                signal_type="market_spread",
                value=spread,
                metadata={},
                ticker=ticker,
                as_of_ts=now,
            ),
        ]

        if m.get("_emit_terms"):
            terms = self._terms(m)
            records.append(
                FeatureRecord(
                    source=self.source,
                    entity=ticker,
                    signal_type=self.SOURCE_TERMS_SIGNAL,
                    # The row exists to carry metadata; value is the number of
                    # terms fields the payload actually supplied, which makes a
                    # thinning upstream response visible in the signal itself.
                    value=float(len(terms)),
                    metadata={
                        **terms,
                        "terms_hash": m.get("_terms_hash", ""),
                        "title": m.get("title", ""),
                        "category": m.get("category", ""),
                        "series_ticker": m.get("series_ticker", ""),
                        "event_ticker": m.get("event_ticker", ""),
                        "close_time": close_str,
                    },
                    ticker=ticker,
                    as_of_ts=now,
                )
            )

        return records

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> list[FeatureRecord]:
        """Poll, write, then remember which terms rows actually persisted.

        The terms cache advances on the write rather than on the fetch for the
        same reason the resolution archive does: BaseIngestor.run logs a
        write_batch failure and returns an empty list, and a terms row skipped
        on that basis would never be re-emitted — the next cycle would see an
        unchanged fingerprint and stay quiet. Price rows are unaffected either
        way, since the next poll re-reads the same quote.
        """
        written = await super().run()
        if written:
            cache = self._ensure_terms_seen()
            for r in written:
                if r.signal_type == self.SOURCE_TERMS_SIGNAL:
                    cache[r.entity] = str(r.metadata.get("terms_hash", ""))
        return written

