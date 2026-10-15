"""Kalshi settlement ingestor — persists resolution outcomes locally.

Why this exists as its own ingestor rather than a branch of KalshiIngestor:
the price poll is a *sampling* job (every 15 min, same tickers, each cycle a
new observation), while this is an *archival* job (each ticker settles exactly
once, and the row is immutable thereafter). They want opposite write
behaviour — append always vs. append once — and opposite cadences, so keeping
them separate avoids a dedupe path in the hot price loop.

The resolution outcome is the supervision label for everything downstream: a
contract's price history is worthless for scoring without knowing how it
settled. Kalshi serves settled markets from /markets?status=settled, but that
window is finite — once a market ages out, the label is gone and no amount of
stored price history can reconstruct it. So the outcome is captured on a
schedule and written to the feature store, where it is as durable as the
prices it labels.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from exo import config
from exo.ingestion.base import BaseIngestor
from exo.ingestion.kalshi import _PROBE_CONCURRENCY, _build_auth_headers
from exo.models import FeatureRecord, RawRecord

logger = logging.getLogger(__name__)

# The series catalog changes slowly; one unpaginated call a day is plenty.
_SERIES_REFRESH_INTERVAL = timedelta(hours=24)

# Kalshi caps /markets page size at 200.
_SETTLED_PAGE_LIMIT = 200

# Pages to walk per series per sweep. One page covers a 6-hour incremental
# sweep many times over, but the *first* sweep faces the whole historical
# backlog — and a series like KXTRUMPAPPROVE settles hundreds of strikes a
# day, so without paging the initial backfill would silently cap at 200 per
# series and lose exactly the oldest labels this archive exists to rescue.
# Paging stops early as soon as a page adds nothing new, so steady-state
# sweeps still cost one request per series.
_MAX_PAGES_PER_SERIES = 25

# Rate-limit retries allowed on a single page before abandoning the series for
# this sweep. The next sweep picks up where this one stopped.
_MAX_RETRIES_PER_PAGE = 3


class KalshiResolutionIngestor(BaseIngestor):
    """Archive settled-market outcomes for focused-category series.

    Parameters
    ----------
    categories:
        Categories whose series to archive. Defaults to
        ``config.FOCUSED_CATEGORIES``.
    """

    source = "kalshi_resolution"
    signal_type = "market_resolution"

    def __init__(self, categories: set[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.categories = categories or config.FOCUSED_CATEGORIES

        self._series_category: dict[str, str] = {}
        self._series_refreshed_at: datetime | None = None

        # Tickers already archived. Seeded lazily from the store on first
        # fetch so a restart doesn't re-append every historical settlement.
        self._seen: set[str] | None = None

    # ------------------------------------------------------------------
    # Dedupe state
    # ------------------------------------------------------------------

    def _ensure_seen(self) -> set[str]:
        """Return the archived-ticker set, caching it in memory.

        A failure here propagates rather than degrading to an empty set: an
        empty set means "archive everything", which on a populated store
        would duplicate every row it already holds.
        """
        if self._seen is None:
            self._seen = self.store.distinct_entities(self.source, self.signal_type)
            logger.info(
                "Kalshi resolution archive seeded with %d previously settled tickers",
                len(self._seen),
            )
        return self._seen

    def _refresh_seen(self) -> set[str]:
        """Re-read the archive and union it into the cached set.

        The in-memory cache alone assumes this process is the only writer. It
        isn't necessarily: a one-off backfill run against the same store while
        the scheduler is up produced 504 duplicate rows, because the scheduler
        had seeded its cache hours earlier and could not know about them. The
        archive lives in its own small source directory, so re-reading it each
        sweep costs ~1s against a 6-hour cadence — cheap insurance against a
        failure mode whose only symptom is a quietly double-counted label.

        Unioning rather than replacing keeps this process's own writes, which
        run() records before they could show up in a fresh read.
        """
        current = self._ensure_seen()
        on_disk = self.store.distinct_entities(self.source, self.signal_type)
        added = on_disk - current
        if added:
            logger.info(
                "Kalshi resolution archive grew by %d tickers written elsewhere "
                "since this process last looked", len(added),
            )
        current |= on_disk
        return current

    # ------------------------------------------------------------------
    # Series discovery
    # ------------------------------------------------------------------

    async def _refresh_series(self, client: httpx.AsyncClient) -> None:
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
            "Kalshi resolution series allowlist refreshed: %d series",
            len(self._series_category),
        )

    async def _probe_settled(
        self,
        client: httpx.AsyncClient,
        series_ticker: str,
        sem: asyncio.Semaphore,
        seen: set[str],
    ) -> list[dict]:
        """Return settled markets for one series, paging past the 200 cap.

        Stops as soon as a page contributes no ticker outside *seen*: on an
        incremental sweep that is page one, so the common case costs a single
        request, while a first run keeps paging through the backlog.
        """
        out: list[dict] = []
        cursor: str | None = None
        pages = 0
        retries = 0

        async with sem:
            while pages < _MAX_PAGES_PER_SERIES:
                path = "/markets"
                headers = _build_auth_headers("GET", "/trade-api/v2" + path)
                params: dict[str, Any] = {
                    "series_ticker": series_ticker,
                    "status": "settled",
                    "limit": _SETTLED_PAGE_LIMIT,
                }
                if cursor:
                    params["cursor"] = cursor
                try:
                    resp = await client.get(
                        config.KALSHI_BASE_URL + path, headers=headers, params=params
                    )
                    if resp.status_code == 429:
                        # A retry re-reads the same page, so it must not count
                        # against the page budget — otherwise a rate-limited
                        # series stops paging early and looks like it ran out
                        # of pages.
                        if retries >= _MAX_RETRIES_PER_PAGE:
                            logger.warning(
                                "Kalshi settled probe for %s rate limited %d times "
                                "on one page; giving up on this series for now",
                                series_ticker, retries,
                            )
                            break
                        retries += 1
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logger.debug(
                            "Kalshi rate limited on settled probe series=%s; sleeping %ds",
                            series_ticker, retry_after,
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.debug(
                        "Kalshi settled probe failed for %s: %s", series_ticker, exc
                    )
                    break

                pages += 1
                retries = 0

                markets = data.get("markets", [])
                if not markets:
                    break
                out.extend(markets)

                # Nothing on this page is new, so older pages are already
                # archived — Kalshi returns settled markets newest-first.
                if all(m.get("ticker", "") in seen for m in markets):
                    break

                cursor = data.get("cursor")
                if not cursor:
                    break
            else:
                logger.warning(
                    "Kalshi settled probe for %s hit the %d-page cap; older "
                    "settlements in this series were not read this sweep",
                    series_ticker, _MAX_PAGES_PER_SERIES,
                )

        return out

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    async def fetch(self) -> list[RawRecord]:
        seen = self._refresh_seen()
        now = self.utcnow()
        raws: list[RawRecord] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            if (
                self._series_refreshed_at is None
                or now - self._series_refreshed_at > _SERIES_REFRESH_INTERVAL
            ):
                await self._refresh_series(client)

            if not self._series_category:
                logger.warning(
                    "Kalshi resolution series allowlist is empty; skipping cycle"
                )
                return raws

            sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
            series = list(self._series_category.keys())
            results = await asyncio.gather(
                *(self._probe_settled(client, s, sem, seen) for s in series)
            )

        skipped_pending = 0
        non_binary = 0
        # Intra-sweep dedupe only. The persisted `seen` set is deliberately not
        # touched here: a ticker is "archived" once its row is on disk, and
        # marking it before the write would make a failed write_batch lose that
        # label silently and permanently. run() commits these after the write.
        batch: set[str] = set()
        for series_ticker, markets in zip(series, results):
            category = self._series_category.get(series_ticker, "")
            for m in markets:
                ticker = m.get("ticker", "")
                if not ticker or ticker in seen or ticker in batch:
                    continue
                # Two different things look like "no result yet":
                #   result="" / absent, not finalized — settled but the outcome
                #     has not been posted. Deferred: a later sweep gets it.
                #   result="scalar" (or any finalized non-yes/no) — settled for
                #     good at a fractional value on the notional: a tie
                #     splitting 50/50, or a proportionally-settled strike
                #     ladder. Archived with that fraction as the value rather
                #     than discarded, because it is a real outcome and carries
                #     magnitude: a contract that traded at 0.20 and settled at
                #     0.44 was underpriced, which is exactly what a mispricing
                #     objective wants to learn from. It is flagged non-binary so
                #     consumers that need a 0/1 label can exclude it — what must
                #     never happen is rounding a $0.50 tie to one side, which
                #     would assert an outcome that did not occur.
                result = m.get("result")
                is_binary = result in ("yes", "no")
                finalized = result == "scalar" or m.get("status") == "finalized"
                if not is_binary:
                    if not finalized:
                        skipped_pending += 1
                        continue
                    if self._settlement_fraction(m) is None:
                        # Finalized, non-binary, and no usable settlement
                        # value — nothing can be said about the outcome.
                        skipped_pending += 1
                        continue
                    non_binary += 1
                raws.append(
                    RawRecord(
                        source=self.source,
                        entity=ticker,
                        raw={**m, "category": category, "series_ticker": series_ticker},
                        fetched_at=now,
                    )
                )
                batch.add(ticker)

        logger.info(
            "Kalshi resolution sweep: %d new settlements across %d series "
            "(%d of them non-binary fractional settlements; %d deferred "
            "awaiting a posted result)",
            len(raws), len(series), non_binary, skipped_pending,
        )
        return raws

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> list[FeatureRecord]:
        """Fetch and write, then mark only the persisted rows as archived.

        BaseIngestor.run swallows a write_batch failure and returns an empty
        list. For a sampling ingestor that is harmless — the next poll
        re-reads the same quote. Here the row is a one-time label, so the
        seen-set must advance on the write, not on the fetch, or a dropped
        batch would be skipped forever.
        """
        written = await super().run()
        if written:
            self._ensure_seen().update(r.entity for r in written)
        return written

    # ------------------------------------------------------------------
    # Normalise
    # ------------------------------------------------------------------

    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        m = raw.raw
        ticker = m.get("ticker", raw.entity)
        result = m.get("result")
        is_binary = result in ("yes", "no")

        # close_time is the actual settlement moment. expiration_time and
        # expected_expiration_time are scheduled backstop deadlines that can
        # land hours or days later, so using either would misdate the label
        # and, in a walk-forward split, leak a resolution into the future.
        close_str = m.get("close_time")
        resolved_at = self._parse_ts(close_str) or raw.fetched_at

        # For a binary market the value *is* the outcome; for a fractional
        # settlement it is the realised fraction of notional, on the same 0-1
        # scale as the stored prices, so both can be scored the same way.
        fraction = self._settlement_fraction(m)
        if is_binary:
            value = 1.0 if result == "yes" else 0.0
        else:
            value = fraction if fraction is not None else 0.0

        return [
            FeatureRecord(
                source=self.source,
                entity=ticker,
                signal_type=self.signal_type,
                value=value,
                metadata={
                    "result": result,
                    "is_binary": is_binary,
                    "settlement_fraction": fraction,
                    "settlement_value_dollars": m.get("settlement_value_dollars"),
                    "notional_value_dollars": m.get("notional_value_dollars"),
                    "resolution_ts": close_str,
                    "category": m.get("category", ""),
                    "series_ticker": m.get("series_ticker", ""),
                    "event_ticker": m.get("event_ticker", ""),
                    "title": m.get("title", ""),
                    # Settlement-time liquidity, for filtering labels that come
                    # from markets nobody actually traded.
                    "volume_lifetime": self._num(m, "volume_fp", "volume"),
                    "open_interest": self._num(m, "open_interest_fp", "open_interest"),
                },
                ticker=ticker,
                # Dated to settlement, not to the sweep that found it, so the
                # date= partition reflects when the outcome became true and a
                # point-in-time read can't see a label before it existed.
                as_of_ts=resolved_at,
            )
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _settlement_fraction(cls, m: dict) -> float | None:
        """Settled value as a fraction of notional, or None if unusable.

        Kalshi reports both in dollars; the ratio is the contract's realised
        probability-equivalent, directly comparable to the prices stored for it.
        A missing or zero notional yields None rather than a divide-by-zero or a
        silently wrong 0.0.
        """
        raw_value = m.get("settlement_value_dollars")
        if raw_value is None:
            return None
        try:
            value = float(raw_value)
            notional = float(m.get("notional_value_dollars") or 0)
        except (TypeError, ValueError):
            return None
        if notional <= 0:
            return None
        return min(max(value / notional, 0.0), 1.0)

    @staticmethod
    def _parse_ts(value: Any) -> datetime | None:
        try:
            return datetime.fromisoformat(str(value).rstrip("Z")).replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _num(m: dict, *keys: str) -> float:
        for k in keys:
            v = m.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0
