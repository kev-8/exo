"""APScheduler job scheduler for all data ingestion and risk index jobs."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.combining import OrTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from exo import config
from exo.event_bus import get_bus
from exo.ingestion.eia import EIAIngestor
from exo.ingestion.finnhub import FinnhubIngestor
from exo.ingestion.fred import FREDIngestor
from exo.ingestion.gdelt import GDELTIngestor
from exo.ingestion.google_trends import GoogleTrendsIngestor
from exo.ingestion.kalshi import KalshiIngestor
from exo.ingestion.kalshi_resolution import KalshiResolutionIngestor
from exo.ingestion.polymarket import PolymarketIngestor
from exo.ingestion.ofac import OFACIngestor
from exo.ingestion.ucdp import UCDPCandidateIngestor, UCDPGEDIngestor, seed_missing
from exo.ingestion.unga_votes import UNGAVotesIngestor
from exo.ingestion.wits import WITSIngestor
from exo.ingestion.wgi import WGIIngestor
from exo.ingestion.world_bank import WorldBankIngestor
from exo.models import StalenessAlert
from exo.risk_index.engine import RiskIndexEngine, COUNTRIES
from exo.store.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class ExoScheduler:
    """Orchestrate all periodic data ingestion and risk index update jobs.

    Parameters
    ----------
    store:
        Shared feature store instance.
    exclude_jobs:
        Job ids to skip — e.g.
        {"kalshi_rest"} on a deployment that doesn't need Kalshi.
    """

    def __init__(self, store: FeatureStore | None = None, exclude_jobs: set[str] = frozenset()) -> None:
        self.store = store or FeatureStore()
        self.bus = get_bus()
        # A job that starts a few seconds late (the event loop was busy with
        # another job) must still run: APScheduler's default 1s grace silently
        # dropped about a quarter of GDELT cycles while Kalshi was processing.
        # coalesce collapses a backlog of missed runs into one.
        self._scheduler = AsyncIOScheduler(
            job_defaults={"misfire_grace_time": 300, "coalesce": True}
        )
        self._exclude_jobs = exclude_jobs
        # Populated as jobs are skipped in _add_ingestor_job — a source this
        # deployment never ingests will never have fresh data, so alerting on
        # it forever is just noise, not a signal.
        self._excluded_sources: set[str] = set()

        _kwargs = {"store": self.store, "bus": self.bus}
        self.gdelt = GDELTIngestor(**_kwargs)
        self.kalshi = KalshiIngestor(**_kwargs)
        self.kalshi_resolution = KalshiResolutionIngestor(**_kwargs)
        self.fred = FREDIngestor(**_kwargs)
        self.google_trends = GoogleTrendsIngestor(**_kwargs)
        self.world_bank = WorldBankIngestor(**_kwargs)
        self.wgi = WGIIngestor(**_kwargs)
        self.unga_votes = UNGAVotesIngestor(**_kwargs)
        self.eia = EIAIngestor(**_kwargs)
        self.polymarket = PolymarketIngestor(**_kwargs)
        self.finnhub = FinnhubIngestor(**_kwargs)
        self.ofac = OFACIngestor(**_kwargs)
        self.wits = WITSIngestor(**_kwargs)
        self.ucdp_ged = UCDPGEDIngestor(**_kwargs)
        self.ucdp_candidate = UCDPCandidateIngestor(**_kwargs)

        self.risk_index = RiskIndexEngine(store=self.store)

    # ------------------------------------------------------------------
    # Schedule registration
    # ------------------------------------------------------------------

    def _add_ingestor_job(self, ingestor, job_id: str, **trigger_kwargs) -> None:
        if job_id in self._exclude_jobs:
            self._excluded_sources.add(ingestor.source)
            logger.info("Skipping job %s (excluded)", job_id)
            return

        async def _run():
            try:
                await ingestor.run()
            except Exception as exc:
                logger.error("Ingestor job %s failed: %s", job_id, exc)

        self._scheduler.add_job(_run, id=job_id, **trigger_kwargs)

    def setup_jobs(self) -> None:
        """Register all scheduled jobs."""

        # GDELT — every 15 min, matching the cadence GDELT publishes its event
        # exports at. The first cycle after a restart backfills a week of files.
        self._add_ingestor_job(self.gdelt, "gdelt_ingest",
                               trigger=IntervalTrigger(minutes=15))

        # Kalshi market data — every 15 min
        self._add_ingestor_job(self.kalshi, "kalshi_rest",
                               trigger=IntervalTrigger(minutes=15))

        # Kalshi settlements — every 6 hours. Unlike the price poll this is
        # archival: each contract settles once and the row never changes, so
        # the cadence only needs to beat the rate at which Kalshi ages settled
        # markets out of /markets. Losing a settlement loses the label for
        # every price observation of that contract, permanently.
        self._add_ingestor_job(self.kalshi_resolution, "kalshi_resolution",
                               trigger=IntervalTrigger(hours=6))

        # Finnhub — every 30 min
        self._add_ingestor_job(self.finnhub, "finnhub_ingest",
                               trigger=IntervalTrigger(minutes=30))

        # Polymarket — every 1 hour
        self._add_ingestor_job(self.polymarket, "polymarket_ingest",
                               trigger=IntervalTrigger(hours=1))

        # Google Trends — every 2 hours
        self._add_ingestor_job(self.google_trends, "google_trends_ingest",
                               trigger=IntervalTrigger(hours=2))

        # FRED — every 12 hours
        self._add_ingestor_job(self.fred, "fred_ingest",
                               trigger=IntervalTrigger(hours=12))

        # World Bank — daily at 06:00 UTC
        self._add_ingestor_job(self.world_bank, "world_bank_ingest",
                               trigger=CronTrigger(hour=6, minute=0, timezone="UTC"))

        # WGI — annual release on an unannounced date, usually Sep–Dec: weekly
        # Mondays through that window, the 1st of the month otherwise, plus once
        # at startup so a fresh deploy doesn't wait for the next poll. A poll
        # with no newer edition is one HEAD request and writes nothing.
        self._add_ingestor_job(
            self.wgi, "wgi_ingest",
            trigger=OrTrigger([
                CronTrigger(month="9-12", day_of_week="mon", hour=6, minute=15, timezone="UTC"),
                CronTrigger(month="1-8", day=1, hour=6, minute=15, timezone="UTC"),
            ]),
            next_run_time=datetime.now(timezone.utc),
        )

        # EIA — daily at 06:00 UTC
        self._add_ingestor_job(self.eia, "eia_ingest",
                               trigger=CronTrigger(hour=6, minute=0, timezone="UTC"))

        # UNGA Votes — weekly Monday 09:00 UTC
        self._add_ingestor_job(self.unga_votes, "unga_votes_ingest",
                               trigger=CronTrigger(day_of_week="mon", hour=9, minute=0, timezone="UTC"))

        # OFAC SDN list — weekly Monday 07:00 UTC (before WITS so SDN data is fresh)
        self._add_ingestor_job(self.ofac, "ofac_ingest",
                               trigger=CronTrigger(day_of_week="mon", hour=7, minute=0, timezone="UTC"))

        # WITS trade concentration + secondary exposure — weekly Monday 07:30 UTC
        # Runs after OFAC so sanctioned-partner list is populated from live SDN data
        self._add_ingestor_job(self.wits, "wits_ingest",
                               trigger=CronTrigger(day_of_week="mon", hour=7, minute=30, timezone="UTC"))

        # UCDP GED (structural conflict) — weekly Monday 08:00 UTC
        self._add_ingestor_job(self.ucdp_ged, "ucdp_ged_ingest",
                               trigger=CronTrigger(day_of_week="mon", hour=8, minute=0, timezone="UTC"))

        # UCDP Candidate (short-term conflict) — weekly Monday 09:30 UTC
        self._add_ingestor_job(self.ucdp_candidate, "ucdp_candidate_ingest",
                               trigger=CronTrigger(day_of_week="mon", hour=9, minute=30, timezone="UTC"))

        # Risk index update — every 6 hours, first run shortly after startup
        # (after the startup WGI poll) so a deploy that adds countries or data
        # doesn't wait 6 hours to show it.
        self._scheduler.add_job(
            self._risk_index_update, id="risk_index_update",
            trigger=IntervalTrigger(hours=6),
            next_run_time=datetime.now(timezone.utc) + timedelta(minutes=2),
        )

        # Staleness check — every 5 minutes
        self._scheduler.add_job(
            self._staleness_check, id="staleness_check",
            trigger=IntervalTrigger(minutes=5),
        )

        logger.info("All jobs registered with scheduler")

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    async def _risk_index_update(self) -> None:
        logger.info("Risk index update triggered")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.risk_index.update_all)

    async def _staleness_check(self) -> None:
        loop = asyncio.get_event_loop()
        items = [
            (source, threshold_sec)
            for source, threshold_sec in config.STALENESS_THRESHOLDS.items()
            if source not in self._excluded_sources
        ]

        # Run filesystem checks sequentially — concurrent NFS stat() calls can
        # trigger kernel-level uninterruptible sleep (D-state) on Railway volumes.
        for source, threshold_sec in items:
            is_fresh = await loop.run_in_executor(
                None, self.store.has_recent_data, source, threshold_sec
            )
            if not is_fresh:
                alert = StalenessAlert(
                    source=source,
                    entity="",
                    last_seen=None,
                    threshold_sec=threshold_sec,
                    message=f"Source {source} has exceeded staleness threshold of {threshold_sec}s",
                )
                await self.bus.publish(alert)
                logger.warning("StalenessAlert: source=%s threshold=%ds", source, threshold_sec)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        # Fill UCDP gaps (newly tracked countries) before the first risk update.
        try:
            await asyncio.to_thread(seed_missing, self.store)
        except Exception as exc:
            logger.warning("UCDP seed failed: %s", exc)
        self.setup_jobs()
        await self.bus.start()
        self._scheduler.start()
        logger.info("ExoScheduler started")

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        await self.bus.stop()
        logger.info("ExoScheduler stopped")
