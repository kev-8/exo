"""APScheduler job scheduler for all data ingestion and risk index jobs."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from exo import config
from exo.event_bus import get_bus
from exo.ingestion.acled import ACLEDIngestor
from exo.ingestion.eia import EIAIngestor
from exo.ingestion.finnhub import FinnhubIngestor
from exo.ingestion.fred import FREDIngestor
from exo.ingestion.gdelt import GDELTIngestor
from exo.ingestion.google_trends import GoogleTrendsIngestor
from exo.ingestion.kalshi import KalshiIngestor
from exo.ingestion.polymarket import PolymarketIngestor
from exo.ingestion.reddit import RedditIngestor
from exo.ingestion.unga_votes import UNGAVotesIngestor
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
    """

    def __init__(self, store: FeatureStore | None = None) -> None:
        self.store = store or FeatureStore()
        self.bus = get_bus()
        self._scheduler = AsyncIOScheduler()

        _kwargs = {"store": self.store, "bus": self.bus}
        self.gdelt = GDELTIngestor(**_kwargs)
        self.kalshi = KalshiIngestor(**_kwargs)
        self.acled = ACLEDIngestor(**_kwargs)
        self.fred = FREDIngestor(**_kwargs)
        self.google_trends = GoogleTrendsIngestor(**_kwargs)
        self.reddit = RedditIngestor(**_kwargs)
        self.world_bank = WorldBankIngestor(**_kwargs)
        self.unga_votes = UNGAVotesIngestor(**_kwargs)
        self.eia = EIAIngestor(**_kwargs)
        self.polymarket = PolymarketIngestor(**_kwargs)
        self.finnhub = FinnhubIngestor(**_kwargs)

        self.risk_index = RiskIndexEngine(store=self.store)

    # ------------------------------------------------------------------
    # Schedule registration
    # ------------------------------------------------------------------

    def _add_ingestor_job(self, ingestor, job_id: str, **trigger_kwargs) -> None:
        async def _run():
            try:
                await ingestor.run()
            except Exception as exc:
                logger.error("Ingestor job %s failed: %s", job_id, exc)

        self._scheduler.add_job(_run, id=job_id, **trigger_kwargs)

    def setup_jobs(self) -> None:
        """Register all scheduled jobs."""

        # GDELT — every 15 min
        self._add_ingestor_job(self.gdelt, "gdelt_ingest",
                               trigger=IntervalTrigger(minutes=15))

        # Kalshi market data — every 15 min
        self._add_ingestor_job(self.kalshi, "kalshi_rest",
                               trigger=IntervalTrigger(minutes=15))

        # Finnhub — every 30 min
        self._add_ingestor_job(self.finnhub, "finnhub_ingest",
                               trigger=IntervalTrigger(minutes=30))

        # Reddit — every 1 hour
        self._add_ingestor_job(self.reddit, "reddit_ingest",
                               trigger=IntervalTrigger(hours=1))

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

        # EIA — daily at 06:00 UTC
        self._add_ingestor_job(self.eia, "eia_ingest",
                               trigger=CronTrigger(hour=6, minute=0, timezone="UTC"))

        # UNGA Votes — weekly Monday 09:00 UTC
        self._add_ingestor_job(self.unga_votes, "unga_votes_ingest",
                               trigger=CronTrigger(day_of_week="mon", hour=9, minute=0, timezone="UTC"))

        # ACLED — weekly Monday 08:00 UTC
        self._add_ingestor_job(self.acled, "acled_ingest",
                               trigger=CronTrigger(day_of_week="mon", hour=8, minute=0, timezone="UTC"))

        # Risk index update — every 6 hours
        self._scheduler.add_job(
            self._risk_index_update, id="risk_index_update",
            trigger=IntervalTrigger(hours=6),
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
        for source, threshold_sec in config.STALENESS_THRESHOLDS.items():
            rec = self.store.get_latest(
                entity=None,
                signal_type=None,
                source=source,
                max_age_sec=threshold_sec,
            )
            if rec is None:
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
        self.setup_jobs()
        await self.bus.start()
        self.kalshi.start_websocket()
        self._scheduler.start()
        logger.info("ExoScheduler started")

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        await self.bus.stop()
        logger.info("ExoScheduler stopped")
