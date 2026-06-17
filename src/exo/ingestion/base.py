"""BaseIngestor abstract base class."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from exo.event_bus import EventBus, get_bus
from exo.models import FeatureRecord, FeatureUpdated, RawRecord
from exo.store.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """Abstract base class for all data ingestors.

    Subclasses must implement :meth:`fetch` and :meth:`normalise`.
    :meth:`run` orchestrates the full fetch → normalise → write cycle.
    """

    source: str  # must be overridden by subclass

    def __init__(
        self,
        store: FeatureStore | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.store = store or FeatureStore()
        self.bus = bus or get_bus()
        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch(self) -> list[RawRecord]:
        """Fetch raw data from the external source.

        Returns a list of :class:`~exo.models.RawRecord` objects.
        Implementations should handle rate-limiting and transient errors
        internally; persistent errors may raise.
        """

    @abstractmethod
    def normalise(self, raw: RawRecord) -> list[FeatureRecord]:
        """Transform a single :class:`~exo.models.RawRecord` into one or
        more typed :class:`~exo.models.FeatureRecord` objects.
        """

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> list[FeatureRecord]:
        """Fetch → normalise → write → publish cycle.

        Returns all written :class:`~exo.models.FeatureRecord` objects.
        """
        self._logger.info("Starting ingest cycle for source=%s", self.source)
        try:
            raws = await self.fetch()
        except Exception as exc:
            self._logger.error("fetch() failed for source=%s: %s", self.source, exc)
            return []

        all_records: list[FeatureRecord] = []
        loop = asyncio.get_event_loop()
        for raw in raws:
            try:
                records = await loop.run_in_executor(None, self.normalise, raw)
            except Exception as exc:
                self._logger.warning(
                    "normalise() failed for source=%s entity=%s: %s",
                    self.source,
                    raw.entity,
                    exc,
                )
                continue
            all_records.extend(records)

        written: list[FeatureRecord] = []
        if all_records:
            try:
                await loop.run_in_executor(None, self.store.write_batch, all_records)
                written = all_records
            except Exception as exc:
                self._logger.error("write_batch() failed for source=%s: %s", self.source, exc)

            for record in written:
                event = FeatureUpdated(record=record)
                await self.bus.publish(event)

        self._logger.info(
            "Ingest cycle complete for source=%s: %d records written",
            self.source,
            len(written),
        )
        return written

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)
