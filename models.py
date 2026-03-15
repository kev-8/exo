"""Shared dataclasses for the exo data pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Raw / Feature records
# ---------------------------------------------------------------------------


@dataclass
class RawRecord:
    """Unprocessed data from an external source."""

    source: str
    entity: str
    raw: dict[str, Any]
    fetched_at: datetime


@dataclass
class FeatureRecord:
    """Normalised, typed feature written to the feature store."""

    source: str                   # e.g. 'gdelt', 'kalshi', 'fred'
    entity: str                   # country code, market identifier, or composite key
    signal_type: str              # e.g. 'news_sentiment', 'market_price'
    value: float
    metadata: dict[str, Any]
    as_of_ts: datetime            # point-in-time timestamp for the observation
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str | None = None     # market identifier if applicable


@dataclass
class FeatureQuery:
    """Query parameters for feature store reads."""

    entity: str | None = None
    signal_type: str | None = None
    source: str | None = None
    ticker: str | None = None
    as_of_ts: datetime | None = None
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    max_age_sec: float | None = None
    limit: int = 1000


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@dataclass
class KalshiMarket:
    """Normalised Kalshi market metadata."""

    ticker: str
    title: str
    category: str
    status: str                   # 'open' | 'closed' | 'resolved'
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    volume: float
    open_interest: float
    close_time: datetime | None
    resolved_yes: bool | None = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Risk index outputs
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    """Score for one risk dimension."""

    name: str
    score: float                  # 0.0 – 1.0 (higher = higher risk)
    weight: float
    contributing_signals: list[str]


@dataclass
class RiskIndexSnapshot:
    """Country risk index snapshot."""

    country: str                  # ISO 3166-1 alpha-2
    composite_score: float        # weighted sum of dimension scores
    dimensions: list[DimensionScore]
    as_of_ts: datetime
    computed_at: datetime = field(default_factory=datetime.utcnow)
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Event bus events
# ---------------------------------------------------------------------------


@dataclass
class PipelineEvent:
    """Base class for pipeline events."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    emitted_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureUpdated(PipelineEvent):
    """Emitted after a FeatureRecord is written to the store."""

    record: FeatureRecord | None = None


@dataclass
class StalenessAlert(PipelineEvent):
    """Emitted when a data source exceeds its staleness threshold."""

    source: str = ""
    entity: str = ""
    last_seen: datetime | None = None
    threshold_sec: float = 0.0
    message: str = ""
