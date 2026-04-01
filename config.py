"""Pipeline constants and API credentials loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"
RISK_INDEX_DIR = DATA_DIR / "risk_index"

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

KALSHI_API_KEY_ID: str = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY: str = os.getenv("KALSHI_PRIVATE_KEY", "")  # PEM content
KALSHI_BASE_URL: str = os.getenv(
    "KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
)
KALSHI_WS_URL: str = os.getenv(
    "KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2"
)

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
ACLED_API_KEY: str = os.getenv("ACLED_API_KEY", "")
ACLED_EMAIL: str = os.getenv("ACLED_EMAIL", "")

FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")

REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "exo-bot/1.0")

EIA_API_KEY: str = os.getenv("EIA_API_KEY", "")

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Staleness thresholds (seconds)
# ---------------------------------------------------------------------------

STALENESS_THRESHOLDS: dict[str, float] = {
    "gdelt": 15 * 60,           # 15 min
    "kalshi": 15 * 60,          # 15 min
    "finnhub": 30 * 60,         # 30 min
    "reddit": 60 * 60,          # 1 hour
    "google_trends": 2 * 3600,  # 2 hours
    "fivethirtyeight": 6 * 3600,
    "fred": 12 * 3600,
    "world_bank": 24 * 3600,
    "eia": 24 * 3600,
    "polymarket": 60 * 60,
    "acled": 7 * 24 * 3600,     # weekly
}

# ---------------------------------------------------------------------------
# Risk index dimension weights
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: dict[str, float] = {
    "political_stability": 0.25,
    "conflict_intensity": 0.25,
    "policy_predictability": 0.20,
    "sanctions_risk": 0.15,
    "economic_stress": 0.15,
}

# ---------------------------------------------------------------------------
# Validation benchmarks
# ---------------------------------------------------------------------------

ICRG_CORRELATION_TARGET: float = 0.65
