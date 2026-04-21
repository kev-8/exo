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
    "google_trends": 2 * 3600,  # 2 hours
    "fred": 12 * 3600,
    "world_bank": 24 * 3600,
    "eia": 24 * 3600,
    "polymarket": 60 * 60,
    "acled": 7 * 24 * 3600,          # weekly
    "unga_votes": 7 * 24 * 3600,     # weekly
    "wits": 7 * 24 * 3600,           # weekly poll, annual data
    "ofac": 7 * 24 * 3600,           # weekly poll (SDN list updated frequently)
    "ucdp_ged": 7 * 24 * 3600,       # weekly poll, annual releases
    "ucdp_candidate": 7 * 24 * 3600, # weekly poll, quarterly revisions
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

# Tier weights per dimension — controls how structural / short_term / acute
# sub-scores blend into the dimension's final score.
DIMENSION_TIER_WEIGHTS: dict[str, dict[str, float]] = {
    "political_stability":   {"structural": 1.0, "short_term": 0.0, "acute": 0.0},
    "conflict_intensity":    {"structural": 0.3, "short_term": 0.4, "acute": 0.3},
    "policy_predictability": {"structural": 1.0, "short_term": 0.0, "acute": 0.0},
    "sanctions_risk":        {"structural": 0.4, "short_term": 0.2, "acute": 0.4},
    "economic_stress":       {"structural": 0.3, "short_term": 0.4, "acute": 0.3},
}

# Minimum normalised SDN score for a country to be included as a bilateral
# trade partner in the WITS secondary_exposure query. Derived from OFAC data
# at fetch() time — no hardcoded sanctioned-country list.
SDN_BILATERAL_THRESHOLD: float = float(os.getenv("SDN_BILATERAL_THRESHOLD", "0.05"))

# ---------------------------------------------------------------------------
# Validation benchmarks
# ---------------------------------------------------------------------------

ICRG_CORRELATION_TARGET: float = 0.65
