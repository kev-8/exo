"""One-off script to seed WGI governance indicators into the feature store.

Data source: World Bank WGI 2024 report (published Sep/Oct 2025), retrieved via
TheGlobalEconomy.com. Most cells are 2024 vintage; some RL/CC/VA cells fall back
to 2023 where 2024 data was not yet indexed. All values normalised: (raw + 2.5) / 5.0.

WGI indicators are no longer available via the World Bank v2 API (archived May 2026).

Signal types written:
  political_stability    ← PV.EST
  voice_accountability   ← VA.EST
  rule_of_law            ← RL.EST
  control_of_corruption  ← CC.EST

Usage:
    cd /Users/kevin/Desktop/ds/exo
    source /Users/kevin/Desktop/ds/environments/exo_env/bin/activate
    python scripts/seed_wgi.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exo.models import FeatureRecord
from exo.store.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Normalised WGI 2024 values: (raw + 2.5) / 5.0
# Format: iso2 -> (PV_norm, VA_norm, RL_norm, CC_norm)
WGI_2024: dict[str, tuple[float, float, float, float]] = {
    "US": (0.480, 0.678, 0.766, 0.702),
    "RU": (0.320, 0.248, 0.268, 0.304),
    "CN": (0.470, 0.172, 0.492, 0.498),
    "UA": (0.414, 0.478, 0.316, 0.362),
    "IL": (0.292, 0.628, 0.656, 0.692),
    "IR": (0.184, 0.208, 0.288, 0.330),
    "IN": (0.342, 0.440, 0.496, 0.426),
    "PK": (0.088, 0.266, 0.328, 0.300),
    "KP": (0.426, 0.104, 0.194, 0.188),
    "TW": (0.684, 0.716, 0.710, 0.704),
    "HT": (0.214, 0.336, 0.190, 0.212),
    "BR": (0.396, 0.568, 0.448, 0.400),
    "MX": (0.356, 0.502, 0.338, 0.312),
    "NG": (0.100, 0.396, 0.322, 0.304),
    "KE": (0.228, 0.450, 0.434, 0.392),
    "ZA": (0.366, 0.618, 0.512, 0.444),
    "FR": (0.452, 0.724, 0.736, 0.770),
    "GB": (0.552, 0.750, 0.844, 0.834),
    "MY": (0.576, 0.412, 0.570, 0.560),
    "CL": (0.524, 0.704, 0.626, 0.752),
}

AS_OF_TS = datetime(2024, 12, 31, tzinfo=timezone.utc)

SIGNAL_TYPES = [
    "political_stability",
    "voice_accountability",
    "rule_of_law",
    "control_of_corruption",
]


def main() -> None:
    store = FeatureStore()
    records: list[FeatureRecord] = []

    for iso2, (pv, va, rl, cc) in WGI_2024.items():
        for signal_type, value in zip(SIGNAL_TYPES, (pv, va, rl, cc)):
            records.append(FeatureRecord(
                source="world_bank",
                entity=iso2,
                signal_type=signal_type,
                value=value,
                metadata={"vintage": "WGI_2024", "normalisation": "(raw + 2.5) / 5.0"},
                as_of_ts=AS_OF_TS,
            ))

    store.write_batch(records)
    logger.info("Wrote %d WGI records for %d countries", len(records), len(WGI_2024))


if __name__ == "__main__":
    main()
