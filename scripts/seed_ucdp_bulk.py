"""One-off script to seed UCDP GED and Candidate data from public bulk downloads.

Downloads directly from https://ucdp.uu.se/downloads/ — no API token required.

  GED 25.1   → structural tier: ucdp_ged_events, ucdp_ged_fatalities
  Candidate  → short-term tier: ucdp_candidate_events

GED is filtered to GED_SINCE_YEAR (default 2015) so the structural score reflects
recent conflict history rather than all-time legacy events back to 1989.
Countries with zero events in the filtered window are skipped — they retain the
scorer's neutral 0.5 default rather than being anchored to 0.0.

Normalisation matches UCDPGEDIngestor / UCDPCandidateIngestor exactly
(log-scale against rolling max across all countries in the filtered window).

Usage:
    cd /Users/kevin/Desktop/ds/exo
    source /Users/kevin/Desktop/ds/environments/exo_env/bin/activate
    python scripts/seed_ucdp_bulk.py
"""

from __future__ import annotations

import io
import logging
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from exo.models import FeatureRecord
from exo.store.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ISO2_TO_UCDP: dict[str, int] = {
    "US": 2,   "RU": 365, "CN": 710, "UA": 369, "IL": 666,
    "IR": 630, "IN": 750, "PK": 770, "KP": 731, "TW": 713,
    "HT": 41,  "BR": 140, "MX": 70,  "NG": 475, "KE": 501,
    "ZA": 560, "FR": 220, "GB": 200, "MY": 820, "CL": 155,
}

GED_URL       = "https://ucdp.uu.se/downloads/ged/ged251-csv.zip"
CANDIDATE_URL = "https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_0_3.csv"

# Structural tier uses recent conflict history only — avoids legacy events
# from the 1980s-90s dominating countries that are peaceful today.
GED_SINCE_YEAR = 2010

_FALLBACK_MAX_EVENTS     = 500.0
_FALLBACK_MAX_FATALITIES = 10_000.0


def _log_normalise(value: float, rolling_max: float, fallback: float) -> float:
    denom = math.log1p(max(rolling_max, fallback))
    return min(1.0, math.log1p(max(0.0, value)) / denom)


def download_ged() -> pd.DataFrame:
    logger.info("Downloading GED 25.1 ...")
    r = httpx.get(GED_URL, timeout=120, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    df_recent = df[df["year"] >= GED_SINCE_YEAR]
    logger.info("GED rows total=%d  filtered (>=%d)=%d", len(df), GED_SINCE_YEAR, len(df_recent))
    return df_recent


def download_candidate() -> pd.DataFrame:
    logger.info("Downloading Candidate GED ...")
    r = httpx.get(CANDIDATE_URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    logger.info("Candidate rows: %d", len(df))
    return df


def aggregate_by_country(df: pd.DataFrame) -> dict[int, dict]:
    result: dict[int, dict] = {}
    fat_col = "best" if "best" in df.columns else None
    for gw_id, grp in df.groupby("country_id"):
        total_fat = int(grp[fat_col].fillna(0).clip(lower=0).sum()) if fat_col else 0
        result[int(gw_id)] = {
            "event_count": len(grp),
            "total_fatalities": total_fat,
        }
    return result


def seed_ged(store: FeatureStore, ged_df: pd.DataFrame, now: datetime) -> None:
    agg = aggregate_by_country(ged_df)
    gw_to_iso2 = {v: k for k, v in ISO2_TO_UCDP.items()}

    # Rolling max across ALL countries in the filtered dataset (not just our 20)
    all_events = [v["event_count"] for v in agg.values()]
    all_fat    = [v["total_fatalities"] for v in agg.values()]
    max_events = max(all_events) if all_events else _FALLBACK_MAX_EVENTS
    max_fat    = max(all_fat)    if all_fat    else _FALLBACK_MAX_FATALITIES

    written = 0
    for gw_id, iso2 in gw_to_iso2.items():
        d = agg.get(gw_id)
        if d is None or d["event_count"] == 0:
            logger.info("GED  %-2s  no events since %d — skipping (scorer defaults to 0.5)", iso2, GED_SINCE_YEAR)
            continue

        ec  = float(d["event_count"])
        fat = float(d["total_fatalities"])
        e_score = _log_normalise(ec,  max_events, _FALLBACK_MAX_EVENTS)
        f_score = _log_normalise(fat, max_fat,    _FALLBACK_MAX_FATALITIES)

        store.write(FeatureRecord(
            source="ucdp_ged", entity=iso2, signal_type="ucdp_ged_events",
            value=round(e_score, 4),
            metadata={"raw_event_count": ec, "rolling_max": max_events, "since_year": GED_SINCE_YEAR},
            as_of_ts=now,
        ))
        store.write(FeatureRecord(
            source="ucdp_ged", entity=iso2, signal_type="ucdp_ged_fatalities",
            value=round(f_score, 4),
            metadata={"raw_fatalities": fat, "rolling_max": max_fat, "since_year": GED_SINCE_YEAR},
            as_of_ts=now,
        ))
        logger.info("GED  %-2s  events=%5.0f (%.3f)  fatalities=%6.0f (%.3f)",
                    iso2, ec, e_score, fat, f_score)
        written += 2

    logger.info("Wrote %d GED feature records", written)


def seed_candidate(store: FeatureStore, cand_df: pd.DataFrame, now: datetime) -> None:
    agg = aggregate_by_country(cand_df)
    gw_to_iso2 = {v: k for k, v in ISO2_TO_UCDP.items()}

    all_events = [v["event_count"] for v in agg.values()]
    max_events = max(all_events) if all_events else _FALLBACK_MAX_EVENTS

    written = 0
    for gw_id, iso2 in gw_to_iso2.items():
        d = agg.get(gw_id)
        if d is None or d["event_count"] == 0:
            logger.info("CAND %-2s  no events — skipping (scorer defaults to 0.5)", iso2)
            continue

        ec = float(d["event_count"])
        e_score = _log_normalise(ec, max_events, _FALLBACK_MAX_EVENTS)
        store.write(FeatureRecord(
            source="ucdp_candidate", entity=iso2, signal_type="ucdp_candidate_events",
            value=round(e_score, 4),
            metadata={"raw_event_count": ec, "rolling_max": max_events},
            as_of_ts=now,
        ))
        logger.info("CAND %-2s  events=%5.0f (%.3f)", iso2, ec, e_score)
        written += 1

    logger.info("Wrote %d candidate feature records", written)


def main() -> None:
    now = datetime.now(timezone.utc)
    store = FeatureStore()

    ged_df  = download_ged()
    cand_df = download_candidate()

    seed_ged(store, ged_df, now)
    seed_candidate(store, cand_df, now)

    logger.info("Done.")


if __name__ == "__main__":
    main()
