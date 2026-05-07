"""Risk index store — append-only DuckDB + Parquet.

Written to by RiskIndexEngine only.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from exo import config
from exo.models import DimensionScore, RiskIndexSnapshot, TierScore

logger = logging.getLogger(__name__)


class IndexStore:
    """Append-only store for :class:`~exo.models.RiskIndexSnapshot` objects.

    Parquet is partitioned as::

        data/risk_index/country={cc}/date={date}/
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir or config.RISK_INDEX_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db = duckdb.connect(":memory:")
        self._db.execute("INSTALL parquet; LOAD parquet;")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, snapshot: RiskIndexSnapshot) -> None:
        date_str = snapshot.as_of_ts.strftime("%Y-%m-%d")
        partition = self.data_dir / f"country={snapshot.country}" / f"date={date_str}"
        partition.mkdir(parents=True, exist_ok=True)

        dims = [
            {
                "name": d.name,
                "score": d.score,
                "weight": d.weight,
                "signals": json.dumps(d.contributing_signals),
                "tier_scores": json.dumps({
                    tier: {"score": ts.score, "signals": ts.contributing_signals}
                    for tier, ts in d.tier_scores.items()
                }),
            }
            for d in snapshot.dimensions
        ]
        row = {
            "snapshot_id": snapshot.snapshot_id,
            "country": snapshot.country,
            "composite_score": snapshot.composite_score,
            "structural_score": snapshot.structural_score,
            "short_term_score": snapshot.short_term_score,
            "acute_score": snapshot.acute_score,
            "dimensions": json.dumps(dims),
            "as_of_ts": snapshot.as_of_ts.isoformat(),
            "computed_at": snapshot.computed_at.isoformat(),
        }
        df = pd.DataFrame([row])
        out_path = partition / f"{snapshot.snapshot_id}.parquet"
        df.to_parquet(out_path, index=False)
        logger.debug(
            "Wrote RiskIndexSnapshot %s for %s → %s",
            snapshot.snapshot_id,
            snapshot.country,
            out_path,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(
        self,
        country: str,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        limit: int = 1000,
    ) -> list[RiskIndexSnapshot]:
        files = [
            str(p)
            for p in (self.data_dir / f"country={country}").glob("date=*/*.parquet")
            if not p.name.startswith("._")
        ]
        if not files:
            return []
        try:
            _db = duckdb.connect(":memory:")
            df = _db.execute(
                "SELECT * FROM read_parquet($1, union_by_name=true)", [files]
            ).fetchdf()
        except Exception as exc:
            logger.warning("No index data for %s: %s", country, exc)
            return []

        if df.empty:
            return []

        df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], utc=True)
        if start_ts:
            df = df[df["as_of_ts"] >= pd.Timestamp(start_ts, tz="UTC")]
        if end_ts:
            df = df[df["as_of_ts"] <= pd.Timestamp(end_ts, tz="UTC")]

        df = df.sort_values("as_of_ts", ascending=False).head(limit)
        snapshots = []
        for _, row in df.iterrows():
            dims_raw = json.loads(row["dimensions"])
            dims = []
            for d in dims_raw:
                raw_tiers = json.loads(d.get("tier_scores") or "{}")
                tier_scores = {
                    tier: TierScore(
                        tier=tier,
                        score=v["score"],
                        contributing_signals=v.get("signals", []),
                    )
                    for tier, v in raw_tiers.items()
                }
                dims.append(DimensionScore(
                    name=d["name"],
                    score=d["score"],
                    weight=d["weight"],
                    contributing_signals=json.loads(d["signals"]),
                    tier_scores=tier_scores,
                ))

            # Derive tier scores from dimension tier_scores when the columns
            # are absent (snapshots written before the schema added these fields).
            def _tier_composite(tier: str) -> float:
                total_w = sum(d.weight for d in dims if tier in d.tier_scores)
                if total_w == 0:
                    return 0.5
                return sum(
                    d.tier_scores[tier].score * d.weight
                    for d in dims if tier in d.tier_scores
                ) / total_w

            cols = df.columns.tolist()
            structural  = float(row["structural_score"])  if "structural_score"  in cols else _tier_composite("structural")
            short_term  = float(row["short_term_score"])  if "short_term_score"  in cols else _tier_composite("short_term")
            acute       = float(row["acute_score"])       if "acute_score"       in cols else _tier_composite("acute")

            as_of = row["as_of_ts"]
            if hasattr(as_of, "to_pydatetime"):
                as_of = as_of.to_pydatetime()
            snapshots.append(
                RiskIndexSnapshot(
                    snapshot_id=str(row["snapshot_id"]),
                    country=str(row["country"]),
                    composite_score=float(row["composite_score"]),
                    structural_score=structural,
                    short_term_score=short_term,
                    acute_score=acute,
                    dimensions=dims,
                    as_of_ts=as_of,
                    computed_at=datetime.fromisoformat(str(row["computed_at"])),
                )
            )
        return snapshots

    def get_latest(self, country: str) -> RiskIndexSnapshot | None:
        results = self.get_history(country, limit=1)
        return results[0] if results else None
