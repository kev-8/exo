"""DuckDB-backed append-only feature store with Redis hot cache.

Write path:
    write(record) → Parquet partition at data/features/source={src}/date={date}/

Read path:
    read(query)           → point-in-time list of FeatureRecords
    get_latest(...)       → most-recent record within max_age window
    get_for_ticker(...)   → all records associated with a market identifier

Redis cache:
    On every write, the latest record per (source, entity, signal_type) is
    serialised to JSON and SET with a TTL equal to 2× staleness threshold.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from exo import config
from exo.models import FeatureQuery, FeatureRecord

logger = logging.getLogger(__name__)

_REDIS_AVAILABLE = False
try:
    import redis

    _REDIS_AVAILABLE = True
except ImportError:
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FeatureStore:
    """Append-only feature store backed by DuckDB + Parquet partitions.

    Parameters
    ----------
    data_dir:
        Root path for Parquet partitions (default: ``config.FEATURES_DIR``).
    redis_url:
        Redis connection string.  Pass ``None`` to disable caching.
    backtest_mode:
        When ``True``, every read must supply ``as_of_ts`` or a
        :class:`ValueError` is raised.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        redis_url: str | None = None,
        backtest_mode: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir or config.FEATURES_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backtest_mode = backtest_mode

        # Redis
        self._redis: Optional["redis.Redis"] = None  # type: ignore[name-defined]
        if _REDIS_AVAILABLE and (redis_url or config.REDIS_URL):
            try:
                import redis as _redis

                self._redis = _redis.from_url(redis_url or config.REDIS_URL, decode_responses=True)
                self._redis.ping()
                logger.info("Redis cache connected: %s", redis_url or config.REDIS_URL)
            except Exception as exc:
                logger.warning("Redis unavailable (%s); continuing without cache", exc)
                self._redis = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, record: FeatureRecord) -> None:
        """Persist *record* to the Parquet partition and update Redis cache."""
        date_str = record.as_of_ts.strftime("%Y-%m-%d")
        partition = self.data_dir / f"source={record.source}" / f"date={date_str}"
        partition.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            [
                {
                    "record_id": record.record_id,
                    "source": record.source,
                    "entity": record.entity,
                    "signal_type": record.signal_type,
                    "value": record.value,
                    "metadata": json.dumps(record.metadata),
                    "ticker": record.ticker,
                    "as_of_ts": record.as_of_ts.isoformat(),
                    "ingested_at": record.ingested_at.isoformat(),
                }
            ]
        )

        out_path = partition / f"{record.record_id}.parquet"
        df.to_parquet(out_path, index=False)
        logger.debug("Wrote feature record %s → %s", record.record_id, out_path)

        self._cache_set(record)

    def write_batch(self, records: list[FeatureRecord]) -> None:
        """Write all records in one parquet file per (source, date) group.

        Far more efficient than calling write() per-record: one to_parquet()
        call instead of N, which prevents pyarrow allocator bloat under high
        write rates (e.g. Kalshi 600 records/15 min → 1 file/15 min).
        """
        if not records:
            return

        import uuid
        from collections import defaultdict

        groups: dict[tuple[str, str], list[FeatureRecord]] = defaultdict(list)
        for r in records:
            date_str = r.as_of_ts.strftime("%Y-%m-%d")
            groups[(r.source, date_str)].append(r)

        for (source, date_str), group in groups.items():
            partition = self.data_dir / f"source={source}" / f"date={date_str}"
            partition.mkdir(parents=True, exist_ok=True)

            rows = [
                {
                    "record_id": r.record_id,
                    "source": r.source,
                    "entity": r.entity,
                    "signal_type": r.signal_type,
                    "value": r.value,
                    "metadata": json.dumps(r.metadata),
                    "ticker": r.ticker,
                    "as_of_ts": r.as_of_ts.isoformat(),
                    "ingested_at": r.ingested_at.isoformat(),
                }
                for r in group
            ]
            df = pd.DataFrame(rows)
            batch_id = str(uuid.uuid4())[:8]
            out_path = partition / f"batch_{batch_id}.parquet"
            df.to_parquet(out_path, index=False)
            logger.debug("Wrote batch of %d records → %s", len(group), out_path)

            for r in group:
                self._cache_set(r)

    # ------------------------------------------------------------------
    # Redis cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, record: FeatureRecord) -> str:
        return f"exo:feature:{record.source}:{record.entity}:{record.signal_type}"

    def _cache_set(self, record: FeatureRecord) -> None:
        if self._redis is None:
            return
        key = self._cache_key(record)
        ttl = int(config.STALENESS_THRESHOLDS.get(record.source, 3600) * 2)
        payload = {
            "record_id": record.record_id,
            "source": record.source,
            "entity": record.entity,
            "signal_type": record.signal_type,
            "value": record.value,
            "metadata": record.metadata,
            "ticker": record.ticker,
            "as_of_ts": record.as_of_ts.isoformat(),
            "ingested_at": record.ingested_at.isoformat(),
        }
        try:
            self._redis.setex(key, ttl, json.dumps(payload))
        except Exception as exc:
            logger.warning("Redis SET failed: %s", exc)

    def _cache_get(self, source: str, entity: str, signal_type: str) -> FeatureRecord | None:
        if self._redis is None:
            return None
        key = f"exo:feature:{source}:{entity}:{signal_type}"
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            d = json.loads(raw)
            return FeatureRecord(
                record_id=d["record_id"],
                source=d["source"],
                entity=d["entity"],
                signal_type=d["signal_type"],
                value=d["value"],
                metadata=d["metadata"],
                ticker=d.get("ticker"),
                as_of_ts=datetime.fromisoformat(d["as_of_ts"]),
                ingested_at=datetime.fromisoformat(d["ingested_at"]),
            )
        except Exception as exc:
            logger.warning("Redis GET failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def has_recent_data(self, source: str, threshold_sec: float) -> bool:
        """Return True if any parquet file for *source* was written within *threshold_sec*.

        Uses filesystem mtime — no DuckDB, no parquet loading.
        Safe to call concurrently from many threads.
        """
        import time
        cutoff = time.time() - threshold_sec
        src_dir = self.data_dir / f"source={source}"
        if not src_dir.exists():
            return False
        date_dirs = sorted(
            [p for p in src_dir.iterdir() if p.name.startswith("date=")],
            reverse=True,
        )
        for date_dir in date_dirs[:3]:
            for p in date_dir.iterdir():
                if p.suffix == ".parquet" and not p.name.startswith("._"):
                    try:
                        if p.stat().st_mtime >= cutoff:
                            return True
                    except OSError:
                        pass
        return False

    def _parquet_files(self, source: str | None = None) -> list[str]:
        src_part = f"source={source}" if source else "source=*"
        return [
            str(p)
            for p in self.data_dir.glob(f"{src_part}/date=*/*.parquet")
            if not p.name.startswith("._")
        ]

    def _load_df(self, source: str | None = None) -> pd.DataFrame | None:
        files = self._parquet_files(source)
        if not files:
            return None
        _db = duckdb.connect(":memory:")
        try:
            result = _db.execute(
                "SELECT * FROM read_parquet($1, union_by_name=true)", [files]
            ).fetchdf()
            return result if not result.empty else None
        except Exception as exc:
            logger.warning("Parquet read failed: %s", exc)
            return None
        finally:
            _db.close()

    def read(self, query: FeatureQuery) -> list[FeatureRecord]:
        """Return records matching *query*, honoring point-in-time semantics."""
        if self.backtest_mode and query.as_of_ts is None:
            raise ValueError("Backtest mode requires as_of_ts on every read")

        df = self._load_df(query.source)
        if df is None:
            return []

        df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], format="ISO8601", utc=True)
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], format="ISO8601", utc=True)

        if query.as_of_ts is not None:
            cutoff = pd.Timestamp(query.as_of_ts, tz="UTC") if query.as_of_ts.tzinfo is None else pd.Timestamp(query.as_of_ts)
            df = df[df["as_of_ts"] <= cutoff]

        if query.entity:
            df = df[df["entity"] == query.entity]
        if query.signal_type:
            df = df[df["signal_type"] == query.signal_type]
        if query.ticker:
            df = df[df["ticker"] == query.ticker]
        if query.start_ts:
            start = pd.Timestamp(query.start_ts, tz="UTC") if query.start_ts.tzinfo is None else pd.Timestamp(query.start_ts)
            df = df[df["as_of_ts"] >= start]
        if query.end_ts:
            end = pd.Timestamp(query.end_ts, tz="UTC") if query.end_ts.tzinfo is None else pd.Timestamp(query.end_ts)
            df = df[df["as_of_ts"] <= end]
        if query.max_age_sec is not None:
            cutoff_age = pd.Timestamp(_utcnow()) - pd.Timedelta(seconds=query.max_age_sec)
            df = df[df["as_of_ts"] >= cutoff_age]

        df = df.sort_values("as_of_ts", ascending=False).head(query.limit)
        return [self._row_to_record(row) for _, row in df.iterrows()]

    def get_latest(
        self,
        entity: str,
        signal_type: str,
        source: str | None = None,
        max_age_sec: float | None = None,
        as_of_ts: datetime | None = None,
    ) -> FeatureRecord | None:
        """Return the most recent record for (entity, signal_type).

        Checks Redis cache first when no ``as_of_ts`` is specified.
        """
        if self.backtest_mode and as_of_ts is None:
            raise ValueError("Backtest mode requires as_of_ts on every read")

        if as_of_ts is None and source is not None:
            cached = self._cache_get(source, entity, signal_type)
            if cached is not None:
                if max_age_sec is None:
                    return cached
                age = (_utcnow() - cached.as_of_ts.replace(tzinfo=timezone.utc)).total_seconds()
                if age <= max_age_sec:
                    return cached

        results = self.read(
            FeatureQuery(
                entity=entity,
                signal_type=signal_type,
                source=source,
                as_of_ts=as_of_ts,
                max_age_sec=max_age_sec,
                limit=1,
            )
        )
        return results[0] if results else None

    def get_for_ticker(
        self, ticker: str, as_of_ts: datetime | None = None
    ) -> list[FeatureRecord]:
        """Return all feature records associated with a Kalshi *ticker*."""
        if self.backtest_mode and as_of_ts is None:
            raise ValueError("Backtest mode requires as_of_ts on every read")
        return self.read(FeatureQuery(ticker=ticker, as_of_ts=as_of_ts))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: pd.Series) -> FeatureRecord:
        meta = row.get("metadata", "{}")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        as_of = row["as_of_ts"]
        if hasattr(as_of, "to_pydatetime"):
            as_of = as_of.to_pydatetime()
        ingested = row["ingested_at"]
        if hasattr(ingested, "to_pydatetime"):
            ingested = ingested.to_pydatetime()

        # Safely handle pandas NA / NaN for nullable string column
        raw_ticker = row.get("ticker")
        try:
            ticker = str(raw_ticker) if raw_ticker is not None and raw_ticker == raw_ticker else None
        except Exception:
            ticker = None

        return FeatureRecord(
            record_id=str(row.get("record_id", "")),
            source=str(row["source"]),
            entity=str(row["entity"]),
            signal_type=str(row["signal_type"]),
            value=float(row["value"]),
            metadata=meta,
            ticker=ticker,
            as_of_ts=as_of,
            ingested_at=ingested,
        )
