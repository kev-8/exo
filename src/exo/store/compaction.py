"""Compact legacy one-file-per-record parquet partitions.

The old `FeatureStore.write()` path wrote a separate UUID-named parquet file
per record. `write_batch()` replaced it (one `batch_*.parquet` per
source/date), but the legacy files remain and dominate both inode usage and
read cost: `_load_df()` globs every file for a source on every read, and
DuckDB's per-file overhead — not row count — is what hurts.

Each partition's legacy files are merged into a single
`compacted_<date>.parquet`, verifying the row count round-trips before
deleting the originals. Existing `batch_*.parquet` files are left untouched.

Shared by `scripts/compact_features.py` (local CLI) and the admin API route
(Railway volume, where there's no shell).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

MEMORY_LIMIT = "2GB"
# Compact in chunks so a partition with 50k+ tiny files doesn't blow memory
# doing the very thing this module exists to avoid.
CHUNK = 2000


# FeatureStore writes these as strings (see write()/write_batch()). A column
# that is *entirely* null in a partition — `ticker` for any source that doesn't
# set one — comes back from DuckDB typed INTEGER, so it would round-trip as
# Int32/<NA> instead of object/None and silently change the schema.
_STRING_COLS = (
    "record_id", "source", "entity", "signal_type",
    "metadata", "ticker", "as_of_ts", "ingested_at",
)


def _restore_schema(df) -> None:
    """Coerce columns back to the dtypes FeatureStore wrote, in place."""
    for col in _STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    if "value" in df.columns:
        import pandas as pd
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")


def legacy_files(partition: Path) -> list[Path]:
    """Legacy per-record files in *partition* — excludes batch/compacted output."""
    return sorted(
        p for p in partition.glob("*.parquet")
        if not p.name.startswith(("batch_", "compacted_", "._"))
    )


def compact_partition(partition: Path, dry_run: bool = False) -> tuple[int, int]:
    """Merge *partition*'s legacy files into one. Returns (files, bytes) removed."""
    files = legacy_files(partition)
    if len(files) < 2:
        return 0, 0

    total_bytes = sum(f.stat().st_size for f in files)
    if dry_run:
        return len(files), total_bytes

    db = duckdb.connect(":memory:")
    db.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    out = partition / f"compacted_{partition.name.replace('date=', '')}.parquet"
    tmp = out.with_suffix(".parquet.tmp")

    try:
        frames = []
        expected = 0
        for i in range(0, len(files), CHUNK):
            chunk = [str(f) for f in files[i:i + CHUNK]]
            df = db.execute(
                "SELECT * FROM read_parquet($1, union_by_name=true)", [chunk]
            ).fetchdf()
            expected += len(df)
            frames.append(df)

        import pandas as pd
        merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        del frames
        _restore_schema(merged)
        merged.to_parquet(tmp, index=False)
        del merged

        # Verify the round-trip before destroying anything.
        actual = db.execute(
            "SELECT COUNT(*) FROM read_parquet($1)", [[str(tmp)]]
        ).fetchone()[0]
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"{partition.name}: row mismatch (expected {expected}, got {actual}) — left untouched"
            )

        tmp.rename(out)
        for f in files:
            f.unlink()
        return len(files), total_bytes
    finally:
        db.close()
        tmp.unlink(missing_ok=True)


def compact_source(source_dir: Path, dry_run: bool = False) -> dict:
    """Compact every partition under *source_dir*. Never raises on a single
    partition — a failure there is recorded and the rest continue."""
    files = nbytes = 0
    partitions = sorted(p for p in source_dir.iterdir() if p.is_dir())
    errors: list[str] = []

    for part in partitions:
        try:
            n, b = compact_partition(part, dry_run)
        except Exception as exc:
            logger.warning("compaction failed for %s: %s", part, exc)
            errors.append(f"{part.name}: {exc}")
            continue
        files += n
        nbytes += b

    return {
        "source": source_dir.name.replace("source=", ""),
        "partitions": len(partitions),
        "files_removed": files,
        "bytes_removed": nbytes,
        "errors": errors,
    }


def compact_all(features_dir: Path, source: str | None = None, dry_run: bool = False) -> dict:
    """Compact one source, or every source under *features_dir*."""
    pattern = f"source={source}" if source else "source=*"
    dirs = sorted(p for p in features_dir.glob(pattern) if p.is_dir())
    results = [compact_source(d, dry_run) for d in dirs]

    return {
        "dry_run": dry_run,
        "sources": results,
        "files_removed": sum(r["files_removed"] for r in results),
        "bytes_removed": sum(r["bytes_removed"] for r in results),
    }
