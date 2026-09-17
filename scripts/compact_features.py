#!/usr/bin/env python3
"""Compact legacy one-file-per-record parquet partitions.

The old `FeatureStore.write()` path wrote a separate UUID-named parquet file
per record. `write_batch()` replaced it (one `batch_*.parquet` per
source/date), but the legacy files remain and dominate read cost:
`_load_df()` globs every file for a source on every read, and DuckDB's
per-file overhead — not row count — is what hurts. Measured: 1,988
single-record files cost +625MB to read; the same 1,988 rows in one file
cost +2MB.

This merges each partition's legacy files into a single
`compacted_<date>.parquet`, verifying the row count round-trips before
deleting the originals. Existing `batch_*.parquet` files are left untouched.

Usage:
    python scripts/compact_features.py --dry-run          # report only
    python scripts/compact_features.py                    # all sources
    python scripts/compact_features.py --source polymarket
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

FEATURES_DIR = Path("/Users/kevin/Desktop/ds/exo/data/features")
MEMORY_LIMIT = "2GB"
# Compact in chunks so a partition with 50k+ tiny files doesn't blow memory
# doing the very thing this script exists to avoid.
CHUNK = 2000


def legacy_files(partition: Path) -> list[Path]:
    return sorted(
        p for p in partition.glob("*.parquet")
        if not p.name.startswith(("batch_", "compacted_", "._"))
    )


def compact_partition(partition: Path, dry_run: bool) -> tuple[int, int]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", help="Only compact this source (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="Report without modifying")
    args = ap.parse_args()

    pattern = f"source={args.source}" if args.source else "source=*"
    sources = sorted(p for p in FEATURES_DIR.glob(pattern) if p.is_dir())
    if not sources:
        print(f"No sources matched {pattern}", file=sys.stderr)
        sys.exit(1)

    grand_files = grand_bytes = 0
    for src in sources:
        partitions = sorted(p for p in src.iterdir() if p.is_dir())
        src_files = src_bytes = 0
        for part in partitions:
            try:
                n, b = compact_partition(part, args.dry_run)
            except Exception as exc:
                print(f"  !! {part.name}: {exc}", file=sys.stderr)
                continue
            src_files += n
            src_bytes += b
        if src_files:
            verb = "would merge" if args.dry_run else "merged"
            print(f"{src.name}: {verb} {src_files} legacy files ({src_bytes/1024/1024:.0f} MB) across {len(partitions)} partitions")
        grand_files += src_files
        grand_bytes += src_bytes

    verb = "Would compact" if args.dry_run else "Compacted"
    print(f"\n{verb} {grand_files} legacy files ({grand_bytes/1024/1024/1024:.2f} GB total)")


if __name__ == "__main__":
    main()
