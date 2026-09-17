#!/usr/bin/env python3
"""Compact legacy one-file-per-record parquet partitions.

Thin CLI over `exo.store.compaction`, which the admin API route also uses.
See that module for why this exists and what it guarantees.

Usage:
    python scripts/compact_features.py --dry-run          # report only
    python scripts/compact_features.py                    # all sources
    python scripts/compact_features.py --source polymarket
"""
from __future__ import annotations

import argparse
import sys

from exo import config
from exo.store.compaction import compact_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", help="Only compact this source (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="Report without modifying")
    args = ap.parse_args()

    result = compact_all(config.FEATURES_DIR, source=args.source, dry_run=args.dry_run)
    if not result["sources"]:
        print(f"No sources matched {args.source or '*'} in {config.FEATURES_DIR}", file=sys.stderr)
        sys.exit(1)

    verb = "would merge" if args.dry_run else "merged"
    for src in result["sources"]:
        for err in src["errors"]:
            print(f"  !! {err}", file=sys.stderr)
        if src["files_removed"]:
            print(
                f"{src['source']}: {verb} {src['files_removed']} legacy files "
                f"({src['bytes_removed']/1024/1024:.0f} MB) across {src['partitions']} partitions"
            )

    verb = "Would compact" if args.dry_run else "Compacted"
    print(f"\n{verb} {result['files_removed']} legacy files "
          f"({result['bytes_removed']/1024/1024/1024:.2f} GB total)")


if __name__ == "__main__":
    main()
