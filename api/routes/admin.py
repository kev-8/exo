"""Temporary admin endpoints for volume maintenance.

Railway's Hobby plan has no in-container shell, so volume inspection and
cleanup have to happen over HTTP. These are destructive and token-guarded —
remove them once the volume is healthy again.

All routes require ?token=<EXO_ADMIN_TOKEN>. If that env var is unset the
routes refuse to run at all, so they're inert unless deliberately enabled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import shutil
import threading
from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from exo import config
from exo.store.compaction import compact_all

logger = logging.getLogger(__name__)

router = APIRouter()

# Every source the risk index or API actually reads. Purging one of these
# degrades the live UI, so it takes an explicit force=true.
_LOAD_BEARING_SOURCES = {
    "eia", "finnhub", "fred", "gdelt", "google_trends", "ofac", "polymarket",
    "ucdp_candidate", "ucdp_ged", "unga_votes", "wits", "world_bank",
}

_SOURCE_RE = re.compile(r"^[a-z0-9_]+$")
_PARTITION_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


def _require_token(token: str | None) -> None:
    expected = os.getenv("EXO_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(503, "EXO_ADMIN_TOKEN not configured — admin routes disabled")
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(403, "invalid admin token")


def _source_dir(source: str):
    """Resolve a source directory, refusing anything that escapes FEATURES_DIR."""
    if not _SOURCE_RE.match(source):
        raise HTTPException(400, f"invalid source name: {source!r}")
    path = (config.FEATURES_DIR / f"source={source}").resolve()
    if not str(path).startswith(str(config.FEATURES_DIR.resolve())):
        raise HTTPException(400, "path traversal rejected")
    return path


@router.get("/admin/storage")
def storage(token: str = Query(default=None)):
    """Disk *and* inode usage, plus per-source file counts.

    Inodes matter more than bytes here: many small parquet files exhaust the
    inode table long before the size cap, which surfaces as ENOSPC while
    df still reports free space.
    """
    _require_token(token)

    st = os.statvfs(config.DATA_DIR)
    blocks_total, blocks_free = st.f_blocks, st.f_bavail
    inodes_total, inodes_free = st.f_files, st.f_favail

    sources = []
    if config.FEATURES_DIR.exists():
        for src in sorted(config.FEATURES_DIR.glob("source=*")):
            if not src.is_dir():
                continue
            files = list(src.rglob("*.parquet"))
            sources.append({
                "source": src.name.replace("source=", ""),
                "files": len(files),
                "bytes": sum(f.stat().st_size for f in files),
                "partitions": sum(1 for p in src.iterdir() if p.is_dir()),
            })
    sources.sort(key=lambda s: s["files"], reverse=True)

    return {
        "data_dir": str(config.DATA_DIR),
        "bytes": {
            "total": blocks_total * st.f_frsize,
            "free": blocks_free * st.f_frsize,
            "used_pct": round(100 * (1 - blocks_free / blocks_total), 1) if blocks_total else None,
        },
        "inodes": {
            "total": inodes_total,
            "free": inodes_free,
            "used_pct": round(100 * (1 - inodes_free / inodes_total), 1) if inodes_total else None,
        },
        "sources": sources,
    }


@router.post("/admin/purge")
def purge(
    source: str = Query(...),
    token: str = Query(default=None),
    dry_run: bool = Query(default=True),
    before: str | None = Query(default=None, description="Only partitions before YYYY-MM-DD"),
    force: bool = Query(default=False, description="Required for load-bearing sources"),
):
    """Delete a source's parquet partitions to reclaim inodes.

    Defaults to dry_run — pass dry_run=false to actually delete.
    """
    _require_token(token)

    if source in _LOAD_BEARING_SOURCES and not force:
        raise HTTPException(
            400,
            f"{source!r} is read by the risk index or API; pass force=true to purge it anyway",
        )

    cutoff = None
    if before:
        try:
            cutoff = date.fromisoformat(before)
        except ValueError:
            raise HTTPException(400, f"invalid date: {before!r} (want YYYY-MM-DD)")

    src_dir = _source_dir(source)
    if not src_dir.exists():
        raise HTTPException(404, f"no such source on this volume: {source}")

    removed_files = removed_bytes = 0
    partitions = []
    for part in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        m = _PARTITION_RE.match(part.name)
        if cutoff and m and date.fromisoformat(m.group(1)) >= cutoff:
            continue
        files = list(part.rglob("*.parquet"))
        nbytes = sum(f.stat().st_size for f in files)
        partitions.append(part.name)
        removed_files += len(files)
        removed_bytes += nbytes
        if not dry_run:
            shutil.rmtree(part)

    if not dry_run:
        logger.warning(
            "admin purge: removed %d files (%d bytes) from source=%s across %d partitions",
            removed_files, removed_bytes, source, len(partitions),
        )

    return {
        "source": source,
        "dry_run": dry_run,
        "partitions": len(partitions),
        "files_removed": removed_files,
        "bytes_removed": removed_bytes,
    }


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

# Compaction is slow and CPU-bound, and the app runs with --workers 1, so it
# must never run on the event loop: a multi-minute block would stall the
# healthcheck and get the container killed. Real runs go to a worker thread
# and are polled via /admin/compact/status.
_job: dict = {"state": "idle"}
_job_lock = threading.Lock()


def _run_compaction(source: str | None) -> None:
    try:
        result = compact_all(config.FEATURES_DIR, source=source, dry_run=False)
        with _job_lock:
            _job.update(
                state="done",
                finished_at=datetime.now(timezone.utc).isoformat(),
                result=result,
            )
        logger.warning(
            "admin compaction finished: removed %d files (%d bytes)",
            result["files_removed"], result["bytes_removed"],
        )
    except Exception as exc:                      # noqa: BLE001 — surfaced via status
        logger.exception("admin compaction failed")
        with _job_lock:
            _job.update(
                state="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
            )


@router.post("/admin/compact")
async def compact(
    source: str | None = Query(default=None),
    token: str = Query(default=None),
    dry_run: bool = Query(default=True),
):
    """Merge legacy per-record parquet files into one file per partition.

    dry_run (the default) reports what would be merged and returns inline.
    A real run is dispatched to a worker thread; poll /admin/compact/status.
    """
    _require_token(token)
    if source is not None:
        _source_dir(source)                      # validates the name

    if dry_run:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: compact_all(config.FEATURES_DIR, source=source, dry_run=True)
        )
        return result

    with _job_lock:
        if _job.get("state") == "running":
            raise HTTPException(409, "a compaction is already running")
        _job.clear()
        _job.update(
            state="running",
            source=source or "*",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    threading.Thread(target=_run_compaction, args=(source,), daemon=True).start()
    return {"state": "running", "source": source or "*", "poll": "/api/admin/compact/status"}


@router.get("/admin/compact/status")
def compact_status(token: str = Query(default=None)):
    _require_token(token)
    with _job_lock:
        return dict(_job)
