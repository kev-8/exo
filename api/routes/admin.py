"""Temporary admin endpoint for seeding data on Railway.

DELETE THIS FILE after the initial data transfer is complete.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from exo import config

router = APIRouter()


@router.post("/admin/seed", include_in_schema=False)
async def seed_from_url(url: str):
    """Download a tarball from url and extract it into DATA_DIR."""
    data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with tmp_path.open("wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)

        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

        tmp_path.unlink(missing_ok=True)

        counts = {p.name: sum(1 for _ in p.rglob("*.parquet")) for p in data_dir.iterdir() if p.is_dir()}
        return JSONResponse({"status": "ok", "data_dir": str(data_dir), "parquet_counts": counts})

    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/admin/cleanup", include_in_schema=False)
def cleanup_stale_data():
    """Delete the stale data directory at /app/src/exo/data."""
    stale = Path("/app/src/exo/data")
    if not stale.exists():
        return JSONResponse({"status": "nothing to delete", "path": str(stale)})
    shutil.rmtree(stale)
    return JSONResponse({"status": "deleted", "path": str(stale)})
