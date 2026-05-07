"""Temporary admin/debug endpoints.

DELETE THIS FILE after diagnosis is complete.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import duckdb
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from exo import config

router = APIRouter()


@router.delete("/admin/dotfiles", include_in_schema=False)
def delete_dotfiles():
    """Delete macOS ._* resource fork files from the data directory."""
    deleted = []
    for p in config.DATA_DIR.rglob("._*"):
        p.unlink()
        deleted.append(str(p))
    return JSONResponse({"deleted": len(deleted), "files": deleted[:20]})


@router.get("/admin/debug", include_in_schema=False)
def debug():
    """Diagnose data directory and DuckDB reads."""
    data_dir = config.DATA_DIR
    features_dir = config.FEATURES_DIR
    risk_dir = config.RISK_INDEX_DIR

    # Directory existence + file counts
    def dir_info(p: Path):
        if not p.exists():
            return {"exists": False}
        files = list(p.rglob("*.parquet"))
        return {"exists": True, "parquet_count": len(files), "sample": str(files[0]) if files else None}

    # Try a DuckDB read on each source
    duckdb_results = {}
    for source in ["gdelt", "world_bank", "fred"]:
        pattern = str(features_dir / f"source={source}" / "date=*" / "*.parquet")
        try:
            db = duckdb.connect(":memory:")
            df = db.execute(
                f"SELECT DISTINCT entity FROM read_parquet('{pattern}', union_by_name=true) LIMIT 5"
            ).fetchdf()
            duckdb_results[source] = {"ok": True, "entities": df["entity"].tolist()}
        except Exception as exc:
            duckdb_results[source] = {"ok": False, "error": str(exc)}

    # Try risk index read for US
    risk_pattern = str(risk_dir / "country=US" / "date=*" / "*.parquet")
    try:
        db = duckdb.connect(":memory:")
        df = db.execute(f"SELECT count(*) as n FROM read_parquet('{risk_pattern}', union_by_name=true)").fetchdf()
        risk_result = {"ok": True, "rows": int(df["n"][0])}
    except Exception as exc:
        risk_result = {"ok": False, "error": str(exc)}

    return JSONResponse({
        "data_dir":    str(data_dir),
        "features_dir": dir_info(features_dir),
        "risk_dir":    dir_info(risk_dir),
        "duckdb":      duckdb_results,
        "risk_us":     risk_result,
    })
