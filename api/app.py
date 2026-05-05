"""exo FastAPI application.

Serves risk index data, trade flows, and signal feeds.
In production (Railway), also serves the Vite static build from ui/dist/.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import countries, risk, signals, trade

logger = logging.getLogger(__name__)

app = FastAPI(
    title="exo Geopolitical Intelligence API",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server during development
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

app.include_router(countries.router, prefix="/api")
app.include_router(risk.router,      prefix="/api")
app.include_router(trade.router,     prefix="/api")
app.include_router(signals.router,   prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static frontend — serve Vite build if present (production)
# ---------------------------------------------------------------------------

_UI_DIST = Path(__file__).parent.parent / "ui" / "dist"

if _UI_DIST.exists():
    # Serve static assets under /assets
    app.mount("/assets", StaticFiles(directory=str(_UI_DIST / "assets")), name="assets")

    # Catch-all: serve index.html for all non-API routes (SPA routing)
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        index = _UI_DIST / "index.html"
        return FileResponse(str(index))
else:
    logger.info("ui/dist not found — running in API-only mode (frontend not built)")
