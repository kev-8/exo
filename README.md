# exo

**[exo.dosi.io](https://exo.dosi.io)**

A geopolitical intelligence platform that produces a real-time country-level risk index.

---

## What it does

exo ingests news events, conflict data, economic indicators, trade flows, export-control lists, digital infrastructure and market data on a rolling schedule. It normalises all signals into a shared feature store, then computes a weighted composite risk score across six dimensions for each tracked country.

Each dimension is decomposed into three tiers — **Structural** (15 year baseline), **Short-term** (1–2 year trends), and **Acute** (7–30 day signals) — which blend into the dimension score using configurable per-dimension weights. The composite and all three tier-level aggregates are persisted with every snapshot.

The risk index is updated every 6 hours across **30 tracked countries**. A React frontend serves a live 3D globe with per-country risk cards, trade flow arcs, and a real-time signal feed.

---

## Data sources

| Source | Cadence | Signal |
|---|---|---|
| GDELT | 15 min | News sentiment and event magnitude by country |
| Kalshi | 15 min | Market prices and volume |
| Finnhub | 30 min | Financial news sentiment (VADER, general/forex/merger) |
| Polymarket | 1 hour | Active market prices filtered by geopolitical tags |
| Google Trends | 2 hours | Search volume for geopolitical keywords |
| FRED | 12 hours | US macro indicators (unemployment, CPI, yield curve, consumer sentiment) |
| World Bank | Daily | GDP growth, debt, unemployment, energy imports, trade openness, gross savings, electricity access |
| EIA | Daily | Crude oil (WTI) and natural gas (Henry Hub) prices |
| Export controls | Daily | Entity counts on the US Entity List (trade.gov), Japan's End User List and China's ECL/UEL (OpenSanctions) |
| WGI | Weekly Sep–Dec, monthly otherwise, and at startup | Worldwide Governance Indicators from the World Bank's annual release file |
| UNGA Votes | Weekly | Ideal point estimates (Harvard Dataverse) — policy predictability proxy |
| OFAC | Weekly | SDN entity count per country, normalised via rolling max |
| WITS | Weekly | Trade concentration with the US/EU and secondary sanctions exposure |
| UCDP GED | Weekly | Georeferenced historical conflict events and fatalities (from 2010) |
| UCDP Candidate | Weekly | Near real-time conflict events for short-term conflict tracking |
| UN Comtrade | Monthly | AI hardware import sources, supplier chokepoints, dependence on Chinese export-controlled minerals |
| PeeringDB + Submarine Cable Map | Monthly | Internet exchanges, data-centre facilities and submarine cables per country |

The export-control, Comtrade, and infrastructure jobs also run at startup while the store holds none of their data, so a fresh deploy fills them the same day.

> **Note:** the World Bank removed the WGI indicators (`PV.EST`, `VA.EST`, `RL.EST`, `CC.EST`) from its v2 API in May 2026. `ingestion/wgi.py` reads the annual WGI release file instead and writes each new edition once per country.

> **UCDP:** UCDP data comes from the public bulk files via `scripts/seed_ucdp_bulk.py`. Its output is committed as `src/exo/ingestion/ucdp_seed.json` and written at startup for any country and signal the store holds nothing for. A country with no recorded events scores 0, except North Korea, where UCDP's silence reflects no reporting, so it keeps the neutral 0.5.

---

## Risk index

Six dimensions, each scored 0–1 (higher = higher risk). Each dimension exposes its tier scores alongside its blended score.

| Dimension | Weight | Structural | Short-term | Acute |
|---|---|---|---|---|
| Political stability | 22.5% | WGI Political Stability, Voice & Accountability | — | — |
| Conflict intensity | 22.5% | UCDP GED (historical fatalities) | UCDP Candidate (recent events) | GDELT event magnitude |
| Policy predictability | 18% | UNGA ideal point variance (10-year window) | — | — |
| Sanctions risk | 13.5% | Trade openness (World Bank), trade concentration and secondary exposure (WITS) | OFAC SDN entity count | GDELT news sentiment, Google Trends sanctions search volume |
| Economic stress | 13.5% | World Bank macro, WGI rule of law and corruption | FRED composite, EIA energy prices weighted by import dependency | Finnhub news sentiment (VADER), Polymarket near-term markets |
| AI exposure | 10% | Hardware, supplier chokepoint, digital access (Comtrade, PeeringDB, cable map, World Bank) | Export-control lists | — |

The composite score is the weighted sum of dimension scores. Each of `structural_score`, `short_term_score`, and `acute_score` is also stored per snapshot as a weighted average of the corresponding tier across all dimensions.

Benchmarks (`risk_index/validation.py`): V-Dem liberal democracy index and the Fragile States Index. On 2026-10-23 the composite correlated at r = 0.703 with V-Dem (2025) and 0.764 with the FSI (2023), across 30 and 29 countries.

### AI Exposure

Exposure to AI supply-chain chokepoints and export controls.

| Component | Tier | Built from |
|---|---|---|
| Hardware | structural | Mean of: share of computer, telecom and chip imports (HS 8471/8517/8542) from China, Hong Kong and Taiwan; and dependence on Chinese supply of China's export-controlled minerals at a fixed 2022 baseline |
| Chokepoint | structural | The country's largest share, as an origin, of 12 large importers' imports of a critical mineral, lithium, refined copper, nickel or chips (50% scores 1) |
| Access | structural | Mean of: scarcity of submarine cables, internet exchanges and data-centre facilities; and the share of the population without electricity |
| Export controls | short-term | `log1p(US Entity List + Japan EUL + 0.5 × China ECL/UEL)`, scaled by the same sum of each list's largest country count |

`ai_exposure = 0.6 × mean(available structural components) + 0.4 × export controls`. A missing component is left out of the mean rather than set to 0.5, and coverage (for example "4/4") is stored in the dimension's `metadata`.

**Rollback.** Set `AI_EXPOSURE_WEIGHT` to `0.0` in `config.py`. The score is still computed and stored, but adds nothing to the composite or tier scores. A scheduler job every Monday at 12:00 UTC logs a warning for any country whose AI score moved more than 0.05 in 7 days, with the inputs before and after, so a real event can be told apart from an unstable signal.

**Licences.** The Japan and China lists come from OpenSanctions (CC BY-NC 4.0, non-commercial use only).

### Sanctions trade signals (WITS)

- `trade_concentration`: share of a country's trade with the US and the EU, leaving out its own bloc (an EU member's trade with the EU, the US's with itself).
- `secondary_exposure`: share of a country's trade with the countries under broad OFAC programmes (Russia, Iran, North Korea, Cuba, Syria, Belarus, Venezuela); a 10% share scores 1. Partners are chosen by programme, not by SDN entry counts, which also reflect front companies registered in hubs such as the UAE and UK.

North Korea, Taiwan, Haiti and Guinea report no trade to WITS and keep the neutral default.

### Signal normalisation

Most signals are stored in **[0, 1]**, where 1.0 represents maximum risk. Sentiment signals (GDELT tone, Finnhub VADER) are transformed at ingest time: `risk_score = (1.0 − raw_sentiment) / 2`. Raw values are preserved in record metadata. The AI-exposure sources keep raw counts and shares in metadata so the scorer can re-weight them without re-ingesting. World Bank macro indicators are stored as raw values and normalised in the scorer.

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+ (frontend build)
- Redis (hot cache for latest feature records)

### Install

```bash
pip install -e ".[dev]"
```

### Build the frontend

```bash
cd ui && npm install && npm run build
```

### Environment variables

```bash
# Required for live data
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY=...
FRED_API_KEY=...
FINNHUB_API_KEY=...
EIA_API_KEY=...

# Optional overrides
REDIS_URL=redis://localhost:6379/0
DATA_DIR=/data               # Path to persistent data directory
ALLOWED_ORIGINS=https://your-app.railway.app  # Comma-separated; default allows localhost dev servers
EXO_DISABLE_SCHEDULER=1      # Serve the API only, with no ingestion
```

GDELT, OFAC, UCDP bulk files, UNGA, WGI, WITS, World Bank, UN Comtrade, the trade.gov screening list, OpenSanctions, PeeringDB and the Submarine Cable Map do not require API keys.

---

## Running

### Full pipeline + API server

```bash
# Start the API (serves frontend + runs the scheduler via lifespan)
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Or run the pipeline without the API
python main.py
```

Locally, the `com.exo.ingestion` launchd service runs the pipeline via `scripts/run_ingestion_campaign.sh` against `DATA_DIR=<repo>/data`.

### Development (frontend hot-reload + API)

```bash
cd ui && npm run dev   # API in serve-only mode against <repo>/data, plus Vite
cd ui && npm run ui    # Vite only, proxying /api to localhost:8000 (or VITE_API_TARGET)
```

The dev API runs with `EXO_DISABLE_SCHEDULER=1`, so it never starts a second scheduler next to the launchd pipeline.

### Single ingestor

```python
import asyncio
from exo.ingestion.gdelt import GDELTIngestor

async def main():
    records = await GDELTIngestor().run()
    print(f"{len(records)} records written")

asyncio.run(main())
```

### Update the risk index

```python
from exo.risk_index.engine import RiskIndexEngine

engine = RiskIndexEngine()
engine.update_all()
```

### Query country history

```python
from exo.store.index_store import IndexStore

store = IndexStore()
history = store.get_history("UA", limit=90)
for snap in history:
    print(
        f"{snap.as_of_ts.date()}  "
        f"composite={snap.composite_score:.3f}  "
        f"structural={snap.structural_score:.3f}  "
        f"short_term={snap.short_term_score:.3f}  "
        f"acute={snap.acute_score:.3f}"
    )
```

---

## Tests

```bash
pytest tests/ -v
```

Covers: schema validation, feature store round-trip, point-in-time query enforcement, ingestor normalisation and fetch parsing (fixture responses, no network), country-code maps for every source, risk index dimension scoring (all tiers), composite weighting, the AI-exposure weight and rollback, and scheduler job registration.

---

## Deployment (Railway)

The app runs as a single service on Railway: one process runs both the FastAPI server and the APScheduler pipeline via a FastAPI lifespan handler (`--workers 1`, so jobs never fire twice). Kalshi jobs are excluded there.

**Services required:**
- Web service (Python 3.11 + Node 20) — builds frontend, serves API + static files
- Redis add-on — injected as `REDIS_URL` automatically
- Volume — mount at `/data`, set `DATA_DIR=/data`

Railway builds with Railpack, which ignores `nixpacks.toml`: build and start commands live in `railway.toml`. Set all env vars listed above in the Railway dashboard before first deploy.

**Data on first deploy:** Most sources fill on their first scheduled run. WGI, export controls, Comtrade and digital infrastructure run at startup while empty, UCDP is written from the committed seed file, and the first risk update runs 2 minutes after boot. `scripts/export_data.sh` exports a local data tarball if a full copy is ever needed.

---

## Project structure

```
exo/
├── main.py                    # Pipeline-only entry point (no API)
├── railway.toml               # Railway build + deploy config (Railpack)
│
├── src/exo/
│   ├── config.py              # Env vars, paths, staleness thresholds, dimension weights
│   ├── models.py              # Shared dataclasses (TierScore, DimensionScore, RiskIndexSnapshot)
│   ├── event_bus.py           # Async pub/sub
│   ├── scheduler.py           # APScheduler — ingestion, index and AI stability jobs
│   ├── observability.py       # Structured logging and metrics
│   │
│   ├── ingestion/
│   │   ├── gdelt.py           #   GDELT news sentiment + magnitude (15 min)
│   │   ├── kalshi.py          #   Kalshi prediction markets (15 min)
│   │   ├── kalshi_resolution.py # Kalshi settlement archiver (6 hr)
│   │   ├── finnhub.py         #   Finnhub financial news via VADER (30 min)
│   │   ├── polymarket.py      #   Polymarket prediction markets (1 hr)
│   │   ├── google_trends.py   #   Google Trends search volume (2 hr)
│   │   ├── fred.py            #   FRED macro indicators (12 hr)
│   │   ├── world_bank.py      #   World Bank macro, trade openness, electricity access (daily)
│   │   ├── eia.py             #   EIA crude oil and gas prices (daily)
│   │   ├── export_controls.py #   US / Japan / China export-control lists (daily)
│   │   ├── wgi.py             #   WGI annual release (weekly Sep–Dec)
│   │   ├── unga_votes.py      #   UNGA ideal point estimates (weekly)
│   │   ├── ofac.py            #   OFAC SDN list entity counts (weekly)
│   │   ├── wits.py            #   WITS trade concentration + secondary exposure (weekly)
│   │   ├── ucdp.py            #   UCDP GED + Candidate conflict data (weekly) + startup seed
│   │   ├── ucdp_seed.json     #   Committed UCDP bulk-seed records
│   │   ├── comtrade.py        #   UN Comtrade AI hardware, chokepoint and mineral signals (monthly)
│   │   └── digital_infra.py   #   Submarine cables, IXPs and facilities (monthly)
│   │
│   ├── store/
│   │   ├── feature_store.py   # DuckDB + Parquet + Redis hot cache
│   │   └── index_store.py     # Risk index snapshot persistence (tier scores, dimension metadata)
│   │
│   └── risk_index/
│       ├── dimensions.py      # Six-dimension scorer with structural/short-term/acute tiers
│       ├── engine.py          # RiskIndexEngine — composite + per-tier aggregation
│       └── validation.py      # Benchmark correlations + AI exposure stability check
│
├── api/
│   ├── app.py                 # FastAPI app with lifespan scheduler + static file serving
│   └── routes/                # /api/countries, /api/risk, /api/trade, /api/signals
│
├── ui/                        # React + Vite + Tailwind frontend
│   └── src/
│       ├── App.jsx
│       ├── components/        # Globe, RiskCard, TradePanel, SignalFeed, ShowcaseCard, CountrySelector
│       ├── data/              # Static fallback data (showcaseData.js)
│       └── lib/               # api.js, riskColor.js
│
└── scripts/
    ├── export_data.sh         # Export a local data tarball
    ├── seed_ucdp_bulk.py      # Seed UCDP GED/Candidate from public bulk downloads
    ├── compact_features.py    # Compact legacy one-file-per-record parquet partitions
    └── run_ingestion_campaign.sh # Local launchd pipeline entry point
```
