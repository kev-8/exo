# exo

A geopolitical risk intelligence platform that produces a real-time country-level risk index.

---

## What it does

exo ingests news events, conflict data, economic indicators, social sentiment, and market data on a rolling schedule. It normalises all signals into a shared feature store, then computes a weighted composite risk score across five dimensions for each tracked country.

Each dimension is decomposed into three tiers — **Structural** (annual baselines), **Short-term** (1–2 year trends), and **Acute** (7–30 day signals) — which blend into the dimension score using configurable per-dimension weights. The composite and all three tier-level aggregates are persisted with every snapshot.

The risk index is updated every 6 hours and validated against established benchmarks (ICRG, V-Dem).

---

## Data sources

| Source | Cadence | Signal |
|---|---|---|
| GDELT | 15 min | News sentiment by country |
| Kalshi | 15 min + WebSocket | Market prices and volume |
| Finnhub | 30 min | Financial news sentiment (FinBERT, general/forex/merger) |
| Polymarket | 1 hour | Active market prices filtered by geopolitical tags |
| Google Trends | 2 hours | Search volume for geopolitical keywords |
| FRED | 12 hours | US macro indicators (unemployment, CPI, yield curve) |
| World Bank | Daily | GDP growth, debt, unemployment, WGI governance indicators, energy imports, trade openness |
| EIA | Daily | Crude oil (WTI) and natural gas (Henry Hub) prices |
| UNGA Votes | Weekly | Ideal point estimates (Harvard Dataverse) — policy predictability proxy |
| OFAC | Weekly | SDN entity count per country, normalised via rolling max |
| WITS | Weekly | Bilateral trade concentration (US/EU dependency) and secondary sanctions exposure |
| UCDP GED | Weekly | Georeferenced historical conflict events and fatalities |
| UCDP Candidate | Weekly | Candidate (near real-time) conflict events for short-term conflict tracking |

---

## Risk index

Five dimensions, each scored 0–1 (higher = higher risk). Each dimension exposes three tier scores alongside its blended score.

| Dimension | Weight | Structural | Short-term | Acute |
|---|---|---|---|---|
| Political stability | 25% | WGI Political Stability, Voice & Accountability | — | — |
| Conflict intensity | 25% | UCDP GED (historical fatalities) | UCDP Candidate (recent events) | GDELT event magnitude |
| Policy predictability | 20% | UNGA ideal point variance (10-year window) | — | — |
| Sanctions risk | 15% | Trade openness (World Bank), trade concentration (WITS), secondary exposure (WITS × OFAC) | OFAC SDN entity count | GDELT news sentiment, Google Trends sanctions search volume |
| Economic stress | 15% | World Bank macro + WGI governance | FRED composite, EIA energy prices weighted by import dependency | Finnhub news sentiment (FinBERT) |

The composite score is the weighted sum of dimension scores. Each of `structural_score`, `short_term_score`, and `acute_score` is also stored per snapshot as a weighted average of the corresponding tier across all dimensions.

Validated against ICRG (Pearson r target > 0.65) and V-Dem (secondary benchmark).

### Secondary sanctions exposure

The `secondary_exposure` signal quantifies how much of a country's trade flows to nations already subject to comprehensive OFAC sanctions. The set of sanctioned trade partners is resolved dynamically at ingestion time from live OFAC SDN data. Any country whose normalised SDN score exceeds `SDN_BILATERAL_THRESHOLD` (default 0.05) is included as a partner in the bilateral trade query. OFAC runs 30 minutes before WITS each week to ensure SDN data is fresh.

### Signal normalisation

All ingestors emit values in **[0, 1]** where 1.0 represents maximum risk. Sentiment signals (GDELT tone, Finnhub FinBERT) are transformed at ingest time: `risk_score = (1.0 − raw_sentiment) / 2`. Raw values are preserved in record metadata.

---

## Setup

### Prerequisites

- Python 3.11+
- Redis (hot cache for latest feature records)

### Install

```bash
pip install -e ".[dev]"
```

### Environment variables

```bash
# Required for live data
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY=...      
FRED_API_KEY=...
FINNHUB_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
EIA_API_KEY=...

# Optional overrides
REDIS_URL=redis://localhost:6379/0
SDN_BILATERAL_THRESHOLD=0.05   # min normalised SDN score to include a country as a sanctions partner
```

GDELT, OFAC SDN, UCDP, UNGA, WITS, and World Bank do not require API keys.

---

## Running

### Full pipeline

```python
# main.py
import asyncio
from exo.scheduler import ExoScheduler

async def main():
    scheduler = ExoScheduler()
    await scheduler.start()
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        await scheduler.stop()

asyncio.run(main())
```

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

Covers: schema validation, feature store round-trip, point-in-time query enforcement, ingestor normalisation, risk index dimension scoring (all tiers), and composite weighting.

---

## Project structure

```
exo/
├── config.py              # Env vars, staleness thresholds, dimension tier weights
├── models.py              # Shared dataclasses (TierScore, DimensionScore, RiskIndexSnapshot)
├── event_bus.py           # Async pub/sub
├── scheduler.py           # APScheduler — all ingestion and index jobs
├── observability.py       # Structured logging and metrics
│
├── ingestion/             # 12 data source ingestors:
│   ├── gdelt.py           #   GDELT news sentiment (15 min)
│   ├── kalshi.py          #   Kalshi prediction markets (15 min + WS)
│   ├── finnhub.py         #   Finnhub financial news via FinBERT (30 min)
│   ├── polymarket.py      #   Polymarket prediction markets (1 hr)
│   ├── google_trends.py   #   Google Trends search volume (2 hr)
│   ├── fred.py            #   FRED macro indicators (12 hr)
│   ├── world_bank.py      #   World Bank WGI + macro + trade openness (daily)
│   ├── eia.py             #   EIA crude oil and gas prices (daily)
│   ├── unga_votes.py      #   UNGA ideal point estimates (weekly)
│   ├── ofac.py            #   OFAC SDN list entity counts (weekly)
│   ├── wits.py            #   WITS bilateral trade + secondary exposure (weekly)
│   ├── ucdp.py            #   UCDP GED + Candidate conflict data (weekly)
│
├── store/
│   ├── feature_store.py   # DuckDB + Parquet + Redis hot cache
│   └── index_store.py     # Risk index snapshot persistence (with tier scores)
│
└── risk_index/
    ├── dimensions.py      # Five-dimension scorer with structural/short-term/acute tiers
    ├── engine.py          # RiskIndexEngine — composite + per-tier aggregation
    └── validation.py      # ICRG / V-Dem correlation checks
```
