# exo

A geopolitical risk intelligence platform that produces a real-time country-level risk index.

---

## What it does

exo ingests news events, conflict data, economic indicators, social sentiment, polling averages, and market data on a rolling schedule. It normalises all signals into a shared feature store, then computes a weighted composite risk score across five dimensions for each tracked country.

The risk index is updated every 6 hours and validated against established benchmarks (ICRG, V-Dem).

---

## Data sources

| Source | Cadence | Signal |
|---|---|---|
| GDELT | 15 min | News sentiment and event magnitude by country |
| Kalshi | 15 min + WebSocket | Market prices and volume |
| Finnhub | 30 min | Financial news sentiment |
| Reddit | 1 hour | Subreddit sentiment (geopolitics, worldnews, politics) |
| Polymarket | 1 hour | Cross-market price data |
| Google Trends | 2 hours | Search volume for geopolitical keywords |
| FiveThirtyEight | 6 hours | Polling averages |
| FRED | 12 hours | US macro indicators (unemployment, CPI, yield curve) |
| World Bank | Daily | GDP growth, debt, unemployment by country |
| EIA | Daily | Crude oil and natural gas prices |
| ACLED | Weekly | Armed conflict events and fatality rates |

---

## Risk index

Five dimensions, each scored 0–1 (higher = higher risk):

| Dimension | Weight | Primary sources |
|---|---|---|
| Political stability | 25% | Polling averages, Reddit political sentiment |
| Conflict intensity | 25% | ACLED events + fatalities, GDELT magnitude |
| Policy predictability | 20% | Variance of forward-looking estimates |
| Sanctions risk | 15% | GDELT negative tone, Google Trends |
| Economic stress | 15% | FRED composite, World Bank indicators |

The composite score is the weighted sum. Validated against ICRG (Pearson r target > 0.65) and V-Dem (secondary benchmark).

---

## Setup

### Prerequisites

- Python 3.11+
- Redis (hot cache for latest feature records)

### Install

```bash
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Fill in your API keys
```

Minimum for core functionality:

```
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=secrets/kalshi_private_key.pem
```

All other sources degrade gracefully if their key is missing.

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
    print(f"{snap.as_of_ts.date()}  composite={snap.composite_score:.3f}")
```

---

## Tests

```bash
pytest tests/ -v
```

Covers: schema validation, feature store round-trip, point-in-time query enforcement, risk index isolation and scoring.

---

## Project structure

```
exo/
├── config.py              # Env vars and constants
├── models.py              # Shared dataclasses
├── event_bus.py           # Async pub/sub
├── scheduler.py           # APScheduler — all ingestion jobs
├── observability.py       # Structured logging and metrics
│
├── ingestion/             # 11 data source ingestors
├── store/
│   ├── feature_store.py   # DuckDB + Parquet + Redis hot cache
│   └── index_store.py     # Risk index persistence
└── risk_index/
    ├── dimensions.py      # Five-dimension scorer
    ├── engine.py          # RiskIndexEngine
    └── validation.py      # ICRG / V-Dem correlation checks
```
