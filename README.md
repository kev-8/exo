# exo

A geopolitical risk intelligence platform that produces a real-time country-level risk index.

---

## What it does

exo ingests news events, conflict data, economic indicators, social sentiment, and market data on a rolling schedule. It normalises all signals into a shared feature store, then computes a weighted composite risk score across five dimensions for each tracked country.

The risk index is updated every 6 hours and validated against established benchmarks (ICRG, V-Dem).

---

## Data sources

| Source | Cadence | Signal |
|---|---|---|
| GDELT | 15 min | News sentiment and event magnitude by country |
| Kalshi | 15 min + WebSocket | Market prices and volume |
| Finnhub | 30 min | Financial news sentiment (FinBERT, general/forex/merger) |
| Polymarket | 1 hour | Active market prices filtered by geopolitical tags |
| Google Trends | 2 hours | Search volume for geopolitical keywords |
| FRED | 12 hours | US macro indicators (unemployment, CPI, yield curve) |
| World Bank | Daily | GDP growth, debt, unemployment, WGI governance indicators, energy imports |
| EIA | Daily | Crude oil (WTI) and natural gas (Henry Hub) prices |
| UNGA Votes | Weekly | Ideal point estimates (Harvard Dataverse) — policy predictability proxy |

---

## Risk index

Five dimensions, each scored 0–1 (higher = higher risk):

| Dimension | Weight | Primary sources |
|---|---|---|
| Political stability | 25% | WGI Political Stability, Voice & Accountability |
| Conflict intensity | 25% | GDELT news magnitude |
| Policy predictability | 20% | UNGA ideal point variance (last 10 years, Voeten dataset) |
| Sanctions risk | 15% | GDELT news sentiment tone, Google Trends sanctions search volume |
| Economic stress | 15% | FRED composite, World Bank macro + WGI governance, EIA energy prices weighted by import dependency, Finnhub sentiment |

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
├── ingestion/             # 9 data source ingestors (gdelt, kalshi, finnhub, polymarket,
│                          #  google_trends, fred, world_bank, eia, unga_votes)
├── store/
│   ├── feature_store.py   # DuckDB + Parquet + Redis hot cache
│   └── index_store.py     # Risk index persistence
└── risk_index/
    ├── dimensions.py      # Five-dimension scorer
    ├── engine.py          # RiskIndexEngine
    └── validation.py      # ICRG / V-Dem correlation checks
```
