"""Five-dimension risk scoring per country.

Dimensions:
  1. political_stability   (weight 0.25): WGI Political Stability, Voice & Accountability 
  2. conflict_intensity    (weight 0.25): ACLED events + fatality rate, GDELT news magnitude
  3. policy_predictability (weight 0.20): UNGA ideal point variance (unga_votes)
  4. sanctions_risk        (weight 0.15): GDELT news sentiment tone, Google Trends sanctions search volume
  5. economic_stress       (weight 0.15): FRED composite, World Bank macro + WGI governance indicators,
                                          EIA crude/gas prices weighted by energy import dependency,
                                          Finnhub financial news sentiment
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from exo import config
from exo.models import DimensionScore, FeatureQuery
from exo.store.feature_store import FeatureStore

logger = logging.getLogger(__name__)


def _safe(val: float | None, default: float = 0.5) -> float:
    if val is None or (val != val):  # NaN check
        return default
    return float(val)


class DimensionScorer:
    """Compute dimension scores for a given country at a point in time."""

    def __init__(self, store: FeatureStore) -> None:
        self.store = store

    def score_all(
        self, country: str, as_of_ts: datetime | None = None
    ) -> list[DimensionScore]:
        now = as_of_ts or datetime.now(timezone.utc)
        return [
            self.political_stability(country, now),
            self.conflict_intensity(country, now),
            self.policy_predictability(country, now),
            self.sanctions_risk(country, now),
            self.economic_stress(country, now),
        ]

    # ------------------------------------------------------------------
    # 1. Political stability (weight 0.25)
    # ------------------------------------------------------------------

    def political_stability(self, country: str, now: datetime) -> DimensionScore:
        signals: list[str] = []
        scores: list[float] = []

        # WGI Political Stability (already normalised to [0,1]; invert: higher = more stable = lower risk)
        ps_rec = self.store.get_latest(
            entity=country, signal_type="political_stability",
            source="world_bank", as_of_ts=now,
        )
        if ps_rec:
            instability = 1.0 - _safe(ps_rec.value)
            scores.append(instability)
            signals.append(f"wgi_political_stability={ps_rec.value:.3f}")

        # WGI Voice & Accountability (invert: higher = more accountable = lower risk)
        va_rec = self.store.get_latest(
            entity=country, signal_type="voice_accountability",
            source="world_bank", as_of_ts=now,
        )
        if va_rec:
            instability = 1.0 - _safe(va_rec.value)
            scores.append(instability)
            signals.append(f"wgi_voice_accountability={va_rec.value:.3f}")

        # Reddit political sentiment (US only, commented out pending API approval)
        # reddit_rec = self.store.get_latest(
        #     entity="US_politics", signal_type="reddit_sentiment",
        #     source="reddit", as_of_ts=now,
        # )
        # if reddit_rec:
        #     instability = (1.0 - _safe(reddit_rec.value)) / 2 + 0.5
        #     scores.append(instability)
        #     signals.append(f"reddit_politics={reddit_rec.value:.3f}")

        score = _safe(sum(scores) / len(scores) if scores else None, default=0.5)
        return DimensionScore(
            name="political_stability",
            score=round(score, 4),
            weight=config.DIMENSION_WEIGHTS["political_stability"],
            contributing_signals=signals,
        )

    # ------------------------------------------------------------------
    # 2. Conflict intensity (weight 0.25)
    # ------------------------------------------------------------------

    def conflict_intensity(self, country: str, now: datetime) -> DimensionScore:
        signals: list[str] = []
        scores: list[float] = []

        week_ago = now - timedelta(days=7)
        acled_recs = self.store.read(FeatureQuery(
            entity=country, signal_type="conflict_event", source="acled",
            as_of_ts=now, start_ts=week_ago, limit=200,
        ))
        if acled_recs:
            mean_intensity = sum(r.value for r in acled_recs) / len(acled_recs)
            total_fatalities = sum(r.metadata.get("fatalities", 0) for r in acled_recs)
            normalised_fatalities = min(1.0, total_fatalities / 1000.0)
            combined = (mean_intensity + normalised_fatalities) / 2
            scores.append(combined)
            signals.append(f"acled_events={len(acled_recs)} fatalities={total_fatalities:.0f}")

        gdelt_rec = self.store.get_latest(
            entity=country, signal_type="news_magnitude", source="gdelt", as_of_ts=now,
        )
        if gdelt_rec and gdelt_rec.value > 0:
            scores.append(_safe(gdelt_rec.value))
            signals.append(f"gdelt_magnitude={gdelt_rec.value:.3f}")

        score = _safe(sum(scores) / len(scores) if scores else None, default=0.3)
        return DimensionScore(
            name="conflict_intensity",
            score=round(score, 4),
            weight=config.DIMENSION_WEIGHTS["conflict_intensity"],
            contributing_signals=signals,
        )

    # ------------------------------------------------------------------
    # 3. Policy predictability (weight 0.20)
    # ------------------------------------------------------------------

    def policy_predictability(self, country: str, now: datetime) -> DimensionScore:
        """Variance in UNGA ideal point estimates indicates low policy predictability."""
        signals: list[str] = []

        rec = self.store.get_latest(
            entity=country, signal_type="ideal_point_variance",
            source="unga_votes", as_of_ts=now,
        )

        if rec is None:
            return DimensionScore(
                name="policy_predictability",
                score=0.5,
                weight=config.DIMENSION_WEIGHTS["policy_predictability"],
                contributing_signals=["insufficient_data"],
            )

        score = _safe(rec.value)
        signals.append(
            f"ideal_point_variance={rec.metadata.get('variance', 0):.4f}"
            f" n_years={rec.metadata.get('n_years')}"
            f" latest_year={rec.metadata.get('latest_year')}"
        )

        return DimensionScore(
            name="policy_predictability",
            score=round(score, 4),
            weight=config.DIMENSION_WEIGHTS["policy_predictability"],
            contributing_signals=signals,
        )

    # ------------------------------------------------------------------
    # 4. Sanctions risk (weight 0.15)
    # ------------------------------------------------------------------

    def sanctions_risk(self, country: str, now: datetime) -> DimensionScore:
        signals: list[str] = []
        scores: list[float] = []

        # Negative news tone as a proxy for sanctions discourse intensity
        gdelt_rec = self.store.get_latest(
            entity=country, signal_type="news_sentiment", source="gdelt", as_of_ts=now,
        )
        if gdelt_rec:
            sanctions_score = max(0.0, min(1.0, (1.0 - gdelt_rec.value) / 2))
            scores.append(sanctions_score)
            signals.append(f"gdelt_sentiment={gdelt_rec.value:.3f}")

        gt_rec = self.store.get_latest(
            entity="sanctions", signal_type="search_volume", source="google_trends", as_of_ts=now,
        )
        if gt_rec:
            scores.append(_safe(gt_rec.value))
            signals.append(f"google_trends_sanctions={gt_rec.value:.3f}")

        score = _safe(sum(scores) / len(scores) if scores else None, default=0.3)
        return DimensionScore(
            name="sanctions_risk",
            score=round(score, 4),
            weight=config.DIMENSION_WEIGHTS["sanctions_risk"],
            contributing_signals=signals,
        )

    # ------------------------------------------------------------------
    # 5. Economic stress (weight 0.15)
    # ------------------------------------------------------------------

    def economic_stress(self, country: str, now: datetime) -> DimensionScore:
        signals: list[str] = []
        scores: list[float] = []

        fred_rec = self.store.get_latest(
            entity="US", signal_type="economic_indicator", source="fred", as_of_ts=now,
        )
        if fred_rec:
            scores.append(_safe(fred_rec.value))
            signals.append(f"fred_composite={fred_rec.value:.3f}")

        wb_recs = {
            "gdp_growth": self.store.get_latest(entity=country, signal_type="gdp_growth", source="world_bank", as_of_ts=now),
            "debt_to_gdp": self.store.get_latest(entity=country, signal_type="debt_to_gdp", source="world_bank", as_of_ts=now),
            "unemployment_rate": self.store.get_latest(entity=country, signal_type="unemployment_rate", source="world_bank", as_of_ts=now),
            "rule_of_law": self.store.get_latest(entity=country, signal_type="rule_of_law", source="world_bank", as_of_ts=now),
            "control_of_corruption": self.store.get_latest(entity=country, signal_type="control_of_corruption", source="world_bank", as_of_ts=now),
        }
        for sig, rec in wb_recs.items():
            if rec is not None:
                raw = rec.value
                if sig == "unemployment_rate":
                    normalised = min(1.0, raw / 20.0)
                elif sig == "debt_to_gdp":
                    normalised = min(1.0, raw / 200.0)
                elif sig == "gdp_growth":
                    normalised = max(0.0, min(1.0, (5.0 - raw) / 10.0))
                elif sig in ("rule_of_law", "control_of_corruption"):
                    # Already normalised to [0,1] in ingestor; invert (higher = better governance = lower stress)
                    normalised = 1.0 - raw
                else:
                    normalised = 0.5
                scores.append(normalised)
                signals.append(f"{sig}={raw:.2f}")

        # EIA energy prices — weighted by country energy import dependency
        energy_imports_rec = self.store.get_latest(
            entity=country, signal_type="energy_imports_pct",
            source="world_bank", as_of_ts=now,
        )
        energy_import_weight = _safe(energy_imports_rec.value, default=0.5) if energy_imports_rec else 0.5

        crude_rec = self.store.get_latest(
            entity="WTI", signal_type="crude_oil_price", source="eia", as_of_ts=now,
        )
        if crude_rec:
            # Normalise WTI price: $40 = low stress, $120 = high stress
            crude_normalised = max(0.0, min(1.0, (crude_rec.value - 40) / 80))
            # Importers: high price = stress; exporters: high price = relief (inverted)
            # energy_import_weight near 1.0 = importer, near 0.0 = exporter
            directional = crude_normalised * energy_import_weight + (1.0 - crude_normalised) * (1.0 - energy_import_weight)
            scores.append(directional)
            signals.append(f"eia_crude={crude_rec.value:.2f} import_weight={energy_import_weight:.2f}")

        gas_rec = self.store.get_latest(
            entity="HH", signal_type="natural_gas_price", source="eia", as_of_ts=now,
        )
        if gas_rec:
            # Normalise Henry Hub: $2 = low stress, $8 = high stress
            gas_normalised = max(0.0, min(1.0, (gas_rec.value - 2) / 6))
            directional = gas_normalised * energy_import_weight + (1.0 - gas_normalised) * (1.0 - energy_import_weight)
            scores.append(directional)
            signals.append(f"eia_gas={gas_rec.value:.2f}")

        # Finnhub financial news sentiment (global signal)
        for category in ("general", "forex", "merger"):
            finnhub_rec = self.store.get_latest(
                entity=category, signal_type="news_sentiment", source="finnhub", as_of_ts=now,
            )
            if finnhub_rec:
                # Sentiment in [-1, 1]; invert and shift to [0, 1]: negative news = higher stress
                stress = (1.0 - _safe(finnhub_rec.value)) / 2
                scores.append(stress)
                signals.append(f"finnhub_{category}={finnhub_rec.value:.3f}")

        score = _safe(sum(scores) / len(scores) if scores else None, default=0.5)
        return DimensionScore(
            name="economic_stress",
            score=round(score, 4),
            weight=config.DIMENSION_WEIGHTS["economic_stress"],
            contributing_signals=signals,
        )
