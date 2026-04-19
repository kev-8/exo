"""Five-dimension risk scoring per country — three-tier architecture.

Each dimension is scored across three time-horizon tiers:

  structural  — annual/multi-year signals (WGI, UNGA, UCDP GED, trade data)
  short_term  — 1-2 year signals (FRED, EIA, UCDP candidate)
  acute       — 7-30 day signals (GDELT, Google Trends, Polymarket, Finnhub)

Tier sub-scores blend into the dimension score via DIMENSION_TIER_WEIGHTS.
If a tier has no data its TierScore defaults to 0.5 (neutral).

Dimensions and weights:
  1. political_stability   (0.25): structural only — WGI PV.EST, VA.EST
  2. conflict_intensity    (0.25): all three tiers
  3. policy_predictability (0.20): structural only — UNGA ideal point variance
  4. sanctions_risk        (0.15): structural + acute
  5. economic_stress       (0.15): all three tiers
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from exo import config
from exo.models import DimensionScore, FeatureQuery, TierScore
from exo.store.feature_store import FeatureStore

logger = logging.getLogger(__name__)

TIERS = ("structural", "short_term", "acute")


def _safe(val: float | None, default: float = 0.5) -> float:
    if val is None or (val != val):  # NaN check
        return default
    return float(val)


def _avg(scores: list[float], default: float = 0.5) -> float:
    return sum(scores) / len(scores) if scores else default


def _tier_score(tier: str, scores: list[float], signals: list[str], default: float = 0.5) -> TierScore:
    return TierScore(
        tier=tier,
        score=round(_avg(scores, default), 4),
        contributing_signals=signals,
    )


def _blend_tiers(
    dimension: str,
    tier_scores: dict[str, TierScore],
) -> float:
    """Weighted blend of tier sub-scores using DIMENSION_TIER_WEIGHTS."""
    weights = config.DIMENSION_TIER_WEIGHTS[dimension]
    total_weight = sum(weights[t] for t in TIERS)
    if total_weight == 0:
        return 0.5
    blended = sum(tier_scores[t].score * weights[t] for t in TIERS) / total_weight
    return round(blended, 4)


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
    # 1. Political stability (weight 0.25) — structural only
    # ------------------------------------------------------------------

    def political_stability(self, country: str, now: datetime) -> DimensionScore:
        s_scores: list[float] = []
        s_signals: list[str] = []

        ps_rec = self.store.get_latest(
            entity=country, signal_type="political_stability",
            source="world_bank", as_of_ts=now,
        )
        if ps_rec:
            s_scores.append(1.0 - _safe(ps_rec.value))
            s_signals.append(f"wgi_political_stability={ps_rec.value:.3f}")

        va_rec = self.store.get_latest(
            entity=country, signal_type="voice_accountability",
            source="world_bank", as_of_ts=now,
        )
        if va_rec:
            s_scores.append(1.0 - _safe(va_rec.value))
            s_signals.append(f"wgi_voice_accountability={va_rec.value:.3f}")

        tier_scores = {
            "structural": _tier_score("structural", s_scores, s_signals),
            "short_term": _tier_score("short_term", [], []),
            "acute":      _tier_score("acute", [], []),
        }
        score = _blend_tiers("political_stability", tier_scores)
        all_signals = s_signals
        return DimensionScore(
            name="political_stability",
            score=score,
            weight=config.DIMENSION_WEIGHTS["political_stability"],
            contributing_signals=all_signals,
            tier_scores=tier_scores,
        )

    # ------------------------------------------------------------------
    # 2. Conflict intensity (weight 0.25) — all three tiers
    # ------------------------------------------------------------------

    def conflict_intensity(self, country: str, now: datetime) -> DimensionScore:
        # Structural: UCDP GED verified events + fatalities
        s_scores: list[float] = []
        s_signals: list[str] = []

        ged_events = self.store.get_latest(
            entity=country, signal_type="ucdp_ged_events",
            source="ucdp_ged", as_of_ts=now,
        )
        if ged_events:
            s_scores.append(_safe(ged_events.value))
            s_signals.append(
                f"ucdp_ged_events={ged_events.value:.3f}"
                f" (raw={ged_events.metadata.get('raw_event_count', '?')})"
            )

        ged_fat = self.store.get_latest(
            entity=country, signal_type="ucdp_ged_fatalities",
            source="ucdp_ged", as_of_ts=now,
        )
        if ged_fat:
            s_scores.append(_safe(ged_fat.value))
            s_signals.append(
                f"ucdp_ged_fatalities={ged_fat.value:.3f}"
                f" (raw={ged_fat.metadata.get('raw_fatalities', '?')})"
            )

        # Short-term: UCDP candidate (current year, subject to revision)
        st_scores: list[float] = []
        st_signals: list[str] = []

        cand = self.store.get_latest(
            entity=country, signal_type="ucdp_candidate_events",
            source="ucdp_candidate", as_of_ts=now,
        )
        if cand:
            st_scores.append(_safe(cand.value))
            st_signals.append(
                f"ucdp_candidate_events={cand.value:.3f}"
                f" (raw={cand.metadata.get('raw_event_count', '?')})"
            )

        # Acute: GDELT news magnitude
        a_scores: list[float] = []
        a_signals: list[str] = []

        gdelt_rec = self.store.get_latest(
            entity=country, signal_type="news_magnitude",
            source="gdelt", as_of_ts=now,
        )
        if gdelt_rec and gdelt_rec.value > 0:
            a_scores.append(_safe(gdelt_rec.value))
            a_signals.append(f"gdelt_magnitude={gdelt_rec.value:.3f}")

        # ACLED (pending institutional approval — included for future use)
        week_ago = now - timedelta(days=7)
        acled_recs = self.store.read(FeatureQuery(
            entity=country, signal_type="conflict_event", source="acled",
            as_of_ts=now, start_ts=week_ago, limit=200,
        ))
        if acled_recs:
            mean_intensity = sum(r.value for r in acled_recs) / len(acled_recs)
            total_fatalities = sum(r.metadata.get("fatalities", 0) for r in acled_recs)
            normalised_fatalities = min(1.0, total_fatalities / 1000.0)
            a_scores.append((mean_intensity + normalised_fatalities) / 2)
            a_signals.append(f"acled_events={len(acled_recs)} fatalities={total_fatalities:.0f}")

        tier_scores = {
            "structural": _tier_score("structural", s_scores, s_signals, default=0.3),
            "short_term": _tier_score("short_term", st_scores, st_signals, default=0.3),
            "acute":      _tier_score("acute", a_scores, a_signals, default=0.3),
        }
        score = _blend_tiers("conflict_intensity", tier_scores)
        all_signals = s_signals + st_signals + a_signals
        return DimensionScore(
            name="conflict_intensity",
            score=score,
            weight=config.DIMENSION_WEIGHTS["conflict_intensity"],
            contributing_signals=all_signals,
            tier_scores=tier_scores,
        )

    # ------------------------------------------------------------------
    # 3. Policy predictability (weight 0.20) — structural only
    # ------------------------------------------------------------------

    def policy_predictability(self, country: str, now: datetime) -> DimensionScore:
        s_scores: list[float] = []
        s_signals: list[str] = []

        rec = self.store.get_latest(
            entity=country, signal_type="ideal_point_variance",
            source="unga_votes", as_of_ts=now,
        )
        if rec:
            s_scores.append(_safe(rec.value))
            s_signals.append(
                f"ideal_point_variance={rec.metadata.get('variance', 0):.4f}"
                f" n_years={rec.metadata.get('n_years')}"
                f" latest_year={rec.metadata.get('latest_year')}"
            )

        tier_scores = {
            "structural": _tier_score("structural", s_scores, s_signals or ["insufficient_data"]),
            "short_term": _tier_score("short_term", [], []),
            "acute":      _tier_score("acute", [], []),
        }
        score = _blend_tiers("policy_predictability", tier_scores)
        return DimensionScore(
            name="policy_predictability",
            score=score,
            weight=config.DIMENSION_WEIGHTS["policy_predictability"],
            contributing_signals=s_signals or ["insufficient_data"],
            tier_scores=tier_scores,
        )

    # ------------------------------------------------------------------
    # 4. Sanctions risk (weight 0.15) — structural + acute
    # ------------------------------------------------------------------

    def sanctions_risk(self, country: str, now: datetime) -> DimensionScore:
        # Structural: trade openness + trade concentration (US/EU) + secondary exposure
        s_scores: list[float] = []
        s_signals: list[str] = []

        openness_rec = self.store.get_latest(
            entity=country, signal_type="trade_openness",
            source="world_bank", as_of_ts=now,
        )
        if openness_rec:
            s_scores.append(_safe(openness_rec.value))
            s_signals.append(
                f"trade_openness={openness_rec.value:.3f}"
                f" (exp={openness_rec.metadata.get('exports_pct_gdp', '?'):.1f}%"
                f" imp={openness_rec.metadata.get('imports_pct_gdp', '?'):.1f}%)"
            )

        concentration_rec = self.store.get_latest(
            entity=country, signal_type="trade_concentration",
            source="wits", as_of_ts=now,
        )
        if concentration_rec:
            s_scores.append(_safe(concentration_rec.value))
            s_signals.append(
                f"trade_concentration={concentration_rec.value:.3f}"
                f" (us={concentration_rec.metadata.get('us_pct', '?'):.2f}"
                f" eu={concentration_rec.metadata.get('eu_pct', '?'):.2f})"
            )

        secondary_rec = self.store.get_latest(
            entity=country, signal_type="secondary_exposure",
            source="wits", as_of_ts=now,
        )
        if secondary_rec:
            s_scores.append(_safe(secondary_rec.value))
            sanctioned_b = secondary_rec.metadata.get("sanctioned_weighted_usd", 0) / 1e9
            s_signals.append(
                f"secondary_exposure={secondary_rec.value:.3f}"
                f" (sanctioned_weighted=${sanctioned_b:.1f}B"
                f" n={secondary_rec.metadata.get('n_sanctioned_partners', '?')})"
            )

        # Short-term: OFAC SDN entity count (reflects active US enforcement attention)
        st_scores: list[float] = []
        st_signals: list[str] = []

        sdn_rec = self.store.get_latest(
            entity=country, signal_type="sdn_entity_count",
            source="ofac", as_of_ts=now,
        )
        if sdn_rec:
            st_scores.append(_safe(sdn_rec.value))
            st_signals.append(
                f"sdn_count={sdn_rec.value:.3f}"
                f" (raw={sdn_rec.metadata.get('raw_count', '?')})"
            )

        # Acute: GDELT negative sentiment tone + Google Trends sanctions search
        a_scores: list[float] = []
        a_signals: list[str] = []

        gdelt_rec = self.store.get_latest(
            entity=country, signal_type="news_sentiment",
            source="gdelt", as_of_ts=now,
        )
        if gdelt_rec:
            sanctions_score = max(0.0, min(1.0, (1.0 - gdelt_rec.value) / 2))
            a_scores.append(sanctions_score)
            a_signals.append(f"gdelt_sentiment={gdelt_rec.value:.3f}")

        gt_rec = self.store.get_latest(
            entity="sanctions", signal_type="search_volume",
            source="google_trends", as_of_ts=now,
        )
        if gt_rec:
            a_scores.append(_safe(gt_rec.value))
            a_signals.append(f"google_trends_sanctions={gt_rec.value:.3f}")

        tier_scores = {
            "structural": _tier_score("structural", s_scores, s_signals, default=0.3),
            "short_term": _tier_score("short_term", st_scores, st_signals, default=0.3),
            "acute":      _tier_score("acute", a_scores, a_signals, default=0.3),
        }
        score = _blend_tiers("sanctions_risk", tier_scores)
        all_signals = s_signals + st_signals + a_signals
        return DimensionScore(
            name="sanctions_risk",
            score=score,
            weight=config.DIMENSION_WEIGHTS["sanctions_risk"],
            contributing_signals=all_signals,
            tier_scores=tier_scores,
        )

    # ------------------------------------------------------------------
    # 5. Economic stress (weight 0.15) — all three tiers
    # ------------------------------------------------------------------

    def economic_stress(self, country: str, now: datetime) -> DimensionScore:
        # Structural: World Bank macro + WGI governance
        s_scores: list[float] = []
        s_signals: list[str] = []

        wb_signals = {
            "gdp_growth": None,
            "debt_to_gdp": None,
            "unemployment_rate": None,
            "gross_savings": None,
            "rule_of_law": None,
            "control_of_corruption": None,
        }
        for sig in wb_signals:
            rec = self.store.get_latest(
                entity=country, signal_type=sig,
                source="world_bank", as_of_ts=now,
            )
            if rec is not None:
                raw = rec.value
                if sig == "unemployment_rate":
                    normalised = min(1.0, raw / 20.0)
                elif sig == "debt_to_gdp":
                    normalised = min(1.0, raw / 200.0)
                elif sig == "gdp_growth":
                    normalised = max(0.0, min(1.0, (5.0 - raw) / 10.0))
                elif sig in ("rule_of_law", "control_of_corruption"):
                    normalised = 1.0 - raw   # higher governance = lower stress
                else:
                    normalised = 0.5
                s_scores.append(normalised)
                s_signals.append(f"{sig}={raw:.2f}")

        # Short-term: FRED composite + EIA energy prices (weighted by import dependency)
        st_scores: list[float] = []
        st_signals: list[str] = []

        fred_rec = self.store.get_latest(
            entity="US", signal_type="economic_indicator",
            source="fred", as_of_ts=now,
        )
        if fred_rec:
            st_scores.append(_safe(fred_rec.value))
            st_signals.append(f"fred_composite={fred_rec.value:.3f}")

        energy_imports_rec = self.store.get_latest(
            entity=country, signal_type="energy_imports_pct",
            source="world_bank", as_of_ts=now,
        )
        energy_import_weight = _safe(energy_imports_rec.value, 0.5) if energy_imports_rec else 0.5

        crude_rec = self.store.get_latest(
            entity="WTI", signal_type="crude_oil_price",
            source="eia", as_of_ts=now,
        )
        if crude_rec:
            crude_norm = max(0.0, min(1.0, (crude_rec.value - 40) / 80))
            directional = crude_norm * energy_import_weight + (1.0 - crude_norm) * (1.0 - energy_import_weight)
            st_scores.append(directional)
            st_signals.append(f"eia_crude={crude_rec.value:.2f} import_weight={energy_import_weight:.2f}")

        gas_rec = self.store.get_latest(
            entity="HH", signal_type="natural_gas_price",
            source="eia", as_of_ts=now,
        )
        if gas_rec:
            gas_norm = max(0.0, min(1.0, (gas_rec.value - 2) / 6))
            directional = gas_norm * energy_import_weight + (1.0 - gas_norm) * (1.0 - energy_import_weight)
            st_scores.append(directional)
            st_signals.append(f"eia_gas={gas_rec.value:.2f}")

        # Acute: Finnhub financial news sentiment + Polymarket near-term markets
        a_scores: list[float] = []
        a_signals: list[str] = []

        for category in ("general", "forex", "merger"):
            finnhub_rec = self.store.get_latest(
                entity=category, signal_type="news_sentiment",
                source="finnhub", as_of_ts=now,
            )
            if finnhub_rec:
                stress = (1.0 - _safe(finnhub_rec.value)) / 2
                a_scores.append(stress)
                a_signals.append(f"finnhub_{category}={finnhub_rec.value:.3f}")

        # Polymarket near-term markets (near_term flag set by ingestor)
        week_ago = now - timedelta(days=7)
        pm_recs = self.store.read(FeatureQuery(
            signal_type="polymarket_price", source="polymarket",
            as_of_ts=now, start_ts=week_ago, limit=50,
        ))
        near_term_pm = [r for r in pm_recs if r.metadata.get("near_term")]
        if near_term_pm:
            avg_pm = sum(r.value for r in near_term_pm) / len(near_term_pm)
            a_scores.append(_safe(avg_pm))
            a_signals.append(f"polymarket_near_term={avg_pm:.3f} n={len(near_term_pm)}")

        tier_scores = {
            "structural": _tier_score("structural", s_scores, s_signals),
            "short_term": _tier_score("short_term", st_scores, st_signals),
            "acute":      _tier_score("acute", a_scores, a_signals),
        }
        score = _blend_tiers("economic_stress", tier_scores)
        all_signals = s_signals + st_signals + a_signals
        return DimensionScore(
            name="economic_stress",
            score=score,
            weight=config.DIMENSION_WEIGHTS["economic_stress"],
            contributing_signals=all_signals,
            tier_scores=tier_scores,
        )
