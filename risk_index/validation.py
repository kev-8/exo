"""Validation against established geopolitical risk benchmarks.

Primary benchmark:  ICRG Political Risk Index (Pearson r target > 0.65)
Secondary benchmark: V-Dem Liberal Democracy Index
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson r between two equal-length lists."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        raise ValueError("Inputs must be equal-length lists with at least 2 elements")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


def validate_against_icrg(
    composite_scores: dict[str, float],
    icrg_scores: dict[str, float],
    target_correlation: float = 0.65,
) -> dict[str, Any]:
    """Validate composite scores against ICRG benchmark scores.

    Parameters
    ----------
    composite_scores:
        Dict mapping country code → our composite risk score (higher = riskier).
    icrg_scores:
        Dict mapping country code → ICRG political risk score (0-100, higher = safer).
        Inverted internally to align conventions.
    target_correlation:
        Minimum required Pearson r.

    Returns
    -------
    dict with keys: ``correlation``, ``passed``, ``n_countries``, ``details``.
    """
    common = sorted(set(composite_scores) & set(icrg_scores))
    if len(common) < 5:
        return {
            "correlation": None,
            "passed": False,
            "n_countries": len(common),
            "details": "Insufficient overlap between our scores and ICRG",
        }

    ours = [composite_scores[c] for c in common]
    icrg_inverted = [1.0 - icrg_scores[c] / 100.0 for c in common]

    r = pearson_correlation(ours, icrg_inverted)
    passed = r >= target_correlation

    if passed:
        logger.info("ICRG validation PASSED: r=%.4f (target=%.2f)", r, target_correlation)
    else:
        logger.warning("ICRG validation FAILED: r=%.4f < target=%.2f", r, target_correlation)

    return {
        "correlation": round(r, 4),
        "passed": passed,
        "n_countries": len(common),
        "target": target_correlation,
        "countries": common,
        "details": f"Pearson r={r:.4f} on {len(common)} countries",
    }


def validate_against_vdem(
    composite_scores: dict[str, float],
    vdem_scores: dict[str, float],
) -> dict[str, Any]:
    """Secondary validation against the V-Dem liberal democracy index.

    V-Dem values are 0–1 (higher = more democratic = lower risk).
    """
    common = sorted(set(composite_scores) & set(vdem_scores))
    if len(common) < 5:
        return {"correlation": None, "passed": False, "n_countries": len(common)}

    ours = [composite_scores[c] for c in common]
    vdem_inverted = [1.0 - vdem_scores[c] for c in common]

    r = pearson_correlation(ours, vdem_inverted)
    return {
        "correlation": round(r, 4),
        "passed": r >= 0.50,
        "n_countries": len(common),
        "countries": common,
    }
