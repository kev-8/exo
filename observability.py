"""Structured JSON logging, metrics, and alerting."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

# ---------------------------------------------------------------------------
# Structured logger setup
# ---------------------------------------------------------------------------


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for structured JSON output."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Metrics store (in-process; replace with Prometheus/StatsD in production)
# ---------------------------------------------------------------------------


class MetricsStore:
    """Simple in-process metrics accumulator."""

    def __init__(self) -> None:
        self._latencies: dict[str, deque] = {}
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._brier_scores: deque = deque(maxlen=1000)

    # Latency
    def record_latency(self, key: str, latency_sec: float) -> None:
        if key not in self._latencies:
            self._latencies[key] = deque(maxlen=1000)
        self._latencies[key].append(latency_sec)

    def get_latency_percentiles(self, key: str) -> dict[str, float]:
        vals = sorted(self._latencies.get(key, []))
        if not vals:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        n = len(vals)
        return {
            "p50": vals[int(n * 0.50)],
            "p95": vals[int(n * 0.95)],
            "p99": vals[int(n * 0.99)],
        }

    # Counters
    def increment(self, key: str, amount: int = 1) -> None:
        self._counters[key] = self._counters.get(key, 0) + amount

    def get_counter(self, key: str) -> int:
        return self._counters.get(key, 0)

    # Gauges
    def set_gauge(self, key: str, value: float) -> None:
        self._gauges[key] = value

    def get_gauge(self, key: str) -> float:
        return self._gauges.get(key, 0.0)

    # Brier score (rolling 30-day)
    def record_brier(self, predicted: float, outcome: int) -> None:
        self._brier_scores.append((predicted - outcome) ** 2)

    def get_brier_score(self) -> float | None:
        if not self._brier_scores:
            return None
        return sum(self._brier_scores) / len(self._brier_scores)

    def snapshot(self) -> dict:
        return {
            "brier_score_rolling": self.get_brier_score(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "llm_latency": self.get_latency_percentiles("llm_tier3"),
            "feature_store_writes": self.get_counter("feature_store_write"),
            "opportunity_signals": self.get_counter("opportunity_signal"),
        }


# Module-level singleton
_metrics = MetricsStore()


def get_metrics() -> MetricsStore:
    return _metrics


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------


def timed(metric_key: str) -> Callable:
    """Decorator that records function latency to the metrics store."""

    def decorator(fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.monotonic() - start
                _metrics.record_latency(metric_key, elapsed)

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = time.monotonic() - start
                _metrics.record_latency(metric_key, elapsed)

        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

DATA_ERROR_RATE_THRESHOLD = 0.02   # 2%
