"""Resampling-based inference for the real-data admission experiments.

Two designs are supported:

1. Calendar-time block bootstrap.  Holding each policy's admission decisions fixed,
   resample calendar months in circular blocks from the common timeline and recompute
   post-decision holdout statistics.  Because the same resampled months are applied to
   every signal, cross-signal correlation within a month is preserved.
2. Signal cluster bootstrap.  Resample whole signals with replacement and recompute
   activation rates and holdout statistics.  This treats signals as the independent
   sampling unit and captures cross-signal heterogeneity in both decisions and
   post-decision returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from sequential_tradeability.realdata import SignalEvidence


@dataclass(frozen=True)
class HoldoutSeries:
    """Post-decision holdout returns for one signal, indexed by calendar month."""

    signal: str
    month_ids: np.ndarray
    returns: np.ndarray

    @property
    def mean_return(self) -> float:
        if self.returns.size == 0:
            return float("nan")
        return float(np.mean(self.returns))


def month_id(value: date) -> int:
    """Return a consecutive integer month index (months since year zero)."""
    return value.year * 12 + (value.month - 1)


def holdout_series(
    evidence: SignalEvidence,
    *,
    stop_index: int,
    holdout_months: int,
) -> HoldoutSeries:
    """Extract the dated post-decision holdout window for one evidence stream.

    Mirrors the window used by ``realdata._holdout_statistics``: holdout returns start
    at the decision month inside the validation window and extend into the reserved
    holdout segment, truncated to ``holdout_months`` observations.
    """
    if stop_index < 0:
        raise ValueError("stop_index must be nonnegative")
    if holdout_months < 1:
        raise ValueError("holdout_months must be positive")

    returns = evidence.validation_returns[stop_index:]
    dates = evidence.validation_dates[stop_index:]
    if evidence.holdout_returns.size:
        returns = np.concatenate([returns, evidence.holdout_returns])
        dates = np.concatenate([dates, evidence.holdout_dates])
    returns = np.asarray(returns[:holdout_months], dtype=float)
    dates = dates[:holdout_months]
    month_ids = np.array([month_id(value) for value in dates], dtype=int)
    return HoldoutSeries(signal=evidence.name, month_ids=month_ids, returns=returns)


def circular_block_indices(
    rng: np.random.Generator,
    *,
    n_periods: int,
    block_length: int,
) -> np.ndarray:
    """Draw a circular block bootstrap resample of ``0..n_periods-1``.

    Blocks of consecutive indices (wrapping at the end) are concatenated until the
    resample reaches ``n_periods`` entries.
    """
    if n_periods < 1:
        raise ValueError("n_periods must be positive")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    n_blocks = int(np.ceil(n_periods / block_length))
    starts = rng.integers(0, n_periods, size=n_blocks)
    offsets = np.arange(block_length)
    indices = (starts[:, None] + offsets[None, :]) % n_periods
    return indices.reshape(-1)[:n_periods]


def time_block_bootstrap_means(
    panel: list[HoldoutSeries],
    rng: np.random.Generator,
    *,
    block_length: int = 12,
) -> np.ndarray:
    """One time-block bootstrap replication of per-signal holdout mean returns.

    A single resample of the common calendar timeline is applied to every signal:
    month ``t`` drawn ``k`` times contributes each signal's return in month ``t`` with
    weight ``k``.  Signals are aligned on calendar months, so cross-signal correlation
    within a month is preserved.  Returns one bootstrap mean per panel entry (``nan``
    if a signal's window receives no sampled months).
    """
    if not panel:
        return np.array([], dtype=float)
    low = min(int(series.month_ids.min()) for series in panel)
    high = max(int(series.month_ids.max()) for series in panel)
    n_periods = high - low + 1
    drawn = circular_block_indices(rng, n_periods=n_periods, block_length=block_length)
    counts = np.bincount(drawn, minlength=n_periods).astype(float)

    means = np.empty(len(panel), dtype=float)
    for i, series in enumerate(panel):
        weights = counts[series.month_ids - low]
        total = float(weights.sum())
        means[i] = float(np.dot(weights, series.returns) / total) if total > 0 else float("nan")
    return means


def percentile_interval(values: np.ndarray, *, coverage: float = 0.95) -> tuple[float, float]:
    """Return the symmetric percentile interval of a bootstrap distribution."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    tail = 100.0 * (1.0 - coverage) / 2.0
    return (
        float(np.percentile(values, tail)),
        float(np.percentile(values, 100.0 - tail)),
    )
