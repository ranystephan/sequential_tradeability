"""Tests for resampling-based inference helpers."""

from __future__ import annotations

from datetime import date

import numpy as np

from sequential_tradeability import (
    HoldoutSeries,
    ReturnSeries,
    build_signal_evidence,
    circular_block_indices,
    holdout_series,
    month_id,
    percentile_interval,
    time_block_bootstrap_means,
)
from sequential_tradeability.realdata import _holdout_statistics


def _toy_evidence(n_months: int = 80, calibration: int = 24, validation: int = 36):
    rng = np.random.default_rng(7)
    dates = np.array(
        [date(1990 + month // 12, month % 12 + 1, 28) for month in range(n_months)],
        dtype=object,
    )
    series = ReturnSeries(name="toy", dates=dates, returns=rng.normal(0.5, 2.0, size=n_months))
    return build_signal_evidence(
        series,
        calibration_months=calibration,
        validation_months=validation,
    )


def test_month_id_is_consecutive_across_year_boundary():
    assert month_id(date(1999, 12, 31)) + 1 == month_id(date(2000, 1, 1))


def test_holdout_series_matches_holdout_statistics():
    evidence = _toy_evidence()
    for stop_index in [0, 5, 30]:
        series = holdout_series(evidence, stop_index=stop_index, holdout_months=12)
        n_months, mean_return, _ = _holdout_statistics(
            evidence, stop_index=stop_index, holdout_months=12
        )
        assert series.returns.size == n_months
        assert np.isclose(series.mean_return, mean_return)
        assert series.month_ids.size == series.returns.size
        assert np.all(np.diff(series.month_ids) == 1)


def test_circular_block_indices_cover_range_and_are_reproducible():
    rng = np.random.default_rng(11)
    indices = circular_block_indices(rng, n_periods=100, block_length=12)
    assert indices.size == 100
    assert indices.min() >= 0 and indices.max() < 100
    rng_repeat = np.random.default_rng(11)
    repeat = circular_block_indices(rng_repeat, n_periods=100, block_length=12)
    assert np.array_equal(indices, repeat)


def test_time_block_bootstrap_preserves_constant_returns():
    months = np.arange(24000, 24060)
    panel = [
        HoldoutSeries(signal="a", month_ids=months, returns=np.full(60, 1.5)),
        HoldoutSeries(signal="b", month_ids=months[10:50], returns=np.full(40, -0.5)),
    ]
    rng = np.random.default_rng(3)
    means = time_block_bootstrap_means(panel, rng, block_length=6)
    assert np.allclose(means, [1.5, -0.5])


def test_time_block_bootstrap_applies_common_months_to_aligned_signals():
    months = np.arange(24000, 24120)
    rng_data = np.random.default_rng(5)
    returns = rng_data.normal(size=120)
    panel = [
        HoldoutSeries(signal="a", month_ids=months, returns=returns),
        HoldoutSeries(signal="b", month_ids=months, returns=2.0 * returns),
    ]
    rng = np.random.default_rng(9)
    means = time_block_bootstrap_means(panel, rng, block_length=12)
    assert np.isclose(means[1], 2.0 * means[0])


def test_time_block_bootstrap_is_unbiased_for_the_sample_mean():
    months = np.arange(24000, 24120)
    rng_data = np.random.default_rng(13)
    returns = rng_data.normal(size=120)
    panel = [HoldoutSeries(signal="a", month_ids=months, returns=returns)]
    rng = np.random.default_rng(17)
    draws = np.array(
        [time_block_bootstrap_means(panel, rng, block_length=12)[0] for _ in range(4000)]
    )
    assert abs(np.mean(draws) - np.mean(returns)) < 0.02
    assert np.std(draws) > 0.0


def test_percentile_interval_brackets_distribution():
    values = np.linspace(0.0, 1.0, 1001)
    low, high = percentile_interval(values, coverage=0.9)
    assert np.isclose(low, 0.05, atol=1e-3)
    assert np.isclose(high, 0.95, atol=1e-3)
