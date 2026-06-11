from datetime import date

import numpy as np
import pytest

from sequential_tradeability import (
    ReturnSeries,
    ThreeStatePrior,
    TradeabilityPayoff,
    apply_static_threshold_to_evidence,
    apply_three_state_solution_to_evidence,
    build_signal_evidence,
    lag1_autocorrelation,
    block_sign_flip_evidence,
    sign_flip_evidence,
    solve_three_state_tradeability,
    static_three_state_score,
    trim_return_series,
)


def test_build_signal_evidence_has_brownian_increment_scaling() -> None:
    dates = np.array([date(2000 + index // 12, index % 12 + 1, 28) for index in range(18)])
    returns = np.array(
        [
            1.0,
            -1.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            1.5,
            -1.5,
            0.75,
            -0.75,
            0.5,
            -0.5,
            0.5,
            -0.5,
            0.25,
            -0.25,
            0.1,
            -0.1,
        ]
    )
    series = ReturnSeries(name="toy", dates=dates, returns=returns)

    evidence = build_signal_evidence(
        series,
        calibration_months=12,
        validation_months=4,
        dt=1.0 / 12.0,
    )
    expected_sigma = np.std(returns[:12], ddof=1)
    expected_increments = returns[12:16] / expected_sigma * np.sqrt(1.0 / 12.0)

    assert evidence.monthly_sigma == pytest.approx(expected_sigma)
    assert np.allclose(evidence.increments, expected_increments)
    assert evidence.observations[0] == 0.0
    assert np.allclose(evidence.observations[1:], np.cumsum(expected_increments))
    assert evidence.times[-1] == pytest.approx(4.0 / 12.0)


def test_lag1_autocorrelation_matches_manual_formula() -> None:
    values = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    left = values[:-1] - np.mean(values[:-1])
    right = values[1:] - np.mean(values[1:])
    expected = np.dot(left, right) / np.sqrt(np.dot(left, left) * np.dot(right, right))

    assert lag1_autocorrelation(values) == pytest.approx(expected)


def test_trim_return_series_respects_date_range() -> None:
    dates = np.array([date(2000, month, 28) for month in range(1, 7)], dtype=object)
    returns = np.arange(6.0)
    series = ReturnSeries(name="toy", dates=dates, returns=returns)

    trimmed = trim_return_series(
        series,
        start_date=date(2000, 3, 1),
        end_date=date(2000, 5, 31),
    )

    assert list(trimmed.dates) == [
        date(2000, 3, 28),
        date(2000, 4, 28),
        date(2000, 5, 28),
    ]
    assert np.allclose(trimmed.returns, [2.0, 3.0, 4.0])


def test_sign_flip_evidence_preserves_increment_magnitudes() -> None:
    dates = np.array([date(2000 + index // 12, index % 12 + 1, 28) for index in range(24)])
    returns = np.array([1.0, -1.0] * 12)
    evidence = build_signal_evidence(
        ReturnSeries(name="toy", dates=dates, returns=returns),
        calibration_months=12,
        validation_months=8,
    )

    placebo = sign_flip_evidence(evidence, np.random.default_rng(3))

    assert placebo.name == "toy_signflip"
    assert np.allclose(np.sort(np.abs(placebo.increments)), np.sort(np.abs(evidence.increments)))
    assert placebo.observations[0] == 0.0
    assert np.allclose(placebo.observations[1:], np.cumsum(placebo.increments))


def test_static_three_state_net_payoff_score_matches_manual_formula() -> None:
    dates = np.array([date(2000 + index // 12, index % 12 + 1, 28) for index in range(24)])
    validation_returns = [2.0, 2.0, -1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.2, -0.2]
    returns = np.array([1.0, -1.0] * 6 + validation_returns)
    evidence = build_signal_evidence(
        ReturnSeries(name="toy", dates=dates, returns=returns),
        calibration_months=12,
        validation_months=8,
    )
    prior = ThreeStatePrior(alpha=0.4, prob_negative=0.15, prob_dead=0.7, prob_positive=0.15)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.005,
    )
    horizon_months = 4
    time = evidence.times[horizon_months]
    observation = evidence.observations[horizon_months]
    posterior_mean = prior.mean(time, observation)
    posterior_variance = prior.variance(time, observation)
    posterior_dead = prior.posterior_dead(time, observation)
    expected = (
        payoff.gross_value(np.array([posterior_mean]), np.array([posterior_variance]))[0]
        - payoff.implementation_hurdle
        - 0.01 * posterior_dead
    )

    score = static_three_state_score(
        prior=prior,
        payoff_model=payoff,
        evidence=evidence,
        horizon_months=horizon_months,
        score_kind="net_payoff",
        dead_activation_penalty=0.01,
    )

    assert score == pytest.approx(expected)


def test_static_threshold_decides_at_requested_horizon_and_holdout_start() -> None:
    dates = np.array([date(2000 + index // 12, index % 12 + 1, 28) for index in range(36)])
    returns = np.array([1.0, -1.0] * 6 + [2.0, 2.0, 2.0, -1.0, -1.0, -1.0] + [0.5] * 18)
    evidence = build_signal_evidence(
        ReturnSeries(name="toy", dates=dates, returns=returns),
        calibration_months=12,
        validation_months=6,
    )
    prior = ThreeStatePrior(alpha=0.4, prob_negative=0.15, prob_dead=0.7, prob_positive=0.15)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.005,
    )

    result = apply_static_threshold_to_evidence(
        prior=prior,
        payoff_model=payoff,
        evidence=evidence,
        horizon_months=3,
        score_kind="z_stat",
        threshold=0.0,
        holdout_months=4,
    )

    assert result.decision == 1
    assert result.stop_index == 3
    assert result.stop_date == evidence.validation_dates[2]
    assert result.holdout_months == 4
    expected_holdout = np.concatenate(
        [evidence.validation_returns[3:], evidence.holdout_returns]
    )[:4]
    assert result.holdout_mean_return_percent == pytest.approx(np.mean(expected_holdout))


def test_apply_three_state_solution_returns_valid_real_data_decision() -> None:
    dates = np.array([date(2000 + index // 12, index % 12 + 1, 28) for index in range(36)])
    returns = np.array(
        [
            1.0,
            -1.0,
            1.2,
            -1.2,
            0.8,
            -0.8,
            1.1,
            -1.1,
            0.9,
            -0.9,
            1.0,
            -1.0,
            3.0,
            2.5,
            2.0,
            1.5,
            1.0,
            0.5,
            0.25,
            0.1,
            -0.2,
            0.2,
            -0.1,
            0.1,
            0.4,
            0.3,
            0.2,
            0.1,
            -0.1,
            -0.2,
            0.1,
            0.2,
            0.3,
            0.1,
            0.0,
            -0.1,
        ]
    )
    evidence = build_signal_evidence(
        ReturnSeries(name="toy", dates=dates, returns=returns),
        calibration_months=12,
        validation_months=12,
    )
    prior = ThreeStatePrior(
        alpha=0.4,
        prob_negative=0.15,
        prob_dead=0.7,
        prob_positive=0.15,
    )
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.005,
    )
    solution = solve_three_state_tradeability(
        prior=prior,
        payoff_model=payoff,
        observation_cost=0.001,
        horizon=1.0,
        evidence_max=4.0,
        activation_decay=1.0,
        dead_activation_penalty=0.002,
        n_time=13,
        n_evidence=101,
    )

    result = apply_three_state_solution_to_evidence(solution, evidence, holdout_months=6)

    assert result.signal == "toy"
    assert result.decision in {-1, 1}
    assert 0 <= result.stop_index <= evidence.validation_months
    if result.stop_index > 0:
        assert result.stop_date == evidence.validation_dates[result.stop_index - 1]
    assert result.holdout_months == 6
    assert np.isfinite(result.validation_increment_variance_over_dt)


def test_block_sign_flip_evidence_flips_whole_blocks() -> None:
    dates = np.array([date(2000 + index // 12, index % 12 + 1, 28) for index in range(60)])
    rng_data = np.random.default_rng(11)
    returns = rng_data.normal(0.5, 1.0, size=60)
    evidence = build_signal_evidence(
        ReturnSeries(name="toy", dates=dates, returns=returns),
        calibration_months=12,
        validation_months=48,
    )

    placebo = block_sign_flip_evidence(evidence, np.random.default_rng(7), block_months=12)

    assert placebo.name == "toy_blockflip"
    assert np.allclose(np.abs(placebo.increments), np.abs(evidence.increments))
    ratios = placebo.increments / evidence.increments
    for start in range(0, 48, 12):
        block = ratios[start : start + 12]
        assert np.all(block == block[0])
        assert block[0] in (-1.0, 1.0)
    assert placebo.observations[0] == 0.0
    assert np.allclose(placebo.observations[1:], np.cumsum(placebo.increments))


def test_block_sign_flip_evidence_handles_partial_final_block() -> None:
    dates = np.array([date(2000 + index // 12, index % 12 + 1, 28) for index in range(31)])
    returns = np.arange(1.0, 32.0)
    evidence = build_signal_evidence(
        ReturnSeries(name="toy", dates=dates, returns=returns),
        calibration_months=12,
        validation_months=19,
    )

    placebo = block_sign_flip_evidence(evidence, np.random.default_rng(5), block_months=12)

    assert placebo.increments.size == 19
    ratios = placebo.increments / evidence.increments
    assert np.all(ratios[:12] == ratios[0])
    assert np.all(ratios[12:] == ratios[12])
