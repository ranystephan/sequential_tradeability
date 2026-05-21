import numpy as np
import pytest

from sequential_tradeability import (
    GaussianPosterior,
    TradeabilityPayoff,
    evaluate_boundary_policy,
    evaluate_static_payoff_policy,
    evaluate_zstat_policy,
    realized_activation_value,
    simulate_gaussian_evidence,
    solve_gaussian_tradeability,
)


def test_gaussian_posterior_is_monte_carlo_calibrated() -> None:
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.25)
    paths = simulate_gaussian_evidence(
        posterior=posterior,
        horizon=1.5,
        n_steps=60,
        n_paths=40_000,
        seed=7,
    )
    error = paths.true_theta - paths.posterior_means[:, -1]
    mse = np.mean(error**2)

    assert mse == pytest.approx(paths.posterior_variance[-1], rel=0.025)


def test_fixed_theta_observation_moments_match_model() -> None:
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.25)
    theta = 0.3
    horizon = 2.0
    paths = simulate_gaussian_evidence(
        posterior=posterior,
        horizon=horizon,
        n_steps=80,
        n_paths=50_000,
        seed=11,
        true_theta=theta,
    )
    terminal_y = paths.observations[:, -1]

    assert np.mean(terminal_y) == pytest.approx(theta * horizon, abs=0.02)
    assert np.var(terminal_y, ddof=1) == pytest.approx(horizon, rel=0.02)


def test_realized_activation_value_uses_posterior_chosen_exposure() -> None:
    payoff = TradeabilityPayoff(
        risk_aversion=2.0,
        return_variance=0.5,
        implementation_hurdle=0.1,
    )
    value = realized_activation_value(
        true_theta=np.array([0.4]),
        posterior_mean=np.array([0.3]),
        posterior_variance=np.array([0.1]),
        stop_time=np.array([0.25]),
        decision=np.array([1]),
        payoff_model=payoff,
        observation_cost=0.01,
        activation_decay=0.2,
    )[0]
    exposure = 0.3 / (2.0 * (0.5 + 0.1))
    expected = np.exp(-0.2 * 0.25) * (0.4 * exposure - 0.5 * 2.0 * 0.5 * exposure**2 - 0.1)
    expected -= 0.01 * 0.25

    assert value == pytest.approx(expected)


def test_policy_evaluators_return_valid_stopping_decisions() -> None:
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.05)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.015,
    )
    solution = solve_gaussian_tradeability(
        posterior=posterior,
        payoff_model=payoff,
        observation_cost=0.02,
        horizon=1.0,
        mean_max=0.8,
        activation_decay=3.0,
        n_time=51,
        n_mean=101,
    )
    paths = simulate_gaussian_evidence(
        posterior=posterior,
        horizon=1.0,
        n_steps=50,
        n_paths=200,
        seed=13,
        true_theta=0.2,
    )

    outcomes = [
        evaluate_boundary_policy(
            solution=solution,
            paths=paths,
            payoff_model=payoff,
            observation_cost=0.02,
            activation_decay=3.0,
        ),
        evaluate_static_payoff_policy(
            paths=paths,
            payoff_model=payoff,
            observation_cost=0.02,
            activation_decay=3.0,
        ),
        evaluate_zstat_policy(
            paths=paths,
            payoff_model=payoff,
            observation_cost=0.02,
            activation_decay=3.0,
            z_threshold=2.0,
        ),
    ]

    for outcome in outcomes:
        assert set(np.unique(outcome.decision)).issubset({-1, 1})
        assert np.all((outcome.stop_index >= 0) & (outcome.stop_index < paths.times.size))
        assert outcome.realized_value.shape == paths.true_theta.shape
