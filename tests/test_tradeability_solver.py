import numpy as np
import pytest

from sequential_tradeability import (
    GaussianPosterior,
    TradeabilityPayoff,
    solve_gaussian_tradeability,
)


def test_tradeability_payoff_activation_threshold_is_exact() -> None:
    payoff = TradeabilityPayoff(
        risk_aversion=3.0,
        return_variance=0.04,
        implementation_hurdle=0.002,
    )
    posterior_variance = 0.01
    threshold = payoff.activation_threshold(posterior_variance)

    assert payoff.value(np.array([0.999 * threshold]), posterior_variance)[0] == 0.0
    assert payoff.value(np.array([threshold]), posterior_variance)[0] == pytest.approx(0.0)
    assert payoff.value(np.array([1.001 * threshold]), posterior_variance)[0] > 0.0


def test_zero_terminal_payoff_implies_no_continuation_value() -> None:
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.5)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=1.0,
        implementation_hurdle=1.0e9,
    )
    solution = solve_gaussian_tradeability(
        posterior=posterior,
        payoff_model=payoff,
        observation_cost=0.1,
        horizon=1.0,
        mean_max=1.0,
        n_time=21,
        n_mean=51,
    )

    assert np.all(solution.values >= solution.payoff)
    assert np.max(solution.values) == pytest.approx(0.0, abs=1e-12)


def test_observation_option_value_is_positive_near_zero_mean() -> None:
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.7)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.015,
    )
    solution = solve_gaussian_tradeability(
        posterior=posterior,
        payoff_model=payoff,
        observation_cost=0.0005,
        horizon=2.0,
        mean_max=1.0,
        n_time=101,
        n_mean=101,
    )

    center = solution.grid.means.size // 2
    assert solution.payoff[0, center] == 0.0
    assert solution.values[0, center] > 0.0
    assert not solution.stop[0, center]


def test_higher_observation_cost_reduces_initial_value() -> None:
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.7)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.015,
    )
    common_kwargs = dict(
        posterior=posterior,
        payoff_model=payoff,
        horizon=2.0,
        mean_max=1.0,
        n_time=101,
        n_mean=101,
    )

    cheap = solve_gaussian_tradeability(observation_cost=0.0005, **common_kwargs)
    expensive = solve_gaussian_tradeability(observation_cost=0.02, **common_kwargs)

    assert expensive.value_at_initial_mean(0.0) < cheap.value_at_initial_mean(0.0)


def test_solution_is_symmetric_in_posterior_mean() -> None:
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.4)
    payoff = TradeabilityPayoff(
        risk_aversion=2.0,
        return_variance=0.05,
        implementation_hurdle=0.01,
    )
    solution = solve_gaussian_tradeability(
        posterior=posterior,
        payoff_model=payoff,
        observation_cost=0.001,
        horizon=1.5,
        mean_max=0.8,
        n_time=81,
        n_mean=81,
    )

    assert np.max(np.abs(solution.values - np.fliplr(solution.values))) < 1e-10


def test_decision_regions_distinguish_reject_observe_and_activate() -> None:
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.7)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.015,
    )
    solution = solve_gaussian_tradeability(
        posterior=posterior,
        payoff_model=payoff,
        observation_cost=0.0005,
        horizon=2.0,
        mean_max=1.0,
        n_time=101,
        n_mean=101,
    )
    regions = solution.decision_regions()
    center = solution.grid.means.size // 2

    assert regions[0, center] == 0
    assert regions[-1, center] == -1
    assert regions[0, 0] == 1
    assert regions[0, -1] == 1


def test_activation_decay_makes_strong_alpha_exercise_early() -> None:
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
        n_time=101,
        n_mean=101,
    )
    regions = solution.decision_regions()

    assert regions[0, 0] == 1
    assert regions[0, -1] == 1
