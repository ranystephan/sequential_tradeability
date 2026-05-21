import numpy as np
import pytest

from sequential_tradeability import (
    GaussianPosterior,
    ThreeStatePrior,
    TradeabilityPayoff,
    solve_gaussian_tradeability,
    solve_three_state_tradeability,
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


def test_three_state_solution_dominates_payoff_and_matches_terminal_condition() -> None:
    prior = ThreeStatePrior(
        alpha=0.4,
        prob_negative=0.25,
        prob_dead=0.5,
        prob_positive=0.25,
    )
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.01,
    )

    solution = solve_three_state_tradeability(
        prior=prior,
        payoff_model=payoff,
        observation_cost=0.005,
        horizon=1.0,
        evidence_max=3.0,
        activation_decay=1.0,
        n_time=61,
        n_evidence=101,
    )

    assert np.all(solution.values >= solution.payoff - 1e-12)
    assert np.max(np.abs(solution.values[-1] - solution.payoff[-1])) < 1e-12
    assert solution.value_at_initial_evidence(0.0) >= 0.0


def test_three_state_solution_is_symmetric_for_symmetric_prior() -> None:
    prior = ThreeStatePrior(
        alpha=0.45,
        prob_negative=0.2,
        prob_dead=0.6,
        prob_positive=0.2,
    )
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.008,
    )

    solution = solve_three_state_tradeability(
        prior=prior,
        payoff_model=payoff,
        observation_cost=0.003,
        horizon=1.0,
        evidence_max=3.0,
        activation_decay=1.0,
        n_time=61,
        n_evidence=101,
    )

    assert np.max(np.abs(solution.values - np.fliplr(solution.values))) < 1e-10
    assert np.max(np.abs(solution.posterior_dead - np.fliplr(solution.posterior_dead))) < 1e-12


def test_three_state_higher_observation_cost_reduces_initial_value() -> None:
    prior = ThreeStatePrior(
        alpha=0.4,
        prob_negative=0.25,
        prob_dead=0.5,
        prob_positive=0.25,
    )
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.01,
    )
    common_kwargs = dict(
        prior=prior,
        payoff_model=payoff,
        horizon=1.0,
        evidence_max=3.0,
        activation_decay=1.0,
        n_time=61,
        n_evidence=101,
    )

    cheap = solve_three_state_tradeability(observation_cost=0.001, **common_kwargs)
    expensive = solve_three_state_tradeability(observation_cost=0.02, **common_kwargs)

    assert expensive.value_at_initial_evidence(0.0) < cheap.value_at_initial_evidence(0.0)


def test_three_state_dead_activation_penalty_reduces_initial_value() -> None:
    prior = ThreeStatePrior(
        alpha=0.45,
        prob_negative=0.15,
        prob_dead=0.7,
        prob_positive=0.15,
    )
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.005,
    )
    common_kwargs = dict(
        prior=prior,
        payoff_model=payoff,
        observation_cost=0.001,
        horizon=1.0,
        evidence_max=4.0,
        activation_decay=1.0,
        n_time=61,
        n_evidence=101,
    )

    no_penalty = solve_three_state_tradeability(dead_activation_penalty=0.0, **common_kwargs)
    penalized = solve_three_state_tradeability(dead_activation_penalty=0.01, **common_kwargs)

    assert penalized.value_at_initial_evidence(0.0) < no_penalty.value_at_initial_evidence(0.0)
    assert np.all(penalized.values <= no_penalty.values + 1e-10)


def test_three_state_regions_include_reject_observe_and_activate() -> None:
    prior = ThreeStatePrior(
        alpha=0.5,
        prob_negative=0.25,
        prob_dead=0.5,
        prob_positive=0.25,
    )
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.01,
    )
    solution = solve_three_state_tradeability(
        prior=prior,
        payoff_model=payoff,
        observation_cost=0.002,
        horizon=1.0,
        evidence_max=4.0,
        activation_decay=1.5,
        n_time=81,
        n_evidence=121,
    )
    regions = solution.decision_regions()
    center = solution.evidence.size // 2

    assert regions[-1, center] == -1
    assert regions[0, 0] == 1
    assert regions[0, -1] == 1
    assert np.any(regions[0] == 0)
