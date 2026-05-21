import numpy as np
import pytest

from sequential_tradeability.ekv import (
    BernoulliPosterior,
    DiscretePrior,
    GaussianPosterior,
    ThreeStatePrior,
    bernoulli_boundary,
    gaussian_stopping_time,
)


def test_gaussian_posterior_matches_precision_update() -> None:
    posterior = GaussianPosterior(prior_mean=0.7, prior_variance=2.5)
    t = 1.3
    y = -0.4

    posterior_precision = 1.0 / 2.5 + t
    expected_variance = 1.0 / posterior_precision
    expected_mean = expected_variance * (0.7 / 2.5 + y)

    assert posterior.mean(t, y) == pytest.approx(expected_mean)
    assert posterior.variance(t) == pytest.approx(expected_variance)


def test_gaussian_stopping_time_stops_when_variance_hits_sqrt_cost() -> None:
    prior_variance = 4.0
    observation_cost = 0.25
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=prior_variance)

    tau_star = gaussian_stopping_time(prior_variance, observation_cost)

    assert tau_star == pytest.approx(1.75)
    assert posterior.variance(tau_star) == pytest.approx(np.sqrt(observation_cost))


def test_gaussian_variance_identity_is_exact_for_deterministic_tau() -> None:
    prior_variance = 1.8
    tau = 2.2
    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=prior_variance)

    integrated_learning = (
        prior_variance - prior_variance / (1.0 + prior_variance * tau)
    )

    assert posterior.variance(tau) == pytest.approx(prior_variance - integrated_learning)


def test_bernoulli_posterior_variance_is_beta_squared_minus_mean_squared() -> None:
    posterior = BernoulliPosterior(beta=1.4, prob_positive=0.35)
    y = 0.8
    mean = posterior.mean(y)

    assert posterior.variance(y) == pytest.approx(1.4**2 - mean**2)


def test_bernoulli_boundary_solves_free_boundary_integral() -> None:
    beta = 1.0
    observation_cost = 0.08

    boundary = bernoulli_boundary(beta, observation_cost)
    gamma = np.sqrt(beta**2 - np.sqrt(observation_cost))

    assert gamma < boundary < beta

    grid = np.linspace(0.0, boundary, 20_001)
    psi = beta**2 - grid**2
    integral = np.trapezoid((observation_cost - psi**2) / psi**2, grid)

    assert integral == pytest.approx(0.0, abs=2e-4)


def test_discrete_prior_derivative_of_posterior_mean_is_posterior_variance() -> None:
    prior = DiscretePrior(
        support=np.array([-1.5, -0.2, 0.8, 2.0]),
        probabilities=np.array([0.2, 0.3, 0.1, 0.4]),
    )
    t = 0.9
    y = 0.35
    step = 1e-5

    finite_difference = (prior.mean(t, y + step) - prior.mean(t, y - step)) / (2.0 * step)

    assert finite_difference == pytest.approx(prior.variance(t, y), rel=1e-7, abs=1e-7)


def test_three_state_posterior_probabilities_are_normalized() -> None:
    prior = ThreeStatePrior(
        alpha=0.45,
        prob_negative=0.2,
        prob_dead=0.55,
        prob_positive=0.25,
    )
    y_grid = np.linspace(-3.0, 3.0, 31)

    posterior = prior.posterior_probabilities(t=4.0, y=y_grid)

    assert posterior.shape == (31, 3)
    assert np.all(posterior > 0.0)
    assert np.allclose(posterior.sum(axis=1), 1.0)


def test_three_state_prior_is_symmetric_when_prior_is_symmetric() -> None:
    prior = ThreeStatePrior(
        alpha=0.8,
        prob_negative=0.25,
        prob_dead=0.5,
        prob_positive=0.25,
    )
    t = 1.7
    y = 0.6

    assert prior.posterior_positive(t, y) == pytest.approx(
        prior.posterior_negative(t, -y)
    )
    assert prior.posterior_dead(t, y) == pytest.approx(prior.posterior_dead(t, -y))
    assert prior.mean(t, y) == pytest.approx(-prior.mean(t, -y))


def test_three_state_dead_probability_increases_when_evidence_stays_flat() -> None:
    prior = ThreeStatePrior(
        alpha=0.7,
        prob_negative=0.3,
        prob_dead=0.4,
        prob_positive=0.3,
    )

    dead_probabilities = np.array([prior.posterior_dead(t, y=0.0) for t in [0.0, 1.0, 3.0]])

    assert np.all(np.diff(dead_probabilities) > 0.0)
    assert dead_probabilities[0] == pytest.approx(0.4)


def test_three_state_large_evidence_identifies_direction() -> None:
    prior = ThreeStatePrior(
        alpha=0.5,
        prob_negative=0.3,
        prob_dead=0.4,
        prob_positive=0.3,
    )

    assert prior.posterior_positive(t=2.0, y=100.0) == pytest.approx(1.0)
    assert prior.posterior_negative(t=2.0, y=-100.0) == pytest.approx(1.0)


def test_three_state_derivative_of_posterior_mean_is_posterior_variance() -> None:
    prior = ThreeStatePrior(
        alpha=0.9,
        prob_negative=0.15,
        prob_dead=0.65,
        prob_positive=0.2,
    )
    t = 2.4
    y = 0.35
    step = 1e-5

    finite_difference = (prior.mean(t, y + step) - prior.mean(t, y - step)) / (
        2.0 * step
    )

    assert finite_difference == pytest.approx(prior.variance(t, y), rel=1e-7, abs=1e-7)


def test_three_state_innovation_increment_is_centered_under_posterior() -> None:
    prior = ThreeStatePrior(
        alpha=0.6,
        prob_negative=0.2,
        prob_dead=0.5,
        prob_positive=0.3,
    )
    t = 1.2
    y = -0.25
    dt = 0.01
    posterior = prior.posterior_probabilities(t, y)
    conditional_drift_residuals = prior.support - prior.mean(t, y)

    assert np.dot(posterior, conditional_drift_residuals * dt) == pytest.approx(0.0)
    assert np.dot(posterior, conditional_drift_residuals**2 * dt) == pytest.approx(
        prior.variance(t, y) * dt
    )
