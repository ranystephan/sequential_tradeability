"""Bayesian drift-estimation identities from Ekstrom-Karatzas-Vaicenavicius.

The observation model is

    Y_t = X t + W_t,

where X is a prior random variable and W is standard Brownian motion.  This module
implements the posterior formulas used in the first project step.  It deliberately keeps
the mathematical objects close to the paper: posterior mean G(t, y), posterior variance
H(t, y), the Gaussian stopping time, and the Bernoulli free-boundary condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number; got {value!r}")


@dataclass(frozen=True)
class GaussianPosterior:
    """Conjugate posterior for a Gaussian prior under Y_t = X t + W_t."""

    prior_mean: float
    prior_variance: float

    def __post_init__(self) -> None:
        _require_positive("prior_variance", self.prior_variance)

    def mean(self, t: float, y: float) -> float:
        """Return E[X | Y_t = y]."""
        if t < 0:
            raise ValueError("t must be nonnegative")
        return (self.prior_mean + self.prior_variance * y) / (
            1.0 + self.prior_variance * t
        )

    def variance(self, t: float) -> float:
        """Return Var(X | Y_t), independent of y in the Gaussian case."""
        if t < 0:
            raise ValueError("t must be nonnegative")
        return self.prior_variance / (1.0 + self.prior_variance * t)

    def learning_rate(self, t: float) -> float:
        """Return Psi(t)^2, the instantaneous posterior-variance decay rate."""
        return self.variance(t) ** 2


def gaussian_stopping_time(prior_variance: float, observation_cost: float) -> float:
    """Optimal EKV stopping time for a Gaussian prior.

    For a Gaussian prior, posterior variance is deterministic:

        Psi(t) = sigma^2 / (1 + sigma^2 t).

    The optimal rule stops when Psi(t) reaches sqrt(c), unless the initial variance is
    already below that threshold.
    """

    _require_positive("prior_variance", prior_variance)
    _require_positive("observation_cost", observation_cost)
    return max(1.0 / np.sqrt(observation_cost) - 1.0 / prior_variance, 0.0)


@dataclass(frozen=True)
class BernoulliPosterior:
    """Posterior for X in {-beta, beta} under Y_t = X t + W_t."""

    beta: float
    prob_positive: float

    def __post_init__(self) -> None:
        _require_positive("beta", self.beta)
        if not 0.0 < self.prob_positive < 1.0:
            raise ValueError("prob_positive must be strictly between 0 and 1")

    def prob_plus(self, y: float) -> float:
        """Return P[X = beta | Y_t = y].

        The t-dependent likelihood terms cancel because both support points have the
        same square u^2 = beta^2.
        """
        log_p_plus = np.log(self.prob_positive) + self.beta * y
        log_p_minus = np.log1p(-self.prob_positive) - self.beta * y
        return float(np.exp(log_p_plus - logsumexp([log_p_plus, log_p_minus])))

    def mean(self, y: float) -> float:
        """Return E[X | Y_t = y]."""
        return self.beta * (2.0 * self.prob_plus(y) - 1.0)

    def variance(self, y: float) -> float:
        """Return Var(X | Y_t = y)."""
        mean = self.mean(y)
        return self.variance_from_mean(self.beta, mean)

    @staticmethod
    def variance_from_mean(beta: float, posterior_mean: float) -> float:
        """Return beta^2 - x^2 for posterior mean x."""
        _require_positive("beta", beta)
        if abs(posterior_mean) > beta * (1.0 + 1e-12):
            raise ValueError("posterior_mean must lie in [-beta, beta]")
        variance = beta**2 - posterior_mean**2
        return max(float(variance), 0.0)


def bernoulli_boundary(beta: float, observation_cost: float) -> float:
    """Compute the symmetric Bernoulli optimal stopping boundary.

    The EKV Bernoulli example has psi(x) = beta^2 - x^2.  If beta^4 <= c, immediate
    stopping is optimal.  Otherwise the boundary a in (gamma, beta) solves

        integral_0^a (c - psi(x)^2) / psi(x)^2 dx = 0,

    where gamma = sqrt(beta^2 - sqrt(c)).
    """

    _require_positive("beta", beta)
    _require_positive("observation_cost", observation_cost)
    if beta**4 <= observation_cost:
        return 0.0

    gamma = np.sqrt(beta**2 - np.sqrt(observation_cost))

    def inverse_variance_integral(a: float) -> float:
        """Return integral_0^a 1 / (beta^2 - x^2)^2 dx."""
        return a / (2.0 * beta**2 * (beta**2 - a**2)) + np.arctanh(a / beta) / (
            2.0 * beta**3
        )

    def root_function(a: float) -> float:
        return observation_cost * inverse_variance_integral(a) - a

    lower = np.nextafter(gamma, beta)
    upper = beta * (1.0 - 1e-12)
    return float(brentq(root_function, lower, upper, xtol=1e-11, rtol=1e-11))


@dataclass(frozen=True)
class DiscretePrior:
    """Finite-support prior for direct posterior moment calculations."""

    support: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if support.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("support and probabilities must be one-dimensional")
        if support.shape != probabilities.shape:
            raise ValueError("support and probabilities must have the same shape")
        if support.size < 2:
            raise ValueError("support must contain at least two points")
        if np.any(probabilities <= 0):
            raise ValueError("all probabilities must be positive")
        total = probabilities.sum()
        if not np.isclose(total, 1.0):
            raise ValueError("probabilities must sum to one")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "probabilities", probabilities)

    def posterior_probabilities(self, t: float, y: float) -> np.ndarray:
        """Return posterior probabilities over the finite support."""
        if t < 0:
            raise ValueError("t must be nonnegative")
        log_weights = np.log(self.probabilities) + self.support * y - 0.5 * self.support**2 * t
        return np.exp(log_weights - logsumexp(log_weights))

    def mean(self, t: float, y: float) -> float:
        """Return posterior mean G(t, y)."""
        weights = self.posterior_probabilities(t, y)
        return float(np.dot(weights, self.support))

    def variance(self, t: float, y: float) -> float:
        """Return posterior variance H(t, y)."""
        weights = self.posterior_probabilities(t, y)
        mean = float(np.dot(weights, self.support))
        second = float(np.dot(weights, self.support**2))
        return max(second - mean**2, 0.0)
