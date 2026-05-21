"""Simulation harness for the Gaussian evidence model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sequential_tradeability.ekv import GaussianPosterior, ThreeStatePrior, _require_positive
from sequential_tradeability.tradeability import (
    StoppingSolution,
    ThreeStateStoppingSolution,
    TradeabilityPayoff,
)


@dataclass(frozen=True)
class EvidencePaths:
    """Simulated observations and posterior means under Y_t = theta t + W_t."""

    times: np.ndarray
    true_theta: np.ndarray
    observations: np.ndarray
    posterior_means: np.ndarray
    posterior_variance: np.ndarray


@dataclass(frozen=True)
class ThreeStateEvidencePaths:
    """Simulated evidence and three-state posterior moments."""

    times: np.ndarray
    true_theta: np.ndarray
    observations: np.ndarray
    posterior_means: np.ndarray
    posterior_variance: np.ndarray
    posterior_dead: np.ndarray


@dataclass(frozen=True)
class PolicyOutcome:
    """Pathwise stopping decisions and realized economic values."""

    name: str
    stop_index: np.ndarray
    decision: np.ndarray
    posterior_mean_at_stop: np.ndarray
    realized_value: np.ndarray

    @property
    def activation_rate(self) -> float:
        return float(np.mean(self.decision == 1))

    @property
    def rejection_rate(self) -> float:
        return float(np.mean(self.decision == -1))

    def mean_stop_time(self, times: np.ndarray) -> float:
        return float(np.mean(times[self.stop_index]))

    @property
    def mean_realized_value(self) -> float:
        return float(np.mean(self.realized_value))

    @property
    def stderr_realized_value(self) -> float:
        if self.realized_value.size < 2:
            return 0.0
        return float(np.std(self.realized_value, ddof=1) / np.sqrt(self.realized_value.size))


def simulate_gaussian_evidence(
    *,
    posterior: GaussianPosterior,
    horizon: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    true_theta: float | None = None,
) -> EvidencePaths:
    """Simulate evidence paths and exact Gaussian posterior means."""
    _require_positive("horizon", horizon)
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, horizon, n_steps + 1)
    dt = times[1] - times[0]

    if true_theta is None:
        theta = rng.normal(
            loc=posterior.prior_mean,
            scale=np.sqrt(posterior.prior_variance),
            size=n_paths,
        )
    else:
        theta = np.full(n_paths, true_theta, dtype=float)

    increments = theta[:, None] * dt + np.sqrt(dt) * rng.normal(size=(n_paths, n_steps))
    observations = np.zeros((n_paths, n_steps + 1))
    observations[:, 1:] = np.cumsum(increments, axis=1)

    denominator = 1.0 + posterior.prior_variance * times
    posterior_means = (
        posterior.prior_mean + posterior.prior_variance * observations
    ) / denominator[None, :]
    posterior_variance = posterior.prior_variance / denominator

    return EvidencePaths(
        times=times,
        true_theta=theta,
        observations=observations,
        posterior_means=posterior_means,
        posterior_variance=posterior_variance,
    )


def simulate_three_state_evidence(
    *,
    prior: ThreeStatePrior,
    horizon: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    true_theta: float | None = None,
) -> ThreeStateEvidencePaths:
    """Simulate evidence paths and exact three-state posterior moments."""
    _require_positive("horizon", horizon)
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, horizon, n_steps + 1)
    dt = times[1] - times[0]

    if true_theta is None:
        theta = rng.choice(prior.support, size=n_paths, p=prior.probabilities)
    else:
        theta = np.full(n_paths, true_theta, dtype=float)

    increments = theta[:, None] * dt + np.sqrt(dt) * rng.normal(size=(n_paths, n_steps))
    observations = np.zeros((n_paths, n_steps + 1))
    observations[:, 1:] = np.cumsum(increments, axis=1)

    posterior_means = np.empty_like(observations)
    posterior_variance = np.empty_like(observations)
    posterior_dead = np.empty_like(observations)
    for time_index, time in enumerate(times):
        y = observations[:, time_index]
        posterior_means[:, time_index] = prior.mean(time, y)
        posterior_variance[:, time_index] = prior.variance(time, y)
        posterior_dead[:, time_index] = prior.posterior_dead(time, y)

    return ThreeStateEvidencePaths(
        times=times,
        true_theta=theta,
        observations=observations,
        posterior_means=posterior_means,
        posterior_variance=posterior_variance,
        posterior_dead=posterior_dead,
    )


def realized_activation_value(
    *,
    true_theta: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    stop_time: np.ndarray,
    decision: np.ndarray,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    activation_decay: float,
    dead_activation_penalty: float = 0.0,
) -> np.ndarray:
    """Evaluate realized value using the exposure chosen from the posterior belief."""
    _require_positive("observation_cost", observation_cost)
    if not np.isfinite(activation_decay) or activation_decay < 0:
        raise ValueError("activation_decay must be nonnegative and finite")
    if not np.isfinite(dead_activation_penalty) or dead_activation_penalty < 0:
        raise ValueError("dead_activation_penalty must be nonnegative and finite")

    exposure = posterior_mean / (
        payoff_model.risk_aversion * (payoff_model.return_variance + posterior_variance)
    )
    false_discovery_cost = dead_activation_penalty * np.isclose(true_theta, 0.0)
    activation_value = np.exp(-activation_decay * stop_time) * (
        true_theta * exposure
        - 0.5 * payoff_model.risk_aversion * payoff_model.return_variance * exposure**2
        - payoff_model.implementation_hurdle
        - false_discovery_cost
    )
    value = np.where(decision == 1, activation_value, 0.0)
    return value - observation_cost * stop_time


def evaluate_boundary_policy(
    *,
    solution: StoppingSolution,
    paths: EvidencePaths,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    activation_decay: float,
    dead_activation_penalty: float = 0.0,
    name: str = "optimal boundary",
) -> PolicyOutcome:
    """Evaluate the finite-difference stopping boundary on simulated paths."""
    regions = solution.decision_regions()
    stop_index = np.full(paths.true_theta.shape, paths.times.size - 1, dtype=int)
    decision = np.full(paths.true_theta.shape, -1, dtype=int)
    stopped = np.zeros(paths.true_theta.shape, dtype=bool)

    for time_index in range(paths.times.size):
        mean_index = np.searchsorted(solution.grid.means, paths.posterior_means[:, time_index])
        mean_index = np.clip(mean_index, 1, solution.grid.means.size - 1)
        left = solution.grid.means[mean_index - 1]
        right = solution.grid.means[mean_index]
        choose_right = np.abs(paths.posterior_means[:, time_index] - right) < np.abs(
            paths.posterior_means[:, time_index] - left
        )
        nearest = mean_index - 1 + choose_right.astype(int)

        path_region = regions[time_index, nearest]
        active = (~stopped) & (path_region != 0)
        stop_index[active] = time_index
        decision[active] = path_region[active]
        stopped[active] = True

    posterior_mean_at_stop = paths.posterior_means[np.arange(paths.true_theta.size), stop_index]
    posterior_variance_at_stop = paths.posterior_variance[stop_index]
    stop_time = paths.times[stop_index]
    realized_value = realized_activation_value(
        true_theta=paths.true_theta,
        posterior_mean=posterior_mean_at_stop,
        posterior_variance=posterior_variance_at_stop,
        stop_time=stop_time,
        decision=decision,
        payoff_model=payoff_model,
        observation_cost=observation_cost,
        activation_decay=activation_decay,
        dead_activation_penalty=dead_activation_penalty,
    )
    return PolicyOutcome(
        name=name,
        stop_index=stop_index,
        decision=decision,
        posterior_mean_at_stop=posterior_mean_at_stop,
        realized_value=realized_value,
    )


def evaluate_three_state_boundary_policy(
    *,
    solution: ThreeStateStoppingSolution,
    paths: ThreeStateEvidencePaths,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    activation_decay: float,
    dead_activation_penalty: float = 0.0,
    name: str = "dead/alive boundary",
) -> PolicyOutcome:
    """Evaluate the three-state stopping boundary on simulated evidence paths."""
    regions = solution.decision_regions()
    stop_index = np.full(paths.true_theta.shape, paths.times.size - 1, dtype=int)
    decision = np.full(paths.true_theta.shape, -1, dtype=int)
    stopped = np.zeros(paths.true_theta.shape, dtype=bool)

    for time_index in range(paths.times.size):
        evidence_index = np.searchsorted(solution.evidence, paths.observations[:, time_index])
        evidence_index = np.clip(evidence_index, 1, solution.evidence.size - 1)
        left = solution.evidence[evidence_index - 1]
        right = solution.evidence[evidence_index]
        choose_right = np.abs(paths.observations[:, time_index] - right) < np.abs(
            paths.observations[:, time_index] - left
        )
        nearest = evidence_index - 1 + choose_right.astype(int)

        path_region = regions[time_index, nearest]
        active = (~stopped) & (path_region != 0)
        stop_index[active] = time_index
        decision[active] = path_region[active]
        stopped[active] = True

    posterior_mean_at_stop = paths.posterior_means[np.arange(paths.true_theta.size), stop_index]
    posterior_variance_at_stop = paths.posterior_variance[
        np.arange(paths.true_theta.size), stop_index
    ]
    stop_time = paths.times[stop_index]
    realized_value = realized_activation_value(
        true_theta=paths.true_theta,
        posterior_mean=posterior_mean_at_stop,
        posterior_variance=posterior_variance_at_stop,
        stop_time=stop_time,
        decision=decision,
        payoff_model=payoff_model,
        observation_cost=observation_cost,
        activation_decay=activation_decay,
        dead_activation_penalty=dead_activation_penalty,
    )
    return PolicyOutcome(
        name=name,
        stop_index=stop_index,
        decision=decision,
        posterior_mean_at_stop=posterior_mean_at_stop,
        realized_value=realized_value,
    )


def evaluate_three_state_static_payoff_policy(
    *,
    paths: ThreeStateEvidencePaths,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    activation_decay: float,
    dead_activation_penalty: float = 0.0,
    name: str = "three-state static payoff threshold",
) -> PolicyOutcome:
    """Activate once the three-state immediate posterior payoff is positive."""
    stop_index = np.full(paths.true_theta.shape, paths.times.size - 1, dtype=int)
    decision = np.full(paths.true_theta.shape, -1, dtype=int)

    for time_index in range(paths.times.size):
        immediate_payoff = payoff_model.value(
            paths.posterior_means[:, time_index],
            paths.posterior_variance[:, time_index],
        )
        immediate_payoff = np.maximum(
            immediate_payoff - dead_activation_penalty * paths.posterior_dead[:, time_index],
            0.0,
        )
        activate = immediate_payoff > 0.0
        active = (decision == -1) & activate
        stop_index[active] = time_index
        decision[active] = 1

    posterior_mean_at_stop = paths.posterior_means[np.arange(paths.true_theta.size), stop_index]
    posterior_variance_at_stop = paths.posterior_variance[
        np.arange(paths.true_theta.size), stop_index
    ]
    stop_time = paths.times[stop_index]
    realized_value = realized_activation_value(
        true_theta=paths.true_theta,
        posterior_mean=posterior_mean_at_stop,
        posterior_variance=posterior_variance_at_stop,
        stop_time=stop_time,
        decision=decision,
        payoff_model=payoff_model,
        observation_cost=observation_cost,
        activation_decay=activation_decay,
        dead_activation_penalty=dead_activation_penalty,
    )
    return PolicyOutcome(
        name=name,
        stop_index=stop_index,
        decision=decision,
        posterior_mean_at_stop=posterior_mean_at_stop,
        realized_value=realized_value,
    )


def evaluate_three_state_zstat_policy(
    *,
    paths: ThreeStateEvidencePaths,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    activation_decay: float,
    z_threshold: float,
    dead_activation_penalty: float = 0.0,
    name: str | None = None,
) -> PolicyOutcome:
    """Activate once |posterior mean| / posterior sd exceeds a fixed threshold."""
    _require_positive("z_threshold", z_threshold)
    stop_index = np.full(paths.true_theta.shape, paths.times.size - 1, dtype=int)
    decision = np.full(paths.true_theta.shape, -1, dtype=int)

    for time_index in range(paths.times.size):
        posterior_sd = np.sqrt(np.maximum(paths.posterior_variance[:, time_index], 1e-15))
        activate = np.abs(paths.posterior_means[:, time_index]) / posterior_sd >= z_threshold
        active = (decision == -1) & activate
        stop_index[active] = time_index
        decision[active] = 1

    posterior_mean_at_stop = paths.posterior_means[np.arange(paths.true_theta.size), stop_index]
    posterior_variance_at_stop = paths.posterior_variance[
        np.arange(paths.true_theta.size), stop_index
    ]
    stop_time = paths.times[stop_index]
    realized_value = realized_activation_value(
        true_theta=paths.true_theta,
        posterior_mean=posterior_mean_at_stop,
        posterior_variance=posterior_variance_at_stop,
        stop_time=stop_time,
        decision=decision,
        payoff_model=payoff_model,
        observation_cost=observation_cost,
        activation_decay=activation_decay,
        dead_activation_penalty=dead_activation_penalty,
    )
    return PolicyOutcome(
        name=name or f"three-state z-stat {z_threshold:g}",
        stop_index=stop_index,
        decision=decision,
        posterior_mean_at_stop=posterior_mean_at_stop,
        realized_value=realized_value,
    )


def evaluate_static_payoff_policy(
    *,
    paths: EvidencePaths,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    activation_decay: float,
    dead_activation_penalty: float = 0.0,
    name: str = "static payoff threshold",
) -> PolicyOutcome:
    """Activate once the immediate posterior payoff is positive; otherwise reject at T."""
    stop_index = np.full(paths.true_theta.shape, paths.times.size - 1, dtype=int)
    decision = np.full(paths.true_theta.shape, -1, dtype=int)

    for time_index, q in enumerate(paths.posterior_variance):
        threshold = payoff_model.activation_threshold(q)
        activate = np.abs(paths.posterior_means[:, time_index]) >= threshold
        active = (decision == -1) & activate
        stop_index[active] = time_index
        decision[active] = 1

    posterior_mean_at_stop = paths.posterior_means[np.arange(paths.true_theta.size), stop_index]
    posterior_variance_at_stop = paths.posterior_variance[stop_index]
    stop_time = paths.times[stop_index]
    realized_value = realized_activation_value(
        true_theta=paths.true_theta,
        posterior_mean=posterior_mean_at_stop,
        posterior_variance=posterior_variance_at_stop,
        stop_time=stop_time,
        decision=decision,
        payoff_model=payoff_model,
        observation_cost=observation_cost,
        activation_decay=activation_decay,
        dead_activation_penalty=dead_activation_penalty,
    )
    return PolicyOutcome(
        name=name,
        stop_index=stop_index,
        decision=decision,
        posterior_mean_at_stop=posterior_mean_at_stop,
        realized_value=realized_value,
    )


def evaluate_zstat_policy(
    *,
    paths: EvidencePaths,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    activation_decay: float,
    z_threshold: float,
    dead_activation_penalty: float = 0.0,
    name: str | None = None,
) -> PolicyOutcome:
    """Activate once |posterior mean| / posterior sd exceeds a fixed threshold."""
    _require_positive("z_threshold", z_threshold)
    stop_index = np.full(paths.true_theta.shape, paths.times.size - 1, dtype=int)
    decision = np.full(paths.true_theta.shape, -1, dtype=int)

    for time_index, q in enumerate(paths.posterior_variance):
        activate = np.abs(paths.posterior_means[:, time_index]) / np.sqrt(q) >= z_threshold
        active = (decision == -1) & activate
        stop_index[active] = time_index
        decision[active] = 1

    posterior_mean_at_stop = paths.posterior_means[np.arange(paths.true_theta.size), stop_index]
    posterior_variance_at_stop = paths.posterior_variance[stop_index]
    stop_time = paths.times[stop_index]
    realized_value = realized_activation_value(
        true_theta=paths.true_theta,
        posterior_mean=posterior_mean_at_stop,
        posterior_variance=posterior_variance_at_stop,
        stop_time=stop_time,
        decision=decision,
        payoff_model=payoff_model,
        observation_cost=observation_cost,
        activation_decay=activation_decay,
        dead_activation_penalty=dead_activation_penalty,
    )
    return PolicyOutcome(
        name=name or f"z-stat {z_threshold:g}",
        stop_index=stop_index,
        decision=decision,
        posterior_mean_at_stop=posterior_mean_at_stop,
        realized_value=realized_value,
    )
