"""Tradeability payoff and finite-difference stopping solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from sequential_tradeability.ekv import GaussianPosterior, ThreeStatePrior, _require_positive


@dataclass(frozen=True)
class TradeabilityPayoff:
    """Mean-variance activation value for a signal with posterior uncertainty."""

    risk_aversion: float
    return_variance: float
    implementation_hurdle: float

    def __post_init__(self) -> None:
        _require_positive("risk_aversion", self.risk_aversion)
        _require_positive("return_variance", self.return_variance)
        if not np.isfinite(self.implementation_hurdle) or self.implementation_hurdle < 0:
            raise ValueError("implementation_hurdle must be nonnegative and finite")

    def gross_value(
        self,
        posterior_mean: np.ndarray,
        posterior_variance: float | np.ndarray,
    ) -> np.ndarray:
        """Return m^2 / (2 lambda (sigma^2 + q))."""
        posterior_variance = np.asarray(posterior_variance, dtype=float)
        if np.any(~np.isfinite(posterior_variance)) or np.any(posterior_variance < 0.0):
            raise ValueError("posterior_variance must be nonnegative and finite")
        denominator = 2.0 * self.risk_aversion * (self.return_variance + posterior_variance)
        return np.asarray(posterior_mean, dtype=float) ** 2 / denominator

    def value(
        self,
        posterior_mean: np.ndarray,
        posterior_variance: float | np.ndarray,
    ) -> np.ndarray:
        """Return max(gross value minus implementation hurdle, reject value)."""
        net = self.gross_value(posterior_mean, posterior_variance) - self.implementation_hurdle
        return np.maximum(net, 0.0)

    def activation_threshold(self, posterior_variance: float) -> float:
        """Return the smallest absolute posterior mean with positive payoff."""
        _require_positive("posterior_variance", posterior_variance)
        if self.implementation_hurdle == 0.0:
            return 0.0
        return float(
            np.sqrt(
                2.0
                * self.risk_aversion
                * (self.return_variance + posterior_variance)
                * self.implementation_hurdle
            )
        )


@dataclass(frozen=True)
class StoppingGrid:
    """Uniform time/mean grid for the Gaussian-belief stopping problem."""

    times: np.ndarray
    means: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])

    @property
    def dm(self) -> float:
        return float(self.means[1] - self.means[0])


@dataclass(frozen=True)
class StoppingSolution:
    """Numerical solution of the finite-horizon tradeability stopping problem."""

    grid: StoppingGrid
    values: np.ndarray
    payoff: np.ndarray
    stop: np.ndarray
    posterior_variance: np.ndarray

    def value_at_initial_mean(self, initial_mean: float) -> float:
        """Interpolate V(0, m0) on the computed grid."""
        return float(np.interp(initial_mean, self.grid.means, self.values[0]))

    def decision_regions(self, payoff_tolerance: float = 1e-10) -> np.ndarray:
        """Classify each grid point as reject, observe, or activate.

        Returns an integer array with -1 for stop/reject, 0 for continue observing,
        and 1 for stop/activate.
        """
        regions = np.zeros_like(self.values, dtype=int)
        regions[self.stop & (self.payoff <= payoff_tolerance)] = -1
        regions[self.stop & (self.payoff > payoff_tolerance)] = 1
        return regions


@dataclass(frozen=True)
class ThreeStateStoppingSolution:
    """Numerical solution of the dead/alive evidence-space stopping problem."""

    times: np.ndarray
    evidence: np.ndarray
    values: np.ndarray
    payoff: np.ndarray
    stop: np.ndarray
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    posterior_dead: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])

    @property
    def dy(self) -> float:
        return float(self.evidence[1] - self.evidence[0])

    def value_at_initial_evidence(self, initial_evidence: float) -> float:
        """Interpolate V(0, y0) on the computed grid."""
        return float(np.interp(initial_evidence, self.evidence, self.values[0]))

    def decision_regions(self, payoff_tolerance: float = 1e-10) -> np.ndarray:
        """Classify each grid point as reject, observe, or activate."""
        regions = np.zeros_like(self.values, dtype=int)
        regions[self.stop & (self.payoff <= payoff_tolerance)] = -1
        regions[self.stop & (self.payoff > payoff_tolerance)] = 1
        return regions


def make_grid(horizon: float, n_time: int, mean_max: float, n_mean: int) -> StoppingGrid:
    """Create a symmetric uniform grid."""
    _require_positive("horizon", horizon)
    _require_positive("mean_max", mean_max)
    if n_time < 2:
        raise ValueError("n_time must be at least 2")
    if n_mean < 5 or n_mean % 2 == 0:
        raise ValueError("n_mean must be an odd integer at least 5")
    return StoppingGrid(
        times=np.linspace(0.0, horizon, n_time),
        means=np.linspace(-mean_max, mean_max, n_mean),
    )


def solve_gaussian_tradeability(
    *,
    posterior: GaussianPosterior,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    horizon: float,
    mean_max: float,
    activation_decay: float = 0.0,
    n_time: int = 401,
    n_mean: int = 401,
) -> StoppingSolution:
    """Solve the finite-horizon Gaussian-belief optimal stopping problem.

    The posterior mean follows

        dm_t = q(t) dW_t,  q(t) = prior_variance / (1 + prior_variance t).

    The reward-form variational inequality is

        V(t,m) = max(exp(-rho t) Phi(t,m), continuation value),
        V_t + 0.5 q(t)^2 V_mm - c = 0      in the continuation region,
        V(T,m) = exp(-rho T) Phi(T,m).

    We use backward implicit Euler on a uniform mean grid. The mean-domain boundaries
    are pinned to the immediate stopping payoff; choose mean_max wide enough that
    stopping is optimal near the edges.
    """
    _require_positive("observation_cost", observation_cost)
    if not np.isfinite(activation_decay) or activation_decay < 0:
        raise ValueError("activation_decay must be nonnegative and finite")
    grid = make_grid(horizon, n_time, mean_max, n_mean)
    n_t = grid.times.size
    n_m = grid.means.size

    posterior_variance = np.array([posterior.variance(t) for t in grid.times])
    payoff = np.vstack(
        [
            np.exp(-activation_decay * t) * payoff_model.value(grid.means, q)
            for t, q in zip(grid.times, posterior_variance, strict=True)
        ]
    )
    values = np.empty_like(payoff)
    values[-1] = payoff[-1]

    dt = grid.dt
    dm = grid.dm
    interior_count = n_m - 2

    for time_index in range(n_t - 2, -1, -1):
        q = posterior_variance[time_index]
        diffusion_number = 0.5 * q**2 * dt / dm**2

        diagonal = np.full(interior_count, 1.0 + 2.0 * diffusion_number)
        off_diagonal = np.full(interior_count - 1, -diffusion_number)

        rhs = values[time_index + 1, 1:-1] - observation_cost * dt
        rhs[0] += diffusion_number * payoff[time_index, 0]
        rhs[-1] += diffusion_number * payoff[time_index, -1]

        banded = np.zeros((3, interior_count))
        banded[0, 1:] = off_diagonal
        banded[1] = diagonal
        banded[2, :-1] = off_diagonal
        continuation = solve_banded((1, 1), banded, rhs)

        values[time_index, 0] = payoff[time_index, 0]
        values[time_index, -1] = payoff[time_index, -1]
        values[time_index, 1:-1] = np.maximum(payoff[time_index, 1:-1], continuation)

    stop = np.isclose(values, payoff, rtol=1e-8, atol=1e-10)
    return StoppingSolution(
        grid=grid,
        values=values,
        payoff=payoff,
        stop=stop,
        posterior_variance=posterior_variance,
    )


def solve_three_state_tradeability(
    *,
    prior: ThreeStatePrior,
    payoff_model: TradeabilityPayoff,
    observation_cost: float,
    horizon: float,
    evidence_max: float,
    activation_decay: float = 0.0,
    dead_activation_penalty: float = 0.0,
    n_time: int = 401,
    n_evidence: int = 401,
) -> ThreeStateStoppingSolution:
    """Solve the finite-horizon dead/alive tradeability stopping problem.

    The evidence process has posterior-predictive dynamics

        dY_t = m(t, Y_t) dt + dW_hat_t,

    where m(t, y) is the three-state posterior mean. The reward-form variational
    inequality is

        max(exp(-rho t) Phi(m(t,y), q(t,y), pi_0(t,y)) - V,
            V_t + m(t,y) V_y + 0.5 V_yy - c) = 0.

    The optional dead_activation_penalty eta prices a false discovery cost paid when an
    activated signal is truly dead. Its posterior expected cost is eta * P(theta = 0 |
    Y_t = y), so the terminal activation payoff is

        (gross_trade_value - implementation_hurdle - eta * posterior_dead)^+.

    We discretize the generator with nonnegative nearest-neighbor transition rates:

        lambda_up   = 0.5 / dy^2 + max(m, 0) / dy,
        lambda_down = 0.5 / dy^2 + max(-m, 0) / dy.

    This gives a monotone implicit Euler continuation step before applying the obstacle.
    The evidence-domain boundaries are pinned to the immediate stopping payoff; choose
    evidence_max wide enough that stopping is optimal at the edges.
    """
    _require_positive("observation_cost", observation_cost)
    _require_positive("horizon", horizon)
    _require_positive("evidence_max", evidence_max)
    if not np.isfinite(activation_decay) or activation_decay < 0:
        raise ValueError("activation_decay must be nonnegative and finite")
    if not np.isfinite(dead_activation_penalty) or dead_activation_penalty < 0:
        raise ValueError("dead_activation_penalty must be nonnegative and finite")
    if n_time < 2:
        raise ValueError("n_time must be at least 2")
    if n_evidence < 5 or n_evidence % 2 == 0:
        raise ValueError("n_evidence must be an odd integer at least 5")

    times = np.linspace(0.0, horizon, n_time)
    evidence = np.linspace(-evidence_max, evidence_max, n_evidence)
    dt = float(times[1] - times[0])
    dy = float(evidence[1] - evidence[0])
    interior_count = n_evidence - 2

    posterior_mean = np.vstack([prior.mean(t, evidence) for t in times])
    posterior_variance = np.vstack([prior.variance(t, evidence) for t in times])
    posterior_dead = np.vstack([prior.posterior_dead(t, evidence) for t in times])
    payoff = np.maximum(
        payoff_model.gross_value(posterior_mean, posterior_variance)
        - payoff_model.implementation_hurdle
        - dead_activation_penalty * posterior_dead,
        0.0,
    )
    payoff *= np.exp(-activation_decay * times)[:, np.newaxis]

    values = np.empty_like(payoff)
    values[-1] = payoff[-1]

    for time_index in range(n_time - 2, -1, -1):
        drift = posterior_mean[time_index, 1:-1]
        rate_up = 0.5 / dy**2 + np.maximum(drift, 0.0) / dy
        rate_down = 0.5 / dy**2 + np.maximum(-drift, 0.0) / dy

        diagonal = 1.0 + dt * (rate_up + rate_down)
        upper = -dt * rate_up[:-1]
        lower = -dt * rate_down[1:]

        rhs = values[time_index + 1, 1:-1] - observation_cost * dt
        rhs[0] += dt * rate_down[0] * payoff[time_index, 0]
        rhs[-1] += dt * rate_up[-1] * payoff[time_index, -1]

        banded = np.zeros((3, interior_count))
        banded[0, 1:] = upper
        banded[1] = diagonal
        banded[2, :-1] = lower
        continuation = solve_banded((1, 1), banded, rhs)

        values[time_index, 0] = payoff[time_index, 0]
        values[time_index, -1] = payoff[time_index, -1]
        values[time_index, 1:-1] = np.maximum(payoff[time_index, 1:-1], continuation)

    stop = np.isclose(values, payoff, rtol=1e-8, atol=1e-10)
    return ThreeStateStoppingSolution(
        times=times,
        evidence=evidence,
        values=values,
        payoff=payoff,
        stop=stop,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        posterior_dead=posterior_dead,
    )
