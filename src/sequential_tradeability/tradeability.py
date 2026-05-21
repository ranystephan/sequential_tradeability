"""Tradeability payoff and finite-difference stopping solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from sequential_tradeability.ekv import GaussianPosterior, _require_positive


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

    def gross_value(self, posterior_mean: np.ndarray, posterior_variance: float) -> np.ndarray:
        """Return m^2 / (2 lambda (sigma^2 + q))."""
        _require_positive("posterior_variance", posterior_variance)
        denominator = 2.0 * self.risk_aversion * (self.return_variance + posterior_variance)
        return np.asarray(posterior_mean, dtype=float) ** 2 / denominator

    def value(self, posterior_mean: np.ndarray, posterior_variance: float) -> np.ndarray:
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
    n_time: int = 401,
    n_mean: int = 401,
) -> StoppingSolution:
    """Solve the finite-horizon Gaussian-belief optimal stopping problem.

    The posterior mean follows

        dm_t = q(t) dW_t,  q(t) = prior_variance / (1 + prior_variance t).

    The reward-form variational inequality is

        V(t,m) = max(Phi(t,m), continuation value),
        V_t + 0.5 q(t)^2 V_mm - c = 0      in the continuation region,
        V(T,m) = Phi(T,m).

    We use backward implicit Euler on a uniform mean grid. The mean-domain boundaries
    are pinned to the immediate stopping payoff; choose mean_max wide enough that
    stopping is optimal near the edges.
    """
    _require_positive("observation_cost", observation_cost)
    grid = make_grid(horizon, n_time, mean_max, n_mean)
    n_t = grid.times.size
    n_m = grid.means.size

    posterior_variance = np.array([posterior.variance(t) for t in grid.times])
    payoff = np.vstack([payoff_model.value(grid.means, q) for q in posterior_variance])
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

