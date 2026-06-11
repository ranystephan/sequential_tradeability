"""Generate the first Gaussian-belief tradeability boundary figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sequential_tradeability import (  # noqa: E402
    GaussianPosterior,
    TradeabilityPayoff,
    solve_gaussian_tradeability,
)


def main() -> None:
    figure_dir = REPO_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)

    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.10)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.025,
    )
    observation_cost = 0.005
    activation_decay = 1.0
    solution = solve_gaussian_tradeability(
        posterior=posterior,
        payoff_model=payoff,
        observation_cost=observation_cost,
        horizon=2.0,
        mean_max=1.0,
        activation_decay=activation_decay,
        n_time=301,
        n_mean=401,
    )

    regions = solution.decision_regions()
    times = solution.grid.times
    means = solution.grid.means
    thresholds = np.array([payoff.activation_threshold(q) for q in solution.posterior_variance])

    cmap = ListedColormap(["#d9d9d9", "#2f6fbb", "#b2182b"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.pcolormesh(times, means, regions.T, cmap=cmap, norm=norm, shading="auto")

    ax.plot(times, thresholds, color="black", linewidth=1.3, linestyle="--")
    ax.plot(times, -thresholds, color="black", linewidth=1.3, linestyle="--")
    ax.axhline(0.0, color="white", linewidth=0.8, alpha=0.8)

    # Overlay two illustrative posterior-mean paths dm_t = q(t) dW_t, truncated at
    # the first exit from the continuation region (one activates, one rejects).
    q0 = posterior.prior_variance
    horizon = times[-1]
    n_steps = 500
    dt = horizon / n_steps
    path_times = np.linspace(0.0, horizon, n_steps + 1)

    def simulate(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        path = np.zeros(n_steps + 1)
        for i in range(n_steps):
            q = q0 / (1.0 + q0 * path_times[i])
            path[i + 1] = path[i] + q * np.sqrt(dt) * rng.standard_normal()
        return path

    def region_at(t: float, m_value: float) -> int:
        ti = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
        mi = int(np.clip(np.searchsorted(means, m_value), 0, len(means) - 1))
        return int(regions[ti, mi])

    def first_exit(path: np.ndarray) -> tuple[int, int]:
        for k in range(1, len(path)):
            r = region_at(path_times[k], path[k])
            if r != 0:
                return k, r
        return len(path) - 1, 0

    activate_path = None
    reject_path = None
    for seed in range(400):
        path = simulate(seed)
        k, outcome = first_exit(path)
        if outcome == 1 and activate_path is None and 0.5 < path_times[k] < 1.9:
            activate_path = (path_times[: k + 1], path[: k + 1], path_times[k], path[k])
        if outcome == -1 and reject_path is None and path_times[k] > 0.95:
            reject_path = (path_times[: k + 1], path[: k + 1], path_times[k], path[k])
        if activate_path is not None and reject_path is not None:
            break

    for entry, exit_color, label, dy in (
        (activate_path, "#b2182b", "activate", 7),
        (reject_path, "#3a3a3a", "reject", -13),
    ):
        if entry is None:
            continue
        pt, pm, ex_t, ex_m = entry
        ax.plot(pt, pm, color="black", linewidth=1.6, solid_capstyle="round", zorder=4)
        ax.plot([ex_t], [ex_m], marker="o", color=exit_color, markersize=6, zorder=5)
        ax.annotate(
            label,
            (ex_t, ex_m),
            textcoords="offset points",
            xytext=(6, dy),
            color=exit_color,
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title("Sequential tradeability boundary, Gaussian belief")
    ax.set_xlabel("validation time")
    ax.set_ylabel("posterior mean alpha")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.25, 0.25)

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#d9d9d9", markersize=9),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#2f6fbb", markersize=9),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#b2182b", markersize=9),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.3),
    ]
    labels = [
        "stop: reject",
        "continue observing",
        "stop: activate",
        "static activation threshold",
    ]
    ax.legend(handles, labels, loc="upper right", frameon=True, fontsize=8)

    note = (
        rf"$q_0=0.10$, $\lambda=1$, $\sigma_r^2=0.05$, "
        rf"$\kappa=0.025$, $c={observation_cost:g}$, $\rho={activation_decay:g}$"
    )
    ax.text(
        0.02,
        0.03,
        note,
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )

    for extension in ("pdf", "png"):
        fig.savefig(figure_dir / f"gaussian_tradeability_boundary.{extension}", dpi=220)


if __name__ == "__main__":
    main()
