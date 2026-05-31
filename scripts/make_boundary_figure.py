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
