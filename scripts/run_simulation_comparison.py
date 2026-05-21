"""Run the first pathwise validation experiment for stopping policies."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sequential_tradeability import (  # noqa: E402
    GaussianPosterior,
    PolicyOutcome,
    TradeabilityPayoff,
    evaluate_boundary_policy,
    evaluate_static_payoff_policy,
    evaluate_zstat_policy,
    simulate_gaussian_evidence,
    solve_gaussian_tradeability,
)


def summarize(theta: float, outcome: PolicyOutcome, times: np.ndarray) -> dict[str, float | str]:
    return {
        "theta": theta,
        "policy": outcome.name,
        "mean_value": outcome.mean_realized_value,
        "stderr_value": outcome.stderr_realized_value,
        "activation_rate": outcome.activation_rate,
        "rejection_rate": outcome.rejection_rate,
        "mean_stop_time": outcome.mean_stop_time(times),
    }


def main() -> None:
    result_dir = REPO_ROOT / "results"
    figure_dir = REPO_ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    posterior = GaussianPosterior(prior_mean=0.0, prior_variance=0.20)
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.005,
    )
    observation_cost = 0.005
    activation_decay = 1.0
    horizon = 2.0
    n_steps = 300
    n_paths = 10_000

    solution = solve_gaussian_tradeability(
        posterior=posterior,
        payoff_model=payoff,
        observation_cost=observation_cost,
        horizon=horizon,
        mean_max=1.2,
        activation_decay=activation_decay,
        n_time=n_steps + 1,
        n_mean=401,
    )

    theta_grid = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
    rows: list[dict[str, float | str]] = []
    for theta_index, theta in enumerate(theta_grid):
        paths = simulate_gaussian_evidence(
            posterior=posterior,
            horizon=horizon,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=10_000 + theta_index,
            true_theta=float(theta),
        )
        outcomes = [
            evaluate_boundary_policy(
                solution=solution,
                paths=paths,
                payoff_model=payoff,
                observation_cost=observation_cost,
                activation_decay=activation_decay,
                name="dynamic boundary",
            ),
            evaluate_static_payoff_policy(
                paths=paths,
                payoff_model=payoff,
                observation_cost=observation_cost,
                activation_decay=activation_decay,
                name="static payoff threshold",
            ),
            evaluate_zstat_policy(
                paths=paths,
                payoff_model=payoff,
                observation_cost=observation_cost,
                activation_decay=activation_decay,
                z_threshold=1.0,
                name="z-stat threshold",
            ),
        ]
        rows.extend(summarize(float(theta), outcome, paths.times) for outcome in outcomes)

    csv_path = result_dir / "simulation_policy_comparison.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    policies = ["dynamic boundary", "static payoff threshold", "z-stat threshold"]
    colors = {
        "dynamic boundary": "#2f6fbb",
        "static payoff threshold": "#b2182b",
        "z-stat threshold": "#4d4d4d",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        theta_values = np.array([float(row["theta"]) for row in policy_rows])
        mean_values = np.array([float(row["mean_value"]) for row in policy_rows])
        stderr = np.array([float(row["stderr_value"]) for row in policy_rows])
        ax.errorbar(
            theta_values,
            mean_values,
            yerr=1.96 * stderr,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=policy,
            color=colors[policy],
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("true alpha strength")
    ax.set_ylabel("mean realized value")
    ax.set_title("Policy comparison under the Gaussian evidence model")
    ax.legend(frameon=True, fontsize=8)
    for extension in ("pdf", "png"):
        fig.savefig(figure_dir / f"simulation_policy_comparison.{extension}", dpi=220)


if __name__ == "__main__":
    main()
