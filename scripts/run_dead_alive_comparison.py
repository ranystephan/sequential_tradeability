"""Compare Gaussian and dead/alive stopping policies on controlled evidence paths."""

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
    ThreeStatePrior,
    TradeabilityPayoff,
    evaluate_boundary_policy,
    evaluate_three_state_boundary_policy,
    evaluate_three_state_static_payoff_policy,
    evaluate_three_state_zstat_policy,
    simulate_gaussian_evidence,
    simulate_three_state_evidence,
    solve_gaussian_tradeability,
    solve_three_state_tradeability,
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

    target_prior_variance = 0.045
    prob_dead = 0.70
    alpha = float(np.sqrt(target_prior_variance / (1.0 - prob_dead)))
    dead_activation_penalty = 0.002
    dead_alive_prior = ThreeStatePrior(
        alpha=alpha,
        prob_negative=(1.0 - prob_dead) / 2.0,
        prob_dead=prob_dead,
        prob_positive=(1.0 - prob_dead) / 2.0,
    )
    matched_gaussian = GaussianPosterior(
        prior_mean=0.0,
        prior_variance=target_prior_variance,
    )
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.005,
    )
    observation_cost = 0.002
    activation_decay = 1.0
    horizon = 2.0
    n_steps = 300
    n_paths = 20_000

    gaussian_solution = solve_gaussian_tradeability(
        posterior=matched_gaussian,
        payoff_model=payoff,
        observation_cost=observation_cost,
        horizon=horizon,
        mean_max=0.8,
        activation_decay=activation_decay,
        n_time=n_steps + 1,
        n_mean=401,
    )
    dead_alive_solution = solve_three_state_tradeability(
        prior=dead_alive_prior,
        payoff_model=payoff,
        observation_cost=observation_cost,
        horizon=horizon,
        evidence_max=5.0,
        activation_decay=activation_decay,
        dead_activation_penalty=dead_activation_penalty,
        n_time=n_steps + 1,
        n_evidence=501,
    )

    theta_grid = np.array([-alpha, -0.30, 0.0, 0.30, alpha])
    rows: list[dict[str, float | str]] = []
    for theta_index, theta in enumerate(theta_grid):
        gaussian_paths = simulate_gaussian_evidence(
            posterior=matched_gaussian,
            horizon=horizon,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=20_000 + theta_index,
            true_theta=float(theta),
        )
        dead_alive_paths = simulate_three_state_evidence(
            prior=dead_alive_prior,
            horizon=horizon,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=20_000 + theta_index,
            true_theta=float(theta),
        )
        outcomes = [
            (
                evaluate_boundary_policy(
                    solution=gaussian_solution,
                    paths=gaussian_paths,
                    payoff_model=payoff,
                    observation_cost=observation_cost,
                    activation_decay=activation_decay,
                    dead_activation_penalty=dead_activation_penalty,
                    name="matched Gaussian boundary",
                ),
                gaussian_paths.times,
            ),
            (
                evaluate_three_state_boundary_policy(
                    solution=dead_alive_solution,
                    paths=dead_alive_paths,
                    payoff_model=payoff,
                    observation_cost=observation_cost,
                    activation_decay=activation_decay,
                    dead_activation_penalty=dead_activation_penalty,
                    name="dead/alive boundary",
                ),
                dead_alive_paths.times,
            ),
            (
                evaluate_three_state_static_payoff_policy(
                    paths=dead_alive_paths,
                    payoff_model=payoff,
                    observation_cost=observation_cost,
                    activation_decay=activation_decay,
                    dead_activation_penalty=dead_activation_penalty,
                    name="dead/alive static payoff",
                ),
                dead_alive_paths.times,
            ),
            (
                evaluate_three_state_zstat_policy(
                    paths=dead_alive_paths,
                    payoff_model=payoff,
                    observation_cost=observation_cost,
                    activation_decay=activation_decay,
                    z_threshold=1.5,
                    dead_activation_penalty=dead_activation_penalty,
                    name="dead/alive z-stat",
                ),
                dead_alive_paths.times,
            ),
        ]
        rows.extend(summarize(float(theta), outcome, times) for outcome, times in outcomes)

    csv_path = result_dir / "dead_alive_policy_comparison.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    policies = [
        "matched Gaussian boundary",
        "dead/alive boundary",
        "dead/alive static payoff",
        "dead/alive z-stat",
    ]
    colors = {
        "matched Gaussian boundary": "#6f6f6f",
        "dead/alive boundary": "#1f78b4",
        "dead/alive static payoff": "#b2182b",
        "dead/alive z-stat": "#33a02c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        theta_values = np.array([float(row["theta"]) for row in policy_rows])
        mean_values = np.array([float(row["mean_value"]) for row in policy_rows])
        stderr = np.array([float(row["stderr_value"]) for row in policy_rows])
        activation = np.array([float(row["activation_rate"]) for row in policy_rows])
        axes[0].errorbar(
            theta_values,
            mean_values,
            yerr=1.96 * stderr,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=policy,
            color=colors[policy],
        )
        axes[1].plot(
            theta_values,
            activation,
            marker="o",
            linewidth=1.5,
            label=policy,
            color=colors[policy],
        )

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("true alpha strength")
    axes[0].set_ylabel("mean realized value")
    axes[0].set_title("Realized value")
    axes[1].set_xlabel("true alpha strength")
    axes[1].set_ylabel("activation rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Activation frequency")
    axes[1].legend(frameon=True, fontsize=7, loc="lower right")
    fig.suptitle("Dead/alive signal-admission stress test")
    for extension in ("pdf", "png"):
        fig.savefig(figure_dir / f"dead_alive_policy_comparison.{extension}", dpi=220)

    center = dead_alive_solution.evidence.size // 2
    diagnostics = {
        "initial_value_at_y0": dead_alive_solution.value_at_initial_evidence(0.0),
        "initial_payoff_at_y0": float(dead_alive_solution.payoff[0, center]),
        "initial_dead_probability_at_y0": float(dead_alive_solution.posterior_dead[0, center]),
        "terminal_dead_probability_at_y0": float(dead_alive_solution.posterior_dead[-1, center]),
        "dead_activation_penalty": dead_activation_penalty,
        "three_state_alpha": alpha,
        "three_state_prior_dead": prob_dead,
        "matched_prior_variance": target_prior_variance,
        "max_obstacle_violation": float(
            np.max(dead_alive_solution.payoff - dead_alive_solution.values)
        ),
        "terminal_value_error": float(
            np.max(np.abs(dead_alive_solution.values[-1] - dead_alive_solution.payoff[-1]))
        ),
        "symmetry_error": float(
            np.max(np.abs(dead_alive_solution.values - np.fliplr(dead_alive_solution.values)))
        ),
    }
    diagnostics_path = result_dir / "dead_alive_diagnostics.csv"
    with diagnostics_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(diagnostics.keys()))
        writer.writeheader()
        writer.writerow(diagnostics)


if __name__ == "__main__":
    main()
