"""Check grid convergence for the three-state OSAP stopping problem."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sequential_tradeability import (  # noqa: E402
    ThreeStatePrior,
    ThreeStateStoppingSolution,
    TradeabilityPayoff,
    solve_three_state_tradeability,
)


def main() -> None:
    result_dir = REPO_ROOT / "results"
    result_dir.mkdir(exist_ok=True)

    prior = ThreeStatePrior(
        alpha=0.38729833462074165,
        prob_negative=0.15,
        prob_dead=0.70,
        prob_positive=0.15,
    )
    payoff = TradeabilityPayoff(
        risk_aversion=1.0,
        return_variance=0.05,
        implementation_hurdle=0.005,
    )
    common_kwargs = {
        "prior": prior,
        "payoff_model": payoff,
        "observation_cost": 0.001,
        "horizon": 20.0,
        "evidence_max": 8.0,
        "activation_decay": 0.25,
        "dead_activation_penalty": 0.04,
    }
    grids = [
        ("coarse", 121, 401),
        ("medium_time", 241, 401),
        ("medium_space", 241, 601),
        ("paper", 241, 801),
        ("fine", 481, 1001),
    ]
    solutions = {
        name: solve_three_state_tradeability(n_time=n_time, n_evidence=n_evidence, **common_kwargs)
        for name, n_time, n_evidence in grids
    }
    reference = solutions["fine"]

    eval_times = np.linspace(0.0, 20.0, 241)
    eval_evidence = np.linspace(-8.0, 8.0, 401)
    reference_regions = classify_on_grid(reference, eval_times, eval_evidence)

    rows = []
    reference_value = reference.value_at_initial_evidence(0.0)
    for name, n_time, n_evidence in grids:
        solution = solutions[name]
        regions = classify_on_grid(solution, eval_times, eval_evidence)
        disagreement = float(np.mean(regions != reference_regions))
        initial_value = solution.value_at_initial_evidence(0.0)
        rows.append(
            {
                "grid": name,
                "n_time": n_time,
                "n_evidence": n_evidence,
                "dt": solution.dt,
                "dy": solution.dy,
                "initial_value": initial_value,
                "abs_initial_value_error_vs_fine": abs(initial_value - reference_value),
                "activation_boundary_y_month_0": positive_activation_boundary(solution, 0.0),
                "activation_boundary_y_month_60": positive_activation_boundary(solution, 5.0),
                "activation_boundary_y_month_120": positive_activation_boundary(solution, 10.0),
                "region_disagreement_vs_fine": disagreement,
            }
        )

    csv_path = result_dir / "grid_convergence.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def positive_activation_boundary(solution: ThreeStateStoppingSolution, time: float) -> float:
    """Return the smallest nonnegative evidence grid point where activation is optimal."""
    time_index = int(np.clip(round(time / solution.dt), 0, solution.times.size - 1))
    regions = solution.decision_regions()
    activate = (regions[time_index] == 1) & (solution.evidence >= 0.0)
    if not np.any(activate):
        return float("nan")
    return float(np.min(solution.evidence[activate]))


def classify_on_grid(
    solution: ThreeStateStoppingSolution,
    eval_times: np.ndarray,
    eval_evidence: np.ndarray,
) -> np.ndarray:
    """Classify stop/continue/activate regions on a common grid by nearest neighbor."""
    regions = solution.decision_regions()
    time_indices = np.clip(np.rint(eval_times / solution.dt).astype(int), 0, regions.shape[0] - 1)
    evidence_indices = np.searchsorted(solution.evidence, eval_evidence)
    evidence_indices = np.clip(evidence_indices, 1, solution.evidence.size - 1)
    left = solution.evidence[evidence_indices - 1]
    right = solution.evidence[evidence_indices]
    choose_right = np.abs(eval_evidence - right) < np.abs(eval_evidence - left)
    nearest_evidence_indices = evidence_indices - 1 + choose_right.astype(int)
    return regions[np.ix_(time_indices, nearest_evidence_indices)]


if __name__ == "__main__":
    main()
