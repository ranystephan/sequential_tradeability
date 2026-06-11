"""Out-of-sample calibration of the false-discovery penalty eta.

The paper selects eta = 0.04 on the same placebo panel used for evaluation.  This
script removes that circularity: in each replication, eta is selected on one
independent placebo draw (the smallest grid value whose calibration placebo
activation rate falls below one third, the paper's stated criterion) and then
evaluated on a fresh placebo draw.  The variational inequality is solved once per
grid value and reused across replications.
"""

from __future__ import annotations

import csv

import numpy as np
from osap_common import (
    HOLDOUT_MONTHS,
    RESULT_DIR,
    load_real_evidence,
    solve_paper_boundary,
)

from sequential_tradeability import (
    apply_three_state_solution_to_evidence,
    percentile_interval,
    sign_flip_evidence,
)

ETA_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
PLACEBO_TARGET_RATE = 1.0 / 3.0
N_REPLICATIONS = 25
CALIBRATION_SEED_BASE = 60_000
EVALUATION_SEED_BASE = 70_000


def activation_rate(solution, evidence_list) -> float:
    activated = sum(
        apply_three_state_solution_to_evidence(
            solution, evidence, holdout_months=HOLDOUT_MONTHS
        ).decision
        == 1
        for evidence in evidence_list
    )
    return activated / len(evidence_list)


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    real_evidence = load_real_evidence()

    solutions = {}
    real_rates = {}
    for eta in ETA_GRID:
        print(f"solving boundary at eta = {eta}")
        solutions[eta] = solve_paper_boundary(dead_activation_penalty=eta)
        real_rates[eta] = activation_rate(solutions[eta], real_evidence)

    rows = []
    for replication in range(N_REPLICATIONS):
        calibration_rng = np.random.default_rng(CALIBRATION_SEED_BASE + replication)
        evaluation_rng = np.random.default_rng(EVALUATION_SEED_BASE + replication)
        calibration_placebo = [
            sign_flip_evidence(evidence, calibration_rng) for evidence in real_evidence
        ]
        evaluation_placebo = [
            sign_flip_evidence(evidence, evaluation_rng) for evidence in real_evidence
        ]

        selected_eta = None
        calibration_rate_at_selected = None
        for eta in ETA_GRID:
            rate = activation_rate(solutions[eta], calibration_placebo)
            if rate < PLACEBO_TARGET_RATE:
                selected_eta = eta
                calibration_rate_at_selected = rate
                break
        if selected_eta is None:
            raise RuntimeError(
                f"replication {replication}: no eta in the grid reaches the placebo target"
            )

        evaluation_rate = activation_rate(solutions[selected_eta], evaluation_placebo)
        rows.append(
            {
                "replication": replication,
                "selected_eta": selected_eta,
                "calibration_placebo_rate": calibration_rate_at_selected,
                "evaluation_placebo_rate": evaluation_rate,
                "real_activation_rate": real_rates[selected_eta],
                "activation_gap": real_rates[selected_eta] - evaluation_rate,
            }
        )
        print(
            f"replication {replication + 1}/{N_REPLICATIONS}: "
            f"eta = {selected_eta}, out-of-sample placebo rate = {evaluation_rate:.4f}"
        )

    csv_path = RESULT_DIR / "osap_split_eta_calibration.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    etas = np.array([row["selected_eta"] for row in rows])
    oos_rates = np.array([row["evaluation_placebo_rate"] for row in rows])
    real = np.array([row["real_activation_rate"] for row in rows])
    low, high = percentile_interval(oos_rates, coverage=0.95)
    summary = {
        "n_replications": N_REPLICATIONS,
        "placebo_target_rate": PLACEBO_TARGET_RATE,
        "selected_eta_min": float(np.min(etas)),
        "selected_eta_median": float(np.median(etas)),
        "selected_eta_max": float(np.max(etas)),
        "oos_placebo_rate_mean": float(np.mean(oos_rates)),
        "oos_placebo_rate_std": float(np.std(oos_rates, ddof=1)),
        "oos_placebo_rate_q025": low,
        "oos_placebo_rate_q975": high,
        "real_activation_rate_mean": float(np.mean(real)),
        "real_activation_rate_min": float(np.min(real)),
        "real_activation_rate_max": float(np.max(real)),
    }
    summary_path = RESULT_DIR / "osap_split_eta_calibration_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
