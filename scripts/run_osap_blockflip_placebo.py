"""Block sign-flip placebo robustness for the OSAP admission experiment.

The main placebo flips the sign of each monthly validation return independently,
which destroys serial dependence of signed returns along with drift.  This script
repeats the placebo replication with one random sign per 12-month block, preserving
within-block dependence, to check that the real-placebo separation does not rest on
the per-month randomization.
"""

from __future__ import annotations

import csv

import numpy as np
from osap_common import (
    HOLDOUT_MONTHS,
    PAPER_PLACEBO_SEED,
    RESULT_DIR,
    load_real_evidence,
    solve_paper_boundary,
)

from sequential_tradeability import (
    apply_three_state_solution_to_evidence,
    block_sign_flip_evidence,
    percentile_interval,
)

N_REPLICATIONS = 200
BLOCK_MONTHS = 12


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    solution = solve_paper_boundary()
    real_evidence = load_real_evidence()
    n_signals = len(real_evidence)

    real_rate = sum(
        apply_three_state_solution_to_evidence(
            solution, evidence, holdout_months=HOLDOUT_MONTHS
        ).decision
        == 1
        for evidence in real_evidence
    ) / n_signals

    rows = []
    for replication in range(N_REPLICATIONS):
        rng = np.random.default_rng(PAPER_PLACEBO_SEED + replication)
        activated = 0
        for evidence in real_evidence:
            placebo = block_sign_flip_evidence(evidence, rng, block_months=BLOCK_MONTHS)
            result = apply_three_state_solution_to_evidence(
                solution, placebo, holdout_months=HOLDOUT_MONTHS
            )
            activated += int(result.decision == 1)
        rows.append(
            {
                "replication": replication,
                "seed": PAPER_PLACEBO_SEED + replication,
                "block_months": BLOCK_MONTHS,
                "n_signals": n_signals,
                "placebo_activation_count": activated,
                "placebo_activation_rate": activated / n_signals,
                "activation_gap": real_rate - activated / n_signals,
            }
        )
        if (replication + 1) % 25 == 0:
            print(f"replication {replication + 1}/{N_REPLICATIONS}")

    csv_path = RESULT_DIR / "osap_blockflip_placebo.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rates = np.array([row["placebo_activation_rate"] for row in rows])
    low, high = percentile_interval(rates, coverage=0.95)
    summary = {
        "n_replications": N_REPLICATIONS,
        "block_months": BLOCK_MONTHS,
        "real_activation_rate": real_rate,
        "placebo_rate_mean": float(np.mean(rates)),
        "placebo_rate_std": float(np.std(rates, ddof=1)),
        "placebo_rate_min": float(np.min(rates)),
        "placebo_rate_q025": low,
        "placebo_rate_q975": high,
        "placebo_rate_max": float(np.max(rates)),
    }
    summary_path = RESULT_DIR / "osap_blockflip_placebo_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
