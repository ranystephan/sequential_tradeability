"""Replicate the OSAP sign-flip placebo experiment over many random draws.

The headline placebo activation rate (27.1%) in the main experiment uses one
fixed-seed sign-flip draw per signal.  This script quantifies the sampling noise of
that number: it redraws the per-month random sign flips many times, reapplies the same
sequential boundary, and reports the distribution of the placebo activation rate.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sequential_tradeability import (  # noqa: E402
    ThreeStatePrior,
    TradeabilityPayoff,
    apply_three_state_solution_to_evidence,
    build_signal_evidence,
    load_osap_long_short_returns,
    percentile_interval,
    sign_flip_evidence,
    solve_three_state_tradeability,
    trim_return_series,
)

N_REPLICATIONS = 200
BASE_SEED = 34_219  # replication index 0 reproduces the paper's placebo draw


def main() -> None:
    raw_path = REPO_ROOT / "data/raw/osap_monthly_long_short_returns.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}; run the OSAP download step first.")
    result_dir = REPO_ROOT / "results"
    result_dir.mkdir(exist_ok=True)

    calibration_months = 120
    validation_months = 240
    holdout_months = 120
    dt = 1.0 / 12.0
    common_start_date = date(1970, 1, 1)

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
    solution = solve_three_state_tradeability(
        prior=prior,
        payoff_model=payoff,
        observation_cost=0.001,
        horizon=validation_months * dt,
        evidence_max=8.0,
        activation_decay=0.25,
        dead_activation_penalty=0.04,
        n_time=validation_months + 1,
        n_evidence=801,
    )

    series = load_osap_long_short_returns(raw_path)
    required = calibration_months + validation_months
    real_evidence = []
    for name in sorted(series):
        trimmed = trim_return_series(series[name], start_date=common_start_date)
        if trimmed.returns.size < required:
            continue
        real_evidence.append(
            build_signal_evidence(
                trimmed,
                calibration_months=calibration_months,
                validation_months=validation_months,
                dt=dt,
            )
        )
    n_signals = len(real_evidence)

    real_results = [
        apply_three_state_solution_to_evidence(solution, evidence, holdout_months=holdout_months)
        for evidence in real_evidence
    ]
    real_rate = sum(result.decision == 1 for result in real_results) / n_signals

    rows = []
    for replication in range(N_REPLICATIONS):
        rng = np.random.default_rng(BASE_SEED + replication)
        activated = 0
        for evidence in real_evidence:
            placebo = sign_flip_evidence(evidence, rng)
            result = apply_three_state_solution_to_evidence(
                solution, placebo, holdout_months=holdout_months
            )
            activated += int(result.decision == 1)
        rows.append(
            {
                "replication": replication,
                "seed": BASE_SEED + replication,
                "n_signals": n_signals,
                "placebo_activation_count": activated,
                "placebo_activation_rate": activated / n_signals,
                "activation_gap": real_rate - activated / n_signals,
            }
        )
        if (replication + 1) % 25 == 0:
            print(f"replication {replication + 1}/{N_REPLICATIONS}")

    csv_path = result_dir / "osap_placebo_replication.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rates = np.array([row["placebo_activation_rate"] for row in rows])
    low, high = percentile_interval(rates, coverage=0.95)
    summary = {
        "n_replications": N_REPLICATIONS,
        "n_signals": n_signals,
        "real_activation_rate": real_rate,
        "placebo_rate_mean": float(np.mean(rates)),
        "placebo_rate_std": float(np.std(rates, ddof=1)),
        "placebo_rate_min": float(np.min(rates)),
        "placebo_rate_q025": low,
        "placebo_rate_q975": high,
        "placebo_rate_max": float(np.max(rates)),
        "paper_draw_rate": rows[0]["placebo_activation_rate"],
    }
    summary_path = result_dir / "osap_placebo_replication_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
