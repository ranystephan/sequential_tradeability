"""Run sequential admission on WRDS stock-level long-short signal returns."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sequential_tradeability import (  # noqa: E402
    ReturnSeries,
    ThreeStatePrior,
    TradeabilityPayoff,
    apply_three_state_solution_to_evidence,
    build_signal_evidence,
    solve_three_state_tradeability,
    trim_return_series,
)


def main() -> None:
    input_path = REPO_ROOT / "data/processed/wrds_signal_returns.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run scripts/download_wrds_stock_level.py first."
        )

    result_dir = REPO_ROOT / "results"
    result_dir.mkdir(exist_ok=True)

    calibration_months = 120
    validation_months = 240
    holdout_months = 120
    common_start_date = date(1970, 1, 1)
    dt = 1.0 / 12.0

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

    returns = pd.read_csv(input_path, parse_dates=["date"])
    rows = []
    for signal in sorted(returns["signal"].unique()):
        signal_frame = returns[returns["signal"] == signal].sort_values("date")
        for return_column, return_kind in [
            ("gross_return_percent", "gross"),
            ("net_return_percent", "net_range_cost"),
        ]:
            series = ReturnSeries(
                name=f"{signal}_{return_kind}",
                dates=np.array([value.date() for value in signal_frame["date"]], dtype=object),
                returns=signal_frame[return_column].to_numpy(dtype=float),
            )
            trimmed = trim_return_series(series, start_date=common_start_date)
            evidence = build_signal_evidence(
                trimmed,
                calibration_months=calibration_months,
                validation_months=validation_months,
                dt=dt,
            )
            result = apply_three_state_solution_to_evidence(
                solution,
                evidence,
                holdout_months=holdout_months,
            )
            row = asdict(result)
            row["signal_base"] = signal
            row["return_kind"] = return_kind
            row["decision_label"] = result.decision_label
            row["standard_direction_decision_label"] = (
                "activate"
                if result.decision == 1 and result.posterior_mean_at_stop > 0.0
                else "reject"
            )
            row["first_date"] = trimmed.dates[0].isoformat()
            row["last_date"] = trimmed.dates[-1].isoformat()
            row["stop_date"] = result.stop_date.isoformat()
            row["calibration_mean_return_percent"] = float(np.mean(evidence.calibration_returns))
            row["validation_mean_return_percent"] = float(np.mean(evidence.validation_returns))
            rows.append(row)

    output_path = result_dir / "wrds_stock_level_admission.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = [summarize(rows, return_kind) for return_kind in ["gross", "net_range_cost"]]
    summary_path = result_dir / "wrds_stock_level_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def summarize(rows: list[dict[str, object]], return_kind: str) -> dict[str, object]:
    selected = [row for row in rows if row["return_kind"] == return_kind]
    activated = [row for row in selected if row["decision_label"] == "activate"]
    standard_activated = [
        row for row in selected if row["standard_direction_decision_label"] == "activate"
    ]
    rejected = [row for row in selected if row["decision_label"] == "reject"]
    return {
        "return_kind": return_kind,
        "n_signals": len(selected),
        "n_activated": len(activated),
        "n_standard_direction_activated": len(standard_activated),
        "n_rejected": len(rejected),
        "activation_rate": len(activated) / len(selected),
        "standard_direction_activation_rate": len(standard_activated) / len(selected),
        "mean_stop_month": float(
            np.mean([12.0 * float(row["stop_time_years"]) for row in selected])
        ),
        "mean_posterior_dead_at_stop": float(
            np.mean([float(row["posterior_dead_at_stop"]) for row in selected])
        ),
        "mean_validation_variance_over_dt": float(
            np.mean([float(row["validation_increment_variance_over_dt"]) for row in selected])
        ),
        "mean_holdout_return_if_activated": float(
            np.mean([float(row["holdout_mean_return_percent"]) for row in activated])
        )
        if activated
        else float("nan"),
        "mean_holdout_return_if_standard_direction_activated": float(
            np.mean([float(row["holdout_mean_return_percent"]) for row in standard_activated])
        )
        if standard_activated
        else float("nan"),
        "mean_holdout_return_if_rejected": float(
            np.mean([float(row["holdout_mean_return_percent"]) for row in rejected])
        )
        if rejected
        else float("nan"),
    }


if __name__ == "__main__":
    main()
