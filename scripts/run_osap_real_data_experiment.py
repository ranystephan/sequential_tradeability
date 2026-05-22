"""Run the first real-data signal-admission experiment on OSAP anomaly returns."""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, replace
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
    solve_three_state_tradeability,
    trim_return_series,
)

OSAP_URL = (
    "https://www.federalreserve.gov/econres/feds/files/"
    "PredictorLSretWide-feds-2021037.csv"
)


def main() -> None:
    raw_path = REPO_ROOT / "data/raw/osap_monthly_long_short_returns.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Download the Federal Reserve OSAP CSV from {OSAP_URL}."
        )

    result_dir = REPO_ROOT / "results"
    result_dir.mkdir(exist_ok=True)

    selected_signals = [
        "STreversal",
        "Mom12m",
        "BM",
        "OperProf",
        "AssetGrowth",
        "GP",
        "Accruals",
        "BidAskSpread",
    ]
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
    missing = [name for name in selected_signals if name not in series]
    if missing:
        raise ValueError(f"Selected signals are missing from OSAP file: {missing}")

    rows = []
    all_rows = []
    placebo_rows = []
    rng = np.random.default_rng(34_219)
    for name in selected_signals:
        row = run_one_signal(
            name=name,
            series=series[name],
            common_start_date=common_start_date,
            calibration_months=calibration_months,
            validation_months=validation_months,
            holdout_months=holdout_months,
            dt=dt,
            solution=solution,
        )
        rows.append(row)

    csv_path = result_dir / "osap_real_data_admission.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    required_observations = calibration_months + validation_months
    for name in sorted(series):
        trimmed = trim_return_series(series[name], start_date=common_start_date)
        if trimmed.returns.size < required_observations:
            continue
        row, placebo_row = run_one_signal_with_placebo(
            name=name,
            series=series[name],
            common_start_date=common_start_date,
            calibration_months=calibration_months,
            validation_months=validation_months,
            holdout_months=holdout_months,
            dt=dt,
            solution=solution,
            rng=rng,
        )
        all_rows.append(row)
        placebo_rows.append(placebo_row)

    all_csv_path = result_dir / "osap_real_data_all_signals.csv"
    with all_csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    placebo_csv_path = result_dir / "osap_real_data_placebo_signflip.csv"
    with placebo_csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(placebo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(placebo_rows)

    activated = [row for row in rows if row["decision_label"] == "activate"]
    rejected = [row for row in rows if row["decision_label"] == "reject"]
    summary = {
        "n_signals": len(rows),
        "n_activated": len(activated),
        "n_rejected": len(rejected),
        "mean_posterior_dead_at_stop": float(
            np.mean([row["posterior_dead_at_stop"] for row in rows])
        ),
        "mean_validation_variance_over_dt": float(
            np.mean([row["validation_increment_variance_over_dt"] for row in rows])
        ),
        "median_abs_lag1_autocorr": float(
            np.median([abs(row["validation_lag1_autocorr"]) for row in rows])
        ),
        "mean_holdout_return_if_activated": float(
            np.mean([row["holdout_mean_return_percent"] for row in activated])
        )
        if activated
        else float("nan"),
    }
    summary_path = result_dir / "osap_real_data_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    all_summary = summarize_rows(all_rows)
    all_summary_path = result_dir / "osap_real_data_all_summary.csv"
    with all_summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_summary.keys()))
        writer.writeheader()
        writer.writerow(all_summary)

    placebo_summary = summarize_rows(placebo_rows)
    placebo_summary_path = result_dir / "osap_real_data_placebo_summary.csv"
    with placebo_summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(placebo_summary.keys()))
        writer.writeheader()
        writer.writerow(placebo_summary)


def run_one_signal(
    *,
    name: str,
    series,
    common_start_date: date,
    calibration_months: int,
    validation_months: int,
    holdout_months: int,
    dt: float,
    solution,
) -> dict[str, object]:
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
    row["decision_label"] = result.decision_label
    row["first_date"] = trimmed.dates[0].isoformat()
    row["last_date"] = trimmed.dates[-1].isoformat()
    row["stop_date"] = result.stop_date.isoformat()
    return row


def run_one_signal_with_placebo(
    *,
    name: str,
    series,
    common_start_date: date,
    calibration_months: int,
    validation_months: int,
    holdout_months: int,
    dt: float,
    solution,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object]]:
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
    row["decision_label"] = result.decision_label
    row["first_date"] = trimmed.dates[0].isoformat()
    row["last_date"] = trimmed.dates[-1].isoformat()
    row["stop_date"] = result.stop_date.isoformat()

    signs = rng.choice(np.array([-1.0, 1.0]), size=evidence.increments.size)
    placebo_increments = evidence.increments * signs
    placebo_evidence = replace(
        evidence,
        name=f"{name}_signflip",
        increments=placebo_increments,
        observations=np.concatenate([[0.0], np.cumsum(placebo_increments)]),
    )
    placebo_result = apply_three_state_solution_to_evidence(
        solution,
        placebo_evidence,
        holdout_months=holdout_months,
    )
    placebo_row = asdict(placebo_result)
    placebo_row["decision_label"] = placebo_result.decision_label
    placebo_row["first_date"] = trimmed.dates[0].isoformat()
    placebo_row["last_date"] = trimmed.dates[-1].isoformat()
    placebo_row["stop_date"] = placebo_result.stop_date.isoformat()
    return row, placebo_row


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, float | int]:
    activated = [row for row in rows if row["decision_label"] == "activate"]
    rejected = [row for row in rows if row["decision_label"] == "reject"]
    return {
        "n_signals": len(rows),
        "n_activated": len(activated),
        "n_rejected": len(rejected),
        "activation_rate": len(activated) / len(rows),
        "mean_posterior_dead_at_stop": float(
            np.mean([float(row["posterior_dead_at_stop"]) for row in rows])
        ),
        "mean_validation_variance_over_dt": float(
            np.mean([float(row["validation_increment_variance_over_dt"]) for row in rows])
        ),
        "median_abs_lag1_autocorr": float(
            np.median([abs(float(row["validation_lag1_autocorr"])) for row in rows])
        ),
        "mean_holdout_return_if_activated": float(
            np.mean([float(row["holdout_mean_return_percent"]) for row in activated])
        )
        if activated
        else float("nan"),
        "mean_holdout_return_if_rejected": float(
            np.mean([float(row["holdout_mean_return_percent"]) for row in rejected])
        )
        if rejected
        else float("nan"),
    }


if __name__ == "__main__":
    main()
