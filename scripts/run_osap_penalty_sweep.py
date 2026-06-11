"""Sweep the false-discovery penalty on real OSAP signals and sign-flip placebos."""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sequential_tradeability import (  # noqa: E402
    ThreeStatePrior,
    TradeabilityPayoff,
    apply_three_state_solution_to_evidence,
    build_signal_evidence,
    load_osap_long_short_returns,
    sign_flip_evidence,
    solve_three_state_tradeability,
    trim_return_series,
)


def main() -> None:
    raw_path = REPO_ROOT / "data/raw/osap_monthly_long_short_returns.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}; run the OSAP download step first.")

    result_dir = REPO_ROOT / "results"
    figure_dir = REPO_ROOT / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    calibration_months = 120
    validation_months = 240
    holdout_months = 120
    dt = 1.0 / 12.0
    common_start_date = date(1970, 1, 1)

    series = load_osap_long_short_returns(raw_path)
    required_observations = calibration_months + validation_months
    rng = np.random.default_rng(34_219)
    evidence_pairs = []
    for name in sorted(series):
        trimmed = trim_return_series(series[name], start_date=common_start_date)
        if trimmed.returns.size < required_observations:
            continue
        evidence = build_signal_evidence(
            trimmed,
            calibration_months=calibration_months,
            validation_months=validation_months,
            dt=dt,
        )
        evidence_pairs.append((evidence, sign_flip_evidence(evidence, rng)))

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
    penalties = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
    rows = []
    for penalty in penalties:
        solution = solve_three_state_tradeability(
            prior=prior,
            payoff_model=payoff,
            observation_cost=0.001,
            horizon=validation_months * dt,
            evidence_max=8.0,
            activation_decay=0.25,
            dead_activation_penalty=penalty,
            n_time=validation_months + 1,
            n_evidence=801,
        )
        real_results = [
            apply_three_state_solution_to_evidence(
                solution,
                evidence,
                holdout_months=holdout_months,
            )
            for evidence, _ in evidence_pairs
        ]
        placebo_results = [
            apply_three_state_solution_to_evidence(
                solution,
                placebo,
                holdout_months=holdout_months,
            )
            for _, placebo in evidence_pairs
        ]
        rows.append(summarize(penalty, real_results, placebo_results))

    csv_path = result_dir / "osap_penalty_sweep.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    penalties_array = np.array([row["dead_activation_penalty"] for row in rows], dtype=float)
    real_activation = np.array([row["real_activation_rate"] for row in rows], dtype=float)
    placebo_activation = np.array([row["placebo_activation_rate"] for row in rows], dtype=float)
    activation_gap = real_activation - placebo_activation

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot(penalties_array, real_activation, marker="o", label="real OSAP signals")
    ax.plot(penalties_array, placebo_activation, marker="o", label="sign-flip placebo")
    ax.plot(penalties_array, activation_gap, marker="o", label="activation gap")
    ax.axvline(0.04, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("false-discovery penalty eta")
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("OSAP admission sensitivity to false-discovery penalty")
    ax.legend(frameon=True, fontsize=8)
    for extension in ("pdf", "png"):
        fig.savefig(figure_dir / f"osap_penalty_sweep.{extension}", dpi=220)


def summarize(penalty, real_results, placebo_results) -> dict[str, float]:
    real_activated = [result for result in real_results if result.decision == 1]
    placebo_activated = [result for result in placebo_results if result.decision == 1]
    return {
        "dead_activation_penalty": penalty,
        "n_signals": len(real_results),
        "real_activation_rate": len(real_activated) / len(real_results),
        "placebo_activation_rate": len(placebo_activated) / len(placebo_results),
        "activation_gap": len(real_activated) / len(real_results)
        - len(placebo_activated) / len(placebo_results),
        "real_mean_holdout_if_activated": mean_holdout(real_activated),
        "placebo_mean_holdout_if_activated": mean_holdout(placebo_activated),
        "real_mean_posterior_dead_at_stop": float(
            np.mean([result.posterior_dead_at_stop for result in real_results])
        ),
        "placebo_mean_posterior_dead_at_stop": float(
            np.mean([result.posterior_dead_at_stop for result in placebo_results])
        ),
    }


def mean_holdout(results) -> float:
    if not results:
        return float("nan")
    return float(np.mean([result.holdout_mean_return_percent for result in results]))


if __name__ == "__main__":
    main()
