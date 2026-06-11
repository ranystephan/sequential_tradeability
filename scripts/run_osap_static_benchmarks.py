"""Compare the sequential OSAP admission rule with calibrated fixed-horizon tests."""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sequential_tradeability import (  # noqa: E402
    RealDataPolicyResult,
    SignalEvidence,
    ThreeStatePrior,
    TradeabilityPayoff,
    apply_static_threshold_to_evidence,
    apply_three_state_solution_to_evidence,
    build_signal_evidence,
    load_osap_long_short_returns,
    sign_flip_evidence,
    solve_three_state_tradeability,
    static_three_state_score,
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
    dead_activation_penalty = 0.04

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
        dead_activation_penalty=dead_activation_penalty,
        n_time=validation_months + 1,
        n_evidence=801,
    )

    evidence_pairs = load_evidence_pairs(
        raw_path=raw_path,
        common_start_date=common_start_date,
        calibration_months=calibration_months,
        validation_months=validation_months,
        dt=dt,
    )
    real_evidence = [pair[0] for pair in evidence_pairs]
    placebo_evidence = [pair[1] for pair in evidence_pairs]

    sequential_real = [
        apply_three_state_solution_to_evidence(
            solution,
            evidence,
            holdout_months=holdout_months,
        )
        for evidence in real_evidence
    ]
    sequential_placebo = [
        apply_three_state_solution_to_evidence(
            solution,
            evidence,
            holdout_months=holdout_months,
        )
        for evidence in placebo_evidence
    ]
    target_placebo_activations = sum(result.decision == 1 for result in sequential_placebo)

    rows = [
        summarize_results(
            method="sequential_boundary",
            score_kind="dynamic_stop",
            horizon_months="adaptive",
            threshold=float("nan"),
            real_results=sequential_real,
            placebo_results=sequential_placebo,
        )
    ]

    score_kinds = ["net_payoff", "z_stat", "alive_probability"]
    horizons = [12, 24, 60, 120, 240]
    for score_kind in score_kinds:
        for horizon_months in horizons:
            placebo_scores = np.array(
                [
                    static_three_state_score(
                        prior=prior,
                        payoff_model=payoff,
                        evidence=evidence,
                        horizon_months=horizon_months,
                        score_kind=score_kind,
                        dead_activation_penalty=dead_activation_penalty,
                    )
                    for evidence in placebo_evidence
                ]
            )
            threshold = threshold_for_top_k(placebo_scores, target_placebo_activations)
            static_real = [
                apply_static_threshold_to_evidence(
                    prior=prior,
                    payoff_model=payoff,
                    evidence=evidence,
                    horizon_months=horizon_months,
                    score_kind=score_kind,
                    threshold=threshold,
                    dead_activation_penalty=dead_activation_penalty,
                    holdout_months=holdout_months,
                )
                for evidence in real_evidence
            ]
            static_placebo = [
                apply_static_threshold_to_evidence(
                    prior=prior,
                    payoff_model=payoff,
                    evidence=evidence,
                    horizon_months=horizon_months,
                    score_kind=score_kind,
                    threshold=threshold,
                    dead_activation_penalty=dead_activation_penalty,
                    holdout_months=holdout_months,
                )
                for evidence in placebo_evidence
            ]
            rows.append(
                summarize_results(
                    method=f"fixed_{score_kind}",
                    score_kind=score_kind,
                    horizon_months=str(horizon_months),
                    threshold=threshold,
                    real_results=static_real,
                    placebo_results=static_placebo,
                )
            )

    csv_path = result_dir / "osap_static_benchmark_comparison.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_figure(rows, figure_dir / "osap_static_benchmark_comparison")


def load_evidence_pairs(
    *,
    raw_path: Path,
    common_start_date: date,
    calibration_months: int,
    validation_months: int,
    dt: float,
) -> list[tuple[SignalEvidence, SignalEvidence]]:
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
    return evidence_pairs


def threshold_for_top_k(scores: np.ndarray, k: int) -> float:
    """Return the cutoff that activates the top k scores."""
    if k < 0 or k > scores.size:
        raise ValueError("k must lie between zero and the number of scores")
    if k == 0:
        return float(np.nextafter(np.max(scores), np.inf))
    if k == scores.size:
        return float(np.min(scores))
    return float(np.sort(scores)[-k])


def summarize_results(
    *,
    method: str,
    score_kind: str,
    horizon_months: str,
    threshold: float,
    real_results: list[RealDataPolicyResult],
    placebo_results: list[RealDataPolicyResult],
) -> dict[str, object]:
    real_activated = [result for result in real_results if result.decision == 1]
    placebo_activated = [result for result in placebo_results if result.decision == 1]
    return {
        "method": method,
        "score_kind": score_kind,
        "horizon_months": horizon_months,
        "threshold": threshold,
        "n_signals": len(real_results),
        "real_activation_count": len(real_activated),
        "placebo_activation_count": len(placebo_activated),
        "real_activation_rate": len(real_activated) / len(real_results),
        "placebo_activation_rate": len(placebo_activated) / len(placebo_results),
        "activation_gap": len(real_activated) / len(real_results)
        - len(placebo_activated) / len(placebo_results),
        "real_mean_decision_month": 12.0
        * float(np.mean([result.stop_time_years for result in real_results])),
        "placebo_mean_decision_month": 12.0
        * float(np.mean([result.stop_time_years for result in placebo_results])),
        "real_mean_holdout_if_activated": mean_holdout(real_activated),
        "placebo_mean_holdout_if_activated": mean_holdout(placebo_activated),
        "real_activation_weighted_holdout": weighted_holdout(real_activated, real_results),
        "placebo_activation_weighted_holdout": weighted_holdout(placebo_activated, placebo_results),
        "real_mean_posterior_dead_if_activated": mean_posterior_dead(real_activated),
        "placebo_mean_posterior_dead_if_activated": mean_posterior_dead(placebo_activated),
    }


def mean_holdout(results: list[RealDataPolicyResult]) -> float:
    if not results:
        return float("nan")
    return float(np.mean([result.holdout_mean_return_percent for result in results]))


def weighted_holdout(
    activated_results: list[RealDataPolicyResult],
    all_results: list[RealDataPolicyResult],
) -> float:
    if not activated_results:
        return 0.0
    return len(activated_results) / len(all_results) * mean_holdout(activated_results)


def mean_posterior_dead(results: list[RealDataPolicyResult]) -> float:
    if not results:
        return float("nan")
    return float(np.mean([result.posterior_dead_at_stop for result in results]))


def make_figure(rows: list[dict[str, object]], output_stem: Path) -> None:
    sequential = rows[0]
    static_rows = [row for row in rows if row["method"] == "fixed_net_payoff"]
    horizons = np.array([int(row["horizon_months"]) for row in static_rows])
    real_rates = np.array([float(row["real_activation_rate"]) for row in static_rows])
    weighted_holdouts = np.array(
        [float(row["real_activation_weighted_holdout"]) for row in static_rows]
    )
    sequential_decision_month = float(sequential["real_mean_decision_month"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), constrained_layout=True)
    axes[0].plot(horizons, real_rates, marker="o", label="fixed horizon")
    axes[0].axhline(
        float(sequential["real_activation_rate"]),
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="sequential boundary",
    )
    axes[0].set_xlabel("fixed decision month")
    axes[0].set_ylabel("real activation rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Admission power")
    axes[0].legend(frameon=True, fontsize=8)

    axes[1].plot(horizons, weighted_holdouts, marker="o", label="fixed horizon")
    axes[1].scatter(
        [sequential_decision_month],
        [float(sequential["real_activation_weighted_holdout"])],
        color="black",
        marker="*",
        s=90,
        label="sequential boundary",
        zorder=3,
    )
    axes[1].set_xlabel("mean decision month")
    axes[1].set_ylabel("activation rate x holdout return")
    axes[1].set_title("Post-decision tradeoff")
    axes[1].legend(frameon=True, fontsize=8)
    for extension in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=220)


if __name__ == "__main__":
    main()
