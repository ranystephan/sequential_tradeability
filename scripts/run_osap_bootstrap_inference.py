"""Bootstrap standard errors for the OSAP holdout-return comparisons.

Two complementary resampling designs:

1. Calendar-time block bootstrap (decisions held fixed).  Resample calendar months in
   circular 12-month blocks from the common timeline and recompute each method's
   holdout statistics.  The same resampled months are applied to every signal, so
   cross-signal correlation within a month is preserved.
2. Signal cluster bootstrap.  Resample the 192 signals with replacement and recompute
   activation rates and holdout statistics, capturing cross-signal heterogeneity in
   both the decisions and the post-decision returns.

Reported statistics: mean holdout return among activated signals, activation-weighted
holdout return, and the sequential-minus-fixed differences at 60, 120, and 240 months.
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
    apply_static_threshold_to_evidence,
    apply_three_state_solution_to_evidence,
    build_signal_evidence,
    holdout_series,
    load_osap_long_short_returns,
    percentile_interval,
    sign_flip_evidence,
    solve_three_state_tradeability,
    static_three_state_score,
    time_block_bootstrap_means,
    trim_return_series,
)

N_BOOTSTRAP = 2000
BLOCK_LENGTH_MONTHS = 12
FIXED_HORIZONS = [60, 120, 240]
SEED_TIME_BLOCK = 52_100
SEED_SIGNAL_CLUSTER = 52_200


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

    series = load_osap_long_short_returns(raw_path)
    required = calibration_months + validation_months
    rng_placebo = np.random.default_rng(34_219)  # the paper's placebo calibration draw
    real_evidence = []
    placebo_evidence = []
    for name in sorted(series):
        trimmed = trim_return_series(series[name], start_date=common_start_date)
        if trimmed.returns.size < required:
            continue
        evidence = build_signal_evidence(
            trimmed,
            calibration_months=calibration_months,
            validation_months=validation_months,
            dt=dt,
        )
        real_evidence.append(evidence)
        placebo_evidence.append(sign_flip_evidence(evidence, rng_placebo))
    n_signals = len(real_evidence)

    # Baseline decisions, identical to the static-benchmark experiment.
    methods: dict[str, list] = {}
    methods["sequential"] = [
        apply_three_state_solution_to_evidence(solution, evidence, holdout_months=holdout_months)
        for evidence in real_evidence
    ]
    sequential_placebo = [
        apply_three_state_solution_to_evidence(solution, evidence, holdout_months=holdout_months)
        for evidence in placebo_evidence
    ]
    target_placebo_activations = sum(result.decision == 1 for result in sequential_placebo)

    for horizon in FIXED_HORIZONS:
        placebo_scores = np.array(
            [
                static_three_state_score(
                    prior=prior,
                    payoff_model=payoff,
                    evidence=evidence,
                    horizon_months=horizon,
                    score_kind="net_payoff",
                    dead_activation_penalty=dead_activation_penalty,
                )
                for evidence in placebo_evidence
            ]
        )
        threshold = float(np.sort(placebo_scores)[-target_placebo_activations])
        methods[f"fixed_{horizon}"] = [
            apply_static_threshold_to_evidence(
                prior=prior,
                payoff_model=payoff,
                evidence=evidence,
                horizon_months=horizon,
                score_kind="net_payoff",
                threshold=threshold,
                dead_activation_penalty=dead_activation_penalty,
                holdout_months=holdout_months,
            )
            for evidence in real_evidence
        ]

    # Per-method state: activation flags, rates, and dated holdout panels for activated signals.
    activated_flags = {
        name: np.array([result.decision == 1 for result in results])
        for name, results in methods.items()
    }
    rates = {name: float(np.mean(flags)) for name, flags in activated_flags.items()}
    panels = {
        name: [
            holdout_series(
                real_evidence[i],
                stop_index=methods[name][i].stop_index,
                holdout_months=holdout_months,
            )
            for i in np.flatnonzero(flags)
        ]
        for name, flags in activated_flags.items()
    }
    baseline_mean = {
        name: float(np.mean([series.mean_return for series in panel]))
        for name, panel in panels.items()
    }
    baseline_weighted = {name: rates[name] * baseline_mean[name] for name in methods}
    for name in methods:
        expected = float(
            np.mean(
                [
                    result.holdout_mean_return_percent
                    for result in methods[name]
                    if result.decision == 1
                ]
            )
        )
        assert abs(baseline_mean[name] - expected) < 1e-12, name

    statistics = build_statistics(baseline_mean, baseline_weighted)

    # Design 1: calendar-time block bootstrap, decisions held fixed.
    rng = np.random.default_rng(SEED_TIME_BLOCK)
    merged_panel = [series for name in methods for series in panels[name]]
    panel_slices = {}
    offset = 0
    for name in methods:
        panel_slices[name] = slice(offset, offset + len(panels[name]))
        offset += len(panels[name])

    time_block_draws = {key: np.empty(N_BOOTSTRAP) for key in statistics}
    for b in range(N_BOOTSTRAP):
        means = time_block_bootstrap_means(merged_panel, rng, block_length=BLOCK_LENGTH_MONTHS)
        mean_by_method = {
            name: float(np.nanmean(means[panel_slices[name]])) for name in methods
        }
        weighted_by_method = {name: rates[name] * mean_by_method[name] for name in methods}
        record_statistics(time_block_draws, b, mean_by_method, weighted_by_method)

    # Design 2: signal cluster bootstrap (decisions resampled with their signals).
    rng = np.random.default_rng(SEED_SIGNAL_CLUSTER)
    holdout_by_method = {
        name: np.array(
            [result.holdout_mean_return_percent for result in results], dtype=float
        )
        for name, results in methods.items()
    }
    cluster_draws = {key: np.empty(N_BOOTSTRAP) for key in statistics}
    for b in range(N_BOOTSTRAP):
        draw = rng.integers(0, n_signals, size=n_signals)
        mean_by_method = {}
        weighted_by_method = {}
        for name in methods:
            flags = activated_flags[name][draw]
            rate = float(np.mean(flags))
            values = holdout_by_method[name][draw][flags]
            mean = float(np.mean(values)) if values.size else float("nan")
            mean_by_method[name] = mean
            weighted_by_method[name] = rate * mean if values.size else 0.0
        record_statistics(cluster_draws, b, mean_by_method, weighted_by_method)

    rows = []
    for key, baseline in statistics.items():
        row = {"statistic": key, "baseline": baseline}
        for design, draws in [("time_block", time_block_draws), ("signal_cluster", cluster_draws)]:
            values = draws[key]
            finite = values[np.isfinite(values)]
            low, high = percentile_interval(values, coverage=0.95)
            row[f"{design}_se"] = float(np.std(finite, ddof=1))
            row[f"{design}_ci_low"] = low
            row[f"{design}_ci_high"] = high
        rows.append(row)

    csv_path = result_dir / "osap_bootstrap_inference.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['statistic']}: {row['baseline']:.4f} "
            f"[time-block se {row['time_block_se']:.4f}, "
            f"95% CI ({row['time_block_ci_low']:.4f}, {row['time_block_ci_high']:.4f})] "
            f"[cluster se {row['signal_cluster_se']:.4f}, "
            f"95% CI ({row['signal_cluster_ci_low']:.4f}, {row['signal_cluster_ci_high']:.4f})]"
        )


def build_statistics(
    baseline_mean: dict[str, float],
    baseline_weighted: dict[str, float],
) -> dict[str, float]:
    statistics: dict[str, float] = {}
    for name in baseline_mean:
        statistics[f"mean_holdout[{name}]"] = baseline_mean[name]
        statistics[f"weighted_holdout[{name}]"] = baseline_weighted[name]
    for horizon in FIXED_HORIZONS:
        statistics[f"diff_mean_holdout[sequential-fixed_{horizon}]"] = (
            baseline_mean["sequential"] - baseline_mean[f"fixed_{horizon}"]
        )
        statistics[f"diff_weighted_holdout[sequential-fixed_{horizon}]"] = (
            baseline_weighted["sequential"] - baseline_weighted[f"fixed_{horizon}"]
        )
    return statistics


def record_statistics(
    draws: dict[str, np.ndarray],
    b: int,
    mean_by_method: dict[str, float],
    weighted_by_method: dict[str, float],
) -> None:
    for name in mean_by_method:
        draws[f"mean_holdout[{name}]"][b] = mean_by_method[name]
        draws[f"weighted_holdout[{name}]"][b] = weighted_by_method[name]
    for horizon in FIXED_HORIZONS:
        draws[f"diff_mean_holdout[sequential-fixed_{horizon}]"][b] = (
            mean_by_method["sequential"] - mean_by_method[f"fixed_{horizon}"]
        )
        draws[f"diff_weighted_holdout[sequential-fixed_{horizon}]"][b] = (
            weighted_by_method["sequential"] - weighted_by_method[f"fixed_{horizon}"]
        )


if __name__ == "__main__":
    main()
