"""One-at-a-time parameter sensitivity for the OSAP admission experiment.

Varies the implementation hurdle kappa, the observation cost c, and the decay rate
rho around the paper configuration (kappa = 0.005, c = 0.001, rho = 0.25), holding
eta = 0.04 and the prior fixed.  The penalty eta has its own sweep in
run_osap_penalty_sweep.py.  For each configuration the script reports real and
placebo activation rates (paper placebo draw), the activation gap, the mean decision
month, and the mean holdout return among activated real signals.
"""

from __future__ import annotations

import csv

import numpy as np
from osap_common import (
    HOLDOUT_MONTHS,
    PAPER_PLACEBO_SEED,
    PAYOFF,
    RESULT_DIR,
    load_real_evidence,
    solve_paper_boundary,
)

from sequential_tradeability import (
    TradeabilityPayoff,
    apply_three_state_solution_to_evidence,
    sign_flip_evidence,
)

BASELINE = {"kappa": 0.005, "observation_cost": 0.001, "rho": 0.25}
VARIANTS = [
    ("baseline", BASELINE),
    ("kappa_half", {**BASELINE, "kappa": 0.0025}),
    ("kappa_double", {**BASELINE, "kappa": 0.01}),
    ("cost_half", {**BASELINE, "observation_cost": 0.0005}),
    ("cost_double", {**BASELINE, "observation_cost": 0.002}),
    ("rho_half", {**BASELINE, "rho": 0.125}),
    ("rho_double", {**BASELINE, "rho": 0.5}),
]


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    real_evidence = load_real_evidence()
    placebo_rng = np.random.default_rng(PAPER_PLACEBO_SEED)
    placebo_evidence = [sign_flip_evidence(evidence, placebo_rng) for evidence in real_evidence]
    n_signals = len(real_evidence)

    rows = []
    for name, params in VARIANTS:
        print(f"solving boundary for {name}: {params}")
        payoff = TradeabilityPayoff(
            risk_aversion=PAYOFF.risk_aversion,
            return_variance=PAYOFF.return_variance,
            implementation_hurdle=params["kappa"],
        )
        solution = solve_paper_boundary(
            payoff=payoff,
            observation_cost=params["observation_cost"],
            activation_decay=params["rho"],
        )

        real_results = [
            apply_three_state_solution_to_evidence(
                solution, evidence, holdout_months=HOLDOUT_MONTHS
            )
            for evidence in real_evidence
        ]
        placebo_results = [
            apply_three_state_solution_to_evidence(
                solution, evidence, holdout_months=HOLDOUT_MONTHS
            )
            for evidence in placebo_evidence
        ]
        real_activated = [result for result in real_results if result.decision == 1]
        placebo_activated_count = sum(result.decision == 1 for result in placebo_results)

        rows.append(
            {
                "variant": name,
                "kappa": params["kappa"],
                "observation_cost": params["observation_cost"],
                "rho": params["rho"],
                "n_signals": n_signals,
                "real_activation_rate": len(real_activated) / n_signals,
                "placebo_activation_rate": placebo_activated_count / n_signals,
                "activation_gap": (len(real_activated) - placebo_activated_count) / n_signals,
                "real_mean_decision_month": 12.0
                * float(np.mean([result.stop_time_years for result in real_results])),
                "real_mean_holdout_if_activated": float(
                    np.mean([result.holdout_mean_return_percent for result in real_activated])
                ),
            }
        )
        print(rows[-1])

    csv_path = RESULT_DIR / "osap_sensitivity_sweep.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
