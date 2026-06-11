"""Canonical configuration and data loading for the OSAP admission experiments.

Every OSAP experiment in the paper uses this exact configuration; scripts import from
here so that parameter values exist in one place.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sequential_tradeability import (  # noqa: E402
    SignalEvidence,
    ThreeStatePrior,
    ThreeStateStoppingSolution,
    TradeabilityPayoff,
    build_signal_evidence,
    load_osap_long_short_returns,
    solve_three_state_tradeability,
    trim_return_series,
)

RAW_OSAP_PATH = REPO_ROOT / "data/raw/osap_monthly_long_short_returns.csv"
RESULT_DIR = REPO_ROOT / "results"

CALIBRATION_MONTHS = 120
VALIDATION_MONTHS = 240
HOLDOUT_MONTHS = 120
DT = 1.0 / 12.0
COMMON_START_DATE = date(1970, 1, 1)
PAPER_PENALTY = 0.04
PAPER_PLACEBO_SEED = 34_219

PRIOR = ThreeStatePrior(
    alpha=0.38729833462074165,
    prob_negative=0.15,
    prob_dead=0.70,
    prob_positive=0.15,
)
PAYOFF = TradeabilityPayoff(
    risk_aversion=1.0,
    return_variance=0.05,
    implementation_hurdle=0.005,
)
OBSERVATION_COST = 0.001
ACTIVATION_DECAY = 0.25
EVIDENCE_MAX = 8.0
N_TIME = VALIDATION_MONTHS + 1
N_EVIDENCE = 801


def solve_paper_boundary(
    *,
    dead_activation_penalty: float = PAPER_PENALTY,
    payoff: TradeabilityPayoff = PAYOFF,
    observation_cost: float = OBSERVATION_COST,
    activation_decay: float = ACTIVATION_DECAY,
) -> ThreeStateStoppingSolution:
    """Solve the three-state variational inequality at the paper configuration."""
    return solve_three_state_tradeability(
        prior=PRIOR,
        payoff_model=payoff,
        observation_cost=observation_cost,
        horizon=VALIDATION_MONTHS * DT,
        evidence_max=EVIDENCE_MAX,
        activation_decay=activation_decay,
        dead_activation_penalty=dead_activation_penalty,
        n_time=N_TIME,
        n_evidence=N_EVIDENCE,
    )


def load_real_evidence() -> list[SignalEvidence]:
    """Load the 192 eligible OSAP evidence streams used throughout the paper."""
    if not RAW_OSAP_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_OSAP_PATH}; run the OSAP download step first.")
    series = load_osap_long_short_returns(RAW_OSAP_PATH)
    required = CALIBRATION_MONTHS + VALIDATION_MONTHS
    evidence = []
    for name in sorted(series):
        trimmed = trim_return_series(series[name], start_date=COMMON_START_DATE)
        if trimmed.returns.size < required:
            continue
        evidence.append(
            build_signal_evidence(
                trimmed,
                calibration_months=CALIBRATION_MONTHS,
                validation_months=VALIDATION_MONTHS,
                dt=DT,
            )
        )
    if len(evidence) != 192:
        raise RuntimeError(f"Expected 192 eligible OSAP signals, found {len(evidence)}")
    return evidence
