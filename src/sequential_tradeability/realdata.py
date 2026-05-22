"""Real-data evidence construction for anomaly long-short returns."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from sequential_tradeability.ekv import _require_positive
from sequential_tradeability.tradeability import ThreeStateStoppingSolution


@dataclass(frozen=True)
class ReturnSeries:
    """One cleaned anomaly return series."""

    name: str
    dates: np.ndarray
    returns: np.ndarray


@dataclass(frozen=True)
class SignalEvidence:
    """Normalized Brownian evidence stream built from real anomaly returns."""

    name: str
    dates: np.ndarray
    raw_returns: np.ndarray
    calibration_returns: np.ndarray
    validation_returns: np.ndarray
    validation_dates: np.ndarray
    holdout_returns: np.ndarray
    holdout_dates: np.ndarray
    monthly_sigma: float
    dt: float
    increments: np.ndarray
    observations: np.ndarray
    times: np.ndarray

    @property
    def calibration_months(self) -> int:
        return int(self.calibration_returns.size)

    @property
    def validation_months(self) -> int:
        return int(self.validation_returns.size)


@dataclass(frozen=True)
class RealDataPolicyResult:
    """Single-signal policy result from a real evidence stream."""

    signal: str
    decision: int
    stop_index: int
    stop_date: date
    stop_time_years: float
    evidence_at_stop: float
    posterior_mean_at_stop: float
    posterior_dead_at_stop: float
    calibration_months: int
    validation_months: int
    monthly_sigma: float
    validation_increment_mean: float
    validation_increment_variance_over_dt: float
    validation_lag1_autocorr: float
    holdout_months: int
    holdout_mean_return_percent: float
    holdout_t_stat: float

    @property
    def decision_label(self) -> str:
        if self.decision > 0:
            return "activate"
        if self.decision < 0:
            return "reject"
        return "continue"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_float(value: str) -> float:
    value = value.strip()
    if value.upper() in {"", "NA", "NAN", "."}:
        return float("nan")
    return float(value)


def load_osap_long_short_returns(path: str | Path) -> dict[str, ReturnSeries]:
    """Load Chen-Zimmermann OSAP monthly long-short returns from the wide CSV."""
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            raise ValueError("OSAP CSV must contain a date column")
        names = [name for name in reader.fieldnames if name != "date"]
        date_rows: list[date] = []
        values = {name: [] for name in names}
        for row in reader:
            date_rows.append(_parse_date(row["date"]))
            for name in names:
                values[name].append(_parse_float(row[name]))

    dates = np.array(date_rows, dtype=object)
    series: dict[str, ReturnSeries] = {}
    for name in names:
        returns = np.asarray(values[name], dtype=float)
        valid = np.isfinite(returns)
        if np.any(valid):
            series[name] = ReturnSeries(
                name=name,
                dates=dates[valid],
                returns=returns[valid],
            )
    return series


def trim_return_series(
    series: ReturnSeries,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> ReturnSeries:
    """Restrict a return series to a date interval."""
    keep = np.ones(series.returns.size, dtype=bool)
    if start_date is not None:
        keep &= series.dates >= start_date
    if end_date is not None:
        keep &= series.dates <= end_date
    if not np.any(keep):
        raise ValueError(f"{series.name} has no observations in requested date range")
    return ReturnSeries(
        name=series.name,
        dates=series.dates[keep],
        returns=series.returns[keep],
    )


def build_signal_evidence(
    series: ReturnSeries,
    *,
    calibration_months: int,
    validation_months: int,
    dt: float = 1.0 / 12.0,
) -> SignalEvidence:
    """Convert monthly returns into Brownian evidence increments.

    OSAP returns are stored in percent.  We estimate the monthly volatility from the
    calibration window and define

        Delta Y_t = (r_t / sigma_month) sqrt(dt).

    The calibration-window increment variance is therefore approximately dt, matching
    the Brownian evidence model dY_t = theta dt + dW_t.
    """
    if calibration_months < 12:
        raise ValueError("calibration_months must be at least 12")
    if validation_months < 1:
        raise ValueError("validation_months must be positive")
    _require_positive("dt", dt)

    required = calibration_months + validation_months
    if series.returns.size < required:
        raise ValueError(
            f"{series.name} has {series.returns.size} observations, but {required} are required"
        )

    calibration_returns = series.returns[:calibration_months]
    validation_returns = series.returns[calibration_months:required]
    holdout_returns = series.returns[required:]
    monthly_sigma = float(np.std(calibration_returns, ddof=1))
    _require_positive("monthly_sigma", monthly_sigma)

    increments = validation_returns / monthly_sigma * np.sqrt(dt)
    observations = np.concatenate([[0.0], np.cumsum(increments)])
    times = np.arange(validation_returns.size + 1, dtype=float) * dt

    return SignalEvidence(
        name=series.name,
        dates=series.dates,
        raw_returns=series.returns,
        calibration_returns=calibration_returns,
        validation_returns=validation_returns,
        validation_dates=series.dates[calibration_months:required],
        holdout_returns=holdout_returns,
        holdout_dates=series.dates[required:],
        monthly_sigma=monthly_sigma,
        dt=dt,
        increments=increments,
        observations=observations,
        times=times,
    )


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Return the sample lag-1 autocorrelation, or nan if not identified."""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return float("nan")
    left = values[:-1] - np.mean(values[:-1])
    right = values[1:] - np.mean(values[1:])
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def apply_three_state_solution_to_evidence(
    solution: ThreeStateStoppingSolution,
    evidence: SignalEvidence,
    *,
    holdout_months: int = 120,
) -> RealDataPolicyResult:
    """Apply a computed three-state boundary to one real evidence stream."""
    if evidence.times[-1] > solution.times[-1] + 1e-12:
        raise ValueError("solution horizon is shorter than the evidence validation window")
    if holdout_months < 1:
        raise ValueError("holdout_months must be positive")

    regions = solution.decision_regions()
    stop_index = evidence.times.size - 1
    decision = -1
    for time_index, observation in enumerate(evidence.observations):
        grid_time_index = int(round(evidence.times[time_index] / solution.dt))
        grid_time_index = min(grid_time_index, solution.times.size - 1)
        evidence_index = np.searchsorted(solution.evidence, observation)
        evidence_index = int(np.clip(evidence_index, 1, solution.evidence.size - 1))
        left = solution.evidence[evidence_index - 1]
        right = solution.evidence[evidence_index]
        nearest = (
            evidence_index
            if abs(observation - right) < abs(observation - left)
            else evidence_index - 1
        )
        region = int(regions[grid_time_index, nearest])
        if region != 0:
            stop_index = time_index
            decision = region
            break

    stop_date_index = min(stop_index, evidence.validation_dates.size - 1)
    stop_time = evidence.times[stop_index]
    grid_time_index = min(int(round(stop_time / solution.dt)), solution.times.size - 1)
    evidence_index = np.searchsorted(solution.evidence, evidence.observations[stop_index])
    evidence_index = int(np.clip(evidence_index, 1, solution.evidence.size - 1))
    left = solution.evidence[evidence_index - 1]
    right = solution.evidence[evidence_index]
    nearest = evidence_index if abs(evidence.observations[stop_index] - right) < abs(
        evidence.observations[stop_index] - left
    ) else evidence_index - 1

    holdout_start = stop_index
    available_holdout = evidence.validation_returns[holdout_start:]
    if evidence.holdout_returns.size:
        available_holdout = np.concatenate([available_holdout, evidence.holdout_returns])
    holdout = available_holdout[:holdout_months]
    if holdout.size > 1:
        holdout_mean = float(np.mean(holdout))
        holdout_std = float(np.std(holdout, ddof=1))
        holdout_t = holdout_mean / (holdout_std / np.sqrt(holdout.size))
        if holdout_std <= 0:
            holdout_t = float("nan")
    else:
        holdout_mean = float("nan")
        holdout_t = float("nan")

    return RealDataPolicyResult(
        signal=evidence.name,
        decision=decision,
        stop_index=stop_index,
        stop_date=evidence.validation_dates[stop_date_index],
        stop_time_years=float(stop_time),
        evidence_at_stop=float(evidence.observations[stop_index]),
        posterior_mean_at_stop=float(solution.posterior_mean[grid_time_index, nearest]),
        posterior_dead_at_stop=float(solution.posterior_dead[grid_time_index, nearest]),
        calibration_months=evidence.calibration_months,
        validation_months=evidence.validation_months,
        monthly_sigma=evidence.monthly_sigma,
        validation_increment_mean=float(np.mean(evidence.increments)),
        validation_increment_variance_over_dt=float(
            np.var(evidence.increments, ddof=1) / evidence.dt
        ),
        validation_lag1_autocorr=lag1_autocorrelation(evidence.increments),
        holdout_months=int(holdout.size),
        holdout_mean_return_percent=holdout_mean,
        holdout_t_stat=holdout_t,
    )
