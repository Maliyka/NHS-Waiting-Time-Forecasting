"""
metrics.py — Forecast accuracy metrics for the NHS Forecasting project.

Implements MAE, RMSE, MAPE, MASE, interval coverage, and a naive seasonal
baseline. All functions handle edge cases (empty arrays, zero denominators).
Francis Kwesi Acquah | B01821156 | UWS
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ── Individual metrics ─────────────────────────────────────────────────────────

def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """
    Mean Absolute Error.

    MAE = mean(|actual - predicted|)

    Returns None if either array is empty or has mismatched length.
    """
    actual, predicted = _validate_arrays(actual, predicted)
    if actual is None:
        return None
    errors = np.abs(actual - predicted)
    return float(np.mean(errors))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """
    Root Mean Squared Error.

    RMSE = sqrt(mean((actual - predicted)^2))

    Returns None if arrays are empty or mismatched.
    """
    actual, predicted = _validate_arrays(actual, predicted)
    if actual is None:
        return None
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """
    Mean Absolute Percentage Error (%).

    MAPE = mean(|actual - predicted| / |actual|) * 100

    Skips time periods where actual == 0 to avoid division by zero.
    Returns None if no valid periods remain.
    """
    actual, predicted = _validate_arrays(actual, predicted)
    if actual is None:
        return None

    nonzero = actual != 0
    if not np.any(nonzero):
        return None

    pct_errors = np.abs(
        (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
    ) * 100
    return float(np.mean(pct_errors))


def compute_mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    seasonal_period: int = 12,
) -> Optional[float]:
    """
    Mean Absolute Scaled Error.

    MASE = MAE_model / MAE_naive_seasonal

    The naive seasonal baseline predicts each value as the value from
    exactly one seasonal period ago (e.g. 12 months ago for monthly data).

    A MASE < 1 means the model beats the naive seasonal baseline.
    A MASE > 1 means the naive baseline was better.

    Args:
        actual:          Array of actual values.
        predicted:       Array of model predictions.
        seasonal_period: Seasonal period (12 for monthly data).

    Returns:
        MASE score, or None if scale cannot be computed.
    """
    actual, predicted = _validate_arrays(actual, predicted)
    if actual is None:
        return None

    mae_model = compute_mae(actual, predicted)
    if mae_model is None:
        return None

    # Naive seasonal scale: mean absolute difference y_t vs y_{t-m}
    if len(actual) <= seasonal_period:
        return None
    naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
    scale = float(np.mean(naive_errors))

    if scale == 0:
        return None

    return float(mae_model / scale)


def compute_interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Optional[float]:
    """
    Proportion of actual values that fall within the prediction interval.

    Returns a value between 0 and 1. For a well-calibrated 80% PI, this
    should be close to 0.80.

    Args:
        actual: Array of actual observed values.
        lower:  Array of lower bound predictions.
        upper:  Array of upper bound predictions.

    Returns:
        Coverage proportion (0.0 – 1.0), or None if arrays are invalid.
    """
    actual = np.asarray(actual, dtype=float)
    lower  = np.asarray(lower,  dtype=float)
    upper  = np.asarray(upper,  dtype=float)

    if len(actual) == 0 or not (len(actual) == len(lower) == len(upper)):
        return None

    within = np.sum((actual >= lower) & (actual <= upper))
    return float(within / len(actual))


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower_80: Optional[np.ndarray] = None,
    upper_80: Optional[np.ndarray] = None,
    lower_95: Optional[np.ndarray] = None,
    upper_95: Optional[np.ndarray] = None,
    seasonal_period: int = 12,
) -> Dict[str, Optional[float]]:
    """
    Compute all forecast accuracy metrics in one call.

    Args:
        actual:          Array of observed values.
        predicted:       Array of point forecasts.
        lower_80/upper_80: 80% prediction interval bounds (optional).
        lower_95/upper_95: 95% prediction interval bounds (optional).
        seasonal_period: Seasonal period for MASE.

    Returns:
        Dict with keys: mae, rmse, mape, mase, coverage_80, coverage_95.
        Values are None if they cannot be computed.
    """
    result = {
        "mae":         compute_mae(actual, predicted),
        "rmse":        compute_rmse(actual, predicted),
        "mape":        compute_mape(actual, predicted),
        "mase":        compute_mase(actual, predicted, seasonal_period),
        "coverage_80": None,
        "coverage_95": None,
    }

    if lower_80 is not None and upper_80 is not None:
        result["coverage_80"] = compute_interval_coverage(actual, lower_80, upper_80)

    if lower_95 is not None and upper_95 is not None:
        result["coverage_95"] = compute_interval_coverage(actual, lower_95, upper_95)

    return result


# ── Naive seasonal baseline ────────────────────────────────────────────────────

def naive_seasonal_forecast(
    series: pd.Series,
    horizon: int,
    seasonal_period: int = 12,
) -> pd.Series:
    """
    Produce a naive seasonal forecast: predict each future value as the
    value from exactly one seasonal period ago.

    This is the standard baseline for MASE computation and also a useful
    sanity check — any model should beat this.

    Args:
        series:          Historical time series (DatetimeIndex or RangeIndex).
        horizon:         Number of periods ahead to forecast.
        seasonal_period: Seasonal period (12 for monthly).

    Returns:
        pd.Series of length `horizon` with naive forecasts.
    """
    values = series.values
    n      = len(values)

    if n < seasonal_period:
        raise ValueError(
            f"Series length ({n}) must be >= seasonal_period ({seasonal_period})"
        )

    forecasts = []
    for h in range(1, horizon + 1):
        # Index into the series: position of the value one seasonal period before
        # the h-th future period
        idx = n - seasonal_period + ((h - 1) % seasonal_period)
        forecasts.append(float(values[idx]))

    return pd.Series(forecasts)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate_arrays(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple:
    """
    Convert inputs to float numpy arrays and validate.

    Returns (None, None) if arrays are empty or have mismatched lengths.
    """
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0 or len(actual) != len(predicted):
        return None, None

    # Remove rows where either value is NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual    = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return None, None

    return actual, predicted
