"""
model_fitters.py
----------------
Low-level curve-fitting helpers used during row-wise foam volume imputation.

Each fitter receives observed day indices (``t``) and observed volumes (``y``)
and returns a ``FitResult`` containing the R² score, fitted values, and the
fitted model object (type depends on the model).

Design notes
~~~~~~~~~~~~
- All fitters operate only on the *observed* (non-NaN) subset passed in.
- Returning a ``FitResult`` dataclass avoids the fragile three-tuple API.
- Adding a new model requires only: a new fitter function + entry in
  ``ALL_FITTERS`` in ``imputers.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .constants import POLY_DEGREE, RF_N_ESTIMATORS, RF_RANDOM_STATE


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Outcome of a single curve-fitting attempt.

    Attributes
    ----------
    r2    : R² score on the observed data points (NaN if fitting failed).
    yhat  : Predicted values at every point in the *full* day range.
    model : Fitted model object; type varies by fitter (may be None).
    """
    r2: float
    yhat: np.ndarray
    model: Optional[Any]


# ---------------------------------------------------------------------------
# Individual fitters
# ---------------------------------------------------------------------------

def fit_linear(t: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Fit a linear model ``y = a*t + b``.

    Parameters
    ----------
    t : 1-D array of observed day indices.
    y : 1-D array of observed foam volumes (same length as *t*).

    Returns
    -------
    FitResult
    """
    model = LinearRegression().fit(t.reshape(-1, 1), y)
    yhat = model.predict(t.reshape(-1, 1))
    r2 = r2_score(y, yhat) if len(y) >= 2 else np.nan
    return FitResult(r2=r2, yhat=yhat, model=model)


def fit_exponential(t: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Fit an exponential model ``y = A * exp(B * t)`` via log-linearisation.

    Only the strictly positive *y* values are used for fitting; the full
    prediction array covers all days in *t*.

    Parameters
    ----------
    t : 1-D array of observed day indices.
    y : 1-D array of observed foam volumes.

    Returns
    -------
    FitResult
        ``model`` is a tuple ``(A, B)``; ``r2`` is NaN if fewer than two
        positive values exist.
    """
    positive_mask = y > 0
    if positive_mask.sum() < 2:
        return FitResult(r2=np.nan, yhat=np.full_like(y, np.nan, dtype=float), model=None)

    t_pos = t[positive_mask].reshape(-1, 1)
    log_y = np.log(y[positive_mask])

    lin_model = LinearRegression().fit(t_pos, log_y)
    B: float = lin_model.coef_[0]
    A: float = np.exp(lin_model.intercept_)

    yhat = A * np.exp(B * t)
    r2 = r2_score(y[positive_mask], yhat[positive_mask]) if positive_mask.sum() >= 2 else np.nan
    return FitResult(r2=r2, yhat=yhat, model=(A, B))


def fit_polynomial(t: np.ndarray, y: np.ndarray, degree: int = POLY_DEGREE) -> FitResult:
    """
    Fit a polynomial model of the given *degree* using ``numpy.polyfit``.

    Parameters
    ----------
    t      : 1-D array of observed day indices.
    y      : 1-D array of observed foam volumes.
    degree : Polynomial degree (default from ``constants.POLY_DEGREE``).

    Returns
    -------
    FitResult
        ``model`` is the numpy coefficient array returned by ``polyfit``.
    """
    if len(y) < degree + 1:
        return FitResult(r2=np.nan, yhat=np.full_like(y, np.nan, dtype=float), model=None)

    coeffs = np.polyfit(t, y, degree)
    yhat = np.polyval(coeffs, t)
    r2 = r2_score(y, yhat)
    return FitResult(r2=r2, yhat=yhat, model=coeffs)


def fit_random_forest(t: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Fit a Random Forest regressor.

    Parameters
    ----------
    t : 1-D array of observed day indices.
    y : 1-D array of observed foam volumes.

    Returns
    -------
    FitResult
        ``model`` is the fitted ``RandomForestRegressor`` instance.
    """
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
    )
    rf.fit(t.reshape(-1, 1), y)
    yhat = rf.predict(t.reshape(-1, 1))
    r2 = r2_score(y, yhat) if len(y) >= 2 else np.nan
    return FitResult(r2=r2, yhat=yhat, model=rf)


# ---------------------------------------------------------------------------
# Registry – maps model key → fitter callable
# ---------------------------------------------------------------------------

FITTER_REGISTRY: dict[str, Any] = {
    "linear": fit_linear,
    "exp": fit_exponential,
    "poly": fit_polynomial,
    "rf": fit_random_forest,
}
"""
Extend this dict to add new models without changing any other module.
The key must match a ``MODEL_*`` constant in ``constants.py``.
"""
