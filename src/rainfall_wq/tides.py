"""Astronomical + seasonal harmonics for a tidal estuary sonde.

Fit
    Y(t) = β0 + Σ_k [ a_k sin(2π t / T_k) + b_k cos(2π t / T_k) ]
by OLS on the observed 15-minute or hourly series. Residuals are the
working series for rainfall / CSO analysis so M2 and the annual cycle
are not mistaken for a storm response.

Periods follow standard tidal constituents (hours). S1 (24 h) is included
because dissolved oxygen has a strong solar-day cycle; Sa / Ssa capture
the seasonal envelope that dominated the daily DO baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Period in hours. Shallow-water M4 is often visible in the Hudson estuary.
CONSTITUENT_PERIOD_H = {
    "M2": 12.4206012,
    "S2": 12.0,
    "N2": 12.65834751,
    "K1": 23.93447213,
    "O1": 25.81933871,
    "M4": 6.2103006,
    "S1": 24.0,
    "Sa": 365.2422 * 24.0,
    "Ssa": 182.6211 * 24.0,
}

def hours_since_epoch(index: pd.DatetimeIndex) -> np.ndarray:
    utc = pd.to_datetime(index, utc=True)
    origin = pd.Timestamp("2000-01-01", tz="UTC")
    return (utc - origin).total_seconds().to_numpy() / 3600.0


def harmonic_design(index: pd.DatetimeIndex, constituents: dict[str, float] | None = None) -> pd.DataFrame:
    constituents = constituents or CONSTITUENT_PERIOD_H
    t = hours_since_epoch(index)
    cols = {}
    for name, period in constituents.items():
        omega = 2.0 * np.pi / period
        cols[f"{name}_sin"] = np.sin(omega * t)
        cols[f"{name}_cos"] = np.cos(omega * t)
    return pd.DataFrame(cols, index=index)


def fit_harmonics(
    series: pd.Series,
    constituents: dict[str, float] | None = None,
) -> dict:
    """Return fitted values, residuals, coefficients, and in-sample R²."""
    y = series.astype(float)
    X = harmonic_design(y.index, constituents)
    pair = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(pair) < X.shape[1] + 10:
        raise ValueError(f"Not enough finite observations to fit harmonics for {series.name}.")
    model = LinearRegression()
    model.fit(pair.drop(columns=["y"]), pair["y"])
    fitted = pd.Series(model.predict(X.fillna(0.0)), index=y.index, name=f"{series.name}_harmonic")
    residual = (y - fitted).rename(f"{series.name}_resid")
    yhat = pd.Series(model.predict(pair.drop(columns=["y"])), index=pair.index)
    sse = float(((pair["y"] - yhat) ** 2).sum())
    sst = float(((pair["y"] - pair["y"].mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst else np.nan
    coef = {"intercept": float(model.intercept_), **dict(zip(X.columns, model.coef_.tolist()))}
    return {
        "name": series.name,
        "fitted": fitted,
        "residual": residual,
        "coef": coef,
        "r2": r2,
        "n": int(len(pair)),
        "design": X,
        "model": model,
    }
