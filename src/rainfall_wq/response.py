"""Rain → water-quality response models: DLM, spline GAM, negative binomial.

Distributed lag (DLM)
    Y_t = harmonics(t) + Σ_{k=0}^{K} β_k P_{t-k} + ε_t

Spline GAM (sklearn SplineTransformer + Ridge)
    Y_t = harmonics(t) + f_0(P_t) + f_3(P_{t,3h}) + f_24(P_{t,24h}) + ε_t

Bacteria (Harbor Survey)
    Enterococci ~ NegativeBinomial(rain class, 48 h rain, antecedent dry days)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from .metrics import skill_table


def lagged_precip(precip: pd.Series, lags: list[int]) -> pd.DataFrame:
    return pd.DataFrame({f"p_lag{k}": precip.shift(k) for k in lags}, index=precip.index)


def rolling_precip(precip: pd.Series) -> pd.DataFrame:
    p = precip.fillna(0.0)
    return pd.DataFrame(
        {
            "p_0": p,
            "p_3h": p.rolling(3, min_periods=1).sum(),
            "p_24h": p.rolling(24, min_periods=6).sum(),
        },
        index=precip.index,
    )


def _time_split(n: int, train_frac: float = 0.8) -> int:
    return max(int(n * train_frac), 10)


def fit_dlm(
    y: pd.Series,
    precip: pd.Series,
    harmonic_design: pd.DataFrame,
    lags: list[int] | None = None,
) -> dict:
    """OLS distributed lag of rain on Y, with tidal/seasonal harmonics."""
    lags = lags or list(range(0, 13))
    rain = lagged_precip(precip, lags)
    X = pd.concat([harmonic_design, rain], axis=1)
    pair = pd.concat([y.rename("y"), X], axis=1).dropna()
    split = _time_split(len(pair))
    train, test = pair.iloc[:split], pair.iloc[split:]
    rain_cols = [c for c in rain.columns if c in pair.columns]
    harm_cols = [c for c in harmonic_design.columns if c in pair.columns]

    harm = LinearRegression().fit(train[harm_cols], train["y"])
    full = LinearRegression().fit(train[harm_cols + rain_cols], train["y"])

    pred_h_tr = harm.predict(train[harm_cols])
    pred_h_te = harm.predict(test[harm_cols])
    pred_f_tr = full.predict(train[harm_cols + rain_cols])
    pred_f_te = full.predict(test[harm_cols + rain_cols])

    lag_coef = pd.Series(full.coef_[len(harm_cols) :], index=rain_cols, name="beta")
    lag_coef.index = [int(c.replace("p_lag", "")) for c in lag_coef.index]
    lag_coef.index.name = "lag_h"

    return {
        "split": split,
        "lag_coef": lag_coef,
        "skill": pd.DataFrame(
            [
                skill_table(train["y"], pred_h_tr, "harmonic cal"),
                skill_table(test["y"], pred_h_te, "harmonic val"),
                skill_table(train["y"], pred_f_tr, "DLM cal"),
                skill_table(test["y"], pred_f_te, "DLM val"),
            ]
        ),
        "pred_val": pd.DataFrame(
            {"obs": test["y"], "harmonic": pred_h_te, "dlm": pred_f_te},
            index=test.index,
        ),
        "n_train": int(len(train)),
        "n_val": int(len(test)),
    }


def fit_gam(
    y: pd.Series,
    precip: pd.Series,
    harmonic_design: pd.DataFrame,
) -> dict:
    """Ridge GAM: harmonics (linear) + B-splines on 0 / 3 h / 24 h rain."""
    rain = rolling_precip(precip)
    X = pd.concat([harmonic_design, rain], axis=1)
    pair = pd.concat([y.rename("y"), X], axis=1).dropna()
    split = _time_split(len(pair))
    train, test = pair.iloc[:split], pair.iloc[split:]
    harm_cols = list(harmonic_design.columns)
    rain_cols = ["p_0", "p_3h", "p_24h"]

    spline = SplineTransformer(n_knots=4, degree=2, include_bias=False, knots="quantile")
    model = Pipeline(
        [
            (
                "cols",
                ColumnTransformer(
                    [
                        ("harm", "passthrough", harm_cols),
                        ("rain", spline, rain_cols),
                    ]
                ),
            ),
            ("ridge", Ridge(alpha=5.0)),
        ]
    )
    try:
        model.fit(train[harm_cols + rain_cols], train["y"])
    except ValueError:
        # Degenerate knots when a rain feature is nearly constant in train.
        model = Pipeline(
            [
                (
                    "cols",
                    ColumnTransformer(
                        [
                            ("harm", "passthrough", harm_cols),
                            ("rain", "passthrough", rain_cols),
                        ]
                    ),
                ),
                ("ridge", Ridge(alpha=5.0)),
            ]
        )
        model.fit(train[harm_cols + rain_cols], train["y"])
    pred_tr = model.predict(train[harm_cols + rain_cols])
    pred_te = model.predict(test[harm_cols + rain_cols])
    return {
        "split": split,
        "skill": pd.DataFrame(
            [
                skill_table(train["y"], pred_tr, "GAM cal"),
                skill_table(test["y"], pred_te, "GAM val"),
            ]
        ),
        "pred_val": pd.Series(pred_te, index=test.index, name="gam"),
        "n_train": int(len(train)),
        "n_val": int(len(test)),
    }


def antecedent_dry_days(daily_precip_mm: pd.Series, wet_mm: float = 0.1) -> pd.Series:
    """Days since last daily rainfall ≥ wet_mm."""
    p = daily_precip_mm.astype(float)
    last = pd.Series(pd.NaT, index=p.index)
    last_wet = pd.NaT
    for t, val in p.items():
        if np.isfinite(val) and val >= wet_mm:
            last_wet = t
        last.loc[t] = last_wet
    add = pd.Series(np.nan, index=p.index, dtype=float)
    for t in p.index:
        lw = last.loc[t]
        if pd.notna(lw):
            add.loc[t] = (t - lw) / pd.Timedelta(days=1)
    return add.rename("antecedent_dry_days")


def _fit_count_glm(formula: str, df: pd.DataFrame, y_col: str):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    used = "nb_glm"
    poi = smf.glm(formula, df, family=sm.families.Poisson()).fit(disp=0)
    mu = np.asarray(poi.fittedvalues, dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    mask = (mu > 0) & np.isfinite(mu) & np.isfinite(y)
    alpha = float(np.mean(((y[mask] - mu[mask]) ** 2 - mu[mask]) / mu[mask] ** 2))
    alpha = float(np.clip(alpha, 0.05, 40.0))
    try:
        model = smf.glm(
            formula, df, family=sm.families.NegativeBinomial(alpha=alpha)
        ).fit(disp=0, maxiter=200)
        mu_hat = pd.Series(model.predict(df), index=df.index)
        return model, mu_hat, used
    except Exception:
        model = smf.glm(formula, df, family=sm.families.Poisson()).fit(cov_type="HC3", disp=0)
        mu_hat = pd.Series(model.predict(df), index=df.index)
        return model, mu_hat, "poisson_hc3"


def fit_bacteria_nb(frame: pd.DataFrame, y_col: str, station: str) -> dict:
    """Two count models so rain class is not collinear with 48 h rainfall.

    intensity:  y ~ rain_48h_mm + antecedent_dry_days
    rain_class: y ~ C(rain_class) + antecedent_dry_days   (sunny = reference)
    """
    df = frame.dropna(subset=[y_col, "precip_mm", "antecedent_dry_days", "rain_48h_mm"]).copy()
    df = df[df[y_col] >= 0]
    df["rain_class"] = pd.Categorical(
        df["rain_class"].astype(str),
        categories=["sunny", "light", "medium", "heavy", "rainstorm", "extreme"],
    )
    df = df[df["rain_class"].notna()]
    specs = {
        "intensity": f"{y_col} ~ rain_48h_mm + antecedent_dry_days",
        "rain_class": f"{y_col} ~ C(rain_class) + antecedent_dry_days",
    }
    skills = []
    param_frames = []
    summaries = []
    primary_mu = None
    primary_used = "nb_glm"
    for spec, formula in specs.items():
        model, mu_hat, used = _fit_count_glm(formula, df, y_col)
        skills.append(skill_table(df[y_col], mu_hat, f"{station} {y_col} {spec} {used}"))
        params = model.params.to_frame("coef")
        params["pvalue"] = model.pvalues
        params["station"] = station
        params["response"] = y_col
        params["family"] = used
        params["spec"] = spec
        param_frames.append(params.reset_index().rename(columns={"index": "term"}))
        summaries.append(f"# {spec} ({used})\n{model.summary()}\n")
        if spec == "intensity":
            primary_mu = mu_hat
            primary_used = used
    return {
        "family": primary_used,
        "skill": skills[0],
        "skills": pd.DataFrame(skills),
        "params": pd.concat(param_frames, ignore_index=True),
        "n": int(len(df)),
        "fitted": primary_mu,
        "obs": df[y_col],
        "summary": "\n".join(summaries),
    }
