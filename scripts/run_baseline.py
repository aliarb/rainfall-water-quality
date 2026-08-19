"""Align rainfall with water quality, classify events, fit a baseline transfer-function model.

The full SWMM–EFDC coupling in Lin et al. (2024) needs a pipe network and
river bathymetry that we do not yet have for Hudson River Park. This script
implements the paper's *problem formulation* on observed data:

1. Rainfall intensity classes (Table 4).
2. Event detection and lag between rain and water-quality peaks.
3. A linear rainfall → water-quality impulse-response model
   (reduced form of land-to-river pollutant transfer).
4. Skill scores R² and NSE (Eqs. 4–5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_wq.config import (  # noqa: E402
    CSO_RAIN_THRESHOLD_MM,
    DATA_PROCESSED,
    DATA_RAW,
    DATA_SAMPLE,
    OUTPUTS,
    classify_daily_rain_mm,
)
from rainfall_wq.figures import plot_timeseries, plot_turbidity_by_class  # noqa: E402

TARGETS = ["turbidity_fnu", "dissolved_oxygen_mg_l", "salinity_psu", "water_temp_c"]
LAGS_DAYS = list(range(0, 8))


def load_daily_frame() -> pd.DataFrame:
    iv_path = DATA_RAW / "usgs" / "pier84_iv.parquet"
    if iv_path.exists():
        iv = pd.read_parquet(iv_path)
        local = iv.index.tz_convert("America/New_York")
        iv = iv.copy()
        iv["date"] = pd.to_datetime(local.strftime("%Y-%m-%d"))
        agg = {
            "water_temp_c": "mean",
            "specific_conductance_uS_cm": "mean",
            "dissolved_oxygen_mg_l": "mean",
            "ph": "mean",
            "turbidity_fnu": "mean",
        }
        agg = {k: v for k, v in agg.items() if k in iv.columns}
        wq = iv.groupby("date").agg(agg)
        if "turbidity_fnu" in iv.columns:
            wq["turbidity_fnu_max"] = iv.groupby("date")["turbidity_fnu"].max()
    else:
        wq = pd.read_csv(DATA_RAW / "usgs" / "pier84_daily.csv")
        dt = pd.to_datetime(wq["datetime"], utc=True).dt.tz_convert("America/New_York")
        wq["date"] = pd.to_datetime(dt.dt.strftime("%Y-%m-%d"))
        wq = wq.drop(columns=["datetime"]).groupby("date").mean(numeric_only=True)

    rain = pd.read_csv(DATA_RAW / "rainfall" / "central_park_daily.csv", parse_dates=["date"])
    rain = rain.set_index("date")
    rain["rain_class"] = rain["precip_mm"].map(classify_daily_rain_mm)
    rain["wet_cso_proxy"] = rain["precip_mm"] >= CSO_RAIN_THRESHOLD_MM

    df = wq.join(rain, how="inner")
    df["doy"] = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "pier84_daily_aligned.csv")
    return df


def event_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per wet day: rainfall class and same-day / lagged WQ anomalies."""
    wet = df[df["precip_mm"] >= 0.1].copy()
    for col in ["turbidity_fnu", "dissolved_oxygen_mg_l"]:
        if col not in df.columns:
            continue
        baseline = df[col].median()
        wet[f"{col}_anomaly"] = wet[col] - baseline
        for lag in (1, 2, 3):
            wet[f"{col}_lead{lag}"] = df[col].shift(-lag).reindex(wet.index) - baseline
    out = DATA_PROCESSED / "rain_events_daily.csv"
    wet.to_csv(out)
    return wet


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("rain_class"):
        row = {
            "rain_class": name,
            "n_days": int(len(g)),
            "precip_mm_median": float(g["precip_mm"].median()),
            "precip_mm_max": float(g["precip_mm"].max()),
        }
        for col in TARGETS:
            if col in g.columns:
                row[f"{col}_median"] = float(g[col].median())
        rows.append(row)
    out = pd.DataFrame(rows).set_index("rain_class")
    order = [c for c in ["sunny", "light", "medium", "heavy", "rainstorm", "extreme"] if c in out.index]
    out = out.reindex(order)
    out.to_csv(DATA_PROCESSED / "wq_by_rain_class.csv")
    return out


def lagged_rain_features(precip: pd.Series, lags: list[int] = LAGS_DAYS) -> pd.DataFrame:
    return pd.DataFrame({f"rain_lag{k}": precip.shift(k) for k in lags}, index=precip.index)


def exponential_kernel(precip: pd.Series, tau_days: float, max_lag: int = 7) -> pd.Series:
    """Discrete exponential impulse response: w_k ∝ exp(-k/τ)."""
    weights = np.array([np.exp(-k / tau_days) for k in range(max_lag + 1)])
    weights = weights / weights.sum()
    x = precip.fillna(0.0).to_numpy(dtype=float)
    conv = np.convolve(x, weights, mode="full")[: len(x)]
    return pd.Series(conv, index=precip.index, name=f"rain_kernel_tau{tau_days:g}")


def fit_transfer_model(df: pd.DataFrame, target: str) -> dict:
    """Y_t = β0 + β_r K_τ(R)_t + γ1 sin(doy) + γ2 cos(doy) [+ δ salinity].

    Salinity is a mixing / tide proxy and is omitted when it *is* the target.
    """
    y = df[target]
    taus = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    best = None
    for tau in taus:
        kernel = exponential_kernel(df["precip_mm"], tau)
        X = pd.DataFrame(
            {
                "rain_kernel": kernel,
                "sin_doy": df["sin_doy"],
                "cos_doy": df["cos_doy"],
            },
            index=df.index,
        )
        if target not in ("salinity_psu", "specific_conductance_uS_cm"):
            if "salinity_psu" in df.columns:
                X["salinity_psu"] = df["salinity_psu"]
            elif "specific_conductance_uS_cm" in df.columns:
                X["specific_conductance_uS_cm"] = df["specific_conductance_uS_cm"]
        pair = pd.concat([y, X], axis=1).dropna()
        if len(pair) < 60:
            continue
        yv = pair[target]
        xv = pair.drop(columns=[target])
        # Chronological hold-out: last 20%
        split = int(len(pair) * 0.8)
        model = LinearRegression()
        model.fit(xv.iloc[:split], yv.iloc[:split])
        pred_all = pd.Series(model.predict(xv), index=pair.index)
        skill_cal = skill_table(yv.iloc[:split], pred_all.iloc[:split], f"{target} cal τ={tau}")
        skill_val = skill_table(yv.iloc[split:], pred_all.iloc[split:], f"{target} val τ={tau}")
        nse = skill_val["NSE"]
        if best is None or (np.isfinite(nse) and nse > best["nse_val"]):
            best = {
                "target": target,
                "tau_days": tau,
                "nse_val": float(nse) if np.isfinite(nse) else -np.inf,
                "skill_cal": skill_cal.to_dict(),
                "skill_val": skill_val.to_dict(),
                "coef": {"intercept": float(model.intercept_), **dict(zip(xv.columns, model.coef_.tolist()))},
                "pred": pred_all,
                "obs": yv,
                "split": split,
            }
    if best is None:
        raise RuntimeError(f"Could not fit model for {target}")
    return best


def lag_correlations(df: pd.DataFrame, target: str = "turbidity_fnu") -> pd.Series:
    if target not in df.columns:
        return pd.Series(dtype=float)
    corrs = {}
    for k in LAGS_DAYS:
        corrs[k] = df[target].corr(df["precip_mm"].shift(k))
    s = pd.Series(corrs, name=f"corr_{target}_vs_rain_lag")
    s.index.name = "lag_days"
    return s


def hourly_lag_analysis(max_lag_h: int = 24) -> tuple[pd.DataFrame, pd.Series]:
    """CSO-relevant timescale: hourly Central Park rain vs Pier 84 turbidity."""
    iv = pd.read_parquet(DATA_RAW / "usgs" / "pier84_iv.parquet")
    rain = pd.read_parquet(DATA_RAW / "rainfall" / "hourly_NYC.parquet")
    turb = iv["turbidity_fnu"].resample("1h").mean().rename("turbidity_fnu")
    do = iv["dissolved_oxygen_mg_l"].resample("1h").mean().rename("dissolved_oxygen_mg_l")
    p = rain["precip_mm"].resample("1h").sum().rename("precip_mm")
    h = pd.concat([turb, do, p], axis=1, sort=True).dropna(how="all")
    rows = []
    for k in range(max_lag_h + 1):
        rows.append(
            {
                "lag_hours": k,
                "corr_turbidity": float(h["turbidity_fnu"].corr(h["precip_mm"].shift(k))),
                "corr_do": float(h["dissolved_oxygen_mg_l"].corr(h["precip_mm"].shift(k))),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "lag_corr_hourly.csv", index=False)

    wet = h["precip_mm"] >= CSO_RAIN_THRESHOLD_MM
    contrast = pd.Series(
        {
            "n_wet_hours": int(wet.sum()),
            "n_dry_hours": int((~wet).sum()),
            "turbidity_wet_median": float(h.loc[wet, "turbidity_fnu"].median()),
            "turbidity_dry_median": float(h.loc[~wet, "turbidity_fnu"].median()),
            "do_wet_median": float(h.loc[wet, "dissolved_oxygen_mg_l"].median()),
            "do_dry_median": float(h.loc[~wet, "dissolved_oxygen_mg_l"].median()),
        }
    )
    contrast.to_csv(DATA_PROCESSED / "hourly_wet_dry_contrast.csv")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(out["lag_hours"], out["corr_turbidity"], marker="o", ms=4, label="turbidity")
    ax.plot(out["lag_hours"], out["corr_do"], marker="o", ms=4, label="dissolved oxygen")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("Lag (hours), rain leading water quality")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Hourly Central Park rain vs USGS Pier 84")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "hourly_rain_wq_lag.png", dpi=140)
    plt.close(fig)
    return out, contrast


def fit_seasonality_only(df: pd.DataFrame, target: str) -> pd.Series:
    pair = df[[target, "sin_doy", "cos_doy"]].dropna()
    split = int(len(pair) * 0.8)
    model = LinearRegression()
    model.fit(pair[["sin_doy", "cos_doy"]].iloc[:split], pair[target].iloc[:split])
    pred = model.predict(pair[["sin_doy", "cos_doy"]])
    return skill_table(pair[target].iloc[split:], pred[split:], f"{target} seasonality-only val")


def make_plots(df: pd.DataFrame, models: dict[str, dict]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_timeseries(df)
    plot_turbidity_by_class(df)

    if "turbidity_fnu" in models:
        m = models["turbidity_fnu"]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(m["obs"].index, m["obs"], color="#F58518", lw=0.9, label="observed turbidity")
        ax.plot(m["pred"].index, m["pred"], color="#4C78A8", lw=0.9, label=f"model τ={m['tau_days']} d")
        split_date = m["obs"].index[m["split"]]
        ax.axvline(split_date, color="0.4", ls="--", lw=0.8, label="validation start")
        ax.set_ylabel("Turbidity (FNU)")
        ax.set_title("Baseline rainfall impulse-response model, USGS Pier 84")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUTS / "model_turbidity_fit.png", dpi=140)
        plt.close(fig)


def harbor_pathogens_vs_rain() -> pd.DataFrame | None:
    path = DATA_RAW / "harbor" / "hudson_n3b_n4_n5.csv"
    rain_path = DATA_RAW / "rainfall" / "central_park_daily.csv"
    if not path.exists() or not rain_path.exists():
        return None
    h = pd.read_csv(path, parse_dates=["sample_date"])
    rain = pd.read_csv(rain_path, parse_dates=["date"]).set_index("date")
    h["date"] = pd.to_datetime(h["sample_date"].dt.strftime("%Y-%m-%d"))
    h = h.merge(rain[["precip_mm"]], left_on="date", right_index=True, how="inner")
    h["rain_class"] = h["precip_mm"].map(classify_daily_rain_mm)
    entero = "top_enterococci_bacteria_cells_100ml"
    fecal = "top_fecal_coliform_bacteria_cells_100ml"
    for col in (entero, fecal):
        if col in h.columns:
            h[col] = pd.to_numeric(h[col], errors="coerce")
    rows = []
    for station, g in h.groupby("sampling_location"):
        for cls, gg in g.groupby("rain_class"):
            rows.append(
                {
                    "station": station,
                    "rain_class": cls,
                    "n": int(len(gg)),
                    "enterococci_median": float(gg[entero].median()) if entero in gg else float("nan"),
                    "fecal_coliform_median": float(gg[fecal].median()) if fecal in gg else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "harbor_pathogens_by_rain_class.csv", index=False)
    return out


def write_sample(df: pd.DataFrame) -> None:
    DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    df.tail(60).to_csv(DATA_SAMPLE / "pier84_daily_aligned_tail60.csv")


def main() -> None:
    df = load_daily_frame()
    print(f"aligned daily rows={len(df)}  {df.index.min().date()} → {df.index.max().date()}")
    events = event_table(df)
    print(f"wet days={len(events)}")
    summary = class_summary(df)
    print(summary)

    corr = lag_correlations(df, "turbidity_fnu")
    corr.to_csv(DATA_PROCESSED / "lag_corr_turbidity.csv")
    print("turbidity vs rain lag correlations")
    print(corr)

    hourly_corr, wet_dry = hourly_lag_analysis()
    print("hourly lag correlations (first 6 h)")
    print(hourly_corr.head(7))
    print(wet_dry)

    models = {}
    skills = []
    for target in ["turbidity_fnu", "dissolved_oxygen_mg_l"]:
        if target not in df.columns:
            continue
        fit = fit_transfer_model(df, target)
        models[target] = fit
        skills.append(pd.Series(fit["skill_cal"]))
        skills.append(pd.Series(fit["skill_val"]))
        skills.append(fit_seasonality_only(df, target))
        print(target, "tau", fit["tau_days"], "val NSE", fit["nse_val"], "val R2", fit["skill_val"].get("R2"))

    skill_df = pd.DataFrame(skills)
    skill_df.to_csv(DATA_PROCESSED / "baseline_skill.csv", index=False)

    coefs = {k: v["coef"] | {"tau_days": v["tau_days"]} for k, v in models.items()}
    (DATA_PROCESSED / "baseline_coefs.json").write_text(json.dumps(coefs, indent=2))

    make_plots(df, models)
    write_sample(df)
    harbor = harbor_pathogens_vs_rain()
    if harbor is not None:
        print("Harbor Survey pathogens by rain class")
        print(harbor)
    print("wrote processed tables and outputs/")


if __name__ == "__main__":
    main()
