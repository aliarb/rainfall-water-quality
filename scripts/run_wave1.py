"""Wave 1–2 analysis: tidal residuals, storm events, first flush, DLM/GAM, N4 bacteria.

Requires data from scripts/fetch_all.py.
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
from rainfall_wq.events import delineate_events, superposed_epoch  # noqa: E402
from rainfall_wq.first_flush import first_flush_summary  # noqa: E402
from rainfall_wq.response import (  # noqa: E402
    antecedent_dry_days,
    fit_bacteria_nb,
    fit_dlm,
    fit_gam,
)
from rainfall_wq.tides import fit_harmonics, harmonic_design  # noqa: E402

TARGETS = ["turbidity_fnu", "dissolved_oxygen_mg_l", "specific_conductance_uS_cm"]


def load_hourly() -> tuple[pd.DataFrame, pd.Series]:
    iv = pd.read_parquet(DATA_RAW / "usgs" / "pier84_iv.parquet")
    if iv.index.tz is None:
        iv.index = iv.index.tz_localize("America/New_York")
    else:
        iv.index = iv.index.tz_convert("America/New_York")
    wq = iv[TARGETS].resample("1h").mean()

    rain = pd.read_parquet(DATA_RAW / "rainfall" / "hourly_NYC.parquet")
    if rain.index.tz is None:
        rain.index = rain.index.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    else:
        rain.index = rain.index.tz_convert("America/New_York")
    precip = rain["precip_mm"].resample("1h").sum().rename("precip_mm")
    return wq, precip


def fit_all_harmonics(wq: pd.DataFrame) -> dict[str, dict]:
    fits = {}
    for col in TARGETS:
        if col not in wq.columns:
            continue
        print(f"  harmonics {col} ...")
        fits[col] = fit_harmonics(wq[col])
        print(f"    n={fits[col]['n']}  R2={fits[col]['r2']:.3f}")
    return fits


def threshold_contrast(residual: pd.Series, precip: pd.Series) -> pd.Series:
    """Lite regression-discontinuity: residual WQ just below vs above 2.54 mm / 24 h."""
    p24 = precip.rolling(24, min_periods=12).sum()
    aligned = pd.concat([residual.rename("r"), p24.rename("p24")], axis=1, sort=True).dropna()
    lo = aligned[(aligned["p24"] >= 1.5) & (aligned["p24"] < CSO_RAIN_THRESHOLD_MM)]
    hi = aligned[(aligned["p24"] >= CSO_RAIN_THRESHOLD_MM) & (aligned["p24"] < 5.0)]
    return pd.Series(
        {
            "n_below": int(len(lo)),
            "n_above": int(len(hi)),
            "mean_resid_below": float(lo["r"].mean()) if len(lo) else np.nan,
            "mean_resid_above": float(hi["r"].mean()) if len(hi) else np.nan,
            "median_resid_below": float(lo["r"].median()) if len(lo) else np.nan,
            "median_resid_above": float(hi["r"].median()) if len(hi) else np.nan,
            "delta_mean": float(hi["r"].mean() - lo["r"].mean()) if len(lo) and len(hi) else np.nan,
        }
    )


def bacteria_frame(precip_hourly: pd.Series) -> pd.DataFrame:
    harbor = pd.read_csv(DATA_RAW / "harbor" / "hudson_n3b_n4_n5.csv", parse_dates=["sample_date"])
    daily = pd.read_csv(DATA_RAW / "rainfall" / "central_park_daily.csv", parse_dates=["date"])
    daily = daily.set_index("date")
    daily["antecedent_dry_days"] = antecedent_dry_days(daily["precip_mm"])

    hourly = precip_hourly.copy()
    idx = hourly.index
    if idx.tz is not None:
        local = idx.tz_convert("America/New_York")
        hourly = pd.Series(hourly.to_numpy(), index=pd.to_datetime(local.strftime("%Y-%m-%d %H:%M:%S")), name=hourly.name)
    rain_day = hourly.resample("D").sum()
    rain_48 = (rain_day + rain_day.shift(1)).rename("rain_48h_mm")

    entero = "top_enterococci_bacteria_cells_100ml"
    fecal = "top_fecal_coliform_bacteria_cells_100ml"
    harbor["date"] = pd.to_datetime(harbor["sample_date"].dt.strftime("%Y-%m-%d"))
    for col in (entero, fecal):
        harbor[col] = pd.to_numeric(harbor[col], errors="coerce")

    out = harbor.merge(daily[["precip_mm", "antecedent_dry_days"]], left_on="date", right_index=True, how="inner")
    out = out.merge(rain_48, left_on="date", right_index=True, how="left")
    out["rain_class"] = out["precip_mm"].map(classify_daily_rain_mm)
    return out.rename(
        columns={
            "sampling_location": "station",
            entero: "enterococci",
            fecal: "fecal_coliform",
        }
    )


def make_plots(
    wq: pd.DataFrame,
    precip: pd.Series,
    fits: dict,
    epoch_turb: pd.DataFrame,
    mv: pd.DataFrame,
    dlm_turb: dict,
) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(precip.index, precip.to_numpy(), width=1 / 24, color="#4C78A8")
    axes[0].set_ylabel("Precip (mm/h)")
    axes[0].set_title("Hourly Central Park rain and Pier 84 turbidity residual")
    if "turbidity_fnu" in fits:
        axes[1].plot(wq.index, wq["turbidity_fnu"], color="#F58518", lw=0.4, alpha=0.7, label="observed")
        axes[1].plot(fits["turbidity_fnu"]["fitted"].index, fits["turbidity_fnu"]["fitted"], color="0.3", lw=0.5, label="tide+season")
        axes[1].set_ylabel("Turbidity (FNU)")
        axes[1].legend(fontsize=8, loc="upper right")
        axes[2].plot(fits["turbidity_fnu"]["residual"].index, fits["turbidity_fnu"]["residual"], color="#E45756", lw=0.4)
        axes[2].set_ylabel("Turbidity residual (FNU)")
    axes[2].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "harmonic_turbidity_residual.png", dpi=140)
    plt.close(fig)

    if not epoch_turb.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for group, color in [("below_cso_threshold", "#4C78A8"), ("cso_proxy", "#E45756")]:
            g = epoch_turb[epoch_turb["group"] == group].sort_values("lead_h")
            if g.empty:
                continue
            ax.plot(g["lead_h"], g["mean"], color=color, lw=1.6, label=group.replace("_", " "))
            ax.fill_between(g["lead_h"], g["q25"], g["q75"], color=color, alpha=0.15)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.axvline(0, color="0.5", lw=0.7, ls="--")
        ax.set_xlabel("Hours from event start")
        ax.set_ylabel("Turbidity residual (FNU)")
        ax.set_title("Superposed epoch: turbidity residual around rain events")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUTS / "superposed_epoch_turbidity.png", dpi=140)
        plt.close(fig)

    if not mv.empty:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1, label="uniform washoff")
        ax.plot(mv["V"], mv["M_median"], color="#E45756", lw=2, label="median M(V)")
        ax.fill_between(mv["V"], mv["M_q25"], mv["M_q75"], color="#E45756", alpha=0.2)
        ax.set_xlabel("Cumulative rain fraction V")
        ax.set_ylabel("Cumulative excess turbidity fraction M")
        n = int(mv["n_events"].iloc[0])
        ax.set_title(f"First flush proxy, CSO-threshold events (n={n})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "first_flush_mv.png", dpi=140)
        plt.close(fig)

    lag = dlm_turb["lag_coef"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(lag.index, lag.to_numpy(), color="#4C78A8", width=0.8)
    ax.axhline(0, color="0.4", lw=0.7)
    ax.set_xlabel("Lag (hours), rain leading turbidity")
    ax.set_ylabel("DLM coefficient (FNU per mm)")
    ax.set_title("Distributed-lag rain effect on Pier 84 turbidity")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "dlm_lag_coefficients.png", dpi=140)
    plt.close(fig)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("Loading hourly WQ and rain")
    wq, precip = load_hourly()
    aligned = wq.join(precip, how="inner")
    print(f"  hourly overlap {aligned.index.min()} → {aligned.index.max()}  n={len(aligned)}")

    print("Fitting tidal + seasonal harmonics")
    fits = fit_all_harmonics(wq)
    resid = pd.DataFrame({k: v["residual"] for k, v in fits.items()})
    resid = resid.join(precip, how="left")
    resid.to_parquet(DATA_PROCESSED / "hourly_harmonic_residuals.parquet")
    harm_r2 = {k: {"r2": v["r2"], "n": v["n"]} for k, v in fits.items()}
    (DATA_PROCESSED / "harmonic_skill.json").write_text(json.dumps(harm_r2, indent=2))

    print("Delineating rain events")
    events = delineate_events(precip)
    events.to_csv(DATA_PROCESSED / "rain_events_hourly.csv", index=False)
    print(f"  events={len(events)}  CSO-proxy={int(events['cso_proxy'].sum())}")

    print("Superposed epoch and first flush")
    epoch_turb = superposed_epoch(fits["turbidity_fnu"]["residual"], events)
    epoch_do = superposed_epoch(fits["dissolved_oxygen_mg_l"]["residual"], events)
    epoch_turb.to_csv(DATA_PROCESSED / "epoch_turbidity.csv", index=False)
    epoch_do.to_csv(DATA_PROCESSED / "epoch_do.csv", index=False)
    mv = first_flush_summary(precip, fits["turbidity_fnu"]["residual"], events)
    if not mv.empty:
        mv.to_csv(DATA_PROCESSED / "first_flush_mv.csv", index=False)
        print(f"  first-flush M(V=0.3)={mv.attrs.get('M_at_V30', float('nan')):.3f}")

    contrast = threshold_contrast(fits["turbidity_fnu"]["residual"], precip)
    contrast.to_csv(DATA_PROCESSED / "cso_threshold_contrast.csv")
    print("  CSO-threshold residual contrast")
    print(contrast.to_string())

    print("DLM and GAM on turbidity / DO")
    design = harmonic_design(aligned.index)
    skills = []
    models = {}
    for target in ["turbidity_fnu", "dissolved_oxygen_mg_l"]:
        dlm = fit_dlm(aligned[target], aligned["precip_mm"], design)
        try:
            gam = fit_gam(aligned[target], aligned["precip_mm"], design)
        except Exception as exc:
            print(f"  GAM failed for {target}: {exc}")
            gam = None
        dlm["skill"].insert(0, "target", target)
        skills.append(dlm["skill"])
        if gam is not None:
            gam["skill"].insert(0, "target", target)
            skills.append(gam["skill"])
        models[target] = {"dlm": dlm, "gam": gam}
        print(dlm["skill"][["label", "R2", "NSE"]].to_string(index=False))
        if gam is not None:
            print(gam["skill"][["label", "R2", "NSE"]].to_string(index=False))

    skill_df = pd.concat(skills, ignore_index=True)
    skill_df.to_csv(DATA_PROCESSED / "wave1_skill.csv", index=False)
    models["turbidity_fnu"]["dlm"]["lag_coef"].to_csv(DATA_PROCESSED / "dlm_turbidity_lags.csv")

    print("Harbor Survey bacteria models")
    bact = bacteria_frame(precip)
    bact.to_csv(DATA_PROCESSED / "harbor_bacteria_covariates.csv", index=False)
    param_rows = []
    bact_skills = []
    for station in ["N4", "N5"]:
        sub = bact[bact["station"] == station].copy()
        for ycol, short in [("enterococci", "enterococci"), ("fecal_coliform", "fecal")]:
            try:
                fit = fit_bacteria_nb(sub, ycol, station)
            except Exception as exc:
                print(f"  {station} {ycol} failed: {exc}")
                continue
            print(f"  {station} {ycol} family={fit['family']} n={fit['n']} R2={fit['skill']['R2']:.3f}")
            bact_skills.append(fit["skills"] if "skills" in fit else fit["skill"])
            param_rows.append(fit["params"])
            summary_path = DATA_PROCESSED / f"bacteria_{station}_{short}_{fit['family']}.txt"
            summary_path.write_text(fit["summary"])
    if param_rows:
        params_all = pd.concat(param_rows, ignore_index=True)
        params_all.to_csv(DATA_PROCESSED / "bacteria_nb_params.csv", index=False)
        params_all.to_csv(DATA_SAMPLE / "bacteria_nb_params.csv", index=False)
        n4 = params_all[
            (params_all["station"] == "N4")
            & (params_all["response"] == "enterococci")
            & (params_all["spec"] == "rain_class")
        ]
        class_rows = n4[n4["term"].str.contains("rain_class", na=False)]
        if not class_rows.empty:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            labels = class_rows["term"].str.replace(r"C\(rain_class\)\[T\.", "", regex=True).str.replace("]", "", regex=False)
            ax.barh(labels, class_rows["coef"], color="#E45756")
            ax.axvline(0, color="0.4", lw=0.7)
            ax.set_xlabel("Log-mean enterococci vs sunny days (N4)")
            ax.set_title("Harbor Survey N4 rain-class effects")
            fig.tight_layout()
            fig.savefig(OUTPUTS / "n4_enterococci_rain_class.png", dpi=140)
            plt.close(fig)
    if bact_skills:
        pd.concat(bact_skills, ignore_index=True).to_csv(DATA_PROCESSED / "bacteria_skill.csv", index=False)

    print("Figures")
    make_plots(wq, precip, fits, epoch_turb, mv, models["turbidity_fnu"]["dlm"])

    DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    events.tail(40).to_csv(DATA_SAMPLE / "rain_events_hourly_tail.csv", index=False)
    skill_df.to_csv(DATA_SAMPLE / "wave1_skill.csv", index=False)
    if not mv.empty:
        mv.to_csv(DATA_SAMPLE / "first_flush_mv.csv", index=False)
    harm_sample = pd.Series({k: v["r2"] for k, v in fits.items()}, name="harmonic_r2")
    harm_sample.to_csv(DATA_SAMPLE / "harmonic_r2.csv")
    print("done")


if __name__ == "__main__":
    main()
