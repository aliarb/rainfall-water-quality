"""Rain-class and event-scale responses for the manuscript hypotheses.

H1  bacteria rise with 48 h rain at N3B, N4, N5 (negative-binomial GLM)
H2  responses grow with rain class after season is removed (daily) and
    after tides are removed (event scale)
H3  timing of the response after rain onset
H4  delivery versus dilution: conductance at Pier 84, grab salinity,
    ammonium, and suspended solids against 48 h rain

Reads outputs of scripts/run_baseline.py and scripts/run_wave1.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_wq.config import CSO_RAIN_THRESHOLD_MM, DATA_PROCESSED, DATA_SAMPLE  # noqa: E402
from rainfall_wq.metrics import skill_table  # noqa: E402
from rainfall_wq.response import _fit_count_glm  # noqa: E402

ORDER = ["sunny", "light", "medium", "heavy", "rainstorm", "extreme"]
DAILY_VARS = ["turbidity_fnu", "dissolved_oxygen_mg_l", "specific_conductance_uS_cm", "ph"]
HOURLY_VARS = ["turbidity_fnu", "dissolved_oxygen_mg_l", "specific_conductance_uS_cm"]
EVENT_CLASSES = [
    ("below_proxy", 0.0, CSO_RAIN_THRESHOLD_MM),
    ("light", CSO_RAIN_THRESHOLD_MM, 10.0),
    ("medium", 10.0, 25.0),
    ("heavy", 25.0, 50.0),
    ("rainstorm", 50.0, 100.0),
    ("extreme", 100.0, np.inf),
]
PLOT_GROUPS = {
    "below_proxy": "< 2.54 mm",
    "light": "2.54-24.9 mm",
    "medium": "2.54-24.9 mm",
    "heavy": ">= 25 mm",
    "rainstorm": ">= 25 mm",
    "extreme": ">= 25 mm",
    "placebo": "placebo",
}
PRE_H = (-6, -1)
POST_WINDOWS = {"0-12h": (0, 12), "0-24h": (0, 24), "24-48h": (24, 48)}
LEADS = np.arange(-6, 49)
GRAB_ANALYTES = {
    "top_salinity_psu": ("salinity", "linear"),
    "top_ammonium_mg_l": ("ammonium", "log"),
    "top_total_suspended_solid_mg_l": ("tss", "log"),
    "winkler_method_top_dissolved_oxygen_mg_l": ("grab_do", "linear"),
}
RNG = np.random.default_rng(20261029)
TZ = "America/New_York"


def boot_median_ci(x: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def seasonal_anomaly(daily: pd.DataFrame, col: str) -> pd.Series:
    """Anomaly from an annual cycle plus year offsets fitted on sunny days only."""
    d = daily[["date", col, "rain_class"]].dropna(subset=[col]).copy()
    w = 2 * np.pi * d["date"].dt.day_of_year.to_numpy() / 365.25
    years = pd.get_dummies(d["date"].dt.year, drop_first=True, dtype=float).to_numpy()
    X = np.column_stack([np.ones(len(d)), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w), years])
    sunny = (d["rain_class"] == "sunny").to_numpy()
    beta, *_ = np.linalg.lstsq(X[sunny], d[col].to_numpy()[sunny], rcond=None)
    return pd.Series(d[col].to_numpy() - X @ beta, index=d["date"], name=col)


def daily_class_tables(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, lag_rows = [], []
    cls = daily.set_index("date")["rain_class"]
    full_idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    for col in DAILY_VARS:
        anom = seasonal_anomaly(daily, col)
        raw = daily.set_index("date")[col]
        sunny_anom = anom[cls.reindex(anom.index) == "sunny"].to_numpy()
        for c in ORDER:
            a = anom[cls.reindex(anom.index) == c].to_numpy()
            r = raw[(cls == c)].dropna().to_numpy()
            lo, hi = boot_median_ci(a)
            p = np.nan
            if c != "sunny" and len(a) >= 3:
                p = float(stats.mannwhitneyu(a, sunny_anom, alternative="two-sided").pvalue)
            rows.append(
                {
                    "variable": col,
                    "rain_class": c,
                    "n_days_class": int((cls == c).sum()),
                    "n_obs": int(len(a)),
                    "raw_median": float(np.median(r)) if len(r) else np.nan,
                    "anomaly_median": float(np.median(a)) if len(a) else np.nan,
                    "anomaly_ci_lo": lo,
                    "anomaly_ci_hi": hi,
                    "p_vs_sunny": p,
                }
            )
        ref = float(np.median(sunny_anom))
        for r in rows:
            if r["variable"] == col:
                r["shift_vs_sunny"] = r["anomaly_median"] - ref
        anom_full = anom.reindex(full_idx)
        for c in ORDER[1:]:
            days = cls.index[cls == c]
            for k in range(0, 4):
                vals = anom_full.reindex(days + pd.Timedelta(days=k)).dropna().to_numpy()
                lag_rows.append(
                    {
                        "variable": col,
                        "rain_class": c,
                        "lag_day": k,
                        "n": int(len(vals)),
                        "anomaly_median": float(np.median(vals)) if len(vals) else np.nan,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(lag_rows)


def event_class(depth: float) -> str:
    for name, lo, hi in EVENT_CLASSES:
        if lo <= depth < hi:
            return name
    return "extreme"


def load_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev["start"] = pd.to_datetime(ev["start"], utc=True).dt.tz_convert(TZ)
    ev["event_class"] = ev["depth_mm"].map(event_class)
    return ev


def placebo_starts(ev: pd.DataFrame, resid: pd.DataFrame, shifts: tuple[int, ...] = (-7, 7)) -> pd.DataFrame:
    """Same clock hour and season as each proxy event, shifted by a week, with no rain in the window."""
    wet = resid["precip_mm"].fillna(0.0) >= 0.1
    rows = []
    for row in ev[ev["depth_mm"] >= CSO_RAIN_THRESHOLD_MM].itertuples(index=False):
        for s in shifts:
            t0 = row.start + pd.Timedelta(days=s)
            win = pd.date_range(t0 - pd.Timedelta(hours=6), t0 + pd.Timedelta(hours=24), freq="h")
            if wet.reindex(win, fill_value=False).any():
                continue
            rows.append(
                {
                    "event_id": f"{row.event_id}{s:+d}d",
                    "start": t0,
                    "event_class": "placebo",
                    "depth_mm": 0.0,
                    "peak_mm_h": 0.0,
                }
            )
    return pd.DataFrame(rows)


def event_responses(resid: pd.DataFrame, ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_event, curves = [], []
    for row in ev.itertuples(index=False):
        times = pd.DatetimeIndex([row.start + pd.Timedelta(hours=int(h)) for h in LEADS])
        for col in HOURLY_VARS:
            vals = resid[col].reindex(times).to_numpy(dtype=float)
            pre_mask = (LEADS >= PRE_H[0]) & (LEADS <= PRE_H[1])
            pre = vals[pre_mask]
            if np.isfinite(pre).mean() < 0.5:
                continue
            base = float(np.nanmean(pre))
            rec = {
                "event_id": row.event_id,
                "variable": col,
                "event_class": row.event_class,
                "depth_mm": row.depth_mm,
                "peak_mm_h": row.peak_mm_h,
            }
            for name, (a, b) in POST_WINDOWS.items():
                post = vals[(LEADS >= a) & (LEADS <= b)]
                rec[f"delta_{name}"] = (
                    float(np.nanmean(post) - base) if np.isfinite(post).mean() >= 0.5 else np.nan
                )
            per_event.append(rec)
            curves.append(
                pd.DataFrame(
                    {
                        "event_id": row.event_id,
                        "variable": col,
                        "event_class": row.event_class,
                        "lead_h": LEADS,
                        "delta": vals - base,
                    }
                )
            )
    return pd.DataFrame(per_event), pd.concat(curves, ignore_index=True)


def summarize_events(per_event: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, rho_rows = [], []
    groups = [c for c, _, _ in EVENT_CLASSES] + ["cso_proxy_all", "placebo"]
    for col in HOURLY_VARS:
        sub = per_event[per_event["variable"] == col]
        real = sub[sub["event_class"] != "placebo"]
        for name, _ in POST_WINDOWS.items():
            key = f"delta_{name}"
            placebo = sub.loc[sub["event_class"] == "placebo", key].dropna().to_numpy()
            for c in groups:
                if c == "cso_proxy_all":
                    mask = (sub["event_class"] != "placebo") & (sub["depth_mm"] >= CSO_RAIN_THRESHOLD_MM)
                else:
                    mask = sub["event_class"] == c
                x = sub.loc[mask, key].dropna().to_numpy()
                lo, hi = boot_median_ci(x)
                p = float(stats.wilcoxon(x).pvalue) if len(x) >= 6 else np.nan
                p_pl = (
                    float(stats.mannwhitneyu(x, placebo, alternative="two-sided").pvalue)
                    if c != "placebo" and len(x) >= 3 and len(placebo) >= 3
                    else np.nan
                )
                rows.append(
                    {
                        "variable": col,
                        "window": name,
                        "event_class": c,
                        "n": int(len(x)),
                        "median_delta": float(np.median(x)) if len(x) else np.nan,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p_wilcoxon": p,
                        "p_vs_placebo": p_pl,
                    }
                )
            ok = real[[key, "depth_mm"]].dropna()
            rho, p = stats.spearmanr(ok["depth_mm"], ok[key])
            rho_rows.append(
                {"variable": col, "window": name, "n": int(len(ok)), "spearman_rho": float(rho), "p": float(p)}
            )
    return pd.DataFrame(rows), pd.DataFrame(rho_rows)


def epoch_by_group(curves: pd.DataFrame) -> pd.DataFrame:
    curves = curves.copy()
    curves["group"] = curves["event_class"].map(PLOT_GROUPS)
    rows = []
    for (col, grp, lead), part in curves.groupby(["variable", "group", "lead_h"]):
        x = part["delta"].dropna().to_numpy()
        lo, hi = boot_median_ci(x, n_boot=1000)
        rows.append(
            {
                "variable": col,
                "group": grp,
                "lead_h": int(lead),
                "n": int(len(x)),
                "median": float(np.median(x)) if len(x) else np.nan,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def dep_flag(x: object) -> str:
    s = str(x).strip().lower()
    if s in {"d", "dry"}:
        return "dry"
    if s in {"w", "wet"}:
        return "wet"
    return "unlabelled"


def bacteria_tables(bact: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    rows = []
    for station in ["N4", "N5", "N3B"]:
        for c in ORDER:
            sub = bact[(bact["station"] == station) & (bact["rain_class"] == c)]
            rows.append(
                {
                    "station": station,
                    "rain_class": c,
                    "n_entero": int(sub["enterococci"].notna().sum()),
                    "n_fecal": int(sub["fecal_coliform"].notna().sum()),
                    "enterococci_median": float(sub["enterococci"].median()) if len(sub) else np.nan,
                    "fecal_coliform_median": float(sub["fecal_coliform"].median()) if len(sub) else np.nan,
                }
            )
    out["by_class"] = pd.DataFrame(rows)

    b = bact.copy()
    b["dep_flag"] = b["weather_condition_dry_or_wet"].map(dep_flag)
    out["by_dep_flag"] = (
        b.groupby(["station", "dep_flag"])
        .agg(
            n=("fecal_coliform", "count"),
            enterococci_median=("enterococci", "median"),
            fecal_coliform_median=("fecal_coliform", "median"),
            rain_48h_median=("rain_48h_mm", "median"),
        )
        .reset_index()
    )

    params, curves = [], []
    for station in ["N4", "N5", "N3B"]:
        for y in ["fecal_coliform", "enterococci"]:
            df = bact[bact["station"] == station].dropna(
                subset=[y, "precip_mm", "rain_48h_mm", "antecedent_dry_days", "rain_class"]
            )
            df = df[df[y] >= 0].copy()
            w = 2 * np.pi * pd.to_datetime(df["date"]).dt.day_of_year.to_numpy() / 365.25
            df["sin_doy"], df["cos_doy"] = np.sin(w), np.cos(w)
            specs = {
                "rain": f"{y} ~ rain_48h_mm + antecedent_dry_days",
                "rain_season": f"{y} ~ rain_48h_mm + antecedent_dry_days + sin_doy + cos_doy",
            }
            for spec, formula in specs.items():
                model, mu_hat, family = _fit_count_glm(formula, df, y)
                sk = skill_table(df[y], mu_hat, f"{station} {y} {spec}")
                for term in ["rain_48h_mm", "antecedent_dry_days"]:
                    params.append(
                        {
                            "station": station,
                            "response": y,
                            "spec": spec,
                            "term": term,
                            "coef": float(model.params[term]),
                            "se": float(model.bse[term]),
                            "p": float(model.pvalues[term]),
                            "rate_ratio_per_10mm": float(np.exp(10 * model.params[term]))
                            if term == "rain_48h_mm"
                            else np.nan,
                            "R2": float(sk["R2"]),
                            "n": int(len(df)),
                            "family": family,
                        }
                    )
                if spec == "rain" and y == "fecal_coliform":
                    grid = pd.DataFrame(
                        {
                            "rain_48h_mm": np.arange(0, 61, 1.0),
                            "antecedent_dry_days": float(df["antecedent_dry_days"].median()),
                        }
                    )
                    pred = model.get_prediction(grid).summary_frame(alpha=0.05)
                    curves.append(
                        pd.DataFrame(
                            {
                                "station": station,
                                "rain_48h_mm": grid["rain_48h_mm"],
                                "mean": pred["mean"].to_numpy(),
                                "ci_lo": pred["mean_ci_lower"].to_numpy(),
                                "ci_hi": pred["mean_ci_upper"].to_numpy(),
                            }
                        )
                    )
    out["nb"] = pd.DataFrame(params)
    out["nb_curves"] = pd.concat(curves, ignore_index=True)
    return out


def grab_analytes(bact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = bact.copy()
    w = 2 * np.pi * pd.to_datetime(b["date"]).dt.day_of_year.to_numpy() / 365.25
    b["sin_doy"], b["cos_doy"] = np.sin(w), np.cos(w)
    rows, eff = [], []
    for col, (short, scale) in GRAB_ANALYTES.items():
        b[col] = pd.to_numeric(b[col], errors="coerce")
        for c in ORDER:
            x = b.loc[b["rain_class"] == c, col].dropna()
            rows.append(
                {"analyte": short, "rain_class": c, "n": int(len(x)), "median": float(x.median()) if len(x) else np.nan}
            )
        df = b.dropna(subset=[col, "rain_48h_mm"]).copy()
        if scale == "log":
            df = df[df[col] > 0]
            df["y"] = np.log(df[col])
        else:
            df["y"] = df[col]
        fit = smf.ols("y ~ rain_48h_mm + sin_doy + cos_doy + C(station)", df).fit(cov_type="HC3")
        coef = float(fit.params["rain_48h_mm"])
        eff.append(
            {
                "analyte": short,
                "scale": scale,
                "n": int(len(df)),
                "coef_per_mm": coef,
                "effect_per_10mm": float(np.exp(10 * coef) - 1) if scale == "log" else 10 * coef,
                "p": float(fit.pvalues["rain_48h_mm"]),
                "spearman_rho": float(stats.spearmanr(df["rain_48h_mm"], df[col])[0]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(eff)


def main() -> None:
    daily = pd.read_csv(DATA_PROCESSED / "pier84_daily_aligned.csv", parse_dates=["date"])
    resid = pd.read_parquet(DATA_PROCESSED / "hourly_harmonic_residuals.parquet")
    events = pd.read_csv(DATA_PROCESSED / "rain_events_hourly.csv")
    bact = pd.read_csv(DATA_PROCESSED / "harbor_bacteria_covariates.csv")

    outputs: dict[str, pd.DataFrame] = {}
    outputs["class_daily_anomaly"], outputs["class_daily_lag"] = daily_class_tables(daily)

    ev = load_events(events)
    starts = pd.concat([ev, placebo_starts(ev, resid)], ignore_index=True)
    per_event, curves = event_responses(resid, starts)
    outputs["event_response_by_class"], outputs["event_response_spearman"] = summarize_events(per_event)
    outputs["epoch_by_group"] = epoch_by_group(curves)

    bt = bacteria_tables(bact)
    outputs["bacteria_by_class_full"] = bt["by_class"]
    outputs["bacteria_by_dep_flag"] = bt["by_dep_flag"]
    outputs["bacteria_nb_all"] = bt["nb"]
    outputs["bacteria_nb_curves"] = bt["nb_curves"]

    outputs["grab_analytes_by_class"], outputs["grab_analytes_rain_effect"] = grab_analytes(bact)

    DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)
        frame.to_csv(DATA_SAMPLE / f"{name}.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for name in [
        "class_daily_anomaly",
        "event_response_by_class",
        "event_response_spearman",
        "bacteria_by_class_full",
        "bacteria_by_dep_flag",
        "bacteria_nb_all",
        "grab_analytes_by_class",
        "grab_analytes_rain_effect",
    ]:
        print(f"\n== {name}")
        print(outputs[name].round(4).to_string(index=False))
    lag = outputs["class_daily_lag"]
    print("\n== class_daily_lag")
    print(
        lag.pivot_table(index=["variable", "rain_class"], columns="lag_day", values="anomaly_median")
        .round(3)
        .to_string()
    )
    ep = outputs["epoch_by_group"]
    print("\n== epoch extremes by group (lead of max/min median within 0-48 h)")
    for (col, grp), part in ep[ep["lead_h"] >= 0].groupby(["variable", "group"]):
        i_max, i_min = part["median"].idxmax(), part["median"].idxmin()
        print(
            f"{col:28s} {grp:14s} n0={int(part['n'].iloc[0]):4d} "
            f"max {part.loc[i_max, 'median']:+.3f} at {int(part.loc[i_max, 'lead_h'])} h, "
            f"min {part.loc[i_min, 'median']:+.3f} "
            f"[{part.loc[i_min, 'ci_lo']:+.3f}, {part.loc[i_min, 'ci_hi']:+.3f}] "
            f"at {int(part.loc[i_min, 'lead_h'])} h"
        )
    print("\n== DO epoch medians at selected leads")
    sel = ep[(ep["variable"] == "dissolved_oxygen_mg_l") & ep["lead_h"].isin([0, 3, 6, 9, 12, 18, 24, 36, 48])]
    print(sel.pivot_table(index="lead_h", columns="group", values="median").round(3).to_string())


if __name__ == "__main__":
    main()
