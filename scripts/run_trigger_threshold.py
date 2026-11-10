"""Rainfall trigger thresholds for dissolved oxygen and fecal coliform.

Rain x(t) is hourly Central Park precipitation. For a look-back of h hours,

    S_h(t) = sum of x over (t - h, t]        cumulative rain
    M_h(t) = max of x over (t - h, t]        peak hourly rain

A trigger T(t) = 1 when R(t) > R* and 0 otherwise, with R = S_h or M_h.
R* is chosen separately for each parameter as the threshold that best
separates the response when the trigger is on from when it is off.

Responses
    DO   change in tide- and season-removed DO at Pier 84 over the window,
         dDO_h(t) = eps(t) - mean eps over [t - h - 6, t - h - 1]
    Bac  log10 fecal coliform at the cast time of each Harbor Survey sample,
         with station offsets and an annual sine and cosine
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_wq.config import DATA_PROCESSED, DATA_SAMPLE  # noqa: E402

TZ = "America/New_York"
WINDOWS_DO = (6, 12, 24)
WINDOWS_BAC = (6, 12, 24, 48, 72)
GRID_SUM = (0.25, 0.5, 1.0, 1.5, 2.0, 2.54, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
GRID_MAX = (0.25, 0.5, 1.0, 1.5, 2.0, 2.54, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0)
MIN_ON_DO_DAYS = 20
MIN_ON_BAC = 15
N_BOOT = 400
RNG = np.random.default_rng(20261110)


def rain_features(rain: pd.Series, h: int) -> pd.DataFrame:
    roll = rain.rolling(h, min_periods=h)
    return pd.DataFrame({"S": roll.sum(), "M": roll.max()})


def welch_t(on: np.ndarray, off: np.ndarray) -> float:
    if len(on) < 2 or len(off) < 2:
        return np.nan
    v = on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off)
    return float((on.mean() - off.mean()) / np.sqrt(v)) if v > 0 else np.nan


def do_frame(resid: pd.DataFrame) -> dict[int, pd.DataFrame]:
    eps = resid["dissolved_oxygen_mg_l"]
    rain = resid["precip_mm"]
    out = {}
    for h in WINDOWS_DO:
        base = eps.shift(h + 1).rolling(6, min_periods=4).mean()
        f = rain_features(rain, h)
        df = pd.DataFrame({"y": eps - base, "S": f["S"], "M": f["M"]}, index=resid.index).dropna()
        df["day"] = df.index.tz_convert(TZ).floor("D")
        out[h] = df
    return out


def bac_frame(resid: pd.DataFrame) -> dict[int, pd.DataFrame]:
    b = pd.read_csv(DATA_PROCESSED / "harbor_bacteria_covariates.csv")
    b = b.dropna(subset=["fecal_coliform", "sample_time"]).copy()
    t = pd.to_datetime(b["sample_date"] + " " + b["sample_time"], errors="coerce")
    b = b[t.notna()].copy()
    b["cast"] = t[t.notna()].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward").dt.floor("h")
    b = b.dropna(subset=["cast"])
    rain = resid["precip_mm"]
    b = b[(b["cast"] >= rain.index.min()) & (b["cast"] <= rain.index.max())].copy()
    b["y"] = np.log10(b["fecal_coliform"].clip(lower=1))
    w = 2 * np.pi * b["cast"].dt.day_of_year / 365.25
    b["sin_doy"], b["cos_doy"] = np.sin(w), np.cos(w)
    b["day"] = b["cast"].dt.floor("D")
    out = {}
    for h in WINDOWS_BAC:
        f = rain_features(rain, h)
        df = b.join(f, on="cast")
        out[h] = df.dropna(subset=["S", "M"]).reset_index(drop=True)
    return out


def scan(df: pd.DataFrame, feature: str, grid: tuple[float, ...], min_on, kind: str) -> pd.DataFrame:
    rows = []
    y = df["y"].to_numpy()
    r = df[feature].to_numpy()
    for thr in grid:
        on = r > thr
        n_on = int(on.sum())
        enough = df.loc[on, "day"].nunique() >= MIN_ON_DO_DAYS if kind == "do" else n_on >= MIN_ON_BAC
        if not enough or (~on).sum() < MIN_ON_BAC:
            continue
        rows.append({"R_star": thr, "n_on": n_on, "n_off": int((~on).sum()), "t": welch_t(y[on], y[~on])})
    return pd.DataFrame(rows)


def pick(table: pd.DataFrame, sign: int) -> pd.Series | None:
    if table.empty:
        return None
    s = table.assign(score=sign * table["t"]).dropna(subset=["score"])
    return s.loc[s["score"].idxmax()] if not s.empty else None


def fit_do(df: pd.DataFrame, feature: str, thr: float) -> dict:
    d = df.assign(T=(df[feature] > thr).astype(int))
    fit = smf.ols("y ~ T", d).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    return {"effect": float(fit.params["T"]), "lo": float(fit.conf_int().loc["T", 0]),
            "hi": float(fit.conf_int().loc["T", 1]), "p": float(fit.pvalues["T"]),
            "mean_on": float(d.loc[d["T"] == 1, "y"].mean()), "mean_off": float(d.loc[d["T"] == 0, "y"].mean())}


def fit_bac(df: pd.DataFrame, feature: str, thr: float) -> dict:
    d = df.assign(T=(df[feature] > thr).astype(int))
    fit = smf.ols("y ~ T + sin_doy + cos_doy + C(station)", d).fit(cov_type="HC3")
    b, (lo, hi) = float(fit.params["T"]), fit.conf_int().loc["T"].to_numpy()
    return {"effect": float(10**b), "lo": float(10**lo), "hi": float(10**hi), "p": float(fit.pvalues["T"]),
            "mean_on": float(10 ** d.loc[d["T"] == 1, "y"].mean()), "mean_off": float(10 ** d.loc[d["T"] == 0, "y"].mean())}


def boot_threshold(df: pd.DataFrame, feature: str, grid, kind: str, sign: int) -> np.ndarray:
    days = df["day"].unique()
    groups = {d: idx for d, idx in df.groupby("day").indices.items()}
    picks = []
    for _ in range(N_BOOT):
        sample = RNG.choice(days, size=len(days), replace=True)
        idx = np.concatenate([groups[d] for d in sample])
        bs = df.iloc[idx].copy()
        bs["day"] = np.repeat(np.arange(len(sample)), [len(groups[d]) for d in sample])
        best = pick(scan(bs, feature, grid, None, kind), sign)
        if best is not None:
            picks.append(best["R_star"])
    return np.asarray(picks, dtype=float)


def run(parameter: str, frames: dict[int, pd.DataFrame], kind: str, sign: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scans, best = [], []
    for h, df in frames.items():
        for feature, grid in (("S", GRID_SUM), ("M", GRID_MAX)):
            table = scan(df, feature, grid, None, kind)
            if table.empty:
                continue
            scans.append(table.assign(parameter=parameter, h=h, feature=feature))
            b = pick(table, sign)
            if b is not None:
                best.append({"parameter": parameter, "h": h, "feature": feature, "R_star": b["R_star"],
                             "t": b["t"], "n_on": b["n_on"], "n_off": b["n_off"]})
    best = pd.DataFrame(best)
    best["score"] = sign * best["t"]
    top = best.loc[best["score"].idxmax()]
    df = frames[int(top["h"])]
    fit = fit_do(df, top["feature"], top["R_star"]) if kind == "do" else fit_bac(df, top["feature"], top["R_star"])
    boots = boot_threshold(df, top["feature"], GRID_SUM if top["feature"] == "S" else GRID_MAX, kind, sign)
    selected = pd.DataFrame([{
        **top.drop("score").to_dict(), **fit,
        "R_star_q05": float(np.quantile(boots, 0.05)), "R_star_median": float(np.median(boots)),
        "R_star_q95": float(np.quantile(boots, 0.95)), "n_boot": int(len(boots)),
    }])
    return pd.concat(scans, ignore_index=True), selected


def main() -> None:
    resid = pd.read_parquet(DATA_PROCESSED / "hourly_harmonic_residuals.parquet").sort_index()
    resid = resid.asfreq("h")
    resid["precip_mm"] = resid["precip_mm"].fillna(0.0)

    do_scan, do_sel = run("DO", do_frame(resid), "do", sign=-1)
    bac_scan, bac_sel = run("Bac", bac_frame(resid), "bac", sign=+1)

    scans = pd.concat([do_scan, bac_scan], ignore_index=True)
    selected = pd.concat([do_sel, bac_sel], ignore_index=True)
    for folder in (DATA_PROCESSED, DATA_SAMPLE):
        scans.to_csv(folder / "trigger_scan.csv", index=False)
        selected.to_csv(folder / "trigger_selected.csv", index=False)
    pd.set_option("display.width", 200)
    print(selected.T.to_string())


if __name__ == "__main__":
    main()
