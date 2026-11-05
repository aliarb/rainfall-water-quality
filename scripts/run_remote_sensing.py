"""Evaluate Sentinel-2 water colour against rain class, USGS turbidity, and the
baseline rain-kernel model.

Requires scripts/fetch_remote_sensing.py (and USGS / rainfall fetches).
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
    DATA_PROCESSED,
    DATA_RAW,
    DATA_SAMPLE,
    OUTPUTS,
    RS_MATCH_MINUTES,
    classify_daily_rain_mm,
)
from rainfall_wq.remote_sensing import (  # noqa: E402
    fit_matchup,
    fit_rain_season_model,
    match_sonde,
)

ORDER = ["sunny", "light", "medium", "heavy", "rainstorm", "extreme"]
RS_TARGETS = ["ndti", "hue_angle_deg", "forel_ule", "sat_turbidity_fnu", "ndci"]


def load_rs() -> pd.DataFrame:
    path = DATA_RAW / "remote_sensing" / "sentinel2_pier84.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run scripts/fetch_remote_sensing.py first ({path})")
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["date"] = pd.to_datetime(df["datetime_utc"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
    # One scene per local day: most water pixels, then least cloud.
    df = df.sort_values(["water_frac", "n_water"], ascending=False)
    df = df.drop_duplicates(subset=["date"], keep="first")
    return df.sort_values("datetime_utc").set_index("datetime_utc")


def load_rain() -> pd.DataFrame:
    rain = pd.read_csv(DATA_RAW / "rainfall" / "central_park_daily.csv", parse_dates=["date"])
    rain = rain.set_index("date")
    rain["rain_class"] = rain["precip_mm"].map(classify_daily_rain_mm)
    rain["rain_48h_mm"] = rain["precip_mm"] + rain["precip_mm"].shift(1)
    return rain


def load_usgs_hourly() -> pd.Series | None:
    path = DATA_RAW / "usgs" / "pier84_iv.parquet"
    if not path.exists():
        return None
    iv = pd.read_parquet(path)
    if "turbidity_fnu" not in iv.columns:
        return None
    s = iv["turbidity_fnu"]
    if s.index.tz is None:
        s.index = s.index.tz_localize("America/New_York")
    return s


def class_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("rain_class"):
        row = {"rain_class": name, "n_scenes": int(len(g))}
        for col in RS_TARGETS + ["usgs_turbidity_fnu"]:
            if col in g.columns:
                row[f"{col}_median"] = float(g[col].median())
        rows.append(row)
    out = pd.DataFrame(rows).set_index("rain_class")
    return out.reindex([c for c in ORDER if c in out.index])


def make_plots(matched: pd.DataFrame, by_class: pd.DataFrame) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    pair = matched.dropna(subset=["usgs_turbidity_fnu", "ndti"])
    if len(pair) >= 8:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.scatter(pair["ndti"], pair["usgs_turbidity_fnu"], s=18, c="#4C78A8", alpha=0.7)
        ax.set_xlabel("Sentinel-2 NDTI (mid-channel)")
        ax.set_ylabel("USGS Pier 84 turbidity (FNU)")
        ax.set_title("Clear-sky matchup, ±1 h of overpass")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "s2_ndti_vs_usgs_turbidity.png", dpi=140)
        plt.close(fig)

    cats = [c for c in ORDER if c in by_class.index]
    if cats and "sat_turbidity_fnu_median" in by_class.columns:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.bar(
            range(len(cats)),
            [by_class.loc[c, "sat_turbidity_fnu_median"] for c in cats],
            color="#54A24B",
        )
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.capitalize() for c in cats])
        ax.set_ylabel("Median Nechad turbidity proxy (FNU-like)")
        ax.set_xlabel("Central Park rainfall class that day")
        ax.set_title("Channel colour vs rain class (cloud-free overpasses)")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "s2_turbidity_by_rain_class.png", dpi=140)
        plt.close(fig)

    if "hue_angle_deg" in matched.columns:
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.scatter(matched.index, matched["hue_angle_deg"], s=12, c="#4C78A8", label="hue angle")
        ax.set_ylabel("Hue angle (deg)")
        ax.set_title("Sentinel-2 water colour, Hudson west of Pier 84")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUTS / "s2_hue_angle_timeseries.png", dpi=140)
        plt.close(fig)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_SAMPLE.mkdir(parents=True, exist_ok=True)

    rs = load_rs()
    rain = load_rain()
    print(f"Sentinel-2 scenes {rs.index.min()} → {rs.index.max()}  n={len(rs)}")

    joined = rs.reset_index().merge(rain, left_on="date", right_index=True, how="left")
    joined = joined.set_index("datetime_utc")

    sonde = load_usgs_hourly()
    if sonde is not None:
        matched = match_sonde(joined, sonde, pd.Timedelta(minutes=RS_MATCH_MINUTES))
        n_match = int(matched["usgs_turbidity_fnu"].notna().sum())
        print(f"  USGS matchups within {RS_MATCH_MINUTES} min: {n_match}")
    else:
        matched = joined.copy()
        matched["usgs_turbidity_fnu"] = np.nan
        print("  no USGS instantaneous file; skipping sonde matchup")

    matched.to_csv(DATA_PROCESSED / "sentinel2_matched.csv")
    by_class = class_table(matched)
    by_class.to_csv(DATA_PROCESSED / "sentinel2_by_rain_class.csv")
    print(by_class.to_string())

    skills = []
    if matched["usgs_turbidity_fnu"].notna().sum() >= 10:
        for sat in ["ndti", "sat_turbidity_fnu", "red_green", "forel_ule"]:
            if sat not in matched.columns:
                continue
            skills.append(fit_matchup(matched["usgs_turbidity_fnu"], matched[sat], f"usgs ~ {sat}"))
        print(pd.DataFrame(skills)[["label", "R2", "NSE", "n"]].to_string(index=False))

    rain_daily = rain["precip_mm"]
    daily_y = matched.copy()
    daily_y.index = pd.to_datetime(matched["date"])
    daily_y = daily_y[~daily_y.index.duplicated(keep="first")].sort_index()

    model_rows = []
    for target in ["ndti", "sat_turbidity_fnu", "hue_angle_deg"]:
        if target not in daily_y.columns:
            continue
        fit = fit_rain_season_model(daily_y[target], rain_daily)
        if not fit or fit.get("n", 0) == 0:
            print(f"  {target}: not enough clear-sky days")
            continue
        model_rows.append(
            {
                "target": target,
                "model": "rain kernel + season",
                "tau_days": fit["tau_days"],
                "n": fit["n"],
                "R2_val": fit["skill_val"]["R2"],
                "NSE_val": fit["skill_val"]["NSE"],
            }
        )
        print(
            f"  {target} rain+season τ={fit['tau_days']}  "
            f"n={fit['n']}  val R2={fit['skill_val']['R2']:.3f}  NSE={fit['skill_val']['NSE']:.3f}"
        )

    usgs = daily_y.get("usgs_turbidity_fnu")
    if usgs is not None and usgs.notna().sum() >= 40 and "ndti" in daily_y.columns:
        extra = daily_y[["ndti"]].rename(columns={"ndti": "ndti"})
        rain_only = fit_rain_season_model(usgs, rain_daily)
        rain_rs = fit_rain_season_model(usgs, rain_daily, extra=extra)
        for name, fit in [("usgs turbidity rain+season", rain_only), ("usgs turbidity rain+season+NDTI", rain_rs)]:
            if not fit or fit.get("n", 0) == 0:
                continue
            model_rows.append(
                {
                    "target": "usgs_turbidity_fnu",
                    "model": name,
                    "tau_days": fit["tau_days"],
                    "n": fit["n"],
                    "R2_val": fit["skill_val"]["R2"],
                    "NSE_val": fit["skill_val"]["NSE"],
                }
            )
            print(
                f"  {name} τ={fit['tau_days']}  n={fit['n']}  "
                f"val R2={fit['skill_val']['R2']:.3f}  NSE={fit['skill_val']['NSE']:.3f}"
            )

    skill_df = pd.DataFrame(skills)
    if not skill_df.empty:
        skill_df.to_csv(DATA_PROCESSED / "sentinel2_matchup_skill.csv", index=False)
        skill_df.to_csv(DATA_SAMPLE / "sentinel2_matchup_skill.csv", index=False)
    if model_rows:
        models = pd.DataFrame(model_rows)
        models.to_csv(DATA_PROCESSED / "sentinel2_model_skill.csv", index=False)
        models.to_csv(DATA_SAMPLE / "sentinel2_model_skill.csv", index=False)

    by_class.to_csv(DATA_SAMPLE / "sentinel2_by_rain_class.csv")
    matched.tail(30).to_csv(DATA_SAMPLE / "sentinel2_matched_tail.csv")
    (DATA_PROCESSED / "sentinel2_summary.json").write_text(
        json.dumps(
            {
                "n_scenes": int(len(rs)),
                "n_match": int(matched["usgs_turbidity_fnu"].notna().sum()),
                "start": str(rs.index.min()),
                "end": str(rs.index.max()),
            },
            indent=2,
        )
    )
    make_plots(matched, by_class)
    print("done")


if __name__ == "__main__":
    main()
