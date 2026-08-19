"""Download USGS NWIS water quality for Hudson River Park (Pier 84)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_wq.config import (  # noqa: E402
    DATA_RAW,
    USGS_PARAMS,
    USGS_SITE_ID,
    USGS_START,
)

IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
SITE_URL = "https://waterservices.usgs.gov/nwis/site/"


def _get(url: str, params: dict) -> requests.Response:
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    return r


def fetch_site_metadata() -> pd.DataFrame:
    r = _get(SITE_URL, {"sites": USGS_SITE_ID, "format": "rdb"})
    lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
    header, _widths, *rows = lines
    cols = header.split("\t")
    data = [row.split("\t") for row in rows]
    df = pd.DataFrame(data, columns=cols)
    out = DATA_RAW / "usgs" / "site_metadata.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def fetch_daily(start: str = USGS_START, end: str | None = None) -> pd.DataFrame:
    """Mean daily values for all Pier 84 sonde parameters."""
    params = {
        "format": "json",
        "sites": USGS_SITE_ID,
        "startDT": start,
        "parameterCd": ",".join(USGS_PARAMS),
        "statCd": "00003",
        "siteStatus": "all",
    }
    if end:
        params["endDT"] = end
    payload = _get(DV_URL, params).json()
    frames = []
    for series in payload.get("value", {}).get("timeSeries", []):
        pcode = series["variable"]["variableCode"][0]["value"]
        name = USGS_PARAMS.get(pcode, pcode)
        pts = series["values"][0]["value"]
        if not pts:
            continue
        part = pd.DataFrame(pts)
        part["datetime"] = pd.to_datetime(part["dateTime"], utc=True)
        part[name] = pd.to_numeric(part["value"], errors="coerce")
        frames.append(part[["datetime", name]].set_index("datetime"))
    if not frames:
        raise RuntimeError("No USGS daily values returned.")
    df = pd.concat(frames, axis=1, sort=True)
    df.index = df.index.tz_convert("America/New_York")
    df.index.name = "datetime"
    out = DATA_RAW / "usgs" / "pier84_daily.csv"
    df.to_csv(out)
    meta = DATA_RAW / "usgs" / "pier84_daily_meta.json"
    meta.write_text(json.dumps({"site": USGS_SITE_ID, "params": USGS_PARAMS, "start": start}, indent=2))
    return df


def fetch_instantaneous_year(year: int) -> pd.DataFrame:
    """15-minute instantaneous values for one calendar year."""
    params = {
        "format": "json",
        "sites": USGS_SITE_ID,
        "startDT": f"{year}-01-01",
        "endDT": f"{year}-12-31",
        "parameterCd": ",".join(USGS_PARAMS),
        "siteStatus": "all",
    }
    payload = _get(IV_URL, params).json()
    frames = []
    for series in payload.get("value", {}).get("timeSeries", []):
        pcode = series["variable"]["variableCode"][0]["value"]
        name = USGS_PARAMS.get(pcode, pcode)
        pts = series["values"][0]["value"]
        if not pts:
            continue
        part = pd.DataFrame(pts)
        part["datetime"] = pd.to_datetime(part["dateTime"], utc=True)
        part[name] = pd.to_numeric(part["value"], errors="coerce")
        frames.append(part[["datetime", name]].set_index("datetime"))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1, sort=True)
    df.index = df.index.tz_convert("America/New_York")
    df.index.name = "datetime"
    out = DATA_RAW / "usgs" / f"pier84_iv_{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def fetch_instantaneous(start_year: int = 2019, end_year: int = 2026) -> pd.DataFrame:
    parts = []
    for year in range(start_year, end_year + 1):
        print(f"USGS IV {year} ...")
        part = fetch_instantaneous_year(year)
        if not part.empty:
            parts.append(part)
            print(f"  {len(part):,} rows")
    if not parts:
        raise RuntimeError("No USGS instantaneous values returned.")
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = DATA_RAW / "usgs" / "pier84_iv.parquet"
    df.to_parquet(out)
    return df


def main() -> None:
    DATA_RAW.joinpath("usgs").mkdir(parents=True, exist_ok=True)
    print("Site metadata")
    print(fetch_site_metadata().head())
    print("Daily means")
    daily = fetch_daily()
    print(daily.tail())
    print(f"daily rows={len(daily)}")
    try:
        iv = fetch_instantaneous()
        print(f"IV rows={len(iv)}  {iv.index.min()} → {iv.index.max()}")
    except Exception as exc:
        print(f"IV download failed ({exc}). Daily values are saved.")


if __name__ == "__main__":
    main()
