"""Download Manhattan rainfall: NOAA ACIS daily (Central Park) + IEM hourly ASOS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_wq.config import ACIS_STATIONS, DATA_RAW, IEM_HOURLY_STATIONS  # noqa: E402

ACIS_URL = "https://data.rcc-acis.org/StnData"
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _parse_acis_pcpn(value: str) -> float:
    if value in (None, "", "M", "S"):
        return float("nan")
    if str(value).startswith("T"):
        return 0.0
    # Accumulated values sometimes look like "1.20A"
    text = str(value).rstrip("A").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def fetch_acis_daily(
    sid: str = ACIS_STATIONS["central_park"]["sid"],
    start: str = "2019-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Daily precipitation (inches) and air temperature from NOAA ACIS.

    ACIS needs no token. Trace precipitation is stored as 0; missing as NaN.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    body = {
        "sid": sid,
        "sdate": start,
        "edate": end,
        "elems": [
            {"name": "pcpn", "interval": "dly", "duration": "dly"},
            {"name": "maxt", "interval": "dly", "duration": "dly"},
            {"name": "mint", "interval": "dly", "duration": "dly"},
        ],
        "output": "json",
    }
    r = requests.post(ACIS_URL, json=body, timeout=120)
    r.raise_for_status()
    payload = r.json()
    rows = []
    for date_str, pcpn, maxt, mint in payload.get("data", []):
        rows.append(
            {
                "date": date_str,
                "precip_in": _parse_acis_pcpn(pcpn),
                "tmax_f": _parse_acis_pcpn(maxt),
                "tmin_f": _parse_acis_pcpn(mint),
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["precip_mm"] = df["precip_in"] * 25.4
    df = df.set_index("date")
    station = next(s for s in ACIS_STATIONS.values() if s["sid"] == sid)
    out = DATA_RAW / "rainfall" / "central_park_daily.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    meta = {
        "source": "NOAA ACIS StnData",
        "url": ACIS_URL,
        "sid": sid,
        "name": station["name"],
        "lat": station["lat"],
        "lon": station["lon"],
        "units": {"precip_in": "inches", "precip_mm": "mm", "tmax_f": "F", "tmin_f": "F"},
    }
    (DATA_RAW / "rainfall" / "central_park_daily_meta.json").write_text(json.dumps(meta, indent=2))
    return df


def fetch_iem_hourly(
    station: str,
    start: str = "2019-05-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Hourly precipitation (inches) from Iowa Environmental Mesonet ASOS archive."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    params = {
        "station": station,
        "data": "p01i",
        "year1": start_ts.year,
        "month1": start_ts.month,
        "day1": start_ts.day,
        "year2": end_ts.year,
        "month2": end_ts.month,
        "day2": end_ts.day,
        "tz": "America/New_York",
        "format": "onlycomma",
        "latlon": "yes",
        "missing": "null",
        "trace": "0.0001",
        "direct": "no",
        "report_type": "3",
    }
    r = requests.get(IEM_ASOS, params=params, timeout=180)
    r.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(r.text), parse_dates=["valid"])
    if df.empty:
        return df
    df = df.rename(columns={"valid": "datetime", "p01i": "precip_in"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")
    df["precip_in"] = pd.to_numeric(df["precip_in"], errors="coerce")
    df["precip_mm"] = df["precip_in"] * 25.4
    df = df.set_index("datetime").sort_index()
    out = DATA_RAW / "rainfall" / f"hourly_{station}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def main() -> None:
    DATA_RAW.joinpath("rainfall").mkdir(parents=True, exist_ok=True)
    daily = fetch_acis_daily()
    print("ACIS Central Park daily")
    print(daily.tail())
    print(f"days={len(daily)}  precip_mm sum={daily['precip_mm'].sum():.1f}")
    for sid in IEM_HOURLY_STATIONS:
        print(f"IEM hourly {sid} ...")
        hourly = fetch_iem_hourly(sid)
        print(f"  rows={len(hourly)}")


if __name__ == "__main__":
    main()
