"""Download NYC DEP Harbor Water Quality grab samples for Hudson River Park stations."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_wq.config import DATA_RAW, HARBOR_BASE, HARBOR_DATASET, HARBOR_STATIONS  # noqa: E402

KEEP = [
    "sampling_location",
    "sample_date",
    "sample_time",
    "weather_condition_dry_or_wet",
    "top_sample_temperature_c",
    "bottom_sample_temperature_c",
    "top_salinity_psu",
    "bottom_salinity_psu",
    "winkler_method_top_dissolved_oxygen_mg_l",
    "winkler_method_bottom_dissolved_oxygen_mg_l",
    "top_ph",
    "bottom_ph",
    "top_fecal_coliform_bacteria_cells_100ml",
    "bottom_fecal_coliform_bacteria_cells_100ml",
    "top_enterococci_bacteria_cells_100ml",
    "bottom_enterococci_bacteria_cells_100ml",
    "top_nitrate_nitrite_mg_l",
    "top_ammonium_mg_l",
    "total_phosphorus_mg_l",
    "top_total_kjeldhal_nitrogen_mg_l",
    "top_total_suspended_solid_mg_l",
    "top_active_chlorophyll_a_g_l",
    "lat",
    "long",
    "type",
]


def fetch_harbor(stations: list[str] | None = None, limit: int = 50000) -> pd.DataFrame:
    stations = stations or list(HARBOR_STATIONS)
    quoted = ",".join(f"'{s}'" for s in stations)
    where = quote(f"sampling_location in({quoted})")
    url = f"{HARBOR_BASE}/{HARBOR_DATASET}.csv?$where={where}&$limit={limit}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]
    present = [c for c in KEEP if c in df.columns]
    df = df[present].copy()
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    out = DATA_RAW / "harbor" / "hudson_n3b_n4_n5.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    df = fetch_harbor()
    print(df["sampling_location"].value_counts())
    print(df["sample_date"].min(), "→", df["sample_date"].max())
    print(f"rows={len(df)}")


if __name__ == "__main__":
    main()
