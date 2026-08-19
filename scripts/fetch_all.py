"""Fetch all public datasets used in the Hudson River Park baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_harbor  # noqa: E402
import fetch_rainfall  # noqa: E402
import fetch_usgs  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-iv", action="store_true", help="Skip 15-minute USGS download")
    args = p.parse_args()

    print("=== USGS site + daily ===")
    fetch_usgs.fetch_site_metadata()
    daily = fetch_usgs.fetch_daily()
    print(f"daily {len(daily)} rows  {daily.index.min()} → {daily.index.max()}")

    if not args.skip_iv:
        print("=== USGS 15-minute ===")
        try:
            iv = fetch_usgs.fetch_instantaneous()
            print(f"IV {len(iv)} rows")
        except Exception as exc:
            print(f"IV download failed ({exc}); daily record is enough for the baseline.")

    print("=== Rainfall ===")
    fetch_rainfall.main()

    print("=== Harbor Survey N3B/N4/N5 ===")
    fetch_harbor.main()


if __name__ == "__main__":
    main()
