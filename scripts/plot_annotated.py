"""Rebuild annotated story figures from saved tables (no model refit)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_wq.config import DATA_PROCESSED  # noqa: E402
from rainfall_wq.figures import annotate_all  # noqa: E402


def main() -> None:
    daily = pd.read_csv(DATA_PROCESSED / "pier84_daily_aligned.csv", parse_dates=["date"], index_col="date")
    paths = annotate_all(daily)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
