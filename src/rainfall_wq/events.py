"""Rainfall event delineation and superposed-epoch windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CSO_RAIN_THRESHOLD_MM, classify_daily_rain_mm

IETD_HOURS = 6
WET_HOUR_MM = 0.1
MIN_EVENT_MM = 1.0
EPOCH_PRE_H = 6
EPOCH_POST_H = 48


def delineate_events(
    precip_mm: pd.Series,
    ietd_hours: int = IETD_HOURS,
    wet_mm: float = WET_HOUR_MM,
    min_depth_mm: float = MIN_EVENT_MM,
) -> pd.DataFrame:
    """Cluster wet hours into storms separated by at least `ietd_hours` dry.

    A wet hour has precipitation ≥ `wet_mm`. Events whose total depth is
    below `min_depth_mm` are dropped. CSO-proxy is total depth ≥ 2.54 mm.
    """
    p = precip_mm.astype(float).fillna(0.0).sort_index()
    wet_times = p.index[p >= wet_mm]
    if len(wet_times) == 0:
        return pd.DataFrame()

    ietd = pd.Timedelta(hours=ietd_hours)
    starts = [wet_times[0]]
    ends = []
    prev = wet_times[0]
    for t in wet_times[1:]:
        if t - prev <= ietd:
            prev = t
        else:
            ends.append(prev)
            starts.append(t)
            prev = t
    ends.append(prev)

    rows = []
    prev_end = None
    for i, (start, end) in enumerate(zip(starts, ends), start=1):
        window = p.loc[start:end]
        depth = float(window.sum())
        if depth < min_depth_mm:
            continue
        duration_h = (end - start) / pd.Timedelta(hours=1) + 1.0
        peak_idx = window.idxmax()
        add_h = np.nan if prev_end is None else (start - prev_end) / pd.Timedelta(hours=1)
        rows.append(
            {
                "event_id": i,
                "start": start,
                "end": end,
                "peak_time": peak_idx,
                "depth_mm": depth,
                "duration_h": float(duration_h),
                "peak_mm_h": float(window.max()),
                "antecedent_dry_h": add_h,
                "cso_proxy": depth >= CSO_RAIN_THRESHOLD_MM,
                "rain_class": classify_daily_rain_mm(depth),
            }
        )
        prev_end = end
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.reset_index(drop=True)
        events["event_id"] = np.arange(1, len(events) + 1)
    return events


def superposed_epoch(
    residual: pd.Series,
    events: pd.DataFrame,
    pre_h: int = EPOCH_PRE_H,
    post_h: int = EPOCH_POST_H,
    origin: str = "start",
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Mean residual WQ aligned on event start or peak rain hour.

    Returns one row per lead hour, with mean / quartiles across events that
    have at least `min_coverage` of the window observed.
    """
    if events.empty:
        return pd.DataFrame()
    residual = residual.sort_index()
    leads = np.arange(-pre_h, post_h + 1)
    origin_col = "peak_time" if origin == "peak" else "start"

    stacked = []
    for row in events.itertuples(index=False):
        t0 = getattr(row, origin_col)
        times = pd.DatetimeIndex([t0 + pd.Timedelta(hours=int(h)) for h in leads])
        vals = residual.reindex(times).to_numpy(dtype=float)
        if np.isfinite(vals).mean() < min_coverage:
            continue
        part = pd.DataFrame(
            {
                "event_id": row.event_id,
                "lead_h": leads,
                "residual": vals,
                "cso_proxy": row.cso_proxy,
                "rain_class": row.rain_class,
                "depth_mm": row.depth_mm,
            }
        )
        stacked.append(part)
    if not stacked:
        return pd.DataFrame()
    long = pd.concat(stacked, ignore_index=True)

    def _summarize(part: pd.DataFrame, group: str) -> pd.DataFrame:
        g = part.groupby("lead_h")["residual"]
        out = pd.DataFrame(
            {
                "n": g.count(),
                "mean": g.mean(),
                "median": g.median(),
                "q25": g.quantile(0.25),
                "q75": g.quantile(0.75),
            }
        ).reset_index()
        out["group"] = group
        return out

    return pd.concat(
        [
            _summarize(long, "all"),
            _summarize(long[long["cso_proxy"]], "cso_proxy"),
            _summarize(long[~long["cso_proxy"]], "below_cso_threshold"),
        ],
        ignore_index=True,
    )
