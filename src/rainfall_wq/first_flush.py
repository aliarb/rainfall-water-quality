"""Event first-flush curves in normalized M(V) space.

Cumulative excess turbidity mass fraction M versus cumulative rain volume
fraction V. A 1:1 line is uniform washoff; M > V at small V is a first flush.
Excess is residual turbidity above the 6-hour pre-event median, floored at 0.
Flow is unknown, so turbidity × Δt is a load *proxy*, not a true mass flux.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

V_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


def event_mv_curve(
    precip_mm: pd.Series,
    residual: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pre_h: int = 6,
) -> pd.Series | None:
    rain = precip_mm.loc[start:end].astype(float).fillna(0.0)
    turb = residual.reindex(rain.index).astype(float)
    if rain.sum() <= 0 or turb.notna().sum() < 3:
        return None
    pre_start = start - pd.Timedelta(hours=pre_h)
    baseline = residual.loc[pre_start:start].median()
    if not np.isfinite(baseline):
        baseline = 0.0
    excess = (turb - baseline).clip(lower=0.0).fillna(0.0)
    if float(excess.sum()) <= 0:
        return None
    v = rain.cumsum() / rain.sum()
    m = excess.cumsum() / excess.sum()
    # Force endpoints so interpolation is well defined.
    v = pd.concat([pd.Series([0.0], index=[v.index[0] - pd.Timedelta(seconds=1)]), v])
    m = pd.concat([pd.Series([0.0], index=[m.index[0] - pd.Timedelta(seconds=1)]), m])
    order = np.argsort(v.to_numpy())
    v_np = np.clip(v.to_numpy()[order], 0.0, 1.0)
    m_np = np.clip(m.to_numpy()[order], 0.0, 1.0)
    # Unique V for interpolation
    _, uniq = np.unique(v_np, return_index=True)
    v_u, m_u = v_np[np.sort(uniq)], m_np[np.sort(uniq)]
    if len(v_u) < 2:
        return None
    interp = np.interp(V_GRID, v_u, m_u)
    return pd.Series(interp, index=V_GRID, name="M")


def first_flush_summary(
    precip_mm: pd.Series,
    residual: pd.Series,
    events: pd.DataFrame,
    min_depth_mm: float = 2.54,
) -> pd.DataFrame:
    """Median M(V) curve across events at/above the CSO-proxy depth."""
    curves = []
    if events.empty:
        return pd.DataFrame()
    subset = events[events["depth_mm"] >= min_depth_mm]
    for row in subset.itertuples(index=False):
        curve = event_mv_curve(precip_mm, residual, row.start, row.end)
        if curve is None:
            continue
        curves.append(curve.to_frame().T.assign(event_id=row.event_id, depth_mm=row.depth_mm))
    if not curves:
        return pd.DataFrame()
    mat = pd.concat(curves, ignore_index=True)
    value_cols = [c for c in mat.columns if c not in {"event_id", "depth_mm"}]
    out = pd.DataFrame(
        {
            "V": [float(c) for c in value_cols],
            "M_median": mat[value_cols].median(axis=0).to_numpy(),
            "M_q25": mat[value_cols].quantile(0.25, axis=0).to_numpy(),
            "M_q75": mat[value_cols].quantile(0.75, axis=0).to_numpy(),
            "n_events": int(len(mat)),
        }
    )
    # First-flush index: M at V=0.3 (30% of runoff). Paper cited ~50% of load.
    at_30 = out.loc[np.isclose(out["V"], 0.30), "M_median"]
    out.attrs["M_at_V30"] = float(at_30.iloc[0]) if len(at_30) else np.nan
    return out
