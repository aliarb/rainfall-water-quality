"""Skill scores from Lin et al. (2024), Eqs. (1)–(5)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align(observed, modeled) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(observed, dtype=float)
    m = np.asarray(modeled, dtype=float)
    mask = np.isfinite(o) & np.isfinite(m)
    if mask.sum() < 2:
        raise ValueError("Need at least two paired finite observations.")
    return o[mask], m[mask]


def relative_errors(observed, modeled) -> dict[str, float]:
    """Eqs. (1)–(3): max / min / mean relative percent error."""
    o, m = _align(observed, modeled)
    denom = (m + o) / 2.0
    pair = np.abs(m - o) / np.where(np.abs(denom) < 1e-12, np.nan, denom) * 100.0
    pair = pair[np.isfinite(pair)]
    mean_rel = np.mean(np.abs(m - o) / (len(o) * np.mean(o))) * 100.0 if np.mean(o) else np.nan
    # Eq. (3) is mean |M-O| / (n * mean(O)) * 100 as written; also report MAPE.
    mape = np.mean(np.abs(m - o) / np.where(np.abs(o) < 1e-12, np.nan, np.abs(o))) * 100.0
    return {
        "Emax_pct": float(np.nanmax(pair)) if len(pair) else np.nan,
        "Emin_pct": float(np.nanmin(pair)) if len(pair) else np.nan,
        "Eaverage_pct": float(mean_rel),
        "MAPE_pct": float(np.nanmean(mape)),
        "n": int(len(o)),
    }


def r_squared(observed, modeled) -> float:
    """Pearson R², Eq. (4)."""
    o, m = _align(observed, modeled)
    o_c = o - o.mean()
    m_c = m - m.mean()
    denom = np.sqrt((o_c**2).sum() * (m_c**2).sum())
    if denom == 0:
        return np.nan
    r = float((o_c * m_c).sum() / denom)
    return r**2


def nash_sutcliffe(observed, modeled) -> float:
    """Nash–Sutcliffe efficiency, Eq. (5)."""
    o, m = _align(observed, modeled)
    num = ((o - m) ** 2).sum()
    den = ((o - o.mean()) ** 2).sum()
    if den == 0:
        return np.nan
    return float(1.0 - num / den)


def skill_table(observed, modeled, label: str = "") -> pd.Series:
    rel = relative_errors(observed, modeled)
    return pd.Series(
        {
            "label": label,
            "R2": r_squared(observed, modeled),
            "NSE": nash_sutcliffe(observed, modeled),
            **rel,
        }
    )


def lateral_diffusion_ez(h_m: float, ux_m3s: float, alpha: float = 0.6) -> float:
    """Lateral diffusion coefficient Ez = α h ux, Eq. (6).

    α is dimensionless (0.4–0.8 for natural rivers; 0.6 in Lin et al.).
    h is mean section depth (m). ux is a resistive flow-rate term (m³/s)
    as used in the paper; treat as a calibrated scale, not a textbook
    Elder/Fischer mixing coefficient, until Hudson bathymetry is available.
    """
    return alpha * h_m * ux_m3s
