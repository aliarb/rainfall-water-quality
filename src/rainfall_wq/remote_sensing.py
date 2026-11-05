"""Sentinel-2 water-colour indices for the Hudson off Pier 84.

Surface reflectance from Sen2Cor L2A (via Planetary Computer) is converted
to hue angle / Forel-Ule colour, NDTI, NDCI, and a Nechad red-band turbidity
proxy. These are clear-sky snapshots, not an hourly CSO sensor: wet days are
preferentially cloudy, so rain-class samples are biased toward the day after
a storm rather than the overflow itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import skill_table


def _ols_fit(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with intercept. Returns (params including intercept, fitted)."""
    A = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(dtype=float), rcond=None)
    fitted = A @ coef
    return coef, fitted

# Pitarch et al.–style Forel-Ule centroids (hue angle, degrees), FU 1 (blue)
# through FU 21 (brown). Hue is scale-invariant in RGB reflectance.
_FU_HUE_CENTROIDS = np.array(
    [
        229.9,
        218.0,
        202.4,
        183.0,
        162.5,
        145.1,
        129.2,
        117.0,
        107.0,
        96.7,
        89.0,
        81.3,
        74.3,
        67.5,
        62.0,
        56.4,
        51.3,
        46.7,
        42.6,
        39.2,
        36.5,
    ]
)

# Nechad (2010) 665 nm: T = A * rho / (1 - rho / C), rho = water-leaving
# reflectance. L2A BOA reflectance is a rough stand-in at this site.
_NECHAD_A = 234.21
_NECHAD_C = 0.1686

_CLOUD_SCL = {0, 1, 3, 8, 9, 10}


def dn_to_reflectance(dn: np.ndarray, processing_baseline: str | float | None) -> np.ndarray:
    """Sen2Cor L2A digital numbers to BOA reflectance.

    Processing baseline >= 04.00 stores an offset of -1000 (scale 1e-4).
    """
    x = np.asarray(dn, dtype=float)
    try:
        pb = float(str(processing_baseline).replace(",", "."))
    except (TypeError, ValueError):
        pb = 4.0
    rho = x / 10000.0
    if pb >= 4.0:
        rho = rho - 0.1
    return rho


def hue_angle_deg(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """CIE hue angle from linear RGB reflectance (van der Woerd / Wernand)."""
    r = np.asarray(red, dtype=float)
    g = np.asarray(green, dtype=float)
    b = np.asarray(blue, dtype=float)
    x = 2.768892 * r + 1.751748 * g + 1.130160 * b
    y = 1.000000 * r + 4.590700 * g + 0.060100 * b
    z = 0.056508 * g + 5.594292 * b
    denom = x + y + z
    ok = np.isfinite(denom) & (denom > 0)
    cx = np.full_like(denom, np.nan)
    cy = np.full_like(denom, np.nan)
    cx[ok] = x[ok] / denom[ok]
    cy[ok] = y[ok] / denom[ok]
    alpha = np.degrees(np.arctan2(cy - 1.0 / 3.0, cx - 1.0 / 3.0))
    return np.mod(alpha, 360.0)


def forel_ule_index(hue_deg: np.ndarray) -> np.ndarray:
    """Nearest Forel-Ule class (1 = blue, 21 = brown)."""
    h = np.asarray(hue_deg, dtype=float)
    out = np.full(h.shape, np.nan)
    ok = np.isfinite(h)
    if not ok.any():
        return out
    d = np.abs(h[ok, None] - _FU_HUE_CENTROIDS[None, :])
    out[ok] = d.argmin(axis=1) + 1
    return out


def nechad_turbidity(red_rho: np.ndarray) -> np.ndarray:
    rho = np.clip(np.asarray(red_rho, dtype=float), 1e-5, 0.95 * _NECHAD_C)
    return _NECHAD_A * rho / (1.0 - rho / _NECHAD_C)


def water_mask(scl: np.ndarray, green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """SCL water, plus unclassified/dark pixels with MNDWI > 0.15."""
    scl = np.asarray(scl)
    g = np.asarray(green, dtype=float)
    s = np.asarray(swir, dtype=float)
    mndwi = (g - s) / np.where((g + s) == 0, np.nan, g + s)
    cloud = np.isin(scl, list(_CLOUD_SCL))
    water = scl == 6
    extra = ((scl == 2) | (scl == 5) | (scl == 7)) & (mndwi > 0.15)
    return (~cloud) & (water | extra)


def scene_indices(
    bands: dict[str, np.ndarray],
    scl: np.ndarray,
    processing_baseline: str | float | None,
    min_water_pixels: int,
) -> dict[str, float] | None:
    """Median indices over water pixels in one cropped scene."""
    mask = water_mask(scl, bands["B03"], bands["B11"])
    n_water = int(mask.sum())
    n_pix = int(mask.size)
    if n_water < min_water_pixels:
        return None
    rho = {k: dn_to_reflectance(v[mask], processing_baseline) for k, v in bands.items()}
    b2, b3, b4 = rho["B02"], rho["B03"], rho["B04"]
    b5, b8, b11 = rho["B05"], rho["B08"], rho["B11"]
    hue = hue_angle_deg(b4, b3, b2)
    ndti = (b4 - b3) / np.where((b4 + b3) == 0, np.nan, b4 + b3)
    ndci = (b5 - b4) / np.where((b5 + b4) == 0, np.nan, b5 + b4)
    mndwi = (b3 - b11) / np.where((b3 + b11) == 0, np.nan, b3 + b11)
    ndwi = (b3 - b8) / np.where((b3 + b8) == 0, np.nan, b3 + b8)
    fu = forel_ule_index(hue)
    return {
        "n_pixels": n_pix,
        "n_water": n_water,
        "water_frac": n_water / n_pix if n_pix else np.nan,
        "rho_blue": float(np.nanmedian(b2)),
        "rho_green": float(np.nanmedian(b3)),
        "rho_red": float(np.nanmedian(b4)),
        "rho_re": float(np.nanmedian(b5)),
        "rho_nir": float(np.nanmedian(b8)),
        "rho_swir": float(np.nanmedian(b11)),
        "ndti": float(np.nanmedian(ndti)),
        "ndci": float(np.nanmedian(ndci)),
        "mndwi": float(np.nanmedian(mndwi)),
        "ndwi": float(np.nanmedian(ndwi)),
        "red_green": float(np.nanmedian(b4 / np.where(b3 == 0, np.nan, b3))),
        "hue_angle_deg": float(np.nanmedian(hue)),
        "forel_ule": float(np.nanmedian(fu)),
        "sat_turbidity_fnu": float(np.nanmedian(nechad_turbidity(b4))),
    }


def parse_crop_array(arr: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Map rio-tiler crop.npy (bands, y, x) onto named Sentinel-2 assets.

    The data API returns requested assets in order and may append an alpha
    mask as a final band (0/255).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected (bands, y, x), got {arr.shape}")
    names = ["B02", "B03", "B04", "B05", "B08", "B11", "SCL"]
    if arr.shape[0] < 7:
        raise ValueError(f"Need 7 band planes, got {arr.shape[0]}")
    bands = {name: arr[i].astype(float) for i, name in enumerate(names[:6])}
    scl = arr[6]
    if arr.shape[0] >= 8:
        alpha = arr[-1]
        valid = alpha > 0
        for k in bands:
            bands[k] = np.where(valid, bands[k], np.nan)
        scl = np.where(valid, scl, 0)
    return bands, scl


def _naive_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def match_sonde(
    rs: pd.DataFrame,
    sonde: pd.Series,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """Nearest USGS sample within ``tolerance`` of each overpass."""
    y = sonde.dropna().sort_index()
    y.index = _naive_utc(y.index)
    rs_times = _naive_utc(rs.index)
    rows = []
    for t, (_, row) in zip(rs_times, rs.iterrows(), strict=True):
        t = pd.Timestamp(t)
        if len(y) == 0:
            break
        pos = int(y.index.searchsorted(t))
        cand = []
        if pos < len(y):
            cand.append(pos)
        if pos > 0:
            cand.append(pos - 1)
        best = None
        for j in cand:
            dt = abs(y.index[j] - t)
            if dt <= tolerance and (best is None or dt < best[0]):
                best = (dt, float(y.iloc[j]))
        rec = row.to_dict()
        rec["datetime"] = pd.Timestamp(t, tz="UTC")
        rec["usgs_turbidity_fnu"] = best[1] if best else np.nan
        rec["match_minutes"] = best[0].total_seconds() / 60.0 if best else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.set_index("datetime")


def fit_matchup(obs: pd.Series, sat: pd.Series, label: str) -> pd.Series:
    pair = pd.concat([obs.rename("obs"), sat.rename("sat")], axis=1).dropna()
    if len(pair) < 10:
        out = skill_table(pair["obs"], pair["sat"], label) if len(pair) >= 2 else pd.Series(
            {"label": label, "R2": np.nan, "NSE": np.nan, "n": float(len(pair))}
        )
        return out
    coef, pred = _ols_fit(pair[["sat"]], pair["obs"])
    skill = skill_table(pair["obs"], pred, label)
    skill["intercept"] = float(coef[0])
    skill["slope"] = float(coef[1])
    return skill


def exponential_kernel(precip: pd.Series, tau_days: float, max_lag: int = 7) -> pd.Series:
    weights = np.array([np.exp(-k / tau_days) for k in range(max_lag + 1)])
    weights = weights / weights.sum()
    x = precip.fillna(0.0).to_numpy(dtype=float)
    conv = np.convolve(x, weights, mode="full")[: len(x)]
    return pd.Series(conv, index=precip.index, name=f"rain_kernel_tau{tau_days:g}")


def fit_rain_season_model(
    y: pd.Series,
    precip: pd.Series,
    extra: pd.DataFrame | None = None,
) -> dict:
    """Y = β0 + β_r K_τ(P) + season [+ extra covariates], last-20% hold-out."""
    doy = y.index.dayofyear.astype(float)
    season = pd.DataFrame(
        {
            "sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        },
        index=y.index,
    )
    best = None
    for tau in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        kernel = exponential_kernel(precip, tau).reindex(y.index)
        parts = [kernel.rename("rain_kernel"), season]
        if extra is not None:
            parts.append(extra.reindex(y.index))
        X = pd.concat(parts, axis=1)
        pair = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(pair) < 40:
            continue
        split = int(len(pair) * 0.8)
        xv = pair.drop(columns=["y"])
        coef, _ = _ols_fit(xv.iloc[:split], pair["y"].iloc[:split])
        A = np.column_stack([np.ones(len(xv)), xv.to_numpy(dtype=float)])
        pred = pd.Series(A @ coef, index=pair.index)
        skill_val = skill_table(pair["y"].iloc[split:], pred.iloc[split:], f"val τ={tau}")
        nse = skill_val["NSE"]
        if best is None or (np.isfinite(nse) and nse > best["nse_val"]):
            best = {
                "tau_days": tau,
                "nse_val": float(nse) if np.isfinite(nse) else -np.inf,
                "skill_cal": skill_table(pair["y"].iloc[:split], pred.iloc[:split], f"cal τ={tau}"),
                "skill_val": skill_val,
                "coef": {"intercept": float(coef[0]), **dict(zip(xv.columns, coef[1:]))},
                "n": int(len(pair)),
                "n_val": int(len(pair) - split),
            }
    return best or {"tau_days": np.nan, "nse_val": np.nan, "n": 0}
