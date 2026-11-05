"""Site identifiers, parameter codes, and rainfall classification.

Hudson River Park focus
-----------------------
USGS 01376515 sits on the left bank at Hudson River Park Pier 84
(lat 40.764639, lon -74.003194), ~4 mi upstream of the mouth.
It is operated with HRECOS / Hudson River Park Trust.

NYC DEP Harbor Survey stations N4 (W. 42nd St) and N5 (Pier A / The Battery)
bracket the park along the Hudson transect; N3B (W. 125th St) is the next
upstream Inner Harbor station.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
OUTPUTS = ROOT / "outputs"

# --- Water quality ----------------------------------------------------------
USGS_SITE_ID = "01376515"
USGS_SITE_NAME = "Hudson River at Pier 84 at New York NY"
USGS_LAT = 40.764639
USGS_LON = -74.003194
USGS_START = "2019-05-01"

# NWIS parameter codes available at this sonde (15-minute)
USGS_PARAMS = {
    "00010": "water_temp_c",
    "00095": "specific_conductance_uS_cm",
    "00300": "dissolved_oxygen_mg_l",
    "00400": "ph",
    "63680": "turbidity_fnu",
    "00480": "salinity_psu",
}

HARBOR_STATIONS = {
    "N4": "W. 42nd Street (Hudson River Park, mid-park)",
    "N5": "Pier A / The Battery (south end of Hudson River Park)",
    "N3B": "W. 125th Street (upstream Inner Harbor)",
}

HARBOR_DATASET = "5uug-f49n"
HARBOR_BASE = "https://data.cityofnewyork.us/resource"

# --- Rainfall ---------------------------------------------------------------
# Central Park is the long-record NOAA/NWS climate station for Manhattan.
# It is ~2.5 km from Pier 84 — the best freely available daily gauge for
# Hudson River Park. Hourly ASOS from LaGuardia (KLGA) supplements
# sub-daily timing; Central Park hourly METAR (KNYC) is used when present.
ACIS_STATIONS = {
    "central_park": {
        "sid": "USW00094728",
        "name": "NY City Central Park",
        "lat": 40.77898,
        "lon": -73.96925,
    },
}

IEM_HOURLY_STATIONS = {
    "NYC": {"name": "Central Park (KNYC)", "lat": 40.7789, "lon": -73.9692},
    "LGA": {"name": "LaGuardia (KLGA)", "lat": 40.7772, "lon": -73.8726},
}

# Lin et al. (2024) Table 4, after Zhou et al. (2021). Daily rainfall (mm).
RAINFALL_CLASSES_MM = {
    "sunny": (0.0, 0.0),
    "light": (0.1, 9.9),
    "medium": (10.0, 24.9),
    "heavy": (25.0, 49.9),
    "rainstorm": (50.0, 99.9),
    "extreme": (100.0, float("inf")),
}

# NYC combined-sewer activation is commonly cited near 0.1 inch (~2.5 mm)
# of rainfall. Used as a dry/wet CSO-proxy threshold, not a physical outfall model.
CSO_RAIN_THRESHOLD_IN = 0.10
CSO_RAIN_THRESHOLD_MM = CSO_RAIN_THRESHOLD_IN * 25.4

# --- Sentinel-2 water colour (mid-channel, west of Pier 84) -----------------
# The sonde sits on the Manhattan bulkhead. Optical pixels at the pier mix
# dock, buildings, and water, so we sample a small Hudson box ~0.5 km offshore.
# Planetary Computer STAC + item crop; no token for sentinel-2-l2a.
RS_COLLECTION = "sentinel-2-l2a"
RS_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
RS_DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1"
RS_BBOX = (-74.012, 40.763, -74.007, 40.766)  # west, south, east, north
RS_MAX_CLOUD_PCT = 85.0
RS_MIN_WATER_PIXELS = 15
RS_MATCH_MINUTES = 60
RS_ASSETS = ("B02", "B03", "B04", "B05", "B08", "B11", "SCL")

SITES = {
    "usgs_pier84": {
        "agency": "USGS / HRECOS",
        "id": USGS_SITE_ID,
        "name": USGS_SITE_NAME,
        "lat": USGS_LAT,
        "lon": USGS_LON,
        "role": "primary continuous WQ at Hudson River Park",
    },
    "harbor_n4": {
        "agency": "NYC DEP Harbor Survey",
        "id": "N4",
        "name": HARBOR_STATIONS["N4"],
        "role": "pathogen / nutrient grab samples, mid-park",
    },
    "harbor_n5": {
        "agency": "NYC DEP Harbor Survey",
        "id": "N5",
        "name": HARBOR_STATIONS["N5"],
        "role": "pathogen / nutrient grab samples, Battery",
    },
    "rain_central_park": {
        "agency": "NOAA / NWS via ACIS",
        "id": "USW00094728",
        "name": "NY City Central Park",
        "role": "primary daily rainfall for Manhattan",
    },
}


def classify_daily_rain_mm(precip_mm: float) -> str:
    """Assign a Lin et al. (2024) rainfall class to a daily total (mm).

    Each class runs up to the next class's lower bound, so totals between the
    table's one-decimal limits (0.39 in = 9.906 mm) stay in the lower class.
    """
    if precip_mm is None or precip_mm != precip_mm:  # NaN
        return "unknown"
    if precip_mm <= 0:
        return "sunny"
    for name in ("extreme", "rainstorm", "heavy", "medium"):
        if precip_mm >= RAINFALL_CLASSES_MM[name][0]:
            return name
    return "light"
