"""Extract Sentinel-2 L2A water-colour samples west of Hudson River Park Pier 84.

Uses Microsoft Planetary Computer STAC (no token) and the item crop API.
Raw rows: data/raw/remote_sensing/sentinel2_pier84.csv
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rainfall_wq.config import (  # noqa: E402
    DATA_RAW,
    RS_ASSETS,
    RS_BBOX,
    RS_COLLECTION,
    RS_DATA_API,
    RS_MAX_CLOUD_PCT,
    RS_MIN_WATER_PIXELS,
    RS_STAC_URL,
    USGS_START,
)
from rainfall_wq.remote_sensing import parse_crop_array, scene_indices  # noqa: E402

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "rainfall-water-quality/hudson-park"


def _bbox_polygon(bbox: tuple[float, float, float, float]) -> dict:
    west, south, east, north = bbox
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]],
        },
    }


def search_items(
    start: str,
    end: str,
    max_cloud: float,
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    body: dict | None = {
        "collections": [RS_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 200,
    }
    url = RS_STAC_URL
    items: list[dict] = []
    while body is not None:
        r = SESSION.post(url, json=body, timeout=120)
        r.raise_for_status()
        payload = r.json()
        items.extend(payload.get("features", []))
        nxt = next((lnk for lnk in payload.get("links", []) if lnk.get("rel") == "next"), None)
        if nxt and "body" in nxt:
            body = nxt["body"]
            url = nxt.get("href", url)
        else:
            body = None
    return items


def crop_item(item: dict, bbox: tuple[float, float, float, float], max_size: int = 32) -> np.ndarray:
    item_id = item["id"]
    r = SESSION.post(
        f"{RS_DATA_API}/item/crop.npy",
        params=[
            ("collection", RS_COLLECTION),
            ("item", item_id),
            *[("assets", a) for a in RS_ASSETS],
            ("max_size", max_size),
        ],
        json=_bbox_polygon(bbox),
        timeout=120,
    )
    r.raise_for_status()
    return np.load(io.BytesIO(r.content))


def summarize_item(item: dict, bbox: tuple[float, float, float, float]) -> dict | None:
    props = item.get("properties", {})
    arr = crop_item(item, bbox)
    bands, scl = parse_crop_array(arr)
    stats = scene_indices(
        bands,
        scl,
        props.get("s2:processing_baseline"),
        RS_MIN_WATER_PIXELS,
    )
    if stats is None:
        return None
    dt = pd.to_datetime(props["datetime"], utc=True)
    stats.update(
        {
            "item_id": item["id"],
            "datetime_utc": dt,
            "mgrs_tile": props.get("s2:mgrs_tile"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "processing_baseline": props.get("s2:processing_baseline"),
            "platform": props.get("platform"),
        }
    )
    return stats


def _out_path() -> Path:
    out = DATA_RAW / "remote_sensing" / "sentinel2_pier84.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def load_existing() -> pd.DataFrame:
    path = _out_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["datetime_utc"])


def save_rows(rows: list[dict], existing: pd.DataFrame) -> pd.DataFrame:
    fresh = pd.DataFrame(rows)
    if existing.empty:
        out = fresh
    else:
        out = pd.concat([existing, fresh], ignore_index=True)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["item_id"], keep="last")
    out = out.sort_values("datetime_utc")
    out.to_csv(_out_path(), index=False)
    meta = DATA_RAW / "remote_sensing" / "sentinel2_pier84_meta.json"
    meta.write_text(
        json.dumps(
            {
                "collection": RS_COLLECTION,
                "bbox": list(RS_BBOX),
                "n": int(len(out)),
                "start": str(out["datetime_utc"].min()),
                "end": str(out["datetime_utc"].max()),
                "source": "Microsoft Planetary Computer STAC + item crop",
            },
            indent=2,
        )
    )
    return out


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=USGS_START)
    p.add_argument("--end", default=None)
    p.add_argument("--max-cloud", type=float, default=RS_MAX_CLOUD_PCT)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--reset", action="store_true", help="Ignore an existing extract")
    args = p.parse_args(argv)

    end = args.end or pd.Timestamp.today(tz="UTC").strftime("%Y-%m-%d")
    existing = pd.DataFrame() if args.reset else load_existing()
    seen = set(existing["item_id"].astype(str)) if not existing.empty else set()

    print(f"STAC search {RS_COLLECTION}  {args.start} → {end}  cloud<{args.max_cloud}")
    items = search_items(args.start, end, args.max_cloud, RS_BBOX)
    todo = [it for it in items if it["id"] not in seen]
    print(f"  items={len(items)}  new={len(todo)}  already={len(seen)}")
    if not todo:
        return existing

    rows: list[dict] = []
    failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(summarize_item, it, RS_BBOX): it["id"] for it in todo}
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                rec = fut.result()
            except Exception as exc:
                failed += 1
                rec = None
                if failed <= 8:
                    print(f"  skip {futs[fut]} ({exc})")
            if rec:
                rows.append(rec)
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  kept={len(rows)}  failed={failed}  {time.time()-t0:.0f}s")
                if rows:
                    existing = save_rows(rows, existing)
                    rows = []

    if rows:
        existing = save_rows(rows, existing)
    print(f"wrote {_out_path()}  n={len(existing)}")
    return existing


if __name__ == "__main__":
    main()
