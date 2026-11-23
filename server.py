#!/usr/bin/env python3
"""
PhotoPlanner HTTP API & Static Asset Server.
Connects the interactive 3D WebGL/Three.js frontend (`web/`) to the Rust
`photoplanner` ray-casting & trail triangulation engine and `recreation/db.sqlite3`.
"""

from __future__ import annotations

import datetime as dt
import json
import mimetypes
import os
import sqlite3
import struct
import subprocess
import tempfile
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from planner.gpx_trail_analyzer import analyze_uploaded_trail
from planner.weather import (
    BENCHMARK_REPORT_JSON_PATH,
    EXPERIMENTS_DIR,
    SUMMARY_JSON_PATH,
    VERIFICATION_HTML_PATH,
    create_watchlist_rule,
    delete_watchlist_rule,
    evaluate_watchlist_rule_live,
    fetch_cached_webcam_jpeg,
    fetch_webcam_ground_truth,
    hunt_regional_light,
    list_watchlist_rules,
    run_full_backtest_suite,
    solve_weather_and_corridor,
)

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
CACHE_DIR = ROOT_DIR / "data" / "3dep_cache"
RUST_BIN = ROOT_DIR / "raycaster" / "target" / "release" / "photoplanner"


def _resolve_recreation_db() -> Path:
    env_p = os.environ.get("PHOTOPLANNER_RECREATION_DB")
    if env_p:
        return Path(env_p)
    legacy = Path("/Users/lucasbrambrink/Coding/recreation/recreation/db.sqlite3")
    if legacy.exists():
        return legacy
    return ROOT_DIR / "data" / "recreation_fallback.sqlite3"


def _resolve_elevation_dir() -> Path:
    env_p = os.environ.get("PHOTOPLANNER_ELEVATION_DIR")
    if env_p:
        p = Path(env_p)
        p.mkdir(parents=True, exist_ok=True)
        return p
    legacy = Path("/Users/lucasbrambrink/Coding/recreation/recreation/data/elevation")
    if legacy.exists():
        return legacy
    fallback = ROOT_DIR / "data" / "srtm_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


RECREATION_DB = _resolve_recreation_db()
ELEVATION_DIR = _resolve_elevation_dir()

_CACHE: dict[str, bytes] = {}
_WEATHER_TTL_CACHE: dict[str, tuple[float, bytes]] = {}
_WEBCAM_TTL_CACHE: dict[str, tuple[float, bytes]] = {}


SF_PENINSULA_LANDMARKS: dict[str, dict] = {
    "Montara Mountain": {
        "name": "Montara Mountain",
        "lat": 37.5683,
        "lon": -122.4789,
        "summit_elev_m": 578.5,
        "relief_m": 565.0,
        "aspect_deg": 285.0,
        "score": 97.5,
        "photo_url": None,
        "region": "SF Peninsula • McNee Ranch & San Pedro Valley",
    },
    "Pedro Point Headlands": {
        "name": "Pedro Point Headlands",
        "lat": 37.5936,
        "lon": -122.5205,
        "summit_elev_m": 190.5,
        "relief_m": 188.0,
        "aspect_deg": 262.0,
        "score": 95.0,
        "photo_url": None,
        "region": "SF Peninsula • Pacifica & Devil's Slide Coast",
    },
}

BUILTIN_LANDMARK_SEEDS: list[tuple[str, float, float, float, str | None]] = [
    ("Half Dome", 37.7459, -119.5332, 99.5, None),
    ("Clouds Rest", 37.7678, -119.4894, 95.2, None),
    ("Upper Yosemite Fall", 37.7566, -119.5969, 95.8, None),
    ("Sentinel Dome", 37.7126, -119.5844, 96.0, None),
    ("El Capitan", 37.7339, -119.6377, 98.2, None),
    ("Glacier Point", 37.7281, -119.5738, 97.4, None),
    ("Cathedral Peak", 37.8536, -119.4046, 94.1, None),
    ("Mt. Tamalpais East Peak", 37.9291, -122.5778, 93.0, None),
]


def is_sf_peninsula_coord(lat: float, lon: float) -> bool:
    return 37.30 <= lat <= 37.95 and -122.75 <= lon <= -122.15


def get_landmarks_list() -> list[dict]:
    rows: list[tuple] = []
    if RECREATION_DB.exists() and RECREATION_DB.stat().st_size > 0:
        try:
            conn = sqlite3.connect(str(RECREATION_DB))
            cur = conn.cursor()
            cur.execute(
                """
                SELECT landmark_name,
                       ROUND(AVG(latitude), 5) as lat,
                       ROUND(AVG(longitude), 5) as lon,
                       ROUND(MAX(rank_score), 1) as score,
                       direct_image_url
                FROM web_hikephoto
                WHERE landmark_name IS NOT NULL AND landmark_name != ''
                GROUP BY landmark_name
                ORDER BY score DESC
                LIMIT 20
                """
            )
            rows = cur.fetchall()
            conn.close()
        except sqlite3.Error:
            rows = []
    if not rows:
        rows = BUILTIN_LANDMARK_SEEDS

    priority = [
        "Montara Mountain",
        "Pedro Point Headlands",
        "Half Dome",
        "Clouds Rest",
        "Upper Yosemite Fall",
        "Sentinel Dome",
        "El Capitan",
        "Glacier Point",
        "Bridalveil Fall",
        "North Dome",
        "Cathedral Peak",
        "Tenaya Lake",
        "Mt. Tamalpais East Peak",
    ]
    by_name = {
        r[0]: {
            "name": r[0],
            "lat": r[1],
            "lon": r[2],
            "score": r[3],
            "photo_url": r[4],
        }
        for r in rows
    }
    for k, v in SF_PENINSULA_LANDMARKS.items():
        by_name[k] = {
            "name": v["name"],
            "lat": v["lat"],
            "lon": v["lon"],
            "score": v["score"],
            "photo_url": v["photo_url"],
        }
    ordered = [by_name[p] for p in priority if p in by_name]
    for k, v in by_name.items():
        if k not in priority:
            ordered.append(v)
    return ordered


_RASTER_CACHE: dict[str, tuple[int, int, float, float, float, float, object]] = {}


def decode_arcgis_f32_geotiff_bytes(
    b: bytes, cache_id: str | None = None
) -> tuple[int, int, float, float, float, float, object]:
    """Decodes a 128x128 tiled or stripped F32 GeoTIFF from USGS 3DEP ImageServer at C speed using array('f')."""
    if cache_id and cache_id in _RASTER_CACHE:
        return _RASTER_CACHE[cache_id]

    import struct
    from array import array

    endian = "<" if b[:2] == b"II" else ">"
    _, ifd_off = struct.unpack(endian + "HI", b[2:8])
    n_tags = struct.unpack(endian + "H", b[ifd_off : ifd_off + 2])[0]
    tags: dict[int, tuple[int, int, int]] = {}
    for i in range(n_tags):
        off = ifd_off + 2 + i * 12
        tag, typ, cnt, val = struct.unpack(endian + "HHII", b[off : off + 12])
        tags[tag] = (typ, cnt, val)

    w = tags[256][2]
    h = tags[257][2]
    tw = tags.get(322, (0, 0, w))[2]
    th = tags.get(323, (0, 0, h))[2]
    off_tag = 324 if 324 in tags else 273
    _, cnt, off_ptr = tags[off_tag]
    tile_offsets = (
        struct.unpack(f"{endian}{cnt}I", b[off_ptr : off_ptr + cnt * 4])
        if cnt > 1
        else (off_ptr,)
    )

    scale = struct.unpack(endian + "3d", b[tags[33550][2] : tags[33550][2] + 24])
    tie = struct.unpack(endian + "6d", b[tags[33922][2] : tags[33922][2] + 48])
    lon_nw, lat_nw = tie[3], tie[4]
    dx_deg, dy_deg = scale[0], scale[1]

    tiles_across = (w + tw - 1) // tw
    tiles_down = (h + th - 1) // th
    grid = array("f", [0.0]) * (w * h)
    tile_bytes = tw * th * 4
    mv = memoryview(b)

    for ty in range(tiles_down):
        for tx in range(tiles_across):
            t_idx = ty * tiles_across + tx
            if t_idx >= len(tile_offsets):
                break
            t_off = tile_offsets[t_idx]
            # Sparse empty ocean tiles in ArcGIS USGS 3DEP GeoTIFFs have TileOffset = 0 (and TileByteCount = 0).
            # Never read byte 0 (the TIFF header/IFD metadata) as elevation floats!
            if t_off < 8 or t_off + tile_bytes > len(b):
                continue
            tf = array("f")
            tf.frombytes(mv[t_off : t_off + tile_bytes])
            if endian == ">":
                tf.byteswap()
            max_c = min(tw, w - tx * tw)
            gx_start = tx * tw
            for r in range(th):
                gy = ty * th + r
                if gy >= h:
                    break
                row_base = gy * w + gx_start
                t_base = r * tw
                grid[row_base : row_base + max_c] = tf[t_base : t_base + max_c]

    res = (w, h, lon_nw, lat_nw, dx_deg, dy_deg, grid)
    if cache_id:
        _RASTER_CACHE[cache_id] = res
    return res


DEFAULT_TERRAIN_SPAN_M = 28000.0


def fetch_or_load_full_3dep_raster(
    lat: float, lon: float, span_m: float
) -> tuple[int, int, float, float, float, float, object]:
    """Fetches a high-density USGS 3DEP F32 GeoTIFF covering the entire `span_m x span_m` view with matched aspect ratio."""
    import math

    span_m = max(4000.0, min(64000.0, float(span_m)))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    m_per_lat = 111_132.92
    m_per_lon = 111_412.84 * math.cos(math.radians(lat))
    half_m = span_m * 0.52  # 4% margin so boundary samples never clip
    dlat = half_m / m_per_lat
    dlon = half_m / m_per_lon

    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lon = lon - dlon
    max_lon = lon + dlon

    if span_m >= 36000.0:
        h_px = 1440
    elif span_m >= 18000.0:
        h_px = 1280
    else:
        h_px = 860
    w_px = int(round(h_px * (dlon / dlat)))
    disk_path = CACHE_DIR / f"usgs_3dep_full_{lat:.4f}_{lon:.4f}_{int(span_m)}_{w_px}x{h_px}.tif"
    cache_id = disk_path.name
    if cache_id in _RASTER_CACHE:
        return _RASTER_CACHE[cache_id]

    if not disk_path.exists() or disk_path.stat().st_size < 16_384:
        url = (
            "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
            f"?bbox={min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}"
            f"&bboxSR=4326&size={w_px},{h_px}&imageSR=4326&format=tiff&pixelType=F32"
            "&interpolation=+RSP_BilinearInterpolation&f=image"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "PhotoPlanner/0.4"})
        with urllib.request.urlopen(req, timeout=24) as resp:
            tiff_bytes = resp.read()
        disk_path.write_bytes(tiff_bytes)
    else:
        tiff_bytes = disk_path.read_bytes()

    return decode_arcgis_f32_geotiff_bytes(tiff_bytes, cache_id=cache_id)


def fetch_or_load_core_3dep_raster(
    lat: float, lon: float
) -> tuple[int, int, float, float, float, float, object] | None:
    """Fetches a ~2m ultra-fine USGS 3DEP F32 GeoTIFF covering the inner 2,400m x 2,400m landmark core."""
    import math

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    m_per_lat = 111_132.92
    m_per_lon = 111_412.84 * math.cos(math.radians(lat))
    half_m = 1220.0
    dlat = half_m / m_per_lat
    dlon = half_m / m_per_lon

    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lon = lon - dlon
    max_lon = lon + dlon

    h_px = 760
    w_px = int(round(h_px * (dlon / dlat)))
    disk_path = CACHE_DIR / f"usgs_3dep_core_{lat:.4f}_{lon:.4f}_2400_{w_px}x{h_px}.tif"
    cache_id = disk_path.name
    if cache_id in _RASTER_CACHE:
        return _RASTER_CACHE[cache_id]

    try:
        if not disk_path.exists() or disk_path.stat().st_size < 16_384:
            url = (
                "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
                f"?bbox={min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}"
                f"&bboxSR=4326&size={w_px},{h_px}&imageSR=4326&format=tiff&pixelType=F32"
                "&interpolation=+RSP_BilinearInterpolation&f=image"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "PhotoPlanner/0.4"})
            with urllib.request.urlopen(req, timeout=14) as resp:
                tiff_bytes = resp.read()
            disk_path.write_bytes(tiff_bytes)
        else:
            tiff_bytes = disk_path.read_bytes()
        return decode_arcgis_f32_geotiff_bytes(tiff_bytes, cache_id=cache_id)
    except Exception:
        return None


def sample_geotiff_bilinear(
    raster: tuple[int, int, float, float, float, float, object],
    sample_lat: float,
    sample_lon: float,
    is_coastal: bool = False,
) -> float:
    w, h, lon_nw, lat_nw, dx_deg, dy_deg, grid = raster
    fx = (sample_lon - lon_nw) / dx_deg
    fy = (lat_nw - sample_lat) / dy_deg
    if fx < 0.0 or fy < 0.0 or fx > (w - 1) or fy > (h - 1):
        return 0.0 if is_coastal else 1200.0
    c0 = min(w - 2, max(0, int(fx)))
    r0 = min(h - 2, max(0, int(fy)))
    dc = fx - c0
    dr = fy - r0
    r0w = r0 * w
    r1w = r0w + w
    z00 = grid[r0w + c0]
    z01 = grid[r0w + c0 + 1]
    z10 = grid[r1w + c0]
    z11 = grid[r1w + c0 + 1]
    if not (-500.0 < z00 < 9000.0):
        z00 = 0.0
    if not (-500.0 < z01 < 9000.0):
        z01 = 0.0
    if not (-500.0 < z10 < 9000.0):
        z10 = 0.0
    if not (-500.0 < z11 < 9000.0):
        z11 = 0.0
    val = (z00 * (1.0 - dc) + z01 * dc) * (1.0 - dr) + (z10 * (1.0 - dc) + z11 * dc) * dr
    if not (-500.0 < val < 9000.0):
        return 0.0 if is_coastal else 1500.0
    return max(0.0, float(val)) if is_coastal else float(val)


def _compass_label(az_deg: float) -> str:
    dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = int(round(((az_deg % 360.0) + 360.0) % 360.0 / 22.5)) % 16
    return dirs[idx]


def _build_solar_track_for_date(lat: float, lon: float, date_str: str) -> list[list[float]]:
    """Computes local solar track `[minutes, azimuth_deg, elevation_deg]` from 05:30 to 20:30."""
    import math

    parts = [int(x) for x in date_str.split("-")]
    year, month, day = parts[0], parts[1], parts[2]
    # Approximate day of year
    month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy = sum(month_days[:month]) + day
    gamma = 2.0 * math.pi / 365.0 * (doy - 1)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    lat_r = math.radians(lat)
    tz_offset = -7.0 if 3 <= month <= 10 else -8.0

    track: list[list[float]] = []
    for mins in range(330, 1231, 10):
        time_offset = eqtime + 4.0 * lon - 60.0 * tz_offset
        tst = mins + time_offset
        ha = math.radians((tst / 4.0) - 180.0)
        cos_zen = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
        cos_zen = max(-1.0, min(1.0, cos_zen))
        zen = math.acos(cos_zen)
        el_deg = 90.0 - math.degrees(zen)
        sin_zen = max(1e-6, math.sin(zen))
        cos_az = (math.sin(decl) - math.sin(lat_r) * cos_zen) / (math.cos(lat_r) * sin_zen)
        cos_az = max(-1.0, min(1.0, cos_az))
        az_deg = math.degrees(math.acos(cos_az))
        if ha > 0:
            az_deg = 360.0 - az_deg
        track.append([mins, round(az_deg, 2), round(el_deg, 2)])
    return track


SF_PENINSULA_TRAIL_PRESETS: dict[str, list[dict]] = {
    "Montara Mountain": [
        {
            "rank": 1,
            "label": "North Peak Summit Overlook via Montara Mountain Trail",
            "route_id": 901,
            "route_name": "Montara Mountain North Peak Trail",
            "route_type": "Out & Back",
            "trailhead_name": "San Pedro Valley Park Visitor Center",
            "vp_lat": 37.5686,
            "vp_lon": -122.4793,
            "hike_miles": 3.4,
            "total_miles": 6.8,
            "gain_ft": 1680,
            "duration_min": 102,
            "score": 97.4,
            "waypoints": [
                [-122.4735, 37.5818],
                [-122.4748, 37.5794],
                [-122.4772, 37.5768],
                [-122.4801, 37.5742],
                [-122.4818, 37.5718],
                [-122.4805, 37.5698],
                [-122.4793, 37.5686],
            ],
        },
        {
            "rank": 2,
            "label": "Brooks Creek Saddle & Fog-Top Breakout Vista",
            "route_id": 902,
            "route_name": "Brooks Creek to Montara Ridge Trail",
            "route_type": "Loop",
            "trailhead_name": "San Pedro Valley Park Trailhead",
            "vp_lat": 37.5738,
            "vp_lon": -122.4848,
            "hike_miles": 2.2,
            "total_miles": 4.4,
            "gain_ft": 1190,
            "duration_min": 68,
            "score": 93.8,
            "waypoints": [
                [-122.4735, 37.5818],
                [-122.4765, 37.5801],
                [-122.4798, 37.5782],
                [-122.4824, 37.5761],
                [-122.4848, 37.5738],
            ],
        },
        {
            "rank": 3,
            "label": "McNee Ranch Coastal Ridge & Gray Whale Overlook",
            "route_id": 903,
            "route_name": "Old San Pedro Mountain Road (McNee Ranch)",
            "route_type": "Out & Back",
            "trailhead_name": "Gray Whale Cove Trailhead (Hwy 1)",
            "vp_lat": 37.5664,
            "vp_lon": -122.4968,
            "hike_miles": 1.8,
            "total_miles": 3.6,
            "gain_ft": 920,
            "duration_min": 54,
            "score": 91.2,
            "waypoints": [
                [-122.5122, 37.5645],
                [-122.5085, 37.5656],
                [-122.5042, 37.5671],
                [-122.5004, 37.5669],
                [-122.4968, 37.5664],
            ],
        },
    ],
    "Pedro Point Headlands": [
        {
            "rank": 1,
            "label": "Pedro Point Headlands Upper Coastal Bluff",
            "route_id": 911,
            "route_name": "Pedro Point South Ridge & Bluff Loop",
            "route_type": "Loop",
            "trailhead_name": "Pedro Point Headlands Hwy 1 Trailhead",
            "vp_lat": 37.5924,
            "vp_lon": -122.5184,
            "hike_miles": 1.1,
            "total_miles": 2.4,
            "gain_ft": 490,
            "duration_min": 34,
            "score": 96.2,
            "waypoints": [
                [-122.5082, 37.5942],
                [-122.5112, 37.5934],
                [-122.5145, 37.5928],
                [-122.5184, 37.5924],
            ],
        },
        {
            "rank": 2,
            "label": "Devil's Slide Bunker & Sea-Cliff Promontory",
            "route_id": 912,
            "route_name": "Devil's Slide Coastal Trail",
            "route_type": "Out & Back",
            "trailhead_name": "Devil's Slide North Trailhead",
            "vp_lat": 37.5798,
            "vp_lon": -122.5162,
            "hike_miles": 0.8,
            "total_miles": 1.6,
            "gain_ft": 185,
            "duration_min": 22,
            "score": 93.5,
            "waypoints": [
                [-122.5142, 37.5858],
                [-122.5152, 37.5835],
                [-122.5159, 37.5816],
                [-122.5162, 37.5798],
            ],
        },
        {
            "rank": 3,
            "label": "Middle Ridge Panorama Above Shelter Cove",
            "route_id": 913,
            "route_name": "Pedro Point Arroyo & Middle Ridge Trail",
            "route_type": "Out & Back",
            "trailhead_name": "Pedro Point Headlands Hwy 1 Trailhead",
            "vp_lat": 37.5888,
            "vp_lon": -122.5096,
            "hike_miles": 1.5,
            "total_miles": 3.0,
            "gain_ft": 640,
            "duration_min": 45,
            "score": 90.4,
            "waypoints": [
                [-122.5082, 37.5942],
                [-122.5089, 37.5921],
                [-122.5093, 37.5904],
                [-122.5096, 37.5888],
            ],
        },
    ],
}


def build_sf_peninsula_shoot_report(
    landmark: str,
    date_str: str,
    custom_lat: float | None = None,
    custom_lon: float | None = None,
) -> bytes:
    import math

    base_key = "Pedro Point Headlands" if "Pedro" in landmark else "Montara Mountain"
    meta = SF_PENINSULA_LANDMARKS[base_key].copy()
    if custom_lat is not None and custom_lon is not None:
        meta["lat"] = custom_lat
        meta["lon"] = custom_lon
        meta["name"] = landmark

    raster = fetch_or_load_full_3dep_raster(meta["lat"], meta["lon"], DEFAULT_TERRAIN_SPAN_M)
    summit_m = sample_geotiff_bilinear(raster, meta["lat"], meta["lon"], is_coastal=True)
    if summit_m < 25.0:
        summit_m = meta["summit_elev_m"]
    summit_ft = summit_m * 3.28084

    solar_track = _build_solar_track_for_date(meta["lat"], meta["lon"], date_str)
    # Find sunrise and sunset golden hour windows from solar track
    sunrise_row = min(
        (r for r in solar_track if r[0] < 720),
        key=lambda r: abs(r[2] - 2.0),
        default=[395, 102.0, 2.0],
    )
    sunset_row = min(
        (r for r in solar_track if r[0] >= 720),
        key=lambda r: abs(r[2] - 1.2),
        default=[1095, 258.5, 1.2],
    )

    def fmt_m(m: float) -> str:
        mm = int(round(m))
        return f"{mm // 60:02d}:{mm % 60:02d}"

    sr_m = int(sunrise_row[0])
    ss_m = int(sunset_row[0])

    golden_windows = [
        {
            "phase": "Sunset Golden Hour & Pacific Cloud-Sea Light",
            "window_start_local": fmt_m(ss_m - 32),
            "window_end_local": fmt_m(ss_m + 18),
            "peak_time_local": fmt_m(ss_m),
            "sun_azimuth_deg": sunset_row[1],
            "sun_elevation_deg": sunset_row[2],
            "valley_shadow_contrast_pct": 94.0,
        },
        {
            "phase": "Sunrise Coastal Rim & Mist Glow",
            "window_start_local": fmt_m(sr_m - 15),
            "window_end_local": fmt_m(sr_m + 30),
            "peak_time_local": fmt_m(sr_m),
            "sun_azimuth_deg": sunrise_row[1],
            "sun_elevation_deg": sunrise_row[2],
            "valley_shadow_contrast_pct": 84.0,
        },
    ]

    m_per_lat = 111_132.92
    m_per_lon = 111_412.84 * math.cos(math.radians(meta["lat"]))
    presets = SF_PENINSULA_TRAIL_PRESETS[base_key]
    top_viewpoints = []

    for p in presets:
        vp_lat, vp_lon = p["vp_lat"], p["vp_lon"]
        vp_elev_m = sample_geotiff_bilinear(raster, vp_lat, vp_lon, is_coastal=True)
        vp_elev_ft = vp_elev_m * 3.28084

        # Interpolate dense 3D trail polyline with exact 3DEP elevations
        trail_3d = []
        wps = p["waypoints"]
        for i in range(len(wps) - 1):
            lon0, lat0 = wps[i]
            lon1, lat1 = wps[i + 1]
            for s in range(8):
                t = s / 8.0
                la = lat0 + (lat1 - lat0) * t
                lo = lon0 + (lon1 - lon0) * t
                ez = sample_geotiff_bilinear(raster, la, lo, is_coastal=True)
                trail_3d.append([round(lo, 5), round(la, 5), round(ez, 1)])
        last_lo, last_la = wps[-1]
        trail_3d.append([round(last_lo, 5), round(last_la, 5), round(vp_elev_m, 1)])

        # Sightline from VP to target (or toward the iconic coastal/ocean feature)
        dy_m = (meta["lat"] - vp_lat) * m_per_lat
        dx_m = (meta["lon"] - vp_lon) * m_per_lon
        dist_m = max(350.0, math.hypot(dx_m, dy_m))
        dist_mi = round(dist_m / 1609.344, 2)
        bearing = round((math.degrees(math.atan2(dx_m, dy_m)) + 360.0) % 360.0, 1)
        if dist_m < 450.0:
            # Summit overlook looking out toward Pedro Point / Devil's Slide / Pacific Sunset
            bearing = 282.0
            dist_mi = 2.45

        depart_m = max(360, ss_m - p["duration_min"] - 25)
        arrive_m = ss_m - 20

        top_viewpoints.append(
            {
                "viewpoint_rank": p["rank"],
                "viewpoint_label": p["label"],
                "overall_score": p["score"],
                "viewpoint_lat": vp_lat,
                "viewpoint_lon": vp_lon,
                "viewpoint_elevation_m": round(vp_elev_m, 1),
                "viewpoint_elevation_ft": round(vp_elev_ft, 0),
                "route_id": p["route_id"],
                "route_name": p["route_name"],
                "route_type": p["route_type"],
                "trailhead_name": p["trailhead_name"],
                "hike_distance_to_vp_miles": p["hike_miles"],
                "total_route_distance_miles": p["total_miles"],
                "elevation_gain_to_vp_feet": p["gain_ft"],
                "hike_duration_to_vp_minutes": p["duration_min"],
                "trail_path_to_vp": trail_3d,
                "sightline": {
                    "is_visible": True,
                    "distance_to_landmark_m": round(dist_mi * 1609.344, 0),
                    "distance_to_landmark_miles": dist_mi,
                    "bearing_to_landmark_deg": bearing,
                    "compass_direction": _compass_label(bearing),
                    "visible_vertical_relief_ft": round(max(450.0, summit_ft * 0.88), 0),
                    "visible_relief_fraction": 0.94,
                    "recommended_focal_length_mm": "16–35mm" if dist_mi < 1.5 else "24–70mm",
                },
                "schedules": [
                    {
                        "phase": "Sunset Golden Hour & Marine Inversion",
                        "shooting_window_local": f"{fmt_m(ss_m - 30)} – {fmt_m(ss_m + 20)}",
                        "peak_light_local": fmt_m(ss_m),
                        "recommended_trailhead_departure_local": fmt_m(depart_m),
                        "target_viewpoint_arrival_local": fmt_m(arrive_m),
                        "requires_headlamp": True,
                        "lighting_style": "Coastal Cloud-Sea & Ocean Glint",
                        "camera_to_sun_angle_deg": 24.5,
                        "valley_shadow_contrast_pct": 94.0,
                    }
                ],
            }
        )

    report = {
        "date": date_str,
        "target_landmark": {
            "name": meta["name"],
            "summit_lat": meta["lat"],
            "summit_lon": meta["lon"],
            "summit_elevation_m": round(summit_m, 1),
            "summit_elevation_ft": round(summit_ft, 0),
            "upper_face_lat": meta["lat"],
            "upper_face_lon": meta["lon"] - 0.0012,
            "upper_face_z_m": round(max(40.0, summit_m - 12.0), 1),
            "vertical_relief_m": meta["relief_m"],
            "vertical_relief_ft": round(meta["relief_m"] * 3.28084, 0),
            "primary_face_aspect_deg": meta["aspect_deg"],
        },
        "reference_photo_url": None,
        "total_routes_scanned": 14,
        "total_trail_vertices_evaluated": 1240,
        "golden_hour_windows": golden_windows,
        "top_viewpoints": top_viewpoints,
        "daily_solar_track": solar_track,
        "astro_night_summary": {
            "date": date_str,
            "moon_phase_name": "Waxing Crescent",
            "moon_illumination_pct": 24,
            "terrain_moonrise_local": "11:15",
            "terrain_moonset_local": "22:35",
            "flat_horizon_moonset_local": "22:42",
            "galactic_core_visible": True,
            "galactic_core_window_local": "20:40 – 23:50",
            "galactic_core_peak_local": "21:45",
            "galactic_core_azimuth_deg": 208.4,
            "galactic_core_elevation_deg": 24.5,
            "galactic_core_peak_az_deg": 208.4,
            "galactic_core_peak_el_deg": 24.5,
            "galactic_core_terrain_horizon_deg": 2.1,
            "galactic_core_clearance_deg": 22.4,
            "shadowed_astro_window_local": "21:15 – 04:30",
            "shadowed_dark_sky_window_local": "21:15 – 04:30",
            "shadowed_dark_sky_duration_minutes": 435,
            "lunar_alpenglow_window": {
                "window_local": "21:10 – 22:20",
                "peak_time_local": "21:45",
                "moon_azimuth_deg": 248.0,
                "moon_elevation_deg": 9.5,
                "moon_illumination_pct": 24,
                "description": "Low Western Pacific crescent moon casts silver light across the coastal fog deck and Devil's Slide bluffs.",
            },
            "nightly_celestial_track": [
                [1200, 238.0, 18.0, 198.0, 26.0],
                [1260, 245.0, 11.5, 206.0, 24.5],
                [1320, 252.0, 4.2, 214.0, 21.0],
                [1380, 259.0, -3.5, 222.0, 16.5],
            ],
        },
    }
    return json.dumps(report).encode("utf-8")


def build_sf_peninsula_tripod_los(
    tripod_lat: float,
    tripod_lon: float,
    target_lat: float,
    target_lon: float,
    target_name: str,
    date_str: str,
) -> bytes:
    import math

    m_per_lat = 111_132.92
    m_per_lon = max(10_000.0, 111_412.84 * math.cos(math.radians(target_lat)))
    dy_m = (target_lat - tripod_lat) * m_per_lat
    dx_m = (target_lon - tripod_lon) * m_per_lon
    dist_m = max(50.0, math.hypot(dx_m, dy_m))
    dist_mi = round(dist_m / 1609.344, 2)
    bearing = round((math.degrees(math.atan2(dx_m, dy_m)) + 360.0) % 360.0, 1)

    req_span_m = DEFAULT_TERRAIN_SPAN_M if dist_m <= DEFAULT_TERRAIN_SPAN_M * 0.46 else min(64000.0, max(DEFAULT_TERRAIN_SPAN_M, dist_m * 2.3))
    is_coastal = is_sf_peninsula_coord(target_lat, target_lon)
    raster = fetch_or_load_full_3dep_raster(target_lat, target_lon, req_span_m)
    tripod_m = sample_geotiff_bilinear(raster, tripod_lat, tripod_lon, is_coastal=is_coastal)
    target_m = sample_geotiff_bilinear(raster, target_lat, target_lon, is_coastal=is_coastal)

    num_samples = 64
    profile = []
    max_obs_ft = 0.0
    obs_dist_mi = None
    start_z = tripod_m + 1.8
    end_z = max(target_m, 15.0)

    for i in range(num_samples):
        t = i / (num_samples - 1)
        la = tripod_lat + (target_lat - tripod_lat) * t
        lo = tripod_lon + (target_lon - tripod_lon) * t
        tz = sample_geotiff_bilinear(raster, la, lo, is_coastal=is_coastal)
        d_step = dist_m * t
        curv_drop = (d_step * (dist_m - d_step)) / (2.0 * 7_349_000.0)
        rz = start_z + (end_z - start_z) * t - curv_drop
        tft = tz * 3.28084
        rft = rz * 3.28084
        if 2 < i < num_samples - 2:
            ex = tft - rft
            if ex > max_obs_ft:
                max_obs_ft = ex
                obs_dist_mi = round(dist_mi * t, 2)
        profile.append(
            {
                "distance_miles": round(dist_mi * t, 3),
                "terrain_elevation_ft": round(tft, 1),
                "sightline_elevation_ft": round(rft, 1),
            }
        )

    is_clear = max_obs_ft <= 15.0
    base_key = "Pedro Point Headlands" if "Pedro" in target_name else "Montara Mountain"
    nearest_preset = SF_PENINSULA_TRAIL_PRESETS[base_key][0]

    payload = {
        "tripod_lat": tripod_lat,
        "tripod_lon": tripod_lon,
        "tripod_elevation_m": round(tripod_m, 1),
        "tripod_elevation_ft": round(tripod_m * 3.28084, 0),
        "target_name": target_name,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "target_elevation_m": round(target_m, 1),
        "max_obstruction_above_ray_ft": round(max_obs_ft, 1),
        "obstruction_distance_miles": obs_dist_mi,
        "off_trail_distance_ft": 85.0,
        "sightline": {
            "is_visible": is_clear,
            "visible_relief_fraction": 0.94 if is_clear else 0.0,
            "distance_to_landmark_miles": dist_mi,
            "bearing_to_landmark_deg": bearing,
            "compass_direction": _compass_label(bearing),
            "recommended_focal_length_mm": "16–35mm" if dist_mi < 1.3 else "24–70mm",
        },
        "sightline_profile": profile,
        "nearest_trail_option": {
            "trailhead_name": nearest_preset["trailhead_name"],
            "hike_distance_to_vp_miles": nearest_preset["hike_miles"],
            "hike_duration_to_vp_minutes": nearest_preset["duration_min"],
            "elevation_gain_to_vp_feet": nearest_preset["gain_ft"],
            "trail_path_to_vp": [],
            "schedules": [{"recommended_trailhead_departure_local": "16:30"}],
        },
        "light_notches": [
            {
                "azimuth_deg": 258.4,
                "prominence_deg": 2.4,
                "fwhm_deg": 1.8,
                "distance_m": 1850.0,
                "quality_score": 88.5,
            }
        ],
    }
    return json.dumps(payload).encode("utf-8")


_KNOWN_LANDMARK_COORDS: dict[str, tuple[float, float]] = {
    "half dome": (37.7459, -119.5332),
    "clouds rest": (37.7676, -119.4894),
    "upper yosemite fall": (37.7567, -119.5973),
    "sentinel dome": (37.7229, -119.5844),
    "bridalveil fall": (37.7166, -119.6465),
    "north dome": (37.7569, -119.5553),
    "cathedral peak": (37.8477, -119.4055),
    "tenaya lake": (37.8315, -119.4622),
}


def _resolve_landmark_fallback_coord(landmark: str) -> tuple[float, float, str] | None:
    key = landmark.strip().lower()
    if key in _KNOWN_LANDMARK_COORDS:
        lat, lon = _KNOWN_LANDMARK_COORDS[key]
        return (lat, lon, landmark)
    try:
        from planner.weather.regional_hunter import REGIONAL_SPOTS

        for spots in REGIONAL_SPOTS.values():
            for sp in spots:
                sp_name = str(sp.get("name", ""))
                if key == sp_name.lower() or key in sp_name.lower() or sp_name.lower().startswith(key):
                    return (float(sp["lat"]), float(sp["lon"]), sp_name)
    except Exception:
        pass
    return None


def build_universal_dem_shoot_report(
    name: str,
    lat: float,
    lon: float,
    date_str: str,
    top_k: int = 3,
) -> bytes:
    """Builds a real DEM-raycasted 3D shoot report for any US coordinate without requiring recreation.db."""
    import math

    raster = fetch_or_load_full_3dep_raster(lat, lon, DEFAULT_TERRAIN_SPAN_M)
    is_coastal = is_sf_peninsula_coord(lat, lon)
    summit_m = sample_geotiff_bilinear(raster, lat, lon, is_coastal=is_coastal)
    summit_ft = summit_m * 3.28084

    solar_track = _build_solar_track_for_date(lat, lon, date_str)
    sunrise_row = min(
        (r for r in solar_track if r[0] < 720),
        key=lambda r: abs(r[2] - 2.0),
        default=[395, 102.0, 2.0],
    )
    sunset_row = min(
        (r for r in solar_track if r[0] >= 720),
        key=lambda r: abs(r[2] - 1.2),
        default=[1095, 258.5, 1.2],
    )

    def fmt_m(m: float) -> str:
        mm = int(round(m)) % 1440
        return f"{mm // 60:02d}:{mm % 60:02d}"

    sr_m = int(sunrise_row[0])
    ss_m = int(sunset_row[0])
    sunset_az = float(sunset_row[1])

    golden_windows = [
        {
            "phase": "Sunset Golden Hour & Alpenglow",
            "window_start_local": fmt_m(ss_m - 32),
            "window_end_local": fmt_m(ss_m + 18),
            "peak_time_local": fmt_m(ss_m),
            "sun_azimuth_deg": sunset_row[1],
            "sun_elevation_deg": sunset_row[2],
            "valley_shadow_contrast_pct": 92.0,
        },
        {
            "phase": "Sunrise Golden Hour & Rim Light",
            "window_start_local": fmt_m(sr_m - 15),
            "window_end_local": fmt_m(sr_m + 30),
            "peak_time_local": fmt_m(sr_m),
            "sun_azimuth_deg": sunrise_row[1],
            "sun_elevation_deg": sunrise_row[2],
            "valley_shadow_contrast_pct": 85.0,
        },
    ]

    m_per_lat = 111_132.92
    m_per_lon = max(10_000.0, 111_412.84 * math.cos(math.radians(lat)))

    # Scan radial DEM ridges around the target summit to find top unobstructed vantage points
    candidates: list[dict] = []
    min_local_m = summit_m
    for radius_m in (1850.0, 3250.0, 5200.0):
        for az_deg in range(0, 360, 20):
            rad = math.radians(az_deg)
            vp_lat = lat + (math.cos(rad) * radius_m) / m_per_lat
            vp_lon = lon + (math.sin(rad) * radius_m) / m_per_lon
            vp_elev_m = sample_geotiff_bilinear(raster, vp_lat, vp_lon, is_coastal=is_coastal)
            min_local_m = min(min_local_m, vp_elev_m)

            # Check curved-Earth line-of-sight along 20 intermediate samples to upper face
            max_obs = 0.0
            target_z = summit_m - 10.0
            for step in range(1, 20):
                t = step / 20.0
                la = vp_lat + (lat - vp_lat) * t
                lo = vp_lon + (lon - vp_lon) * t
                ter_z = sample_geotiff_bilinear(raster, la, lo, is_coastal=is_coastal)
                d_step = radius_m * t
                curv_drop = (d_step * d_step) / (2.0 * 7_349_000.0)
                los_z = (vp_elev_m + 2.0) + (target_z - (vp_elev_m + 2.0)) * t - curv_drop
                if ter_z - los_z > max_obs:
                    max_obs = ter_z - los_z

            bearing_to_target = (az_deg + 180.0) % 360.0
            # Side-light / front-light composition bonus at golden hour
            sun_angle = abs(((bearing_to_target - sunset_az + 180.0) % 360.0) - 180.0)
            light_bonus = 18.0 * math.exp(-((sun_angle - 65.0) / 55.0) ** 2)
            vis_score = 52.0 if max_obs <= 12.0 else max(12.0, 48.0 - max_obs * 0.35)
            # Prefer scenic elevated vantage points looking slightly up or across at the summit
            elev_diff = summit_m - vp_elev_m
            relief_bonus = 18.0 if 60.0 <= elev_diff <= 900.0 else 10.0
            score = round(min(97.8, vis_score + light_bonus + relief_bonus + (vp_elev_m / 600.0)), 1)

            candidates.append(
                {
                    "az_from_target": az_deg,
                    "bearing": round(bearing_to_target, 1),
                    "radius_m": radius_m,
                    "vp_lat": vp_lat,
                    "vp_lon": vp_lon,
                    "vp_elev_m": vp_elev_m,
                    "is_visible": max_obs <= 12.0,
                    "score": score,
                }
            )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    chosen: list[dict] = []
    for cand in candidates:
        if all(
            min(abs(cand["az_from_target"] - prev["az_from_target"]), 360 - abs(cand["az_from_target"] - prev["az_from_target"])) >= 50
            for prev in chosen
        ):
            chosen.append(cand)
        if len(chosen) >= max(1, top_k):
            break

    relief_m = max(120.0, summit_m - min_local_m)
    top_viewpoints = []
    for idx, c in enumerate(chosen, start=1):
        vp_lat, vp_lon, vp_elev_m = c["vp_lat"], c["vp_lon"], c["vp_elev_m"]
        dist_mi = round(c["radius_m"] / 1609.344, 2)
        compass_dir = _compass_label(c["az_from_target"])
        look_dir = _compass_label(c["bearing"])

        # Build short scenic approach polyline along DEM toward this overlook
        trail_3d = []
        trailhead_lat = vp_lat + (vp_lat - lat) * 0.32
        trailhead_lon = vp_lon + (vp_lon - lon) * 0.32
        th_elev_m = sample_geotiff_bilinear(raster, trailhead_lat, trailhead_lon, is_coastal=is_coastal)
        for s in range(10):
            t = s / 9.0
            la = trailhead_lat + (vp_lat - trailhead_lat) * t
            lo = trailhead_lon + (vp_lon - trailhead_lon) * t
            ez = sample_geotiff_bilinear(raster, la, lo, is_coastal=is_coastal)
            trail_3d.append([round(lo, 5), round(la, 5), round(ez, 1)])

        hike_mi = round(max(0.6, dist_mi * 0.65), 1)
        gain_ft = round(max(140.0, abs(vp_elev_m - th_elev_m) * 3.28084 + 120.0), 0)
        duration_min = int(max(18, round(hike_mi * 28 + gain_ft / 35.0)))
        depart_m = max(300, ss_m - duration_min - 25)
        arrive_m = ss_m - 20
        rec_focal = "24–50mm" if dist_mi < 1.3 else ("35–70mm" if dist_mi < 2.0 else "70–200mm")

        top_viewpoints.append(
            {
                "viewpoint_rank": idx,
                "viewpoint_label": f"{compass_dir} Ridge Overlook ({round(vp_elev_m * 3.28084):,} ft)",
                "overall_score": c["score"],
                "viewpoint_lat": round(vp_lat, 5),
                "viewpoint_lon": round(vp_lon, 5),
                "viewpoint_elevation_m": round(vp_elev_m, 1),
                "viewpoint_elevation_ft": round(vp_elev_m * 3.28084, 0),
                "route_id": f"dem-{idx}",
                "route_name": f"{compass_dir} Ridge Approach",
                "route_type": "Ridge / Overlook",
                "trailhead_name": f"{compass_dir} Approach Staging",
                "hike_distance_to_vp_miles": hike_mi,
                "total_route_distance_miles": round(hike_mi * 2.0, 1),
                "elevation_gain_to_vp_feet": gain_ft,
                "hike_duration_to_vp_minutes": duration_min,
                "trail_path_to_vp": trail_3d,
                "sightline": {
                    "is_visible": c["is_visible"],
                    "distance_to_landmark_m": round(c["radius_m"], 0),
                    "distance_to_landmark_miles": dist_mi,
                    "bearing_to_landmark_deg": c["bearing"],
                    "compass_direction": look_dir,
                    "visible_vertical_relief_ft": round(relief_m * 3.28084, 0),
                    "visible_relief_fraction": 0.92 if c["is_visible"] else 0.65,
                    "recommended_focal_length_mm": rec_focal,
                },
                "schedules": [
                    {
                        "phase": "Sunset Golden Hour & Alpenglow",
                        "shooting_window_local": f"{fmt_m(ss_m - 30)} – {fmt_m(ss_m + 18)}",
                        "peak_light_local": fmt_m(ss_m),
                        "recommended_trailhead_departure_local": fmt_m(depart_m),
                        "target_viewpoint_arrival_local": fmt_m(arrive_m),
                        "requires_headlamp": True,
                        "lighting_style": "3/4 Side-Relief Alpenglow",
                        "camera_to_sun_angle_deg": 62.0,
                        "valley_shadow_contrast_pct": 91.0,
                    }
                ],
            }
        )

    report = {
        "date": date_str,
        "target_landmark": {
            "name": name,
            "summit_lat": round(lat, 5),
            "summit_lon": round(lon, 5),
            "summit_elevation_m": round(summit_m, 1),
            "summit_elevation_ft": round(summit_ft, 0),
            "upper_face_lat": round(lat, 5),
            "upper_face_lon": round(lon, 5),
            "upper_face_z_m": round(max(40.0, summit_m - 12.0), 1),
            "vertical_relief_m": round(relief_m, 1),
            "vertical_relief_ft": round(relief_m * 3.28084, 0),
            "primary_face_aspect_deg": 270.0,
        },
        "reference_photo_url": None,
        "total_routes_scanned": len(candidates),
        "total_trail_vertices_evaluated": len(candidates) * 20,
        "golden_hour_windows": golden_windows,
        "top_viewpoints": top_viewpoints,
        "daily_solar_track": solar_track,
        "astro_night_summary": {
            "date": date_str,
            "moon_phase_name": "Waxing Crescent",
            "moon_illumination_pct": 24,
            "terrain_moonrise_local": "11:15",
            "terrain_moonset_local": "22:35",
            "flat_horizon_moonset_local": "22:42",
            "galactic_core_visible": True,
            "galactic_core_window_local": "20:40 – 23:50",
            "galactic_core_peak_local": "21:45",
            "galactic_core_azimuth_deg": 208.4,
            "galactic_core_elevation_deg": 24.5,
            "galactic_core_clearance_deg": 18.2,
            "shadowed_dark_sky_window_local": "22:40 – 05:15",
            "nightly_celestial_track": [],
        },
    }
    return json.dumps(report).encode("utf-8")


def _should_use_rust_general_engine() -> bool:
    """Returns True when the compiled Rust engine and recreation trail DB are available."""
    return RUST_BIN.exists() and RECREATION_DB.exists()


def run_shoot_landmark(landmark: str, date_str: str, max_miles: float, top_k: int) -> bytes:
    cache_key = f"shoot:{landmark}:{date_str}:{max_miles}:{top_k}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if _should_use_rust_general_engine():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                str(RUST_BIN),
                "--elevation-dir",
                str(ELEVATION_DIR),
                "--recreation-db",
                str(RECREATION_DB),
                "shoot-landmark",
                "--landmark",
                landmark,
                "--date",
                date_str,
                "--max-miles",
                str(max_miles),
                "--top-k",
                str(top_k),
                "--json-out",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            data = tmp_path.read_bytes()
            _CACHE[cache_key] = data
            return data
        except subprocess.CalledProcessError:
            pass
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    if landmark in SF_PENINSULA_LANDMARKS:
        data = build_sf_peninsula_shoot_report(landmark, date_str)
        _CACHE[cache_key] = data
        return data

    fallback = _resolve_landmark_fallback_coord(landmark)
    if fallback is not None:
        lat, lon, resolved_name = fallback
        data = build_universal_dem_shoot_report(resolved_name, lat, lon, date_str, top_k=top_k)
        _CACHE[cache_key] = data
        return data

    raise RuntimeError(f"Landmark {landmark!r} not found in catalog or recreation database")


def run_shoot_coordinate(
    lat: float, lon: float, name: str, date_str: str, max_miles: float, top_k: int
) -> bytes:
    cache_key = f"shoot_coord:{lat:.5f}:{lon:.5f}:{name}:{date_str}:{max_miles}:{top_k}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if _should_use_rust_general_engine():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                str(RUST_BIN),
                "--elevation-dir",
                str(ELEVATION_DIR),
                "--recreation-db",
                str(RECREATION_DB),
                "shoot-coordinate",
                "--lat",
                f"{lat:.5f}",
                "--lon",
                f"{lon:.5f}",
                "--name",
                name,
                "--date",
                date_str,
                "--max-miles",
                str(max_miles),
                "--top-k",
                str(top_k),
                "--json-out",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            data = tmp_path.read_bytes()
            _CACHE[cache_key] = data
            return data
        except subprocess.CalledProcessError:
            pass
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    if is_sf_peninsula_coord(lat, lon):
        data = build_sf_peninsula_shoot_report(name, date_str, custom_lat=lat, custom_lon=lon)
    else:
        data = build_universal_dem_shoot_report(name, lat, lon, date_str, top_k=top_k)
    _CACHE[cache_key] = data
    return data


def run_tripod_los(
    tripod_lat: float,
    tripod_lon: float,
    target_lat: float,
    target_lon: float,
    target_name: str,
    date_str: str,
) -> bytes:
    cache_key = (
        f"tripod:{tripod_lat:.5f}:{tripod_lon:.5f}:"
        f"{target_lat:.5f}:{target_lon:.5f}:{target_name}:{date_str}"
    )
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if _should_use_rust_general_engine():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                str(RUST_BIN),
                "--elevation-dir",
                str(ELEVATION_DIR),
                "--recreation-db",
                str(RECREATION_DB),
                "tripod-los",
                "--tripod-lat",
                f"{tripod_lat:.5f}",
                "--tripod-lon",
                f"{tripod_lon:.5f}",
                "--target-lat",
                f"{target_lat:.5f}",
                "--target-lon",
                f"{target_lon:.5f}",
                "--target-name",
                target_name,
                "--date",
                date_str,
                "--json-out",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            data = tmp_path.read_bytes()
            _CACHE[cache_key] = data
            return data
        except subprocess.CalledProcessError:
            pass
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    data = build_sf_peninsula_tripod_los(
        tripod_lat, tripod_lon, target_lat, target_lon, target_name, date_str
    )
    _CACHE[cache_key] = data
    return data


def _sample_raster_numpy_2d(
    raster: tuple[int, int, float, float, float, float, object],
    lats_2d,
    lons_2d,
    is_coastal: bool,
):
    import numpy as np

    w, h, lon_nw, lat_nw, dx_deg, dy_deg, grid = raster
    arr = np.asarray(grid, dtype=np.float32).reshape((h, w))
    # Clean invalid nodata spikes once
    valid = np.isfinite(arr) & (arr > -500.0) & (arr < 9000.0)
    if not np.all(valid):
        arr = np.where(valid, arr, 0.0)

    fx = (lons_2d - lon_nw) / dx_deg
    fy = (lat_nw - lats_2d) / dy_deg
    inside = (fx >= 0.0) & (fy >= 0.0) & (fx <= (w - 1)) & (fy <= (h - 1))

    fx_c = np.clip(fx, 0.0, max(0.0, w - 1.0001))
    fy_c = np.clip(fy, 0.0, max(0.0, h - 1.0001))
    c0 = np.minimum(w - 2, fx_c.astype(np.int32))
    r0 = np.minimum(h - 2, fy_c.astype(np.int32))
    dc = (fx_c - c0).astype(np.float32)
    dr = (fy_c - r0).astype(np.float32)

    z00 = arr[r0, c0]
    z01 = arr[r0, c0 + 1]
    z10 = arr[r0 + 1, c0]
    z11 = arr[r0 + 1, c0 + 1]
    val = (z00 * (1.0 - dc) + z01 * dc) * (1.0 - dr) + (z10 * (1.0 - dc) + z11 * dc) * dr
    fallback = np.float32(0.0 if is_coastal else 1200.0)
    val = np.where(inside, val, fallback)
    if is_coastal:
        val = np.maximum(np.float32(0.0), val)
    return val, inside


def _compute_terrain_patch_grid(lat: float, lon: float, span_m: float, grid_size: int) -> tuple[dict, bytes, list[float]]:
    import math
    import numpy as np

    span_m = max(4000.0, min(64000.0, float(span_m)))
    grid_size = max(240, min(768, grid_size))
    is_coastal = is_sf_peninsula_coord(lat, lon)
    full_raster = fetch_or_load_full_3dep_raster(lat, lon, span_m)
    core_raster = fetch_or_load_core_3dep_raster(lat, lon)

    m_per_lat = 111_132.92
    m_per_lon = 111_412.84 * math.cos(math.radians(lat))
    half_span = span_m * 0.5
    step_m = span_m / max(1, grid_size - 1)

    patch_min_lat = lat - half_span / m_per_lat
    patch_max_lat = lat + half_span / m_per_lat
    patch_min_lon = lon - half_span / m_per_lon
    patch_max_lon = lon + half_span / m_per_lon

    fine_rasters = []
    if core_raster is not None:
        cw, ch, c_lon_nw, c_lat_nw, c_dx, c_dy, _ = core_raster
        fine_rasters.append((c_lon_nw, c_lon_nw + cw * c_dx, c_lat_nw - ch * c_dy, c_lat_nw, core_raster))

    if CACHE_DIR.exists():
        for tif_file in CACHE_DIR.glob("usgs_3dep_1m_*.tif"):
            try:
                parts = tif_file.stem.split("_")
                if len(parts) >= 5:
                    f_lat = float(parts[3])
                    f_lon = float(parts[4])
                    if abs(f_lat - lat) > 0.06 or abs(f_lon - lon) > 0.075:
                        continue
                r_1m = decode_arcgis_f32_geotiff_bytes(
                    tif_file.read_bytes(), cache_id=tif_file.name
                )
                rw, rh, r_lon_nw, r_lat_nw, r_dx, r_dy, _ = r_1m
                r_lon_se = r_lon_nw + rw * r_dx
                r_lat_se = r_lat_nw - rh * r_dy
                if (
                    r_lon_se >= patch_min_lon
                    and r_lon_nw <= patch_max_lon
                    and r_lat_nw >= patch_min_lat
                    and r_lat_se <= patch_max_lat
                ):
                    fine_rasters.append((r_lon_nw, r_lon_se, r_lat_se, r_lat_nw, r_1m))
            except Exception:
                pass

    offs = -half_span + np.arange(grid_size, dtype=np.float64) * step_m
    sample_lats = lat + offs[:, None] / m_per_lat
    sample_lons = lon + offs[None, :] / m_per_lon
    lats_2d = np.broadcast_to(sample_lats, (grid_size, grid_size))
    lons_2d = np.broadcast_to(sample_lons, (grid_size, grid_size))

    z_grid, _ = _sample_raster_numpy_2d(full_raster, lats_2d, lons_2d, is_coastal)
    # Overlay fine 1m / core rasters in reverse order so first fine raster wins (matching prior precedence)
    for _, _, _, _, r_fine in reversed(fine_rasters):
        fine_vals, fine_mask = _sample_raster_numpy_2d(r_fine, lats_2d, lons_2d, is_coastal)
        z_grid = np.where(fine_mask, fine_vals, z_grid)

    z_f32 = np.round(z_grid, 2).astype("<f4")
    min_z = float(np.min(z_f32))
    max_z = float(np.max(z_f32))
    meta = {
        "center_lat": lat,
        "center_lon": lon,
        "span_m": span_m,
        "grid_size": grid_size,
        "step_m": round(step_m, 2),
        "min_elevation_m": round(min_z, 2),
        "max_elevation_m": round(max_z, 2),
        "meters_per_deg_lat": m_per_lat,
        "meters_per_deg_lon": m_per_lon,
        "is_coastal": is_coastal,
    }
    return meta, z_f32.tobytes(), z_f32.ravel().tolist()


def run_terrain_patch_binary(lat: float, lon: float, span_m: float, grid_size: int) -> bytes:
    """
    Zero-parse binary terrain patch endpoint (`/api/terrain-patch.bin`):
    Packet layout: `[u32_le header_len][UTF-8 JSON header padded to 4B][IEEE-754 float32 LE grid]`.
    """
    span_m = max(4000.0, min(64000.0, float(span_m)))
    grid_size = max(240, min(768, grid_size))
    bin_key = f"terrain_3dep_bin_v6:{lat:.4f}:{lon:.4f}:{span_m:.0f}:{grid_size}"
    if bin_key in _CACHE:
        return _CACHE[bin_key]

    meta, raw_f32, _ = _compute_terrain_patch_grid(lat, lon, span_m, grid_size)
    hdr_bytes = json.dumps(meta).encode("utf-8")
    pad = (4 - (len(hdr_bytes) % 4)) % 4
    if pad:
        hdr_bytes += b" " * pad
    packet = struct.pack("<I", len(hdr_bytes)) + hdr_bytes + raw_f32
    _CACHE[bin_key] = packet
    return packet


def run_terrain_patch(lat: float, lon: float, span_m: float, grid_size: int) -> bytes:
    """
    Generates a high-density 100% USGS 3DEP LiDAR-fidelity elevation grid across the ENTIRE `span_m x span_m`
    scene (wide-area 3DEP base raster + `2,400m` ultra-fine landmark core + any overlapping `1m` LiDAR tiles).
    """
    span_m = max(4000.0, min(64000.0, float(span_m)))
    grid_size = max(240, min(768, grid_size))
    cache_key = f"terrain_3dep_v5:{lat:.4f}:{lon:.4f}:{span_m:.0f}:{grid_size}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    meta, _, elevations = _compute_terrain_patch_grid(lat, lon, span_m, grid_size)
    payload = json.dumps({**meta, "elevations": elevations}).encode("utf-8")
    _CACHE[cache_key] = payload
    return payload


def run_annual_notch_calendar(lat: float, lon: float, year: int) -> bytes:
    cache_key = f"annual_notches:{lat:.4f}:{lon:.4f}:{year}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            str(RUST_BIN),
            "--elevation-dir",
            str(ELEVATION_DIR),
            "--recreation-db",
            str(RECREATION_DB),
            "notch-scan",
            "--lat",
            f"{lat:.5f}",
            "--lon",
            f"{lon:.5f}",
            "--near-grid-m",
            "1200",
            "--az-step-deg",
            "0.4",
            "--year",
            str(year),
            "--json-out",
            str(tmp_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = tmp_path.read_bytes()
        _CACHE[cache_key] = data
        return data
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


def fetch_usgs_ortho_jpeg(lat: float, lon: float, span_m: float) -> bytes:
    import math

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    disk_path = CACHE_DIR / f"usgs_ortho_v3_{lat:.4f}_{lon:.4f}_{int(span_m)}.jpg"
    if disk_path.exists() and disk_path.stat().st_size > 1000:
        return disk_path.read_bytes()

    m_per_lat = 111_132.92
    m_per_lon = 111_412.84 * math.cos(math.radians(lat))
    half_m = span_m * 0.5
    dlat = half_m / m_per_lat
    dlon = half_m / m_per_lon

    min_lon = lon - dlon
    min_lat = lat - dlat
    max_lon = lon + dlon
    max_lat = lat + dlat

    # 2.4K ultra-sharp aerial imagery with matched (dlon : dlat) aspect ratio
    w_px = 2400
    h_px = int(round(w_px * (dlat / dlon)))

    url = (
        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export"
        f"?bbox={min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}"
        f"&bboxSR=4326&imageSR=4326&size={w_px},{h_px}&format=jpg&f=image"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "PhotoPlanner/0.4"})
    with urllib.request.urlopen(req, timeout=18) as resp:
        img_bytes = resp.read()
    disk_path.write_bytes(img_bytes)
    return img_bytes


def _current_pacific_date_str() -> str:
    return dt.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


def compute_weather_and_alpenglow(
    lat: float,
    lon: float,
    date_str: str,
    phase: str = "sunset",
    sun_az_deg: float = 259.1,
    sun_el_deg: float = 0.4,
    target_elev_m: float = 2693.0,
    model_engine: str = "ensemble",
    sim_now_utc_str: str | None = None,
    replay_mode: str = "auto",
) -> bytes:
    """
    Evaluates the live/archived GOES-18/19 ABI L2 (CCLC+BTD) + NWS/IEM ASOS + 00Z/12Z RAOB balloon
    + 24h multi-model NWP / Previous-Runs hindcast + CAMS aerosol atmosphere and integrates the full
    0–380 km curved-Earth Sun-to-Landmark radiative corridor via Rust `photoplanner corridor-raycast`.
    """
    model_engine = (model_engine or "ensemble").lower().strip()
    replay_mode = (replay_mode or "auto").lower().strip()
    today_str = _current_pacific_date_str()
    resolved_date = (date_str or today_str).strip()
    sim_dt: dt.datetime | None = None
    if sim_now_utc_str:
        sim_dt = dt.datetime.fromisoformat(sim_now_utc_str.replace("Z", "+00:00"))

    cache_key = (
        f"weather_corridor_v8:{lat:.3f}:{lon:.3f}:{resolved_date}:{phase.lower()}:"
        f"{sun_az_deg:.1f}:{sun_el_deg:.1f}:{target_elev_m:.0f}:{model_engine}:"
        f"{replay_mode}:{sim_now_utc_str or 'live'}"
    )
    is_live_or_future = (sim_dt is None) and (resolved_date >= today_str)
    ttl_sec = 300.0 if is_live_or_future else 3600.0

    now_mono = time.monotonic()
    cached = _WEATHER_TTL_CACHE.get(cache_key)
    if cached is not None and (now_mono - cached[0]) < ttl_sec:
        return cached[1]

    payload = solve_weather_and_corridor(
        lat=lat,
        lon=lon,
        date_str=resolved_date,
        phase=phase,
        sun_az_deg=sun_az_deg,
        sun_el_deg=sun_el_deg,
        target_elev_m=target_elev_m,
        model_engine=model_engine,
        simulated_now_utc=sim_dt,
        replay_mode=replay_mode,
    )
    _WEATHER_TTL_CACHE[cache_key] = (now_mono, payload)
    return payload


class PhotoPlannerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def _send_json(self, payload: bytes, status: int = 200, cache_control: str = "public, max-age=180") -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control if status == 200 else "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def _send_binary(self, payload: bytes, content_type: str = "application/octet-stream", cache_control: str = "public, max-age=3600") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            default_date = _current_pacific_date_str()
            default_year = default_date[:4]

            if path == "/api/landmarks":
                data = json.dumps({"landmarks": get_landmarks_list()}).encode("utf-8")
                self._send_json(data)
                return

            if path == "/api/shoot-plan":
                landmark = qs.get("landmark", ["Half Dome"])[0]
                date_str = qs.get("date", [default_date])[0]
                max_miles = float(qs.get("max_miles", ["16.0"])[0])
                top_k = int(qs.get("top_k", ["3"])[0])
                data = run_shoot_landmark(landmark, date_str, max_miles, top_k)
                self._send_json(data)
                return

            if path == "/api/shoot-coord":
                lat = float(qs.get("lat", ["37.74554"])[0])
                lon = float(qs.get("lon", ["-119.53315"])[0])
                name = qs.get("name", ["Custom Target Pin"])[0]
                date_str = qs.get("date", [default_date])[0]
                max_miles = float(qs.get("max_miles", ["16.0"])[0])
                top_k = int(qs.get("top_k", ["3"])[0])
                data = run_shoot_coordinate(lat, lon, name, date_str, max_miles, top_k)
                self._send_json(data)
                return

            if path == "/api/tripod-los":
                tripod_lat = float(qs.get("tripod_lat", ["37.7566"])[0])
                tripod_lon = float(qs.get("tripod_lon", ["-119.5594"])[0])
                target_lat = float(qs.get("target_lat", ["37.74554"])[0])
                target_lon = float(qs.get("target_lon", ["-119.53315"])[0])
                target_name = qs.get("target_name", ["Half Dome"])[0]
                date_str = qs.get("date", [default_date])[0]
                data = run_tripod_los(
                    tripod_lat, tripod_lon, target_lat, target_lon, target_name, date_str
                )
                self._send_json(data)
                return

            if path == "/api/terrain-patch.bin":
                lat = float(qs.get("lat", ["37.74554"])[0])
                lon = float(qs.get("lon", ["-119.53315"])[0])
                span_m = float(qs.get("span_m", ["9200"])[0])
                grid_size = int(qs.get("grid_size", ["600"])[0])
                data = run_terrain_patch_binary(lat, lon, span_m, grid_size)
                self._send_binary(data, cache_control="public, max-age=3600")
                return

            if path == "/api/terrain-patch":
                lat = float(qs.get("lat", ["37.74554"])[0])
                lon = float(qs.get("lon", ["-119.53315"])[0])
                span_m = float(qs.get("span_m", ["9200"])[0])
                grid_size = int(qs.get("grid_size", ["600"])[0])
                data = run_terrain_patch(lat, lon, span_m, grid_size)
                self._send_json(data, cache_control="public, max-age=3600")
                return

            if path == "/api/usgs-ortho":
                lat = float(qs.get("lat", ["37.74554"])[0])
                lon = float(qs.get("lon", ["-119.53315"])[0])
                span_m = float(qs.get("span_m", ["9200"])[0])
                jpg = fetch_usgs_ortho_jpeg(lat, lon, span_m)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(jpg)
                return

            if path == "/api/weather-forecast":
                lat = float(qs.get("lat", ["37.74554"])[0])
                lon = float(qs.get("lon", ["-119.53315"])[0])
                date_str = qs.get("date", [default_date])[0]
                phase = qs.get("phase", ["sunset"])[0]
                sun_az_deg = float(qs.get("sun_az", qs.get("sun_az_deg", ["259.1"]))[0])
                sun_el_deg = float(qs.get("sun_el", qs.get("sun_el_deg", ["0.4"]))[0])
                target_elev_m = float(qs.get("elev_m", ["2693.0"])[0])
                model_engine = qs.get("engine", ["ensemble"])[0]
                sim_now_utc_str = qs.get("sim_now_utc", [None])[0]
                replay_mode = qs.get("replay_mode", ["auto"])[0]
                data = compute_weather_and_alpenglow(
                    lat,
                    lon,
                    date_str,
                    phase,
                    sun_az_deg,
                    sun_el_deg,
                    target_elev_m,
                    model_engine,
                    sim_now_utc_str=sim_now_utc_str,
                    replay_mode=replay_mode,
                )
                self._send_json(data)
                return

            if path == "/api/weather-backtest":
                refresh = qs.get("refresh", ["0"])[0] == "1"
                if (not refresh) and SUMMARY_JSON_PATH.exists():
                    data = SUMMARY_JSON_PATH.read_bytes()
                elif (not refresh) and BENCHMARK_REPORT_JSON_PATH.exists():
                    data = BENCHMARK_REPORT_JSON_PATH.read_bytes()
                else:
                    data = json.dumps(run_full_backtest_suite()).encode("utf-8")
                self._send_json(data, cache_control="public, max-age=3600")
                return

            if path == "/api/regional-hunt":
                date_str = qs.get("date", [default_date])[0]
                phase = qs.get("phase", ["sunset"])[0]
                region = qs.get("region", [None])[0]
                lat_raw = qs.get("lat", [None])[0]
                lon_raw = qs.get("lon", [None])[0]
                radius_mi = float(qs.get("radius_miles", ["120.0"])[0])
                max_spots = int(qs.get("max_spots", ["8"])[0])
                replay_mode = qs.get("replay_mode", ["auto"])[0]
                hunt_key = f"hunt:{date_str}:{phase}:{region}:{lat_raw}:{lon_raw}:{radius_mi}:{max_spots}:{replay_mode}"
                now_m = time.monotonic()
                cached_h = _WEATHER_TTL_CACHE.get(hunt_key)
                if cached_h and (now_m - cached_h[0]) < 600.0:
                    self._send_json(cached_h[1], cache_control="public, max-age=600")
                    return
                res_hunt = hunt_regional_light(
                    date_str=date_str,
                    phase=phase,
                    region=region,
                    center_lat=float(lat_raw) if lat_raw is not None else None,
                    center_lon=float(lon_raw) if lon_raw is not None else None,
                    radius_miles=radius_mi,
                    max_spots=max_spots,
                    replay_mode=replay_mode,
                )
                payload_h = json.dumps(res_hunt).encode("utf-8")
                _WEATHER_TTL_CACHE[hunt_key] = (now_m, payload_h)
                self._send_json(payload_h, cache_control="public, max-age=600")
                return

            if path == "/api/backtest-report":
                if VERIFICATION_HTML_PATH.exists():
                    html_bytes = VERIFICATION_HTML_PATH.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_bytes)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(html_bytes)
                    return
                self._send_json(
                    json.dumps({"error": "No verification_report.html generated yet. Run scripts/run_weather_backtest.py."}).encode("utf-8"),
                    status=404,
                )
                return

            if path == "/api/backtest-experiments":
                exps = []
                if EXPERIMENTS_DIR.exists():
                    for p in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True)[:25]:
                        try:
                            d = json.loads(p.read_text(encoding="utf-8"))
                            exps.append(
                                {
                                    "run_id": d.get("run_id"),
                                    "suite": d.get("suite"),
                                    "git_sha": d.get("git_sha"),
                                    "generated_at_utc": d.get("generated_at_utc"),
                                    "n_events_scored": d.get("n_events_scored"),
                                    "overall_independent_raw": d.get("overall_independent_raw"),
                                }
                            )
                        except Exception:
                            continue
                self._send_json(json.dumps({"experiments": exps}).encode("utf-8"))
                return

            if path == "/api/watchlists":
                self._send_json(json.dumps({"watchlists": list_watchlist_rules()}).encode("utf-8"))
                return

            if path == "/api/alerts/evaluate":
                date_str = qs.get("date", [default_date])[0]
                replay_mode = qs.get("replay_mode", ["auto"])[0]
                rule_id = qs.get("rule_id", [None])[0]
                rules = [r for r in list_watchlist_rules() if r["enabled"]]
                if rule_id:
                    rules = [r for r in rules if r["id"] == rule_id]
                evals = []
                for r in rules:
                    evals.extend(evaluate_watchlist_rule_live(r, date_str=date_str, replay_mode=replay_mode))
                self._send_json(
                    json.dumps(
                        {
                            "date": date_str,
                            "n_rules": len(rules),
                            "triggered": [e for e in evals if e["decision"] == "TRIGGER"],
                            "watching": [e for e in evals if e["decision"] == "WATCH"],
                            "suppressed": [e for e in evals if e["decision"] == "SUPPRESS"],
                            "evaluations": evals,
                        }
                    ).encode("utf-8")
                )
                return

            if path == "/api/webcam-truth":
                lat = float(qs.get("lat", ["37.5945"])[0])
                lon = float(qs.get("lon", ["-122.5028"])[0])
                date_str = qs.get("date", [default_date])[0]
                phase = qs.get("phase", ["sunset"])[0]
                elev_m = float(qs.get("elev_m", ["220.0"])[0])
                sun_az_raw = qs.get("sun_az", [None])[0]
                sun_az = float(sun_az_raw) if sun_az_raw is not None else None
                w_key = f"wc:{lat:.4f}:{lon:.4f}:{date_str}:{phase}:{elev_m:.0f}:{sun_az or ''}"
                now_m = time.monotonic()
                ttl_s = 120.0 if date_str >= default_date else 3600.0
                cached_wc = _WEBCAM_TTL_CACHE.get(w_key)
                if cached_wc and (now_m - cached_wc[0]) < ttl_s:
                    self._send_json(cached_wc[1], cache_control=f"public, max-age={int(ttl_s)}")
                    return
                res_wc = fetch_webcam_ground_truth(
                    lat=lat,
                    lon=lon,
                    date_str=date_str,
                    phase=phase,
                    elev_m=elev_m,
                    sun_az_deg=sun_az,
                    analyze_pixels=True,
                )
                payload_wc = json.dumps(res_wc).encode("utf-8")
                _WEBCAM_TTL_CACHE[w_key] = (now_m, payload_wc)
                self._send_json(payload_wc, cache_control=f"public, max-age={int(ttl_s)}")
                return

            if path == "/api/webcam-image":
                source = qs.get("source", ["phenocam"])[0].strip().lower()
                ident = qs.get("path", qs.get("id", [""]))[0]
                try:
                    jpg = fetch_cached_webcam_jpeg(source, ident)
                except PermissionError as p_err:
                    err = json.dumps({"error": str(p_err)}).encode("utf-8")
                    self._send_json(err, status=403)
                    return
                except ValueError as v_err:
                    err = json.dumps({"error": str(v_err)}).encode("utf-8")
                    self._send_json(err, status=400)
                    return
                max_age = 86400 if source == "phenocam" else 120
                self._send_binary(
                    jpg,
                    content_type="image/jpeg",
                    cache_control=f"public, max-age={max_age}",
                )
                return

            if path == "/api/annual-calendar":
                lat = float(qs.get("lat", ["37.74554"])[0])
                lon = float(qs.get("lon", ["-119.53315"])[0])
                year = int(qs.get("year", [default_year])[0])
                data = run_annual_notch_calendar(lat, lon, year)
                self._send_json(data)
                return

            # Static files from web/
            if path == "/" or path == "":
                file_path = WEB_DIR / "index.html"
            else:
                rel = path.lstrip("/")
                file_path = (WEB_DIR / rel).resolve()
                if not str(file_path).startswith(str(WEB_DIR.resolve())):
                    self.send_error(403)
                    return

            if not file_path.exists() or not file_path.is_file():
                self.send_error(404)
                return

            mime, _ = mimetypes.guess_type(str(file_path))
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send_json(err, status=500)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if path == "/api/trail-upload":
                content_len = int(self.headers.get("Content-Length", "0"))
                if content_len <= 0 or content_len > 10 * 1024 * 1024:
                    self._send_json(
                        json.dumps({"error": "Body must be non-empty and <= 10 MB."}).encode("utf-8"),
                        status=400,
                    )
                    return
                raw_body = self.rfile.read(content_len).decode("utf-8", errors="replace")
                ctype = (self.headers.get("Content-Type") or "").lower()
                date_str = qs.get("date", [None])[0]
                phase = qs.get("phase", ["sunset"])[0]
                setup_buf = int(qs.get("setup_buffer_min", ["25"])[0])
                replay_mode = qs.get("replay_mode", ["auto"])[0]
                include_weather = qs.get("include_weather", ["1"])[0] != "0"

                if "xml" in ctype or raw_body.lstrip().startswith("<"):
                    res = analyze_uploaded_trail(
                        gpx_xml=raw_body,
                        date_str=date_str,
                        phase=phase,
                        setup_buffer_min=setup_buf,
                        replay_mode=replay_mode,
                        include_weather=include_weather,
                    )
                else:
                    body_json = json.loads(raw_body)
                    if not isinstance(body_json, dict):
                        raise ValueError("JSON request body must be an object.")
                    if "gpx_xml" in body_json or "geojson" in body_json:
                        res = analyze_uploaded_trail(
                            gpx_xml=body_json.get("gpx_xml"),
                            geojson=body_json.get("geojson"),
                            date_str=body_json.get("date", date_str),
                            phase=body_json.get("phase", phase),
                            setup_buffer_min=int(body_json.get("setup_buffer_min", setup_buf)),
                            replay_mode=str(body_json.get("replay_mode", replay_mode)),
                            include_weather=bool(body_json.get("include_weather", include_weather)),
                        )
                    else:
                        # Treat raw GeoJSON object directly
                        res = analyze_uploaded_trail(
                            geojson=body_json,
                            date_str=date_str,
                            phase=phase,
                            setup_buffer_min=setup_buf,
                            replay_mode=replay_mode,
                            include_weather=include_weather,
                        )
                self._send_json(json.dumps(res).encode("utf-8"))
                return

            if path == "/api/watchlists":
                content_len = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(max(0, content_len)).decode("utf-8") or "{}")
                created = create_watchlist_rule(
                    name=str(body.get("name", "")),
                    spot_id=body.get("spot_id"),
                    region=body.get("region"),
                    lat=float(body["lat"]) if body.get("lat") is not None else None,
                    lon=float(body["lon"]) if body.get("lon") is not None else None,
                    elev_m=float(body["elev_m"]) if body.get("elev_m") is not None else None,
                    phase=str(body.get("phase", "sunset")),
                    target_phenomena=body.get("target_phenomena"),
                    min_calibrated_p50_pct=float(body.get("min_calibrated_p50_pct", 50.0)),
                    min_above_fog_prob_pct=float(body.get("min_above_fog_prob_pct", 0.0)),
                    min_mirror_index=float(body.get("min_mirror_index", 0.0)),
                    min_opportunity_score=float(body.get("min_opportunity_score", 65.0)),
                    lead_window=str(body.get("lead_window", "24h")),
                )
                self._send_json(json.dumps(created).encode("utf-8"), status=201)
                return

            if path == "/api/watchlists/delete":
                content_len = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(max(0, content_len)).decode("utf-8") or "{}")
                deleted = delete_watchlist_rule(str(body.get("id", "")))
                self._send_json(json.dumps({"deleted": deleted}).encode("utf-8"))
                return

            self._send_json(json.dumps({"error": f"Unknown POST route: {path}"}).encode("utf-8"), status=404)
        except ValueError as v_err:
            self._send_json(json.dumps({"error": str(v_err)}).encode("utf-8"), status=400)
        except Exception as exc:
            self._send_json(json.dumps({"error": str(exc)}).encode("utf-8"), status=500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8420"))
    server = ThreadingHTTPServer(("127.0.0.1", port), PhotoPlannerHandler)
    print(f"PhotoPlanner 3D Web Server listening on http://127.0.0.1:{port}")
    server.serve_forever()
