"""
Mountain & Coastal Webcam Ground-Truth Verification, Sky RGB Chromaticity Calibration,
and 2 km GOES-18 NREL NSRDB v4.0.0 Golden-Hour Irradiance (`planner/weather/webcam_truth.py`, v4.2).

Provides:
1. **Live Mountain Network (`ALERTCalifornia`)**: 1,310+ California summit, ridge, and coastal
   HD cameras with live compass azimuth (`az_current`), pitch (`tilt_current`), horizontal FOV
   (`fov`), GPS elevation, and `1920x1080` frames (`cameras.alertcalifornia.org`, zero API keys).
2. **Historical + Live Scientific Archive (`PhenoCam` / `NEON`)**: Timestamped multi-year `1296x960`
   RGB archives across coastal headlands, Coast Range ridges, Monterey Bay, and High Sierra towers
   (`phenocam.nau.edu`, zero API keys; filenames stamped in year-round `UTC-8` LST).
3. **Sky & Horizon RGB Chromaticity Analyzer (`analyze_webcam_sky_chromaticity`)**: Segments the
   upper sky & cloud-underside horizon band (`y in [4%, 20%]`), automatically rejects StarDot NIR
   monochrome/pink night-filter frames, and extracts `R/B` warmth (`P90`), `R/G` crimson index,
   HSV saturation (`P85`), spatial cloud contrast, empirical color temperature (`K`), and an
   observationally grounded `0-100` `empirical_alpenglow_score`.
4. **2 km Satellite-Derived Golden-Hour Irradiance (`fetch_nsrdb_2km_golden_hour_dni`)**: Keyless
   AWS S3 HTTP byte-range reader for `NREL NSRDB GOES CONUS v4.0.0` (`2 km`, `5-minute` GOES-18
   `dni` / `clearsky_dni`), upgrading blurred `31 km` `ERA5` coastal grid cells.
"""

from __future__ import annotations

import atexit
import datetime as dt
import hashlib
import io
import json
import math
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional runtime dependency
    Image = None  # type: ignore[assignment]

try:
    import h5py
except ImportError:  # pragma: no cover - optional runtime dependency
    h5py = None  # type: ignore[assignment]

from .corridor_solver import (
    _find_sunrise_sunset_golden_hours,
    _is_coastal_california,
    _solar_ephemeris_local,
)
from .nwp_corridor import offset_lat_lon_km

ROOT_DIR = Path(__file__).resolve().parents[2]
WEBCAM_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "webcams"
WEBCAM_FRAMES_DIR = WEBCAM_CACHE_DIR / "frames"
WEBCAM_BROWSE_DIR = WEBCAM_CACHE_DIR / "browse"
NSRDB_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "nsrdb"

USER_AGENT = "PhotoPlanner/4.2 (Atmospheric Webcam & NSRDB Verification)"

PHENOCAM_CAMERAS_URL = "https://phenocam.nau.edu/api/cameras/?limit=2000&format=json"
PHENOCAM_BASE_URL = "https://phenocam.nau.edu"
ALERTCA_CATALOG_URL = "https://cameras.alertcalifornia.org/public-camera-data/all_cameras-v3.json"
ALERTCA_FRAME_BASE = "https://cameras.alertcalifornia.org/public-camera-data"
NSRDB_S3_BASE = "https://nrel-pds-nsrdb.s3.amazonaws.com/GOES/conus/v4.0.0"

_CATALOG_LOCK = threading.Lock()
_MEM_PHENOCAM_CATALOG: tuple[float, list[dict[str, Any]]] | None = None
_MEM_ALERTCA_CATALOG: tuple[float, list[dict[str, Any]]] | None = None

_NSRDB_LOCK = threading.Lock()
_NSRDB_HANDLES: dict[int, dict[str, Any]] = {}
_NSRDB_CA_COORDS: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _close_nsrdb_handles() -> None:
    with _NSRDB_LOCK:
        for hnd in list(_NSRDB_HANDLES.values()):
            for k in ("hf_irr", "hf_clr", "sf_irr", "sf_clr"):
                obj = hnd.get(k)
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass
        _NSRDB_HANDLES.clear()


atexit.register(_close_nsrdb_handles)

_ORIENT_TO_DEG: dict[str, float] = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


def _compass_label(az_deg: float) -> str:
    dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = int(round((az_deg % 360.0) / 22.5)) % 16
    return dirs[idx]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r_km * math.asin(min(1.0, math.sqrt(a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _ang_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180.0 else d


def _orientation_to_azimuth_deg(orient: str | None, fallback_az: float = 270.0) -> tuple[float, bool]:
    if not orient:
        return fallback_az, False
    tok = str(orient).strip().upper()
    if tok in _ORIENT_TO_DEG:
        return _ORIENT_TO_DEG[tok], True
    try:
        val = float(tok)
        if 0.0 <= val <= 360.0:
            return val % 360.0, True
    except ValueError:
        pass
    return fallback_az, False


# ==============================================================================
# 1. Sky & Horizon RGB Chromaticity Extraction (PhenoCam + ALERTCalifornia)
# ==============================================================================


def analyze_webcam_sky_chromaticity(jpeg_bytes: bytes) -> dict[str, Any]:
    """
    Extracts quantitative golden-hour sky & cloud-underside chromaticity metrics from a
    mountain/coastal webcam JPEG (`y in [4%, 20%]`, `x in [4%, 96%]`):
      - Excludes top OSD timestamp banners (`y < 4%`) and golden California grass/canopy (`y >= 22%`).
      - Detects and flags StarDot monochrome / pink NIR night-mode frames (`is_ir_night_mode`).
      - Computes `rb_warmth_p90`, `rg_ratio_p90`, `saturation_p85`, `sky_contrast_std`,
        `estimated_color_temp_kelvin`, and `empirical_alpenglow_score` (`0-100`).
    """
    if Image is None or not jpeg_bytes:
        return {"available": False, "reason": "pillow_unavailable_or_empty"}

    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as im:
            im_rgb = im.convert("RGB")
            w_px, h_px = im_rgb.size
            arr = np.asarray(im_rgb, dtype=np.float32) / 255.0
    except Exception as exc:
        return {"available": False, "reason": f"decode_error:{exc}"}

    if h_px < 64 or w_px < 64:
        return {"available": False, "reason": "frame_too_small"}

    y0 = max(2, int(round(0.04 * h_px)))
    y1 = max(y0 + 8, int(round(0.20 * h_px)))
    x0 = max(2, int(round(0.04 * w_px)))
    x1 = max(x0 + 16, int(round(0.96 * w_px)))

    roi = arr[y0:y1, x0:x1, :]
    r = roi[:, :, 0]
    g = roi[:, :, 1]
    b = roi[:, :, 2]

    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    sat = (c_max - c_min) / np.maximum(c_max, 1e-4)

    sub_full = arr[::4, ::4, :]
    full_chroma_diff = float(
        np.mean(np.abs(sub_full[:, :, 0] - sub_full[:, :, 1]) + np.abs(sub_full[:, :, 1] - sub_full[:, :, 2]))
    )
    chroma_diff = float(np.mean(np.abs(r - g) + np.abs(g - b)))
    lum_mean_all = float(np.mean(lum))
    rg_all = float(np.mean((r + 0.02) / (g + 0.02)))
    bg_all = float(np.mean((b + 0.02) / (g + 0.02)))

    # StarDot / security cameras switch to monochrome NIR (greyscale across the entire image)
    # or pink-tinted NIR after civil twilight. Bright west-facing skies (`lum_mean_all > 0.65`)
    # with colored landscape (`full_chroma_diff >= 0.015`) are true RGB daylight/golden-hour frames.
    is_monochrome_ir = (chroma_diff < 0.016 and full_chroma_diff < 0.015)
    is_dark_night = lum_mean_all < 0.082 or (lum_mean_all < 0.145 and chroma_diff < 0.026)
    is_pink_nir = (rg_all > 1.38 and bg_all > 0.90 and lum_mean_all < 0.28)
    is_ir_night = bool(is_monochrome_ir or is_dark_night or is_pink_nir)

    valid_mask = (lum > 0.085) & (lum < 0.975)
    if int(np.sum(valid_mask)) < 120:
        valid_mask = lum >= 0.0

    rv = r[valid_mask]
    gv = g[valid_mask]
    bv = b[valid_mask]
    lv = lum[valid_mask]
    sv = sat[valid_mask]

    rb = (rv + 0.018) / (bv + 0.018)
    rg = (rv + 0.018) / (gv + 0.018)

    rb_mean = float(np.mean(rb))
    rb_p90 = float(np.percentile(rb, 90))
    rg_p90 = float(np.percentile(rg, 90))
    sat_mean = float(np.mean(sv))
    sat_p85 = float(np.percentile(sv, 85))
    lum_mean = float(np.mean(lv))
    lum_std = float(np.std(lv))

    rayleigh_blue_frac = float(np.mean((rb < 0.84) & (sv > 0.17)))
    warm_glow_frac = float(np.mean((rb > 1.14) & (lv > 0.16)))
    overcast_grey_frac = float(np.mean((rb >= 0.86) & (rb <= 1.13) & (sv < 0.165)))
    warm_sat = sv * np.clip((rb - 0.96) / 0.30, 0.0, 1.0)
    warm_sat_p85 = float(np.percentile(warm_sat, 85))

    # Map observed sky R/B + R/G warmth to effective correlated color temperature (2050 K .. 6400 K)
    warm_idx = max(0.0, min(1.35, (rb_p90 - 0.92) * 0.85 + max(0.0, rg_p90 - 1.02) * 0.65))
    if overcast_grey_frac > 0.55 and rb_p90 < 1.18:
        est_kelvin = int(round(5550.0 + min(750.0, overcast_grey_frac * 700.0)))
    else:
        est_kelvin = int(round(max(2050.0, min(6300.0, 5200.0 - warm_idx * 2450.0))))

    # Empirical 0-100 golden-hour sky & cloud-canvas quality score:
    #   1. Direct warm horizon / Chappuis cloud-underside warmth (`rb_p90`, `rg_p90`, `warm_glow_frac`)
    #   2. Warm-hue saturation (`warm_sat_p85`, excluding cold blue Rayleigh/storm skies where B > R)
    #   3. Crisp clear-above-inversion Rayleigh gradient & cloud texture (`rayleigh_blue_frac`, `lum_std`)
    #   4. Penalty for flat, desaturated grey overcast (`overcast_grey_frac`)
    warm_pts = min(56.0, max(0.0, (rb_p90 - 0.98) * 52.0 + max(0.0, rg_p90 - 1.02) * 34.0 + warm_glow_frac * 22.0))
    sat_pts = min(22.0, warm_sat_p85 * 54.0)
    clarity_pts = min(14.0, rayleigh_blue_frac * 12.0 + min(6.0, lum_std * 36.0))
    grey_pen = overcast_grey_frac * 18.0

    raw_emp = 16.0 + warm_pts + sat_pts + clarity_pts - grey_pen
    if is_ir_night:
        emp_score = 0
    else:
        emp_score = int(round(max(12.0, min(97.0, raw_emp))))

    if is_ir_night:
        sky_grade = "Post-Twilight NIR Camera Mode"
    elif rb_p90 >= 1.52 and sat_p85 >= 0.30:
        sky_grade = "Vivid Crimson-Amber Golden Hour"
    elif rb_p90 >= 1.24 or warm_glow_frac >= 0.22:
        sky_grade = "Warm Golden-Hour Horizon Glow"
    elif rayleigh_blue_frac >= 0.38 and overcast_grey_frac < 0.32:
        sky_grade = "Crisp Clear Sky / Above-Inversion Dome"
    elif overcast_grey_frac >= 0.48:
        sky_grade = "Cool Diffuse Overcast / Marine Lid"
    else:
        sky_grade = "Mixed Broken Cloud & Twilight Scatter"

    # Sample average RGB swatch of the brightest warm upper-horizon quartile for UI badge display
    q75_rb = float(np.percentile(rb, 75))
    swatch_mask = rb >= q75_rb
    swatch_rgb = [
        int(round(float(np.clip(np.mean(rv[swatch_mask]) * 255.0, 0.0, 255.0)))),
        int(round(float(np.clip(np.mean(gv[swatch_mask]) * 255.0, 0.0, 255.0)))),
        int(round(float(np.clip(np.mean(bv[swatch_mask]) * 255.0, 0.0, 255.0)))),
    ]

    return {
        "available": True,
        "width_px": w_px,
        "height_px": h_px,
        "sky_roi_y_pct": [4, 20],
        "is_ir_night_mode": is_ir_night,
        "rb_warmth_mean": round(rb_mean, 3),
        "rb_warmth_p90": round(rb_p90, 3),
        "rg_ratio_p90": round(rg_p90, 3),
        "saturation_mean": round(sat_mean, 3),
        "saturation_p85": round(sat_p85, 3),
        "sky_luminance_mean": round(lum_mean, 3),
        "sky_contrast_std": round(lum_std, 3),
        "rayleigh_blue_fraction": round(rayleigh_blue_frac, 3),
        "warm_glow_fraction": round(warm_glow_frac, 3),
        "overcast_grey_fraction": round(overcast_grey_frac, 3),
        "estimated_color_temp_kelvin": est_kelvin,
        "empirical_alpenglow_score": emp_score,
        "sky_grade": sky_grade,
        "horizon_swatch_rgb": swatch_rgb,
    }


# ==============================================================================
# 2. Webcam Catalogs & Frame Fetchers (PhenoCam + ALERTCalifornia)
# ==============================================================================


def _fetch_url_bytes(url: str, timeout: float = 12.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_cached_webcam_jpeg(source: str, identifier: str) -> bytes:
    """
    Safely fetches and caches a JPEG frame from an allowlisted webcam provider:
      - `source == "phenocam"`: `identifier` is `/data/archive/{site}/{YYYY}/{MM}/{file}.jpg`
      - `source == "alertca"`: `identifier` is `{camera_id}` (e.g. `mttam-cam01`)
    """
    WEBCAM_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    if source == "phenocam":
        clean_path = identifier.strip()
        if not re.match(r"^/data/archive/[A-Za-z0-9._-]+/\d{4}/\d{2}/[A-Za-z0-9._-]+\.jpg$", clean_path):
            raise ValueError("Invalid PhenoCam archive path")
        key_hash = hashlib.sha256(clean_path.encode("utf-8")).hexdigest()[:20]
        cfile = WEBCAM_FRAMES_DIR / f"pheno_{key_hash}.jpg"
        if cfile.exists() and cfile.stat().st_size > 2048:
            return cfile.read_bytes()
        raw = _fetch_url_bytes(f"{PHENOCAM_BASE_URL}{clean_path}", timeout=14.0)
        if len(raw) > 2048:
            cfile.write_bytes(raw)
        return raw

    if source == "alertca":
        cam_id = identifier.strip()
        if not re.match(r"^[A-Za-z0-9_-]{3,64}$", cam_id):
            raise ValueError("Invalid ALERTCalifornia camera ID")
        cfile = WEBCAM_FRAMES_DIR / f"alertca_{cam_id}.jpg"
        # Live camera frames refresh every 2 minutes
        if cfile.exists() and (time.time() - cfile.stat().st_mtime) < 120.0 and cfile.stat().st_size > 2048:
            return cfile.read_bytes()
        raw = _fetch_url_bytes(f"{ALERTCA_FRAME_BASE}/{cam_id}/latest-frame.jpg", timeout=12.0)
        if len(raw) > 2048:
            cfile.write_bytes(raw)
        return raw

    raise ValueError(f"Unsupported webcam source: {source}")


def _get_phenocam_catalog() -> list[dict[str, Any]]:
    global _MEM_PHENOCAM_CATALOG
    now_s = time.time()
    with _CATALOG_LOCK:
        if _MEM_PHENOCAM_CATALOG and (now_s - _MEM_PHENOCAM_CATALOG[0]) < 3600.0:
            return _MEM_PHENOCAM_CATALOG[1]

    WEBCAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfile = WEBCAM_CACHE_DIR / "phenocam_catalog_v1.json"
    if cfile.exists() and (now_s - cfile.stat().st_mtime) < 86400.0:
        try:
            items = json.loads(cfile.read_text(encoding="utf-8"))
            with _CATALOG_LOCK:
                _MEM_PHENOCAM_CATALOG = (now_s, items)
            return items
        except Exception:
            pass

    try:
        raw = json.loads(_fetch_url_bytes(PHENOCAM_CAMERAS_URL, timeout=14.0).decode("utf-8"))
        results = raw.get("results") if isinstance(raw, dict) else raw
        items = list(results or [])
        cfile.write_text(json.dumps(items), encoding="utf-8")
        with _CATALOG_LOCK:
            _MEM_PHENOCAM_CATALOG = (now_s, items)
        return items
    except Exception:
        if cfile.exists():
            try:
                return json.loads(cfile.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []


def _get_alertca_catalog() -> list[dict[str, Any]]:
    global _MEM_ALERTCA_CATALOG
    now_s = time.time()
    with _CATALOG_LOCK:
        if _MEM_ALERTCA_CATALOG and (now_s - _MEM_ALERTCA_CATALOG[0]) < 1800.0:
            return _MEM_ALERTCA_CATALOG[1]

    WEBCAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfile = WEBCAM_CACHE_DIR / "alertca_catalog_v3.json"
    if cfile.exists() and (now_s - cfile.stat().st_mtime) < 21600.0:
        try:
            feats = json.loads(cfile.read_text(encoding="utf-8"))
            with _CATALOG_LOCK:
                _MEM_ALERTCA_CATALOG = (now_s, feats)
            return feats
        except Exception:
            pass

    try:
        raw = json.loads(_fetch_url_bytes(ALERTCA_CATALOG_URL, timeout=14.0).decode("utf-8"))
        feats = list(raw.get("features") or [])
        cfile.write_text(json.dumps(feats), encoding="utf-8")
        with _CATALOG_LOCK:
            _MEM_ALERTCA_CATALOG = (now_s, feats)
        return feats
    except Exception:
        if cfile.exists():
            try:
                return json.loads(cfile.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []


def _list_phenocam_daily_frames(site: str, req_date: dt.date, is_today: bool = False) -> list[dict[str, Any]]:
    """
    Lists daily RGB JPEG frames from `https://phenocam.nau.edu/webcam/browse/{site}/{YYYY}/{MM}/{DD}/`.
    Note: PhenoCam filename timestamps (`_YYYY_MM_DD_HHMMSS.jpg`) are recorded in Local Standard
    Time (`UTC-8` PST year-round in Pacific timezones, with no DST shift).
    """
    WEBCAM_BROWSE_DIR.mkdir(parents=True, exist_ok=True)
    cfile = WEBCAM_BROWSE_DIR / f"{site}_{req_date.isoformat()}.json"
    now_s = time.time()
    if cfile.exists():
        ttl = 900.0 if is_today else 365 * 86400.0
        if (now_s - cfile.stat().st_mtime) < ttl:
            try:
                return json.loads(cfile.read_text(encoding="utf-8"))
            except Exception:
                pass

    url = (
        f"{PHENOCAM_BASE_URL}/webcam/browse/{urllib.parse.quote(site)}/"
        f"{req_date.year:04d}/{req_date.month:02d}/{req_date.day:02d}/"
    )
    try:
        html = _fetch_url_bytes(url, timeout=12.0).decode("utf-8", errors="ignore")
    except Exception:
        return []

    matches = sorted(set(re.findall(r"(/data/archive/[A-Za-z0-9._-]+/\d{4}/\d{2}/[A-Za-z0-9._-]+\.jpg)", html)))
    tz_pac = ZoneInfo("America/Los_Angeles")
    frames: list[dict[str, Any]] = []
    seen_ts: set[str] = set()
    for rel_path in matches:
        fname = rel_path.rsplit("/", 1)[-1]
        if "_IR_" in fname:
            continue
        m = re.search(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})\.jpg$", fname)
        if not m:
            continue
        yr, mo, dy, hh, mm, ss = (int(x) for x in m.groups())
        ts_key = f"{yr:04d}-{mo:02d}-{dy:02d}T{hh:02d}:{mm:02d}:{ss:02d}"
        if ts_key in seen_ts:
            continue
        seen_ts.add(ts_key)
        # PhenoCam California timestamps are UTC-8 Standard Time year-round
        dt_lst = dt.datetime(yr, mo, dy, hh, mm, ss, tzinfo=dt.timezone(dt.timedelta(hours=-8)))
        dt_utc = dt_lst.astimezone(dt.timezone.utc)
        dt_civil = dt_utc.astimezone(tz_pac)
        frames.append(
            {
                "archive_path": rel_path,
                "url": f"{PHENOCAM_BASE_URL}{rel_path}",
                "proxy_url": f"/api/webcam-image?source=phenocam&path={urllib.parse.quote(rel_path, safe='')}",
                "timestamp_utc": dt_utc.isoformat(),
                "timestamp_local": dt_civil.strftime("%H:%M"),
                "minutes_local": dt_civil.hour * 60 + dt_civil.minute,
                "epoch_s": int(dt_utc.timestamp()),
            }
        )

    frames.sort(key=lambda x: x["epoch_s"])
    if frames or not is_today:
        try:
            cfile.write_text(json.dumps(frames), encoding="utf-8")
        except Exception:
            pass
    return frames


def _match_phenocam_archive_camera(
    lat: float,
    lon: float,
    elev_m: float,
    req_date: dt.date,
    target_utc: dt.datetime,
    sun_az_deg: float,
    is_today: bool,
    analyze_pixels: bool = True,
) -> dict[str, Any] | None:
    catalog = _get_phenocam_catalog()
    if not catalog:
        return None

    date_str = req_date.isoformat()
    is_sierra = elev_m >= 1350.0 and not _is_coastal_california(lat, lon)
    candidates: list[tuple[float, dict[str, Any], float]] = []

    _NON_SKY_SITES = {"sequoia1", "sequoia2"}
    for cam in catalog:
        site = str(cam.get("Sitename") or "")
        if not site or "DP1.00042" in site or "DP1.20002" in site or site.lower() in _NON_SKY_SITES:
            continue
        clat = cam.get("Lat")
        clon = cam.get("Lon")
        if clat is None or clon is None:
            continue
        clat_f, clon_f = float(clat), float(clon)
        dist_km = _haversine_km(lat, lon, clat_f, clon_f)
        if dist_km > 135.0:
            continue

        smeta = cam.get("sitemetadata") or {}
        desc = str(smeta.get("camera_description") or "").lower()
        if any(k in desc for k in ("understory", "soil", "litter", "stream", "creek", "aquatic", "bole", "trunk")):
            continue

        d_first = str(cam.get("date_first") or "1999-01-01")[:10]
        d_last = str(cam.get("date_last") or "2099-12-31")[:10]
        if date_str < d_first:
            continue
        if (not is_today) and date_str > d_last:
            continue

        celev_m = float(cam.get("Elev") or 0.0)
        elev_diff_m = abs(celev_m - elev_m)
        orient_raw = smeta.get("camera_orientation")
        az_deg, has_orient = _orientation_to_azimuth_deg(orient_raw, fallback_az=sun_az_deg)
        az_pen_km = (_ang_diff_deg(az_deg, sun_az_deg) / 180.0) * 18.0 if has_orient else 8.0
        tower_bonus_km = -18.0 if "DP1.00033" in site else 0.0

        # For High Sierra targets, strongly prefer high-elevation Sierra towers (> 1,000 m ASL)
        regime_pen_km = 0.0
        if is_sierra and celev_m < 900.0:
            regime_pen_km = 65.0
        elif (not is_sierra) and celev_m > 1600.0:
            regime_pen_km = 45.0

        rank_cost = dist_km + elev_diff_m * 0.012 + az_pen_km + regime_pen_km + tower_bonus_km
        candidates.append((rank_cost, cam, dist_km))

    candidates.sort(key=lambda t: t[0])
    target_epoch = int(target_utc.timestamp())

    for _, cam, dist_km in candidates[:6]:
        site = str(cam["Sitename"])
        frames = _list_phenocam_daily_frames(site, req_date, is_today=is_today)
        if not frames:
            continue

        # Select golden-hour window [-110m, +25m] around target_utc
        win_frames = [
            f for f in frames
            if -6600 <= (f["epoch_s"] - target_epoch) <= 1500
        ]
        if not win_frames:
            # Fallback to closest 4 frames to target_utc within 3 hours
            win_frames = sorted(frames, key=lambda f: abs(f["epoch_s"] - target_epoch))[:4]
            win_frames.sort(key=lambda f: f["epoch_s"])
            if not win_frames or abs(win_frames[0]["epoch_s"] - target_epoch) > 12600:
                continue

        # Subsample up to 5 timeline frames spanning golden hour
        if len(win_frames) > 5:
            idxs = np.linspace(0, len(win_frames) - 1, 5).round().astype(int)
            timeline = [win_frames[int(i)] for i in sorted(set(idxs))]
        else:
            timeline = list(win_frames)

        # Pick primary frame closest to -18 min before geometric sunset (sun_el ~ +1.5..+4 deg) that isn't NIR night mode
        ideal_epoch = target_epoch - 1080
        ordered_for_peak = sorted(
            win_frames,
            key=lambda f: abs(f["epoch_s"] - ideal_epoch) + (1200 if f["epoch_s"] > target_epoch + 180 else 0),
        )

        primary_frame = ordered_for_peak[0]
        primary_chroma: dict[str, Any] = {"available": False}

        if analyze_pixels and Image is not None:
            for cand_f in ordered_for_peak[:5]:
                try:
                    jpg_b = fetch_cached_webcam_jpeg("phenocam", cand_f["archive_path"])
                    ch = analyze_webcam_sky_chromaticity(jpg_b)
                    if ch.get("available") and not ch.get("is_ir_night_mode"):
                        primary_frame = cand_f
                        primary_chroma = ch
                        break
                    if not primary_chroma.get("available"):
                        primary_chroma = ch
                except Exception:
                    continue

        if primary_frame not in timeline:
            timeline.append(primary_frame)
            timeline.sort(key=lambda f: f["epoch_s"])

        for f in timeline:
            delta_min = int(round((f["epoch_s"] - target_epoch) / 60.0))
            f["offset_vs_sunset_min"] = delta_min
            f["chip_label"] = (
                "Peak" if abs(delta_min) <= 4 else (f"{delta_min:+d}m")
            )
            f["is_primary"] = f["archive_path"] == primary_frame["archive_path"]

        clat_f = float(cam["Lat"])
        clon_f = float(cam["Lon"])
        celev_m = float(cam.get("Elev") or 0.0)
        smeta = cam.get("sitemetadata") or {}
        orient_raw = smeta.get("camera_orientation")
        az_deg, has_orient = _orientation_to_azimuth_deg(orient_raw, fallback_az=sun_az_deg)
        desc_title = str(smeta.get("site_acknowledgements") or smeta.get("camera_description") or site).split(".")[0].strip()
        if len(desc_title) > 54 or not desc_title:
            desc_title = site

        d_east_m = (clon_f - lon) * 111412.84 * math.cos(math.radians(lat))
        d_north_m = (clat_f - lat) * 111132.92
        in_patch = abs(d_east_m) <= 5600.0 and abs(d_north_m) <= 5600.0

        return {
            "source": "phenocam",
            "provider_label": "PhenoCam / NEON Scientific Archive",
            "camera_id": site,
            "name": f"{site} ({desc_title})" if site not in desc_title else desc_title,
            "short_name": site,
            "is_live": is_today,
            "is_historical_archive": not is_today,
            "lat": round(clat_f, 5),
            "lon": round(clon_f, 5),
            "elev_m": round(celev_m, 1),
            "elev_ft": int(round(celev_m * 3.28084)),
            "distance_km": round(dist_km, 1),
            "distance_miles": round(dist_km * 0.621371, 1),
            "bearing_from_target_deg": round(_bearing_deg(lat, lon, clat_f, clon_f), 1),
            "azimuth_deg": round(az_deg, 1),
            "azimuth_cardinal": _compass_label(az_deg),
            "has_calibrated_azimuth": has_orient,
            "tilt_deg": -1.2,
            "hfov_deg": 62.0,
            "is_inside_terrain_patch": in_patch,
            "snap_mode": "exact_webcam_gps" if in_patch else "regional_azimuth_fov",
            "primary_frame": primary_frame,
            "timeline_frames": timeline,
            "sky_chromaticity": primary_chroma,
        }

    return None


def _match_alertca_live_cameras(
    lat: float,
    lon: float,
    elev_m: float,
    sun_az_deg: float,
    analyze_pixels: bool = True,
) -> list[dict[str, Any]]:
    feats = _get_alertca_catalog()
    if not feats:
        return []

    now_epoch = int(time.time())
    tz_pac = ZoneInfo("America/Los_Angeles")
    scored: list[tuple[float, dict[str, Any]]] = []

    for ft in feats:
        geom = ft.get("geometry") or {}
        coords = geom.get("coordinates") or []
        props = ft.get("properties") or {}
        if len(coords) < 2:
            continue
        clon_f, clat_f = float(coords[0]), float(coords[1])
        celev_m = float(coords[2]) if len(coords) >= 3 and coords[2] is not None else 0.0
        cam_id = str(props.get("id") or "").strip()
        if not cam_id:
            continue

        dist_km = _haversine_km(lat, lon, clat_f, clon_f)
        if dist_km > 52.0:
            continue

        last_ts = int(props.get("last_frame_ts") or 0)
        if last_ts > 0 and (now_epoch - last_ts) > 86400 * 2:
            continue

        az_raw = props.get("az_current")
        fov_center = props.get("fov_center")
        brg_to_target = _bearing_deg(clat_f, clon_f, lat, lon)
        has_az = False
        az_deg = brg_to_target
        if isinstance(az_raw, (int, float, str)):
            try:
                az_deg = float(az_raw) % 360.0
                has_az = True
            except ValueError:
                pass
        elif isinstance(fov_center, (list, tuple)) and len(fov_center) >= 2:
            try:
                az_deg = _bearing_deg(clat_f, clon_f, float(fov_center[1]), float(fov_center[0]))
                has_az = True
            except (TypeError, ValueError):
                pass

        tilt_raw = props.get("tilt_current")
        tilt_val = float(tilt_raw) if isinstance(tilt_raw, (int, float)) else -1.8
        fov_raw = props.get("fov")
        fov_val = float(fov_raw) if isinstance(fov_raw, (int, float)) and float(fov_raw) > 5.0 else 62.0

        # Reward cameras either inside the 3D terrain patch, looking toward the landmark, or looking along sunset azimuth
        align_target = _ang_diff_deg(az_deg, brg_to_target)
        align_sun = _ang_diff_deg(az_deg, sun_az_deg)
        best_align = min(align_target, align_sun)
        cost = dist_km * 1.15 + (best_align / 180.0) * 10.0 + abs(celev_m - elev_m) * 0.004
        scored.append((cost, {
            "cam_id": cam_id,
            "name": str(props.get("name") or cam_id),
            "lat": clat_f,
            "lon": clon_f,
            "elev_m": celev_m,
            "dist_km": dist_km,
            "az_deg": az_deg,
            "has_az": has_az,
            "tilt_deg": tilt_val,
            "hfov_deg": fov_val,
            "last_ts": last_ts,
        }))

    scored.sort(key=lambda t: t[0])
    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for idx, (_, c) in enumerate(scored[:4]):
        if len(out) >= 2:
            break
        base_prefix = c["cam_id"].split("-")[0]
        if base_prefix in seen_names and len(scored) > 2:
            continue
        seen_names.add(base_prefix)

        ts_epoch = c["last_ts"] if c["last_ts"] > 0 else now_epoch
        dt_utc = dt.datetime.fromtimestamp(ts_epoch, tz=dt.timezone.utc)
        dt_loc = dt_utc.astimezone(tz_pac)
        proxy_url = f"/api/webcam-image?source=alertca&id={urllib.parse.quote(c['cam_id'], safe='')}"

        chroma: dict[str, Any] = {"available": False}
        if analyze_pixels and idx == 0 and Image is not None:
            try:
                jpg_b = fetch_cached_webcam_jpeg("alertca", c["cam_id"])
                chroma = analyze_webcam_sky_chromaticity(jpg_b)
            except Exception:
                chroma = {"available": False}

        d_east_m = (c["lon"] - lon) * 111412.84 * math.cos(math.radians(lat))
        d_north_m = (c["lat"] - lat) * 111132.92
        in_patch = abs(d_east_m) <= 5600.0 and abs(d_north_m) <= 5600.0

        frame_obj = {
            "archive_path": c["cam_id"],
            "url": f"{ALERTCA_FRAME_BASE}/{c['cam_id']}/latest-frame.jpg",
            "proxy_url": proxy_url,
            "timestamp_utc": dt_utc.isoformat(),
            "timestamp_local": dt_loc.strftime("%H:%M"),
            "minutes_local": dt_loc.hour * 60 + dt_loc.minute,
            "epoch_s": ts_epoch,
            "offset_vs_sunset_min": 0,
            "chip_label": "Live HD",
            "is_primary": True,
        }

        out.append(
            {
                "source": "alertca",
                "provider_label": "ALERTCalifornia Live Mountain HD",
                "camera_id": c["cam_id"],
                "name": f"{c['name']} ({c['cam_id']})",
                "short_name": c["name"],
                "is_live": True,
                "is_historical_archive": False,
                "lat": round(c["lat"], 5),
                "lon": round(c["lon"], 5),
                "elev_m": round(c["elev_m"], 1),
                "elev_ft": int(round(c["elev_m"] * 3.28084)),
                "distance_km": round(c["dist_km"], 1),
                "distance_miles": round(c["dist_km"] * 0.621371, 1),
                "bearing_from_target_deg": round(_bearing_deg(lat, lon, c["lat"], c["lon"]), 1),
                "azimuth_deg": round(c["az_deg"], 1),
                "azimuth_cardinal": _compass_label(c["az_deg"]),
                "has_calibrated_azimuth": c["has_az"],
                "tilt_deg": round(max(-14.0, min(8.0, c["tilt_deg"])), 1),
                "hfov_deg": round(max(24.0, min(92.0, c["hfov_deg"])), 1),
                "is_inside_terrain_patch": in_patch,
                "snap_mode": "exact_webcam_gps" if in_patch else "regional_azimuth_fov",
                "primary_frame": frame_obj,
                "timeline_frames": [frame_obj],
                "sky_chromaticity": chroma,
            }
        )

    return out


def fetch_webcam_ground_truth(
    lat: float,
    lon: float,
    date_str: str,
    phase: str = "sunset",
    elev_m: float = 220.0,
    sun_az_deg: float | None = None,
    analyze_pixels: bool = True,
) -> dict[str, Any]:
    """
    Matches real timestamped mountain/coastal webcam imagery to `(lat, lon, date_str, phase)`:
      - For historical dates (`date < today_pacific`): returns archived golden-hour `PhenoCam` /
        `NEON` imagery (`-75m .. +20m` timeline around sunrise/sunset) + RGB sky chromaticity.
      - For live/today dates (`date >= today_pacific`): returns live `1920x1080` `ALERTCalifornia`
        summit/ridge cameras (with live compass azimuth, pitch, and FOV) PLUS today's `PhenoCam`
        scientific reference when available.
    """
    tz_pac = ZoneInfo("America/Los_Angeles")
    today_pac = dt.datetime.now(tz_pac).date()
    req_date = dt.date.fromisoformat(date_str)
    is_past = req_date < today_pac
    is_today = req_date == today_pac
    is_future = req_date > today_pac

    noon_loc = dt.datetime(req_date.year, req_date.month, req_date.day, 12, 0, tzinfo=tz_pac)
    tz_off_hr = noon_loc.utcoffset().total_seconds() / 3600.0  # type: ignore[union-attr]
    sr_info, ss_info = _find_sunrise_sunset_golden_hours(lat, lon, req_date, tz_off_hr)
    is_sunset = phase.lower() != "sunrise"
    target_mins = int(ss_info[0] if is_sunset else sr_info[0])
    calc_az, calc_el = _solar_ephemeris_local(lat, lon, req_date, float(target_mins), tz_off_hr)
    eff_sun_az = float(sun_az_deg) if sun_az_deg is not None else float(calc_az)

    target_loc = dt.datetime(
        req_date.year, req_date.month, req_date.day,
        max(0, min(23, target_mins // 60)), target_mins % 60, tzinfo=tz_pac
    )
    target_utc = target_loc.astimezone(dt.timezone.utc)

    cameras: list[dict[str, Any]] = []
    if not is_past:
        # Live ridge cameras from ALERTCalifornia (current atmospheric conditions on mountain!)
        alert_cams = _match_alertca_live_cameras(
            lat, lon, elev_m, eff_sun_az, analyze_pixels=analyze_pixels
        )
        cameras.extend(alert_cams)

    pheno_date = today_pac if is_future else req_date
    pheno_target_utc = (
        dt.datetime.now(dt.timezone.utc)
        if is_future
        else target_utc
    )
    pheno_cam = _match_phenocam_archive_camera(
        lat,
        lon,
        elev_m,
        pheno_date,
        pheno_target_utc,
        eff_sun_az,
        is_today=(not is_past),
        analyze_pixels=analyze_pixels and (is_past or not cameras),
    )
    if pheno_cam is not None:
        cameras.append(pheno_cam)

    primary = cameras[0] if cameras else None
    return {
        "available": primary is not None,
        "date": date_str,
        "phase": "sunset" if is_sunset else "sunrise",
        "temporal_mode": "historical_archive" if is_past else ("live_today" if is_today else "live_nowcast_for_future_date"),
        "target_local_time": target_loc.strftime("%H:%M"),
        "target_utc": target_utc.isoformat(),
        "sun_azimuth_deg": round(eff_sun_az, 1),
        "sun_elevation_deg": round(float(calc_el), 2),
        "primary_camera": primary,
        "cameras": cameras,
    }


# ==============================================================================
# 3. 2 km GOES-18 NREL NSRDB CONUS v4.0.0 Keyless S3 Byte-Range Irradiance
# ==============================================================================


class _S3RangeFile(io.RawIOBase):
    """Seekable `1 MB` block-cached HTTP `Range:` stream for NREL NSRDB HDF5 files on AWS S3."""

    def __init__(self, url: str, size: int, block_size: int = 1048576) -> None:
        self.url = url
        self.size = size
        self.pos = 0
        self.bs = block_size
        self.cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, b: Any) -> int:
        if self.pos >= self.size:
            return 0
        n = min(len(b), self.size - self.pos)
        out = bytearray()
        cur, end = self.pos, self.pos + n
        while cur < end:
            blk = cur // self.bs
            if blk not in self.cache:
                b0 = blk * self.bs
                b1 = min(self.size - 1, b0 + self.bs - 1)
                req = urllib.request.Request(
                    self.url,
                    headers={"Range": f"bytes={b0}-{b1}", "User-Agent": USER_AGENT},
                )
                with urllib.request.urlopen(req, timeout=16.0) as resp:
                    self.cache[blk] = resp.read()
                if len(self.cache) > 48:
                    oldest = next(iter(self.cache))
                    self.cache.pop(oldest, None)
            data = self.cache[blk]
            off = cur - blk * self.bs
            take = min(end - cur, len(data) - off)
            out.extend(data[off : off + take])
            cur += take
        b[: len(out)] = out
        self.pos += len(out)
        return len(out)


def _head_content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        return int(resp.headers["Content-Length"])


def _get_nsrdb_handles_and_ca_coords(year: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray] | None:
    if h5py is None:
        return None
    with _NSRDB_LOCK:
        if year not in _NSRDB_HANDLES:
            irr_url = f"{NSRDB_S3_BASE}/nsrdb_conus_irradiance_{year}.h5"
            clr_url = f"{NSRDB_S3_BASE}/nsrdb_conus_clearsky_{year}.h5"
            irr_sz = _head_content_length(irr_url)
            clr_sz = _head_content_length(clr_url)
            sf_irr = _S3RangeFile(irr_url, irr_sz, block_size=2097152)
            sf_clr = _S3RangeFile(clr_url, clr_sz, block_size=2097152)
            hf_irr = h5py.File(sf_irr, "r")
            hf_clr = h5py.File(sf_clr, "r")
            _NSRDB_HANDLES[year] = {
                "irr_url": irr_url,
                "clr_url": clr_url,
                "sf_irr": sf_irr,
                "sf_clr": sf_clr,
                "hf_irr": hf_irr,
                "hf_clr": hf_clr,
                "dni_scale": float(hf_irr["dni"].attrs.get("scale_factor", 1.0)),
                "cdni_scale": float(hf_clr["clearsky_dni"].attrs.get("scale_factor", 1.0)),
            }

        hnd = _NSRDB_HANDLES[year]
        if year not in _NSRDB_CA_COORDS:
            NSRDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            npz_path = NSRDB_CACHE_DIR / f"ca_coords_{year}.npz"
            if npz_path.exists():
                z = np.load(npz_path)
                _NSRDB_CA_COORDS[year] = (z["gids"], z["lats"], z["lons"])
            else:
                coords = hnd["hf_irr"]["coordinates"][:]
                mask = (
                    (coords[:, 0] >= 32.0)
                    & (coords[:, 0] <= 42.5)
                    & (coords[:, 1] >= -125.5)
                    & (coords[:, 1] <= -114.0)
                )
                gids = np.nonzero(mask)[0].astype(np.int32)
                lats = coords[gids, 0].astype(np.float32)
                lons = coords[gids, 1].astype(np.float32)
                np.savez_compressed(npz_path, gids=gids, lats=lats, lons=lons)
                _NSRDB_CA_COORDS[year] = (gids, lats, lons)

        gids, lats, lons = _NSRDB_CA_COORDS[year]
        return hnd, gids, lats, lons


def _read_nsrdb_uint16_series(url: str, ds: Any, i0: int, i1: int, gid: int) -> np.ndarray:
    """
    Reads `ds[i0:i1+1, gid]` via `libhdf5` `O(log N)` chunk B-tree indexing over `2 MB` HTTP `Range:` blocks.
    """
    _ = url
    return np.asarray(ds[i0 : i1 + 1, gid], dtype=np.uint16)


def fetch_nsrdb_2km_golden_hour_dni(
    lat: float,
    lon: float,
    elev_m: float,
    date_str: str,
    sun_az_deg: float | None = None,
) -> dict[str, Any] | None:
    """
    Fetches `2 km`, `5-minute` GOES-18 satellite-derived Direct Normal Irradiance (`dni` vs
    `clearsky_dni`) from `NREL NSRDB CONUS v4.0.0` for the 2-hour pre-sunset golden-hour window.
    For coastal sunsets, samples both the `0 km` headland/summit `2 km` cell and an `8 km`
    offshore slant-corridor cell along `sun_az_deg` so marine stratus decks and coastal gaps
    are resolved without `31 km` `ERA5` land/ocean blurring.
    """
    if h5py is None:
        return None
    req_d = dt.date.fromisoformat(date_str)
    year = req_d.year
    if year < 2022 or year > 2025:
        return None

    NSRDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfile = NSRDB_CACHE_DIR / f"nsrdb_gh_v42_{lat:.3f}_{lon:.3f}_{date_str}.json"
    if cfile.exists():
        try:
            return json.loads(cfile.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        res_h = _get_nsrdb_handles_and_ca_coords(year)
        if res_h is None:
            return None
        hnd, gids, lats, lons = res_h

        tz_pac = ZoneInfo("America/Los_Angeles")
        noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz_pac)
        tz_off_hr = noon_loc.utcoffset().total_seconds() / 3600.0  # type: ignore[union-attr]
        _, ss_info = _find_sunrise_sunset_golden_hours(lat, lon, req_d, tz_off_hr)
        ss_mins = int(ss_info[0])
        calc_az, _ = _solar_ephemeris_local(lat, lon, req_d, float(ss_mins), tz_off_hr)
        eff_az = float(sun_az_deg) if sun_az_deg is not None else float(calc_az)

        ss_loc = dt.datetime(req_d.year, req_d.month, req_d.day, ss_mins // 60, ss_mins % 60, tzinfo=tz_pac)
        ss_utc = ss_loc.astimezone(dt.timezone.utc).replace(tzinfo=None)
        jan1_utc = dt.datetime(year, 1, 1, 0, 0, 0)

        # 5-minute steps spanning [sunset - 115 min, sunset - 15 min] (sun el ~ 3 deg .. 22 deg)
        i1 = int((ss_utc - dt.timedelta(minutes=15) - jan1_utc).total_seconds() // 300)
        i0 = max(0, i1 - 20)
        if i1 <= i0:
            return None

        cos_lat = math.cos(math.radians(lat))
        idx0 = int(np.argmin((lats - lat) ** 2 + ((lons - lon) * cos_lat) ** 2))
        gid0 = int(gids[idx0])
        grid_lat0, grid_lon0 = float(lats[idx0]), float(lons[idx0])

        is_coastal = _is_coastal_california(lat, lon)
        az_rad = math.radians(eff_az)
        off_lat, off_lon = offset_lat_lon_km(lat, lon, 8.0 * math.sin(az_rad), 8.0 * math.cos(az_rad))
        idx_cor = int(np.argmin((lats - off_lat) ** 2 + ((lons - off_lon) * cos_lat) ** 2))
        gid_cor = int(gids[idx_cor])

        def _eval_gid(gid_val: int) -> tuple[float, float, float]:
            with _NSRDB_LOCK:
                dv = _read_nsrdb_uint16_series(hnd["irr_url"], hnd["hf_irr"]["dni"], i0, i1, gid_val).astype(np.float64) / hnd["dni_scale"]
                cv = _read_nsrdb_uint16_series(hnd["clr_url"], hnd["hf_clr"]["clearsky_dni"], i0, i1, gid_val).astype(np.float64) / hnd["cdni_scale"]
            mask = cv >= 75.0
            if not np.any(mask):
                return 0.0, 0.0, 0.0
            # Weight later golden-hour steps (closer to low-angle sunset) higher: 0.55 -> 1.45
            w = np.linspace(0.55, 1.45, int(np.sum(mask)))
            ratios = np.clip((dv[mask] / cv[mask]) * 100.0, 0.0, 100.0)
            return (
                float(np.average(ratios, weights=w)),
                float(np.mean(dv[mask])),
                float(np.mean(cv[mask])),
            )

        pct_0km, dni_m0, cdni_m0 = _eval_gid(gid0)
        if is_coastal and gid_cor != gid0:
            pct_8km, _, _ = _eval_gid(gid_cor)
            # For coastal summits above marine fog (elev_m >= 480 m), overhead/summit sun stays lit even when
            # 8 km offshore sea-level pixel is capped under low stratus; for low bluffs, offshore stratus blocks low sun!
            w_offshore = 0.18 if elev_m >= 480.0 else 0.42
            golden_pct = round((1.0 - w_offshore) * pct_0km + w_offshore * pct_8km, 1)
        else:
            pct_8km = pct_0km
            golden_pct = round(pct_0km, 1)

        out = {
            "available": True,
            "source": f"NREL NSRDB GOES-18 CONUS v4.0.0 (2 km, 5-min, {year})",
            "date": date_str,
            "gid_0km": gid0,
            "grid_lat": round(grid_lat0, 4),
            "grid_lon": round(grid_lon0, 4),
            "onshore_0km_transmittance_pct": round(pct_0km, 1),
            "corridor_8km_transmittance_pct": round(pct_8km, 1),
            "golden_hour_rel_transmittance_pct": max(0.0, min(99.0, golden_pct)),
            "mean_dni_wm2": round(dni_m0, 1),
            "mean_clearsky_dni_wm2": round(cdni_m0, 1),
        }
        try:
            cfile.write_text(json.dumps(out), encoding="utf-8")
        except Exception:
            pass
        return out
    except Exception:
        return None
