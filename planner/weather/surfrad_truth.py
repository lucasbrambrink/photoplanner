"""
NOAA SURFRAD 1-Minute Ground Pyrheliometer Ground-Truth Reader (`planner/weather/surfrad_truth.py`).

Provides **100% independent, research-grade ground measurements** of golden-hour Direct Normal
Irradiance (`direct_n`, W/m²) from NOAA Global Monitoring Laboratory's Surface Radiation Budget
Network (`https://gml.noaa.gov/aftp/data/radiation/surfrad/`).

Why SURFRAD is the gold standard for PhotoPlanner backtesting:
- Unlike ERA5 (31 km reanalysis) or NSRDB (2 km GOES-derived irradiance), SURFRAD uses physically
  calibrated Normal Incidence Pyrheliometers (NIPs) tracking the solar disk every **1 minute**.
- Completely independent of GOES satellites, HRRR, GFS, and ECMWF — zero shared model/retrieval error.
- Captures BOTH **sunrise** and **sunset** golden-hour windows (`el in [1.5°, 12.0°]`), including:
  1. Mean relative direct transmittance (`direct_n / clear_sky_dni`, `0..100%`) weighted toward
     low golden-hour angles (`1.5°..8°`).
  2. Binary WMO direct-sunshine breakout (`golden_hour_sun_hit`: whether sustained golden-hour
     direct beam exceeded `>= 120 W/m²` or `>= 35%` of clear-sky DNI) for calibrating hit/bust odds.
  3. Exact diffuse-to-direct ratio (`diffuse / (direct_n * sin(el) + diffuse)`) and co-located
     1-minute surface temperature, RH, wind speed, and station pressure.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
import urllib.request

import numpy as np

from .nwp_corridor import local_tz_for_lon

ROOT_DIR = Path(__file__).resolve().parents[2]
SURFRAD_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "surfrad"
SURFRAD_BASE_URL = "https://gml.noaa.gov/aftp/data/radiation/surfrad"
USER_AGENT = "PhotoPlanner/4.5 (NOAA SURFRAD Independent Ground Truth)"
_SURFRAD_DL_LOCK = threading.Lock()

# Official NOAA SURFRAD network stations (western/mountain stations first)
SURFRAD_STATIONS: dict[str, dict[str, Any]] = {
    "tbl": {
        "code": "tbl",
        "dir": "Boulder_CO",
        "name": "Table Mountain, Boulder CO",
        "region": "rockies",
        "subregion": "front_range",
        "lat": 40.1250,
        "lon": -105.2368,
        "elev_m": 1689.0,
    },
    "dra": {
        "code": "dra",
        "dir": "Desert_Rock_NV",
        "name": "Desert Rock, NV (Mojave)",
        "region": "southwest",
        "subregion": "mojave_desert",
        "lat": 36.6237,
        "lon": -116.0195,
        "elev_m": 1007.0,
    },
    "fpk": {
        "code": "fpk",
        "dir": "Fort_Peck_MT",
        "name": "Fort Peck, MT (Northern Plains)",
        "region": "northern_rockies",
        "subregion": "montana_plains",
        "lat": 48.3078,
        "lon": -105.1017,
        "elev_m": 634.0,
    },
    "sxf": {
        "code": "sxf",
        "dir": "Sioux_Falls_SD",
        "name": "Sioux Falls, SD",
        "region": "plains",
        "subregion": "dakota_plains",
        "lat": 43.7340,
        "lon": -96.6233,
        "elev_m": 473.0,
    },
    "bon": {
        "code": "bon",
        "dir": "Bondville_IL",
        "name": "Bondville, IL",
        "region": "midwest",
        "subregion": "midwest_plains",
        "lat": 40.0519,
        "lon": -88.3731,
        "elev_m": 213.0,
    },
    "gwn": {
        "code": "gwn",
        "dir": "Goodwin_Creek_MS",
        "name": "Goodwin Creek, MS",
        "region": "southeast",
        "subregion": "midsouth",
        "lat": 34.2547,
        "lon": -89.8729,
        "elev_m": 98.0,
    },
    "psu": {
        "code": "psu",
        "dir": "Penn_State_PA",
        "name": "Penn State, PA (Appalachians)",
        "region": "east",
        "subregion": "appalachian_ridge",
        "lat": 40.7201,
        "lon": -77.9309,
        "elev_m": 376.0,
    },
}


def clear_sky_dni_kasten_ineichen(
    solar_el_deg: float, elev_m: float, day_of_year: int, linke_turbidity: float = 2.6
) -> float:
    """
    Physical clear-sky Direct Normal Irradiance (`W/m²`) using Kasten-Young (1989) airmass
    with barometric altitude pressure scaling and Ineichen-Perez (2002) transmittance.
    """
    if solar_el_deg <= 0.2:
        return 0.0
    zen_deg = 90.0 - solar_el_deg
    # Extraterrestrial normal irradiance corrected for Earth-Sun orbital eccentricity
    b = 2.0 * math.pi * (day_of_year - 1) / 365.0
    e0 = (
        1.000110
        + 0.034221 * math.cos(b)
        + 0.001280 * math.sin(b)
        + 0.000719 * math.cos(2.0 * b)
        + 0.000077 * math.sin(2.0 * b)
    )
    i0 = 1361.0 * e0

    # Kasten-Young (1989) relative optical airmass
    sin_el = math.sin(math.radians(solar_el_deg))
    am = 1.0 / (sin_el + 0.50572 * ((solar_el_deg + 6.07995) ** -1.6364))
    p_ratio = math.exp(-max(0.0, elev_m) / 8434.5)
    am_p = am * p_ratio

    # Ineichen-Perez (2002) broadband direct beam transmittance
    b_coef = 0.664 + 0.163 / max(0.25, p_ratio)
    tau_beam = math.exp(-0.09 * am_p * (max(1.8, linke_turbidity) - 1.0))
    return max(0.0, b_coef * i0 * tau_beam)


def parse_surfrad_dat_lines(lines: list[str]) -> list[dict[str, Any]]:
    """
    Parses a NOAA SURFRAD 1-minute `.dat` file (2 header lines + space-delimited rows)
    into structured UTC minute records.
    """
    rows: list[dict[str, Any]] = []
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < 16:
            continue
        try:
            yr = int(parts[0])
            jday = int(parts[1])
            mo = int(parts[2])
            dy = int(parts[3])
            hr = int(parts[4])
            mn = int(parts[5])
            zen = float(parts[7])
            dw_solar = float(parts[8])
            direct_n = float(parts[12])
            qc_direct = int(parts[13])
            diffuse = float(parts[14])
            qc_diffuse = int(parts[15])
        except (ValueError, IndexError):
            continue

        temp_c = None
        rh_pct = None
        wind_ms = None
        pres_hpa = None
        if len(parts) >= 48:
            try:
                t_v, t_qc = float(parts[38]), int(parts[39])
                r_v, r_qc = float(parts[40]), int(parts[41])
                w_v, w_qc = float(parts[42]), int(parts[43])
                p_v, p_qc = float(parts[46]), int(parts[47])
                if t_qc == 0 and t_v > -90.0:
                    temp_c = t_v
                if r_qc == 0 and 0.0 <= r_v <= 105.0:
                    rh_pct = r_v
                if w_qc == 0 and w_v >= 0.0:
                    wind_ms = w_v
                if p_qc == 0 and p_v > 500.0:
                    pres_hpa = p_v
            except (ValueError, IndexError):
                pass

        # Guard against minute=60 edge rows in some legacy files
        if hr >= 24 or mn >= 60:
            continue
        t_utc = dt.datetime(yr, mo, dy, hr, mn, tzinfo=dt.timezone.utc)
        rows.append(
            {
                "utc": t_utc,
                "jday": jday,
                "zen_deg": zen,
                "el_deg": 90.0 - zen,
                "dw_solar_wm2": dw_solar,
                "direct_n_wm2": direct_n,
                "qc_direct": qc_direct,
                "diffuse_wm2": diffuse,
                "qc_diffuse": qc_diffuse,
                "temp_c": temp_c,
                "rh_pct": rh_pct,
                "wind_ms": wind_ms,
                "pres_hpa": pres_hpa,
            }
        )
    return rows


def _download_surfrad_utc_day(station_code: str, utc_date: dt.date) -> list[dict[str, Any]]:
    st = SURFRAD_STATIONS.get(station_code.lower())
    if st is None:
        return []
    SURFRAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yy = utc_date.year % 100
    doy = utc_date.timetuple().tm_yday
    fname = f"{st['code']}{yy:02d}{doy:03d}.dat"
    raw_cache = SURFRAD_CACHE_DIR / f"{utc_date.year}_{fname}"

    text: str | None = None
    if raw_cache.exists() and raw_cache.stat().st_size > 1000:
        try:
            text = raw_cache.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = None

    if text is None:
        url = f"{SURFRAD_BASE_URL}/{st['dir']}/{utc_date.year}/{fname}"
        with _SURFRAD_DL_LOCK:
            if raw_cache.exists() and raw_cache.stat().st_size > 1000:
                try:
                    text = raw_cache.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = None
            if text is None:
                for attempt in range(3):
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                        with urllib.request.urlopen(req, timeout=18.0) as resp:
                            body = resp.read().decode("utf-8", errors="ignore")
                        if len(body) > 1000:
                            tmp_p = raw_cache.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                            tmp_p.write_text(body, encoding="utf-8")
                            os.replace(tmp_p, raw_cache)
                            text = body
                            break
                    except Exception:
                        if attempt < 2:
                            time.sleep(1.0 * (attempt + 1))
                        continue
        if text is None:
            return []

    return parse_surfrad_dat_lines(text.splitlines())


def summarize_surfrad_golden_hour(
    rows: list[dict[str, Any]],
    station_meta: dict[str, Any],
    req_date: dt.date,
    phase: str = "sunset",
    el_min_deg: float = 1.5,
    el_max_deg: float = 12.0,
) -> dict[str, Any] | None:
    """
    Aggregates 1-minute SURFRAD pyrheliometer measurements across the local sunrise or sunset
    golden-hour window (`el in [1.5°, 12.0°]`).

    Uses local civil date + morning/afternoon split (`hour_local < 12` for sunrise,
    `hour_local >= 12` for sunset) so UTC midnight rollovers are handled seamlessly.
    """
    is_sunrise = phase.lower() == "sunrise"
    tz_obj, _ = local_tz_for_lon(float(station_meta["lon"]))
    elev_m = float(station_meta["elev_m"])
    # Slightly cleaner Linke turbidity at high western desert/mountain sites
    tl = 2.3 if elev_m >= 950.0 else 3.0

    selected: list[tuple[float, float, float, float, dict[str, Any]]] = []
    for r in rows:
        if r["qc_direct"] != 0 or r["direct_n_wm2"] < -10.0:
            continue
        el = float(r["el_deg"])
        if not (el_min_deg <= el <= el_max_deg):
            continue
        loc_dt = r["utc"].astimezone(tz_obj)
        if loc_dt.date() != req_date:
            continue
        if is_sunrise and loc_dt.hour >= 12:
            continue
        if (not is_sunrise) and loc_dt.hour < 12:
            continue

        clear_dni = clear_sky_dni_kasten_ineichen(el, elev_m, int(r["jday"]), linke_turbidity=tl)
        if clear_dni < 65.0:
            continue
        meas_dni = max(0.0, float(r["direct_n_wm2"]))
        rel_pct = max(0.0, min(100.0, (meas_dni / clear_dni) * 100.0))
        # Weight low golden-hour elevations (1.5..6 deg) higher than 10..12 deg
        w = max(0.45, 1.55 - 0.095 * (el - el_min_deg))
        selected.append((el, meas_dni, clear_dni, rel_pct, {"w": w, "row": r}))

    if len(selected) < 8:
        return None

    weights = np.array([x[4]["w"] for x in selected], dtype=np.float64)
    rels = np.array([x[3] for x in selected], dtype=np.float64)
    dnis = np.array([x[1] for x in selected], dtype=np.float64)
    clears = np.array([x[2] for x in selected], dtype=np.float64)
    els = np.array([x[0] for x in selected], dtype=np.float64)

    # Low-angle alpenglow core (el <= 6.0 deg)
    low_mask = els <= 6.0
    low_rel_pct = (
        float(np.mean(rels[low_mask])) if np.any(low_mask) else float(np.average(rels, weights=weights))
    )

    # WMO sunshine duration threshold: direct_n >= 120 W/m² (or >= 35% of clear-sky DNI)
    wmo_hits = (dnis >= 120.0) | (rels >= 35.0)
    sunshine_frac = float(np.mean(wmo_hits))
    golden_hour_sun_hit = bool(sunshine_frac >= 0.35 or float(np.percentile(rels, 80)) >= 45.0)

    # Surface conditions from valid 1-min rows
    temps = [x[4]["row"]["temp_c"] for x in selected if x[4]["row"]["temp_c"] is not None]
    rhs = [x[4]["row"]["rh_pct"] for x in selected if x[4]["row"]["rh_pct"] is not None]
    winds = [x[4]["row"]["wind_ms"] for x in selected if x[4]["row"]["wind_ms"] is not None]

    golden_pct = float(np.average(rels, weights=weights))
    return {
        "available": True,
        "source": f"NOAA SURFRAD 1-min NIP Pyrheliometer ({station_meta['code'].upper()})",
        "station_code": station_meta["code"],
        "station_name": station_meta["name"],
        "date": req_date.isoformat(),
        "phase": "sunrise" if is_sunrise else "sunset",
        "n_valid_minutes": len(selected),
        "golden_hour_rel_transmittance_pct": round(max(0.0, min(99.0, golden_pct)), 1),
        "low_angle_rel_transmittance_pct": round(max(0.0, min(99.0, low_rel_pct)), 1),
        "p90_rel_transmittance_pct": round(float(np.percentile(rels, 90)), 1),
        "mean_dni_wm2": round(float(np.mean(dnis)), 1),
        "max_dni_wm2": round(float(np.max(dnis)), 1),
        "mean_clearsky_dni_wm2": round(float(np.mean(clears)), 1),
        "wmo_sunshine_fraction": round(sunshine_frac, 3),
        "golden_hour_sun_hit": golden_hour_sun_hit,
        "mean_temp_c": round(float(np.mean(temps)), 1) if temps else None,
        "mean_rh_pct": round(float(np.mean(rhs)), 1) if rhs else None,
        "mean_wind_ms": round(float(np.mean(winds)), 1) if winds else None,
    }


def fetch_surfrad_golden_hour_truth(
    station_code: str,
    date_str: str,
    phase: str = "sunset",
) -> dict[str, Any] | None:
    """
    Fetches and caches NOAA SURFRAD 1-minute pyrheliometer golden-hour ground truth
    for `station_code` (`tbl`, `dra`, `fpk`, `sxf`, `bon`, `gwn`, `psu`), `date_str`,
    and `phase` (`"sunrise"` or `"sunset"`).
    """
    st = SURFRAD_STATIONS.get(station_code.lower())
    if st is None:
        return None
    req_d = dt.date.fromisoformat(date_str)
    phase_norm = "sunrise" if phase.lower() == "sunrise" else "sunset"

    SURFRAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfile = SURFRAD_CACHE_DIR / f"surfrad_gh_v1_{st['code']}_{phase_norm}_{date_str}.json"
    if cfile.exists():
        try:
            return json.loads(cfile.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Load both req_d and req_d + 1 day UTC so Western US sunsets past 00:00 UTC are never truncated
    rows = _download_surfrad_utc_day(st["code"], req_d)
    if phase_norm == "sunset":
        rows = rows + _download_surfrad_utc_day(st["code"], req_d + dt.timedelta(days=1))

    summary = summarize_surfrad_golden_hour(rows, st, req_d, phase=phase_norm)
    if summary is not None:
        try:
            cfile.write_text(json.dumps(summary), encoding="utf-8")
        except Exception:
            pass
    return summary
