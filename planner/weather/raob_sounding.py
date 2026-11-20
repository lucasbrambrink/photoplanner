"""
IGRA / NWS Upper-Air Radiosonde Balloon Sounding Ingestion (`00Z` Sunset & `12Z` Sunrise).

Fetches high-vertical-resolution (~100-200 level) observed weather balloon soundings
from the Iowa Environmental Mesonet (IEM) / NOAA RAOB archive (`mesonet.agron.iastate.edu/json/raob.py`)
with permanent local disk caching (`data/weather_cache/raob/`).

Why `00:00 UTC` & `12:00 UTC` Radiosondes Are Gold-Standard Ground Truth for Western US Golden Hour:
- NWS `00:00 UTC` balloons are physically released at ~`23:00 UTC` (`15:00 PST` / `16:00 PDT`),
  ascending through the boundary layer and tropopause right into California sunset golden hour!
- NWS `12:00 UTC` balloons are released at ~`11:00 UTC` (`03:00 PST` / `04:00 PDT`), right before
  California dawn / sunrise golden hour!
- Resolves 20-40 m vertical marine-layer inversions (`KOAK` Oakland 28 km from Montara/Pedro Point,
  `KVBG` Vandenberg on the Central Coast, `KNKX` Miramar, `KREV` Reno for Sierra/Yosemite lee side)
  that 6-level NWP pressure grids (`1000, 925, 850 hPa`) completely smear out.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
RAOB_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "raob"
USER_AGENT = "PhotoPlanner/3.1 (raob-sounding-verification; github.com/lucasbrambrink/photoplanner)"

_MEM_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}
_MEM_TTL_SECONDS = 900.0

# Curated Western & CONUS NWS Upper-Air Radiosonde Stations: (icao, name, lat, lon, elev_m)
RAOB_STATIONS: tuple[tuple[str, str, float, float, float], ...] = (
    ("KOAK", "Oakland Intl Upper Air", 37.7283, -122.2158, 3.0),
    ("KVBG", "Vandenberg SFB Upper Air", 34.7500, -120.5667, 121.0),
    ("KNKX", "San Diego Miramar Upper Air", 32.8667, -117.1500, 134.0),
    ("KREV", "Reno NWS Upper Air", 39.5667, -119.8000, 1516.0),
    ("KMFR", "Medford Upper Air", 42.3667, -122.8667, 405.0),
    ("KVEF", "Las Vegas Upper Air", 36.0500, -115.1833, 664.0),
    ("KLKN", "Elko Upper Air", 40.8667, -115.7333, 1595.0),
    ("KSLE", "Salem Upper Air", 44.9167, -123.0000, 61.0),
    ("KUIL", "Quillayute Coastal Upper Air", 47.9500, -124.5500, 56.0),
    ("KOTX", "Spokane Upper Air", 47.6833, -117.6333, 728.0),
    ("KBOI", "Boise Upper Air", 43.5667, -116.2167, 871.0),
    ("KSLC", "Salt Lake City Upper Air", 40.7700, -111.9600, 1288.0),
    ("KFGZ", "Flagstaff Upper Air", 35.2333, -111.8167, 2192.0),
    ("KTUS", "Tucson Upper Air", 32.1167, -110.9333, 779.0),
    ("KGJT", "Grand Junction Upper Air", 39.1167, -108.5333, 1475.0),
    ("KDNR", "Denver Upper Air", 39.7667, -104.8667, 1611.0),
    ("KABQ", "Albuquerque Upper Air", 35.0333, -106.6167, 1619.0),
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r_km * math.asin(min(1.0, math.sqrt(a)))


def _rh_liquid_or_ice_pct(temp_c: float, dew_c: float) -> tuple[float, float]:
    """
    Returns `(rh_water_pct, rh_effective_cloud_pct)` where `rh_effective_cloud_pct`
    evaluates saturation with respect to liquid water for `T >= -12 C` and seamlessly
    transitions to saturation with respect to hexagonal ice (`e_si`) for `T <= -28 C`
    (essential for detecting upper-tropospheric cirrus decks at `-35 C..-60 C`).
    """
    if dew_c > temp_c + 0.2:
        dew_c = temp_c
    # Alduchov & Eskridge (1996) / Magnus over liquid water
    es_w = 6.1094 * math.exp((17.625 * temp_c) / max(15.0, temp_c + 243.04))
    e_vap = 6.1094 * math.exp((17.625 * dew_c) / max(15.0, dew_c + 243.04))
    rh_w = max(1.0, min(100.0, 100.0 * e_vap / max(1e-5, es_w)))

    if temp_c >= -12.0:
        return rh_w, rh_w

    # Saturation vapor pressure over ice (Magnus-Tetens ice)
    es_i = 6.1121 * math.exp((22.587 * temp_c) / max(15.0, temp_c + 273.86))
    rh_i = max(1.0, min(115.0, 100.0 * e_vap / max(1e-5, es_i)))
    w_ice = max(0.0, min(1.0, (-12.0 - temp_c) / 16.0))
    rh_eff = (1.0 - w_ice) * rh_w + w_ice * min(100.0, rh_i)
    return rh_w, rh_eff


def _wind_uv_from_dir_sknt(drct_deg: float | None, sknt: float | None) -> tuple[float, float] | None:
    if drct_deg is None or sknt is None:
        return None
    spd_ms = float(sknt) * 0.514444
    rad = math.radians(float(drct_deg))
    return round(-spd_ms * math.sin(rad), 2), round(-spd_ms * math.cos(rad), 2)


def _candidate_synoptic_times(target_utc: dt.datetime, causal_only: bool) -> list[dt.datetime]:
    """
    Returns ordered `00Z` / `12Z` synoptic sounding timestamps near `target_utc`.
    Note: `00Z` balloons launch at `~23:00 UTC` (`T - 1h`), so a `00Z` sounding is already
    in flight / completed by `00:15 UTC` (`16:15 PST` / `17:15 PDT` sunset!).
    """
    t_u = (
        target_utc.replace(tzinfo=dt.timezone.utc)
        if target_utc.tzinfo is None
        else target_utc.astimezone(dt.timezone.utc)
    )
    base_day = dt.datetime(t_u.year, t_u.month, t_u.day, 0, 0, tzinfo=dt.timezone.utc)
    candidates = [
        base_day - dt.timedelta(hours=12),
        base_day,
        base_day + dt.timedelta(hours=12),
        base_day + dt.timedelta(hours=24),
        base_day - dt.timedelta(hours=24),
    ]
    if causal_only:
        # Operational radiosonde balloons launched ~45 min pre-nominal take ~90-120 min to reach
        # 10 hPa and typically appear on GTS/IEM ~45-75 min AFTER nominal synoptic time.
        # Enforce `nominal + 60 min <= t_u` so causal nowcasts & backtests never leak future 00Z/12Z soundings.
        candidates = [c for c in candidates if (c + dt.timedelta(minutes=60)) <= t_u]

    candidates.sort(key=lambda c: abs((t_u - c).total_seconds()))
    return [c for c in candidates if abs((t_u - c).total_seconds()) <= 18.0 * 3600.0]


def _analyze_raob_profile(
    raw_levels: list[dict[str, Any]],
    station_id: str,
    station_name: str,
    st_lat: float,
    st_lon: float,
    st_elev_m: float,
    dist_km: float,
    valid_utc: dt.datetime,
    target_utc: dt.datetime,
) -> dict[str, Any] | None:
    levels: list[dict[str, float]] = []
    winds_by_pres: list[tuple[float, float, float]] = []

    for row in raw_levels:
        h = row.get("hght")
        tc = row.get("tmpc")
        dc = row.get("dwpc")
        pr = row.get("pres")
        if h is None or tc is None or pr is None:
            continue
        try:
            h_f = float(h)
            tc_f = float(tc)
            pr_f = float(pr)
            dc_f = float(dc) if (dc is not None and str(dc).strip() not in ("", "M")) else (tc_f - 18.0)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(h_f) and math.isfinite(tc_f) and math.isfinite(pr_f) and math.isfinite(dc_f)):
            continue
        rh_w, rh_eff = _rh_liquid_or_ice_pct(tc_f, dc_f)
        levels.append(
            {
                "pres_hpa": pr_f,
                "hght_m": h_f,
                "tmpc": tc_f,
                "dwpc": dc_f,
                "rh_water_pct": rh_w,
                "rh_eff_pct": rh_eff,
            }
        )
        uv = _wind_uv_from_dir_sknt(row.get("drct"), row.get("sknt"))
        if uv is not None:
            winds_by_pres.append((pr_f, uv[0], uv[1]))

    if len(levels) < 8:
        return None

    levels.sort(key=lambda x: x["hght_m"])
    sfc = levels[0]
    sfc_lcl_asl_m = round(
        sfc["hght_m"] + max(35.0, 125.0 * max(0.0, sfc["tmpc"] - sfc["dwpc"])), 1
    )

    # 1. Detect primary boundary-layer / subsidence temperature inversion (`0 .. 3,400 m` ASL)
    # Merge adjacent warming segments separated by thin isothermal/micro-cooling steps (< 0.35 K over < 160 m)
    best_inv: dict[str, float] | None = None
    n_lv = len(levels)
    for i in range(n_lv - 1):
        z_base = levels[i]["hght_m"]
        t_base = levels[i]["tmpc"]
        if z_base - st_elev_m > 2800.0 or z_base > 3400.0:
            break
        if levels[i + 1]["tmpc"] <= t_base:
            continue

        peak_idx = i + 1
        peak_t = levels[i + 1]["tmpc"]
        j = i + 1
        while j < n_lv:
            zj = levels[j]["hght_m"]
            tj = levels[j]["tmpc"]
            if zj - z_base > 1400.0:
                break
            if tj >= peak_t:
                peak_t = tj
                peak_idx = j
            elif peak_t - tj > 0.65:
                break
            j += 1

        strength_k = peak_t - t_base
        z_top = levels[peak_idx]["hght_m"]
        # Exclude 0-35 m nocturnal surface skin inversions when a real marine inversion exists aloft,
        # unless the surface inversion itself is strong (>= 2.2 K)
        agl_base = max(0.0, z_base - st_elev_m)
        if strength_k < 0.55:
            continue
        score = strength_k * (1.25 if 90.0 <= agl_base <= 1650.0 else 0.72)
        if best_inv is None or score > best_inv["score"]:
            best_inv = {
                "score": score,
                "base_asl_m": round(z_base, 1),
                "top_asl_m": round(z_top, 1),
                "strength_k": round(strength_k, 2),
                "base_temp_c": round(t_base, 2),
                "top_temp_c": round(peak_t, 2),
            }

    inv_base_m = best_inv["base_asl_m"] if best_inv else None
    inv_top_m = best_inv["top_asl_m"] if best_inv else None
    inv_strength_k = best_inv["strength_k"] if best_inv else 0.0

    # 2. Sub-inversion marine stratus / fog saturation analysis
    lid_search_m = (inv_base_m + 95.0) if inv_base_m is not None else (st_elev_m + 1400.0)
    bl_levels = [lv for lv in levels if lv["hght_m"] <= lid_search_m]
    sub_inv_max_rh = max((lv["rh_water_pct"] for lv in bl_levels), default=sfc["rh_water_pct"])

    sat_bl = [
        lv for lv in bl_levels
        if lv["rh_water_pct"] >= 89.0 or (lv["tmpc"] - lv["dwpc"]) <= 1.75
    ]
    stratus_base_m = round(sat_bl[0]["hght_m"], 1) if sat_bl else None
    stratus_top_m = round(sat_bl[-1]["hght_m"], 1) if sat_bl else None
    if stratus_top_m is not None and inv_base_m is not None:
        # Marine stratus cloud tops physically terminate right at the sharp inversion base jump
        stratus_top_m = round(max(stratus_top_m, min(inv_base_m + 35.0, stratus_top_m + 60.0)), 1)

    # 3. Multi-layer radiosonde cloud-fraction proxies (WMO / Wang-Rossow RH profiles)
    # Use consistent ASL bounds (clamped >= st_elev_m) so high-elevation mountain stations
    # like KREV (1,516 m) and KFGZ (2,192 m) never suffer multi-kilometer blind gaps between bands.
    def _band_cloud_proxy_asl(z_min_asl: float, z_max_asl: float, rh_crit: float, rh_full: float) -> float:
        lo_m = max(st_elev_m, z_min_asl)
        hi_m = max(lo_m + 150.0, z_max_asl)
        band = [
            lv["rh_eff_pct"]
            for lv in levels
            if (lv["hght_m"] >= lo_m) and (lv["hght_m"] <= hi_m)
        ]
        if not band:
            return 0.0
        peak_rh = max(band)
        if peak_rh <= rh_crit:
            return 0.0
        return round(min(1.0, ((peak_rh - rh_crit) / max(2.0, rh_full - rh_crit)) ** 1.15), 3)

    fog_cf_proxy = _band_cloud_proxy_asl(st_elev_m, max(st_elev_m + 650.0, inv_base_m or 750.0), 84.0, 96.5)
    low_cf_proxy = _band_cloud_proxy_asl(st_elev_m, max(st_elev_m + 1800.0, 2600.0), 82.0, 96.0)
    mid_cf_proxy = _band_cloud_proxy_asl(max(st_elev_m + 1200.0, 2200.0), 6500.0, 78.0, 94.0)
    high_cf_proxy = _band_cloud_proxy_asl(6200.0, 14800.0, 72.0, 93.0)

    def _wind_at_pres(target_p: float) -> list[float] | None:
        if not winds_by_pres:
            return None
        best_w = min(winds_by_pres, key=lambda t: abs(t[0] - target_p))
        if abs(best_w[0] - target_p) > 85.0:
            return None
        return [best_w[1], best_w[2]]

    age_hr = round((target_utc - valid_utc).total_seconds() / 3600.0, 2)
    return {
        "station_id": station_id,
        "station_name": station_name,
        "station_lat": st_lat,
        "station_lon": st_lon,
        "station_elevation_m": st_elev_m,
        "distance_km": round(dist_km, 1),
        "valid_utc": valid_utc.isoformat(),
        "offset_hours_from_target": age_hr,
        "n_levels": len(levels),
        "surface_temp_c": round(sfc["tmpc"], 2),
        "surface_dewpoint_c": round(sfc["dwpc"], 2),
        "surface_rh_pct": round(sfc["rh_water_pct"], 1),
        "lcl_asl_m": sfc_lcl_asl_m,
        "has_inversion": best_inv is not None,
        "inversion_base_asl_m": inv_base_m,
        "inversion_top_asl_m": inv_top_m,
        "inversion_strength_k": round(inv_strength_k, 2),
        "sub_inversion_max_rh_pct": round(sub_inv_max_rh, 1),
        "saturated_stratus_base_asl_m": stratus_base_m,
        "saturated_stratus_top_asl_m": stratus_top_m,
        "fog_cloud_fraction_proxy": fog_cf_proxy,
        "low_cloud_fraction_proxy": low_cf_proxy,
        "mid_cloud_fraction_proxy": mid_cf_proxy,
        "high_cloud_fraction_proxy": high_cf_proxy,
        "wind_925_uv_ms": _wind_at_pres(925.0),
        "wind_850_uv_ms": _wind_at_pres(850.0),
        "wind_500_uv_ms": _wind_at_pres(500.0),
        "wind_250_uv_ms": _wind_at_pres(250.0),
    }


def fetch_raob_sounding(
    lat: float,
    lon: float,
    target_utc: dt.datetime | None = None,
    max_distance_km: float = 290.0,
    causal_only: bool = False,
) -> dict[str, Any] | None:
    """
    Fetches the nearest NWS `00Z` / `12Z` radiosonde vertical sounding within `max_distance_km`
    of `(lat, lon)` around `target_utc`.

    - When `causal_only=False` (Historical Observed Truth mode), selects the closest synoptic
      balloon launch (`00Z` for West Coast sunset, `12Z` for sunrise).
    - When `causal_only=True` (Nowcast / Hindcast simulation), strictly requires the balloon
      release to precede `target_utc` so zero future data leaks into forecasts.
    """
    t_u = (
        dt.datetime.now(dt.timezone.utc)
        if target_utc is None
        else (
            target_utc.replace(tzinfo=dt.timezone.utc)
            if target_utc.tzinfo is None
            else target_utc.astimezone(dt.timezone.utc)
        )
    )
    if target_utc is None:
        causal_only = True

    ranked_stations: list[tuple[float, str, str, float, float, float]] = []
    for sid, sname, slat, slon, selev in RAOB_STATIONS:
        d_km = _haversine_km(lat, lon, slat, slon)
        if d_km <= max_distance_km:
            ranked_stations.append((d_km, sid, sname, slat, slon, selev))
    if not ranked_stations:
        return None
    ranked_stations.sort(key=lambda x: x[0])

    synoptic_times = _candidate_synoptic_times(t_u, causal_only=causal_only)
    if not synoptic_times:
        return None

    RAOB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    now_mono = time.monotonic()

    for syn_dt in synoptic_times[:2]:
        for d_km, sid, sname, slat, slon, selev in ranked_stations[:2]:
            ts_str = syn_dt.strftime("%Y%m%d%H00")
            cache_key = f"{sid}_{ts_str}"
            raw_levels: list[dict[str, Any]] | None = None

            if cache_key in _MEM_CACHE:
                ts_m, cached_doc = _MEM_CACHE[cache_key]
                if cached_doc is not None:
                    raw_levels = cached_doc.get("profile")
                elif now_mono - ts_m < _MEM_TTL_SECONDS:
                    continue

            disk_path = RAOB_CACHE_DIR / f"raob_{cache_key}.json"
            if raw_levels is None and disk_path.exists():
                try:
                    saved = json.loads(disk_path.read_text(encoding="utf-8"))
                    raw_levels = saved.get("profile") or []
                    if raw_levels:
                        _MEM_CACHE[cache_key] = (now_mono, {"profile": raw_levels})
                except Exception:
                    raw_levels = None

            if raw_levels is None:
                url = f"https://mesonet.agron.iastate.edu/json/raob.py?ts={ts_str}&station={sid}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req, timeout=6.5) as resp:
                        doc = json.loads(resp.read().decode("utf-8"))
                    profs = doc.get("profiles") or []
                    if profs and profs[0].get("profile"):
                        raw_levels = profs[0]["profile"]
                        _MEM_CACHE[cache_key] = (now_mono, {"profile": raw_levels})
                        try:
                            disk_path.write_text(
                                json.dumps({"station": sid, "valid_utc": syn_dt.isoformat(), "profile": raw_levels}),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                    else:
                        _MEM_CACHE[cache_key] = (now_mono, None)
                        continue
                except Exception:
                    continue

            if raw_levels:
                analyzed = _analyze_raob_profile(
                    raw_levels, sid, sname, slat, slon, selev, d_km, syn_dt, t_u
                )
                if analyzed is not None:
                    return analyzed

    return None
