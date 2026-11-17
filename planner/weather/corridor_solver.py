"""
Unified Corridor Atmospheric Solver (`planner/weather/corridor_solver.py`).

Couples:
1. Live `0-3.5h` GOES-18/19 multi-spectral cloud retrieval + 2-frame FFT optical-flow advection
2. Live NWS `api.weather.gov` METAR / RAWS surface ceilometer & visibility assimilation
3. 24-hour (`00:00..23:00` local) multi-station pressure-level NWP soundings (`0..320 km`),
   live multi-model runs (`HRRR 3km`, `ECMWF IFS 0.25`, `ECMWF AIFS 0.25 AI`, `GFS`), and CAMS AOD
4. Rust `./raycaster/target/release/photoplanner corridor-raycast` (`0-380 km` curved-Earth DEM +
   6-band Rayleigh/Chappuis/Mie + altitude-exact 3D stochastic cloud-chord integrator)

Never fabricates MD5 pseudo-climatology when live APIs fail — always reports explicit
`data_provenance` (`is_live`, `degraded`, `sources`, `nowcast_active`, `goes_scan_utc`, `metar_station`).
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .goes_nowcast import fetch_goes_corridor_nowcast, sample_advected_goes_state
from .nwp_corridor import (
    _safe_arr_val,
    extract_hour_sounding,
    fetch_24h_corridor_and_models,
    local_tz_for_lon,
    offset_lat_lon_km,
)
from .raob_sounding import fetch_raob_sounding
from .station_obs import fetch_nearest_station_observation

ROOT_DIR = Path(__file__).resolve().parents[2]
RUST_BIN = ROOT_DIR / "raycaster" / "target" / "release" / "photoplanner"
DEFAULT_ELEVATION_DIR = Path(
    os.environ.get(
        "PHOTOPLANNER_ELEVATION_DIR",
        str(
            Path("/Users/lucasbrambrink/Coding/recreation/recreation/data/elevation")
            if Path("/Users/lucasbrambrink/Coding/recreation/recreation/data/elevation").exists()
            else (ROOT_DIR / "data" / "srtm_cache")
        ),
    )
)
DEFAULT_RECREATION_DB = Path(
    os.environ.get(
        "PHOTOPLANNER_RECREATION_DB",
        str(
            Path("/Users/lucasbrambrink/Coding/recreation/recreation/db.sqlite3")
            if Path("/Users/lucasbrambrink/Coding/recreation/recreation/db.sqlite3").exists()
            else (ROOT_DIR / "data" / "recreation_fallback.sqlite3")
        ),
    )
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r_km * math.asin(min(1.0, math.sqrt(a)))


def _is_coastal_california(lat: float, lon: float) -> bool:
    if 36.8 <= lat <= 38.4 and -123.2 <= lon <= -122.10:
        return True
    coast_lon = -124.2 + max(0.0, 40.5 - lat) * 0.72
    return (32.5 <= lat <= 42.0) and (abs(lon - coast_lon) <= 0.45)


def _solar_ephemeris_local(
    lat: float, lon: float, req_date: dt.date, minutes_local: float, tz_offset_hours: float
) -> tuple[float, float]:
    """Returns `(azimuth_deg, elevation_deg)` at `minutes_local` (`0..1440`)."""
    doy = req_date.timetuple().tm_yday
    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (minutes_local / 60.0 - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.001480 * math.sin(3.0 * gamma)
    )
    tst = minutes_local + eqtime + 4.0 * lon - 60.0 * tz_offset_hours
    ha = math.radians(tst / 4.0 - 180.0)
    lat_r = math.radians(lat)
    cos_zen = max(
        -1.0,
        min(1.0, math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)),
    )
    zen = math.acos(cos_zen)
    el_deg = 90.0 - math.degrees(zen)
    sin_zen = max(1e-6, math.sin(zen))
    cos_az = max(
        -1.0,
        min(1.0, (math.sin(decl) - math.sin(lat_r) * cos_zen) / (math.cos(lat_r) * sin_zen)),
    )
    az_deg = math.degrees(math.acos(cos_az))
    if ha > 0:
        az_deg = 360.0 - az_deg
    return round(az_deg, 2), round(el_deg, 2)


def _find_sunrise_sunset_golden_hours(
    lat: float, lon: float, req_date: dt.date, tz_offset_hours: float
) -> tuple[tuple[int, float, float], tuple[int, float, float]]:
    """Finds `(minutes_local, az_deg, el_deg)` for sunrise (`el ~ +1.0°`) and sunset (`el ~ +0.8°`)."""
    best_sr = (390, 102.0, 1.0)
    best_sr_err = 999.0
    best_ss = (1050, 258.0, 0.8)
    best_ss_err = 999.0
    for mins in range(270, 1320, 4):
        az, el = _solar_ephemeris_local(lat, lon, req_date, float(mins), tz_offset_hours)
        if mins < 720:
            err = abs(el - 1.0)
            if err < best_sr_err:
                best_sr_err = err
                best_sr = (mins, az, el)
        else:
            err = abs(el - 0.8)
            if err < best_ss_err:
                best_ss_err = err
                best_ss = (mins, az, el)
    return best_sr, best_ss


def _compass_cardinal(az_deg: float) -> str:
    dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    return dirs[int(round(((az_deg % 360.0) + 360.0) % 360.0 / 22.5)) % 16]


def _blend_nowcast_into_station(
    sounding: dict[str, Any],
    goes_data: dict[str, Any] | None,
    station_obs: dict[str, Any] | None,
    lead_minutes: float,
    w_nowcast: float,
    raob_obs: dict[str, Any] | None = None,
    is_coastal_loc: bool = False,
) -> dict[str, Any]:
    """
    Fuses live/archived GOES-18/19 advected multi-layer cloud fractions (`CCLC` + `BTD` + `ACTP/ACHA/COD`),
    NWS/IEM ASOS laser ceilometer & visibility, and `00Z`/`12Z` NWS IGRA/RAOB weather-balloon soundings
    into a `CorridorStationProfile`.
    """
    if w_nowcast <= 0.02 and raob_obs is None:
        return sounding

    st = json.loads(json.dumps(sounding))
    rs = st["rust_station"]
    dist_km = float(rs["distance_km"])
    elev_m = float(rs["surface_elevation_m"])

    # 1. Assimilate 00Z / 12Z RAOB radiosonde boundary-layer inversion & upper-air winds
    w_raob = 0.0
    if raob_obs is not None and w_nowcast > 0.02:
        raob_dist_km = _haversine_km(
            float(rs["lat"]),
            float(rs["lon"]),
            float(raob_obs["station_lat"]),
            float(raob_obs["station_lon"]),
        )
        dt_hr = abs(float(raob_obs.get("offset_hours_from_target", 0.0)))
        if raob_dist_km <= 215.0 and dt_hr <= 9.5:
            dist_w = math.exp(-max(0.0, raob_dist_km - 15.0) / 145.0)
            time_w = 1.0 if dt_hr <= 4.0 else math.exp(-(dt_hr - 4.0) / 7.5)
            w_raob = min(0.92, max(w_nowcast, 0.45) * dist_w * time_w)

            for w_key in ("wind_850_uv_ms", "wind_500_uv_ms", "wind_250_uv_ms"):
                r_uv = raob_obs.get(w_key)
                if isinstance(r_uv, (list, tuple)) and len(r_uv) == 2:
                    s_uv = st[w_key]
                    wu = (1.0 - 0.55 * w_raob) * float(s_uv[0]) + 0.55 * w_raob * float(r_uv[0])
                    wv = (1.0 - 0.55 * w_raob) * float(s_uv[1]) + 0.55 * w_raob * float(r_uv[1])
                    st[w_key] = (round(wu, 2), round(wv, 2))

            r_strength = float(raob_obs.get("inversion_strength_k") or 0.0)
            r_inv_base = raob_obs.get("inversion_base_asl_m")
            r_inv_top = raob_obs.get("inversion_top_asl_m")
            if r_strength >= 0.6 and r_inv_base is not None:
                rb_m = float(r_inv_base)
                rt_m = float(r_inv_top) if r_inv_top is not None else (rb_m + 150.0)
                sat_stratus_top = raob_obs.get("saturated_stratus_top_asl_m")
                r_lid_m = (
                    float(sat_stratus_top)
                    if sat_stratus_top is not None
                    else (rb_m + 0.18 * max(0.0, rt_m - rb_m))
                )
                st["inv_base_asl_m"] = round((1.0 - w_raob) * float(st["inv_base_asl_m"]) + w_raob * rb_m, 1)
                st["inv_top_asl_m"] = round((1.0 - w_raob) * float(st["inv_top_asl_m"]) + w_raob * r_lid_m, 1)
                st["inv_jump_k"] = round((1.0 - w_raob) * float(st["inv_jump_k"]) + w_raob * r_strength, 2)
                st["raob_assimilated"] = True
                st["raob_weight"] = round(w_raob, 3)
                st["raob_lid_asl_m"] = round(r_lid_m, 1)
                st["raob_inv_base_asl_m"] = round(rb_m, 1)
                st["raob_inv_strength_k"] = round(r_strength, 2)
            elif w_raob >= 0.45:
                st["inv_jump_k"] = round((1.0 - 0.65 * w_raob) * float(st["inv_jump_k"]) + 0.65 * w_raob * r_strength, 2)
                st["raob_assimilated"] = True
                st["raob_weight"] = round(w_raob, 3)
                st["raob_inv_strength_k"] = round(r_strength, 2)

    if w_nowcast <= 0.02:
        return st

    # 2. Assimilate surface METAR / ASOS ceilometer, visibility, and boundary-layer LCL
    if station_obs is not None and dist_km <= 65.0:
        obs_dist_decay = math.exp(-dist_km / 45.0)
        w_obs = min(0.88, w_nowcast * obs_dist_decay)
        st_elev_obs = float(station_obs.get("station_elevation_m", 0.0))
        elev_diff_m = abs(elev_m - st_elev_obs)
        if station_obs.get("temp_c") is not None and dist_km <= 25.0 and elev_diff_m <= 240.0:
            obs_t = float(station_obs["temp_c"]) - 0.0065 * (elev_m - st_elev_obs)
            st["temp_2m_c"] = round((1.0 - w_obs) * st["temp_2m_c"] + w_obs * obs_t, 2)
            rs["surface_temp_c"] = st["temp_2m_c"]
        if station_obs.get("rh_pct") is not None and dist_km <= 25.0 and elev_diff_m <= 240.0:
            st["rh_2m_pct"] = round(
                (1.0 - w_obs) * st["rh_2m_pct"] + w_obs * float(station_obs["rh_pct"]), 1
            )
        if station_obs.get("visibility_m") is not None and elev_diff_m <= 380.0:
            obs_vis_m = float(station_obs["visibility_m"])
            if obs_vis_m < 15_250.0 or st["visibility_m"] < obs_vis_m:
                st["visibility_m"] = round(
                    (1.0 - w_obs) * st["visibility_m"] + w_obs * obs_vis_m, 0
                )
        if station_obs.get("aod_550nm_proxy") is not None:
            rs["aod_550nm"] = round(
                (1.0 - 0.55 * w_obs) * rs["aod_550nm"]
                + 0.55 * w_obs * float(station_obs["aod_550nm_proxy"]),
                3,
            )
        if station_obs.get("lcl_asl_m") is not None and dist_km <= 25.0 and elev_diff_m <= 280.0:
            st["lcl_asl_m"] = round(
                (1.0 - w_obs) * st["lcl_asl_m"] + w_obs * float(station_obs["lcl_asl_m"]), 1
            )

    # 3. Assimilate 3-frame GOES-18/19 L2 (CCLC 5-layer + C13-C07 fog BTD + ACTP/ACHA/COD)
    if goes_data is not None:
        adv = sample_advected_goes_state(
            goes_data,
            float(rs["lat"]),
            float(rs["lon"]),
            lead_minutes,
            tuple(st["wind_850_uv_ms"]),
            tuple(st["wind_500_uv_ms"]),
            tuple(st["wind_250_uv_ms"]),
        )
        sat_low_pct = adv["low_cf"] * 100.0
        sat_fog_pct = float(adv.get("fog_cf", adv["low_cf"] * (0.82 if is_coastal_loc else 0.35))) * 100.0
        sat_mid_pct = adv["mid_cf"] * 100.0
        sat_high_pct = adv["high_cf"] * 100.0
        has_cclc_ml = bool(goes_data.get("has_cclc_multilayer") or goes_data.get("has_cclc"))

        raw_sat_low_pct = sat_low_pct
        upper_shield = max(0.0, min(1.0, (sat_high_pct + 0.75 * sat_mid_pct - 45.0) / 40.0))
        is_space_occluded = upper_shield > 0.22 and raw_sat_low_pct < 38.0
        w_asos_used = 0.0

        if station_obs is not None and dist_km <= 45.0:
            obs_low_pct = float(station_obs.get("low_cloud_fraction", 0.0)) * 100.0
            obs_fog_pct = float(
                station_obs.get("fog_cloud_fraction", station_obs.get("low_cloud_fraction", 0.0) * 0.65)
            ) * 100.0
            # Bottom-up laser ceilometer looks directly into the boundary layer without upper-cirrus/frontal masking
            if is_space_occluded:
                base_asos_w = 0.56 + 0.32 * upper_shield
                w_asos_used = base_asos_w * math.exp(-dist_km / 48.0)
            else:
                base_asos_w = 0.56 if (sat_high_pct >= 55.0 and not has_cclc_ml) else (0.48 if sat_high_pct >= 60.0 else 0.38)
                w_asos_used = base_asos_w * math.exp(-dist_km / 42.0)
            sat_low_pct = (1.0 - w_asos_used) * sat_low_pct + w_asos_used * obs_low_pct
            sat_fog_pct = (1.0 - w_asos_used) * sat_fog_pct + w_asos_used * obs_fog_pct
        elif raob_obs is not None and w_raob >= 0.35 and (sat_high_pct >= 55.0 or is_space_occluded):
            r_low_pct = float(raob_obs.get("low_cloud_fraction_proxy", sat_low_pct / 100.0)) * 100.0
            w_r_low = min(0.72, (0.34 + 0.32 * upper_shield) * w_raob)
            sat_low_pct = (1.0 - w_r_low) * sat_low_pct + w_r_low * r_low_pct
            sat_fog_pct = (1.0 - w_r_low) * sat_fog_pct + w_r_low * (r_low_pct * (0.85 if is_coastal_loc else 0.45))

        # Differentiate layer blend weights: pinned boundary-layer stratus persists longer
        # than fast-moving upper-tropospheric cirrus (`25-45 m/s` jet advection).
        # Under an optically thick upper shield without a co-located ceilometer, attenuate spaceborne low-cloud weight
        # so occluded `0%` satellite retrievals never wipe out genuine NWP/RAOB boundary-layer stratus.
        pin_boost = 0.12 * float(adv.get("pin_weight_low", 0.0))
        occ_atten = (1.0 - 0.72 * upper_shield * (1.0 - w_asos_used)) if is_space_occluded else 1.0
        w_low_eff = min(0.96, w_nowcast * (1.06 + pin_boost) * max(0.22, occ_atten))
        w_mid_eff = min(0.94, w_nowcast * 0.96)
        w_hi_eff = min(0.92, w_nowcast * 0.88)

        new_low = int(round((1.0 - w_low_eff) * st["cloud_low_pct"] + w_low_eff * sat_low_pct))
        new_mid = int(round((1.0 - w_mid_eff) * st["cloud_mid_pct"] + w_mid_eff * sat_mid_pct))
        new_high = int(round((1.0 - w_hi_eff) * st["cloud_high_pct"] + w_hi_eff * sat_high_pct))

        st["cloud_low_pct"] = max(0, min(100, new_low))
        st["cloud_mid_pct"] = max(0, min(100, new_mid))
        st["cloud_high_pct"] = max(0, min(100, new_high))

        orig_fog_band = next((b for b in rs.get("cloud_bands", []) if b.get("label") == "fog"), None)
        orig_cu_band = next((b for b in rs.get("cloud_bands", []) if b.get("label") == "low"), None)
        new_bands: list[dict[str, Any]] = []
        low_cf = st["cloud_low_pct"] / 100.0
        mid_cf = st["cloud_mid_pct"] / 100.0
        high_cf = st["cloud_high_pct"] / 100.0

        nwp_fog_cf = float(orig_fog_band["cloud_fraction"]) if orig_fog_band is not None else (
            (int(sounding["cloud_low_pct"]) / 100.0) * (0.80 if is_coastal_loc else 0.25)
        )
        blended_fog_cf = min(low_cf, (1.0 - w_low_eff) * nwp_fog_cf + w_low_eff * (sat_fog_pct / 100.0))

        ceil_asl = (
            float(station_obs["lowest_ceiling_asl_m"])
            if (station_obs and station_obs.get("lowest_ceiling_asl_m") and dist_km <= 40.0)
            else None
        )
        fog_ceil_asl = (
            float(station_obs["fog_base_asl_m"])
            if (station_obs and station_obs.get("fog_base_asl_m") and dist_km <= 40.0)
            else ceil_asl
        )
        is_cal_l2 = bool(goes_data.get("is_calibrated_l2", False))

        # Coastal marine stratus forms over sea/bay (0 m ASL) and laps against ridges up to the inversion lid,
        # so do NOT artificially clamp marine stratus base/top above a coastal summit's elevation!
        mbl_floor_m = 0.0 if is_coastal_loc else max(0.0, elev_m - 350.0)

        if low_cf >= 0.03:
            sat_low_top_m = float(adv.get("low_top_asl_m") or 0.0)
            has_sat_low_top = (
                is_cal_l2
                and adv["low_cf"] >= 0.12
                and sat_low_top_m > mbl_floor_m + 95.0
            )
            is_fog_like = (is_coastal_loc or elev_m < 1350.0) and (
                blended_fog_cf >= 0.06
                or st["inv_jump_k"] >= 1.0
                or st["rh_2m_pct"] >= 86.0
                or (has_sat_low_top and sat_low_top_m < mbl_floor_m + 950.0)
            )
            nwp_had_low = int(sounding["cloud_low_pct"]) >= 20
            raw_nwp_base = max(mbl_floor_m + (35.0 if is_fog_like else 220.0), st["lcl_asl_m"])
            if is_coastal_loc and is_fog_like:
                raw_nwp_base = min(raw_nwp_base, max(45.0, float(st["inv_top_asl_m"]) - 160.0))
            nwp_top_m = max(raw_nwp_base + (150.0 if is_fog_like else 480.0), float(st["inv_top_asl_m"]))

            has_raob_lid = bool(st.get("raob_assimilated") and st.get("raob_lid_asl_m") is not None and is_fog_like)
            if has_raob_lid:
                # 00Z/12Z balloon sounding resolves marine inversion lid to ~15m vertical precision
                raob_lid = float(st["raob_lid_asl_m"])
                if has_sat_low_top and abs(sat_low_top_m - raob_lid) <= 220.0:
                    top_m = (1.0 - 0.18 * w_low_eff) * raob_lid + (0.18 * w_low_eff) * sat_low_top_m
                else:
                    top_m = raob_lid
            elif has_sat_low_top:
                w_ht = 0.88 if not nwp_had_low else (0.76 * w_low_eff)
                top_m = (1.0 - w_ht) * nwp_top_m + w_ht * sat_low_top_m
            else:
                top_m = nwp_top_m

            raob_stratus_base = (
                float(raob_obs["saturated_stratus_base_asl_m"])
                if (raob_obs and w_raob >= 0.45 and raob_obs.get("saturated_stratus_base_asl_m") is not None)
                else None
            )
            max_fog_ceil = (top_m - 45.0) if has_raob_lid else (top_m + 180.0)
            if fog_ceil_asl is not None and fog_ceil_asl > mbl_floor_m + 20.0 and (not is_fog_like or fog_ceil_asl <= max_fog_ceil):
                base_m = fog_ceil_asl
                if not has_raob_lid:
                    top_m = max(base_m + 95.0, top_m)
            elif raob_stratus_base is not None and is_fog_like and raob_stratus_base < top_m + 60.0:
                base_m = max(mbl_floor_m + 30.0, min(raob_stratus_base, top_m - 85.0))
            elif has_sat_low_top and (not nwp_had_low or raw_nwp_base > top_m - 110.0):
                deck_thick = 210.0 if is_fog_like else 380.0
                base_m = max(mbl_floor_m + (35.0 if is_fog_like else 140.0), top_m - deck_thick)
            else:
                base_m = max(mbl_floor_m + 35.0, min(raw_nwp_base, top_m - 95.0))

            top_m = max(base_m + 85.0, top_m)
            if is_fog_like:
                st["inv_top_asl_m"] = round(top_m, 1)

            primary_low_cf = max(blended_fog_cf, low_cf * 0.78) if is_fog_like else low_cf
            tau_nwp = primary_low_cf * min(26.0, 0.022 * (top_m - base_m))
            sat_tau_low = float(adv.get("tau_low", adv["tau_total"] * 0.72))
            if adv["low_cf"] > 0.05:
                incloud_low = max(1.2, min(32.0, sat_tau_low / adv["low_cf"]))
                tau_sat = primary_low_cf * incloud_low
            else:
                tau_sat = primary_low_cf * 6.8
            tau_low = (1.0 - w_low_eff) * tau_nwp + w_low_eff * tau_sat
            new_bands.append(
                {
                    "label": "fog" if is_fog_like else "low",
                    "base_asl_m": round(base_m, 1),
                    "top_asl_m": round(top_m, 1),
                    "cloud_fraction": round(primary_low_cf, 3),
                    "optical_depth_vertical": round(max(0.08, tau_low), 2),
                    "aspect_ratio_h_over_d": 0.07 if is_fog_like else 0.28,
                    "characteristic_cell_km": 11.0 if is_fog_like else 3.8,
                    "ice_fraction": 0.0,
                }
            )
            # Emit simultaneous above-inversion cumulus/stratocumulus deck (5-10 kft CCLC CF2 or NWP 850 hPa Sc)
            residual_cu_cf = max(0.0, low_cf - 0.72 * primary_low_cf) if is_fog_like else 0.0
            orig_cu_cf = (
                float(orig_cu_band["cloud_fraction"])
                if (orig_cu_band is not None and float(orig_cu_band["base_asl_m"]) > top_m + 130.0)
                else 0.0
            )
            cu_cf = max(residual_cu_cf, orig_cu_cf * (1.0 - 0.35 * w_low_eff))
            if is_fog_like and cu_cf >= 0.05:
                cu_base_m = max(
                    top_m + 180.0,
                    elev_m + 180.0,
                    float(orig_cu_band["base_asl_m"]) if orig_cu_band is not None else 1550.0,
                )
                cu_top_m = max(
                    cu_base_m + 420.0,
                    float(orig_cu_band["top_asl_m"]) if orig_cu_band is not None else 2450.0,
                )
                new_bands.append(
                    {
                        "label": "low",
                        "base_asl_m": round(cu_base_m, 1),
                        "top_asl_m": round(cu_top_m, 1),
                        "cloud_fraction": round(min(0.92, cu_cf), 3),
                        "optical_depth_vertical": round(max(0.15, cu_cf * 8.2), 2),
                        "aspect_ratio_h_over_d": float(orig_cu_band.get("aspect_ratio_h_over_d", 0.32)) if orig_cu_band else 0.32,
                        "characteristic_cell_km": float(orig_cu_band.get("characteristic_cell_km", 2.8)) if orig_cu_band else 2.8,
                        "ice_fraction": 0.0,
                    }
                )
        if mid_cf >= 0.03:
            sat_mid_top = float(adv.get("mid_top_asl_m") or 4350.0)
            mid_top = max(
                elev_m + 1450.0,
                (1.0 - 0.55 * w_mid_eff) * max(elev_m + 2350.0, 4350.0) + (0.55 * w_mid_eff) * sat_mid_top,
            )
            mid_base = max(elev_m + 750.0, mid_top - 1350.0)
            tau_mid_nwp = mid_cf * 6.2
            sat_tau_mid = float(adv.get("tau_mid", mid_cf * 5.6))
            if adv["mid_cf"] > 0.05:
                tau_mid_sat = mid_cf * max(0.8, min(24.0, sat_tau_mid / adv["mid_cf"]))
            else:
                tau_mid_sat = mid_cf * 5.4
            tau_mid = (1.0 - w_mid_eff) * tau_mid_nwp + w_mid_eff * tau_mid_sat
            new_bands.append(
                {
                    "label": "mid",
                    "base_asl_m": round(mid_base, 1),
                    "top_asl_m": round(mid_top, 1),
                    "cloud_fraction": round(mid_cf, 3),
                    "optical_depth_vertical": round(max(0.06, tau_mid), 2),
                    "aspect_ratio_h_over_d": 0.22,
                    "characteristic_cell_km": 6.5,
                    "ice_fraction": 0.35,
                }
            )
        if high_cf >= 0.03:
            sat_hi_top = float(adv.get("high_top_asl_m") or 10200.0)
            hi_top = max(
                7800.0,
                (1.0 - 0.65 * w_hi_eff) * 10400.0 + (0.65 * w_hi_eff) * sat_hi_top,
            )
            hi_base = max(6600.0, hi_top - 2100.0)
            tau_hi_nwp = high_cf * (0.55 + 1.35 * high_cf)
            sat_tau_hi = float(adv.get("tau_high", tau_hi_nwp))
            if is_cal_l2 and adv["high_cf"] > 0.05:
                # ABI-L2-CODC distinguishes optically thin alpenglow cirrus (COD ~0.6) from opaque anvil (COD ~8)
                incloud_hi = max(0.30, min(14.0, sat_tau_hi / adv["high_cf"]))
                tau_hi_sat = high_cf * incloud_hi
            else:
                tau_hi_sat = tau_hi_nwp
            tau_hi = (1.0 - w_hi_eff) * tau_hi_nwp + w_hi_eff * tau_hi_sat
            new_bands.append(
                {
                    "label": "high",
                    "base_asl_m": round(hi_base, 1),
                    "top_asl_m": round(hi_top, 1),
                    "cloud_fraction": round(high_cf, 3),
                    "optical_depth_vertical": round(max(0.05, tau_hi), 2),
                    "aspect_ratio_h_over_d": 0.08,
                    "characteristic_cell_km": 16.0,
                    "ice_fraction": 1.0,
                }
            )
        rs["cloud_bands"] = new_bands

    return st


def _build_fluid_params_from_sounding(
    st0: dict[str, Any],
    is_coastal: bool,
    seed: int,
) -> dict[str, Any]:
    """Maps a real multi-layer pressure-level sounding + GOES L2 / RAOB cloud bands into unified 3D fluid boundary params."""
    u_sfc, v_sfc = st0["wind_sfc_uv_ms"]
    u_850, v_850 = st0["wind_850_uv_ms"]
    u_500, v_500 = st0["wind_500_uv_ms"]
    u_250, v_250 = st0["wind_250_uv_ms"]
    spd_850 = math.hypot(u_850, v_850)
    spd_700 = float(st0.get("wind_700_kmh", spd_850 * 3.6)) / 3.6

    rs = st0["rust_station"]
    elev_m = float(rs.get("surface_elevation_m", 0.0))
    bands = rs.get("cloud_bands") or []
    fog_band = next((b for b in bands if b.get("label") == "fog"), None)
    low_band = next((b for b in bands if b.get("label") == "low"), None)
    mid_band = next((b for b in bands if b.get("label") == "mid"), None)
    high_band = next((b for b in bands if b.get("label") == "high"), None)

    mbl_floor_m = 0.0 if is_coastal else max(0.0, elev_m - 350.0)
    fog_pct = round(float(fog_band["cloud_fraction"]) * 100.0, 1) if fog_band else 0.0
    default_fog_top = max(mbl_floor_m + 110.0, float(st0.get("inv_top_asl_m", 480.0)))
    default_fog_base = max(mbl_floor_m + 30.0, min(default_fog_top - 95.0, float(st0.get("lcl_asl_m", 180.0))))
    fog_base_m = round(float(fog_band["base_asl_m"]), 1) if fog_band else round(default_fog_base, 1)
    fog_top_m = round(max(fog_base_m + 85.0, float(fog_band["top_asl_m"])), 1) if fog_band else round(default_fog_top, 1)

    if low_band is not None:
        cu_pct = round(float(low_band["cloud_fraction"]) * 100.0, 1)
        low_base_m = round(float(low_band["base_asl_m"]), 1)
        low_top_m = round(max(low_base_m + 280.0, float(low_band["top_asl_m"])), 1)
    else:
        cu_pct = 0.0 if fog_band is not None else float(st0["cloud_low_pct"])
        min_cu_base = max(fog_top_m + 180.0, elev_m + 180.0) if fog_band is not None else (elev_m + 220.0)
        low_base_m = round(max(min_cu_base, float(st0.get("lcl_asl_m", 1100.0))), 1)
        conv_boost_m = min(1600.0, math.sqrt(max(0.0, float(st0.get("cape_j_kg", 0.0)))) * 40.0)
        low_top_m = round(max(low_base_m + 520.0, low_base_m + 450.0 + conv_boost_m), 1)

    mid_pct = round(float(mid_band["cloud_fraction"]) * 100.0, 1) if mid_band else float(st0["cloud_mid_pct"])
    mid_base_m = round(float(mid_band["base_asl_m"]), 1) if mid_band else round(max(low_top_m + 350.0, elev_m + 1100.0, 3150.0), 1)
    mid_top_m = round(max(mid_base_m + 520.0, float(mid_band["top_asl_m"])), 1) if mid_band else round(max(mid_base_m + 1150.0, 4650.0), 1)
    mid_ice_frac = round(float(mid_band.get("ice_fraction", 0.30)), 2) if mid_band else 0.30

    high_pct = round(float(high_band["cloud_fraction"]) * 100.0, 1) if high_band else float(st0["cloud_high_pct"])
    high_base_m = round(float(high_band["base_asl_m"]), 1) if high_band else 7650.0
    high_top_m = round(max(high_base_m + 1150.0, float(high_band["top_asl_m"])), 1) if high_band else 10350.0
    high_tau = round(float(high_band["optical_depth_vertical"]), 2) if high_band else round((high_pct / 100.0) * 1.15, 2)

    raw_inv_k = float(st0.get("raob_inv_strength_k", st0["inv_jump_k"]))
    inv_strength_k = max(
        0.5,
        min(
            16.0,
            raw_inv_k if st0.get("raob_assimilated") else (raw_inv_k + (2.6 if is_coastal else 1.0)),
        ),
    )
    cape_eff = max(
        40.0,
        st0["cape_j_kg"]
        if st0["cape_j_kg"] > 25.0
        else ((120.0 if is_coastal else 210.0) + max(0.0, st0["temp_2m_c"] - 12.0) * 28.0),
    )

    lapse_k_km = float(rs.get("lapse_rate_k_km", 6.2))
    n_bv = math.sqrt(max(4e-5, (9.81 / 285.0) * max(0.0015, (9.8 - lapse_k_km) / 1000.0)))
    barrier_h_m = max(350.0, min(2200.0, elev_m if elev_m > 320.0 else 550.0))
    froude_num = round(max(0.15, min(3.5, max(2.0, spd_700) / max(1.5, n_bv * barrier_h_m))), 2)
    wave_gain = round(
        min(
            1.85,
            0.72
            + max(spd_850, spd_700) * 0.052
            + (0.28 if 0.55 <= froude_num <= 1.65 and spd_700 >= 7.5 else 0.0),
        ),
        2,
    )

    return {
        "surfaceTempC": round(st0["temp_2m_c"], 2),
        "surfaceRhPct": round(st0["rh_2m_pct"], 1),
        "fogCloudPct": fog_pct,
        "fogBaseAslM": fog_base_m,
        "fogTopAslM": fog_top_m,
        "cumulusCloudPct": cu_pct,
        "lowCloudPct": float(st0["cloud_low_pct"]),
        "lowBaseAslM": low_base_m,
        "lowTopAslM": low_top_m,
        "midCloudPct": mid_pct,
        "midBaseAslM": mid_base_m,
        "midTopAslM": mid_top_m,
        "midIceFrac": mid_ice_frac,
        "highCloudPct": high_pct,
        "highBaseAslM": high_base_m,
        "highTopAslM": high_top_m,
        "highOpticalDepth": high_tau,
        "inversionBaseM": round(max(40.0, float(st0.get("raob_inv_base_asl_m") or (fog_base_m if fog_band else st0["inv_base_asl_m"]))), 1),
        "inversionThicknessM": round(max(95.0, fog_top_m - fog_base_m), 1) if fog_band else (180.0 if is_coastal else 300.0),
        "inversionStrengthK": round(inv_strength_k, 2),
        "windSfcU": u_sfc,
        "windSfcV": v_sfc,
        "wind850U": u_850,
        "wind850V": v_850,
        "wind500U": u_500,
        "wind500V": v_500,
        "wind250U": u_250,
        "wind250V": v_250,
        "convectiveCape": round(cape_eff, 1),
        "orographicWaveGain": wave_gain,
        "froudeNumber": froude_num,
        "vorticityConfinement": 1.18,
        "cirrusIceBoost": round(0.75 + 0.52 * min(1.35, high_tau / max(0.25, (high_pct / 100.0) + 0.15)), 2),
        "jet250Mph": int(round(st0["wind_250_kmh"] * 0.621371)),
        "jet300Mph": int(round(st0["wind_300_kmh"] * 0.621371)),
        "seed": int(seed & 0xFFFF),
    }


def _build_atmospheric_layers_provenance(
    st0: dict[str, Any],
    fluid_params: dict[str, Any],
    *,
    is_coastal: bool,
    is_historical: bool,
    lead_hours: float,
    w_nowcast: float,
    has_live_nowcast: bool,
    goes_data: dict[str, Any] | None,
    station_obs: dict[str, Any] | None,
    raob_obs: dict[str, Any] | None = None,
    has_hrrr: bool,
    hrrr_lmh: tuple[int, int, int],
    ecmwf_lmh: tuple[int, int, int],
    aifs_lmh: tuple[int, int, int],
    gfs_lmh: tuple[int, int, int],
    nowcast_lmh: tuple[int, int, int],
    is_historical_truth_replay: bool = False,
    hindcast_lead_days: int = 0,
) -> dict[str, Any]:
    """Builds per-layer altitude, data provenance, and calibrated confidence metadata for 3D UI & cloud hover."""
    rs = st0["rust_station"]
    bands = rs.get("cloud_bands") or []
    band_map = {str(b.get("label")): b for b in bands}

    h_lo, h_mi, h_hi = hrrr_lmh
    e_lo, e_mi, e_hi = ecmwf_lmh
    a_lo, a_mi, a_hi = aifs_lmh
    g_lo, g_mi, g_hi = gfs_lmh
    n_lo, n_mi, n_hi = nowcast_lmh

    raw_sat_name = str(goes_data.get("satellite", "GOES-18")) if goes_data else "GOES-18"
    sat_name = "GOES-19" if "19" in raw_sat_name else "GOES-18"
    is_cal_l2 = bool(goes_data and goes_data.get("is_calibrated_l2"))
    has_cclc_ml = bool(goes_data and (goes_data.get("has_cclc_multilayer") or goes_data.get("has_cclc")))
    has_btd_fog = bool(goes_data and (goes_data.get("has_twilight_btd_fog") or goes_data.get("has_btd_fog")))
    sat_age_min = float(goes_data.get("scan_age_minutes", 8.0)) if goes_data else 0.0
    motion_conf = float(goes_data.get("motion_confidence", 0.72)) if goes_data else 0.0

    has_close_metar = bool(station_obs and float(station_obs.get("distance_km", 999.0)) <= 45.0)
    metar_id = str(station_obs["station_id"]) if station_obs else None
    metar_dist_km = round(float(station_obs["distance_km"]), 1) if station_obs else None
    has_metar_ceil = bool(has_close_metar and station_obs and station_obs.get("lowest_ceiling_asl_m"))

    has_raob = bool(raob_obs and float(raob_obs.get("distance_km", 999.0)) <= 185.0 and w_nowcast >= 0.05)
    raob_id = str(raob_obs["station_id"]) if raob_obs else None
    raob_dist_km = round(float(raob_obs["distance_km"]), 1) if raob_obs else None

    if is_historical_truth_replay and has_live_nowcast and w_nowcast >= 0.12:
        temporal_regime = "historical_observed_truth"
    elif hindcast_lead_days > 0:
        temporal_regime = f"historical_hindcast_day{hindcast_lead_days}"
    elif has_live_nowcast and w_nowcast >= 0.12:
        temporal_regime = "live_nowcast"
    elif is_historical:
        temporal_regime = "historical_archive"
    else:
        temporal_regime = "nwp_forecast"

    # Empirically calibrated base skill vs lead time (calibrated against 384 multi-sensor layer verifications in backtest.py)
    if is_historical and hindcast_lead_days == 0 and not has_live_nowcast:
        base_nwp_skill = 83.5
    else:
        pos_lead = max(0.0, lead_hours)
        base_nwp_skill = 86.5 * math.exp(-pos_lead / 440.0)

    obs_base_skill = (
        (94.0 if (is_cal_l2 and has_cclc_ml) else (91.0 if is_cal_l2 else 84.0))
        * (0.91 + 0.09 * motion_conf)
        * math.exp(-max(0.0, lead_hours) / 22.0)
        if (has_live_nowcast and goes_data is not None)
        else base_nwp_skill
    )

    high_cf_total = float(fluid_params.get("highCloudPct", st0["cloud_high_pct"])) / 100.0

    def _tier(c: int) -> str:
        if c >= 74:
            return "high"
        if c >= 50:
            return "moderate"
        return "low"

    def _layer_conf_and_sources(
        layer_id: str,
        cov_pct: float,
        m_vals: tuple[int, int, int, int],
        now_val: int,
    ) -> tuple[int, list[str], dict[str, int], str]:
        mh, me, ma, mg = m_vals
        spread = max(mh, me, ma, mg) - min(mh, me, ma, mg)
        mean_m = 0.25 * (mh + me + ma + mg)
        std_m = math.sqrt(
            0.25 * ((mh - mean_m) ** 2 + (me - mean_m) ** 2 + (ma - mean_m) ** 2 + (mg - mean_m) ** 2)
        )
        regime_pen = 4.5 if (layer_id == "fog" and is_coastal and w_nowcast < 0.35) else 0.0
        spread_pen = min(20.0, 0.45 * std_m) if is_historical else min(24.0, 0.52 * std_m)
        nwp_conf = base_nwp_skill - spread_pen - regime_pen

        w_obs = 0.0
        if has_live_nowcast and w_nowcast > 0.04:
            if layer_id in ("fog", "low"):
                w_obs = min(0.95, w_nowcast * 1.06)
            elif layer_id == "mid":
                w_obs = min(0.93, w_nowcast * 0.96)
            else:
                w_obs = min(0.91, w_nowcast * 0.88)

        obs_conf = obs_base_skill
        obscured_from_space = False
        if w_obs > 0.05:
            if layer_id in ("fog", "low") and high_cf_total >= 0.62:
                if has_close_metar:
                    obs_conf -= 2.5
                elif has_cclc_ml or has_raob:
                    obs_conf -= 6.5
                    obscured_from_space = True
                else:
                    obs_conf -= 15.0
                    obscured_from_space = True
            elif layer_id == "mid" and high_cf_total >= 0.72:
                obs_conf -= 6.0 if has_cclc_ml else 9.0
                obscured_from_space = True
            if layer_id in ("fog", "low") and has_close_metar:
                obs_conf += 4.5 if has_metar_ceil else 3.0
            if layer_id == "fog" and has_raob:
                obs_conf += 3.5

        raw_conf = w_obs * obs_conf + (1.0 - w_obs) * nwp_conf
        conf_pct = int(round(max(20.0, min(95.0, raw_conf))))

        sources: list[str] = []
        if w_obs >= 0.12 and goes_data is not None:
            if has_cclc_ml:
                btd_tag = "+BTD" if (layer_id == "fog" and has_btd_fog) else ""
                l2_tag = f"ABI L2 CCLC-5Layer{btd_tag}/ACHA/COD"
            elif is_cal_l2:
                l2_tag = "ABI L2 ACTP/ACHA/COD"
            else:
                l2_tag = "ABI IR/Vis"
            sources.append(f"{sat_name} {l2_tag} ({int(round(w_obs * 100))}%)")
        if layer_id in ("fog", "low") and has_close_metar and metar_id:
            ceil_str = "ceilometer" if has_metar_ceil else "ASOS"
            sources.append(f"METAR {metar_id} {ceil_str} ({metar_dist_km:.0f} km)")
        if layer_id in ("fog", "low") and has_raob and raob_id:
            sources.append(f"RAOB {raob_id} Balloon ({raob_dist_km:.0f} km)")
        nwp_w_pct = int(round((1.0 - w_obs) * 100))
        if hindcast_lead_days > 0:
            sources.append(f"Archived +{hindcast_lead_days * 24}h NWP Hindcast (HRRR/IFS/GFS)")
        elif is_historical and nwp_w_pct >= 8:
            sources.append(f"Archived HRRR 3km + ERA5/IFS ({nwp_w_pct}%)")
        elif nwp_w_pct >= 8:
            hrrr_lbl = "NOAA HRRR 3km" if has_hrrr else "NOAA GFS"
            sources.append(f"{hrrr_lbl} + ECMWF IFS/AIFS ({nwp_w_pct}%)")

        breakdown = {
            "hrrr": int(mh),
            "ecmwf_ifs": int(me),
            "ecmwf_aifs": int(ma),
            "gfs": int(mg),
        }
        if has_live_nowcast:
            breakdown["goes_l2"] = int(now_val)

        if w_obs >= 0.45 and goes_data is not None:
            mode_prefix = "Observed truth: " if is_historical_truth_replay else ""
            if obscured_from_space and not has_close_metar:
                reason = (
                    f"{mode_prefix}{sat_name} CCLC under {int(round(high_cf_total * 100))}% cirrus"
                    + (f" + {raob_id} balloon lid" if (has_raob and raob_id) else "")
                    + "; 3D 1m eddies simulated."
                )
            elif layer_id in ("fog", "low") and (has_metar_ceil or has_raob):
                parts_m = []
                if has_cclc_ml:
                    parts_m.append(f"{sat_name} CCLC" + ("+3.9µm BTD" if has_btd_fog else ""))
                else:
                    parts_m.append(f"{sat_name} L2")
                if has_metar_ceil and metar_id:
                    parts_m.append(f"{metar_id} ceilometer base")
                if has_raob and raob_id:
                    parts_m.append(f"{raob_id} balloon inversion")
                reason = f"{mode_prefix}{' + '.join(parts_m)}; 3D 1m billows simulated."
            elif layer_id == "high" and is_cal_l2:
                reason = (
                    f"{mode_prefix}{sat_name} CCLC/ACHAC ice top & COD optical depth; "
                    f"3D ice fallstreaks simulated."
                )
            else:
                reason = (
                    f"{mode_prefix}{sat_name} L2 macro deck & parallax drift ({sat_age_min:.0f}m scan); "
                    f"3D 1m structure simulated."
                )
        elif hindcast_lead_days > 0:
            reason = (
                f"Archived +{hindcast_lead_days * 24}h Day-{hindcast_lead_days} forecast run "
                f"(HRRR {mh}% / IFS {me}% / GFS {mg}%); 3D wisps simulated."
            )
        elif is_historical:
            reason = (
                f"Archived pressure-level reanalysis sounding (±{int(round(std_m))}% model σ); "
                f"3D 1m cloud field simulated."
            )
        else:
            lead_tag = f"+{max(0.0, lead_hours):.0f}h forecast"
            if spread >= 34:
                reason = (
                    f"{lead_tag} • Models split (HRRR {mh}% vs IFS {me}% vs AIFS {ma}%); "
                    f"3D cloud structure simulated."
                )
            elif spread >= 16:
                reason = (
                    f"{lead_tag} • Moderate model spread (±{int(round(std_m))}% σ); "
                    f"macro sounding from NWP, 3D wisps simulated."
                )
            else:
                reason = (
                    f"{lead_tag} • Strong 4-model consensus (HRRR {mh}% / IFS {me}%); "
                    f"macro sounding from NWP, 3D wisps simulated."
                )

        return conf_pct, sources, breakdown, reason

    fog_cov = float(fluid_params["fogCloudPct"])
    cu_cov = float(fluid_params["cumulusCloudPct"])
    mid_cov = float(fluid_params["midCloudPct"])
    hi_cov = float(fluid_params["highCloudPct"])

    fog_conf, fog_src, fog_brk, fog_note = _layer_conf_and_sources(
        "fog", fog_cov, (h_lo, e_lo, a_lo, g_lo), n_lo
    )
    low_conf, low_src, low_brk, low_note = _layer_conf_and_sources(
        "low", cu_cov, (h_lo, e_lo, a_lo, g_lo), n_lo
    )
    mid_conf, mid_src, mid_brk, mid_note = _layer_conf_and_sources(
        "mid", mid_cov, (h_mi, e_mi, a_mi, g_mi), n_mi
    )
    hi_conf, hi_src, hi_brk, hi_note = _layer_conf_and_sources(
        "high", hi_cov, (h_hi, e_hi, a_hi, g_hi), n_hi
    )

    def _tau_of(lbl: str, fallback_cov: float, scale: float) -> float:
        if lbl in band_map:
            return round(float(band_map[lbl].get("optical_depth_vertical", 0.0)), 2)
        return round((fallback_cov / 100.0) * scale, 2)

    froude = float(fluid_params.get("froudeNumber", 1.0))
    mid_name = (
        "Mid Altocumulus & Standing Wave"
        if (0.55 <= froude <= 1.65 and float(st0.get("wind_700_kmh", 0.0)) >= 25.0)
        else "Mid-Tropospheric Altocumulus"
    )

    layers = [
        {
            "id": "high",
            "name": "Upper-Tropospheric Ice Cirrus",
            "short_name": "High Ice Cirrus",
            "phase": "ice",
            "coverage_pct": int(round(hi_cov)),
            "base_asl_m": round(float(fluid_params["highBaseAslM"]), 0),
            "top_asl_m": round(float(fluid_params["highTopAslM"]), 0),
            "base_ft": int(round(float(fluid_params["highBaseAslM"]) * 3.28084)),
            "top_ft": int(round(float(fluid_params["highTopAslM"]) * 3.28084)),
            "optical_depth": _tau_of("high", hi_cov, 1.25),
            "wind_mph": int(round(float(st0.get("wind_250_kmh", 40.0)) * 0.621371)),
            "confidence_pct": hi_conf,
            "confidence_tier": _tier(hi_conf),
            "sources": hi_src,
            "model_breakdown_pct": hi_brk,
            "hover_disclosure": hi_note,
        },
        {
            "id": "mid",
            "name": mid_name,
            "short_name": "Mid Altocumulus / Wave",
            "phase": "mixed" if float(fluid_params.get("midIceFrac", 0.3)) >= 0.2 else "liquid",
            "coverage_pct": int(round(mid_cov)),
            "base_asl_m": round(float(fluid_params["midBaseAslM"]), 0),
            "top_asl_m": round(float(fluid_params["midTopAslM"]), 0),
            "base_ft": int(round(float(fluid_params["midBaseAslM"]) * 3.28084)),
            "top_ft": int(round(float(fluid_params["midTopAslM"]) * 3.28084)),
            "optical_depth": _tau_of("mid", mid_cov, 5.5),
            "wind_mph": int(round(math.hypot(*st0.get("wind_500_uv_ms", (8.0, 0.0))) * 2.23694)),
            "confidence_pct": mid_conf,
            "confidence_tier": _tier(mid_conf),
            "sources": mid_src,
            "model_breakdown_pct": mid_brk,
            "hover_disclosure": mid_note,
        },
        {
            "id": "low",
            "name": "Boundary-Layer Cumulus / Sc",
            "short_name": "Low Cumulus / Sc",
            "phase": "liquid",
            "coverage_pct": int(round(cu_cov)),
            "base_asl_m": round(float(fluid_params["lowBaseAslM"]), 0),
            "top_asl_m": round(float(fluid_params["lowTopAslM"]), 0),
            "base_ft": int(round(float(fluid_params["lowBaseAslM"]) * 3.28084)),
            "top_ft": int(round(float(fluid_params["lowTopAslM"]) * 3.28084)),
            "optical_depth": _tau_of("low", cu_cov, 8.5),
            "wind_mph": int(round(math.hypot(*st0.get("wind_850_uv_ms", (5.0, 0.0))) * 2.23694)),
            "confidence_pct": low_conf,
            "confidence_tier": _tier(low_conf),
            "sources": low_src,
            "model_breakdown_pct": low_brk,
            "hover_disclosure": low_note,
        },
        {
            "id": "fog",
            "name": "Sub-Inversion Marine Stratus / Fog",
            "short_name": "Marine Stratus / Fog",
            "phase": "liquid",
            "coverage_pct": int(round(fog_cov)),
            "base_asl_m": round(float(fluid_params["fogBaseAslM"]), 0),
            "top_asl_m": round(float(fluid_params["fogTopAslM"]), 0),
            "base_ft": int(round(float(fluid_params["fogBaseAslM"]) * 3.28084)),
            "top_ft": int(round(float(fluid_params["fogTopAslM"]) * 3.28084)),
            "optical_depth": _tau_of("fog", fog_cov, 10.5),
            "wind_mph": int(round(float(st0.get("wind_10m_kmh", 12.0)) * 0.621371)),
            "confidence_pct": fog_conf,
            "confidence_tier": _tier(fog_conf),
            "sources": fog_src,
            "model_breakdown_pct": fog_brk,
            "hover_disclosure": fog_note,
        },
    ]

    w_tot = (fog_cov + 5.0) + (cu_cov + 5.0) + (mid_cov + 4.0) + (hi_cov + 5.0)
    overall_conf = int(
        round(
            (
                fog_conf * (fog_cov + 5.0)
                + low_conf * (cu_cov + 5.0)
                + mid_conf * (mid_cov + 4.0)
                + hi_conf * (hi_cov + 5.0)
            )
            / max(1.0, w_tot)
        )
    )

    if temporal_regime == "historical_observed_truth":
        metar_part = f" + {metar_id}" if (has_close_metar and metar_id) else ""
        raob_part = f" + {raob_id} Balloon" if (has_raob and raob_id) else ""
        pipeline_label = f"Observed {sat_name} CCLC{metar_part}{raob_part}"
    elif temporal_regime.startswith("historical_hindcast_day"):
        pipeline_label = f"Archived +{hindcast_lead_days * 24}h Day-{hindcast_lead_days} Forecast (HRRR/IFS/GFS)"
    elif temporal_regime == "live_nowcast":
        metar_part = f" + METAR {metar_id}" if (has_close_metar and metar_id) else ""
        raob_part = f" + {raob_id} Balloon" if (has_raob and raob_id) else ""
        pipeline_label = f"Live {sat_name} ABI L2{metar_part}{raob_part} + 4-Model NWP"
    elif temporal_regime == "historical_archive":
        pipeline_label = "ERA5 / Open-Meteo Pressure-Level Reanalysis"
    else:
        hrrr_part = "HRRR 3km + " if has_hrrr else ""
        pipeline_label = f"{hrrr_part}ECMWF IFS/AIFS + GFS Ensemble (+{max(0.0, lead_hours):.0f}h)"

    return {
        "overall_confidence_pct": overall_conf,
        "overall_confidence_tier": _tier(overall_conf),
        "temporal_regime": temporal_regime,
        "lead_hours": round(lead_hours, 2),
        "nowcast_weight_pct": int(round(w_nowcast * 100.0)),
        "primary_pipeline_label": pipeline_label,
        "metar_station_id": metar_id if has_close_metar else None,
        "metar_distance_km": metar_dist_km if has_close_metar else None,
        "raob_station_id": raob_id if has_raob else None,
        "raob_distance_km": raob_dist_km if has_raob else None,
        "satellite_name": sat_name if (has_live_nowcast and goes_data) else None,
        "satellite_scan_age_min": round(sat_age_min, 1) if (has_live_nowcast and goes_data) else None,
        "simulation_note": (
            "Layer heights, macro coverage & wind shear come from satellite/METAR/RAOB/NWP soundings; "
            "fine 1m billows, wave lenses & cirrus fallstreaks are physically simulated on GPU."
        ),
        "layers": layers,
    }


def _build_16x16_mesoscale_cloud_grid(
    center_lat: float,
    center_lon: float,
    anchor_samples: list[tuple[float, float, float, float, float, float]],
    goes_data: dict[str, Any] | None,
    lead_minutes: float,
    w_nowcast: float,
    wind_850_uv: tuple[float, float],
    wind_500_uv: tuple[float, float],
    wind_250_uv: tuple[float, float],
) -> dict[str, Any]:
    grid_n = 16
    span_m = 160_000.0
    half_km = 80.0
    raw_bytes = bytearray(grid_n * grid_n * 4)
    cells_preview: list[dict[str, Any]] = []

    for gy in range(grid_n):
        ny_km = -half_km + (gy / (grid_n - 1)) * (2.0 * half_km)
        for gx in range(grid_n):
            ex_km = -half_km + (gx / (grid_n - 1)) * (2.0 * half_km)
            w_sum = 0.0
            low_acc = 0.0
            mid_acc = 0.0
            high_acc = 0.0
            fog_acc = 0.0
            for ax_km, ay_km, alow, amid, ahigh, afog in anchor_samples:
                d2 = (ex_km - ax_km) ** 2 + (ny_km - ay_km) ** 2
                w = 1.0 / (d2 + 140.0) + 0.15 * math.exp(-d2 / (2.0 * 48.0 * 48.0))
                w_sum += w
                low_acc += w * alow
                mid_acc += w * amid
                high_acc += w * ahigh
                fog_acc += w * afog

            low_p = low_acc / max(1e-6, w_sum)
            mid_p = mid_acc / max(1e-6, w_sum)
            high_p = high_acc / max(1e-6, w_sum)
            fog_p = fog_acc / max(1e-6, w_sum)

            if goes_data is not None and w_nowcast > 0.03:
                clat, clon = offset_lat_lon_km(center_lat, center_lon, ex_km, ny_km)
                adv = sample_advected_goes_state(
                    goes_data,
                    clat,
                    clon,
                    lead_minutes,
                    wind_850_uv,
                    wind_500_uv,
                    wind_250_uv,
                )
                low_p = (1.0 - w_nowcast) * low_p + w_nowcast * (adv["low_cf"] * 100.0)
                mid_p = (1.0 - w_nowcast) * mid_p + w_nowcast * (adv["mid_cf"] * 100.0)
                high_p = (1.0 - w_nowcast) * high_p + w_nowcast * (adv["high_cf"] * 100.0)
                fog_p = (1.0 - w_nowcast) * fog_p + w_nowcast * (float(adv.get("fog_cf", adv["low_cf"] * 0.85)) * 100.0)

            r_b = max(0, min(255, int(round((low_p / 100.0) * 255.0))))
            g_b = max(0, min(255, int(round((mid_p / 100.0) * 255.0))))
            b_b = max(0, min(255, int(round((high_p / 100.0) * 255.0))))
            a_b = max(0, min(255, int(round((fog_p / 100.0) * 255.0))))
            off = (gy * grid_n + gx) * 4
            raw_bytes[off : off + 4] = bytes((r_b, g_b, b_b, a_b))

            cells_preview.append(
                {
                    "gx": gx,
                    "gy": gy,
                    "east_km": round(ex_km, 1),
                    "north_km": round(ny_km, 1),
                    "low_pct": int(round(max(0.0, min(100.0, low_p)))),
                    "mid_pct": int(round(max(0.0, min(100.0, mid_p)))),
                    "high_pct": int(round(max(0.0, min(100.0, high_p)))),
                    "fog_pct": int(round(max(0.0, min(100.0, fog_p)))),
                }
            )

    return {
        "grid_size": grid_n,
        "span_m": span_m,
        "rgba8_b64": base64.b64encode(bytes(raw_bytes)).decode("ascii"),
        "cells": cells_preview,
    }


def _invoke_rust_corridor_raycast(
    lat: float,
    lon: float,
    target_elev_m: float,
    sun_az_deg: float,
    sun_el_deg: float,
    batch_input: dict[str, Any],
) -> dict[str, Any]:
    """Invokes `./raycaster/target/release/photoplanner corridor-raycast` with `CorridorBatchInput`."""
    DEFAULT_ELEVATION_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f_in:
        json.dump(batch_input, f_in)
        in_path = Path(f_in.name)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_out:
        out_path = Path(f_out.name)

    try:
        cmd = [
            str(RUST_BIN),
            "--elevation-dir",
            str(DEFAULT_ELEVATION_DIR),
            "--recreation-db",
            str(DEFAULT_RECREATION_DB),
            "corridor-raycast",
            "--lat",
            f"{lat:.5f}",
            "--lon",
            f"{lon:.5f}",
            "--target-asl-m",
            f"{target_elev_m:.2f}",
            "--sun-az-deg",
            f"{sun_az_deg:.2f}",
            "--sun-el-deg",
            f"{sun_el_deg:.2f}",
            "--near-grid-m",
            "1800",
            "--az-bins",
            "180",
            "--alt-bins",
            "32",
            "--corridor-json",
            str(in_path),
            "--json-out",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=35)
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        for p in (in_path, out_path):
            if p.exists():
                try:
                    os.remove(p)
                except OSError:
                    pass


# Empirically calibrated nowcast-to-NWP fusion & probabilistic interval parameters
# (verified via `planner/weather/backtest.py` against held-out GOES-18 CCLC/BTD + 00Z RAOB + golden-hour DNI + ASOS)
CALIBRATION_CONFIG: dict[str, float] = {
    "nowcast_decay_hours": 2.65,
    "nowcast_max_ensemble_weight": 0.92,
    "coastal_hrrr_low_weight": 0.28,
    "coastal_ecmwf_low_weight": 0.44,
    "coastal_aifs_low_weight": 0.28,
    "epistemic_base_spread_pct": 5.5,
    "epistemic_lead_slope_pct_per_hr": 1.25,
    "epistemic_model_spread_gain": 0.22,
    "aod_excess_extinction_slope": 2.8,
}


def solve_weather_and_corridor(
    lat: float,
    lon: float,
    date_str: str,
    phase: str = "sunset",
    sun_az_deg: float = 259.1,
    sun_el_deg: float = 0.4,
    target_elev_m: float = 2693.0,
    model_engine: str = "ensemble",
    simulated_now_utc: dt.datetime | None = None,
    calibration_overrides: dict[str, float] | None = None,
    replay_mode: str = "auto",
    enable_multilayer_goes: bool = True,
    enable_raob: bool = True,
) -> bytes:
    cal = {**CALIBRATION_CONFIG, **(calibration_overrides or {})}
    model_engine = (model_engine or "ensemble").lower().strip()
    replay_mode = (replay_mode or "auto").lower().strip()
    is_coastal = _is_coastal_california(lat, lon)
    is_sunset = phase.lower().startswith("sunset") or sun_az_deg >= 180.0

    tz_obj, tz_name = local_tz_for_lon(lon)
    real_now_utc = dt.datetime.now(dt.timezone.utc)
    real_today_local = real_now_utc.astimezone(tz_obj).date()
    try:
        req_date = dt.date.fromisoformat(date_str)
    except ValueError:
        req_date = real_today_local

    noon_local = dt.datetime(req_date.year, req_date.month, req_date.day, 12, 0, tzinfo=tz_obj)
    tz_offset_hr = (noon_local.utcoffset() or dt.timedelta(0)).total_seconds() / 3600.0

    sr_info, ss_info = _find_sunrise_sunset_golden_hours(lat, lon, req_date, tz_offset_hr)
    primary_mins = ss_info[0] if is_sunset else sr_info[0]
    conjugate_mins, conjugate_az, conjugate_el = sr_info if is_sunset else ss_info
    primary_hr_idx = max(0, min(23, int(round(primary_mins / 60.0))))
    conjugate_hr_idx = max(0, min(23, int(round(conjugate_mins / 60.0))))

    target_local_dt = dt.datetime(
        req_date.year,
        req_date.month,
        req_date.day,
        primary_mins // 60,
        primary_mins % 60,
        tzinfo=tz_obj,
    )
    target_utc = target_local_dt.astimezone(dt.timezone.utc)
    is_past_date = req_date < real_today_local

    # Resolve replay/hindcast clock & multi-sensor gating
    hindcast_lead_days = 0
    is_historical_truth_replay = False
    raob_causal = True
    obs_query_utc: dt.datetime | None = None

    if simulated_now_utc is not None:
        now_utc = (
            simulated_now_utc.replace(tzinfo=dt.timezone.utc)
            if simulated_now_utc.tzinfo is None
            else simulated_now_utc.astimezone(dt.timezone.utc)
        )
        obs_query_utc = now_utc
        raob_causal = True
    elif replay_mode == "truth" or (is_past_date and replay_mode == "auto"):
        # Past dates default to archived multi-sensor Observed Truth at golden hour!
        now_utc = target_utc
        obs_query_utc = target_utc
        is_historical_truth_replay = True
        raob_causal = False
        replay_mode = "truth"
    elif replay_mode == "nowcast_1h":
        now_utc = target_utc - dt.timedelta(hours=1.0)
        obs_query_utc = now_utc
        raob_causal = True
    elif replay_mode == "nowcast_3h":
        now_utc = target_utc - dt.timedelta(hours=3.0)
        obs_query_utc = now_utc
        raob_causal = True
    elif replay_mode == "hindcast_24h":
        now_utc = target_utc - dt.timedelta(hours=24.0)
        hindcast_lead_days = 1
    elif replay_mode == "hindcast_72h":
        now_utc = target_utc - dt.timedelta(hours=72.0)
        hindcast_lead_days = 3
    elif replay_mode == "nwp_only":
        now_utc = target_utc - dt.timedelta(hours=12.0)
    else:
        now_utc = real_now_utc
        obs_query_utc = None

    now_local = now_utc.astimezone(tz_obj)
    lead_hours = (target_utc - now_utc).total_seconds() / 3600.0
    is_today_or_near = is_historical_truth_replay or (-1.5 <= lead_hours <= 5.5)

    if replay_mode in ("hindcast_24h", "hindcast_72h", "nwp_only"):
        should_query_live_obs = False
        should_query_raob = False
    else:
        should_query_live_obs = (
            is_historical_truth_replay
            or (req_date == now_local.date())
            or (-2.0 <= lead_hours <= 6.5)
        )
        should_query_raob = bool(enable_raob and should_query_live_obs)

    raob_ref_utc = target_utc if is_historical_truth_replay else now_utc

    # Fetch 24h NWP soundings (+ optional Previous-Runs hindcasts), METAR/ASOS surface obs,
    # 3-frame GOES-18/19 S3 L2 (CCLC+BTD+ACTP/ACHA/COD), and 00Z/12Z RAOB balloon concurrently!
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_nwp = pool.submit(
            fetch_24h_corridor_and_models,
            lat,
            lon,
            req_date.isoformat(),
            sun_az_deg,
            conjugate_az,
            is_coastal,
            hindcast_lead_days,
        )
        f_obs = (
            pool.submit(
                fetch_nearest_station_observation,
                lat,
                lon,
                85.0,
                obs_query_utc,
            )
            if should_query_live_obs
            else None
        )
        f_goes = (
            pool.submit(
                fetch_goes_corridor_nowcast,
                lat,
                lon,
                15.0,
                is_coastal,
                obs_query_utc,
                enable_multilayer_goes,
            )
            if should_query_live_obs
            else None
        )
        f_raob = (
            pool.submit(
                fetch_raob_sounding,
                lat,
                lon,
                raob_ref_utc,
                220.0,
                raob_causal,
            )
            if should_query_raob
            else None
        )
        nwp_bundle = f_nwp.result()
        station_obs = f_obs.result() if f_obs else None
        goes_data = f_goes.result() if f_goes else None
        raob_obs = f_raob.result() if f_raob else None

    if not nwp_bundle["ok"]:
        raise RuntimeError(
            "Upstream NWP sounding service unreachable; refusing to synthesize fake weather. "
            f"Details: {'; '.join(nwp_bundle.get('errors') or ['empty point_docs'])}"
        )

    point_docs = nwp_bundle["point_docs"]
    coords_meta = nwp_bundle["coords_meta"]
    is_historical = (
        bool(nwp_bundle["is_historical"])
        and (simulated_now_utc is None)
        and (replay_mode in ("truth", "nwp_only"))
    )

    # Open-Meteo endpoints return `utc_offset_seconds = -28800` (`GMT-8` standard time year-round)
    # even during Daylight Saving Time (`PDT = GMT-7`). Shift array index `hr` by `(api_off_hr - tz_offset_hr)`
    # (`-1` in March–November PDT, `0` in PST) so civil hour `hr` reads the true UTC-coincident sounding.
    api_off_s = point_docs[0].get("utc_offset_seconds") if point_docs else None
    api_hr_shift = int(round(float(api_off_s) / 3600.0 - tz_offset_hr)) if api_off_s is not None else 0
    primary_api_hr = max(0, min(23, primary_hr_idx + api_hr_shift))

    mm_doc = nwp_bundle.get("multimodel_doc") or {}
    mm_off_s = mm_doc.get("utc_offset_seconds")
    mm_hr_shift = int(round(float(mm_off_s) / 3600.0 - tz_offset_hr)) if mm_off_s is not None else api_hr_shift

    st0_base = extract_hour_sounding(
        point_docs[0],
        primary_api_hr,
        0.0,
        sun_az_deg,
        lat,
        lon,
        aod_550nm=nwp_bundle["aod_0km_24h"][primary_api_hr],
        angstrom_exp=nwp_bundle["angstrom_24h"][primary_api_hr],
        ozone_du=nwp_bundle["ozone_du_24h"][primary_api_hr],
        is_coastal=is_coastal,
    )

    mm_h = mm_doc.get("hourly") or {}

    def _model_lmh_at_hr(
        suffix: str, hr_idx: int, fallback_low: int, fallback_mid: int, fallback_high: int
    ) -> tuple[int, int, int, bool]:
        arr_tot = mm_h.get(f"cloud_cover_{suffix}") or []
        arr_lo = mm_h.get(f"cloud_cover_low_{suffix}") or []
        if not any(v is not None for v in arr_tot) and not any(v is not None for v in arr_lo):
            return fallback_low, fallback_mid, fallback_high, False
        m_idx = max(0, min(23, hr_idx + mm_hr_shift))
        lo = int(round(max(0.0, min(100.0, _safe_arr_val(mm_h, f"cloud_cover_low_{suffix}", m_idx, fallback_low)))))
        mi = int(round(max(0.0, min(100.0, _safe_arr_val(mm_h, f"cloud_cover_mid_{suffix}", m_idx, fallback_mid)))))
        hi = int(round(max(0.0, min(100.0, _safe_arr_val(mm_h, f"cloud_cover_high_{suffix}", m_idx, fallback_high)))))
        return lo, mi, hi, True

    def _model_lmh(suffix: str, fallback_low: int, fallback_mid: int, fallback_high: int) -> tuple[int, int, int, bool]:
        return _model_lmh_at_hr(suffix, primary_hr_idx, fallback_low, fallback_mid, fallback_high)

    hrrr_low, hrrr_mid, hrrr_high, has_hrrr = _model_lmh(
        "ncep_hrrr_conus",
        st0_base["cloud_low_pct"],
        st0_base["cloud_mid_pct"],
        st0_base["cloud_high_pct"],
    )
    ecmwf_low, ecmwf_mid, ecmwf_high, has_ecmwf = _model_lmh(
        "ecmwf_ifs025",
        st0_base["cloud_low_pct"],
        st0_base["cloud_mid_pct"],
        st0_base["cloud_high_pct"],
    )
    aifs_low, aifs_mid, aifs_high, has_aifs = _model_lmh(
        "ecmwf_aifs025_single",
        ecmwf_low,
        ecmwf_mid,
        ecmwf_high,
    )
    gfs_low, gfs_mid, gfs_high, has_gfs = _model_lmh(
        "gfs_seamless",
        st0_base["cloud_low_pct"],
        st0_base["cloud_mid_pct"],
        st0_base["cloud_high_pct"],
    )
    if not has_hrrr and has_gfs:
        hrrr_low, hrrr_mid, hrrr_high = gfs_low, gfs_mid, gfs_high

    lead_minutes_primary = max(0.0, lead_hours * 60.0)
    if goes_data is not None:
        adv0 = sample_advected_goes_state(
            goes_data,
            lat,
            lon,
            min(220.0, lead_minutes_primary),
            st0_base["wind_850_uv_ms"],
            st0_base["wind_500_uv_ms"],
            st0_base["wind_250_uv_ms"],
        )
        nowcast_low = int(round(adv0["low_cf"] * 100.0))
        if station_obs is not None and float(station_obs.get("distance_km", 999.0)) <= 45.0:
            obs_low = float(station_obs.get("low_cloud_fraction", 0.0)) * 100.0
            up_sh0 = max(0.0, min(1.0, ((adv0["high_cf"] + 0.75 * adv0["mid_cf"]) * 100.0 - 45.0) / 40.0))
            if up_sh0 > 0.22 and adv0["low_cf"] < 0.38:
                w_asos0 = (0.56 + 0.32 * up_sh0) * math.exp(-float(station_obs.get("distance_km", 0.0)) / 48.0)
            else:
                w_asos0 = (0.50 if adv0["high_cf"] >= 0.60 else 0.38) * math.exp(-float(station_obs.get("distance_km", 0.0)) / 42.0)
            nowcast_low = int(round((1.0 - w_asos0) * nowcast_low + w_asos0 * obs_low))
        nowcast_mid = int(round(adv0["mid_cf"] * 100.0))
        nowcast_high = int(round(adv0["high_cf"] * 100.0))
        has_live_nowcast = True
    else:
        nowcast_low, nowcast_mid, nowcast_high = hrrr_low, hrrr_mid, hrrr_high
        has_live_nowcast = False

    decay_tau_hr = max(0.5, float(cal["nowcast_decay_hours"]))
    max_ens_w = max(0.0, min(0.96, float(cal["nowcast_max_ensemble_weight"])))
    if has_live_nowcast and is_today_or_near:
        base_w_now = 1.0 / (1.0 + (max(0.0, lead_hours) / decay_tau_hr) ** 1.8)
    else:
        base_w_now = 0.0

    if is_historical_truth_replay and has_live_nowcast and model_engine in ("ensemble", "nowcast"):
        w_nowcast_primary = 0.94
    elif model_engine == "nowcast" and has_live_nowcast:
        w_nowcast_primary = 0.94 if is_today_or_near else 0.55
    elif model_engine == "ensemble":
        w_nowcast_primary = base_w_now * max_ens_w
    else:
        w_nowcast_primary = 0.0

    def _compute_nwp_consensus_for_hr(hr_idx: int, fb_lo: int, fb_mi: int, fb_hi: int) -> tuple[int, int, int]:
        h_lo, h_mi, h_hi, ok_h = _model_lmh_at_hr("ncep_hrrr_conus", hr_idx, fb_lo, fb_mi, fb_hi)
        e_lo, e_mi, e_hi, _ = _model_lmh_at_hr("ecmwf_ifs025", hr_idx, fb_lo, fb_mi, fb_hi)
        a_lo, a_mi, a_hi, _ = _model_lmh_at_hr("ecmwf_aifs025_single", hr_idx, e_lo, e_mi, e_hi)
        g_lo, g_mi, g_hi, ok_g = _model_lmh_at_hr("gfs_seamless", hr_idx, fb_lo, fb_mi, fb_hi)
        if not ok_h and ok_g:
            h_lo, h_mi, h_hi = g_lo, g_mi, g_hi
        if model_engine == "graphcast":
            return a_lo, a_mi, a_hi
        if model_engine == "hrrr":
            return h_lo, h_mi, h_hi
        if model_engine == "ecmwf":
            return e_lo, e_mi, e_hi
        if is_coastal and (ok_h or ok_g):
            w_h_lo = float(cal["coastal_hrrr_low_weight"])
            w_e_lo = float(cal["coastal_ecmwf_low_weight"])
            w_a_lo = float(cal["coastal_aifs_low_weight"])
            c_lo = int(round(w_h_lo * h_lo + w_e_lo * e_lo + w_a_lo * a_lo))
        else:
            c_lo = int(round(0.45 * h_lo + 0.30 * e_lo + 0.25 * a_lo))
        c_mi = int(round(0.45 * h_mi + 0.30 * e_mi + 0.25 * a_mi))
        c_hi = int(round(0.40 * h_hi + 0.35 * e_hi + 0.25 * a_hi))
        return c_lo, c_mi, c_hi

    ens_nwp_low, ens_nwp_mid, ens_nwp_high = _compute_nwp_consensus_for_hr(
        primary_hr_idx, st0_base["cloud_low_pct"], st0_base["cloud_mid_pct"], st0_base["cloud_high_pct"]
    )
    ens_low = int(round((1.0 - w_nowcast_primary) * ens_nwp_low + w_nowcast_primary * nowcast_low))
    ens_mid = int(round((1.0 - w_nowcast_primary) * ens_nwp_mid + w_nowcast_primary * nowcast_mid))
    ens_high = int(round((1.0 - w_nowcast_primary) * ens_nwp_high + w_nowcast_primary * nowcast_high))

    def _apply_nwp_prior_scaling(
        st: dict[str, Any],
        base0_hr: dict[str, Any],
        target_lmh_hr: tuple[int, int, int],
        is_origin: bool,
        pdist_km: float,
    ) -> dict[str, Any]:
        if model_engine == "nowcast":
            return st
        t_lo, t_mi, t_hi = target_lmh_hr
        b_lo = base0_hr["cloud_low_pct"]
        b_mi = base0_hr["cloud_mid_pct"]
        b_hi = base0_hr["cloud_high_pct"]
        if abs(t_lo - b_lo) <= 2 and abs(t_mi - b_mi) <= 3 and abs(t_hi - b_hi) <= 3:
            return st
        spatial_w = 1.0 if (is_origin or model_engine != "ensemble") else math.exp(-pdist_km / 140.0)
        s_copy = json.loads(json.dumps(st))
        raw_r_lo = (t_lo + 5.0) / (b_lo + 5.0)
        raw_r_mi = (t_mi + 5.0) / (b_mi + 5.0)
        raw_r_hi = (t_hi + 5.0) / (b_hi + 5.0)
        r_low = 1.0 + spatial_w * (raw_r_lo - 1.0)
        r_mid = 1.0 + spatial_w * (raw_r_mi - 1.0)
        r_hi = 1.0 + spatial_w * (raw_r_hi - 1.0)
        s_copy["cloud_low_pct"] = t_lo if is_origin else max(0, min(100, int(round(s_copy["cloud_low_pct"] * r_low))))
        s_copy["cloud_mid_pct"] = t_mi if is_origin else max(0, min(100, int(round(s_copy["cloud_mid_pct"] * r_mid))))
        s_copy["cloud_high_pct"] = t_hi if is_origin else max(0, min(100, int(round(s_copy["cloud_high_pct"] * r_hi))))
        for band in s_copy["rust_station"]["cloud_bands"]:
            lbl = band["label"]
            scale = r_low if lbl in ("fog", "low") else (r_mid if lbl == "mid" else r_hi)
            band["cloud_fraction"] = round(max(0.0, min(1.0, band["cloud_fraction"] * scale)), 3)
            band["optical_depth_vertical"] = round(max(0.04, band["optical_depth_vertical"] * scale), 2)
        return s_copy

    hourly_soundings_0km: list[dict[str, Any]] = []
    hourly_corridor_states: list[dict[str, Any]] = []
    hourly_eval_inputs: list[dict[str, Any]] = []
    primary_corridor_stations_ui: list[dict[str, Any]] = []

    for hr in range(24):
        ahr = max(0, min(23, hr + api_hr_shift))
        hr_mins = hr * 60 + 30
        hr_dt_local = dt.datetime(req_date.year, req_date.month, req_date.day, hr, 30, tzinfo=tz_obj)
        hr_lead_hr = (hr_dt_local.astimezone(dt.timezone.utc) - now_utc).total_seconds() / 3600.0
        min_hr_lead = -2.5 if is_historical_truth_replay else -0.8
        if has_live_nowcast and min_hr_lead <= hr_lead_hr <= 4.5:
            eff_lead = abs(hr_lead_hr) if is_historical_truth_replay else max(0.0, hr_lead_hr)
            if is_historical_truth_replay and abs(hr_lead_hr) <= 0.75 and model_engine in ("ensemble", "nowcast"):
                w_hr_now = 0.94
            elif model_engine == "nowcast":
                w_hr_now = 0.94
            elif model_engine == "ensemble":
                w_hr_now = max_ens_w / (1.0 + (eff_lead / decay_tau_hr) ** 1.8)
            else:
                w_hr_now = 0.0
        else:
            w_hr_now = 0.0

        base0_hr = extract_hour_sounding(
            point_docs[0],
            ahr,
            0.0,
            sun_az_deg,
            lat,
            lon,
            aod_550nm=nwp_bundle["aod_0km_24h"][ahr],
            angstrom_exp=nwp_bundle["angstrom_24h"][ahr],
            ozone_du=nwp_bundle["ozone_du_24h"][ahr],
            is_coastal=is_coastal,
        )
        target_lmh_hr = _compute_nwp_consensus_for_hr(
            hr, base0_hr["cloud_low_pct"], base0_hr["cloud_mid_pct"], base0_hr["cloud_high_pct"]
        )

        rust_stations_hr: list[dict[str, Any]] = []
        st_primary_list_hr: list[dict[str, Any]] = []

        for idx_pt in range(9):
            _, plat, plon, pdist_km, paz_deg = coords_meta[idx_pt]
            aod_interp = (
                nwp_bundle["aod_0km_24h"][ahr]
                if pdist_km < 50.0
                else nwp_bundle["aod_100km_24h"][ahr]
            )
            pt_is_coastal = _is_coastal_california(plat, plon)
            snd = (
                base0_hr
                if idx_pt == 0
                else extract_hour_sounding(
                    point_docs[idx_pt],
                    ahr,
                    pdist_km,
                    paz_deg,
                    plat,
                    plon,
                    aod_550nm=aod_interp,
                    angstrom_exp=nwp_bundle["angstrom_24h"][ahr],
                    ozone_du=nwp_bundle["ozone_du_24h"][ahr],
                    is_coastal=pt_is_coastal,
                )
            )
            snd = _apply_nwp_prior_scaling(snd, base0_hr, target_lmh_hr, is_origin=(idx_pt == 0), pdist_km=pdist_km)
            snd = _blend_nowcast_into_station(
                snd,
                goes_data,
                station_obs if idx_pt == 0 else None,
                max(0.0, hr_lead_hr * 60.0),
                w_hr_now,
                raob_obs=raob_obs,
                is_coastal_loc=pt_is_coastal,
            )
            rust_stations_hr.append(snd["rust_station"])
            if idx_pt < 6:
                st_primary_list_hr.append(snd)

        hourly_soundings_0km.append(st_primary_list_hr[0])
        if hr == primary_hr_idx:
            primary_corridor_stations_ui = st_primary_list_hr

        corr_state_hr = {
            "azimuth_center_deg": round(sun_az_deg, 2),
            "max_corridor_km": 380.0,
            "stations": rust_stations_hr,
        }
        hourly_corridor_states.append(corr_state_hr)

        if hr == conjugate_hr_idx:
            eval_az, eval_el = conjugate_az, conjugate_el
            eval_time_str = f"{conjugate_mins // 60:02d}:{conjugate_mins % 60:02d}"
            inc_lut = True
        elif hr == primary_hr_idx:
            eval_az, eval_el = sun_az_deg, sun_el_deg
            eval_time_str = f"{primary_mins // 60:02d}:{primary_mins % 60:02d}"
            inc_lut = False
        else:
            eval_az, eval_el = _solar_ephemeris_local(lat, lon, req_date, float(hr_mins), tz_offset_hr)
            eval_time_str = f"{hr:02d}:30"
            inc_lut = False

        hourly_eval_inputs.append(
            {
                "hour_index": hr,
                "time_local": eval_time_str,
                "sun_az_deg": eval_az,
                "sun_el_deg": eval_el,
                "corridor": corr_state_hr,
                "include_lut": inc_lut,
            }
        )

    primary_corridor_state = hourly_corridor_states[primary_hr_idx]
    batch_input = {
        "corridor": primary_corridor_state,
        "hourly_evals": hourly_eval_inputs,
    }

    rust_out = _invoke_rust_corridor_raycast(
        lat, lon, target_elev_m, sun_az_deg, sun_el_deg, batch_input
    )

    primary_ray = rust_out["primary_ray"]
    overhead_canvas = rust_out["overhead_canvas"]
    corridor_lut = rust_out["corridor_lut"]
    corridor_lut["has_weather"] = True
    hourly_rust_results = rust_out.get("hourly_results") or []

    conjugate_lut: dict[str, Any] | None = None
    if conjugate_hr_idx < len(hourly_rust_results):
        conj_b64 = hourly_rust_results[conjugate_hr_idx].get("lut_rgba8_b64")
        if conj_b64:
            conjugate_lut = {
                **corridor_lut,
                "sun_elevation_deg": conjugate_el,
                "rgba8_b64": conj_b64,
                "has_weather": True,
            }

    sunset_corridor_lut = corridor_lut if is_sunset else (conjugate_lut or corridor_lut)
    sunrise_corridor_lut = (conjugate_lut or corridor_lut) if is_sunset else corridor_lut

    meso_anchors: list[tuple[float, float, float, float, float, float]] = []
    m_per_lat = 111.13292
    m_per_lon = max(22.0, 111.41284 * math.cos(math.radians(lat)))
    for idx_pt, (_, plat, plon, pdist_km, paz_deg) in enumerate(coords_meta):
        snd_pt = extract_hour_sounding(
            point_docs[idx_pt],
            primary_api_hr,
            pdist_km,
            paz_deg,
            plat,
            plon,
            aod_550nm=nwp_bundle["aod_0km_24h"][primary_api_hr],
            angstrom_exp=nwp_bundle["angstrom_24h"][primary_api_hr],
            ozone_du=nwp_bundle["ozone_du_24h"][primary_api_hr],
            is_coastal=_is_coastal_california(plat, plon),
        )
        ex_km = (plon - lon) * m_per_lon
        ny_km = (plat - lat) * m_per_lat
        fog_pct = snd_pt["cloud_low_pct"] if snd_pt["inv_jump_k"] >= 0.8 else snd_pt["cloud_low_pct"] * 0.45
        meso_anchors.append(
            (
                ex_km,
                ny_km,
                float(snd_pt["cloud_low_pct"]),
                float(snd_pt["cloud_mid_pct"]),
                float(snd_pt["cloud_high_pct"]),
                float(fog_pct),
            )
        )

    st0 = primary_corridor_stations_ui[0]
    mesoscale_cloud_grid = _build_16x16_mesoscale_cloud_grid(
        lat,
        lon,
        meso_anchors,
        goes_data,
        min(220.0, lead_minutes_primary),
        w_nowcast_primary,
        st0["wind_850_uv_ms"],
        st0["wind_500_uv_ms"],
        st0["wind_250_uv_ms"],
    )

    spread_low_models = max(hrrr_low, ecmwf_low, aifs_low) - min(hrrr_low, ecmwf_low, aifs_low)
    spread_high_models = max(hrrr_high, ecmwf_high, aifs_high) - min(hrrr_high, ecmwf_high, aifs_high)
    model_disagreement_pct = 0.65 * spread_low_models + 0.35 * spread_high_models
    epistemic_sigma_pct = max(
        0.0,
        min(
            32.0,
            float(cal["epistemic_base_spread_pct"])
            + float(cal["epistemic_lead_slope_pct_per_hr"]) * min(18.0, max(0.0, lead_hours))
            + float(cal["epistemic_model_spread_gain"]) * model_disagreement_pct,
        ),
    )

    primary_col_aod = (
        0.65 * float(st0["rust_station"]["aod_550nm"])
        + 0.35 * float(nwp_bundle["aod_100km_24h"][primary_hr_idx])
    )

    def _normalize_corridor_ray_pct(
        ray_doc: dict[str, Any], col_aod_550: float = primary_col_aod
    ) -> tuple[int, int, int, int, int]:
        c_rgb = ray_doc.get("clear_sky_rgb") or [0.0, 0.0, 0.0]
        clear_photopic = 0.24 * float(c_rgb[0]) + 0.68 * float(c_rgb[1]) + 0.08 * float(c_rgb[2])
        aer_slope = float(cal["aod_excess_extinction_slope"])
        aer_excess_trans = math.exp(-aer_slope * max(0.0, col_aod_550 - 0.065))
        direct_prob = max(0.0, min(1.0, float(ray_doc.get("direct_beam_probability") or 0.0)))
        terr_vis = max(0.0, min(1.0, float(ray_doc.get("terrain_disk_visibility", 1.0))))

        if clear_photopic > 1e-11:
            scale = aer_excess_trans / clear_photopic
            exp_rel = float(ray_doc.get("transmittance_expected") or 0.0) * scale
            p10_rel = float(ray_doc.get("transmittance_p10") or 0.0) * scale
            p50_rel = float(ray_doc.get("transmittance_p50") or 0.0) * scale
            p90_rel = float(ray_doc.get("transmittance_p90") or 0.0) * scale
        else:
            exp_rel = direct_prob * aer_excess_trans
            p10_rel = (1.0 if direct_prob > 0.92 else 0.0) * aer_excess_trans
            p50_rel = (1.0 if direct_prob >= 0.50 else 0.0) * aer_excess_trans
            p90_rel = (1.0 if direct_prob >= 0.12 else 0.0) * aer_excess_trans

        e_f = max(0.0, min(99.0, exp_rel * 100.0))
        q10_sub = max(0.0, min(99.0, p10_rel * 100.0))
        q50_sub = max(q10_sub, min(99.0, p50_rel * 100.0))
        q90_sub = max(q50_sub, min(99.0, p90_rel * 100.0))

        if terr_vis > 0.05 and epistemic_sigma_pct > 0.05:
            epi_eff = epistemic_sigma_pct * terr_vis
            center_q = 0.55 * q50_sub + 0.45 * e_f
            lo_half = math.hypot(max(0.0, center_q - q10_sub), epi_eff)
            hi_half = math.hypot(max(0.0, q90_sub - center_q), epi_eff)
            q10 = int(round(max(0.0, min(q50_sub, center_q - lo_half))))
            q50 = int(round(max(q10, min(99.0, center_q))))
            q90 = int(round(max(q50, min(99.0, center_q + hi_half))))
        else:
            q10 = int(round(q10_sub))
            q50 = int(round(q50_sub))
            q90 = int(round(q90_sub))

        e_pct = int(round(e_f))
        pr_pct = int(round(direct_prob * 100.0))
        return e_pct, pr_pct, q10, q50, q90

    exp_trans_pct, direct_prob_pct, p10_pct, p50_pct, p90_pct = _normalize_corridor_ray_pct(
        primary_ray, primary_col_aod
    )
    kelvin = int(round(float(primary_ray["beam_color_kelvin"])))
    chappuis_rose = round(float(primary_ray["chappuis_rose_index"]), 3)
    dom_blk = primary_ray.get("dominant_blocker") or {}
    blocker_label = str(dom_blk.get("blocker_type") or "clear_corridor")
    blocker_dist_km = round(float(dom_blk.get("distance_km") or 0.0), 1)
    blocker_alt_m = round(float(dom_blk.get("altitude_asl_m") or 0.0), 0)
    blocker_summary = str(dom_blk.get("summary") or "Unobstructed 380 km corridor")

    canvas_glow_score = int(round(float(overhead_canvas.get("composite_canvas_score") or 55.0)))
    dominant_glow_band = str(overhead_canvas.get("dominant_glow_band") or "high")

    temp_sunset_f = st0["temp_2m_c"] * 9.0 / 5.0 + 32.0
    sr_sounding = hourly_soundings_0km[max(0, min(23, sr_info[0] // 60))]
    temp_sunrise_f = int(round(sr_sounding["temp_2m_c"] * 9.0 / 5.0 + 32.0))
    temp_950_f = round(st0["temp_950_c"] * 9.0 / 5.0 + 32.0, 1)
    temp_925_f = round(st0["temp_925_c"] * 9.0 / 5.0 + 32.0, 1)
    temp_850_f = round(st0["temp_850_c"] * 9.0 / 5.0 + 32.0, 1)
    raw_plev_delta_f = (max(temp_950_f, temp_925_f) - temp_sunset_f) if is_coastal else (temp_850_f - temp_sunset_f)
    if st0.get("raob_assimilated") and st0.get("raob_inv_strength_k") is not None:
        w_r0 = float(st0.get("raob_weight", 0.75))
        raob_delta_f = float(st0["raob_inv_strength_k"]) * 1.8
        inversion_delta_t_f = round((1.0 - w_r0) * raw_plev_delta_f + w_r0 * raob_delta_f, 1)
    else:
        inversion_delta_t_f = round(raw_plev_delta_f, 1)

    lcl_cloud_base_ft = int(round(st0["lcl_asl_m"] * 3.28084))
    fog_band = next((b for b in st0["rust_station"]["cloud_bands"] if b["label"] == "fog"), None)
    if fog_band is None:
        fog_band = next((b for b in st0["rust_station"]["cloud_bands"] if b["label"] == "low"), None)
    if fog_band:
        fog_base_ft = int(round(float(fog_band["base_asl_m"]) * 3.28084))
        fog_top_ft = int(round(float(fog_band["top_asl_m"]) * 3.28084))
    else:
        fog_top_ft = int(round(st0["inv_top_asl_m"] * 3.28084))
        fog_base_ft = max(120, fog_top_ft - (650 if is_coastal else 900))

    target_elev_ft = int(round(target_elev_m * 3.28084))
    summit_clearance_ft = target_elev_ft - fog_top_ft
    pbl_height_ft = int(round(st0["pbl_agl_m"] * 3.28084))
    vis_miles = round(max(0.2, min(48.0, st0["visibility_m"] / 1609.344)), 1)
    wind_mph = int(round(st0["wind_10m_kmh"] * 0.621371))
    wind_dir_deg = int(round(st0["wind_10m_dir_deg"]))

    offshore_low_pct = (
        int(primary_corridor_stations_ui[1]["cloud_low_pct"])
        if (is_coastal and len(primary_corridor_stations_ui) > 1)
        else int(st0["cloud_low_pct"])
    )
    sea_deck_pct = max(int(st0["cloud_low_pct"]), offshore_low_pct) if is_coastal else int(st0["cloud_low_pct"])

    inversion_cap_score = min(45.0, max(0.0, (inversion_delta_t_f + 2.0) * 4.0))
    stratus_deck_score = (
        min(42.0, sea_deck_pct * 0.42)
        if summit_clearance_ft > 80
        else max(5.0, 22.0 - abs(summit_clearance_ft) * 0.03)
    )
    cloud_sea_score = int(
        round(max(10.0, min(98.0, inversion_cap_score + stratus_deck_score + (12.0 if st0["cloud_mid_pct"] < 40 else 3.0))))
    )
    under_deck_sunset_score = int(
        round(
            max(
                10.0,
                min(
                    97.0,
                    0.55 * p90_pct + 0.42 * canvas_glow_score - 0.12 * max(0, st0["cloud_low_pct"] - 75),
                ),
            )
        )
    )

    beam_quality = 0.58 * exp_trans_pct + 0.27 * p90_pct + 0.15 * direct_prob_pct
    effective_canvas = max(
        float(canvas_glow_score),
        0.82 * beam_quality if (st0["cloud_high_pct"] + st0["cloud_mid_pct"]) < 25 else float(canvas_glow_score),
    )
    alpenglow_score = int(
        round(
            max(
                12.0,
                min(
                    98.0,
                    max(
                        0.58 * beam_quality + 0.32 * effective_canvas + min(10.0, chappuis_rose * 18.0),
                        cloud_sea_score * 0.94 if (is_coastal and summit_clearance_ft > 140 and sea_deck_pct >= 50) else 0.0,
                    ),
                ),
            )
        )
    )

    inversion_chance_pct = int(
        round(
            max(
                5.0,
                min(
                    97.0,
                    (st0["rh_2m_pct"] - 45.0) * 0.80
                    + sea_deck_pct * 0.50
                    + max(0.0, inversion_delta_t_f * 3.4),
                ),
            )
        )
    )

    nowcast_tag = ""
    if is_historical_truth_replay and w_nowcast_primary > 0.12 and goes_data is not None:
        raob_lbl = f" + {raob_obs['station_id']}" if (raob_obs and raob_obs.get("distance_km", 999.0) <= 185.0) else ""
        nowcast_tag = f" [Observed GOES-18 + {station_obs['station_id'] if station_obs else 'ABI'}{raob_lbl} Truth]"
    elif hindcast_lead_days > 0:
        nowcast_tag = f" [+{hindcast_lead_days * 24}h Historical Hindcast]"
    elif w_nowcast_primary > 0.12 and goes_data is not None:
        nowcast_tag = f" [Live GOES-18 + {station_obs['station_id'] if station_obs else 'ABI'} Nowcast]"

    if float(primary_ray.get("terrain_disk_visibility", 1.0)) <= 0.05 and not (is_coastal and sea_deck_pct >= 55 and summit_clearance_ft > 130 and inversion_delta_t_f >= 1.8):
        verdict = f"Topographic Ridge Occlusion ({blocker_dist_km:.0f} km){nowcast_tag}"
        recommendation = (
            f"At El {sun_el_deg:+.1f}°, terrain at {blocker_dist_km:.1f} km ({int(blocker_alt_m * 3.28084):,} ft ASL) "
            f"blocks the direct solar disk, while `{dominant_glow_band}` clouds score {canvas_glow_score}/100 for twilight afterglow."
        )
    elif is_coastal and sea_deck_pct >= 55 and summit_clearance_ft > 130 and inversion_delta_t_f >= 1.8:
        verdict = f"Marine Cloud-Sea Inversion (+{inversion_delta_t_f}°F Cap){nowcast_tag}"
        recommendation = (
            f"Capped marine stratus ({fog_base_ft:,}–{fog_top_ft:,} ft ASL, {sea_deck_pct}% offshore deck) sits {summit_clearance_ft:,} ft below summit. "
            f"380 km ray corridor delivers {direct_prob_pct}% direct-beam odds (P10 {p10_pct}% / P50 {p50_pct}% / P90 {p90_pct}%) at {kelvin} K."
        )
    elif exp_trans_pct >= 58 and canvas_glow_score >= 62:
        verdict = f"High-Canvas Alpenglow Corridor ({kelvin} K){nowcast_tag}"
        recommendation = (
            f"380 km curved-Earth corridor yields {direct_prob_pct}% clear-beam survival (P50 {p50_pct}% / P90 {p90_pct}% transmittance, {kelvin} K Chappuis tint) "
            f"under a {canvas_glow_score}/100 overhead cloud reflector canvas."
        )
    elif exp_trans_pct >= 50:
        verdict = f"Clean 380 km Solar Beam Corridor ({kelvin} K){nowcast_tag}"
        recommendation = (
            f"Unobstructed solar corridor ({direct_prob_pct}% direct sunbeam probability, P50 {p50_pct}% / P90 {p90_pct}%, AOD {st0['rust_station']['aod_550nm']:.2f}) "
            f"casting crisp {kelvin} K golden-hour light onto the landmark."
        )
    elif blocker_label == "aerosol_haze":
        blk_ft = int(round(blocker_alt_m * 3.28084))
        verdict = f"Boundary-Layer Smoke/Haze Extinction ({kelvin} K){nowcast_tag}"
        recommendation = (
            f"Elevated boundary-layer aerosol/haze around {blocker_dist_km:.0f} km ({blk_ft:,} ft ASL, AOD {st0['rust_station']['aod_550nm']:.2f}) "
            f"dims direct beam to {exp_trans_pct}% (P50 {p50_pct}% / P90 {p90_pct}%) with deep {kelvin} K reddening."
        )
    else:
        blk_ft = int(round(blocker_alt_m * 3.28084))
        verdict = f"Corridor Cloud Attenuation @ {blocker_dist_km:.0f} km{nowcast_tag}"
        recommendation = (
            f"Primary attenuation from `{blocker_label}` deck at {blocker_dist_km:.0f} km ({blk_ft:,} ft ASL): "
            f"bimodal sunbeam odds {direct_prob_pct}% (cloudy P10 {p10_pct}% · median P50 {p50_pct}% · sun-break P90 {p90_pct}%)."
        )

    el_rad = math.radians(max(0.05, sun_el_deg))
    corridor_stations_6: list[dict[str, Any]] = []
    for st_s in primary_corridor_stations_ui:
        rs_s = st_s["rust_station"]
        d_km = float(rs_s["distance_km"])
        d_m = d_km * 1000.0
        ray_alt_m = target_elev_m + d_m * math.tan(el_rad) + (0.85 * d_m * d_m) / (2.0 * 6_371_000.0)
        tau_gate = 0.0
        for b in rs_s["cloud_bands"]:
            if b["base_asl_m"] - 250.0 <= ray_alt_m <= b["top_asl_m"] + 250.0:
                tau_gate += b["optical_depth_vertical"] * 0.65
        gate_pass_pct = int(round(max(3.0, min(99.0, 100.0 * math.exp(-tau_gate)))))
        corridor_stations_6.append(
            {
                "distance_km": int(round(d_km)),
                "distance_miles": round(d_km * 0.621371, 1),
                "lat": rs_s["lat"],
                "lon": rs_s["lon"],
                "surface_elevation_m": rs_s["surface_elevation_m"],
                "ray_altitude_m": round(ray_alt_m, 1),
                "ray_altitude_ft": int(round(ray_alt_m * 3.28084)),
                "cloud_low_pct": st_s["cloud_low_pct"],
                "cloud_mid_pct": st_s["cloud_mid_pct"],
                "cloud_high_pct": st_s["cloud_high_pct"],
                "aod_550nm": rs_s["aod_550nm"],
                "gate_transmittance_pct": gate_pass_pct,
                "cloud_bands": rs_s["cloud_bands"],
            }
        )

    st_far = corridor_stations_6[4]
    st_mid = corridor_stations_6[1]
    st_near = corridor_stations_6[0]
    ray_path_stations = [
        {
            **st_far,
            "id": "entry_190km",
            "name": "Far-Corridor Entry (190 km)",
            "region": "Offshore Pacific Limb (190 km)" if (is_coastal and is_sunset) else f"{_compass_cardinal(sun_az_deg)} Upper Corridor (190 km)",
            "role": "190 km Curved-Earth Gate",
        },
        {
            **st_mid,
            "id": "notch_15km",
            "name": "Keyhole / Offshore Gate (15 km)",
            "region": "Gulf of the Farallones (15 km)" if (is_coastal and is_sunset) else f"{_compass_cardinal(sun_az_deg)} Ridge Gate (15 km)",
            "role": "15 km Keyhole Gate",
        },
        {
            **st_near,
            "id": "target_0km",
            "name": "Target Overhead Canvas",
            "region": "Landmark & Overhead Cloud Deck",
            "canvas_score": canvas_glow_score,
            "role": "Alpenglow Sky Canvas",
        },
    ]

    transect_indices = [9, 10, 0, 11, 12]
    transect_labels_coastal = ["11km W Offshore", "5.5km W Coast", "0km Ridge/Peak", "5.5km E Slope", "11km E SF Bay"]
    transect_labels_alpine = ["11km W Gorge", "5.5km W Valley", "0km Monolith", "5.5km E Rim", "11km E Crest"]
    transect_west_to_east: list[dict[str, Any]] = []

    for pos_i, pt_idx in enumerate(transect_indices):
        _, tlat, tlon, tdist, taz = coords_meta[pt_idx]
        t_coastal = _is_coastal_california(tlat, tlon)
        tsnd = extract_hour_sounding(
            point_docs[pt_idx],
            primary_hr_idx,
            tdist,
            taz,
            tlat,
            tlon,
            aod_550nm=nwp_bundle["aod_0km_24h"][primary_hr_idx],
            angstrom_exp=nwp_bundle["angstrom_24h"][primary_hr_idx],
            ozone_du=nwp_bundle["ozone_du_24h"][primary_hr_idx],
            is_coastal=t_coastal,
        )
        tsnd = _blend_nowcast_into_station(
            tsnd,
            goes_data,
            station_obs if pt_idx == 0 else None,
            min(210.0, lead_minutes_primary),
            w_nowcast_primary,
            raob_obs=raob_obs,
            is_coastal_loc=t_coastal,
        )
        east_km_val = [-11.0, -5.5, 0.0, 5.5, 11.0][pos_i]
        t_fog_ft = int(round(tsnd["inv_top_asl_m"] * 3.28084))
        p_mm = tsnd["precip_mm"]
        dbz = round(max(12.0, min(56.0, 10.0 * math.log10(200.0 * (p_mm ** 1.6)))), 1) if p_mm >= 0.04 else 0.0
        transect_west_to_east.append(
            {
                "label": transect_labels_coastal[pos_i] if is_coastal else transect_labels_alpine[pos_i],
                "east_km": east_km_val,
                "lon": tlon,
                "low_cloud_pct": tsnd["cloud_low_pct"],
                "mid_cloud_pct": tsnd["cloud_mid_pct"],
                "high_cloud_pct": tsnd["cloud_high_pct"],
                "rh_1000_pct": int(round(tsnd["rh_2m_pct"])),
                "rh_975_pct": int(round(tsnd["rh_975_pct"])),
                "rh_950_pct": int(round(tsnd["rh_950_pct"])),
                "fog_top_ft": t_fog_ft,
                "fog_top_m": round(tsnd["inv_top_asl_m"], 1),
                "precip_mm_hr": p_mm,
                "radar_dbz": dbz,
            }
        )

    voxel_cells: list[dict[str, Any]] = []
    for gy in range(8):
        ny_m = -9200.0 + (gy / 7.0) * 18400.0
        for gx in range(8):
            ex_m = -9200.0 + (gx / 7.0) * 18400.0
            fx = max(0.0, min(4.0, (ex_m + 11000.0) / 5500.0))
            i0 = min(3, int(fx))
            tx = fx - i0
            tr0, tr1 = transect_west_to_east[i0], transect_west_to_east[i0 + 1]
            low_v = (1.0 - tx) * tr0["low_cloud_pct"] + tx * tr1["low_cloud_pct"]
            mid_v = (1.0 - tx) * tr0["mid_cloud_pct"] + tx * tr1["mid_cloud_pct"]
            high_v = (1.0 - tx) * tr0["high_cloud_pct"] + tx * tr1["high_cloud_pct"]
            top_m = (1.0 - tx) * tr0["fog_top_m"] + tx * tr1["fog_top_m"]

            if goes_data is not None and w_nowcast_primary > 0.05:
                clat, clon = offset_lat_lon_km(lat, lon, ex_m / 1000.0, ny_m / 1000.0)
                adv_c = sample_advected_goes_state(
                    goes_data,
                    clat,
                    clon,
                    min(210.0, lead_minutes_primary),
                    st0["wind_850_uv_ms"],
                    st0["wind_500_uv_ms"],
                    st0["wind_250_uv_ms"],
                )
                low_v = (1.0 - 0.5 * w_nowcast_primary) * low_v + 0.5 * w_nowcast_primary * (adv_c["low_cf"] * 100.0)
                mid_v = (1.0 - 0.5 * w_nowcast_primary) * mid_v + 0.5 * w_nowcast_primary * (adv_c["mid_cf"] * 100.0)
                high_v = (1.0 - 0.5 * w_nowcast_primary) * high_v + 0.5 * w_nowcast_primary * (adv_c["high_cf"] * 100.0)

            tau_col = round(min(6.5, max(0.04, 0.025 * low_v + 0.018 * mid_v + 0.007 * high_v)), 2)
            col_trans = int(round(max(2.0, min(99.0, 100.0 * math.exp(-tau_col * 0.65)))))
            voxel_cells.append(
                {
                    "gx": gx,
                    "gy": gy,
                    "east_m": round(ex_m, 1),
                    "north_m": round(ny_m, 1),
                    "stratus_pct": int(round(max(0.0, min(100.0, low_v)))),
                    "fog_base_m": round(max(35.0, top_m - 240.0), 1),
                    "fog_top_m": round(top_m, 1),
                    "fog_top_ft": int(round(top_m * 3.28084)),
                    "cumulus_pct": int(round(max(0.0, min(100.0, low_v * 0.75)))),
                    "cumulus_alt_m": round(top_m + 350.0, 1),
                    "altocumulus_pct": int(round(max(0.0, min(100.0, mid_v)))),
                    "alto_alt_m": 3400.0,
                    "cirrus_pct": int(round(max(0.0, min(100.0, high_v)))),
                    "cirrus_alt_m": 8800.0,
                    "optical_depth_tau": tau_col,
                    "transmittance_pct": col_trans,
                    "precip_mm_hr": tr0["precip_mm_hr"],
                    "radar_dbz": tr0["radar_dbz"],
                }
            )

    mean_tau = sum(c["optical_depth_tau"] for c in voxel_cells) / max(1, len(voxel_cells))
    crepuscular_gaps = []
    for cand in sorted(
        [c for c in voxel_cells if c["east_m"] <= 2800.0],
        key=lambda c: c["optical_depth_tau"],
    )[:4]:
        crepuscular_gaps.append(
            {
                "gx": cand["gx"],
                "gy": cand["gy"],
                "east_m": round(max(-3200.0, min(-320.0, cand["east_m"] * 0.36 - 220.0)), 1),
                "north_m": round(cand["north_m"] * 0.38, 1),
                "gap_alt_m": round(cand["fog_top_m"] + 180.0, 1),
                "radius_m": 560.0,
                "optical_depth_tau": round(min(0.55, cand["optical_depth_tau"] * 0.38), 2),
                "transmittance_pct": max(32, min(95, cand["transmittance_pct"] + 18)),
            }
        )

    spatial_cloud_field = {
        "grid_size": 8,
        "span_m": 18400.0,
        "wind_dir_deg": wind_dir_deg,
        "wind_speed_mph": wind_mph,
        "mean_optical_depth_tau": round(mean_tau, 2),
        "is_thick_overcast_with_gaps": bool(mean_tau >= 1.45),
        "crepuscular_gaps": crepuscular_gaps,
        "west_offshore_low_pct": transect_west_to_east[0]["low_cloud_pct"],
        "coast_ridge_low_pct": transect_west_to_east[1]["low_cloud_pct"],
        "summit_low_pct": transect_west_to_east[2]["low_cloud_pct"],
        "east_leeward_low_pct": transect_west_to_east[4]["low_cloud_pct"],
        "max_radar_dbz": max((c["radar_dbz"] for c in voxel_cells), default=0.0),
        "transect_west_to_east": transect_west_to_east,
        "cells": voxel_cells,
    }

    def _score_col(lo: int, mi: int, hi: int) -> tuple[float, int]:
        tau_m = round(min(6.5, max(0.05, 0.026 * lo + 0.017 * mi + 0.007 * hi)), 2)
        sc = int(round(max(24.0, min(97.0, 84.0 - lo * 0.32 - mi * 0.14 + (14.0 if 18 <= hi <= 72 else 0.0)))))
        return tau_m, sc

    engine_profiles = {
        "ensemble": {
            "id": "ensemble",
            "name": "Lead-Time Weighted Hybrid (GOES-18 CCLC/BTD + RAOB + HRRR 3km + ECMWF + AIFS)",
            "short_name": "⚡ Hybrid Ensemble",
            "provider": "GOES-18 CCLC/BTD + RAOB Balloon + NOAA HRRR 3km + ECMWF IFS/AIFS",
            "resolution_3d": "3 km NWP + 2-10 km GOES CCLC/BTD + 00Z Balloon × 380 km Corridor",
            "low": ens_low,
            "mid": ens_mid,
            "high": ens_high,
        },
        "graphcast": {
            "id": "graphcast",
            "name": "ECMWF AIFS 0.25° Neural Weather Transformer",
            "short_name": "🧠 ECMWF AIFS (AI)",
            "provider": "ECMWF AIFS Single (`ecmwf_aifs025_single`)",
            "resolution_3d": "0.25° (~25 km) AI Forecast",
            "low": aifs_low,
            "mid": aifs_mid,
            "high": aifs_high,
        },
        "hrrr": {
            "id": "hrrr",
            "name": "NOAA 3km HRRR Mesoscale Radar-Assimilated",
            "short_name": "🛰️ NOAA 3km HRRR",
            "provider": "NCEP HRRR CONUS (`ncep_hrrr_conus`)" if has_hrrr else "NOAA GFS Seamless (`gfs_seamless`)",
            "resolution_3d": "3.0 km native grid × pressure levels",
            "low": hrrr_low,
            "mid": hrrr_mid,
            "high": hrrr_high,
        },
        "ecmwf": {
            "id": "ecmwf",
            "name": "ECMWF IFS 0.25° Global Physics Model",
            "short_name": "🌍 ECMWF 0.25° IFS",
            "provider": "ECMWF (`ecmwf_ifs025`)",
            "resolution_3d": "0.25° (~25 km) × pressure levels",
            "low": ecmwf_low,
            "mid": ecmwf_mid,
            "high": ecmwf_high,
        },
        "nowcast": {
            "id": "nowcast",
            "name": "3-Frame GOES-18/19 L2 (CCLC+BTD+ACTP+ACHA+COD) + ASOS + RAOB Balloon",
            "short_name": "📡 Multi-Sensor Truth / Nowcast",
            "provider": (
                f"{goes_data['satellite']} CCLC/L2 + {station_obs['station_id'] if station_obs else 'NWS'}"
                + (f" + {raob_obs['station_id']} Balloon" if raob_obs else "")
                if (has_live_nowcast and goes_data and goes_data.get("is_calibrated_l2"))
                else (
                    f"{goes_data['satellite']} + {station_obs['station_id'] if station_obs else 'NWS Surface'}"
                    if has_live_nowcast
                    else "Outside Satellite Nowcast Window (NWP Forecast)"
                )
            ),
            "resolution_3d": "2 km COD/BTD + 10 km 5-Layer CCLC/ACHA + 00Z Balloon Sounding",
            "low": nowcast_low,
            "mid": nowcast_mid,
            "high": nowcast_high,
        },
    }
    for prof in engine_profiles.values():
        t_v, s_v = _score_col(prof["low"], prof["mid"], prof["high"])
        prof["optical_depth_tau"] = t_v
        prof["alpenglow_score"] = s_v

    active_prof = engine_profiles.get(model_engine, engine_profiles["ensemble"])
    spread_low = max(hrrr_low, ecmwf_low, aifs_low) - min(hrrr_low, ecmwf_low, aifs_low)
    spread_high = max(hrrr_high, ecmwf_high, aifs_high) - min(hrrr_high, ecmwf_high, aifs_high)
    ensemble_agreement_pct = max(42, min(99, int(round(100.0 - spread_low * 0.42 - spread_high * 0.28))))

    ai_ensemble = {
        "selected_engine": model_engine if model_engine in engine_profiles else "ensemble",
        "selected_engine_name": active_prof["name"],
        "selected_short_name": active_prof["short_name"],
        "resolution_3d": active_prof["resolution_3d"],
        "ensemble_agreement_pct": ensemble_agreement_pct,
        "atmospheric_feel_archetype": (
            f"380 km Ray Corridor • {direct_prob_pct}% Beam Odds (P10 {p10_pct}% · P50 {p50_pct}% · P90 {p90_pct}%)"
        ),
        "synthesis_note": (
            f"Spectral Chappuis {kelvin} K · Overhead Canvas {canvas_glow_score}/100 · "
            f"Blocker: {blocker_label} @ {blocker_dist_km:.0f} km"
            + (
                f" · GOES Drift ({goes_data['motion_u_ms']:+.1f}, {goes_data['motion_v_ms']:+.1f}) m/s"
                if goes_data
                else ""
            )
        ),
        "advection_wind_mph": wind_mph,
        "advection_from_deg": wind_dir_deg,
        "models": [
            engine_profiles["ensemble"],
            engine_profiles["graphcast"],
            engine_profiles["hrrr"],
            engine_profiles["ecmwf"],
            engine_profiles["nowcast"],
        ],
    }

    st_night = hourly_soundings_0km[22]
    jet_250_mph = int(round(st_night["wind_250_kmh"] * 0.621371))
    jet_300_mph = int(round(st_night["wind_300_kmh"] * 0.621371))
    crest_700_mph = int(round(st_night["wind_700_kmh"] * 0.621371))
    wind_shear_mph = max(4, abs(jet_250_mph - crest_700_mph))
    seeing_score = int(round(max(18.0, min(97.0, 102.0 - 0.55 * jet_250_mph - 0.25 * wind_shear_mph))))
    aod_night = nwp_bundle["aod_0km_24h"][22]
    pm25_night = nwp_bundle["pm25_24h"][22]
    dust_night = nwp_bundle["dust_24h"][22]
    transparency_score = int(round(max(18.0, min(99.0, 100.0 * math.exp(-2.1 * aod_night) - pm25_night * 0.35))))
    night_clarity = int(
        round(
            max(
                5.0,
                100.0
                - (
                    0.75 * st_night["cloud_high_pct"]
                    + 0.85 * st_night["cloud_mid_pct"]
                    + (0.25 if (is_coastal and summit_clearance_ft > 150) else 0.70) * st_night["cloud_low_pct"]
                ),
            )
        )
    )
    astro_atmosphere = {
        "source": "Live 250 hPa Sounding + CAMS 550 nm Aerosol" if nwp_bundle["has_live_aq"] else nwp_bundle["source_label"],
        "astro_atmosphere_score": int(round(0.42 * night_clarity + 0.32 * transparency_score + 0.26 * seeing_score)),
        "seeing_score": seeing_score,
        "seeing_grade": "Pickering 8/10 Calm" if seeing_score >= 78 else ("Moderate Jet Ripple" if seeing_score >= 52 else "Strong Jet Turbulence"),
        "seeing_note": f"22:00 local 250 hPa jet is {jet_250_mph} mph (Δv {wind_shear_mph} mph shear vs 700 hPa).",
        "jet_stream_250hpa_mph": jet_250_mph,
        "jet_stream_300hpa_mph": jet_300_mph,
        "crest_wind_700hpa_mph": crest_700_mph,
        "vertical_wind_shear_mph": wind_shear_mph,
        "transparency_score": transparency_score,
        "haze_grade": "Pristine Rayleigh Air" if aod_night < 0.08 else ("Light Boundary Haze" if aod_night < 0.18 else "Elevated Smoke/Aerosol"),
        "haze_note": f"Column 550 nm AOD is {aod_night:.3f} (PM2.5 {pm25_night:.1f} µg/m³, Dust {dust_night:.1f} µg/m³).",
        "aod_550nm": aod_night,
        "dust_ug_m3": dust_night,
        "pm25_ug_m3": pm25_night,
        "night_cloud_high_pct": st_night["cloud_high_pct"],
        "night_cloud_mid_pct": st_night["cloud_mid_pct"],
        "night_cloud_low_pct": st_night["cloud_low_pct"],
        "night_clarity_score": night_clarity,
    }

    seed_int = int(hashlib.sha256(f"{lat:.3f}:{lon:.3f}:{req_date.isoformat()}".encode()).hexdigest()[:8], 16)
    fluid_atmosphere_params = _build_fluid_params_from_sounding(st0, is_coastal, seed_int)
    atmospheric_layers = _build_atmospheric_layers_provenance(
        st0,
        fluid_atmosphere_params,
        is_coastal=is_coastal,
        is_historical=is_historical,
        lead_hours=lead_hours,
        w_nowcast=w_nowcast_primary,
        has_live_nowcast=has_live_nowcast,
        goes_data=goes_data,
        station_obs=station_obs,
        raob_obs=raob_obs,
        has_hrrr=has_hrrr,
        hrrr_lmh=(hrrr_low, hrrr_mid, hrrr_high),
        ecmwf_lmh=(ecmwf_low, ecmwf_mid, ecmwf_high),
        aifs_lmh=(aifs_low, aifs_mid, aifs_high),
        gfs_lmh=(gfs_low, gfs_mid, gfs_high),
        nowcast_lmh=(nowcast_low, nowcast_mid, nowcast_high),
        is_historical_truth_replay=is_historical_truth_replay,
        hindcast_lead_days=hindcast_lead_days,
    )

    hourly_timeline: list[dict[str, Any]] = []
    for hr in range(24):
        snd_h = hourly_soundings_0km[hr]
        r_out = hourly_rust_results[hr] if hr < len(hourly_rust_results) else None
        h_ray = (r_out or {}).get("ray") or {}
        h_canv = (r_out or {}).get("overhead_canvas") or {}
        h_blk = h_ray.get("dominant_blocker") or {}
        hr_col_aod = (
            0.65 * float(snd_h["rust_station"]["aod_550nm"])
            + 0.35 * float(nwp_bundle["aod_100km_24h"][hr])
        )
        h_exp_pct, h_prob_pct, h_p10, h_p50, h_p90 = _normalize_corridor_ray_pct(h_ray, hr_col_aod)
        h_kelvin = int(round(float(h_ray.get("beam_color_kelvin") or 5400.0)))
        h_glow = int(round(float(h_canv.get("composite_canvas_score") or 40.0)))

        hr_dt_loc = dt.datetime(req_date.year, req_date.month, req_date.day, hr, 30, tzinfo=tz_obj)
        hr_lead = (hr_dt_loc.astimezone(dt.timezone.utc) - now_utc).total_seconds() / 3600.0
        min_hr_l = -2.5 if is_historical_truth_replay else -0.8
        if has_live_nowcast and min_hr_l <= hr_lead <= 4.5:
            eff_l = abs(hr_lead) if is_historical_truth_replay else max(0.0, hr_lead)
            if is_historical_truth_replay and abs(hr_lead) <= 0.75 and model_engine in ("ensemble", "nowcast"):
                w_hr = 0.94
            elif model_engine == "nowcast":
                w_hr = 0.94
            elif model_engine == "ensemble":
                w_hr = max_ens_w / (1.0 + (eff_l / decay_tau_hr) ** 1.8)
            else:
                w_hr = 0.0
        else:
            w_hr = 0.0

        fb_lo, fb_mi, fb_hi = snd_h["cloud_low_pct"], snd_h["cloud_mid_pct"], snd_h["cloud_high_pct"]
        mh_lo, mh_mi, mh_hi, ok_mh = _model_lmh_at_hr("ncep_hrrr_conus", hr, fb_lo, fb_mi, fb_hi)
        me_lo, me_mi, me_hi, _ = _model_lmh_at_hr("ecmwf_ifs025", hr, fb_lo, fb_mi, fb_hi)
        ma_lo, ma_mi, ma_hi, _ = _model_lmh_at_hr("ecmwf_aifs025_single", hr, me_lo, me_mi, me_hi)
        mg_lo, mg_mi, mg_hi, ok_mg = _model_lmh_at_hr("gfs_seamless", hr, fb_lo, fb_mi, fb_hi)
        if not ok_mh and ok_mg:
            mh_lo, mh_mi, mh_hi = mg_lo, mg_mi, mg_hi

        hr_fluid_params = _build_fluid_params_from_sounding(snd_h, is_coastal, seed_int + hr)
        hr_layers = _build_atmospheric_layers_provenance(
            snd_h,
            hr_fluid_params,
            is_coastal=is_coastal,
            is_historical=is_historical,
            lead_hours=hr_lead,
            w_nowcast=w_hr,
            has_live_nowcast=bool(has_live_nowcast and w_hr > 0.04),
            goes_data=goes_data if w_hr > 0.04 else None,
            station_obs=station_obs if w_hr > 0.04 else None,
            raob_obs=raob_obs if w_hr > 0.04 else None,
            has_hrrr=ok_mh,
            hrrr_lmh=(mh_lo, mh_mi, mh_hi),
            ecmwf_lmh=(me_lo, me_mi, me_hi),
            aifs_lmh=(ma_lo, ma_mi, ma_hi),
            gfs_lmh=(mg_lo, mg_mi, mg_hi),
            nowcast_lmh=(nowcast_low, nowcast_mid, nowcast_high),
            is_historical_truth_replay=is_historical_truth_replay,
            hindcast_lead_days=hindcast_lead_days,
        )

        hourly_timeline.append(
            {
                "hour": hr,
                "minutes_local": hr * 60 + 30,
                "time_local": (r_out or {}).get("time_local") or f"{hr:02d}:30",
                "sun_az_deg": (r_out or {}).get("sun_az_deg") or 180.0,
                "sun_el_deg": (r_out or {}).get("sun_el_deg") or 0.0,
                "temp_f": int(round(snd_h["temp_2m_c"] * 9.0 / 5.0 + 32.0)),
                "wind_mph": int(round(snd_h["wind_10m_kmh"] * 0.621371)),
                "cloud_low_pct": snd_h["cloud_low_pct"],
                "cloud_mid_pct": snd_h["cloud_mid_pct"],
                "cloud_high_pct": snd_h["cloud_high_pct"],
                "fog_top_ft": int(round(snd_h["inv_top_asl_m"] * 3.28084)),
                "aod_550nm": snd_h["rust_station"]["aod_550nm"],
                "expected_transmittance_pct": h_exp_pct,
                "direct_beam_pct": h_prob_pct,
                "p10_pct": h_p10,
                "p50_pct": h_p50,
                "p90_pct": h_p90,
                "kelvin": h_kelvin,
                "canvas_glow_score": h_glow,
                "blocker_label": h_blk.get("blocker_type") or "clear",
                "blocker_distance_km": round(float(h_blk.get("distance_km") or 0.0), 1),
                "fluid_atmosphere_params": hr_fluid_params,
                "atmospheric_layers": hr_layers,
            }
        )

    inversion_model = {
        "is_coastal": is_coastal,
        "is_historical": is_historical,
        "temp_2m_f": round(temp_sunset_f, 1),
        "temp_950hpa_f": temp_950_f,
        "temp_925hpa_f": temp_925_f,
        "temp_850hpa_f": temp_850_f,
        "rh_2m_pct": int(round(st0["rh_2m_pct"])),
        "rh_950hpa_pct": int(round(st0["rh_950_pct"])),
        "rh_925hpa_pct": int(round(st0["rh_925_pct"])),
        "inversion_delta_t_f": inversion_delta_t_f,
        "fog_base_ft": fog_base_ft,
        "fog_top_ft": fog_top_ft,
        "pbl_height_ft": pbl_height_ft,
        "target_elev_ft": target_elev_ft,
        "summit_clearance_ft": summit_clearance_ft,
        "cloud_sea_score": cloud_sea_score,
        "under_deck_sunset_score": under_deck_sunset_score,
        "raob_station_id": raob_obs["station_id"] if raob_obs else None,
        "raob_inversion_base_asl_m": raob_obs.get("inversion_base_asl_m") if raob_obs else None,
        "raob_inversion_top_asl_m": raob_obs.get("inversion_top_asl_m") if raob_obs else None,
        "raob_inversion_strength_k": raob_obs.get("inversion_strength_k") if raob_obs else None,
    }

    nowcast_status = {
        "active": bool(w_nowcast_primary > 0.05 and has_live_nowcast),
        "lead_hours": round(lead_hours, 2),
        "nowcast_blend_weight": round(w_nowcast_primary, 3),
        "satellite": goes_data["satellite"] if goes_data else None,
        "source": goes_data.get("source") if goes_data else None,
        "is_calibrated_l2": bool(goes_data.get("is_calibrated_l2")) if goes_data else False,
        "has_cclc_multilayer": bool(goes_data.get("has_cclc_multilayer") or goes_data.get("has_cclc")) if goes_data else False,
        "has_twilight_btd_fog": bool(goes_data.get("has_twilight_btd_fog") or goes_data.get("has_btd_fog")) if goes_data else False,
        "parallax_corrected": bool(goes_data.get("parallax_corrected")) if goes_data else False,
        "history_span_minutes": goes_data.get("history_span_minutes") if goes_data else 0.0,
        "goes_scan_utc": goes_data["scan_timestamp_utc"] if goes_data else None,
        "goes_scan_age_minutes": goes_data["scan_age_minutes"] if goes_data else None,
        "goes_motion_uv_ms": [goes_data["motion_u_ms"], goes_data["motion_v_ms"]] if goes_data else None,
        "low_motion_uv_ms": list(goes_data.get("low_motion_uv_ms") or [0.0, 0.0]) if goes_data else None,
        "high_motion_uv_ms": list(goes_data.get("high_motion_uv_ms") or [0.0, 0.0]) if goes_data else None,
        "goes_motion_confidence": goes_data["motion_confidence"] if goes_data else 0.0,
        "epistemic_sigma_pct": round(epistemic_sigma_pct, 2),
        "metar_station": (
            {
                "id": station_obs["station_id"],
                "name": station_obs["station_name"],
                "distance_km": station_obs["distance_km"],
                "age_minutes": station_obs["age_minutes"],
                "temp_c": station_obs["temp_c"],
                "visibility_miles": station_obs["visibility_miles"],
                "fog_base_asl_m": station_obs.get("fog_base_asl_m"),
                "lowest_ceiling_asl_m": station_obs["lowest_ceiling_asl_m"],
            }
            if station_obs
            else None
        ),
        "raob_station": (
            {
                "id": raob_obs["station_id"],
                "name": raob_obs["station_name"],
                "distance_km": raob_obs["distance_km"],
                "valid_utc": raob_obs["valid_utc"],
                "inversion_base_asl_m": raob_obs.get("inversion_base_asl_m"),
                "inversion_top_asl_m": raob_obs.get("inversion_top_asl_m"),
                "inversion_strength_k": raob_obs.get("inversion_strength_k"),
            }
            if raob_obs
            else None
        ),
    }

    historical_replay_meta = {
        "is_past_date": is_past_date,
        "replay_mode": replay_mode,
        "is_observed_truth": is_historical_truth_replay,
        "hindcast_lead_days": hindcast_lead_days,
        "lead_hours": round(lead_hours, 2),
        "target_utc": target_utc.isoformat(),
        "eval_now_utc": now_utc.isoformat(),
    }

    data_provenance = {
        "is_live": not is_past_date and (simulated_now_utc is None),
        "degraded": len(nwp_bundle.get("errors") or []) > 0,
        "warnings": nwp_bundle.get("errors") or [],
        "timezone": tz_name,
        "compute_backend": rust_out.get("compute_backend"),
        "srtm_tiles_loaded": rust_out.get("srtm_tiles_loaded"),
        "timing_ms": rust_out.get("timing_ms"),
        "nwp_source": nwp_bundle["source_label"],
        "cams_aerosol_live": nwp_bundle["has_live_aq"],
        "nowcast": nowcast_status,
        "historical_replay": historical_replay_meta,
    }

    corridor_raycast_summary = {
        "max_corridor_km": 380.0,
        "direct_beam_probability_pct": direct_prob_pct,
        "expected_transmittance_pct": exp_trans_pct,
        "transmittance_p10_pct": p10_pct,
        "transmittance_p50_pct": p50_pct,
        "transmittance_p90_pct": p90_pct,
        "color_temperature_kelvin": kelvin,
        "chappuis_rose_index": chappuis_rose,
        "terrain_disk_fraction": round(float(primary_ray.get("terrain_disk_visibility", 1.0)), 3),
        "terrain_horizon_deg": round(float(primary_ray.get("terrain_horizon_deg", 0.0)), 2),
        "far_field_horizon_deg": round(float(primary_ray.get("far_field_horizon_deg", 0.0)), 2),
        "dominant_blocker_label": blocker_label,
        "dominant_blocker_distance_km": blocker_dist_km,
        "dominant_blocker_altitude_asl_m": blocker_alt_m,
        "dominant_blocker_altitude_ft": int(round(blocker_alt_m * 3.28084)),
        "dominant_blocker_summary": blocker_summary,
        "clear_sky_rgb": primary_ray.get("clear_sky_rgb"),
        "expected_beam_rgb": primary_ray.get("expected_beam_rgb"),
        "segments": primary_ray.get("segments"),
        "overhead_canvas": overhead_canvas,
    }

    payload_dict = {
        "date": req_date.isoformat(),
        "phase": "Sunset" if is_sunset else "Sunrise",
        "sun_azimuth_deg": round(sun_az_deg, 1),
        "sun_elevation_deg": round(sun_el_deg, 1),
        "corridor_direction": _compass_cardinal(sun_az_deg),
        "source": (
            f"{active_prof['short_name']} • {nwp_bundle['source_label']}"
            + (f" + {goes_data['satellite']}" if nowcast_status["active"] and goes_data else "")
        ),
        "alpenglow_score": alpenglow_score,
        "canvas_score": canvas_glow_score,
        "verdict": verdict,
        "recommendation": recommendation,
        "ray_transmittance_pct": exp_trans_pct,
        "direct_beam_probability_pct": direct_prob_pct,
        "transmittance_p10_pct": p10_pct,
        "transmittance_p50_pct": p50_pct,
        "transmittance_p90_pct": p90_pct,
        "color_temperature_kelvin": kelvin,
        "chappuis_rose_index": chappuis_rose,
        "lcl_cloud_base_ft": lcl_cloud_base_ft,
        "fog_base_ft": fog_base_ft,
        "fog_top_ft": fog_top_ft,
        "cloud_high_pct": st0["cloud_high_pct"],
        "cloud_mid_pct": st0["cloud_mid_pct"],
        "cloud_low_pct": st0["cloud_low_pct"],
        "visibility_miles": vis_miles,
        "temp_sunset_f": int(round(temp_sunset_f)),
        "temp_sunrise_f": temp_sunrise_f,
        "wind_mph": wind_mph,
        "wind_dir_deg": wind_dir_deg,
        "valley_inversion_fog_pct": inversion_chance_pct,
        "inversion_model": inversion_model,
        "ai_ensemble": ai_ensemble,
        "spatial_cloud_field": spatial_cloud_field,
        "mesoscale_cloud_grid": mesoscale_cloud_grid,
        "fluid_atmosphere_params": fluid_atmosphere_params,
        "atmospheric_layers": atmospheric_layers,
        "ray_path_stations": ray_path_stations,
        "corridor_stations_6": corridor_stations_6,
        "corridor_raycast": corridor_raycast_summary,
        "corridor_lut": corridor_lut,
        "sunset_corridor_lut": sunset_corridor_lut,
        "sunrise_corridor_lut": sunrise_corridor_lut,
        "hourly_timeline": hourly_timeline,
        "nowcast_status": nowcast_status,
        "raob_sounding": raob_obs,
        "historical_replay": historical_replay_meta,
        "data_provenance": data_provenance,
        "astro_atmosphere": astro_atmosphere,
    }
    return json.dumps(payload_dict).encode("utf-8")
