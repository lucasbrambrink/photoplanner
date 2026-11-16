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
) -> dict[str, Any]:
    """
    Fuses live GOES-18/19 advected satellite cloud fractions + optical depths and
    NWS METAR/RAWS surface ceilometer/visibility into a `CorridorStationProfile`.
    """
    if w_nowcast <= 0.02:
        return sounding

    st = json.loads(json.dumps(sounding))
    rs = st["rust_station"]
    dist_km = float(rs["distance_km"])

    if station_obs is not None and dist_km <= 65.0:
        obs_dist_decay = math.exp(-dist_km / 45.0)
        w_obs = min(0.88, w_nowcast * obs_dist_decay)
        if station_obs.get("temp_c") is not None and dist_km <= 25.0:
            obs_t = float(station_obs["temp_c"])
            st["temp_2m_c"] = round((1.0 - w_obs) * st["temp_2m_c"] + w_obs * obs_t, 2)
            rs["surface_temp_c"] = st["temp_2m_c"]
        if station_obs.get("rh_pct") is not None and dist_km <= 25.0:
            st["rh_2m_pct"] = round(
                (1.0 - w_obs) * st["rh_2m_pct"] + w_obs * float(station_obs["rh_pct"]), 1
            )
        if station_obs.get("visibility_m") is not None:
            st["visibility_m"] = round(
                (1.0 - w_obs) * st["visibility_m"] + w_obs * float(station_obs["visibility_m"]), 0
            )
        if station_obs.get("aod_550nm_proxy") is not None:
            rs["aod_550nm"] = round(
                (1.0 - 0.55 * w_obs) * rs["aod_550nm"]
                + 0.55 * w_obs * float(station_obs["aod_550nm_proxy"]),
                3,
            )
        if station_obs.get("lcl_asl_m") is not None and dist_km <= 25.0:
            st["lcl_asl_m"] = round(
                (1.0 - w_obs) * st["lcl_asl_m"] + w_obs * float(station_obs["lcl_asl_m"]), 1
            )

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
        sat_mid_pct = adv["mid_cf"] * 100.0
        sat_high_pct = adv["high_cf"] * 100.0

        if station_obs is not None and dist_km <= 30.0:
            obs_low_pct = float(station_obs.get("low_cloud_fraction", 0.0)) * 100.0
            sat_low_pct = 0.62 * sat_low_pct + 0.38 * obs_low_pct

        new_low = int(round((1.0 - w_nowcast) * st["cloud_low_pct"] + w_nowcast * sat_low_pct))
        new_mid = int(round((1.0 - w_nowcast) * st["cloud_mid_pct"] + w_nowcast * sat_mid_pct))
        new_high = int(round((1.0 - w_nowcast) * st["cloud_high_pct"] + w_nowcast * sat_high_pct))

        st["cloud_low_pct"] = max(0, min(100, new_low))
        st["cloud_mid_pct"] = max(0, min(100, new_mid))
        st["cloud_high_pct"] = max(0, min(100, new_high))

        elev_m = float(rs["surface_elevation_m"])
        new_bands: list[dict[str, Any]] = []
        low_cf = st["cloud_low_pct"] / 100.0
        mid_cf = st["cloud_mid_pct"] / 100.0
        high_cf = st["cloud_high_pct"] / 100.0

        ceil_asl = (
            float(station_obs["lowest_ceiling_asl_m"])
            if (station_obs and station_obs.get("lowest_ceiling_asl_m") and dist_km <= 35.0)
            else None
        )

        if low_cf >= 0.03:
            is_fog_like = elev_m < 850.0 and (st["inv_jump_k"] >= 1.0 or st["rh_2m_pct"] >= 88.0)
            base_m = (
                ceil_asl
                if (ceil_asl is not None and ceil_asl > elev_m + 30.0)
                else max(elev_m + (35.0 if is_fog_like else 220.0), st["lcl_asl_m"])
            )
            top_m = max(base_m + (240.0 if is_fog_like else 580.0), st["inv_top_asl_m"])
            tau_nwp = low_cf * min(26.0, 0.022 * (top_m - base_m))
            tau_sat = low_cf * max(1.5, min(28.0, adv["tau_total"] * 0.72))
            tau_low = (1.0 - w_nowcast) * tau_nwp + w_nowcast * tau_sat
            new_bands.append(
                {
                    "label": "fog" if is_fog_like else "low",
                    "base_asl_m": round(base_m, 1),
                    "top_asl_m": round(top_m, 1),
                    "cloud_fraction": round(low_cf, 3),
                    "optical_depth_vertical": round(max(0.08, tau_low), 2),
                    "aspect_ratio_h_over_d": 0.07 if is_fog_like else 0.28,
                    "characteristic_cell_km": 11.0 if is_fog_like else 3.8,
                    "ice_fraction": 0.0,
                }
            )
        if mid_cf >= 0.03:
            mid_base = max(elev_m + 900.0, 2900.0)
            mid_top = mid_base + 1450.0
            new_bands.append(
                {
                    "label": "mid",
                    "base_asl_m": round(mid_base, 1),
                    "top_asl_m": round(mid_top, 1),
                    "cloud_fraction": round(mid_cf, 3),
                    "optical_depth_vertical": round(mid_cf * 7.5, 2),
                    "aspect_ratio_h_over_d": 0.22,
                    "characteristic_cell_km": 6.5,
                    "ice_fraction": 0.35,
                }
            )
        if high_cf >= 0.03:
            new_bands.append(
                {
                    "label": "high",
                    "base_asl_m": 8200.0,
                    "top_asl_m": 10600.0,
                    "cloud_fraction": round(high_cf, 3),
                    "optical_depth_vertical": round(high_cf * (0.55 + 1.40 * high_cf), 2),
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
    """Maps a real pressure-level sounding + true vector winds into WebGL2/WASM fluid boundary params."""
    u_sfc, v_sfc = st0["wind_sfc_uv_ms"]
    u_850, v_850 = st0["wind_850_uv_ms"]
    u_500, v_500 = st0["wind_500_uv_ms"]
    u_250, v_250 = st0["wind_250_uv_ms"]
    spd_850 = math.hypot(u_850, v_850)

    inv_strength_k = max(1.0, min(15.0, st0["inv_jump_k"] + (3.2 if is_coastal else 1.2)))
    cape_eff = max(
        40.0,
        st0["cape_j_kg"]
        if st0["cape_j_kg"] > 25.0
        else ((120.0 if is_coastal else 210.0) + max(0.0, st0["temp_2m_c"] - 12.0) * 28.0),
    )

    return {
        "surfaceTempC": round(st0["temp_2m_c"], 2),
        "surfaceRhPct": round(st0["rh_2m_pct"], 1),
        "lowCloudPct": float(st0["cloud_low_pct"]),
        "midCloudPct": float(st0["cloud_mid_pct"]),
        "highCloudPct": float(st0["cloud_high_pct"]),
        "inversionBaseM": round(max(110.0, st0["inv_base_asl_m"]), 1),
        "inversionThicknessM": 220.0 if is_coastal else 320.0,
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
        "orographicWaveGain": round(min(1.65, 0.85 + spd_850 * 0.055), 2),
        "vorticityConfinement": 1.18,
        "cirrusIceBoost": round(0.82 + 0.58 * (st0["cloud_high_pct"] / 100.0), 2),
        "jet250Mph": int(round(st0["wind_250_kmh"] * 0.621371)),
        "jet300Mph": int(round(st0["wind_300_kmh"] * 0.621371)),
        "seed": int(seed & 0xFFFF),
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
                fog_p = (1.0 - w_nowcast) * fog_p + w_nowcast * (adv["low_cf"] * 85.0)

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


def solve_weather_and_corridor(
    lat: float,
    lon: float,
    date_str: str,
    phase: str = "sunset",
    sun_az_deg: float = 259.1,
    sun_el_deg: float = 0.4,
    target_elev_m: float = 2693.0,
    model_engine: str = "ensemble",
) -> bytes:
    model_engine = (model_engine or "ensemble").lower().strip()
    is_coastal = _is_coastal_california(lat, lon)
    is_sunset = phase.lower().startswith("sunset") or sun_az_deg >= 180.0

    tz_obj, tz_name = local_tz_for_lon(lon)
    now_utc = dt.datetime.now(dt.timezone.utc)
    now_local = now_utc.astimezone(tz_obj)
    try:
        req_date = dt.date.fromisoformat(date_str)
    except ValueError:
        req_date = now_local.date()

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
    lead_hours = (target_local_dt.astimezone(dt.timezone.utc) - now_utc).total_seconds() / 3600.0
    is_today_or_near = -1.5 <= lead_hours <= 5.5
    should_query_live_obs = (req_date == now_local.date()) or (-2.0 <= lead_hours <= 6.5)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_nwp = pool.submit(
            fetch_24h_corridor_and_models,
            lat,
            lon,
            req_date.isoformat(),
            sun_az_deg,
            conjugate_az,
            is_coastal,
        )
        f_obs = (
            pool.submit(fetch_nearest_station_observation, lat, lon)
            if should_query_live_obs
            else None
        )
        nwp_bundle = f_nwp.result()
        station_obs = f_obs.result() if f_obs else None

    est_sfc_temp_c = 15.0
    if station_obs and station_obs.get("temp_c") is not None:
        est_sfc_temp_c = float(station_obs["temp_c"])
    elif nwp_bundle["ok"]:
        now_hr = max(0, min(23, now_local.hour))
        est_sfc_temp_c = _safe_arr_val(
            nwp_bundle["point_docs"][0].get("hourly") or {}, "temperature_2m", now_hr, 15.0
        )

    goes_data = (
        fetch_goes_corridor_nowcast(lat, lon, surface_temp_c=est_sfc_temp_c, is_coastal=is_coastal)
        if should_query_live_obs
        else None
    )

    if not nwp_bundle["ok"]:
        raise RuntimeError(
            "Upstream NWP sounding service unreachable; refusing to synthesize fake weather. "
            f"Details: {'; '.join(nwp_bundle.get('errors') or ['empty point_docs'])}"
        )

    point_docs = nwp_bundle["point_docs"]
    coords_meta = nwp_bundle["coords_meta"]
    is_historical = bool(nwp_bundle["is_historical"])

    st0_base = extract_hour_sounding(
        point_docs[0],
        primary_hr_idx,
        0.0,
        sun_az_deg,
        lat,
        lon,
        aod_550nm=nwp_bundle["aod_0km_24h"][primary_hr_idx],
        angstrom_exp=nwp_bundle["angstrom_24h"][primary_hr_idx],
        ozone_du=nwp_bundle["ozone_du_24h"][primary_hr_idx],
        is_coastal=is_coastal,
    )

    mm_h = (nwp_bundle.get("multimodel_doc") or {}).get("hourly") or {}

    def _model_lmh(suffix: str, fallback_low: int, fallback_mid: int, fallback_high: int) -> tuple[int, int, int, bool]:
        arr_tot = mm_h.get(f"cloud_cover_{suffix}") or []
        if not any(v is not None for v in arr_tot):
            return fallback_low, fallback_mid, fallback_high, False
        lo = int(round(max(0.0, min(100.0, _safe_arr_val(mm_h, f"cloud_cover_low_{suffix}", primary_hr_idx, fallback_low)))))
        mi = int(round(max(0.0, min(100.0, _safe_arr_val(mm_h, f"cloud_cover_mid_{suffix}", primary_hr_idx, fallback_mid)))))
        hi = int(round(max(0.0, min(100.0, _safe_arr_val(mm_h, f"cloud_cover_high_{suffix}", primary_hr_idx, fallback_high)))))
        return lo, mi, hi, True

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
            min(210.0, lead_minutes_primary),
            st0_base["wind_850_uv_ms"],
            st0_base["wind_500_uv_ms"],
            st0_base["wind_250_uv_ms"],
        )
        nowcast_low = int(round(adv0["low_cf"] * 100.0))
        if station_obs is not None:
            obs_low = float(station_obs.get("low_cloud_fraction", 0.0)) * 100.0
            nowcast_low = int(round(0.60 * nowcast_low + 0.40 * obs_low))
        nowcast_mid = int(round(adv0["mid_cf"] * 100.0))
        nowcast_high = int(round(adv0["high_cf"] * 100.0))
        has_live_nowcast = True
    else:
        nowcast_low, nowcast_mid, nowcast_high = hrrr_low, hrrr_mid, hrrr_high
        has_live_nowcast = False

    if has_live_nowcast and is_today_or_near:
        base_w_now = math.exp(-max(0.0, lead_hours) / 2.2)
    else:
        base_w_now = 0.0

    if model_engine == "nowcast" and has_live_nowcast:
        w_nowcast_primary = 0.92 if is_today_or_near else 0.55
    elif model_engine == "ensemble":
        w_nowcast_primary = base_w_now * 0.72
    else:
        w_nowcast_primary = 0.0

    ens_nwp_low = int(round(0.45 * hrrr_low + 0.30 * ecmwf_low + 0.25 * aifs_low))
    ens_nwp_mid = int(round(0.45 * hrrr_mid + 0.30 * ecmwf_mid + 0.25 * aifs_mid))
    ens_nwp_high = int(round(0.40 * hrrr_high + 0.35 * ecmwf_high + 0.25 * aifs_high))

    ens_low = int(round((1.0 - w_nowcast_primary) * ens_nwp_low + w_nowcast_primary * nowcast_low))
    ens_mid = int(round((1.0 - w_nowcast_primary) * ens_nwp_mid + w_nowcast_primary * nowcast_mid))
    ens_high = int(round((1.0 - w_nowcast_primary) * ens_nwp_high + w_nowcast_primary * nowcast_high))

    engine_target_lmh: dict[str, tuple[int, int, int]] = {
        "ensemble": (ens_low, ens_mid, ens_high),
        "graphcast": (aifs_low, aifs_mid, aifs_high),
        "hrrr": (hrrr_low, hrrr_mid, hrrr_high),
        "ecmwf": (ecmwf_low, ecmwf_mid, ecmwf_high),
        "nowcast": (nowcast_low, nowcast_mid, nowcast_high),
    }
    sel_low, sel_mid, sel_high = engine_target_lmh.get(model_engine, (ens_low, ens_mid, ens_high))

    def _apply_engine_choice(st: dict[str, Any], is_origin: bool = False) -> dict[str, Any]:
        if model_engine == "ensemble" and abs(sel_low - st0_base["cloud_low_pct"]) <= 3:
            return st
        s_copy = json.loads(json.dumps(st))
        r_low = (sel_low + 5.0) / (st0_base["cloud_low_pct"] + 5.0)
        r_mid = (sel_mid + 5.0) / (st0_base["cloud_mid_pct"] + 5.0)
        r_hi = (sel_high + 5.0) / (st0_base["cloud_high_pct"] + 5.0)
        s_copy["cloud_low_pct"] = sel_low if is_origin else max(0, min(100, int(round(s_copy["cloud_low_pct"] * r_low))))
        s_copy["cloud_mid_pct"] = sel_mid if is_origin else max(0, min(100, int(round(s_copy["cloud_mid_pct"] * r_mid))))
        s_copy["cloud_high_pct"] = sel_high if is_origin else max(0, min(100, int(round(s_copy["cloud_high_pct"] * r_hi))))
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
        hr_mins = hr * 60 + 30
        hr_dt_local = dt.datetime(req_date.year, req_date.month, req_date.day, hr, 30, tzinfo=tz_obj)
        hr_lead_hr = (hr_dt_local.astimezone(dt.timezone.utc) - now_utc).total_seconds() / 3600.0
        if has_live_nowcast and -0.8 <= hr_lead_hr <= 3.8:
            w_hr_now = (
                0.90
                if model_engine == "nowcast"
                else (0.72 * math.exp(-max(0.0, hr_lead_hr) / 2.2) if model_engine == "ensemble" else 0.0)
            )
        else:
            w_hr_now = 0.0

        rust_stations_hr: list[dict[str, Any]] = []
        st_primary_list_hr: list[dict[str, Any]] = []

        for idx_pt in range(9):
            _, plat, plon, pdist_km, paz_deg = coords_meta[idx_pt]
            aod_interp = (
                nwp_bundle["aod_0km_24h"][hr]
                if pdist_km < 50.0
                else nwp_bundle["aod_100km_24h"][hr]
            )
            snd = extract_hour_sounding(
                point_docs[idx_pt],
                hr,
                pdist_km,
                paz_deg,
                plat,
                plon,
                aod_550nm=aod_interp,
                angstrom_exp=nwp_bundle["angstrom_24h"][hr],
                ozone_du=nwp_bundle["ozone_du_24h"][hr],
                is_coastal=_is_coastal_california(plat, plon),
            )
            snd = _blend_nowcast_into_station(
                snd,
                goes_data,
                station_obs if idx_pt == 0 else None,
                max(0.0, hr_lead_hr * 60.0),
                w_hr_now,
            )
            if w_hr_now < 0.5:
                snd = _apply_engine_choice(snd, is_origin=(idx_pt == 0))
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
            primary_hr_idx,
            pdist_km,
            paz_deg,
            plat,
            plon,
            aod_550nm=nwp_bundle["aod_0km_24h"][primary_hr_idx],
            angstrom_exp=nwp_bundle["angstrom_24h"][primary_hr_idx],
            ozone_du=nwp_bundle["ozone_du_24h"][primary_hr_idx],
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
        min(210.0, lead_minutes_primary),
        w_nowcast_primary,
        st0["wind_850_uv_ms"],
        st0["wind_500_uv_ms"],
        st0["wind_250_uv_ms"],
    )

    def _normalize_corridor_ray_pct(ray_doc: dict[str, Any]) -> tuple[int, int, int, int, int]:
        c_rgb = ray_doc.get("clear_sky_rgb") or [0.0, 0.0, 0.0]
        clear_photopic = 0.24 * float(c_rgb[0]) + 0.68 * float(c_rgb[1]) + 0.08 * float(c_rgb[2])
        aer_od = float(ray_doc.get("total_aerosol_od_550") or 0.0)
        ray_od = float(ray_doc.get("total_rayleigh_od_550") or 0.095)
        slant_airmass = max(1.0, ray_od / 0.095)
        equiv_vert_aod = aer_od / slant_airmass
        aer_excess_trans = math.exp(-3.8 * max(0.0, equiv_vert_aod - 0.055))
        direct_prob = max(0.0, min(1.0, float(ray_doc.get("direct_beam_probability") or 0.0)))

        if clear_photopic > 1.5e-3:
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

        e_pct = int(round(max(0.0, min(99.0, exp_rel * 100.0))))
        pr_pct = int(round(direct_prob * 100.0))
        q10 = int(round(max(0.0, min(99.0, p10_rel * 100.0))))
        q50 = int(round(max(q10, min(99.0, p50_rel * 100.0))))
        q90 = int(round(max(q50, min(99.0, p90_rel * 100.0))))
        return e_pct, pr_pct, q10, q50, q90

    exp_trans_pct, direct_prob_pct, p10_pct, p50_pct, p90_pct = _normalize_corridor_ray_pct(primary_ray)
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
    inversion_delta_t_f = round(
        (max(temp_950_f, temp_925_f) - temp_sunset_f) if is_coastal else (temp_850_f - temp_sunset_f),
        1,
    )

    lcl_cloud_base_ft = int(round(st0["lcl_asl_m"] * 3.28084))
    fog_band = next((b for b in st0["rust_station"]["cloud_bands"] if b["label"] in ("fog", "low")), None)
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

    inversion_cap_score = min(45.0, max(0.0, (inversion_delta_t_f + 2.0) * 4.0))
    stratus_deck_score = (
        min(42.0, st0["cloud_low_pct"] * 0.42)
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
    # Sheer granite/headland faces act as a direct specular/Lambertian alpenglow canvas even when
    # overhead clouds are sparse (`cloud_high + cloud_mid < 25%`), while high cirrus boosts sky fire to 90-98.
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
                        cloud_sea_score * 0.94 if (is_coastal and summit_clearance_ft > 140 and st0["cloud_low_pct"] >= 55) else 0.0,
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
                    + st0["cloud_low_pct"] * 0.50
                    + max(0.0, inversion_delta_t_f * 3.4),
                ),
            )
        )
    )

    nowcast_tag = ""
    if w_nowcast_primary > 0.12 and goes_data is not None:
        nowcast_tag = f" [Live GOES-18 + {station_obs['station_id'] if station_obs else 'ABI'} Nowcast]"

    if float(primary_ray.get("terrain_disk_visibility", 1.0)) <= 0.05:
        verdict = f"Topographic Ridge Occlusion ({blocker_dist_km:.0f} km){nowcast_tag}"
        recommendation = (
            f"At El {sun_el_deg:+.1f}°, terrain at {blocker_dist_km:.1f} km ({int(blocker_alt_m * 3.28084):,} ft ASL) "
            f"blocks the direct solar disk, while `{dominant_glow_band}` clouds score {canvas_glow_score}/100 for twilight afterglow."
        )
    elif is_coastal and st0["cloud_low_pct"] >= 60 and summit_clearance_ft > 130 and inversion_delta_t_f >= 1.8:
        verdict = f"Marine Cloud-Sea Inversion (+{inversion_delta_t_f}°F Cap){nowcast_tag}"
        recommendation = (
            f"Capped marine stratus ({fog_base_ft:,}–{fog_top_ft:,} ft ASL) sits {summit_clearance_ft:,} ft below summit. "
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
            is_coastal=_is_coastal_california(tlat, tlon),
        )
        tsnd = _blend_nowcast_into_station(
            tsnd, goes_data, station_obs if pt_idx == 0 else None, min(210.0, lead_minutes_primary), w_nowcast_primary
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
            "name": "Lead-Time Weighted Hybrid (GOES-18 + HRRR 3km + ECMWF + AIFS)",
            "short_name": "⚡ Hybrid Ensemble",
            "provider": "GOES-18 Nowcast (0-3h) + NOAA HRRR 3km + ECMWF IFS/AIFS",
            "resolution_3d": "3 km NWP + 6.8 km GOES Advection × 380 km Corridor",
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
            "name": "GOES-18/19 ABI + 2-Frame FFT Advection + NWS METAR",
            "short_name": "📡 Live GOES Nowcast",
            "provider": (
                f"{goes_data['satellite']} + {station_obs['station_id'] if station_obs else 'NWS Surface'}"
                if has_live_nowcast
                else "Outside 0-4h Live Satellite Window (HRRR Fallback)"
            ),
            "resolution_3d": "Multi-spectral 0.64/3.9/8.4 µm + FFT Optical Flow",
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

    hourly_timeline: list[dict[str, Any]] = []
    for hr in range(24):
        snd_h = hourly_soundings_0km[hr]
        r_out = hourly_rust_results[hr] if hr < len(hourly_rust_results) else None
        h_ray = (r_out or {}).get("ray") or {}
        h_canv = (r_out or {}).get("overhead_canvas") or {}
        h_blk = h_ray.get("dominant_blocker") or {}
        h_exp_pct, h_prob_pct, h_p10, h_p50, h_p90 = _normalize_corridor_ray_pct(h_ray)
        h_kelvin = int(round(float(h_ray.get("beam_color_kelvin") or 5400.0)))
        h_glow = int(round(float(h_canv.get("composite_canvas_score") or 40.0)))
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
                "fluid_atmosphere_params": _build_fluid_params_from_sounding(snd_h, is_coastal, seed_int + hr),
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
    }

    nowcast_status = {
        "active": bool(w_nowcast_primary > 0.05 and has_live_nowcast),
        "lead_hours": round(lead_hours, 2),
        "nowcast_blend_weight": round(w_nowcast_primary, 3),
        "satellite": goes_data["satellite"] if goes_data else None,
        "goes_scan_utc": goes_data["scan_timestamp_utc"] if goes_data else None,
        "goes_scan_age_minutes": goes_data["scan_age_minutes"] if goes_data else None,
        "goes_motion_uv_ms": [goes_data["motion_u_ms"], goes_data["motion_v_ms"]] if goes_data else None,
        "goes_motion_confidence": goes_data["motion_confidence"] if goes_data else 0.0,
        "metar_station": (
            {
                "id": station_obs["station_id"],
                "name": station_obs["station_name"],
                "distance_km": station_obs["distance_km"],
                "age_minutes": station_obs["age_minutes"],
                "temp_c": station_obs["temp_c"],
                "visibility_miles": station_obs["visibility_miles"],
                "lowest_ceiling_asl_m": station_obs["lowest_ceiling_asl_m"],
            }
            if station_obs
            else None
        ),
    }

    data_provenance = {
        "is_live": not is_historical,
        "degraded": len(nwp_bundle.get("errors") or []) > 0,
        "warnings": nwp_bundle.get("errors") or [],
        "timezone": tz_name,
        "compute_backend": rust_out.get("compute_backend"),
        "srtm_tiles_loaded": rust_out.get("srtm_tiles_loaded"),
        "timing_ms": rust_out.get("timing_ms"),
        "nwp_source": nwp_bundle["source_label"],
        "cams_aerosol_live": nwp_bundle["has_live_aq"],
        "nowcast": nowcast_status,
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
        "ray_path_stations": ray_path_stations,
        "corridor_stations_6": corridor_stations_6,
        "corridor_raycast": corridor_raycast_summary,
        "corridor_lut": corridor_lut,
        "sunset_corridor_lut": sunset_corridor_lut,
        "sunrise_corridor_lut": sunrise_corridor_lut,
        "hourly_timeline": hourly_timeline,
        "nowcast_status": nowcast_status,
        "data_provenance": data_provenance,
        "astro_atmosphere": astro_atmosphere,
    }
    return json.dumps(payload_dict).encode("utf-8")
