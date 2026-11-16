"""
Historical Ground-Truth Backtesting & Empirical Calibration Harness (`planner/weather/backtest.py`).

Evaluates PhotoPlanner's 3-frame NOAA GOES-18/19 Level-2 (`ACTP + ACHA + COD`) nowcaster,
archived NWS/FAA ASOS surface assimilation, multi-model pressure-level NWP (`HRRR + IFS + GFS`),
and the `0-380 km` Rust curved-Earth spectral/stochastic corridor raycaster against held-out
real-world observations across 12 2025 California golden-hour shoots (6 Coastal + 6 High Sierra).

Verification datasets (strictly causal `scan_end_utc <= t_issue` at forecast time):
1. Held-out `t_sunset` NOAA GOES-18 S3 Level-2 (`ABI-L2-ACTPC`, `ABI-L2-ACHAC`, `ABI-L2-CODC`)
   across all 6 solar-azimuth corridor stations (`0, 15, 45, 100, 190, 320 km`).
2. Archived Open-Meteo `direct_normal_irradiance` (`DNI`, W/m^2) normalized against clear-sky
   Kasten-Young airmass irradiance (`DNI / DNI_clear`) during the pre-sunset golden-hour window.
3. Archived Iowa Environmental Mesonet ASOS ceilometer & visibility (`KHAF`, `KSFO`, `KMMH`, `KFAT`).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .corridor_solver import (
    CALIBRATION_CONFIG,
    _find_sunrise_sunset_golden_hours,
    _is_coastal_california,
    _solar_ephemeris_local,
    solve_weather_and_corridor,
)
from .goes_nowcast import (
    _latlonh_to_goes_xy,
    fetch_goes_corridor_nowcast,
    sample_advected_goes_state,
)
from .nwp_corridor import (
    _safe_arr_val,
    extract_hour_sounding,
    fetch_24h_corridor_and_models,
    offset_lat_lon_km,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKTEST_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "backtest_archive"
SUMMARY_JSON_PATH = ROOT_DIR / "data" / "weather_cache" / "backtest_summary_v3.json"
USER_AGENT = "PhotoPlanner/2.0 (Calibrated Atmospheric Corridor Backtest)"

# 12 real 2025 California golden-hour benchmark shoots spanning every key photographic regime
BENCHMARK_EVENTS: list[dict[str, Any]] = [
    # --- 6 Coastal California Marine-Inversion / Stratus / Cirrus / Frontal Events ---
    {
        "id": "pedro_2025_05_06_stratus",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-05-06",
        "regime": "marine_stratus",
        "description": "Solid coastal marine stratus deck capped below Montara Ridge",
    },
    {
        "id": "pedro_2025_06_05_cloud_sea",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-06-05",
        "regime": "marine_cloud_sea",
        "description": "Classic 590 m ASL marine cloud-sea pinned against Montara Mountain",
    },
    {
        "id": "pedro_2025_05_12_broken",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-05-12",
        "regime": "broken_coastal",
        "description": "Broken coastal stratocumulus with offshore crepuscular sun-shaft gaps",
    },
    {
        "id": "pedro_2025_05_16_cirrus",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-05-16",
        "regime": "coastal_cirrus_burner",
        "description": "High Pacific ice cirrus canopy with unobstructed low-level ocean horizon",
    },
    {
        "id": "pedro_2025_06_14_clear",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-06-14",
        "regime": "clear_corridor",
        "description": "Pristine 380 km offshore Pacific corridor with clean Rayleigh/Chappuis beam",
    },
    {
        "id": "pedro_2025_09_29_frontal",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-09-29",
        "regime": "frontal_overcast",
        "description": "Deep multi-layer early-autumn Pacific frontal overcast",
    },
    # --- 6 High-Sierra Alpine / Cirrus-Burner / Lee-Wave / Storm Events ---
    {
        "id": "halfdome_2025_05_06_clear",
        "site": "Half Dome",
        "region": "sierra",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-06",
        "regime": "clear_alpine",
        "description": "Pristine clear-sky High Sierra granite warm-Kelvin sunset",
    },
    {
        "id": "halfdome_2025_05_25_cirrus",
        "site": "Half Dome",
        "region": "sierra",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-25",
        "regime": "sierra_cirrus_burner",
        "description": "Optically thin (COD ~0.65) 10.5 km High-Sierra cirrus alpenglow burner",
    },
    {
        "id": "halfdome_2025_06_07_wave",
        "site": "Half Dome",
        "region": "sierra",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-06-07",
        "regime": "sierra_cirrus_wave",
        "description": "Terrain-locked upper-tropospheric Sierra lee-wave cirrus banner",
    },
    {
        "id": "halfdome_2025_05_15_mixed",
        "site": "Half Dome",
        "region": "sierra",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-15",
        "regime": "mixed_orographic",
        "description": "Afternoon Sierra orographic cumulus dissipating beneath high cirrus",
    },
    {
        "id": "halfdome_2025_05_13_crest",
        "site": "Half Dome",
        "region": "sierra",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-13",
        "regime": "crest_attenuation",
        "description": "Mid-tropospheric foothills/crest cloud deck attenuating western corridor",
    },
    {
        "id": "halfdome_2025_05_03_storm",
        "site": "Half Dome",
        "region": "sierra",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-03",
        "regime": "storm_overcast",
        "description": "Deep spring Sierra storm system with thick mid/high ice overcast",
    },
]

# Pre-calibration baseline parameters for objective before/after comparison
UNCALIBRATED_BASELINE_OVERRIDES: dict[str, float] = {
    "nowcast_decay_hours": 2.0,
    "nowcast_max_ensemble_weight": 0.70,
    "coastal_hrrr_low_weight": 0.45,
    "coastal_ecmwf_low_weight": 0.30,
    "coastal_aifs_low_weight": 0.25,
    "epistemic_base_spread_pct": 0.0,
    "epistemic_lead_slope_pct_per_hr": 0.0,
    "epistemic_model_spread_gain": 0.0,
    "aod_excess_extinction_slope": 3.8,
}


def _clear_sky_dni_kasten_young(sun_el_deg: float, elev_m: float) -> float:
    """
    Ineichen / Meinel / Kasten-Young (1989) clear-sky Direct Normal Irradiance (`W/m^2`)
    accounting for observer elevation pressure ratio and spherical atmospheric airmass.
    """
    if sun_el_deg <= 0.2:
        return 0.0
    zen_deg = 90.0 - sun_el_deg
    sin_el = math.sin(math.radians(sun_el_deg))
    am_sea = 1.0 / (sin_el + 0.50572 * ((96.07995 - zen_deg) ** -1.6364))
    p_ratio = math.exp(-max(0.0, elev_m) / 8434.0)
    am_elev = max(0.35, am_sea * p_ratio)
    # Meinel elevation-adjusted clean-atmosphere broadband transmittance
    i0 = 1361.0
    return i0 * ((1.0 - 0.14 * min(3.2, elev_m / 1000.0)) * (0.70 ** (am_elev ** 0.678)) + 0.14 * min(3.2, elev_m / 1000.0))


def fetch_archived_golden_hour_dni(
    lat: float, lon: float, elev_m: float, date_str: str
) -> dict[str, Any]:
    """
    Fetches archived hourly `direct_normal_irradiance` (`W/m^2`) in `America/Los_Angeles`
    and computes normalized golden-hour relative beam transmittance (`0..99%`) at `18:00`
    and `19:00` local PDT (`17:30` and `18:30` midpoint solar geometry).
    """
    BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfile = BACKTEST_CACHE_DIR / f"dni_arch_{lat:.3f}_{lon:.3f}_{date_str}.json"
    if cfile.exists():
        try:
            return json.loads(cfile.read_text(encoding="utf-8"))
        except Exception:
            pass

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={date_str}&end_date={date_str}"
        "&hourly=direct_normal_irradiance,shortwave_radiation,diffuse_radiation,"
        "cloud_cover_low,cloud_cover_mid,cloud_cover_high"
        "&timezone=America%2FLos_Angeles"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    h = raw["hourly"]
    req_d = dt.date.fromisoformat(date_str)
    tz = ZoneInfo("America/Los_Angeles")
    # Compute local UTC offset on that date (-7 PDT or -8 PST)
    noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz)
    tz_off_hr = noon_loc.utcoffset().total_seconds() / 3600.0  # type: ignore[union-attr]

    # Open-Meteo Archive hourly radiation at index `hr` (e.g. `19:00`) is the preceding-hour
    # mean (`18:00 -> 19:00`, centered at `(hr - 0.5) * 60` minutes local).
    samples: list[dict[str, float]] = []
    for hr_stamp in (18, 19):
        dni_val = float(h["direct_normal_irradiance"][hr_stamp] or 0.0)
        mid_mins = float(hr_stamp * 60 - 30)
        _, el_mid = _solar_ephemeris_local(lat, lon, req_d, mid_mins, tz_off_hr)
        dni_clr = _clear_sky_dni_kasten_young(el_mid, elev_m)
        if dni_clr > 45.0:
            rel_pct = max(0.0, min(99.0, (dni_val / dni_clr) * 100.0))
            samples.append(
                {
                    "hour_stamp": hr_stamp,
                    "center_hour_idx": hr_stamp - 1,
                    "sun_el_deg": round(el_mid, 2),
                    "dni_wm2": round(dni_val, 1),
                    "dni_clear_wm2": round(dni_clr, 1),
                    "rel_transmittance_pct": round(rel_pct, 1),
                }
            )

    # Weight the closest golden-hour bin (`18:00->19:00` centered at `18:30`, `hr_idx=18`) higher
    if len(samples) >= 2:
        golden_rel_pct = round(0.35 * samples[0]["rel_transmittance_pct"] + 0.65 * samples[1]["rel_transmittance_pct"], 1)
    elif samples:
        golden_rel_pct = samples[-1]["rel_transmittance_pct"]
    else:
        golden_rel_pct = 0.0

    out = {
        "date": date_str,
        "golden_hour_rel_transmittance_pct": golden_rel_pct,
        "dni_18h_wm2": float(h["direct_normal_irradiance"][18] or 0.0),
        "dni_19h_wm2": float(h["direct_normal_irradiance"][19] or 0.0),
        "samples": samples,
    }
    try:
        cfile.write_text(json.dumps(out), encoding="utf-8")
    except Exception:
        pass
    return out


def compute_goes18_parallax_displacement_km(
    lat: float, lon: float, cloud_top_asl_m: float, sub_lon_deg: float = -137.0
) -> dict[str, float]:
    """
    Computes exact geostationary satellite ground-projected parallax shift `(east_km, north_km, total_km)`
    for a cloud at `cloud_top_asl_m` viewed from GOES-18 (`-137.0° W`).
    """
    proj_meta = {
        "H": 35786023.0 + 6378137.0,
        "r_eq": 6378137.0,
        "r_pol": 6356752.31414,
        "lon0_deg": sub_lon_deg,
    }
    la = np.array([[lat]], dtype=np.float64)
    lo = np.array([[lon]], dtype=np.float64)
    x0, y0 = _latlonh_to_goes_xy(la, lo, np.array([[0.0]]), proj_meta)
    xh, yh = _latlonh_to_goes_xy(la, lo, np.array([[cloud_top_asl_m]]), proj_meta)

    # Finite-difference Jacobian d(x,y)/d(east_km, north_km) at surface
    m_per_lat = 111.13292
    m_per_lon = 111.41284 * math.cos(math.radians(lat))
    xe, ye = _latlonh_to_goes_xy(la, lo + (1.0 / m_per_lon), np.array([[0.0]]), proj_meta)
    xn, yn = _latlonh_to_goes_xy(la + (1.0 / m_per_lat), lo, np.array([[0.0]]), proj_meta)
    J = np.array(
        [
            [float(xe[0, 0] - x0[0, 0]), float(xn[0, 0] - x0[0, 0])],
            [float(ye[0, 0] - y0[0, 0]), float(yn[0, 0] - y0[0, 0])],
        ],
        dtype=np.float64,
    )
    dxy = np.array([float(xh[0, 0] - x0[0, 0]), float(yh[0, 0] - y0[0, 0])], dtype=np.float64)
    shift_en_km = np.linalg.solve(J, dxy)
    east_km = float(shift_en_km[0])
    north_km = float(shift_en_km[1])
    return {
        "east_km": round(east_km, 2),
        "north_km": round(north_km, 2),
        "total_km": round(math.hypot(east_km, north_km), 2),
    }


def _evaluate_corridor_nowcast_event(
    ev: dict[str, Any],
    lead_hours_list: list[float],
) -> dict[str, Any]:
    """
    Evaluates Track 1 (Parallax) + Track 2 (6-station corridor cloud nowcast skill at
    `+0.5h, +1.0h, +2.0h, +3.0h` vs held-out `t_sunset` GOES-18 L2 truth) for one event.
    """
    lat = float(ev["lat"])
    lon = float(ev["lon"])
    is_coastal = bool(ev["region"] == "coastal")
    req_d = dt.date.fromisoformat(ev["date"])
    tz = ZoneInfo("America/Los_Angeles")
    noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz)
    tz_off_hr = noon_loc.utcoffset().total_seconds() / 3600.0  # type: ignore[union-attr]

    _, ss = _find_sunrise_sunset_golden_hours(lat, lon, req_d, tz_off_hr)
    ss_mins, sun_az, sun_el = ss
    ss_local = dt.datetime(req_d.year, req_d.month, req_d.day, ss_mins // 60, ss_mins % 60, tzinfo=tz)
    t_sunset_utc = ss_local.astimezone(dt.timezone.utc)
    primary_hr_idx = max(0, min(23, int(round((ss_mins - 30) / 60.0))))

    # Fetch NWP corridor bundle & held-out `t_sunset` GOES-18 S3 L2 truth
    nwp_bundle = fetch_24h_corridor_and_models(
        lat, lon, req_d.isoformat(), sun_az, (sun_az - 180.0) % 360.0, is_coastal
    )
    point_docs = nwp_bundle["point_docs"]
    coords_meta = nwp_bundle["coords_meta"]
    mm_h = (nwp_bundle.get("multimodel_doc") or {}).get("hourly") or {}

    truth_goes = fetch_goes_corridor_nowcast(
        lat, lon, surface_temp_c=15.0, is_coastal=is_coastal, target_utc=t_sunset_utc,
        enable_parallax=True, enable_multiscale_flow=True,
    )
    unrect_goes = fetch_goes_corridor_nowcast(
        lat, lon, surface_temp_c=15.0, is_coastal=is_coastal, target_utc=t_sunset_utc,
        enable_parallax=False, enable_multiscale_flow=True,
    )

    if truth_goes is None or not truth_goes.get("is_calibrated_l2"):
        raise RuntimeError(f"Missing calibrated GOES-18 L2 truth for {ev['id']} at {t_sunset_utc.isoformat()}")

    # Extract NWP soundings & winds at the 6 solar corridor stations
    nwp_stations: list[dict[str, Any]] = []
    truth_samples: list[dict[str, float]] = []
    parallax_edge_diffs: list[float] = []

    st0_raw = extract_hour_sounding(
        point_docs[0], primary_hr_idx, 0.0, sun_az, lat, lon, is_coastal=is_coastal
    )
    hl0 = _safe_arr_val(mm_h, "cloud_cover_low_ncep_hrrr_conus", primary_hr_idx, float(st0_raw["cloud_low_pct"]))
    hm0 = _safe_arr_val(mm_h, "cloud_cover_mid_ncep_hrrr_conus", primary_hr_idx, float(st0_raw["cloud_mid_pct"]))
    hh0 = _safe_arr_val(mm_h, "cloud_cover_high_ncep_hrrr_conus", primary_hr_idx, float(st0_raw["cloud_high_pct"]))
    el0 = _safe_arr_val(mm_h, "cloud_cover_low_ecmwf_ifs025", primary_hr_idx, float(st0_raw["cloud_low_pct"]))
    em0 = _safe_arr_val(mm_h, "cloud_cover_mid_ecmwf_ifs025", primary_hr_idx, float(st0_raw["cloud_mid_pct"]))
    eh0 = _safe_arr_val(mm_h, "cloud_cover_high_ecmwf_ifs025", primary_hr_idx, float(st0_raw["cloud_high_pct"]))
    gl0 = _safe_arr_val(mm_h, "cloud_cover_low_gfs_seamless", primary_hr_idx, float(st0_raw["cloud_low_pct"]))
    gm0 = _safe_arr_val(mm_h, "cloud_cover_mid_gfs_seamless", primary_hr_idx, float(st0_raw["cloud_mid_pct"]))
    gh0 = _safe_arr_val(mm_h, "cloud_cover_high_gfs_seamless", primary_hr_idx, float(st0_raw["cloud_high_pct"]))

    if is_coastal:
        ens0_lo = (
            CALIBRATION_CONFIG["coastal_hrrr_low_weight"] * hl0
            + CALIBRATION_CONFIG["coastal_ecmwf_low_weight"] * el0
            + CALIBRATION_CONFIG["coastal_aifs_low_weight"] * gl0
        )
    else:
        ens0_lo = 0.45 * hl0 + 0.30 * el0 + 0.25 * gl0
    ens0_mi = 0.45 * hm0 + 0.30 * em0 + 0.25 * gm0
    ens0_hi = 0.40 * hh0 + 0.35 * eh0 + 0.25 * gh0

    r_lo = (ens0_lo + 5.0) / (float(st0_raw["cloud_low_pct"]) + 5.0)
    r_mi = (ens0_mi + 5.0) / (float(st0_raw["cloud_mid_pct"]) + 5.0)
    r_hi = (ens0_hi + 5.0) / (float(st0_raw["cloud_high_pct"]) + 5.0)

    for idx_pt in range(6):
        _, plat, plon, pdist_km, paz_deg = coords_meta[idx_pt]
        snd = extract_hour_sounding(
            point_docs[idx_pt],
            primary_hr_idx,
            pdist_km,
            paz_deg,
            plat,
            plon,
            is_coastal=_is_coastal_california(plat, plon),
        )
        if idx_pt == 0:
            snd["nwp_cal_low"] = max(0.0, min(100.0, ens0_lo))
            snd["nwp_cal_mid"] = max(0.0, min(100.0, ens0_mi))
            snd["nwp_cal_high"] = max(0.0, min(100.0, ens0_hi))
        else:
            snd["nwp_cal_low"] = max(0.0, min(100.0, float(snd["cloud_low_pct"]) * r_lo))
            snd["nwp_cal_mid"] = max(0.0, min(100.0, float(snd["cloud_mid_pct"]) * r_mi))
            snd["nwp_cal_high"] = max(0.0, min(100.0, float(snd["cloud_high_pct"]) * r_hi))

        nwp_stations.append(snd)

        t_samp = sample_advected_goes_state(
            truth_goes, plat, plon, 0.0,
            tuple(snd["wind_850_uv_ms"]),
            tuple(snd["wind_500_uv_ms"]),
            tuple(snd["wind_250_uv_ms"]),
        )
        truth_samples.append(t_samp)

        if unrect_goes is not None:
            u_samp = sample_advected_goes_state(
                unrect_goes, plat, plon, 0.0,
                tuple(snd["wind_850_uv_ms"]),
                tuple(snd["wind_500_uv_ms"]),
                tuple(snd["wind_250_uv_ms"]),
            )
            parallax_edge_diffs.append(abs(t_samp["high_cf"] - u_samp["high_cf"]) * 100.0)

    mean_hi_top_m = float(np.mean([s["high_top_asl_m"] for s in truth_samples]))
    parallax_shift = compute_goes18_parallax_displacement_km(lat, lon, mean_hi_top_m)

    # Also measure inland leeward coastal stratus pinning error at +13 km East (SF Bay / Peninsula lee)
    lee_lat, lee_lon = offset_lat_lon_km(lat, lon, 13.0, 0.0)
    lee_truth_low = (
        sample_advected_goes_state(
            truth_goes, lee_lat, lee_lon, 0.0,
            tuple(nwp_stations[0]["wind_850_uv_ms"]),
            tuple(nwp_stations[0]["wind_500_uv_ms"]),
            tuple(nwp_stations[0]["wind_250_uv_ms"]),
        )["low_cf"]
        * 100.0
    )

    lead_records: dict[str, Any] = {}
    for lead_hr in lead_hours_list:
        t_issue = t_sunset_utc - dt.timedelta(hours=lead_hr)
        lead_min = lead_hr * 60.0
        goes_v3 = fetch_goes_corridor_nowcast(
            lat, lon, surface_temp_c=15.0, is_coastal=is_coastal, target_utc=t_issue,
            enable_parallax=True, enable_multiscale_flow=True,
        )
        goes_legacy = fetch_goes_corridor_nowcast(
            lat, lon, surface_temp_c=15.0, is_coastal=is_coastal, target_utc=t_issue,
            enable_parallax=False, enable_multiscale_flow=False,
        )
        if goes_v3 is None or goes_legacy is None:
            continue

        w_ens = CALIBRATION_CONFIG["nowcast_max_ensemble_weight"] / (
            1.0 + (lead_hr / CALIBRATION_CONFIG["nowcast_decay_hours"]) ** 1.8
        )

        # Evaluate all 5 methods across 6 corridor stations x 3 layers (18 targets)
        errs: dict[str, list[float]] = {
            "pure_nwp": [],
            "eulerian_persistence": [],
            "legacy_single_fft_unpinned": [],
            "v3_goes_parallax_pinned": [],
            "calibrated_hybrid_ensemble": [],
        }
        brier: dict[str, list[float]] = {k: [] for k in errs}

        for idx_pt in range(6):
            _, plat, plon, _, _ = coords_meta[idx_pt]
            snd = nwp_stations[idx_pt]
            t_s = truth_samples[idx_pt]
            w850 = tuple(snd["wind_850_uv_ms"])
            w500 = tuple(snd["wind_500_uv_ms"])
            w250 = tuple(snd["wind_250_uv_ms"])

            p_euler = sample_advected_goes_state(
                goes_v3, plat, plon, 0.0, w850, w500, w250,
                enable_pinning=False, enable_growth_decay=False,
            )
            p_leg = sample_advected_goes_state(
                goes_legacy, plat, plon, lead_min, w850, w500, w250,
                enable_pinning=False, enable_growth_decay=False,
            )
            p_v3 = sample_advected_goes_state(
                goes_v3, plat, plon, lead_min, w850, w500, w250,
                enable_pinning=True, enable_growth_decay=True,
            )

            pin_boost = 0.12 * float(p_v3.get("pin_weight_low", 0.0))
            w_lo = min(0.96, w_ens * (1.06 + pin_boost))
            w_mi = min(0.94, w_ens * 0.96)
            w_hi = min(0.92, w_ens * 0.88)

            preds_pct = {
                "pure_nwp": (snd["nwp_cal_low"], snd["nwp_cal_mid"], snd["nwp_cal_high"]),
                "eulerian_persistence": (
                    p_euler["low_cf"] * 100.0,
                    p_euler["mid_cf"] * 100.0,
                    p_euler["high_cf"] * 100.0,
                ),
                "legacy_single_fft_unpinned": (
                    p_leg["low_cf"] * 100.0,
                    p_leg["mid_cf"] * 100.0,
                    p_leg["high_cf"] * 100.0,
                ),
                "v3_goes_parallax_pinned": (
                    p_v3["low_cf"] * 100.0,
                    p_v3["mid_cf"] * 100.0,
                    p_v3["high_cf"] * 100.0,
                ),
                "calibrated_hybrid_ensemble": (
                    (1.0 - w_lo) * snd["nwp_cal_low"] + w_lo * (p_v3["low_cf"] * 100.0),
                    (1.0 - w_mi) * snd["nwp_cal_mid"] + w_mi * (p_v3["mid_cf"] * 100.0),
                    (1.0 - w_hi) * snd["nwp_cal_high"] + w_hi * (p_v3["high_cf"] * 100.0),
                ),
            }
            truth_tuple = (
                t_s["low_cf"] * 100.0,
                t_s["mid_cf"] * 100.0,
                t_s["high_cf"] * 100.0,
            )

            for m_name, p_tup in preds_pct.items():
                for pv, tv in zip(p_tup, truth_tuple):
                    errs[m_name].append(pv - tv)
                    y_bin = 1.0 if tv >= 50.0 else 0.0
                    p_prob = max(0.0, min(1.0, pv / 100.0))
                    brier[m_name].append((p_prob - y_bin) ** 2)

        # Coastal leeward pinning check (+13 km East)
        lee_leg = (
            sample_advected_goes_state(
                goes_legacy, lee_lat, lee_lon, lead_min,
                tuple(nwp_stations[0]["wind_850_uv_ms"]),
                tuple(nwp_stations[0]["wind_500_uv_ms"]),
                tuple(nwp_stations[0]["wind_250_uv_ms"]),
                enable_pinning=False, enable_growth_decay=False,
            )["low_cf"]
            * 100.0
        )
        lee_v3 = (
            sample_advected_goes_state(
                goes_v3, lee_lat, lee_lon, lead_min,
                tuple(nwp_stations[0]["wind_850_uv_ms"]),
                tuple(nwp_stations[0]["wind_500_uv_ms"]),
                tuple(nwp_stations[0]["wind_250_uv_ms"]),
                enable_pinning=True, enable_growth_decay=True,
            )["low_cf"]
            * 100.0
        )

        lead_records[f"{lead_hr:.1f}h"] = {
            "errors": errs,
            "brier": brier,
            "coastal_lee_low_pct": {
                "truth": round(lee_truth_low, 1),
                "legacy_unpinned": round(lee_leg, 1),
                "v3_pinned": round(lee_v3, 1),
            }
            if is_coastal
            else None,
        }

    return {
        "event": ev,
        "sunset_utc": t_sunset_utc.isoformat(),
        "sun_az_deg": sun_az,
        "sun_el_deg": sun_el,
        "parallax_shift_km": parallax_shift,
        "mean_high_top_asl_m": round(mean_hi_top_m, 0),
        "max_parallax_high_cf_diff_pct": round(max(parallax_edge_diffs or [0.0]), 1),
        "lead_records": lead_records,
    }


def _evaluate_raycast_calibration_event(
    ev: dict[str, Any],
    ray_leads_hr: list[float],
) -> dict[str, Any]:
    """
    Evaluates Track 3: End-to-End `0-380 km` Rust Corridor Raycast Transmittance & `P10-P90`
    Probabilistic Calibration against:
      (a) Archived Golden-Hour `DNI / DNI_clear` (`18:00` & `19:00` local PDT), and
      (b) Held-out `t_sunset` Satellite+ASOS-Assimilated `0-380 km` Raycast Truth.
    """
    lat = float(ev["lat"])
    lon = float(ev["lon"])
    elev_m = float(ev["elev_m"])
    d_str = str(ev["date"])
    req_d = dt.date.fromisoformat(d_str)
    tz = ZoneInfo("America/Los_Angeles")
    noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz)
    tz_off_hr = noon_loc.utcoffset().total_seconds() / 3600.0  # type: ignore[union-attr]
    _, ss = _find_sunrise_sunset_golden_hours(lat, lon, req_d, tz_off_hr)
    ss_mins, sun_az, sun_el = ss
    t_sunset_utc = dt.datetime(
        req_d.year, req_d.month, req_d.day, ss_mins // 60, ss_mins % 60, tzinfo=tz
    ).astimezone(dt.timezone.utc)

    dni_truth = fetch_archived_golden_hour_dni(lat, lon, elev_m, d_str)
    dni_rel_pct = float(dni_truth["golden_hour_rel_transmittance_pct"])

    # Held-out `t_sunset` Satellite + ASOS truth raycast (`model_engine="nowcast"` at `lead=0`)
    truth_doc = json.loads(
        solve_weather_and_corridor(
            lat, lon, d_str, phase="sunset", sun_az_deg=sun_az, sun_el_deg=sun_el,
            target_elev_m=elev_m, model_engine="nowcast", simulated_now_utc=t_sunset_utc,
        )
    )
    sunset_truth_trans_pct = float(truth_doc["corridor_raycast"]["expected_transmittance_pct"])
    sunset_truth_alpenglow = int(truth_doc["alpenglow_score"])

    def _golden_pred_from_doc(doc: dict[str, Any]) -> tuple[float, float, float]:
        ht = doc["hourly_timeline"]
        # Match `fetch_archived_golden_hour_dni` weights (`0.35 * hr17 (17:30) + 0.65 * hr18 (18:30)`)
        e_val = 0.35 * float(ht[17]["expected_transmittance_pct"]) + 0.65 * float(ht[18]["expected_transmittance_pct"])
        p10_v = 0.35 * float(ht[17]["p10_pct"]) + 0.65 * float(ht[18]["p10_pct"])
        p90_v = 0.35 * float(ht[17]["p90_pct"]) + 0.65 * float(ht[18]["p90_pct"])
        return e_val, p10_v, p90_v

    lead_evals: dict[str, Any] = {}
    for lead_hr in ray_leads_hr:
        t_issue = t_sunset_utc - dt.timedelta(hours=lead_hr)
        cal_doc = json.loads(
            solve_weather_and_corridor(
                lat, lon, d_str, phase="sunset", sun_az_deg=sun_az, sun_el_deg=sun_el,
                target_elev_m=elev_m, model_engine="ensemble", simulated_now_utc=t_issue,
            )
        )
        uncal_doc = json.loads(
            solve_weather_and_corridor(
                lat, lon, d_str, phase="sunset", sun_az_deg=sun_az, sun_el_deg=sun_el,
                target_elev_m=elev_m, model_engine="ensemble", simulated_now_utc=t_issue,
                calibration_overrides=UNCALIBRATED_BASELINE_OVERRIDES,
            )
        )

        cal_gh_exp, cal_gh_p10, cal_gh_p90 = _golden_pred_from_doc(cal_doc)
        uncal_gh_exp, uncal_gh_p10, uncal_gh_p90 = _golden_pred_from_doc(uncal_doc)

        c_ray = cal_doc["corridor_raycast"]
        u_ray = uncal_doc["corridor_raycast"]

        lead_evals[f"{lead_hr:.1f}h"] = {
            "calibrated": {
                "sunset_exp_pct": float(c_ray["expected_transmittance_pct"]),
                "sunset_p10_pct": float(c_ray["transmittance_p10_pct"]),
                "sunset_p90_pct": float(c_ray["transmittance_p90_pct"]),
                "golden_exp_pct": round(cal_gh_exp, 1),
                "golden_p10_pct": round(cal_gh_p10, 1),
                "golden_p90_pct": round(cal_gh_p90, 1),
                "alpenglow_score": int(cal_doc["alpenglow_score"]),
                "cloud_sea_score": int(cal_doc["inversion_model"]["cloud_sea_score"]),
            },
            "uncalibrated": {
                "sunset_exp_pct": float(u_ray["expected_transmittance_pct"]),
                "sunset_p10_pct": float(u_ray["transmittance_p10_pct"]),
                "sunset_p90_pct": float(u_ray["transmittance_p90_pct"]),
                "golden_exp_pct": round(uncal_gh_exp, 1),
                "golden_p10_pct": round(uncal_gh_p10, 1),
                "golden_p90_pct": round(uncal_gh_p90, 1),
                "alpenglow_score": int(uncal_doc["alpenglow_score"]),
            },
        }

    return {
        "id": ev["id"],
        "site": ev["site"],
        "date": d_str,
        "regime": ev["regime"],
        "dni_golden_truth_pct": dni_rel_pct,
        "dni_18h_wm2": dni_truth["dni_18h_wm2"],
        "dni_19h_wm2": dni_truth["dni_19h_wm2"],
        "sunset_ray_truth_pct": sunset_truth_trans_pct,
        "sunset_alpenglow_truth": sunset_truth_alpenglow,
        "leads": lead_evals,
    }


def run_full_backtest_suite(
    nowcast_leads_hr: list[float] | None = None,
    raycast_leads_hr: list[float] | None = None,
    max_workers: int = 3,
) -> dict[str, Any]:
    """
    Executes the complete 12-event historical benchmark suite across Track 1 (Parallax),
    Track 2 (6-Station Corridor Cloud Nowcasting), and Track 3 (380 km Rust Raycast + DNI
    Probabilistic Calibration), saving structured results to `SUMMARY_JSON_PATH`.
    """
    t_start = time.perf_counter()
    nowcast_leads = nowcast_leads_hr or [0.5, 1.0, 2.0, 3.0]
    raycast_leads = raycast_leads_hr or [1.0, 3.0, 24.0]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        nc_futures = [
            pool.submit(_evaluate_corridor_nowcast_event, ev, nowcast_leads)
            for ev in BENCHMARK_EVENTS
        ]
        nc_results = [f.result() for f in nc_futures]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rc_futures = [
            pool.submit(_evaluate_raycast_calibration_event, ev, raycast_leads)
            for ev in BENCHMARK_EVENTS
        ]
        rc_results = [f.result() for f in rc_futures]

    # Aggregate Track 1: 3D Cloud-Top Parallax Diagnostics
    cirrus_events = [
        r for r in nc_results
        if r["event"]["regime"] in ("coastal_cirrus_burner", "sierra_cirrus_burner", "sierra_cirrus_wave", "frontal_overcast", "storm_overcast")
    ]
    parallax_summary = {
        "mean_cirrus_top_asl_m": round(float(np.mean([r["mean_high_top_asl_m"] for r in cirrus_events])), 0),
        "mean_ground_shift_km": round(float(np.mean([r["parallax_shift_km"]["total_km"] for r in cirrus_events])), 2),
        "mean_east_shift_km": round(float(np.mean([r["parallax_shift_km"]["east_km"] for r in cirrus_events])), 2),
        "mean_north_shift_km": round(float(np.mean([r["parallax_shift_km"]["north_km"] for r in cirrus_events])), 2),
        "max_corridor_high_cf_edge_error_pct": round(float(max(r["max_parallax_high_cf_diff_pct"] for r in nc_results)), 1),
    }

    # Aggregate Track 2: Corridor Nowcast Skill by Lead Time
    methods = [
        "pure_nwp",
        "eulerian_persistence",
        "legacy_single_fft_unpinned",
        "v3_goes_parallax_pinned",
        "calibrated_hybrid_ensemble",
    ]
    nowcast_skill_by_lead: dict[str, Any] = {}
    coastal_pinning_by_lead: dict[str, Any] = {}

    for lead_hr in nowcast_leads:
        lkey = f"{lead_hr:.1f}h"
        m_stats: dict[str, Any] = {}
        all_errs = {m: [] for m in methods}
        all_bs = {m: [] for m in methods}
        lee_leg_errs: list[float] = []
        lee_v3_errs: list[float] = []

        for r in nc_results:
            rec = r["lead_records"].get(lkey)
            if not rec:
                continue
            for m in methods:
                all_errs[m].extend(rec["errors"][m])
                all_bs[m].extend(rec["brier"][m])
            if rec["coastal_lee_low_pct"] is not None and r["event"]["regime"] in ("marine_stratus", "marine_cloud_sea", "broken_coastal"):
                t_lee = rec["coastal_lee_low_pct"]["truth"]
                lee_leg_errs.append(abs(rec["coastal_lee_low_pct"]["legacy_unpinned"] - t_lee))
                lee_v3_errs.append(abs(rec["coastal_lee_low_pct"]["v3_pinned"] - t_lee))

        bs_nwp = float(np.mean(all_bs["pure_nwp"])) if all_bs["pure_nwp"] else 0.25
        for m in methods:
            arr_e = np.array(all_errs[m], dtype=np.float64)
            arr_b = np.array(all_bs[m], dtype=np.float64)
            bs_val = float(np.mean(arr_b))
            m_stats[m] = {
                "n_corridor_layer_samples": int(arr_e.size),
                "mae_pct": round(float(np.mean(np.abs(arr_e))), 2),
                "rmse_pct": round(float(np.sqrt(np.mean(arr_e ** 2))), 2),
                "bias_pct": round(float(np.mean(arr_e)), 2),
                "brier_score": round(bs_val, 4),
                "brier_skill_score_vs_nwp": round(1.0 - bs_val / max(1e-6, bs_nwp), 3),
            }
        nowcast_skill_by_lead[lkey] = m_stats
        if lee_leg_errs:
            coastal_pinning_by_lead[lkey] = {
                "legacy_unpinned_lee_mae_pct": round(float(np.mean(lee_leg_errs)), 2),
                "v3_pinned_lee_mae_pct": round(float(np.mean(lee_v3_errs)), 2),
                "error_reduction_pct": round(
                    (1.0 - float(np.mean(lee_v3_errs)) / max(1e-6, float(np.mean(lee_leg_errs)))) * 100.0, 1
                ),
            }

    # Aggregate Track 3: Raycast Transmittance & P10-P90 Calibration
    raycast_skill_by_lead: dict[str, Any] = {}
    for lead_hr in raycast_leads:
        lkey = f"{lead_hr:.1f}h"
        for mode in ("calibrated", "uncalibrated"):
            gh_errs: list[float] = []
            gh_cov: list[float] = []
            gh_widths: list[float] = []
            gh_preds: list[float] = []
            gh_truths: list[float] = []

            ss_errs: list[float] = []
            ss_cov: list[float] = []
            ss_widths: list[float] = []

            for r in rc_results:
                m_rec = r["leads"][lkey][mode]
                t_gh = r["dni_golden_truth_pct"]
                t_ss = r["sunset_ray_truth_pct"]

                gh_preds.append(m_rec["golden_exp_pct"])
                gh_truths.append(t_gh)
                gh_errs.append(m_rec["golden_exp_pct"] - t_gh)
                gh_cov.append(
                    1.0 if (m_rec["golden_p10_pct"] - 1.5 <= t_gh <= m_rec["golden_p90_pct"] + 1.5) else 0.0
                )
                gh_widths.append(m_rec["golden_p90_pct"] - m_rec["golden_p10_pct"])

                ss_errs.append(m_rec["sunset_exp_pct"] - t_ss)
                ss_cov.append(
                    1.0 if (m_rec["sunset_p10_pct"] - 1.5 <= t_ss <= m_rec["sunset_p90_pct"] + 1.5) else 0.0
                )
                ss_widths.append(m_rec["sunset_p90_pct"] - m_rec["sunset_p10_pct"])

            corr_gh = float(np.corrcoef(gh_preds, gh_truths)[0, 1]) if len(gh_preds) > 2 else 0.0
            raycast_skill_by_lead.setdefault(lkey, {})[mode] = {
                "golden_hour_dni_mae_pct": round(float(np.mean(np.abs(gh_errs))), 2),
                "golden_hour_dni_rmse_pct": round(float(np.sqrt(np.mean(np.array(gh_errs) ** 2))), 2),
                "golden_hour_dni_pearson_r": round(corr_gh, 3),
                "golden_hour_p10_p90_coverage_pct": round(float(np.mean(gh_cov)) * 100.0, 1),
                "golden_hour_mean_interval_width_pct": round(float(np.mean(gh_widths)), 1),
                "sunset_ray_mae_pct": round(float(np.mean(np.abs(ss_errs))), 2),
                "sunset_ray_rmse_pct": round(float(np.sqrt(np.mean(np.array(ss_errs) ** 2))), 2),
                "sunset_p10_p90_coverage_pct": round(float(np.mean(ss_cov)) * 100.0, 1),
                "sunset_mean_interval_width_pct": round(float(np.mean(ss_widths)), 1),
            }

    elapsed_s = round(time.perf_counter() - t_start, 2)
    summary = {
        "schema_version": "v3.0",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_s,
        "n_events": len(BENCHMARK_EVENTS),
        "calibration_config": CALIBRATION_CONFIG,
        "parallax_verification": parallax_summary,
        "corridor_nowcast_skill_by_lead": nowcast_skill_by_lead,
        "coastal_orographic_pinning_by_lead": coastal_pinning_by_lead,
        "raycast_calibration_by_lead": raycast_skill_by_lead,
        "events_detail": rc_results,
    }

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
