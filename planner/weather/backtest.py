"""
Historical Ground-Truth Backtesting & Multi-Sensor Calibration Harness (`planner/weather/backtest.py`, v4.0).

Evaluates PhotoPlanner's unified multi-layer atmospheric nowcasting & forecasting engine across
**24 real 2025 California golden-hour shoots** spanning **all 4 seasons** (Winter, Spring, Summer,
Autumn) and **4 distinct geographic/elevation regimes**:
  1. Coastal Headlands (`Pedro Point`, `220 m` / `722 ft` ASL)
  2. Coastal Summit Cloud-Sea Ridges (`Mt. Tamalpais` `784 m` / `2,572 ft` & `Montara Mtn` `578 m` / `1,896 ft`)
  3. Central Coast / Monterey Bay (`Garrapata / Big Sur`, `180 m` / `591 ft` ASL)
  4. High Sierra Alpine (`Half Dome` `2,693 m` / `8,835 ft` & `Clouds Rest` `3,026 m` / `9,928 ft`)

Four Empirical Verification Tracks (strictly causal `scan_end_utc <= t_issue` at forecast time):
- **Track 1 (Vertical Altitude & Inversion Accuracy)**: Evaluates boundary-layer fog lid altitude,
  inversion strength (`°F`), ASOS ceilometer cloud-base alignment, and summit-above-fog vs
  immersed-in-fog classification against independent `00Z` NWS radiosonde balloons (`KOAK`, `KVBG`,
  `KREV`) and 5-minute ASOS laser ceilometers (`KHAF`, `KSFO`, `KDVO`, `KMRY`, `KMMH`).
- **Track 2 (Multi-Layer Cloud Capture, Parallax & Under-Cirrus Stratus Recall)**: Evaluates
  5-layer `ABI-L2-CCLC` + twilight `3.9 µm` `BTD` + ASOS/RAOB fusion vs legacy single-layer
  `ABI-L2-ACTPC`, 3D cloud-top parallax displacement (`GOES-18` `-137.0° W`), coastal leeward
  orographic stratus pinning, and 6-station `0–320 km` corridor cloud skill (`+0.5h..+3.0h`).
- **Track 3 (True Lead-Time Forecast Skill Curve: `+1h`, `+3h`, `+24h`, `+72h` Hindcasts)**:
  Evaluates causal nowcasts (`+1h`, `+3h`) and archived Open-Meteo `previous-runs-api` historical
  forecast runs (`+24h` Day-1, `+72h` Day-3) against both Multi-Sensor Observed Truth and
  independent pre-sunset pyranometer/irradiance transmittance (`DNI / DNI_clear`).
- **Track 4 (3D Cloud Hover-Card Confidence Calibration — `ECE` & Reliability Curve)**:
  Scores predicted per-layer hover-card `confidence_pct` (`fog`, `low`, `mid`, `high`) across
  all events and lead times against realized empirical layer accuracy, computing Expected
  Calibration Error (`ECE`) across confidence bins.
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
from .raob_sounding import fetch_raob_sounding
from .station_obs import fetch_nearest_station_observation

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKTEST_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "backtest_archive"
SUMMARY_JSON_PATH = ROOT_DIR / "data" / "weather_cache" / "backtest_summary_v3.json"
BENCHMARK_REPORT_JSON_PATH = ROOT_DIR / "data" / "weather_benchmarks" / "backtest_summary.json"
USER_AGENT = "PhotoPlanner/4.0 (Multi-Sensor Historical Atmospheric Benchmark)"

# 24 real 2025 California golden-hour benchmark shoots spanning all 4 seasons & 4 geographic regimes
BENCHMARK_EVENTS: list[dict[str, Any]] = [
    # --- Region 1: Coastal Headlands (Pedro Point, 220 m / 722 ft ASL) ---
    {
        "id": "pedro_2025_01_18_winter_clear",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "subregion": "coastal_headland",
        "season": "winter",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-01-18",
        "regime": "clear_corridor",
        "description": "Crisp winter offshore Pacific solar corridor with shallow marine haze",
    },
    {
        "id": "pedro_2025_05_06_stratus",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "subregion": "coastal_headland",
        "season": "spring",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-05-06",
        "regime": "marine_stratus",
        "description": "Solid coastal marine stratus deck capped below Montara Ridge",
    },
    {
        "id": "pedro_2025_05_12_broken",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "subregion": "coastal_headland",
        "season": "spring",
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
        "subregion": "coastal_headland",
        "season": "spring",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-05-16",
        "regime": "coastal_cirrus_burner",
        "description": "High Pacific ice cirrus canopy overlapping low marine boundary layer",
    },
    {
        "id": "pedro_2025_06_14_clear",
        "site": "Pedro Point Headlands",
        "region": "coastal",
        "subregion": "coastal_headland",
        "season": "summer",
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
        "subregion": "coastal_headland",
        "season": "autumn",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "date": "2025-09-29",
        "regime": "frontal_overcast",
        "description": "Deep multi-layer early-autumn Pacific frontal overcast",
    },
    # --- Region 2: Coastal Summit Cloud-Sea Ridges (Mt. Tamalpais 784 m & Montara Mtn 578 m) ---
    {
        "id": "tam_2025_02_08_winter_ridge",
        "site": "Mt. Tamalpais East Peak",
        "region": "coastal",
        "subregion": "coastal_summit",
        "season": "winter",
        "lat": 37.9235,
        "lon": -122.5965,
        "elev_m": 784.0,
        "date": "2025-02-08",
        "regime": "broken_coastal",
        "description": "Post-frontal winter coastal summit above scattered low stratocumulus",
    },
    {
        "id": "montara_2025_04_12_spring_lid",
        "site": "Montara Mountain North Peak",
        "region": "coastal",
        "subregion": "coastal_summit",
        "season": "spring",
        "lat": 37.5619,
        "lon": -122.4764,
        "elev_m": 578.0,
        "date": "2025-04-12",
        "regime": "marine_cloud_sea",
        "description": "Spring marine inversion lid testing 578 m coastal ridge clearance",
    },
    {
        "id": "tam_2025_06_05_cloud_sea",
        "site": "Mt. Tamalpais East Peak",
        "region": "coastal",
        "subregion": "coastal_summit",
        "season": "summer",
        "lat": 37.9235,
        "lon": -122.5965,
        "elev_m": 784.0,
        "date": "2025-06-05",
        "regime": "marine_cloud_sea",
        "description": "Classic 616 m KOAK balloon-capped marine cloud sea below 784 m summit",
    },
    {
        "id": "montara_2025_07_15_summer_stratus",
        "site": "Montara Mountain North Peak",
        "region": "coastal",
        "subregion": "coastal_summit",
        "season": "summer",
        "lat": 37.5619,
        "lon": -122.4764,
        "elev_m": 578.0,
        "date": "2025-07-15",
        "regime": "marine_stratus",
        "description": "Mid-July marine stratus deck testing coastal summit immersion vs clearance",
    },
    {
        "id": "tam_2025_08_09_summer_inversion",
        "site": "Mt. Tamalpais East Peak",
        "region": "coastal",
        "subregion": "coastal_summit",
        "season": "summer",
        "lat": 37.9235,
        "lon": -122.5965,
        "elev_m": 784.0,
        "date": "2025-08-09",
        "regime": "marine_cloud_sea",
        "description": "Strong August subsidence inversion capping Gulf of the Farallones stratus",
    },
    {
        "id": "tam_2025_10_18_autumn_inversion",
        "site": "Mt. Tamalpais East Peak",
        "region": "coastal",
        "subregion": "coastal_summit",
        "season": "autumn",
        "lat": 37.9235,
        "lon": -122.5965,
        "elev_m": 784.0,
        "date": "2025-10-18",
        "regime": "coastal_cirrus_burner",
        "description": "Autumn coastal marine inversion beneath upper-tropospheric Pacific cirrus",
    },
    # --- Region 3: Central Coast / Monterey Bay (Garrapata / Big Sur, 180 m / 591 ft ASL) ---
    {
        "id": "garrapata_2025_01_25_winter_coast",
        "site": "Garrapata Big Sur Bluffs",
        "region": "coastal",
        "subregion": "central_coast",
        "season": "winter",
        "lat": 36.4675,
        "lon": -121.9286,
        "elev_m": 180.0,
        "date": "2025-01-25",
        "regime": "broken_coastal",
        "description": "Winter Central Coast cool airmass with offshore stratocumulus bands",
    },
    {
        "id": "garrapata_2025_04_20_spring_sc",
        "site": "Garrapata Big Sur Bluffs",
        "region": "coastal",
        "subregion": "central_coast",
        "season": "spring",
        "lat": 36.4675,
        "lon": -121.9286,
        "elev_m": 180.0,
        "date": "2025-04-20",
        "regime": "broken_coastal",
        "description": "Spring Monterey Bay broken coastal stratocumulus & sunbeam corridor",
    },
    {
        "id": "garrapata_2025_07_22_summer_fog",
        "site": "Garrapata Big Sur Bluffs",
        "region": "coastal",
        "subregion": "central_coast",
        "season": "summer",
        "lat": 36.4675,
        "lon": -121.9286,
        "elev_m": 180.0,
        "date": "2025-07-22",
        "regime": "marine_stratus",
        "description": "Deep July Monterey Bay coastal stratus & twilight 3.9 µm BTD fog bank",
    },
    {
        "id": "garrapata_2025_10_11_autumn_multilayer",
        "site": "Garrapata Big Sur Bluffs",
        "region": "coastal",
        "subregion": "central_coast",
        "season": "autumn",
        "lat": 36.4675,
        "lon": -121.9286,
        "elev_m": 180.0,
        "date": "2025-10-11",
        "regime": "frontal_overcast",
        "description": "Autumn Central Coast multi-layer stratus beneath mid/high Pacific cloud",
    },
    # --- Region 4: High Sierra Alpine (Half Dome 2,693 m & Clouds Rest 3,026 m) ---
    {
        "id": "halfdome_2025_01_11_winter_alpine",
        "site": "Half Dome",
        "region": "sierra",
        "subregion": "high_sierra",
        "season": "winter",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-01-11",
        "regime": "clear_alpine",
        "description": "Winter High Sierra cold granite alpenglow & high jet stream wisps",
    },
    {
        "id": "halfdome_2025_05_03_storm",
        "site": "Half Dome",
        "region": "sierra",
        "subregion": "high_sierra",
        "season": "spring",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-03",
        "regime": "storm_overcast",
        "description": "Deep spring Sierra storm system with thick mid/high ice overcast",
    },
    {
        "id": "halfdome_2025_05_06_clear",
        "site": "Half Dome",
        "region": "sierra",
        "subregion": "high_sierra",
        "season": "spring",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-06",
        "regime": "clear_alpine",
        "description": "Pristine clear-sky High Sierra granite warm-Kelvin sunset",
    },
    {
        "id": "halfdome_2025_05_13_crest",
        "site": "Half Dome",
        "region": "sierra",
        "subregion": "high_sierra",
        "season": "spring",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-13",
        "regime": "crest_attenuation",
        "description": "Mid-tropospheric foothills/crest cloud deck attenuating western corridor",
    },
    {
        "id": "halfdome_2025_05_25_cirrus",
        "site": "Half Dome",
        "region": "sierra",
        "subregion": "high_sierra",
        "season": "spring",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-05-25",
        "regime": "sierra_cirrus_burner",
        "description": "Optically thin (COD ~0.65) 10.5 km High-Sierra cirrus alpenglow burner",
    },
    {
        "id": "cloudsrest_2025_06_07_wave",
        "site": "Clouds Rest Summit",
        "region": "sierra",
        "subregion": "high_sierra",
        "season": "summer",
        "lat": 37.7678,
        "lon": -119.4893,
        "elev_m": 3026.0,
        "date": "2025-06-07",
        "regime": "sierra_cirrus_wave",
        "description": "Terrain-locked upper-tropospheric Sierra lee-wave cirrus banner",
    },
    {
        "id": "cloudsrest_2025_08_16_monsoon",
        "site": "Clouds Rest Summit",
        "region": "sierra",
        "subregion": "high_sierra",
        "season": "summer",
        "lat": 37.7678,
        "lon": -119.4893,
        "elev_m": 3026.0,
        "date": "2025-08-16",
        "regime": "mixed_orographic",
        "description": "Summer High Sierra monsoonal mid-level altocumulus & anvil cirrus",
    },
    {
        "id": "halfdome_2025_10_25_autumn_wave",
        "site": "Half Dome",
        "region": "sierra",
        "subregion": "high_sierra",
        "season": "autumn",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "date": "2025-10-25",
        "regime": "sierra_cirrus_wave",
        "description": "Autumn Sierra frontal approach with standing wave cloud deck",
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
    i0 = 1361.0
    return i0 * (
        (1.0 - 0.14 * min(3.2, elev_m / 1000.0)) * (0.70 ** (am_elev ** 0.678))
        + 0.14 * min(3.2, elev_m / 1000.0)
    )


def fetch_archived_golden_hour_dni(
    lat: float, lon: float, elev_m: float, date_str: str
) -> dict[str, Any]:
    """
    Fetches archived hourly `direct_normal_irradiance` (`W/m^2`) in `America/Los_Angeles`
    and dynamically identifies the exact 2-hour pre-sunset golden-hour window for any season
    (e.g. `15:00–17:00` PST in January vs `18:00–20:00` PDT in June/July).
    """
    BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfile = BACKTEST_CACHE_DIR / f"dni_arch_v4b_{lat:.3f}_{lon:.3f}_{date_str}.json"
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
    with urllib.request.urlopen(req, timeout=14.0) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    h = raw["hourly"]
    req_d = dt.date.fromisoformat(date_str)
    tz = ZoneInfo("America/Los_Angeles")
    noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz)
    tz_off_hr = noon_loc.utcoffset().total_seconds() / 3600.0  # type: ignore[union-attr]
    api_off_s = raw.get("utc_offset_seconds")
    api_off_hr = (float(api_off_s) / 3600.0) if api_off_s is not None else tz_off_hr
    hr_shift_to_civil = int(round(tz_off_hr - api_off_hr))
    grid_elev_m = float(raw.get("elevation")) if raw.get("elevation") is not None else elev_m

    _, ss_civil = _find_sunrise_sunset_golden_hours(lat, lon, req_d, tz_off_hr)
    ss_mins = int(ss_civil[0])
    _, ss_api = _find_sunrise_sunset_golden_hours(lat, lon, req_d, api_off_hr)
    ss_mins_api = int(ss_api[0])

    # Open-Meteo Archive returns `utc_offset_seconds = -28800` (`GMT-8` year-round, without DST shift).
    # Each radiation bin `hr_stamp` is the preceding-hour mean `[hr_stamp - 1, hr_stamp]` in `api_off_hr`,
    # centered at `(hr_stamp - 0.5) * 60` minutes in `api_off_hr`, which maps to civil `hourly_timeline`
    # index `(hr_stamp - 1) + hr_shift_to_civil`.
    candidate_samples: list[dict[str, Any]] = []
    for hr_stamp in range(13, 22):
        mid_mins_api = float(hr_stamp * 60 - 30)
        if mid_mins_api > ss_mins_api + 12:
            continue
        dni_val = float(h["direct_normal_irradiance"][hr_stamp] or 0.0)
        _, el_mid = _solar_ephemeris_local(lat, lon, req_d, mid_mins_api, api_off_hr)
        dni_clr = _clear_sky_dni_kasten_young(el_mid, grid_elev_m)
        if el_mid >= 2.2 and dni_clr > 55.0:
            rel_pct = max(0.0, min(99.0, (dni_val / dni_clr) * 100.0))
            civil_center_idx = max(0, min(23, int(hr_stamp - 1 + hr_shift_to_civil)))
            candidate_samples.append(
                {
                    "hour_stamp": int(hr_stamp),
                    "center_hour_idx": civil_center_idx,
                    "sun_el_deg": round(el_mid, 2),
                    "dni_wm2": round(dni_val, 1),
                    "dni_clear_wm2": round(dni_clr, 1),
                    "rel_transmittance_pct": round(rel_pct, 1),
                }
            )

    samples = candidate_samples[-2:] if len(candidate_samples) >= 2 else candidate_samples
    if len(samples) >= 2:
        golden_rel_pct = round(
            0.35 * float(samples[0]["rel_transmittance_pct"])
            + 0.65 * float(samples[1]["rel_transmittance_pct"]),
            1,
        )
    elif samples:
        golden_rel_pct = float(samples[-1]["rel_transmittance_pct"])
    else:
        golden_rel_pct = 0.0

    out = {
        "date": date_str,
        "sunset_mins_local": ss_mins,
        "api_utc_offset_hr": api_off_hr,
        "civil_utc_offset_hr": tz_off_hr,
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


def _layer_map(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    layers = (doc.get("atmospheric_layers") or {}).get("layers") or []
    return {str(l["id"]): l for l in layers if "id" in l}


def _golden_pred_from_doc(
    doc: dict[str, Any], dni_samples: list[dict[str, Any]]
) -> tuple[float, float, float]:
    ht = doc["hourly_timeline"]
    if len(dni_samples) >= 2:
        i0 = int(dni_samples[0]["center_hour_idx"])
        i1 = int(dni_samples[1]["center_hour_idx"])
        e_val = 0.35 * float(ht[i0]["expected_transmittance_pct"]) + 0.65 * float(ht[i1]["expected_transmittance_pct"])
        p10_v = 0.35 * float(ht[i0]["p10_pct"]) + 0.65 * float(ht[i1]["p10_pct"])
        p90_v = 0.35 * float(ht[i0]["p90_pct"]) + 0.65 * float(ht[i1]["p90_pct"])
        return e_val, p10_v, p90_v
    if len(dni_samples) == 1:
        i0 = int(dni_samples[0]["center_hour_idx"])
        return (
            float(ht[i0]["expected_transmittance_pct"]),
            float(ht[i0]["p10_pct"]),
            float(ht[i0]["p90_pct"]),
        )
    return (
        float(doc["corridor_raycast"]["expected_transmittance_pct"]),
        float(doc["corridor_raycast"]["transmittance_p10_pct"]),
        float(doc["corridor_raycast"]["transmittance_p90_pct"]),
    )


def _evaluate_corridor_nowcast_event(
    ev: dict[str, Any],
    lead_hours_list: list[float],
    refresh: bool = False,
) -> dict[str, Any]:
    """
    Evaluates satellite corridor cloud nowcast skill (`+0.5h..+3.0h`), 3D cloud-top parallax,
    coastal leeward pinning, and single-layer `ACTPC` vs multi-layer `CCLC + BTD` under-cirrus recall.
    """
    BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    leads_tag = "_".join(f"{x:.1f}" for x in lead_hours_list)
    ev_cache = BACKTEST_CACHE_DIR / f"nc_ev_v4_{ev['id']}_{leads_tag}.json"
    if (not refresh) and ev_cache.exists():
        try:
            cached = json.loads(ev_cache.read_text(encoding="utf-8"))
            print(f"  [Track 2 GOES] {ev['id']} ({ev['date']}) cached", flush=True)
            return cached
        except Exception:
            pass

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

    nwp_bundle = fetch_24h_corridor_and_models(
        lat, lon, req_d.isoformat(), sun_az, (sun_az - 180.0) % 360.0, is_coastal
    )
    point_docs = nwp_bundle["point_docs"]
    coords_meta = nwp_bundle["coords_meta"]
    mm_h = (nwp_bundle.get("multimodel_doc") or {}).get("hourly") or {}

    truth_goes = fetch_goes_corridor_nowcast(
        lat, lon, surface_temp_c=15.0, is_coastal=is_coastal, target_utc=t_sunset_utc,
        enable_parallax=True, enable_multiscale_flow=True, enable_multilayer_fusion=True,
    )
    single_layer_goes = fetch_goes_corridor_nowcast(
        lat, lon, surface_temp_c=15.0, is_coastal=is_coastal, target_utc=t_sunset_utc,
        enable_parallax=True, enable_multiscale_flow=True, enable_multilayer_fusion=False,
    )
    unrect_goes = fetch_goes_corridor_nowcast(
        lat, lon, surface_temp_c=15.0, is_coastal=is_coastal, target_utc=t_sunset_utc,
        enable_parallax=False, enable_multiscale_flow=True, enable_multilayer_fusion=True,
    )

    asos_truth = fetch_nearest_station_observation(lat, lon, target_utc=t_sunset_utc)
    raob_truth = fetch_raob_sounding(lat, lon, target_utc=t_sunset_utc, causal_only=False)

    if truth_goes is None or not truth_goes.get("is_calibrated_l2"):
        return {
            "event": ev,
            "sunset_utc": t_sunset_utc.isoformat(),
            "sun_az_deg": sun_az,
            "sun_el_deg": sun_el,
            "goes_available": False,
            "parallax_shift_km": {"east_km": 0.0, "north_km": 0.0, "total_km": 0.0},
            "mean_high_top_asl_m": 0.0,
            "max_parallax_high_cf_diff_pct": 0.0,
            "multilayer_vs_single_sat": None,
            "lead_records": {},
        }

    nwp_stations: list[dict[str, Any]] = []
    truth_samples: list[dict[str, float]] = []
    parallax_edge_diffs: list[float] = []

    api_off_s = point_docs[0].get("utc_offset_seconds") if point_docs else None
    api_hr_shift = int(round(float(api_off_s) / 3600.0 - tz_off_hr)) if api_off_s is not None else 0
    api_hr_idx = max(0, min(23, primary_hr_idx + api_hr_shift))
    mm_off_s = (nwp_bundle.get("multimodel_doc") or {}).get("utc_offset_seconds")
    mm_hr_shift = int(round(float(mm_off_s) / 3600.0 - tz_off_hr)) if mm_off_s is not None else api_hr_shift
    mm_hr_idx = max(0, min(23, primary_hr_idx + mm_hr_shift))

    st0_raw = extract_hour_sounding(
        point_docs[0], api_hr_idx, 0.0, sun_az, lat, lon, is_coastal=is_coastal
    )
    hl0 = _safe_arr_val(mm_h, "cloud_cover_low_ncep_hrrr_conus", mm_hr_idx, float(st0_raw["cloud_low_pct"]))
    hm0 = _safe_arr_val(mm_h, "cloud_cover_mid_ncep_hrrr_conus", mm_hr_idx, float(st0_raw["cloud_mid_pct"]))
    hh0 = _safe_arr_val(mm_h, "cloud_cover_high_ncep_hrrr_conus", mm_hr_idx, float(st0_raw["cloud_high_pct"]))
    el0 = _safe_arr_val(mm_h, "cloud_cover_low_ecmwf_ifs025", mm_hr_idx, float(st0_raw["cloud_low_pct"]))
    em0 = _safe_arr_val(mm_h, "cloud_cover_mid_ecmwf_ifs025", mm_hr_idx, float(st0_raw["cloud_mid_pct"]))
    eh0 = _safe_arr_val(mm_h, "cloud_cover_high_ecmwf_ifs025", mm_hr_idx, float(st0_raw["cloud_high_pct"]))
    gl0 = _safe_arr_val(mm_h, "cloud_cover_low_gfs_seamless", mm_hr_idx, float(st0_raw["cloud_low_pct"]))
    gm0 = _safe_arr_val(mm_h, "cloud_cover_mid_gfs_seamless", mm_hr_idx, float(st0_raw["cloud_mid_pct"]))
    gh0 = _safe_arr_val(mm_h, "cloud_cover_high_gfs_seamless", mm_hr_idx, float(st0_raw["cloud_high_pct"]))

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
            api_hr_idx,
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

    # Compare Multi-Layer CCLC+BTD vs Legacy Single-Layer ACTPC at nearest ASOS station (bottom-up truth)
    multilayer_vs_single_sat: dict[str, Any] | None = None
    if single_layer_goes is not None and asos_truth is not None and float(asos_truth.get("distance_km", 999.0)) <= 55.0:
        st_lat = float(asos_truth["station_lat"])
        st_lon = float(asos_truth["station_lon"])
        w850_0 = tuple(nwp_stations[0]["wind_850_uv_ms"])
        w500_0 = tuple(nwp_stations[0]["wind_500_uv_ms"])
        w250_0 = tuple(nwp_stations[0]["wind_250_uv_ms"])
        ml_at_asos = sample_advected_goes_state(truth_goes, st_lat, st_lon, 0.0, w850_0, w500_0, w250_0)
        sl_at_asos = sample_advected_goes_state(single_layer_goes, st_lat, st_lon, 0.0, w850_0, w500_0, w250_0)
        asos_low_pct = float(asos_truth.get("low_cloud_fraction", 0.0)) * 100.0
        raob_low_pct = (
            float(raob_truth.get("low_cloud_fraction_proxy", 0.0)) * 100.0
            if (raob_truth and float(raob_truth.get("distance_km", 999.0)) <= 120.0)
            else asos_low_pct
        )
        ref_low_pct = max(asos_low_pct, 0.7 * asos_low_pct + 0.3 * raob_low_pct) if is_coastal else asos_low_pct
        multilayer_vs_single_sat = {
            "asos_station_id": asos_truth["station_id"],
            "asos_low_pct": round(asos_low_pct, 1),
            "bottom_up_ref_low_pct": round(ref_low_pct, 1),
            "upper_high_cf_pct": round(ml_at_asos["high_cf"] * 100.0, 1),
            "upper_mid_cf_pct": round(ml_at_asos["mid_cf"] * 100.0, 1),
            "single_layer_sat_low_pct": round(sl_at_asos["low_cf"] * 100.0, 1),
            "multilayer_cclc_btd_low_pct": round(ml_at_asos["low_cf"] * 100.0, 1),
            "has_cclc": bool(truth_goes.get("has_cclc_multilayer") or truth_goes.get("has_cclc")),
            "has_btd": bool(truth_goes.get("has_twilight_btd_fog") or truth_goes.get("has_btd_fog")),
        }

    mean_hi_top_m = float(np.mean([s["high_top_asl_m"] for s in truth_samples]))
    parallax_shift = compute_goes18_parallax_displacement_km(lat, lon, mean_hi_top_m)

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
            enable_parallax=True, enable_multiscale_flow=True, enable_multilayer_fusion=True,
        )
        goes_legacy = fetch_goes_corridor_nowcast(
            lat, lon, surface_temp_c=15.0, is_coastal=is_coastal, target_utc=t_issue,
            enable_parallax=False, enable_multiscale_flow=False, enable_multilayer_fusion=False,
        )
        if goes_v3 is None or goes_legacy is None:
            continue

        w_ens = CALIBRATION_CONFIG["nowcast_max_ensemble_weight"] / (
            1.0 + (lead_hr / CALIBRATION_CONFIG["nowcast_decay_hours"]) ** 1.8
        )

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

    out_nc = {
        "event": ev,
        "sunset_utc": t_sunset_utc.isoformat(),
        "sun_az_deg": sun_az,
        "sun_el_deg": sun_el,
        "goes_available": True,
        "parallax_shift_km": parallax_shift,
        "mean_high_top_asl_m": round(mean_hi_top_m, 0),
        "max_parallax_high_cf_diff_pct": round(max(parallax_edge_diffs or [0.0]), 1),
        "multilayer_vs_single_sat": multilayer_vs_single_sat,
        "lead_records": lead_records,
    }
    try:
        ev_cache.write_text(json.dumps(out_nc), encoding="utf-8")
    except Exception:
        pass
    print(f"  [Track 2 GOES] {ev['id']} ({ev['date']}) done", flush=True)
    return out_nc


def _evaluate_multisensor_solver_event(ev: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
    """
    Evaluates end-to-end `solve_weather_and_corridor` across:
      - `truth` (Observed GOES-18 CCLC+BTD + ASOS ceilometer + 00Z RAOB balloon)
      - `ablation_single_layer` (Ablation: single-layer GOES ACTPC without CCLC/BTD/RAOB)
      - `nowcast_1h` (Causal `t_sunset - 1.0h` forecast)
      - `nowcast_3h` (Causal `t_sunset - 3.0h` forecast)
      - `hindcast_24h` (Archived Open-Meteo `previous-runs-api` Day-1 forecast)
      - `hindcast_72h` (Archived Open-Meteo `previous-runs-api` Day-3 forecast)
      - `nwp_only` (Raw pressure-level NWP reanalysis without satellite/ASOS/RAOB)
    Plus backward-compatible `1.0h, 3.0h, 24.0h` calibrated vs uncalibrated DNI/Raycast metrics.
    """
    BACKTEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ev_cache = BACKTEST_CACHE_DIR / f"rc_ev_v41_{ev['id']}.json"
    if (not refresh) and ev_cache.exists():
        try:
            cached = json.loads(ev_cache.read_text(encoding="utf-8"))
            print(f"  [Track 1/3/4 Solver] {ev['id']} ({ev['date']}) cached", flush=True)
            return cached
        except Exception:
            pass

    lat = float(ev["lat"])
    lon = float(ev["lon"])
    elev_m = float(ev["elev_m"])
    is_coastal = bool(ev["region"] == "coastal")
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
    dni_samples = dni_truth.get("samples") or []

    # Independent evening 00Z RAOB balloon & surface ASOS ceilometer truth
    raob_evening = fetch_raob_sounding(lat, lon, target_utc=t_sunset_utc, causal_only=False)
    asos_evening = fetch_nearest_station_observation(lat, lon, target_utc=t_sunset_utc)

    # 1. Multi-Sensor Observed Truth (`replay_mode="truth"`)
    truth_doc = json.loads(
        solve_weather_and_corridor(
            lat, lon, d_str, phase="sunset", sun_az_deg=sun_az, sun_el_deg=sun_el,
            target_elev_m=elev_m, model_engine="ensemble", replay_mode="truth",
        )
    )
    # 2. Ablation: Legacy single-layer satellite without CCLC/BTD/RAOB
    ablation_doc = json.loads(
        solve_weather_and_corridor(
            lat, lon, d_str, phase="sunset", sun_az_deg=sun_az, sun_el_deg=sun_el,
            target_elev_m=elev_m, model_engine="ensemble", replay_mode="truth",
            enable_multilayer_goes=False, enable_raob=False,
        )
    )

    mode_docs: dict[str, dict[str, Any]] = {
        "truth": truth_doc,
        "ablation_single_layer": ablation_doc,
    }
    rem_modes = ("nowcast_1h", "nowcast_3h", "hindcast_24h", "hindcast_72h", "nwp_only")
    with ThreadPoolExecutor(max_workers=2) as m_pool:
        m_futs = {
            m_key: m_pool.submit(
                solve_weather_and_corridor,
                lat, lon, d_str, phase="sunset", sun_az_deg=sun_az, sun_el_deg=sun_el,
                target_elev_m=elev_m, model_engine="ensemble", replay_mode=m_key,
            )
            for m_key in rem_modes
        }
        for m_key, m_fut in m_futs.items():
            mode_docs[m_key] = json.loads(m_fut.result())

    # Uncalibrated baseline runs for Track 3 interval calibration comparison (pre-HRRR-S3 baseline)
    uncal_docs: dict[str, dict[str, Any]] = {}
    for lead_hr, l_key in ((1.0, "1.0h"), (3.0, "3.0h"), (24.0, "24.0h")):
        t_issue = t_sunset_utc - dt.timedelta(hours=lead_hr)
        uncal_docs[l_key] = json.loads(
            solve_weather_and_corridor(
                lat, lon, d_str, phase="sunset", sun_az_deg=sun_az, sun_el_deg=sun_el,
                target_elev_m=elev_m, model_engine="ensemble", simulated_now_utc=t_issue,
                replay_mode="hindcast_24h" if lead_hr >= 20.0 else "auto",
                calibration_overrides=UNCALIBRATED_BASELINE_OVERRIDES,
                enable_hrrr_s3=False,
            )
        )

    sunset_truth_trans_pct = float(truth_doc["corridor_raycast"]["expected_transmittance_pct"])
    sunset_truth_alpenglow = int(truth_doc["alpenglow_score"])
    truth_layers = _layer_map(truth_doc)

    def _summarize_doc(doc: dict[str, Any]) -> dict[str, Any]:
        gh_exp, gh_p10, gh_p90 = _golden_pred_from_doc(doc, dni_samples)
        c_ray = doc["corridor_raycast"]
        inv_m = doc["inversion_model"]
        lmap = _layer_map(doc)
        return {
            "sunset_exp_pct": float(c_ray["expected_transmittance_pct"]),
            "sunset_p10_pct": float(c_ray["transmittance_p10_pct"]),
            "sunset_p90_pct": float(c_ray["transmittance_p90_pct"]),
            "golden_exp_pct": round(gh_exp, 1),
            "golden_p10_pct": round(gh_p10, 1),
            "golden_p90_pct": round(gh_p90, 1),
            "alpenglow_score": int(doc["alpenglow_score"]),
            "canvas_score": int(doc["canvas_score"]),
            "cloud_sea_score": int(inv_m["cloud_sea_score"]),
            "fog_base_ft": int(doc["fog_base_ft"]),
            "fog_top_ft": int(doc["fog_top_ft"]),
            "summit_clearance_ft": int(inv_m["summit_clearance_ft"]),
            "inversion_delta_t_f": float(inv_m["inversion_delta_t_f"]),
            "cloud_low_pct": int(doc["cloud_low_pct"]),
            "cloud_mid_pct": int(doc["cloud_mid_pct"]),
            "cloud_high_pct": int(doc["cloud_high_pct"]),
            "layers": {
                lid: {
                    "coverage_pct": int(lmap[lid]["coverage_pct"]),
                    "confidence_pct": int(lmap[lid]["confidence_pct"]),
                    "base_ft": int(lmap[lid]["base_ft"]),
                    "top_ft": int(lmap[lid]["top_ft"]),
                }
                for lid in ("fog", "low", "mid", "high")
                if lid in lmap
            },
        }

    modes_summary = {mk: _summarize_doc(mdoc) for mk, mdoc in mode_docs.items()}

    # Backward-compatible `leads` dict (`1.0h`, `3.0h`, `24.0h`)
    lead_map_modes = {"1.0h": "nowcast_1h", "3.0h": "nowcast_3h", "24.0h": "hindcast_24h"}
    lead_evals: dict[str, Any] = {}
    for l_key, m_name in lead_map_modes.items():
        lead_evals[l_key] = {
            "calibrated": modes_summary[m_name],
            "uncalibrated": _summarize_doc(uncal_docs[l_key]),
        }

    # Independent RAOB marine/boundary-layer inversion reference
    raob_ref: dict[str, Any] | None = None
    if raob_evening is not None and float(raob_evening.get("distance_km", 999.0)) <= 185.0:
        r_inv_base = raob_evening.get("inversion_base_asl_m")
        r_inv_top = raob_evening.get("inversion_top_asl_m")
        r_sat_top = raob_evening.get("saturated_stratus_top_asl_m")
        r_strength_k = float(raob_evening.get("inversion_strength_k") or 0.0)
        lid_m: float | None = None
        if r_sat_top is not None:
            lid_m = float(r_sat_top)
        elif r_inv_base is not None and r_strength_k >= 0.6:
            rt = float(r_inv_top) if r_inv_top is not None else (float(r_inv_base) + 150.0)
            lid_m = float(r_inv_base) + 0.18 * max(0.0, rt - float(r_inv_base))
        raob_ref = {
            "station_id": raob_evening["station_id"],
            "distance_km": round(float(raob_evening["distance_km"]), 1),
            "valid_utc": raob_evening["valid_utc"],
            "lid_asl_m": round(lid_m, 1) if lid_m is not None else None,
            "lid_asl_ft": int(round(lid_m * 3.28084)) if lid_m is not None else None,
            "inversion_base_asl_m": r_inv_base,
            "inversion_strength_k": round(r_strength_k, 2),
            "inversion_strength_f": round(r_strength_k * 1.8, 1),
            "summit_truly_above_lid": bool(elev_m > lid_m + 20.0) if (lid_m is not None and is_coastal) else None,
        }

    asos_ref: dict[str, Any] | None = None
    if asos_evening is not None and float(asos_evening.get("distance_km", 999.0)) <= 55.0:
        ceil_m = asos_evening.get("fog_base_asl_m") or asos_evening.get("lowest_ceiling_asl_m")
        asos_ref = {
            "station_id": asos_evening["station_id"],
            "distance_km": round(float(asos_evening["distance_km"]), 1),
            "low_cloud_pct": round(float(asos_evening.get("low_cloud_fraction", 0.0)) * 100.0, 1),
            "fog_cloud_pct": round(float(asos_evening.get("fog_cloud_fraction", 0.0)) * 100.0, 1),
            "ceilometer_base_asl_m": round(float(ceil_m), 1) if (ceil_m is not None and float(ceil_m) <= 1800.0) else None,
            "ceilometer_base_ft": int(round(float(ceil_m) * 3.28084)) if (ceil_m is not None and float(ceil_m) <= 1800.0) else None,
            "visibility_miles": asos_evening.get("visibility_miles"),
        }

    out_rc = {
        "id": ev["id"],
        "site": ev["site"],
        "region": ev["region"],
        "subregion": ev.get("subregion", ev["region"]),
        "season": ev.get("season", "spring"),
        "elev_m": elev_m,
        "date": d_str,
        "regime": ev["regime"],
        "dni_golden_truth_pct": dni_rel_pct,
        "dni_18h_wm2": dni_truth["dni_18h_wm2"],
        "dni_19h_wm2": dni_truth["dni_19h_wm2"],
        "dni_samples": dni_samples,
        "sunset_ray_truth_pct": sunset_truth_trans_pct,
        "sunset_alpenglow_truth": sunset_truth_alpenglow,
        "truth_layers": {
            lid: int(truth_layers[lid]["coverage_pct"])
            for lid in ("fog", "low", "mid", "high")
            if lid in truth_layers
        },
        "raob_ref": raob_ref,
        "asos_ref": asos_ref,
        "modes": modes_summary,
        "leads": lead_evals,
    }
    try:
        ev_cache.write_text(json.dumps(out_rc), encoding="utf-8")
    except Exception:
        pass
    print(f"  [Track 1/3/4 Solver] {ev['id']} ({ev['date']}) done", flush=True)
    return out_rc


def _compute_confidence_calibration(rc_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Track 4: Evaluates 3D cloud hover-card `confidence_pct` reliability across
    `nowcast_1h`, `nowcast_3h`, `hindcast_24h`, and `hindcast_72h` for all 4 cloud layers
    (`fog`, `low`, `mid`, `high`).

    Empirical layer skill score on `[0, 1]`:
      `skill = 1.0 - min(1.0, abs(pred_cov_pct - truth_cov_pct) / 55.0)`
    Plus binary hit rate (`|pred_cov_pct - truth_cov_pct| <= 20 pp`).
    """
    eval_modes = ("nowcast_1h", "nowcast_3h", "hindcast_24h", "hindcast_72h")
    bins_spec = [
        ("low_lt_55", 0, 55),
        ("moderate_55_69", 55, 70),
        ("good_70_81", 70, 82),
        ("high_ge_82", 82, 101),
    ]
    records: list[tuple[float, float, float, str]] = []
    mode_conf_vs_skill: dict[str, dict[str, float]] = {}

    for m_key in eval_modes:
        m_confs: list[float] = []
        m_skills: list[float] = []
        m_hits20: list[float] = []
        for r in rc_results:
            t_lmap = r["modes"]["truth"]["layers"]
            p_lmap = r["modes"][m_key]["layers"]
            for lid in ("fog", "low", "mid", "high"):
                if lid not in t_lmap or lid not in p_lmap:
                    continue
                conf = float(p_lmap[lid]["confidence_pct"]) / 100.0
                err = abs(float(p_lmap[lid]["coverage_pct"]) - float(t_lmap[lid]["coverage_pct"]))
                cont_skill = max(0.0, 1.0 - err / 55.0)
                hit20 = 1.0 if err <= 20.0 else 0.0
                emp_acc = 0.55 * cont_skill + 0.45 * hit20
                records.append((conf, emp_acc, hit20, m_key))
                m_confs.append(conf)
                m_skills.append(emp_acc)
                m_hits20.append(hit20)
        if m_confs:
            mode_conf_vs_skill[m_key] = {
                "mean_predicted_confidence_pct": round(float(np.mean(m_confs)) * 100.0, 1),
                "mean_empirical_accuracy_pct": round(float(np.mean(m_skills)) * 100.0, 1),
                "within_20pp_hit_rate_pct": round(float(np.mean(m_hits20)) * 100.0, 1),
                "calibration_gap_pct": round((float(np.mean(m_confs)) - float(np.mean(m_skills))) * 100.0, 1),
            }

    total_n = max(1, len(records))
    ece = 0.0
    bin_stats: dict[str, Any] = {}
    for b_name, lo_b, hi_b in bins_spec:
        sub = [rec for rec in records if lo_b <= rec[0] * 100.0 < hi_b]
        if not sub:
            continue
        mean_c = float(np.mean([x[0] for x in sub]))
        mean_a = float(np.mean([x[1] for x in sub]))
        mean_h = float(np.mean([x[2] for x in sub]))
        ece += (len(sub) / total_n) * abs(mean_c - mean_a)
        bin_stats[b_name] = {
            "n_layer_predictions": len(sub),
            "mean_predicted_confidence_pct": round(mean_c * 100.0, 1),
            "mean_empirical_accuracy_pct": round(mean_a * 100.0, 1),
            "within_20pp_hit_rate_pct": round(mean_h * 100.0, 1),
            "abs_gap_pct": round(abs(mean_c - mean_a) * 100.0, 2),
        }

    return {
        "n_total_layer_predictions": len(records),
        "expected_calibration_error_pct": round(ece * 100.0, 2),
        "by_lead_mode": mode_conf_vs_skill,
        "reliability_bins": bin_stats,
    }


def run_full_backtest_suite(
    nowcast_leads_hr: list[float] | None = None,
    raycast_leads_hr: list[float] | None = None,
    max_workers: int = 3,
    refresh: bool = False,
) -> dict[str, Any]:
    """
    Executes the complete 24-event multi-sensor historical benchmark suite (`v4.0`) across
    all 4 seasons, 4 California regions, and 4 verification tracks, persisting structured
    JSON to both `SUMMARY_JSON_PATH` and `BENCHMARK_REPORT_JSON_PATH`.
    """
    t_start = time.perf_counter()
    nowcast_leads = nowcast_leads_hr or [0.5, 1.0, 2.0, 3.0]
    raycast_leads = raycast_leads_hr or [1.0, 3.0, 24.0]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        nc_futures = [
            pool.submit(_evaluate_corridor_nowcast_event, ev, nowcast_leads, refresh)
            for ev in BENCHMARK_EVENTS
        ]
        nc_results = [f.result() for f in nc_futures]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rc_futures = [
            pool.submit(_evaluate_multisensor_solver_event, ev, refresh)
            for ev in BENCHMARK_EVENTS
        ]
        rc_results = [f.result() for f in rc_futures]

    # --- Track 1: Vertical Altitude & Inversion Accuracy vs 00Z RAOB Balloons & ASOS Ceilometers ---
    coastal_raob_events = [
        r for r in rc_results
        if r["region"] == "coastal"
        and r["raob_ref"] is not None
        and r["raob_ref"]["lid_asl_ft"] is not None
        and r["raob_ref"]["lid_asl_ft"] <= 5500
    ]
    summit_raob_events = [
        r for r in coastal_raob_events
        if r["subregion"] == "coastal_summit"
        and r["raob_ref"]["summit_truly_above_lid"] is not None
    ]
    ceilometer_events = [
        r for r in rc_results
        if r["asos_ref"] is not None
        and r["asos_ref"]["ceilometer_base_ft"] is not None
        and r["asos_ref"]["low_cloud_pct"] >= 25.0
    ]

    vertical_by_mode: dict[str, Any] = {}
    for m_key in ("nwp_only", "hindcast_72h", "hindcast_24h", "nowcast_3h", "nowcast_1h", "ablation_single_layer", "truth"):
        lid_errs_ft: list[float] = []
        inv_str_errs_f: list[float] = []
        summit_hits: list[float] = []
        ceil_base_errs_ft: list[float] = []

        for r in coastal_raob_events:
            ref_ft = float(r["raob_ref"]["lid_asl_ft"])
            pred_ft = float(r["modes"][m_key]["fog_top_ft"])
            lid_errs_ft.append(pred_ft - ref_ft)
            ref_df = float(r["raob_ref"]["inversion_strength_f"])
            pred_df = float(r["modes"][m_key]["inversion_delta_t_f"])
            inv_str_errs_f.append(pred_df - ref_df)

        for r in summit_raob_events:
            truth_above = bool(r["raob_ref"]["summit_truly_above_lid"])
            pred_above = int(r["modes"][m_key]["summit_clearance_ft"]) > 65
            summit_hits.append(1.0 if pred_above == truth_above else 0.0)

        for r in ceilometer_events:
            ref_base_ft = float(r["asos_ref"]["ceilometer_base_ft"])
            pred_base_ft = float(r["modes"][m_key]["fog_base_ft"])
            ceil_base_errs_ft.append(pred_base_ft - ref_base_ft)

        vertical_by_mode[m_key] = {
            "n_coastal_raob_events": len(lid_errs_ft),
            "fog_lid_mae_ft": round(float(np.mean(np.abs(lid_errs_ft))), 0) if lid_errs_ft else None,
            "fog_lid_mae_m": round(float(np.mean(np.abs(lid_errs_ft))) / 3.28084, 0) if lid_errs_ft else None,
            "fog_lid_bias_ft": round(float(np.mean(lid_errs_ft)), 0) if lid_errs_ft else None,
            "inversion_strength_mae_f": round(float(np.mean(np.abs(inv_str_errs_f))), 2) if inv_str_errs_f else None,
            "summit_above_fog_accuracy_pct": round(float(np.mean(summit_hits)) * 100.0, 1) if summit_hits else None,
            "asos_ceilometer_base_mae_ft": round(float(np.mean(np.abs(ceil_base_errs_ft))), 0) if ceil_base_errs_ft else None,
        }

    # --- Track 2A: Multi-Layer Under-Cirrus & Twilight Fog Capture (`CCLC + BTD + ASOS` vs Single-Layer `ACTPC`) ---
    overlap_sat_events = [
        r["multilayer_vs_single_sat"]
        for r in nc_results
        if r.get("multilayer_vs_single_sat") is not None
        and (
            r["multilayer_vs_single_sat"]["upper_high_cf_pct"] >= 20.0
            or r["multilayer_vs_single_sat"]["upper_mid_cf_pct"] >= 25.0
            or r["multilayer_vs_single_sat"]["bottom_up_ref_low_pct"] >= 25.0
        )
    ]
    sl_sat_errs = [abs(x["single_layer_sat_low_pct"] - x["bottom_up_ref_low_pct"]) for x in overlap_sat_events]
    ml_sat_errs = [abs(x["multilayer_cclc_btd_low_pct"] - x["bottom_up_ref_low_pct"]) for x in overlap_sat_events]

    asos_eval_events = [r for r in rc_results if r["asos_ref"] is not None]
    ablation_low_errs = [
        abs(float(r["modes"]["ablation_single_layer"]["cloud_low_pct"]) - float(r["modes"]["truth"]["cloud_low_pct"]))
        for r in asos_eval_events
    ]
    multilayer_verification = {
        "n_overlap_or_stratus_events": len(overlap_sat_events),
        "single_layer_actpc_low_mae_vs_asos_pct": round(float(np.mean(sl_sat_errs)), 2) if sl_sat_errs else 0.0,
        "multilayer_cclc_btd_low_mae_vs_asos_pct": round(float(np.mean(ml_sat_errs)), 2) if ml_sat_errs else 0.0,
        "satellite_low_cloud_error_reduction_pct": round(
            (1.0 - float(np.mean(ml_sat_errs)) / max(1e-6, float(np.mean(sl_sat_errs)))) * 100.0, 1
        ) if sl_sat_errs else 0.0,
        "mean_solver_low_cloud_shift_with_multilayer_raob_pp": round(float(np.mean(ablation_low_errs)), 2) if ablation_low_errs else 0.0,
        "events_sample": overlap_sat_events,
    }

    # --- Track 2B: 3D Cloud-Top Parallax & 6-Station Corridor Nowcast Skill ---
    valid_nc = [r for r in nc_results if r.get("goes_available")]
    cirrus_events = [
        r for r in valid_nc
        if r["event"]["regime"] in (
            "coastal_cirrus_burner", "sierra_cirrus_burner", "sierra_cirrus_wave", "frontal_overcast", "storm_overcast"
        )
    ]
    parallax_summary = {
        "mean_cirrus_top_asl_m": round(float(np.mean([r["mean_high_top_asl_m"] for r in cirrus_events])), 0) if cirrus_events else 0.0,
        "mean_ground_shift_km": round(float(np.mean([r["parallax_shift_km"]["total_km"] for r in cirrus_events])), 2) if cirrus_events else 0.0,
        "mean_east_shift_km": round(float(np.mean([r["parallax_shift_km"]["east_km"] for r in cirrus_events])), 2) if cirrus_events else 0.0,
        "mean_north_shift_km": round(float(np.mean([r["parallax_shift_km"]["north_km"] for r in cirrus_events])), 2) if cirrus_events else 0.0,
        "max_corridor_high_cf_edge_error_pct": round(float(max((r["max_parallax_high_cf_diff_pct"] for r in valid_nc), default=0.0)), 1),
    }

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

        for r in valid_nc:
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
            bs_val = float(np.mean(arr_b)) if arr_b.size else 0.25
            m_stats[m] = {
                "n_corridor_layer_samples": int(arr_e.size),
                "mae_pct": round(float(np.mean(np.abs(arr_e))), 2) if arr_e.size else 0.0,
                "rmse_pct": round(float(np.sqrt(np.mean(arr_e ** 2))), 2) if arr_e.size else 0.0,
                "bias_pct": round(float(np.mean(arr_e)), 2) if arr_e.size else 0.0,
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

    # --- Track 3: True Lead-Time Hindcast Skill Curve (`+1h`, `+3h`, `+24h`, `+72h`, `nwp_only`) ---
    lead_time_hindcast_skill: dict[str, Any] = {}
    for m_key in ("nowcast_1h", "nowcast_3h", "hindcast_24h", "hindcast_72h", "nwp_only"):
        low_errs = [abs(r["modes"][m_key]["cloud_low_pct"] - r["modes"]["truth"]["cloud_low_pct"]) for r in rc_results]
        mid_errs = [abs(r["modes"][m_key]["cloud_mid_pct"] - r["modes"]["truth"]["cloud_mid_pct"]) for r in rc_results]
        high_errs = [abs(r["modes"][m_key]["cloud_high_pct"] - r["modes"]["truth"]["cloud_high_pct"]) for r in rc_results]
        ray_errs = [abs(r["modes"][m_key]["sunset_exp_pct"] - r["sunset_ray_truth_pct"]) for r in rc_results]
        alp_errs = [abs(r["modes"][m_key]["alpenglow_score"] - r["sunset_alpenglow_truth"]) for r in rc_results]
        dni_errs = [abs(r["modes"][m_key]["golden_exp_pct"] - r["dni_golden_truth_pct"]) for r in rc_results]
        dni_preds = [r["modes"][m_key]["golden_exp_pct"] for r in rc_results]
        dni_truths = [r["dni_golden_truth_pct"] for r in rc_results]
        dni_cov = [
            1.0 if (r["modes"][m_key]["golden_p10_pct"] - 1.5 <= r["dni_golden_truth_pct"] <= r["modes"][m_key]["golden_p90_pct"] + 1.5) else 0.0
            for r in rc_results
        ]
        ss_cov = [
            1.0 if (r["modes"][m_key]["sunset_p10_pct"] - 1.5 <= r["sunset_ray_truth_pct"] <= r["modes"][m_key]["sunset_p90_pct"] + 1.5) else 0.0
            for r in rc_results
        ]
        r_dni = float(np.corrcoef(dni_preds, dni_truths)[0, 1]) if len(dni_preds) > 2 else 0.0
        lead_time_hindcast_skill[m_key] = {
            "cloud_low_mae_pp": round(float(np.mean(low_errs)), 2),
            "cloud_mid_mae_pp": round(float(np.mean(mid_errs)), 2),
            "cloud_high_mae_pp": round(float(np.mean(high_errs)), 2),
            "all_layer_mean_mae_pp": round(float(np.mean(low_errs + mid_errs + high_errs)), 2),
            "sunset_ray_transmittance_mae_pp": round(float(np.mean(ray_errs)), 2),
            "alpenglow_score_mae_pts": round(float(np.mean(alp_errs)), 2),
            "golden_hour_dni_mae_pp": round(float(np.mean(dni_errs)), 2),
            "golden_hour_dni_pearson_r": round(r_dni, 3),
            "golden_hour_p10_p90_coverage_pct": round(float(np.mean(dni_cov)) * 100.0, 1),
            "sunset_p10_p90_coverage_pct": round(float(np.mean(ss_cov)) * 100.0, 1),
        }

    # Backward-compatible `raycast_calibration_by_lead` (`1.0h`, `3.0h`, `24.0h`)
    raycast_skill_by_lead: dict[str, Any] = {}
    for lead_hr in raycast_leads:
        lkey = f"{lead_hr:.1f}h"
        if lkey not in rc_results[0]["leads"]:
            continue
        for mode in ("calibrated", "uncalibrated"):
            gh_errs: list[float] = []
            gh_cov: list[float] = []
            gh_widths: list[float] = []
            gh_preds: list[float] = []
            gh_truths: list[float] = []
            ss_errs: list[float] = []
            ss_cov_l: list[float] = []
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
                ss_cov_l.append(
                    1.0 if (m_rec["sunset_p10_pct"] - 1.5 <= t_ss <= m_rec["sunset_p90_pct"] + 1.5) else 0.0
                )
                ss_widths.append(m_rec["sunset_p90_pct"] - m_rec["sunset_p90_pct"])

            corr_gh = float(np.corrcoef(gh_preds, gh_truths)[0, 1]) if len(gh_preds) > 2 else 0.0
            raycast_skill_by_lead.setdefault(lkey, {})[mode] = {
                "golden_hour_dni_mae_pct": round(float(np.mean(np.abs(gh_errs))), 2),
                "golden_hour_dni_rmse_pct": round(float(np.sqrt(np.mean(np.array(gh_errs) ** 2))), 2),
                "golden_hour_dni_pearson_r": round(corr_gh, 3),
                "golden_hour_p10_p90_coverage_pct": round(float(np.mean(gh_cov)) * 100.0, 1),
                "golden_hour_mean_interval_width_pct": round(float(np.mean(gh_widths)), 1),
                "sunset_ray_mae_pct": round(float(np.mean(np.abs(ss_errs))), 2),
                "sunset_ray_rmse_pct": round(float(np.sqrt(np.mean(np.array(ss_errs) ** 2))), 2),
                "sunset_p10_p90_coverage_pct": round(float(np.mean(ss_cov_l)) * 100.0, 1),
                "sunset_mean_interval_width_pct": round(float(np.mean(ss_widths)), 1),
            }

    # --- Seasonal & Regional Breakdown (`+1h` vs `+24h` vs `+72h`) ---
    seasonal_breakdown: dict[str, Any] = {}
    for season_name in ("winter", "spring", "summer", "autumn"):
        sub_r = [r for r in rc_results if r["season"] == season_name]
        if not sub_r:
            continue
        seasonal_breakdown[season_name] = {
            "n_events": len(sub_r),
            "nowcast_1h_ray_mae_pp": round(float(np.mean([abs(r["modes"]["nowcast_1h"]["sunset_exp_pct"] - r["sunset_ray_truth_pct"]) for r in sub_r])), 2),
            "hindcast_24h_ray_mae_pp": round(float(np.mean([abs(r["modes"]["hindcast_24h"]["sunset_exp_pct"] - r["sunset_ray_truth_pct"]) for r in sub_r])), 2),
            "hindcast_72h_ray_mae_pp": round(float(np.mean([abs(r["modes"]["hindcast_72h"]["sunset_exp_pct"] - r["sunset_ray_truth_pct"]) for r in sub_r])), 2),
            "nowcast_1h_dni_mae_pp": round(float(np.mean([abs(r["modes"]["nowcast_1h"]["golden_exp_pct"] - r["dni_golden_truth_pct"]) for r in sub_r])), 2),
            "hindcast_24h_dni_mae_pp": round(float(np.mean([abs(r["modes"]["hindcast_24h"]["golden_exp_pct"] - r["dni_golden_truth_pct"]) for r in sub_r])), 2),
        }

    regional_breakdown: dict[str, Any] = {}
    for subreg in ("coastal_headland", "coastal_summit", "central_coast", "high_sierra"):
        sub_r = [r for r in rc_results if r["subregion"] == subreg]
        if not sub_r:
            continue
        regional_breakdown[subreg] = {
            "n_events": len(sub_r),
            "nowcast_1h_low_cloud_mae_pp": round(float(np.mean([abs(r["modes"]["nowcast_1h"]["cloud_low_pct"] - r["modes"]["truth"]["cloud_low_pct"]) for r in sub_r])), 2),
            "hindcast_24h_low_cloud_mae_pp": round(float(np.mean([abs(r["modes"]["hindcast_24h"]["cloud_low_pct"] - r["modes"]["truth"]["cloud_low_pct"]) for r in sub_r])), 2),
            "hindcast_72h_low_cloud_mae_pp": round(float(np.mean([abs(r["modes"]["hindcast_72h"]["cloud_low_pct"] - r["modes"]["truth"]["cloud_low_pct"]) for r in sub_r])), 2),
            "nowcast_1h_alpenglow_mae_pts": round(float(np.mean([abs(r["modes"]["nowcast_1h"]["alpenglow_score"] - r["sunset_alpenglow_truth"]) for r in sub_r])), 2),
            "hindcast_24h_alpenglow_mae_pts": round(float(np.mean([abs(r["modes"]["hindcast_24h"]["alpenglow_score"] - r["sunset_alpenglow_truth"]) for r in sub_r])), 2),
        }

    # --- Track 4: 3D Cloud Hover-Card Confidence Calibration ---
    confidence_calibration = _compute_confidence_calibration(rc_results)

    elapsed_s = round(time.perf_counter() - t_start, 2)
    summary = {
        "schema_version": "v4.1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_s,
        "n_events": len(BENCHMARK_EVENTS),
        "calibration_config": CALIBRATION_CONFIG,
        "vertical_inversion_verification": vertical_by_mode,
        "multilayer_under_cirrus_verification": multilayer_verification,
        "parallax_verification": parallax_summary,
        "corridor_nowcast_skill_by_lead": nowcast_skill_by_lead,
        "coastal_orographic_pinning_by_lead": coastal_pinning_by_lead,
        "lead_time_hindcast_skill": lead_time_hindcast_skill,
        "raycast_calibration_by_lead": raycast_skill_by_lead,
        "confidence_calibration_verification": confidence_calibration,
        "seasonal_breakdown": seasonal_breakdown,
        "regional_breakdown": regional_breakdown,
        "events_detail": rc_results,
    }

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    BENCHMARK_REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_REPORT_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print("Running PhotoPlanner v4.1 Multi-Sensor + Native HRRR S3 Benchmark (N=24 events across 4 seasons)...", flush=True)
    res = run_full_backtest_suite(max_workers=2)
    print(f"\nCompleted in {res['elapsed_seconds']}s — saved to {SUMMARY_JSON_PATH} and {BENCHMARK_REPORT_JSON_PATH}", flush=True)
    print("\n=== TRACK 1: Vertical Altitude & Inversion Accuracy vs 00Z RAOB Balloons & ASOS ===")
    for mk, mv in res["vertical_inversion_verification"].items():
        print(f"  {mk:22s} | Lid MAE: {str(mv['fog_lid_mae_ft']):>6s} ft ({str(mv['fog_lid_mae_m']):>5s} m) | Bias: {str(mv['fog_lid_bias_ft']):>6s} ft | Summit Above/In Fog Acc: {mv['summit_above_fog_accuracy_pct']}% | ASOS Base MAE: {mv['asos_ceilometer_base_mae_ft']} ft")
    print("\n=== TRACK 2: Multi-Layer CCLC+BTD vs Single-Layer ACTPC ===")
    mlv = res["multilayer_under_cirrus_verification"]
    print(f"  Overlap/Stratus Events (N={mlv['n_overlap_or_stratus_events']}): Single-Layer MAE={mlv['single_layer_actpc_low_mae_vs_asos_pct']}% -> Multi-Layer CCLC+BTD MAE={mlv['multilayer_cclc_btd_low_mae_vs_asos_pct']}% ({mlv['satellite_low_cloud_error_reduction_pct']}% reduction)")
    print("\n=== TRACK 3: Lead-Time Forecast Skill Curve (+1h -> +72h Hindcasts) ===")
    for mk, mv in res["lead_time_hindcast_skill"].items():
        print(f"  {mk:14s} | Low MAE: {mv['cloud_low_mae_pp']:5.2f} pp | High MAE: {mv['cloud_high_mae_pp']:5.2f} pp | Ray MAE: {mv['sunset_ray_transmittance_mae_pp']:5.2f} pp | DNI MAE: {mv['golden_hour_dni_mae_pp']:5.2f} pp (r={mv['golden_hour_dni_pearson_r']:.3f}) | Alpenglow MAE: {mv['alpenglow_score_mae_pts']:5.2f}")
    print("\n=== TRACK 4: 3D Hover-Card Confidence Calibration ===")
    cc = res["confidence_calibration_verification"]
    print(f"  ECE = {cc['expected_calibration_error_pct']}% across {cc['n_total_layer_predictions']} layer predictions")
    for b_k, b_v in cc["reliability_bins"].items():
        print(f"    {b_k:16s} (n={b_v['n_layer_predictions']:3d}): Pred Conf={b_v['mean_predicted_confidence_pct']:5.1f}% vs Empirical={b_v['mean_empirical_accuracy_pct']:5.1f}% (<=20pp hit={b_v['within_20pp_hit_rate_pct']:5.1f}%)")
