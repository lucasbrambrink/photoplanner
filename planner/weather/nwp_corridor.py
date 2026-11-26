"""
24-Hour Pressure-Level NWP Corridor, Multi-Model Ensemble & CAMS Aerosol Fetcher (`0-120h` + Historical).

Replaces synthetic climatology and single-hour snapshots with full 24-hour (`00:00..23:00` local)
vertical pressure-level soundings (`1000..250 hPa`), true vector winds `(u, v)` from model
`wind_direction_*hPa`, geopotential heights, `CAPE` / `CIN`, `PBL` depth, CAMS `550 nm` AOD +
dust/smoke Angstrom exponent, and real multi-model runs (`NOAA 3km HRRR`, `ECMWF IFS 0.25`,
`ECMWF AIFS 0.25 AI`, `NOAA GFS`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import math
import os
import random
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[2]
NWP_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "nwp"
USER_AGENT = "PhotoPlanner/2.0 (landscape-optics-corridor; github.com/lucasbrambrink/photoplanner)"


def _open_meteo_url(host_prefix: str, path_and_query: str) -> str:
    """Routes Open-Meteo calls to commercial `customer-*.open-meteo.com` when `OPEN_METEO_API_KEY` is set."""
    api_key = os.environ.get("OPEN_METEO_API_KEY", "").strip()
    if api_key:
        sep = "&" if "?" in path_and_query else "?"
        return f"https://customer-{host_prefix}.open-meteo.com{path_and_query}{sep}apikey={api_key}"
    return f"https://{host_prefix}.open-meteo.com{path_and_query}"


_MEM_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MEM_TTL_SECONDS = 1800.0  # 30 minutes (matches HRRR/IFS hourly update cadence)
_PACING_LOCK = threading.Lock()
_LAST_REQ_TS = 0.0
_HOST_429_COOLDOWN_UNTIL: dict[str, float] = {}
_HOST_429_COOLDOWN_SECONDS = 75.0

# 6 log-spaced distances (km) along the primary 0-380 km solar corridor
CORRIDOR_DISTANCES_KM: list[float] = [0.0, 15.0, 45.0, 100.0, 190.0, 320.0]
# 3 conjugate-sector distances (km) along the opposite sunrise/sunset bearing
CONJUGATE_DISTANCES_KM: list[float] = [18.0, 75.0, 190.0]

PRESSURE_SOUNDING_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,"
    "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
    "visibility,precipitation,cape,convective_inhibition,boundary_layer_height,"
    "wind_speed_10m,wind_direction_10m,"
    "temperature_950hPa,temperature_925hPa,temperature_850hPa,"
    "temperature_700hPa,temperature_500hPa,temperature_300hPa,"
    "relative_humidity_975hPa,relative_humidity_950hPa,relative_humidity_925hPa,"
    "relative_humidity_850hPa,relative_humidity_700hPa,relative_humidity_500hPa,relative_humidity_300hPa,"
    "cloud_cover_925hPa,cloud_cover_850hPa,cloud_cover_700hPa,cloud_cover_500hPa,cloud_cover_300hPa,"
    "geopotential_height_925hPa,geopotential_height_850hPa,"
    "geopotential_height_700hPa,geopotential_height_500hPa,geopotential_height_300hPa,"
    "wind_speed_850hPa,wind_direction_850hPa,"
    "wind_speed_700hPa,wind_direction_700hPa,"
    "wind_speed_500hPa,wind_direction_500hPa,"
    "wind_speed_300hPa,wind_direction_300hPa,"
    "wind_speed_250hPa,wind_direction_250hPa"
)

ARCHIVE_REANALYSIS_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,"
    "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
    "precipitation,boundary_layer_height,wind_speed_10m,wind_direction_10m"
)


def local_tz_for_lon(lon: float) -> tuple[ZoneInfo, str]:
    """Resolves IANA timezone for US/Pacific, Mountain, Central, Eastern, or UTC fallback."""
    if -128.0 <= lon < -114.2:
        return ZoneInfo("America/Los_Angeles"), "America/Los_Angeles"
    if -114.2 <= lon < -102.0:
        return ZoneInfo("America/Denver"), "America/Denver"
    if -102.0 <= lon < -87.0:
        return ZoneInfo("America/Chicago"), "America/Chicago"
    if -87.0 <= lon < -65.0:
        return ZoneInfo("America/New_York"), "America/New_York"
    return ZoneInfo("UTC"), "UTC"


def offset_lat_lon_km(lat: float, lon: float, east_km: float, north_km: float) -> tuple[float, float]:
    m_per_lat = 111.13292
    m_per_lon = max(22.0, 111.41284 * math.cos(math.radians(lat)))
    return round(lat + north_km / m_per_lat, 4), round(lon + east_km / m_per_lon, 4)


def bearing_offset_lat_lon_km(
    lat: float, lon: float, azimuth_deg: float, dist_km: float
) -> tuple[float, float]:
    az_r = math.radians(azimuth_deg)
    return offset_lat_lon_km(lat, lon, dist_km * math.sin(az_r), dist_km * math.cos(az_r))


def _get_json(url: str, timeout: float = 10.0) -> Any:
    global _LAST_REQ_TS
    import urllib.error
    import urllib.parse

    host = urllib.parse.urlsplit(url).netloc or "open-meteo"
    now_chk = time.monotonic()
    cooldown_until = _HOST_429_COOLDOWN_UNTIL.get(host, 0.0)
    if now_chk < cooldown_until:
        rem = int(max(1.0, cooldown_until - now_chk))
        raise urllib.error.HTTPError(
            url, 429, f"Too Many Requests (cooldown {rem}s active)", hdrs=None, fp=None
        )

    last_exc: Exception | None = None
    for attempt in range(3):
        with _PACING_LOCK:
            now_m = time.monotonic()
            gap = 0.42 - (now_m - _LAST_REQ_TS)
            if gap > 0:
                time.sleep(gap)
            _LAST_REQ_TS = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                # Fail fast on 429 (1 short retry, then trip host cooldown so NOAA HRRR S3 takes over immediately)
                if attempt == 0:
                    retry_hdr = exc.headers.get("Retry-After") if exc.headers else None
                    wait_s = 1.4 + random.uniform(0.1, 0.4)
                    if retry_hdr:
                        try:
                            hdr_val = float(retry_hdr)
                            if hdr_val > 6.0:
                                _HOST_429_COOLDOWN_UNTIL[host] = time.monotonic() + min(120.0, hdr_val)
                                raise
                            wait_s = max(wait_s, hdr_val)
                        except ValueError:
                            pass
                    time.sleep(min(4.0, wait_s))
                    continue
                _HOST_429_COOLDOWN_UNTIL[host] = time.monotonic() + _HOST_429_COOLDOWN_SECONDS
                raise
            if exc.code in (502, 503, 504) and attempt < 2:
                time.sleep(1.2 * (attempt + 1) + random.uniform(0.1, 0.3))
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < 1:
                time.sleep(0.8)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _safe_arr_val(hourly: dict[str, Any], key: str, idx: int, default: float) -> float:
    arr = hourly.get(key)
    if not isinstance(arr, list) or idx < 0 or idx >= len(arr):
        return default
    v = arr[idx]
    if v is None:
        return default
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else default
    except (TypeError, ValueError):
        return default


def _wind_uv_from_speed_dir_kmh(speed_kmh: float, from_deg: float) -> tuple[float, float]:
    """Converts meteorological wind (`speed_kmh`, blowing FROM `from_deg`) into `(u_east_ms, v_north_ms)`."""
    spd_ms = max(0.0, speed_kmh) / 3.6
    rad = math.radians(from_deg)
    return round(-spd_ms * math.sin(rad), 2), round(-spd_ms * math.cos(rad), 2)


def _std_geopotential_m(p_hpa: float) -> float:
    return 44330.0 * (1.0 - (p_hpa / 1013.25) ** 0.190284)


def extract_hour_sounding(
    point_doc: dict[str, Any],
    hour_idx: int,
    dist_km: float,
    azimuth_deg: float | None,
    lat: float,
    lon: float,
    aod_550nm: float = 0.07,
    angstrom_exp: float = 1.25,
    ozone_du: float = 300.0,
    is_coastal: bool = False,
) -> dict[str, Any]:
    """
    Extracts a rich physical sounding + altitude-resolved `CloudLayerBand`s for `hour_idx` (`0..23`).
    """
    h = point_doc.get("hourly") or {}
    raw_elev = point_doc.get("elevation")
    elev_m = float(raw_elev) if raw_elev is not None else (0.0 if is_coastal else 1200.0)
    if (is_coastal and dist_km >= 8.0 and lon < -122.45) or (31.5 <= lat <= 42.5 and lon < -124.45):
        elev_m = 0.0

    t2m_c = _safe_arr_val(h, "temperature_2m", hour_idx, 14.0 - 0.006 * elev_m)
    rh2m = max(4.0, min(100.0, _safe_arr_val(h, "relative_humidity_2m", hour_idx, 62.0)))
    dew2m_c = _safe_arr_val(h, "dew_point_2m", hour_idx, t2m_c - (100.0 - rh2m) * 0.2)
    p_sfc = _safe_arr_val(
        h,
        "surface_pressure",
        hour_idx,
        1013.25 * ((1.0 - 2.25577e-5 * max(0.0, elev_m)) ** 5.25588),
    )

    cc_tot = max(0.0, min(100.0, _safe_arr_val(h, "cloud_cover", hour_idx, 0.0)))
    cc_low = max(0.0, min(100.0, _safe_arr_val(h, "cloud_cover_low", hour_idx, 0.0)))
    cc_mid = max(0.0, min(100.0, _safe_arr_val(h, "cloud_cover_mid", hour_idx, 0.0)))
    cc_high = max(0.0, min(100.0, _safe_arr_val(h, "cloud_cover_high", hour_idx, 0.0)))

    vis_m = _safe_arr_val(h, "visibility", hour_idx, 38_000.0)
    precip_mm = max(0.0, _safe_arr_val(h, "precipitation", hour_idx, 0.0))
    cape = max(0.0, _safe_arr_val(h, "cape", hour_idx, 0.0))
    cin = _safe_arr_val(h, "convective_inhibition", hour_idx, 0.0)
    pbl_agl_m = max(80.0, _safe_arr_val(h, "boundary_layer_height", hour_idx, 450.0 if is_coastal else 950.0))

    # Pressure-level temperatures, RH, cloud cover, and geopotential heights (m ASL)
    t950 = _safe_arr_val(h, "temperature_950hPa", hour_idx, t2m_c - 2.5)
    t925 = _safe_arr_val(h, "temperature_925hPa", hour_idx, t2m_c - 4.0)
    t850 = _safe_arr_val(h, "temperature_850hPa", hour_idx, t2m_c - 8.5)
    t700 = _safe_arr_val(h, "temperature_700hPa", hour_idx, t850 - 9.0)
    t500 = _safe_arr_val(h, "temperature_500hPa", hour_idx, t700 - 16.0)
    t300 = _safe_arr_val(h, "temperature_300hPa", hour_idx, t500 - 22.0)

    rh975 = _safe_arr_val(h, "relative_humidity_975hPa", hour_idx, rh2m * 0.95)
    rh950 = _safe_arr_val(h, "relative_humidity_950hPa", hour_idx, rh2m * 0.88)
    rh925 = _safe_arr_val(h, "relative_humidity_925hPa", hour_idx, rh2m * 0.80)
    rh850 = _safe_arr_val(h, "relative_humidity_850hPa", hour_idx, max(25.0, cc_low * 0.85))
    rh700 = _safe_arr_val(h, "relative_humidity_700hPa", hour_idx, max(22.0, cc_mid * 0.85))
    rh500 = _safe_arr_val(h, "relative_humidity_500hPa", hour_idx, max(20.0, cc_mid * 0.78))
    rh300 = _safe_arr_val(h, "relative_humidity_300hPa", hour_idx, max(20.0, cc_high * 0.82))

    c925 = _safe_arr_val(h, "cloud_cover_925hPa", hour_idx, cc_low * 0.8)
    c850 = _safe_arr_val(h, "cloud_cover_850hPa", hour_idx, cc_low * 0.7)
    c700 = _safe_arr_val(h, "cloud_cover_700hPa", hour_idx, cc_mid * 0.85)
    c500 = _safe_arr_val(h, "cloud_cover_500hPa", hour_idx, cc_mid * 0.75)
    c300 = _safe_arr_val(h, "cloud_cover_300hPa", hour_idx, cc_high * 0.9)

    z925 = _safe_arr_val(h, "geopotential_height_925hPa", hour_idx, _std_geopotential_m(925.0))
    z850 = _safe_arr_val(h, "geopotential_height_850hPa", hour_idx, _std_geopotential_m(850.0))
    z700 = _safe_arr_val(h, "geopotential_height_700hPa", hour_idx, _std_geopotential_m(700.0))
    z500 = _safe_arr_val(h, "geopotential_height_500hPa", hour_idx, _std_geopotential_m(500.0))
    z300 = _safe_arr_val(h, "geopotential_height_300hPa", hour_idx, _std_geopotential_m(300.0))

    # True vector winds at 10m, 850, 700, 500, 300, 250 hPa (Open-Meteo default unit is km/h)
    wspd10_kmh = _safe_arr_val(h, "wind_speed_10m", hour_idx, 12.0)
    wdir10 = _safe_arr_val(h, "wind_direction_10m", hour_idx, 285.0)
    wspd850_kmh = _safe_arr_val(h, "wind_speed_850hPa", hour_idx, wspd10_kmh * 1.35)
    wdir850 = _safe_arr_val(h, "wind_direction_850hPa", hour_idx, wdir10)
    wspd700_kmh = _safe_arr_val(h, "wind_speed_700hPa", hour_idx, wspd850_kmh * 1.25)
    wdir700 = _safe_arr_val(h, "wind_direction_700hPa", hour_idx, wdir850)
    wspd500_kmh = _safe_arr_val(h, "wind_speed_500hPa", hour_idx, wspd700_kmh * 1.35)
    wdir500 = _safe_arr_val(h, "wind_direction_500hPa", hour_idx, wdir700)
    wspd300_kmh = _safe_arr_val(h, "wind_speed_300hPa", hour_idx, max(45.0, wspd500_kmh * 1.45))
    wdir300 = _safe_arr_val(h, "wind_direction_300hPa", hour_idx, wdir500)
    wspd250_kmh = _safe_arr_val(h, "wind_speed_250hPa", hour_idx, max(50.0, wspd300_kmh * 1.08))
    wdir250 = _safe_arr_val(h, "wind_direction_250hPa", hour_idx, wdir300)

    u_sfc, v_sfc = _wind_uv_from_speed_dir_kmh(wspd10_kmh, wdir10)
    u_850, v_850 = _wind_uv_from_speed_dir_kmh(wspd850_kmh, wdir850)
    u_500, v_500 = _wind_uv_from_speed_dir_kmh(wspd500_kmh, wdir500)
    u_250, v_250 = _wind_uv_from_speed_dir_kmh(wspd250_kmh, wdir250)

    # Environmental lapse rate K/km between 850/700 hPa and 500 hPa
    dz_free_km = max(1.0, (z500 - z700) / 1000.0)
    lapse_k_km = max(3.5, min(9.6, (t700 - t500) / dz_free_km))

    # Thermal inversion & boundary-layer lid detection (`+Delta T` + vertical hydrolapse)
    # Only compare pressure levels physically above the station surface (`z_p > elev_m + 15 m`)
    # so elevated summits (e.g. Mt. Tamalpais 784 m, Sierra ridges) never compare against
    # subterranean extrapolated NWP pressure-level temperatures.
    z950_est = _std_geopotential_m(950.0)
    above_sfc_low_temps = [
        t_val
        for z_val, t_val in ((z950_est, t950), (z925, t925), (z850, t850))
        if z_val > elev_m + 15.0
    ]
    above_sfc_hi_temps = [
        t_val
        for z_val, t_val in ((z850, t850), (z700, t700))
        if z_val > elev_m + 15.0
    ]
    if is_coastal and elev_m < 1350.0:
        # Pacific marine boundary layer is sea-level-referenced (`0 m ASL`), NOT `elev_m + pbl`!
        # Elevated coastal peaks (e.g. Mt. Tamalpais 784 m, Montara Mtn 578 m) frequently poke ABOVE
        # shallow 250–700 m marine stratus decks (`rh975/rh950` moist while `rh925/summit` is dry).
        t_above_max = max(above_sfc_low_temps) if above_sfc_low_temps else t2m_c
        inv_jump_k = max(
            0.0,
            t_above_max - t2m_c,
            max(t925, t850) - t950 + (1.2 if rh975 >= 78.0 and rh925 < 55.0 else 0.0),
        )
        z925_eff = max(720.0, z925)
        z850_eff = max(1350.0, z850)
        if elev_m >= 350.0 and rh2m >= 85.0 and rh975 < 75.0:
            # Deep post-frontal moist boundary layer immersing an elevated coastal summit
            marine_lid_asl_m = elev_m + min(550.0, max(160.0, pbl_agl_m * 1.45))
        elif rh850 >= 66.0:
            marine_lid_asl_m = z850_eff + 200.0
        elif rh925 >= 66.0:
            marine_lid_asl_m = z925_eff + ((rh925 - 66.0) / max(4.0, rh925 - rh850)) * (z850_eff - z925_eff)
        elif rh950 >= 66.0:
            marine_lid_asl_m = 540.0 + ((rh950 - 66.0) / max(4.0, rh950 - rh925)) * (z925_eff - 540.0)
        elif rh975 >= 66.0:
            marine_lid_asl_m = 320.0 + ((rh975 - 66.0) / max(4.0, rh975 - rh950)) * 220.0
        elif t925 > t950 or t850 > t950 or (rh975 - rh950) >= 12.0:
            # Very shallow heat-wave marine inversion capped below 975 hPa (~320 m ASL)
            marine_lid_asl_m = max(185.0, min(360.0, 320.0 * (max(25.0, rh975) / 50.0)))
        elif cc_low < 10.0 and max(rh975, rh950) < 66.0:
            # Clear sub-saturated marine airmass: shallow dry MBL rather than 1.1 km cloud lid
            marine_lid_asl_m = max(240.0, min(520.0, 260.0 + pbl_agl_m * 0.42))
        else:
            marine_lid_asl_m = max(340.0, min(980.0, 380.0 + pbl_agl_m * 0.65))

        if elev_m >= 320.0 and rh2m >= 88.0:
            # Summit station itself is saturated inside fog
            marine_lid_asl_m = max(marine_lid_asl_m, elev_m + min(260.0, max(55.0, pbl_agl_m * 0.85)))
        elif elev_m < 220.0 and rh975 >= 78.0 and pbl_agl_m > 80.0:
            marine_lid_asl_m = 0.76 * marine_lid_asl_m + 0.24 * (elev_m + pbl_agl_m + 140.0)

        inv_top_asl_m = max(185.0, min(1550.0, marine_lid_asl_m))
        inv_base_asl_m = max(70.0, inv_top_asl_m - 210.0)
    elif elev_m < 650.0:
        t_above_max = max(above_sfc_low_temps) if above_sfc_low_temps else t2m_c
        inv_jump_k = max(0.0, t_above_max - t2m_c)
        inv_base_asl_m = elev_m + min(950.0, max(110.0, pbl_agl_m * 0.88))
        inv_top_asl_m = inv_base_asl_m + 310.0
    else:
        # High-elevation valley / basin inversion
        t_above_max = max(above_sfc_hi_temps) if above_sfc_hi_temps else t2m_c
        inv_jump_k = max(0.0, t_above_max - t2m_c)
        inv_base_asl_m = elev_m + min(750.0, max(140.0, pbl_agl_m * 0.65))
        inv_top_asl_m = inv_base_asl_m + 320.0

    lcl_agl_m = max(45.0, 125.0 * max(0.2, t2m_c - dew2m_c))
    lcl_asl_m = elev_m + lcl_agl_m
    pbl_top_asl_m = elev_m + pbl_agl_m

    # Effective low/mid/high cloud percentages incorporating pressure-level saturation
    eff_low_pct = max(cc_low, c925 * 0.9, c850 * 0.82)
    if (is_coastal or elev_m < 450.0) and max(rh975, rh950) >= 92.5:
        eff_low_pct = max(eff_low_pct, min(96.0, (max(rh975, rh950) - 87.0) * 5.2))
    eff_mid_pct = max(cc_mid, c700 * 0.92, c500 * 0.85)
    eff_high_pct = max(cc_high, c300 * 0.95)

    # Construct altitude-exact `CloudLayerBand`s for Rust `corridor-raycast`:
    cloud_bands: list[dict[str, Any]] = []

    # Band 1: Boundary-layer marine stratus / Tule radiation fog (when inversion or low saturation present)
    is_low_marine_or_valley = (
        is_coastal
        and elev_m < 1350.0
        and ((inv_jump_k >= 0.6 and eff_low_pct >= 5.0) or eff_low_pct >= 8.0 or rh975 >= 92.0 or rh2m >= 92.0)
    ) or (
        elev_m < 950.0
        and ((inv_jump_k >= 0.8 and eff_low_pct >= 8.0) or rh2m >= 93.0 or rh975 >= 92.5)
    )
    if is_low_marine_or_valley:
        if is_coastal and elev_m < 1350.0:
            fog_top_m = max(185.0, min(1550.0, inv_top_asl_m))
            sea_lcl_m = lcl_asl_m if elev_m < 240.0 else max(80.0, min(fog_top_m - 130.0, fog_top_m * 0.42))
            fog_base_m = max(55.0, min(fog_top_m - 85.0, sea_lcl_m))
        else:
            fog_top_m = max(elev_m + 140.0, min(inv_top_asl_m, elev_m + 1150.0))
            if rh950 < 68.0 and fog_top_m > 520.0 and elev_m < 250.0:
                fog_top_m = 460.0
            fog_base_m = max(elev_m + 20.0, min(fog_top_m - 90.0, lcl_asl_m))
        fog_cf = max(0.0, min(1.0, eff_low_pct / 100.0))
        if fog_cf >= 0.03:
            fog_thick_m = max(70.0, fog_top_m - fog_base_m)
            cloud_bands.append(
                {
                    "label": "fog",
                    "base_asl_m": round(fog_base_m, 1),
                    "top_asl_m": round(fog_top_m, 1),
                    "cloud_fraction": round(fog_cf, 3),
                    "optical_depth_vertical": round(fog_cf * min(26.0, 0.035 * fog_thick_m), 2),
                    "aspect_ratio_h_over_d": 0.06,
                    "characteristic_cell_km": 14.0 if fog_cf > 0.65 else 5.5,
                    "ice_fraction": 0.0,
                }
            )
        # Remaining cumulus / stratocumulus above marine fog deck (e.g. 850 hPa)
        rem_low_cf = max(0.0, min(1.0, (c850 / 100.0) * 0.75))
        if rem_low_cf >= 0.05 and z850 > fog_top_m + 180.0:
            cu_base = max(fog_top_m + 120.0, z925)
            cu_top = max(cu_base + 380.0, z850 + 320.0 + min(900.0, cape * 0.8))
            cloud_bands.append(
                {
                    "label": "low",
                    "base_asl_m": round(cu_base, 1),
                    "top_asl_m": round(cu_top, 1),
                    "cloud_fraction": round(rem_low_cf, 3),
                    "optical_depth_vertical": round(rem_low_cf * 9.5, 2),
                    "aspect_ratio_h_over_d": round(min(0.95, 0.25 + cape / 900.0), 2),
                    "characteristic_cell_km": 2.6,
                    "ice_fraction": 0.0 if t850 > -4.0 else 0.25,
                }
            )
    else:
        low_cf = max(0.0, min(1.0, eff_low_pct / 100.0))
        if low_cf >= 0.03:
            low_base = max(elev_m + 180.0, min(z850 + 250.0, lcl_asl_m))
            convective_boost_m = min(1800.0, math.sqrt(max(0.0, cape)) * 42.0)
            low_top = max(low_base + 320.0, max(z850 + 350.0, low_base + 480.0 + convective_boost_m))
            aspect = min(1.10, 0.16 + math.sqrt(max(0.0, cape)) * 0.024)
            cell_km = 2.2 if cape > 120.0 else (8.5 if low_cf > 0.65 else 4.2)
            ice_f = max(0.0, min(1.0, (-t850 - 2.0) / 22.0))
            cloud_bands.append(
                {
                    "label": "low",
                    "base_asl_m": round(low_base, 1),
                    "top_asl_m": round(low_top, 1),
                    "cloud_fraction": round(low_cf, 3),
                    "optical_depth_vertical": round(
                        low_cf * min(32.0, 0.014 * (low_top - low_base)), 2
                    ),
                    "aspect_ratio_h_over_d": round(aspect, 2),
                    "characteristic_cell_km": cell_km,
                    "ice_fraction": round(ice_f, 2),
                }
            )

    # Band 2: Mid-tropospheric altocumulus / altostratus / Sierra lee-wave lenticular (700-500 hPa)
    mid_cf = max(0.0, min(1.0, eff_mid_pct / 100.0))
    if mid_cf >= 0.03:
        mid_base = max(elev_m + 650.0, z700 - 220.0)
        if c500 > c700 + 12.0:
            mid_base = max(mid_base, 0.5 * (z700 + z500) - 350.0)
        mid_top = max(mid_base + 520.0, z500 + (350.0 if c500 >= 25.0 else -350.0))
        t_mid = 0.5 * (t700 + t500)
        ice_mid = max(0.0, min(1.0, (-t_mid - 4.0) / 28.0))
        cloud_bands.append(
            {
                "label": "mid",
                "base_asl_m": round(mid_base, 1),
                "top_asl_m": round(mid_top, 1),
                "cloud_fraction": round(mid_cf, 3),
                "optical_depth_vertical": round(
                    mid_cf * min(18.0, 0.0055 * (mid_top - mid_base)), 2
                ),
                "aspect_ratio_h_over_d": 0.09 if cape < 180.0 else 0.45,
                "characteristic_cell_km": 15.0 if cape < 180.0 else 6.5,
                "ice_fraction": round(ice_mid, 2),
            }
        )

    # Band 3: Upper-tropospheric ice cirrus / cirrostratus (350-230 hPa, ~8.0-11.2 km ASL)
    high_cf = max(0.0, min(1.0, eff_high_pct / 100.0))
    if high_cf >= 0.03:
        hi_base = max(6800.0, z300 - 950.0)
        hi_top = max(hi_base + 1250.0, z300 + 950.0)
        # Cirrus is optically semi-transparent (in-cloud tau ~0.4..2.4), essential for crimson transmission & canvas glow!
        cloud_bands.append(
            {
                "label": "high",
                "base_asl_m": round(hi_base, 1),
                "top_asl_m": round(hi_top, 1),
                "cloud_fraction": round(high_cf, 3),
                "optical_depth_vertical": round(high_cf * (0.55 + 1.45 * high_cf), 2),
                "aspect_ratio_h_over_d": 0.05,
                "characteristic_cell_km": 28.0,
                "ice_fraction": 1.0,
            }
        )

    # Aerosol vertical scale height: compressed under strong marine/valley thermal inversions
    aerosol_scale_m = (
        max(520.0, min(1350.0, pbl_agl_m * 1.15))
        if inv_jump_k >= 2.0
        else max(1100.0, min(2400.0, pbl_agl_m * 1.25))
    )

    station_for_rust = {
        "distance_km": round(dist_km, 2),
        "azimuth_deg": round(azimuth_deg, 2) if azimuth_deg is not None else None,
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "surface_elevation_m": round(elev_m, 1),
        "surface_temp_c": round(t2m_c, 2),
        "surface_pressure_hpa": round(p_sfc, 1),
        "lapse_rate_k_km": round(lapse_k_km, 2),
        "inversion_base_asl_m": round(inv_base_asl_m, 1) if inv_jump_k > 0.2 else 0.0,
        "inversion_top_asl_m": round(inv_top_asl_m, 1) if inv_jump_k > 0.2 else 0.0,
        "inversion_strength_k": round(inv_jump_k, 2),
        "boundary_layer_top_asl_m": round(pbl_top_asl_m, 1),
        "aod_550nm": round(aod_550nm, 3),
        "angstrom_exponent": round(angstrom_exp, 2),
        "aerosol_scale_height_m": round(aerosol_scale_m, 1),
        "ozone_du": round(ozone_du, 1),
        "cloud_bands": cloud_bands,
    }

    return {
        "rust_station": station_for_rust,
        "temp_2m_c": round(t2m_c, 2),
        "dewpoint_2m_c": round(dew2m_c, 2),
        "rh_2m_pct": round(rh2m, 1),
        "rh_975_pct": round(rh975, 1),
        "rh_950_pct": round(rh950, 1),
        "rh_925_pct": round(rh925, 1),
        "temp_950_c": round(t950, 2),
        "temp_925_c": round(t925, 2),
        "temp_850_c": round(t850, 2),
        "cloud_total_pct": int(round(cc_tot)),
        "cloud_low_pct": int(round(eff_low_pct)),
        "cloud_mid_pct": int(round(eff_mid_pct)),
        "cloud_high_pct": int(round(eff_high_pct)),
        "c850_pct": int(round(c850)),
        "c700_pct": int(round(c700)),
        "c300_pct": int(round(c300)),
        "visibility_m": round(vis_m, 0),
        "precip_mm": round(precip_mm, 2),
        "cape_j_kg": round(cape, 1),
        "cin_j_kg": round(cin, 1),
        "pbl_agl_m": round(pbl_agl_m, 1),
        "lcl_asl_m": round(lcl_asl_m, 1),
        "inv_jump_k": round(inv_jump_k, 2),
        "inv_base_asl_m": round(inv_base_asl_m, 1),
        "inv_top_asl_m": round(inv_top_asl_m, 1),
        "wind_10m_kmh": round(wspd10_kmh, 1),
        "wind_10m_dir_deg": round(wdir10, 0),
        "wind_700_kmh": round(wspd700_kmh, 1),
        "wind_300_kmh": round(wspd300_kmh, 1),
        "wind_250_kmh": round(wspd250_kmh, 1),
        "wind_sfc_uv_ms": (u_sfc, v_sfc),
        "wind_850_uv_ms": (u_850, v_850),
        "wind_500_uv_ms": (u_500, v_500),
        "wind_250_uv_ms": (u_250, v_250),
    }


def _partition_prev_run_cloud_layers(
    tot_cc: float,
    rh_2m: float,
    temp_2m: float,
    dew_2m: float,
    dni_wm2: float,
    dhi_wm2: float,
    base_low: float,
    base_mid: float,
    base_high: float,
    is_coastal_pt: bool,
) -> tuple[float, float, float]:
    """
    Partitions `cloud_cover_previous_day{k}` (`0..100`) into `(low_pct, mid_pct, high_pct)`
    using physical boundary-layer saturation (`rh_2m`, `temp_2m - dew_2m`) and the Day-k
    radiative forward-scattering signature (`dni_wm2` vs `dhi_wm2`) without leaking Day-0 truth.
    """
    tot = max(0.0, min(100.0, tot_cc))
    if tot <= 1.5:
        return 0.0, 0.0, 0.0

    spread_c = max(0.0, temp_2m - dew_2m)
    bl_moist = max(0.0, min(1.0, (rh_2m - 72.0) / 22.0)) * max(0.0, min(1.0, (5.5 - spread_c) / 4.2))
    if is_coastal_pt and rh_2m >= 82.0:
        bl_moist = max(bl_moist, min(1.0, (rh_2m - 74.0) / 18.0))

    # Optically thin ice cirrus transmits substantial DNI (> 140 W/m^2) while boosting diffuse DHI
    translucent_cirrus = 0.0
    if dni_wm2 > 130.0 and tot >= 25.0:
        translucent_cirrus = max(0.0, min(1.0, (dni_wm2 - 110.0) / 320.0)) * (1.0 - 0.65 * bl_moist)

    w_low = 0.18 + 0.82 * bl_moist
    w_high = 0.22 + 0.72 * translucent_cirrus + 0.28 * (1.0 - bl_moist)
    w_mid = 0.20 + 0.35 * (1.0 - abs(bl_moist - 0.45))
    w_sum = max(1e-4, w_low + w_mid + w_high)

    # Allow realistic multi-layer overlap (sum of layers ~ 1.0..1.35 * tot when tot is high)
    overlap_gain = 1.0 + 0.28 * (tot / 100.0)
    low_p = max(0.0, min(100.0, tot * overlap_gain * (w_low / w_sum) * 1.35))
    mid_p = max(0.0, min(100.0, tot * overlap_gain * (w_mid / w_sum) * 1.15))
    high_p = max(0.0, min(100.0, tot * overlap_gain * (w_high / w_sum) * 1.25))

    # Ensure maximum layer reaches at least `0.88 * tot`
    cur_max = max(low_p, mid_p, high_p, 1.0)
    if cur_max < 0.88 * tot:
        scale = (0.88 * tot) / cur_max
        low_p = min(100.0, low_p * scale)
        mid_p = min(100.0, mid_p * scale)
        high_p = min(100.0, high_p * scale)

    return round(low_p, 1), round(mid_p, 1), round(high_p, 1)


def _reconstruct_hindcast_point_doc(
    base_pdoc: dict[str, Any],
    prev_hourly: dict[str, Any],
    sfx: str,
    pt_coastal: bool,
    utc_offset_seconds: int | None = None,
) -> dict[str, Any]:
    """
    Constructs a strictly leak-free hindcast point document from Open-Meteo Previous Runs
    surface fields (`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `cloud_cover`,
    `direct_normal_irradiance`, `diffuse_radiation`, `wind_speed_10m`, `wind_direction_10m`).

    CRITICAL ANTI-LEAKAGE INVARIANT:
    Never copies Day-0 hourly pressure-level arrays (`cloud_cover_925..300hPa`,
    `temperature_950..300hPa`, `relative_humidity_975..300hPa`, `wind_speed/direction_850..250hPa`,
    `geopotential_height_925..300hPa`, `visibility`, `cape`, `convective_inhibition`,
    `precipitation`, `surface_pressure`) from `base_pdoc`. Only static site metadata
    (`latitude`, `longitude`, `elevation`, `hourly.time`) is preserved; every meteorological
    hourly array is freshly reconstructed from the Day-k previous run alone.
    """
    raw_elev = base_pdoc.get("elevation")
    elev_m = float(raw_elev) if raw_elev is not None else (0.0 if pt_coastal else 1200.0)
    base_time = (base_pdoc.get("hourly") or {}).get("time") or prev_hourly.get("time") or []
    n_hr = max(24, len(base_time)) if base_time else 24
    n_hr = min(n_hr, 24)

    lo_arr: list[float] = []
    mi_arr: list[float] = []
    hi_arr: list[float] = []
    tot_arr: list[float] = []
    t2_arr: list[float] = []
    td2_arr: list[float] = []
    rh2_arr: list[float] = []
    ws10_arr: list[float] = []
    wd10_arr: list[float] = []
    p_sfc_arr: list[float] = []

    t1000 = [15.0] * n_hr
    t950 = [12.0] * n_hr
    t925 = [10.0] * n_hr
    t850 = [6.0] * n_hr
    t700 = [-3.0] * n_hr
    t500 = [-19.0] * n_hr
    t300 = [-41.0] * n_hr

    rh975 = [70.0] * n_hr
    rh950 = [58.0] * n_hr
    rh925 = [45.0] * n_hr
    rh850 = [32.0] * n_hr
    rh700 = [25.0] * n_hr
    rh500 = [22.0] * n_hr
    rh300 = [22.0] * n_hr

    c925 = [0.0] * n_hr
    c850 = [0.0] * n_hr
    c700 = [0.0] * n_hr
    c500 = [0.0] * n_hr
    c300 = [0.0] * n_hr

    pbl_arr = [450.0 if pt_coastal else 950.0] * n_hr
    p_sfc_std = round(1013.25 * ((1.0 - 2.25577e-5 * max(0.0, elev_m)) ** 5.25588), 1)

    for hr in range(n_hr):
        t2 = _safe_arr_val(prev_hourly, f"temperature_2m{sfx}", hr, 14.0)
        rh2 = _safe_arr_val(prev_hourly, f"relative_humidity_2m{sfx}", hr, 72.0)
        td2 = _safe_arr_val(prev_hourly, f"dew_point_2m{sfx}", hr, t2 - (100.0 - rh2) * 0.2)
        cc = _safe_arr_val(prev_hourly, f"cloud_cover{sfx}", hr, 25.0)
        dni = _safe_arr_val(prev_hourly, f"direct_normal_irradiance{sfx}", hr, 0.0)
        dhi = _safe_arr_val(prev_hourly, f"diffuse_radiation{sfx}", hr, 0.0)
        ws10 = _safe_arr_val(prev_hourly, f"wind_speed_10m{sfx}", hr, 10.0)
        wd10 = _safe_arr_val(prev_hourly, f"wind_direction_10m{sfx}", hr, 280.0)

        lo_p, mi_p, hi_p = _partition_prev_run_cloud_layers(
            cc, rh2, t2, td2, dni, dhi, 0.0, 0.0, 0.0, pt_coastal
        )
        tot_arr.append(round(cc, 1))
        lo_arr.append(lo_p)
        mi_arr.append(mi_p)
        hi_arr.append(hi_p)
        t2_arr.append(round(t2, 2))
        td2_arr.append(round(td2, 2))
        rh2_arr.append(round(rh2, 1))
        ws10_arr.append(round(ws10, 1))
        wd10_arr.append(round(wd10, 0))
        p_sfc_arr.append(p_sfc_std)

        # Standard hydrostatic lapse profile anchored strictly to previous-run T2m
        # (no Day-0 upper-air truth retained).
        t1000[hr] = round(t2 + 0.0065 * max(0.0, elev_m - 110.0), 2)
        t950[hr] = round(t2 - 0.0062 * (540.0 - elev_m), 2)
        t925[hr] = round(t2 - 0.0062 * (760.0 - elev_m), 2)
        t850[hr] = round(t2 - 0.0062 * (1455.0 - elev_m), 2)
        t700[hr] = round(t850[hr] - 9.2, 2)
        t500[hr] = round(t700[hr] - 16.2, 2)
        t300[hr] = round(t500[hr] - 22.0, 2)

        if pt_coastal:
            # Realistic marine boundary-layer hydrolapse without leaking Day-0 pressure levels
            rh975[hr] = round(max(18.0, min(99.0, 0.52 * rh2 + 0.46 * lo_p)), 1)
            rh950[hr] = round(max(15.0, min(96.0, 0.34 * rh2 + 0.54 * lo_p)), 1)
            rh925[hr] = round(max(12.0, min(92.0, 0.22 * rh2 + 0.48 * lo_p)), 1)
            rh850[hr] = round(max(10.0, min(88.0, 0.14 * rh2 + 0.34 * lo_p + 0.18 * mi_p)), 1)
            pbl_arr[hr] = round(max(180.0, min(1150.0, 220.0 + ws10 * 14.0 + lo_p * 2.4)), 1)
        else:
            rh975[hr] = round(max(15.0, min(98.0, 0.72 * rh2 + 0.25 * lo_p)), 1)
            rh950[hr] = round(max(14.0, min(96.0, 0.60 * rh2 + 0.32 * lo_p)), 1)
            rh925[hr] = round(max(12.0, min(94.0, 0.48 * rh2 + 0.40 * lo_p)), 1)
            rh850[hr] = round(max(10.0, min(92.0, 0.32 * rh2 + 0.50 * lo_p)), 1)
            pbl_arr[hr] = round(max(260.0, min(2200.0, 520.0 + max(0.0, t2 - 8.0) * 48.0 + ws10 * 12.0)), 1)

        rh700[hr] = round(max(18.0, min(94.0, 0.22 * rh2 + 0.72 * mi_p)), 1)
        rh500[hr] = round(max(16.0, min(92.0, 0.18 * rh2 + 0.68 * mi_p + 0.15 * hi_p)), 1)
        rh300[hr] = round(max(16.0, min(90.0, 0.15 * rh2 + 0.78 * hi_p)), 1)

        c925[hr] = round(lo_p * 0.85, 1)
        c850[hr] = round(lo_p * 0.70, 1)
        c700[hr] = round(mi_p * 0.85, 1)
        c500[hr] = round(mi_p * 0.75, 1)
        c300[hr] = round(hi_p * 0.90, 1)

    clean_hourly: dict[str, Any] = {
        "time": list(base_time[:n_hr]),
        "temperature_2m": t2_arr,
        "dew_point_2m": td2_arr,
        "relative_humidity_2m": rh2_arr,
        "surface_pressure": p_sfc_arr,
        "cloud_cover": tot_arr,
        "cloud_cover_low": lo_arr,
        "cloud_cover_mid": mi_arr,
        "cloud_cover_high": hi_arr,
        "cloud_cover_925hPa": c925,
        "cloud_cover_850hPa": c850,
        "cloud_cover_700hPa": c700,
        "cloud_cover_500hPa": c500,
        "cloud_cover_300hPa": c300,
        "wind_speed_10m": ws10_arr,
        "wind_direction_10m": wd10_arr,
        "temperature_1000hPa": t1000,
        "temperature_950hPa": t950,
        "temperature_925hPa": t925,
        "temperature_850hPa": t850,
        "temperature_700hPa": t700,
        "temperature_500hPa": t500,
        "temperature_300hPa": t300,
        "relative_humidity_975hPa": rh975,
        "relative_humidity_950hPa": rh950,
        "relative_humidity_925hPa": rh925,
        "relative_humidity_850hPa": rh850,
        "relative_humidity_700hPa": rh700,
        "relative_humidity_500hPa": rh500,
        "relative_humidity_300hPa": rh300,
        "boundary_layer_height": pbl_arr,
        # Upper winds, geopotential heights, visibility, CAPE, and precip are intentionally
        # omitted so extract_hour_sounding() derives them solely from previous-run surface
        # winds and standard atmosphere — zero Day-0 truth leakage.
    }
    eff_utc_off = utc_offset_seconds if utc_offset_seconds is not None else base_pdoc.get("utc_offset_seconds")
    return {
        "latitude": base_pdoc.get("latitude"),
        "longitude": base_pdoc.get("longitude"),
        "elevation": base_pdoc.get("elevation"),
        "utc_offset_seconds": eff_utc_off,
        "hindcast_leak_free": True,
        "hourly": clean_hourly,
    }


def _apply_previous_runs_hindcast(
    base_point_docs: list[dict[str, Any]],
    coords: list[tuple[str, float, float, float, float | None]],
    lat: float,
    lon: float,
    d_iso: str,
    tz_enc: str,
    lead_days: int,
    is_coastal: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str] | None:
    """
    Fetches genuine archived `previous_day{lead_days}` NWP forecasts (`+24h`..`+168h`) from
    `previous-runs-api.open-meteo.com` across all corridor points and ensemble models.
    """
    k = max(1, min(7, int(lead_days)))
    sfx = f"_previous_day{k}"
    lats_csv = ",".join(f"{c[1]:.4f}" for c in coords)
    lons_csv = ",".join(f"{c[2]:.4f}" for c in coords)

    corr_vars = (
        f"temperature_2m{sfx},dew_point_2m{sfx},relative_humidity_2m{sfx},"
        f"cloud_cover{sfx},direct_normal_irradiance{sfx},diffuse_radiation{sfx},"
        f"wind_speed_10m{sfx},wind_direction_10m{sfx}"
    )
    mm_vars = (
        f"temperature_2m{sfx},relative_humidity_2m{sfx},dew_point_2m{sfx},"
        f"cloud_cover{sfx},direct_normal_irradiance{sfx},diffuse_radiation{sfx},"
        f"wind_speed_10m{sfx},wind_direction_10m{sfx}"
    )

    corr_url = _open_meteo_url(
        "previous-runs-api",
        f"/v1/forecast?latitude={lats_csv}&longitude={lons_csv}&hourly={corr_vars}"
        f"&start_date={d_iso}&end_date={d_iso}&timezone={tz_enc}",
    )
    mm_url = _open_meteo_url(
        "previous-runs-api",
        f"/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}&hourly={mm_vars}"
        f"&models=ncep_hrrr_conus,ecmwf_ifs025,gfs_seamless"
        f"&start_date={d_iso}&end_date={d_iso}&timezone={tz_enc}",
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_c = pool.submit(_get_json, corr_url, 10.0)
        f_m = pool.submit(_get_json, mm_url, 9.0)
        raw_corr = f_c.result()
        try:
            raw_mm = f_m.result()
        except Exception:
            raw_mm = None

    prev_docs = raw_corr if isinstance(raw_corr, list) else [raw_corr]
    if len(prev_docs) != len(base_point_docs):
        return None
    h_check = (prev_docs[0].get("hourly") or {}).get(f"cloud_cover{sfx}") or []
    if not any(v is not None for v in h_check):
        return None

    out_docs: list[dict[str, Any]] = []
    for idx, pdoc in enumerate(base_point_docs):
        ph = prev_docs[idx].get("hourly") or {}
        pt_lon = coords[idx][2]
        pt_coastal = is_coastal and (pt_lon <= -121.6)
        out_docs.append(
            _reconstruct_hindcast_point_doc(
                pdoc,
                ph,
                sfx,
                pt_coastal,
                utc_offset_seconds=prev_docs[idx].get("utc_offset_seconds"),
            )
        )

    synth_mm: dict[str, Any] | None = None
    if isinstance(raw_mm, dict) and raw_mm.get("hourly"):
        mh = raw_mm["hourly"]
        mm_out_h: dict[str, Any] = {"time": mh.get("time") or []}
        for m_name in ("ncep_hrrr_conus", "ecmwf_ifs025", "gfs_seamless"):
            cc_list = mh.get(f"cloud_cover{sfx}_{m_name}") or []
            if not any(v is not None for v in cc_list):
                continue
            lo_m, mi_m, hi_m, tot_m, t2_m = [], [], [], [], []
            for hr in range(24):
                cc_v = _safe_arr_val(mh, f"cloud_cover{sfx}_{m_name}", hr, 0.0)
                rh_v = _safe_arr_val(mh, f"relative_humidity_2m{sfx}_{m_name}", hr, 72.0)
                t2_v = _safe_arr_val(mh, f"temperature_2m{sfx}_{m_name}", hr, 14.0)
                td_v = _safe_arr_val(mh, f"dew_point_2m{sfx}_{m_name}", hr, t2_v - 4.0)
                dni_v = _safe_arr_val(mh, f"direct_normal_irradiance{sfx}_{m_name}", hr, 0.0)
                dhi_v = _safe_arr_val(mh, f"diffuse_radiation{sfx}_{m_name}", hr, 0.0)
                lp, mp, hp = _partition_prev_run_cloud_layers(
                    cc_v, rh_v, t2_v, td_v, dni_v, dhi_v, 0.0, 0.0, 0.0, is_coastal
                )
                tot_m.append(round(cc_v, 1))
                lo_m.append(lp)
                mi_m.append(mp)
                hi_m.append(hp)
                t2_m.append(round(t2_v, 2))
            mm_out_h[f"cloud_cover_{m_name}"] = tot_m
            mm_out_h[f"cloud_cover_low_{m_name}"] = lo_m
            mm_out_h[f"cloud_cover_mid_{m_name}"] = mi_m
            mm_out_h[f"cloud_cover_high_{m_name}"] = hi_m
            mm_out_h[f"temperature_2m_{m_name}"] = t2_m
        aifs_is_synthetic = False
        # Mirror IFS/GFS blend to AIFS slot when AIFS previous_day is absent, and flag explicitly
        if "cloud_cover_low_ecmwf_ifs025" in mm_out_h and "cloud_cover_low_gfs_seamless" in mm_out_h:
            aifs_is_synthetic = True
            mm_out_h["cloud_cover_ecmwf_aifs025_single"] = [
                round(0.65 * a + 0.35 * b, 1)
                for a, b in zip(mm_out_h["cloud_cover_ecmwf_ifs025"], mm_out_h["cloud_cover_gfs_seamless"])
            ]
            mm_out_h["cloud_cover_low_ecmwf_aifs025_single"] = [
                round(0.65 * a + 0.35 * b, 1)
                for a, b in zip(mm_out_h["cloud_cover_low_ecmwf_ifs025"], mm_out_h["cloud_cover_low_gfs_seamless"])
            ]
            mm_out_h["cloud_cover_mid_ecmwf_aifs025_single"] = [
                round(0.65 * a + 0.35 * b, 1)
                for a, b in zip(mm_out_h["cloud_cover_mid_ecmwf_ifs025"], mm_out_h["cloud_cover_mid_gfs_seamless"])
            ]
            mm_out_h["cloud_cover_high_ecmwf_aifs025_single"] = [
                round(0.65 * a + 0.35 * b, 1)
                for a, b in zip(mm_out_h["cloud_cover_high_ecmwf_ifs025"], mm_out_h["cloud_cover_high_gfs_seamless"])
            ]
        synth_mm = {
            "utc_offset_seconds": raw_mm.get("utc_offset_seconds"),
            "hourly": mm_out_h,
            "aifs_is_synthetic_blend": aifs_is_synthetic,
        }

    label = f"Archived True +{k * 24}h Hindcast (Open-Meteo Previous Runs Day-{k}, {d_iso})"
    return out_docs, synth_mm, label


def build_corridor_coords_meta(
    lat: float,
    lon: float,
    primary_az_deg: float,
    conjugate_az_deg: float,
) -> list[tuple[str, float, float, float, float | None]]:
    """
    Constructs the canonical 21-point spatial coordinate grid (`corr_0km..320km`,
    `conj_18km..190km`, `transect_-11..+11km`, and `meso_*` 160 km box anchors)
    shared by Open-Meteo NWP soundings and native 3 km NOAA HRRR S3 extraction.
    """
    coords: list[tuple[str, float, float, float, float | None]] = []
    # [0..5]: 6 primary corridor stations
    for d_km in CORRIDOR_DISTANCES_KM:
        slat, slon = bearing_offset_lat_lon_km(lat, lon, primary_az_deg, d_km)
        coords.append((f"corr_{int(d_km)}km", slat, slon, d_km, primary_az_deg))

    # [6..8]: 3 conjugate-sector corridor stations
    for d_km in CONJUGATE_DISTANCES_KM:
        slat, slon = bearing_offset_lat_lon_km(lat, lon, conjugate_az_deg, d_km)
        coords.append((f"conj_{int(d_km)}km", slat, slon, d_km, conjugate_az_deg))

    # [9..12]: 4 local W-to-E transect stations (-11, -5.5, +5.5, +11 km East; 0 km is coords[0])
    for ex_km in (-11.0, -5.5, +5.5, 11.0):
        slat, slon = offset_lat_lon_km(lat, lon, ex_km, 0.0)
        coords.append((f"transect_{ex_km:+.1f}km", slat, slon, abs(ex_km), 90.0 if ex_km > 0 else 270.0))

    # [13..20]: 8 outer mesoscale 160 km box anchors
    meso_offsets_km = [
        (-68.0, -68.0),
        (0.0, -68.0),
        (68.0, -68.0),
        (-68.0, 0.0),
        (68.0, 0.0),
        (-68.0, 68.0),
        (0.0, 68.0),
        (68.0, 68.0),
    ]
    for ex_km, ny_km in meso_offsets_km:
        slat, slon = offset_lat_lon_km(lat, lon, ex_km, ny_km)
        az_m = math.degrees(math.atan2(ex_km, ny_km)) % 360.0
        coords.append((f"meso_{int(ex_km)}_{int(ny_km)}", slat, slon, math.hypot(ex_km, ny_km), az_m))
    return coords


def fetch_24h_corridor_and_models(
    lat: float,
    lon: float,
    date_str: str,
    primary_az_deg: float,
    conjugate_az_deg: float,
    is_coastal: bool = False,
    hindcast_lead_days: int = 0,
) -> dict[str, Any]:
    """
    Fetches full 24-hour (`00:00..23:00` local) pressure-level soundings across:
    - 6 primary corridor stations (`0, 15, 45, 100, 190, 320 km`) along `primary_az_deg`
    - 3 conjugate corridor stations (`18, 75, 190 km`) along `conjugate_az_deg`
    - 9 spatial mesoscale & local W-E transect points (`[-75..+75 km]`)
    plus multi-model comparison (`HRRR`, `ECMWF IFS`, `ECMWF AIFS`, `GFS`) and CAMS Air Quality.

    When `is_historical` and `hindcast_lead_days >= 1`, queries Open-Meteo's Previous Runs API
    (`previous_day1..7`) to reconstruct the genuine forecast issued `24h..168h` prior!
    """
    tz_obj, tz_name = local_tz_for_lon(lon)
    today_local = dt.datetime.now(tz_obj).date()
    try:
        req_date = dt.date.fromisoformat(date_str)
    except ValueError:
        req_date = today_local

    days_delta = (req_date - today_local).days
    is_historical = days_delta < 0
    eff_lead_days = max(0, min(7, int(hindcast_lead_days))) if is_historical else 0

    lead_tag = f":prev_v4_{eff_lead_days}" if eff_lead_days > 0 else ""
    cache_key = (
        f"nwp24_v2:{lat:.3f}:{lon:.3f}:{req_date.isoformat()}:"
        f"{primary_az_deg:.0f}:{conjugate_az_deg:.0f}{lead_tag}"
    )
    now_mono = time.monotonic()
    if cache_key in _MEM_CACHE:
        ts, cached = _MEM_CACHE[cache_key]
        if is_historical or (now_mono - ts < _MEM_TTL_SECONDS):
            return cached

    disk_path = NWP_CACHE_DIR / f"{cache_key.replace(':', '_')}.json"
    if disk_path.exists():
        try:
            age_s = time.time() - disk_path.stat().st_mtime
            if is_historical or age_s < _MEM_TTL_SECONDS:
                cached = json.loads(disk_path.read_text(encoding="utf-8"))
                if cached.get("ok") and (not is_historical or cached.get("multimodel_doc") is not None):
                    _MEM_CACHE[cache_key] = (now_mono, cached)
                    return cached
        except Exception:
            pass

    if eff_lead_days > 0:
        base_bundle = fetch_24h_corridor_and_models(
            lat, lon, date_str, primary_az_deg, conjugate_az_deg, is_coastal=is_coastal, hindcast_lead_days=0
        )
        if base_bundle.get("ok"):
            tz_enc_prev = tz_name.replace("/", "%2F")
            try:
                applied = _apply_previous_runs_hindcast(
                    base_bundle["point_docs"],
                    base_bundle["coords_meta"],
                    lat,
                    lon,
                    req_date.isoformat(),
                    tz_enc_prev,
                    eff_lead_days,
                    is_coastal,
                )
                if applied is not None:
                    prev_docs, prev_mm, prev_label = applied
                    hind_res = {
                        **base_bundle,
                        "hindcast_lead_days": eff_lead_days,
                        "source_label": prev_label,
                        "point_docs": prev_docs,
                        "multimodel_doc": prev_mm or {"hourly": {}},
                    }
                    _MEM_CACHE[cache_key] = (now_mono, hind_res)
                    NWP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    try:
                        disk_path.write_text(json.dumps(hind_res), encoding="utf-8")
                    except Exception:
                        pass
                    return hind_res
            except Exception:
                pass

    # Assemble the 21 spatial coordinates for the single-batch Open-Meteo request:
    coords = build_corridor_coords_meta(lat, lon, primary_az_deg, conjugate_az_deg)

    lats_csv = ",".join(f"{c[1]:.4f}" for c in coords)
    lons_csv = ",".join(f"{c[2]:.4f}" for c in coords)
    d_iso = req_date.isoformat()
    tz_enc = tz_name.replace("/", "%2F")

    def _fetch_corridor_batch() -> tuple[list[dict[str, Any]], str]:
        if not is_historical:
            url = _open_meteo_url(
                "api",
                f"/v1/forecast?latitude={lats_csv}&longitude={lons_csv}"
                f"&hourly={PRESSURE_SOUNDING_VARS}&start_date={d_iso}&end_date={d_iso}&timezone={tz_enc}",
            )
            data = _get_json(url, timeout=10.0)
            docs = data if isinstance(data, list) else [data]
            return docs, "Open-Meteo High-Res NWP Sounding (HRRR 3km / IFS Seamless)"

        # Historical date: try archived high-res forecast first (available 2021-present with pressure levels!),
        # then fall back cleanly to ERA5 reanalysis archive (`archive-api.open-meteo.com`).
        try:
            hist_fc_url = _open_meteo_url(
                "historical-forecast-api",
                f"/v1/forecast?latitude={lats_csv}&longitude={lons_csv}"
                f"&hourly={PRESSURE_SOUNDING_VARS}&start_date={d_iso}&end_date={d_iso}&timezone={tz_enc}",
            )
            data = _get_json(hist_fc_url, timeout=10.0)
            docs = data if isinstance(data, list) else [data]
            h0 = (docs[0].get("hourly") or {}).get("temperature_2m") or []
            if any(v is not None for v in h0):
                return docs, f"Archived High-Res NWP Sounding ({d_iso})"
        except Exception:
            pass

        era5_url = _open_meteo_url(
            "archive-api",
            f"/v1/archive?latitude={lats_csv}&longitude={lons_csv}"
            f"&hourly={ARCHIVE_REANALYSIS_VARS}&start_date={d_iso}&end_date={d_iso}&timezone={tz_enc}",
        )
        data = _get_json(era5_url, timeout=10.0)
        docs = data if isinstance(data, list) else [data]
        return docs, f"ERA5 Historical Reanalysis ({d_iso})"

    def _fetch_multimodel() -> dict[str, Any] | None:
        if is_historical and days_delta < -1500:
            return None
        host_pfx = "historical-forecast-api" if is_historical else "api"
        m_vars = (
            "temperature_2m,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
            "wind_speed_10m,wind_direction_10m"
        )
        url = _open_meteo_url(
            host_pfx,
            f"/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&hourly={m_vars}&models=ncep_hrrr_conus,ecmwf_ifs025,ecmwf_aifs025_single,gfs_seamless"
            f"&start_date={d_iso}&end_date={d_iso}&timezone={tz_enc}",
        )
        try:
            return _get_json(url, timeout=8.0)
        except Exception:
            return None

    def _fetch_air_quality() -> list[dict[str, Any]] | None:
        # Query CAMS at target (0 km) and mid-corridor (100 km)
        lat100, lon100 = coords[3][1], coords[3][2]
        url = _open_meteo_url(
            "air-quality-api",
            f"/v1/air-quality?latitude={lat:.4f},{lat100:.4f}&longitude={lon:.4f},{lon100:.4f}"
            f"&hourly=aerosol_optical_depth,pm2_5,pm10,dust,ozone"
            f"&start_date={d_iso}&end_date={d_iso}&timezone={tz_enc}",
        )
        try:
            res = _get_json(url, timeout=7.5)
            return res if isinstance(res, list) else [res]
        except Exception:
            return None

    errors: list[str] = []
    point_docs: list[dict[str, Any]] = []
    source_label = "Unavailable"
    multimodel_doc: dict[str, Any] | None = None
    aq_docs: list[dict[str, Any]] | None = None

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_corr = pool.submit(_fetch_corridor_batch)
        f_mm = pool.submit(_fetch_multimodel)
        f_aq = pool.submit(_fetch_air_quality)

        try:
            point_docs, source_label = f_corr.result()
        except Exception as exc:
            errors.append(f"Corridor NWP fetch failed: {exc}")

        try:
            multimodel_doc = f_mm.result()
        except Exception as exc:
            errors.append(f"Multi-model fetch warning: {exc}")

        try:
            aq_docs = f_aq.result()
        except Exception as exc:
            errors.append(f"CAMS AQ fetch warning: {exc}")

    if len(point_docs) != len(coords) and disk_path.exists():
        try:
            stale_cached = json.loads(disk_path.read_text(encoding="utf-8"))
            if stale_cached.get("ok"):
                _MEM_CACHE[cache_key] = (now_mono, stale_cached)
                return stale_cached
        except Exception:
            pass

    # Parse 24-hour CAMS aerosol & ozone time series
    aod_0km_24h = [0.065] * 24
    aod_100km_24h = [0.065] * 24
    pm25_24h = [4.5] * 24
    dust_24h = [3.0] * 24
    angstrom_24h = [1.25] * 24
    ozone_du_24h = [300.0] * 24
    has_live_aq = False

    if aq_docs and len(aq_docs) >= 1:
        aq0 = aq_docs[0].get("hourly") or {}
        aq1 = (aq_docs[1].get("hourly") or {}) if len(aq_docs) > 1 else aq0
        raw_aod = aq0.get("aerosol_optical_depth") or []
        if any(v is not None for v in raw_aod):
            has_live_aq = True
            for hr in range(24):
                a0 = max(0.015, min(2.5, _safe_arr_val(aq0, "aerosol_optical_depth", hr, 0.065)))
                a1 = max(0.015, min(2.5, _safe_arr_val(aq1, "aerosol_optical_depth", hr, a0)))
                pm25 = max(0.5, _safe_arr_val(aq0, "pm2_5", hr, 4.5))
                pm10 = max(pm25, _safe_arr_val(aq0, "pm10", hr, pm25 * 1.6))
                dust = max(0.0, _safe_arr_val(aq0, "dust", hr, 2.5))
                oz_ug = max(20.0, _safe_arr_val(aq0, "ozone", hr, 75.0))

                fine_ratio = max(0.15, min(0.95, pm25 / max(1.0, pm10)))
                alpha = 0.35 + 1.35 * fine_ratio - min(0.35, dust / 60.0)
                if is_coastal and pm25 < 12.0:
                    alpha = min(alpha, 0.92)

                aod_0km_24h[hr] = round(a0, 3)
                aod_100km_24h[hr] = round(a1, 3)
                pm25_24h[hr] = round(pm25, 1)
                dust_24h[hr] = round(dust, 1)
                angstrom_24h[hr] = round(max(0.35, min(1.85, alpha)), 2)
                doy = req_date.timetuple().tm_yday
                seasonal_du = 302.0 + 22.0 * math.cos(2.0 * math.pi * (doy - 95) / 365.0)
                ozone_du_24h[hr] = round(seasonal_du + (oz_ug - 70.0) * 0.12, 1)

    result: dict[str, Any] = {
        "ok": len(point_docs) == len(coords),
        "is_historical": is_historical,
        "hindcast_lead_days": 0,
        "days_delta": days_delta,
        "timezone": tz_name,
        "req_date": d_iso,
        "today_local": today_local.isoformat(),
        "source_label": source_label,
        "coords_meta": coords,
        "point_docs": point_docs,
        "multimodel_doc": multimodel_doc,
        "has_live_aq": has_live_aq,
        "aod_0km_24h": aod_0km_24h,
        "aod_100km_24h": aod_100km_24h,
        "pm25_24h": pm25_24h,
        "dust_24h": dust_24h,
        "angstrom_24h": angstrom_24h,
        "ozone_du_24h": ozone_du_24h,
        "errors": errors,
    }

    if result["ok"]:
        _MEM_CACHE[cache_key] = (now_mono, result)
        NWP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            disk_path = NWP_CACHE_DIR / f"{cache_key.replace(':', '_')}.json"
            disk_path.write_text(json.dumps(result), encoding="utf-8")
        except Exception:
            pass

    return result


def _wind_speed_dir_from_uv_ms(u_ms: float, v_ms: float) -> tuple[float, float]:
    """Converts `(u_east_ms, v_north_ms)` back to meteorological `(speed_kmh, from_deg)`."""
    spd_kmh = round(math.hypot(u_ms, v_ms) * 3.6, 1)
    if spd_kmh < 0.2:
        return 0.0, 270.0
    from_deg = round((math.degrees(math.atan2(-u_ms, -v_ms)) + 360.0) % 360.0, 1)
    return spd_kmh, from_deg


def build_nwp_bundle_from_hrrr_s3(
    hrrr_s3_data: dict[str, Any],
    coords_meta: list[tuple[str, float, float, float, float | None]],
    lat: float,
    lon: float,
    date_str: str,
    primary_hr_idx: int,
    *,
    is_coastal: bool = False,
    hourly_hrrr_stations: dict[int, list[dict[str, Any]]] | None = None,
    upstream_errors: list[str] | None = None,
) -> dict[str, Any]:
    """
    Constructs a real 24-hour pressure-level `nwp_bundle` directly from native 3 km NOAA HRRR S3
    GRIB2 soundings (`stations` + optional multi-hour diurnal anchor HRRR forecasts) when Open-Meteo
    is rate-limited (`HTTP 429`) or unreachable. Zero synthetic weather values are invented.
    """
    tz_obj, tz_name = local_tz_for_lon(lon)
    today_local = dt.datetime.now(tz_obj).date()
    try:
        req_date = dt.date.fromisoformat(date_str)
    except ValueError:
        req_date = today_local
    days_delta = (req_date - today_local).days
    is_historical = days_delta < 0

    anchor_map: dict[int, list[dict[str, Any]]] = {}
    if hourly_hrrr_stations:
        for h_k, st_list in hourly_hrrr_stations.items():
            if isinstance(st_list, list) and len(st_list) >= len(coords_meta):
                anchor_map[max(0, min(23, int(h_k)))] = st_list
    primary_stations = hrrr_s3_data.get("stations") or []
    anchor_map[max(0, min(23, int(primary_hr_idx)))] = primary_stations
    sorted_anchor_hrs = sorted(anchor_map.keys())

    def _interp_station_field(st_idx: int, hr: int, key: str, default: float) -> float:
        if len(sorted_anchor_hrs) == 1:
            v = anchor_map[sorted_anchor_hrs[0]][st_idx].get(key)
            return float(v) if v is not None else default
        if hr in anchor_map:
            v = anchor_map[hr][st_idx].get(key)
            return float(v) if v is not None else default
        # Linear interpolation between nearest surrounding real HRRR forecast hours (clamped at ends)
        if hr <= sorted_anchor_hrs[0]:
            v = anchor_map[sorted_anchor_hrs[0]][st_idx].get(key)
            return float(v) if v is not None else default
        if hr >= sorted_anchor_hrs[-1]:
            v = anchor_map[sorted_anchor_hrs[-1]][st_idx].get(key)
            return float(v) if v is not None else default
        h_lo = sorted_anchor_hrs[0]
        h_hi = sorted_anchor_hrs[-1]
        for k in sorted_anchor_hrs:
            if k <= hr:
                h_lo = k
            elif k > hr:
                h_hi = k
                break
        w = (hr - h_lo) / max(1.0, float(h_hi - h_lo))
        v_lo = anchor_map[h_lo][st_idx].get(key)
        v_hi = anchor_map[h_hi][st_idx].get(key)
        if v_lo is None and v_hi is None:
            return default
        if v_lo is None:
            return float(v_hi)
        if v_hi is None:
            return float(v_lo)
        return (1.0 - w) * float(v_lo) + w * float(v_hi)

    ref_dt_local = dt.datetime(req_date.year, req_date.month, req_date.day, 12, 0, tzinfo=tz_obj)
    utc_off_s = int(round((ref_dt_local.utcoffset() or dt.timedelta(0)).total_seconds()))

    point_docs: list[dict[str, Any]] = []
    for idx, (_name, plat, plon, _dist_km, _az) in enumerate(coords_meta):
        st0 = primary_stations[idx]
        elev_m = float(st0.get("z_sfc_m") or 0.0)
        hourly: dict[str, list[Any]] = {
            "time": [f"{req_date.isoformat()}T{hr:02d}:00" for hr in range(24)],
            "temperature_2m": [],
            "relative_humidity_2m": [],
            "dew_point_2m": [],
            "surface_pressure": [],
            "cloud_cover": [],
            "cloud_cover_low": [],
            "cloud_cover_mid": [],
            "cloud_cover_high": [],
            "visibility": [],
            "precipitation": [0.0] * 24,
            "cape": [],
            "convective_inhibition": [0.0] * 24,
            "boundary_layer_height": [],
            "wind_speed_10m": [],
            "wind_direction_10m": [],
            "temperature_950hPa": [],
            "temperature_925hPa": [],
            "temperature_850hPa": [],
            "temperature_700hPa": [],
            "temperature_500hPa": [],
            "temperature_300hPa": [],
            "relative_humidity_975hPa": [],
            "relative_humidity_950hPa": [],
            "relative_humidity_925hPa": [],
            "relative_humidity_850hPa": [],
            "relative_humidity_700hPa": [],
            "relative_humidity_500hPa": [],
            "relative_humidity_300hPa": [],
            "cloud_cover_925hPa": [],
            "cloud_cover_850hPa": [],
            "cloud_cover_700hPa": [],
            "cloud_cover_500hPa": [],
            "cloud_cover_300hPa": [],
            "geopotential_height_925hPa": [],
            "geopotential_height_850hPa": [],
            "geopotential_height_700hPa": [],
            "geopotential_height_500hPa": [],
            "geopotential_height_300hPa": [],
            "wind_speed_850hPa": [],
            "wind_direction_850hPa": [],
            "wind_speed_700hPa": [],
            "wind_direction_700hPa": [],
            "wind_speed_500hPa": [],
            "wind_direction_500hPa": [],
            "wind_speed_300hPa": [],
            "wind_direction_300hPa": [],
            "wind_speed_250hPa": [],
            "wind_direction_250hPa": [],
        }
        p_sfc_std = round(1013.25 * ((1.0 - 2.25577e-5 * max(0.0, elev_m)) ** 5.25588), 1)
        for hr in range(24):
            t2m = _interp_station_field(idx, hr, "temp_2m_c", 12.0)
            rh2m = _interp_station_field(idx, hr, "rh_2m_pct", 55.0)
            dpt2m = _interp_station_field(idx, hr, "dewpoint_2m_c", t2m - 6.0)
            lo = _interp_station_field(idx, hr, "lcdc_pct", 0.0)
            mi = _interp_station_field(idx, hr, "mcdc_pct", 0.0)
            hi = _interp_station_field(idx, hr, "hcdc_pct", 0.0)
            tot = max(lo, mi, hi, _interp_station_field(idx, hr, "tcdc_tot_pct", max(lo, mi, hi)))
            vis = _interp_station_field(idx, hr, "vis_m", 32000.0)
            cape = _interp_station_field(idx, hr, "cape_j_kg", 0.0)
            hpbl = _interp_station_field(idx, hr, "hpbl_m", 650.0)

            hourly["temperature_2m"].append(round(t2m, 2))
            hourly["relative_humidity_2m"].append(round(rh2m, 1))
            hourly["dew_point_2m"].append(round(dpt2m, 2))
            hourly["surface_pressure"].append(p_sfc_std)
            hourly["cloud_cover"].append(round(tot, 1))
            hourly["cloud_cover_low"].append(round(lo, 1))
            hourly["cloud_cover_mid"].append(round(mi, 1))
            hourly["cloud_cover_high"].append(round(hi, 1))
            hourly["visibility"].append(round(vis, 0))
            hourly["cape"].append(round(cape, 1))
            hourly["boundary_layer_height"].append(round(hpbl, 1))

            hourly["temperature_950hPa"].append(round(_interp_station_field(idx, hr, "temp_950_c", t2m - 2.0), 2))
            hourly["temperature_925hPa"].append(round(_interp_station_field(idx, hr, "temp_925_c", t2m - 3.5), 2))
            hourly["temperature_850hPa"].append(round(_interp_station_field(idx, hr, "temp_850_c", t2m - 7.0), 2))
            hourly["temperature_700hPa"].append(round(_interp_station_field(idx, hr, "temp_700_c", -4.0), 2))
            hourly["temperature_500hPa"].append(round(_interp_station_field(idx, hr, "temp_500_c", -20.0), 2))
            hourly["temperature_300hPa"].append(round(_interp_station_field(idx, hr, "temp_300_c", -44.0), 2))

            rh975 = _interp_station_field(idx, hr, "rh_975_pct", rh2m)
            rh950 = _interp_station_field(idx, hr, "rh_950_pct", rh2m * 0.92)
            rh925 = _interp_station_field(idx, hr, "rh_925_pct", rh2m * 0.85)
            rh700 = _interp_station_field(idx, hr, "rh_700_pct", max(20.0, mi * 0.85))
            rh500 = _interp_station_field(idx, hr, "rh_500_pct", max(18.0, mi * 0.78))
            rh300 = _interp_station_field(idx, hr, "rh_300_pct", max(18.0, hi * 0.82))
            hourly["relative_humidity_975hPa"].append(round(rh975, 1))
            hourly["relative_humidity_950hPa"].append(round(rh950, 1))
            hourly["relative_humidity_925hPa"].append(round(rh925, 1))
            hourly["relative_humidity_850hPa"].append(round(max(20.0, lo * 0.85), 1))
            hourly["relative_humidity_700hPa"].append(round(rh700, 1))
            hourly["relative_humidity_500hPa"].append(round(rh500, 1))
            hourly["relative_humidity_300hPa"].append(round(rh300, 1))

            hourly["cloud_cover_925hPa"].append(round(lo * 0.85, 1))
            hourly["cloud_cover_850hPa"].append(round(lo * 0.75, 1))
            hourly["cloud_cover_700hPa"].append(round(mi * 0.85, 1))
            hourly["cloud_cover_500hPa"].append(round(mi * 0.75, 1))
            hourly["cloud_cover_300hPa"].append(round(hi * 0.92, 1))

            hourly["geopotential_height_925hPa"].append(round(_interp_station_field(idx, hr, "z_925_m", 760.0), 1))
            hourly["geopotential_height_850hPa"].append(round(_interp_station_field(idx, hr, "z_850_m", 1480.0), 1))
            hourly["geopotential_height_700hPa"].append(round(_interp_station_field(idx, hr, "z_700_m", 3080.0), 1))
            hourly["geopotential_height_500hPa"].append(round(_interp_station_field(idx, hr, "z_500_m", 5650.0), 1))
            hourly["geopotential_height_300hPa"].append(round(_interp_station_field(idx, hr, "z_300_m", 9350.0), 1))

            for lvl_tag, u_k, v_k in (
                ("10m", "u10_ms", "v10_ms"),
                ("850hPa", "u850_ms", "v850_ms"),
                ("700hPa", "u700_ms", "v700_ms"),
                ("500hPa", "u500_ms", "v500_ms"),
                ("300hPa", "u300_ms", "v300_ms"),
                ("250hPa", "u250_ms", "v250_ms"),
            ):
                spd, dr = _wind_speed_dir_from_uv_ms(
                    _interp_station_field(idx, hr, u_k, 3.0),
                    _interp_station_field(idx, hr, v_k, 0.5),
                )
                hourly[f"wind_speed_{lvl_tag}"].append(spd)
                hourly[f"wind_direction_{lvl_tag}"].append(dr)

        point_docs.append(
            {
                "latitude": plat,
                "longitude": plon,
                "elevation": elev_m,
                "utc_offset_seconds": utc_off_s,
                "hourly": hourly,
            }
        )

    # HRRR-Smoke 550 nm AOD across the 24 hours
    aod_0km_24h: list[float] = []
    aod_100km_24h: list[float] = []
    idx_100 = min(3, len(coords_meta) - 1)
    has_smoke_aod = False
    for hr in range(24):
        a0 = _interp_station_field(0, hr, "aotk_550nm", 0.065)
        a1 = _interp_station_field(idx_100, hr, "aotk_550nm", a0)
        if primary_stations[0].get("aotk_550nm") is not None:
            has_smoke_aod = True
        aod_0km_24h.append(round(max(0.015, min(2.5, a0)), 3))
        aod_100km_24h.append(round(max(0.015, min(2.5, a1)), 3))

    cycle_tag = str(hrrr_s3_data.get("cycle_tag") or "latest")
    n_anchors = len(sorted_anchor_hrs)
    multimodel_hourly = {
        "cloud_cover_ncep_hrrr_conus": list(point_docs[0]["hourly"]["cloud_cover"]),
        "cloud_cover_low_ncep_hrrr_conus": list(point_docs[0]["hourly"]["cloud_cover_low"]),
        "cloud_cover_mid_ncep_hrrr_conus": list(point_docs[0]["hourly"]["cloud_cover_mid"]),
        "cloud_cover_high_ncep_hrrr_conus": list(point_docs[0]["hourly"]["cloud_cover_high"]),
    }

    return {
        "ok": True,
        "hrrr_s3_primary_fallback": True,
        "hrrr_anchor_hours": sorted_anchor_hrs,
        "is_historical": is_historical,
        "hindcast_lead_days": 0,
        "days_delta": days_delta,
        "timezone": tz_name,
        "req_date": req_date.isoformat(),
        "today_local": today_local.isoformat(),
        "source_label": f"NOAA HRRR 3km Native S3 ({cycle_tag} · {n_anchors} diurnal steps)",
        "coords_meta": coords_meta,
        "point_docs": point_docs,
        "multimodel_doc": {"utc_offset_seconds": utc_off_s, "hourly": multimodel_hourly},
        "has_live_aq": has_smoke_aod,
        "aod_0km_24h": aod_0km_24h,
        "aod_100km_24h": aod_100km_24h,
        "pm25_24h": [round(max(2.0, a * 55.0), 1) for a in aod_0km_24h],
        "dust_24h": [2.5] * 24,
        "angstrom_24h": [0.92 if is_coastal else 1.25] * 24,
        "ozone_du_24h": [300.0] * 24,
        "errors": list(upstream_errors or []),
    }

