"""
Live GOES-18 (West) & GOES-19 (East) Multi-Spectral Satellite Nowcasting (`0-3.5h`).

Ingests real-time geo-aligned NOAA GOES ABI channels over a `440 km x 440 km`
domain surrounding the target landmark and solar corridor:
- Channel 02 (`0.64 um` Red Visible reflectance) — daytime optical depth & fine cloud edges
- Channel 07 (`3.9 um` Shortwave IR window) — twilight/night marine stratus & valley fog
- Channel 11 (`8.4 um` Clean Longwave IR window) — cloud-top brightness temperature `T_B`
  and altitude classification (`low` < 2.5 km, `mid` 2.5-6.5 km, `high cirrus` > 6.5 km)

Computes:
1. Exact latest GOES ABI L2 scan timestamp from NOAA's public S3 bucket (`noaa-goes18` / `noaa-goes19`).
2. Real 2-frame satellite cloud motion vector `(u_ms, v_ms)` via Hanning-windowed 2D FFT
   phase correlation (`t - 25 min` -> `t_now`), blended with NWP steering winds when sky is clear.
3. Lagrangian upstream advection (`x_src = x - u_layer * dt`) out to `+210 minutes` (`3.5h`)
   for both the 6 log-spaced solar corridor stations (`0..320 km`) and the `16 x 16`
   (`160 km x 160 km`) mesoscale cloud texture fed into WebGL2 (`uMesoscaleCloudTex`).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
GOES_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "goes"
USER_AGENT = "PhotoPlanner/2.0 (landscape-optics-corridor; github.com/lucasbrambrink/photoplanner)"

_MEM_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MEM_TTL_SECONDS = 300.0  # 5 minutes (matches GOES CONUS 5-minute rapid scan cadence!)


def _solar_elevation_utc(lat: float, lon: float, when_utc: dt.datetime) -> float:
    """Fast solar elevation angle in degrees at `(lat, lon, when_utc)`."""
    doy = when_utc.timetuple().tm_yday
    frac_hr = when_utc.hour + when_utc.minute / 60.0 + when_utc.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (frac_hr - 12.0) / 24.0)
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
    tst = (frac_hr * 60.0 + eqtime + 4.0 * lon) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)
    lat_r = math.radians(lat)
    cos_z = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    cos_z = max(-1.0, min(1.0, cos_z))
    return 90.0 - math.degrees(math.acos(cos_z))


def _query_latest_goes_s3_scan_utc(is_west: bool, now_utc: dt.datetime) -> tuple[str | None, str]:
    """Queries NOAA's public S3 bucket XML listing for the latest ABI-L2-CMIPC scan timestamp."""
    bucket = "noaa-goes18" if is_west else "noaa-goes19"
    sat_name = "GOES-18 West" if is_west else "GOES-19 East"
    for dt_back_hr in (0, 1):
        probe_dt = now_utc - dt.timedelta(hours=dt_back_hr)
        doy = probe_dt.timetuple().tm_yday
        prefix = f"ABI-L2-CMIPC/{probe_dt.year}/{doy:03d}/{probe_dt.hour:02d}/"
        url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                xml_text = resp.read().decode("utf-8", errors="ignore")
            # Parse latest `_sYYYYDDDHHMMSS` start-scan token
            idx = xml_text.rfind("_s20")
            if idx != -1 and len(xml_text) >= idx + 16:
                tok = xml_text[idx + 2 : idx + 15]  # YYYYDDDHHMMSS (13 chars)
                yr = int(tok[0:4])
                dy = int(tok[4:7])
                hh = int(tok[7:9])
                mm = int(tok[9:11])
                ss = int(tok[11:13])
                scan_dt = dt.datetime(yr, 1, 1, hh, mm, ss, tzinfo=dt.timezone.utc) + dt.timedelta(
                    days=dy - 1
                )
                return scan_dt.isoformat(), f"NOAA {sat_name} ABI L2 ({bucket}.s3)"
        except Exception:
            continue
    return None, f"NOAA {sat_name} ABI CONUS Sector"


def _fetch_wms_channel_grid(
    cgi_name: str,
    layer: str,
    bbox: tuple[float, float, float, float],
    grid_px: int = 96,
    time_iso_z: str | None = None,
) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = bbox
    time_param = f"&TIME={time_iso_z}" if time_iso_z else ""
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/wms/{cgi_name}.cgi"
        f"?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}"
        f"&SRS=EPSG:4326&BBOX={min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}"
        f"&WIDTH={grid_px}&HEIGHT={grid_px}&FORMAT=image/png{time_param}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=5.5) as resp:
        raw = resp.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    # Row 0 in WMS GetMap image is `max_lat` (North), Row H-1 is `min_lat` (South).
    return arr[:, :, 0]


def _estimate_optical_flow_fft_ms(
    frame_prev: np.ndarray,
    frame_now: np.ndarray,
    dx_m: float,
    dy_m: float,
    dt_seconds: float,
) -> tuple[float, float, float]:
    """
    2D Hanning-windowed FFT phase correlation between `frame_prev` and `frame_now`.
    Returns `(u_east_ms, v_north_ms, confidence_0_to_1)`.
    Note: image row index increases southward (`+row = -north`), column increases eastward (`+col = +east`).
    """
    if dt_seconds <= 60.0 or frame_prev.shape != frame_now.shape:
        return 0.0, 0.0, 0.0

    p0 = frame_prev.astype(np.float64) - float(np.mean(frame_prev))
    p1 = frame_now.astype(np.float64) - float(np.mean(frame_now))
    if np.std(p0) < 2.2 or np.std(p1) < 2.2:
        # Almost completely uniform clear sky or featureless uniform deck
        return 0.0, 0.0, 0.05

    # Check if frames are identical (e.g. archive lag)
    if float(np.max(np.abs(frame_now - frame_prev))) < 0.5:
        return 0.0, 0.0, 0.0

    h, w = p0.shape
    win_y = np.hanning(h)[:, None]
    win_x = np.hanning(w)[None, :]
    window = win_y * win_x

    f0 = np.fft.fft2(p0 * window)
    f1 = np.fft.fft2(p1 * window)
    cross = f0 * np.conj(f1)
    denom = np.abs(cross) + 1e-8
    corr = np.real(np.fft.ifft2(cross / denom))
    corr_shifted = np.fft.fftshift(corr)

    cy, cx = h // 2, w // 2
    # Restrict physically plausible cloud speed to <= 55 m/s (~123 mph)
    max_shift_x = max(2, min(cx - 2, int(math.ceil(55.0 * dt_seconds / max(200.0, dx_m)))))
    max_shift_y = max(2, min(cy - 2, int(math.ceil(55.0 * dt_seconds / max(200.0, dy_m)))))

    sub = corr_shifted[cy - max_shift_y : cy + max_shift_y + 1, cx - max_shift_x : cx + max_shift_x + 1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    peak_val = float(sub[iy, ix])
    med_val = float(np.median(np.abs(sub))) + 1e-6

    # Sub-pixel parabolic refinement
    sub_dx = 0.0
    sub_dy = 0.0
    if 0 < ix < sub.shape[1] - 1:
        l_v, c_v, r_v = float(sub[iy, ix - 1]), peak_val, float(sub[iy, ix + 1])
        d2 = l_v - 2.0 * c_v + r_v
        if abs(d2) > 1e-6:
            sub_dx = max(-0.5, min(0.5, 0.5 * (l_v - r_v) / d2))
    if 0 < iy < sub.shape[0] - 1:
        u_v, c_v, d_v = float(sub[iy - 1, ix]), peak_val, float(sub[iy + 1, ix])
        d2 = u_v - 2.0 * c_v + d_v
        if abs(d2) > 1e-6:
            sub_dy = max(-0.5, min(0.5, 0.5 * (u_v - d_v) / d2))

    # Note: `corr = ifft(F(prev) * conj(F(now)))` peaks at `(prev_pos - now_pos)`, so displacement `now - prev` is negated:
    shift_cols_east = -((ix - max_shift_x) + sub_dx)
    shift_rows_south = -((iy - max_shift_y) + sub_dy)

    u_east_ms = (shift_cols_east * dx_m) / dt_seconds
    v_north_ms = (-shift_rows_south * dy_m) / dt_seconds
    snr = peak_val / med_val
    conf = max(0.0, min(0.96, (snr - 2.8) / 8.5))
    return round(u_east_ms, 2), round(v_north_ms, 2), round(conf, 3)


def _sample_grid_bilinear(
    grid: np.ndarray,
    bbox: tuple[float, float, float, float],
    q_lat: float,
    q_lon: float,
) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    h, w = grid.shape
    fx = (q_lon - min_lon) / max(1e-6, max_lon - min_lon) * (w - 1)
    fy = (max_lat - q_lat) / max(1e-6, max_lat - min_lat) * (h - 1)
    fx = max(0.0, min(float(w - 1.001), fx))
    fy = max(0.0, min(float(h - 1.001), fy))
    x0 = int(fx)
    y0 = int(fy)
    tx = fx - x0
    ty = fy - y0
    v00 = float(grid[y0, x0])
    v10 = float(grid[y0, x0 + 1])
    v01 = float(grid[y0 + 1, x0])
    v11 = float(grid[y0 + 1, x0 + 1])
    return (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty) + (v01 * (1.0 - tx) + v11 * tx) * ty


def fetch_goes_corridor_nowcast(
    lat: float,
    lon: float,
    surface_temp_c: float = 15.0,
    is_coastal: bool = False,
) -> dict[str, Any] | None:
    """
    Fetches live GOES-18 / GOES-19 multi-spectral grids over a `760 km x 760 km`
    domain around `(lat, lon)` (covering the full `380 km` corridor in any azimuth!)
    and classifies spatially resolved Low (`0-2.5 km`), Mid (`2.5-6.5 km`), and
    High Cirrus (`> 6.5 km`) cloud fraction + vertical optical depth grids, plus
    2-frame optical-flow motion `(u_ms, v_ms)`.
    """
    # CONUS / PACUS GOES sector coverage check
    if not (21.0 <= lat <= 53.0 and -138.0 <= lon <= -65.0):
        return None

    cache_key = f"goes_v2:{lat:.2f}:{lon:.2f}"
    now_mono = time.monotonic()
    if cache_key in _MEM_CACHE:
        ts, cached = _MEM_CACHE[cache_key]
        if now_mono - ts < _MEM_TTL_SECONDS:
            return cached

    is_west = lon < -102.0
    cgi_name = "goes_west" if is_west else "goes_east"
    now_utc = dt.datetime.now(dt.timezone.utc)

    # 760 km x 760 km domain (+/- 380 km) at 112x112 (~6.8 km/px)
    half_km = 385.0
    dlat = half_km / 111.13
    dlon = half_km / max(25.0, 111.41 * math.cos(math.radians(lat)))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    grid_px = 112

    try:
        scan_iso, sat_label = _query_latest_goes_s3_scan_utc(is_west, now_utc)
        t_now_dt = (
            dt.datetime.fromisoformat(scan_iso)
            if scan_iso
            else (now_utc - dt.timedelta(minutes=7))
        )
        t_prev_dt = t_now_dt - dt.timedelta(minutes=25)
        t_now_z = t_now_dt.strftime("%Y-%m-%dT%H:%M:00Z")
        t_prev_z = t_prev_dt.strftime("%Y-%m-%dT%H:%M:00Z")

        ch02_now = _fetch_wms_channel_grid(cgi_name, "conus_ch02", bbox, grid_px, t_now_z)
        ch07_now = _fetch_wms_channel_grid(cgi_name, "conus_ch07", bbox, grid_px, t_now_z)
        ch11_now = _fetch_wms_channel_grid(cgi_name, "conus_ch11", bbox, grid_px, t_now_z)

        sun_el_now = _solar_elevation_utc(lat, lon, t_now_dt)
        flow_layer = "conus_ch02" if sun_el_now > 8.0 else "conus_ch11"
        flow_now = ch02_now if sun_el_now > 8.0 else ch11_now
        try:
            flow_prev = _fetch_wms_channel_grid(cgi_name, flow_layer, bbox, grid_px, t_prev_z)
        except Exception:
            flow_prev = flow_now

        dx_m = (2.0 * half_km * 1000.0) / grid_px
        dy_m = dx_m
        u_sat_ms, v_sat_ms, flow_conf = _estimate_optical_flow_fft_ms(
            flow_prev, flow_now, dx_m, dy_m, 25.0 * 60.0
        )

        # Physical multi-spectral cloud classification across the 112x112 grid:
        # 1. Brightness temperature T_B (K) from pure grayscale 8.4 um clean IR window `ch11_now`:
        #    Empirically verified on live IEM GOES-18: warm clear desert ~213 -> ~298 K,
        #    1 km coastal marine overcast ~167.5 -> ~276 K, upper cirrus (<115) -> <250 K.
        tb_k = 192.0 + 0.50 * ch11_now
        sfc_temp_k = 273.15 + surface_temp_c
        # Thermal depression below surface temperature (larger = colder, higher cloud top!)
        dt_top_k = np.maximum(0.0, sfc_temp_k - tb_k)

        # 2. Visible optical depth & cloudiness when daytime (`sun_el_now > 5 deg`):
        has_daylight_vis = sun_el_now > 5.0
        if has_daylight_vis:
            # Ocean background ~62, vegetated/mountain land ~105, desert ~132;
            # Build smooth longitude-based coast/ocean clear-sky albedo baseline
            lons_1d = np.linspace(bbox[0], bbox[2], grid_px, dtype=np.float32)[None, :]
            if is_coastal:
                # West of -122.48 is low-albedo Pacific water (~68), east is coastal hills/bay (~104)
                ocean_mask = np.clip((-122.45 - lons_1d) / 0.25, 0.0, 1.0)
                clear_vis_bg = 104.0 * (1.0 - ocean_mask) + 66.0 * ocean_mask
            else:
                clear_vis_bg = np.full((1, grid_px), 114.0, dtype=np.float32)

            vis_excess = np.clip((ch02_now - clear_vis_bg) / (238.0 - clear_vis_bg), 0.0, 1.0)
            # Asymptotic two-stream reflectance -> vertical optical depth tau_vis
            mu0 = max(0.15, math.sin(math.radians(sun_el_now)))
            refl_cloud = np.clip(vis_excess * (0.85 + 0.15 * mu0), 0.0, 0.94)
            tau_total_grid = np.clip((refl_cloud / np.maximum(0.06, 1.0 - refl_cloud)) * 4.8, 0.0, 48.0)
        else:
            vis_excess = np.zeros_like(ch11_now)
            tau_total_grid = np.zeros_like(ch11_now)

        # 3. Nocturnal / twilight 3.9 um vs 8.4 um liquid stratus/fog emissive index:
        #    Verified on live GOES-18: SF marine stratus = 82.2 - 0.31*167.5 = +30.3, Clear desert = +0.3!
        swir_fog_idx = np.clip((ch07_now - 0.31 * ch11_now - 6.0) / 26.0, 0.0, 1.0)

        # 4. Partition into High Cirrus (T_B < 246 K or dt_top > 34 K), Mid (dt_top 15..34 K), and Low (< 15 K):
        high_ir = np.clip((250.0 - tb_k) / 24.0, 0.0, 1.0)
        mid_ir = np.clip((dt_top_k - 13.5) / 14.0, 0.0, 1.0) * (1.0 - 0.75 * high_ir)

        if has_daylight_vis:
            low_cf_grid = np.clip(
                np.maximum(vis_excess * (1.0 - 0.65 * high_ir - 0.50 * mid_ir), swir_fog_idx * 0.85),
                0.0,
                1.0,
            )
            mid_cf_grid = np.clip(mid_ir * (0.45 + 0.55 * vis_excess), 0.0, 1.0)
            # Thin cirrus has modest visible excess (~0.10-0.35) but cold IR signature
            high_cf_grid = np.clip(np.maximum(high_ir, vis_excess * 0.25 * (dt_top_k / 28.0)), 0.0, 1.0)
        else:
            low_cf_grid = np.clip(
                np.maximum(swir_fog_idx, np.clip((dt_top_k - 3.5) / 11.0, 0.0, 1.0) * (1.0 - mid_ir - high_ir)),
                0.0,
                1.0,
            )
            mid_cf_grid = np.clip(mid_ir, 0.0, 1.0)
            high_cf_grid = np.clip(high_ir, 0.0, 1.0)
            tau_total_grid = np.clip(
                low_cf_grid * 9.5 + mid_cf_grid * 6.5 + high_cf_grid * 1.8,
                0.0,
                42.0,
            )

        age_min = max(0.0, (now_utc - t_now_dt).total_seconds() / 60.0)
        GOES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        result: dict[str, Any] = {
            "satellite": sat_label,
            "is_west": is_west,
            "scan_timestamp_utc": t_now_dt.isoformat(),
            "scan_age_minutes": round(age_min, 1),
            "sun_el_at_scan_deg": round(sun_el_now, 2),
            "has_daylight_visible": has_daylight_vis,
            "bbox": bbox,
            "grid_px": grid_px,
            "motion_u_ms": u_sat_ms,
            "motion_v_ms": v_sat_ms,
            "motion_confidence": flow_conf,
            "low_cf_grid": low_cf_grid,
            "mid_cf_grid": mid_cf_grid,
            "high_cf_grid": high_cf_grid,
            "tau_grid": tau_total_grid,
            "tb_k_grid": tb_k,
        }
        _MEM_CACHE[cache_key] = (now_mono, result)
        return result
    except Exception:
        return None


def sample_advected_goes_state(
    goes_data: dict[str, Any],
    q_lat: float,
    q_lon: float,
    lead_minutes: float,
    wind_low_uv_ms: tuple[float, float],
    wind_mid_uv_ms: tuple[float, float],
    wind_high_uv_ms: tuple[float, float],
) -> dict[str, float]:
    """
    Lagrangian back-trajectory evaluation at `(q_lat, q_lon)` for `+lead_minutes` (`0..210 min`).
    Blends 2-frame GOES FFT optical-flow motion with layer-specific NWP steering winds so
    differentially sheared low marine decks and upper jet-stream cirrus advect along their
    true respective trajectories!
    """
    bbox = goes_data["bbox"]
    dt_sec = max(0.0, min(240.0, lead_minutes)) * 60.0
    conf = float(goes_data.get("motion_confidence", 0.0))
    u_sat = float(goes_data.get("motion_u_ms", 0.0))
    v_sat = float(goes_data.get("motion_v_ms", 0.0))

    m_per_lat = 111_132.92
    m_per_lon = max(25_000.0, 111_412.84 * math.cos(math.radians(q_lat)))

    def _advect_latlon(w_uv: tuple[float, float], sat_weight: float) -> tuple[float, float]:
        w_eff = conf * sat_weight
        u_eff = w_eff * u_sat + (1.0 - w_eff) * w_uv[0]
        v_eff = w_eff * v_sat + (1.0 - w_eff) * w_uv[1]
        # Upstream source point at t_scan that arrives at (q_lat, q_lon) after dt_sec:
        src_lat = q_lat - (v_eff * dt_sec) / m_per_lat
        src_lon = q_lon - (u_eff * dt_sec) / m_per_lon
        return src_lat, src_lon

    lat_low, lon_low = _advect_latlon(wind_low_uv_ms, 0.72)
    lat_mid, lon_mid = _advect_latlon(wind_mid_uv_ms, 0.60)
    lat_hi, lon_hi = _advect_latlon(wind_high_uv_ms, 0.55)

    low_cf = _sample_grid_bilinear(goes_data["low_cf_grid"], bbox, lat_low, lon_low)
    mid_cf = _sample_grid_bilinear(goes_data["mid_cf_grid"], bbox, lat_mid, lon_mid)
    high_cf = _sample_grid_bilinear(goes_data["high_cf_grid"], bbox, lat_hi, lon_hi)
    tau_vis = _sample_grid_bilinear(goes_data["tau_grid"], bbox, lat_low, lon_low)
    tb_k = _sample_grid_bilinear(goes_data["tb_k_grid"], bbox, lat_hi, lon_hi)

    return {
        "low_cf": max(0.0, min(1.0, low_cf)),
        "mid_cf": max(0.0, min(1.0, mid_cf)),
        "high_cf": max(0.0, min(1.0, high_cf)),
        "tau_total": max(0.0, min(48.0, tau_vis)),
        "tb_k": tb_k,
    }
