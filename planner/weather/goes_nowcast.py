"""
Calibrated Multi-Frame GOES-18 (West) & GOES-19 (East) Satellite Nowcasting (`0-3.5h`).

Ingests real 5-minute NOAA GOES-R Level-2 geophysical NetCDF4/HDF5 products directly
from NOAA's public S3 buckets (`noaa-goes18`, `noaa-goes19`, `noaa-goes16`) across a
`768 km x 768 km` (`128 x 128`, `6.0 km/px`) domain surrounding the target landmark
and solar corridor:
- `ABI-L2-ACTPC` (`2 km` CONUS): Enterprise Cloud Top Phase (`clear`, `liquid_water`,
  `supercooled`, `mixed`, `ice`) — continuous 24h day/twilight/night IR retrieval.
- `ABI-L2-ACHAC` (`10 km` CONUS): Enterprise AWG Cloud Top Height (`HT` in meters ASL)
  for physical altitude assignment and 3D geostationary parallax rectification.
- `ABI-L2-CODC` (`2 km` CONUS): Enterprise Cloud Optical Depth at `640 nm` (`COD`,
  seamless day visible + twilight/night IR optical depth).

Computes:
1. Exact WGS84 -> GOES-R Fixed Grid (`x`, `y` scan-angle radians) projection with
   layer-specific 3D Cloud-Top Parallax Correction (~8-11 km NNE shift for 10 km cirrus).
2. Sub-grid `3 x 3` (`2 km` -> `6 km`) fractional cloud cover across 3 causal frames
   (`t - 30 min`, `t - 15 min`, `t_0`), zero future leakage for historical backtesting.
3. Multi-scale `3 x 3` regional tile 2D FFT phase-correlation optical flow solved
   independently for Low (`< 2.8 km`) vs Mid/High (`>= 2.8 km`) cloud decks.
4. Continuity-based Orographic Stationary-Feature Pinning (`w_pin` in `[0, 0.88]`):
   identifies coastal marine stratus banks and Sierra lee-wave lenticulars whose edges
   remain stationary across 3 consecutive GOES scans despite non-zero ambient wind.
5. Damped Lagrangian cloud growth/dissipation tendency (`DC/Dt`) along back-trajectories.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    Image = None  # type: ignore
    HAS_PIL = False

try:
    import h5py  # type: ignore

    HAS_H5PY = True
except Exception:
    h5py = None  # type: ignore
    HAS_H5PY = False

ROOT_DIR = Path(__file__).resolve().parents[2]
GOES_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "goes"
USER_AGENT = "PhotoPlanner/3.0 (landscape-optics-corridor; github.com/lucasbrambrink/photoplanner)"

_MEM_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MEM_TTL_SECONDS = 300.0  # 5 minutes (matches GOES CONUS 5-minute rapid-scan cadence)
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


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


def _parse_goes_scan_times(key: str) -> tuple[dt.datetime, dt.datetime] | None:
    """Extracts `(start_utc, end_utc)` from a GOES-R S3 object key."""
    idx_s = key.find("_s20")
    idx_e = key.find("_e20")
    if idx_s == -1 or idx_e == -1 or len(key) < idx_e + 15:
        return None
    try:
        def _tok_dt(tok: str) -> dt.datetime:
            yr = int(tok[0:4])
            dy = int(tok[4:7])
            hh = int(tok[7:9])
            mm = int(tok[9:11])
            ss = int(tok[11:13])
            return dt.datetime(yr, 1, 1, hh, mm, ss, tzinfo=dt.timezone.utc) + dt.timedelta(days=dy - 1)

        return _tok_dt(key[idx_s + 2 : idx_s + 15]), _tok_dt(key[idx_e + 2 : idx_e + 15])
    except Exception:
        return None


def _read_cf_unsigned_array(ds: Any, raw: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Reinterprets NetCDF4/HDF5 integer arrays carrying CF `_Unsigned="true"` (common across
    NOAA GOES-R ABI L2 `HT`, `COD`, `CMI`, and `DQF` datasets) because `h5py` ignores the
    CF `_Unsigned` attribute and returns signed `int16`/`int8` (causing wraparound above
    32767, e.g. `HT > 10,000 m` ASL wrapping to `-9,000 m`).
    Returns `(unpacked_raw, fill_value)` in matching integer dtype.
    """
    unsigned_attr = ds.attrs.get("_Unsigned", "")
    if isinstance(unsigned_attr, bytes):
        unsigned_attr = unsigned_attr.decode("utf-8", errors="ignore")
    is_unsigned = str(unsigned_attr).strip().lower() in ("true", "1", "yes")

    fv_attr = ds.attrs.get("_FillValue")
    fv_raw = int(fv_attr[0]) if hasattr(fv_attr, "__len__") else (int(fv_attr) if fv_attr is not None else -1)

    if is_unsigned and np.issubdtype(raw.dtype, np.signedinteger):
        if raw.dtype == np.int16:
            return raw.view(np.uint16), fv_raw & 0xFFFF
        if raw.dtype == np.int8:
            return raw.view(np.uint8), fv_raw & 0xFF
        if raw.dtype == np.int32:
            return raw.view(np.uint32), fv_raw & 0xFFFFFFFF
    return raw, fv_raw


def _select_goes_bucket(is_west: bool, ref_utc: dt.datetime) -> tuple[str, str]:
    if is_west:
        # GOES-18 became operational GOES-West on 2023-01-04; GOES-17 served prior dates
        if ref_utc < dt.datetime(2023, 1, 4, tzinfo=dt.timezone.utc):
            return "noaa-goes17", "GOES-17 West"
        return "noaa-goes18", "GOES-18 West"
    # GOES-19 became operational GOES-East on 2025-04-07; GOES-16 served prior dates
    if ref_utc < dt.datetime(2025, 4, 8, tzinfo=dt.timezone.utc):
        return "noaa-goes16", "GOES-16 East"
    return "noaa-goes19", "GOES-19 East"


def _list_s3_product_keys_for_window(
    bucket: str,
    product: str,
    ref_utc: dt.datetime,
    lookback_minutes: int = 55,
) -> list[tuple[dt.datetime, dt.datetime, str]]:
    """Lists S3 keys for `product` covering `[ref_utc - lookback_minutes, ref_utc]`."""
    t_earliest = ref_utc - dt.timedelta(minutes=lookback_minutes)
    hours_to_check: list[tuple[int, int, int]] = []
    cur = t_earliest.replace(minute=0, second=0, microsecond=0)
    end_hr = ref_utc.replace(minute=0, second=0, microsecond=0)
    while cur <= end_hr:
        hours_to_check.append((cur.year, cur.timetuple().tm_yday, cur.hour))
        cur += dt.timedelta(hours=1)

    found: list[tuple[dt.datetime, dt.datetime, str]] = []
    for yr, doy, hh in hours_to_check:
        prefix = f"{product}/{yr}/{doy:03d}/{hh:02d}/"
        url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=5.5) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
            for item in root.findall("s3:Contents", _S3_NS):
                k_el = item.find("s3:Key", _S3_NS)
                if k_el is None or not k_el.text:
                    continue
                parsed = _parse_goes_scan_times(k_el.text)
                if parsed is not None:
                    found.append((parsed[0], parsed[1], k_el.text))
        except Exception:
            continue
    found.sort(key=lambda t: t[0])
    return found


def _pick_closest_causal_key(
    entries: list[tuple[dt.datetime, dt.datetime, str]],
    desired_utc: dt.datetime,
    cutoff_utc: dt.datetime,
    max_slop_minutes: float = 18.0,
) -> tuple[dt.datetime, str] | None:
    """Selects the scan closest to `desired_utc` whose scan end time `<= cutoff_utc`."""
    best: tuple[float, dt.datetime, str] | None = None
    for s_dt, e_dt, key in entries:
        if e_dt > cutoff_utc:
            continue
        mid_dt = s_dt + 0.5 * (e_dt - s_dt)
        diff_min = abs((mid_dt - desired_utc).total_seconds()) / 60.0
        if diff_min <= max_slop_minutes:
            if best is None or diff_min < best[0]:
                best = (diff_min, mid_dt, key)
    if best is None:
        return None
    return best[1], best[2]


def _download_s3_bytes(bucket: str, key: str, timeout: float = 12.0) -> bytes:
    """Downloads a GOES S3 object with local raw-file caching for small L2 products."""
    GOES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = key.rsplit("/", 1)[-1]
    raw_cache = GOES_CACHE_DIR / "s3_raw" / bucket / fname
    if raw_cache.exists() and raw_cache.stat().st_size > 1024:
        return raw_cache.read_bytes()

    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()

    # Persist compact L2 files (< 8 MB) on disk so repeat backtests & reloads are instant
    if len(data) < 8_500_000:
        try:
            raw_cache.parent.mkdir(parents=True, exist_ok=True)
            raw_cache.write_bytes(data)
        except Exception:
            pass
    return data


def _latlonh_to_goes_xy(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    h_asl_m: np.ndarray | float,
    proj_meta: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact GOES-R PUG Volume 3 geostationary projection from geodetic `(lat, lon, h_asl_m)`
    to ABI Fixed Grid scan angles `(x_rad, y_rad)` (`sweep_angle_axis = 'x'`).
    Including `h_asl_m > 0` corrects geostationary cloud-top parallax displacement!
    """
    r_eq = proj_meta["r_eq"]
    r_pol = proj_meta["r_pol"]
    h_sat = proj_meta["H"]
    lon0 = math.radians(proj_meta["lon0_deg"])
    e2 = 1.0 - (r_pol * r_pol) / (r_eq * r_eq)

    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    n_rad = r_eq / np.sqrt(1.0 - e2 * sin_lat * sin_lat)

    ox = (n_rad + h_asl_m) * cos_lat * np.cos(lon_r - lon0)
    oy = (n_rad + h_asl_m) * cos_lat * np.sin(lon_r - lon0)
    oz = (n_rad * (1.0 - e2) + h_asl_m) * sin_lat

    sx = h_sat - ox
    sy = -oy
    sz = oz
    norm_s = np.sqrt(sx * sx + sy * sy + sz * sz)
    y_rad = np.arctan2(sz, sx)
    x_rad = np.arcsin(np.clip(-sy / np.maximum(1.0, norm_s), -1.0, 1.0))
    return x_rad, y_rad


def _xy_to_fractional_indices(
    x_rad: np.ndarray,
    y_rad: np.ndarray,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maps scan-angle arrays `(x_rad, y_rad)` to continuous `(row_f, col_f, valid_mask)`."""
    x0, dx = float(x_vec[0]), float(x_vec[1] - x_vec[0])
    y0, dy = float(y_vec[0]), float(y_vec[1] - y_vec[0])
    col_f = (x_rad - x0) / dx
    row_f = (y_rad - y0) / dy
    h_n, w_n = len(y_vec), len(x_vec)
    valid = (col_f >= 1.0) & (col_f <= w_n - 2.0) & (row_f >= 1.0) & (row_f <= h_n - 2.0)
    return (
        np.clip(row_f, 1.0, h_n - 2.001),
        np.clip(col_f, 1.0, w_n - 2.001),
        valid,
    )


def _box3x3_smooth(arr: np.ndarray) -> np.ndarray:
    """Fast pure-NumPy 3x3 edge-replicated box filter."""
    p = np.pad(arr, 1, mode="edge")
    return (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:]
        + p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


def _solar_elevation_grid_utc(
    lat_grid: np.ndarray, lon_grid: np.ndarray, when_utc: dt.datetime
) -> np.ndarray:
    """Vectorized solar elevation angle (deg) across `(lat_grid, lon_grid)` at `when_utc`."""
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
    tst = np.mod(frac_hr * 60.0 + eqtime + 4.0 * lon_grid, 1440.0)
    ha = np.radians(tst / 4.0 - 180.0)
    lat_r = np.radians(lat_grid)
    cos_z = np.clip(
        np.sin(lat_r) * math.sin(decl) + np.cos(lat_r) * math.cos(decl) * np.cos(ha),
        -1.0,
        1.0,
    )
    return (90.0 - np.degrees(np.arccos(cos_z))).astype(np.float32)


def _decode_goes_l2_frame(
    actpc_bytes: bytes,
    achac_bytes: bytes | None,
    codc_bytes: bytes | None,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    is_coastal: bool,
    enable_parallax: bool = True,
    cclc_bytes: bytes | None = None,
    cmip_c07_bytes: bytes | None = None,
    cmip_c13_bytes: bytes | None = None,
    scan_utc: dt.datetime | None = None,
    enable_multilayer_fusion: bool = True,
) -> dict[str, Any]:
    """
    Decodes a single timestamp's GOES L2 `ACTPC` + `ACHAC` (+ optional `CODC`, 5-layer `CCLC`,
    and twilight `CMIPC C13 - C07` IR brightness-temperature difference) onto the
    `(grid_px, grid_px)` geographic corridor grid with 3D cloud-top parallax correction.
    """
    assert h5py is not None

    with h5py.File(io.BytesIO(actpc_bytes), "r") as f_p:
        p_attrs = f_p["goes_imager_projection"].attrs
        proj_meta = {
            "r_eq": float(p_attrs["semi_major_axis"][0]),
            "r_pol": float(p_attrs["semi_minor_axis"][0]),
            "H": float(p_attrs["semi_major_axis"][0] + p_attrs["perspective_point_height"][0]),
            "lon0_deg": float(p_attrs["longitude_of_projection_origin"][0]),
        }
        x_2km_all = f_p["x"][:].astype(np.float64) * float(f_p["x"].attrs["scale_factor"][0]) + float(
            f_p["x"].attrs["add_offset"][0]
        )
        y_2km_all = f_p["y"][:].astype(np.float64) * float(f_p["y"].attrs["scale_factor"][0]) + float(
            f_p["y"].attrs["add_offset"][0]
        )
        # Compute scan-angle envelope across the domain (0..16 km altitude margin) so h5py
        # only decompresses the local ~420x420 corridor window instead of all 1500x2500 CONUS pixels!
        corners_lat = np.array([lat_grid[0, 0], lat_grid[0, -1], lat_grid[-1, 0], lat_grid[-1, -1]])
        corners_lon = np.array([lon_grid[0, 0], lon_grid[0, -1], lon_grid[-1, 0], lon_grid[-1, -1]])
        xc0, yc0 = _latlonh_to_goes_xy(corners_lat, corners_lon, 0.0, proj_meta)
        xc1, yc1 = _latlonh_to_goes_xy(corners_lat, corners_lon, 16000.0, proj_meta)
        rf0, cf0, _ = _xy_to_fractional_indices(
            np.concatenate([xc0, xc1]), np.concatenate([yc0, yc1]), x_2km_all, y_2km_all
        )
        r2_min = max(0, int(np.floor(np.min(rf0))) - 8)
        r2_max = min(len(y_2km_all), int(np.ceil(np.max(rf0))) + 9)
        c2_min = max(0, int(np.floor(np.min(cf0))) - 8)
        c2_max = min(len(x_2km_all), int(np.ceil(np.max(cf0))) + 9)

        x_2km = x_2km_all[c2_min:c2_max]
        y_2km = y_2km_all[r2_min:r2_max]
        phase_full = f_p["Phase"][r2_min:r2_max, c2_min:c2_max]
        dqf_full = f_p["DQF"][r2_min:r2_max, c2_min:c2_max]

    default_low_h = 850.0 if is_coastal else 1850.0
    default_mid_h = 4300.0
    default_high_h = 9600.0

    ht_10km_full: np.ndarray | None = None
    x_10km: np.ndarray | None = None
    y_10km: np.ndarray | None = None

    if achac_bytes is not None:
        try:
            with h5py.File(io.BytesIO(achac_bytes), "r") as f_h:
                x_10km = f_h["x"][:].astype(np.float64) * float(
                    f_h["x"].attrs["scale_factor"][0]
                ) + float(f_h["x"].attrs["add_offset"][0])
                y_10km = f_h["y"][:].astype(np.float64) * float(
                    f_h["y"].attrs["scale_factor"][0]
                ) + float(f_h["y"].attrs["add_offset"][0])
                ht_raw, ht_fv = _read_cf_unsigned_array(f_h["HT"], f_h["HT"][:])
                ht_dqf, _ = _read_cf_unsigned_array(f_h["DQF"], f_h["DQF"][:])
                ht_sf = float(f_h["HT"].attrs["scale_factor"][0])
                ht_ao = float(f_h["HT"].attrs["add_offset"][0])
                ht_10km_full = np.where(
                    (ht_raw != ht_fv) & ((ht_dqf & 3) == 0),
                    ht_raw.astype(np.float32) * ht_sf + ht_ao,
                    np.nan,
                )
        except Exception:
            ht_10km_full = None

    # Sample cloud-top height onto target grid (with 1 parallax pass if enabled)
    if ht_10km_full is not None and x_10km is not None and y_10km is not None:
        x_h0, y_h0 = _latlonh_to_goes_xy(
            lat_grid, lon_grid, 3000.0 if enable_parallax else 0.0, proj_meta
        )
        rf_h, cf_h, val_h0 = _xy_to_fractional_indices(x_h0, y_h0, x_10km, y_10km)
        ri_h = np.rint(rf_h).astype(np.int32)
        ci_h = np.rint(cf_h).astype(np.int32)
        ht_pass1 = np.where(val_h0, ht_10km_full[ri_h, ci_h], np.nan)
        if enable_parallax:
            h_est = np.where(np.isfinite(ht_pass1), np.clip(ht_pass1, 0.0, 16500.0), 2500.0)
            x_h1, y_h1 = _latlonh_to_goes_xy(lat_grid, lon_grid, h_est, proj_meta)
            rf_h, cf_h, val_h1 = _xy_to_fractional_indices(x_h1, y_h1, x_10km, y_10km)
            ht_grid = np.where(
                val_h1,
                ht_10km_full[np.rint(rf_h).astype(np.int32), np.rint(cf_h).astype(np.int32)],
                np.nan,
            )
        else:
            ht_grid = ht_pass1

        low_top_m = np.where(
            np.isfinite(ht_grid) & (ht_grid < 2900.0),
            np.clip(ht_grid, 180.0, 2850.0),
            default_low_h,
        ).astype(np.float32)
        mid_top_m = np.where(
            np.isfinite(ht_grid) & (ht_grid >= 2600.0) & (ht_grid < 6800.0),
            ht_grid,
            default_mid_h,
        ).astype(np.float32)
        high_top_m = np.where(
            np.isfinite(ht_grid) & (ht_grid >= 6500.0),
            np.clip(ht_grid, 6500.0, 15500.0),
            default_high_h,
        ).astype(np.float32)
    else:
        ht_grid = np.full_like(lat_grid, np.nan, dtype=np.float32)
        low_top_m = np.full_like(lat_grid, default_low_h, dtype=np.float32)
        mid_top_m = np.full_like(lat_grid, default_mid_h, dtype=np.float32)
        high_top_m = np.full_like(lat_grid, default_high_h, dtype=np.float32)

    cod_full: np.ndarray | None = None
    if codc_bytes is not None:
        try:
            with h5py.File(io.BytesIO(codc_bytes), "r") as f_c:
                cod_raw, cod_fv = _read_cf_unsigned_array(
                    f_c["COD"], f_c["COD"][r2_min:r2_max, c2_min:c2_max]
                )
                cod_sf = float(f_c["COD"].attrs["scale_factor"][0])
                cod_ao = float(f_c["COD"].attrs["add_offset"][0])
                cod_full = np.where(
                    cod_raw != cod_fv,
                    np.clip(cod_raw.astype(np.float32) * cod_sf + cod_ao, 0.0, 65.0),
                    np.nan,
                )
        except Exception:
            cod_full = None

    # High-cordillera snowpack/high-terrain mask (Sierra Nevada crest > ~2.4 km ASL)
    alpine_snow_zone = (lat_grid >= 36.0) & (lat_grid <= 41.5) & (lon_grid >= -120.25) & (lon_grid <= -118.1)

    def _sample_layer_3x3(
        h_layer_m: np.ndarray | float,
        layer_kind: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_eff = h_layer_m if enable_parallax else 0.0
        xl, yl = _latlonh_to_goes_xy(lat_grid, lon_grid, h_eff, proj_meta)
        r_f, c_f, valid_m = _xy_to_fractional_indices(xl, yl, x_2km, y_2km)
        r0 = np.rint(r_f).astype(np.int32)
        c0 = np.rint(c_f).astype(np.int32)

        cf_acc = np.zeros_like(lat_grid, dtype=np.float32)
        cod_acc = np.zeros_like(lat_grid, dtype=np.float32)
        cod_cnt = np.zeros_like(lat_grid, dtype=np.float32)

        # Also lookup local cloud-top height at the projected pixel if ACHAC is available
        if ht_10km_full is not None and x_10km is not None and y_10km is not None:
            rh_f, ch_f, _ = _xy_to_fractional_indices(xl, yl, x_10km, y_10km)
            ht_loc = ht_10km_full[np.rint(rh_f).astype(np.int32), np.rint(ch_f).astype(np.int32)]
        else:
            ht_loc = ht_grid

        has_ht = np.isfinite(ht_loc)

        for dr in (-1, 0, 1):
            rr = r0 + dr
            for dc in (-1, 0, 1):
                cc = c0 + dc
                ph = phase_full[rr, cc]
                good_qf = (dqf_full[rr, cc] & 1) == 0
                if layer_kind == "low":
                    # Require NOAA overall_good_quality_qf (`(dqf & 1) == 0`) so clear-sky / twilight
                    # surface-restoral `DQF == 5` pixels are never misclassified as low liquid stratus!
                    is_hit = good_qf & (
                        ((ph == 1) & (~has_ht | (ht_loc < 2900.0)))
                        | ((ph == 2) & has_ht & (~alpine_snow_zone) & (ht_loc < 2300.0))
                        | ((ph == 5) & has_ht & (~alpine_snow_zone) & (ht_loc < 2600.0))
                    )
                elif layer_kind == "mid":
                    min_mid_ht = np.where(alpine_snow_zone, 4150.0, 2300.0)
                    is_hit = (
                        good_qf
                        & (
                            ((ph == 2) & ((~has_ht & ~alpine_snow_zone) | (has_ht & (ht_loc >= min_mid_ht) & (ht_loc < 6800.0))))
                            | ((ph == 3) & ((~has_ht & ~alpine_snow_zone) | (has_ht & (ht_loc >= min_mid_ht) & (ht_loc < 6800.0))))
                            | ((ph == 1) & has_ht & (ht_loc >= np.maximum(2900.0, min_mid_ht)) & (ht_loc < 6800.0))
                            | ((ph == 4) & has_ht & (ht_loc >= np.maximum(2800.0, min_mid_ht)) & (ht_loc < 6000.0))
                            | ((ph == 5) & has_ht & (ht_loc >= np.maximum(2600.0, min_mid_ht)) & (ht_loc < 6600.0))
                        )
                    )
                else:
                    # Ice cirrus / anvil (`Phase 4` with good QF, or valid ACHA cloud top >= 6.6 km ASL
                    # which captures optically thin sunset cirrus even when IR beta-ratio QF is weak)
                    is_hit = (
                        ((ph == 4) & good_qf & (~has_ht | (ht_loc >= 6000.0)))
                        | ((ph >= 1) & (ph <= 5) & has_ht & (ht_loc >= 6600.0))
                    )

                hit_f = (is_hit & valid_m).astype(np.float32)
                cf_acc += hit_f
                if cod_full is not None:
                    cv = cod_full[rr, cc]
                    ok_c = is_hit & valid_m & np.isfinite(cv)
                    cod_acc += np.where(ok_c, cv, 0.0)
                    cod_cnt += ok_c.astype(np.float32)

        cf_layer = np.clip(cf_acc / 9.0, 0.0, 1.0)
        if layer_kind == "low":
            default_incloud_tau = 6.8
        elif layer_kind == "mid":
            default_incloud_tau = 5.2
        else:
            default_incloud_tau = 1.15

        incloud_tau = np.where(cod_cnt > 0.5, cod_acc / np.maximum(1.0, cod_cnt), default_incloud_tau)
        tau_layer = np.clip(cf_layer * incloud_tau, 0.0, 48.0).astype(np.float32)
        return cf_layer.astype(np.float32), tau_layer

    actp_low_cf, tau_low = _sample_layer_3x3(low_top_m, "low")
    actp_mid_cf, tau_mid = _sample_layer_3x3(mid_top_m, "mid")
    actp_high_cf, tau_high = _sample_layer_3x3(high_top_m, "high")

    low_cf = actp_low_cf.copy()
    mid_cf = actp_mid_cf.copy()
    high_cf = actp_high_cf.copy()
    fog_cf = np.where(low_top_m < 1450.0, actp_low_cf * 0.85, actp_low_cf * 0.35).astype(np.float32)
    has_cclc = False
    has_btd_fog = False

    # --- Multi-Layer Upgrade 1: NOAA ABI-L2-CCLC (5-Band Cloud Cover Layers: CF1..CF5) ---
    # Recovers sub-cirrus marine stratus (`CF1`: 0-5 kft / 0-1,524 m ASL, `CF2`: 5-10 kft) and
    # mid-level altocumulus (`CF3`: 10-18 kft) obscured from single-layer top-down `ACTPC`!
    if enable_multilayer_fusion and cclc_bytes is not None:
        try:
            with h5py.File(io.BytesIO(cclc_bytes), "r") as f_cc:
                xc_10 = f_cc["x"][:].astype(np.float64) * float(
                    f_cc["x"].attrs["scale_factor"][0]
                ) + float(f_cc["x"].attrs["add_offset"][0])
                yc_10 = f_cc["y"][:].astype(np.float64) * float(
                    f_cc["y"].attrs["scale_factor"][0]
                ) + float(f_cc["y"].attrs["add_offset"][0])
                cc_dqf, _ = _read_cf_unsigned_array(f_cc["DQF"], f_cc["DQF"][:])
                ok_cc = (cc_dqf >= 0) & (cc_dqf <= 1)

                def _proj_cclc_band(ds_name: str, h_band_m: float) -> np.ndarray:
                    raw_b, fv_b = _read_cf_unsigned_array(f_cc[ds_name], f_cc[ds_name][:])
                    sf_attr = f_cc[ds_name].attrs.get("scale_factor")
                    ao_attr = f_cc[ds_name].attrs.get("add_offset")
                    sf_b = float(sf_attr[0]) if hasattr(sf_attr, "__len__") else (float(sf_attr) if sf_attr is not None else 1.0)
                    ao_b = float(ao_attr[0]) if hasattr(ao_attr, "__len__") else (float(ao_attr) if ao_attr is not None else 0.0)
                    arr_pct = np.where(
                        (raw_b != fv_b) & ok_cc,
                        np.clip(raw_b.astype(np.float32) * sf_b + ao_b, 0.0, 100.0) / 100.0,
                        np.nan,
                    )
                    xb, yb = _latlonh_to_goes_xy(
                        lat_grid, lon_grid, h_band_m if enable_parallax else 0.0, proj_meta
                    )
                    rf_b, cf_b, val_b = _xy_to_fractional_indices(xb, yb, xc_10, yc_10)
                    sampled = arr_pct[np.rint(rf_b).astype(np.int32), np.rint(cf_b).astype(np.int32)]
                    return np.where(val_b & np.isfinite(sampled), sampled, 0.0).astype(np.float32)

                cf1_0_5kft = _proj_cclc_band("CF1", 650.0)     # 0 - 1,524 m ASL (marine fog/stratus)
                cf2_5_10kft = _proj_cclc_band("CF2", 2200.0)   # 1,524 - 3,048 m ASL (cumulus / lower deck)
                cf3_10_18kft = _proj_cclc_band("CF3", 4200.0)  # 3,048 - 5,486 m ASL (altocumulus / altostratus)
                cf4_18_24kft = _proj_cclc_band("CF4", 6300.0)  # 5,486 - 7,315 m ASL (upper-mid / low cirrus)
                cf5_gt24kft = _proj_cclc_band("CF5", 9800.0)   # > 7,315 m ASL (high ice cirrus)

                # Mask CF1 (< 1,524 m ASL) and CF2 (< 3,048 m ASL) over high Sierra granite/snow
                cf1_eff = np.where(alpine_snow_zone, 0.0, cf1_0_5kft)
                cf2_eff = np.where(alpine_snow_zone, 0.0, cf2_5_10kft)
                cclc_low = np.clip(np.maximum(cf1_eff, 0.88 * cf2_eff), 0.0, 1.0)
                cclc_mid = np.clip(np.maximum(cf3_10_18kft, 0.62 * cf4_18_24kft), 0.0, 1.0)
                cclc_high = np.clip(np.maximum(cf5_gt24kft, 0.65 * cf4_18_24kft), 0.0, 1.0)

                # Blend 2 km sharp-edged ACTPC with 10 km multi-layer CCLC:
                # Where upper decks obscure ACTPC (`high > 0.25` or `mid > 0.30`), trust CCLC's
                # multi-layer retrieval to fill in obscured lower/middle cloud decks!
                obscured_low = np.clip((actp_high_cf + 0.8 * actp_mid_cf) / 0.45, 0.0, 1.0)
                low_cf = np.clip(
                    np.maximum(
                        actp_low_cf,
                        cclc_low * (0.58 + 0.42 * obscured_low),
                    ),
                    0.0,
                    1.0,
                ).astype(np.float32)
                obscured_mid = np.clip(actp_high_cf / 0.45, 0.0, 1.0)
                mid_cf = np.clip(
                    np.maximum(
                        actp_mid_cf,
                        cclc_mid * (0.55 + 0.45 * obscured_mid),
                    ),
                    0.0,
                    1.0,
                ).astype(np.float32)
                high_cf = np.clip(np.maximum(actp_high_cf, 0.88 * cclc_high), 0.0, 1.0).astype(np.float32)
                fog_cf = np.clip(np.maximum(fog_cf, cf1_eff * 0.94), 0.0, 1.0).astype(np.float32)
                has_cclc = True
        except Exception:
            has_cclc = False

    # --- Multi-Layer Upgrade 2: Twilight/Night `10.3 um - 3.9 um` (`C13 - C07`) Fog/Stratus BTD ---
    # Solves sunset/sunrise terminator blindness (`sun_el <= 8.5 deg`) when visible `CODC` drops out
    # and `ACTPC` surface-restoral (`DQF == 5`) erases shallow 200-600 m coastal stratus!
    tb13_grid: np.ndarray | None = None
    if (
        enable_multilayer_fusion
        and cmip_c07_bytes is not None
        and cmip_c13_bytes is not None
        and scan_utc is not None
    ):
        try:
            with h5py.File(io.BytesIO(cmip_c07_bytes), "r") as f7, h5py.File(
                io.BytesIO(cmip_c13_bytes), "r"
            ) as f13:
                r7, fv7 = _read_cf_unsigned_array(f7["CMI"], f7["CMI"][r2_min:r2_max, c2_min:c2_max])
                sf7 = float(f7["CMI"].attrs["scale_factor"][0])
                ao7 = float(f7["CMI"].attrs["add_offset"][0])
                tb07_2km = np.where(r7 != fv7, r7.astype(np.float32) * sf7 + ao7, np.nan)

                r13, fv13 = _read_cf_unsigned_array(f13["CMI"], f13["CMI"][r2_min:r2_max, c2_min:c2_max])
                sf13 = float(f13["CMI"].attrs["scale_factor"][0])
                ao13 = float(f13["CMI"].attrs["add_offset"][0])
                tb13_2km = np.where(r13 != fv13, r13.astype(np.float32) * sf13 + ao13, np.nan)

                xl_b, yl_b = _latlonh_to_goes_xy(
                    lat_grid, lon_grid, 550.0 if enable_parallax else 0.0, proj_meta
                )
                rf_b, cf_b, val_b = _xy_to_fractional_indices(xl_b, yl_b, x_2km, y_2km)
                ri_b = np.rint(rf_b).astype(np.int32)
                ci_b = np.rint(cf_b).astype(np.int32)

                tb07_px = tb07_2km[ri_b, ci_b]
                tb13_px = tb13_2km[ri_b, ci_b]
                tb13_grid = np.where(np.isfinite(tb13_px), tb13_px, np.nan).astype(np.float32)

                sun_el_px = _solar_elevation_grid_utc(lat_grid, lon_grid, scan_utc)
                # Warm boundary-layer top (`270 K .. 290 K`), excluding cold mid/high ice clouds (`< 266 K`)
                warm_bl_top = (tb13_px >= 270.0) & (tb13_px <= 290.5) & (~alpine_snow_zone) & val_b

                # 1) Dark/civil-twilight emissive droplet regime (`sun_el <= 1.2 deg`):
                # Liquid stratus has lower 3.9 um emissivity than 10.3 um -> `tb13 - tb07 >= +1.6 K` over ocean/coast.
                # Over inland continental soil/prairie, clear silicate land has natural `+1.5..+3.0 K` BTD
                # at night, so NOAA operational FLS requires `>= +3.3 K` inland to prevent clear-sky false fog.
                btd_emissive = tb13_px - tb07_px
                btd_thresh = 1.55 if is_coastal else 3.30
                btd_span = 2.55 if is_coastal else 2.80
                night_fog = np.where(
                    warm_bl_top & (sun_el_px <= 1.2),
                    np.clip((btd_emissive - btd_thresh) / btd_span, 0.0, 1.0),
                    0.0,
                )
                # 2) Grazing low-sun twilight regime (`1.2 < sun_el <= 7.5 deg` over cool marine sector):
                # Small liquid stratus droplets strongly scatter 3.9 um (`tb07 - tb13 >= +6.5 K`) while
                # `tb13` stays cool-marine (`271..285.5 K`, ruling out sun-heated inland soil)
                ocean_or_coast = lon_grid <= -121.65 if is_coastal else np.zeros_like(warm_bl_top)
                grazing_fog = np.where(
                    warm_bl_top & ocean_or_coast & (tb13_px <= 285.5) & (sun_el_px > 1.2) & (sun_el_px <= 7.5),
                    np.clip(((tb07_px - tb13_px) - 6.5) / 9.5, 0.0, 0.92),
                    0.0,
                )
                btd_fog = _box3x3_smooth(np.maximum(night_fog, grazing_fog)).astype(np.float32)
                low_cf = np.clip(np.maximum(low_cf, btd_fog), 0.0, 1.0).astype(np.float32)
                fog_cf = np.clip(np.maximum(fog_cf, btd_fog * 0.95), 0.0, 1.0).astype(np.float32)
                has_btd_fog = True
        except Exception:
            has_btd_fog = False

    # Preserve real GOES `CODC` optical depths on the `ACTPC`-detected fraction (especially
    # optically thin golden-hour cirrus `tau ~ 0.2..0.8`), and only add climatological optical
    # depth for the newly uncovered cloud fraction contributed by `CCLC` or twilight `BTD`.
    delta_low_cf = np.maximum(0.0, low_cf - actp_low_cf)
    delta_mid_cf = np.maximum(0.0, mid_cf - actp_mid_cf)
    delta_high_cf = np.maximum(0.0, high_cf - actp_high_cf)
    tau_low = np.clip(tau_low + delta_low_cf * 6.4, 0.0, 48.0).astype(np.float32)
    tau_mid = np.clip(tau_mid + delta_mid_cf * 4.8, 0.0, 48.0).astype(np.float32)
    tau_high = np.clip(tau_high + delta_high_cf * 0.95, 0.0, 48.0).astype(np.float32)
    tau_total = np.clip(tau_low + tau_mid + tau_high, 0.0, 52.0).astype(np.float32)

    if tb13_grid is not None and np.any(np.isfinite(tb13_grid)):
        tb_k_proxy = np.where(
            np.isfinite(tb13_grid),
            tb13_grid,
            288.0 - 0.0062 * high_top_m * high_cf,
        ).astype(np.float32)
    else:
        tb_k_proxy = np.where(
            high_cf > 0.20,
            288.0 - 0.0062 * high_top_m,
            np.where(mid_cf > 0.20, 288.0 - 0.0060 * mid_top_m, 290.0 - 0.0055 * low_top_m * low_cf),
        ).astype(np.float32)

    return {
        "fog_cf": fog_cf,
        "low_cf": low_cf,
        "mid_cf": mid_cf,
        "high_cf": high_cf,
        "actp_low_cf": actp_low_cf,
        "actp_mid_cf": actp_mid_cf,
        "actp_high_cf": actp_high_cf,
        "has_cclc": has_cclc,
        "has_btd_fog": has_btd_fog,
        "tau_low": tau_low,
        "tau_mid": tau_mid,
        "tau_high": tau_high,
        "tau_total": tau_total,
        "low_top_m": low_top_m,
        "mid_top_m": mid_top_m,
        "high_top_m": high_top_m,
        "tb_k": tb_k_proxy,
    }


def _estimate_optical_flow_fft_ms(
    frame_prev: np.ndarray,
    frame_now: np.ndarray,
    dx_m: float,
    dy_m: float,
    dt_seconds: float,
    min_std: float = 0.045,
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
    if np.std(p0) < min_std or np.std(p1) < min_std:
        return 0.0, 0.0, 0.04

    if float(np.max(np.abs(frame_now - frame_prev))) < 1e-4:
        return 0.0, 0.0, 0.0

    h, w = p0.shape
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]

    f0 = np.fft.fft2(p0 * window)
    f1 = np.fft.fft2(p1 * window)
    cross = f0 * np.conj(f1)
    corr = np.real(np.fft.ifft2(cross / (np.abs(cross) + 1e-8)))
    corr_shifted = np.fft.fftshift(corr)

    cy, cx = h // 2, w // 2
    max_shift_x = max(2, min(cx - 2, int(math.ceil(52.0 * dt_seconds / max(200.0, dx_m)))))
    max_shift_y = max(2, min(cy - 2, int(math.ceil(52.0 * dt_seconds / max(200.0, dy_m)))))

    sub = corr_shifted[cy - max_shift_y : cy + max_shift_y + 1, cx - max_shift_x : cx + max_shift_x + 1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    peak_val = float(sub[iy, ix])
    med_val = float(np.median(np.abs(sub))) + 1e-6

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

    shift_cols_east = -((ix - max_shift_x) + sub_dx)
    shift_rows_south = -((iy - max_shift_y) + sub_dy)

    u_east_ms = (shift_cols_east * dx_m) / dt_seconds
    v_north_ms = (-shift_rows_south * dy_m) / dt_seconds
    snr = peak_val / med_val
    conf = max(0.0, min(0.95, (snr - 2.7) / 8.2))
    return round(u_east_ms, 2), round(v_north_ms, 2), round(conf, 3)


def _compute_regional_flow_field(
    f_m30: np.ndarray,
    f_m15: np.ndarray,
    f_0: np.ndarray,
    dx_m: float,
    dy_m: float,
    dt_30_0_s: float,
    dt_30_15_s: float,
    dt_15_0_s: float,
    enable_multiscale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Computes spatially varying `(u_grid, v_grid, conf_grid)` across a `3 x 3` overlapping
    tile layout (`64 x 64` patches over `128 x 128`) with 3-scan temporal consistency checking.
    """
    h, w = f_0.shape
    u_glob, v_glob, c_glob = _estimate_optical_flow_fft_ms(
        f_m30, f_0, dx_m, dy_m, dt_30_0_s, min_std=0.04
    )
    if not enable_multiscale or h < 96 or w < 96:
        return (
            np.full((h, w), u_glob, dtype=np.float32),
            np.full((h, w), v_glob, dtype=np.float32),
            np.full((h, w), c_glob, dtype=np.float32),
            u_glob,
            v_glob,
            c_glob,
        )

    patch = 64
    starts = [0, (h - patch) // 2, h - patch]
    u_tiles = np.zeros((3, 3), dtype=np.float32)
    v_tiles = np.zeros((3, 3), dtype=np.float32)
    c_tiles = np.zeros((3, 3), dtype=np.float32)

    for ty_i, y0 in enumerate(starts):
        for tx_i, x0 in enumerate(starts):
            s30 = f_m30[y0 : y0 + patch, x0 : x0 + patch]
            s15 = f_m15[y0 : y0 + patch, x0 : x0 + patch]
            s0 = f_0[y0 : y0 + patch, x0 : x0 + patch]

            u_long, v_long, c_long = _estimate_optical_flow_fft_ms(
                s30, s0, dx_m, dy_m, dt_30_0_s, min_std=0.045
            )
            u_a, v_a, c_a = _estimate_optical_flow_fft_ms(
                s30, s15, dx_m, dy_m, dt_30_15_s, min_std=0.045
            )
            u_b, v_b, c_b = _estimate_optical_flow_fft_ms(
                s15, s0, dx_m, dy_m, dt_15_0_s, min_std=0.045
            )

            if c_long > 0.12 and (c_a > 0.10 or c_b > 0.10):
                u_step = 0.5 * (u_a + u_b) if (c_a > 0.10 and c_b > 0.10) else (u_b if c_b >= c_a else u_a)
                v_step = 0.5 * (v_a + v_b) if (c_a > 0.10 and c_b > 0.10) else (v_b if c_b >= c_a else v_a)
                discrep = math.hypot(u_long - u_step, v_long - v_step)
                consistency = max(0.25, min(1.12, 1.12 - discrep / 14.0))
                u_tiles[ty_i, tx_i] = 0.70 * u_long + 0.30 * u_step
                v_tiles[ty_i, tx_i] = 0.70 * v_long + 0.30 * v_step
                c_tiles[ty_i, tx_i] = min(0.95, c_long * consistency)
            elif c_long > 0.12:
                u_tiles[ty_i, tx_i] = u_long
                v_tiles[ty_i, tx_i] = v_long
                c_tiles[ty_i, tx_i] = c_long * 0.85
            else:
                u_tiles[ty_i, tx_i] = u_glob
                v_tiles[ty_i, tx_i] = v_glob
                c_tiles[ty_i, tx_i] = c_glob * 0.55

    # Bilinear upsample from 3x3 tile centers (at 32, 64, 96) to (h, w)
    centers = np.array([32.0, 64.0, 96.0], dtype=np.float32)
    ys = np.clip((np.arange(h, dtype=np.float32) - centers[0]) / 32.0, 0.0, 1.999)
    xs = np.clip((np.arange(w, dtype=np.float32) - centers[0]) / 32.0, 0.0, 1.999)
    yi = ys.astype(np.int32)[:, None]
    xi = xs.astype(np.int32)[None, :]
    wy = (ys - ys.astype(np.int32))[:, None]
    wx = (xs - xs.astype(np.int32))[None, :]

    def _bilerp_3x3(tile3: np.ndarray) -> np.ndarray:
        return (
            (tile3[yi, xi] * (1.0 - wx) + tile3[yi, xi + 1] * wx) * (1.0 - wy)
            + (tile3[yi + 1, xi] * (1.0 - wx) + tile3[yi + 1, xi + 1] * wx) * wy
        ).astype(np.float32)

    u_grid = _bilerp_3x3(u_tiles)
    v_grid = _bilerp_3x3(v_tiles)
    c_grid = _bilerp_3x3(c_tiles)
    return (
        u_grid,
        v_grid,
        c_grid,
        round(float(u_tiles[1, 1]), 2),
        round(float(v_tiles[1, 1]), 2),
        round(float(c_tiles[1, 1]), 3),
    )


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


def _fetch_iem_wms_fallback(
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float],
    grid_px: int,
    surface_temp_c: float,
    is_coastal: bool,
    is_west: bool,
    now_utc: dt.datetime,
) -> dict[str, Any] | None:
    """Zero-h5py fallback using current IEM GOES WMS imagery when S3/h5py is unavailable."""
    cgi_name = "goes_west" if is_west else "goes_east"
    min_lon, min_lat, max_lon, max_lat = bbox

    def _wms_layer(layer: str) -> np.ndarray:
        url = (
            f"https://mesonet.agron.iastate.edu/cgi-bin/wms/{cgi_name}.cgi"
            f"?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}"
            f"&SRS=EPSG:4326&BBOX={min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}"
            f"&WIDTH={grid_px}&HEIGHT={grid_px}&FORMAT=image/png"
        )
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=5.5) as resp:
            raw = resp.read()
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32)[:, :, 0]

    ch02 = _wms_layer("conus_ch02")
    ch07 = _wms_layer("conus_ch07")
    ch11 = _wms_layer("conus_ch11")
    sun_el = _solar_elevation_utc(lat, lon, now_utc)
    tb_k = 192.0 + 0.50 * ch11
    dt_top_k = np.maximum(0.0, (273.15 + surface_temp_c) - tb_k)
    has_vis = sun_el > 5.0

    if has_vis:
        lons_1d = np.linspace(bbox[0], bbox[2], grid_px, dtype=np.float32)[None, :]
        if is_coastal:
            ocean_m = np.clip((-122.45 - lons_1d) / 0.25, 0.0, 1.0)
            clear_bg = 104.0 * (1.0 - ocean_m) + 66.0 * ocean_m
        else:
            clear_bg = np.full((1, grid_px), 114.0, dtype=np.float32)
        vis_ex = np.clip((ch02 - clear_bg) / (238.0 - clear_bg), 0.0, 1.0)
        mu0 = max(0.15, math.sin(math.radians(sun_el)))
        refl = np.clip(vis_ex * (0.85 + 0.15 * mu0), 0.0, 0.94)
        tau_tot = np.clip((refl / np.maximum(0.06, 1.0 - refl)) * 4.8, 0.0, 48.0)
    else:
        vis_ex = np.zeros_like(ch11)
        tau_tot = np.zeros_like(ch11)

    swir_fog = np.clip((ch07 - 0.31 * ch11 - 6.0) / 26.0, 0.0, 1.0)
    high_ir = np.clip((250.0 - tb_k) / 24.0, 0.0, 1.0)
    mid_ir = np.clip((dt_top_k - 13.5) / 14.0, 0.0, 1.0) * (1.0 - 0.75 * high_ir)

    if has_vis:
        low_cf = np.clip(np.maximum(vis_ex * (1.0 - 0.65 * high_ir - 0.50 * mid_ir), swir_fog * 0.85), 0.0, 1.0)
        mid_cf = np.clip(mid_ir * (0.45 + 0.55 * vis_ex), 0.0, 1.0)
        high_cf = np.clip(np.maximum(high_ir, vis_ex * 0.25 * (dt_top_k / 28.0)), 0.0, 1.0)
    else:
        low_cf = np.clip(np.maximum(swir_fog, np.clip((dt_top_k - 3.5) / 11.0, 0.0, 1.0) * (1.0 - mid_ir - high_ir)), 0.0, 1.0)
        mid_cf = np.clip(mid_ir, 0.0, 1.0)
        high_cf = np.clip(high_ir, 0.0, 1.0)
        tau_tot = np.clip(low_cf * 7.5 + mid_cf * 5.6 + high_cf * 1.85, 0.0, 42.0)

    zeros = np.zeros_like(low_cf, dtype=np.float32)
    return {
        "satellite": f"NOAA {'GOES-18 West' if is_west else 'GOES-19 East'} (IEM WMS Snapshot)",
        "is_west": is_west,
        "is_calibrated_l2": False,
        "scan_timestamp_utc": (now_utc - dt.timedelta(minutes=7)).isoformat(),
        "scan_age_minutes": 7.0,
        "history_span_minutes": 0.0,
        "sun_el_at_scan_deg": round(sun_el, 2),
        "has_daylight_visible": has_vis,
        "bbox": bbox,
        "grid_px": grid_px,
        "motion_u_ms": 0.0,
        "motion_v_ms": 0.0,
        "motion_confidence": 0.0,
        "low_motion_uv_ms": [0.0, 0.0],
        "high_motion_uv_ms": [0.0, 0.0],
        "u_low_grid": zeros,
        "v_low_grid": zeros,
        "conf_low_grid": zeros,
        "u_high_grid": zeros,
        "v_high_grid": zeros,
        "conf_high_grid": zeros,
        "low_cf_grid": low_cf,
        "mid_cf_grid": mid_cf,
        "high_cf_grid": high_cf,
        "low_m30_grid": low_cf,
        "mid_m30_grid": mid_cf,
        "high_m30_grid": high_cf,
        "euler_low_grid": zeros,
        "euler_high_grid": zeros,
        "tau_low_grid": (low_cf * 7.4).astype(np.float32),
        "tau_mid_grid": (mid_cf * 5.6).astype(np.float32),
        "tau_high_grid": (high_cf * 1.85).astype(np.float32),
        "tau_grid": tau_tot,
        "low_top_m_grid": np.full_like(low_cf, 850.0 if is_coastal else 1850.0),
        "mid_top_m_grid": np.full_like(low_cf, 4300.0),
        "high_top_m_grid": np.full_like(low_cf, 9600.0),
        "tb_k_grid": tb_k,
    }


def _list_cmipc_c07_c13_keys(
    bucket: str, ref_utc: dt.datetime, lookback_minutes: int = 55
) -> tuple[list[tuple[dt.datetime, dt.datetime, str]], list[tuple[dt.datetime, dt.datetime, str]]]:
    """Lists `ABI-L2-CMIPC` Channel 07 (`3.9 um`) and Channel 13 (`10.3 um`) keys across any ABI scan mode."""
    t_earliest = ref_utc - dt.timedelta(minutes=lookback_minutes)
    cur = t_earliest.replace(minute=0, second=0, microsecond=0)
    end_hr = ref_utc.replace(minute=0, second=0, microsecond=0)
    c07_list: list[tuple[dt.datetime, dt.datetime, str]] = []
    c13_list: list[tuple[dt.datetime, dt.datetime, str]] = []
    while cur <= end_hr:
        yr, doy, hh = cur.year, cur.timetuple().tm_yday, cur.hour
        prefix = f"ABI-L2-CMIPC/{yr}/{doy:03d}/{hh:02d}/OR_ABI-L2-CMIPC-M"
        url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=5.5) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
            for item in root.findall("s3:Contents", _S3_NS):
                k_el = item.find("s3:Key", _S3_NS)
                if k_el is None or not k_el.text:
                    continue
                key_txt = k_el.text
                if "C07_" in key_txt:
                    parsed = _parse_goes_scan_times(key_txt)
                    if parsed is not None:
                        c07_list.append((parsed[0], parsed[1], key_txt))
                elif "C13_" in key_txt:
                    parsed = _parse_goes_scan_times(key_txt)
                    if parsed is not None:
                        c13_list.append((parsed[0], parsed[1], key_txt))
        except Exception:
            pass
        cur += dt.timedelta(hours=1)
    c07_list.sort(key=lambda t: t[0])
    c13_list.sort(key=lambda t: t[0])
    return c07_list, c13_list


def fetch_goes_corridor_nowcast(
    lat: float,
    lon: float,
    surface_temp_c: float = 15.0,
    is_coastal: bool = False,
    target_utc: dt.datetime | None = None,
    enable_parallax: bool = True,
    enable_multiscale_flow: bool = True,
    enable_multilayer_fusion: bool = True,
) -> dict[str, Any] | None:
    """
    Fetches 3-frame (`t - 30m`, `t - 15m`, `t_0`) calibrated NOAA GOES-18/19 S3 Level-2
    products (`ACTPC` + `ACHAC` + `CODC` + 5-layer `CCLC` + twilight `CMIPC C13 - C07` fog BTD)
    over a `768 km x 768 km` (`128 x 128`, `6.0 km/px`) domain around `(lat, lon)`.

    When `target_utc` is provided (e.g., historical replay or backtesting), strictly enforces
    causal scan selection (`scan_end_utc <= target_utc`) with zero future data leakage.
    """
    if not (21.0 <= lat <= 53.0 and -138.0 <= lon <= -65.0):
        return None

    is_live_mode = target_utc is None
    ref_utc = (
        dt.datetime.now(dt.timezone.utc)
        if target_utc is None
        else (
            target_utc.replace(tzinfo=dt.timezone.utc)
            if target_utc.tzinfo is None
            else target_utc.astimezone(dt.timezone.utc)
        )
    )

    time_bucket_str = (
        f"live_{int(time.time() // 300)}"
        if is_live_mode
        else ref_utc.strftime("%Y%m%d_%H%M")
    )
    cache_key = (
        f"goes_v7_{lat:.2f}_{lon:.2f}_{time_bucket_str}_"
        f"p{int(enable_parallax)}_m{int(enable_multiscale_flow)}_ml{int(enable_multilayer_fusion)}"
    )
    now_mono = time.monotonic()
    if cache_key in _MEM_CACHE:
        ts, cached = _MEM_CACHE[cache_key]
        if (not is_live_mode) or (now_mono - ts < _MEM_TTL_SECONDS):
            return cached

    npz_cache_file = GOES_CACHE_DIR / "decoded_v7" / f"{cache_key}.npz"
    meta_cache_file = GOES_CACHE_DIR / "decoded_v7" / f"{cache_key}.json"
    if npz_cache_file.exists() and meta_cache_file.exists():
        age_disk = time.time() - meta_cache_file.stat().st_mtime
        if (not is_live_mode) or (age_disk < _MEM_TTL_SECONDS):
            try:
                import json as _json

                meta_d = _json.loads(meta_cache_file.read_text(encoding="utf-8"))
                meta_d["bbox"] = tuple(meta_d["bbox"])
                with np.load(npz_cache_file) as zf:
                    for k_arr in zf.files:
                        meta_d[k_arr] = zf[k_arr]
                _MEM_CACHE[cache_key] = (now_mono, meta_d)
                return meta_d
            except Exception:
                pass

    is_west = lon < -100.0
    bucket, sat_short = _select_goes_bucket(is_west, ref_utc)

    # 768 km x 768 km domain (+/- 384 km) at 128 x 128 (6.0 km/px)
    half_km = 384.0
    grid_px = 128
    dlat = half_km / 111.13292
    dlon = half_km / max(25.0, 111.41284 * math.cos(math.radians(lat)))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

    sun_el_ref = _solar_elevation_utc(lat, lon, ref_utc)
    need_twilight_btd = enable_multilayer_fusion and (sun_el_ref <= 8.5)

    if HAS_H5PY:
        try:
            with ThreadPoolExecutor(max_workers=5) as list_pool:
                f_ctp_keys = list_pool.submit(_list_s3_product_keys_for_window, bucket, "ABI-L2-ACTPC", ref_utc, 55)
                f_ach_keys = list_pool.submit(_list_s3_product_keys_for_window, bucket, "ABI-L2-ACHAC", ref_utc, 55)
                f_cod_keys = list_pool.submit(_list_s3_product_keys_for_window, bucket, "ABI-L2-CODC", ref_utc, 35)
                f_ccl_keys = (
                    list_pool.submit(_list_s3_product_keys_for_window, bucket, "ABI-L2-CCLC", ref_utc, 55)
                    if enable_multilayer_fusion
                    else None
                )
                f_btd_keys = (
                    list_pool.submit(_list_cmipc_c07_c13_keys, bucket, ref_utc, 55)
                    if need_twilight_btd
                    else None
                )
                ctp_entries = f_ctp_keys.result()
                ach_entries = f_ach_keys.result()
                cod_entries = f_cod_keys.result()
                ccl_entries = f_ccl_keys.result() if f_ccl_keys is not None else []
                c07_entries, c13_entries = f_btd_keys.result() if f_btd_keys is not None else ([], [])

            pick_0 = _pick_closest_causal_key(ctp_entries, ref_utc, ref_utc, max_slop_minutes=22.0)
            if pick_0 is not None:
                t_0_dt, ctp_key_0 = pick_0
                pick_15 = _pick_closest_causal_key(
                    ctp_entries,
                    t_0_dt - dt.timedelta(minutes=15),
                    t_0_dt - dt.timedelta(minutes=6),
                    max_slop_minutes=14.0,
                ) or (t_0_dt, ctp_key_0)
                pick_30 = _pick_closest_causal_key(
                    ctp_entries,
                    t_0_dt - dt.timedelta(minutes=30),
                    t_0_dt - dt.timedelta(minutes=16),
                    max_slop_minutes=16.0,
                ) or pick_15

                t_15_dt, ctp_key_15 = pick_15
                t_30_dt, ctp_key_30 = pick_30

                ach_0 = _pick_closest_causal_key(ach_entries, t_0_dt, ref_utc, 12.0)
                ach_15 = _pick_closest_causal_key(ach_entries, t_15_dt, ref_utc, 12.0)
                ach_30 = _pick_closest_causal_key(ach_entries, t_30_dt, ref_utc, 12.0)
                cod_0 = _pick_closest_causal_key(cod_entries, t_0_dt, ref_utc, 15.0)
                ccl_0 = _pick_closest_causal_key(ccl_entries, t_0_dt, ref_utc, 12.0) if ccl_entries else None
                ccl_15 = _pick_closest_causal_key(ccl_entries, t_15_dt, ref_utc, 12.0) if ccl_entries else None
                ccl_30 = _pick_closest_causal_key(ccl_entries, t_30_dt, ref_utc, 12.0) if ccl_entries else None
                # Match twilight BTD channels symmetrically across all 3 causal frames so
                # golden-hour optical flow, pinning, and growth tendency see consistent fog edges.
                c07_0 = _pick_closest_causal_key(c07_entries, t_0_dt, ref_utc, 10.0) if c07_entries else None
                c13_0 = _pick_closest_causal_key(c13_entries, t_0_dt, ref_utc, 10.0) if c13_entries else None
                c07_15 = _pick_closest_causal_key(c07_entries, t_15_dt, ref_utc, 10.0) if c07_entries else None
                c13_15 = _pick_closest_causal_key(c13_entries, t_15_dt, ref_utc, 10.0) if c13_entries else None
                c07_30 = _pick_closest_causal_key(c07_entries, t_30_dt, ref_utc, 10.0) if c07_entries else None
                c13_30 = _pick_closest_causal_key(c13_entries, t_30_dt, ref_utc, 10.0) if c13_entries else None

                needed_keys: set[str] = {ctp_key_0, ctp_key_15, ctp_key_30}
                for opt_p in (
                    ach_0, ach_15, ach_30, cod_0, ccl_0, ccl_15, ccl_30,
                    c07_0, c13_0, c07_15, c13_15, c07_30, c13_30,
                ):
                    if opt_p is not None:
                        needed_keys.add(opt_p[1])

                downloaded: dict[str, bytes] = {}
                with ThreadPoolExecutor(max_workers=min(8, len(needed_keys))) as dl_pool:
                    fut_map = {k: dl_pool.submit(_download_s3_bytes, bucket, k) for k in needed_keys}
                    for k, fut in fut_map.items():
                        downloaded[k] = fut.result()

                lats_1d = np.linspace(bbox[3], bbox[1], grid_px, dtype=np.float64)
                lons_1d = np.linspace(bbox[0], bbox[2], grid_px, dtype=np.float64)
                lat_grid, lon_grid = np.meshgrid(lats_1d, lons_1d, indexing="ij")

                dec_0 = _decode_goes_l2_frame(
                    downloaded[ctp_key_0],
                    downloaded.get(ach_0[1]) if ach_0 else None,
                    downloaded.get(cod_0[1]) if cod_0 else None,
                    lat_grid,
                    lon_grid,
                    is_coastal,
                    enable_parallax=enable_parallax,
                    cclc_bytes=downloaded.get(ccl_0[1]) if ccl_0 else None,
                    cmip_c07_bytes=downloaded.get(c07_0[1]) if c07_0 else None,
                    cmip_c13_bytes=downloaded.get(c13_0[1]) if c13_0 else None,
                    scan_utc=t_0_dt,
                    enable_multilayer_fusion=enable_multilayer_fusion,
                )
                dec_15 = (
                    dec_0
                    if ctp_key_15 == ctp_key_0
                    else _decode_goes_l2_frame(
                        downloaded[ctp_key_15],
                        downloaded.get(ach_15[1]) if ach_15 else None,
                        None,
                        lat_grid,
                        lon_grid,
                        is_coastal,
                        enable_parallax=enable_parallax,
                        cclc_bytes=downloaded.get(ccl_15[1]) if ccl_15 else None,
                        cmip_c07_bytes=downloaded.get(c07_15[1]) if c07_15 else None,
                        cmip_c13_bytes=downloaded.get(c13_15[1]) if c13_15 else None,
                        scan_utc=t_15_dt,
                        enable_multilayer_fusion=enable_multilayer_fusion,
                    )
                )
                dec_30 = (
                    dec_15
                    if ctp_key_30 == ctp_key_15
                    else _decode_goes_l2_frame(
                        downloaded[ctp_key_30],
                        downloaded.get(ach_30[1]) if ach_30 else None,
                        None,
                        lat_grid,
                        lon_grid,
                        is_coastal,
                        enable_parallax=enable_parallax,
                        cclc_bytes=downloaded.get(ccl_30[1]) if ccl_30 else None,
                        cmip_c07_bytes=downloaded.get(c07_30[1]) if c07_30 else None,
                        cmip_c13_bytes=downloaded.get(c13_30[1]) if c13_30 else None,
                        scan_utc=t_30_dt,
                        enable_multilayer_fusion=enable_multilayer_fusion,
                    )
                )

                dx_m = (2.0 * half_km * 1000.0) / grid_px
                dy_m = dx_m
                dt_30_0_s = max(0.0, (t_0_dt - t_30_dt).total_seconds())
                dt_30_15_s = max(0.0, (t_15_dt - t_30_dt).total_seconds())
                dt_15_0_s = max(0.0, (t_0_dt - t_15_dt).total_seconds())

                u_low_g, v_low_g, c_low_g, u_low_c, v_low_c, c_low_c = _compute_regional_flow_field(
                    dec_30["low_cf"],
                    dec_15["low_cf"],
                    dec_0["low_cf"],
                    dx_m,
                    dy_m,
                    dt_30_0_s,
                    dt_30_15_s,
                    dt_15_0_s,
                    enable_multiscale=enable_multiscale_flow,
                )

                up_30 = np.clip(dec_30["mid_cf"] + dec_30["high_cf"], 0.0, 1.0)
                up_15 = np.clip(dec_15["mid_cf"] + dec_15["high_cf"], 0.0, 1.0)
                up_0 = np.clip(dec_0["mid_cf"] + dec_0["high_cf"], 0.0, 1.0)
                u_hi_g, v_hi_g, c_hi_g, u_hi_c, v_hi_c, c_hi_c = _compute_regional_flow_field(
                    up_30,
                    up_15,
                    up_0,
                    dx_m,
                    dy_m,
                    dt_30_0_s,
                    dt_30_15_s,
                    dt_15_0_s,
                    enable_multiscale=enable_multiscale_flow,
                )

                # Apples-to-apples 30-minute Eulerian stationarity residual (`|c(t0) - c(t-30)|`)
                # matching the 30-minute Lagrangian hindcast window so moving fronts aren't falsely pinned.
                euler_low = np.abs(dec_0["low_cf"] - dec_30["low_cf"]).astype(np.float32)
                euler_high = np.abs(up_0 - up_30).astype(np.float32)

                # Dominant motion summary for UI telemetry
                if c_low_c >= c_hi_c:
                    dom_u, dom_v, dom_c = u_low_c, v_low_c, max(c_low_c, c_hi_c)
                else:
                    dom_u, dom_v, dom_c = u_hi_c, v_hi_c, max(c_low_c, c_hi_c)

                sun_el_now = _solar_elevation_utc(lat, lon, t_0_dt)
                age_min = max(0.0, (ref_utc - t_0_dt).total_seconds() / 60.0)
                prod_tags = ["ACTP", "ACHA", "COD"]
                if dec_0.get("has_cclc"):
                    prod_tags.append("CCLC-5Layer")
                if dec_0.get("has_btd_fog"):
                    prod_tags.append("C13-C07-BTD")

                result: dict[str, Any] = {
                    "satellite": f"NOAA {sat_short} ABI L2 ({'+'.join(prod_tags)})",
                    "is_west": is_west,
                    "is_calibrated_l2": True,
                    "has_cclc": bool(dec_0.get("has_cclc")),
                    "has_cclc_multilayer": bool(dec_0.get("has_cclc")),
                    "has_btd_fog": bool(dec_0.get("has_btd_fog")),
                    "has_twilight_btd_fog": bool(dec_0.get("has_btd_fog")),
                    "parallax_corrected": enable_parallax,
                    "multiscale_flow": enable_multiscale_flow,
                    "scan_timestamp_utc": t_0_dt.isoformat(),
                    "scan_age_minutes": round(age_min, 1),
                    "history_span_minutes": round(dt_30_0_s / 60.0, 1),
                    "sun_el_at_scan_deg": round(sun_el_now, 2),
                    "has_daylight_visible": sun_el_now > 3.0,
                    "bbox": bbox,
                    "grid_px": grid_px,
                    "motion_u_ms": dom_u,
                    "motion_v_ms": dom_v,
                    "motion_confidence": dom_c,
                    "low_motion_uv_ms": [u_low_c, v_low_c],
                    "high_motion_uv_ms": [u_hi_c, v_hi_c],
                    "low_motion_confidence": c_low_c,
                    "high_motion_confidence": c_hi_c,
                    "u_low_grid": u_low_g,
                    "v_low_grid": v_low_g,
                    "conf_low_grid": c_low_g,
                    "u_high_grid": u_hi_g,
                    "v_high_grid": v_hi_g,
                    "conf_high_grid": c_hi_g,
                    "fog_cf_grid": dec_0["fog_cf"],
                    "low_cf_grid": dec_0["low_cf"],
                    "mid_cf_grid": dec_0["mid_cf"],
                    "high_cf_grid": dec_0["high_cf"],
                    "actp_low_cf_grid": dec_0["actp_low_cf"],
                    "actp_mid_cf_grid": dec_0["actp_mid_cf"],
                    "actp_high_cf_grid": dec_0["actp_high_cf"],
                    "fog_m30_grid": dec_30["fog_cf"],
                    "low_m30_grid": dec_30["low_cf"],
                    "mid_m30_grid": dec_30["mid_cf"],
                    "high_m30_grid": dec_30["high_cf"],
                    "euler_low_grid": euler_low,
                    "euler_high_grid": euler_high,
                    "tau_low_grid": dec_0["tau_low"],
                    "tau_mid_grid": dec_0["tau_mid"],
                    "tau_high_grid": dec_0["tau_high"],
                    "tau_grid": dec_0["tau_total"],
                    "low_top_m_grid": dec_0["low_top_m"],
                    "mid_top_m_grid": dec_0["mid_top_m"],
                    "high_top_m_grid": dec_0["high_top_m"],
                    "tb_k_grid": dec_0["tb_k"],
                }
                _MEM_CACHE[cache_key] = (now_mono, result)
                try:
                    import json as _json

                    npz_cache_file.parent.mkdir(parents=True, exist_ok=True)
                    arr_dict = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
                    scalar_dict = {
                        k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in result.items()
                        if not isinstance(v, np.ndarray)
                    }
                    np.savez_compressed(npz_cache_file, **arr_dict)
                    meta_cache_file.write_text(_json.dumps(scalar_dict), encoding="utf-8")
                except Exception:
                    pass
                return result
        except Exception:
            pass

    if is_live_mode:
        try:
            fallback = _fetch_iem_wms_fallback(
                lat, lon, bbox, 112, surface_temp_c, is_coastal, is_west, ref_utc
            )
            if fallback is not None:
                _MEM_CACHE[cache_key] = (now_mono, fallback)
                return fallback
        except Exception:
            return None
    return None


def sample_advected_goes_state(
    goes_data: dict[str, Any],
    q_lat: float,
    q_lon: float,
    lead_minutes: float,
    wind_low_uv_ms: tuple[float, float],
    wind_mid_uv_ms: tuple[float, float],
    wind_high_uv_ms: tuple[float, float],
    enable_pinning: bool = True,
    enable_growth_decay: bool = True,
) -> dict[str, float]:
    """
    Lagrangian back-trajectory evaluation at `(q_lat, q_lon)` for `+lead_minutes` (`0..240 min`)
    with:
    - Spatially varying layered GOES optical-flow + NWP steering-wind blending
    - Continuity-tested Orographic Stationary-Feature Pinning (`w_pin` in `[0, 0.88]`)
    - Exponentially damped Lagrangian cloud growth/dissipation tendency (`DC/Dt`)
    """
    bbox = goes_data["bbox"]
    lead_clamped = max(0.0, min(240.0, lead_minutes))
    dt_sec = lead_clamped * 60.0
    hist_min = float(goes_data.get("history_span_minutes") or 0.0)
    hist_sec = hist_min * 60.0

    m_per_lat = 111_132.92
    m_per_lon = max(25_000.0, 111_412.84 * math.cos(math.radians(q_lat)))

    def _advect_layer(
        cf0_grid: np.ndarray,
        cfm30_grid: np.ndarray | None,
        euler_grid: np.ndarray | None,
        u_sat_grid: np.ndarray | None,
        v_sat_grid: np.ndarray | None,
        c_sat_grid: np.ndarray | None,
        w_nwp_uv: tuple[float, float],
        sat_max_weight: float,
    ) -> tuple[float, float, float, float]:
        if u_sat_grid is not None and v_sat_grid is not None and c_sat_grid is not None:
            u_sat = _sample_grid_bilinear(u_sat_grid, bbox, q_lat, q_lon)
            v_sat = _sample_grid_bilinear(v_sat_grid, bbox, q_lat, q_lon)
            conf = _sample_grid_bilinear(c_sat_grid, bbox, q_lat, q_lon)
        else:
            u_sat = float(goes_data.get("motion_u_ms", 0.0))
            v_sat = float(goes_data.get("motion_v_ms", 0.0))
            conf = float(goes_data.get("motion_confidence", 0.0))

        w_eff = max(0.0, min(0.96, (conf / 0.95) * sat_max_weight))
        u_raw = w_eff * u_sat + (1.0 - w_eff) * w_nwp_uv[0]
        v_raw = w_eff * v_sat + (1.0 - w_eff) * w_nwp_uv[1]

        w_pin = 0.0
        if (
            enable_pinning
            and dt_sec > 60.0
            and hist_sec >= 720.0
            and cfm30_grid is not None
            and euler_grid is not None
            and math.hypot(u_raw, v_raw) >= 1.1
        ):
            c_now_here = _sample_grid_bilinear(cf0_grid, bbox, q_lat, q_lon)
            hind_lat = q_lat - (v_raw * hist_sec) / m_per_lat
            hind_lon = q_lon - (u_raw * hist_sec) / m_per_lon
            c_adv_hind = _sample_grid_bilinear(cfm30_grid, bbox, hind_lat, hind_lon)
            e_adv = abs(c_now_here - c_adv_hind)
            e_stat = _sample_grid_bilinear(euler_grid, bbox, q_lat, q_lon)
            if e_adv > e_stat + 0.025:
                pin_raw = min(0.92, (e_adv - e_stat - 0.025) / 0.18)
                w_pin = pin_raw * math.exp(-lead_clamped / 220.0)

        u_eff = (1.0 - w_pin) * u_raw
        v_eff = (1.0 - w_pin) * v_raw

        src_lat = q_lat - (v_eff * dt_sec) / m_per_lat
        src_lon = q_lon - (u_eff * dt_sec) / m_per_lon
        cf_adv = _sample_grid_bilinear(cf0_grid, bbox, src_lat, src_lon)

        if (
            enable_growth_decay
            and dt_sec > 60.0
            and hist_sec >= 720.0
            and cfm30_grid is not None
        ):
            prev_lat = src_lat - (v_eff * hist_sec) / m_per_lat
            prev_lon = src_lon - (u_eff * hist_sec) / m_per_lon
            cf_prev = _sample_grid_bilinear(cfm30_grid, bbox, prev_lat, prev_lon)
            d_cf = cf_adv - cf_prev
            tau_g_min = 55.0
            horizon_factor = (1.0 - math.exp(-lead_clamped / tau_g_min)) / max(
                0.25, hist_min / tau_g_min
            )
            delta_tend = max(-0.22, min(0.22, 0.36 * d_cf * horizon_factor))
            cf_out = max(0.0, min(1.0, cf_adv + delta_tend))
        else:
            cf_out = max(0.0, min(1.0, cf_adv))

        return cf_out, src_lat, src_lon, w_pin

    low_cf, lat_lo, lon_lo, pin_lo = _advect_layer(
        goes_data["low_cf_grid"],
        goes_data.get("low_m30_grid"),
        goes_data.get("euler_low_grid"),
        goes_data.get("u_low_grid"),
        goes_data.get("v_low_grid"),
        goes_data.get("conf_low_grid"),
        wind_low_uv_ms,
        0.95,
    )
    fog_grid = goes_data.get("fog_cf_grid")
    if fog_grid is not None:
        fog_cf, _, _, _ = _advect_layer(
            fog_grid,
            goes_data.get("fog_m30_grid"),
            goes_data.get("euler_low_grid"),
            goes_data.get("u_low_grid"),
            goes_data.get("v_low_grid"),
            goes_data.get("conf_low_grid"),
            wind_low_uv_ms,
            0.95,
        )
    else:
        fog_cf = low_cf * 0.80

    mid_cf, lat_mi, lon_mi, _ = _advect_layer(
        goes_data["mid_cf_grid"],
        goes_data.get("mid_m30_grid"),
        goes_data.get("euler_high_grid"),
        goes_data.get("u_high_grid"),
        goes_data.get("v_high_grid"),
        goes_data.get("conf_high_grid"),
        wind_mid_uv_ms,
        0.85,
    )
    high_cf, lat_hi, lon_hi, pin_hi = _advect_layer(
        goes_data["high_cf_grid"],
        goes_data.get("high_m30_grid"),
        goes_data.get("euler_high_grid"),
        goes_data.get("u_high_grid"),
        goes_data.get("v_high_grid"),
        goes_data.get("conf_high_grid"),
        wind_high_uv_ms,
        0.88,
    )

    def _scaled_tau(tau_key: str, src_cf_grid: np.ndarray, new_cf: float, slat: float, slon: float, default_mult: float) -> float:
        t_grid = goes_data.get(tau_key)
        if t_grid is None:
            return new_cf * default_mult
        raw_tau = _sample_grid_bilinear(t_grid, bbox, slat, slon)
        raw_cf = _sample_grid_bilinear(src_cf_grid, bbox, slat, slon)
        if raw_cf > 0.06:
            return max(0.0, min(48.0, raw_tau * (new_cf / raw_cf)))
        return new_cf * default_mult

    tau_low = _scaled_tau("tau_low_grid", goes_data["low_cf_grid"], low_cf, lat_lo, lon_lo, 7.4)
    tau_mid = _scaled_tau("tau_mid_grid", goes_data["mid_cf_grid"], mid_cf, lat_mi, lon_mi, 5.6)
    tau_high = _scaled_tau("tau_high_grid", goes_data["high_cf_grid"], high_cf, lat_hi, lon_hi, 1.85)
    tau_total = max(0.0, min(52.0, tau_low + tau_mid + tau_high))

    low_top_m = (
        _sample_grid_bilinear(goes_data["low_top_m_grid"], bbox, lat_lo, lon_lo)
        if "low_top_m_grid" in goes_data
        else 1100.0
    )
    mid_top_m = (
        _sample_grid_bilinear(goes_data["mid_top_m_grid"], bbox, lat_mi, lon_mi)
        if "mid_top_m_grid" in goes_data
        else 4300.0
    )
    high_top_m = (
        _sample_grid_bilinear(goes_data["high_top_m_grid"], bbox, lat_hi, lon_hi)
        if "high_top_m_grid" in goes_data
        else 9600.0
    )
    tb_k = _sample_grid_bilinear(goes_data["tb_k_grid"], bbox, lat_hi, lon_hi)

    return {
        "fog_cf": fog_cf,
        "low_cf": low_cf,
        "mid_cf": mid_cf,
        "high_cf": high_cf,
        "tau_low": tau_low,
        "tau_mid": tau_mid,
        "tau_high": tau_high,
        "tau_total": tau_total,
        "low_top_asl_m": low_top_m,
        "mid_top_asl_m": mid_top_m,
        "high_top_asl_m": high_top_m,
        "pin_weight_low": pin_lo,
        "pin_weight_high": pin_hi,
        "tb_k": tb_k,
    }


def _sample_grid_bilinear_array(
    grid: np.ndarray,
    bbox: tuple[float, float, float, float],
    q_lat: np.ndarray,
    q_lon: np.ndarray,
) -> np.ndarray:
    """Vectorized bilinear sampler over `(grid_px, grid_px)` GOES L2 arrays."""
    min_lon, min_lat, max_lon, max_lat = bbox
    h, w = grid.shape
    fx = (q_lon - min_lon) / max(1e-6, max_lon - min_lon) * (w - 1)
    fy = (max_lat - q_lat) / max(1e-6, max_lat - min_lat) * (h - 1)
    fx = np.clip(fx, 0.0, float(w - 1.001))
    fy = np.clip(fy, 0.0, float(h - 1.001))
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    tx = (fx - x0).astype(np.float32)
    ty = (fy - y0).astype(np.float32)
    x1 = np.minimum(w - 1, x0 + 1)
    y1 = np.minimum(h - 1, y0 + 1)

    v00 = grid[y0, x0]
    v10 = grid[y0, x1]
    v01 = grid[y1, x0]
    v11 = grid[y1, x1]
    return ((v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty) + (v01 * (1.0 - tx) + v11 * tx) * ty).astype(np.float32)


def sample_advected_goes_grid_64x64(
    goes_data: dict[str, Any],
    center_lat: float,
    center_lon: float,
    lead_minutes: float,
    wind_low_uv_ms: tuple[float, float],
    wind_mid_uv_ms: tuple[float, float],
    wind_high_uv_ms: tuple[float, float],
    grid_size: int = 64,
    span_m: float = 112000.0,
) -> dict[str, np.ndarray]:
    """
    Vectorized `64x64` (`112 km x 112 km`, `1.75 km` texel) Lagrangian GOES-R multi-layer sampler.
    Evaluates all 4,096 spatial texels simultaneously in `< 0.8 ms`, preserving sharp coastal stratus
    edges, Golden Gate / valley fog tongues, stationary orographic pinning (`w_pin`), and optical depth.
    Row 0 = South (`-half_span`), Row 63 = North (`+half_span`); Col 0 = West, Col 63 = East.
    """
    bbox = goes_data["bbox"]
    lead_clamped = max(0.0, min(240.0, float(lead_minutes)))
    dt_sec = lead_clamped * 60.0
    hist_min = float(goes_data.get("history_span_minutes") or 0.0)
    hist_sec = hist_min * 60.0

    half_km = (span_m * 0.5) / 1000.0
    m_per_lat_km = 111.13292
    m_per_lon_km = max(22.0, 111.41284 * math.cos(math.radians(center_lat)))
    m_per_lat_m = 111_132.92
    m_per_lon_m = max(25_000.0, 111_412.84 * math.cos(math.radians(center_lat)))

    u_1d = (np.arange(grid_size, dtype=np.float32) + 0.5) / float(grid_size) - 0.5
    east_km_1d = u_1d * (2.0 * half_km)
    north_km_1d = u_1d * (2.0 * half_km)
    east_grid, north_grid = np.meshgrid(east_km_1d, north_km_1d, indexing="xy")

    lat_grid = (center_lat + north_grid / m_per_lat_km).astype(np.float32)
    lon_grid = (center_lon + east_grid / m_per_lon_km).astype(np.float32)

    def _advect_layer_vec(
        cf0_grid: np.ndarray,
        cfm30_grid: np.ndarray | None,
        euler_grid: np.ndarray | None,
        u_sat_grid: np.ndarray | None,
        v_sat_grid: np.ndarray | None,
        c_sat_grid: np.ndarray | None,
        w_nwp_uv: tuple[float, float],
        sat_max_weight: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if u_sat_grid is not None and v_sat_grid is not None and c_sat_grid is not None:
            u_sat = _sample_grid_bilinear_array(u_sat_grid, bbox, lat_grid, lon_grid)
            v_sat = _sample_grid_bilinear_array(v_sat_grid, bbox, lat_grid, lon_grid)
            conf = _sample_grid_bilinear_array(c_sat_grid, bbox, lat_grid, lon_grid)
        else:
            u_sat = np.full_like(lat_grid, float(goes_data.get("motion_u_ms", 0.0)))
            v_sat = np.full_like(lat_grid, float(goes_data.get("motion_v_ms", 0.0)))
            conf = np.full_like(lat_grid, float(goes_data.get("motion_confidence", 0.0)))

        w_eff = np.clip((conf / 0.95) * sat_max_weight, 0.0, 0.96)
        u_raw = w_eff * u_sat + (1.0 - w_eff) * float(w_nwp_uv[0])
        v_raw = w_eff * v_sat + (1.0 - w_eff) * float(w_nwp_uv[1])

        if (
            dt_sec > 60.0
            and hist_sec >= 720.0
            and cfm30_grid is not None
            and euler_grid is not None
        ):
            c_now = _sample_grid_bilinear_array(cf0_grid, bbox, lat_grid, lon_grid)
            hind_lat = lat_grid - (v_raw * hist_sec) / m_per_lat_m
            hind_lon = lon_grid - (u_raw * hist_sec) / m_per_lon_m
            c_hind = _sample_grid_bilinear_array(cfm30_grid, bbox, hind_lat, hind_lon)
            e_adv = np.abs(c_now - c_hind)
            e_stat = _sample_grid_bilinear_array(euler_grid, bbox, lat_grid, lon_grid)
            spd_m = np.hypot(u_raw, v_raw)
            pin_raw = np.where(
                (spd_m >= 1.1) & (e_adv > e_stat + 0.025),
                np.minimum(0.92, (e_adv - e_stat - 0.025) / 0.18),
                0.0,
            )
            w_pin = (pin_raw * math.exp(-lead_clamped / 220.0)).astype(np.float32)
        else:
            w_pin = np.zeros_like(lat_grid)

        u_eff = (1.0 - w_pin) * u_raw
        v_eff = (1.0 - w_pin) * v_raw

        src_lat = lat_grid - (v_eff * dt_sec) / m_per_lat_m
        src_lon = lon_grid - (u_eff * dt_sec) / m_per_lon_m
        cf_adv = _sample_grid_bilinear_array(cf0_grid, bbox, src_lat, src_lon)

        if dt_sec > 60.0 and hist_sec >= 720.0 and cfm30_grid is not None:
            prev_lat = src_lat - (v_eff * hist_sec) / m_per_lat_m
            prev_lon = src_lon - (u_eff * hist_sec) / m_per_lon_m
            cf_prev = _sample_grid_bilinear_array(cfm30_grid, bbox, prev_lat, prev_lon)
            tau_g_min = 55.0
            horizon_factor = (1.0 - math.exp(-lead_clamped / tau_g_min)) / max(0.25, hist_min / tau_g_min)
            delta_tend = np.clip(0.36 * (cf_adv - cf_prev) * horizon_factor, -0.22, 0.22)
            cf_out = np.clip(cf_adv + delta_tend, 0.0, 1.0)
        else:
            cf_out = np.clip(cf_adv, 0.0, 1.0)
        return cf_out.astype(np.float32), src_lat, src_lon, w_pin.astype(np.float32)

    low_cf, lat_lo, lon_lo, pin_lo = _advect_layer_vec(
        goes_data["low_cf_grid"],
        goes_data.get("low_m30_grid"),
        goes_data.get("euler_low_grid"),
        goes_data.get("u_low_grid"),
        goes_data.get("v_low_grid"),
        goes_data.get("conf_low_grid"),
        wind_low_uv_ms,
        0.95,
    )
    fog_grid = goes_data.get("fog_cf_grid")
    if fog_grid is not None:
        fog_cf, _, _, _ = _advect_layer_vec(
            fog_grid,
            goes_data.get("fog_m30_grid"),
            goes_data.get("euler_low_grid"),
            goes_data.get("u_low_grid"),
            goes_data.get("v_low_grid"),
            goes_data.get("conf_low_grid"),
            wind_low_uv_ms,
            0.95,
        )
    else:
        fog_cf = (low_cf * 0.80).astype(np.float32)

    mid_cf, lat_mi, lon_mi, _ = _advect_layer_vec(
        goes_data["mid_cf_grid"],
        goes_data.get("mid_m30_grid"),
        goes_data.get("euler_high_grid"),
        goes_data.get("u_high_grid"),
        goes_data.get("v_high_grid"),
        goes_data.get("conf_high_grid"),
        wind_mid_uv_ms,
        0.85,
    )
    high_cf, lat_hi, lon_hi, pin_hi = _advect_layer_vec(
        goes_data["high_cf_grid"],
        goes_data.get("high_m30_grid"),
        goes_data.get("euler_high_grid"),
        goes_data.get("u_high_grid"),
        goes_data.get("v_high_grid"),
        goes_data.get("conf_high_grid"),
        wind_high_uv_ms,
        0.88,
    )

    if "low_top_m_grid" in goes_data:
        low_top_m = _sample_grid_bilinear_array(goes_data["low_top_m_grid"], bbox, lat_lo, lon_lo)
    else:
        low_top_m = np.full_like(low_cf, 1100.0)

    if "tau_grid" in goes_data:
        tau_tot = np.clip(_sample_grid_bilinear_array(goes_data["tau_grid"], bbox, lat_lo, lon_lo), 0.0, 32.0)
    else:
        tau_tot = np.clip(low_cf * 7.4 + mid_cf * 5.6 + high_cf * 1.85, 0.0, 32.0)

    return {
        "fog_cf": fog_cf,
        "low_cf": low_cf,
        "mid_cf": mid_cf,
        "high_cf": high_cf,
        "low_top_m": low_top_m,
        "tau_total": tau_tot,
        "pin_weight_low": pin_lo,
        "pin_weight_high": pin_hi,
    }

