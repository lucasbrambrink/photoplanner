"""
Zero-Dependency Native NOAA HRRR 3 km S3 `.idx` + GRIB2 Byte-Range Sounding Client (`planner/weather/hrrr_s3.py`).

Directly queries NOAA's authoritative `s3://noaa-hrrr-bdp-pds` bucket (`https://noaa-hrrr-bdp-pds.s3.amazonaws.com/`)
for both **live operational forecasts (`+0h..+48h`)** and **strictly causal historical hindcasts (`+1h..+48h`)**,
unpacking targeted GRIB2 byte ranges (`GDT=3.30` Lambert Conformal Conic + `DRT=5.0/5.2/5.3` Complex Packing
with 1st/2nd-order spatial differencing) in pure vectorized `numpy` without requiring native C `eccodes`/`pygrib`.

Retrieved 3 km Native Fields:
- **`wrfsfcf{FF}.grib2` (2D Surface, Cloud Boundaries, Microphysics & HRRR-Smoke)**:
  `HGT:surface`, `VIS:surface`, `TMP:2 m`, `DPT:2 m`, `RH:2 m`, `UGRD:10 m`, `VGRD:10 m`,
  `CAPE:surface`, `HPBL:surface`, `AOTK:entire atmosphere` (HRRR-Smoke 550 nm AOD),
  `TCOLW:entire atmosphere` (Liquid Water Path `kg/m^2`), `TCOLI:entire atmosphere` (Ice Water Path `kg/m^2`),
  `TCDC:boundary layer cloud layer`, `LCDC:low cloud layer`, `MCDC:middle cloud layer`,
  `HCDC:high cloud layer`, `TCDC:entire atmosphere`, `HGT:cloud ceiling`, `HGT:cloud base`, `HGT:cloud top`.
- **`wrfprsf{FF}.grib2` (25 hPa Boundary-Layer & Tropospheric Sounding)**:
  `HGT`, `TMP`, `RH`, and `CLMR` (Cloud Liquid Water Mixing Ratio `kg/kg`) at
  `1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700 mb` (`~210 m` vertical spacing across
  coastal marine & valley inversions), plus `500 mb`, `300 mb`, and `250 mb` upper-tropospheric state
  (`HGT`, `TMP`, `RH`, `CIMIXR`, `UGRD`, `VGRD`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import math
from pathlib import Path
import struct
import threading
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
HRRR_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "hrrr_s3"
HRRR_S3_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
USER_AGENT = "PhotoPlanner/4.1 (hrrr-s3-native-sounding; github.com/lucasbrambrink/photoplanner)"

# NCEP HRRR CONUS 3 km Lambert Conformal Conic grid constants (GRIB2 Template 3.30)
HRRR_NX = 1799
HRRR_NY = 1059
_HRRR_R_M = 6371229.0
_HRRR_PHI0 = math.radians(38.5)
_HRRR_LAM0 = math.radians(262.5 - 360.0)
_HRRR_N = math.sin(_HRRR_PHI0)
_HRRR_F = (math.cos(_HRRR_PHI0) * (math.tan(math.pi * 0.25 + _HRRR_PHI0 * 0.5) ** _HRRR_N)) / _HRRR_N
_HRRR_DX_M = 3000.0

# 25 hPa marine boundary-layer levels (1000..850 mb) + 50 hPa lower-troposphere levels (800, 750, 700 mb)
BL_PRESSURE_LEVELS_MB: tuple[int, ...] = (
    1000, 975, 950, 925, 900, 875, 850, 800, 750, 700
)
UPPER_PRESSURE_LEVELS_MB: tuple[int, ...] = (500, 300, 250)

_MEM_TILE_CACHE: dict[str, tuple[float, dict[str, np.ndarray], dict[str, Any]]] = {}
_MEM_LOCK = threading.Lock()
_TILE_LOCK_MAP: dict[str, threading.Lock] = {}
_MEM_TTL_SECONDS = 1800.0


def _lcc_xy_m(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    phi = math.radians(max(-80.0, min(85.0, lat_deg)))
    lon_norm = ((lon_deg + 180.0) % 360.0) - 180.0
    lam = math.radians(lon_norm)
    rho = _HRRR_R_M * _HRRR_F / (math.tan(math.pi * 0.25 + phi * 0.5) ** _HRRR_N)
    theta = _HRRR_N * (lam - _HRRR_LAM0)
    return rho * math.sin(theta), -rho * math.cos(theta)


_HRRR_X1_M, _HRRR_Y1_M = _lcc_xy_m(21.138123, 237.280472 - 360.0)


def hrrr_latlon_to_ij(lat: float, lon: float) -> tuple[float, float]:
    """Converts `(lat, lon)` to fractional HRRR 3 km grid indices `(i_east, j_north)`."""
    x, y = _lcc_xy_m(lat, lon)
    return (x - _HRRR_X1_M) / _HRRR_DX_M, (y - _HRRR_Y1_M) / _HRRR_DX_M


def is_in_hrrr_conus(lat: float, lon: float, margin_cells: float = 4.0) -> bool:
    """Returns `True` when `(lat, lon)` lies safely inside the 3 km HRRR CONUS domain."""
    if not (21.0 <= lat <= 53.0 and -135.0 <= lon <= -60.0):
        return False
    fi, fj = hrrr_latlon_to_ij(lat, lon)
    return (margin_cells <= fi <= HRRR_NX - 1.0 - margin_cells) and (
        margin_cells <= fj <= HRRR_NY - 1.0 - margin_cells
    )


def _unpack_bits_u64(data_bytes: bytes, bit_offset: int, count: int, width: int) -> np.ndarray:
    if width <= 0 or count <= 0:
        return np.zeros(max(0, count), dtype=np.int64)
    bit_pos = bit_offset + np.arange(count, dtype=np.int64) * int(width)
    byte_idx = (bit_pos >> 3).astype(np.int64)
    bit_in_byte = (bit_pos & 7).astype(np.uint64)
    padded = np.zeros(len(data_bytes) + 8, dtype=np.uint8)
    padded[: len(data_bytes)] = np.frombuffer(data_bytes, dtype=np.uint8)
    word = (
        (padded[byte_idx].astype(np.uint64) << np.uint64(24))
        | (padded[byte_idx + 1].astype(np.uint64) << np.uint64(16))
        | (padded[byte_idx + 2].astype(np.uint64) << np.uint64(8))
        | padded[byte_idx + 3].astype(np.uint64)
    )
    shift = np.uint64(32) - bit_in_byte - np.uint64(width)
    mask = np.uint64((1 << int(width)) - 1)
    return ((word >> shift) & mask).astype(np.int64)


def decode_grib2_message(buf: bytes) -> np.ndarray:
    """
    Vectorized pure-`numpy` GRIB2 message decoder for NCEP HRRR (`DRT=5.0`, `5.2`, `5.3` + Section 6 bitmap).
    Returns a 2D `float32` array of shape `(HRRR_NY, HRRR_NX)` (`+j` South->North, `+i` West->East).
    """
    if len(buf) < 32 or buf[:4] != b"GRIB":
        raise ValueError("Invalid GRIB2 header")

    pos = 16
    nx = HRRR_NX
    ny = HRRR_NY
    drt = 0
    s5 = b""
    s6 = b""
    s7 = b""

    while pos + 4 <= len(buf):
        if buf[pos : pos + 4] == b"7777":
            break
        slen = struct.unpack(">I", buf[pos : pos + 4])[0]
        if slen < 5 or pos + slen > len(buf):
            break
        sec = buf[pos + 4]
        body = buf[pos + 5 : pos + slen]
        if sec == 3 and len(body) >= 33:
            nx, ny = struct.unpack(">II", body[25:33])
        elif sec == 5:
            drt = struct.unpack(">H", body[4:6])[0]
            s5 = body
        elif sec == 6:
            s6 = body
        elif sec == 7:
            s7 = body
        pos += slen

    if not s5:
        raise ValueError("Missing GRIB2 Section 5")

    npts_pack = struct.unpack(">I", s5[0:4])[0]
    ref_f = struct.unpack(">f", s5[6:10])[0]
    bscale = struct.unpack(">h", s5[10:12])[0]
    if bscale & 0x8000:
        bscale = -(bscale & 0x7FFF)
    dscale = struct.unpack(">h", s5[12:14])[0]
    if dscale & 0x8000:
        dscale = -(dscale & 0x7FFF)
    nbits = s5[14]
    dec_factor = 10.0 ** dscale
    bin_factor = 2.0 ** bscale

    if drt == 0:
        if nbits == 0 or len(s7) == 0:
            vals = np.full(npts_pack, ref_f / dec_factor, dtype=np.float64)
        else:
            raw = _unpack_bits_u64(s7, 0, npts_pack, nbits).astype(np.float64)
            vals = (ref_f + raw * bin_factor) / dec_factor
    elif drt in (2, 3):
        ng = struct.unpack(">I", s5[26:30])[0]
        ref_gw = s5[30]
        wid_bits = s5[31]
        ref_gl = struct.unpack(">I", s5[32:36])[0]
        len_inc = s5[36]
        true_last_len = struct.unpack(">I", s5[37:41])[0]
        len_bits = s5[41]
        order = s5[42] if (drt == 3 and len(s5) > 43) else 0
        extra_oct = s5[43] if (drt == 3 and len(s5) > 43) else 0

        pos_b = 0
        init_vals: list[int] = []
        gmin = 0
        if drt == 3 and extra_oct > 0 and len(s7) >= (order + 1) * extra_oct:
            sign_bit = 1 << (8 * extra_oct - 1)
            mag_mask = sign_bit - 1

            def _read_sm(chunk: bytes) -> int:
                v = int.from_bytes(chunk, "big")
                return -(v & mag_mask) if (v & sign_bit) else v

            for _ in range(order):
                init_vals.append(_read_sm(s7[pos_b : pos_b + extra_oct]))
                pos_b += extra_oct
            gmin = _read_sm(s7[pos_b : pos_b + extra_oct])
            pos_b += extra_oct

        if ng == 0 or len(s7) <= pos_b:
            vals = np.full(npts_pack, ref_f / dec_factor, dtype=np.float64)
        else:
            bit_cur = pos_b * 8
            grefs = _unpack_bits_u64(s7, bit_cur, ng, nbits) if nbits > 0 else np.zeros(ng, dtype=np.int64)
            bit_cur = ((bit_cur + ng * nbits + 7) // 8) * 8
            gwids = ref_gw + (
                _unpack_bits_u64(s7, bit_cur, ng, wid_bits) if wid_bits > 0 else np.zeros(ng, dtype=np.int64)
            )
            bit_cur = ((bit_cur + ng * wid_bits + 7) // 8) * 8
            glens = ref_gl + len_inc * (
                _unpack_bits_u64(s7, bit_cur, ng, len_bits) if len_bits > 0 else np.zeros(ng, dtype=np.int64)
            )
            glens[-1] = true_last_len
            bit_cur = ((bit_cur + ng * len_bits + 7) // 8) * 8

            total_pts = int(np.sum(glens))
            out_int = np.empty(total_pts, dtype=np.int64)
            g_bit_starts = bit_cur + np.concatenate(([0], np.cumsum(glens * gwids)[:-1]))
            g_pt_starts = np.concatenate(([0], np.cumsum(glens)[:-1]))
            padded = np.zeros(len(s7) + 8, dtype=np.uint8)
            padded[: len(s7)] = np.frombuffer(s7, dtype=np.uint8)

            for w_val in np.unique(gwids):
                idxs_g = np.nonzero(gwids == w_val)[0]
                tot_w_pts = int(np.sum(glens[idxs_g]))
                if tot_w_pts <= 0:
                    continue
                intra = np.arange(tot_w_pts, dtype=np.int64) - np.repeat(
                    np.concatenate(([0], np.cumsum(glens[idxs_g])[:-1])), glens[idxs_g]
                )
                target_idx = np.repeat(g_pt_starts[idxs_g], glens[idxs_g]) + intra
                if w_val == 0:
                    out_int[target_idx] = np.repeat(grefs[idxs_g], glens[idxs_g])
                else:
                    bpos = np.repeat(g_bit_starts[idxs_g], glens[idxs_g]) + intra * int(w_val)
                    b_idx = (bpos >> 3).astype(np.int64)
                    b_off = (bpos & 7).astype(np.uint64)
                    word = (
                        (padded[b_idx].astype(np.uint64) << np.uint64(24))
                        | (padded[b_idx + 1].astype(np.uint64) << np.uint64(16))
                        | (padded[b_idx + 2].astype(np.uint64) << np.uint64(8))
                        | padded[b_idx + 3].astype(np.uint64)
                    )
                    sh = np.uint64(32) - b_off - np.uint64(w_val)
                    msk = np.uint64((1 << int(w_val)) - 1)
                    out_int[target_idx] = ((word >> sh) & msk).astype(np.int64) + np.repeat(
                        grefs[idxs_g], glens[idxs_g]
                    )

            if drt == 3 and order > 0 and len(init_vals) >= order and total_pts >= order:
                if order == 1:
                    d = out_int + gmin
                    d[0] = init_vals[0]
                    out_int = np.cumsum(d)
                elif order == 2:
                    d = out_int + gmin
                    d[0] = init_vals[0]
                    if total_pts > 1:
                        d[1] = init_vals[1] - init_vals[0]
                        d1 = np.cumsum(d[1:])
                        restored = np.empty_like(out_int)
                        restored[0] = init_vals[0]
                        restored[1:] = init_vals[0] + np.cumsum(d1)
                        out_int = restored
            vals = (ref_f + out_int.astype(np.float64) * bin_factor) / dec_factor
    else:
        raise ValueError(f"Unsupported GRIB2 Data Representation Template 5.{drt}")

    bmap_ind = s6[0] if len(s6) > 0 else 255
    total_grid = nx * ny
    if bmap_ind == 0:
        bits = np.unpackbits(np.frombuffer(s6[1:], dtype=np.uint8))[:total_grid].astype(bool)
        full = np.full(total_grid, np.nan, dtype=np.float32)
        n_valid = int(np.sum(bits))
        full[bits] = vals[:n_valid].astype(np.float32)
    else:
        full = np.full(total_grid, np.nan, dtype=np.float32)
        n_copy = min(total_grid, len(vals))
        full[:n_copy] = vals[:n_copy].astype(np.float32)
    return full.reshape((ny, nx))


def _http_get_bytes(url: str, byte_range: tuple[int, int] | None = None, timeout: float = 16.0) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        s_b, e_b = byte_range
        headers["Range"] = f"bytes={s_b}-{e_b}" if e_b >= s_b else f"bytes={s_b}-"
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                raise
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _parse_idx_entries(idx_text: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in idx_text.splitlines() if ln.strip()]
    entries: list[dict[str, Any]] = []
    for k, ln in enumerate(lines):
        parts = ln.split(":")
        if len(parts) < 6:
            continue
        start_b = int(parts[1])
        end_b = (int(lines[k + 1].split(":")[1]) - 1) if (k + 1 < len(lines)) else -1
        entries.append(
            {
                "var": parts[3],
                "level": parts[4],
                "key": f"{parts[3]}:{parts[4]}",
                "start": start_b,
                "end": end_b,
            }
        )
    return entries


def candidate_hrrr_cycles(
    target_utc: dt.datetime,
    eval_now_utc: dt.datetime,
    replay_mode: str = "auto",
    hindcast_lead_days: int = 0,
) -> list[tuple[dt.datetime, int]]:
    """
    Returns ordered candidate `(init_utc, forecast_hour)` pairs respecting strict causality
    and HRRR's `48h` synoptic (`00/06/12/18Z`) vs `18h` hourly (`01..23Z`) horizons.
    """
    t_u = (
        target_utc.replace(tzinfo=dt.timezone.utc)
        if target_utc.tzinfo is None
        else target_utc.astimezone(dt.timezone.utc)
    )
    n_u = (
        eval_now_utc.replace(tzinfo=dt.timezone.utc)
        if eval_now_utc.tzinfo is None
        else eval_now_utc.astimezone(dt.timezone.utc)
    )

    if hindcast_lead_days >= 2:
        return []

    target_round = dt.datetime(t_u.year, t_u.month, t_u.day, t_u.hour, 0, tzinfo=dt.timezone.utc)
    if t_u.minute >= 30:
        target_round += dt.timedelta(hours=1)

    # Zero-lead historical analysis (`truth` or `nwp_only`): use spun-up `f01` valid at `target_round`
    if replay_mode in ("truth", "nwp_only") and hindcast_lead_days == 0:
        c1 = target_round - dt.timedelta(hours=1)
        c2 = target_round - dt.timedelta(hours=2)
        return [(c1, 1), (c2, 2)]

    # True +24h historical hindcast (`hindcast_24h` / `hindcast_lead_days == 1`):
    # Strictly causal `init_utc <= target_utc - 24h`, requiring a 48h synoptic cycle (`00/06/12/18Z`)
    if replay_mode == "hindcast_24h" or hindcast_lead_days == 1:
        cutoff = t_u - dt.timedelta(hours=24)
        cand_hr = dt.datetime(cutoff.year, cutoff.month, cutoff.day, cutoff.hour, 0, tzinfo=dt.timezone.utc)
        out_24: list[tuple[dt.datetime, int]] = []
        for back_h in range(0, 18):
            init_dt = cand_hr - dt.timedelta(hours=back_h)
            if init_dt.hour not in (0, 6, 12, 18):
                continue
            fhr = int(round((target_round - init_dt).total_seconds() / 3600.0))
            if 24 <= fhr <= 48:
                out_24.append((init_dt, fhr))
            if len(out_24) >= 3:
                break
        return out_24

    # Causal live forecast or historical `+1h`/`+3h` nowcast:
    # Account for ~65 min operational S3 upload latency after `init_utc`
    lead_hr = (t_u - n_u).total_seconds() / 3600.0
    if lead_hr > 48.5 or lead_hr < -6.0:
        return []

    avail_cutoff = n_u - dt.timedelta(minutes=65)
    start_init = dt.datetime(
        avail_cutoff.year, avail_cutoff.month, avail_cutoff.day, avail_cutoff.hour, 0, tzinfo=dt.timezone.utc
    )
    out_live: list[tuple[dt.datetime, int]] = []
    for back_h in range(0, 14):
        init_dt = start_init - dt.timedelta(hours=back_h)
        max_fhr = 48 if (init_dt.hour in (0, 6, 12, 18)) else 18
        fhr = int(round((target_round - init_dt).total_seconds() / 3600.0))
        if fhr == 0:
            # Prefer `f01` from `init_dt - 1h` so microphysical cloud diagnostics are spun up
            continue
        if 1 <= fhr <= max_fhr:
            out_live.append((init_dt, fhr))
        if len(out_live) >= 4:
            break
    return out_live


# Desired `wrfsfcf` fields mapped to compact internal keys
_SFC_WANTED: dict[str, str] = {
    "HGT:surface": "sfc_hgt",
    "VIS:surface": "sfc_vis",
    "TMP:2 m above ground": "t2m_k",
    "DPT:2 m above ground": "dpt2m_k",
    "RH:2 m above ground": "rh2m",
    "UGRD:10 m above ground": "u10",
    "VGRD:10 m above ground": "v10",
    "CAPE:surface": "cape",
    "HPBL:surface": "hpbl",
    "AOTK:entire atmosphere (considered as a single layer)": "aotk",
    "TCOLW:entire atmosphere": "tcolw",
    "TCOLI:entire atmosphere": "tcoli",
    "TCDC:boundary layer cloud layer": "tcdc_bl",
    "LCDC:low cloud layer": "lcdc",
    "MCDC:middle cloud layer": "mcdc",
    "HCDC:high cloud layer": "hcdc",
    "TCDC:entire atmosphere": "tcdc_tot",
    "HGT:cloud ceiling": "ceil_asl",
    "HGT:cloud base": "cbase_asl",
    "HGT:cloud top": "ctop_asl",
}


def _build_wanted_prs_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for p in BL_PRESSURE_LEVELS_MB:
        m[f"HGT:{p} mb"] = f"z_{p}"
        m[f"TMP:{p} mb"] = f"t_{p}"
        m[f"RH:{p} mb"] = f"rh_{p}"
        m[f"CLMR:{p} mb"] = f"clmr_{p}"
    for p in (850, 700, 500, 300, 250):
        m[f"UGRD:{p} mb"] = f"u_{p}"
        m[f"VGRD:{p} mb"] = f"v_{p}"
    for p in (500, 300):
        m[f"HGT:{p} mb"] = f"z_{p}"
        m[f"TMP:{p} mb"] = f"t_{p}"
        m[f"RH:{p} mb"] = f"rh_{p}"
        m[f"CIMIXR:{p} mb"] = f"cimr_{p}"
    return m


_PRS_WANTED = _build_wanted_prs_map()


def _coalesce_ranges(
    matched: list[tuple[str, int, int]], max_gap_bytes: int = 65_000, max_span_bytes: int = 14_500_000
) -> list[tuple[int, int, list[tuple[str, int, int]]]]:
    """Groups nearby GRIB2 messages into minimal HTTP `Range:` requests."""
    if not matched:
        return []
    ordered = sorted(matched, key=lambda x: x[1])
    groups: list[tuple[int, int, list[tuple[str, int, int]]]] = []
    cur_s = ordered[0][1]
    cur_e = ordered[0][2]
    cur_items = [ordered[0]]

    for item in ordered[1:]:
        _, s_b, e_b = item
        if (
            cur_e > 0
            and e_b > 0
            and (s_b - cur_e) <= max_gap_bytes
            and (e_b - cur_s) <= max_span_bytes
        ):
            cur_e = max(cur_e, e_b)
            cur_items.append(item)
        else:
            groups.append((cur_s, cur_e, cur_items))
            cur_s, cur_e, cur_items = s_b, e_b, [item]
    groups.append((cur_s, cur_e, cur_items))
    return groups


def _fetch_and_crop_cycle_tile(
    init_utc: dt.datetime,
    fhr: int,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Downloads & decodes `wrfsfcf{fhr:02d}` + `wrfprsf{fhr:02d}` via coalesced HTTP `Range:` requests,
    cropping to the regional Lambert subgrid `[j0:j1, i0:i1]` and persisting a compact `.npz` cache.
    """
    ymd = init_utc.strftime("%Y%m%d")
    hh = init_utc.strftime("%H")
    tile_id = f"{ymd}_{hh}z_f{fhr:02d}_i{i0}_{i1}_j{j0}_{j1}"

    now_mono = time.monotonic()
    with _MEM_LOCK:
        if tile_id in _MEM_TILE_CACHE:
            ts, arrs, meta = _MEM_TILE_CACHE[tile_id]
            if now_mono - ts < _MEM_TTL_SECONDS:
                return arrs, meta
        t_lock = _TILE_LOCK_MAP.setdefault(tile_id, threading.Lock())

    with t_lock:
        with _MEM_LOCK:
            if tile_id in _MEM_TILE_CACHE:
                ts, arrs, meta = _MEM_TILE_CACHE[tile_id]
                if now_mono - ts < _MEM_TTL_SECONDS:
                    return arrs, meta

        HRRR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        npz_path = HRRR_CACHE_DIR / f"tile_v1_{tile_id}.npz"
        meta_path = HRRR_CACHE_DIR / f"tile_v1_{tile_id}.json"

        if npz_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                with np.load(npz_path) as npz_doc:
                    arrs = {k: np.array(npz_doc[k], dtype=np.float32) for k in npz_doc.files}
                if "sfc_hgt" in arrs and "z_950" in arrs:
                    with _MEM_LOCK:
                        _MEM_TILE_CACHE[tile_id] = (now_mono, arrs, meta)
                    return arrs, meta
            except Exception:
                pass

        t0 = time.perf_counter()
        sfc_base = f"{HRRR_S3_BASE}/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf{fhr:02d}.grib2"
        prs_base = f"{HRRR_S3_BASE}/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfprsf{fhr:02d}.grib2"

        with ThreadPoolExecutor(max_workers=2) as idx_pool:
            f_s_idx = idx_pool.submit(_http_get_bytes, sfc_base + ".idx", None, 12.0)
            f_p_idx = idx_pool.submit(_http_get_bytes, prs_base + ".idx", None, 12.0)
            sfc_entries = _parse_idx_entries(f_s_idx.result().decode("utf-8", errors="replace"))
            prs_entries = _parse_idx_entries(f_p_idx.result().decode("utf-8", errors="replace"))

        sfc_matched: list[tuple[str, int, int]] = []
        seen_sfc: set[str] = set()
        for e in sfc_entries:
            out_k = _SFC_WANTED.get(e["key"])
            if out_k and out_k not in seen_sfc:
                seen_sfc.add(out_k)
                sfc_matched.append((out_k, int(e["start"]), int(e["end"])))

        prs_matched: list[tuple[str, int, int]] = []
        seen_prs: set[str] = set()
        for e in prs_entries:
            out_k = _PRS_WANTED.get(e["key"])
            if out_k and out_k not in seen_prs:
                seen_prs.add(out_k)
                prs_matched.append((out_k, int(e["start"]), int(e["end"])))

        tasks: list[tuple[str, int, int, list[tuple[str, int, int]]]] = []
        for gs, ge, gitems in _coalesce_ranges(sfc_matched, max_gap_bytes=65_000, max_span_bytes=14_500_000):
            tasks.append((sfc_base, gs, ge, gitems))
        for gs, ge, gitems in _coalesce_ranges(prs_matched, max_gap_bytes=65_000, max_span_bytes=8_500_000):
            tasks.append((prs_base, gs, ge, gitems))

        cropped_fields: dict[str, np.ndarray] = {}
        bytes_downloaded = 0

        def _run_group(task_spec: tuple[str, int, int, list[tuple[str, int, int]]]) -> tuple[int, dict[str, np.ndarray]]:
            g_url, gs, ge, gitems = task_spec
            raw_block = _http_get_bytes(g_url, (gs, ge), timeout=22.0)
            local_out: dict[str, np.ndarray] = {}
            for out_k, ms, me in gitems:
                rel_s = ms - gs
                rel_e = (me - gs + 1) if me >= ms else len(raw_block)
                grid_2d = decode_grib2_message(raw_block[rel_s:rel_e])
                local_out[out_k] = np.array(grid_2d[j0:j1, i0:i1], dtype=np.float32, copy=True)
            return len(raw_block), local_out

        with ThreadPoolExecutor(max_workers=4) as dl_pool:
            for nb, partial in dl_pool.map(_run_group, tasks):
                bytes_downloaded += nb
                cropped_fields.update(partial)

        valid_utc = init_utc + dt.timedelta(hours=fhr)
        meta = {
            "init_utc": init_utc.isoformat(),
            "valid_utc": valid_utc.isoformat(),
            "cycle_tag": f"{ymd} {hh}Z f{fhr:02d}",
            "fhr": fhr,
            "i0": i0,
            "i1": i1,
            "j0": j0,
            "j1": j1,
            "bytes_downloaded": bytes_downloaded,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }

        try:
            np.savez_compressed(npz_path, **cropped_fields)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            pass

        with _MEM_LOCK:
            if len(_MEM_TILE_CACHE) >= 8:
                oldest_k = min(_MEM_TILE_CACHE, key=lambda k: _MEM_TILE_CACHE[k][0])
                _MEM_TILE_CACHE.pop(oldest_k, None)
            _MEM_TILE_CACHE[tile_id] = (now_mono, cropped_fields, meta)
        return cropped_fields, meta


def _sample_tile_bilinear(arr: np.ndarray | None, fi_local: float, fj_local: float) -> float:
    """NaN-aware bilinear sampler on a cropped HRRR tile."""
    if arr is None:
        return float("nan")
    ny, nx = arr.shape
    x = max(0.0, min(float(nx - 1), fi_local))
    y = max(0.0, min(float(ny - 1), fj_local))
    ix = min(nx - 2, int(math.floor(x))) if nx > 1 else 0
    iy = min(ny - 2, int(math.floor(y))) if ny > 1 else 0
    fx = x - ix
    fy = y - iy

    q00 = float(arr[iy, ix])
    q10 = float(arr[iy, min(nx - 1, ix + 1)])
    q01 = float(arr[min(ny - 1, iy + 1), ix])
    q11 = float(arr[min(ny - 1, iy + 1), min(nx - 1, ix + 1)])

    vals = [(q00, (1.0 - fx) * (1.0 - fy)), (q10, fx * (1.0 - fy)), (q01, (1.0 - fx) * fy), (q11, fx * fy)]
    w_sum = 0.0
    v_sum = 0.0
    for v, w in vals:
        if math.isfinite(v) and w > 0.0:
            w_sum += w
            v_sum += w * v
    if w_sum > 1e-6:
        return v_sum / w_sum
    # Nearest finite fallback in 3x3 neighborhood
    iy_n = int(round(y))
    ix_n = int(round(x))
    sub = arr[max(0, iy_n - 1) : min(ny, iy_n + 2), max(0, ix_n - 1) : min(nx, ix_n + 2)]
    fin = sub[np.isfinite(sub)]
    return float(np.median(fin)) if fin.size > 0 else float("nan")


def _analyze_hrrr_profile_at_ij(
    arrs: dict[str, np.ndarray],
    fi_loc: float,
    fj_loc: float,
) -> dict[str, Any]:
    """
    Extracts a complete radiosonde-equivalent thermodynamic & microphysical column
    from the 3 km HRRR 25-hPa `wrfprsf` + `wrfsfcf` grids at `(fi_loc, fj_loc)`.
    """
    def g(key: str, default: float = float("nan")) -> float:
        v = _sample_tile_bilinear(arrs.get(key), fi_loc, fj_loc)
        return v if math.isfinite(v) else default

    z_sfc = max(0.0, g("sfc_hgt", 0.0))
    t2m_c = g("t2m_k", 287.15) - 273.15
    dpt2m_c = g("dpt2m_k", t2m_c - 4.0) - 273.15
    rh2m = max(2.0, min(100.0, g("rh2m", 65.0)))
    vis_m = max(50.0, min(60_000.0, g("sfc_vis", 32_000.0)))
    hpbl_m = max(60.0, g("hpbl", 450.0))
    cape = max(0.0, g("cape", 0.0))
    u10 = g("u10", 0.0)
    v10 = g("v10", 0.0)
    aotk = g("aotk", float("nan"))
    tcolw = max(0.0, g("tcolw", 0.0))
    tcoli = max(0.0, g("tcoli", 0.0))

    tcdc_bl = max(0.0, min(100.0, g("tcdc_bl", 0.0)))
    lcdc = max(0.0, min(100.0, g("lcdc", 0.0)))
    mcdc = max(0.0, min(100.0, g("mcdc", 0.0)))
    hcdc = max(0.0, min(100.0, g("hcdc", 0.0)))
    tcdc_tot = max(0.0, min(100.0, g("tcdc_tot", max(lcdc, mcdc, hcdc))))

    ceil_asl = g("ceil_asl", float("nan"))
    cbase_asl = g("cbase_asl", float("nan"))
    ctop_asl = g("ctop_asl", float("nan"))

    # Build ascending above-ground 25 hPa boundary-layer levels (`0 .. ~3,100 m` ASL)
    levels: list[dict[str, float]] = [
        {
            "p_mb": 1013.0,
            "z_m": z_sfc + 2.0,
            "t_c": t2m_c,
            "rh": rh2m,
            "clmr_g_kg": 0.12 if (vis_m <= 900.0 and rh2m >= 95.0) else 0.0,
        }
    ]
    plev_map: dict[int, dict[str, float]] = {}
    for p in BL_PRESSURE_LEVELS_MB:
        zp = g(f"z_{p}", float("nan"))
        tp = g(f"t_{p}", float("nan")) - 273.15
        rhp = max(1.0, min(100.0, g(f"rh_{p}", 50.0)))
        clmr_g = max(0.0, g(f"clmr_{p}", 0.0)) * 1000.0
        if math.isfinite(zp) and math.isfinite(tp):
            rec = {"p_mb": float(p), "z_m": zp, "t_c": tp, "rh": rhp, "clmr_g_kg": clmr_g}
            plev_map[p] = rec
            if zp > z_sfc + 14.0:
                levels.append(rec)

    levels.sort(key=lambda r: r["z_m"])

    # 1. Radiosonde-identical continuous temperature-inversion detector (`raob_sounding.py` alignment)
    best_inv: dict[str, float] | None = None
    n_lv = len(levels)
    for i in range(n_lv - 1):
        zb = levels[i]["z_m"]
        tb = levels[i]["t_c"]
        agl_b = max(0.0, zb - z_sfc)
        if agl_b > 2700.0 or zb > 3300.0:
            break
        # Exclude 2 m cold sea-surface/grass skin contact layers unless surface fog is saturated (`raob_sounding.py` L206)
        if agl_b < 35.0 and not (rh2m >= 92.0 or vis_m <= 2500.0):
            continue
        if levels[i + 1]["t_c"] <= tb - 0.15:
            continue
        peak_idx = i
        peak_t = tb
        for j in range(i + 1, n_lv):
            zj = levels[j]["z_m"]
            tj = levels[j]["t_c"]
            if zj - zb > 1450.0:
                break
            if tj >= peak_t:
                peak_t = tj
                peak_idx = j
            elif peak_t - tj > 0.75:
                break
        strength_k = peak_t - tb
        if peak_idx > i and strength_k >= 0.45:
            zt = levels[peak_idx]["z_m"]
            score = strength_k * (1.25 if 75.0 <= agl_b <= 1700.0 else 0.75)
            if best_inv is None or score > best_inv["score"]:
                best_inv = {
                    "score": score,
                    "base_m": zb,
                    "top_m": zt,
                    "strength_k": strength_k,
                }

    # Sub-grid 25 hPa static-stability kink detector (for thin 0.6-1.2 K synoptic subsidence caps
    # smoothed across a ~230 m pressure slice into a < 3.6 K/km stable layer atop a >= 5.2 K/km mixed layer)
    static_stab_lid_m: float | None = None
    static_stab_strength_k = 0.0
    if best_inv is None and len(levels) >= 3:
        for k_idx in range(1, len(levels) - 1):
            lv_prev = levels[k_idx - 1]
            lv0 = levels[k_idx]
            lv1 = levels[k_idx + 1]
            if lv0["z_m"] < max(z_sfc + 160.0, 520.0) or lv0["z_m"] > 1650.0:
                continue
            dz_b = (lv0["z_m"] - lv_prev["z_m"]) / 1000.0
            dz_u = (lv1["z_m"] - lv0["z_m"]) / 1000.0
            if dz_b < 0.08 or dz_u < 0.08:
                continue
            lr_b = -(lv0["t_c"] - lv_prev["t_c"]) / dz_b
            lr_u = -(lv1["t_c"] - lv0["t_c"]) / dz_u
            if lr_b >= 5.2 and lr_u <= 3.6 and (lr_b - lr_u) >= 1.8:
                static_stab_lid_m = 0.35 * lv0["z_m"] + 0.65 * lv1["z_m"]
                static_stab_strength_k = round(min(2.2, (lr_b - lr_u) * 0.25), 2)
                break

    inv_base_m = (
        best_inv["base_m"]
        if best_inv
        else (
            (static_stab_lid_m - 90.0)
            if static_stab_lid_m is not None
            else (z_sfc + min(850.0, max(120.0, hpbl_m * 0.85)))
        )
    )
    inv_top_m = best_inv["top_m"] if best_inv else (inv_base_m + 240.0)
    inv_strength_k = best_inv["strength_k"] if best_inv else static_stab_strength_k

    # 2. Sub-inversion liquid cloud top (`CLMR` + `RH >= 88%` + uncontaminated `HGT:cloud top`)
    bl_lv = [lv for lv in levels if lv["z_m"] <= max(z_sfc + 2200.0, 2450.0)]
    clmr_lv_idxs = [idx for idx, lv in enumerate(bl_lv) if lv["clmr_g_kg"] >= 0.012]
    clmr_top_m: float | None = None
    clmr_base_m: float | None = None
    if clmr_lv_idxs:
        i_lo = clmr_lv_idxs[0]
        i_hi = clmr_lv_idxs[-1]
        clmr_base_m = bl_lv[i_lo]["z_m"]
        if i_lo > 0:
            clmr_base_m = 0.55 * bl_lv[i_lo]["z_m"] + 0.45 * bl_lv[i_lo - 1]["z_m"]
        if i_hi + 1 < len(bl_lv):
            c0 = bl_lv[i_hi]["clmr_g_kg"]
            c1 = bl_lv[i_hi + 1]["clmr_g_kg"]
            frac = min(0.75, max(0.15, c0 / max(0.02, c0 - c1 + 0.12)))
            clmr_top_m = bl_lv[i_hi]["z_m"] + frac * (bl_lv[i_hi + 1]["z_m"] - bl_lv[i_hi]["z_m"])
        else:
            clmr_top_m = bl_lv[i_hi]["z_m"] + 90.0

    upper_clear = (mcdc < 18.0) and (hcdc < 18.0)
    explicit_bl_top_m = (
        ctop_asl
        if (
            math.isfinite(ctop_asl)
            and upper_clear
            and (tcdc_bl >= 35.0 or lcdc >= 40.0)
            and (z_sfc + 55.0 <= ctop_asl <= z_sfc + 2450.0)
        )
        else None
    )
    explicit_bl_base_m = (
        cbase_asl
        if (math.isfinite(cbase_asl) and z_sfc + 15.0 <= cbase_asl <= z_sfc + 2100.0)
        else (
            ceil_asl
            if (math.isfinite(ceil_asl) and z_sfc + 15.0 <= ceil_asl <= z_sfc + 2100.0)
            else None
        )
    )

    # Saturated sub-inversion RH fallback (`RH >= 89%` above 40 m AGL, identical to `raob_sounding.py` line 233)
    lid_cap_search = (best_inv["base_m"] + 160.0) if best_inv else (z_sfc + 1450.0)
    sat_bl = [
        lv
        for lv in bl_lv
        if lv["z_m"] > z_sfc + 40.0 and lv["z_m"] <= lid_cap_search and lv["rh"] >= 89.0
    ]
    sat_rh_top_m = sat_bl[-1]["z_m"] if sat_bl else None
    if sat_rh_top_m is not None and best_inv is not None:
        sat_rh_top_m = max(sat_rh_top_m, min(best_inv["base_m"] + 45.0, sat_rh_top_m + 65.0))

    # Dry shallow boundary-layer hydrolapse detector (e.g. winter clear MBL capped at ~350 m where RH
    # drops >= 16% into < 48% dry subsiding air while 25-hPa temperature smoothing masks a thin cap)
    hydrolapse_lid_m: float | None = None
    if z_sfc < 400.0 and clmr_top_m is None and explicit_bl_top_m is None:
        for k_idx in range(len(bl_lv) - 1):
            lv0 = bl_lv[k_idx]
            lv1 = bl_lv[k_idx + 1]
            if lv0["z_m"] > z_sfc + 40.0 and lv0["z_m"] <= 560.0:
                if lv0["rh"] >= 54.0 and (lv0["rh"] - lv1["rh"]) >= 16.0 and lv1["rh"] < 48.0:
                    hydrolapse_lid_m = 0.72 * lv0["z_m"] + 0.28 * lv1["z_m"]
                    break

    if explicit_bl_top_m is not None and clmr_top_m is not None:
        lid_asl_m = 0.62 * explicit_bl_top_m + 0.38 * clmr_top_m
    elif explicit_bl_top_m is not None:
        lid_asl_m = explicit_bl_top_m
    elif clmr_top_m is not None:
        lid_asl_m = clmr_top_m
    elif sat_rh_top_m is not None:
        lid_asl_m = sat_rh_top_m
    elif hydrolapse_lid_m is not None and (
        best_inv is None or (inv_base_m > 850.0 and inv_strength_k < 2.2)
    ):
        lid_asl_m = hydrolapse_lid_m
    elif best_inv is not None and inv_strength_k >= 0.55:
        # Exact dry-inversion lid definition matching `raob_sounding.py` / `backtest.py`!
        lid_asl_m = inv_base_m + 0.18 * max(0.0, inv_top_m - inv_base_m)
    elif static_stab_lid_m is not None:
        lid_asl_m = static_stab_lid_m
    elif z_sfc < 600.0 and 925 in plev_map:
        # Deep winter/synoptic mixed boundary layer without a shallow marine cap (`nwp_corridor.py` L274)
        lid_asl_m = max(z_sfc + max(220.0, hpbl_m), min(1450.0, plev_map[925]["z_m"] + 0.72 * hpbl_m))
    else:
        lid_asl_m = z_sfc + min(1350.0, max(220.0, hpbl_m))

    lcl_asl_m = z_sfc + max(35.0, 125.0 * max(0.15, t2m_c - dpt2m_c))
    if explicit_bl_base_m is not None and explicit_bl_base_m < lid_asl_m - 35.0:
        stratus_base_m = explicit_bl_base_m
    elif clmr_base_m is not None and clmr_base_m < lid_asl_m - 35.0:
        stratus_base_m = max(z_sfc + 25.0, clmr_base_m)
    else:
        stratus_base_m = max(z_sfc + 30.0, min(lid_asl_m - 95.0, lcl_asl_m))

    # Upper-tropospheric levels & vector winds
    z500 = g("z_500", 5680.0)
    t500_c = g("t_500", 257.15) - 273.15
    rh500 = max(1.0, min(100.0, g("rh_500", 30.0)))
    z300 = g("z_300", 9380.0)
    t300_c = g("t_300", 231.15) - 273.15
    rh300 = max(1.0, min(100.0, g("rh_300", 35.0)))
    cimr_hi_g = (max(0.0, g("cimr_500", 0.0)) + max(0.0, g("cimr_300", 0.0))) * 1000.0

    u850, v850 = g("u_850", u10 * 1.3), g("v_850", v10 * 1.3)
    u700, v700 = g("u_700", u850 * 1.25), g("v_700", v850 * 1.25)
    u500, v500 = g("u_500", u700 * 1.35), g("v_500", v700 * 1.35)
    u300, v300 = g("u_300", u500 * 1.35), g("v_300", v500 * 1.35)
    u250, v250 = g("u_250", u300 * 1.08), g("v_250", v300 * 1.08)

    def _p_val(p_key: int, field: str, fb: float) -> float:
        return plev_map[p_key][field] if p_key in plev_map else fb

    return {
        "z_sfc_m": round(z_sfc, 1),
        "temp_2m_c": round(t2m_c, 2),
        "dewpoint_2m_c": round(dpt2m_c, 2),
        "rh_2m_pct": round(rh2m, 1),
        "vis_m": round(vis_m, 0),
        "hpbl_m": round(hpbl_m, 1),
        "cape_j_kg": round(cape, 1),
        "aotk_550nm": round(aotk, 3) if (math.isfinite(aotk) and 0.005 <= aotk <= 3.5) else None,
        "tcolw_kg_m2": round(tcolw, 4),
        "tcoli_kg_m2": round(tcoli, 4),
        "cimr_hi_g_kg": round(cimr_hi_g, 4),
        "tcdc_bl_pct": round(tcdc_bl, 1),
        "lcdc_pct": round(lcdc, 1),
        "mcdc_pct": round(mcdc, 1),
        "hcdc_pct": round(hcdc, 1),
        "tcdc_tot_pct": round(tcdc_tot, 1),
        "ceil_asl_m": round(ceil_asl, 1) if math.isfinite(ceil_asl) else None,
        "cbase_asl_m": round(cbase_asl, 1) if math.isfinite(cbase_asl) else None,
        "ctop_asl_m": round(ctop_asl, 1) if math.isfinite(ctop_asl) else None,
        "has_inversion": best_inv is not None,
        "inv_base_asl_m": round(inv_base_m, 1),
        "inv_top_asl_m": round(inv_top_m, 1),
        "inv_strength_k": round(inv_strength_k, 2),
        "lid_asl_m": round(lid_asl_m, 1),
        "stratus_base_asl_m": round(stratus_base_m, 1),
        "lcl_asl_m": round(lcl_asl_m, 1),
        "max_bl_clmr_g_kg": round(max((lv["clmr_g_kg"] for lv in bl_lv), default=0.0), 4),
        "rh_975_pct": round(_p_val(975, "rh", rh2m * 0.92), 1),
        "rh_950_pct": round(_p_val(950, "rh", rh2m * 0.85), 1),
        "rh_925_pct": round(_p_val(925, "rh", 55.0), 1),
        "temp_950_c": round(_p_val(950, "t_c", t2m_c - 2.0), 2),
        "temp_925_c": round(_p_val(925, "t_c", t2m_c - 3.5), 2),
        "temp_850_c": round(_p_val(850, "t_c", t2m_c - 7.5), 2),
        "temp_700_c": round(_p_val(700, "t_c", t500_c + 15.0), 2),
        "temp_500_c": round(t500_c, 2),
        "temp_300_c": round(t300_c, 2),
        "z_925_m": round(_p_val(925, "z_m", 760.0), 1),
        "z_850_m": round(_p_val(850, "z_m", 1480.0), 1),
        "z_700_m": round(_p_val(700, "z_m", 3080.0), 1),
        "z_500_m": round(z500, 1),
        "z_300_m": round(z300, 1),
        "rh_700_pct": round(_p_val(700, "rh", 35.0), 1),
        "rh_500_pct": round(rh500, 1),
        "rh_300_pct": round(rh300, 1),
        "u10_ms": round(u10, 2),
        "v10_ms": round(v10, 2),
        "u850_ms": round(u850, 2),
        "v850_ms": round(v850, 2),
        "u700_ms": round(u700, 2),
        "v700_ms": round(v700, 2),
        "u500_ms": round(u500, 2),
        "v500_ms": round(v500, 2),
        "u300_ms": round(u300, 2),
        "v300_ms": round(v300, 2),
        "u250_ms": round(u250, 2),
        "v250_ms": round(v250, 2),
    }


def fetch_hrrr_s3_corridor(
    lat: float,
    lon: float,
    coords_meta: list[tuple[str, float, float, float, float | None]],
    target_utc: dt.datetime,
    eval_now_utc: dt.datetime,
    is_coastal: bool = False,
    replay_mode: str = "auto",
    hindcast_lead_days: int = 0,
) -> dict[str, Any] | None:
    """
    Fetches native 3 km NOAA HRRR S3 soundings across all `coords_meta` points (`0..320 km`)
    plus an undisturbed offshore marine reference column (when `is_coastal=True`) and a
    `64x64` local 3 km spatial cloud/ceiling patch around `(lat, lon)`.
    """
    if not is_in_hrrr_conus(lat, lon):
        return None

    cands = candidate_hrrr_cycles(target_utc, eval_now_utc, replay_mode, hindcast_lead_days)
    if not cands:
        return None

    # Compute Lambert `(i, j)` bounding box covering all `coords_meta` + 100 km local halo
    ij_pts: list[tuple[float, float]] = [hrrr_latlon_to_ij(lat, lon)]
    for _, plat, plon, _, _ in coords_meta:
        ij_pts.append(hrrr_latlon_to_ij(plat, plon))

    min_i = max(0, int(math.floor(min(p[0] for p in ij_pts))) - 36)
    max_i = min(HRRR_NX, int(math.ceil(max(p[0] for p in ij_pts))) + 36)
    min_j = max(0, int(math.floor(min(p[1] for p in ij_pts))) - 36)
    max_j = min(HRRR_NY, int(math.ceil(max(p[1] for p in ij_pts))) + 36)

    # Quantize tile bounds to multiples of 128 grid cells (`384 km`) so all azimuths & nearby landmarks share `.npz` tiles!
    tile_i0 = max(0, (min_i // 128) * 128)
    tile_i1 = min(HRRR_NX, ((max_i + 127) // 128) * 128)
    tile_j0 = max(0, (min_j // 128) * 128)
    tile_j1 = min(HRRR_NY, ((max_j + 127) // 128) * 128)

    arrs: dict[str, np.ndarray] | None = None
    meta: dict[str, Any] | None = None
    for init_dt, fhr in cands:
        try:
            arrs, meta = _fetch_and_crop_cycle_tile(init_dt, fhr, tile_i0, tile_i1, tile_j0, tile_j1)
            break
        except Exception:
            continue

    if arrs is None or meta is None:
        return None

    stations: list[dict[str, Any]] = []
    for name, plat, plon, pdist_km, paz_deg in coords_meta:
        fi, fj = hrrr_latlon_to_ij(plat, plon)
        col = _analyze_hrrr_profile_at_ij(arrs, fi - tile_i0, fj - tile_j0)
        col["name"] = name
        col["lat"] = plat
        col["lon"] = plon
        col["distance_km"] = pdist_km
        col["azimuth_deg"] = paz_deg
        stations.append(col)

    # For coastal sites, also sample the nearest open-ocean column (`HGT:surface < 20 m`, 6-18 km West)
    # where `1000/975/950 mb` are all in free air above sea level without coastal-cliff terrain smearing.
    marine_ref: dict[str, Any] | None = None
    if is_coastal:
        fi0, fj0 = hrrr_latlon_to_ij(lat, lon)
        sfc_grid = arrs.get("sfc_hgt")
        best_ocean_ij: tuple[float, float] | None = None
        for step_w in (2.0, 3.0, 4.0, 5.0, 6.0):
            cand_i = fi0 - tile_i0 - step_w
            cand_j = fj0 - tile_j0
            if sfc_grid is not None and 1.0 <= cand_i < sfc_grid.shape[1] - 1.0:
                h_val = _sample_tile_bilinear(sfc_grid, cand_i, cand_j)
                if math.isfinite(h_val) and h_val < 18.0:
                    best_ocean_ij = (cand_i, cand_j)
                    break
        if best_ocean_ij is not None:
            marine_ref = _analyze_hrrr_profile_at_ij(arrs, best_ocean_ij[0], best_ocean_ij[1])

    return {
        "ok": True,
        "provider": "NOAA HRRR 3km Native S3 (GRIB2 .idx)",
        "cycle_tag": meta["cycle_tag"],
        "init_utc": meta["init_utc"],
        "valid_utc": meta["valid_utc"],
        "fhr": meta["fhr"],
        "bytes_downloaded": meta["bytes_downloaded"],
        "elapsed_ms": meta["elapsed_ms"],
        "stations": stations,
        "marine_ref": marine_ref,
    }


def apply_hrrr_column_to_sounding(
    base_sounding: dict[str, Any],
    hrrr_col: dict[str, Any],
    *,
    marine_ref: dict[str, Any] | None = None,
    is_coastal: bool = False,
    target_elev_m: float | None = None,
    blend_weight: float = 1.0,
) -> dict[str, Any]:
    """
    Fuses a native 3 km HRRR thermodynamic + microphysical (`CLMR` / `TCOLW` / `HGT:cloud top`)
    column into a PhotoPlanner `CorridorStationProfile` sounding dict (`blend_weight` in `[0, 1]`).
    """
    w = max(0.0, min(1.0, float(blend_weight)))
    if w <= 0.02:
        return base_sounding

    st = json.loads(json.dumps(base_sounding))
    rs = st["rust_station"]
    elev_m = float(target_elev_m if target_elev_m is not None else rs.get("surface_elevation_m", 0.0))

    # Over steep coastal cliffs/peaks (`0 km`), weight the open-ocean marine reference (`6-15 km` West)
    # smoothly by the 3 km grid-cell terrain elevation `z_sfc_m` when a moist marine boundary layer is present,
    # so underground `1000/975 mb` extrapolation over coastal ridges never corrupts the sea-level marine lid.
    if is_coastal and marine_ref is not None and elev_m < 1350.0:
        z_grid = float(hrrr_col["z_sfc_m"])
        has_offshore_mbl = (
            float(marine_ref["lcdc_pct"]) >= 14.0
            or float(marine_ref["tcdc_bl_pct"]) >= 14.0
            or float(marine_ref["rh_975_pct"]) >= 66.0
            or float(marine_ref["max_bl_clmr_g_kg"]) >= 0.008
        )
        w_mar = min(0.92, 0.55 + 0.37 * min(1.0, z_grid / 260.0)) if has_offshore_mbl else 0.32
        m_col = marine_ref if has_offshore_mbl else hrrr_col
        h_lid_m = w_mar * float(marine_ref["lid_asl_m"]) + (1.0 - w_mar) * float(hrrr_col["lid_asl_m"])
        h_inv_base_m = w_mar * float(marine_ref["inv_base_asl_m"]) + (1.0 - w_mar) * float(hrrr_col["inv_base_asl_m"])
        h_inv_strength_k = max(float(marine_ref["inv_strength_k"]), float(hrrr_col["inv_strength_k"]))
        # Coastal airport/shoreline stratus ceilings lift over land relative to cold offshore sea mist:
        land_ceil = hrrr_col.get("cbase_asl_m") or hrrr_col.get("ceil_asl_m")
        if land_ceil is not None and 40.0 <= float(land_ceil) <= h_lid_m - 45.0:
            h_stratus_base_m = float(land_ceil)
        elif z_grid < 320.0:
            land_lcl = max(45.0, min(h_lid_m - 85.0, float(hrrr_col["lcl_asl_m"])))
            h_stratus_base_m = 0.72 * land_lcl + 0.28 * float(marine_ref["stratus_base_asl_m"])
        else:
            h_stratus_base_m = max(75.0, min(h_lid_m - 120.0, 0.55 * float(marine_ref["stratus_base_asl_m"]) + 0.45 * (h_lid_m * 0.48)))
    else:
        m_col = hrrr_col
        h_lid_m = float(hrrr_col["lid_asl_m"])
        h_inv_base_m = float(hrrr_col["inv_base_asl_m"])
        h_inv_strength_k = float(hrrr_col["inv_strength_k"])
        h_stratus_base_m = float(hrrr_col["stratus_base_asl_m"])

    # Cloud layer percentages from HRRR native `LCDC`, `TCDC:boundary layer`, `MCDC`, `HCDC`
    h_low_pct = max(float(hrrr_col["lcdc_pct"]), float(hrrr_col["tcdc_bl_pct"]))
    if is_coastal and marine_ref is not None and elev_m < h_lid_m - 25.0:
        # Coastal headlands/bluffs sitting BELOW the marine inversion lid experience coastal stratus
        # even when the 3 km grid box straddles dry inland hills (`Pedro Point` / `Garrapata`).
        offshore_low = max(float(marine_ref["lcdc_pct"]), float(marine_ref["tcdc_bl_pct"]))
        h_low_pct = max(h_low_pct, 0.58 * offshore_low + 0.42 * h_low_pct)

    h_mid_pct = float(hrrr_col["mcdc_pct"])
    h_high_pct = float(hrrr_col["hcdc_pct"])

    st["temp_2m_c"] = round((1.0 - w) * float(st["temp_2m_c"]) + w * float(hrrr_col["temp_2m_c"]), 2)
    st["dewpoint_2m_c"] = round((1.0 - w) * float(st["dewpoint_2m_c"]) + w * float(hrrr_col["dewpoint_2m_c"]), 2)
    st["rh_2m_pct"] = round((1.0 - w) * float(st["rh_2m_pct"]) + w * float(hrrr_col["rh_2m_pct"]), 1)
    st["rh_975_pct"] = round((1.0 - w) * float(st["rh_975_pct"]) + w * float(m_col["rh_975_pct"]), 1)
    st["rh_950_pct"] = round((1.0 - w) * float(st["rh_950_pct"]) + w * float(m_col["rh_950_pct"]), 1)
    st["rh_925_pct"] = round((1.0 - w) * float(st["rh_925_pct"]) + w * float(m_col["rh_925_pct"]), 1)
    st["temp_950_c"] = round((1.0 - w) * float(st["temp_950_c"]) + w * float(m_col["temp_950_c"]), 2)
    st["temp_925_c"] = round((1.0 - w) * float(st["temp_925_c"]) + w * float(m_col["temp_925_c"]), 2)
    st["temp_850_c"] = round((1.0 - w) * float(st["temp_850_c"]) + w * float(m_col["temp_850_c"]), 2)

    st["visibility_m"] = round((1.0 - w) * float(st["visibility_m"]) + w * float(hrrr_col["vis_m"]), 0)
    st["pbl_agl_m"] = round((1.0 - w) * float(st["pbl_agl_m"]) + w * float(hrrr_col["hpbl_m"]), 1)
    st["cape_j_kg"] = round((1.0 - w) * float(st["cape_j_kg"]) + w * float(hrrr_col["cape_j_kg"]), 1)
    st["lcl_asl_m"] = round((1.0 - w) * float(st["lcl_asl_m"]) + w * float(hrrr_col["lcl_asl_m"]), 1)

    st["inv_base_asl_m"] = round((1.0 - w) * float(st["inv_base_asl_m"]) + w * h_inv_base_m, 1)
    st["inv_top_asl_m"] = round((1.0 - w) * float(st["inv_top_asl_m"]) + w * h_lid_m, 1)
    st["inv_jump_k"] = round((1.0 - w) * float(st["inv_jump_k"]) + w * h_inv_strength_k, 2)
    st["hrrr_native_s3"] = True
    st["hrrr_inv_strength_k"] = round(h_inv_strength_k, 2)
    st["hrrr_lid_asl_m"] = round(h_lid_m, 1)
    st["stratus_base_asl_m"] = round(h_stratus_base_m, 1)

    st["cloud_low_pct"] = int(round(max(0.0, min(100.0, (1.0 - w) * float(st["cloud_low_pct"]) + w * h_low_pct))))
    st["cloud_mid_pct"] = int(round(max(0.0, min(100.0, (1.0 - w) * float(st["cloud_mid_pct"]) + w * h_mid_pct))))
    st["cloud_high_pct"] = int(round(max(0.0, min(100.0, (1.0 - w) * float(st["cloud_high_pct"]) + w * h_high_pct))))

    # Blend 3D vector winds (m/s)
    for k_uv, (hu, hv) in (
        ("wind_sfc_uv_ms", (hrrr_col["u10_ms"], hrrr_col["v10_ms"])),
        ("wind_850_uv_ms", (hrrr_col["u850_ms"], hrrr_col["v850_ms"])),
        ("wind_500_uv_ms", (hrrr_col["u500_ms"], hrrr_col["v500_ms"])),
        ("wind_250_uv_ms", (hrrr_col["u250_ms"], hrrr_col["v250_ms"])),
    ):
        bu, bv = st[k_uv]
        st[k_uv] = (round((1.0 - w) * float(bu) + w * float(hu), 2), round((1.0 - w) * float(bv) + w * float(hv), 2))

    spd10_ms = math.hypot(*st["wind_sfc_uv_ms"])
    st["wind_10m_kmh"] = round(spd10_ms * 3.6, 1)
    if spd10_ms > 0.2:
        u_s, v_s = st["wind_sfc_uv_ms"]
        st["wind_10m_dir_deg"] = round((math.degrees(math.atan2(-u_s, -v_s)) + 360.0) % 360.0, 0)

    # Blend HRRR-Smoke 550 nm AOD when available
    if hrrr_col.get("aotk_550nm") is not None:
        rs["aod_550nm"] = round(
            (1.0 - 0.65 * w) * float(rs["aod_550nm"]) + (0.65 * w) * float(hrrr_col["aotk_550nm"]), 3
        )

    rs["surface_temp_c"] = st["temp_2m_c"]
    rs["inversion_base_asl_m"] = st["inv_base_asl_m"] if st["inv_jump_k"] > 0.2 else 0.0
    rs["inversion_top_asl_m"] = st["inv_top_asl_m"] if st["inv_jump_k"] > 0.2 else 0.0
    rs["inversion_strength_k"] = st["inv_jump_k"]
    rs["boundary_layer_top_asl_m"] = round(float(rs["surface_elevation_m"]) + float(st["pbl_agl_m"]), 1)

    # Reconstruct altitude-exact `CloudLayerBand`s using HRRR `CLMR`, `TCOLW` (`LWP`), `TCOLI` (`IWP`), & ceiling/top
    new_bands: list[dict[str, Any]] = []
    low_cf = st["cloud_low_pct"] / 100.0
    mid_cf = st["cloud_mid_pct"] / 100.0
    high_cf = st["cloud_high_pct"] / 100.0

    mbl_floor_m = 0.0 if (is_coastal and elev_m < 1350.0) else max(0.0, float(rs["surface_elevation_m"]))
    is_bl_stratus = (
        is_coastal
        and elev_m < 1350.0
        and (
            float(m_col["tcdc_bl_pct"]) >= 10.0
            or st["inv_jump_k"] >= 0.6
            or float(m_col["rh_975_pct"]) >= 78.0
            or float(st["rh_2m_pct"]) >= 88.0
        )
    ) or (
        elev_m < 950.0
        and (float(hrrr_col["tcdc_bl_pct"]) >= 12.0 or float(st["rh_2m_pct"]) >= 91.0)
        and st["inv_jump_k"] >= 0.6
    )

    if low_cf >= 0.03:
        if is_bl_stratus:
            top_m = max(mbl_floor_m + 135.0, float(st["inv_top_asl_m"]))
            base_m = max(mbl_floor_m + 30.0, min(top_m - 85.0, h_stratus_base_m))
            # Physical Slingo/Stephens LWP optical depth (`tau ~ 145 * LWP` for r_eff ~ 10.3 um)
            lwp_src = max(float(hrrr_col["tcolw_kg_m2"]), 0.65 * float(m_col["tcolw_kg_m2"]))
            tau_lwp = min(28.0, max(1.2, 145.0 * lwp_src)) if lwp_src >= 0.008 else min(24.0, 0.030 * (top_m - base_m))
            new_bands.append(
                {
                    "label": "fog",
                    "base_asl_m": round(base_m, 1),
                    "top_asl_m": round(top_m, 1),
                    "cloud_fraction": round(low_cf, 3),
                    "optical_depth_vertical": round(max(0.08, low_cf * tau_lwp), 2),
                    "aspect_ratio_h_over_d": 0.06,
                    "characteristic_cell_km": 13.0 if low_cf > 0.65 else 5.5,
                    "ice_fraction": 0.0,
                }
            )
        else:
            cbase = hrrr_col.get("cbase_asl_m")
            low_base = (
                float(cbase)
                if (cbase is not None and mbl_floor_m + 90.0 <= float(cbase) <= mbl_floor_m + 2600.0)
                else max(mbl_floor_m + 180.0, min(float(hrrr_col["z_850_m"]) + 250.0, float(st["lcl_asl_m"])))
            )
            conv_boost = min(1600.0, math.sqrt(max(0.0, float(st["cape_j_kg"]))) * 40.0)
            low_top = max(low_base + 360.0, float(hrrr_col["z_850_m"]) + 320.0 + conv_boost)
            lwp_v = float(hrrr_col["tcolw_kg_m2"])
            tau_cu = min(28.0, max(2.2, 115.0 * lwp_v)) if lwp_v >= 0.015 else min(26.0, 0.014 * (low_top - low_base))
            new_bands.append(
                {
                    "label": "low",
                    "base_asl_m": round(low_base, 1),
                    "top_asl_m": round(low_top, 1),
                    "cloud_fraction": round(low_cf, 3),
                    "optical_depth_vertical": round(max(0.08, low_cf * tau_cu), 2),
                    "aspect_ratio_h_over_d": round(min(1.05, 0.18 + math.sqrt(max(0.0, float(st["cape_j_kg"]))) * 0.022), 2),
                    "characteristic_cell_km": 2.5 if float(st["cape_j_kg"]) > 120.0 else 6.0,
                    "ice_fraction": round(max(0.0, min(1.0, (-float(st["temp_850_c"]) - 2.0) / 22.0)), 2),
                }
            )

    if mid_cf >= 0.03:
        z700 = float(hrrr_col["z_700_m"])
        z500 = float(hrrr_col["z_500_m"])
        mid_base = max(float(rs["surface_elevation_m"]) + 650.0, z700 - 200.0)
        mid_top = max(mid_base + 550.0, z500 + (250.0 if float(hrrr_col["rh_500_pct"]) >= 65.0 else -300.0))
        t_mid = 0.5 * (float(hrrr_col["temp_700_c"]) + float(hrrr_col["temp_500_c"]))
        ice_mid = max(0.0, min(1.0, (-t_mid - 4.0) / 28.0))
        new_bands.append(
            {
                "label": "mid",
                "base_asl_m": round(mid_base, 1),
                "top_asl_m": round(mid_top, 1),
                "cloud_fraction": round(mid_cf, 3),
                "optical_depth_vertical": round(max(0.06, mid_cf * min(16.0, 0.0052 * (mid_top - mid_base))), 2),
                "aspect_ratio_h_over_d": 0.20,
                "characteristic_cell_km": 6.5,
                "ice_fraction": round(ice_mid, 2),
            }
        )

    if high_cf >= 0.03:
        z300 = float(hrrr_col["z_300_m"])
        ctop_h = hrrr_col.get("ctop_asl_m")
        hi_top = (
            max(7800.0, min(13_200.0, float(ctop_h)))
            if (ctop_h is not None and float(ctop_h) >= 6800.0)
            else max(8200.0, z300 + 900.0)
        )
        hi_base = max(6600.0, min(hi_top - 1250.0, z300 - 950.0))
        iwp_v = float(hrrr_col["tcoli_kg_m2"])
        # Optically thin cirrus (`IWP ~ 0.015 kg/m^2 -> tau ~ 0.65`) vs thick anvil (`IWP ~ 0.20 -> tau ~ 7`)
        incloud_hi_tau = (
            max(0.42, min(11.0, 38.0 * iwp_v / max(0.20, high_cf)))
            if iwp_v >= 0.003
            else (0.55 + 1.40 * high_cf)
        )
        new_bands.append(
            {
                "label": "high",
                "base_asl_m": round(hi_base, 1),
                "top_asl_m": round(hi_top, 1),
                "cloud_fraction": round(high_cf, 3),
                "optical_depth_vertical": round(max(0.05, high_cf * incloud_hi_tau), 2),
                "aspect_ratio_h_over_d": 0.08,
                "characteristic_cell_km": 16.0,
                "ice_fraction": 1.0,
            }
        )

    rs["cloud_bands"] = new_bands
    return st
