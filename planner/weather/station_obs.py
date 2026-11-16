"""
Live NWS / FAA Surface Station Assimilation (`api.weather.gov` METAR & RAWS).

Fetches the nearest active surface weather stations around a target landmark or
coastal corridor point to anchor `0-3h` nowcasts with real-time observed:
- 2 m temperature, dewpoint, and relative humidity (true surface LCL cloud base)
- Nephelometer horizontal visibility (surface haze / marine mist extinction)
- Ceilometer cloud layers (`FEW`, `SCT`, `BKN`, `OVC`, `VV`) & exact base altitudes
- 10 m observed wind vector and sea-level / station pressure
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
OBS_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "nws_obs"
USER_AGENT = "PhotoPlanner/2.0 (landscape-optics-corridor; github.com/lucasbrambrink/photoplanner)"

_MEM_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MEM_TTL_SECONDS = 600.0  # 10 minutes

_CLOUD_AMOUNT_FRACTION: dict[str, float] = {
    "CLR": 0.0,
    "SKC": 0.0,
    "NCD": 0.0,
    "NSC": 0.02,
    "FEW": 0.18,
    "SCT": 0.42,
    "BKN": 0.76,
    "OVC": 0.96,
    "VV": 1.00,
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r_km * math.asin(min(1.0, math.sqrt(a)))


def _get_json(url: str, timeout: float = 5.5) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/geo+json, application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_val(prop: Any) -> float | None:
    if not isinstance(prop, dict):
        return None
    v = prop.get("value")
    if v is None:
        return None
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


# Reference ASOS stations for fast historical archive queries (IEM ASOS network)
_ASOS_ARCHIVE_STATIONS: list[tuple[str, str, float, float, float]] = [
    ("HAF", "Half Moon Bay Airport", 37.5134, -122.5012, 20.0),
    ("SFO", "San Francisco Intl", 37.6196, -122.3656, 4.0),
    ("OAK", "Oakland Intl", 37.7213, -122.2208, 3.0),
    ("STS", "Charles M. Schulz Sonoma", 38.5090, -122.8129, 38.0),
    ("MRY", "Monterey Regional", 36.5870, -121.8430, 78.0),
    ("SJC", "San Jose Intl", 37.3626, -121.9290, 19.0),
    ("MOD", "Modesto City-County", 37.6258, -120.9544, 30.0),
    ("MCE", "Merced Regional", 37.2847, -120.5139, 47.0),
    ("FAT", "Fresno Yosemite Intl", 36.7762, -119.7181, 102.0),
    ("MMH", "Mammoth Yosemite Airport", 37.6241, -118.8378, 2173.0),
    ("BIH", "Eastern Sierra Regional Bishop", 37.3731, -118.3636, 1257.0),
    ("TVL", "Lake Tahoe Airport", 38.8939, -119.9953, 1909.0),
    ("RNO", "Reno-Tahoe Intl", 39.4991, -119.7681, 1346.0),
    ("LAX", "Los Angeles Intl", 33.9425, -118.4081, 38.0),
    ("SBA", "Santa Barbara Municipal", 34.4262, -119.8404, 4.0),
    ("SAN", "San Diego Intl", 32.7336, -117.1897, 5.0),
    ("SEA", "Seattle-Tacoma Intl", 47.4490, -122.3093, 132.0),
    ("PDX", "Portland Intl", 45.5887, -122.5975, 9.0),
]


def fetch_historical_asos_observation(
    lat: float,
    lon: float,
    target_utc: dt.datetime,
    max_distance_km: float = 115.0,
) -> dict[str, Any] | None:
    """
    Fetches archived hourly/5-minute ASOS surface ceilometer, visibility, temperature,
    and RH observations nearest to `(lat, lon)` at `target_utc` from the Iowa
    Environmental Mesonet (`mesonet.agron.iastate.edu`) ASOS archive.
    """
    if target_utc.tzinfo is None:
        target_utc = target_utc.replace(tzinfo=dt.timezone.utc)
    else:
        target_utc = target_utc.astimezone(dt.timezone.utc)

    candidates = sorted(
        (
            (_haversine_km(lat, lon, slat, slon), sid, sname, slat, slon, selev)
            for sid, sname, slat, slon, selev in _ASOS_ARCHIVE_STATIONS
        ),
        key=lambda item: item[0],
    )
    candidates = [c for c in candidates if c[0] <= max_distance_km][:3]
    if not candidates:
        return None

    OBS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    day_start = (target_utc - dt.timedelta(hours=3)).date()
    day_end = (target_utc + dt.timedelta(hours=3)).date() + dt.timedelta(days=1)

    for dist_km, sid, sname, st_lat, st_lon, st_elev_m in candidates:
        cache_file = (
            OBS_CACHE_DIR
            / f"asos_v2_hist_{sid}_{target_utc.strftime('%Y%m%d_%H%M')}.json"
        )
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        url = (
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            f"?station={sid}&data=tmpf&data=dwpf&data=relh&data=vsby"
            f"&data=sknt&data=drct&data=alti"
            f"&data=skyc1&data=skyc2&data=skyc3&data=skyc4"
            f"&data=skyl1&data=skyl2&data=skyl3&data=skyl4"
            f"&year1={day_start.year}&month1={day_start.month}&day1={day_start.day}"
            f"&year2={day_end.year}&month2={day_end.month}&day2={day_end.day}"
            "&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                csv_text = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            continue

        lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip() and not ln.startswith("#")]
        if len(lines) < 2:
            continue
        headers = lines[0].split(",")
        best_row: dict[str, str] | None = None
        best_dt: dt.datetime | None = None
        best_abs_s = 1e12

        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) != len(headers):
                continue
            row = dict(zip(headers, parts))
            valid_str = row.get("valid", "")
            try:
                obs_dt = dt.datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(
                    tzinfo=dt.timezone.utc
                )
            except ValueError:
                continue
            diff_s = abs((obs_dt - target_utc).total_seconds())
            # Prefer rows that actually include sky cover or temperature within +/- 75 min
            has_sky = row.get("skyc1", "M") != "M"
            score_s = diff_s - (300.0 if has_sky else 0.0)
            if diff_s <= 5400.0 and score_s < best_abs_s:
                best_abs_s = score_s
                best_row = row
                best_dt = obs_dt

        if best_row is None or best_dt is None:
            continue

        def _fval(key: str) -> float | None:
            raw = best_row.get(key, "M") if best_row else "M"
            if raw in ("M", "T", "", None):
                return None
            try:
                return float(raw)
            except ValueError:
                return None

        tmpf = _fval("tmpf")
        dwpf = _fval("dwpf")
        temp_c = ((tmpf - 32.0) * 5.0 / 9.0) if tmpf is not None else None
        dew_c = ((dwpf - 32.0) * 5.0 / 9.0) if dwpf is not None else None
        rh_pct = _fval("relh")
        vsby_mi = _fval("vsby")
        vis_m = (vsby_mi * 1609.344) if vsby_mi is not None else None
        sknt = _fval("sknt")
        wind_ms = (sknt * 0.514444) if sknt is not None else None
        wind_dir = _fval("drct")
        alti_inhg = _fval("alti")
        pressure_hpa = (alti_inhg * 33.8639) if alti_inhg is not None else None

        parsed_layers: list[dict[str, Any]] = []
        low_frac = 0.0
        mid_frac = 0.0
        high_frac = 0.0
        lowest_ceiling_asl_m: float | None = None

        for idx_l in (1, 2, 3, 4):
            amt = (best_row.get(f"skyc{idx_l}", "M") or "M").strip().upper()
            if amt in ("M", ""):
                continue
            frac = _CLOUD_AMOUNT_FRACTION.get(amt, 0.0)
            skyl_ft = _fval(f"skyl{idx_l}")
            base_agl_m = (skyl_ft * 0.3048) if skyl_ft is not None else None
            base_asl_m = (st_elev_m + base_agl_m) if base_agl_m is not None else None
            parsed_layers.append(
                {
                    "amount": amt,
                    "cloud_fraction": frac,
                    "base_agl_m": round(base_agl_m, 1) if base_agl_m is not None else None,
                    "base_asl_m": round(base_asl_m, 1) if base_asl_m is not None else None,
                }
            )
            if frac >= 0.65 and base_asl_m is not None:
                if lowest_ceiling_asl_m is None or base_asl_m < lowest_ceiling_asl_m:
                    lowest_ceiling_asl_m = base_asl_m
            if base_agl_m is not None:
                if base_agl_m < 2200.0:
                    low_frac = max(low_frac, frac)
                elif base_agl_m < 6200.0:
                    mid_frac = max(mid_frac, frac)
                else:
                    high_frac = max(high_frac, frac)

        lcl_asl_m: float | None = None
        if temp_c is not None and dew_c is not None:
            lcl_asl_m = round(st_elev_m + max(40.0, 125.0 * max(0.0, temp_c - dew_c)), 1)

        # US ASOS visibility sensors cap unrestricted visibility at 10.0 SM (~16,093 m).
        # Only infer surface haze/smoke AOD when visibility is genuinely restricted (< 9.5 SM).
        aod_proxy: float | None = None
        if vis_m is not None and 2500.0 < vis_m < 15_250.0:
            beta_sfc = 3.912 / max(2500.0, vis_m)
            aod_proxy = round(max(0.04, min(0.85, beta_sfc * 1100.0)), 3)

        result: dict[str, Any] = {
            "station_id": f"K{sid}" if len(sid) == 3 else sid,
            "station_name": sname,
            "station_lat": round(st_lat, 4),
            "station_lon": round(st_lon, 4),
            "station_elevation_m": round(st_elev_m, 1),
            "distance_km": round(dist_km, 1),
            "timestamp_utc": best_dt.isoformat(),
            "age_minutes": round(abs((target_utc - best_dt).total_seconds()) / 60.0, 1),
            "text_description": "Archived ASOS Observation (IEM)",
            "temp_c": round(temp_c, 2) if temp_c is not None else None,
            "dewpoint_c": round(dew_c, 2) if dew_c is not None else None,
            "rh_pct": round(rh_pct, 1) if rh_pct is not None else None,
            "visibility_m": round(vis_m, 0) if vis_m is not None else None,
            "visibility_miles": round(vsby_mi, 2) if vsby_mi is not None else None,
            "wind_speed_ms": round(wind_ms, 2) if wind_ms is not None else None,
            "wind_dir_deg": round(wind_dir, 0) if wind_dir is not None else None,
            "pressure_hpa": round(pressure_hpa, 1) if pressure_hpa is not None else None,
            "lcl_asl_m": lcl_asl_m,
            "lowest_ceiling_asl_m": round(lowest_ceiling_asl_m, 1)
            if lowest_ceiling_asl_m is not None
            else None,
            "low_cloud_fraction": round(low_frac, 3),
            "mid_cloud_fraction": round(mid_frac, 3),
            "high_cloud_fraction": round(high_frac, 3),
            "aod_550nm_proxy": aod_proxy,
            "cloud_layers": parsed_layers,
        }
        try:
            cache_file.write_text(json.dumps(result), encoding="utf-8")
        except Exception:
            pass
        return result

    return None


def fetch_nearest_station_observation(
    lat: float,
    lon: float,
    max_distance_km: float = 85.0,
    target_utc: dt.datetime | None = None,
) -> dict[str, Any] | None:
    """
    Queries `api.weather.gov` for the closest reporting METAR/RAWS station within
    `max_distance_km` when live (`target_utc is None`), or queries the IEM ASOS
    historical archive when `target_utc` is more than 2.5 hours in the past.
    """
    if target_utc is not None:
        now_u = dt.datetime.now(dt.timezone.utc)
        t_u = (
            target_utc.replace(tzinfo=dt.timezone.utc)
            if target_utc.tzinfo is None
            else target_utc.astimezone(dt.timezone.utc)
        )
        if (now_u - t_u).total_seconds() > 9000.0:
            return fetch_historical_asos_observation(
                lat, lon, t_u, max_distance_km=max(max_distance_km, 115.0)
            )

    # Quick CONUS / Alaska / Hawaii bounding check for api.weather.gov
    if not (18.0 <= lat <= 72.0 and -170.0 <= lon <= -64.0):
        return None

    cache_key = f"nws:{lat:.2f}:{lon:.2f}"
    now_mono = time.monotonic()
    if cache_key in _MEM_CACHE:
        ts, cached = _MEM_CACHE[cache_key]
        if now_mono - ts < _MEM_TTL_SECONDS:
            return cached

    OBS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    disk_file = OBS_CACHE_DIR / f"obs_{lat:.2f}_{lon:.2f}.json"
    if disk_file.exists():
        age_s = time.time() - disk_file.stat().st_mtime
        if age_s < _MEM_TTL_SECONDS:
            try:
                cached = json.loads(disk_file.read_text(encoding="utf-8"))
                _MEM_CACHE[cache_key] = (now_mono, cached)
                return cached
            except Exception:
                pass

    try:
        pts = _get_json(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", timeout=5.0)
        stations_url = pts.get("properties", {}).get("observationStations")
        if not stations_url:
            return None

        st_collection = _get_json(stations_url, timeout=5.0)
        features = st_collection.get("features") or []
        now_utc = dt.datetime.now(dt.timezone.utc)

        for feat in features[:4]:
            props = feat.get("properties") or {}
            sid = props.get("stationIdentifier")
            if not sid:
                continue
            coords = (feat.get("geometry") or {}).get("coordinates") or [lon, lat]
            st_lon, st_lat = float(coords[0]), float(coords[1])
            dist_km = _haversine_km(lat, lon, st_lat, st_lon)
            if dist_km > max_distance_km:
                continue

            st_elev_m = _extract_val(props.get("elevation")) or 0.0
            try:
                obs_doc = _get_json(
                    f"https://api.weather.gov/stations/{sid}/observations/latest",
                    timeout=4.5,
                )
            except Exception:
                continue

            op = obs_doc.get("properties") or {}
            ts_str = op.get("timestamp")
            if not ts_str:
                continue
            try:
                obs_dt = dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            age_min = (now_utc - obs_dt).total_seconds() / 60.0
            if age_min > 180.0:
                continue

            temp_c = _extract_val(op.get("temperature"))
            dew_c = _extract_val(op.get("dewpoint"))
            rh_pct = _extract_val(op.get("relativeHumidity"))
            vis_m = _extract_val(op.get("visibility"))
            wind_dir = _extract_val(op.get("windDirection"))
            wind_val = _extract_val(op.get("windSpeed"))
            wind_unit = ((op.get("windSpeed") or {}).get("unitCode") or "").lower()
            if wind_val is not None:
                if "km_h" in wind_unit or "km/h" in wind_unit:
                    wind_ms = wind_val / 3.6
                elif "kt" in wind_unit or "knot" in wind_unit:
                    wind_ms = wind_val * 0.514444
                else:
                    wind_ms = wind_val
            else:
                wind_ms = None

            baro_pa = _extract_val(op.get("barometricPressure")) or _extract_val(
                op.get("seaLevelPressure")
            )
            pressure_hpa = (baro_pa / 100.0) if (baro_pa and baro_pa > 20000.0) else None

            # Parse ceilometer cloud layers (AGL above station -> ASL)
            raw_layers = op.get("cloudLayers") or []
            parsed_layers: list[dict[str, Any]] = []
            low_frac = 0.0
            mid_frac = 0.0
            high_frac = 0.0
            lowest_ceiling_asl_m: float | None = None

            for cl in raw_layers:
                amt = str(cl.get("amount") or "").upper().strip()
                frac = _CLOUD_AMOUNT_FRACTION.get(amt, 0.0)
                base_agl_m = _extract_val(cl.get("base"))
                base_asl_m = (st_elev_m + base_agl_m) if base_agl_m is not None else None
                parsed_layers.append(
                    {
                        "amount": amt,
                        "cloud_fraction": frac,
                        "base_agl_m": base_agl_m,
                        "base_asl_m": round(base_asl_m, 1) if base_asl_m is not None else None,
                    }
                )
                if frac >= 0.65 and base_asl_m is not None:
                    if lowest_ceiling_asl_m is None or base_asl_m < lowest_ceiling_asl_m:
                        lowest_ceiling_asl_m = base_asl_m

                if base_agl_m is None:
                    if amt in ("CLR", "SKC"):
                        pass
                elif base_agl_m < 2200.0:
                    low_frac = max(low_frac, frac)
                elif base_agl_m < 6200.0:
                    mid_frac = max(mid_frac, frac)
                else:
                    high_frac = max(high_frac, frac)

            if temp_c is None and vis_m is None and not parsed_layers:
                continue

            if rh_pct is None and temp_c is not None and dew_c is not None:
                # Magnus formula RH
                es = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
                e = 6.112 * math.exp((17.67 * dew_c) / (dew_c + 243.5))
                rh_pct = max(2.0, min(100.0, 100.0 * e / max(0.01, es)))

            lcl_asl_m: float | None = None
            if temp_c is not None and dew_c is not None:
                lcl_agl_m = max(40.0, 125.0 * max(0.0, temp_c - dew_c))
                lcl_asl_m = round(st_elev_m + lcl_agl_m, 1)

            vis_miles = round(vis_m / 1609.344, 2) if vis_m is not None else None
            # Koschmieder surface extinction proxy only when visibility is restricted (< 9.5 SM;
            # automated ASOS sensors report 10.0 SM for unlimited clear-air visibility).
            aod_proxy: float | None = None
            if vis_m is not None and 2500.0 < vis_m < 15_250.0:
                beta_sfc = 3.912 / max(2500.0, vis_m)
                aod_proxy = round(max(0.04, min(0.85, beta_sfc * 1100.0)), 3)

            result: dict[str, Any] = {
                "station_id": sid,
                "station_name": props.get("name") or sid,
                "station_lat": round(st_lat, 4),
                "station_lon": round(st_lon, 4),
                "station_elevation_m": round(st_elev_m, 1),
                "distance_km": round(dist_km, 1),
                "timestamp_utc": obs_dt.isoformat(),
                "age_minutes": round(max(0.0, age_min), 1),
                "text_description": op.get("textDescription") or "",
                "temp_c": round(temp_c, 2) if temp_c is not None else None,
                "dewpoint_c": round(dew_c, 2) if dew_c is not None else None,
                "rh_pct": round(rh_pct, 1) if rh_pct is not None else None,
                "visibility_m": round(vis_m, 0) if vis_m is not None else None,
                "visibility_miles": vis_miles,
                "wind_speed_ms": round(wind_ms, 2) if wind_ms is not None else None,
                "wind_dir_deg": round(wind_dir, 0) if wind_dir is not None else None,
                "pressure_hpa": round(pressure_hpa, 1) if pressure_hpa is not None else None,
                "lcl_asl_m": lcl_asl_m,
                "lowest_ceiling_asl_m": round(lowest_ceiling_asl_m, 1)
                if lowest_ceiling_asl_m is not None
                else None,
                "low_cloud_fraction": round(low_frac, 3),
                "mid_cloud_fraction": round(mid_frac, 3),
                "high_cloud_fraction": round(high_frac, 3),
                "aod_550nm_proxy": aod_proxy,
                "cloud_layers": parsed_layers,
            }

            _MEM_CACHE[cache_key] = (now_mono, result)
            try:
                disk_file.write_text(json.dumps(result), encoding="utf-8")
            except Exception:
                pass
            return result
    except Exception:
        return None

    return None
