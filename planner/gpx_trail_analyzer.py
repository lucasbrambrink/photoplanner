"""
Custom GPX / GeoJSON Trail Upload Analyzer (`planner/gpx_trail_analyzer.py`).

Implements Roadmap Phase 3 (Task 3.3):
  - Parses user-uploaded `.gpx` (XML `<trkpt>` / `<rtept>`) or `.geojson` (`LineString`,
    `MultiLineString`, `Feature`, `FeatureCollection`) tracks for any arbitrary location.
  - Computes haversine cumulative mileage, elevation gain/loss, loop vs. out-and-back
    classification (mirroring `pp-trails::effective_scan_len`), and exact **Tobler's Hiking
    Function** ascent/descent pacing (`v = 6 * exp(-3.5 * |slope + 0.05|)` km/h).
  - Solves live/historical 380 km curved-Earth corridor meteorology & calibrated
    `P10/P50/P90` transmittance at the route's peak/apex via `solve_weather_and_corridor`.
  - Intersects the route's 3D elevation profile with probabilistic `P10 / P50 / P90`
    inversion/fog-lid bands (`compute_trail_fog_breakout_mileposts`) to output exact
    **Breakout Mileposts**.
  - Emits 4-category non-prescriptive **Trail Nudges** (`HorizonOpening`, `LightNotchWindow`,
    `FogBreakout`, `HeadlampSafety`), **Time-Reversed Trailhead Departure Logistics**, and a
    self-contained **Offline Backcountry Field Card**.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import xml.etree.ElementTree as ET
from typing import Any

from .weather.corridor_solver import (
    _find_sunrise_sunset_golden_hours,
    solve_weather_and_corridor,
)
from .weather.nwp_corridor import local_tz_for_lon
from .weather.terrain_coupling import compute_trail_fog_breakout_mileposts


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_m = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r_m * math.asin(min(1.0, math.sqrt(a)))


def _strip_xml_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if "}" in tag else tag.lower()


def parse_gpx_xml(gpx_text: str) -> dict[str, Any]:
    """
    Parses a `.gpx` XML document into `{"name": str, "coords": [[lon, lat, elev_m], ...]}`.
    Supports `<trk>/<trkseg>/<trkpt>` and `<rte>/<rtept>` tracks with or without XML namespaces.
    """
    root = ET.fromstring(gpx_text.strip())
    name = "Uploaded GPX Trail"
    coords: list[list[float]] = []

    for elem in root.iter():
        tag = _strip_xml_ns(elem.tag)
        if tag == "name" and elem.text and elem.text.strip() and name == "Uploaded GPX Trail":
            name = elem.text.strip()
        elif tag in ("trkpt", "rtept"):
            lat_s = elem.attrib.get("lat")
            lon_s = elem.attrib.get("lon")
            if lat_s is None or lon_s is None:
                continue
            lat = float(lat_s)
            lon = float(lon_s)
            elev_m = 0.0
            for child in elem:
                if _strip_xml_ns(child.tag) == "ele" and child.text:
                    try:
                        elev_m = float(child.text.strip())
                    except ValueError:
                        elev_m = 0.0
                    break
            coords.append([lon, lat, elev_m])

    if len(coords) < 2:
        raise ValueError("GPX document contains fewer than 2 valid track points.")
    return {"name": name, "coords": coords}


def parse_geojson_trail(geojson_obj: dict[str, Any] | str) -> dict[str, Any]:
    """
    Parses GeoJSON (`LineString`, `MultiLineString`, `Feature`, or `FeatureCollection`)
    into `{"name": str, "coords": [[lon, lat, elev_m], ...]}`.
    """
    doc = json.loads(geojson_obj) if isinstance(geojson_obj, str) else geojson_obj
    if not isinstance(doc, dict):
        raise ValueError("GeoJSON payload must be a JSON object.")

    name = "Uploaded GeoJSON Trail"
    raw_lines: list[list[Any]] = []

    def _collect(obj: dict[str, Any]) -> None:
        nonlocal name
        gtype = str(obj.get("type", ""))
        props = obj.get("properties") or {}
        if isinstance(props, dict) and props.get("name") and name == "Uploaded GeoJSON Trail":
            name = str(props["name"])
        if gtype == "FeatureCollection":
            for feat in obj.get("features") or []:
                if isinstance(feat, dict):
                    _collect(feat)
        elif gtype == "Feature":
            geom = obj.get("geometry")
            if isinstance(geom, dict):
                _collect(geom)
        elif gtype == "LineString":
            raw_lines.append(obj.get("coordinates") or [])
        elif gtype == "MultiLineString":
            for line in obj.get("coordinates") or []:
                raw_lines.append(line)

    _collect(doc)
    coords: list[list[float]] = []
    for line in raw_lines:
        for pt in line:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                lon = float(pt[0])
                lat = float(pt[1])
                elev_m = float(pt[2]) if len(pt) >= 3 and pt[2] is not None else 0.0
                coords.append([lon, lat, elev_m])

    if len(coords) < 2:
        raise ValueError("GeoJSON payload contains fewer than 2 valid LineString coordinates.")
    return {"name": name, "coords": coords}


def compute_tobler_trail_profile(coords: list[list[float]]) -> dict[str, Any]:
    """
    Computes cumulative mileage, elevation gain/loss, loop vs. out-and-back topology,
    apex coordinates, and Tobler's hiking-function ascent/descent durations (`v = 6 * exp(-3.5 * |s + 0.05|)` km/h).
    """
    n = len(coords)
    cum_m = [0.0] * n
    gain_m = 0.0
    loss_m = 0.0
    tobler_min_cum = [0.0] * n

    for i in range(1, n):
        d_m = _haversine_m(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0])
        dz_m = coords[i][2] - coords[i - 1][2]
        cum_m[i] = cum_m[i - 1] + d_m
        if dz_m > 0:
            gain_m += dz_m
        else:
            loss_m += -dz_m

        if d_m > 0.5:
            slope = max(-0.65, min(0.65, dz_m / d_m))
            speed_kmh = max(0.8, 6.0 * math.exp(-3.5 * abs(slope + 0.05)))
            seg_min = (d_m / 1000.0) / speed_kmh * 60.0
        else:
            seg_min = 0.0
        tobler_min_cum[i] = tobler_min_cum[i - 1] + seg_min

    total_m = cum_m[-1]
    total_mi = total_m / 1609.344

    # Detect loop vs. doubled out-and-back vs. one-way (matching pp-trails::effective_scan_len)
    start_end_m = _haversine_m(coords[0][1], coords[0][0], coords[-1][1], coords[-1][0])
    is_closed = total_m > 300.0 and start_end_m < max(120.0, 0.08 * total_m)
    is_doubled_oab = False
    if is_closed and n >= 8:
        q1 = coords[n // 4]
        q3 = coords[(3 * n) // 4]
        if _haversine_m(q1[1], q1[0], q3[1], q3[0]) < max(90.0, 0.06 * total_m):
            is_doubled_oab = True

    topology = (
        "doubled_out_and_back"
        if is_doubled_oab
        else ("loop" if is_closed else "one_way_or_out_and_back")
    )

    apex_idx = max(range(n), key=lambda i: coords[i][2])
    apex_pt = coords[apex_idx]
    ascent_to_apex_min = tobler_min_cum[apex_idx]
    if is_closed:
        return_from_apex_min = max(5.0, tobler_min_cum[-1] - ascent_to_apex_min)
    else:
        # Estimate reverse-leg Tobler descent back to trailhead
        desc_min = 0.0
        for i in range(apex_idx, 0, -1):
            d_m = _haversine_m(coords[i][1], coords[i][0], coords[i - 1][1], coords[i - 1][0])
            dz_m = coords[i - 1][2] - coords[i][2]
            if d_m > 0.5:
                slope = max(-0.65, min(0.65, dz_m / d_m))
                speed_kmh = max(0.8, 6.0 * math.exp(-3.5 * abs(slope + 0.05)))
                desc_min += (d_m / 1000.0) / speed_kmh * 60.0
        return_from_apex_min = max(5.0, desc_min)

    return {
        "n_points": n,
        "topology": topology,
        "total_distance_miles": round(total_mi, 2),
        "elevation_gain_ft": int(round(gain_m * 3.28084)),
        "elevation_loss_ft": int(round(loss_m * 3.28084)),
        "min_elev_ft": int(round(min(c[2] for c in coords) * 3.28084)),
        "max_elev_ft": int(round(apex_pt[2] * 3.28084)),
        "trailhead": {
            "lat": round(coords[0][1], 5),
            "lon": round(coords[0][0], 5),
            "elev_m": round(coords[0][2], 1),
            "elev_ft": int(round(coords[0][2] * 3.28084)),
        },
        "apex": {
            "index": apex_idx,
            "lat": round(apex_pt[1], 5),
            "lon": round(apex_pt[0], 5),
            "elev_m": round(apex_pt[2], 1),
            "elev_ft": int(round(apex_pt[2] * 3.28084)),
            "milepost": round(cum_m[apex_idx] / 1609.344, 2),
        },
        "tobler_ascent_to_apex_min": int(round(ascent_to_apex_min)),
        "tobler_return_from_apex_min": int(round(return_from_apex_min)),
        "tobler_total_roundtrip_min": int(round(ascent_to_apex_min + return_from_apex_min)),
    }


def _fmt_local_minutes(mins: int) -> str:
    m_mod = mins % (24 * 60)
    return f"{m_mod // 60:02d}:{m_mod % 60:02d}"


def build_trail_nudges_from_analysis(
    profile: dict[str, Any],
    breakout: dict[str, Any],
    weather_doc: dict[str, Any],
    phase: str,
    event_mins_local: int,
    setup_buffer_min: int = 25,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Generates the 4 non-prescriptive trail nudge categories (`HorizonOpening`,
    `LightNotchWindow`, `FogBreakout`, `HeadlampSafety`) plus time-reversed
    departure & return logistics.
    """
    nudges: list[dict[str, Any]] = []
    apex = profile["apex"]
    c_ray = weather_doc.get("corridor_raycast") or {}
    inv = weather_doc.get("inversion_model") or {}
    cal = weather_doc.get("calibrated_probabilities") or {}

    is_sunrise = phase.lower().startswith("sunrise")
    cal_p50 = float(cal.get("transmittance_p50_pct", c_ray.get("expected_transmittance_pct", 0.0)))
    cal_p10 = float(cal.get("transmittance_p10_pct", weather_doc.get("transmittance_p10_pct", 0.0)))
    cal_p90 = float(cal.get("transmittance_p90_pct", weather_doc.get("transmittance_p90_pct", 100.0)))
    terr_hor = float(c_ray.get("terrain_horizon_deg", 0.0))
    sun_el = float(weather_doc.get("sun_elevation_deg", 1.0))

    # 1. FogBreakout nudge
    p50_bk = breakout.get("expected_breakout_p50")
    p90_bk = breakout.get("confident_breakout_p90")
    above_prob = float(inv.get("summit_above_fog_prob_pct", 50.0))
    if p50_bk is not None:
        p50_elev_ft = int(round(float(p50_bk.get("elevation_ft", p50_bk.get("elev_ft", 0)))))
        p90_elev_ft = int(round(float(p90_bk.get("elevation_ft", p90_bk.get("elev_ft", 0))))) if p90_bk is not None else None
        p90_clause = (
            f" Confident (P90) breakout above {p90_elev_ft:,} ft at Mile {p90_bk['milepost']}."
            if p90_bk is not None and p90_elev_ft is not None
            else ""
        )
        nudges.append(
            {
                "category": "FogBreakout",
                "milepost": p50_bk["milepost"],
                "elev_ft": p50_elev_ft,
                "title": f"Inversion Breakout around Mile {p50_bk['milepost']}",
                "detail": (
                    f"Trail climbs through expected (P50) fog lid ({p50_elev_ft:,} ft ASL) at Mile {p50_bk['milepost']}; "
                    f"apex has {above_prob:.0f}% cloud-sea breakout odds.{p90_clause}"
                ),
            }
        )
    elif inv.get("breakout_verdict") == "IMMERSED_IN_FOG":
        nudges.append(
            {
                "category": "FogBreakout",
                "milepost": apex["milepost"],
                "elev_ft": apex["elev_ft"],
                "title": "Fog Immersion Risk Across Entire Route",
                "detail": (
                    f"Highest point ({apex['elev_ft']:,} ft) sits below expected fog lid "
                    f"({inv.get('fog_top_p50_ft', 0):,} ft ASL; only {above_prob:.0f}% breakout odds)."
                ),
            }
        )

    # 2. HorizonOpening / LightNotchWindow nudges
    if terr_hor > sun_el + 1.5:
        nudges.append(
            {
                "category": "HorizonOpening",
                "milepost": apex["milepost"],
                "elev_ft": apex["elev_ft"],
                "title": f"Early Terrain Horizon Cutoff ({terr_hor:.1f}°)",
                "detail": (
                    f"Distant ridge shadows the low sun ({sun_el:.1f}°) along the {weather_doc.get('corridor_direction', 'W')} "
                    f"corridor; plan primary compositions 20–35 min earlier while sun clears {terr_hor:.1f}°."
                ),
            }
        )
    else:
        nudges.append(
            {
                "category": "LightNotchWindow",
                "milepost": apex["milepost"],
                "elev_ft": apex["elev_ft"],
                "title": f"Apex Light Window ({_fmt_local_minutes(event_mins_local)} local)",
                "detail": (
                    f"Unobstructed {weather_doc.get('corridor_direction', 'W')} horizon ({terr_hor:.1f}°) at Mile {apex['milepost']}. "
                    f"Calibrated direct-beam transmittance: {cal_p50:.0f}% (P10–P90: {cal_p10:.0f}–{cal_p90:.0f}%), "
                    f"Alpenglow {weather_doc.get('alpenglow_score', 0)}/100."
                ),
            }
        )

    # 3. Time-Reversed Departure & HeadlampSafety nudge
    ascent_min = int(profile["tobler_ascent_to_apex_min"])
    ret_min = int(profile["tobler_return_from_apex_min"])
    # Arrive setup_buffer_min before peak golden-hour window
    target_arrival_min = event_mins_local - (setup_buffer_min if is_sunrise else (setup_buffer_min + 20))
    depart_min = target_arrival_min - ascent_min
    return_finish_min = (event_mins_local + (35 if not is_sunrise else 60)) + ret_min
    needs_headlamp = bool(is_sunrise or return_finish_min > event_mins_local + 25)

    nudges.append(
        {
            "category": "HeadlampSafety",
            "milepost": 0.0 if is_sunrise else apex["milepost"],
            "elev_ft": profile["trailhead"]["elev_ft"] if is_sunrise else apex["elev_ft"],
            "title": (
                f"Pre-Dawn Headlamp Ascent — Depart {_fmt_local_minutes(depart_min)}"
                if is_sunrise
                else f"Post-Sunset Headlamp Descent — Back ~{_fmt_local_minutes(return_finish_min)}"
            ),
            "detail": (
                f"Tobler ascent is {ascent_min} min ({profile['elevation_gain_ft']:,} ft gain) + {setup_buffer_min} min tripod setup; "
                f"depart trailhead by {_fmt_local_minutes(depart_min)} local. Pack headlamp for dark "
                f"{'approach' if is_sunrise else 'descent (~' + str(ret_min) + ' min)'}."
            ),
        }
    )

    logistics = {
        "phase": "sunrise" if is_sunrise else "sunset",
        "golden_hour_anchor_local": _fmt_local_minutes(event_mins_local),
        "recommended_trailhead_departure_local": _fmt_local_minutes(depart_min),
        "target_apex_arrival_local": _fmt_local_minutes(target_arrival_min),
        "estimated_trailhead_return_local": _fmt_local_minutes(return_finish_min),
        "setup_buffer_min": int(setup_buffer_min),
        "headlamp_required": needs_headlamp,
    }
    return nudges, logistics


def analyze_uploaded_trail(
    *,
    gpx_xml: str | None = None,
    geojson: dict[str, Any] | str | None = None,
    date_str: str | None = None,
    phase: str = "sunset",
    setup_buffer_min: int = 25,
    replay_mode: str = "auto",
    include_weather: bool = True,
) -> dict[str, Any]:
    """
    End-to-end Roadmap Task 3.3 pipeline for an uploaded `.gpx` or `.geojson` trail.
    """
    if gpx_xml:
        parsed = parse_gpx_xml(gpx_xml)
    elif geojson is not None:
        parsed = parse_geojson_trail(geojson)
    else:
        raise ValueError("Must supply either `gpx_xml` or `geojson`.")

    coords = parsed["coords"]
    profile = compute_tobler_trail_profile(coords)
    apex = profile["apex"]

    tz_obj, tz_name = local_tz_for_lon(apex["lon"])
    if date_str is None:
        date_str = dt.datetime.now(tz_obj).date().isoformat()
    req_d = dt.date.fromisoformat(date_str)
    noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz_obj)
    tz_off = (noon_loc.utcoffset() or dt.timedelta(0)).total_seconds() / 3600.0
    sr_info, ss_info = _find_sunrise_sunset_golden_hours(apex["lat"], apex["lon"], req_d, tz_off)
    is_sunrise = phase.strip().lower().startswith("sunrise")
    ev_mins, sun_az, sun_el = sr_info if is_sunrise else ss_info

    weather_doc: dict[str, Any] = {}
    if include_weather:
        raw_w = solve_weather_and_corridor(
            apex["lat"],
            apex["lon"],
            date_str,
            phase="sunrise" if is_sunrise else "sunset",
            sun_az_deg=sun_az,
            sun_el_deg=sun_el,
            target_elev_m=max(10.0, float(apex["elev_m"])),
            model_engine="ensemble",
            replay_mode=replay_mode,
            include_lut=False,
        )
        weather_doc = json.loads(raw_w)

    inv = weather_doc.get("inversion_model") or {}
    breakout = (
        compute_trail_fog_breakout_mileposts(coords, inv)
        if inv.get("fog_top_p50_ft") is not None
        else {"available": False, "reason": "no_inversion_model"}
    )
    nudges, logistics = build_trail_nudges_from_analysis(
        profile=profile,
        breakout=breakout,
        weather_doc=weather_doc,
        phase="sunrise" if is_sunrise else "sunset",
        event_mins_local=int(ev_mins),
        setup_buffer_min=setup_buffer_min,
    )

    c_ray = weather_doc.get("corridor_raycast") or {}
    cal = weather_doc.get("calibrated_probabilities") or {}
    field_card = {
        "trail_name": parsed["name"],
        "date": date_str,
        "timezone": tz_name,
        "phase": "Sunrise" if is_sunrise else "Sunset",
        "depart_trailhead": logistics["recommended_trailhead_departure_local"],
        "golden_hour_anchor": logistics["golden_hour_anchor_local"],
        "return_trailhead": logistics["estimated_trailhead_return_local"],
        "headlamp_required": logistics["headlamp_required"],
        "distance_mi": profile["total_distance_miles"],
        "gain_ft": profile["elevation_gain_ft"],
        "apex_elev_ft": apex["elev_ft"],
        "alpenglow_score": weather_doc.get("alpenglow_score"),
        "calibrated_transmittance_p10_p50_p90": [
            cal.get("transmittance_p10_pct", weather_doc.get("transmittance_p10_pct")),
            cal.get("transmittance_p50_pct", weather_doc.get("transmittance_p50_pct")),
            cal.get("transmittance_p90_pct", weather_doc.get("transmittance_p90_pct")),
        ],
        "fog_breakout_p50_milepost": (breakout.get("expected_breakout_p50") or {}).get("milepost"),
        "verdict": weather_doc.get("verdict"),
    }

    return {
        "schema_version": "trail_upload_v1",
        "trail_name": parsed["name"],
        "date": date_str,
        "timezone": tz_name,
        "phase": "sunrise" if is_sunrise else "sunset",
        "sun_azimuth_deg": round(sun_az, 1),
        "sun_elevation_deg": round(sun_el, 1),
        "profile": profile,
        "fog_breakout": breakout,
        "logistics": logistics,
        "nudges": nudges,
        "field_card": field_card,
        "weather_summary": {
            "verdict": weather_doc.get("verdict"),
            "recommendation": weather_doc.get("recommendation"),
            "alpenglow_score": weather_doc.get("alpenglow_score"),
            "canvas_score": weather_doc.get("canvas_score"),
            "dominant_blocker": c_ray.get("dominant_blocker_label"),
            "calibrated_probabilities": cal,
            "inversion_model": inv,
            "lake_mirror_reflection": weather_doc.get("lake_mirror_reflection"),
        },
    }
