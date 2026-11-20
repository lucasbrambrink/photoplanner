"""
3D Regional "Light-Hunting" Engine & Causal Bust Attribution (`planner/weather/regional_hunter.py`).

Implements Roadmap Phase 4 (Tasks 4.1 & 4.2):
  - Evaluates curated regional photography spots (coastal headlands, cloud-sea ridges,
    alpine granite peaks, alpine mirror lakes, desert canyons) using PhotoPlanner's
    380 km curved-Earth radiative corridor + calibrated probabilistic transmittance +
    probabilistic 3D inversion breakout + topographic lake Mirror Index.
  - Ranks tonight's / sunrise's top opportunities by specific photographic phenomenon:
      * Cloud-Sea Breakout
      * Open-Horizon Cirrus Burner
      * Direct Alpine Alpenglow
      * Alpine Mirror Reflection
      * Moody Storm-Break Shafts
      * Clear Dark-Sky Astro
  - Produces plain-English **Causal Bust Attributions** explaining *why* lower-ranked
    spots will fail (immersed below fog lid, distant 180 km corridor cloud wall, terrain
    horizon shadowing, wind-chopped lake surface, or cloudless bluebird sky).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .corridor_solver import (
    _find_sunrise_sunset_golden_hours,
    solve_weather_and_corridor,
)
from .nwp_corridor import local_tz_for_lon
from .terrain_coupling import compute_lake_mirror_index


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_mi = 3958.7613
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r_mi * math.asin(min(1.0, math.sqrt(a)))


# Curated multi-region photography catalog covering coastal cloud-sea ridges,
# alpine granite peaks, alpine mirror lakes, and red-rock canyons.
REGIONAL_SPOT_CATALOG: list[dict[str, Any]] = [
    # --- Bay Area & Central Coast ---
    {
        "id": "mt_tam_east",
        "name": "Mt. Tamalpais East Peak",
        "region": "bay_area",
        "state": "CA",
        "lat": 37.9235,
        "lon": -122.5965,
        "elev_m": 784.0,
        "archetype": "coastal_ridge",
        "is_lake": False,
        "upwind_shelter_deg": 2.0,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "montara_mtn",
        "name": "Montara Mountain North Peak",
        "region": "bay_area",
        "state": "CA",
        "lat": 37.5619,
        "lon": -122.4764,
        "elev_m": 578.0,
        "archetype": "coastal_ridge",
        "is_lake": False,
        "upwind_shelter_deg": 2.0,
        "best_phases": ("sunset",),
    },
    {
        "id": "pedro_point",
        "name": "Pedro Point Headlands",
        "region": "bay_area",
        "state": "CA",
        "lat": 37.5945,
        "lon": -122.5028,
        "elev_m": 220.0,
        "archetype": "coastal_headland",
        "is_lake": False,
        "upwind_shelter_deg": 1.0,
        "best_phases": ("sunset",),
    },
    {
        "id": "hawk_hill",
        "name": "Hawk Hill, Marin Headlands",
        "region": "bay_area",
        "state": "CA",
        "lat": 37.8270,
        "lon": -122.4994,
        "elev_m": 281.0,
        "archetype": "coastal_headland",
        "is_lake": False,
        "upwind_shelter_deg": 2.5,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "grizzly_peak",
        "name": "Grizzly Peak, Berkeley Hills",
        "region": "bay_area",
        "state": "CA",
        "lat": 37.8819,
        "lon": -122.2328,
        "elev_m": 535.0,
        "archetype": "coastal_ridge",
        "is_lake": False,
        "upwind_shelter_deg": 3.0,
        "best_phases": ("sunset",),
    },
    {
        "id": "mt_diablo",
        "name": "Mount Diablo Summit",
        "region": "bay_area",
        "state": "CA",
        "lat": 37.8816,
        "lon": -121.9142,
        "elev_m": 1173.0,
        "archetype": "coastal_ridge",
        "is_lake": False,
        "upwind_shelter_deg": 1.0,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "garrapata_bluffs",
        "name": "Garrapata Big Sur Bluffs",
        "region": "central_coast",
        "state": "CA",
        "lat": 36.4470,
        "lon": -121.9245,
        "elev_m": 180.0,
        "archetype": "coastal_headland",
        "is_lake": False,
        "upwind_shelter_deg": 1.5,
        "best_phases": ("sunset",),
    },
    {
        "id": "bixby_overlook",
        "name": "Bixby Creek Overlook, Big Sur",
        "region": "central_coast",
        "state": "CA",
        "lat": 36.3714,
        "lon": -121.9017,
        "elev_m": 115.0,
        "archetype": "coastal_headland",
        "is_lake": False,
        "upwind_shelter_deg": 4.0,
        "best_phases": ("sunset",),
    },
    # --- Sierra Nevada ---
    {
        "id": "glacier_point",
        "name": "Glacier Point (Half Dome View)",
        "region": "sierra",
        "state": "CA",
        "lat": 37.7281,
        "lon": -119.5738,
        "elev_m": 2199.0,
        "archetype": "alpine_granite",
        "is_lake": False,
        "upwind_shelter_deg": 5.0,
        "best_phases": ("sunset",),
    },
    {
        "id": "clouds_rest",
        "name": "Clouds Rest, Yosemite",
        "region": "sierra",
        "state": "CA",
        "lat": 37.7678,
        "lon": -119.4894,
        "elev_m": 3026.0,
        "archetype": "alpine_granite",
        "is_lake": False,
        "upwind_shelter_deg": 1.5,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "tenaya_lake",
        "name": "Tenaya Lake, Tuolumne",
        "region": "sierra",
        "state": "CA",
        "lat": 37.8315,
        "lon": -119.4622,
        "elev_m": 2484.0,
        "archetype": "alpine_lake",
        "is_lake": True,
        "upwind_shelter_deg": 14.0,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "mono_lake_tufa",
        "name": "Mono Lake South Tufa",
        "region": "sierra",
        "state": "CA",
        "lat": 37.9430,
        "lon": -119.0273,
        "elev_m": 1946.0,
        "archetype": "alpine_lake",
        "is_lake": True,
        "upwind_shelter_deg": 4.5,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "alabama_hills",
        "name": "Alabama Hills (Mt. Whitney View)",
        "region": "sierra",
        "state": "CA",
        "lat": 36.6063,
        "lon": -118.1165,
        "elev_m": 1380.0,
        "archetype": "alpine_granite",
        "is_lake": False,
        "upwind_shelter_deg": 6.0,
        "best_phases": ("sunrise",),
    },
    # --- Colorado Rockies ---
    {
        "id": "maroon_bells",
        "name": "Maroon Bells & Maroon Lake",
        "region": "rockies",
        "state": "CO",
        "lat": 39.0989,
        "lon": -106.9410,
        "elev_m": 2920.0,
        "archetype": "alpine_lake",
        "is_lake": True,
        "upwind_shelter_deg": 20.0,
        "best_phases": ("sunrise",),
    },
    {
        "id": "dream_lake_rmnp",
        "name": "Dream Lake, Rocky Mountain NP",
        "region": "rockies",
        "state": "CO",
        "lat": 40.3095,
        "lon": -105.6596,
        "elev_m": 3025.0,
        "archetype": "alpine_lake",
        "is_lake": True,
        "upwind_shelter_deg": 18.0,
        "best_phases": ("sunrise",),
    },
    {
        "id": "loveland_pass",
        "name": "Loveland Pass Continental Divide",
        "region": "rockies",
        "state": "CO",
        "lat": 39.6636,
        "lon": -105.8792,
        "elev_m": 3655.0,
        "archetype": "alpine_granite",
        "is_lake": False,
        "upwind_shelter_deg": 1.0,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "grand_teton_schwabacher",
        "name": "Schwabacher Landing, Grand Teton",
        "region": "rockies",
        "state": "WY",
        "lat": 43.7114,
        "lon": -110.6716,
        "elev_m": 2005.0,
        "archetype": "alpine_lake",
        "is_lake": True,
        "upwind_shelter_deg": 12.0,
        "best_phases": ("sunrise",),
    },
    # --- Pacific Northwest ---
    {
        "id": "rainier_paradise",
        "name": "Mt. Rainier Paradise",
        "region": "pnw",
        "state": "WA",
        "lat": 46.7865,
        "lon": -121.7353,
        "elev_m": 1650.0,
        "archetype": "alpine_granite",
        "is_lake": False,
        "upwind_shelter_deg": 6.0,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "reflection_lakes_rainier",
        "name": "Reflection Lakes, Mt. Rainier",
        "region": "pnw",
        "state": "WA",
        "lat": 46.7686,
        "lon": -121.7290,
        "elev_m": 1482.0,
        "archetype": "alpine_lake",
        "is_lake": True,
        "upwind_shelter_deg": 15.0,
        "best_phases": ("sunrise", "sunset"),
    },
    {
        "id": "rialto_beach",
        "name": "Rialto Beach, Olympic NP",
        "region": "pnw",
        "state": "WA",
        "lat": 47.9211,
        "lon": -124.6378,
        "elev_m": 15.0,
        "archetype": "coastal_headland",
        "is_lake": False,
        "upwind_shelter_deg": 1.0,
        "best_phases": ("sunset",),
    },
    {
        "id": "hurricane_ridge",
        "name": "Hurricane Ridge, Olympic NP",
        "region": "pnw",
        "state": "WA",
        "lat": 47.9691,
        "lon": -123.4986,
        "elev_m": 1598.0,
        "archetype": "coastal_ridge",
        "is_lake": False,
        "upwind_shelter_deg": 2.0,
        "best_phases": ("sunrise", "sunset"),
    },
    # --- Southwest ---
    {
        "id": "mesa_arch",
        "name": "Mesa Arch, Canyonlands",
        "region": "southwest",
        "state": "UT",
        "lat": 38.3880,
        "lon": -109.8634,
        "elev_m": 1875.0,
        "archetype": "canyon_rim",
        "is_lake": False,
        "upwind_shelter_deg": 3.0,
        "best_phases": ("sunrise",),
    },
    {
        "id": "grand_canyon_hopi",
        "name": "Grand Canyon Hopi Point",
        "region": "southwest",
        "state": "AZ",
        "lat": 36.0746,
        "lon": -112.1549,
        "elev_m": 2155.0,
        "archetype": "canyon_rim",
        "is_lake": False,
        "upwind_shelter_deg": 2.0,
        "best_phases": ("sunset",),
    },
    {
        "id": "zabriskie_point",
        "name": "Zabriskie Point, Death Valley",
        "region": "southwest",
        "state": "CA",
        "lat": 36.4201,
        "lon": -116.8120,
        "elev_m": 216.0,
        "archetype": "canyon_rim",
        "is_lake": False,
        "upwind_shelter_deg": 4.0,
        "best_phases": ("sunrise",),
    },
]


def classify_spot_phenomena_and_attribution(
    spot: dict[str, Any],
    doc: dict[str, Any],
) -> dict[str, Any]:
    """
    Scores a solved spot document into specific photographic phenomenon tags, an overall
    regional Light-Hunting Opportunity Score (`0–100`), a GO/GAMBLE/BUST verdict, and a
    plain-English **Causal Bust / Opportunity Attribution**.
    """
    c_ray = doc.get("corridor_raycast") or {}
    inv = doc.get("inversion_model") or {}
    cal = doc.get("calibrated_probabilities") or {}
    astro = doc.get("astro_atmosphere") or {}

    raw_exp = float(c_ray.get("expected_transmittance_pct", doc.get("ray_transmittance_pct", 0.0)))
    cal_p50 = float(cal.get("transmittance_p50_pct", raw_exp))
    cal_p10 = float(cal.get("transmittance_p10_pct", doc.get("transmittance_p10_pct", 0.0)))
    cal_p90 = float(cal.get("transmittance_p90_pct", doc.get("transmittance_p90_pct", 100.0)))

    alpenglow = int(doc.get("alpenglow_score", 0))
    canvas = int(doc.get("canvas_score", 0))
    low_cf = int(doc.get("cloud_low_pct", 0))
    mid_cf = int(doc.get("cloud_mid_pct", 0))
    high_cf = int(doc.get("cloud_high_pct", 0))

    fog_top_ft = int(inv.get("fog_top_p50_ft", doc.get("fog_top_ft", 0)))
    fog_p90_ft = int(inv.get("fog_top_p90_ft", fog_top_ft + 300))
    elev_ft = int(round(float(spot["elev_m"]) * 3.28084))
    clearance_ft = int(inv.get("summit_clearance_ft", elev_ft - fog_top_ft))
    above_fog_prob = float(inv.get("summit_above_fog_prob_pct", cal.get("summit_above_fog_prob_pct", 50.0)))
    cloud_sea_score = int(inv.get("cloud_sea_score", 0))

    blocker_label = str(c_ray.get("dominant_blocker_label") or "clear_corridor")
    blocker_dist_km = float(c_ray.get("dominant_blocker_distance_km", 0.0))
    terrain_hor_deg = float(c_ray.get("terrain_horizon_deg", 0.0))
    sun_el_deg = float(doc.get("sun_elevation_deg", 1.0))

    wind_mph = float(doc.get("wind_mph", 0.0))
    wind_dir_deg = float(doc.get("wind_dir_deg", 270.0))
    mirror = compute_lake_mirror_index(
        wind_speed_10m_mph=wind_mph,
        wind_dir_deg=wind_dir_deg,
        upwind_horizon_deg=max(float(spot.get("upwind_shelter_deg", 0.0)), terrain_hor_deg * 0.5),
    )

    is_coastal_like = spot.get("archetype") in ("coastal_ridge", "coastal_headland")
    immersed_in_fog = bool(is_coastal_like and low_cf >= 45 and above_fog_prob < 35.0 and clearance_ft < -50)

    phenomena: list[dict[str, Any]] = []

    # 1. Cloud-Sea Breakout
    if is_coastal_like and above_fog_prob >= 60.0 and (low_cf >= 30 or cloud_sea_score >= 60):
        cs_score = int(round(min(98.0, 0.50 * cloud_sea_score + 0.35 * above_fog_prob + 0.15 * cal_p50)))
        phenomena.append({"tag": "Cloud-Sea Breakout", "score": cs_score})

    # 2. Open-Horizon Cirrus Burner
    if (not immersed_in_fog) and (high_cf >= 22 or canvas >= 60) and cal_p50 >= 40.0:
        cb_score = int(round(min(98.0, 0.45 * canvas + 0.35 * cal_p50 + 0.20 * alpenglow)))
        phenomena.append({"tag": "Open-Horizon Cirrus Burner", "score": cb_score})

    # 3. Direct Granite / Alpine Alpenglow
    if spot.get("archetype") in ("alpine_granite", "canyon_rim", "alpine_lake") and cal_p50 >= 50.0 and alpenglow >= 58:
        ag_score = int(round(min(98.0, 0.55 * alpenglow + 0.45 * cal_p50)))
        phenomena.append({"tag": "Direct Alpine Alpenglow", "score": ag_score})

    # 4. Alpine Mirror Reflection
    if spot.get("is_lake") and mirror["mirror_index"] >= 65 and (cal_p50 >= 35.0 or alpenglow >= 50):
        mr_score = int(round(min(98.0, 0.50 * mirror["mirror_index"] + 0.30 * alpenglow + 0.20 * cal_p50)))
        phenomena.append({"tag": "Alpine Mirror Reflection", "score": mr_score})

    # 5. Moody Storm-Break Shafts
    if (cal_p90 - cal_p10) >= 32.0 and cal_p90 >= 55.0 and not immersed_in_fog:
        ms_score = int(round(min(92.0, 0.50 * cal_p90 + 0.30 * canvas + 10.0)))
        phenomena.append({"tag": "Moody Storm-Break Shafts", "score": ms_score})

    # 6. Clear Dark-Sky Astro
    seeing_idx = float(astro.get("seeing_index", 70.0)) if isinstance(astro, dict) else 70.0
    if low_cf <= 18 and mid_cf <= 18 and high_cf <= 22:
        as_score = int(round(min(96.0, 0.50 * seeing_idx + 0.50 * (100.0 - 0.8 * (low_cf + mid_cf + high_cf)))))
        phenomena.append({"tag": "Clear Dark-Sky Astro", "score": as_score})

    phenomena.sort(key=lambda x: x["score"], reverse=True)
    primary_phenomenon = phenomena[0]["tag"] if phenomena else "Muted / Flat Light"

    # Overall Regional Opportunity Score (0..100)
    base_opp = 0.48 * alpenglow + 0.32 * cal_p50 + 0.20 * canvas
    if phenomena:
        opp_score = int(round(max(base_opp, 0.65 * phenomena[0]["score"] + 0.35 * base_opp)))
    else:
        opp_score = int(round(base_opp))
    if immersed_in_fog:
        opp_score = min(opp_score, 32)
    if terrain_hor_deg > sun_el_deg + 1.8 and cal_p50 < 25.0:
        opp_score = min(opp_score, 36)
    opp_score = max(5, min(98, opp_score))

    # Verdict + Causal Attribution
    if opp_score >= 68:
        verdict = "GO"
    elif opp_score >= 46:
        verdict = "GAMBLE"
    else:
        verdict = "BUST"

    # Build plain-English causal explanation
    if immersed_in_fog:
        causal_reason = (
            f"BUST — Immersed ~{abs(clearance_ft):,} ft inside the {fog_top_ft:,} ft marine stratus deck "
            f"(only {above_fog_prob:.0f}% breakout odds). Climb above {fog_p90_ft:,} ft for clear cloud-sea views."
        )
    elif terrain_hor_deg > sun_el_deg + 1.8 and blocker_label == "terrain":
        causal_reason = (
            f"Terrain horizon ({terrain_hor_deg:.1f}°) shadows the low sun ({sun_el_deg:.1f}°) "
            f"at {blocker_dist_km:.0f} km — light cuts off early; shift to an elevated ridge or earlier golden hour."
        )
    elif cal_p50 < 30.0 and blocker_label in ("low", "mid", "high", "fog"):
        canvas_clause = (
            f"Overhead canvas looks promising ({high_cf}% high / {mid_cf}% mid), but "
            if (high_cf >= 25 or mid_cf >= 25)
            else ""
        )
        causal_reason = (
            f"{canvas_clause}a {blocker_label}-level cloud wall ~{blocker_dist_km:.0f} km {doc.get('corridor_direction', 'W')} "
            f"blocks the grazing solar beam (calibrated transmittance {cal_p50:.0f}%, P10–P90: {cal_p10:.0f}–{cal_p90:.0f}%)."
        )
    elif spot.get("is_lake") and mirror["mirror_index"] < 45:
        causal_reason = (
            f"Light is viable (cal. beam {cal_p50:.0f}%), but {mirror['wind_from_cardinal']} wind "
            f"({wind_mph:.0f} mph, Mirror Index {mirror['mirror_index']}/100) chops lake reflections."
        )
    elif low_cf < 15 and mid_cf < 15 and high_cf < 15 and cal_p50 >= 65.0:
        causal_reason = (
            f"Crystal-clear 380 km corridor (cal. beam {cal_p50:.0f}%, P10–P90: {cal_p10:.0f}–{cal_p90:.0f}%) "
            f"with cloudless overhead sky — prime direct alpine alpenglow & Clear Dark-Sky Astro, minimal cloud color."
        )
    else:
        causal_reason = (
            f"{primary_phenomenon}: calibrated beam {cal_p50:.0f}% (P10–P90: {cal_p10:.0f}–{cal_p90:.0f}%), "
            f"Alpenglow {alpenglow}/100, Canvas {canvas}/100"
            + (
                f", summit {clearance_ft:+,} ft vs fog lid ({above_fog_prob:.0f}% breakout odds)."
                if is_coastal_like
                else (f", Mirror Index {mirror['mirror_index']}/100." if spot.get("is_lake") else ".")
            )
        )

    return {
        "spot_id": spot["id"],
        "name": spot["name"],
        "region": spot["region"],
        "state": spot["state"],
        "lat": spot["lat"],
        "lon": spot["lon"],
        "elev_m": spot["elev_m"],
        "elev_ft": elev_ft,
        "archetype": spot["archetype"],
        "verdict": verdict,
        "opportunity_score": opp_score,
        "primary_phenomenon": primary_phenomenon,
        "phenomena": phenomena,
        "causal_attribution": causal_reason,
        "metrics": {
            "alpenglow_score": alpenglow,
            "canvas_score": canvas,
            "raw_transmittance_pct": round(raw_exp, 1),
            "calibrated_p10_pct": round(cal_p10, 1),
            "calibrated_p50_pct": round(cal_p50, 1),
            "calibrated_p90_pct": round(cal_p90, 1),
            "cloud_low_pct": low_cf,
            "cloud_mid_pct": mid_cf,
            "cloud_high_pct": high_cf,
            "fog_top_p50_ft": fog_top_ft,
            "fog_top_p90_ft": fog_p90_ft,
            "summit_above_fog_prob_pct": round(above_fog_prob, 1),
            "dominant_blocker": blocker_label,
            "blocker_distance_km": round(blocker_dist_km, 1),
            "mirror_index": mirror["mirror_index"] if spot.get("is_lake") else None,
        },
    }


def hunt_regional_light(
    date_str: str | None = None,
    phase: str = "sunset",
    region: str | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_miles: float = 120.0,
    max_spots: int = 8,
    max_workers: int = 2,
    replay_mode: str = "auto",
) -> dict[str, Any]:
    """
    Task 4.1 & 4.2: Runs 3D inverse-raycasted regional light hunting across matching spots,
    returning ranked `top_picks` (GO / GAMBLE) and `bust_warnings` (BUST with causal reasons).
    """
    spots = list(REGIONAL_SPOT_CATALOG)
    if region:
        reg_l = region.strip().lower()
        spots = [s for s in spots if s["region"].lower() == reg_l]
    if center_lat is not None and center_lon is not None:
        with_dist = []
        for s in spots:
            d_mi = _haversine_miles(center_lat, center_lon, s["lat"], s["lon"])
            if d_mi <= radius_miles:
                row = dict(s)
                row["distance_miles"] = round(d_mi, 1)
                with_dist.append(row)
        with_dist.sort(key=lambda x: x["distance_miles"])
        spots = with_dist
    spots = spots[: max(1, max_spots)]

    if not spots:
        return {
            "date": date_str,
            "phase": phase,
            "n_spots_evaluated": 0,
            "top_picks": [],
            "bust_warnings": [],
            "all_ranked": [],
        }

    if date_str is None:
        tz0, _ = local_tz_for_lon(spots[0]["lon"])
        date_str = dt.datetime.now(tz0).date().isoformat()
    req_d = dt.date.fromisoformat(date_str)
    phase_l = phase.strip().lower()

    # Deduplicate NWP solver calls on a 0.05° (~5.5 km) grid cell when two spots sit in the same cell
    def _eval_spot(spot: dict[str, Any]) -> dict[str, Any]:
        tz_obj, _ = local_tz_for_lon(spot["lon"])
        noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz_obj)
        tz_off = (noon_loc.utcoffset() or dt.timedelta(0)).total_seconds() / 3600.0
        sr, ss = _find_sunrise_sunset_golden_hours(spot["lat"], spot["lon"], req_d, tz_off)
        _, az, el = sr if phase_l == "sunrise" else ss
        raw = solve_weather_and_corridor(
            spot["lat"],
            spot["lon"],
            date_str,
            phase=phase_l,
            sun_az_deg=az,
            sun_el_deg=el,
            target_elev_m=float(spot["elev_m"]),
            model_engine="ensemble",
            replay_mode=replay_mode,
            include_lut=bool(spot.get("is_lake", False)),
        )
        doc = json.loads(raw)
        scored = classify_spot_phenomena_and_attribution(spot, doc)
        if "distance_miles" in spot:
            scored["distance_miles"] = spot["distance_miles"]
        return scored

    ranked: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(spots)))) as pool:
        futs = [pool.submit(_eval_spot, sp) for sp in spots]
        for fut in as_completed(futs):
            try:
                ranked.append(fut.result())
            except Exception:
                continue

    ranked.sort(key=lambda r: (r["opportunity_score"], r["metrics"]["calibrated_p50_pct"]), reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx

    return {
        "schema_version": "regional_hunt_v1",
        "date": date_str,
        "phase": phase_l,
        "region": region,
        "n_spots_evaluated": len(ranked),
        "top_picks": [r for r in ranked if r["verdict"] in ("GO", "GAMBLE")],
        "bust_warnings": [r for r in ranked if r["verdict"] == "BUST"],
        "all_ranked": ranked,
    }
