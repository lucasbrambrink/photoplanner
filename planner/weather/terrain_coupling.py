"""
Terrain-Coupled Atmospheric Intelligence (`planner/weather/terrain_coupling.py`).

Implements Roadmap Phase 2 core physics coupling 3D topography with probabilistic meteorology:
  - Task 2.1: Probabilistic 3D Inversion-Lid Altitude Bands (`P10 / P50 / P90` ASL) and
    exact Trail Fog-Breakout Milepost intersection along any 3D trail polyline.
  - Task 2.2: Topographic Lake Reflection "Mirror Index" (`0–100`) combining 10 m wind/gusts
    with upwind terrain horizon sheltering `H(phi_wind)`.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_probabilistic_fog_bands(
    fog_top_ft: float,
    fog_base_ft: float,
    target_elev_ft: float,
    inversion_delta_t_f: float,
    lead_hours: float = 1.0,
    raob_assimilated: bool = False,
    hrrr_s3_active: bool = False,
    is_coastal: bool = False,
    calibrated_above_fog_prob_pct: float | None = None,
) -> dict[str, Any]:
    """
    Computes calibrated probabilistic uncertainty bands (`P10 / P50 / P90` ASL) for the
    marine/valley inversion fog lid and summit breakout odds.

    Physical basis:
      - Observed RAOB / native 3 km HRRR tighten lid-altitude 1-sigma error to ~180–220 ft.
      - Longer lead times (+24h..+72h) grow lid-height spread toward ~340–520 ft.
      - Strong subsidence inversions (`ΔT >= 6°F`) sharpen the thermodynamic lid boundary.
    """
    lead_h = max(0.0, float(lead_hours))
    base_sigma = 175.0 if raob_assimilated else (225.0 if hrrr_s3_active else 285.0)
    lead_growth = 70.0 * math.log1p(lead_h / 2.5)
    # Strong inversion caps the stratus top more crisply than a weak/diffuse inversion
    inv_sharpness = 0.85 if inversion_delta_t_f >= 6.0 else (1.0 if inversion_delta_t_f >= 2.0 else 1.20)
    regime_scale = 0.95 if is_coastal else 1.10
    sigma_lid_ft = max(130.0, min(680.0, (base_sigma + lead_growth) * inv_sharpness * regime_scale))

    z80 = 1.28155
    p50_ft = float(fog_top_ft)
    p10_ft = max(float(fog_base_ft), p50_ft - z80 * sigma_lid_ft)
    p90_ft = p50_ft + z80 * sigma_lid_ft

    clr_p50 = float(target_elev_ft) - p50_ft
    clr_p10 = float(target_elev_ft) - p90_ft  # conservative clearance (when lid is high)
    clr_p90 = float(target_elev_ft) - p10_ft  # optimistic clearance (when lid is low)

    if calibrated_above_fog_prob_pct is not None:
        above_prob_pct = max(1.0, min(99.0, float(calibrated_above_fog_prob_pct)))
    else:
        above_prob_pct = max(1.0, min(99.0, _normal_cdf((clr_p50 - 30.0) / sigma_lid_ft) * 100.0))

    if above_prob_pct >= 75.0:
        verdict = "ABOVE_CLOUD_SEA"
        summary = (
            f"Summit ({target_elev_ft:,.0f} ft) sits ~{clr_p50:+,.0f} ft above the expected "
            f"{p50_ft:,.0f} ft fog lid ({above_prob_pct:.0f}% breakout odds)."
        )
    elif above_prob_pct >= 30.0:
        verdict = "BORDERLINE_RIM"
        summary = (
            f"Borderline fog-rim conditions: lid P10–P90 spans {p10_ft:,.0f}–{p90_ft:,.0f} ft "
            f"near your {target_elev_ft:,.0f} ft elevation ({above_prob_pct:.0f}% breakout odds)."
        )
    else:
        verdict = "IMMERSED_IN_FOG"
        summary = (
            f"Likely immersed inside the deck: expected lid {p50_ft:,.0f} ft sits above "
            f"{target_elev_ft:,.0f} ft ({above_prob_pct:.0f}% breakout odds; aim above {p90_ft:,.0f} ft)."
        )

    return {
        "fog_top_p10_ft": int(round(p10_ft)),
        "fog_top_p50_ft": int(round(p50_ft)),
        "fog_top_p90_ft": int(round(p90_ft)),
        "fog_lid_sigma_ft": int(round(sigma_lid_ft)),
        "summit_clearance_p10_ft": int(round(clr_p10)),
        "summit_clearance_p50_ft": int(round(clr_p50)),
        "summit_clearance_p90_ft": int(round(clr_p90)),
        "summit_above_fog_prob_pct": round(above_prob_pct, 1),
        "breakout_verdict": verdict,
        "summary": summary,
    }


def compute_trail_fog_breakout_mileposts(
    trail_coords_lon_lat_elev_m: Sequence[Sequence[float]],
    fog_bands: dict[str, Any],
) -> dict[str, Any]:
    """
    Intersects a 3D trail polyline (`[[lon, lat, elev_m], ...]`) with probabilistic
    inversion-lid bands (`fog_top_p10_ft`, `fog_top_p50_ft`, `fog_top_p90_ft`) to locate
    exact **Trail Breakout Mileposts** where hikers emerge above the cloud sea.
    """
    if len(trail_coords_lon_lat_elev_m) < 2:
        return {
            "available": False,
            "reason": "Trail polyline requires at least 2 vertices.",
        }

    p10_ft = float(fog_bands["fog_top_p10_ft"])
    p50_ft = float(fog_bands["fog_top_p50_ft"])
    p90_ft = float(fog_bands["fog_top_p90_ft"])

    cum_m = 0.0
    elevs_ft: list[float] = []
    dists_mi: list[float] = []

    for i, pt in enumerate(trail_coords_lon_lat_elev_m):
        lon, lat, z_m = float(pt[0]), float(pt[1]), float(pt[2])
        if i > 0:
            prev = trail_coords_lon_lat_elev_m[i - 1]
            cum_m += _haversine_m(float(prev[1]), float(prev[0]), lat, lon)
        dists_mi.append(cum_m / 1609.344)
        elevs_ft.append(z_m * 3.28084)

    def _find_first_crossing(threshold_ft: float) -> dict[str, Any] | None:
        if elevs_ft[0] >= threshold_ft:
            return {
                "milepost": 0.0,
                "elevation_ft": int(round(elevs_ft[0])),
                "lat": round(float(trail_coords_lon_lat_elev_m[0][1]), 5),
                "lon": round(float(trail_coords_lon_lat_elev_m[0][0]), 5),
                "starts_above": True,
            }
        for i in range(1, len(elevs_ft)):
            z0, z1 = elevs_ft[i - 1], elevs_ft[i]
            if z0 < threshold_ft <= z1:
                frac = (threshold_ft - z0) / max(1e-6, z1 - z0)
                mi = dists_mi[i - 1] + frac * (dists_mi[i] - dists_mi[i - 1])
                lat = float(trail_coords_lon_lat_elev_m[i - 1][1]) + frac * (
                    float(trail_coords_lon_lat_elev_m[i][1]) - float(trail_coords_lon_lat_elev_m[i - 1][1])
                )
                lon = float(trail_coords_lon_lat_elev_m[i - 1][0]) + frac * (
                    float(trail_coords_lon_lat_elev_m[i][0]) - float(trail_coords_lon_lat_elev_m[i - 1][0])
                )
                return {
                    "milepost": round(mi, 2),
                    "elevation_ft": int(round(threshold_ft)),
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "starts_above": False,
                }
        return None

    early_mp = _find_first_crossing(p10_ft)
    expected_mp = _find_first_crossing(p50_ft)
    confident_mp = _find_first_crossing(p90_ft)

    above_p50_dist = 0.0
    total_dist = max(1e-6, dists_mi[-1])
    for i in range(1, len(elevs_ft)):
        seg_mi = dists_mi[i] - dists_mi[i - 1]
        if 0.5 * (elevs_ft[i - 1] + elevs_ft[i]) >= p50_ft:
            above_p50_dist += seg_mi

    above_frac_pct = round(100.0 * above_p50_dist / total_dist, 1)
    max_elev_ft = int(round(max(elevs_ft)))
    min_elev_ft = int(round(min(elevs_ft)))

    if expected_mp is None:
        guidance = (
            f"Trail tops out at {max_elev_ft:,} ft, below the expected {p50_ft:,.0f} ft fog lid."
            + (
                f" Earliest optimistic rim breakout (P10) is Mile {early_mp['milepost']:.2f} ({early_mp['elevation_ft']:,} ft)."
                if early_mp
                else " Entire route stays inside the boundary-layer deck."
            )
        )
    elif expected_mp["starts_above"]:
        guidance = f"Trailhead ({min_elev_ft:,} ft) already starts above the expected {p50_ft:,.0f} ft cloud-sea ceiling."
    else:
        guidance = (
            f"Break above expected cloud-sea lid around Mile {expected_mp['milepost']:.2f} ({expected_mp['elevation_ft']:,} ft)"
            + (
                f"; >90% clear-sky odds past Mile {confident_mp['milepost']:.2f} ({confident_mp['elevation_ft']:,} ft)."
                if confident_mp and not confident_mp["starts_above"]
                else "."
            )
        )

    return {
        "available": True,
        "trail_length_miles": round(dists_mi[-1], 2),
        "trail_min_elev_ft": min_elev_ft,
        "trail_max_elev_ft": max_elev_ft,
        "above_expected_lid_pct_of_trail": above_frac_pct,
        "early_breakout_p10": early_mp,
        "expected_breakout_p50": expected_mp,
        "confident_breakout_p90": confident_mp,
        "guidance": guidance,
    }


def compute_lake_mirror_index(
    wind_speed_10m_mph: float,
    wind_dir_deg: float,
    upwind_horizon_deg: float = 0.0,
    wind_gust_mph: float | None = None,
    fetch_km: float = 0.8,
) -> dict[str, Any]:
    """
    Task 2.2: Topographic Lake Reflection "Mirror Index" (`0–100`).

    Combines 10 m synoptic wind speed, gustiness, lake fetch, and upwind terrain horizon
    sheltering angle `H(phi_wind)` (degrees above horizontal in the upwind direction).
    Cirque walls and deep canyon rims decouple the lake surface from synoptic flow,
    preserving glass-calm alpine reflections even when ridgelines are breezy.
    """
    v10 = max(0.0, float(wind_speed_10m_mph))
    gust = max(v10, float(wind_gust_mph)) if wind_gust_mph is not None else v10 * 1.25
    h_up = max(0.0, min(60.0, float(upwind_horizon_deg)))
    fetch = max(0.15, min(15.0, float(fetch_km)))

    # Orographic lee sheltering attenuation of wind kinetic energy:
    # H = 0° -> 1.00 (fully exposed), H = 10° -> 0.54, H = 20° -> 0.26, H = 30° -> 0.14
    shelter_energy_factor = 1.0 / (1.0 + (h_up / 10.5) ** 1.85)
    shelter_speed_factor = math.sqrt(shelter_energy_factor)
    shelter_reduction_pct = round((1.0 - shelter_speed_factor) * 100.0, 1)

    fetch_factor = (fetch / 0.8) ** 0.14
    gust_excess = max(0.0, gust - v10)
    eff_wind_mph = (v10 + 0.35 * gust_excess) * shelter_speed_factor * fetch_factor

    # Smooth logistic-like decay from glass (0–2.5 mph) to capillary ripples (4–8 mph) to chop (>11 mph)
    raw_score = 100.0 * math.exp(-((eff_wind_mph / 7.8) ** 1.65))
    mirror_index = int(round(max(0.0, min(100.0, raw_score))))

    if mirror_index >= 85:
        grade = "Glass Mirror"
    elif mirror_index >= 65:
        grade = "Calm / Soft Ripples"
    elif mirror_index >= 40:
        grade = "Broken Reflections (Use ND Long Exposure)"
    else:
        grade = "Wind-Chopped Surface"

    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    cardinal = dirs[int(round(((wind_dir_deg % 360.0) + 360.0) % 360.0 / 22.5)) % 16]

    if h_up >= 8.0:
        note = (
            f"{cardinal} flow ({v10:.0f} mph) is sheltered by a {h_up:.1f}° upwind terrain wall "
            f"(-{shelter_reduction_pct:.0f}% wind), yielding ~{eff_wind_mph:.1f} mph lake-surface breeze ({grade})."
        )
    else:
        note = (
            f"Exposed to {cardinal} wind ({v10:.0f} mph, upwind horizon {h_up:.1f}°); "
            f"effective surface breeze ~{eff_wind_mph:.1f} mph ({grade})."
        )

    return {
        "mirror_index": mirror_index,
        "grade": grade,
        "raw_wind_10m_mph": round(v10, 1),
        "effective_sheltered_wind_mph": round(eff_wind_mph, 1),
        "wind_from_deg": round(wind_dir_deg % 360.0, 1),
        "wind_from_cardinal": cardinal,
        "upwind_horizon_deg": round(h_up, 2),
        "shelter_reduction_pct": shelter_reduction_pct,
        "summary": note,
    }
