"""
Probabilistic 7-Archetype Hourly Light Engine (`planner/weather/light_archetypes.py`).

Computes mutually exclusive probabilities (strictly summing to 100%) across 7 distinct
photographic light archetypes for every hour `0..23` and for the primary sunrise/sunset
phase, driven directly by:
  - 380 km curved-Earth solar corridor quantiles (`P10`, `P50`, `P90`, `direct_beam_pct`)
  - 4-layer cloud column (`fog`, `low`, `mid`, `high` + high optical depth)
  - Inversion lid vs. summit elevation (`fog_top_ft`, `target_elev_ft`, `inversion_delta_t_f`)
  - Overhead sky reflector score (`canvas_glow_score`)
  - CAMS 550 nm Aerosol Optical Depth (`aod_550nm`), visibility, and surface humidity
"""

from __future__ import annotations

import math
from typing import Any


ARCHETYPE_ORDER: tuple[str, ...] = (
    "filtered_gold",
    "clearing_storm",
    "atmospheric_diffusion",
    "high_thin_veil",
    "giant_softbox",
    "blue_hour",
    "harsh_sun",
)

VIABLE_THRESHOLD_PCT: int = 20

ARCHETYPE_CATALOG: dict[str, dict[str, Any]] = {
    "giant_softbox": {
        "id": "giant_softbox",
        "name": "Giant Softbox",
        "short_name": "Softbox",
        "subtitle": "Thick, Uniform Overcast",
        "summary": "Cool 6500–7500K wrap-around overcast sky with virtually no shadows and very low contrast.",
        "accent_hex": "#94a3b8",
        "swatch_gradient": "linear-gradient(135deg, #475569 0%, #94a3b8 55%, #cbd5e1 100%)",
        "field_reference": {
            "shadow_edge": "Shadowless wrap-around",
            "contrast": "Very low",
            "color_cast": "Cool 6500–7500K",
            "best_for": "Forests, waterfalls, saturated foliage, intimate rock detail",
            "tactical_lever": "Exclude sky, boost micro-contrast",
        },
    },
    "clearing_storm": {
        "id": "clearing_storm",
        "name": "Clearing Storm",
        "short_name": "Storm Breaks",
        "subtitle": "\"God Beams\" & Spotlight Drama",
        "summary": "Extreme warm/cool split with fractured storm breaks carving sunlit spotlight shafts.",
        "accent_hex": "#f59e0b",
        "swatch_gradient": "linear-gradient(135deg, #1e293b 0%, #334155 48%, #f59e0b 78%, #fde68a 100%)",
        "field_reference": {
            "shadow_edge": "Hard breaks to moody umbra",
            "contrast": "Extreme",
            "color_cast": "Warm / cool split",
            "best_for": "Stormy peaks, spotlighted valleys, dramatic cloud shafts",
            "tactical_lever": "Expose for sunlit highlights, let shadows drop",
        },
    },
    "atmospheric_diffusion": {
        "id": "atmospheric_diffusion",
        "name": "Atmospheric Diffusion",
        "short_name": "Mist & Haze",
        "subtitle": "Fog, Mist & Coastal Haze",
        "summary": "Graduated depth layers and aerial perspective softening successive ridges into cool-neutral mist.",
        "accent_hex": "#7dd3fc",
        "swatch_gradient": "linear-gradient(135deg, #334e68 0%, #7aa5c2 52%, #d9e8f5 100%)",
        "field_reference": {
            "shadow_edge": "Diffused / graduated",
            "contrast": "Low–moderate layered",
            "color_cast": "Neutral / cool mist",
            "best_for": "Layered ridges, sea stacks, minimalist coastal silhouettes",
            "tactical_lever": "Use distance separation to build depth layers",
        },
    },
    "high_thin_veil": {
        "id": "high_thin_veil",
        "name": "High Thin Veil",
        "short_name": "Cirrus Scrim",
        "subtitle": "Cirrus / Bright Overcast Scrim",
        "summary": "1–2 stop translucent cirrus scrim creating soft-edged shadows and clean neutral-warm balance.",
        "accent_hex": "#e2e8f0",
        "swatch_gradient": "linear-gradient(135deg, #3b6ea5 0%, #9dbcd4 56%, #f1ece1 100%)",
        "field_reference": {
            "shadow_edge": "Soft-edged (1–2 stop scrim)",
            "contrast": "Moderate",
            "color_cast": "Neutral–warm",
            "best_for": "Midday alpine texture, open vistas, glare-free rock & foliage",
            "tactical_lever": "Ideal for midday subjects without harsh glare",
        },
    },
    "filtered_gold": {
        "id": "filtered_gold",
        "name": "Filtered Gold",
        "short_name": "Golden Light",
        "subtitle": "Low-Angle Golden Light & Alpenglow",
        "summary": "Sun within ~6–10° of horizon casting 3000–4500K raking amber light and long crisp shadows.",
        "accent_hex": "#fb923c",
        "swatch_gradient": "linear-gradient(135deg, #1e3a6e 0%, #c2410c 52%, #fbbf24 100%)",
        "field_reference": {
            "shadow_edge": "Long, crisp raking",
            "contrast": "High warm/cool drama",
            "color_cast": "Warm 3000–4500K amber",
            "best_for": "Granite walls, textured cliffs, backlit ridges, alpenglow",
            "tactical_lever": "Side/back-lighting to reveal texture",
        },
    },
    "blue_hour": {
        "id": "blue_hour",
        "name": "Blue Hour",
        "short_name": "Blue Hour",
        "subtitle": "Blue Hour & Twilight Glow",
        "summary": "Sun −4° to −8° below horizon bathing the landscape in virtually shadowless cobalt/violet ambient scatter.",
        "accent_hex": "#6366f1",
        "swatch_gradient": "linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #8b5cf6 100%)",
        "field_reference": {
            "shadow_edge": "Virtually shadowless",
            "contrast": "Low–silhouetted",
            "color_cast": "Cobalt / violet ambient",
            "best_for": "Alpine mirror lakes, twilight ridgelines, long-exposure surf",
            "tactical_lever": "Balance cool ambient with warm accents or long exposure",
        },
    },
    "harsh_sun": {
        "id": "harsh_sun",
        "name": "Harsh Sun",
        "short_name": "Harsh Sun",
        "subtitle": "Clear Midday High-Pressure",
        "summary": "Unfiltered high-pressure 5200–5600K sun casting razor-sharp shadows and very high contrast.",
        "accent_hex": "#38bdf8",
        "swatch_gradient": "linear-gradient(135deg, #0369a1 0%, #38bdf8 56%, #fef08a 100%)",
        "field_reference": {
            "shadow_edge": "Razor-sharp",
            "contrast": "Very high",
            "color_cast": "Crisp 5200–5600K",
            "best_for": "Graphic shadows, high-contrast B&W, polarized alpine water",
            "tactical_lever": "Lean into shadows/silhouettes or use a CPL",
        },
    },
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _largest_remainder_100(raw_scores: dict[str, float]) -> tuple[dict[str, int], dict[str, float]]:
    """
    Normalizes non-negative scores to integer percentages summing STRICTLY to 100
    using Hamilton's Largest Remainder Method, alongside exact rounded floats.
    """
    cleaned = {k: max(0.0, float(raw_scores.get(k, 0.0))) for k in ARCHETYPE_ORDER}
    total = sum(cleaned.values())
    if total <= 1e-12:
        # Safe fallback distribution (never divides by zero)
        cleaned = {k: (1.0 if k == "high_thin_veil" else 0.0) for k in ARCHETYPE_ORDER}
        total = 1.0

    exact = {k: (100.0 * cleaned[k] / total) for k in ARCHETYPE_ORDER}
    floors = {k: int(math.floor(exact[k])) for k in ARCHETYPE_ORDER}
    remainder = 100 - sum(floors.values())

    # Deterministic tie-breaking by descending fractional part, then raw score, then canonical order
    order_for_rem = sorted(
        ARCHETYPE_ORDER,
        key=lambda k: (exact[k] - floors[k], exact[k], -ARCHETYPE_ORDER.index(k)),
        reverse=True,
    )
    ints = dict(floors)
    for idx in range(max(0, remainder)):
        ints[order_for_rem[idx % len(order_for_rem)]] += 1

    # Round floats for display while keeping exact internal order
    exact_rounded = {k: round(exact[k], 2) for k in ARCHETYPE_ORDER}
    return ints, exact_rounded


def compute_light_archetypes(
    *,
    sun_el_deg: float,
    p10_pct: float,
    p50_pct: float,
    p90_pct: float,
    cloud_low_pct: float,
    cloud_mid_pct: float,
    cloud_high_pct: float,
    fog_pct: float | None = None,
    fog_top_ft: float = 0.0,
    target_elev_ft: float = 0.0,
    canvas_glow_score: float = 40.0,
    aod_550nm: float = 0.07,
    high_optical_depth: float | None = None,
    inversion_delta_t_f: float = 0.0,
    visibility_miles: float = 18.0,
    rh_pct: float = 55.0,
    direct_beam_pct: float | None = None,
    terrain_disk_fraction: float = 1.0,
    is_coastal: bool = False,
    compact: bool = False,
) -> dict[str, Any]:
    """
    Computes mutually exclusive probabilities across all 7 Light Archetypes for a single
    time slice (hour or golden-hour phase), guaranteed to sum strictly to 100%.
    """
    el = float(sun_el_deg)
    q10 = _clamp(float(p10_pct), 0.0, 100.0)
    q50 = _clamp(float(p50_pct), q10, 100.0)
    q90 = _clamp(float(p90_pct), q50, 100.0)
    spread = max(0.0, q90 - q10)

    low = _clamp(float(cloud_low_pct) / 100.0, 0.0, 1.0)
    mid = _clamp(float(cloud_mid_pct) / 100.0, 0.0, 1.0)
    high = _clamp(float(cloud_high_pct) / 100.0, 0.0, 1.0)
    fog = _clamp(
        (float(fog_pct) / 100.0) if fog_pct is not None else (low * (0.82 if is_coastal else 0.40)),
        0.0,
        1.0,
    )
    high_tau = (
        max(0.0, float(high_optical_depth))
        if high_optical_depth is not None
        else (high * (0.65 + 1.15 * high))
    )

    aod = max(0.0, float(aod_550nm))
    vis_mi = max(0.1, float(visibility_miles))
    rh = _clamp(float(rh_pct), 0.0, 100.0)
    canvas = _clamp(float(canvas_glow_score) / 100.0, 0.0, 1.0)
    direct = _clamp((float(direct_beam_pct) if direct_beam_pct is not None else q50) / 100.0, 0.0, 1.0)
    t_disk = _clamp(float(terrain_disk_fraction), 0.0, 1.0)

    clearance_ft = float(target_elev_ft) - float(fog_top_ft) if fog_top_ft > 0.0 else 9999.0
    # True when summit sits well above a capped coastal/valley fog sea
    above_fog_sea = bool(fog_top_ft > 0.0 and clearance_ft >= 120.0)
    # Effective low-cloud overhead obscuration at the photographer's elevation
    low_overhead = low * 0.18 if (above_fog_sea and (is_coastal or inversion_delta_t_f >= 1.2)) else low
    opaque_overhead = 1.0 - (1.0 - low_overhead) * (1.0 - mid)
    total_overhead = 1.0 - (1.0 - opaque_overhead) * (1.0 - 0.60 * high)

    p50_n = q50 / 100.0
    p90_n = q90 / 100.0
    p10_n = q10 / 100.0
    spread_n = _clamp(spread / 70.0, 0.0, 1.0)

    # Solar geometry gates
    # Blue hour peaks around -4..-8 deg and remains the dominant nocturnal/twilight ambient state below -2 deg.
    blue_solar_gate = 1.0 - _smoothstep(-2.2, 1.6, el)
    # Filtered gold peaks for sun within ~-1.5..+10 deg of horizon
    gold_solar_gate = _smoothstep(-3.2, -0.4, el) * (1.0 - _smoothstep(8.5, 14.5, el))
    # Daylight gate for midday/cloud regimes
    day_solar_gate = _smoothstep(-1.8, 3.5, el)
    # High-sun gate for Harsh Sun (midday high pressure, > ~10-16 deg)
    harsh_solar_gate = _smoothstep(8.5, 17.5, el)

    # 1. Blue Hour & Twilight Glow
    # Pure ambient cobalt/violet scatter when sun is below horizon; slightly reduced if socked in by dense fog/overcast
    twilight_sweet = math.exp(-((el + 5.5) / 4.2) ** 2)
    blue_clear_bonus = 0.58 + 0.42 * (1.0 - 0.55 * opaque_overhead)
    blue_canvas_bonus = 1.0 + 0.28 * canvas * _smoothstep(-9.0, -2.0, el)
    raw_blue_hour = (
        blue_solar_gate
        * (0.85 + 0.65 * twilight_sweet)
        * blue_clear_bonus
        * blue_canvas_bonus
        * 1.45
    )

    # 2. Filtered Gold (Low-Angle Golden Light & Alpenglow)
    # High corridor transmittance OR strong overhead canvas glow near the horizon (alpenglow)
    gold_beam_signal = max(
        0.55 * p50_n + 0.30 * p90_n + 0.15 * direct,
        0.72 * canvas * (1.0 - 0.65 * low_overhead),
    )
    # When summit sits above a capped cloud-sea with clear corridor, golden cloud-sea alpenglow spikes
    cloud_sea_gold_bonus = (
        0.28 * fog * _smoothstep(100.0, 450.0, clearance_ft) * (0.4 + 0.6 * p90_n)
        if above_fog_sea
        else 0.0
    )
    # Attenuate if deeply buried under opaque fog with no break
    buried_penalty = (
        1.0 - 0.68 * _smoothstep(80.0, -350.0, clearance_ft) * fog * (1.0 - p90_n)
        if fog_top_ft > 0.0
        else 1.0
    )
    # Terrain occlusion dims direct face gold unless overhead cirrus canvas catches alpenglow
    disk_factor = max(0.35 + 0.65 * t_disk, 0.78 * canvas)
    raw_filtered_gold = (
        gold_solar_gate
        * max(0.03, (gold_beam_signal + cloud_sea_gold_bonus) ** 1.15)
        * buried_penalty
        * disk_factor
        * 1.65
    )

    # 3. Harsh Sun (Clear Midday High-Pressure)
    # High solar angle + reliable direct beam (high P10 & P50) + minimal cloud veil + clean air
    clear_sky_factor = (1.0 - opaque_overhead) * (1.0 - 0.78 * high)
    aod_clarity = 1.0 - _smoothstep(0.18, 0.58, aod)
    vis_clarity = _smoothstep(5.0, 14.0, vis_mi)
    beam_reliability = 0.55 * p50_n + 0.45 * p10_n
    raw_harsh_sun = (
        harsh_solar_gate
        * (beam_reliability ** 1.25)
        * (clear_sky_factor ** 1.35)
        * (0.35 + 0.65 * aod_clarity)
        * (0.40 + 0.60 * vis_clarity)
        * 1.55
    )

    # 4. High Thin Veil (Cirrus / Bright Overcast Scrim)
    # Significant high cirrus or translucent bright mid/high veil, low opaque stratus, 1-2 stop scrim transmittance
    # Peak when high cloud >= 35% with optically thin/moderate tau (0.25..2.8), OR bright thin mid deck
    cirrus_scrim_signal = (
        _smoothstep(0.18, 0.65, high)
        * (1.0 - 0.55 * _smoothstep(3.2, 7.5, high_tau))
        * (1.0 - 0.78 * low_overhead)
    )
    thin_bright_mid = (
        _smoothstep(0.25, 0.70, mid)
        * (1.0 - _smoothstep(0.55, 0.90, low_overhead))
        * _smoothstep(0.25, 0.55, p50_n)
        * (1.0 - _smoothstep(0.76, 0.94, p50_n))
    )
    scrim_trans_bell = math.exp(-((p50_n - 0.58) / 0.30) ** 2)
    # High thin veil works across daylight and late golden hour (less so in deep blue hour)
    veil_solar_gate = _smoothstep(1.0, 9.0, el) * 0.92 + gold_solar_gate * 0.35
    raw_high_thin_veil = (
        veil_solar_gate
        * max(cirrus_scrim_signal * (0.45 + 0.55 * scrim_trans_bell), thin_bright_mid * 0.85)
        * (1.0 - 0.35 * spread_n)
        * 1.25
    )

    # 5. Giant Softbox (Thick, Uniform Overcast)
    # High opaque cover + low direct beam (low P50 AND low P90 -> uniform shadowless deck without breaks)
    uniformity = 1.0 - _smoothstep(18.0, 56.0, spread)
    low_beam_darkness = (1.0 - _smoothstep(0.15, 0.52, p50_n)) * (1.0 - _smoothstep(0.32, 0.70, p90_n))
    heavy_deck = _smoothstep(0.55, 0.92, opaque_overhead)
    # If buried in close surface fog with <3 mi visibility, diffusion shares mass; if under a solid cloud ceiling, softbox dominates
    ceiling_vs_mist = 0.55 + 0.45 * _smoothstep(1.5, 7.0, vis_mi)
    softbox_solar_gate = 0.22 * blue_solar_gate + 0.95 * _smoothstep(-2.0, 4.0, el)
    raw_giant_softbox = (
        softbox_solar_gate
        * heavy_deck
        * (0.35 + 0.65 * low_beam_darkness)
        * (0.35 + 0.65 * uniformity)
        * ceiling_vs_mist
        * 1.45
    )

    # 6. Clearing Storm ("God Beams" & Spotlight Drama)
    # Driven by fractured opaque low/mid storm clouds (NOT pure high cirrus and NOT dense surface fog immersion)
    # + wide bimodal corridor spread (P90 - P10) + sunlit break potential (high P90).
    opaque_storm_presence = _smoothstep(0.18, 0.45, opaque_overhead)
    broken_cloud_bell = math.exp(-((opaque_overhead - 0.56) / 0.30) ** 2) * opaque_storm_presence
    fracture_signal = _smoothstep(22.0, 58.0, spread) * _smoothstep(0.38, 0.80, p90_n)
    spotlight_contrast = _smoothstep(0.12, 0.42, p50_n) * (1.0 - _smoothstep(0.74, 0.94, p50_n))
    # Suppress storm-beam drama when photographer is buried in thick low-visibility surface fog/mist
    surface_fog_damping = 1.0 - 0.68 * _smoothstep(5.5, 1.5, vis_mi) * _smoothstep(0.50, 0.85, fog)
    # Also require meaningful mid-level or broken cumulus structure rather than laminar fog-only stratus
    storm_verticality = 0.45 + 0.55 * _smoothstep(0.14, 0.48, mid + 0.55 * max(0.0, low_overhead - fog * 0.7))
    broken_storm_structure = broken_cloud_bell * (0.48 * fracture_signal + 0.34 * spotlight_contrast + 0.18 * canvas)
    corridor_bimodal_drama = fracture_signal * opaque_storm_presence * (0.55 + 0.45 * spotlight_contrast)
    storm_solar_gate = _smoothstep(-1.0, 3.5, el)
    raw_clearing_storm = (
        storm_solar_gate
        * max(broken_storm_structure * 1.40, corridor_bimodal_drama * 1.42)
        * surface_fog_damping
        * storm_verticality
        * 1.45
    )

    # 7. Atmospheric Diffusion (Fog, Mist & Coastal Haze)
    # Driven by (a) summit in/near inversion fog layer, (b) overlooking a thick valley/coastal fog sea,
    # (c) elevated CAMS AOD aerosol/smoke/coastal haze, or (d) low surface visibility / high RH mist.
    if fog_top_ft > 0.0:
        # In fog or skimming the top edge (-600..+450 ft): very strong mist & graduated layers
        in_or_near_fog = (
            _smoothstep(0.18, 0.65, fog)
            * _smoothstep(-900.0, -100.0, clearance_ft)
            * (1.0 - _smoothstep(240.0, 980.0, clearance_ft))
        )
        # Deeply inside thick coastal fog bank (below lid)
        deep_in_fog = _smoothstep(0.35, 0.78, fog) * _smoothstep(120.0, -220.0, clearance_ft)
        # Overlooking a layered fog sea from above (adds atmospheric depth layers)
        overlooking_fog_sea = (
            _smoothstep(0.40, 0.82, fog)
            * _smoothstep(120.0, 420.0, clearance_ft)
            * (1.0 - _smoothstep(1500.0, 3200.0, clearance_ft))
            * 0.58
        )
        inversion_fog_signal = max(in_or_near_fog, deep_in_fog * 0.96, overlooking_fog_sea)
    else:
        inversion_fog_signal = 0.0

    aod_haze_signal = _smoothstep(0.15, 0.52, aod) * 0.98
    vis_mist_signal = _smoothstep(12.0, 2.2, vis_mi) * 1.05
    humid_coast_mist = _smoothstep(78.0, 95.0, rh) * (0.50 if is_coastal else 0.28) * (0.35 + 0.65 * fog)
    diffusion_core = max(inversion_fog_signal, aod_haze_signal, vis_mist_signal, humid_coast_mist)
    # Additive reinforcement when multiple haze/mist mechanisms coincide (e.g. coastal fog + low visibility + AOD)
    diffusion_combo = _clamp(
        diffusion_core + 0.24 * min(inversion_fog_signal, aod_haze_signal + vis_mist_signal),
        0.0,
        1.25,
    )
    diff_solar_gate = 0.35 * blue_solar_gate + 1.0 * _smoothstep(-3.0, 2.5, el)
    raw_atmospheric_diffusion = diff_solar_gate * (diffusion_combo ** 1.05) * 1.50

    raw_scores = {
        "filtered_gold": raw_filtered_gold,
        "clearing_storm": raw_clearing_storm,
        "atmospheric_diffusion": raw_atmospheric_diffusion,
        "high_thin_veil": raw_high_thin_veil,
        "giant_softbox": raw_giant_softbox,
        "blue_hour": raw_blue_hour,
        "harsh_sun": raw_harsh_sun,
    }

    probs_int, probs_exact = _largest_remainder_100(raw_scores)

    ranked_ids = sorted(
        ARCHETYPE_ORDER,
        key=lambda k: (probs_int[k], probs_exact[k], -ARCHETYPE_ORDER.index(k)),
        reverse=True,
    )
    dominant_id = ranked_ids[0]
    viable_ids = [k for k in ranked_ids if probs_int[k] >= VIABLE_THRESHOLD_PCT]
    if not viable_ids:
        viable_ids = [dominant_id]

    result: dict[str, Any] = {
        "dominant_id": dominant_id,
        "dominant_name": ARCHETYPE_CATALOG[dominant_id]["name"],
        "dominant_probability_pct": probs_int[dominant_id],
        "viable_threshold_pct": VIABLE_THRESHOLD_PCT,
        "viable_ids": viable_ids,
        "ranked_ids": ranked_ids,
        "probabilities": probs_int,
        "probabilities_exact": probs_exact,
    }

    if not compact:
        result["archetypes"] = [
            {
                **ARCHETYPE_CATALOG[k],
                "probability_pct": probs_int[k],
                "probability_exact": probs_exact[k],
                "is_dominant": k == dominant_id,
                "is_viable": k in viable_ids,
            }
            for k in ranked_ids
        ]
        result["diagnostics"] = {
            "sun_el_deg": round(el, 2),
            "p10_pct": round(q10, 1),
            "p50_pct": round(q50, 1),
            "p90_pct": round(q90, 1),
            "spread_pct": round(spread, 1),
            "fog_pct": round(fog * 100.0, 1),
            "low_pct": round(low * 100.0, 1),
            "mid_pct": round(mid * 100.0, 1),
            "high_pct": round(high * 100.0, 1),
            "clearance_ft": round(clearance_ft, 0) if fog_top_ft > 0.0 else None,
            "canvas_glow_score": round(float(canvas_glow_score), 1),
            "aod_550nm": round(aod, 3),
        }

    return result
