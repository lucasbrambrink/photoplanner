"""
Dynamic SEO Light Guides & B2B API Tier (`planner/seo_guides.py`).

Implements Roadmap Phase 4 — Task 4.4:
  - Daily-updated, search-engine-indexable landing pages (`/guides` and `/guides/{slug}`)
    for iconic regional peaks, coastal cloud-sea ridges, alpine mirror lakes, and red-rock
    canyons across 7 US regions (40 curated destinations in `REGIONAL_SPOT_CATALOG`).
  - Rich technical SEO markup: canonical links, OpenGraph + Twitter metadata, Schema.org
    JSON-LD (`TouristAttraction` + `GeoCoordinates` + dynamic `FAQPage`), internal regional
    cross-links, `/sitemap.xml`, and `/robots.txt`.
  - Clean versioned B2B JSON API payload (`/api/v1/guides` and `/api/v1/guides/{slug}`)
    for photo workshops, outdoor apps, and backcountry lodges, with optional `X-API-Key`
    tier gating via `PHOTOPLANNER_B2B_API_KEYS`.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
from typing import Any

from planner.weather.corridor_solver import (
    _compass_cardinal,
    _find_sunrise_sunset_golden_hours,
    solve_weather_and_corridor,
)
from planner.weather.nwp_corridor import local_tz_for_lon
from planner.weather.regional_hunter import (
    REGIONAL_SPOT_CATALOG,
    classify_spot_phenomena_and_attribution,
)


REGION_LABELS: dict[str, str] = {
    "bay_area": "San Francisco Bay Area & Marin Coast",
    "central_coast": "Big Sur & California Central Coast",
    "sierra": "Sierra Nevada & Yosemite",
    "pnw": "Pacific Northwest & Cascades",
    "rockies": "Rocky Mountains & Glacier",
    "southwest": "American Southwest & Colorado Plateau",
    "east_coast": "East Coast, Acadia & Appalachians",
}

ARCHETYPE_PROFILES: dict[str, dict[str, str]] = {
    "coastal_ridge": {
        "focus_keyword": "Cloud-Sea & Golden Hour Forecast",
        "recommended_lens": "24–70 mm (wide fog waves) + 70–200 mm (layered ridges)",
        "headline_phenomenon": "Marine Layer Cloud-Sea & Alpenglow",
        "field_tip": (
            "Compare the probabilistic P10–P90 fog-sea lid height against summit elevation before "
            "driving up—being 150+ ft above the P90 lid yields clean cloud-sea waves, while sitting "
            "below the P50 lid drops visibility to near zero."
        ),
    },
    "coastal_headland": {
        "focus_keyword": "Sunset & Horizon Slot Forecast",
        "recommended_lens": "16–35 mm (sea stacks & surf) + 24–70 mm (headland cliffs)",
        "headline_phenomenon": "Open-Horizon Cirrus Burner & Twilight Glow",
        "field_tip": (
            "Coastal sunsets live or die on the distant 120–320 km offshore marine stratus wall. "
            "Even clear local skies fail to light up if a solid low cloud deck blocks the grazing "
            "solar beam 180 km out to sea."
        ),
    },
    "alpine_granite": {
        "focus_keyword": "Alpenglow & Light Corridor Forecast",
        "recommended_lens": "24–70 mm (granite context) + 70–200 mm (peak alpenglow compression)",
        "headline_phenomenon": "Direct Granite Alpenglow & High-Altitude Cirrus",
        "field_tip": (
            "High granite faces catch direct crimson light 10–15 minutes after valley floors fall "
            "into shadow. Watch the 380 km curved-Earth beam transmittance alongside terrain horizon "
            "blocking to time last light cleanly."
        ),
    },
    "alpine_lake": {
        "focus_keyword": "Mirror Reflection & Sunrise/Sunset Forecast",
        "recommended_lens": "16–35 mm (shoreline reflection) + 24–70 mm (peak symmetry)",
        "headline_phenomenon": "Alpine Mirror Reflection & First-Light Alpenglow",
        "field_tip": (
            "Glass-calm reflections require wind speeds under ~5 mph or strong upwind cirque "
            "sheltering. Dawn typically beats dusk for mirror calm before daytime thermal winds "
            "spin up."
        ),
    },
    "canyon_rim": {
        "focus_keyword": "Golden Hour & Canyon Rim Light Forecast",
        "recommended_lens": "16–35 mm (rim depth & arches) + 24–70 mm (canyon walls)",
        "headline_phenomenon": "Direct Rim Alpenglow & Desert Cirrus",
        "field_tip": (
            "High-desert clarity produces intense low-angle contrast when the 0–380 km corridor is "
            "unobstructed. Arrive 30 minutes before geometric horizon crossing to catch pre-dawn "
            "Belt of Venus or post-sunset crimson cloud bounce."
        ),
    },
}

# Spot-specific overrides for top flagship destinations (adds custom lens / trailhead context)
SPOT_EDITORIAL_OVERRIDES: dict[str, dict[str, str]] = {
    "mt_tam_east": {
        "slug": "mt-tamalpais-east-peak",
        "title_focus": "Cloud-Sea & Sunset Forecast",
        "access_note": "East Peak lot + 0.3 mi Plankwalk / Verna Dunshee loop (2,572 ft ASL).",
        "recommended_lens": "24–70 mm + 70–200 mm telephoto for rolling Marin fog waves",
        "viewfinder_landmark": "Pedro Point Headlands",
    },
    "montara_mtn": {
        "slug": "montara-mountain",
        "title_focus": "Marine Cloud-Sea & Sunset Forecast",
        "access_note": "Gray Whale Cove / McNee Ranch trailhead (3.6 mi, +1,650 ft gain).",
        "recommended_lens": "24–70 mm + 70–200 mm",
        "viewfinder_landmark": "Montara Mountain",
    },
    "pedro_point": {
        "slug": "pedro-point-headlands",
        "title_focus": "Coastal Sunset & Cirrus Forecast",
        "access_note": "Pedro Point Headlands South Trail from Linda Mar (2.2 mi round trip).",
        "recommended_lens": "16–35 mm + 24–70 mm",
        "viewfinder_landmark": "Pedro Point Headlands",
    },
    "half_dome": {
        "slug": "half-dome-yosemite",
        "title_focus": "Alpenglow & Sunset Forecast",
        "access_note": "Best viewed from Glacier Point, Sentinel Dome, Olmsted Point, or Cook's Meadow.",
        "recommended_lens": "50–200 mm (Glacier Point) or 24–70 mm (Valley floor)",
        "viewfinder_landmark": "Half Dome",
    },
    "glacier_point": {
        "slug": "glacier-point-half-dome",
        "title_focus": "Half Dome Alpenglow & Sunset Forecast",
        "access_note": "Glacier Point overlook (7,214 ft ASL); seasonal road access May–Nov.",
        "recommended_lens": "24–70 mm + 70–200 mm",
        "viewfinder_landmark": "Half Dome",
    },
    "clouds_rest": {
        "slug": "clouds-rest-yosemite",
        "title_focus": "Sierra Alpenglow & Wave Cloud Forecast",
        "access_note": "Sunrise Lakes Trailhead at Tenaya Lake (14.5 mi RT, +2,300 ft gain).",
        "recommended_lens": "24–70 mm + 70–200 mm",
        "viewfinder_landmark": "Clouds Rest",
    },
    "tenaya_lake": {
        "slug": "tenaya-lake-yosemite",
        "title_focus": "Alpine Mirror Reflection & Alpenglow Forecast",
        "access_note": "Tioga Road pullouts along East Beach & West Shore (8,150 ft ASL).",
        "recommended_lens": "16–35 mm + 24–70 mm",
        "viewfinder_landmark": "Tenaya Lake",
    },
    "maroon_bells": {
        "slug": "maroon-bells-maroon-lake",
        "title_focus": "Sunrise Alpenglow & Mirror Reflection Forecast",
        "access_note": "Maroon Lake shoreline (9,580 ft ASL); RFTA shuttle / parking reservation required.",
        "recommended_lens": "24–70 mm (classic reflection) + 16–35 mm (autumn aspen foreground)",
    },
    "mesa_arch": {
        "slug": "mesa-arch-canyonlands",
        "title_focus": "Sunrise Underside Glow Forecast",
        "access_note": "Island in the Sky Mesa Arch Trail (0.6 mi easy loop; arrive 45+ min pre-dawn).",
        "recommended_lens": "16–24 mm ultra-wide for full arch + La Sal Mountains",
    },
}


def slugify_spot(spot: dict[str, Any]) -> str:
    """Returns the canonical SEO URL slug for a catalog spot."""
    override = SPOT_EDITORIAL_OVERRIDES.get(spot["id"], {})
    if override.get("slug"):
        return override["slug"]
    raw = str(spot["id"]).strip().lower().replace("_", "-")
    return re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")


def _fmt_hhmm(minutes_local: int) -> str:
    m_norm = int(minutes_local) % 1440
    return f"{m_norm // 60:02d}:{m_norm % 60:02d}"


def _default_date_for_lon(lon: float) -> str:
    tz_obj, _ = local_tz_for_lon(lon)
    return dt.datetime.now(tz_obj).date().isoformat()


def list_seo_guides(region: str | None = None) -> list[dict[str, Any]]:
    """Lists all SEO Light Guide catalog entries with canonical slugs and editorial metadata."""
    reg_filter = region.strip().lower() if region else None
    out: list[dict[str, Any]] = []
    for spot in REGIONAL_SPOT_CATALOG:
        if reg_filter and spot["region"].lower() != reg_filter:
            continue
        slug = slugify_spot(spot)
        arch = ARCHETYPE_PROFILES.get(spot["archetype"], ARCHETYPE_PROFILES["alpine_granite"])
        ov = SPOT_EDITORIAL_OVERRIDES.get(spot["id"], {})
        elev_ft = int(round(float(spot["elev_m"]) * 3.28084))
        out.append(
            {
                "id": spot["id"],
                "slug": slug,
                "name": spot["name"],
                "region": spot["region"],
                "region_label": REGION_LABELS.get(spot["region"], spot["region"].replace("_", " ").title()),
                "state": spot["state"],
                "lat": float(spot["lat"]),
                "lon": float(spot["lon"]),
                "elev_m": float(spot["elev_m"]),
                "elev_ft": elev_ft,
                "archetype": spot["archetype"],
                "is_lake": bool(spot.get("is_lake", False)),
                "best_phases": list(spot.get("best_phases", ("sunset",))),
                "primary_phase": spot.get("best_phases", ("sunset",))[0],
                "title_focus": ov.get("title_focus", arch["focus_keyword"]),
                "recommended_lens": ov.get("recommended_lens", arch["recommended_lens"]),
                "access_note": ov.get(
                    "access_note",
                    f"{spot['name']} ({elev_ft:,} ft ASL, {spot['state']}). Verify seasonal road & trail conditions.",
                ),
                "headline_phenomenon": arch["headline_phenomenon"],
                "field_tip": arch["field_tip"],
                "viewfinder_landmark": ov.get("viewfinder_landmark", spot["name"]),
                "guide_path": f"/guides/{slug}",
                "api_path": f"/api/v1/guides/{slug}",
            }
        )
    return out


def get_seo_spot(slug_or_id: str) -> dict[str, Any] | None:
    """Resolves either a hyphenated SEO slug (`mt-tamalpais-east-peak`) or catalog ID (`mt_tam_east`)."""
    needle = slug_or_id.strip().lower().strip("/")
    for item in list_seo_guides():
        if item["slug"] == needle or item["id"].lower() == needle:
            return item
    return None


def is_b2b_api_key_valid(provided_key: str | None) -> tuple[bool, str]:
    """
    Validates optional B2B API keys configured via `PHOTOPLANNER_B2B_API_KEYS` (comma-separated).
    When `PHOTOPLANNER_B2B_API_KEYS` is unset/empty, returns `(True, "open_preview")` so local
    development and evaluation work seamlessly; when configured, enforces exact key match.
    """
    configured = [
        k.strip()
        for k in os.environ.get("PHOTOPLANNER_B2B_API_KEYS", "").split(",")
        if k.strip()
    ]
    if not configured:
        return True, "open_preview"
    if provided_key and provided_key.strip() in configured:
        return True, "b2b_licensed"
    return False, "unauthorized"


def build_spot_guide_payload(
    slug_or_id: str,
    date_str: str | None = None,
    phase: str | None = None,
    include_live_forecast: bool = True,
    base_url: str = "",
) -> dict[str, Any]:
    """
    Builds the complete structured SEO Light Guide + B2B API JSON payload for a destination.

    Computes exact local sunrise & sunset golden-hour ephemeris (zero network cost) and, when
    `include_live_forecast=True`, hydrates live calibrated 380 km corridor transmittance,
    probabilistic P10/P50/P90 fog-sea lid, Lake Mirror Index, and causal attribution.
    """
    guide = get_seo_spot(slug_or_id)
    if guide is None:
        raise KeyError(f"Unknown SEO Light Guide spot: {slug_or_id}")

    raw_spot = next(s for s in REGIONAL_SPOT_CATALOG if s["id"] == guide["id"])
    resolved_date = (date_str or _default_date_for_lon(guide["lon"])).strip()
    req_d = dt.date.fromisoformat(resolved_date)
    resolved_phase = (phase or guide["primary_phase"]).strip().lower()
    if resolved_phase not in ("sunrise", "sunset"):
        resolved_phase = "sunset"

    tz_obj, tz_name = local_tz_for_lon(guide["lon"])
    noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz_obj)
    tz_off = (noon_loc.utcoffset() or dt.timedelta(0)).total_seconds() / 3600.0
    (sr_min, sr_az, sr_el), (ss_min, ss_az, ss_el) = _find_sunrise_sunset_golden_hours(
        guide["lat"], guide["lon"], req_d, tz_off
    )

    ephemeris = {
        "timezone": tz_name,
        "sunrise_peak_local": _fmt_hhmm(sr_min),
        "sunrise_window_local": f"{_fmt_hhmm(sr_min - 28)} – {_fmt_hhmm(sr_min + 36)}",
        "sunrise_azimuth_deg": round(sr_az, 1),
        "sunrise_compass": _compass_cardinal(sr_az),
        "sunset_peak_local": _fmt_hhmm(ss_min),
        "sunset_window_local": f"{_fmt_hhmm(ss_min - 42)} – {_fmt_hhmm(ss_min + 26)}",
        "sunset_azimuth_deg": round(ss_az, 1),
        "sunset_compass": _compass_cardinal(ss_az),
    }

    forecast_hydrated = False
    forecast_error: str | None = None
    evaluation: dict[str, Any] | None = None
    raw_weather_meta: dict[str, Any] = {}

    if include_live_forecast:
        try:
            _, az, el = (sr_min, sr_az, sr_el) if resolved_phase == "sunrise" else (ss_min, ss_az, ss_el)
            raw_json = solve_weather_and_corridor(
                guide["lat"],
                guide["lon"],
                resolved_date,
                phase=resolved_phase,
                sun_az_deg=az,
                sun_el_deg=el,
                summit_elev_m=guide["elev_m"],
                landmark_name=guide["name"],
                replay_mode="auto",
            )
            doc = json.loads(raw_json)
            evaluation = classify_spot_phenomena_and_attribution(raw_spot, doc)
            inv = doc.get("inversion_model") or {}
            raw_weather_meta = {
                "color_temperature_kelvin": doc.get("color_temperature_kelvin", 2400),
                "temp_f": doc.get("temp_sunset_f"),
                "wind_mph": doc.get("wind_mph"),
                "fog_top_p10_ft": inv.get("fog_top_p10_ft"),
                "fog_top_p50_ft": inv.get("fog_top_p50_ft", inv.get("fog_top_ft")),
                "fog_top_p90_ft": inv.get("fog_top_p90_ft"),
                "breakout_verdict": inv.get("breakout_verdict"),
                "recommendation": doc.get("recommendation"),
            }
            forecast_hydrated = True
        except Exception as exc:
            forecast_error = str(exc)

    if evaluation is None:
        # Clean deterministic fallback when `include_live_forecast=False` or upstream NWP is unreachable
        evaluation = {
            "spot_id": guide["id"],
            "name": guide["name"],
            "verdict": "EPHEMERIS_READY",
            "opportunity_score": None,
            "primary_phenomenon": guide["headline_phenomenon"],
            "phenomena": [],
            "causal_attribution": (
                f"Live 380 km corridor forecast ready on demand for {guide['name']} ({resolved_date}, "
                f"{resolved_phase}). Open the interactive 3D Viewfinder to run real-time HRRR + GOES-18 + "
                f"calibrated transmittance raycasting."
            ),
            "metrics": {
                "alpenglow_score": None,
                "canvas_score": None,
                "calibrated_p10_pct": None,
                "calibrated_p50_pct": None,
                "calibrated_p90_pct": None,
                "fog_top_p50_ft": None,
                "fog_top_p90_ft": None,
                "summit_above_fog_prob_pct": None,
                "dominant_blocker": None,
                "blocker_distance_km": None,
                "mirror_index": None,
            },
        }

    m = evaluation["metrics"]
    canonical_root = base_url.rstrip("/")
    canonical_url = f"{canonical_root}{guide['guide_path']}" if canonical_root else guide["guide_path"]

    phase_title = "Sunrise" if resolved_phase == "sunrise" else "Sunset"
    seo_title = f"{guide['name']} {guide['title_focus']} ({resolved_date}) | LightFinder"
    if forecast_hydrated and m.get("calibrated_p50_pct") is not None:
        seo_description = (
            f"{guide['name']} ({guide['elev_ft']:,} ft) {phase_title.lower()} light forecast for {resolved_date}: "
            f"{evaluation['verdict']} ({evaluation['opportunity_score']}/100), calibrated beam P50 "
            f"{m['calibrated_p50_pct']:.0f}% (P10–P90 {m['calibrated_p10_pct']:.0f}–{m['calibrated_p90_pct']:.0f}%), "
            f"golden hour {ephemeris[f'{resolved_phase}_window_local']}."
        )
    else:
        seo_description = (
            f"Daily 3D raycasted {guide['title_focus'].lower()} for {guide['name']} ({guide['elev_ft']:,} ft, "
            f"{guide['state']}). Exact {phase_title.lower()} golden hour {ephemeris[f'{resolved_phase}_window_local']}, "
            f"380 km light corridor odds, fog-sea lid height, and recommended lenses."
        )

    # Dynamic location-specific FAQs for rich Google SERP snippets
    faqs: list[dict[str, str]] = [
        {
            "question": f"What time is golden hour at {guide['name']} on {resolved_date}?",
            "answer": (
                f"Sunrise golden hour peaks around {ephemeris['sunrise_peak_local']} "
                f"(window {ephemeris['sunrise_window_local']}, solar azimuth {ephemeris['sunrise_azimuth_deg']}° "
                f"{ephemeris['sunrise_compass']}). Sunset golden hour peaks around {ephemeris['sunset_peak_local']} "
                f"(window {ephemeris['sunset_window_local']}, azimuth {ephemeris['sunset_azimuth_deg']}° "
                f"{ephemeris['sunset_compass']}) local time ({tz_name})."
            ),
        },
        {
            "question": f"Will {guide['name']} have good {phase_title.lower()} light or clear the fog on {resolved_date}?",
            "answer": str(evaluation["causal_attribution"]),
        },
        {
            "question": f"What lens focal length works best at {guide['name']}?",
            "answer": (
                f"For full-frame 35mm cameras, bring {guide['recommended_lens']}. "
                f"{guide['field_tip']}"
            ),
        },
    ]

    # Related spots in the same region for internal SEO link equity
    nearby = [
        {
            "slug": s["slug"],
            "name": s["name"],
            "elev_ft": s["elev_ft"],
            "state": s["state"],
            "guide_path": s["guide_path"],
            "title_focus": s["title_focus"],
        }
        for s in list_seo_guides(region=guide["region"])
        if s["id"] != guide["id"]
    ][:6]

    json_ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "TouristAttraction",
                "name": guide["name"],
                "description": seo_description,
                "url": canonical_url,
                "geo": {
                    "@type": "GeoCoordinates",
                    "latitude": guide["lat"],
                    "longitude": guide["lon"],
                    "elevation": f"{guide['elev_m']:.0f} m",
                },
                "address": {
                    "@type": "PostalAddress",
                    "addressRegion": guide["state"],
                    "addressCountry": "US",
                },
            },
            {
                "@type": "FAQPage",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": f["question"],
                        "acceptedAnswer": {"@type": "Answer", "text": f["answer"]},
                    }
                    for f in faqs
                ],
            },
        ],
    }

    return {
        "schema_version": "seo_guide_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "date": resolved_date,
        "phase": resolved_phase,
        "forecast_hydrated": forecast_hydrated,
        "forecast_error": forecast_error,
        "spot": guide,
        "ephemeris": ephemeris,
        "evaluation": evaluation,
        "weather_context": raw_weather_meta,
        "seo": {
            "title": seo_title,
            "description": seo_description,
            "canonical_url": canonical_url,
            "faqs": faqs,
            "json_ld": json_ld,
        },
        "nearby_guides": nearby,
    }


def _verdict_badge_class(verdict: str) -> str:
    v = str(verdict or "").upper()
    if v == "GO":
        return "badge-go"
    if v == "GAMBLE":
        return "badge-gamble"
    if v == "BUST":
        return "badge-bust"
    return "badge-neutral"


def render_spot_guide_html(payload: dict[str, Any]) -> str:
    """Renders a fast, zero-dependency, SEO-optimized HTML landing page for a single destination."""
    spot = payload["spot"]
    eph = payload["ephemeris"]
    ev = payload["evaluation"]
    m = ev["metrics"]
    seo = payload["seo"]
    ctx = payload.get("weather_context") or {}
    phase = payload["phase"]
    other_phase = "sunrise" if phase == "sunset" else "sunset"
    date_str = payload["date"]

    def esc(val: Any) -> str:
        return html.escape(str(val if val is not None else ""))

    p10 = m.get("calibrated_p10_pct")
    p50 = m.get("calibrated_p50_pct")
    p90 = m.get("calibrated_p90_pct")
    has_probs = p50 is not None
    p10_num = max(0.0, min(100.0, float(p10 or 0.0)))
    p50_num = max(0.0, min(100.0, float(p50 or 0.0)))
    p90_num = max(p10_num, min(100.0, float(p90 or p50_num)))

    fog_p50 = ctx.get("fog_top_p50_ft") or m.get("fog_top_p50_ft")
    fog_p10 = ctx.get("fog_top_p10_ft")
    fog_p90 = ctx.get("fog_top_p90_ft") or m.get("fog_top_p90_ft")
    above_fog = m.get("summit_above_fog_prob_pct")
    mirror_idx = m.get("mirror_index")

    verdict_cls = _verdict_badge_class(ev.get("verdict", ""))
    score_str = f"{ev['opportunity_score']}/100" if ev.get("opportunity_score") is not None else "Live On-Demand"

    viewfinder_href = (
        f"/?landmark={html.escape(spot['viewfinder_landmark'])}&date={esc(date_str)}"
    )
    json_ld_str = json.dumps(seo["json_ld"], ensure_ascii=False).replace("</", "<\\/")

    phenomena_pills = "".join(
        f'<span class="tag">{esc(p["label"])} · {int(p["score"])}</span>'
        for p in (ev.get("phenomena") or [])[:4]
    ) or f'<span class="tag">{esc(spot["headline_phenomenon"])}</span>'

    faq_html = "".join(
        f"""
        <details class="faq-item" open>
          <summary>{esc(f['question'])}</summary>
          <p>{esc(f['answer'])}</p>
        </details>"""
        for f in seo["faqs"]
    )

    nearby_html = "".join(
        f"""
        <a class="nearby-card" href="{esc(n['guide_path'])}?date={esc(date_str)}">
          <strong>{esc(n['name'])}</strong>
          <span>{n['elev_ft']:,} ft · {esc(n['state'])} · {esc(n['title_focus'])}</span>
        </a>"""
        for n in payload.get("nearby_guides", [])
    )

    fog_block_val = (
        f"{int(round(fog_p50)):,} ft ASL"
        if fog_p50 is not None
        else "Run Live Forecast"
    )
    fog_sub = (
        f"Band {int(round(fog_p10)):,}–{int(round(fog_p90)):,} ft · Breakout Odds {above_fog:.0f}%"
        if (fog_p10 is not None and fog_p90 is not None and above_fog is not None)
        else f"Summit {spot['elev_ft']:,} ft ASL ({esc(ctx.get('breakout_verdict') or 'Probabilistic P10–P90 Inversion Lid')})"
    )

    mirror_card = (
        f"""
      <div class="stat-card">
        <div class="stat-label">Lake Mirror Reflection Index</div>
        <div class="stat-val">{int(round(mirror_idx)) if mirror_idx is not None else '--'} / 100</div>
        <div class="stat-sub">Sheltered upwind cirque horizon + 10 m wind calm check</div>
      </div>"""
        if spot["is_lake"]
        else f"""
      <div class="stat-card">
        <div class="stat-label">Recommended Full-Frame Optics</div>
        <div class="stat-val" style="font-size:1.05rem;">{esc(spot['recommended_lens'])}</div>
        <div class="stat-sub">Test exact 16–200 mm framing in the 3D Lens Viewfinder</div>
      </div>"""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{esc(seo['title'])}</title>
  <meta name="description" content="{esc(seo['description'])}" />
  <link rel="canonical" href="{esc(seo['canonical_url'])}" />
  <meta property="og:type" content="article" />
  <meta property="og:title" content="{esc(seo['title'])}" />
  <meta property="og:description" content="{esc(seo['description'])}" />
  <meta property="og:url" content="{esc(seo['canonical_url'])}" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="{esc(seo['title'])}" />
  <meta name="twitter:description" content="{esc(seo['description'])}" />
  <script type="application/ld+json">{json_ld_str}</script>
  <style>
    :root {{
      --bg: #0a0f1a; --surface: #111827; --surface-2: #162033;
      --border: #24324a; --text: #f1f5f9; --muted: #94a3b8;
      --amber: #f59e0b; --gold: #fde68a; --sky: #38bdf8;
      --go: #22c55e; --gamble: #f59e0b; --bust: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.55; }}
    a {{ color: var(--sky); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 20px 18px 56px; }}
    nav.crumb {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; font-size: 0.84rem; color: var(--muted); padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
    .nav-links {{ display: flex; gap: 14px; align-items: center; }}
    header.hero {{ padding: 26px 0 20px; display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; flex-wrap: wrap; }}
    .eyebrow {{ font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--amber); font-weight: 700; }}
    h1 {{ margin: 6px 0 8px; font-size: clamp(1.55rem, 3.2vw, 2.2rem); line-height: 1.18; letter-spacing: -0.02em; }}
    .subline {{ color: var(--muted); font-size: 0.94rem; margin: 0; }}
    .verdict-box {{ display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }}
    .badge {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px; font-weight: 800; font-size: 0.88rem; letter-spacing: 0.03em; font-variant-numeric: tabular-nums; }}
    .badge-go {{ background: rgba(34,197,94,0.16); border: 1px solid rgba(34,197,94,0.6); color: #86efac; }}
    .badge-gamble {{ background: rgba(245,158,11,0.16); border: 1px solid rgba(245,158,11,0.6); color: var(--gold); }}
    .badge-bust {{ background: rgba(239,68,68,0.16); border: 1px solid rgba(239,68,68,0.6); color: #fca5a5; }}
    .badge-neutral {{ background: rgba(56,189,248,0.14); border: 1px solid rgba(56,189,248,0.5); color: #bae6fd; }}
    .cta-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }}
    .btn {{ display: inline-flex; align-items: center; gap: 6px; padding: 9px 15px; border-radius: 8px; font-weight: 700; font-size: 0.86rem; border: 1px solid var(--border); background: var(--surface-2); color: var(--text); }}
    .btn:hover {{ text-decoration: none; border-color: var(--amber); }}
    .btn-primary {{ background: #d97706; border-color: #f59e0b; color: #0a0f1a; }}
    .btn-primary:hover {{ background: #f59e0b; }}
    .tags {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 0; }}
    .tag {{ padding: 4px 10px; border-radius: 999px; background: var(--surface-2); border: 1px solid var(--border); font-size: 0.76rem; color: #cbd5e1; font-weight: 600; }}
    .attribution {{ margin: 18px 0; padding: 16px 18px; border-radius: 12px; background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--amber); }}
    .attribution strong {{ display: block; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--gold); margin-bottom: 6px; }}
    .attribution p {{ margin: 0; font-size: 0.96rem; color: #e2e8f0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
    .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 15px; }}
    .stat-label {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); font-weight: 700; }}
    .stat-val {{ font-size: 1.32rem; font-weight: 800; margin-top: 6px; font-variant-numeric: tabular-nums; color: #f8fafc; }}
    .stat-sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 5px; }}
    .prob-bar {{ position: relative; height: 10px; border-radius: 999px; background: #1e293b; margin-top: 10px; overflow: hidden; }}
    .prob-range {{ position: absolute; top: 0; bottom: 0; background: linear-gradient(90deg, rgba(56,189,248,0.5), rgba(245,158,11,0.75)); border-radius: 999px; }}
    .prob-median {{ position: absolute; top: -2px; bottom: -2px; width: 3px; background: #fef08a; }}
    h2 {{ font-size: 1.05rem; margin: 28px 0 12px; letter-spacing: -0.01em; color: #f8fafc; }}
    .two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; }}
    .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
    .panel p {{ margin: 6px 0; font-size: 0.9rem; color: #cbd5e1; }}
    .faq-item {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 15px; margin-bottom: 10px; }}
    .faq-item summary {{ cursor: pointer; font-weight: 700; font-size: 0.92rem; color: #f8fafc; }}
    .faq-item p {{ margin: 8px 0 0; font-size: 0.88rem; color: #cbd5e1; }}
    .nearby-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }}
    .nearby-card {{ display: flex; flex-direction: column; gap: 3px; padding: 12px 14px; border-radius: 10px; background: var(--surface); border: 1px solid var(--border); color: var(--text); }}
    .nearby-card:hover {{ border-color: var(--sky); text-decoration: none; }}
    .nearby-card strong {{ font-size: 0.9rem; }}
    .nearby-card span {{ font-size: 0.77rem; color: var(--muted); }}
    footer {{ margin-top: 34px; padding-top: 16px; border-top: 1px solid var(--border); display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; font-size: 0.78rem; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <nav class="crumb" aria-label="Breadcrumb">
      <div><a href="/">LightFinder 3D</a> / <a href="/guides">Light Guides</a> / <a href="/guides?region={esc(spot['region'])}">{esc(spot['region_label'])}</a></div>
      <div class="nav-links">
        <a href="/api/backtest-report">Accuracy Scoreboard</a>
        <a href="{esc(spot['api_path'])}?date={esc(date_str)}&amp;phase={esc(phase)}">JSON API</a>
      </div>
    </nav>

    <header class="hero">
      <div>
        <div class="eyebrow">{esc(spot['region_label'])} · {spot['elev_ft']:,} FT ASL · {esc(phase.upper())} BRIEFING</div>
        <h1>{esc(spot['name'])} {esc(spot['title_focus'])}</h1>
        <p class="subline">Calibrated 380 km curved-Earth solar corridor &amp; 3D terrain light forecast for <strong>{esc(date_str)}</strong> ({esc(eph['timezone'])})</p>
        <div class="tags">{phenomena_pills}</div>
        <div class="cta-row">
          <a class="btn btn-primary" href="{viewfinder_href}">📷 Open in 3D Lens Viewfinder</a>
          <a class="btn" href="{esc(spot['guide_path'])}?date={esc(date_str)}&amp;phase={esc(other_phase)}">Switch to {other_phase.title()} Forecast</a>
        </div>
      </div>
      <div class="verdict-box">
        <span class="badge {verdict_cls}">{esc(ev['verdict'])} · {esc(score_str)}</span>
        <span style="font-size:0.78rem;color:var(--muted);">Updated {esc(payload['generated_at_utc'])}</span>
      </div>
    </header>

    <section class="attribution" aria-label="Causal Light Attribution">
      <strong>Causal Light &amp; Bust Attribution ({esc(ev['primary_phenomenon'])})</strong>
      <p>{esc(ev['causal_attribution'])}</p>
    </section>

    <section class="grid" aria-label="Key Probabilistic Light Metrics">
      <div class="stat-card">
        <div class="stat-label">Calibrated Beam Odds (380 km)</div>
        <div class="stat-val">{f"P50 {p50_num:.0f}%" if has_probs else "On Demand"}</div>
        <div class="stat-sub">{f"80% Band: P10 {p10_num:.0f}% – P90 {p90_num:.0f}% · Alpenglow {m['alpenglow_score']}/100" if has_probs else "Calibrated against NOAA SURFRAD & 2 km NSRDB"}</div>
        <div class="prob-bar" aria-hidden="true">
          <div class="prob-range" style="left:{p10_num:.1f}%;width:{max(2.0, p90_num - p10_num):.1f}%;"></div>
          <div class="prob-median" style="left:{p50_num:.1f}%;"></div>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Golden Hour ({esc(phase.title())})</div>
        <div class="stat-val">{esc(eph[f'{phase}_peak_local'])}</div>
        <div class="stat-sub">Window {esc(eph[f'{phase}_window_local'])} · Az {eph[f'{phase}_azimuth_deg']}° {esc(eph[f'{phase}_compass'])}</div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Probabilistic Fog-Sea Lid</div>
        <div class="stat-val">{esc(fog_block_val)}</div>
        <div class="stat-sub">{esc(fog_sub)}</div>
      </div>

      {mirror_card}
    </section>

    <section class="two-col">
      <article class="panel">
        <h2 style="margin-top:0;">Solar Geometry &amp; Field Logistics</h2>
        <p><strong>Sunrise Golden Hour:</strong> {esc(eph['sunrise_window_local'])} (peak {esc(eph['sunrise_peak_local'])}, Az {eph['sunrise_azimuth_deg']}° {esc(eph['sunrise_compass'])})</p>
        <p><strong>Sunset Golden Hour:</strong> {esc(eph['sunset_window_local'])} (peak {esc(eph['sunset_peak_local'])}, Az {eph['sunset_azimuth_deg']}° {esc(eph['sunset_compass'])})</p>
        <p><strong>GPS &amp; Elevation:</strong> {spot['lat']:.4f}, {spot['lon']:.4f} · {spot['elev_ft']:,} ft ({spot['elev_m']:.0f} m ASL)</p>
        <p><strong>Access &amp; Vantage:</strong> {esc(spot['access_note'])}</p>
      </article>

      <article class="panel">
        <h2 style="margin-top:0;">Photographer Field Strategy</h2>
        <p><strong>Optics:</strong> {esc(spot['recommended_lens'])}</p>
        <p>{esc(spot['field_tip'])}</p>
        <p><strong>Independent Verification:</strong> Every LightFinder light probability is out-of-sample calibrated against NOAA SURFRAD pyrheliometers and 2 km GOES-18 NSRDB irradiance (<a href="/api/backtest-report">inspect public scoreboard</a>).</p>
      </article>
    </section>

    <section aria-label="Frequently Asked Questions">
      <h2>Frequently Asked Questions — {esc(spot['name'])}</h2>
      {faq_html}
    </section>

    <section aria-label="Nearby Regional Light Guides">
      <h2>Nearby Light Guides in {esc(spot['region_label'])}</h2>
      <div class="nearby-grid">{nearby_html}</div>
    </section>

    <footer>
      <span>LightFinder 3D · Raycasted Terrain + Calibrated Atmospheric Physics</span>
      <span>B2B Workshop &amp; Outdoor App API: <a href="/api/v1/guides">/api/v1/guides</a> · <a href="/sitemap.xml">Sitemap</a></span>
    </footer>
  </div>
</body>
</html>
"""


def render_guides_directory_html(
    region: str | None = None,
    date_str: str | None = None,
    base_url: str = "",
) -> str:
    """Renders the crawlable `/guides` directory page linking all regional SEO Light Guides."""
    items = list_seo_guides(region=region)
    resolved_date = date_str or _default_date_for_lon(-120.0)
    canonical_root = base_url.rstrip("/")
    canonical_url = f"{canonical_root}/guides" if canonical_root else "/guides"

    by_region: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_region.setdefault(item["region"], []).append(item)

    filter_pills = ['<a class="pill' + (" is-active" if not region else "") + '" href="/guides">All Regions (' + str(len(list_seo_guides())) + ")</a>"]
    for reg_key, reg_label in REGION_LABELS.items():
        cnt = len(list_seo_guides(region=reg_key))
        if cnt == 0:
            continue
        active = " is-active" if (region and region.lower() == reg_key) else ""
        filter_pills.append(
            f'<a class="pill{active}" href="/guides?region={html.escape(reg_key)}">{html.escape(reg_label)} ({cnt})</a>'
        )

    sections_html = []
    for reg_key, reg_spots in by_region.items():
        reg_label = REGION_LABELS.get(reg_key, reg_key.title())
        cards = "".join(
            f"""
            <a class="guide-card" href="{html.escape(s['guide_path'])}?date={html.escape(resolved_date)}">
              <div class="guide-top">
                <strong>{html.escape(s['name'])}</strong>
                <span class="elev">{s['elev_ft']:,} ft</span>
              </div>
              <div class="guide-focus">{html.escape(s['title_focus'])}</div>
              <div class="guide-meta">{html.escape(s['state'])} · Best: {html.escape(', '.join(p.title() for p in s['best_phases']))} · {html.escape(s['archetype'].replace('_', ' ').title())}</div>
            </a>"""
            for s in reg_spots
        )
        sections_html.append(
            f"""
            <section class="region-block">
              <h2>{html.escape(reg_label)}</h2>
              <div class="cards-grid">{cards}</div>
            </section>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Daily Landscape Photography Light Guides ({html.escape(resolved_date)}) | LightFinder 3D</title>
  <meta name="description" content="Daily calibrated sunrise, sunset, cloud-sea breakout, alpenglow, and mirror lake light forecasts for {len(list_seo_guides())} iconic US landscape photography destinations." />
  <link rel="canonical" href="{html.escape(canonical_url)}" />
  <style>
    :root {{ --bg:#0a0f1a; --surface:#111827; --border:#24324a; --text:#f1f5f9; --muted:#94a3b8; --amber:#f59e0b; --sky:#38bdf8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
    a {{ color:var(--sky); text-decoration:none; }}
    .wrap {{ max-width:1080px; margin:0 auto; padding:22px 18px 56px; }}
    nav {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; padding-bottom:14px; border-bottom:1px solid var(--border); font-size:0.86rem; }}
    header {{ padding:24px 0 16px; }}
    h1 {{ margin:6px 0; font-size:clamp(1.6rem, 3.2vw, 2.25rem); letter-spacing:-0.02em; }}
    .lead {{ margin:0; color:var(--muted); max-width:740px; }}
    .pills {{ display:flex; gap:8px; flex-wrap:wrap; margin:18px 0 10px; }}
    .pill {{ padding:6px 12px; border-radius:999px; border:1px solid var(--border); background:var(--surface); color:#cbd5e1; font-size:0.79rem; font-weight:600; }}
    .pill.is-active, .pill:hover {{ border-color:var(--amber); color:#fde68a; }}
    .region-block {{ margin-top:26px; }}
    h2 {{ font-size:1.1rem; margin:0 0 12px; color:#f8fafc; }}
    .cards-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:12px; }}
    .guide-card {{ display:flex; flex-direction:column; gap:5px; padding:14px; border-radius:12px; background:var(--surface); border:1px solid var(--border); color:var(--text); }}
    .guide-card:hover {{ border-color:var(--amber); }}
    .guide-top {{ display:flex; justify-content:space-between; gap:8px; align-items:baseline; }}
    .guide-top strong {{ font-size:0.94rem; }}
    .elev {{ font-size:0.74rem; color:#fde68a; font-variant-numeric:tabular-nums; white-space:nowrap; }}
    .guide-focus {{ font-size:0.82rem; color:#bae6fd; font-weight:600; }}
    .guide-meta {{ font-size:0.75rem; color:var(--muted); }}
    footer {{ margin-top:36px; padding-top:16px; border-top:1px solid var(--border); font-size:0.78rem; color:var(--muted); display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <nav>
      <div><a href="/">← LightFinder 3D Viewfinder</a></div>
      <div style="display:flex;gap:14px;">
        <a href="/api/backtest-report">Public Accuracy Scoreboard</a>
        <a href="/api/v1/guides">B2B JSON API</a>
        <a href="/sitemap.xml">Sitemap</a>
      </div>
    </nav>

    <header>
      <div style="font-size:0.73rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--amber);font-weight:700;">DAILY 3D LIGHT INTELLIGENCE DIRECTORY</div>
      <h1>Dynamic Landscape Light Guides</h1>
      <p class="lead">Daily-updated sunrise, sunset, cloud-sea breakout, alpenglow, and mirror-lake light forecasts powered by 1m LiDAR + SRTM raycasting and calibrated 380 km solar corridor physics.</p>
      <div class="pills">{"".join(filter_pills)}</div>
    </header>

    {"".join(sections_html)}

    <footer>
      <span>{len(items)} Active Light Guides · Calibrated against NOAA SURFRAD &amp; 2 km GOES-18 NSRDB</span>
      <span><a href="/docs">OpenAPI / B2B Docs</a> · <a href="/robots.txt">robots.txt</a></span>
    </footer>
  </div>
</body>
</html>
"""


def render_sitemap_xml(base_url: str, date_str: str | None = None) -> str:
    """Generates a valid XML sitemap covering the app, directory, scoreboard, and all guide pages."""
    root = base_url.rstrip("/") or "https://photoplanner.app"
    lastmod = date_str or _default_date_for_lon(-120.0)
    urls = [
        (f"{root}/", "1.0", "daily"),
        (f"{root}/guides", "0.9", "daily"),
        (f"{root}/api/backtest-report", "0.7", "weekly"),
    ]
    for g in list_seo_guides():
        urls.append((f"{root}{g['guide_path']}", "0.8", "daily"))

    entries = "\n".join(
        f"  <url><loc>{html.escape(loc)}</loc><lastmod>{html.escape(lastmod)}</lastmod>"
        f"<changefreq>{freq}</changefreq><priority>{prio}</priority></url>"
        for loc, prio, freq in urls
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}\n"
        "</urlset>\n"
    )


def render_robots_txt(base_url: str) -> str:
    """Generates `robots.txt` permitting search indexing of Light Guides and pointing to `/sitemap.xml`."""
    root = base_url.rstrip("/") or "https://photoplanner.app"
    return (
        "User-agent: *\n"
        "Allow: /\n"
        "Allow: /guides\n"
        "Allow: /guides/\n"
        "Disallow: /api/watchlists\n"
        "Disallow: /api/webcam-image\n"
        f"Sitemap: {root}/sitemap.xml\n"
    )
