"""
Scalable Multi-Regime Backtesting, Diagnostics & Verification Suite (`planner/weather/benchmark_suite.py`).

Evaluates PhotoPlanner's atmospheric corridor & light-casting engine across independent
physical ground-truth sensors, multiple US geographic/elevation regimes, all 4 seasons,
and BOTH sunrises and sunsets:

Verification Tiers (strictly separated — never mixed):
  - Tier 1A: NOAA SURFRAD 1-minute Normal Incidence Pyrheliometers (ground-based direct beam)
  - Tier 1B: NREL NSRDB 2 km 5-minute GOES CONUS v4.0.0 Direct Normal Irradiance (satellite)
  - Tier 1C: NWS 00Z/12Z RAOB radiosondes & ASOS laser ceilometers (inversion lid & cloud base)
  - Tier 2 : Internal `t=0` model consistency (diagnostic only, never mixed into Tier 1)

Capabilities:
  - Stratified suites: `ca24`, `surfrad`, `conus_multi`, `all`, plus programmatic generator
    scaling cleanly to 500–1,000+ events.
  - Full proper scoring rules: MAE, RMSE, Bias, Pearson r, Spearman rho, CRPS, Pinball Loss
    (P10/P50/P90), 80% PI coverage/width, plus binary golden-hour hit Brier, BSS, ECE, POD, FAR, CSI.
  - Automated failure-mode diagnostics (`diagnose_failures`) pinpointing physical root causes.
  - Experiment ledger (`data/weather_benchmarks/experiments/`) with paired bootstrap CIs (`--compare`).
  - Self-contained interactive HTML verification dashboard with inline SVG plots.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .backtest import BENCHMARK_EVENTS as CA24_BENCHMARK_EVENTS
from .calibration import (
    DEFAULT_CALIBRATION_MODEL_PATH,
    TransmittanceCalibrator,
    brier_score,
    brier_skill_score,
    crps_from_quantiles,
    evaluate_calibration_out_of_sample,
    expected_calibration_error,
    pinball_loss,
    save_calibration_bundle,
)
from .corridor_solver import (
    _find_sunrise_sunset_golden_hours,
    solve_weather_and_corridor,
)
from .nwp_corridor import local_tz_for_lon
from .raob_sounding import fetch_raob_sounding
from .station_obs import fetch_nearest_station_observation
from .surfrad_truth import SURFRAD_STATIONS, fetch_surfrad_golden_hour_truth
from .webcam_truth import fetch_nsrdb_2km_golden_hour_dni

ROOT_DIR = Path(__file__).resolve().parents[2]
SUITE_CACHE_DIR = ROOT_DIR / "data" / "weather_cache" / "benchmark_suite_v1"
EXPERIMENTS_DIR = ROOT_DIR / "data" / "weather_benchmarks" / "experiments"
VERIFICATION_HTML_PATH = ROOT_DIR / "data" / "weather_benchmarks" / "verification_report.html"

DEFAULT_EVAL_MODES: tuple[str, ...] = (
    "nowcast_1h",
    "nowcast_3h",
    "hindcast_24h",
    "hindcast_72h",
)

# Stratified 2024 seasonal dates (all >= 2024-02-10 so Open-Meteo previous-runs-api has genuine Day-1/Day-3 hindcasts)
_SEASONAL_DATES_2024: dict[str, list[str]] = {
    "winter": ["2024-02-10", "2024-02-24", "2024-12-14"],
    "spring": ["2024-03-16", "2024-04-20", "2024-05-18"],
    "summer": ["2024-06-15", "2024-07-20", "2024-08-17"],
    "autumn": ["2024-09-21", "2024-10-19", "2024-11-16"],
}

# Stratified 2025 seasonal dates spanning clear, frontal, monsoon, and marine-layer patterns
_SEASONAL_DATES_2025: dict[str, list[str]] = {
    "winter": ["2025-01-18", "2025-02-08", "2025-02-22", "2025-12-06"],
    "spring": ["2025-03-22", "2025-04-12", "2025-05-06", "2025-05-15"],
    "summer": ["2025-06-14", "2025-07-15", "2025-08-09", "2025-08-23"],
    "autumn": ["2025-09-20", "2025-10-11", "2025-10-25", "2025-11-15"],
}

ALL_SURFRAD_STATION_CODES: tuple[str, ...] = ("tbl", "dra", "fpk", "sxf", "bon", "gwn", "psu")

# Iconic CONUS landscape photography sites across 5 distinct climate regimes
CONUS_PHOTO_SITES: list[dict[str, Any]] = [
    {
        "slug": "rainier",
        "site": "Mt. Rainier Paradise, WA",
        "region": "pnw_alpine",
        "subregion": "pnw_cascades",
        "lat": 46.7865,
        "lon": -121.7353,
        "elev_m": 1650.0,
        "phases": ("sunrise", "sunset"),
    },
    {
        "slug": "rialto",
        "site": "Rialto Beach, Olympic NP WA",
        "region": "pnw_coastal",
        "subregion": "pnw_coast",
        "lat": 47.9211,
        "lon": -124.6378,
        "elev_m": 15.0,
        "phases": ("sunset",),
    },
    {
        "slug": "maroon_bells",
        "site": "Maroon Bells, CO",
        "region": "rockies_alpine",
        "subregion": "colorado_rockies",
        "lat": 39.0989,
        "lon": -106.9410,
        "elev_m": 2920.0,
        "phases": ("sunrise",),
    },
    {
        "slug": "grand_teton",
        "site": "Grand Teton Schwabacher, WY",
        "region": "rockies_alpine",
        "subregion": "tetons_wy",
        "lat": 43.7114,
        "lon": -110.6716,
        "elev_m": 2005.0,
        "phases": ("sunrise",),
    },
    {
        "slug": "mesa_arch",
        "site": "Mesa Arch, Canyonlands UT",
        "region": "southwest_plateau",
        "subregion": "colorado_plateau",
        "lat": 38.3880,
        "lon": -109.8634,
        "elev_m": 1875.0,
        "phases": ("sunrise",),
    },
    {
        "slug": "grand_canyon",
        "site": "Grand Canyon Hopi Point, AZ",
        "region": "southwest_plateau",
        "subregion": "grand_canyon",
        "lat": 36.0746,
        "lon": -112.1549,
        "elev_m": 2155.0,
        "phases": ("sunset",),
    },
    {
        "slug": "half_dome_sr",
        "site": "Half Dome, Yosemite CA",
        "region": "sierra",
        "subregion": "high_sierra",
        "lat": 37.7459,
        "lon": -119.5332,
        "elev_m": 2693.0,
        "phases": ("sunrise",),
    },
    {
        "slug": "tam_sr",
        "site": "Mt. Tamalpais East Peak, CA",
        "region": "coastal",
        "subregion": "coastal_summit",
        "lat": 37.9235,
        "lon": -122.5965,
        "elev_m": 784.0,
        "phases": ("sunrise",),
    },
]


def _season_for_date(d: dt.date) -> str:
    m = d.month
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def build_ca24_events() -> list[dict[str, Any]]:
    """Returns the canonical 24 California 2025 sunset events with normalized metadata."""
    out: list[dict[str, Any]] = []
    for ev in CA24_BENCHMARK_EVENTS:
        row = dict(ev)
        row.setdefault("phase", "sunset")
        row.setdefault("suite", "ca24")
        row.setdefault("surfrad_station", None)
        out.append(row)
    return out


def build_surfrad_events(
    stations: tuple[str, ...] = ("tbl", "dra", "fpk", "sxf"),
    dates_per_season: int = 1,
    phases: tuple[str, ...] = ("sunrise", "sunset"),
    seasonal_dates: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Builds a stratified NOAA SURFRAD 1-minute ground pyrheliometer event catalog
    spanning multiple US climates, all 4 seasons, and both sunrises & sunsets.
    """
    date_map = seasonal_dates if seasonal_dates is not None else _SEASONAL_DATES_2025
    events: list[dict[str, Any]] = []
    for st_code in stations:
        meta = SURFRAD_STATIONS[st_code]
        for season, dates in date_map.items():
            for d_str in dates[: max(1, dates_per_season)]:
                for ph in phases:
                    events.append(
                        {
                            "id": f"surfrad_{st_code}_{d_str.replace('-', '_')}_{ph}",
                            "site": meta["name"],
                            "region": meta["region"],
                            "subregion": f"surfrad_{st_code}",
                            "season": season,
                            "lat": float(meta["lat"]),
                            "lon": float(meta["lon"]),
                            "elev_m": float(meta["elev_m"]),
                            "date": d_str,
                            "phase": ph,
                            "regime": f"{meta['region']}_{season}",
                            "suite": "surfrad",
                            "surfrad_station": st_code,
                            "description": f"SURFRAD {st_code.upper()} 1-min pyrheliometer ({season} {ph})",
                        }
                    )
    return events


def build_conus_multi_events(
    dates_per_season: int = 1,
    seasonal_dates: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Builds a multi-region CONUS landscape photography benchmark spanning PNW, Rockies,
    Colorado Plateau, Southwest, High Sierra sunrises, and Coastal Cloud-Sea sunrises.
    """
    date_map = seasonal_dates if seasonal_dates is not None else _SEASONAL_DATES_2025
    events: list[dict[str, Any]] = []
    for sp in CONUS_PHOTO_SITES:
        for season, dates in date_map.items():
            for d_str in dates[: max(1, dates_per_season)]:
                for ph in sp["phases"]:
                    events.append(
                        {
                            "id": f"conus_{sp['slug']}_{d_str.replace('-', '_')}_{ph}",
                            "site": sp["site"],
                            "region": sp["region"],
                            "subregion": sp["subregion"],
                            "season": season,
                            "lat": float(sp["lat"]),
                            "lon": float(sp["lon"]),
                            "elev_m": float(sp["elev_m"]),
                            "date": d_str,
                            "phase": ph,
                            "regime": f"{sp['region']}_{season}",
                            "suite": "conus_multi",
                            "surfrad_station": None,
                            "description": f"{sp['site']} ({season} {ph})",
                        }
                    )
    return events


def build_cross_year_events(
    surfrad_dates_per_season: int = 3,
    conus_dates_per_season: int = 2,
) -> list[dict[str, Any]]:
    """
    Builds a 500+ event cross-year (2024 vs 2025) independent multi-regime benchmark:
      - All 7 NOAA SURFRAD 1-min ground pyrheliometer stations (sunrise + sunset, 2024 & 2025)
      - All 8 iconic CONUS landscape photography landmarks (NSRDB 2 km GOES, 2024 & 2025)
      - Canonical 24 California coastal & Sierra sunsets (NSRDB 2 km + RAOB inversion truth)
    At defaults (3 SURFRAD + 2 CONUS dates/season/year): 336 Tier 1A + 144 Tier 1B + 24 CA24
    = 504 unique independent events (240 in 2024 + 264 in 2025) -> 2,016 leak-free forecast rows.
    """
    events: list[dict[str, Any]] = []
    events.extend(
        build_surfrad_events(
            stations=ALL_SURFRAD_STATION_CODES,
            dates_per_season=surfrad_dates_per_season,
            seasonal_dates=_SEASONAL_DATES_2024,
        )
    )
    events.extend(
        build_surfrad_events(
            stations=ALL_SURFRAD_STATION_CODES,
            dates_per_season=surfrad_dates_per_season,
            seasonal_dates=_SEASONAL_DATES_2025,
        )
    )
    events.extend(
        build_conus_multi_events(
            dates_per_season=conus_dates_per_season,
            seasonal_dates=_SEASONAL_DATES_2024,
        )
    )
    events.extend(
        build_conus_multi_events(
            dates_per_season=conus_dates_per_season,
            seasonal_dates=_SEASONAL_DATES_2025,
        )
    )
    events.extend(build_ca24_events())
    # Deduplicate defensively by event id
    seen: set[str] = set()
    unique_events: list[dict[str, Any]] = []
    for ev in events:
        if ev["id"] not in seen:
            seen.add(ev["id"])
            unique_events.append(ev)
    return unique_events


def build_stratified_events(
    suite: str = "all",
    limit: int | None = None,
    phases: tuple[str, ...] | None = None,
    dates_per_season: int = 1,
    year: int = 2025,
    cadence_days: int = 15,
) -> list[dict[str, Any]]:
    """
    Constructs a stratified benchmark event list by suite name:
      - `ca24`: Canonical 24 CA 2025 sunsets
      - `surfrad`: NOAA SURFRAD 1-min ground pyrheliometers (sunrise + sunset, 4 seasons)
      - `conus_multi`: Iconic US landscapes across 5 climates (sunrise + sunset, 4 seasons)
      - `all`: Combined `ca24` + `surfrad` + `conus_multi` (~92 events at dates_per_season=1)
      - `cross_year`: 504-event (2,016-row) 2024 vs 2025 multi-regime suite across all 7 SURFRAD + CONUS + CA24
      - `scaled`: Programmatic cadence grid across all 7 SURFRAD + CONUS sites for 500–1,000+ runs
    """
    s = suite.strip().lower()
    if s == "ca24":
        events = build_ca24_events()
    elif s == "surfrad":
        events = build_surfrad_events(dates_per_season=dates_per_season)
    elif s == "conus_multi":
        events = build_conus_multi_events(dates_per_season=dates_per_season)
    elif s == "all":
        events = (
            build_ca24_events()
            + build_surfrad_events(dates_per_season=dates_per_season)
            + build_conus_multi_events(dates_per_season=dates_per_season)
        )
    elif s == "cross_year":
        s_dps = max(1, dates_per_season) if dates_per_season > 1 else 3
        c_dps = max(1, min(3, dates_per_season)) if dates_per_season > 1 else 2
        events = build_cross_year_events(
            surfrad_dates_per_season=s_dps,
            conus_dates_per_season=c_dps,
        )
    elif s == "scaled":
        events = []
        d_start = dt.date(year, 2, 10) if year == 2024 else dt.date(year, 1, 5)
        d_end = dt.date(year, 12, 26)
        grid_dates: list[dt.date] = []
        d_cur = d_start
        while d_cur <= d_end:
            grid_dates.append(d_cur)
            d_cur += dt.timedelta(days=max(1, cadence_days))
        # Site-first ordering maximizes persistent Rust worker SRTM i16 + 1m grid LRU cache hits
        for st_code in ALL_SURFRAD_STATION_CODES:
            meta = SURFRAD_STATIONS[st_code]
            for d in grid_dates:
                d_str = d.isoformat()
                seas = _season_for_date(d)
                for ph in ("sunrise", "sunset"):
                    events.append(
                        {
                            "id": f"scaled_surfrad_{st_code}_{d_str.replace('-', '_')}_{ph}",
                            "site": meta["name"],
                            "region": meta["region"],
                            "subregion": f"surfrad_{st_code}",
                            "season": seas,
                            "lat": float(meta["lat"]),
                            "lon": float(meta["lon"]),
                            "elev_m": float(meta["elev_m"]),
                            "date": d_str,
                            "phase": ph,
                            "regime": f"{meta['region']}_{seas}",
                            "suite": "scaled",
                            "surfrad_station": st_code,
                        }
                    )
        for sp in CONUS_PHOTO_SITES:
            for d in grid_dates:
                d_str = d.isoformat()
                seas = _season_for_date(d)
                for ph in sp["phases"]:
                    events.append(
                        {
                            "id": f"scaled_{sp['slug']}_{d_str.replace('-', '_')}_{ph}",
                            "site": sp["site"],
                            "region": sp["region"],
                            "subregion": sp["subregion"],
                            "season": seas,
                            "lat": float(sp["lat"]),
                            "lon": float(sp["lon"]),
                            "elev_m": float(sp["elev_m"]),
                            "date": d_str,
                            "phase": ph,
                            "regime": f"{sp['region']}_{seas}",
                            "suite": "scaled",
                            "surfrad_station": None,
                        }
                    )
    else:
        raise ValueError(f"Unknown benchmark suite: {suite!r}")

    if phases:
        allowed = {p.lower() for p in phases}
        events = [e for e in events if str(e.get("phase", "sunset")).lower() in allowed]
    if limit is not None and limit > 0:
        events = events[:limit]
    return events


def extract_golden_hour_prediction(
    doc: dict[str, Any],
    phase: str = "sunset",
    min_el_deg: float = 1.2,
    max_el_deg: float = 16.0,
) -> tuple[float, float, float]:
    """
    Extracts predicted golden-hour `(expected_pct, p10_pct, p90_pct)` from `hourly_timeline`
    matched to the solar-elevation window (`1.5°..12°`) used by SURFRAD and NSRDB 2km ground truth.

    Works symmetrically for both `sunrise` (hours `0..11`) and `sunset` (hours `12..23`),
    weighting samples smoothly toward mid-golden-hour (`~5.5°` solar elevation) so hour-center
    offsets don't bias sunrise vs sunset comparisons.
    """
    c_ray_top = doc.get("corridor_raycast") or {}
    if c_ray_top.get("golden_hour_expected_transmittance_pct") is not None:
        return (
            round(float(c_ray_top["golden_hour_expected_transmittance_pct"]), 1),
            round(float(c_ray_top.get("golden_hour_p10_pct", c_ray_top["golden_hour_expected_transmittance_pct"])), 1),
            round(float(c_ray_top.get("golden_hour_p90_pct", c_ray_top["golden_hour_expected_transmittance_pct"])), 1),
        )

    ht = doc.get("hourly_timeline") or []
    is_sunrise = str(phase).strip().lower() == "sunrise"

    cands: list[tuple[float, float, float, float]] = []
    for row in ht:
        hr = int(row.get("hour", 12))
        if is_sunrise and hr >= 12:
            continue
        if (not is_sunrise) and hr < 12:
            continue
        el = float(row.get("sun_el_deg", -99.0))
        if min_el_deg <= el <= max_el_deg:
            # Weight centered around 5.5° solar elevation (middle of 1.5°..12° golden hour)
            w = math.exp(-0.5 * ((el - 5.5) / 4.2) ** 2)
            cands.append(
                (
                    w,
                    float(row.get("expected_transmittance_pct", 0.0)),
                    float(row.get("p10_pct", 0.0)),
                    float(row.get("p90_pct", 0.0)),
                )
            )

    if cands:
        w_sum = sum(c[0] for c in cands)
        if w_sum > 1e-9:
            exp_v = sum(c[0] * c[1] for c in cands) / w_sum
            p10_v = sum(c[0] * c[2] for c in cands) / w_sum
            p90_v = sum(c[0] * c[3] for c in cands) / w_sum
            return round(exp_v, 1), round(p10_v, 1), round(p90_v, 1)

    c_ray = doc.get("corridor_raycast") or {}
    return (
        round(float(c_ray.get("expected_transmittance_pct", 0.0)), 1),
        round(float(c_ray.get("transmittance_p10_pct", 0.0)), 1),
        round(float(c_ray.get("transmittance_p90_pct", 0.0)), 1),
    )


def evaluate_benchmark_event(
    ev: dict[str, Any],
    modes: tuple[str, ...] = DEFAULT_EVAL_MODES,
    refresh: bool = False,
    cache_tag: str = "v1",
) -> dict[str, Any]:
    """
    Evaluates a single historical sunrise or sunset event across the requested leak-free
    forecast modes and pairs it with independent Tier 1A (SURFRAD), Tier 1B (NSRDB 2km),
    and Tier 1C (RAOB/ASOS) ground-truth observations.
    """
    SUITE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    modes_sig = "_".join(modes)
    cache_path = SUITE_CACHE_DIR / f"{cache_tag}_{ev['id']}_{modes_sig}.json"
    if (not refresh) and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    lat = float(ev["lat"])
    lon = float(ev["lon"])
    elev_m = float(ev["elev_m"])
    d_str = str(ev["date"])
    phase = str(ev.get("phase", "sunset")).lower()
    req_d = dt.date.fromisoformat(d_str)

    tz, tz_name = local_tz_for_lon(lon)
    noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz)
    tz_off_hr = noon_loc.utcoffset().total_seconds() / 3600.0  # type: ignore[union-attr]
    sr, ss = _find_sunrise_sunset_golden_hours(lat, lon, req_d, tz_off_hr)
    event_mins, sun_az, sun_el = sr if phase == "sunrise" else ss
    target_utc = dt.datetime(
        req_d.year, req_d.month, req_d.day, event_mins // 60, event_mins % 60, tzinfo=tz
    ).astimezone(dt.timezone.utc)

    # --- Tier 1A: NOAA SURFRAD 1-minute ground pyrheliometer ---
    surfrad_ref: dict[str, Any] | None = None
    if ev.get("surfrad_station"):
        try:
            s_doc = fetch_surfrad_golden_hour_truth(str(ev["surfrad_station"]), d_str, phase=phase)
            if s_doc and s_doc.get("available"):
                surfrad_ref = s_doc
        except Exception:
            surfrad_ref = None

    # --- Tier 1B: NREL NSRDB 2km 5-minute GOES CONUS DNI (only needed when Tier 1A SURFRAD is absent) ---
    nsrdb_ref: dict[str, Any] | None = None
    if surfrad_ref is None:
        try:
            n_doc = fetch_nsrdb_2km_golden_hour_dni(lat, lon, elev_m, d_str, phase=phase)
            if n_doc and n_doc.get("available"):
                nsrdb_ref = n_doc
        except Exception:
            nsrdb_ref = None

    # Primary independent transmittance truth (Tier 1A SURFRAD > Tier 1B NSRDB 2km)
    primary_truth_pct: float | None = None
    primary_truth_tier: str | None = None
    primary_sun_hit: bool | None = None
    if surfrad_ref is not None:
        primary_truth_pct = float(surfrad_ref["golden_hour_rel_transmittance_pct"])
        primary_truth_tier = "tier_1a_surfrad"
        primary_sun_hit = bool(surfrad_ref["golden_hour_sun_hit"])
    elif nsrdb_ref is not None:
        primary_truth_pct = float(nsrdb_ref["golden_hour_rel_transmittance_pct"])
        primary_truth_tier = "tier_1b_nsrdb_2km"
        primary_sun_hit = bool(primary_truth_pct >= 35.0)

    # --- Tier 1C: NWS RAOB balloon & ASOS surface station ---
    raob_ref: dict[str, Any] | None = None
    try:
        raob_obs = fetch_raob_sounding(lat, lon, target_utc=target_utc, causal_only=False)
        if raob_obs is not None and float(raob_obs.get("distance_km", 999.0)) <= 185.0:
            r_inv_base = raob_obs.get("inversion_base_asl_m")
            r_inv_top = raob_obs.get("inversion_top_asl_m")
            r_sat_top = raob_obs.get("saturated_stratus_top_asl_m")
            r_strength_k = float(raob_obs.get("inversion_strength_k") or 0.0)
            lid_m: float | None = None
            if r_sat_top is not None:
                lid_m = float(r_sat_top)
            elif r_inv_base is not None and r_strength_k >= 0.6:
                rt = float(r_inv_top) if r_inv_top is not None else (float(r_inv_base) + 150.0)
                lid_m = float(r_inv_base) + 0.18 * max(0.0, rt - float(r_inv_base))
            raob_ref = {
                "station_id": raob_obs["station_id"],
                "distance_km": round(float(raob_obs["distance_km"]), 1),
                "valid_utc": raob_obs["valid_utc"],
                "lid_asl_m": round(lid_m, 1) if lid_m is not None else None,
                "lid_asl_ft": int(round(lid_m * 3.28084)) if lid_m is not None else None,
                "inversion_strength_f": round(r_strength_k * 1.8, 1),
                "summit_truly_above_lid": bool(elev_m > lid_m + 20.0) if lid_m is not None else None,
            }
    except Exception:
        raob_ref = None

    asos_ref: dict[str, Any] | None = None
    try:
        asos_obs = fetch_nearest_station_observation(lat, lon, target_utc=target_utc)
        if asos_obs is not None and float(asos_obs.get("distance_km", 999.0)) <= 65.0:
            ceil_m = asos_obs.get("fog_base_asl_m") or asos_obs.get("lowest_ceiling_asl_m")
            asos_ref = {
                "station_id": asos_obs["station_id"],
                "distance_km": round(float(asos_obs["distance_km"]), 1),
                "low_cloud_pct": round(float(asos_obs.get("low_cloud_fraction", 0.0)) * 100.0, 1),
                "ceilometer_base_ft": int(round(float(ceil_m) * 3.28084))
                if (ceil_m is not None and float(ceil_m) <= 2200.0)
                else None,
            }
    except Exception:
        asos_ref = None

    # --- Execute Solver across requested forecast modes ---
    modes_out: dict[str, Any] = {}
    for m_key in modes:
        raw_bytes = solve_weather_and_corridor(
            lat,
            lon,
            d_str,
            phase=phase,
            sun_az_deg=sun_az,
            sun_el_deg=sun_el,
            target_elev_m=elev_m,
            model_engine="ensemble",
            replay_mode=m_key,
            include_lut=False,
        )
        doc = json.loads(raw_bytes)
        gh_exp, gh_p10, gh_p90 = extract_golden_hour_prediction(doc, phase=phase)
        c_ray = doc.get("corridor_raycast") or {}
        inv_m = doc.get("inversion_model") or {}
        cal_p = doc.get("calibrated_probabilities") or {}
        prov = doc.get("data_provenance") or {}
        nc_s = prov.get("nowcast") or {}
        modes_out[m_key] = {
            "golden_exp_pct": gh_exp,
            "golden_p10_pct": gh_p10,
            "golden_p90_pct": gh_p90,
            "sunset_exp_pct": float(c_ray.get("expected_transmittance_pct", 0.0)),
            "sunset_p10_pct": float(c_ray.get("transmittance_p10_pct", 0.0)),
            "sunset_p90_pct": float(c_ray.get("transmittance_p90_pct", 0.0)),
            "direct_beam_prob_pct": float(c_ray.get("direct_beam_probability_pct", 0.0)),
            "calibrated_p10_pct": cal_p.get("transmittance_p10_pct"),
            "calibrated_p50_pct": cal_p.get("transmittance_p50_pct"),
            "calibrated_p90_pct": cal_p.get("transmittance_p90_pct"),
            "alpenglow_score": int(doc.get("alpenglow_score", 0)),
            "canvas_score": int(doc.get("canvas_score", 0)),
            "cloud_low_pct": int(doc.get("cloud_low_pct", 0)),
            "cloud_mid_pct": int(doc.get("cloud_mid_pct", 0)),
            "cloud_high_pct": int(doc.get("cloud_high_pct", 0)),
            "fog_top_ft": int(doc.get("fog_top_ft", 0)),
            "summit_clearance_ft": int(inv_m.get("summit_clearance_ft", 0)),
            "blocker_label": str(c_ray.get("dominant_blocker_label") or "clear_corridor"),
            "blocker_distance_km": float(c_ray.get("dominant_blocker_distance_km", 0.0)),
            "terrain_horizon_deg": float(c_ray.get("terrain_horizon_deg", 0.0)),
            "goes_active": bool(nc_s.get("active")),
            "hrrr_s3_active": bool((prov.get("hrrr_s3") or {}).get("active")),
        }

    result = {
        "id": ev["id"],
        "site": ev["site"],
        "suite": ev.get("suite", "custom"),
        "region": ev["region"],
        "subregion": ev.get("subregion", ev["region"]),
        "season": ev.get("season", _season_for_date(req_d)),
        "phase": phase,
        "date": d_str,
        "lat": lat,
        "lon": lon,
        "elev_m": elev_m,
        "timezone": tz_name,
        "sun_az_deg": sun_az,
        "sun_el_deg": sun_el,
        "regime": ev.get("regime", "general"),
        "primary_truth_tier": primary_truth_tier,
        "dni_golden_truth_pct": primary_truth_pct,
        "golden_hour_sun_hit": primary_sun_hit,
        "surfrad_ref": surfrad_ref,
        "nsrdb_2km_ref": nsrdb_ref,
        "nsrdb_2km_golden_truth_pct": float(nsrdb_ref["golden_hour_rel_transmittance_pct"])
        if nsrdb_ref is not None
        else None,
        "raob_ref": raob_ref,
        "asos_ref": asos_ref,
        "modes": modes_out,
    }

    try:
        cache_path.write_text(json.dumps(result), encoding="utf-8")
    except Exception:
        pass
    return result


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def score_predictions(
    rows: list[dict[str, Any]],
    pred_key: str = "golden_exp_pct",
    p10_key: str = "golden_p10_pct",
    p90_key: str = "golden_p90_pct",
    hit_threshold_pct: float = 35.0,
) -> dict[str, Any]:
    """
    Computes continuous proper scoring rules (MAE, RMSE, Bias, Pearson r, Spearman rho,
    CRPS, Pinball Loss, 80% PI Coverage & Width) plus binary golden-hour hit metrics
    (Brier, BSS vs climatology, ECE, POD, FAR, CSI).
    """
    if not rows:
        return {"n": 0}

    y = np.array([float(r["truth_pct"]) for r in rows], dtype=np.float64)
    p50 = np.array([float(r[pred_key]) for r in rows], dtype=np.float64)
    p10 = np.array([float(r[p10_key]) for r in rows], dtype=np.float64)
    p90 = np.array([float(r[p90_key]) for r in rows], dtype=np.float64)
    err = p50 - y

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    pearson_r = (
        float(np.corrcoef(p50, y)[0, 1])
        if (y.size >= 3 and np.std(p50) > 1e-9 and np.std(y) > 1e-9)
        else 0.0
    )
    spearman = _spearman_rho(p50, y)

    crps_val = crps_from_quantiles(y, p10, p50, p90)
    pin10 = pinball_loss(y, p10, 0.10)
    pin50 = pinball_loss(y, p50, 0.50)
    pin90 = pinball_loss(y, p90, 0.90)
    cov = float(np.mean((y >= p10 - 1.0) & (y <= p90 + 1.0))) * 100.0
    width = float(np.mean(np.maximum(0.0, p90 - p10)))

    # Binary "Did golden-hour sunlight break through?" scoring
    y_hit = np.array(
        [
            1.0
            if (r.get("truth_hit") is True or (r.get("truth_hit") is None and float(r["truth_pct"]) >= hit_threshold_pct))
            else 0.0
            for r in rows
        ],
        dtype=np.float64,
    )
    # Derive smooth hit probability from predicted P10..P50..P90 relative to threshold
    hit_probs: list[float] = []
    for q10, q50, q90 in zip(p10, p50, p90):
        spread = max(12.0, (q90 - q10) / 2.56)
        z = (q50 - hit_threshold_pct) / spread
        prob = 1.0 / (1.0 + math.exp(-max(-18.0, min(18.0, z * 1.45))))
        hit_probs.append(prob)
    p_hit = np.array(hit_probs, dtype=np.float64)
    brier = brier_score(y_hit, p_hit)
    bss = brier_skill_score(y_hit, p_hit)
    ece = expected_calibration_error(y_hit, p_hit, n_bins=5)

    pred_yes = p50 >= hit_threshold_pct
    obs_yes = y_hit >= 0.5
    tp = int(np.sum(pred_yes & obs_yes))
    fp = int(np.sum(pred_yes & (~obs_yes)))
    fn = int(np.sum((~pred_yes) & obs_yes))
    tn = int(np.sum((~pred_yes) & (~obs_yes)))
    pod = tp / max(1, tp + fn)
    far = fp / max(1, tp + fp)
    csi = tp / max(1, tp + fp + fn)

    return {
        "n": int(y.size),
        "mae_pct": round(mae, 2),
        "rmse_pct": round(rmse, 2),
        "bias_pct": round(bias, 2),
        "pearson_r": round(pearson_r, 3),
        "spearman_rho": round(spearman, 3),
        "crps_pct": round(crps_val, 2),
        "pinball_p10": round(pin10, 2),
        "pinball_p50": round(pin50, 2),
        "pinball_p90": round(pin90, 2),
        "pi80_coverage_pct": round(cov, 1),
        "pi80_mean_width_pct": round(width, 1),
        "hit_base_rate": round(float(np.mean(y_hit)), 3),
        "brier_score": round(brier, 4),
        "brier_skill_score_vs_climo": round(bss, 3),
        "ece": round(ece, 4),
        "pod_hit_rate": round(pod, 3),
        "far_false_alarm_ratio": round(far, 3),
        "csi_threat_score": round(csi, 3),
        "contingency": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _flatten_scored_rows(
    event_results: list[dict[str, Any]],
    modes: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ev in event_results:
        truth = ev.get("dni_golden_truth_pct")
        if truth is None:
            continue
        for m_key in modes:
            m_rec = (ev.get("modes") or {}).get(m_key)
            if not m_rec:
                continue
            rows.append(
                {
                    "event_id": ev["id"],
                    "site": ev["site"],
                    "suite": ev.get("suite", "custom"),
                    "region": ev["region"],
                    "subregion": ev["subregion"],
                    "season": ev["season"],
                    "phase": ev.get("phase", "sunset"),
                    "date": ev["date"],
                    "elev_m": float(ev.get("elev_m", 0.0)),
                    "mode": m_key,
                    "truth_tier": ev.get("primary_truth_tier", "unknown"),
                    "truth_pct": float(truth),
                    "truth_hit": ev.get("golden_hour_sun_hit"),
                    "golden_exp_pct": float(m_rec["golden_exp_pct"]),
                    "golden_p10_pct": float(m_rec["golden_p10_pct"]),
                    "golden_p90_pct": float(m_rec["golden_p90_pct"]),
                    "sunset_exp_pct": float(m_rec["sunset_exp_pct"]),
                    "cloud_low_pct": int(m_rec["cloud_low_pct"]),
                    "cloud_mid_pct": int(m_rec["cloud_mid_pct"]),
                    "cloud_high_pct": int(m_rec["cloud_high_pct"]),
                    "blocker_label": m_rec.get("blocker_label", "clear_corridor"),
                    "blocker_distance_km": float(m_rec.get("blocker_distance_km", 0.0)),
                    "terrain_horizon_deg": float(m_rec.get("terrain_horizon_deg", 0.0)),
                    "goes_active": bool(m_rec.get("goes_active")),
                    "hrrr_s3_active": bool(m_rec.get("hrrr_s3_active")),
                }
            )
    return rows


def diagnose_failures(
    event_results: list[dict[str, Any]],
    modes: tuple[str, ...] = DEFAULT_EVAL_MODES,
    miss_threshold_pp: float = 25.0,
) -> dict[str, Any]:
    """
    Classifies large forecast misses (`|err| >= 25 pp`) into actionable physical failure modes
    and ranks the worst individual misses for empirical physics debugging.
    """
    rows = _flatten_scored_rows(event_results, modes)
    if not rows:
        return {"n_total_rows": 0, "n_large_misses": 0, "categories": {}, "top_misses": []}

    categories: dict[str, list[dict[str, Any]]] = {
        "FALSE_NEGATIVE_OVER_ATTENUATION": [],
        "FALSE_POSITIVE_MISSED_CLOUD_DECK": [],
        "BROKEN_SKY_TIMING": [],
        "CLEAR_SKY_UNDERPREDICTION": [],
    }
    all_misses: list[dict[str, Any]] = []

    for r in rows:
        err = r["golden_exp_pct"] - r["truth_pct"]
        if abs(err) < miss_threshold_pp:
            continue
        if err <= -miss_threshold_pp:
            if r["truth_pct"] >= 75.0 and r["cloud_low_pct"] < 30 and r["cloud_mid_pct"] < 30:
                cat = "CLEAR_SKY_UNDERPREDICTION"
            elif r["truth_pct"] >= 55.0:
                cat = "FALSE_NEGATIVE_OVER_ATTENUATION"
            else:
                cat = "BROKEN_SKY_TIMING"
        else:
            if r["truth_pct"] <= 30.0:
                cat = "FALSE_POSITIVE_MISSED_CLOUD_DECK"
            else:
                cat = "BROKEN_SKY_TIMING"

        rec = {
            "event_id": r["event_id"],
            "site": r["site"],
            "date": r["date"],
            "phase": r["phase"],
            "region": r["region"],
            "season": r["season"],
            "mode": r["mode"],
            "truth_tier": r["truth_tier"],
            "pred_pct": r["golden_exp_pct"],
            "truth_pct": r["truth_pct"],
            "error_pp": round(err, 1),
            "abs_error_pp": round(abs(err), 1),
            "cloud_lmh": [r["cloud_low_pct"], r["cloud_mid_pct"], r["cloud_high_pct"]],
            "blocker_label": r["blocker_label"],
            "blocker_distance_km": r["blocker_distance_km"],
            "category": cat,
        }
        categories[cat].append(rec)
        all_misses.append(rec)

    all_misses.sort(key=lambda x: x["abs_error_pp"], reverse=True)
    cat_summary = {
        k: {
            "count": len(v),
            "share_of_misses_pct": round(100.0 * len(v) / max(1, len(all_misses)), 1),
            "mean_error_pp": round(float(np.mean([x["error_pp"] for x in v])), 1) if v else 0.0,
        }
        for k, v in categories.items()
    }

    # Also summarize mean bias by dominant blocker type across ALL rows
    by_blocker: dict[str, list[float]] = {}
    for r in rows:
        by_blocker.setdefault(str(r["blocker_label"]), []).append(r["golden_exp_pct"] - r["truth_pct"])
    blocker_bias = {
        k: {
            "n": len(errs),
            "mae_pp": round(float(np.mean(np.abs(errs))), 2),
            "bias_pp": round(float(np.mean(errs)), 2),
        }
        for k, errs in sorted(by_blocker.items(), key=lambda kv: -len(kv[1]))
    }

    return {
        "n_total_rows": len(rows),
        "miss_threshold_pp": miss_threshold_pp,
        "n_large_misses": len(all_misses),
        "large_miss_rate_pct": round(100.0 * len(all_misses) / max(1, len(rows)), 1),
        "categories": cat_summary,
        "error_by_blocker": blocker_bias,
        "top_misses": all_misses[:15],
    }


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT_DIR),
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _preregister_hrrr_cycle_boxes(
    events: list[dict[str, Any]],
    modes: tuple[str, ...],
) -> int:
    """
    Pre-registers all `(i0, i1, j0, j1)` HRRR tile boxes per shared `(YYYYMMDD_HHz_fFF)` cycle
    so any site downloading a CONUS GRIB2 cycle automatically slices & caches sibling sites' tiles.
    """
    from .hrrr_s3 import (
        candidate_hrrr_cycles,
        is_in_hrrr_conus,
        register_cycle_crop_boxes,
        tile_box_for_corridor,
    )
    from .nwp_corridor import build_corridor_coords_meta

    cycle_boxes: dict[str, set[tuple[int, int, int, int]]] = {}
    for ev in events:
        lat = float(ev["lat"])
        lon = float(ev["lon"])
        if not is_in_hrrr_conus(lat, lon):
            continue
        req_d = dt.date.fromisoformat(str(ev["date"]))
        phase = str(ev.get("phase", "sunset")).lower()
        tz, _ = local_tz_for_lon(lon)
        noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz)
        tz_off_hr = (noon_loc.utcoffset() or dt.timedelta(0)).total_seconds() / 3600.0
        sr, ss = _find_sunrise_sunset_golden_hours(lat, lon, req_d, tz_off_hr)
        ev_mins, sun_az, _ = sr if phase == "sunrise" else ss
        conj_az = ss[1] if phase == "sunrise" else sr[1]
        t_utc = dt.datetime(
            req_d.year, req_d.month, req_d.day, ev_mins // 60, ev_mins % 60, tzinfo=tz
        ).astimezone(dt.timezone.utc)
        coords_meta = build_corridor_coords_meta(lat, lon, sun_az, conj_az)
        box = tile_box_for_corridor(lat, lon, coords_meta)
        for m in modes:
            if m == "nowcast_1h":
                cands = candidate_hrrr_cycles(t_utc, t_utc - dt.timedelta(hours=1), replay_mode=m, hindcast_lead_days=0)
            elif m == "nowcast_3h":
                cands = candidate_hrrr_cycles(t_utc, t_utc - dt.timedelta(hours=3), replay_mode=m, hindcast_lead_days=0)
            elif m == "hindcast_24h":
                cands = candidate_hrrr_cycles(t_utc, t_utc - dt.timedelta(hours=24), replay_mode=m, hindcast_lead_days=1)
            else:
                cands = []
            if cands:
                init_dt, fhr = cands[0]
                ck = f"{init_dt.strftime('%Y%m%d')}_{init_dt.strftime('%H')}z_f{fhr:02d}"
                cycle_boxes.setdefault(ck, set()).add(box)

    shared_only = {ck: sorted(bxs) for ck, bxs in cycle_boxes.items() if len(bxs) > 1}
    if shared_only:
        register_cycle_crop_boxes(shared_only)
    return len(shared_only)


def _prune_cache_dir_to_budget(cache_dir: Path, max_bytes: int, patterns: tuple[str, ...]) -> int:
    """Deletes oldest raw cache files (by mtime) when a directory exceeds `max_bytes`."""
    if not cache_dir.exists():
        return 0
    files: list[tuple[float, int, Path]] = []
    total = 0
    for pat in patterns:
        for p in cache_dir.rglob(pat):
            if p.is_file():
                try:
                    st = p.stat()
                    files.append((st.st_mtime, st.st_size, p))
                    total += st.st_size
                except OSError:
                    continue
    if total <= max_bytes:
        return 0
    files.sort(key=lambda x: x[0])
    removed = 0
    for _, sz, p in files:
        if total <= max_bytes:
            break
        try:
            p.unlink()
            total -= sz
            removed += 1
            if p.suffix == ".npz":
                meta_p = p.with_suffix(".json")
                if meta_p.exists():
                    meta_p.unlink()
        except OSError:
            pass
    return removed


def run_benchmark_suite(
    suite: str = "all",
    modes: tuple[str, ...] = DEFAULT_EVAL_MODES,
    limit: int | None = None,
    phases: tuple[str, ...] | None = None,
    dates_per_season: int = 1,
    max_workers: int = 4,
    refresh: bool = False,
    cache_tag: str = "v1",
    run_name: str | None = None,
    save_calibration: bool = False,
    write_html: bool = True,
    max_uncached: int | None = None,
    cached_only: bool = False,
) -> dict[str, Any]:
    """
    Runs the multi-regime independent verification suite, evaluates strict out-of-sample
    calibration CV, generates automated failure-mode diagnostics, persists a timestamped
    experiment snapshot, and optionally renders the interactive HTML dashboard.
    """
    from .hrrr_s3 import HRRR_CACHE_DIR, clear_cycle_crop_boxes

    t0 = time.perf_counter()
    events = build_stratified_events(
        suite=suite,
        limit=limit,
        phases=phases,
        dates_per_season=dates_per_season,
    )
    # Date-first ordering maximizes shared CONUS HRRR cycle & GOES-16/18 scan hits across sites
    # while persistent Rust worker's i16 SRTM LRU cache holds all sites simultaneously in RAM.
    events_exec = sorted(
        events,
        key=lambda e: (str(e.get("date", "")), str(e.get("phase", "")), str(e.get("site", ""))),
    )

    modes_sig = "_".join(modes)
    cached_events: list[dict[str, Any]] = []
    uncached_events: list[dict[str, Any]] = []
    for ev in events_exec:
        cp = SUITE_CACHE_DIR / f"{cache_tag}_{ev['id']}_{modes_sig}.json"
        if (not refresh) and cp.exists():
            cached_events.append(ev)
        else:
            uncached_events.append(ev)

    remaining_after_batch = 0
    if cached_only:
        run_list = cached_events
        uncached_events = []
    elif max_uncached is not None and max_uncached >= 0 and len(uncached_events) > max_uncached:
        remaining_after_batch = len(uncached_events) - max_uncached
        batch_uncached = uncached_events[:max_uncached]
        print(
            f"[BATCH MODE] {len(cached_events)} cached | running {len(batch_uncached)} uncached | "
            f"{remaining_after_batch} remaining after this batch",
            flush=True,
        )
        run_list = batch_uncached
    else:
        run_list = cached_events + uncached_events

    # Register sibling HRRR boxes across ALL uncached events so every downloaded CONUS cycle fans out!
    n_shared = _preregister_hrrr_cycle_boxes(uncached_events, modes)
    if n_shared > 0:
        print(f"[HRRR FAN-OUT] Pre-registered {n_shared} shared CONUS HRRR cycles across sites", flush=True)

    event_results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futs = {
                pool.submit(
                    evaluate_benchmark_event,
                    ev,
                    modes=modes,
                    refresh=refresh,
                    cache_tag=cache_tag,
                ): ev
                for ev in run_list
            }
            done_n = 0
            for fut in as_completed(futs):
                ev = futs[fut]
                done_n += 1
                try:
                    res = fut.result()
                    event_results.append(res)
                    t_val = res.get("dni_golden_truth_pct")
                    m0 = (res.get("modes") or {}).get(modes[0], {})
                    p_val = m0.get("golden_exp_pct")
                    print(
                        f"  [{done_n:3d}/{len(run_list)}] {ev['id']:<42} "
                        f"tier={res.get('primary_truth_tier') or 'none':<16} "
                        f"truth={str(t_val):>5}% pred({modes[0]})={str(p_val):>5}%",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"  [{done_n:3d}/{len(run_list)}] {ev['id']} FAILED: {exc}", flush=True)
    finally:
        clear_cycle_crop_boxes()
        # Keep raw GOES NetCDF and HRRR tile caches bounded on disk during large multi-year runs
        goes_cache_dir = ROOT_DIR / "data" / "weather_cache" / "goes"
        _prune_cache_dir_to_budget(goes_cache_dir, int(1.8 * 1024**3), ("*.nc",))
        _prune_cache_dir_to_budget(HRRR_CACHE_DIR, int(2.2 * 1024**3), ("*.npz",))

    if remaining_after_batch > 0:
        elapsed_s = round(time.perf_counter() - t0, 2)
        return {
            "partial_batch": True,
            "suite": suite,
            "run_id": f"batch_{suite}",
            "git_sha": _git_short_sha(),
            "elapsed_seconds": elapsed_s,
            "n_events_requested": len(events),
            "n_events_cached_total": len(cached_events) + len(event_results),
            "n_events_remaining": remaining_after_batch,
            "n_events_scored": len(event_results),
            "n_scored_mode_rows": 0,
        }

    # Sort deterministically by date, id
    event_results.sort(key=lambda r: (r["date"], r["id"]))
    rows = _flatten_scored_rows(event_results, modes)

    overall_raw = score_predictions(rows)
    by_mode = {
        m: score_predictions([r for r in rows if r["mode"] == m])
        for m in modes
    }
    from .calibration import is_flat_nsrdb_terrain_shadow_artifact

    by_tier = {
        t: score_predictions([r for r in rows if r["truth_tier"] == t])
        for t in sorted({r["truth_tier"] for r in rows})
    }
    by_terrain_shadow_audit = {
        "open_horizon_clean": score_predictions(
            [
                r
                for r in rows
                if not is_flat_nsrdb_terrain_shadow_artifact(
                    r["truth_tier"], r["terrain_horizon_deg"], r["blocker_label"]
                )
            ]
        ),
        "flat_nsrdb_3d_shadow": score_predictions(
            [
                r
                for r in rows
                if is_flat_nsrdb_terrain_shadow_artifact(
                    r["truth_tier"], r["terrain_horizon_deg"], r["blocker_label"]
                )
            ]
        ),
    }
    by_year = {
        yr: score_predictions([r for r in rows if str(r["date"])[:4] == yr])
        for yr in sorted({str(r["date"])[:4] for r in rows})
    }
    by_phase = {
        p: score_predictions([r for r in rows if r["phase"] == p])
        for p in sorted({r["phase"] for r in rows})
    }
    by_season = {
        s: score_predictions([r for r in rows if r["season"] == s])
        for s in ("winter", "spring", "summer", "autumn")
        if any(r["season"] == s for r in rows)
    }
    by_region = {
        reg: score_predictions([r for r in rows if r["region"] == reg])
        for reg in sorted({r["region"] for r in rows})
    }

    # Cross-sensor audit where BOTH SURFRAD 1-min and NSRDB 2km exist on the same event
    cross_sensor_pairs = [
        (
            float(ev["surfrad_ref"]["golden_hour_rel_transmittance_pct"]),
            float(ev["nsrdb_2km_ref"]["golden_hour_rel_transmittance_pct"]),
        )
        for ev in event_results
        if ev.get("surfrad_ref") and ev.get("nsrdb_2km_ref")
    ]
    if cross_sensor_pairs:
        s_arr = np.array([p[0] for p in cross_sensor_pairs])
        n_arr = np.array([p[1] for p in cross_sensor_pairs])
        cross_sensor_audit = {
            "n_colocated_events": len(cross_sensor_pairs),
            "surfrad_vs_nsrdb_mae_pp": round(float(np.mean(np.abs(n_arr - s_arr))), 2),
            "surfrad_vs_nsrdb_bias_pp": round(float(np.mean(n_arr - s_arr)), 2),
            "surfrad_vs_nsrdb_pearson_r": round(float(np.corrcoef(s_arr, n_arr)[0, 1]), 3)
            if len(cross_sensor_pairs) >= 3
            else 0.0,
        }
    else:
        cross_sensor_audit = {"n_colocated_events": 0}

    # Strict Out-of-Sample Calibration CV on independent Tier 1A + open-horizon Tier 1B truth
    cal_eval = evaluate_calibration_out_of_sample(event_results, eval_modes=modes)
    if save_calibration and cal_eval.get("fitted_models", {}).get("transmittance"):
        save_calibration_bundle(cal_eval, DEFAULT_CALIBRATION_MODEL_PATH)

    failures = diagnose_failures(event_results, modes=modes)

    elapsed_s = round(time.perf_counter() - t0, 2)
    ts_tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts_tag}_{run_name or suite}"

    report: dict[str, Any] = {
        "schema_version": "benchmark_suite_v1",
        "run_id": run_id,
        "git_sha": _git_short_sha(),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": suite,
        "modes": list(modes),
        "cache_tag": cache_tag,
        "elapsed_seconds": elapsed_s,
        "n_events_requested": len(events),
        "n_events_scored": sum(1 for e in event_results if e.get("dni_golden_truth_pct") is not None),
        "n_scored_mode_rows": len(rows),
        "overall_independent_raw": overall_raw,
        "by_mode": by_mode,
        "by_truth_tier": by_tier,
        "by_terrain_shadow_audit": by_terrain_shadow_audit,
        "by_year": by_year,
        "by_phase": by_phase,
        "by_season": by_season,
        "by_region": by_region,
        "cross_sensor_surfrad_vs_nsrdb": cross_sensor_audit,
        "out_of_sample_calibration": cal_eval,
        "failure_diagnostics": failures,
        "events_detail": event_results,
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_path = EXPERIMENTS_DIR / f"{run_id}.json"
    exp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["experiment_path"] = str(exp_path)

    if write_html:
        html_str = generate_html_verification_report(report)
        VERIFICATION_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
        VERIFICATION_HTML_PATH.write_text(html_str, encoding="utf-8")
        report["html_report_path"] = str(VERIFICATION_HTML_PATH)

    return report


def compare_experiments(
    path_a: str | Path,
    path_b: str | Path,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compares two saved experiment snapshots (`A = baseline`, `B = candidate`) on their exact
    intersecting `(event_id, mode)` pairs using paired bootstrap resampling (95% CIs) for
    `ΔMAE`, `ΔRMSE`, `ΔCRPS`, and `Δ|Bias|`. Negative deltas mean `B` improved over `A`.
    """
    doc_a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    doc_b = json.loads(Path(path_b).read_text(encoding="utf-8"))

    modes_a = tuple(doc_a.get("modes") or DEFAULT_EVAL_MODES)
    modes_b = tuple(doc_b.get("modes") or DEFAULT_EVAL_MODES)
    rows_a = {(r["event_id"], r["mode"]): r for r in _flatten_scored_rows(doc_a.get("events_detail", []), modes_a)}
    rows_b = {(r["event_id"], r["mode"]): r for r in _flatten_scored_rows(doc_b.get("events_detail", []), modes_b)}

    common_keys = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    if not common_keys:
        return {"n_paired_rows": 0, "error": "No overlapping (event_id, mode) rows between experiments."}

    list_a = [rows_a[k] for k in common_keys]
    list_b = [rows_b[k] for k in common_keys]
    sc_a = score_predictions(list_a)
    sc_b = score_predictions(list_b)

    y = np.array([r["truth_pct"] for r in list_a], dtype=np.float64)
    ea = np.array([r["golden_exp_pct"] - r["truth_pct"] for r in list_a], dtype=np.float64)
    eb = np.array([r["golden_exp_pct"] - r["truth_pct"] for r in list_b], dtype=np.float64)
    ca = np.array(
        [crps_from_quantiles([r["truth_pct"]], [r["golden_p10_pct"]], [r["golden_exp_pct"]], [r["golden_p90_pct"]]) for r in list_a],
        dtype=np.float64,
    )
    cb = np.array(
        [crps_from_quantiles([r["truth_pct"]], [r["golden_p10_pct"]], [r["golden_exp_pct"]], [r["golden_p90_pct"]]) for r in list_b],
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    n = y.size
    d_mae = np.empty(n_boot, dtype=np.float64)
    d_rmse = np.empty(n_boot, dtype=np.float64)
    d_crps = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d_mae[b] = float(np.mean(np.abs(eb[idx])) - np.mean(np.abs(ea[idx])))
        d_rmse[b] = float(np.sqrt(np.mean(eb[idx] ** 2)) - np.sqrt(np.mean(ea[idx] ** 2)))
        d_crps[b] = float(np.mean(cb[idx]) - np.mean(ca[idx]))

    def _ci_block(delta_obs: float, samples: np.ndarray) -> dict[str, Any]:
        lo, hi = float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
        return {
            "delta_b_minus_a": round(delta_obs, 3),
            "ci95_low": round(lo, 3),
            "ci95_high": round(hi, 3),
            "statistically_significant_improvement": bool(hi < 0.0),
            "statistically_significant_regression": bool(lo > 0.0),
        }

    return {
        "baseline_run_id": doc_a.get("run_id", str(path_a)),
        "candidate_run_id": doc_b.get("run_id", str(path_b)),
        "n_paired_rows": len(common_keys),
        "baseline_metrics": sc_a,
        "candidate_metrics": sc_b,
        "paired_bootstrap_deltas": {
            "mae_pct": _ci_block(sc_b["mae_pct"] - sc_a["mae_pct"], d_mae),
            "rmse_pct": _ci_block(sc_b["rmse_pct"] - sc_a["rmse_pct"], d_rmse),
            "crps_pct": _ci_block(sc_b["crps_pct"] - sc_a["crps_pct"], d_crps),
        },
    }


def _svg_scatter(rows: list[dict[str, Any]], width: int = 440, height: int = 340) -> str:
    pad_l, pad_r, pad_t, pad_b = 48, 18, 20, 42
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    def sx(v: float) -> float:
        return pad_l + max(0.0, min(100.0, v)) / 100.0 * pw

    def sy(v: float) -> float:
        return pad_t + (1.0 - max(0.0, min(100.0, v)) / 100.0) * ph

    tier_colors = {
        "tier_1a_surfrad": "#f59e0b",
        "tier_1b_nsrdb_2km": "#38bdf8",
    }
    grid_lines = []
    for tick in (0, 25, 50, 75, 100):
        x = sx(tick)
        y = sy(tick)
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#273142" stroke-width="1"/>'
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height - pad_b}" stroke="#273142" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#94a3b8" font-size="11">{tick}%</text>'
            f'<text x="{x:.1f}" y="{height - pad_b + 16}" text-anchor="middle" fill="#94a3b8" font-size="11">{tick}%</text>'
        )

    # +/- 20 pp band and 1:1 diagonal
    diag = (
        f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(100):.1f}" y2="{sy(100):.1f}" '
        f'stroke="#64748b" stroke-dasharray="5,4" stroke-width="1.5"/>'
    )
    dots = []
    for r in rows:
        col = tier_colors.get(r["truth_tier"], "#a78bfa")
        cx = sx(r["truth_pct"])
        cy = sy(r["golden_exp_pct"])
        tip = html.escape(
            f"{r['site']} {r['date']} ({r['phase']}, {r['mode']}): "
            f"Truth {r['truth_pct']:.0f}% vs Pred {r['golden_exp_pct']:.0f}%"
        )
        dots.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.6" fill="{col}" fill-opacity="0.72" stroke="#0f172a" stroke-width="0.6">'
            f"<title>{tip}</title></circle>"
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block">'
        + "".join(grid_lines)
        + diag
        + "".join(dots)
        + f'<text x="{pad_l + pw / 2:.1f}" y="{height - 6}" text-anchor="middle" fill="#cbd5e1" font-size="12">Independent Observed Transmittance (%)</text>'
        + f'<text x="14" y="{pad_t + ph / 2:.1f}" text-anchor="middle" fill="#cbd5e1" font-size="12" transform="rotate(-90 14 {pad_t + ph / 2:.1f})">Predicted Transmittance (%)</text>'
        + "</svg>"
    )


def _svg_lead_curve(by_mode: dict[str, dict[str, Any]], width: int = 440, height: int = 340) -> str:
    order = [m for m in ("nowcast_1h", "nowcast_3h", "hindcast_24h", "hindcast_72h") if m in by_mode and by_mode[m].get("n", 0) > 0]
    if not order:
        return "<p style='color:#94a3b8'>No lead-time data.</p>"

    pad_l, pad_r, pad_t, pad_b = 48, 20, 24, 50
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b
    max_y = max(35.0, max(float(by_mode[m]["mae_pct"]) for m in order) + 8.0)

    def sy(v: float) -> float:
        return pad_t + (1.0 - max(0.0, min(max_y, v)) / max_y) * ph

    bars = []
    labels = {"nowcast_1h": "+1h", "nowcast_3h": "+3h", "hindcast_24h": "+24h", "hindcast_72h": "+72h"}
    step = pw / max(1, len(order))
    for i, m in enumerate(order):
        st = by_mode[m]
        x_center = pad_l + (i + 0.5) * step
        mae_h = (height - pad_b) - sy(st["mae_pct"])
        crps_h = (height - pad_b) - sy(st["crps_pct"])
        bars.append(
            f'<rect x="{x_center - 24:.1f}" y="{sy(st["mae_pct"]):.1f}" width="22" height="{mae_h:.1f}" rx="3" fill="#38bdf8"/>'
            f'<text x="{x_center - 13:.1f}" y="{sy(st["mae_pct"]) - 5:.1f}" text-anchor="middle" fill="#e2e8f0" font-size="11">{st["mae_pct"]:.1f}</text>'
            f'<rect x="{x_center + 2:.1f}" y="{sy(st["crps_pct"]):.1f}" width="22" height="{crps_h:.1f}" rx="3" fill="#f59e0b"/>'
            f'<text x="{x_center + 13:.1f}" y="{sy(st["crps_pct"]) - 5:.1f}" text-anchor="middle" fill="#fde68a" font-size="11">{st["crps_pct"]:.1f}</text>'
            f'<text x="{x_center:.1f}" y="{height - pad_b + 18}" text-anchor="middle" fill="#cbd5e1" font-size="12">{labels.get(m, m)}</text>'
            f'<text x="{x_center:.1f}" y="{height - pad_b + 34}" text-anchor="middle" fill="#94a3b8" font-size="10">r={st["pearson_r"]:.2f}</text>'
        )

    grid = []
    for t in range(0, int(max_y) + 1, 10):
        y = sy(t)
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#273142" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#94a3b8" font-size="11">{t}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block">'
        + "".join(grid)
        + "".join(bars)
        + f'<rect x="{pad_l}" y="4" width="10" height="10" rx="2" fill="#38bdf8"/><text x="{pad_l + 15}" y="13" fill="#cbd5e1" font-size="11">MAE (pp)</text>'
        + f'<rect x="{pad_l + 82}" y="4" width="10" height="10" rx="2" fill="#f59e0b"/><text x="{pad_l + 97}" y="13" fill="#cbd5e1" font-size="11">CRPS (pp)</text>'
        + "</svg>"
    )


def generate_html_verification_report(report: dict[str, Any]) -> str:
    """Renders a self-contained, zero-dependency interactive HTML verification dashboard."""
    modes = tuple(report.get("modes") or DEFAULT_EVAL_MODES)
    rows = _flatten_scored_rows(report.get("events_detail") or [], modes)
    ov = report.get("overall_independent_raw") or {}
    cal = report.get("out_of_sample_calibration") or {}
    lso = cal.get("transmittance_leave_site_out_cv") or {}
    lso_cal = lso.get("out_of_sample_calibrated") or {}
    lso_raw = lso.get("raw_uncalibrated") or {}
    chrono = cal.get("transmittance_chronological_cv") or {}
    chrono_cal = chrono.get("out_of_sample_calibrated") or {}
    xyear = cal.get("transmittance_cross_year_cv") or {}
    xyear_by_fold = xyear.get("by_fold") or {}
    fail = report.get("failure_diagnostics") or {}

    scatter_svg = _svg_scatter(rows)
    lead_svg = _svg_lead_curve(report.get("by_mode") or {})

    def _table_rows(d: dict[str, dict[str, Any]]) -> str:
        out = []
        for k, v in d.items():
            if not v or v.get("n", 0) == 0:
                continue
            out.append(
                f"<tr><td><strong>{html.escape(str(k))}</strong></td>"
                f"<td>{v['n']}</td><td>{v['mae_pct']:.2f}</td><td>{v['rmse_pct']:.2f}</td>"
                f"<td>{v['bias_pct']:+.2f}</td><td>{v['pearson_r']:.3f}</td>"
                f"<td>{v['crps_pct']:.2f}</td><td>{v['pi80_coverage_pct']:.1f}%</td>"
                f"<td>{v['brier_score']:.3f}</td><td>{v['csi_threat_score']:.2f}</td></tr>"
            )
        return "".join(out)

    xyear_card = ""
    if xyear_by_fold:
        xy_rows = []
        for fl, fd in xyear_by_fold.items():
            fr = fd.get("raw_uncalibrated") or {}
            fc = fd.get("out_of_sample_calibrated") or {}
            xy_rows.append(
                f"<tr><td><strong>{html.escape(fl)}</strong></td>"
                f"<td>{fd.get('n_train', 0)}</td><td>{fd.get('n_test', 0)}</td>"
                f"<td>{fr.get('mae_pct', 0):.2f} &rarr; <strong>{fc.get('mae_pct', 0):.2f}</strong></td>"
                f"<td>{fr.get('crps_pct', 0):.2f} &rarr; <strong>{fc.get('crps_pct', 0):.2f}</strong></td>"
                f"<td>{fr.get('bias_pct', 0):+.2f} &rarr; <strong>{fc.get('bias_pct', 0):+.2f}</strong></td>"
                f"<td>{fr.get('pi80_coverage_pct', 0):.1f}% &rarr; <strong>{fc.get('pi80_coverage_pct', 0):.1f}%</strong></td></tr>"
            )
        xyear_card = f"""
  <div class="card">
    <h2>Strict Cross-Year Holdout Calibration (2024 vs 2025)</h2>
    <table>
      <thead><tr><th>Fold</th><th>Train N</th><th>Test N</th><th>MAE (raw &rarr; cal)</th><th>CRPS (raw &rarr; cal)</th><th>Bias (raw &rarr; cal)</th><th>80% PI Cov (raw &rarr; cal)</th></tr></thead>
      <tbody>{"".join(xy_rows)}</tbody>
    </table>
  </div>"""

    miss_rows = []
    for m in fail.get("top_misses") or []:
        lmh = m.get("cloud_lmh") or [0, 0, 0]
        miss_rows.append(
            f"<tr><td>{html.escape(m['site'])}</td><td>{m['date']}</td><td>{m['phase']}</td>"
            f"<td>{m['mode']}</td><td>{m['truth_pct']:.0f}%</td><td>{m['pred_pct']:.0f}%</td>"
            f"<td style='color:{'#f87171' if m['error_pp'] > 0 else '#60a5fa'}'>{m['error_pp']:+.1f}</td>"
            f"<td>{lmh[0]}/{lmh[1]}/{lmh[2]}</td><td>{html.escape(str(m['blocker_label']))}</td>"
            f"<td><span class='badge'>{html.escape(m['category'])}</span></td></tr>"
        )

    event_rows_json = json.dumps(
        [
            {
                "site": r["site"],
                "date": r["date"],
                "phase": r["phase"],
                "region": r["region"],
                "season": r["season"],
                "mode": r["mode"],
                "tier": r["truth_tier"],
                "truth": r["truth_pct"],
                "pred": r["golden_exp_pct"],
                "err": round(r["golden_exp_pct"] - r["truth_pct"], 1),
                "lmh": f"{r['cloud_low_pct']}/{r['cloud_mid_pct']}/{r['cloud_high_pct']}",
                "blocker": r["blocker_label"],
            }
            for r in rows
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PhotoPlanner Verification Dashboard — {html.escape(str(report.get('run_id', '')))}</title>
<style>
  :root {{ --bg:#0b1120; --card:#111827; --border:#1f2937; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --amber:#f59e0b; --green:#34d399; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--text); font-family:Inter,system-ui,-apple-system,sans-serif; line-height:1.45; }}
  .wrap {{ max-width:1280px; margin:0 auto; }}
  h1 {{ margin:0 0 6px; font-size:24px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
  .kpi .lbl {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; }}
  .kpi .val {{ font-size:24px; font-weight:700; margin-top:4px; }}
  .kpi .delta {{ font-size:12px; color:var(--green); margin-top:2px; }}
  .grid-2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:16px; margin-bottom:20px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; margin-bottom:16px; }}
  .card h2 {{ margin:0 0 12px; font-size:16px; color:#f8fafc; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  th, td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; }}
  .badge {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:10.5px; background:#1e293b; color:#cbd5e1; }}
  .controls {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }}
  select, input {{ background:#0f172a; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:6px 10px; font-size:12.5px; }}
  .legend {{ display:flex; gap:14px; font-size:12px; color:var(--muted); margin-top:6px; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>PhotoPlanner Independent Forecast Verification</h1>
  <div class="sub">
    Run: <code>{html.escape(str(report.get('run_id')))}</code> · Suite: <strong>{html.escape(str(report.get('suite')))}</strong> ·
    Git SHA: <code>{html.escape(str(report.get('git_sha')))}</code> ·
    Events Scored: <strong>{report.get('n_events_scored', 0)}</strong> ({report.get('n_scored_mode_rows', 0)} leak-free forecast rows)
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="lbl">Raw Physics MAE (Indep.)</div><div class="val">{ov.get('mae_pct', 0):.2f} pp</div><div class="delta">RMSE {ov.get('rmse_pct', 0):.2f} · r = {ov.get('pearson_r', 0):.3f}</div></div>
    <div class="kpi"><div class="lbl">Out-of-Sample Cal. MAE</div><div class="val">{lso_cal.get('mae_pct', ov.get('mae_pct', 0)):.2f} pp</div><div class="delta">LSO CV (raw {lso_raw.get('mae_pct', 0):.2f} pp)</div></div>
    <div class="kpi"><div class="lbl">Calibrated CRPS (LSO)</div><div class="val">{lso_cal.get('crps_pct', ov.get('crps_pct', 0)):.2f} pp</div><div class="delta">Chrono CV {chrono_cal.get('crps_pct', 0):.2f} pp</div></div>
    <div class="kpi"><div class="lbl">80% PI Empirical Coverage</div><div class="val">{lso_cal.get('pi80_coverage_pct', ov.get('pi80_coverage_pct', 0)):.1f}%</div><div class="delta">Target: 80.0% (raw {ov.get('pi80_coverage_pct', 0):.1f}%)</div></div>
    <div class="kpi"><div class="lbl">Golden-Hour Hit Brier / BSS</div><div class="val">{ov.get('brier_score', 0):.3f}</div><div class="delta">BSS {ov.get('brier_skill_score_vs_climo', 0):+.3f} · CSI {ov.get('csi_threat_score', 0):.2f}</div></div>
    <div class="kpi"><div class="lbl">Large Miss Rate (≥25 pp)</div><div class="val">{fail.get('large_miss_rate_pct', 0):.1f}%</div><div class="delta">{fail.get('n_large_misses', 0)} / {fail.get('n_total_rows', 0)} rows</div></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Predicted vs Independent Ground-Truth Transmittance</h2>
      {scatter_svg}
      <div class="legend">
        <span><i class="dot" style="background:#f59e0b"></i>Tier 1A: NOAA SURFRAD 1-min Pyrheliometer</span>
        <span><i class="dot" style="background:#38bdf8"></i>Tier 1B: NREL NSRDB 2km GOES DNI</span>
      </div>
    </div>
    <div class="card">
      <h2>Lead-Time Forecast Skill Curve (Leak-Free Operational Runs)</h2>
      {lead_svg}
    </div>
  </div>
{xyear_card}
  <div class="card">
    <h2>Stratified Skill Breakdown (Tier, Year, Phase, Season & Region)</h2>
    <table>
      <thead><tr><th>Stratum</th><th>N</th><th>MAE</th><th>RMSE</th><th>Bias</th><th>r</th><th>CRPS</th><th>80% Cov</th><th>Brier</th><th>CSI</th></tr></thead>
      <tbody>
        {_table_rows(report.get('by_truth_tier') or {})}
        {_table_rows(report.get('by_terrain_shadow_audit') or {})}
        {_table_rows(report.get('by_year') or {})}
        {_table_rows(report.get('by_phase') or {})}
        {_table_rows(report.get('by_season') or {})}
        {_table_rows(report.get('by_region') or {})}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Top Physical Forecast Misses (Failure-Mode Audit)</h2>
    <table>
      <thead><tr><th>Site</th><th>Date</th><th>Phase</th><th>Mode</th><th>Truth</th><th>Pred</th><th>Err</th><th>L/M/H%</th><th>Blocker</th><th>Root Cause</th></tr></thead>
      <tbody>{"".join(miss_rows) or "<tr><td colspan='10'>No misses ≥ 25 pp</td></tr>"}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Interactive Event-Level Explorer</h2>
    <div class="controls">
      <select id="fMode"><option value="">All Modes</option><option>nowcast_1h</option><option>nowcast_3h</option><option>hindcast_24h</option><option>hindcast_72h</option></select>
      <select id="fPhase"><option value="">Sunrise + Sunset</option><option>sunrise</option><option>sunset</option></select>
      <select id="fTier"><option value="">All Truth Tiers</option><option>tier_1a_surfrad</option><option>tier_1b_nsrdb_2km</option></select>
      <input id="fSearch" type="text" placeholder="Filter site / region / date..."/>
    </div>
    <table>
      <thead><tr><th>Site</th><th>Date</th><th>Phase</th><th>Region</th><th>Mode</th><th>Tier</th><th>Truth</th><th>Pred</th><th>Err</th><th>L/M/H%</th><th>Blocker</th></tr></thead>
      <tbody id="expBody"></tbody>
    </table>
  </div>
</div>
<script>
const ROWS = {event_rows_json};
function render() {{
  const fm = document.getElementById('fMode').value;
  const fp = document.getElementById('fPhase').value;
  const ft = document.getElementById('fTier').value;
  const fq = document.getElementById('fSearch').value.toLowerCase();
  const body = document.getElementById('expBody');
  const filtered = ROWS.filter(r =>
    (!fm || r.mode === fm) && (!fp || r.phase === fp) && (!ft || r.tier === ft) &&
    (!fq || (r.site + ' ' + r.region + ' ' + r.date + ' ' + r.blocker).toLowerCase().includes(fq))
  ).sort((a,b) => Math.abs(b.err) - Math.abs(a.err));
  body.innerHTML = filtered.slice(0, 120).map(r => `
    <tr>
      <td>${{r.site}}</td><td>${{r.date}}</td><td>${{r.phase}}</td><td>${{r.region}}</td>
      <td>${{r.mode}}</td><td>${{r.tier}}</td><td>${{r.truth.toFixed(0)}}%</td><td>${{r.pred.toFixed(0)}}%</td>
      <td style="color:${{Math.abs(r.err)>=25 ? (r.err>0?'#f87171':'#60a5fa') : '#94a3b8'}}">${{r.err>=0?'+':''}}${{r.err.toFixed(1)}}</td>
      <td>${{r.lmh}}</td><td>${{r.blocker}}</td>
    </tr>`).join('');
}}
['fMode','fPhase','fTier','fSearch'].forEach(id => document.getElementById(id).addEventListener('input', render));
render();
</script>
</body>
</html>"""
