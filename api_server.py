#!/usr/bin/env python3
"""
PhotoPlanner Production ASGI Server (FastAPI + Uvicorn).

Replaces single-process `ThreadingHTTPServer` for Linux production deployments while maintaining
100% route compatibility with `server.py`, adding:
- OpenAPI 3.1 schema & interactive docs (`/docs`, `/openapi.json`)
- Health & readiness probe (`/api/health`)
- Bounded LRU `0.05°` (~5 km) spatial grid-cell TTL forecast cache to cut upstream NWP cost
- Commercial-safe endpoint surface (no ALERTCalifornia proxying required)
- Full support for 3D Corridor Weather, Regional Light Hunting, Custom GPX/GeoJSON Trail Upload,
  Calibrated Watchlists & Alerts, and the Independent Verification Scoreboard.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8420 --workers 2
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
import datetime as dt
import json
import os
import threading
import time
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from planner.gpx_trail_analyzer import analyze_uploaded_trail
from planner.rust_worker import get_rust_worker, shutdown_rust_worker
from planner.weather import (
    BENCHMARK_REPORT_JSON_PATH,
    EXPERIMENTS_DIR,
    SUMMARY_JSON_PATH,
    VERIFICATION_HTML_PATH,
    create_watchlist_rule,
    delete_watchlist_rule,
    evaluate_watchlist_rule_live,
    hunt_regional_light,
    list_watchlist_rules,
    run_full_backtest_suite,
    solve_weather_and_corridor,
)
from planner.weather.calibration import load_calibration_bundle
import server as legacy_server


class GridTTLCache:
    """Thread-safe bounded LRU + TTL cache keyed on 0.05° (~5 km) spatial cells."""

    def __init__(self, max_entries: int = 512) -> None:
        self._max_entries = max(16, int(max_entries))
        self._lock = threading.Lock()
        self._store: OrderedDict[str, tuple[float, float, bytes]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def quantize_cell(lat: float, lon: float, step_deg: float = 0.05) -> tuple[float, float]:
        return (
            round(round(lat / step_deg) * step_deg, 3),
            round(round(lon / step_deg) * step_deg, 3),
        )

    def get(self, key: str) -> bytes | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, _, payload = entry
            if now >= expires_at:
                self._store.pop(key, None)
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return payload

    def set(self, key: str, payload: bytes, ttl_sec: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._store[key] = (now + ttl_sec, now, payload)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "hits": self.hits,
                "misses": self.misses,
            }


FORECAST_CACHE = GridTTLCache(max_entries=int(os.environ.get("PHOTOPLANNER_CACHE_MAX", "512")))
HUNT_CACHE = GridTTLCache(max_entries=128)


def _today_pacific() -> str:
    return dt.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


class WatchlistCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    spot_id: str | None = None
    region: str | None = None
    lat: float | None = None
    lon: float | None = None
    elev_m: float | None = None
    phase: str = "sunset"
    target_phenomena: list[str] | None = None
    min_calibrated_p50_pct: float = 50.0
    min_above_fog_prob_pct: float = 0.0
    min_mirror_index: float = 0.0
    min_opportunity_score: float = 65.0
    lead_window: str = "24h"


class WatchlistDeleteRequest(BaseModel):
    id: str


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        yield
    finally:
        shutdown_rust_worker()


app = FastAPI(
    title="PhotoPlanner 3D Terrain-Coupled Light Intelligence API",
    version="2.0.0",
    description=(
        "3D curved-Earth radiative corridor solver, probabilistic inversion breakout predictor, "
        "regional inverse-raycasting light hunter, GPX/GeoJSON trail analyzer, and calibrated alert engine."
    ),
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    from planner.weather.webcam_truth import is_alertca_proxy_enabled

    cal_bundle = load_calibration_bundle()
    return {
        "status": "ok",
        "service": "photoplanner-asgi",
        "commercial_safe_mode": not is_alertca_proxy_enabled(),
        "rust_binary_ready": legacy_server.RUST_BIN.exists(),
        "rust_worker": get_rust_worker().health_snapshot(probe=False),
        "elevation_dir": str(legacy_server.ELEVATION_DIR),
        "calibration_loaded": cal_bundle is not None,
        "calibration_training_events": int(cal_bundle[0].n_unique_events) if cal_bundle else 0,
        "open_meteo_commercial_key": bool(os.environ.get("OPEN_METEO_API_KEY")),
        "forecast_grid_cache": FORECAST_CACHE.stats(),
        "regional_hunt_cache": HUNT_CACHE.stats(),
    }


@app.get("/api/landmarks")
def api_landmarks() -> list[dict[str, Any]]:
    return legacy_server.get_landmarks_list()


@app.get("/api/shoot-plan")
def api_shoot_plan(
    landmark: str = "Half Dome",
    date: str | None = None,
    max_miles: float = 16.0,
    top_k: int = 3,
) -> Response:
    resolved_date = date or _today_pacific()
    payload = legacy_server.run_shoot_landmark(landmark, resolved_date, max_miles, top_k)
    return Response(content=payload, media_type="application/json")


@app.get("/api/shoot-coord")
def api_shoot_coord(
    lat: float = 37.74554,
    lon: float = -119.53315,
    date: str | None = None,
    name: str = "Custom Terrain Pin",
    max_miles: float = 16.0,
    top_k: int = 3,
) -> Response:
    resolved_date = date or _today_pacific()
    payload = legacy_server.run_shoot_coordinate(lat, lon, name, resolved_date, max_miles, top_k)
    return Response(content=payload, media_type="application/json")


@app.get("/api/tripod-los")
def api_tripod_los(
    tripod_lat: float,
    tripod_lon: float,
    target_lat: float,
    target_lon: float,
    target_name: str = "Target Landmark",
    date: str | None = None,
) -> Response:
    resolved_date = date or _today_pacific()
    payload = legacy_server.run_tripod_los(
        tripod_lat, tripod_lon, target_lat, target_lon, target_name, resolved_date
    )
    return Response(content=payload, media_type="application/json")


@app.get("/api/terrain-patch.bin")
def api_terrain_patch_bin(
    lat: float = 37.74554,
    lon: float = -119.53315,
    span_m: float = 9500.0,
    grid_size: int = 385,
) -> Response:
    payload = legacy_server.run_terrain_patch_binary(lat, lon, span_m, grid_size)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/terrain-patch")
def api_terrain_patch(
    lat: float = 37.74554,
    lon: float = -119.53315,
    span_m: float = 9500.0,
    grid_size: int = 257,
) -> Response:
    payload = legacy_server.run_terrain_patch(lat, lon, span_m, grid_size)
    return Response(content=payload, media_type="application/json")


@app.get("/api/usgs-ortho")
def api_usgs_ortho(
    lat: float = 37.74554,
    lon: float = -119.53315,
    span_m: float = 9500.0,
) -> Response:
    jpg = legacy_server.fetch_usgs_ortho_jpeg(lat, lon, span_m)
    return Response(
        content=jpg,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/webcam-truth")
def api_webcam_truth(
    lat: float = 37.5945,
    lon: float = -122.5028,
    date: str | None = None,
    phase: str = "sunset",
    elev_m: float = 220.0,
    sun_az: float | None = None,
) -> dict[str, Any]:
    from planner.weather.webcam_truth import fetch_webcam_ground_truth

    resolved_date = date or _today_pacific()
    return fetch_webcam_ground_truth(
        lat=lat,
        lon=lon,
        date_str=resolved_date,
        phase=phase,
        elev_m=elev_m,
        sun_az_deg=sun_az,
        analyze_pixels=True,
    )


@app.get("/api/webcam-image")
def api_webcam_image(
    source: str = "phenocam",
    path: str = "",
    id: str = "",
) -> Response:
    from planner.weather.webcam_truth import fetch_cached_webcam_jpeg

    ident = path or id
    try:
        jpg = fetch_cached_webcam_jpeg(source.strip().lower(), ident)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    max_age = 86400 if source.strip().lower() == "phenocam" else 120
    return Response(
        content=jpg,
        media_type="image/jpeg",
        headers={"Cache-Control": f"public, max-age={max_age}"},
    )


@app.get("/api/weather-forecast")
def api_weather_forecast(
    lat: float = 37.74554,
    lon: float = -119.53315,
    date: str | None = None,
    phase: str = "sunset",
    sun_az: float = 259.1,
    sun_el: float = 0.4,
    elev_m: float = 2693.0,
    engine: str = "ensemble",
    sim_now_utc: str | None = None,
    replay_mode: str = "auto",
    include_lut: bool = True,
) -> Response:
    resolved_date = (date or _today_pacific()).strip()
    today_str = _today_pacific()
    sim_dt = dt.datetime.fromisoformat(sim_now_utc.replace("Z", "+00:00")) if sim_now_utc else None

    # 0.05° (~5 km) spatial grid-cell cache quantization with 25 m vertical elevation bucket
    cell_lat, cell_lon = GridTTLCache.quantize_cell(lat, lon, step_deg=0.05)
    elev_bucket = int(round(elev_m / 25.0) * 25)
    az_bucket = round(sun_az * 2.0) / 2.0
    el_bucket = round(sun_el * 2.0) / 2.0
    cache_key = (
        f"wx_g05:{cell_lat:.3f}:{cell_lon:.3f}:{elev_bucket}:{resolved_date}:"
        f"{phase.lower()}:{az_bucket:.1f}:{el_bucket:.1f}:{engine.lower()}:"
        f"{replay_mode.lower()}:{sim_now_utc or 'live'}:lut={int(include_lut)}"
    )
    cached = FORECAST_CACHE.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="application/json",
            headers={"X-PhotoPlanner-Cache": "HIT", "Cache-Control": "public, max-age=180"},
        )

    payload = solve_weather_and_corridor(
        lat=lat,
        lon=lon,
        date_str=resolved_date,
        phase=phase,
        sun_az_deg=sun_az,
        sun_el_deg=sun_el,
        target_elev_m=elev_m,
        model_engine=engine,
        simulated_now_utc=sim_dt,
        replay_mode=replay_mode,
        include_lut=include_lut,
    )
    ttl_sec = 300.0 if (sim_dt is None and resolved_date >= today_str) else 3600.0
    FORECAST_CACHE.set(cache_key, payload, ttl_sec=ttl_sec)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"X-PhotoPlanner-Cache": "MISS", "Cache-Control": "public, max-age=180"},
    )


@app.get("/api/regional-hunt")
def api_regional_hunt(
    date: str | None = None,
    phase: str = "sunset",
    region: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_miles: float = 120.0,
    max_spots: int = Query(default=8, ge=1, le=24),
    replay_mode: str = "auto",
) -> Response:
    resolved_date = (date or _today_pacific()).strip()
    h_key = f"hunt:{resolved_date}:{phase}:{region}:{lat}:{lon}:{radius_miles}:{max_spots}:{replay_mode}"
    cached = HUNT_CACHE.get(h_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="application/json",
            headers={"X-PhotoPlanner-Cache": "HIT", "Cache-Control": "public, max-age=600"},
        )
    res = hunt_regional_light(
        date_str=resolved_date,
        phase=phase,
        region=region,
        center_lat=lat,
        center_lon=lon,
        radius_miles=radius_miles,
        max_spots=max_spots,
        replay_mode=replay_mode,
    )
    payload = json.dumps(res).encode("utf-8")
    HUNT_CACHE.set(h_key, payload, ttl_sec=600.0)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"X-PhotoPlanner-Cache": "MISS", "Cache-Control": "public, max-age=600"},
    )


@app.post("/api/trail-upload")
async def api_trail_upload(
    request: Request,
    date: str | None = None,
    phase: str = "sunset",
    setup_buffer_min: int = 25,
    replay_mode: str = "auto",
    include_weather: bool = True,
) -> dict[str, Any]:
    raw = await request.body()
    if not raw or len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Body must be non-empty and <= 10 MB.")
    raw_text = raw.decode("utf-8", errors="replace")
    ctype = (request.headers.get("content-type") or "").lower()
    try:
        if "xml" in ctype or raw_text.lstrip().startswith("<"):
            return analyze_uploaded_trail(
                gpx_xml=raw_text,
                date_str=date,
                phase=phase,
                setup_buffer_min=setup_buffer_min,
                replay_mode=replay_mode,
                include_weather=include_weather,
            )
        body_json = json.loads(raw_text)
        if not isinstance(body_json, dict):
            raise ValueError("JSON request body must be an object.")
        if "gpx_xml" in body_json or "geojson" in body_json:
            return analyze_uploaded_trail(
                gpx_xml=body_json.get("gpx_xml"),
                geojson=body_json.get("geojson"),
                date_str=body_json.get("date", date),
                phase=body_json.get("phase", phase),
                setup_buffer_min=int(body_json.get("setup_buffer_min", setup_buffer_min)),
                replay_mode=str(body_json.get("replay_mode", replay_mode)),
                include_weather=bool(body_json.get("include_weather", include_weather)),
            )
        return analyze_uploaded_trail(
            geojson=body_json,
            date_str=date,
            phase=phase,
            setup_buffer_min=setup_buffer_min,
            replay_mode=replay_mode,
            include_weather=include_weather,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/watchlists")
def api_list_watchlists() -> dict[str, Any]:
    return {"watchlists": list_watchlist_rules()}


@app.post("/api/watchlists", status_code=201)
def api_create_watchlist(req: WatchlistCreateRequest) -> dict[str, Any]:
    try:
        return create_watchlist_rule(**req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/watchlists/delete")
def api_delete_watchlist_post(req: WatchlistDeleteRequest) -> dict[str, bool]:
    return {"deleted": delete_watchlist_rule(req.id)}


@app.delete("/api/watchlists/{rule_id}")
def api_delete_watchlist_rest(rule_id: str) -> dict[str, bool]:
    return {"deleted": delete_watchlist_rule(rule_id)}


@app.get("/api/alerts/evaluate")
def api_evaluate_alerts(
    date: str | None = None,
    replay_mode: str = "auto",
    rule_id: str | None = None,
) -> dict[str, Any]:
    resolved_date = (date or _today_pacific()).strip()
    rules = [r for r in list_watchlist_rules() if r["enabled"]]
    if rule_id:
        rules = [r for r in rules if r["id"] == rule_id]
    evals: list[dict[str, Any]] = []
    for r in rules:
        evals.extend(evaluate_watchlist_rule_live(r, date_str=resolved_date, replay_mode=replay_mode))
    return {
        "date": resolved_date,
        "n_rules": len(rules),
        "triggered": [e for e in evals if e["decision"] == "TRIGGER"],
        "watching": [e for e in evals if e["decision"] == "WATCH"],
        "suppressed": [e for e in evals if e["decision"] == "SUPPRESS"],
        "evaluations": evals,
    }


@app.get("/api/weather-backtest")
def api_weather_backtest(refresh: bool = False) -> Response:
    if (not refresh) and SUMMARY_JSON_PATH.exists():
        data = SUMMARY_JSON_PATH.read_bytes()
    elif (not refresh) and BENCHMARK_REPORT_JSON_PATH.exists():
        data = BENCHMARK_REPORT_JSON_PATH.read_bytes()
    else:
        data = json.dumps(run_full_backtest_suite()).encode("utf-8")
    return Response(
        content=data,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/backtest-report", response_class=HTMLResponse)
def api_backtest_report() -> Response:
    if VERIFICATION_HTML_PATH.exists():
        return HTMLResponse(content=VERIFICATION_HTML_PATH.read_text(encoding="utf-8"))
    return JSONResponse(
        status_code=404,
        content={"error": "No verification_report.html generated yet. Run scripts/run_weather_backtest.py."},
    )


@app.get("/api/backtest-experiments")
def api_backtest_experiments() -> dict[str, Any]:
    exps: list[dict[str, Any]] = []
    if EXPERIMENTS_DIR.exists():
        for p in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True)[:25]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                exps.append(
                    {
                        "run_id": d.get("run_id"),
                        "suite": d.get("suite"),
                        "git_sha": d.get("git_sha"),
                        "generated_at_utc": d.get("generated_at_utc"),
                        "n_events_scored": d.get("n_events_scored"),
                        "overall_independent_raw": d.get("overall_independent_raw"),
                    }
                )
            except Exception:
                continue
    return {"experiments": exps}


@app.get("/api/annual-calendar")
def api_annual_calendar(
    lat: float = 37.74554,
    lon: float = -119.53315,
    year: int | None = None,
) -> Response:
    resolved_year = year or dt.datetime.now(ZoneInfo("America/Los_Angeles")).year
    payload = legacy_server.run_annual_notch_calendar(lat, lon, resolved_year)
    return Response(content=payload, media_type="application/json")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(legacy_server.WEB_DIR / "index.html")


@app.get("/{file_path:path}")
def serve_static(file_path: str) -> FileResponse:
    candidate = (legacy_server.WEB_DIR / file_path).resolve()
    web_root = legacy_server.WEB_DIR.resolve()
    if not str(candidate).startswith(str(web_root)) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(candidate)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8420"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
