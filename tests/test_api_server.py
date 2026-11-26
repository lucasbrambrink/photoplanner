"""Tests for the FastAPI/Uvicorn ASGI server and 0.05-degree grid-cell TTL cache."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

import api_server
from planner.weather import alert_engine


def _call_asgi(method: str, path: str, body: bytes = b"", headers: list[tuple[bytes, bytes]] | None = None) -> tuple[int, dict[str, str], bytes]:
    """Minimal zero-dependency ASGI caller so tests don't depend on httpx."""
    raw_path, _, query = path.partition("?")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": raw_path,
        "raw_path": raw_path.encode("utf-8"),
        "query_string": query.encode("utf-8"),
        "headers": headers or [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8420),
    }
    sent_body = False
    status_code = 500
    resp_headers: dict[str, str] = {}
    chunks: list[bytes] = []

    async def _run() -> None:
        nonlocal sent_body, status_code, resp_headers
        done = asyncio.Event()

        async def receive() -> dict:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            await done.wait()
            return {"type": "http.disconnect"}

        async def send(msg: dict) -> None:
            nonlocal status_code, resp_headers
            if msg["type"] == "http.response.start":
                status_code = int(msg["status"])
                resp_headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in msg.get("headers", [])}
            elif msg["type"] == "http.response.body":
                chunks.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    done.set()

        await api_server.app(scope, receive, send)

    asyncio.run(_run())
    return status_code, resp_headers, b"".join(chunks)


class TestApiServerAndGridCache(unittest.TestCase):
    def test_grid_ttl_cache_quantization_lru_and_expiry(self) -> None:
        c = api_server.GridTTLCache(max_entries=16)
        # Two nearby coordinates within 0.05 deg snap to the same 5 km grid cell
        cell_a = c.quantize_cell(37.921, -122.581)
        cell_b = c.quantize_cell(37.912, -122.592)
        self.assertEqual(cell_a, (37.9, -122.6))
        self.assertEqual(cell_a, cell_b)

        c.set("k1", b"v1", ttl_sec=0.05)
        self.assertEqual(c.get("k1"), b"v1")
        time.sleep(0.06)
        self.assertIsNone(c.get("k1"))

        for i in range(20):
            c.set(f"item_{i}", f"val_{i}".encode(), ttl_sec=60.0)
        self.assertEqual(c.stats()["entries"], 16)
        self.assertIsNone(c.get("item_0"))
        self.assertEqual(c.get("item_19"), b"val_19")

    def test_health_and_openapi_endpoints(self) -> None:
        st, _, body = _call_asgi("GET", "/api/health")
        self.assertEqual(st, 200)
        doc = json.loads(body)
        self.assertEqual(doc["status"], "ok")
        self.assertTrue(doc["commercial_safe_mode"])
        self.assertIn("forecast_grid_cache", doc)
        self.assertIn("rust_worker", doc)
        self.assertIn("srtm_max_land_tiles", doc["rust_worker"])
        self.assertIn("srtm_resident_mb", doc["rust_worker"])

        # Commercial-safe mode blocks third-party ALERTCalifornia frame proxying with HTTP 403
        st_ac, _, body_ac = _call_asgi("GET", "/api/webcam-image?source=alertca&id=mttam-cam01")
        self.assertEqual(st_ac, 403)
        self.assertIn("commercial-safe", body_ac.decode("utf-8").lower())

        st_lm, _, body_lm = _call_asgi("GET", "/api/landmarks")
        self.assertEqual(st_lm, 200)
        self.assertIsInstance(json.loads(body_lm), list)

        st_doc, _, body_doc = _call_asgi("GET", "/openapi.json")
        self.assertEqual(st_doc, 200)
        spec = json.loads(body_doc)
        self.assertIn("/api/weather-forecast", spec["paths"])
        self.assertIn("/api/regional-hunt", spec["paths"])
        self.assertIn("/api/trail-upload", spec["paths"])
        self.assertIn("/api/watchlists", spec["paths"])

    def test_watchlist_crud_over_asgi(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            orig_db = alert_engine.DEFAULT_WATCHLIST_DB
            alert_engine.DEFAULT_WATCHLIST_DB = Path(td) / "test_wl.sqlite3"
            try:
                payload = json.dumps({
                    "name": "Mt Tam Sunset Watch",
                    "spot_id": "mt_tam_east_peak",
                    "phase": "sunset",
                    "min_calibrated_p50_pct": 55.0,
                }).encode("utf-8")
                st_c, _, body_c = _call_asgi("POST", "/api/watchlists", body=payload)
                self.assertEqual(st_c, 201)
                created = json.loads(body_c)
                rule_id = created["id"]

                st_l, _, body_l = _call_asgi("GET", "/api/watchlists")
                self.assertEqual(st_l, 200)
                rules = json.loads(body_l)["watchlists"]
                self.assertEqual(len(rules), 1)
                self.assertEqual(rules[0]["id"], rule_id)

                st_d, _, body_d = _call_asgi("DELETE", f"/api/watchlists/{rule_id}")
                self.assertEqual(st_d, 200)
                self.assertTrue(json.loads(body_d)["deleted"])
            finally:
                alert_engine.DEFAULT_WATCHLIST_DB = orig_db

    def test_trail_upload_gpx_over_asgi(self) -> None:
        gpx = (
            b'<?xml version="1.0"?><gpx version="1.1"><trk><name>Test Ridge</name><trkseg>'
            b'<trkpt lat="37.9000" lon="-122.5800"><ele>150</ele></trkpt>'
            b'<trkpt lat="37.9100" lon="-122.5800"><ele>450</ele></trkpt>'
            b'<trkpt lat="37.9200" lon="-122.5800"><ele>720</ele></trkpt>'
            b'</trkseg></trk></gpx>'
        )
        st, _, body = _call_asgi(
            "POST",
            "/api/trail-upload?include_weather=false&date=2025-06-14&phase=sunset",
            body=gpx,
            headers=[(b"content-type", b"application/gpx+xml")],
        )
        self.assertEqual(st, 200)
        doc = json.loads(body)
        self.assertEqual(doc["trail_name"], "Test Ridge")
        self.assertIn("field_card", doc)
        self.assertIn("nudges", doc)

    def test_frontend_lightweight_viewfinder_and_field_card_ui(self) -> None:
        st_html, _, body_html = _call_asgi("GET", "/")
        self.assertEqual(st_html, 200)
        html = body_html.decode("utf-8")
        # Default-on 3D Volumetric Clouds & Light Rays + USGS Satellite photorealistic terrain
        self.assertIn('id="toggle-atmosphere" checked />', html)
        self.assertIn('class="seg-btn is-active" id="btn-surf-aerial" data-surface="1"', html)
        # Commercial-safe scientific camera copy (no ALERTCalifornia branding in UI)
        self.assertIn("PhenoCam / NEON Scientific Sky Chromaticity", html)
        self.assertNotIn("ALERTCalifornia", html)
        # Full-frame 16/24/35/50/70/200mm camera lens viewfinder switcher
        for mm in ("16", "24", "35", "50", "70", "200"):
            self.assertIn(f'data-focal-mm="{mm}"', html)
        # Probabilistic Fog-Sea Elevation Slice P10 / P50 / P90 controls
        for band in ("p10", "p50", "p90", "off"):
            self.assertIn(f'data-fog-band="{band}"', html)
        # Full 00:00–24:00 (0..1440) golden-hour + astro scrubber & Offline Field Card button
        self.assertIn('min="0"', html)
        self.assertIn('max="1440"', html)
        self.assertIn('id="btn-export-field-card"', html)
        # LightFinder Canvas-First Floating Sheet, 7-Archetype Scenario Switcher, Field Reference & Viewport Pills
        self.assertIn('id="planner-sheet"', html)
        self.assertIn('id="archetype-mode-label"', html)
        self.assertIn('id="scenario-cards-list"', html)
        self.assertIn('id="archetype-field-ref"', html)
        self.assertIn('id="archetype-breakdown-details"', html)
        self.assertIn('id="viewport-scenario-pills"', html)
        self.assertIn('id="btn-archetype-auto"', html)
        self.assertIn('id="regional-hunt-results"', html)
        self.assertIn('id="gpx-file-input"', html)
        self.assertIn('id="lab-drawer" hidden', html)

        st_js, _, body_js = _call_asgi("GET", "/app.js")
        self.assertEqual(st_js, 200)
        js = body_js.decode("utf-8")
        self.assertIn("showAtmosphere: true", js)
        self.assertIn("surfaceStyle: 1", js)
        self.assertIn("function focalLengthToFov(", js)
        self.assertIn("function exportOfflineFieldCard(", js)
        self.assertIn("function renderLightArchetypeScenarioUI(", js)
        self.assertIn("function selectLightArchetype(", js)
        self.assertIn("hybridGodRaysGroup", js)
        self.assertIn("fogSeaSliceMesh", js)


if __name__ == "__main__":
    unittest.main()
