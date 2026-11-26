"""End-to-end tests for Launch Readiness Goals 1–4:
  - Goal 1: Nationwide US spot, GPS & geocoding search (`/api/search-spots`)
  - Goal 2: Calibrated alert delivery (email/webhook) + deduplication (`/api/alerts/dispatch`)
  - Goal 3: Production readiness probe (`/api/ready`), IP rate limiter (`429`), Docker/Fly manifests
  - Goal 4: Dynamic OpenGraph share previews (`GET /?name=...`) + 1-tap crowd field reports (`/api/field-reports`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import unittest

import api_server
from planner import field_reports
from planner.spot_search import parse_gps_query, search_photography_spots
from planner.weather import alert_engine
from planner.weather.alert_delivery import build_alert_message, dispatch_due_alerts
from tests.test_api_server import _call_asgi


ROOT = Path(__file__).resolve().parents[1]


class TestLaunchReadinessGoals(unittest.TestCase):
    def setUp(self) -> None:
        api_server.RATE_LIMITER.reset()

    def tearDown(self) -> None:
        api_server.RATE_LIMITER.reset()

    # --- Goal 1: Nationwide US Spot & GPS Search ---
    def test_goal1_gps_and_curated_nationwide_search_over_asgi(self) -> None:
        gps = parse_gps_query("46.7865, -121.7353")
        self.assertIsNotNone(gps)
        self.assertAlmostEqual(gps["lat"], 46.7865, places=4)
        self.assertAlmostEqual(gps["lon"], -121.7353, places=4)

        gps_hem = parse_gps_query("37.7459° N, 119.5332° W")
        self.assertIsNotNone(gps_hem)
        self.assertAlmostEqual(gps_hem["lon"], -119.5332, places=4)
        self.assertIsNone(parse_gps_query("99.5, -200.0"))

        # Curated nationwide search covers non-CA states (WA, UT, CO, ME)
        for q, expected_state in (("rainier", "WA"), ("maroon", "CO"), ("cadillac", "ME"), ("mesa arch", "UT")):
            res = search_photography_spots(q, allow_remote=False)
            self.assertGreaterEqual(res["count"], 1, msg=f"No match for {q}")
            self.assertEqual(res["results"][0]["state"], expected_state)

        st, _, body = _call_asgi("GET", "/api/search-spots?q=37.9235,-122.5965&remote=false")
        self.assertEqual(st, 200)
        doc = json.loads(body)
        self.assertEqual(doc["count"], 1)
        self.assertEqual(doc["results"][0]["source"], "gps")

        st2, _, body2 = _call_asgi("GET", "/api/search-spots?q=acadia&remote=false")
        self.assertEqual(st2, 200)
        self.assertIn("Cadillac Mountain", body2.decode("utf-8"))

    # --- Goal 2: Alert Delivery, Deduplication & Admin Dispatch ---
    def test_goal2_alert_delivery_formatting_dedup_and_admin_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "wl_delivery.sqlite3"
            rule = alert_engine.create_watchlist_rule(
                name="Mt Tam Sunset Cloud Sea",
                spot_id="mt_tam_east",
                phase="sunset",
                min_calibrated_p50_pct=60.0,
                notify_email="shooter@example.com",
                notify_webhook_url="https://ntfy.sh/photoplanner-test",
                db_path=db,
            )
            self.assertEqual(rule["notify_email"], "shooter@example.com")
            self.assertEqual(rule["notify_webhook_url"], "https://ntfy.sh/photoplanner-test")

            fake_eval = {
                "rule_id": rule["id"],
                "spot_name": "Mt. Tamalpais East Peak",
                "phase": "sunset",
                "decision": "TRIGGER",
                "opportunity_score": 84,
                "primary_phenomenon": "Cloud-Sea Breakout",
                "calibrated_p10_p50_p90": [58.0, 79.0, 92.0],
                "reason": "Summit +1,190 ft above fog lid with clear Pacific corridor.",
            }
            msg = build_alert_message(rule, fake_eval, date_str="2026-11-20", base_url="https://photoplanner.app")
            self.assertIn("[TRIGGER]", msg["subject"])
            self.assertIn("79% Light Odds", msg["subject"])
            self.assertIn("https://photoplanner.app/?", msg["deep_link"])
            self.assertIn("Open 3D Lens Viewfinder", msg["html"])

            # Dry-run dispatch previews both channels without mutating dedup state
            dry = dispatch_due_alerts(
                date_str="2026-11-20",
                dry_run=True,
                db_path=db,
                evaluator_fn=lambda _r, **_kw: [fake_eval],
            )
            self.assertEqual(dry["n_sent"], 1)
            self.assertEqual(dry["sent"][0]["channels"], ["email:shooter@example.com", "webhook"])

            # Marking notified suppresses duplicate alerts on the same (date, phase)
            alert_engine.mark_watchlist_notified(rule["id"], "2026-11-20", "sunset", db_path=db)
            dedup = dispatch_due_alerts(
                date_str="2026-11-20",
                dry_run=False,
                db_path=db,
                evaluator_fn=lambda _r, **_kw: [fake_eval],
            )
            self.assertEqual(dedup["n_sent"], 0)
            self.assertEqual(dedup["n_skipped_dedup"], 1)

        # Admin token enforcement on POST /api/alerts/dispatch
        orig_tok = os.environ.get("PHOTOPLANNER_ADMIN_TOKEN")
        orig_db = alert_engine.DEFAULT_WATCHLIST_DB
        with tempfile.TemporaryDirectory() as td:
            alert_engine.DEFAULT_WATCHLIST_DB = Path(td) / "empty_wl.sqlite3"
            os.environ["PHOTOPLANNER_ADMIN_TOKEN"] = "secret_dispatch_tok"
            try:
                st_no, _, _ = _call_asgi("POST", "/api/alerts/dispatch?dry_run=true")
                self.assertEqual(st_no, 401)
                st_ok, _, body_ok = _call_asgi(
                    "POST",
                    "/api/alerts/dispatch?dry_run=true",
                    headers=[(b"x-admin-token", b"secret_dispatch_tok")],
                )
                self.assertEqual(st_ok, 200)
                self.assertTrue(json.loads(body_ok)["dry_run"])
            finally:
                alert_engine.DEFAULT_WATCHLIST_DB = orig_db
                if orig_tok is None:
                    os.environ.pop("PHOTOPLANNER_ADMIN_TOKEN", None)
                else:
                    os.environ["PHOTOPLANNER_ADMIN_TOKEN"] = orig_tok

    # --- Goal 3: Production Readiness, Rate Limiter & Container Manifests ---
    def test_goal3_ready_probe_rate_limiter_and_deploy_files(self) -> None:
        st_r, _, body_r = _call_asgi("GET", "/api/ready")
        self.assertEqual(st_r, 200)
        self.assertTrue(json.loads(body_r)["ready"])

        # Rate limiter enforces cap with HTTP 429 + Retry-After, while exempting /api/ready
        orig_rpm = api_server.RATE_LIMITER.max_rpm
        api_server.RATE_LIMITER.max_rpm = 3
        api_server.RATE_LIMITER.reset()
        try:
            for i in range(3):
                st, hdrs, _ = _call_asgi(
                    "GET",
                    "/api/search-spots?q=tam&remote=false",
                    headers=[(b"x-forwarded-for", b"203.0.113.9")],
                )
                self.assertEqual(st, 200)
                self.assertEqual(hdrs.get("x-ratelimit-limit"), "3")
                self.assertEqual(hdrs.get("x-ratelimit-remaining"), str(2 - i))

            st_429, hdrs_429, _ = _call_asgi(
                "GET",
                "/api/search-spots?q=tam&remote=false",
                headers=[(b"x-forwarded-for", b"203.0.113.9")],
            )
            self.assertEqual(st_429, 429)
            self.assertIn("retry-after", hdrs_429)

            # Health/readiness probes remain exempt even when IP is throttled
            st_exempt, _, _ = _call_asgi(
                "GET", "/api/ready", headers=[(b"x-forwarded-for", b"203.0.113.9")]
            )
            self.assertEqual(st_exempt, 200)
        finally:
            api_server.RATE_LIMITER.max_rpm = orig_rpm
            api_server.RATE_LIMITER.reset()

        # Verify deployment artifacts exist & are well-formed
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("cargo build --release --bin photoplanner", dockerfile)
        self.assertIn("USER appuser", dockerfile)
        self.assertIn("/api/ready", dockerfile)
        self.assertTrue((ROOT / "docker-compose.yml").is_file())
        self.assertTrue((ROOT / "fly.toml").is_file())
        self.assertTrue((ROOT / "DEPLOY.md").is_file())

    # --- Goal 4: Dynamic OpenGraph Share Links & 1-Tap Crowd Ground-Truth Reports ---
    def test_goal4_dynamic_og_share_preview_crowd_reports_and_dom_parity(self) -> None:
        st_og, _, body_og = _call_asgi(
            "GET", "/?name=Half+Dome&date=2026-11-20&phase=sunset&mm=70"
        )
        self.assertEqual(st_og, 200)
        og_html = body_og.decode("utf-8")
        self.assertIn('property="og:title" content="Half Dome — Sunset Light &amp; 3D Viewfinder (2026-11-20) · 70mm FF | LightFinder"', og_html)
        self.assertIn('name="twitter:card"', og_html)

        # 1-tap crowd field report submission & summary over ASGI
        orig_fr_db = field_reports.DEFAULT_FIELD_REPORTS_DB
        with tempfile.TemporaryDirectory() as td:
            field_reports.DEFAULT_FIELD_REPORTS_DB = Path(td) / "field_reports.sqlite3"
            try:
                for outcome, pred in (("epic", 82.0), ("partial", 58.0), ("bust", 22.0)):
                    payload = json.dumps(
                        {
                            "spot_name": "Mt. Tamalpais East Peak",
                            "lat": 37.9235,
                            "lon": -122.5965,
                            "date": "2026-11-20",
                            "phase": "sunset",
                            "outcome": outcome,
                            "phenomenon": "cloud_sea",
                            "predicted_p50_pct": pred,
                            "predicted_alpenglow_score": 78.0,
                            "predicted_fog_top_ft": 1350.0,
                        }
                    ).encode("utf-8")
                    st_p, _, _ = _call_asgi("POST", "/api/field-reports", body=payload)
                    self.assertEqual(st_p, 201)

                st_s, _, body_s = _call_asgi(
                    "GET", "/api/field-reports/summary?spot_name=Mt.+Tamalpais+East+Peak"
                )
                self.assertEqual(st_s, 200)
                summary = json.loads(body_s)
                self.assertEqual(summary["total_reports"], 3)
                self.assertEqual(summary["counts"], {"epic": 1, "partial": 1, "bust": 1})
                self.assertAlmostEqual(summary["positive_rate_pct"], 66.7, places=1)
                self.assertEqual(summary["mean_predicted_p50_by_outcome"]["epic"], 82.0)
            finally:
                field_reports.DEFAULT_FIELD_REPORTS_DB = orig_fr_db

        # Verify 100% DOM ID coverage between web/app.js and web/index.html (zero missing hooks)
        js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        html_src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        js_ids = set(re.findall(r"getElementById\([\"']([^\"']+)[\"']\)", js))
        html_ids = set(re.findall(r'id="([^"]+)"', html_src))
        self.assertEqual(sorted(js_ids - html_ids), [])


if __name__ == "__main__":
    unittest.main()
