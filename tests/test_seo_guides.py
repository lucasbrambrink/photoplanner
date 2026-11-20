"""Tests for Task 4.4: Dynamic SEO Light Guides, Sitemap/Robots, and B2B JSON API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from xml.etree import ElementTree as ET

from planner.seo_guides import (
    REGION_LABELS,
    build_spot_guide_payload,
    get_seo_spot,
    list_seo_guides,
    render_robots_txt,
    render_sitemap_xml,
    render_spot_guide_html,
)
from tests.test_api_server import _call_asgi


class TestSeoLightGuides(unittest.TestCase):
    def test_catalog_and_slug_resolution_cover_40_spots_and_7_regions(self) -> None:
        guides = list_seo_guides()
        self.assertGreaterEqual(len(guides), 40)
        regions = {g["region"] for g in guides}
        for expected in ("bay_area", "central_coast", "sierra", "pnw", "rockies", "southwest", "east_coast"):
            self.assertIn(expected, regions)
            self.assertIn(expected, REGION_LABELS)

        # Unique slugs across the entire catalog
        slugs = [g["slug"] for g in guides]
        self.assertEqual(len(slugs), len(set(slugs)))

        # Accepts both hyphenated SEO slug and raw catalog ID
        tam_by_slug = get_seo_spot("mt-tamalpais-east-peak")
        tam_by_id = get_seo_spot("mt_tam_east")
        self.assertIsNotNone(tam_by_slug)
        self.assertEqual(tam_by_slug["id"], tam_by_id["id"])
        self.assertIsNone(get_seo_spot("nonexistent-mountain-999"))

    def test_payload_and_html_contain_valid_jsonld_opengraph_and_faqs(self) -> None:
        payload = build_spot_guide_payload(
            "mt-tamalpais-east-peak",
            date_str="2025-06-14",
            phase="sunset",
            include_live_forecast=False,
            base_url="https://photoplanner.app",
        )
        self.assertEqual(payload["date"], "2025-06-14")
        self.assertEqual(payload["phase"], "sunset")
        self.assertIn("sunrise_peak_local", payload["ephemeris"])
        self.assertIn("sunset_peak_local", payload["ephemeris"])
        self.assertGreaterEqual(len(payload["nearby_guides"]), 3)

        html_doc = render_spot_guide_html(payload)
        self.assertIn('<link rel="canonical" href="https://photoplanner.app/guides/mt-tamalpais-east-peak"', html_doc)
        self.assertIn('property="og:title"', html_doc)
        self.assertIn('name="twitter:card"', html_doc)
        self.assertIn("Open in 3D Lens Viewfinder", html_doc)

        # Extract and parse embedded JSON-LD structured data
        m = re.search(r'<script type="application/ld\+json">(.*?)</script>', html_doc, re.S)
        self.assertIsNotNone(m)
        ld = json.loads(m.group(1))
        types = {node["@type"] for node in ld["@graph"]}
        self.assertIn("TouristAttraction", types)
        self.assertIn("FAQPage", types)

    def test_sitemap_xml_and_robots_txt_are_valid(self) -> None:
        xml_str = render_sitemap_xml("https://photoplanner.app", date_str="2025-06-14")
        root = ET.fromstring(xml_str)
        locs = [el.text for el in root.iter() if el.tag.endswith("loc")]
        self.assertIn("https://photoplanner.app/", locs)
        self.assertIn("https://photoplanner.app/guides", locs)
        self.assertIn("https://photoplanner.app/guides/mt-tamalpais-east-peak", locs)
        self.assertIn("https://photoplanner.app/guides/half-dome-yosemite", locs)
        self.assertGreaterEqual(len(locs), 43)

        robots = render_robots_txt("https://photoplanner.app")
        self.assertIn("Sitemap: https://photoplanner.app/sitemap.xml", robots)
        self.assertIn("Allow: /guides/", robots)

    def test_asgi_guides_sitemap_robots_and_b2b_api_key_gating(self) -> None:
        # 1. HTML directory & single spot guide (offline ephemeris mode for fast deterministic test)
        st_dir, _, body_dir = _call_asgi("GET", "/guides?region=bay_area&date=2025-06-14")
        self.assertEqual(st_dir, 200)
        dir_html = body_dir.decode("utf-8")
        self.assertIn("Mt. Tamalpais East Peak", dir_html)
        self.assertIn("/guides/mt-tamalpais-east-peak", dir_html)

        st_spot, hdrs_spot, body_spot = _call_asgi(
            "GET", "/guides/mt-tamalpais-east-peak?date=2025-06-14&phase=sunset&live=false"
        )
        self.assertEqual(st_spot, 200)
        self.assertIn("cache-control", hdrs_spot)
        self.assertIn("Mt. Tamalpais East Peak", body_spot.decode("utf-8"))

        # Second identical request hits TTL cache
        st_spot2, hdrs_spot2, _ = _call_asgi(
            "GET", "/guides/mt-tamalpais-east-peak?date=2025-06-14&phase=sunset&live=false"
        )
        self.assertEqual(st_spot2, 200)
        self.assertEqual(hdrs_spot2.get("x-photoplanner-cache"), "HIT")

        # Unknown slug returns 404
        st_404, _, _ = _call_asgi("GET", "/guides/does-not-exist-peak?live=false")
        self.assertEqual(st_404, 404)

        # Sitemap & robots served over ASGI
        st_sm, hdrs_sm, body_sm = _call_asgi("GET", "/sitemap.xml")
        self.assertEqual(st_sm, 200)
        self.assertIn("xml", hdrs_sm.get("content-type", ""))
        self.assertIn(b"<urlset", body_sm)

        st_rb, _, body_rb = _call_asgi("GET", "/robots.txt")
        self.assertEqual(st_rb, 200)
        self.assertIn(b"Sitemap:", body_rb)

        # B2B /api/v1/guides works open when env var unset, and enforces 401 when PHOTOPLANNER_B2B_API_KEYS is set
        orig_keys = os.environ.pop("PHOTOPLANNER_B2B_API_KEYS", None)
        try:
            st_open, hdrs_open, body_open = _call_asgi("GET", "/api/v1/guides?region=sierra")
            self.assertEqual(st_open, 200)
            self.assertEqual(hdrs_open.get("x-photoplanner-tier"), "open_preview")
            self.assertGreaterEqual(json.loads(body_open)["count"], 5)

            os.environ["PHOTOPLANNER_B2B_API_KEYS"] = "workshop_secret_123,partner_xyz"
            st_unauth, _, _ = _call_asgi("GET", "/api/v1/guides/half-dome-yosemite?live=false")
            self.assertEqual(st_unauth, 401)

            st_auth, hdrs_auth, body_auth = _call_asgi(
                "GET",
                "/api/v1/guides/half-dome-yosemite?date=2025-06-14&live=false",
                headers=[(b"x-api-key", b"workshop_secret_123")],
            )
            self.assertEqual(st_auth, 200)
            self.assertEqual(hdrs_auth.get("x-photoplanner-tier"), "b2b_licensed")
            self.assertEqual(json.loads(body_auth)["spot"]["slug"], "half-dome-yosemite")
        finally:
            if orig_keys is None:
                os.environ.pop("PHOTOPLANNER_B2B_API_KEYS", None)
            else:
                os.environ["PHOTOPLANNER_B2B_API_KEYS"] = orig_keys

    def test_static_seo_generator_cli_builds_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "seo_out"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_seo_guides.py",
                    "--out-dir",
                    str(out),
                    "--region",
                    "central_coast",
                    "--date",
                    "2025-06-14",
                    "--base-url",
                    "https://photoplanner.app",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue((out / "sitemap.xml").is_file())
            self.assertTrue((out / "robots.txt").is_file())
            self.assertTrue((out / "guides" / "index.html").is_file())
            self.assertTrue((out / "guides" / "garrapata-bluffs" / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
