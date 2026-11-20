"""
Unit tests for Roadmap Task 3.3: Custom GPX / GeoJSON Trail Upload Analyzer
(`planner/gpx_trail_analyzer.py`).
"""

from __future__ import annotations

import unittest

from planner.gpx_trail_analyzer import (
    analyze_uploaded_trail,
    build_trail_nudges_from_analysis,
    compute_tobler_trail_profile,
    parse_geojson_trail,
    parse_gpx_xml,
)

SAMPLE_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="PhotoPlannerTest" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Mt. Tamalpais Dipsea to East Peak</name>
    <trkseg>
      <trkpt lat="37.9010" lon="-122.6200"><ele>150.0</ele></trkpt>
      <trkpt lat="37.9060" lon="-122.6150"><ele>310.0</ele></trkpt>
      <trkpt lat="37.9110" lon="-122.6100"><ele>480.0</ele></trkpt>
      <trkpt lat="37.9160" lon="-122.6040"><ele>640.0</ele></trkpt>
      <trkpt lat="37.9235" lon="-122.5965"><ele>784.0</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""


class TestGpxTrailUploadAnalyzer(unittest.TestCase):
    def test_parse_gpx_xml_extracts_name_and_3d_coordinates(self) -> None:
        parsed = parse_gpx_xml(SAMPLE_GPX)
        self.assertEqual(parsed["name"], "Mt. Tamalpais Dipsea to East Peak")
        self.assertEqual(len(parsed["coords"]), 5)
        self.assertAlmostEqual(parsed["coords"][-1][2], 784.0, places=1)

    def test_parse_geojson_feature_collection(self) -> None:
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Half Dome Cables Approach"},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [-119.558, 37.732, 1220.0],
                            [-119.545, 37.739, 1950.0],
                            [-119.533, 37.746, 2693.0],
                        ],
                    },
                }
            ],
        }
        parsed = parse_geojson_trail(fc)
        self.assertEqual(parsed["name"], "Half Dome Cables Approach")
        self.assertEqual(len(parsed["coords"]), 3)
        self.assertAlmostEqual(parsed["coords"][-1][2], 2693.0, places=1)

    def test_tobler_profile_and_loop_vs_oneway_detection(self) -> None:
        parsed = parse_gpx_xml(SAMPLE_GPX)
        prof = compute_tobler_trail_profile(parsed["coords"])
        self.assertEqual(prof["topology"], "one_way_or_out_and_back")
        self.assertGreater(prof["total_distance_miles"], 1.5)
        self.assertGreater(prof["elevation_gain_ft"], 1900)
        self.assertGreater(prof["tobler_ascent_to_apex_min"], prof["tobler_return_from_apex_min"])
        self.assertAlmostEqual(prof["apex"]["elev_m"], 784.0, places=1)

        # True circular loop should be detected as "loop"
        loop_coords = [
            [-122.600, 37.900, 200.0],
            [-122.590, 37.905, 420.0],
            [-122.585, 37.915, 650.0],
            [-122.595, 37.920, 500.0],
            [-122.605, 37.910, 350.0],
            [-122.600, 37.900, 200.0],
        ]
        loop_prof = compute_tobler_trail_profile(loop_coords)
        self.assertEqual(loop_prof["topology"], "loop")

    def test_analyze_uploaded_trail_offline_weather_and_nudges(self) -> None:
        res = analyze_uploaded_trail(
            gpx_xml=SAMPLE_GPX,
            date_str="2025-06-05",
            phase="sunset",
            setup_buffer_min=25,
            include_weather=False,
        )
        self.assertEqual(res["schema_version"], "trail_upload_v1")
        self.assertEqual(res["trail_name"], "Mt. Tamalpais Dipsea to East Peak")
        self.assertTrue(res["logistics"]["headlamp_required"])
        cats = {n["category"] for n in res["nudges"]}
        self.assertIn("HeadlampSafety", cats)
        self.assertIn("LightNotchWindow", cats)

    def test_build_nudges_includes_fog_breakout_when_inversion_crosses_trail(self) -> None:
        from planner.weather.terrain_coupling import compute_trail_fog_breakout_mileposts

        parsed = parse_gpx_xml(SAMPLE_GPX)
        prof = compute_tobler_trail_profile(parsed["coords"])
        inv_model = {
            "fog_top_p10_ft": 1150,
            "fog_top_p50_ft": 1520,
            "fog_top_p90_ft": 1820,
            "summit_above_fog_prob_pct": 91.0,
            "breakout_verdict": "ABOVE_CLOUD_SEA",
        }
        breakout = compute_trail_fog_breakout_mileposts(parsed["coords"], inv_model)
        w_doc = {
            "corridor_direction": "WNW",
            "alpenglow_score": 84,
            "sun_elevation_deg": 1.1,
            "corridor_raycast": {"expected_transmittance_pct": 78.0, "terrain_horizon_deg": 0.2},
            "calibrated_probabilities": {"transmittance_p10_pct": 58.0, "transmittance_p50_pct": 76.0, "transmittance_p90_pct": 92.0},
            "inversion_model": inv_model,
        }
        nudges, logistics = build_trail_nudges_from_analysis(
            profile=prof, breakout=breakout, weather_doc=w_doc, phase="sunset", event_mins_local=1215, setup_buffer_min=25
        )
        cats = {n["category"] for n in nudges}
        self.assertIn("FogBreakout", cats)
        self.assertIn("LightNotchWindow", cats)
        self.assertIn("HeadlampSafety", cats)
        self.assertEqual(logistics["golden_hour_anchor_local"], "20:15")


class TestCalibratedAlertEngine(unittest.TestCase):
    def test_watchlist_sqlite_crud_and_trigger_watch_suppress_gating(self) -> None:
        import tempfile
        from pathlib import Path

        from planner.weather.alert_engine import (
            create_watchlist_rule,
            delete_watchlist_rule,
            evaluate_alert_gate,
            list_watchlist_rules,
        )

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "wl.sqlite3"
            rule = create_watchlist_rule(
                name="Mt. Tam Cloud-Sea Breakout",
                spot_id="mt_tam_east",
                phase="sunset",
                target_phenomena=["Cloud-Sea Breakout"],
                min_calibrated_p50_pct=55.0,
                min_above_fog_prob_pct=70.0,
                min_opportunity_score=68.0,
                db_path=db,
            )
            self.assertEqual(len(list_watchlist_rules(db_path=db)), 1)

            go_spot = {
                "spot_id": "mt_tam_east",
                "name": "Mt. Tamalpais East Peak",
                "opportunity_score": 84,
                "primary_phenomenon": "Cloud-Sea Breakout",
                "phenomena": [{"tag": "Cloud-Sea Breakout", "score": 88}],
                "causal_attribution": "Summit cleanly above fog lid.",
                "metrics": {
                    "calibrated_p10_pct": 48.0,
                    "calibrated_p50_pct": 72.0,
                    "calibrated_p90_pct": 91.0,
                    "summit_above_fog_prob_pct": 89.0,
                },
            }
            watch_spot = {
                **go_spot,
                "opportunity_score": 60,
                "metrics": {
                    "calibrated_p10_pct": 22.0,
                    "calibrated_p50_pct": 44.0,
                    "calibrated_p90_pct": 74.0,
                    "summit_above_fog_prob_pct": 62.0,
                },
            }
            bust_spot = {
                **go_spot,
                "opportunity_score": 25,
                "phenomena": [],
                "causal_attribution": "BUST — Immersed inside marine stratus.",
                "metrics": {
                    "calibrated_p10_pct": 2.0,
                    "calibrated_p50_pct": 9.0,
                    "calibrated_p90_pct": 24.0,
                    "summit_above_fog_prob_pct": 12.0,
                },
            }
            self.assertEqual(evaluate_alert_gate(rule, go_spot)["decision"], "TRIGGER")
            self.assertEqual(evaluate_alert_gate(rule, watch_spot)["decision"], "WATCH")
            self.assertEqual(evaluate_alert_gate(rule, bust_spot)["decision"], "SUPPRESS")
            self.assertTrue(delete_watchlist_rule(rule["id"], db_path=db))
            self.assertEqual(len(list_watchlist_rules(db_path=db)), 0)


if __name__ == "__main__":
    unittest.main()
