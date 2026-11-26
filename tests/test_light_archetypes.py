"""Unit & integration tests for the LightFinder rebrand and 7-archetype hourly light engine."""

from __future__ import annotations

import json
from pathlib import Path
import random
import unittest
from unittest import mock

from planner.seo_guides import build_spot_guide_payload, render_guides_directory_html, render_spot_guide_html
from planner.weather.corridor_solver import solve_weather_and_corridor
from planner.weather.light_archetypes import (
    ARCHETYPE_CATALOG,
    ARCHETYPE_ORDER,
    VIABLE_THRESHOLD_PCT,
    compute_light_archetypes,
)
from tests.test_api_server import _call_asgi


ROOT = Path(__file__).resolve().parents[1]


class TestSevenArchetypeLightEngine(unittest.TestCase):
    def test_catalog_contains_all_7_archetypes_and_field_reference(self) -> None:
        expected_ids = {
            "giant_softbox",
            "clearing_storm",
            "atmospheric_diffusion",
            "high_thin_veil",
            "filtered_gold",
            "blue_hour",
            "harsh_sun",
        }
        self.assertEqual(set(ARCHETYPE_ORDER), expected_ids)
        self.assertEqual(set(ARCHETYPE_CATALOG.keys()), expected_ids)
        for aid in expected_ids:
            meta = ARCHETYPE_CATALOG[aid]
            fr = meta["field_reference"]
            for field in ("shadow_edge", "contrast", "color_cast", "best_for", "tactical_lever"):
                self.assertTrue(fr.get(field), msg=f"Missing {field} on {aid}")

    def test_all_7_archetypes_dominate_their_canonical_physical_regimes(self) -> None:
        regimes = [
            (
                "giant_softbox",
                dict(
                    sun_el_deg=28.0,
                    p10_pct=3,
                    p50_pct=8,
                    p90_pct=16,
                    cloud_low_pct=96,
                    cloud_mid_pct=88,
                    cloud_high_pct=65,
                    fog_top_ft=4500,
                    target_elev_ft=1500,
                    visibility_miles=12.0,
                ),
            ),
            (
                "clearing_storm",
                dict(
                    sun_el_deg=16.0,
                    p10_pct=12,
                    p50_pct=44,
                    p90_pct=86,
                    cloud_low_pct=58,
                    cloud_mid_pct=52,
                    cloud_high_pct=40,
                    canvas_glow_score=74,
                    aod_550nm=0.09,
                ),
            ),
            (
                "atmospheric_diffusion",
                dict(
                    sun_el_deg=14.0,
                    p10_pct=15,
                    p50_pct=38,
                    p90_pct=68,
                    cloud_low_pct=88,
                    cloud_mid_pct=10,
                    cloud_high_pct=10,
                    fog_pct=88,
                    fog_top_ft=1100,
                    target_elev_ft=1050,
                    visibility_miles=1.8,
                    rh_pct=95,
                    aod_550nm=0.28,
                    is_coastal=True,
                ),
            ),
            (
                "high_thin_veil",
                dict(
                    sun_el_deg=36.0,
                    p10_pct=45,
                    p50_pct=60,
                    p90_pct=74,
                    cloud_low_pct=8,
                    cloud_mid_pct=18,
                    cloud_high_pct=78,
                    high_optical_depth=1.1,
                    aod_550nm=0.09,
                ),
            ),
            (
                "filtered_gold",
                dict(
                    sun_el_deg=4.0,
                    p10_pct=55,
                    p50_pct=74,
                    p90_pct=88,
                    cloud_low_pct=10,
                    cloud_mid_pct=10,
                    cloud_high_pct=45,
                    canvas_glow_score=82,
                    aod_550nm=0.08,
                ),
            ),
            (
                "blue_hour",
                dict(
                    sun_el_deg=-5.5,
                    p10_pct=0,
                    p50_pct=0,
                    p90_pct=0,
                    cloud_low_pct=10,
                    cloud_mid_pct=15,
                    cloud_high_pct=35,
                    canvas_glow_score=60,
                    aod_550nm=0.08,
                ),
            ),
            (
                "harsh_sun",
                dict(
                    sun_el_deg=48.0,
                    p10_pct=82,
                    p50_pct=90,
                    p90_pct=96,
                    cloud_low_pct=5,
                    cloud_mid_pct=5,
                    cloud_high_pct=5,
                    aod_550nm=0.06,
                    visibility_miles=22.0,
                ),
            ),
        ]

        for expected_dom, kwargs in regimes:
            res = compute_light_archetypes(**kwargs)
            self.assertEqual(res["dominant_id"], expected_dom, msg=f"Got {res['probabilities']}")
            self.assertEqual(sum(res["probabilities"].values()), 100)
            self.assertIn(expected_dom, res["viable_ids"])
            self.assertGreaterEqual(len(res["viable_ids"]), 1)

    def test_strict_sum_100_and_viable_invariants_across_100_random_hours(self) -> None:
        rng = random.Random(20261125)
        for _ in range(100):
            p10 = rng.uniform(0, 90)
            p50 = rng.uniform(p10, 96)
            p90 = rng.uniform(p50, 100)
            res = compute_light_archetypes(
                sun_el_deg=rng.uniform(-30.0, 75.0),
                p10_pct=p10,
                p50_pct=p50,
                p90_pct=p90,
                cloud_low_pct=rng.uniform(0, 100),
                cloud_mid_pct=rng.uniform(0, 100),
                cloud_high_pct=rng.uniform(0, 100),
                fog_pct=rng.uniform(0, 100),
                fog_top_ft=rng.choice([0, 400, 1200, 2200, 4000]),
                target_elev_ft=rng.uniform(200, 9000),
                canvas_glow_score=rng.uniform(0, 100),
                aod_550nm=rng.uniform(0.02, 0.75),
                visibility_miles=rng.uniform(0.4, 30.0),
                rh_pct=rng.uniform(15, 100),
                is_coastal=rng.choice([True, False]),
            )
            probs = res["probabilities"]
            self.assertEqual(set(probs.keys()), set(ARCHETYPE_ORDER))
            self.assertEqual(sum(probs.values()), 100)
            for k, v in probs.items():
                self.assertIsInstance(v, int)
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 100)
            # Viable IDs contain every archetype >= 20% and always at least the dominant 1
            above_20 = [k for k in res["ranked_ids"] if probs[k] >= VIABLE_THRESHOLD_PCT]
            if above_20:
                self.assertEqual(res["viable_ids"], above_20)
            else:
                self.assertEqual(res["viable_ids"], [res["dominant_id"]])

    def test_corridor_solver_emits_primary_and_24_hourly_archetypes(self) -> None:
        # Build a clean deterministic mock NWP + Rust raycast bundle to verify full hourly_timeline integration
        hours_24 = [f"2026-06-15T{h:02d}:00" for h in range(24)]
        mock_point_doc = {
            "elevation": 350.0,
            "utc_offset_seconds": -28800,
            "hourly": {
                "time": hours_24,
                "temperature_2m": [14.0] * 24,
                "relative_humidity_2m": [72.0] * 24,
                "dew_point_2m": [9.0] * 24,
                "cloud_cover": [45.0] * 24,
                "cloud_cover_low": [30.0] * 24,
                "cloud_cover_mid": [20.0] * 24,
                "cloud_cover_high": [45.0] * 24,
                "visibility": [18000.0] * 24,
                "wind_speed_10m": [14.0] * 24,
                "wind_direction_10m": [290.0] * 24,
                "precipitation": [0.0] * 24,
                "cape": [120.0] * 24,
                "boundary_layer_height": [600.0] * 24,
                "temperature_1000hPa": [14.0] * 24,
                "temperature_975hPa": [13.5] * 24,
                "temperature_950hPa": [15.0] * 24,
                "temperature_925hPa": [16.0] * 24,
                "temperature_850hPa": [12.0] * 24,
                "temperature_700hPa": [2.0] * 24,
                "relative_humidity_975hPa": [78.0] * 24,
                "relative_humidity_950hPa": [68.0] * 24,
                "relative_humidity_925hPa": [45.0] * 24,
                "relative_humidity_850hPa": [35.0] * 24,
                "wind_speed_850hPa": [20.0] * 24,
                "wind_direction_850hPa": [285.0] * 24,
                "wind_speed_700hPa": [32.0] * 24,
                "wind_direction_700hPa": [280.0] * 24,
                "wind_speed_500hPa": [48.0] * 24,
                "wind_direction_500hPa": [275.0] * 24,
                "wind_speed_300hPa": [72.0] * 24,
                "wind_direction_300hPa": [270.0] * 24,
                "wind_speed_250hPa": [84.0] * 24,
                "wind_direction_250hPa": [270.0] * 24,
            },
        }
        coords_meta = [
            (f"p{i}", 37.5945, -122.5028 - i * 0.05, float(i * 15), 259.0)
            for i in range(13)
        ]
        mock_nwp = {
            "ok": True,
            "errors": [],
            "is_historical": False,
            "source_label": "Mock NWP Ensemble",
            "has_live_aq": True,
            "point_docs": [mock_point_doc for _ in range(13)],
            "coords_meta": coords_meta,
            "multimodel_doc": {},
            "aod_0km_24h": [0.09] * 24,
            "aod_100km_24h": [0.09] * 24,
            "angstrom_24h": [1.3] * 24,
            "ozone_du_24h": [300.0] * 24,
            "pm25_24h": [6.0] * 24,
            "dust_24h": [2.0] * 24,
        }

        def _fake_rust(lat, lon, target_elev_m, sun_az_deg, sun_el_deg, batch_input, skip_lut=False):
            hourly_res = []
            for ev in batch_input["hourly_evals"]:
                el = float(ev["sun_el_deg"])
                t_exp = 0.65 if el > 0 else 0.0
                hourly_res.append(
                    {
                        "hour_index": ev["hour_index"],
                        "time_local": ev["time_local"],
                        "sun_az_deg": ev["sun_az_deg"],
                        "sun_el_deg": el,
                        "ray": {
                            "clear_sky_rgb": [1.0, 1.0, 1.0],
                            "transmittance_expected": t_exp,
                            "transmittance_p10": max(0.0, t_exp - 0.25),
                            "transmittance_p50": t_exp,
                            "transmittance_p90": min(0.95, t_exp + 0.22),
                            "direct_beam_probability": t_exp,
                            "terrain_disk_visibility": 1.0,
                            "beam_color_kelvin": 3400.0 if 0 <= el <= 10 else 5400.0,
                            "chappuis_rose_index": 0.2,
                            "dominant_blocker": {"blocker_type": "clear_corridor", "distance_km": 0.0, "altitude_asl_m": 0.0},
                        },
                        "overhead_canvas": {"composite_canvas_score": 68.0, "dominant_glow_band": "high"},
                    }
                )
            return {
                "compute_backend": "mock_rust",
                "srtm_tiles_loaded": 0,
                "timing_ms": 1.0,
                "primary_ray": hourly_res[19]["ray"],
                "overhead_canvas": {"composite_canvas_score": 72.0, "dominant_glow_band": "high"},
                "corridor_lut": {"az_bins": 180, "alt_bins": 32, "rgba8_b64": ""},
                "hourly_results": hourly_res,
            }

        with (
            mock.patch("planner.weather.corridor_solver.fetch_24h_corridor_and_models", return_value=mock_nwp),
            mock.patch("planner.weather.corridor_solver._invoke_rust_corridor_raycast", side_effect=_fake_rust),
            mock.patch("planner.weather.corridor_solver.fetch_hrrr_s3_corridor", return_value=None),
            mock.patch("planner.weather.corridor_solver.fetch_nearest_station_observation", return_value=None),
            mock.patch("planner.weather.corridor_solver.fetch_goes_corridor_nowcast", return_value=None),
            mock.patch("planner.weather.corridor_solver.fetch_raob_sounding", return_value=None),
        ):
            raw = solve_weather_and_corridor(
                lat=37.5945,
                lon=-122.5028,
                date_str="2026-06-15",
                phase="sunset",
                sun_az_deg=298.0,
                sun_el_deg=3.5,
                target_elev_m=190.0,
                replay_mode="nwp_only",
                enable_multilayer_goes=False,
                enable_raob=False,
                enable_hrrr_s3=False,
                include_lut=False,
            )

        doc = json.loads(raw)
        self.assertIn("light_archetypes", doc)
        self.assertIn("light_archetype_catalog", doc)
        self.assertEqual(set(doc["light_archetype_catalog"].keys()), set(ARCHETYPE_ORDER))
        self.assertEqual(sum(doc["light_archetypes"]["probabilities"].values()), 100)
        self.assertGreaterEqual(len(doc["light_archetypes"]["viable_ids"]), 1)

        timeline = doc["hourly_timeline"]
        self.assertEqual(len(timeline), 24)
        for hr_idx, row in enumerate(timeline):
            self.assertEqual(row["hour"], hr_idx)
            la = row["light_archetypes"]
            self.assertEqual(sum(la["probabilities"].values()), 100)
            self.assertIn(la["dominant_id"], ARCHETYPE_ORDER)
            self.assertGreaterEqual(len(la["viable_ids"]), 1)

        # Midday vs night should reflect solar physics (daylight archetype vs blue_hour at night)
        self.assertEqual(timeline[2]["light_archetypes"]["dominant_id"], "blue_hour")
        self.assertNotEqual(timeline[13]["light_archetypes"]["dominant_id"], "blue_hour")


        # Verify all 5 exact solar-ephemeris-anchored Light Chapters & sun_events
        chapters = doc.get("light_chapters")
        self.assertIsInstance(chapters, list)
        self.assertEqual(
            [c["id"] for c in chapters],
            ["dawn_blue", "morning_gold", "midday", "evening_gold", "dusk_blue"],
        )
        prev_peak = -1
        for ch in chapters:
            self.assertGreater(ch["peak_minutes"], prev_peak)
            prev_peak = ch["peak_minutes"]
            self.assertLessEqual(ch["start_minutes"], ch["peak_minutes"])
            self.assertLessEqual(ch["peak_minutes"], ch["end_minutes"])
            ch_probs = ch["light_archetypes"]["probabilities"]
            self.assertEqual(set(ch_probs.keys()), set(ARCHETYPE_ORDER))
            self.assertEqual(sum(ch_probs.values()), 100)

        # Under the same evening weather, Evening Gold and Dusk Blue separate Filtered Gold vs Blue Hour
        eve_gold = next(c for c in chapters if c["id"] == "evening_gold")
        dusk_blue = next(c for c in chapters if c["id"] == "dusk_blue")
        self.assertGreater(eve_gold["sun_el_deg"], -1.5)
        self.assertLess(dusk_blue["sun_el_deg"], -2.0)
        self.assertEqual(dusk_blue["light_archetypes"]["dominant_id"], "blue_hour")

        sun_events = doc.get("sun_events") or {}
        self.assertLess(sun_events["dawn_blue_min"], sun_events["sunrise_min"])
        self.assertLess(sun_events["sunrise_min"], sun_events["solar_noon_min"])
        self.assertLess(sun_events["solar_noon_min"], sun_events["sunset_min"])
        self.assertLess(sun_events["sunset_min"], sun_events["dusk_blue_min"])

        # A sunset Blue Hour alert catches the exact dusk_blue chapter even if golden hour is Filtered Gold
        from planner.weather.alert_engine import evaluate_alert_gate

        blue_rule = {
            "id": "wl_blue",
            "name": "Pedro Point Dusk Blue",
            "phase": "sunset",
            "target_archetypes": ["blue_hour"],
            "min_archetype_prob_pct": 50.0,
        }
        ev_blue = evaluate_alert_gate(
            blue_rule,
            {
                "spot_id": "pedro_point",
                "name": "Pedro Point Headlands",
                "eval_phase": "sunset",
                "light_archetypes": doc["light_archetypes"],
                "light_chapters": chapters,
                "hourly_timeline": [],
                "metrics": {"calibrated_p10_pct": 30.0, "calibrated_p50_pct": 65.0, "calibrated_p90_pct": 85.0},
            },
        )
        self.assertEqual(ev_blue["decision"], "TRIGGER")
        self.assertEqual(ev_blue["matched_archetype_id"], "blue_hour")
        self.assertEqual(ev_blue["best_time_local"], dusk_blue["peak_local"])


class TestLightFinderRebrand(unittest.TestCase):
    def test_user_facing_rebrand_across_html_og_seo_and_field_card(self) -> None:
        index_html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("<title>LightFinder — 3D Light &amp; Lens Viewfinder</title>", index_html)
        self.assertIn('<span class="brand-wordmark">LightFinder</span>', index_html)
        self.assertNotIn("PhotoPlanner", index_html)

        # Dynamic OpenGraph share link uses LightFinder
        st, _, body = _call_asgi("GET", "/?name=Montara+Mountain&date=2026-11-25&phase=sunset&mm=35")
        self.assertEqual(st, 200)
        og_html = body.decode("utf-8")
        self.assertIn("| LightFinder", og_html)
        self.assertNotIn("| PhotoPlanner", og_html)

        # SEO guides use LightFinder
        guide_payload = build_spot_guide_payload(
            "mt-tamalpais-east-peak",
            date_str="2026-06-14",
            phase="sunset",
            include_live_forecast=False,
        )
        self.assertIn("| LightFinder", guide_payload["seo"]["title"])
        guide_html = render_spot_guide_html(guide_payload)
        self.assertIn("LightFinder 3D", guide_html)
        self.assertNotIn("PhotoPlanner", guide_html)

        dir_html = render_guides_directory_html(date_str="2026-06-14")
        self.assertIn("LightFinder 3D", dir_html)
        self.assertNotIn("PhotoPlanner", dir_html)

        # Offline Field Card in web/app.js uses LightFinder
        app_js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn("<title>LightFinder Field Card —", app_js)
        self.assertIn("LightFinder_FieldCard_", app_js)
        self.assertIn("Generated offline by LightFinder 3D", app_js)
        self.assertNotIn("PhotoPlanner Field Card", app_js)
        self.assertNotIn("PhotoPlanner_FieldCard_", app_js)


class TestLightFinderArchetypeAlerts(unittest.TestCase):
    def test_archetype_watchlist_crud_validation_and_union_gating(self) -> None:
        import tempfile
        from planner.weather import alert_engine

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "wl_arch.sqlite3"
            with self.assertRaises(ValueError):
                alert_engine.create_watchlist_rule(
                    name="Bad Archetype",
                    lat=37.59,
                    lon=-122.50,
                    target_archetypes=["not_a_real_archetype"],
                    db_path=db,
                )

            # Single non-golden-hour archetype: Giant Softbox should TRIGGER even with low direct beam P50!
            softbox_rule = alert_engine.create_watchlist_rule(
                name="Redwood Creek Softbox",
                lat=37.89,
                lon=-122.57,
                phase="any",
                target_archetypes=["giant_softbox"],
                min_archetype_prob_pct=50.0,
                db_path=db,
            )
            self.assertEqual(softbox_rule["phase"], "any")
            self.assertEqual(softbox_rule["target_archetypes"], ["giant_softbox"])
            self.assertEqual(softbox_rule["min_archetype_prob_pct"], 50.0)

            overcast_spot = {
                "spot_id": "redwood",
                "name": "Redwood Creek",
                "opportunity_score": 18,
                "primary_phenomenon": "Overcast Stratus",
                "phenomena": [],
                "causal_attribution": "Thick marine stratus blanket.",
                "metrics": {
                    "calibrated_p10_pct": 2.0,
                    "calibrated_p50_pct": 8.0,
                    "calibrated_p90_pct": 18.0,
                    "summit_above_fog_prob_pct": 10.0,
                },
                "light_archetypes": {
                    "probabilities": {
                        "giant_softbox": 74,
                        "clearing_storm": 6,
                        "atmospheric_diffusion": 12,
                        "high_thin_veil": 8,
                        "filtered_gold": 0,
                        "blue_hour": 0,
                        "harsh_sun": 0,
                    },
                    "dominant": {"id": "giant_softbox", "name": "Giant Softbox", "probability_pct": 74},
                },
            }
            ev_soft = alert_engine.evaluate_alert_gate(softbox_rule, overcast_spot)
            self.assertEqual(ev_soft["decision"], "TRIGGER")
            self.assertEqual(ev_soft["matched_archetype_id"], "giant_softbox")
            self.assertEqual(ev_soft["combined_target_prob_pct"], 74.0)
            self.assertIn("Exclude sky", ev_soft["tactical_lever"])

            # Mutually-exclusive probability union: Filtered Gold (32%) + Clearing Storm (26%) = 58% >= 50% -> TRIGGER
            combo_rule = alert_engine.create_watchlist_rule(
                name="Coastal Drama Combo",
                lat=37.59,
                lon=-122.50,
                phase="sunset",
                target_archetypes=["filtered_gold", "clearing_storm"],
                min_archetype_prob_pct=50.0,
                db_path=db,
            )
            split_spot = {
                **overcast_spot,
                "name": "Pedro Point Headlands",
                "light_archetypes": {
                    "probabilities": {
                        "giant_softbox": 10,
                        "clearing_storm": 26,
                        "atmospheric_diffusion": 18,
                        "high_thin_veil": 14,
                        "filtered_gold": 32,
                        "blue_hour": 0,
                        "harsh_sun": 0,
                    },
                    "dominant": {"id": "filtered_gold", "name": "Filtered Gold", "probability_pct": 32},
                },
            }
            ev_combo = alert_engine.evaluate_alert_gate(combo_rule, split_spot)
            self.assertEqual(ev_combo["decision"], "TRIGGER")
            self.assertEqual(ev_combo["combined_target_prob_pct"], 58.0)
            self.assertEqual(ev_combo["matched_archetype_id"], "filtered_gold")

            # WATCH tier: 42% (within 15% of 50% threshold and >= 20%)
            watch_spot = {
                **split_spot,
                "light_archetypes": {
                    "probabilities": {
                        "giant_softbox": 44,
                        "clearing_storm": 20,
                        "atmospheric_diffusion": 10,
                        "high_thin_veil": 4,
                        "filtered_gold": 22,
                        "blue_hour": 0,
                        "harsh_sun": 0,
                    },
                    "dominant": {"id": "giant_softbox", "name": "Giant Softbox", "probability_pct": 44},
                },
            }
            ev_watch = alert_engine.evaluate_alert_gate(combo_rule, watch_spot)
            self.assertEqual(ev_watch["decision"], "WATCH")
            self.assertEqual(ev_watch["combined_target_prob_pct"], 42.0)

            # SUPPRESS tier: 12% vs 50% threshold names dominant archetype
            ev_sup = alert_engine.evaluate_alert_gate(combo_rule, overcast_spot)
            self.assertEqual(ev_sup["decision"], "SUPPRESS")
            self.assertIn("Giant Softbox", ev_sup["reason"])

    def test_24h_any_hour_scan_vs_sunset_window_and_delivery_message(self) -> None:
        from planner.weather.alert_delivery import build_alert_message
        from planner.weather.alert_engine import evaluate_alert_gate

        midday_storm = {
            "hour": 13,
            "minutes_local": 810,
            "time_local": "13:30",
            "light_archetypes": {
                "probabilities": {
                    "giant_softbox": 8,
                    "clearing_storm": 68,
                    "atmospheric_diffusion": 14,
                    "high_thin_veil": 6,
                    "filtered_gold": 0,
                    "blue_hour": 0,
                    "harsh_sun": 4,
                },
                "dominant": {"id": "clearing_storm", "name": "Clearing Storm", "probability_pct": 68},
            },
        }
        sunset_quiet = {
            "hour": 17,
            "minutes_local": 1020,
            "time_local": "17:00",
            "light_archetypes": {
                "probabilities": {
                    "giant_softbox": 72,
                    "clearing_storm": 8,
                    "atmospheric_diffusion": 14,
                    "high_thin_veil": 6,
                    "filtered_gold": 0,
                    "blue_hour": 0,
                    "harsh_sun": 0,
                },
                "dominant": {"id": "giant_softbox", "name": "Giant Softbox", "probability_pct": 72},
            },
        }
        spot = {
            "spot_id": "pedro_point",
            "name": "Pedro Point Headlands",
            "opportunity_score": 45,
            "primary_phenomenon": "Storm Breaks",
            "phenomena": [],
            "causal_attribution": "Post-frontal instability.",
            "phase_minutes_local": 1020,
            "metrics": {
                "calibrated_p10_pct": 25.0,
                "calibrated_p50_pct": 48.0,
                "calibrated_p90_pct": 78.0,
                "summit_above_fog_prob_pct": 85.0,
            },
            "light_archetypes": sunset_quiet["light_archetypes"],
            "hourly_timeline": [midday_storm, sunset_quiet],
        }
        rule_sunset = {
            "id": "wl_ss",
            "name": "Pedro Point Headlands",
            "lat": 37.5945,
            "lon": -122.5028,
            "phase": "sunset",
            "target_archetypes": ["clearing_storm"],
            "min_archetype_prob_pct": 50.0,
        }
        rule_any = {**rule_sunset, "id": "wl_any", "phase": "any"}

        ev_ss = evaluate_alert_gate(rule_sunset, {**spot, "eval_phase": "sunset"})
        ev_any = evaluate_alert_gate(rule_any, {**spot, "eval_phase": "any"})
        self.assertEqual(ev_ss["decision"], "SUPPRESS")
        self.assertEqual(ev_any["decision"], "TRIGGER")
        self.assertEqual(ev_any["best_hour"], 13)
        self.assertEqual(ev_any["best_time_local"], "13:30")

        msg = build_alert_message(
            rule_any,
            {**ev_any, "phase": "any"},
            date_str="2026-11-25",
            base_url="https://photoplanner.app",
        )
        self.assertIn("[TRIGGER]", msg["subject"])
        self.assertIn("68% Clearing Storm", msg["subject"])
        self.assertIn("LightFinder", msg["text"])
        self.assertIn("LightFinder", msg["html"])
        self.assertIn("Expose for sunlit highlights", msg["html"])
        self.assertIn("archetype=clearing_storm", msg["deep_link"])
        self.assertIn("hour=13", msg["deep_link"])

    def test_asgi_watchlist_archetype_roundtrip(self) -> None:
        import tempfile
        from planner.weather import alert_engine

        orig_db = alert_engine.DEFAULT_WATCHLIST_DB
        with tempfile.TemporaryDirectory() as td:
            alert_engine.DEFAULT_WATCHLIST_DB = Path(td) / "asgi_arch_wl.sqlite3"
            try:
                bad_payload = json.dumps(
                    {
                        "name": "Invalid Test",
                        "lat": 37.59,
                        "lon": -122.50,
                        "target_archetypes": ["invalid_light"],
                    }
                ).encode("utf-8")
                st_bad, _, _ = _call_asgi("POST", "/api/watchlists", body=bad_payload)
                self.assertEqual(st_bad, 400)

                good_payload = json.dumps(
                    {
                        "name": "Pedro Point Storm Watch",
                        "lat": 37.5945,
                        "lon": -122.5028,
                        "phase": "any",
                        "target_archetypes": ["clearing_storm", "filtered_gold"],
                        "min_archetype_prob_pct": 50.0,
                        "notify_webhook_url": "https://ntfy.sh/lightfinder-test",
                    }
                ).encode("utf-8")
                st_c, _, body_c = _call_asgi("POST", "/api/watchlists", body=good_payload)
                self.assertEqual(st_c, 201)
                created = json.loads(body_c)
                self.assertEqual(created["phase"], "any")
                self.assertEqual(created["target_archetypes"], ["clearing_storm", "filtered_gold"])
                self.assertEqual(created["min_archetype_prob_pct"], 50.0)

                st_l, _, body_l = _call_asgi("GET", "/api/watchlists")
                self.assertEqual(st_l, 200)
                rules = json.loads(body_l)["watchlists"]
                self.assertEqual(len(rules), 1)
                self.assertEqual(rules[0]["target_archetypes"], ["clearing_storm", "filtered_gold"])
            finally:
                alert_engine.DEFAULT_WATCHLIST_DB = orig_db


if __name__ == "__main__":
    unittest.main()
