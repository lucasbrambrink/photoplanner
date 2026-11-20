"""
Unit tests for NOAA SURFRAD ground truth parsing, terrain-coupled meteorology
(Task 2.1 Probabilistic Fog-Lid Bands & Trail Breakout Mileposts, Task 2.2 Lake Mirror Index),
Regional Light-Hunting & Causal Bust Attribution (Tasks 4.1 & 4.2), and Multi-Regime
Benchmark Scoring, Diagnostics & Paired Bootstrap Comparison.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from planner.weather.benchmark_suite import (
    build_stratified_events,
    compare_experiments,
    diagnose_failures,
    extract_golden_hour_prediction,
    generate_html_verification_report,
    score_predictions,
)
from planner.weather.regional_hunter import (
    REGIONAL_SPOT_CATALOG,
    classify_spot_phenomena_and_attribution,
)
from planner.weather.surfrad_truth import parse_surfrad_dat_lines
from planner.weather.terrain_coupling import (
    compute_lake_mirror_index,
    compute_probabilistic_fog_bands,
    compute_trail_fog_breakout_mileposts,
)


class TestSurfradParser(unittest.TestCase):
    def test_parse_surfrad_dat_filters_qc_and_extracts_direct_dni(self) -> None:
        # Synthetic 2-line SURFRAD header + 2 data lines (1 valid, 1 QC-flagged)
        def _row(hr: int, mn: int, zen: float, dni: float, qc_dni: int) -> str:
            cols = ["0.0"] * 48
            cols[0], cols[1], cols[2], cols[3], cols[4], cols[5] = "2025", "135", "5", "15", str(hr), str(mn)
            cols[7] = f"{zen:.2f}"
            cols[8], cols[9] = "120.0", "0"
            cols[12], cols[13] = f"{dni:.1f}", str(qc_dni)
            cols[14], cols[15] = "40.0", "0"
            cols[38], cols[39] = "12.5", "0"
            cols[40], cols[41] = "32.0", "0"
            cols[42], cols[43] = "4.2", "0"
            cols[46], cols[47] = "830.0", "0"
            return " ".join(cols)

        raw = "Table Mountain\n40.125 -105.237 1689 m version 1\n" + _row(12, 10, 84.0, 520.0, 0) + "\n" + _row(12, 11, 83.8, 999.0, 1) + "\n"
        parsed = parse_surfrad_dat_lines(raw.splitlines())
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["qc_direct"], 0)
        self.assertAlmostEqual(parsed[0]["direct_n_wm2"], 520.0, places=1)
        self.assertAlmostEqual(parsed[0]["el_deg"], 6.0, places=2)
        self.assertEqual(parsed[1]["qc_direct"], 1)


class TestTerrainCoupling(unittest.TestCase):
    def test_probabilistic_fog_bands_widen_with_lead_and_tighten_with_raob(self) -> None:
        b_raob_1h = compute_probabilistic_fog_bands(
            fog_top_ft=1500, fog_base_ft=600, target_elev_ft=2572,
            inversion_delta_t_f=8.0, lead_hours=1.0, raob_assimilated=True, is_coastal=True,
        )
        b_nwp_72h = compute_probabilistic_fog_bands(
            fog_top_ft=1500, fog_base_ft=600, target_elev_ft=2572,
            inversion_delta_t_f=3.0, lead_hours=72.0, raob_assimilated=False, is_coastal=True,
        )
        self.assertLess(b_raob_1h["fog_lid_sigma_ft"], b_nwp_72h["fog_lid_sigma_ft"])
        self.assertLess(b_raob_1h["fog_top_p10_ft"], b_raob_1h["fog_top_p50_ft"])
        self.assertLess(b_raob_1h["fog_top_p50_ft"], b_raob_1h["fog_top_p90_ft"])
        self.assertEqual(b_raob_1h["breakout_verdict"], "ABOVE_CLOUD_SEA")

    def test_trail_fog_breakout_mileposts_locates_p10_p50_p90_crossings(self) -> None:
        # Ascending 10-point trail from 300 m (~984 ft) to 900 m (~2953 ft)
        trail = [[-122.50 + i * 0.004, 37.90, 300.0 + i * 66.67] for i in range(10)]
        bands = {"fog_top_p10_ft": 1350, "fog_top_p50_ft": 1700, "fog_top_p90_ft": 2050}
        res = compute_trail_fog_breakout_mileposts(trail, bands)
        self.assertTrue(res["available"])
        self.assertIsNotNone(res["early_breakout_p10"])
        self.assertIsNotNone(res["expected_breakout_p50"])
        self.assertIsNotNone(res["confident_breakout_p90"])
        self.assertLess(res["early_breakout_p10"]["milepost"], res["expected_breakout_p50"]["milepost"])
        self.assertLess(res["expected_breakout_p50"]["milepost"], res["confident_breakout_p90"]["milepost"])
        self.assertGreater(res["above_expected_lid_pct_of_trail"], 40.0)

    def test_lake_mirror_index_sheltered_cirque_beats_exposed_plain(self) -> None:
        exposed = compute_lake_mirror_index(wind_speed_10m_mph=12.0, wind_dir_deg=270.0, upwind_horizon_deg=0.5)
        sheltered = compute_lake_mirror_index(wind_speed_10m_mph=12.0, wind_dir_deg=270.0, upwind_horizon_deg=22.0)
        self.assertGreater(sheltered["mirror_index"], exposed["mirror_index"] + 25)
        self.assertGreater(sheltered["shelter_reduction_pct"], 50.0)
        self.assertLess(sheltered["effective_sheltered_wind_mph"], exposed["effective_sheltered_wind_mph"])


class TestRegionalHunter(unittest.TestCase):
    def test_catalog_covers_multiple_regions_and_archetypes(self) -> None:
        regions = {s["region"] for s in REGIONAL_SPOT_CATALOG}
        self.assertTrue({"bay_area", "sierra", "rockies", "pnw", "southwest"}.issubset(regions))

    def test_causal_bust_attribution_for_fog_immersion_vs_cloud_sea_breakout(self) -> None:
        pedro = next(s for s in REGIONAL_SPOT_CATALOG if s["id"] == "pedro_point")
        tam = next(s for s in REGIONAL_SPOT_CATALOG if s["id"] == "mt_tam_east")

        fog_doc_pedro = {
            "corridor_direction": "W",
            "alpenglow_score": 25,
            "canvas_score": 20,
            "cloud_low_pct": 92,
            "cloud_mid_pct": 5,
            "cloud_high_pct": 10,
            "fog_top_ft": 1650,
            "sun_elevation_deg": 0.8,
            "wind_mph": 10,
            "wind_dir_deg": 280,
            "corridor_raycast": {"expected_transmittance_pct": 6.0, "dominant_blocker_label": "fog", "dominant_blocker_distance_km": 4.0, "terrain_horizon_deg": 0.0},
            "calibrated_probabilities": {"transmittance_p10_pct": 2.0, "transmittance_p50_pct": 8.0, "transmittance_p90_pct": 22.0, "summit_above_fog_prob_pct": 8.0},
            "inversion_model": {"fog_top_p50_ft": 1650, "fog_top_p90_ft": 1950, "summit_clearance_ft": -928, "summit_above_fog_prob_pct": 8.0, "cloud_sea_score": 24},
        }
        clear_doc_tam = {
            "corridor_direction": "W",
            "alpenglow_score": 88,
            "canvas_score": 76,
            "cloud_low_pct": 85,
            "cloud_mid_pct": 8,
            "cloud_high_pct": 38,
            "fog_top_ft": 1650,
            "sun_elevation_deg": 0.8,
            "wind_mph": 8,
            "wind_dir_deg": 280,
            "corridor_raycast": {"expected_transmittance_pct": 74.0, "dominant_blocker_label": "clear_corridor", "dominant_blocker_distance_km": 0.0, "terrain_horizon_deg": 0.0},
            "calibrated_probabilities": {"transmittance_p10_pct": 55.0, "transmittance_p50_pct": 78.0, "transmittance_p90_pct": 94.0, "summit_above_fog_prob_pct": 91.0},
            "inversion_model": {"fog_top_p50_ft": 1650, "fog_top_p90_ft": 1950, "summit_clearance_ft": 922, "summit_above_fog_prob_pct": 91.0, "cloud_sea_score": 90},
        }

        res_pedro = classify_spot_phenomena_and_attribution(pedro, fog_doc_pedro)
        res_tam = classify_spot_phenomena_and_attribution(tam, clear_doc_tam)
        self.assertEqual(res_pedro["verdict"], "BUST")
        self.assertIn("Immersed", res_pedro["causal_attribution"])
        self.assertEqual(res_tam["verdict"], "GO")
        self.assertEqual(res_tam["primary_phenomenon"], "Cloud-Sea Breakout")
        self.assertGreater(res_tam["opportunity_score"], res_pedro["opportunity_score"] + 45)


class TestBenchmarkSuite(unittest.TestCase):
    def test_stratified_suite_builders_and_sunrise_sunset_coverage(self) -> None:
        all_ev = build_stratified_events("all")
        self.assertGreaterEqual(len(all_ev), 80)
        phases = {e["phase"] for e in all_ev}
        self.assertEqual(phases, {"sunrise", "sunset"})
        scaled = build_stratified_events("scaled", limit=250)
        self.assertEqual(len(scaled), 250)

        xyear = build_stratified_events("cross_year")
        self.assertEqual(len(xyear), 504)
        self.assertEqual(len({e["id"] for e in xyear}), 504)
        years = {e["date"][:4] for e in xyear}
        self.assertEqual(years, {"2024", "2025"})
        surfrad_sts = {e["surfrad_station"] for e in xyear if e.get("surfrad_station")}
        self.assertEqual(len(surfrad_sts), 7)
        # Every 2024 date must be >= 2024-02-10 for genuine previous-runs-api coverage
        dates_2024 = [e["date"] for e in xyear if e["date"].startswith("2024-")]
        self.assertEqual(len(dates_2024), 240)
        self.assertTrue(all(d >= "2024-02-10" for d in dates_2024))

    def test_golden_hour_extraction_respects_sunrise_vs_sunset_phase(self) -> None:
        doc = {
            "hourly_timeline": [
                {"hour": 5, "sun_el_deg": 1.0, "expected_transmittance_pct": 30.0, "p10_pct": 15.0, "p90_pct": 55.0},
                {"hour": 6, "sun_el_deg": 6.5, "expected_transmittance_pct": 70.0, "p10_pct": 50.0, "p90_pct": 90.0},
                {"hour": 18, "sun_el_deg": 6.0, "expected_transmittance_pct": 15.0, "p10_pct": 5.0, "p90_pct": 30.0},
                {"hour": 19, "sun_el_deg": 0.8, "expected_transmittance_pct": 10.0, "p10_pct": 0.0, "p90_pct": 25.0},
            ],
            "corridor_raycast": {"expected_transmittance_pct": 10.0, "transmittance_p10_pct": 0.0, "transmittance_p90_pct": 25.0},
        }
        sr_exp, _, _ = extract_golden_hour_prediction(doc, phase="sunrise")
        ss_exp, _, _ = extract_golden_hour_prediction(doc, phase="sunset")
        self.assertGreater(sr_exp, 60.0)
        self.assertLess(ss_exp, 20.0)

    def test_scoring_diagnostics_paired_bootstrap_and_html_report(self) -> None:
        def _make_ev(idx: int, truth: float, pred: float) -> dict:
            return {
                "id": f"ev_{idx}",
                "site": f"Site {idx % 3}",
                "suite": "test",
                "region": "rockies",
                "subregion": "tbl",
                "season": "summer",
                "phase": "sunrise" if idx % 2 == 0 else "sunset",
                "date": f"2025-06-{idx + 1:02d}",
                "elev_m": 1689.0,
                "primary_truth_tier": "tier_1a_surfrad",
                "dni_golden_truth_pct": truth,
                "golden_hour_sun_hit": truth >= 35.0,
                "modes": {
                    "nowcast_1h": {
                        "golden_exp_pct": pred,
                        "golden_p10_pct": max(0.0, pred - 15.0),
                        "golden_p90_pct": min(100.0, pred + 15.0),
                        "sunset_exp_pct": pred,
                        "cloud_low_pct": 10,
                        "cloud_mid_pct": 10,
                        "cloud_high_pct": 15,
                        "blocker_label": "low" if pred < 40 else "clear_corridor",
                        "blocker_distance_km": 40.0,
                    }
                },
            }

        ev_a = [_make_ev(i, 80.0 if i % 2 == 0 else 15.0, 45.0 if i % 2 == 0 else 45.0) for i in range(12)]
        ev_b = [_make_ev(i, 80.0 if i % 2 == 0 else 15.0, 76.0 if i % 2 == 0 else 18.0) for i in range(12)]

        diag = diagnose_failures(ev_a, modes=("nowcast_1h",))
        self.assertEqual(diag["n_large_misses"], 12)

        with tempfile.TemporaryDirectory() as td:
            pa = Path(td) / "a.json"
            pb = Path(td) / "b.json"
            pa.write_text(json.dumps({"run_id": "base", "modes": ["nowcast_1h"], "events_detail": ev_a}), encoding="utf-8")
            pb.write_text(json.dumps({"run_id": "cand", "modes": ["nowcast_1h"], "events_detail": ev_b}), encoding="utf-8")
            comp = compare_experiments(pa, pb, n_boot=500)
            self.assertEqual(comp["n_paired_rows"], 12)
            self.assertTrue(comp["paired_bootstrap_deltas"]["mae_pct"]["statistically_significant_improvement"])
            self.assertTrue(comp["paired_bootstrap_deltas"]["crps_pct"]["statistically_significant_improvement"])

        html_out = generate_html_verification_report(
            {
                "run_id": "test_run",
                "suite": "test",
                "git_sha": "abc123",
                "modes": ["nowcast_1h"],
                "n_events_scored": 12,
                "n_scored_mode_rows": 12,
                "overall_independent_raw": score_predictions(
                    [{"truth_pct": 80.0, "truth_hit": True, "golden_exp_pct": 76.0, "golden_p10_pct": 60.0, "golden_p90_pct": 92.0}]
                ),
                "by_mode": {},
                "failure_diagnostics": diag,
                "events_detail": ev_b,
            }
        )
        self.assertIn("<svg", html_out)
        self.assertIn("PhotoPlanner Independent Forecast Verification", html_out)


if __name__ == "__main__":
    unittest.main()
