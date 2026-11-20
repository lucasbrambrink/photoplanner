"""
Unit and out-of-sample integration tests for `planner/weather/calibration.py`.

Verifies:
1. Proper scoring rules (`pinball_loss`, `crps_from_quantiles`, `brier_score`, `brier_skill_score`, `ECE`).
2. Strict zero-leakage train/test splitters (`leave_group_out_splits`, `chronological_blocked_splits`).
3. Monotonicity, physical `[0, 100]` boundary preservation, and JSON round-trip persistence.
4. Synthetic out-of-sample bias + conformal interval recovery on unseen sites.
5. Real out-of-sample CV improvement on `data/weather_benchmarks/backtest_summary.json`
   evaluated strictly against independent 2 km NSRDB DNI & NWS RAOB radiosondes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from planner.weather.calibration import (
    InversionClearanceCalibrator,
    TransmittanceCalibrator,
    brier_score,
    brier_skill_score,
    build_transmittance_feature_vector,
    chronological_blocked_splits,
    crps_from_quantiles,
    evaluate_calibration_out_of_sample,
    expected_calibration_error,
    interval_coverage_and_width,
    leave_group_out_splits,
    load_calibration_bundle,
    pinball_loss,
    save_calibration_bundle,
    summarize_transmittance_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SUMMARY_PATH = REPO_ROOT / "data" / "weather_benchmarks" / "backtest_summary.json"


class TestProperScoringRules(unittest.TestCase):
    def test_pinball_and_crps_zero_for_perfect_forecast(self) -> None:
        y = np.array([10.0, 50.0, 90.0])
        self.assertAlmostEqual(pinball_loss(y, y, 0.5), 0.0, places=9)
        self.assertAlmostEqual(crps_from_quantiles(y, y, y, y), 0.0, places=9)
        cov, width = interval_coverage_and_width(y, y - 5.0, y + 5.0)
        self.assertAlmostEqual(cov, 1.0, places=9)
        self.assertAlmostEqual(width, 10.0, places=9)

    def test_brier_skill_score_and_ece(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=np.float64)
        p_perfect = np.array([0.0, 0.0, 1.0, 1.0])
        p_climo = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(brier_score(y, p_perfect), 0.0, places=9)
        self.assertAlmostEqual(brier_skill_score(y, p_perfect), 1.0, places=9)
        self.assertAlmostEqual(brier_skill_score(y, p_climo), 0.0, places=9)
        self.assertAlmostEqual(expected_calibration_error(y, p_perfect, n_bins=5), 0.0, places=9)


class TestLeakFreeSplitters(unittest.TestCase):
    def test_leave_group_out_zero_overlap(self) -> None:
        sites = ["A", "A", "B", "B", "C", "C", "D", "D"]
        splits = leave_group_out_splits(sites)
        self.assertEqual(len(splits), 4)
        seen_test = set()
        for tr, te, label in splits:
            tr_sites = {sites[i] for i in tr}
            te_sites = {sites[i] for i in te}
            self.assertEqual(tr_sites & te_sites, set())
            self.assertEqual(te_sites, {label})
            seen_test.update(te.tolist())
        self.assertEqual(seen_test, set(range(len(sites))))

    def test_chronological_blocked_zero_date_overlap(self) -> None:
        dates = [
            "2025-01-05", "2025-01-05", "2025-02-10", "2025-03-12",
            "2025-04-15", "2025-05-20", "2025-06-25", "2025-08-01",
            "2025-09-10", "2025-10-15", "2025-11-20", "2025-12-01",
        ]
        splits = chronological_blocked_splits(dates, n_folds=4)
        self.assertEqual(len(splits), 4)
        for tr, te, _ in splits:
            tr_dates = {dates[i] for i in tr}
            te_dates = {dates[i] for i in te}
            self.assertEqual(tr_dates & te_dates, set())

    def test_cross_year_holdout_zero_year_overlap(self) -> None:
        from planner.weather.calibration import cross_year_holdout_splits

        dates = ["2024-03-16", "2024-07-20", "2024-11-16", "2025-01-18", "2025-06-14", "2025-10-11"]
        splits = cross_year_holdout_splits(dates)
        self.assertEqual(len(splits), 2)
        labels = [lbl for _, _, lbl in splits]
        self.assertEqual(labels, ["train_2025_test_2024", "train_2024_test_2025"])
        for tr, te, _ in splits:
            tr_yrs = {dates[i][:4] for i in tr}
            te_yrs = {dates[i][:4] for i in te}
            self.assertEqual(tr_yrs & te_yrs, set())
        # Single-year input returns empty list
        self.assertEqual(cross_year_holdout_splits(["2025-01-18", "2025-06-14"]), [])

    def test_flat_nsrdb_terrain_shadow_exclusion_and_tier_weights(self) -> None:
        from planner.weather.calibration import (
            TIER_1A_SURFRAD_WEIGHT,
            TIER_1B_NSRDB_OPEN_WEIGHT,
            extract_independent_records_from_benchmark,
            is_flat_nsrdb_terrain_shadow_artifact,
        )

        self.assertTrue(is_flat_nsrdb_terrain_shadow_artifact("tier_1b_nsrdb_2km", 6.2, "terrain"))
        self.assertTrue(is_flat_nsrdb_terrain_shadow_artifact("tier_1b_nsrdb_2km", 4.0, "clear_corridor"))
        self.assertFalse(is_flat_nsrdb_terrain_shadow_artifact("tier_1b_nsrdb_2km", 1.2, "mid"))
        # Tier 1A ground pyrheliometers physically observe real mountain shadows -> never excluded
        self.assertFalse(is_flat_nsrdb_terrain_shadow_artifact("tier_1a_surfrad", 5.5, "terrain"))

        mock_events = [
            {
                "id": "ev_shadow_nsrdb",
                "site": "Maroon Bells, CO",
                "region": "rockies_alpine",
                "date": "2024-02-24",
                "primary_truth_tier": "tier_1b_nsrdb_2km",
                "nsrdb_2km_golden_truth_pct": 99.0,
                "modes": {
                    "nowcast_1h": {
                        "golden_exp_pct": 0.0,
                        "golden_p10_pct": 0.0,
                        "golden_p90_pct": 5.0,
                        "terrain_horizon_deg": 7.5,
                        "blocker_label": "terrain",
                    }
                },
            },
            {
                "id": "ev_open_nsrdb",
                "site": "Grand Canyon, AZ",
                "region": "southwest_plateau",
                "date": "2024-06-15",
                "primary_truth_tier": "tier_1b_nsrdb_2km",
                "nsrdb_2km_golden_truth_pct": 95.0,
                "modes": {
                    "nowcast_1h": {
                        "golden_exp_pct": 88.0,
                        "golden_p10_pct": 70.0,
                        "golden_p90_pct": 98.0,
                        "terrain_horizon_deg": 0.4,
                        "blocker_label": "clear_corridor",
                    }
                },
            },
            {
                "id": "ev_surfrad",
                "site": "Table Mountain, CO",
                "region": "rockies",
                "date": "2024-06-15",
                "primary_truth_tier": "tier_1a_surfrad",
                "surfrad_ref": {"golden_hour_rel_transmittance_pct": 45.0},
                "modes": {
                    "nowcast_1h": {
                        "golden_exp_pct": 42.0,
                        "golden_p10_pct": 20.0,
                        "golden_p90_pct": 65.0,
                        "terrain_horizon_deg": 2.1,
                        "blocker_label": "mid",
                    }
                },
            },
        ]
        clean_recs, _ = extract_independent_records_from_benchmark(mock_events, modes=("nowcast_1h",))
        self.assertEqual(len(clean_recs), 2)
        by_id = {r["event_id"]: r for r in clean_recs}
        self.assertNotIn("ev_shadow_nsrdb", by_id)
        self.assertAlmostEqual(by_id["ev_surfrad"]["tier_weight"], TIER_1A_SURFRAD_WEIGHT)
        self.assertAlmostEqual(by_id["ev_open_nsrdb"]["tier_weight"], TIER_1B_NSRDB_OPEN_WEIGHT)


class TestCalibratorModels(unittest.TestCase):
    def test_synthetic_optical_depth_bias_recovery_and_serialization(self) -> None:
        rng = np.random.default_rng(42)
        n_events = 60
        raw_exp = rng.uniform(3.0, 97.0, size=n_events)
        raw_p10 = np.clip(raw_exp - 8.0, 0.0, 100.0)
        raw_p90 = np.clip(raw_exp + 8.0, 0.0, 100.0)
        lead_h = rng.choice([1.0, 3.0, 24.0], size=n_events)

        # True transmittance has positive logit bias (+0.9) + heteroscedastic noise
        p_raw = np.clip(raw_exp / 100.0, 0.015, 0.985)
        z_true = np.log(p_raw / (1.0 - p_raw)) + 0.9 + rng.normal(0.0, 0.45, size=n_events)
        y_true = 100.0 / (1.0 + np.exp(-z_true))

        X = np.vstack(
            [
                build_transmittance_feature_vector(
                    raw_exp[i], raw_p10[i], raw_p90[i], 30.0, 10.0, 10.0, True, lead_h[i]
                )
                for i in range(n_events)
            ]
        )
        ev_ids = [f"ev_{i}" for i in range(n_events)]

        # Split train / unseen test
        tr, te = slice(0, 40), slice(40, 60)
        cal = TransmittanceCalibrator().fit(
            X[tr], raw_exp[tr], y_true[tr], raw_p90[tr] - raw_p10[tr], lead_h[tr], event_ids=ev_ids[tr]
        )
        p10, p50, p90 = cal.predict(X[te], raw_exp[te], raw_p90[te] - raw_p10[te], lead_h[te])

        # Bounds & strict quantile monotonicity
        self.assertTrue(np.all(p10 >= 0.0))
        self.assertTrue(np.all(p10 <= p50 + 1e-9))
        self.assertTrue(np.all(p50 <= p90 + 1e-9))
        self.assertTrue(np.all(p90 <= 100.0))

        # Out-of-sample MAE & CRPS must significantly beat raw biased forecast
        raw_m = summarize_transmittance_metrics(y_true[te], raw_exp[te], raw_p10[te], raw_p90[te])
        cal_m = summarize_transmittance_metrics(y_true[te], p50, p10, p90)
        self.assertLess(cal_m["mae_pct"], raw_m["mae_pct"])
        self.assertLess(cal_m["crps_pct"], raw_m["crps_pct"])
        self.assertGreaterEqual(cal_m["pi80_coverage_pct"], 65.0)

        # Boundary preservation: solid overcast (0%) stays low, clear (99%) stays high
        _, p50_overcast, _ = cal.predict_one(0.5, 0.0, 10.0, 95.0, 50.0, 20.0, True, 1.0)
        _, p50_clear, _ = cal.predict_one(99.0, 90.0, 100.0, 0.0, 0.0, 0.0, True, 1.0)
        self.assertLess(p50_overcast, 10.0)
        self.assertGreater(p50_clear, 95.0)

        # Round-trip JSON persistence
        inv = InversionClearanceCalibrator().fit(
            [-600, -250, -50, 100, 350, 800],
            [1.0, 3.0, 24.0, 1.0, 3.0, 24.0],
            [0, 0, 0, 1, 1, 1],
        )
        with tempfile.TemporaryDirectory() as tmpd:
            path = Path(tmpd) / "cal_bundle.json"
            save_calibration_bundle(
                {"fitted_models": {"transmittance": cal.to_dict(), "inversion_clearance": inv.to_dict()}},
                path,
            )
            loaded = load_calibration_bundle(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            t_loaded, i_loaded = loaded
            lp10, lp50, lp90 = t_loaded.predict(X[te], raw_exp[te], raw_p90[te] - raw_p10[te], lead_h[te])
            np.testing.assert_allclose(lp50, p50, atol=1e-3)
            np.testing.assert_allclose(lp10, p10, atol=1e-3)
            np.testing.assert_allclose(lp90, p90, atol=1e-3)
            self.assertAlmostEqual(
                i_loaded.predict_one(300.0, 3.0), inv.predict_one(300.0, 3.0), places=4
            )

    def test_real_benchmark_out_of_sample_calibration_improves_all_proper_scores(self) -> None:
        if not BENCHMARK_SUMMARY_PATH.exists():
            self.skipTest("Benchmark summary archive not present")
        doc = json.loads(BENCHMARK_SUMMARY_PATH.read_text(encoding="utf-8"))
        rep = evaluate_calibration_out_of_sample(doc["events_detail"])

        for key in ("transmittance_leave_site_out_cv", "transmittance_chronological_cv"):
            cv = rep[key]
            raw_m = cv["raw_uncalibrated"]
            cal_m = cv["out_of_sample_calibrated"]
            self.assertGreater(cv["n_samples"], 0)
            self.assertLess(cal_m["mae_pct"], raw_m["mae_pct"])
            self.assertLess(cal_m["rmse_pct"], raw_m["rmse_pct"])
            self.assertLess(cal_m["crps_pct"], raw_m["crps_pct"])
            self.assertLess(abs(cal_m["bias_pct"]), abs(raw_m["bias_pct"]))
            self.assertGreaterEqual(cal_m["pi80_coverage_pct"], 75.0)

        for key in ("inversion_leave_site_out_cv", "inversion_chronological_cv"):
            cv = rep[key]
            raw_i = cv["raw_deterministic_step"]
            cal_i = cv["out_of_sample_calibrated"]
            self.assertLess(cal_i["brier_score"], raw_i["brier_score"])
            self.assertLess(cal_i["ece"], raw_i["ece"])


if __name__ == "__main__":
    unittest.main()
