#!/usr/bin/env python3
"""
CLI runner for the 12-event Historical Ground-Truth Nowcasting & Raycast Calibration Suite.

Usage:
    python3 scripts/run_weather_backtest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from planner.weather.backtest import SUMMARY_JSON_PATH, run_full_backtest_suite


def _print_report(summary: dict) -> None:
    print("=" * 92)
    print(
        f"PHOTOPLANNER HISTORICAL GROUND-TRUTH BACKTEST & CALIBRATION SUITE ({summary['n_events']} EVENTS)"
    )
    print(f"Elapsed: {summary['elapsed_seconds']} s | Saved: {SUMMARY_JSON_PATH}")
    print("=" * 92)

    px = summary["parallax_verification"]
    print("\n[TRACK 1] GOES-18 3D Cloud-Top Parallax Rectification (ABI-L2-ACHAC):")
    print(
        f"  Mean Cirrus Cloud-Top Height : {px['mean_cirrus_top_asl_m']:.0f} m ASL"
    )
    print(
        f"  Mean Geostationary Shift     : {px['mean_ground_shift_km']:.2f} km "
        f"(dE {px['mean_east_shift_km']:+.2f} km, dN {px['mean_north_shift_km']:+.2f} km)"
    )
    print(
        f"  Max Corridor Cirrus Edge ΔCF : {px['max_corridor_high_cf_edge_error_pct']:.1f} pct pts"
    )

    print("\n[TRACK 2] 6-Station Corridor Cloud Nowcast Skill vs Held-Out t_sunset GOES-18 L2:")
    for lead_k, m_dict in summary["corridor_nowcast_skill_by_lead"].items():
        print(f"\n  --- Lead Time +{lead_k} ---")
        print(f"  {'Method':<30} {'MAE (%)':>9} {'RMSE (%)':>10} {'Brier':>9} {'BSS vs NWP':>12}")
        for m_name, st in m_dict.items():
            print(
                f"  {m_name:<30} {st['mae_pct']:>9.2f} {st['rmse_pct']:>10.2f} "
                f"{st['brier_score']:>9.4f} {st['brier_skill_score_vs_nwp']:>+12.3f}"
            )

    if summary["coastal_orographic_pinning_by_lead"]:
        print("\n[TRACK 2B] Coastal Montara/SF-Bay Leeward (+13 km E) Stratus Pinning Skill:")
        for lead_k, p_st in summary["coastal_orographic_pinning_by_lead"].items():
            print(
                f"  Lead +{lead_k:<5}: Legacy Unpinned MAE = {p_st['legacy_unpinned_lee_mae_pct']:5.1f}%  ->  "
                f"v3 Pinned MAE = {p_st['v3_pinned_lee_mae_pct']:5.1f}%  "
                f"(-{p_st['error_reduction_pct']:.1f}% error)"
            )

    print("\n[TRACK 3] 380 km Rust Corridor Raycast & P10-P90 Probabilistic Calibration:")
    for lead_k, modes in summary["raycast_calibration_by_lead"].items():
        print(f"\n  --- Lead Time +{lead_k} ---")
        for mode_name in ("uncalibrated", "calibrated"):
            st = modes[mode_name]
            print(
                f"  {mode_name.upper():<13} | Golden-Hr DNI: MAE={st['golden_hour_dni_mae_pct']:5.2f}% "
                f"RMSE={st['golden_hour_dni_rmse_pct']:5.2f}% r={st['golden_hour_dni_pearson_r']:.3f} "
                f"P10-P90 Cov={st['golden_hour_p10_p90_coverage_pct']:5.1f}% (W={st['golden_hour_mean_interval_width_pct']:4.1f}%) "
                f"| Sunset Ray: MAE={st['sunset_ray_mae_pct']:5.2f}% Cov={st['sunset_p10_p90_coverage_pct']:5.1f}%"
            )

    print("\n[PER-EVENT SUMMARY @ +1.0h Lead]:")
    print(
        f"  {'Event ID':<29} {'DNI Truth':>9} {'Cal GH (P10..P90)':>20} {'SS Truth':>9} {'Cal SS (P10..P90)':>20} {'Glow':>5}"
    )
    for ev in summary["events_detail"]:
        if "1.0h" not in ev["leads"]:
            continue
        c = ev["leads"]["1.0h"]["calibrated"]
        gh_str = f"{c['golden_exp_pct']:4.0f}% ({c['golden_p10_pct']:2.0f}..{c['golden_p90_pct']:2.0f}%)"
        ss_str = f"{c['sunset_exp_pct']:4.0f}% ({c['sunset_p10_pct']:2.0f}..{c['sunset_p90_pct']:2.0f}%)"
        print(
            f"  {ev['id']:<29} {ev['dni_golden_truth_pct']:>8.1f}% {gh_str:>20} "
            f"{ev['sunset_ray_truth_pct']:>8.1f}% {ss_str:>20} {c['alpenglow_score']:>5d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhotoPlanner historical weather backtest")
    parser.add_argument("--workers", type=int, default=3, help="Parallel event workers")
    parser.add_argument("--json-only", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args()

    summary = run_full_backtest_suite(max_workers=args.workers)
    if args.json_only:
        print(json.dumps(summary, indent=2))
    else:
        _print_report(summary)


if __name__ == "__main__":
    main()
