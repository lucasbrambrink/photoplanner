#!/usr/bin/env python3
"""
CLI runner for PhotoPlanner's Multi-Regime Independent Backtesting, Diagnostics & Experiment Comparison Suite.

Examples:
    # Run the combined multi-regime independent benchmark (CA24 + SURFRAD + CONUS, sunrises & sunsets)
    .venv/bin/python scripts/run_weather_backtest.py --suite all --workers 4 --diagnose --save-calibration

    # Run NOAA SURFRAD 1-min ground pyrheliometer verification only
    .venv/bin/python scripts/run_weather_backtest.py --suite surfrad --workers 4 --diagnose

    # Compare two saved experiment snapshots with paired bootstrap 95% CIs
    .venv/bin/python scripts/run_weather_backtest.py --compare data/weather_benchmarks/experiments/A.json data/weather_benchmarks/experiments/B.json

    # Run legacy 6-track CA24 deep backtest (including Track 2 GOES parallax & Track 5 PhenoCam)
    .venv/bin/python scripts/run_weather_backtest.py --legacy-ca24 --workers 2
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
from planner.weather.benchmark_suite import (
    DEFAULT_EVAL_MODES,
    compare_experiments,
    run_benchmark_suite,
)


def _print_suite_report(report: dict, show_diagnose: bool = False) -> None:
    ov = report.get("overall_independent_raw") or {}
    print("=" * 98)
    print(
        f"PHOTOPLANNER MULTI-REGIME INDEPENDENT VERIFICATION ({report['suite'].upper()} | "
        f"{report['n_events_scored']}/{report['n_events_requested']} events | "
        f"{report['n_scored_mode_rows']} forecast rows)"
    )
    print(
        f"Run ID: {report['run_id']} | Git: {report['git_sha']} | "
        f"Elapsed: {report['elapsed_seconds']}s"
    )
    print(f"Snapshot: {report.get('experiment_path')} | HTML: {report.get('html_report_path')}")
    print("=" * 98)

    if ov.get("n", 0) > 0:
        print(
            f"\n[OVERALL INDEPENDENT RAW PHYSICS] N={ov['n']} | "
            f"MAE={ov['mae_pct']:.2f} pp | RMSE={ov['rmse_pct']:.2f} pp | Bias={ov['bias_pct']:+.2f} pp | "
            f"r={ov['pearson_r']:.3f} | rho={ov['spearman_rho']:.3f}"
        )
        print(
            f"  Probabilistic: CRPS={ov['crps_pct']:.2f} pp | "
            f"80% PI Cov={ov['pi80_coverage_pct']:.1f}% (W={ov['pi80_mean_width_pct']:.1f} pp) | "
            f"Hit Brier={ov['brier_score']:.4f} (BSS={ov['brier_skill_score_vs_climo']:+.3f}, CSI={ov['csi_threat_score']:.2f})"
        )

    print("\n[BY FORECAST LEAD MODE (LEAK-FREE)]:")
    print(f"  {'Mode':<16} {'N':>4} {'MAE':>7} {'RMSE':>7} {'Bias':>7} {'r':>6} {'CRPS':>7} {'80%Cov':>7} {'Brier':>7} {'CSI':>6}")
    for m, st in (report.get("by_mode") or {}).items():
        if not st or st.get("n", 0) == 0:
            continue
        print(
            f"  {m:<16} {st['n']:>4d} {st['mae_pct']:>7.2f} {st['rmse_pct']:>7.2f} "
            f"{st['bias_pct']:>+7.2f} {st['pearson_r']:>6.3f} {st['crps_pct']:>7.2f} "
            f"{st['pi80_coverage_pct']:>6.1f}% {st['brier_score']:>7.4f} {st['csi_threat_score']:>6.2f}"
        )

    for section_title, key in (
        ("BY INDEPENDENT TRUTH TIER", "by_truth_tier"),
        ("BY 3D TERRAIN SHADOW AUDIT (OPEN HORIZON vs FLAT-NSRDB RIDGE SHADOW)", "by_terrain_shadow_audit"),
        ("BY YEAR (CROSS-YEAR STABILITY)", "by_year"),
        ("BY PHASE (SUNRISE vs SUNSET)", "by_phase"),
        ("BY SEASON", "by_season"),
        ("BY REGION", "by_region"),
    ):
        block = report.get(key) or {}
        if not block:
            continue
        print(f"\n[{section_title}]:")
        print(f"  {'Stratum':<22} {'N':>4} {'MAE':>7} {'RMSE':>7} {'Bias':>7} {'r':>6} {'CRPS':>7} {'80%Cov':>7}")
        for k, st in block.items():
            if not st or st.get("n", 0) == 0:
                continue
            print(
                f"  {k:<22} {st['n']:>4d} {st['mae_pct']:>7.2f} {st['rmse_pct']:>7.2f} "
                f"{st['bias_pct']:>+7.2f} {st['pearson_r']:>6.3f} {st['crps_pct']:>7.2f} "
                f"{st['pi80_coverage_pct']:>6.1f}%"
            )

    xs = report.get("cross_sensor_surfrad_vs_nsrdb") or {}
    if xs.get("n_colocated_events", 0) > 0:
        print(
            f"\n[CROSS-SENSOR AUDIT: SURFRAD 1-min vs NSRDB 2km] N={xs['n_colocated_events']} | "
            f"MAE={xs['surfrad_vs_nsrdb_mae_pp']:.2f} pp | Bias={xs['surfrad_vs_nsrdb_bias_pp']:+.2f} pp | "
            f"r={xs['surfrad_vs_nsrdb_pearson_r']:.3f}"
        )

    cal = report.get("out_of_sample_calibration") or {}
    n_excl = cal.get("n_excluded_flat_nsrdb_terrain_shadow_rows", 0)
    print(f"\n[STRICT OUT-OF-SAMPLE CALIBRATION CV] (excluded {n_excl} flat-NSRDB 3D mountain-shadow rows):")
    for label, key in (
        ("Transmittance Leave-Site-Out", "transmittance_leave_site_out_cv"),
        ("Transmittance Chronological", "transmittance_chronological_cv"),
        ("Transmittance Cross-Year", "transmittance_cross_year_cv"),
    ):
        tr = cal.get(key) or {}
        raw_m = tr.get("raw_uncalibrated") or {}
        cal_m = tr.get("out_of_sample_calibrated") or {}
        if raw_m and cal_m:
            print(
                f"  {label:<30} (N={tr.get('n_samples', 0):4d}): "
                f"MAE {raw_m['mae_pct']:.2f}->{cal_m['mae_pct']:.2f} pp | "
                f"CRPS {raw_m['crps_pct']:.2f}->{cal_m['crps_pct']:.2f} pp | "
                f"Bias {raw_m['bias_pct']:+.2f}->{cal_m['bias_pct']:+.2f} pp | "
                f"80% Cov {raw_m['pi80_coverage_pct']:.1f}%->{cal_m['pi80_coverage_pct']:.1f}%"
            )
            t1a = (tr.get("by_truth_tier") or {}).get("tier_1a_surfrad")
            if t1a:
                tr1, tc1 = t1a["raw_uncalibrated"], t1a["out_of_sample_calibrated"]
                print(
                    f"    [Tier 1A SURFRAD only]     (N={t1a.get('n', 0):4d}): "
                    f"MAE {tr1['mae_pct']:.2f}->{tc1['mae_pct']:.2f} pp | "
                    f"CRPS {tr1['crps_pct']:.2f}->{tc1['crps_pct']:.2f} pp | "
                    f"Bias {tr1['bias_pct']:+.2f}->{tc1['bias_pct']:+.2f} pp | "
                    f"80% Cov {tr1['pi80_coverage_pct']:.1f}%->{tc1['pi80_coverage_pct']:.1f}%"
                )
            if key == "transmittance_cross_year_cv" and tr.get("by_fold"):
                for fl, fd in tr["by_fold"].items():
                    fr = fd.get("raw_uncalibrated") or {}
                    fc = fd.get("out_of_sample_calibrated") or {}
                    print(
                        f"    -> {fl:<25} (N={fd.get('n_test', 0):4d}): "
                        f"MAE {fr['mae_pct']:.2f}->{fc['mae_pct']:.2f} pp | "
                        f"CRPS {fr['crps_pct']:.2f}->{fc['crps_pct']:.2f} pp | "
                        f"Bias {fr['bias_pct']:+.2f}->{fc['bias_pct']:+.2f} pp | "
                        f"80% Cov {fr['pi80_coverage_pct']:.1f}%->{fc['pi80_coverage_pct']:.1f}%"
                    )

    if show_diagnose:
        fail = report.get("failure_diagnostics") or {}
        print(
            f"\n[FAILURE-MODE DIAGNOSTICS] Large Misses (|err| >= {fail.get('miss_threshold_pp', 25)} pp): "
            f"{fail.get('n_large_misses', 0)}/{fail.get('n_total_rows', 0)} ({fail.get('large_miss_rate_pct', 0):.1f}%)"
        )
        for cat, cinfo in (fail.get("categories") or {}).items():
            if cinfo["count"] > 0:
                print(
                    f"  {cat:<34}: {cinfo['count']:3d} ({cinfo['share_of_misses_pct']:5.1f}% of misses, "
                    f"mean err {cinfo['mean_error_pp']:+.1f} pp)"
                )
        print("\n  Error by Dominant Corridor Blocker:")
        for blk, binfo in (fail.get("error_by_blocker") or {}).items():
            print(f"    {blk:<26}: N={binfo['n']:3d} | MAE={binfo['mae_pp']:5.2f} pp | Bias={binfo['bias_pp']:+6.2f} pp")
        print("\n  Top-12 Worst Misses:")
        print(f"  {'Site':<28} {'Date':<11} {'Ph':<3} {'Mode':<13} {'Truth':>6} {'Pred':>6} {'Err':>7} {'L/M/H':>10} {'Blocker':<18}")
        for m in (fail.get("top_misses") or [])[:12]:
            lmh = "/".join(str(x) for x in m["cloud_lmh"])
            ph = "SR" if m["phase"] == "sunrise" else "SS"
            print(
                f"  {m['site'][:27]:<28} {m['date']:<11} {ph:<3} {m['mode']:<13} "
                f"{m['truth_pct']:>5.0f}% {m['pred_pct']:>5.0f}% {m['error_pp']:>+6.1f} {lmh:>10} {m['blocker_label'][:17]:<18}"
            )


def _print_comparison(comp: dict) -> None:
    print("=" * 92)
    print(f"PAIRED BOOTSTRAP EXPERIMENT COMPARISON (N = {comp.get('n_paired_rows', 0)} matched rows)")
    print(f"  Baseline  (A): {comp.get('baseline_run_id')}")
    print(f"  Candidate (B): {comp.get('candidate_run_id')}")
    print("=" * 92)
    if comp.get("error"):
        print(f"ERROR: {comp['error']}")
        return
    a = comp["baseline_metrics"]
    b = comp["candidate_metrics"]
    print(f"  {'Metric':<16} {'Baseline A':>12} {'Candidate B':>12} {'Delta (B-A)':>13} {'95% Boot CI':>22} {'Verdict'}")
    for metric, label in (("mae_pct", "MAE (pp)"), ("rmse_pct", "RMSE (pp)"), ("crps_pct", "CRPS (pp)")):
        d = comp["paired_bootstrap_deltas"][metric]
        verdict = (
            "SIGNIFICANT IMPROVEMENT"
            if d["statistically_significant_improvement"]
            else ("REGRESSION" if d["statistically_significant_regression"] else "within CI")
        )
        ci = f"[{d['ci95_low']:+.2f}, {d['ci95_high']:+.2f}]"
        print(f"  {label:<16} {a[metric]:>12.2f} {b[metric]:>12.2f} {d['delta_b_minus_a']:>+13.2f} {ci:>22} {verdict}")
    print(f"  {'Bias (pp)':<16} {a['bias_pct']:>+12.2f} {b['bias_pct']:>+12.2f} {b['bias_pct'] - a['bias_pct']:>+13.2f}")
    print(f"  {'Pearson r':<16} {a['pearson_r']:>12.3f} {b['pearson_r']:>12.3f} {b['pearson_r'] - a['pearson_r']:>+13.3f}")
    print(f"  {'80% PI Cov (%)':<16} {a['pi80_coverage_pct']:>12.1f} {b['pi80_coverage_pct']:>12.1f}")
    print(f"  {'Hit Brier':<16} {a['brier_score']:>12.4f} {b['brier_score']:>12.4f} {b['brier_score'] - a['brier_score']:>+13.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PhotoPlanner Historical Weather Backtest & Verification CLI")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["ca24", "surfrad", "conus_multi", "all", "cross_year", "scaled"],
        help="Stratified benchmark suite to evaluate",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_EVAL_MODES),
        help="Comma-separated forecast replay modes (e.g. nowcast_1h,nowcast_3h,hindcast_24h,hindcast_72h)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of events evaluated")
    parser.add_argument("--max-uncached", type=int, default=None, help="Evaluate at most N uncached events per invocation (resumable batching)")
    parser.add_argument("--cached-only", action="store_true", help="Score and fit calibration strictly on already-cached events (zero network calls)")
    parser.add_argument("--phases", default=None, help="Comma-separated phases filter (sunrise,sunset)")
    parser.add_argument("--dates-per-season", type=int, default=1, help="Dates per season for surfrad/conus_multi")
    parser.add_argument("--workers", type=int, default=4, help="Parallel event workers")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached per-event benchmark outputs")
    parser.add_argument("--cache-tag", default="v1", help="Cache namespace tag for physics ablations")
    parser.add_argument("--run-name", default=None, help="Human label for saved experiment snapshot")
    parser.add_argument("--save-calibration", action="store_true", help="Persist newly fit calibration artifact")
    parser.add_argument("--diagnose", action="store_true", help="Print failure-mode diagnostics & top misses")
    parser.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"), help="Compare two saved experiment JSONs")
    parser.add_argument("--legacy-ca24", action="store_true", help="Run legacy 6-track CA24 backtest")
    parser.add_argument("--json-only", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args()

    if args.compare:
        comp = compare_experiments(args.compare[0], args.compare[1])
        if args.json_only:
            print(json.dumps(comp, indent=2))
        else:
            _print_comparison(comp)
        return

    if args.legacy_ca24:
        summary = run_full_backtest_suite(max_workers=args.workers, refresh=args.refresh)
        if args.json_only:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Legacy CA24 completed in {summary['elapsed_seconds']}s -> {SUMMARY_JSON_PATH}")
        return

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    phases = tuple(p.strip() for p in args.phases.split(",") if p.strip()) if args.phases else None
    report = run_benchmark_suite(
        suite=args.suite,
        modes=modes,
        limit=args.limit,
        phases=phases,
        dates_per_season=args.dates_per_season,
        max_workers=args.workers,
        refresh=args.refresh,
        cache_tag=args.cache_tag,
        run_name=args.run_name,
        save_calibration=args.save_calibration,
        max_uncached=args.max_uncached,
        cached_only=args.cached_only,
    )
    if args.json_only:
        print(json.dumps(report, indent=2))
    elif report.get("partial_batch"):
        print(
            f"\n[BATCH COMPLETE] Cached {report['n_events_cached_total']}/{report['n_events_requested']} "
            f"events ({report['n_events_remaining']} remaining) in {report['elapsed_seconds']}s."
        )
    else:
        _print_suite_report(report, show_diagnose=args.diagnose)


if __name__ == "__main__":
    main()
