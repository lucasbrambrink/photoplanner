"""
Live Calibrated Nowcasting (0-3h 3-Frame GOES-18/19 L2 ACTP+ACHA+COD + Parallax + Orographic Pinning
+ NWS/IEM ASOS) and Pressure-Level Corridor Forecasting (0-120h HRRR / ECMWF / AIFS / GFS + CAMS Aerosol)
coupled to the Rust `pp-raycast` 0-380 km curved-Earth spectral & stochastic corridor solver.
"""

from .alert_engine import (
    create_watchlist_rule,
    delete_watchlist_rule,
    evaluate_alert_gate,
    evaluate_watchlist_rule_live,
    list_watchlist_rules,
)
from .backtest import BENCHMARK_REPORT_JSON_PATH, SUMMARY_JSON_PATH, run_full_backtest_suite
from .benchmark_suite import (
    EXPERIMENTS_DIR,
    VERIFICATION_HTML_PATH,
    compare_experiments,
    run_benchmark_suite,
)
from .corridor_solver import CALIBRATION_CONFIG, solve_weather_and_corridor
from .hrrr_s3 import fetch_hrrr_s3_corridor
from .light_archetypes import (
    ARCHETYPE_CATALOG,
    ARCHETYPE_ORDER,
    VIABLE_THRESHOLD_PCT,
    compute_light_archetypes,
)
from .raob_sounding import fetch_raob_sounding
from .regional_hunter import REGIONAL_SPOT_CATALOG, hunt_regional_light
from .surfrad_truth import SURFRAD_STATIONS, fetch_surfrad_golden_hour_truth
from .terrain_coupling import (
    compute_lake_mirror_index,
    compute_probabilistic_fog_bands,
    compute_trail_fog_breakout_mileposts,
)
from .webcam_truth import (
    analyze_webcam_sky_chromaticity,
    fetch_cached_webcam_jpeg,
    fetch_nsrdb_2km_golden_hour_dni,
    fetch_webcam_ground_truth,
)

__all__ = [
    "ARCHETYPE_CATALOG",
    "ARCHETYPE_ORDER",
    "BENCHMARK_REPORT_JSON_PATH",
    "CALIBRATION_CONFIG",
    "EXPERIMENTS_DIR",
    "REGIONAL_SPOT_CATALOG",
    "SUMMARY_JSON_PATH",
    "SURFRAD_STATIONS",
    "VERIFICATION_HTML_PATH",
    "VIABLE_THRESHOLD_PCT",
    "analyze_webcam_sky_chromaticity",
    "compare_experiments",
    "compute_lake_mirror_index",
    "compute_light_archetypes",
    "compute_probabilistic_fog_bands",
    "compute_trail_fog_breakout_mileposts",
    "create_watchlist_rule",
    "delete_watchlist_rule",
    "evaluate_alert_gate",
    "evaluate_watchlist_rule_live",
    "fetch_cached_webcam_jpeg",
    "fetch_hrrr_s3_corridor",
    "fetch_nsrdb_2km_golden_hour_dni",
    "fetch_raob_sounding",
    "fetch_surfrad_golden_hour_truth",
    "fetch_webcam_ground_truth",
    "hunt_regional_light",
    "list_watchlist_rules",
    "run_benchmark_suite",
    "run_full_backtest_suite",
    "solve_weather_and_corridor",
]
