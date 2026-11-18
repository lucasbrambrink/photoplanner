"""
Live Calibrated Nowcasting (0-3h 3-Frame GOES-18/19 L2 ACTP+ACHA+COD + Parallax + Orographic Pinning
+ NWS/IEM ASOS) and Pressure-Level Corridor Forecasting (0-120h HRRR / ECMWF / AIFS / GFS + CAMS Aerosol)
coupled to the Rust `pp-raycast` 0-380 km curved-Earth spectral & stochastic corridor solver.
"""

from .backtest import BENCHMARK_REPORT_JSON_PATH, SUMMARY_JSON_PATH, run_full_backtest_suite
from .corridor_solver import CALIBRATION_CONFIG, solve_weather_and_corridor
from .hrrr_s3 import fetch_hrrr_s3_corridor
from .raob_sounding import fetch_raob_sounding
from .webcam_truth import (
    analyze_webcam_sky_chromaticity,
    fetch_cached_webcam_jpeg,
    fetch_nsrdb_2km_golden_hour_dni,
    fetch_webcam_ground_truth,
)

__all__ = [
    "BENCHMARK_REPORT_JSON_PATH",
    "CALIBRATION_CONFIG",
    "SUMMARY_JSON_PATH",
    "analyze_webcam_sky_chromaticity",
    "fetch_cached_webcam_jpeg",
    "fetch_hrrr_s3_corridor",
    "fetch_nsrdb_2km_golden_hour_dni",
    "fetch_raob_sounding",
    "fetch_webcam_ground_truth",
    "run_full_backtest_suite",
    "solve_weather_and_corridor",
]
