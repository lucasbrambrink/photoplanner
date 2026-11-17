"""
Live Calibrated Nowcasting (0-3h 3-Frame GOES-18/19 L2 ACTP+ACHA+COD + Parallax + Orographic Pinning
+ NWS/IEM ASOS) and Pressure-Level Corridor Forecasting (0-120h HRRR / ECMWF / AIFS / GFS + CAMS Aerosol)
coupled to the Rust `pp-raycast` 0-380 km curved-Earth spectral & stochastic corridor solver.
"""

from .backtest import BENCHMARK_REPORT_JSON_PATH, SUMMARY_JSON_PATH, run_full_backtest_suite
from .corridor_solver import CALIBRATION_CONFIG, solve_weather_and_corridor
from .raob_sounding import fetch_raob_sounding

__all__ = [
    "BENCHMARK_REPORT_JSON_PATH",
    "CALIBRATION_CONFIG",
    "SUMMARY_JSON_PATH",
    "fetch_raob_sounding",
    "run_full_backtest_suite",
    "solve_weather_and_corridor",
]
