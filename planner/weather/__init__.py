"""
Live Nowcasting (0-3h GOES-18/19 + NWS METAR/RAWS) and Pressure-Level Corridor
Forecasting (0-120h HRRR / ECMWF / AIFS / GFS + CAMS Aerosol) coupled to the
Rust `pp-raycast` 0-380 km curved-Earth spectral & stochastic corridor solver.
"""

from .corridor_solver import solve_weather_and_corridor

__all__ = ["solve_weather_and_corridor"]
