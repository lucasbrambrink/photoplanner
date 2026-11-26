"""
Regression tests for Phase 0 meteorological ingestion, causality, and physics fixes.

All tests run hermetically on pure NumPy + stdlib (no network calls, no h5py required).
"""

from __future__ import annotations

import copy
import datetime as dt
import math
import unittest
from unittest.mock import patch

import numpy as np

from planner.weather.goes_nowcast import _read_cf_unsigned_array, _select_goes_bucket
from planner.weather.hrrr_s3 import (
    _unpack_bits_u64,
    apply_hrrr_column_to_sounding,
    rotate_hrrr_uv_to_earth,
)
from planner.weather.nwp_corridor import (
    _reconstruct_hindcast_point_doc,
    extract_hour_sounding,
)
from planner.weather.raob_sounding import (
    _analyze_raob_profile,
    _candidate_synoptic_times,
)
from planner.weather.station_obs import (
    _compute_koschmieder_aod_proxy,
    _parse_nws_cloud_layers,
)
from planner.weather.webcam_truth import _match_phenocam_archive_camera


class _MockCFDataset:
    def __init__(self, attrs: dict):
        self.attrs = attrs


class TestGoesNowcastFixes(unittest.TestCase):
    def test_cf_unsigned_int16_preserves_high_cirrus_above_10km(self) -> None:
        """
        GOES-18 ACHA HT uses int16 with CF `_Unsigned='true'`, scale=0.3052 m, offset=0.
        Unsigned 39320 -> ~12,000 m cirrus, which wraps to signed -26216 in int16!
        Fill value is unsigned 65535 (signed -1).
        """
        ds = _MockCFDataset({"_Unsigned": "true", "_FillValue": np.int16(-1)})
        raw = np.array([-26216, 10000, -1], dtype=np.int16)
        vals, fill = _read_cf_unsigned_array(ds, raw)
        self.assertEqual(fill, 65535)
        self.assertEqual(int(vals[0]), 39320)
        self.assertEqual(int(vals[1]), 10000)
        self.assertEqual(int(vals[2]), 65535)
        # Scaled cirrus height must be ~12 km, NOT negative/masked!
        cirrus_h_m = float(vals[0]) * 0.3052
        self.assertGreater(cirrus_h_m, 11_500.0)

    def test_goes17_vs_goes18_historical_bucket_routing(self) -> None:
        b_pre, sat_pre = _select_goes_bucket(
            True, dt.datetime(2022, 9, 15, 2, 0, tzinfo=dt.timezone.utc)
        )
        self.assertEqual(b_pre, "noaa-goes17")
        self.assertIn("GOES-17", sat_pre)

        b_post, sat_post = _select_goes_bucket(
            True, dt.datetime(2025, 5, 16, 2, 0, tzinfo=dt.timezone.utc)
        )
        self.assertEqual(b_post, "noaa-goes18")
        self.assertIn("GOES-18", sat_post)

        b_east, sat_east = _select_goes_bucket(
            False, dt.datetime(2025, 6, 1, 0, 0, tzinfo=dt.timezone.utc)
        )
        self.assertEqual(b_east, "noaa-goes19")
        self.assertIn("GOES-19", sat_east)


class TestHrrrS3Fixes(unittest.TestCase):
    def test_lcc_wind_rotation_preserves_speed_and_sign_in_california(self) -> None:
        # In LCC (lon_0 = -97.5, lat_1=lat_2=38.5):
        # alpha = sin(38.5 deg) * (lon - lon_0) ~ -15.56 deg at lon = -122.5.
        # u_earth = u_g * cos(alpha) + v_g * sin(alpha)
        # v_earth = -u_g * sin(alpha) + v_g * cos(alpha)
        # So pure grid-eastward wind (10, 0) has a positive northward Earth component (+2.68 m/s).
        u_e, v_e = rotate_hrrr_uv_to_earth(10.0, 0.0, -122.5)
        self.assertAlmostEqual(math.hypot(u_e, v_e), 10.0, places=6)
        self.assertGreater(u_e, 9.0)
        self.assertGreater(v_e, 2.0)

        # At central meridian lon = -97.5, rotation angle is 0.
        u0, v0 = rotate_hrrr_uv_to_earth(7.5, -4.2, -97.5)
        self.assertAlmostEqual(u0, 7.5, places=6)
        self.assertAlmostEqual(v0, -4.2, places=6)

    def test_unpack_bits_u64_handles_30bit_cross_boundary_values(self) -> None:
        # Pack three 30-bit integers: [0x3FFFFFFF, 0x12345678, 0x2AAAAAAA]
        vals_in = [0x3FFFFFFF, 0x12345678, 0x2AAAAAAA]
        bit_str = "".join(f"{v:030b}" for v in vals_in)
        pad = (-len(bit_str)) % 8
        bit_str_padded = bit_str + ("0" * pad)
        raw_bytes = int(bit_str_padded, 2).to_bytes(len(bit_str_padded) // 8, "big")
        unpacked = _unpack_bits_u64(raw_bytes, 0, 3, 30)
        self.assertEqual(unpacked.tolist(), vals_in)

    def test_tcolw_not_double_scaled_by_cloud_fraction(self) -> None:
        # Build minimal base sounding and HRRR column with known TCOLW = 0.25 kg/m^2, low cloud = 50%.
        base_sounding = {
            "cloud_low_pct": 50,
            "cloud_mid_pct": 0,
            "cloud_high_pct": 0,
            "temp_2m_c": 14.0,
            "dewpoint_2m_c": 12.0,
            "rh_2m_pct": 88.0,
            "rh_975_pct": 85.0,
            "rh_950_pct": 60.0,
            "rh_925_pct": 35.0,
            "temp_950_c": 16.0,
            "temp_925_c": 18.0,
            "temp_850_c": 14.0,
            "lcl_asl_m": 250.0,
            "inv_base_asl_m": 450.0,
            "inv_top_asl_m": 650.0,
            "inv_jump_k": 4.0,
            "visibility_m": 12000.0,
            "pbl_agl_m": 500.0,
            "cape_j_kg": 0.0,
            "wind_sfc_uv_ms": (4.0, 0.0),
            "wind_850_uv_ms": (4.0, 1.0),
            "wind_500_uv_ms": (12.0, 0.0),
            "wind_250_uv_ms": (25.0, 0.0),
            "rust_station": {
                "distance_km": 0.0,
                "lat": 37.6,
                "lon": -122.5,
                "surface_elevation_m": 100.0,
                "surface_temp_c": 14.0,
                "aod_550nm": 0.07,
                "cloud_bands": [
                    {
                        "label": "fog",
                        "base_asl_m": 180.0,
                        "top_asl_m": 550.0,
                        "cloud_fraction": 0.50,
                        "vertical_optical_depth": 10.0,
                        "ice_fraction": 0.0,
                        "asymmetry_g": 0.85,
                    }
                ],
            },
        }
        hrrr_col = {
            "ok": True,
            "z_sfc_m": 100.0,
            "temp_2m_c": 13.5,
            "dewpoint_2m_c": 12.5,
            "rh_2m_pct": 93.0,
            "rh_975_pct": 92.0,
            "rh_950_pct": 55.0,
            "rh_925_pct": 30.0,
            "temp_950_c": 17.0,
            "temp_925_c": 19.0,
            "temp_850_c": 14.5,
            "lcdc_pct": 50.0,
            "tcdc_bl_pct": 50.0,
            "mcdc_pct": 0.0,
            "hcdc_pct": 0.0,
            "tcolw_kg_m2": 0.25,
            "tcoli_kg_m2": 0.0,
            "vis_m": 9000.0,
            "hpbl_m": 480.0,
            "cape_j_kg": 0.0,
            "lcl_asl_m": 225.0,
            "lid_asl_m": 580.0,
            "inv_base_asl_m": 420.0,
            "inv_strength_k": 5.0,
            "stratus_base_asl_m": 225.0,
            "aotk_550nm": 0.07,
            "u10_ms": 4.0,
            "v10_ms": 0.5,
            "u850_ms": 5.0,
            "v850_ms": 1.0,
            "u500_ms": 12.0,
            "v500_ms": 2.0,
            "u250_ms": 26.0,
            "v250_ms": 3.0,
            "z_850_m": 1500.0,
            "z_700_m": 3100.0,
            "z_500_m": 5650.0,
            "z_300_m": 9350.0,
            "rh_500_pct": 25.0,
            "temp_700_c": 3.0,
            "temp_500_c": -15.0,
        }
        out = apply_hrrr_column_to_sounding(base_sounding, hrrr_col, is_coastal=True, blend_weight=1.0)
        fog_band = next(b for b in out["rust_station"]["cloud_bands"] if b["label"] == "fog")
        # Previously multiplying grid-mean LWP by cf=0.50 drove tau to ~14; without double-scaling it is 28.0.
        self.assertGreaterEqual(float(fog_band["optical_depth_vertical"]), 20.0)


class TestRaobSoundingFixes(unittest.TestCase):
    def test_operational_60min_causality_lag(self) -> None:
        # At 00:30 UTC, the 00:00 UTC balloon is still ascending/being transmitted and MUST NOT be used.
        t_0030 = dt.datetime(2025, 6, 5, 0, 30, tzinfo=dt.timezone.utc)
        cands_0030 = _candidate_synoptic_times(t_0030, causal_only=True)
        self.assertNotIn(dt.datetime(2025, 6, 5, 0, 0, tzinfo=dt.timezone.utc), cands_0030)
        self.assertIn(dt.datetime(2025, 6, 4, 12, 0, tzinfo=dt.timezone.utc), cands_0030)

        # At 01:05 UTC (+65 min post-nominal), the 00:00 UTC sounding IS operationally available.
        t_0105 = dt.datetime(2025, 6, 5, 1, 5, tzinfo=dt.timezone.utc)
        cands_0105 = _candidate_synoptic_times(t_0105, causal_only=True)
        self.assertEqual(cands_0105[0], dt.datetime(2025, 6, 5, 0, 0, tzinfo=dt.timezone.utc))

        # Non-causal observed-truth mode may still use 00:00 UTC at 00:30 UTC.
        cands_truth = _candidate_synoptic_times(t_0030, causal_only=False)
        self.assertEqual(cands_truth[0], dt.datetime(2025, 6, 5, 0, 0, tzinfo=dt.timezone.utc))

    def test_mountain_station_krev_gap_free_asl_bands_and_missing_tokens(self) -> None:
        # Reno (KREV) at 1516 m ASL with >= 8 valid IEM levels, a saturated cloud deck at 2300-2520 m ASL,
        # and a corrupt "M" string level that must be skipped without crashing.
        raw_levels = [
            {"pres": "M", "hght": "M", "tmpc": "M", "dwpc": "M", "drct": "M", "sknt": "M"},
            {"pres": 850, "hght": 1520, "tmpc": 14.0, "dwpc": 4.0, "drct": 260, "sknt": 8},
            {"pres": 820, "hght": 1820, "tmpc": 11.5, "dwpc": 3.5, "drct": 255, "sknt": 10},
            {"pres": 790, "hght": 2120, "tmpc": 9.0, "dwpc": 7.5, "drct": 250, "sknt": 12},
            {"pres": 770, "hght": 2300, "tmpc": 7.0, "dwpc": 6.8, "drct": 250, "sknt": 15},
            {"pres": 750, "hght": 2520, "tmpc": 5.5, "dwpc": 5.3, "drct": 250, "sknt": 18},
            {"pres": 700, "hght": 3080, "tmpc": 1.5, "dwpc": -12.0, "drct": 260, "sknt": 22},
            {"pres": 600, "hght": 4300, "tmpc": -6.0, "dwpc": -22.0, "drct": 265, "sknt": 28},
            {"pres": 500, "hght": 5600, "tmpc": -14.0, "dwpc": -32.0, "drct": 270, "sknt": 35},
            {"pres": 250, "hght": 10500, "tmpc": -48.0, "dwpc": -65.0, "drct": 275, "sknt": 60},
        ]
        valid = dt.datetime(2025, 5, 13, 0, 0, tzinfo=dt.timezone.utc)
        prof = _analyze_raob_profile(
            raw_levels, "KREV", "Reno NV", 39.57, -119.80, 1516.0, 95.0, valid, valid
        )
        self.assertIsNotNone(prof)
        assert prof is not None
        # Saturated deck at 2300-2520 m ASL (~800-1000 m AGL at KREV) must register in low_cloud_fraction_proxy!
        self.assertGreater(float(prof["low_cloud_fraction_proxy"]), 0.70)


class TestStationObsFixes(unittest.TestCase):
    def test_vv_obscured_fog_with_none_base_sets_full_fog(self) -> None:
        raw_layers = [{"amount": "VV", "base": {"value": None}}]
        layers, fog_f, low_f, mid_f, high_f, fog_base, lowest_ceil = _parse_nws_cloud_layers(
            raw_layers, st_elev_m=10.0
        )
        self.assertEqual(fog_f, 1.0)
        self.assertEqual(low_f, 1.0)
        self.assertEqual(mid_f, 0.0)
        self.assertEqual(high_f, 0.0)
        self.assertIsNotNone(fog_base)
        self.assertAlmostEqual(float(fog_base), 25.0, places=1)
        self.assertAlmostEqual(float(lowest_ceil), 25.0, places=1)
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0]["amount"], "VV")

    def test_koschmieder_aod_gates_humid_marine_mist_but_keeps_dry_smoke(self) -> None:
        # Humid coastal mist (vis=2 km, RH=88%) must NOT become dry-aerosol smoke AOD!
        self.assertIsNone(_compute_koschmieder_aod_proxy(2000.0, 88.0))
        self.assertIsNone(_compute_koschmieder_aod_proxy(3000.0, 82.0))

        # Dry wildfire smoke (vis=4 km, RH=30%) MUST yield elevated AOD proxy.
        dry_aod = _compute_koschmieder_aod_proxy(4000.0, 30.0)
        self.assertIsNotNone(dry_aod)
        assert dry_aod is not None
        self.assertGreater(dry_aod, 0.5)

        # Intermediate humidity (76% RH) must be smoothly attenuated compared to dry air.
        mid_aod = _compute_koschmieder_aod_proxy(4000.0, 76.0)
        self.assertIsNotNone(mid_aod)
        assert mid_aod is not None
        self.assertLess(mid_aod, dry_aod)


class TestNwpCorridorLeakFreeHindcast(unittest.TestCase):
    def _build_synthetic_pdoc(self, elev_m: float = 200.0) -> tuple[dict, dict]:
        n = 24
        times = [f"2025-05-13T{h:02d}:00" for h in range(n)]
        base_hourly = {
            "time": times,
            "temperature_2m": [16.0] * n,
            "dew_point_2m": [11.0] * n,
            "relative_humidity_2m": [72.0] * n,
            "cloud_cover": [80.0] * n,
            "cloud_cover_low": [80.0] * n,
            "cloud_cover_mid": [20.0] * n,
            "cloud_cover_high": [10.0] * n,
            "visibility": [5000.0] * n,
            "wind_speed_10m": [15.0] * n,
            "wind_direction_10m": [280.0] * n,
            "precipitation": [2.0] * n,
            "cape": [500.0] * n,
            "boundary_layer_height": [900.0] * n,
        }
        # Populate Day-0 pressure levels with deliberate values
        for p, z, t, rh, cc, ws, wd in (
            (1000, 120.0, 15.0, 85.0, 85.0, 10.0, 270.0),
            (950, 560.0, 19.0, 40.0, 20.0, 14.0, 280.0),
            (925, 790.0, 20.0, 30.0, 10.0, 16.0, 285.0),
            (900, 1030.0, 18.0, 25.0, 5.0, 18.0, 290.0),
            (850, 1510.0, 14.0, 25.0, 5.0, 20.0, 295.0),
            (700, 3100.0, 3.0, 30.0, 15.0, 35.0, 280.0),
            (500, 5680.0, -15.0, 35.0, 25.0, 55.0, 275.0),
            (300, 9350.0, -42.0, 40.0, 30.0, 80.0, 270.0),
            (250, 10550.0, -50.0, 45.0, 35.0, 90.0, 268.0),
        ):
            base_hourly[f"geopotential_height_{p}hPa"] = [z] * n
            base_hourly[f"temperature_{p}hPa"] = [t] * n
            base_hourly[f"relative_humidity_{p}hPa"] = [rh] * n
            base_hourly[f"cloud_cover_{p}hPa"] = [cc] * n
            base_hourly[f"wind_speed_{p}hPa"] = [ws] * n
            base_hourly[f"wind_direction_{p}hPa"] = [wd] * n

        base_pdoc = {"elevation": elev_m, "utc_offset_seconds": -28800, "hourly": base_hourly}

        # Previous-runs archived Day-1 forecast (only surface + cloud columns available in API)
        sfx = "_previous_day1"
        prev_hourly = {
            "time": times,
            f"temperature_2m{sfx}": [18.0] * n,
            f"dew_point_2m{sfx}": [9.0] * n,
            f"relative_humidity_2m{sfx}": [55.0] * n,
            f"cloud_cover{sfx}": [25.0] * n,
            f"cloud_cover_low{sfx}": [10.0] * n,
            f"cloud_cover_mid{sfx}": [15.0] * n,
            f"cloud_cover_high{sfx}": [20.0] * n,
            f"wind_speed_10m{sfx}": [8.0] * n,
            f"wind_direction_10m{sfx}": [240.0] * n,
            f"precipitation{sfx}": [0.0] * n,
        }
        return base_pdoc, prev_hourly

    def test_hindcast_reconstruction_has_zero_day0_leakage(self) -> None:
        base_a, prev_h = self._build_synthetic_pdoc(elev_m=200.0)
        base_b = copy.deepcopy(base_a)

        # Violently mutate EVERY Day-0 field in base_b (pressure levels, winds, vis, CAPE, precip, BLH)
        hb = base_b["hourly"]
        hb["visibility"] = [300.0] * 24
        hb["cape"] = [3500.0] * 24
        hb["precipitation"] = [50.0] * 24
        hb["boundary_layer_height"] = [150.0] * 24
        hb["temperature_2m"] = [2.0] * 24
        hb["cloud_cover_low"] = [100.0] * 24
        for p in (1000, 950, 925, 900, 850, 700, 500, 300, 250):
            hb[f"temperature_{p}hPa"] = [-30.0] * 24
            hb[f"relative_humidity_{p}hPa"] = [100.0] * 24
            hb[f"cloud_cover_{p}hPa"] = [100.0] * 24
            hb[f"wind_speed_{p}hPa"] = [140.0] * 24
            hb[f"wind_direction_{p}hPa"] = [90.0] * 24

        rec_a = _reconstruct_hindcast_point_doc(base_a, prev_h, "_previous_day1", pt_coastal=True)
        rec_b = _reconstruct_hindcast_point_doc(base_b, prev_h, "_previous_day1", pt_coastal=True)

        self.assertTrue(rec_a.get("hindcast_leak_free"))
        self.assertEqual(rec_a, rec_b)

        snd_a = extract_hour_sounding(rec_a, 18, 0.0, 270.0, 37.6, -122.5, is_coastal=True)
        snd_b = extract_hour_sounding(rec_b, 18, 0.0, 270.0, 37.6, -122.5, is_coastal=True)
        self.assertEqual(snd_a, snd_b)

    def test_underground_pressure_levels_filtered_for_alpine_inversion(self) -> None:
        # At Half Dome (2700 m ASL), 1000/950/925/850 hPa are underground!
        # Even if underground 1000 hPa is artificially cold (0 C) and 850 hPa is warm (25 C),
        # extract_hour_sounding must NOT hallucinate a 25 K alpine inversion.
        base_pdoc, _ = self._build_synthetic_pdoc(elev_m=2700.0)
        h = base_pdoc["hourly"]
        h["temperature_2m"] = [8.0] * 24
        h["dew_point_2m"] = [-2.0] * 24
        h["temperature_1000hPa"] = [0.0] * 24
        h["temperature_925hPa"] = [22.0] * 24
        h["temperature_850hPa"] = [25.0] * 24
        # Free-air levels above 2700 m (700 hPa at 3100 m = 5 C, 500 hPa at 5680 m = -15 C) cool normally.
        h["temperature_700hPa"] = [5.0] * 24
        h["temperature_500hPa"] = [-15.0] * 24

        snd = extract_hour_sounding(base_pdoc, 18, 0.0, 270.0, 37.746, -119.533, is_coastal=False)
        self.assertLessEqual(float(snd["inv_jump_k"]), 0.5)

    def test_hindcast_preserves_utc_offset_seconds_and_commercial_url(self) -> None:
        import os
        from planner.weather.nwp_corridor import _open_meteo_url

        base_pdoc, prev_h = self._build_synthetic_pdoc(elev_m=150.0)
        base_pdoc["utc_offset_seconds"] = -28800
        rec = _reconstruct_hindcast_point_doc(
            base_pdoc, prev_h, "_previous_day1", pt_coastal=True, utc_offset_seconds=-28800
        )
        self.assertEqual(rec.get("utc_offset_seconds"), -28800)

        with patch.dict(os.environ, {"OPEN_METEO_API_KEY": ""}, clear=False):
            self.assertTrue(_open_meteo_url("api", "/v1/forecast?x=1").startswith("https://api.open-meteo.com/"))
        with patch.dict(os.environ, {"OPEN_METEO_API_KEY": "demo_key_123"}, clear=False):
            u = _open_meteo_url("previous-runs-api", "/v1/forecast?x=1")
            self.assertTrue(u.startswith("https://customer-previous-runs-api.open-meteo.com/"))
            self.assertIn("apikey=demo_key_123", u)


class TestWebcamTruthFixes(unittest.TestCase):
    def test_phenocam_window_respects_sunrise_vs_sunset_phase(self) -> None:
        fake_catalog = [
            {
                "Sitename": "mockcam",
                "Lat": 37.6,
                "Lon": -122.5,
                "Elev": 200.0,
                "sitemetadata": {
                    "long_name": "Mock Coastal Cam",
                    "camera_orientation": "W",
                },
                "date_first": "2020-01-01",
                "date_last": "2026-12-31",
            }
        ]
        req_date = dt.date(2025, 6, 14)

        def _mk_frame(when_utc: dt.datetime, local_str: str, fname: str) -> dict:
            return {
                "archive_path": f"/data/archive/mockcam/2025/06/{fname}",
                "url": f"https://phenocam.nau.edu/data/archive/mockcam/2025/06/{fname}",
                "proxy_url": f"/api/webcam-image?path={fname}",
                "timestamp_utc": when_utc.isoformat(),
                "timestamp_local": local_str,
                "minutes_local": int(local_str.split(":")[0]) * 60 + int(local_str.split(":")[1]),
                "epoch_s": int(when_utc.timestamp()),
            }

        # Sunrise at 14:00Z (= 07:00 PDT): frames at -30m, +15m, +90m -> ideal +12m selects +15m!
        sr_utc = dt.datetime(2025, 6, 14, 14, 0, tzinfo=dt.timezone.utc)
        sr_frames = [
            _mk_frame(sr_utc - dt.timedelta(minutes=30), "06:30", "mockcam_2025_06_14_063000.jpg"),
            _mk_frame(sr_utc + dt.timedelta(minutes=15), "07:15", "mockcam_2025_06_14_071500.jpg"),
            _mk_frame(sr_utc + dt.timedelta(minutes=90), "08:30", "mockcam_2025_06_14_083000.jpg"),
        ]
        with (
            patch("planner.weather.webcam_truth._get_phenocam_catalog", return_value=fake_catalog),
            patch("planner.weather.webcam_truth._list_phenocam_daily_frames", return_value=sr_frames),
        ):
            res_sr = _match_phenocam_archive_camera(
                37.6, -122.5, 200.0, req_date, sr_utc, 62.0, False, analyze_pixels=False, phase="sunrise"
            )
            self.assertIsNotNone(res_sr)
            assert res_sr is not None
            self.assertEqual(res_sr["primary_frame"]["timestamp_local"], "07:15")

        # Sunset at 03:30Z next day (= 20:30 PDT): frames at -40m, -15m, +25m -> ideal -12m selects -15m!
        ss_utc = dt.datetime(2025, 6, 15, 3, 30, tzinfo=dt.timezone.utc)
        ss_frames = [
            _mk_frame(ss_utc - dt.timedelta(minutes=40), "19:50", "mockcam_2025_06_14_195000.jpg"),
            _mk_frame(ss_utc - dt.timedelta(minutes=15), "20:15", "mockcam_2025_06_14_201500.jpg"),
            _mk_frame(ss_utc + dt.timedelta(minutes=25), "20:55", "mockcam_2025_06_14_205500.jpg"),
        ]
        with (
            patch("planner.weather.webcam_truth._get_phenocam_catalog", return_value=fake_catalog),
            patch("planner.weather.webcam_truth._list_phenocam_daily_frames", return_value=ss_frames),
        ):
            res_ss = _match_phenocam_archive_camera(
                37.6, -122.5, 200.0, req_date, ss_utc, 298.0, False, analyze_pixels=False, phase="sunset"
            )
            self.assertIsNotNone(res_ss)
            assert res_ss is not None
            self.assertEqual(res_ss["primary_frame"]["timestamp_local"], "20:15")


class TestHrrrS3FallbackBundle(unittest.TestCase):
    def test_build_nwp_bundle_from_hrrr_s3_diurnal_interpolation(self) -> None:
        from planner.weather.nwp_corridor import build_corridor_coords_meta, build_nwp_bundle_from_hrrr_s3

        coords = build_corridor_coords_meta(37.7459, -119.5332, 241.0, 118.0)

        def _mk_station(low: float, mid: float, high: float, temp: float) -> dict:
            return {
                "z_sfc_m": 2500.0,
                "temp_2m_c": temp,
                "rh_2m_pct": 48.0,
                "dewpoint_2m_c": temp - 8.0,
                "lcdc_pct": low,
                "mcdc_pct": mid,
                "hcdc_pct": high,
                "tcdc_tot_pct": max(low, mid, high),
                "vis_m": 45000.0,
                "cape_j_kg": 15.0,
                "hpbl_m": 820.0,
                "temp_950_c": temp + 4.0,
                "temp_925_c": temp + 2.0,
                "temp_850_c": temp - 2.0,
                "temp_700_c": -3.5,
                "temp_500_c": -19.0,
                "temp_300_c": -44.0,
                "rh_975_pct": 52.0,
                "rh_950_pct": 50.0,
                "rh_925_pct": 46.0,
                "rh_700_pct": 35.0,
                "rh_500_pct": 28.0,
                "rh_300_pct": max(20.0, high * 0.85),
                "z_925_m": 780.0,
                "z_850_m": 1510.0,
                "z_700_m": 3100.0,
                "z_500_m": 5690.0,
                "z_300_m": 9380.0,
                "u10_ms": 3.5,
                "v10_ms": -1.2,
                "u850_ms": 5.5,
                "v850_ms": -1.5,
                "u700_ms": 9.0,
                "v700_ms": -2.0,
                "u500_ms": 18.0,
                "v500_ms": -4.0,
                "u300_ms": 30.0,
                "v300_ms": -6.0,
                "u250_ms": 34.0,
                "v250_ms": -7.0,
                "aotk_550nm": 0.08,
            }

        hrrr_primary = {
            "ok": True,
            "cycle_tag": "20261126 19Z f06",
            "stations": [_mk_station(0.0, 0.0, 88.0, 10.0) for _ in coords],
        }
        morning_stations = [_mk_station(20.0, 10.0, 30.0, 4.0) for _ in coords]
        bundle = build_nwp_bundle_from_hrrr_s3(
            hrrr_primary,
            coords,
            37.7459,
            -119.5332,
            "2026-11-26",
            primary_hr_idx=17,
            is_coastal=False,
            hourly_hrrr_stations={6: morning_stations},
            upstream_errors=["HTTP 429"],
        )
        self.assertTrue(bundle["ok"])
        self.assertTrue(bundle["hrrr_s3_primary_fallback"])
        self.assertEqual(len(bundle["point_docs"]), len(coords))
        h0 = bundle["point_docs"][0]["hourly"]
        self.assertEqual(len(h0["cloud_cover_high"]), 24)
        self.assertAlmostEqual(h0["cloud_cover_high"][6], 30.0, places=1)
        self.assertAlmostEqual(h0["cloud_cover_high"][17], 88.0, places=1)
        # Midday hr=12 must smoothly interpolate between hr=6 (30%) and hr=17 (88%)
        self.assertGreater(h0["cloud_cover_high"][12], 55.0)
        self.assertLess(h0["cloud_cover_high"][12], 68.0)
        # Verify extract_hour_sounding parses the HRRR S3 doc cleanly
        snd = extract_hour_sounding(bundle["point_docs"][0], 17, 0.0, 241.0, 37.7459, -119.5332)
        self.assertEqual(snd["cloud_high_pct"], 88)
        self.assertEqual(snd["cloud_low_pct"], 0)


if __name__ == "__main__":
    unittest.main()
