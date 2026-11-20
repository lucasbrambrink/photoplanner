"""Tests for the persistent Rust `photoplanner worker` daemon and `i16` SRTM LRU cache."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
import time
import unittest

from planner.rust_worker import RustWorkerClient, DEFAULT_ELEVATION_DIR, DEFAULT_RECREATION_DB, DEFAULT_RUST_BIN


class TestPersistentRustWorker(unittest.TestCase):
    def setUp(self) -> None:
        if not DEFAULT_RUST_BIN.exists():
            self.skipTest(f"Rust binary not built at {DEFAULT_RUST_BIN}")
        self.worker = RustWorkerClient(max_srtm_tiles=24, max_near_grids=4)

    def tearDown(self) -> None:
        self.worker.shutdown()

    def test_ping_telemetry_and_lru_config(self) -> None:
        pong = self.worker.ping()
        self.assertTrue(pong["ok"])
        self.assertEqual(pong["op"], "ping")
        self.assertIsInstance(pong["pid"], int)
        self.assertEqual(pong["srtm_max_land_tiles"], 24)
        self.assertEqual(pong["max_near_grids"], 4)
        self.assertEqual(pong["requests_served"], 0)
        snap = self.worker.health_snapshot(probe=False)
        self.assertTrue(snap["running"])
        self.assertEqual(snap["pid"], pong["pid"])

    def test_corridor_raycast_exact_parity_with_oneshot_cli(self) -> None:
        lat, lon = 37.9235, -122.5965
        target_elev_m = 784.0
        sun_az, sun_el = 252.0, 1.5

        worker_res = self.worker.corridor_raycast(
            lat=lat,
            lon=lon,
            target_elev_m=target_elev_m,
            sun_az_deg=sun_az,
            sun_el_deg=sun_el,
            batch_input=None,
            skip_lut=True,
            near_grid_m=1200,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_out:
            out_path = Path(f_out.name)
        try:
            cmd = [
                str(DEFAULT_RUST_BIN),
                "--elevation-dir",
                str(DEFAULT_ELEVATION_DIR),
                "--recreation-db",
                str(DEFAULT_RECREATION_DB),
                "corridor-raycast",
                "--lat",
                f"{lat:.5f}",
                "--lon",
                f"{lon:.5f}",
                "--target-asl-m",
                f"{target_elev_m:.2f}",
                "--sun-az-deg",
                f"{sun_az:.2f}",
                "--sun-el-deg",
                f"{sun_el:.2f}",
                "--near-grid-m",
                "1200",
                "--skip-lut",
                "--json-out",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
            cli_res = json.loads(out_path.read_text(encoding="utf-8"))
        finally:
            if out_path.exists():
                out_path.unlink()

        # Bit-for-bit physics parity across primary ray & overhead canvas
        for key in (
            "origin_asl_m",
            "terrain_horizon_deg",
            "far_field_horizon_deg",
            "terrain_disk_visibility",
            "direct_beam_probability",
            "transmittance_p10",
            "transmittance_p50",
            "transmittance_p90",
            "transmittance_expected",
            "beam_color_kelvin",
            "chappuis_rose_index",
            "total_rayleigh_od_550",
        ):
            self.assertIn(key, cli_res["primary_ray"])
            self.assertAlmostEqual(
                worker_res["primary_ray"][key],
                cli_res["primary_ray"][key],
                places=6,
                msg=f"Mismatch on primary_ray.{key}",
            )
        self.assertEqual(worker_res["primary_ray"], cli_res["primary_ray"])
        self.assertEqual(worker_res["overhead_canvas"], cli_res["overhead_canvas"])
        self.assertIn("worker_meta", worker_res)
        self.assertGreaterEqual(worker_res["worker_meta"]["srtm_resident_mb"], 0.0)

    def test_warm_cache_hit_and_speedup(self) -> None:
        lat, lon = 37.9235, -122.5965
        cold = self.worker.corridor_raycast(
            lat=lat,
            lon=lon,
            target_elev_m=784.0,
            sun_az_deg=252.0,
            sun_el_deg=1.5,
            skip_lut=True,
            near_grid_m=1200,
        )
        self.assertFalse(cold["worker_meta"]["worker_cache_hit"])

        t0 = time.perf_counter()
        warm = self.worker.corridor_raycast(
            lat=lat,
            lon=lon,
            target_elev_m=784.0,
            sun_az_deg=254.0,
            sun_el_deg=0.8,
            skip_lut=True,
            near_grid_m=1200,
        )
        warm_wall_ms = (time.perf_counter() - t0) * 1000.0
        self.assertTrue(warm["worker_meta"]["worker_cache_hit"])
        self.assertEqual(warm["worker_meta"]["srtm_newly_loaded"], 0)
        self.assertTrue(warm["worker_meta"]["near_grid_cache_hit"])
        self.assertLess(warm["timing_ms"]["terrain_load_ms"], 10.0)
        self.assertLess(warm_wall_ms, 150.0)

    def test_terrain_patch_over_worker(self) -> None:
        res = self.worker.terrain_patch(
            lat=37.9235,
            lon=-122.5965,
            span_m=4000.0,
            grid_size=32,
            near_grid_m=1200,
        )
        self.assertEqual(res["grid_size"], 32)
        raw = base64.b64decode(res["elevations_f32_le_b64"])
        self.assertEqual(len(raw), 32 * 32 * 4)
        vals = struct.unpack(f"<{32 * 32}f", raw)
        self.assertAlmostEqual(min(vals), res["min_elevation_m"], places=2)
        self.assertAlmostEqual(max(vals), res["max_elevation_m"], places=2)

    def test_automatic_crash_recovery(self) -> None:
        first_pong = self.worker.ping()
        old_pid = first_pong["pid"]
        self.assertIsNotNone(old_pid)

        # Simulate unexpected worker crash (SIGKILL)
        os.kill(old_pid, signal.SIGKILL)
        # Wait briefly for kernel to reap / mark exited
        deadline = time.monotonic() + 2.0
        while self.worker._proc is not None and self.worker._proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)

        # Next call must transparently respawn the worker and succeed
        second_pong = self.worker.ping()
        self.assertTrue(second_pong["ok"])
        self.assertNotEqual(second_pong["pid"], old_pid)
        self.assertGreaterEqual(self.worker._restarts, 1)


if __name__ == "__main__":
    unittest.main()
