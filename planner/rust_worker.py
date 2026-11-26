"""
Persistent Rust Raycaster Worker Client (`planner/rust_worker.py`).

Maintains a long-lived `./raycaster/target/release/photoplanner worker` process communicating
over line-delimited JSON (NDJSON) stdin/stdout pipes so that:
- `i16` SRTM 1-arcsecond tiles remain resident in an LRU-bounded RAM cache (~26 MB/land tile)
- 1-meter USGS 3DEP `MaxMipmapGrid1m` near-field grids remain cached in memory (zero cloning)
- Per-request subprocess startup, temp-file JSON I/O, and `.hgt.gz` decompression overhead
  drops from ~200-450 ms down to <1 ms on warm requests.

Thread-safe across FastAPI / Uvicorn worker threads via an internal lock (Rust's Rayon already
parallelizes across CPU cores inside each request while sharing one unified SRTM LRU cache),
with automatic crash recovery, stderr draining, and clean `atexit` / FastAPI lifespan shutdown.
"""

from __future__ import annotations

import atexit
from collections import deque
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUST_BIN = ROOT_DIR / "raycaster" / "target" / "release" / "photoplanner"
DEFAULT_ELEVATION_DIR = Path(
    os.environ.get(
        "PHOTOPLANNER_ELEVATION_DIR",
        str(
            Path("/Users/lucasbrambrink/Coding/recreation/recreation/data/elevation")
            if Path("/Users/lucasbrambrink/Coding/recreation/recreation/data/elevation").exists()
            else (ROOT_DIR / "data" / "srtm_cache")
        ),
    )
)
DEFAULT_RECREATION_DB = Path(
    os.environ.get(
        "PHOTOPLANNER_RECREATION_DB",
        str(
            Path("/Users/lucasbrambrink/Coding/recreation/recreation/db.sqlite3")
            if Path("/Users/lucasbrambrink/Coding/recreation/recreation/db.sqlite3").exists()
            else (ROOT_DIR / "data" / "recreation_fallback.sqlite3")
        ),
    )
)


def is_rust_worker_enabled() -> bool:
    """Returns False only when explicitly disabled via `PHOTOPLANNER_DISABLE_RUST_WORKER=1`."""
    flag = os.environ.get("PHOTOPLANNER_DISABLE_RUST_WORKER", "").strip().lower()
    return flag not in ("1", "true", "yes", "on")


class RustWorkerError(RuntimeError):
    """Raised when the persistent Rust worker returns an error or fails to respond."""


class RustWorkerClient:
    """Thread-safe, self-healing NDJSON client for `photoplanner worker`."""

    def __init__(
        self,
        rust_bin: Path | None = None,
        elevation_dir: Path | None = None,
        recreation_db: Path | None = None,
        max_srtm_tiles: int | None = None,
        max_near_grids: int | None = None,
    ) -> None:
        self.rust_bin = Path(rust_bin) if rust_bin is not None else DEFAULT_RUST_BIN
        self.elevation_dir = Path(elevation_dir) if elevation_dir is not None else DEFAULT_ELEVATION_DIR
        self.recreation_db = Path(recreation_db) if recreation_db is not None else DEFAULT_RECREATION_DB
        self.max_srtm_tiles = max(
            1,
            int(
                max_srtm_tiles
                if max_srtm_tiles is not None
                else os.environ.get("PHOTOPLANNER_MAX_SRTM_TILES", "48")
            ),
        )
        self.max_near_grids = max(
            0,
            int(
                max_near_grids
                if max_near_grids is not None
                else os.environ.get("PHOTOPLANNER_MAX_NEAR_GRIDS", "6")
            ),
        )

        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=32)
        self._restarts: int = 0
        self._requests_served: int = 0
        self._started_at: float | None = None

    @property
    def pid(self) -> int | None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            return proc.pid
        return None

    def _drain_stderr(self, stream: Any) -> None:
        try:
            for line in stream:
                cleaned = line.rstrip()
                if cleaned:
                    self._stderr_tail.append(cleaned)
        except Exception:
            pass

    def _close_proc_locked(self, graceful: bool = False, timeout: float = 3.0) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if graceful and proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write('{"op":"shutdown"}\n')
                    proc.stdin.flush()
                except Exception:
                    pass
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.poll() is None:
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2.0)
        finally:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def _spawn_locked(self) -> subprocess.Popen[str]:
        if not self.rust_bin.exists():
            raise RustWorkerError(f"Rust photoplanner binary not found at {self.rust_bin}")
        self.elevation_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.rust_bin),
            "--elevation-dir",
            str(self.elevation_dir),
            "--recreation-db",
            str(self.recreation_db),
            "worker",
            "--max-srtm-tiles",
            str(self.max_srtm_tiles),
            "--max-near-grids",
            str(self.max_near_grids),
        ]
        self._stderr_tail.clear()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._proc = proc
        self._started_at = time.monotonic()
        if proc.stderr is not None:
            t = threading.Thread(
                target=self._drain_stderr,
                args=(proc.stderr,),
                name=f"pp-rust-stderr-{proc.pid}",
                daemon=True,
            )
            t.start()
        return proc

    def _ensure_proc_locked(self) -> subprocess.Popen[str]:
        if self._proc is None:
            return self._spawn_locked()
        if self._proc.poll() is not None:
            self._close_proc_locked(graceful=False)
            self._restarts += 1
            return self._spawn_locked()
        return self._proc

    def _readline_with_timeout(self, stream: Any, timeout: float) -> str:
        if hasattr(select, "select"):
            fd = stream.fileno()
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                raise TimeoutError(f"Rust worker timed out after {timeout:.1f}s")
        return stream.readline()

    def _send_request(self, payload: dict[str, Any], timeout: float = 90.0) -> dict[str, Any]:
        encoded = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._lock:
            last_exc: Exception | None = None
            for attempt in range(2):
                proc = self._ensure_proc_locked()
                assert proc.stdin is not None and proc.stdout is not None
                try:
                    proc.stdin.write(encoded)
                    proc.stdin.flush()
                    raw_line = self._readline_with_timeout(proc.stdout, timeout=timeout)
                    if not raw_line:
                        rc = proc.poll()
                        tail = " | ".join(list(self._stderr_tail)[-4:])
                        raise BrokenPipeError(
                            f"Rust worker closed stdout (exit={rc}, stderr={tail or 'none'})"
                        )
                    resp = json.loads(raw_line)
                    if not isinstance(resp, dict):
                        raise RustWorkerError("Rust worker returned non-object JSON payload")
                    if not resp.get("ok"):
                        raise RustWorkerError(str(resp.get("error") or "Unknown Rust worker error"))
                    self._requests_served += 1
                    return resp
                except (BrokenPipeError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                    last_exc = exc
                    self._close_proc_locked(graceful=False)
                    if attempt == 0:
                        self._restarts += 1
                        continue
                    raise RustWorkerError(f"Rust worker failed after restart: {exc}") from exc
            raise RustWorkerError(f"Rust worker failed: {last_exc}")

    def ping(self, timeout: float = 10.0) -> dict[str, Any]:
        """Pings the persistent worker (spawning it if not yet running) and returns live cache stats."""
        return self._send_request({"op": "ping"}, timeout=timeout)

    def corridor_raycast(
        self,
        lat: float,
        lon: float,
        target_elev_m: float | None,
        sun_az_deg: float,
        sun_el_deg: float,
        batch_input: dict[str, Any] | None = None,
        skip_lut: bool = False,
        near_grid_m: int = 1200,
        az_bins: int = 180,
        alt_bins: int = 32,
        timeout: float = 90.0,
    ) -> dict[str, Any]:
        """Executes a curved-Earth 0-380 km corridor raycast batch on the warm Rust worker."""
        req: dict[str, Any] = {
            "op": "corridor_raycast",
            "lat": float(lat),
            "lon": float(lon),
            "target_asl_m": float(target_elev_m) if target_elev_m is not None else None,
            "sun_az_deg": float(sun_az_deg),
            "sun_el_deg": float(sun_el_deg),
            "near_grid_m": int(near_grid_m),
            "az_bins": int(az_bins),
            "alt_bins": int(alt_bins),
            "skip_lut": bool(skip_lut),
        }
        if batch_input is not None:
            if "corridor" in batch_input:
                req["corridor"] = batch_input["corridor"]
                req["hourly_evals"] = batch_input.get("hourly_evals") or []
            else:
                req["corridor"] = batch_input
                req["hourly_evals"] = []
        resp = self._send_request(req, timeout=timeout)
        return resp["result"]

    def terrain_patch(
        self,
        lat: float,
        lon: float,
        span_m: float = 9000.0,
        grid_size: int = 256,
        near_grid_m: int = 2000,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Evaluates a hybrid 1m + SRTM 3D elevation patch via the warm Rust worker."""
        req = {
            "op": "terrain_patch",
            "lat": float(lat),
            "lon": float(lon),
            "span_m": float(span_m),
            "grid_size": int(grid_size),
            "near_grid_m": int(near_grid_m),
        }
        resp = self._send_request(req, timeout=timeout)
        return resp["result"]

    def health_snapshot(self, probe: bool = False) -> dict[str, Any]:
        """Returns worker health and SRTM i16 LRU memory telemetry without unnecessary cold spawning."""
        running = self._proc is not None and self._proc.poll() is None
        if not running and not probe:
            return {
                "enabled": is_rust_worker_enabled(),
                "running": False,
                "pid": None,
                "srtm_tiles_loaded": 0,
                "srtm_land_tiles_cached": 0,
                "srtm_max_land_tiles": self.max_srtm_tiles,
                "srtm_resident_mb": 0.0,
                "near_grids_cached": 0,
                "max_near_grids": self.max_near_grids,
                "requests_served": self._requests_served,
                "restarts": self._restarts,
            }
        try:
            info = self.ping(timeout=5.0)
            return {
                "enabled": is_rust_worker_enabled(),
                "running": True,
                "pid": info.get("pid"),
                "compute_backend": info.get("compute_backend"),
                "srtm_tiles_loaded": info.get("srtm_tiles_loaded", 0),
                "srtm_land_tiles_cached": info.get("srtm_land_tiles_cached", 0),
                "srtm_max_land_tiles": info.get("srtm_max_land_tiles", self.max_srtm_tiles),
                "srtm_resident_mb": info.get("srtm_resident_mb", 0.0),
                "near_grids_cached": info.get("near_grids_cached", 0),
                "max_near_grids": info.get("max_near_grids", self.max_near_grids),
                "requests_served": info.get("requests_served", self._requests_served),
                "restarts": self._restarts,
            }
        except Exception as exc:
            return {
                "enabled": is_rust_worker_enabled(),
                "running": False,
                "pid": None,
                "error": str(exc),
                "restarts": self._restarts,
            }

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shuts down the persistent Rust worker subprocess."""
        with self._lock:
            self._close_proc_locked(graceful=True, timeout=timeout)


_SINGLETON_LOCK = threading.Lock()
_DEFAULT_WORKER: RustWorkerClient | None = None


def get_rust_worker() -> RustWorkerClient:
    """Returns the process-wide persistent `RustWorkerClient` singleton."""
    global _DEFAULT_WORKER
    if _DEFAULT_WORKER is None:
        with _SINGLETON_LOCK:
            if _DEFAULT_WORKER is None:
                _DEFAULT_WORKER = RustWorkerClient()
    return _DEFAULT_WORKER


def shutdown_rust_worker() -> None:
    """Shuts down the process-wide persistent Rust worker if it was started."""
    global _DEFAULT_WORKER
    with _SINGLETON_LOCK:
        if _DEFAULT_WORKER is not None:
            _DEFAULT_WORKER.shutdown()
            _DEFAULT_WORKER = None


atexit.register(shutdown_rust_worker)
