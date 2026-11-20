# syntax=docker/dockerfile:1.7
# Multi-stage production build: compiles Rust `photoplanner` release binary + slim Python ASGI runtime

FROM rust:1.85-slim-bookworm AS rust-builder
WORKDIR /build/raycaster

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY raycaster/Cargo.toml raycaster/Cargo.lock ./
COPY raycaster/crates ./crates

RUN cargo build --release --bin photoplanner \
    && strip target/release/photoplanner

FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8420 \
    PHOTOPLANNER_WORKERS=2 \
    PHOTOPLANNER_RATE_LIMIT_RPM=120 \
    PHOTOPLANNER_SRTM_MAX_TILES=20

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy compiled Rust raycaster binary into expected workspace location
COPY --from=rust-builder /build/raycaster/target/release/photoplanner /app/raycaster/target/release/photoplanner

# Copy application source, static frontend, and calibrated model artifacts
COPY api_server.py server.py ./
COPY planner ./planner
COPY web ./web
COPY scripts ./scripts
COPY data/weather_benchmarks/calibration_model_v1.json ./data/weather_benchmarks/calibration_model_v1.json
COPY data/weather_benchmarks/verification_report.html ./data/weather_benchmarks/verification_report.html

# Non-root user with writable persistent /app/data directory (SRTM tiles, caches, SQLite watchlists/reports)
RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/data/srtm_cache /app/data/3dep_cache /app/data/weather_cache \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8420

VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/api/ready || exit 1

CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8420} --workers ${PHOTOPLANNER_WORKERS:-2} --proxy-headers"]
