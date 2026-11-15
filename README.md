# PhotoPlanner

**PhotoPlanner** is a specialized landscape photography toolkit that predicts **Light Angles**, **Light Notches** (narrow mountain gaps, couloirs, ridgeline keyholes), and **Shadow/Spotlight Envelopes** at **1-meter resolution**, connected directly to **hiking trail networks and scenic points of interest** from the `recreation` ("TrailBinder") database.

## Directory Structure

```text
photoplanner/
├── raycaster/                     # Rust Core Light-Ray & Notch Prediction Service
│   ├── Cargo.toml                 # Cargo workspace
│   └── crates/
│       ├── pp-solar/              # High-precision NOAA/Meeus Solar Ephemeris & Bennett refraction
│       ├── pp-terrain/            # Hybrid 1m USGS 3DEP Bare-Earth DEM + Far-Field SRTM .hgt pyramid
│       │                          # with 7-level Hierarchical Max-Mipmap (1m..64m blocks)
│       ├── pp-raycast/            # CPU-parallel (Rayon) Hierarchical DDA ray-caster, Horizon Profile H(φ),
│       │                          # Light-Notch prominence/FWHM detector, and 2D Shadow/Spotlight rasterizer
│       ├── pp-trails/             # SQLite harness reading `recreation/db.sqlite3` routes, trailheads,
│       │                          # and geotagged photos + Tobler's Hiking Function itinerary builder
│       └── pp-cli/                # Fast CLI (`photoplanner`)
└── planner/
    └── engine_bridge.py           # Python wrapper for integrating `photoplanner` into Django/scripts
```

## Quick Start (Rust CLI)

```bash
cd raycaster

# 1. Scan a 360° horizon at 1m near-field + 45km far-field resolution to find Light Notches
cargo run --release -- notch-scan --lat 37.7156 --lon -119.6776 --near-grid-m 3000

# 2. Generate time-reversed hiking & shooting itineraries from recreation's trail database
cargo run --release -- trail-plan --landmark "Sentinel Dome" --max-miles 10.0

# 3. Render a 2D directional ray-cast shadow & golden-hour spotlight map (PNG)
cargo run --release -- shadow-map --lat 37.7459 --lon -119.5332 --size-m 2400 --out-png ../yosemite_light_map.png

# 4. Query live USGS 3DEP 1-meter bare-earth LiDAR GeoTIFF tiles
cargo run --release -- usgs3dep --lat 37.7156 --lon -119.6776
```
