# PhotoPlanner Product Roadmap & Engineering Task Plan

> **Core Positioning ("The Backcountry Serendipity Engine"):** Rather than competing with Viewfindr or PhotoPills as a generic roadside tripod-spot directory, PhotoPlanner serves **trail-based landscape and nightscape photographers**. It combines **1m LiDAR + SRTM raycasting**, **trail kinematics (Tobler's Hiking Function)**, and **3D atmospheric/astrophotography physics** to tell photographers *when to leave the trailhead*, *what atmospheric phenomena are forming*, and *gentle horizon/light windows along the trail*—without dictating rigid compositions.

---

## 1. Competitive Differentiation Matrix

| Capability | Viewfindr | PhotoPills | AllTrails / Gaia | **PhotoPlanner (Target Niche)** |
| :--- | :--- | :--- | :--- | :--- |
| **Solar / Ray Geometry** | 2D azimuth + cloud model | 2D map pins + AR | None | **1m LiDAR + 45 km SRTM 3D Light Notches & Shadow Envelopes** |
| **Afterglow / Alpenglow** | Cloud layer heuristic | None | None | **3-Station Ray-Corridor Transmittance + DEM Horizon Occlusion** |
| **Inversion / Fog Height** | Regional fog-top estimate | None | None | **LCL / Pressure-Level Inversion Intersected with 3D Trail Elevation** |
| **Astrophotography** | Cloud cover + Aurora Kp | Moon phase + Galactic core | None | **Jet Stream Seeing (`250hPa`), Aerosol Haze (AOD), Terrain-Blocked Moon** |
| **Trail & Pacing Logistics** | Static roadside spots | None | Distance & elevation | **Time-Reversed Trailhead Departure + Gentle Along-the-Trail Nudges** |

---

## 2. Phased Engineering Roadmap & Task Breakdown

### Phase 1: Astrophotography Physics & Celestial Ephemeris
**Goal:** Upgrade `raycaster/crates/pp-solar` and `server.py` to model upper-atmospheric seeing, high-altitude haze/smoke, and terrain-occluded lunar/Milky Way geometry.

- [x] **Task 1.1 — Lunar & Galactic Core Ephemeris in `pp-solar`**
  - Add `lunar_position(lat, lon, dt)` returning lunar azimuth, apparent elevation, illuminated fraction (0.0–1.0), and phase angle using Meeus Ch. 47 equations in `raycaster/crates/pp-solar/src/lib.rs`.
  - Add `galactic_core_position(lat, lon, dt)` (Sagittarius A*: RA 17h 45.7m, Dec -29° 00') to track Galactic Center azimuth and elevation across the night.
- [x] **Task 1.2 — Topographic Moonrise/Moonset & "Moonlight Alpenglow" Windows in `pp-raycast`**
  - Intersect the lunar trajectory with the 360° horizon profile `H(phi)` in `raycaster/crates/pp-raycast/src/lib.rs` to compute **true terrain moonrise and moonset** (often 30–90 minutes different from flat-horizon tables in deep valleys).
  - Detect **"Shadowed Astro Windows"** (moon is geometrically above the flat horizon, or setting behind a local ridge, casting the tripod into shadow while leaving high peaks or the Galactic Core visible).
  - Detect **"Lunar Alpenglow"** windows: when lunar illumination is 15%–45% and low lunar rays pass through a terrain notch to side-light a monolith while the sky remains dark enough for stars.
- [x] **Task 1.3 — Upper-Atmosphere Seeing & Jet Stream Turbulence API (`server.py`)**
  - Extend the Open-Meteo query in `server.py` to fetch pressure-level winds: `wind_speed_300hPa`, `wind_speed_250hPa` (~30,000–34,000 ft jet stream), and `wind_speed_700hPa` (~10,000 ft mountain crest).
  - Compute an **Astronomical Seeing Index (0–100)** penalized by high upper-tropospheric wind shear (`|v_250hPa - v_700hPa|`) and surface thermal gradients, warning photographers when star bloating/scintillation will degrade tracked or telephoto nightscapes.
- [x] **Task 1.4 — Upper-Atmospheric Haze, Wildfire Smoke & Transparency (`server.py`)**
  - Integrate the free **Open-Meteo Air Quality API** (`https://air-quality-api.open-meteo.com/v1/air-quality`) for `aerosol_optical_depth` (AOD at 550nm), `dust`, and `pm2_5`.
  - Compute a **Sky Transparency Score (0–100)**:
    - `AOD < 0.08`: Crystalline Alpine Sky (Grade A)
    - `0.08 <= AOD < 0.20`: Mild Scattering / Good Milky Way Contrast (Grade B)
    - `AOD >= 0.25`: High Haze / Wildfire Smoke Warning (star halos, washed-out sky glow)

---

### Phase 2: Terrain-Intersected Meteorology (Viewfindr Parity + 3D Upgrade)
**Goal:** Evolve the weather engine in `server.py` from a single-point + 3-station cloud check into a **topographic weather model** that interacts with trail elevations.

- [ ] **Task 2.1 — 3D Valley Inversion & Cloud-Sea Altitude Predictor**
  - Query Open-Meteo boundary-layer and pressure-level temperature/humidity (`temperature_925hPa`, `temperature_850hPa`, `temperature_700hPa`, `relative_humidity_925hPa`, `relative_humidity_850hPa`, and `boundary_layer_height`).
  - Detect thermal inversions (`T_850hPa > T_surface` with high sub-inversion RH and calm winds < 6 mph) and output both:
    - **Fog Base Elevation (ft ASL)** and **Fog Top Elevation (ft ASL)**.
    - **Trail Breakout Milepost:** Cross-reference the fog-top elevation against the route geometry in `pp-trails` to tell the hiker the exact mile marker where the trail climbs above the cloud sea.
- [ ] **Task 2.2 — Topographic Lake Reflection ("Mirror Index")**
  - Combine `wind_speed_10m` and `wind_gusts_10m` with `wind_direction_10m` and the local horizon profile `H(phi_wind)` from `pp-raycast`.
  - If a lake or tarn sits in a wind-sheltered basin where the upwind ridge subtends an elevation angle > 12°, boost the **Water Reflection Calmness Score** even when regional ridge winds are moderate.
- [ ] **Task 2.3 — DEM-Occluded 3-Station Solar Ray Corridor**
  - Enhance `compute_weather_and_alpenglow` in `server.py` so that the 18 km and 75 km ray altitude checks also verify that the solar ray clears the far-field SRTM terrain profile before reaching the overhead cloud canvas.

---

### Phase 3: "Gentle Nudges" & Trail Serendipity Engine
**Goal:** Expand `raycaster/crates/pp-trails/src/lib.rs` so trail itineraries feel like a knowledgeable local hiking partner rather than a rigid checklist.

- [ ] **Task 3.1 — Continuous Trail-Corridor Horizon & Openness Scanner (`pp-trails`)**
  - Sample trail polylines every 150–250 meters along `TrailRouteRecord.geometry`.
  - Compute a lightweight openness and landmark visibility metric at each sample to identify **"Breakout Zones"** (where a hiker transitions from forest/canyon floor to an open aspect facing a major landmark or solar/lunar azimuth).
- [ ] **Task 3.2 — Non-Prescriptive "Trail Nudge" Generator**
  - Introduce a `TrailNudge` struct in `pp-trails` with four gentle categories:
    1. **Horizon Opening Nudge:** *"Around Mile 1.9 (4,850 ft), the south rim opens toward Half Dome—good stretch to slow down if golden hour light is building."*
    2. **Light Notch Window Nudge:** *"Between 6:18 PM and 6:42 PM along the switchbacks at Mile 2.4–2.7, the sun threads the Cathedral notch."*
    3. **Micro-Climate / Atmospheric Nudge:** *"Meadow crossing at Mile 0.7 holds high morning moisture (RH 91%, wind 2 mph)—watch for low radiation mist at dawn."*
    4. **Pacing & Headlamp Safety Nudge:** *"Staying at the summit through Blue Hour (7:40 PM) means 3.1 miles of descent after civil twilight; expect trailhead return around 9:05 PM with a headlamp."*
- [ ] **Task 3.3 — Custom GPX / GeoJSON Trail Upload**
  - Add an endpoint `POST /api/trail-upload` in `server.py` that accepts any user `.gpx` or `.geojson` track (from Gaia, CalTopo, or AllTrails) and runs the exact same `pp-trails` Tobler timing + light-notch + nudge analysis without requiring the route to exist in `recreation/db.sqlite3`.

---

### Phase 4: Web UI & Side-Hustle Product Packaging
**Goal:** Package the capabilities in `web/index.html` and `web/app.js` into a cohesive, niche-defining product experience with near-zero recurring infrastructure cost.

- [ ] **Task 4.1 — Astro & Atmospheric Dashboard Panel (`web/`)**
  - Add a toggleable **"Nightscape & Upper Atmosphere"** mode to the UI alongside Sunrise/Sunset, displaying:
    - Jet Stream Seeing (`250hPa` mph), Aerosol Optical Depth (AOD), and True Terrain Moonrise/Moonset time vs. flat-horizon time.
    - Interactive timeline slider showing Solar Golden Hour -> Blue Hour -> Terrain Moonset -> Galactic Core Window.
- [ ] **Task 4.2 — Printable / Offline "Backcountry Field Card" Export**
  - Build a one-click **"Export Trail Field Card" (PDF / Mobile Image / JSON)** for hikers entering zero-cell-service wilderness areas:
    - Trailhead departure time, elevation profile with milepost markers, 2–3 gentle trail nudges, and exact terrain sunrise/sunset/moonset bearings.
- [ ] **Task 4.3 — Grid-Cell Weather Caching & Cost Guardrails (`server.py`)**
  - Round weather/air-quality lookup coordinates to 0.05° (~5 km) grid cells with a 60-minute TTL cache in SQLite/disk so that 1,000 concurrent users querying Yosemite trails consume only a handful of free Open-Meteo API calls per hour.
- [ ] **Task 4.4 — SEO Landing Pages for Iconic National Park Trails (Growth Engine)**
  - Pre-render static/cached trail light guides for the top 50 trails in `recreation/db.sqlite3` (e.g., *Sentinel Dome, Mist Trail, Clouds Rest, Cathedral Lakes, Parker Lake*) showing annual light-notch calendars and seasonal astro windows to drive organic search traffic from photographers planning trips.
