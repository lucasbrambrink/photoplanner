# PhotoPlanner Product Roadmap & Engineering Task Plan

> **Core Positioning ("3D Terrain-Coupled Light Intelligence"):** Flat-grid sunset apps (SunsetWx, Alpenglow, Viewfindr) forecast clouds over a 2D pixel and ignore terrain; geometry apps (PhotoPills, TPE) ignore live atmosphere. PhotoPlanner fuses **1m LiDAR + SRTM raycasting**, **380 km curved-Earth radiative corridor physics**, and **calibrated probabilistic meteorology** into two tightly integrated workflows:
> 1. **3D Regional "Light-Hunting" Engine (Inverse Raycasting):** Pre-solves hundreds of regional peaks, overlooks, and trails to rank *where* rare alignments (cloud-sea breakouts, direct granite alpenglow, canyon notch spotlights, open-horizon cirrus burners) are actually forming tonight or this weekend—and explains *why* other spots will bust.
> 2. **Trail & Subject "Will This Shot Happen?" 3D Simulator:** Pick any peak, trail, or uploaded GPX track + lens focal length to inspect exact raycasted shadow vs. light timing, fog-sea elevation bands slicing through the trail (with breakout mileposts), calibrated hit/bust odds, and time-reversed trailhead departure logistics.

---

## 1. Competitive Differentiation Matrix

| Capability | Viewfindr / Alpenglow / SunsetWx | PhotoPills / TPE | AllTrails / Gaia | **PhotoPlanner (Unified Moat)** |
| :--- | :--- | :--- | :--- | :--- |
| **Solar / Terrain Light Geometry** | Flat 2D grid cell | 2D pins / static 3D shadows (no weather) | None | **1m LiDAR + SRTM 3D Light Notches, Shadow Envelopes & Corridor Occlusion** |
| **Afterglow / Alpenglow Physics** | Local cloud heuristic (blind to distant horizon wall) | None | None | **0–380 km Curved-Earth Spectral Corridor + Calibrated Probabilistic Transmittance (`P10/P50/P90`)** |
| **Inversion / Cloud-Sea Height** | Coarse flat-grid fog flag | None | None | **Probabilistic Inversion-Lid Altitude Intersected with 3D Terrain & Trail Mileposts** |
| **Regional Discovery ("Where Tonight?")** | Generic regional heatmap | None | Trail directory | **Inverse-Raycasted Regional Ranking of Specific Peaks/Trails Catching Light or Breaking Above Fog** |
| **Astrophotography** | Cloud cover + Kp | Moon/Milky Way ephemeris | None | **Jet Stream Seeing (`250 hPa`), AOD Haze, Terrain-Blocked Moon & Lunar Alpenglow** |
| **Trail & Field Logistics** | Static spots | None | Distance & elevation | **GPX/Trail Breakout Scanner, Time-Reversed Departure & Offline Field Cards** |

---

## 2. Phased Engineering Roadmap & Task Breakdown

### Phase 0: Core Engine Integrity, Physics Fixes & Out-of-Sample Calibration *(Foundation — Active)*
**Goal:** Make the underlying meteorological inputs, 3D radiative raycaster, and probabilistic verification genuinely rock-solid before scaling product features on top.

- [x] **Task 0.1 — Fix Critical Meteorological Ingestion & Physics Bugs**
  - Fix `h5py` signed `int16` wraparound (`_Unsigned="true"`) on GOES-R `ABI-L2-ACHAC` (`HT`) and `ABI-L2-CODC` (`COD`) in `planner/weather/goes_nowcast.py` so cirrus above 10 km is preserved and parallax-rectified.
  - Apply twilight `C13 - C07` fog BTD symmetrically across all 3 causal GOES frames (`t0`, `t-15m`, `t-30m`), align 30-minute Eulerian vs. Lagrangian pinning timescales, and stop flooring real thin-cirrus `CODC` optical depths upward.
  - Rotate grid-relative NOAA HRRR Lambert Conformal Conic (`uvRelativeToGrid=1`) winds to true Earth-relative `(u_east, v_north)`, fix `dpt2m_k` Kelvin fallback, stop double-multiplying `TCOLW` by `low_cf`, filter underground pressure levels, and fix 64-bit GRIB2 bit unpacking in `planner/weather/hrrr_s3.py`.
  - Fix live NWS `VV///` fog parsing and humid-mist Koschmieder AOD false positives in `planner/weather/station_obs.py`; fix mountain station AGL/ASL cloud-band blind gaps and synoptic freshness ordering in `planner/weather/raob_sounding.py`; fix sunrise webcam windowing and NSRDB domain/Dec-31 bounds in `planner/weather/webcam_truth.py`.
- [x] **Task 0.2 — Fix 3D Radiative Transfer & Geometric Raycasting Bugs (`raycaster/`)**
  - Fix Rayleigh molecular density elevation double-counting, restrict surface-inversion refraction `k` to the lower troposphere instead of bending stratospheric rays by ~3 km, renormalize inversion-trapped aerosol profiles conservatively, and eliminate overhead cloud-canvas self-attenuation in `raycaster/crates/pp-raycast/src/corridor.rs`.
  - Bring macOS Metal GPU shaders (`gpu_metal.rs`) to parity with Linux CPU Delta-Eddington and slant-airmass math, and fix Max-Mipmap empty-space cell-exit stepping (`pp-raycast/src/lib.rs`) + curved-Earth `aim_z` sightline bias (`pp-trails`, `pp-wasm`) + HTTP 403 `.ocean` cache poisoning (`pp-terrain`).
- [x] **Task 0.3 — Eliminate Benchmark Leakage & Build Out-of-Sample Probabilistic Calibration**
  - Remove Day-0 pressure-level truth leakage from `+24h/+48h/+72h` previous-run hindcasts in `planner/weather/nwp_corridor.py` and enforce realistic operational RAOB ingestion lag (`+60 min` post-nominal) in `planner/weather/raob_sounding.py` & `planner/weather/backtest.py`.
  - Clearly separate **independent physical ground truth** (2 km NSRDB DNI, SURFRAD/ASOS, operationally lagged RAOB) from internal model self-consistency diagnostics.
  - Build `planner/weather/calibration.py` for strict out-of-sample (leave-region-out & chronological) probabilistic post-processing: calibrated `P10 / P50 / P90` transmittance quantiles (~80% empirical coverage), calibrated summit-above-fog breakout probabilities, and proper scoring rules (CRPS, Pinball Loss, Brier Skill Score, ECE).
  - Add automated `pytest` + `cargo test` regression suites verifying every physics, geometry, and anti-leakage invariant.
- [ ] **Task 0.4 — Expand Out-of-Sample Multi-Regime Benchmark (500–1,000+ Events)** *(169-event / 676-row cross-year 2024 vs 2025 benchmark + tier-weighted 3D-shadow-clean `cal_v2` verified; 504-event resumable suite ready)*
  - **Shipped:**
    - Built the 504-event / 2,016-row `cross_year` suite (`benchmark_suite.py`) spanning **all 7 NOAA SURFRAD 1-min pyrheliometer stations** (`tbl`, `dra`, `fpk`, `sxf`, `bon`, `gwn`, `psu`), 8 iconic CONUS landscape sites (fixed 2022–2024 NSRDB `meta`-table HDF5 schema with fast verified grid reuse), and CA24 across 2024 (`>= 2024-02-10` for genuine `previous-runs-api` Day-1/Day-3 hindcasts) and 2025, with **Cross-Year Holdout CV** (`train 2024 → blind test 2025` & `train 2025 → blind test 2024` in `calibration.py`), multi-site HRRR CONUS cycle fan-out + `np.savez` fast caching (`41.6×` faster tile writes), hardened SURFRAD retries, resumable `--max-uncached` batching, and automatic disk-bounded raw cache pruning.
    - **Shipped `cal_v2_tier_weighted` 3D-Shadow Exclusion & Tier Reliability Weighting (`planner/weather/calibration.py`)**: excludes steep-mountain 3D shadow rows (`terrain_horizon_deg >= 3.5°` or `blocker == "terrain"`, `120` rows) when scored against flat-horizon 2 km NSRDB satellite DNI, weights Tier 1A NOAA SURFRAD pyrheliometers `1.0` vs Tier 1B open-horizon NSRDB `0.45`, and reports `by_terrain_shadow_audit` (`open_horizon_clean` vs `flat_nsrdb_3d_shadow`) transparently in JSON/HTML scoreboards.
    - **Verified across 169 independent events / 676 leak-free forecast rows across 21 US sites** (`61` in 2024 + `108` in 2025; `344` Tier 1A SURFRAD + `212` open-horizon NSRDB + `120` 3D-shadowed NSRDB audit rows):
      - **Pure Tier 1A NOAA SURFRAD (`N=344`, all 7 stations, 2024+2025)**: Chronological CV MAE `16.97 → 16.48 pp`, CRPS `12.42 → 11.92 pp`, Bias `-8.59 → +0.64 pp`; Cross-Year (`2024 ↔ 2025`) MAE `16.97 → 16.60 pp`, CRPS `12.42 → 11.88 pp`, Bias `-8.59 → -1.69 pp`.
      - **Clean Open-Horizon (`N=556`)**: LSO CV MAE `21.24 → 19.23 pp`, CRPS `15.60 → 13.98 pp`, 80% PI Cov `71.6% → 80.6%`; Chrono CV MAE `21.24 → 18.27 pp`, CRPS `15.60 → 13.32 pp`, 80% Cov `82.4%` (vs `flat_nsrdb_3d_shadow` `N=120`: raw MAE `35.55 pp`, Bias `-30.66 pp`).
      - Updated production `calibration_model_v1.json` (`cal_v2_tier_weighted`, `N=556` clean training rows) and refreshed `verification_report.html`.
  - **Next:** Complete the remaining `335/504` cached events (paused by Open-Meteo's free daily 10k-call cap — resumes automatically with `OPEN_METEO_API_KEY` or tomorrow's quota via `--suite cross_year --cache-tag multi92_v3 --max-uncached 35`).

---

### Phase 1: Astrophotography Physics & Celestial Ephemeris *(Completed)*
**Goal:** Upgrade `raycaster/crates/pp-solar` and backend weather pipelines to model upper-atmospheric seeing, high-altitude haze/smoke, and terrain-occluded lunar/Milky Way geometry.

- [x] **Task 1.1 — Lunar & Galactic Core Ephemeris in `pp-solar`**
  - `lunar_position(lat, lon, dt)` and `galactic_core_position(lat, lon, dt)` (Sagittarius A*) in `raycaster/crates/pp-solar/src/lib.rs`.
- [x] **Task 1.2 — Topographic Moonrise/Moonset & "Moonlight Alpenglow" Windows in `pp-raycast`**
  - True terrain moonrise/moonset, shadowed astro windows, and lunar alpenglow notch detection in `raycaster/crates/pp-raycast/src/lib.rs`.
- [x] **Task 1.3 — Upper-Atmosphere Seeing & Jet Stream Turbulence**
  - Pressure-level jet shear (`250/300/700 hPa`) and Astronomical Seeing Index (0–100).
- [x] **Task 1.4 — Upper-Atmospheric Haze, Wildfire Smoke & Transparency**
  - 550 nm AOD + PM2.5 Sky Transparency grading (A/B/smoke warning).

---

### Phase 2: Terrain-Intersected Meteorology & Commercial-Safe Production Engine *(Completed)*
**Goal:** Evolve the weather and raycaster stack into a fast, Linux-ready, commercial-safe production service with low unit cost.

- [x] **Task 2.1 — Probabilistic 3D Inversion & Trail Breakout Milepost Predictor**
  - Combine calibrated inversion/fog-top probability bands (`P10 / P50 / P90` ASL) with `pp-trails` route elevation profiles to output exact **Trail Breakout Mileposts** and summit-above-fog odds (`planner/weather/terrain_coupling.py` + `corridor_solver.py`).
- [x] **Task 2.2 — Topographic Lake Reflection ("Mirror Index")**
  - Combine `wind_speed_10m`, gusts, direction, and upwind terrain horizon sheltering `H(phi_wind)` from `pp-raycast` to score alpine lake reflection calmness (`compute_lake_mirror_index`).
- [x] **Task 2.3 — DEM-Occluded 380 km Solar Ray Corridor**
  - Curved-Earth spectral + 3D stochastic cloud-chord radiative corridor with far-field SRTM horizon occlusion (`corridor.rs`).
- [x] **Task 2.4 — Commercial-Safe Data Stack & Persistent Linux Service**
  - FastAPI/Uvicorn ASGI server (`api_server.py`) with thread-safe `0.05°` (~5 km) spatial + 25 m elevation `GridTTLCache`, `/api/health` live worker & commercial-safe telemetry, `/docs` OpenAPI schema, optional `OPEN_METEO_API_KEY` commercial endpoint routing (`nwp_corridor.py`), and a persistent NDJSON Rust worker daemon (`photoplanner worker` + `planner/rust_worker.py`) with an `i16` SRTM LRU cache (50% RAM cut) cutting warm corridor raycasts from **1,783 ms → 2.31 ms** (**771× speedup**).
  - Commercial-safe data stack: ALERTCalifornia proxying disabled by default (`PHOTOPLANNER_ENABLE_ALERTCA_PROXY=1` opt-in; HTTP `403` in commercial-safe mode across `webcam_truth.py`, `server.py`, and `api_server.py`), unconditional SF Peninsula bypasses retired in `server.py` in favor of the general Rust engine with graceful fallback, and full `api_server.py` route/parameter parity verified.

---

### Phase 3: "Will This Shot Happen?" 3D Simulator & Trail Serendipity Engine *(Completed)*
**Goal:** Turn the raycaster + trails engine into a fast, mobile-friendly simulator replacing the 4-app workflow (PhotoPills + CalTopo + Windy + webcams).

- [x] **Task 3.1 — Continuous Trail-Corridor Horizon & Openness Scanner (`pp-trails`)**
  - Sample trail polylines every 150–250 m (fixing loop-trail truncation via `effective_scan_len`) to identify **"Breakout Zones"** where hikers emerge into open sightlines toward lit subjects or above the inversion lid (`scan_trail_corridor_breakouts`).
- [x] **Task 3.2 — Non-Prescriptive "Trail Nudge" Generator**
  - Generate four gentle, actionable categories in `pp-trails`: Horizon Opening, Light Notch Window, Micro-Climate/Fog Breakout, and Pacing/Headlamp Safety nudges (`generate_trail_nudges`).
- [x] **Task 3.3 — Custom GPX / GeoJSON Trail Upload (`POST /api/trail-upload`)**
  - Allow photographers to upload any `.gpx` or `.geojson` route and run full Tobler timing + fog breakout + light-notch + corridor transmittance analysis + Offline Field Card for arbitrary locations (`planner/gpx_trail_analyzer.py`, `POST /api/trail-upload`).
- [x] **Task 3.4 — Lightweight Raycasted 3D Terrain & Lens Viewfinder UX**
  - Heavy WebGL2 Navier-Stokes fluid lab frozen & opt-in by default (`🧪 Fluid Lab: Off`); fast, mobile-compatible 3D terrain scene active out-of-the-box with **raycasted alpenglow/shadow envelopes**, **probabilistic 3D Fog-Sea elevation slice (`P10 / P50 / P90 / Off`)**, **exact full-frame 35mm camera focal-length viewfinder (`16 / 24 / 35 / 50 / 70 / 200 mm`)**, full **`00:00–24:00` golden-hour + astro timeline scrubber**, and one-click **Offline Backcountry Field Card** HTML/Print export (`web/index.html`, `web/app.js`, `web/styles.css`).

---

### Phase 4: 3D Regional "Light-Hunting" Engine (Inverse Raycasting) & Launch *(Completed)*
**Goal:** Differentiate decisively from flat-grid sunset radars by answering *"Where within driving distance are terrain + light + atmosphere aligning tonight?"*

- [x] **Task 4.1 — Regional Inverse-Raycasting Pre-Solver (Core Engine)**
  - Precompute/solve regional corridor transmittance + calibrated probabilities + fog-lid breakout odds + lake Mirror Index across a curated 40-spot multi-region catalog (`planner/weather/regional_hunter.py`, `/api/regional-hunt`), ready to scale to 200–500 spots per region.
- [x] **Task 4.2 — Ranked Phenomenon Discovery & Causal Bust Explanations (Core Engine)**
  - Rank tonight / sunrise opportunities by specific photographic phenomenon (*Cloud-Sea Breakout*, *Direct Alpine Alpenglow*, *Open-Horizon Cirrus Burner*, *Alpine Mirror Reflection*, *Moody Storm-Break Shafts*, *Clear Dark-Sky Astro*) with plain-English causal bust attributions (`classify_spot_phenomena_and_attribution`).
- [x] **Task 4.3 — Proactive Calibrated Alerts & Public Accuracy Scoreboard**
  - User saved spots & regional watchlists persisted in SQLite triggering 24h / 3h calibrated `TRIGGER` / `WATCH` / `SUPPRESS` alerts only when calibrated probability exceeds user thresholds, with plain-English bust reasons (`planner/weather/alert_engine.py`, `/api/watchlists`, `/api/alerts/evaluate`).
  - Publish a live, transparent **Verification Scoreboard** grading hits and misses against independent NOAA SURFRAD pyrheliometers, 2 km NSRDB GOES irradiance, and NWS RAOB soundings to build trust (`/api/backtest-report`, `data/weather_benchmarks/verification_report.html`).
- [x] **Task 4.4 — Dynamic SEO Light Guides (Organic Growth Engine) & B2B API Tier**
  - Daily-updated server-rendered SEO landing pages (`/guides`, `/guides/{slug}`) across 40 iconic US destinations in 7 regions with Schema.org JSON-LD (`TouristAttraction` + `FAQPage`), OpenGraph/Twitter tags, `/sitemap.xml`, `/robots.txt`, deep links into the 3D Lens Viewfinder, static CDN pre-generator (`scripts/generate_seo_guides.py`), and versioned B2B JSON API (`/api/v1/guides`, `/api/v1/guides/{slug}`) with `PHOTOPLANNER_B2B_API_KEYS` tier gating (`planner/seo_guides.py`, `api_server.py`).

---

### Phase 5 (Future Horizon): Wildlife & Seasonal Phenology — Nature Photographer One-Stop-Shop
**Goal:** Once the landscape light-forecasting & 3D terrain engine is rock-solid, expand modularly into a complete nature photography field companion without diluting core meteorological accuracy.

- [ ] **Task 5.1 — Seasonal Phenology Layer (Wildflowers, Fall Foliage & Waterfall Flow)**
  - Couple NOAA precipitation/snowpack (SNOTEL) + USGS stream gauges + Growing Degree Days (GDD) to predict peak wildflower superblooms, autumn aspen/larch color windows, and ephemeral waterfall flow rates alongside golden-hour light.
- [ ] **Task 5.2 — Ethical Wildlife Activity & Migration Intelligence**
  - Integrate public-domain ecological & observation signals (eBird migration radar/BirdCast, iNaturalist seasonal ranges, tidal charts for coastal marine life/seabirds, and crepuscular wind-scent / sun-angle positioning) while enforcing strict location-privacy ethics for sensitive/endangered species.
- [ ] **Task 5.3 — Unified Field Trip Planner**
  - Combine 3D light windows, trail breakout logistics, tidal/stream conditions, and seasonal wildlife/phenology peaks into a single multi-day trip itinerary for nature photographers.
