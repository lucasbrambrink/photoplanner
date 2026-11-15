use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use clap::{Parser, Subcommand};
use pp_raycast::{compute_horizon_profile, compute_shadow_and_light_map, detect_light_notches};
use pp_solar::solar_position;
use pp_terrain::{
    meters_per_deg_lon, query_usgs_3dep_1m_tiles, ElevationSurface, HybridTerrain,
    MaxMipmapGrid1m, SrtmCache, METERS_PER_DEG_LAT,
};
use pp_trails::{
    analyze_custom_tripod_point, build_itinerary_for_route,
    find_best_viewpoints_and_hikes_for_landmark, find_best_viewpoints_for_seed,
    RecreationDbHarness,
};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "photoplanner",
    version = "0.1.0",
    about = "1m LiDAR Light-Notch & Trail Itinerary Prediction Engine"
)]
struct Cli {
    /// Path to SRTM .hgt directory (defaults to recreation/data/elevation)
    #[arg(
        long,
        default_value = "/Users/lucasbrambrink/Coding/recreation/recreation/data/elevation"
    )]
    elevation_dir: PathBuf,

    /// Path to recreation SQLite database (defaults to recreation/db.sqlite3)
    #[arg(
        long,
        default_value = "/Users/lucasbrambrink/Coding/recreation/recreation/db.sqlite3"
    )]
    recreation_db: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan a 360° or solar-window horizon from a coordinate using the 1m Max-Mipmap + SRTM pyramid
    /// to detect Light Notches and exact dates/times of solar alignment
    NotchScan {
        /// Latitude in decimal degrees (default: Tunnel View / Yosemite Valley 37.7156)
        #[arg(long, default_value_t = 37.7156, allow_hyphen_values = true)]
        lat: f64,
        /// Longitude in decimal degrees (default: -119.6776)
        #[arg(long, default_value_t = -119.6776, allow_hyphen_values = true)]
        lon: f64,
        /// Near-field 1m grid size in meters (e.g., 3000 = 3km x 3km = 9,000,000 1m cells)
        #[arg(long, default_value_t = 3000)]
        near_grid_m: usize,
        /// Azimuth resolution in degrees
        #[arg(long, default_value_t = 0.1)]
        az_step_deg: f64,
        /// Target calendar year for alignment search
        #[arg(long, default_value_t = 2026)]
        year: i32,
        /// Optional path to export JSON output
        #[arg(long)]
        json_out: Option<PathBuf>,
    },

    /// Connect to the recreation database, scan trails & scenic landmarks, and generate
    /// time-reversed hiking itineraries around predicted Light Notch alignments
    TrailPlan {
        /// Landmark or trail keyword filter (e.g. "Sentinel Dome", "Half Dome", "Yosemite Fall")
        #[arg(long, default_value = "Sentinel Dome")]
        landmark: String,
        /// Maximum round-trip hike distance in miles
        #[arg(long, default_value_t = 12.0)]
        max_miles: f64,
        /// Target calendar year
        #[arg(long, default_value_t = 2026)]
        year: i32,
        /// Optional JSON output path
        #[arg(long)]
        json_out: Option<PathBuf>,
    },

    /// Render a 2D directional ray-cast shadow & golden-hour light-shaft map (PNG + stats)
    ShadowMap {
        /// Center latitude (default: Half Dome / Yosemite Valley 37.7459)
        #[arg(long, default_value_t = 37.7459, allow_hyphen_values = true)]
        lat: f64,
        /// Center longitude (default: -119.5332)
        #[arg(long, default_value_t = -119.5332, allow_hyphen_values = true)]
        lon: f64,
        /// Size of the 1m grid in meters (e.g., 2400 = 2.4km x 2.4km)
        #[arg(long, default_value_t = 2400)]
        size_m: usize,
        /// Resolution step in meters per pixel for output image (e.g., 2m or 4m)
        #[arg(long, default_value_t = 3)]
        step_m: usize,
        /// ISO-8601 UTC timestamp (default: golden hour sunset in Yosemite)
        #[arg(long, default_value = "2026-10-18T00:52:00Z")]
        datetime_utc: String,
        /// Output PNG file path
        #[arg(long, default_value = "shadow_light_map.png")]
        out_png: PathBuf,
    },

    /// Query the live USGS 3DEP 1-meter National Map API for available 1m bare-earth LiDAR GeoTIFF tiles
    Usgs3dep {
        #[arg(long, default_value_t = 37.7156, allow_hyphen_values = true)]
        lat: f64,
        #[arg(long, default_value_t = -119.6776, allow_hyphen_values = true)]
        lon: f64,
    },

    /// Pinpoint the Top K best trail viewpoints (VP1, VP2, VP3) and hikes to photograph a specific
    /// landmark (e.g. "Half Dome") during Golden Hour, with exact trailhead departure & arrival windows
    ShootLandmark {
        /// Target landmark to photograph (e.g. "Half Dome", "Upper Yosemite Fall", "Bridalveil Fall", "Clouds Rest")
        #[arg(long, default_value = "Half Dome")]
        landmark: String,
        /// Target shoot date (YYYY-MM-DD)
        #[arg(long, default_value = "2026-10-15")]
        date: String,
        /// Maximum hike distance in miles
        #[arg(long, default_value_t = 16.0)]
        max_miles: f64,
        /// Number of distinct top viewpoints/hikes to return (VP1, VP2, VP3...)
        #[arg(long, default_value_t = 3)]
        top_k: usize,
        /// Optional JSON output path
        #[arg(long)]
        json_out: Option<PathBuf>,
    },

    /// Export a blended 1m 3DEP + SRTM 3D elevation patch (JSON metadata + Float32 grid) for the WebGL 3D Viewer
    TerrainPatch {
        #[arg(long, default_value_t = 37.74554, allow_hyphen_values = true)]
        lat: f64,
        #[arg(long, default_value_t = -119.53315, allow_hyphen_values = true)]
        lon: f64,
        /// Physical span of the 3D terrain box in meters (e.g. 9000 = 9km x 9km)
        #[arg(long, default_value_t = 9000.0)]
        span_m: f64,
        /// Grid resolution (grid_size x grid_size vertices)
        #[arg(long, default_value_t = 256)]
        grid_size: usize,
        /// Optional output file path for JSON terrain payload
        #[arg(long)]
        json_out: Option<PathBuf>,
        /// Optional output file path for zero-parse binary `[u32 header_len][JSON header][Float32 LE]` payload
        #[arg(long)]
        bin_out: Option<PathBuf>,
    },

    /// Compute Top K viewpoints & hikes for any arbitrary clicked coordinate `(lat, lon)` on the 3D terrain
    ShootCoordinate {
        #[arg(long, allow_hyphen_values = true)]
        lat: f64,
        #[arg(long, allow_hyphen_values = true)]
        lon: f64,
        #[arg(long, default_value = "Custom Terrain Pin")]
        name: String,
        #[arg(long, default_value = "2026-10-15")]
        date: String,
        #[arg(long, default_value_t = 16.0)]
        max_miles: f64,
        #[arg(long, default_value_t = 3)]
        top_k: usize,
        #[arg(long)]
        json_out: PathBuf,
    },

    /// Evaluate a user-clicked custom tripod coordinate `(tripod_lat, tripod_lon)` against a target coordinate
    TripodLos {
        #[arg(long, allow_hyphen_values = true)]
        tripod_lat: f64,
        #[arg(long, allow_hyphen_values = true)]
        tripod_lon: f64,
        #[arg(long, allow_hyphen_values = true)]
        target_lat: f64,
        #[arg(long, allow_hyphen_values = true)]
        target_lon: f64,
        #[arg(long, default_value = "Target Landmark")]
        target_name: String,
        #[arg(long, default_value = "2026-10-15")]
        date: String,
        #[arg(long)]
        json_out: PathBuf,
    },
}

fn load_hybrid_terrain(
    elevation_dir: &PathBuf,
    center_lat: f64,
    center_lon: f64,
    near_grid_m: usize,
) -> HybridTerrain {
    let mut srtm = SrtmCache::new(vec![elevation_dir.clone()]);
    srtm.preload_bbox(
        center_lat - 0.6,
        center_lon - 0.6,
        center_lat + 0.6,
        center_lon + 0.6,
    );

    let cache_dir = PathBuf::from("/Users/lucasbrambrink/Coding/photoplanner/data/3dep_cache");
    let near_1m = if near_grid_m > 0 {
        let clamped_m = near_grid_m.min(2000);
        Some(MaxMipmapGrid1m::fetch_or_load_3dep_1m(
            &cache_dir,
            &srtm,
            center_lat,
            center_lon,
            clamped_m,
            clamped_m,
        ))
    } else {
        None
    };

    HybridTerrain::new(near_1m, srtm)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::NotchScan {
            lat,
            lon,
            near_grid_m,
            az_step_deg,
            year,
            json_out,
        } => {
            let t0 = Instant::now();
            let terrain = load_hybrid_terrain(&cli.elevation_dir, lat, lon, near_grid_m);
            let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t_ray = Instant::now();
            let profile = compute_horizon_profile(
                &terrain,
                lat,
                lon,
                1.65,
                0.0,
                360.0,
                az_step_deg,
                45_000.0,
            );
            let ray_ms = t_ray.elapsed().as_secs_f64() * 1000.0;

            let t_notch = Instant::now();
            let notches = detect_light_notches(&profile, year, 1.2, 8.0);
            let notch_ms = t_notch.elapsed().as_secs_f64() * 1000.0;

            println!("==================================================================");
            println!(" PHOTOPLANNER 1M HIERARCHICAL RAY-CASTER — LIGHT NOTCH SCAN");
            println!("==================================================================");
            println!(
                " Origin          : ({:.5}, {:.5}) | Ground Elev: {:.1} m ({:.0} ft)",
                lat,
                lon,
                profile.origin_elevation_m,
                profile.origin_elevation_m * 3.28084
            );
            println!(
                " Near-Field Grid : {}m x {}m ({:.2}M 1m cells, 7-level Max-Mipmap)",
                near_grid_m,
                near_grid_m,
                (near_grid_m * near_grid_m) as f64 / 1e6
            );
            println!(
                " Ray Sweep       : {} rays (step {:.2}°, reach 45.0 km, curvature+refraction on)",
                profile.samples.len(),
                az_step_deg
            );
            println!(
                " Performance     : Pyramid Build = {:.1} ms | 360° Ray Sweep = {:.2} ms | Notch + Ephemeris Solver = {:.2} ms",
                build_ms, ray_ms, notch_ms
            );
            println!("------------------------------------------------------------------");
            println!(
                " Detected {} High-Prominence Solar Light Notches:",
                notches.len()
            );

            for (idx, n) in notches.iter().take(8).enumerate() {
                println!(
                    "\n  [{}] {} @ Azimuth {:.2}° (Quality Score: {:.1})",
                    idx + 1,
                    n.event_type,
                    n.azimuth_deg,
                    n.notch_quality_score
                );
                println!(
                    "      Aperture   : Floor {:.2}° | Flanking Walls {:.2}° (L: {:.2}°, R: {:.2}°) | Prominence: {:.2}°",
                    n.floor_elevation_deg,
                    n.wall_elevation_deg,
                    n.left_wall_deg,
                    n.right_wall_deg,
                    n.prominence_deg
                );
                println!(
                    "      Geometry   : FWHM Width {:.2}° (~{:.1} m wide gap at {:.0} m distance)",
                    n.angular_width_fwhm_deg, n.physical_aperture_width_m, n.notch_distance_m
                );
                for al in n.alignments.iter().take(3) {
                    println!(
                        "      -> Alignment: {} @ {} local solar ({} UTC) | Sun Az {:.2}°, El {:.2}° (+{:.2}° above floor, -{:.2}° below walls)",
                        al.date,
                        al.peak_local_solar_time,
                        al.peak_utc.format("%H:%M"),
                        al.sun_azimuth_deg,
                        al.sun_elevation_deg,
                        al.clearance_above_floor_deg,
                        al.depth_below_walls_deg
                    );
                }
            }

            if let Some(out_path) = json_out {
                let payload = serde_json::json!({
                    "origin_lat": lat,
                    "origin_lon": lon,
                    "origin_elevation_m": profile.origin_elevation_m,
                    "timing_ms": {
                        "pyramid_build_ms": build_ms,
                        "ray_sweep_ms": ray_ms,
                        "notch_solver_ms": notch_ms,
                    },
                    "notches": notches,
                });
                std::fs::write(&out_path, serde_json::to_string_pretty(&payload)?)?;
                println!("\n Saved JSON report to {}", out_path.display());
            }
        }

        Commands::TrailPlan {
            landmark,
            max_miles,
            year,
            json_out,
        } => {
            let t0 = Instant::now();
            let db = RecreationDbHarness::open(&cli.recreation_db)?;
            let top_landmarks = db.list_top_scenic_landmarks(15)?;
            let routes = db.find_routes(Some(&landmark), max_miles, 5)?;

            println!("==================================================================");
            println!(" PHOTOPLANNER x RECREATION — TRAIL LIGHT-NOTCH ITINERARY PLANNER");
            println!("==================================================================");
            println!(
                " Connected to Recreation DB: {} ({} top scenic landmarks loaded)",
                cli.recreation_db.display(),
                top_landmarks.len()
            );

            if routes.is_empty() {
                println!(" No routes found matching '{}'. Available top landmarks:", landmark);
                for lm in top_landmarks.iter().take(10) {
                    println!(
                        "   - {} ({:.4}, {:.4}) | max photo score {:.1}",
                        lm.name, lm.latitude, lm.longitude, lm.max_rank_score
                    );
                }
                return Ok(());
            }

            let center_lat = routes[0].trailhead_lat;
            let center_lon = routes[0].trailhead_lon;
            let terrain = load_hybrid_terrain(&cli.elevation_dir, center_lat, center_lon, 3000);

            let matching_photos: Vec<_> = top_landmarks
                .into_iter()
                .filter(|p| {
                    p.name
                        .to_lowercase()
                        .contains(&landmark.to_lowercase())
                        || routes[0]
                            .landmarks
                            .iter()
                            .any(|l| l.to_lowercase() == p.name.to_lowercase())
                })
                .collect();

            let mut itineraries = Vec::new();
            for route in &routes {
                let itin =
                    build_itinerary_for_route(&terrain, route, matching_photos.clone(), year, 6);
                itineraries.push(itin);
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

            println!(
                " Evaluated {} trail routes in {:.1} ms:\n",
                itineraries.len(),
                elapsed_ms
            );

            for itin in &itineraries {
                println!(
                    " * Route [{}]: {} ({})",
                    itin.route_id, itin.route_name, itin.route_type
                );
                println!(
                    "   Trailhead : {} ({:.4}, {:.4}) | Total: {:.2} mi, +{:.0} ft | Landmarks: {:?}",
                    itin.trailhead_name,
                    itin.trailhead_coords.0,
                    itin.trailhead_coords.1,
                    itin.total_distance_miles,
                    itin.total_elevation_gain_feet,
                    itin.landmarks
                );
                if let Some(photo) = itin.reference_photos.first() {
                    println!(
                        "   Ref Photo : \"{}\" (Score {:.1}) -> {}",
                        photo.sample_photo_title.as_deref().unwrap_or(&photo.name),
                        photo.max_rank_score,
                        photo.sample_photo_url.as_deref().unwrap_or("N/A")
                    );
                }

                for (v_idx, vp) in itin.vantage_plans.iter().take(2).enumerate() {
                    let n = &vp.best_notch;
                    let al = &n.alignments[0];
                    println!(
                        "   -> Shoot Plan #{}: Hike {:.2} mi (+{:.0} ft, ~{} min via Tobler's Function) to Vantage ({:.5}, {:.5}, {:.0} m)",
                        v_idx + 1,
                        vp.distance_from_trailhead_miles,
                        vp.elevation_gain_to_vantage_feet,
                        vp.hike_duration_minutes,
                        vp.vantage_lat,
                        vp.vantage_lon,
                        vp.vantage_elevation_m
                    );
                    println!(
                        "      Schedule   : Depart Trailhead @ {} local {}-> Arrive Vantage @ {} -> Peak Light @ {} on {}",
                        vp.recommended_trailhead_departure_local,
                        if vp.requires_headlamp_on_approach {
                            "(Headlamp Required) "
                        } else {
                            ""
                        },
                        vp.target_arrival_local,
                        al.peak_local_solar_time,
                        al.date
                    );
                    println!(
                        "      Light Notch: {} @ Az {:.2}°, Floor {:.2}°, Walls {:.2}° (Prominence {:.2}°, FWHM {:.2}° / {:.1}m wide)",
                        n.event_type,
                        n.azimuth_deg,
                        n.floor_elevation_deg,
                        n.wall_elevation_deg,
                        n.prominence_deg,
                        n.angular_width_fwhm_deg,
                        n.physical_aperture_width_m
                    );
                }
                println!();
            }

            if let Some(out_path) = json_out {
                std::fs::write(&out_path, serde_json::to_string_pretty(&itineraries)?)?;
                println!(" Saved itinerary JSON to {}", out_path.display());
            }
        }

        Commands::ShadowMap {
            lat,
            lon,
            size_m,
            step_m,
            datetime_utc,
            out_png,
        } => {
            let dt: DateTime<Utc> = datetime_utc.parse()?;
            let sun = solar_position(lat, lon, dt);

            let t0 = Instant::now();
            let terrain = load_hybrid_terrain(&cli.elevation_dir, lat, lon, size_m);
            let grid = terrain.near_1m.as_ref().unwrap();

            let t_cast = Instant::now();
            let stats =
                compute_shadow_and_light_map(&terrain, grid, sun, step_m, Some(&out_png))?;
            let cast_ms = t_cast.elapsed().as_secs_f64() * 1000.0;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            println!("==================================================================");
            println!(" PHOTOPLANNER 2D DIRECTIONAL SHADOW & LIGHT-SHAFT MAP");
            println!("==================================================================");
            println!(
                " Center          : ({:.5}, {:.5}) | Area: {}m x {}m ({}x{} px @ {}m step)",
                lat, lon, size_m, size_m, stats.width_px, stats.height_px, stats.step_m
            );
            println!(
                " Timestamp (UTC) : {} | Sun Azimuth: {:.2}°, Apparent Elevation: {:.2}°",
                dt.to_rfc3339(),
                sun.azimuth_deg,
                sun.apparent_elevation_deg
            );
            println!(
                " Illumination    : {:.1}% lit | {} High-Contrast Spotlight/Light-Shaft pixels",
                stats.illuminated_fraction * 100.0,
                stats.spotlight_shaft_pixels
            );
            if let Some((hlat, hlon, hscore)) = stats.max_contrast_hotspot {
                println!(
                    " Best Spotlight  : ({:.5}, {:.5}) with contrast score {:.1}",
                    hlat, hlon, hscore
                );
            }
            println!(
                " Performance     : Ray-Cast Render = {:.1} ms (Total w/ 1m Pyramid = {:.1} ms)",
                cast_ms, total_ms
            );
            println!(" Saved PNG Map   : {}", out_png.display());
        }

        Commands::Usgs3dep { lat, lon } => {
            println!(
                "Querying USGS 3D Elevation Program (3DEP) 1-meter DEM catalog around ({:.4}, {:.4})...",
                lat, lon
            );
            let tiles = query_usgs_3dep_1m_tiles(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)?;
            println!("Found {} USGS 3DEP 1m GeoTIFF tiles:", tiles.len());
            for (i, t) in tiles.iter().enumerate() {
                println!(
                    "  [{}] {} (Published: {})",
                    i + 1,
                    t.title,
                    t.publication_date.as_deref().unwrap_or("Unknown")
                );
                println!("      URL : {}", t.download_url);
                println!(
                    "      BBox: [{:.4}, {:.4}, {:.4}, {:.4}]",
                    t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]
                );
            }
        }

        Commands::ShootLandmark {
            landmark,
            date,
            max_miles,
            top_k,
            json_out,
        } => {
            let shoot_date: NaiveDate = date.parse()?;
            let t0 = Instant::now();
            let db = RecreationDbHarness::open(&cli.recreation_db)?;
            let (_, seed_lat, seed_lon, _) = db.resolve_landmark_seed(&landmark)?;

            // Load hybrid 1m + SRTM terrain centered on the target landmark
            let terrain = load_hybrid_terrain(&cli.elevation_dir, seed_lat, seed_lon, 3000);
            let report = find_best_viewpoints_and_hikes_for_landmark(
                &db,
                &terrain,
                &landmark,
                shoot_date,
                max_miles,
                top_k,
            )?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let lm = &report.target_landmark;
            println!("================================================================================");
            println!(
                " PHOTOPLANNER — \"I WANT TO PHOTOGRAPH {}\" ({})",
                lm.name.to_uppercase(),
                report.date
            );
            println!("================================================================================");
            println!(
                " Target Summit   : ({:.5}, {:.5}) | Elev: {:.0} ft ({:.1} m) | Vertical Relief: {:.0} ft ({:.0} m)",
                lm.summit_lat,
                lm.summit_lon,
                lm.summit_elevation_ft,
                lm.summit_elevation_m,
                lm.vertical_relief_ft,
                lm.vertical_relief_m
            );
            println!(
                " Primary Face    : Aspect {:.1}° | Scanned {} routes ({} trail vertices) in {:.1} ms",
                lm.primary_face_aspect_deg,
                report.total_routes_scanned,
                report.total_trail_vertices_evaluated,
                elapsed_ms
            );
            if let Some(ref url) = report.reference_photo_url {
                println!(" Reference Photo : {}", url);
            }
            println!("--------------------------------------------------------------------------------");
            println!(" TOPOGRAPHIC SUN-RAY ILLUMINATION WINDOWS ON {}:", lm.name.to_uppercase());
            for win in &report.golden_hour_windows {
                println!(
                    "   * {} : {} – {} local (Peak @ {}) | Sun Az {:.1}°, El {:.1}° | Valley Shadow Contrast: {:.0}%",
                    win.phase,
                    win.window_start_local,
                    win.window_end_local,
                    win.peak_time_local,
                    win.sun_azimuth_deg,
                    win.sun_elevation_deg,
                    win.valley_shadow_contrast_pct
                );
            }
            println!("--------------------------------------------------------------------------------");
            println!(
                " TOP {} HIKEABLE VIEWPOINTS (VP1..VP{}) WITH UNOBSTRUCTED LINE-OF-SIGHT:\n",
                report.top_viewpoints.len(),
                report.top_viewpoints.len()
            );

            for vp in &report.top_viewpoints {
                println!(
                    " [VP{}] {} — Score: {:.1}/100",
                    vp.viewpoint_rank, vp.viewpoint_label, vp.overall_score
                );
                println!(
                    "   Viewpoint Coords : ({:.5}, {:.5}) @ {:.0} ft ({:.0} m)",
                    vp.viewpoint_lat,
                    vp.viewpoint_lon,
                    vp.viewpoint_elevation_ft,
                    vp.viewpoint_elevation_m
                );
                println!(
                    "   Hike from TH     : {:.2} miles to VP{} (+{:.0} ft gain, ~{} min hike) from Trailhead \"{}\"",
                    vp.hike_distance_to_vp_miles,
                    vp.viewpoint_rank,
                    vp.elevation_gain_to_vp_feet,
                    vp.hike_duration_to_vp_minutes,
                    vp.trailhead_name
                );
                println!(
                    "   Full Trail Route : [Route #{}] {} ({}, {:.2} mi total)",
                    vp.route_id, vp.route_name, vp.route_type, vp.total_route_distance_miles
                );
                println!(
                    "   Sightline to {} : {:.2} miles ({:.0} m) looking {} ({:.1}°) | {:.0} ft ({:.0}%) of sheer relief visible | Lens: {}",
                    lm.name,
                    vp.sightline.distance_to_landmark_miles,
                    vp.sightline.distance_to_landmark_m,
                    vp.sightline.compass_direction,
                    vp.sightline.bearing_to_landmark_deg,
                    vp.sightline.visible_vertical_relief_ft,
                    vp.sightline.visible_relief_fraction * 100.0,
                    vp.sightline.recommended_focal_length_mm
                );
                for sched in &vp.schedules {
                    println!(
                        "   -> [{}] Shoot Window: {} (Peak @ {})",
                        sched.phase, sched.shooting_window_local, sched.peak_light_local
                    );
                    println!(
                        "      Trip Plan     : Depart \"{}\" @ {} -> Arrive VP{} @ {} (20m setup buffer)",
                        vp.trailhead_name,
                        sched.recommended_trailhead_departure_local,
                        vp.viewpoint_rank,
                        sched.target_viewpoint_arrival_local
                    );
                    println!(
                        "      Light Geometry: {} (Camera-to-Sun angle {:.1}°, Valley Shadow Contrast {:.0}%)",
                        sched.lighting_style,
                        sched.camera_to_sun_angle_deg,
                        sched.valley_shadow_contrast_pct
                    );
                }
                println!();
            }

            if let Some(out_path) = json_out {
                std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)?;
                println!(" Saved shoot plan JSON to {}", out_path.display());
            }
        }

        Commands::TerrainPatch {
            lat,
            lon,
            span_m,
            grid_size,
            json_out,
            bin_out,
        } => {
            let terrain = load_hybrid_terrain(&cli.elevation_dir, lat, lon, 2000);
            let m_per_lat = METERS_PER_DEG_LAT;
            let m_per_lon = meters_per_deg_lon(lat);

            let half_span = span_m * 0.5;
            let step_m = span_m / ((grid_size - 1).max(1) as f64);

            // Parallel row evaluation across all CPU cores via Rayon
            let rows: Vec<(f64, f64, Vec<f32>)> = (0..grid_size)
                .into_par_iter()
                .map(|iy| {
                    let dy_m = -half_span + (iy as f64) * step_m;
                    let sample_lat = lat + dy_m / m_per_lat;
                    let mut row_vals = Vec::with_capacity(grid_size);
                    let mut r_min = f64::INFINITY;
                    let mut r_max = f64::NEG_INFINITY;
                    for ix in 0..grid_size {
                        let dx_m = -half_span + (ix as f64) * step_m;
                        let sample_lon = lon + dx_m / m_per_lon;
                        let z = terrain.elevation_m(sample_lat, sample_lon);
                        r_min = r_min.min(z);
                        r_max = r_max.max(z);
                        row_vals.push((z * 100.0).round() as f32 / 100.0);
                    }
                    (r_min, r_max, row_vals)
                })
                .collect();

            let mut elevations = Vec::with_capacity(grid_size * grid_size);
            let mut min_z = f64::INFINITY;
            let mut max_z = f64::NEG_INFINITY;
            for (r_min, r_max, mut r_vals) in rows {
                min_z = min_z.min(r_min);
                max_z = max_z.max(r_max);
                elevations.append(&mut r_vals);
            }

            if let Some(ref bpath) = bin_out {
                let header = serde_json::json!({
                    "center_lat": lat,
                    "center_lon": lon,
                    "span_m": span_m,
                    "grid_size": grid_size,
                    "step_m": step_m,
                    "min_elevation_m": min_z,
                    "max_elevation_m": max_z,
                    "meters_per_deg_lat": m_per_lat,
                    "meters_per_deg_lon": m_per_lon,
                });
                let mut hdr_bytes = serde_json::to_vec(&header)?;
                while hdr_bytes.len() % 4 != 0 {
                    hdr_bytes.push(b' ');
                }
                let hdr_len = hdr_bytes.len() as u32;
                let mut packet = Vec::with_capacity(4 + hdr_bytes.len() + elevations.len() * 4);
                packet.extend_from_slice(&hdr_len.to_le_bytes());
                packet.extend_from_slice(&hdr_bytes);
                for v in &elevations {
                    packet.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(bpath, packet)?;
            }

            if let Some(ref jpath) = json_out {
                let payload = serde_json::json!({
                    "center_lat": lat,
                    "center_lon": lon,
                    "span_m": span_m,
                    "grid_size": grid_size,
                    "step_m": step_m,
                    "min_elevation_m": min_z,
                    "max_elevation_m": max_z,
                    "meters_per_deg_lat": m_per_lat,
                    "meters_per_deg_lon": m_per_lon,
                    "elevations": elevations,
                });
                std::fs::write(jpath, serde_json::to_string(&payload)?)?;
            }
        }

        Commands::ShootCoordinate {
            lat,
            lon,
            name,
            date,
            max_miles,
            top_k,
            json_out,
        } => {
            let shoot_date: NaiveDate = date.parse()?;
            let db = RecreationDbHarness::open(&cli.recreation_db)?;
            let terrain = load_hybrid_terrain(&cli.elevation_dir, lat, lon, 2000);
            let report = find_best_viewpoints_for_seed(
                &db,
                &terrain,
                &name,
                lat,
                lon,
                None,
                shoot_date,
                max_miles,
                top_k,
            )?;
            std::fs::write(&json_out, serde_json::to_string(&report)?)?;
        }

        Commands::TripodLos {
            tripod_lat,
            tripod_lon,
            target_lat,
            target_lon,
            target_name,
            date,
            json_out,
        } => {
            let shoot_date: NaiveDate = date.parse()?;
            let db = RecreationDbHarness::open(&cli.recreation_db)?;
            let terrain = load_hybrid_terrain(&cli.elevation_dir, target_lat, target_lon, 2000);
            let report = analyze_custom_tripod_point(
                &db,
                &terrain,
                tripod_lat,
                tripod_lon,
                target_lat,
                target_lon,
                &target_name,
                shoot_date,
            )?;
            std::fs::write(&json_out, serde_json::to_string(&report)?)?;
        }
    }

    Ok(())
}
