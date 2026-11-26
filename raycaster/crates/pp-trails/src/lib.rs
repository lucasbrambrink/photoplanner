use anyhow::{Context, Result};
use pp_raycast::{
    cast_camera_line_of_sight, cast_camera_line_of_sight_batch, compute_horizon_profile,
    compute_horizon_profiles_batch, compute_subject_astro_night_summary, detect_light_notches,
    extract_landmark_3d_geometry, find_subject_golden_hour_windows, HorizonProfileRequest,
    LandmarkTarget3D, LightNotch, LineOfSightResult, SubjectAstroNightSummary,
    SubjectGoldenHourWindow,
};
use pp_terrain::{meters_per_deg_lon, ElevationSurface, HybridTerrain, METERS_PER_DEG_LAT};
use rayon::prelude::*;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenicLandmark {
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub photo_count: usize,
    pub max_rank_score: f64,
    pub sample_photo_url: Option<String>,
    pub sample_photo_title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailRouteRecord {
    pub id: i64,
    pub name: String,
    pub route_type: String,
    pub trailhead_id: String,
    pub trailhead_name: String,
    pub trailhead_lat: f64,
    pub trailhead_lon: f64,
    pub distance_miles: f64,
    pub elevation_gain_feet: f64,
    /// `[[lon, lat], ...]`
    pub geometry: Vec<[f64; 2]>,
    pub landmarks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailVantageShootPlan {
    pub vantage_lat: f64,
    pub vantage_lon: f64,
    pub vantage_elevation_m: f64,
    pub distance_from_trailhead_miles: f64,
    pub elevation_gain_to_vantage_feet: f64,
    /// Estimated hike time from trailhead to this vantage point using Tobler's Hiking Function
    pub hike_duration_minutes: u32,
    pub requires_headlamp_on_approach: bool,
    pub recommended_trailhead_departure_local: String,
    pub target_arrival_local: String,
    pub best_notch: LightNotch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotoHikeItinerary {
    pub route_id: i64,
    pub route_name: String,
    pub route_type: String,
    pub trailhead_name: String,
    pub trailhead_coords: (f64, f64),
    pub total_distance_miles: f64,
    pub total_elevation_gain_feet: f64,
    pub landmarks: Vec<String>,
    pub reference_photos: Vec<ScenicLandmark>,
    pub vantage_plans: Vec<TrailVantageShootPlan>,
}

/// Connects to the `recreation` ("TrailBinder") SQLite database (`db.sqlite3`).
pub struct RecreationDbHarness {
    conn: Connection,
}

impl RecreationDbHarness {
    pub fn open(db_path: &Path) -> Result<Self> {
        if db_path.exists() {
            let conn = Connection::open(db_path).with_context(|| {
                format!("Failed to open recreation DB at {}", db_path.display())
            })?;
            return Ok(Self { conn });
        }
        Self::open_builtin_fallback()
    }

    fn open_builtin_fallback() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "CREATE TABLE web_hikephoto (
                landmark_name TEXT,
                latitude REAL,
                longitude REAL,
                rank_score REAL,
                direct_image_url TEXT,
                title TEXT
            );
            CREATE TABLE ridb_permitentrance (
                permit_entrance_id TEXT PRIMARY KEY,
                permit_entrance_name TEXT,
                latitude REAL,
                longitude REAL
            );
            CREATE TABLE web_precomputedroute (
                id INTEGER PRIMARY KEY,
                name TEXT,
                route_type TEXT,
                start_trailhead_id TEXT,
                distance_miles REAL,
                elevation_gain_feet REAL,
                geometry TEXT,
                landmarks TEXT
            );
            CREATE TABLE web_trailroute (
                route_id TEXT PRIMARY KEY,
                name TEXT
            );
            CREATE TABLE web_trailsegment (
                route_id TEXT,
                geometry TEXT
            );",
        )?;

        let seeds = [
            ("Half Dome", 37.7459, -119.5332, 99.5, "Half Dome Northwest Face"),
            ("El Capitan", 37.7339, -119.6377, 98.2, "El Capitan Monolith"),
            ("Glacier Point", 37.7281, -119.5738, 97.4, "Glacier Point Amphitheater"),
            ("Sentinel Dome", 37.7126, -119.5844, 96.0, "Sentinel Dome 360 Crown"),
            ("Upper Yosemite Fall", 37.7566, -119.5969, 95.8, "Upper Yosemite Fall"),
            ("Clouds Rest", 37.7678, -119.4894, 95.2, "Clouds Rest Knife-Edge"),
            ("Cathedral Peak", 37.8536, -119.4046, 94.1, "Cathedral Peak High Sierra"),
            ("Purisima Creek Redwoods", 37.4365, -122.3450, 91.5, "Purisima Coastal Ridge"),
            ("Mt. Tamalpais East Peak", 37.9291, -122.5778, 93.0, "Mt. Tamalpais Marine Layer"),
        ];
        for (name, lat, lon, score, title) in seeds {
            conn.execute(
                "INSERT INTO web_hikephoto VALUES (?1, ?2, ?3, ?4, NULL, ?5)",
                params![name, lat, lon, score, title],
            )?;
        }

        conn.execute(
            "INSERT INTO ridb_permitentrance VALUES
             ('TH_GP', 'Glacier Point / Sentinel TH', 37.7118, -119.5865),
             ('TH_ND', 'Porcupine Creek / North Dome TH', 37.8065, -119.5662),
             ('TH_HI', 'Washburn & Panorama Rim TH', 37.7196, -119.5728),
             ('TH_TV', 'Tunnel View / Pohono TH', 37.7156, -119.6776),
             ('TH_OL', 'Sunrise / Olmsted Point TH', 37.8108, -119.4850),
             ('TH_CP', 'Cathedral Lakes Trailhead', 37.8732, -119.3812),
             ('TH_PC', 'Purisima Skyline Trailhead', 37.4428, -122.3322),
             ('TH_MT', 'Pantoll / East Peak Trailhead', 37.9052, -122.6042)",
            [],
        )?;

        let routes = [
            (
                101_i64,
                "Sentinel Dome & Glacier Point Rim Loop",
                "Loop",
                "TH_GP",
                4.8_f64,
                820.0_f64,
                r#"[[-119.5865,37.7118],[-119.5858,37.7121],[-119.5850,37.7124],[-119.5844,37.7126],[-119.5810,37.7180],[-119.5772,37.7235],[-119.5738,37.7281],[-119.5772,37.7235],[-119.5810,37.7180],[-119.5844,37.7126],[-119.5855,37.7122],[-119.5865,37.7118]]"#,
                r#"["Half Dome","Sentinel Dome","Glacier Point","Upper Yosemite Fall"]"#,
            ),
            (
                102_i64,
                "North Dome & Indian Ridge Trail",
                "Out & Back",
                "TH_ND",
                8.8_f64,
                1150.0_f64,
                r#"[[-119.5662,37.8065],[-119.5635,37.8005],[-119.5605,37.7945],[-119.5582,37.7892],[-119.5570,37.7860],[-119.5562,37.7845],[-119.5558,37.7832],[-119.5562,37.7845],[-119.5570,37.7860],[-119.5582,37.7892],[-119.5605,37.7945],[-119.5662,37.8065]]"#,
                r#"["Half Dome","Clouds Rest","Upper Yosemite Fall"]"#,
            ),
            (
                103_i64,
                "Washburn Point & Panorama Cliff Overlook",
                "Out & Back",
                "TH_HI",
                5.4_f64,
                1050.0_f64,
                r#"[[-119.5728,37.7196],[-119.5705,37.7210],[-119.5678,37.7218],[-119.5648,37.7220],[-119.5612,37.7222],[-119.5585,37.7224],[-119.5568,37.7225],[-119.5585,37.7224],[-119.5612,37.7222],[-119.5648,37.7220],[-119.5705,37.7210],[-119.5728,37.7196]]"#,
                r#"["Half Dome","Clouds Rest","Glacier Point"]"#,
            ),
            (
                104_i64,
                "Pohono Trail — Tunnel View to Inspiration & Taft Point",
                "Out & Back",
                "TH_TV",
                7.2_f64,
                2100.0_f64,
                r#"[[-119.6776,37.7156],[-119.6750,37.7140],[-119.6720,37.7128],[-119.6690,37.7130],[-119.6580,37.7122],[-119.6440,37.7125],[-119.6048,37.7131],[-119.6440,37.7125],[-119.6580,37.7122],[-119.6690,37.7130],[-119.6750,37.7140],[-119.6776,37.7156]]"#,
                r#"["El Capitan","Half Dome"]"#,
            ),
            (
                105_i64,
                "Olmsted Point & Sunrise Lakes Ridge",
                "Out & Back",
                "TH_OL",
                6.8_f64,
                1420.0_f64,
                r#"[[-119.4850,37.8108],[-119.4840,37.8065],[-119.4828,37.8012],[-119.4815,37.7960],[-119.4835,37.7895],[-119.4860,37.7820],[-119.4882,37.7745],[-119.4860,37.7820],[-119.4835,37.7895],[-119.4815,37.7960],[-119.4840,37.8065],[-119.4850,37.8108]]"#,
                r#"["Clouds Rest","Half Dome"]"#,
            ),
            (
                106_i64,
                "Upper Cathedral Lake & Echo Peaks Bench",
                "Out & Back",
                "TH_CP",
                7.4_f64,
                1180.0_f64,
                r#"[[-119.3812,37.8732],[-119.3865,37.8690],[-119.3920,37.8648],[-119.3985,37.8602],[-119.4035,37.8578],[-119.4090,37.8555],[-119.4138,37.8535],[-119.4090,37.8555],[-119.4035,37.8578],[-119.3985,37.8602],[-119.3865,37.8690],[-119.3812,37.8732]]"#,
                r#"["Cathedral Peak"]"#,
            ),
            (
                107_i64,
                "Purisima Creek Skyline Ridge Trail",
                "Loop",
                "TH_PC",
                6.2_f64,
                1450.0_f64,
                r#"[[-122.3322,37.4428],[-122.3355,37.4418],[-122.3380,37.4410],[-122.3418,37.4388],[-122.3450,37.4365],[-122.3490,37.4348],[-122.3520,37.4330],[-122.3490,37.4348],[-122.3450,37.4365],[-122.3380,37.4410],[-122.3355,37.4418],[-122.3322,37.4428]]"#,
                r#"["Purisima Creek Redwoods"]"#,
            ),
            (
                108_i64,
                "Matt Davis & East Peak Ridge Traverse",
                "Loop",
                "TH_MT",
                5.6_f64,
                1380.0_f64,
                r#"[[-122.6042,37.9052],[-122.5998,37.9092],[-122.5960,37.9125],[-122.5915,37.9168],[-122.5870,37.9210],[-122.5822,37.9252],[-122.5778,37.9291],[-122.5822,37.9252],[-122.5870,37.9210],[-122.5960,37.9125],[-122.5998,37.9092],[-122.6042,37.9052]]"#,
                r#"["Mt. Tamalpais East Peak"]"#,
            ),
        ];
        for (id, name, rtype, th, dist, gain, geom, lms) in routes {
            conn.execute(
                "INSERT INTO web_precomputedroute VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![id, name, rtype, th, dist, gain, geom, lms],
            )?;
        }

        Ok(Self { conn })
    }

    /// Lists top scenic landmarks ranked by geotagged photo scores from `web_hikephoto`.
    pub fn list_top_scenic_landmarks(&self, limit: usize) -> Result<Vec<ScenicLandmark>> {
        let mut stmt = self.conn.prepare(
            "SELECT landmark_name,
                    COUNT(*) as cnt,
                    AVG(latitude) as avg_lat,
                    AVG(longitude) as avg_lon,
                    MAX(rank_score) as max_score,
                    direct_image_url,
                    title
             FROM web_hikephoto
             WHERE landmark_name IS NOT NULL AND landmark_name != ''
             GROUP BY landmark_name
             ORDER BY max_score DESC
             LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(ScenicLandmark {
                name: row.get(0)?,
                photo_count: row.get::<_, i64>(1)? as usize,
                latitude: row.get(2)?,
                longitude: row.get(3)?,
                max_rank_score: row.get(4)?,
                sample_photo_url: row.get(5).ok(),
                sample_photo_title: row.get(6).ok(),
            })
        })?;

        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Queries precomputed hiking routes from `web_precomputedroute` joined with `ridb_permitentrance`.
    /// Optional `keyword` filters by landmark or route name (e.g. "Half Dome", "Sentinel Dome", "Yosemite Fall").
    pub fn find_routes(
        &self,
        keyword: Option<&str>,
        max_miles: f64,
        limit: usize,
    ) -> Result<Vec<TrailRouteRecord>> {
        let pattern = format!("%{}%", keyword.unwrap_or(""));
        let mut stmt = self.conn.prepare(
            "SELECT r.id,
                    r.name,
                    r.route_type,
                    r.start_trailhead_id,
                     COALESCE(th.permit_entrance_name, r.start_trailhead_id) as th_name,
                    COALESCE(th.latitude, 37.7456) as th_lat,
                    COALESCE(th.longitude, -119.5383) as th_lon,
                    r.distance_miles,
                    r.elevation_gain_feet,
                    r.geometry,
                    r.landmarks
             FROM web_precomputedroute r
             LEFT JOIN ridb_permitentrance th ON r.start_trailhead_id = th.permit_entrance_id
             WHERE r.distance_miles <= ?1
               AND (r.name LIKE ?2 OR r.landmarks LIKE ?2)
             ORDER BY r.distance_miles ASC
             LIMIT ?3",
        )?;

        let rows = stmt.query_map(params![max_miles, pattern, limit as i64], |row| {
            let geom_str: String = row.get(9)?;
            let lms_str: Option<String> = row.get(10).ok();

            let geometry: Vec<[f64; 2]> = serde_json::from_str(&geom_str).unwrap_or_default();
            let landmarks: Vec<String> = lms_str
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();

            Ok(TrailRouteRecord {
                id: row.get(0)?,
                name: row.get(1)?,
                route_type: row.get(2)?,
                trailhead_id: row.get(3)?,
                trailhead_name: row.get(4)?,
                trailhead_lat: row.get(5)?,
                trailhead_lon: row.get(6)?,
                distance_miles: row.get(7)?,
                elevation_gain_feet: row.get(8)?,
                geometry,
                landmarks,
            })
        })?;

        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Resolves a landmark name (e.g. "Half Dome") to its canonical name, seed `(lat, lon)`, and top reference photo URL.
    pub fn resolve_landmark_seed(
        &self,
        landmark_query: &str,
    ) -> Result<(String, f64, f64, Option<String>)> {
        let pattern = format!("%{}%", landmark_query);
        let mut stmt = self.conn.prepare(
            "SELECT landmark_name,
                    AVG(latitude) as avg_lat,
                    AVG(longitude) as avg_lon,
                    direct_image_url
             FROM web_hikephoto
             WHERE landmark_name LIKE ?1
             GROUP BY landmark_name
             ORDER BY MAX(rank_score) DESC
             LIMIT 1",
        )?;

        if let Ok(res) = stmt.query_row(params![pattern], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, f64>(2)?,
                row.get::<_, Option<String>>(3).unwrap_or(None),
            ))
        }) {
            return Ok(res);
        }

        // Fallback: search trail segment geometries whose route name matches `landmark_query`
        let mut seg_stmt = self.conn.prepare(
            "SELECT r.name, s.geometry
             FROM web_trailsegment s
             JOIN web_trailroute r ON s.route_id = r.route_id
             WHERE r.name LIKE ?1
             LIMIT 1",
        )?;
        let (rname, geom_str): (String, String) =
            seg_stmt.query_row(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let coords: Vec<[f64; 2]> = serde_json::from_str(&geom_str).unwrap_or_default();
        if let Some(mid) = coords.get(coords.len() / 2) {
            return Ok((rname, mid[1], mid[0], None));
        }

        anyhow::bail!("Could not resolve landmark '{}'", landmark_query)
    }

    /// Queries all precomputed routes starting within `radius_deg` of `(center_lat, center_lon)` up to `max_miles`.
    pub fn find_all_routes_near(
        &self,
        center_lat: f64,
        center_lon: f64,
        radius_deg: f64,
        max_miles: f64,
    ) -> Result<Vec<TrailRouteRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.id,
                    r.name,
                    r.route_type,
                    r.start_trailhead_id,
                    COALESCE(th.permit_entrance_name, r.start_trailhead_id) as th_name,
                    COALESCE(th.latitude, 37.7456) as th_lat,
                    COALESCE(th.longitude, -119.5383) as th_lon,
                    r.distance_miles,
                    r.elevation_gain_feet,
                    r.geometry,
                    r.landmarks
             FROM web_precomputedroute r
             LEFT JOIN ridb_permitentrance th ON r.start_trailhead_id = th.permit_entrance_id
             WHERE r.distance_miles <= ?1
               AND th.latitude BETWEEN ?2 AND ?3
               AND th.longitude BETWEEN ?4 AND ?5
             ORDER BY r.distance_miles ASC",
        )?;

        let rows = stmt.query_map(
            params![
                max_miles,
                center_lat - radius_deg,
                center_lat + radius_deg,
                center_lon - radius_deg,
                center_lon + radius_deg
            ],
            |row| {
                let geom_str: String = row.get(9)?;
                let lms_str: Option<String> = row.get(10).ok();
                let geometry: Vec<[f64; 2]> = serde_json::from_str(&geom_str).unwrap_or_default();
                let landmarks: Vec<String> = lms_str
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();

                Ok(TrailRouteRecord {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    route_type: row.get(2)?,
                    trailhead_id: row.get(3)?,
                    trailhead_name: row.get(4)?,
                    trailhead_lat: row.get(5)?,
                    trailhead_lon: row.get(6)?,
                    distance_miles: row.get(7)?,
                    elevation_gain_feet: row.get(8)?,
                    geometry,
                    landmarks,
                })
            },
        )?;

        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }
}

/// Computes great-circle distance in meters between two `(lat, lon)` points.
pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

/// Tobler's Hiking Function: returns walking speed in meters per minute given slope `dz / dx`.
#[inline]
pub fn tobler_speed_m_per_min(slope: f64) -> f64 {
    let speed_km_h = 6.0 * (-3.5 * (slope + 0.05).abs()).exp();
    // Apply a conservative 0.85 photography pack factor (camera + tripod)
    (speed_km_h * 0.85) * (1000.0 / 60.0)
}

/// Determines how many vertices of `route.geometry` represent unique trail path.
///
/// - **Loop** and **Point to Point** routes always scan 100% of vertices (never truncated).
/// - **Out & Back** routes that are stored as a doubled track returning within 250 m of the
///   trailhead stop at the true turnaround apex (maximum distance from the trailhead).
/// - Single-leg Out & Back geometries (start and end > 250 m apart) scan 100% of vertices.
pub fn effective_scan_len(route: &TrailRouteRecord) -> usize {
    let n = route.geometry.len();
    if n < 2 {
        return n;
    }
    let rt = route.route_type.to_lowercase();
    if rt.contains("loop") || rt.contains("point") {
        return n;
    }
    let (lon0, lat0) = (route.geometry[0][0], route.geometry[0][1]);
    let (lon_end, lat_end) = (route.geometry[n - 1][0], route.geometry[n - 1][1]);
    let end_to_start_m = haversine_m(lat0, lon0, lat_end, lon_end);
    if end_to_start_m > 250.0 {
        // Single-leg geometry (does not double back to start)
        return n;
    }
    // Doubled Out & Back: locate the actual turnaround apex farthest from trailhead
    let mut apex_idx = n / 2;
    let mut max_d = -1.0_f64;
    for (i, pt) in route.geometry.iter().enumerate() {
        let d = haversine_m(lat0, lon0, pt[1], pt[0]);
        if d > max_d {
            max_d = d;
            apex_idx = i;
        }
    }
    if max_d <= 250.0 {
        return n;
    }
    (apex_idx + 1).max(2).min(n)
}

/// Samples candidate vantage points along a hike geometry, runs parallel 1m+SRTM horizon ray-casting
/// to detect Light Notches from the trail, and calculates Tobler hike durations and departure times.
pub fn build_itinerary_for_route(
    terrain: &HybridTerrain,
    route: &TrailRouteRecord,
    reference_photos: Vec<ScenicLandmark>,
    year: i32,
    sample_count: usize,
) -> PhotoHikeItinerary {
    if route.geometry.len() < 2 {
        return PhotoHikeItinerary {
            route_id: route.id,
            route_name: route.name.clone(),
            route_type: route.route_type.clone(),
            trailhead_name: route.trailhead_name.clone(),
            trailhead_coords: (route.trailhead_lat, route.trailhead_lon),
            total_distance_miles: route.distance_miles,
            total_elevation_gain_feet: route.elevation_gain_feet,
            landmarks: route.landmarks.clone(),
            reference_photos,
            vantage_plans: Vec::new(),
        };
    }

    // 1. Walk the full unique trail geometry (full loop/point-to-point, or up to turnaround apex
    // on doubled out-and-back tracks) to compute cumulative distance, gain, and Tobler time.
    let scan_len = effective_scan_len(route);
    let mut cum_dist_m = vec![0.0_f64; scan_len];
    let mut cum_gain_m = vec![0.0_f64; scan_len];
    let mut cum_time_min = vec![0.0_f64; scan_len];
    let mut vertex_z = vec![0.0_f64; scan_len];

    vertex_z[0] = terrain.elevation_m(route.geometry[0][1], route.geometry[0][0]);

    for i in 1..scan_len {
        let (lon0, lat0) = (route.geometry[i - 1][0], route.geometry[i - 1][1]);
        let (lon1, lat1) = (route.geometry[i][0], route.geometry[i][1]);
        let seg_d_m = haversine_m(lat0, lon0, lat1, lon1);
        let z1 = terrain.elevation_m(lat1, lon1);
        vertex_z[i] = z1;

        let dz = z1 - vertex_z[i - 1];
        let gain = dz.max(0.0);
        let slope = if seg_d_m > 1.0 { dz / seg_d_m } else { 0.0 };
        let dt_min = if seg_d_m > 0.1 {
            seg_d_m / tobler_speed_m_per_min(slope).max(10.0)
        } else {
            0.0
        };

        cum_dist_m[i] = cum_dist_m[i - 1] + seg_d_m;
        cum_gain_m[i] = cum_gain_m[i - 1] + gain;
        cum_time_min[i] = cum_time_min[i - 1] + dt_min;
    }

    // 2. Select `sample_count` spatially diverse candidate vertices along the unique trail corridor
    let mut candidate_indices = Vec::new();
    let step = (scan_len / sample_count.max(1)).max(1);
    for idx in (step..scan_len).step_by(step) {
        candidate_indices.push(idx);
    }
    if !candidate_indices.contains(&(scan_len - 1)) {
        candidate_indices.push(scan_len - 1);
    }

    // 3. Batch all sunrise & sunset 0.10° horizon windows across all candidate trail points into
    //    a single Metal GPU compute dispatch (with automatic parallel Rayon CPU fallback)
    let ((sr_min, sr_max), (ss_min, ss_max)) =
        pp_solar::annual_solar_azimuth_windows(route.trailhead_lat);

    let mut batch_reqs = Vec::with_capacity(candidate_indices.len() * 2);
    for &idx in &candidate_indices {
        let lon = route.geometry[idx][0];
        let lat = route.geometry[idx][1];
        batch_reqs.push(HorizonProfileRequest {
            origin_lat: lat,
            origin_lon: lon,
            eye_offset_m: 1.65,
            az_start_deg: sr_min,
            az_end_deg: sr_max,
            az_step_deg: 0.10,
            max_distance_m: 35_000.0,
        });
        batch_reqs.push(HorizonProfileRequest {
            origin_lat: lat,
            origin_lon: lon,
            eye_offset_m: 1.65,
            az_start_deg: ss_min,
            az_end_deg: ss_max,
            az_step_deg: 0.10,
            max_distance_m: 35_000.0,
        });
    }

    let batched_profiles = compute_horizon_profiles_batch(terrain, &batch_reqs);

    let mut vantage_plans: Vec<TrailVantageShootPlan> = candidate_indices
        .par_iter()
        .enumerate()
        .filter_map(|(cand_pos, &idx)| {
            let lon = route.geometry[idx][0];
            let lat = route.geometry[idx][1];
            let sr_profile = &batched_profiles[cand_pos * 2];
            let ss_profile = &batched_profiles[cand_pos * 2 + 1];

            let mut notches = detect_light_notches(sr_profile, year, 0.55, 8.0);
            notches.extend(detect_light_notches(ss_profile, year, 0.55, 8.0));
            notches.sort_by(|a, b| {
                b.notch_quality_score
                    .partial_cmp(&a.notch_quality_score)
                    .unwrap()
            });
            let best_notch = notches.into_iter().next()?;
            let first_align = best_notch.alignments.first()?;

            let hike_min = cum_time_min[idx].round() as u32;
            // Buffer 25 minutes before alignment for tripod & composition setup
            let total_lead_min = (hike_min + 25) as i64;

            // Parse peak_local_solar_time ("HH:MM")
            let parts: Vec<&str> = first_align.peak_local_solar_time.split(':').collect();
            let (hr, mn) = if parts.len() == 2 {
                (
                    parts[0].parse::<i64>().unwrap_or(6),
                    parts[1].parse::<i64>().unwrap_or(0),
                )
            } else {
                (6, 0)
            };
            let align_total_min = hr * 60 + mn;
            let arrival_min = (align_total_min - 25).rem_euclid(1440);
            let depart_min = (align_total_min - total_lead_min).rem_euclid(1440);

            let depart_str = format!("{:02}:{:02}", depart_min / 60, depart_min % 60);
            let arrival_str = format!("{:02}:{:02}", arrival_min / 60, arrival_min % 60);
            let requires_headlamp =
                best_notch.event_type.contains("Sunrise") || best_notch.event_type.contains("Sunset");

            Some(TrailVantageShootPlan {
                vantage_lat: (lat * 100_000.0).round() / 100_000.0,
                vantage_lon: (lon * 100_000.0).round() / 100_000.0,
                vantage_elevation_m: (vertex_z[idx] * 10.0).round() / 10.0,
                distance_from_trailhead_miles: ((cum_dist_m[idx] / 1609.344) * 100.0).round()
                    / 100.0,
                elevation_gain_to_vantage_feet: (cum_gain_m[idx] * 3.28084).round(),
                hike_duration_minutes: hike_min,
                requires_headlamp_on_approach: requires_headlamp,
                recommended_trailhead_departure_local: depart_str,
                target_arrival_local: arrival_str,
                best_notch,
            })
        })
        .collect();

    vantage_plans.sort_by(|a, b| {
        b.best_notch
            .notch_quality_score
            .partial_cmp(&a.best_notch.notch_quality_score)
            .unwrap()
    });

    PhotoHikeItinerary {
        route_id: route.id,
        route_name: route.name.clone(),
        route_type: route.route_type.clone(),
        trailhead_name: route.trailhead_name.clone(),
        trailhead_coords: (route.trailhead_lat, route.trailhead_lon),
        total_distance_miles: route.distance_miles,
        total_elevation_gain_feet: route.elevation_gain_feet,
        landmarks: route.landmarks.clone(),
        reference_photos,
        vantage_plans,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewpointWindowSchedule {
    pub phase: String,
    pub shooting_window_local: String,
    pub peak_light_local: String,
    pub recommended_trailhead_departure_local: String,
    pub target_viewpoint_arrival_local: String,
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    pub camera_to_sun_angle_deg: f64,
    pub lighting_style: String,
    pub valley_shadow_contrast_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkViewpointHikeOption {
    pub viewpoint_rank: usize,
    pub viewpoint_label: String,
    pub viewpoint_lat: f64,
    pub viewpoint_lon: f64,
    pub viewpoint_elevation_m: f64,
    pub viewpoint_elevation_ft: f64,
    pub route_id: i64,
    pub route_name: String,
    pub route_type: String,
    pub trailhead_name: String,
    pub trailhead_lat: f64,
    pub trailhead_lon: f64,
    pub hike_distance_to_vp_miles: f64,
    pub total_route_distance_miles: f64,
    pub elevation_gain_to_vp_feet: f64,
    pub hike_duration_to_vp_minutes: u32,
    pub sightline: LineOfSightResult,
    pub overall_score: f64,
    pub schedules: Vec<ViewpointWindowSchedule>,
    /// 3D trail coordinates `[lon, lat, elevation_m]` from the trailhead up to this viewpoint
    pub trail_path_to_vp: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkShootPlanReport {
    pub target_landmark: LandmarkTarget3D,
    pub reference_photo_url: Option<String>,
    pub date: String,
    pub golden_hour_windows: Vec<SubjectGoldenHourWindow>,
    pub astro_night_summary: SubjectAstroNightSummary,
    pub total_routes_scanned: usize,
    pub total_trail_vertices_evaluated: usize,
    pub top_viewpoints: Vec<LandmarkViewpointHikeOption>,
    /// Precomputed daily solar trajectory: `[local_minutes, azimuth_deg, apparent_elevation_deg]` from 04:30 to 20:30
    pub daily_solar_track: Vec<[f64; 3]>,
}

fn subtract_minutes_from_hhmm(hhmm: &str, mins_to_sub: i64) -> String {
    let parts: Vec<&str> = hhmm.split(':').collect();
    let (h, m) = if parts.len() == 2 {
        (
            parts[0].parse::<i64>().unwrap_or(17),
            parts[1].parse::<i64>().unwrap_or(0),
        )
    } else {
        (17, 0)
    };
    let total = (h * 60 + m - mins_to_sub).rem_euclid(1440);
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Solves: "I want to take a photo of Landmark X on `date` — what are the Top K hikes & viewpoints
/// (VP1, VP2, VP3) that have an unobstructed view of Landmark X's illuminated face during golden hour,
/// and when must I depart each trailhead?"
pub fn find_best_viewpoints_and_hikes_for_landmark(
    db: &RecreationDbHarness,
    terrain: &HybridTerrain,
    landmark_query: &str,
    date: chrono::NaiveDate,
    max_miles: f64,
    top_k: usize,
) -> Result<LandmarkShootPlanReport> {
    let (canonical_name, seed_lat, seed_lon, ref_photo_url) =
        db.resolve_landmark_seed(landmark_query)?;
    find_best_viewpoints_for_seed(
        db,
        terrain,
        &canonical_name,
        seed_lat,
        seed_lon,
        ref_photo_url,
        date,
        max_miles,
        top_k,
    )
}

/// Computes Top K viewpoints & hikes for any arbitrary clicked coordinate `(seed_lat, seed_lon)` or named landmark.
pub fn find_best_viewpoints_for_seed(
    db: &RecreationDbHarness,
    terrain: &HybridTerrain,
    canonical_name: &str,
    seed_lat: f64,
    seed_lon: f64,
    ref_photo_url: Option<String>,
    date: chrono::NaiveDate,
    max_miles: f64,
    top_k: usize,
) -> Result<LandmarkShootPlanReport> {
    // 1. Extract 3D topographic summit, vertical relief, and primary sheer cliff normal around seed
    let target_3d = extract_landmark_3d_geometry(terrain, canonical_name, seed_lat, seed_lon);

    // 2. Trace solar rays onto Landmark X across Sunrise and Sunset golden hours
    let gh_windows = find_subject_golden_hour_windows(terrain, &target_3d, date);

    // 3. Load all hikeable trail routes within ~15 km (0.14°) of Landmark X
    let all_routes =
        db.find_all_routes_near(target_3d.summit_lat, target_3d.summit_lon, 0.14, max_miles)?;
    let total_routes_scanned = all_routes.len();

    // 4. Walk all routes in parallel to accumulate Tobler hike metrics & 3D trail paths, then evaluate
    //    ALL candidate 3D camera sightlines across every route in ONE batched Metal GPU compute pass!
    struct RouteWalkData {
        path_3d: Vec<[f64; 3]>,
        samples: Vec<(usize, f64, f64, f64, f64, f64, f64)>, // (vertex_idx, lat, lon, z, cum_dist_m, cum_gain_m, cum_time_min)
    }

    let walked_routes: Vec<Option<RouteWalkData>> = all_routes
        .par_iter()
        .map(|route| {
            if route.geometry.len() < 2 {
                return None;
            }
            let scan_len = effective_scan_len(route);

            let mut cum_dist_m = 0.0_f64;
            let mut cum_gain_m = 0.0_f64;
            let mut cum_time_min = 0.0_f64;
            let mut prev_z = terrain.elevation_m(route.geometry[0][1], route.geometry[0][0]);

            let mut path_3d: Vec<[f64; 3]> = Vec::with_capacity(scan_len);
            path_3d.push([
                route.geometry[0][0],
                route.geometry[0][1],
                (prev_z * 10.0).round() / 10.0,
            ]);

            let mut samples = Vec::with_capacity(scan_len / 3 + 2);

            for i in 1..scan_len {
                let (lon0, lat0) = (route.geometry[i - 1][0], route.geometry[i - 1][1]);
                let (lon1, lat1) = (route.geometry[i][0], route.geometry[i][1]);
                let seg_d_m = haversine_m(lat0, lon0, lat1, lon1);
                let z1 = terrain.elevation_m(lat1, lon1);
                path_3d.push([lon1, lat1, (z1 * 10.0).round() / 10.0]);

                let dz = z1 - prev_z;
                prev_z = z1;
                let gain = dz.max(0.0);
                let slope = if seg_d_m > 1.0 { dz / seg_d_m } else { 0.0 };
                let dt_min = if seg_d_m > 0.1 {
                    seg_d_m / tobler_speed_m_per_min(slope).max(10.0)
                } else {
                    0.0
                };

                cum_dist_m += seg_d_m;
                cum_gain_m += gain;
                cum_time_min += dt_min;

                if i % 3 != 0 && i != scan_len - 1 {
                    continue;
                }
                samples.push((i, lat1, lon1, z1, cum_dist_m, cum_gain_m, cum_time_min));
            }

            Some(RouteWalkData { path_3d, samples })
        })
        .collect();

    let mut flat_vps = Vec::new();
    let mut route_sample_ranges = Vec::with_capacity(all_routes.len());
    for wd_opt in &walked_routes {
        let start = flat_vps.len();
        if let Some(wd) = wd_opt {
            for &(_, lat, lon, z, _, _, _) in &wd.samples {
                flat_vps.push((lat, lon, z));
            }
        }
        route_sample_ranges.push((start, flat_vps.len()));
    }

    let mut total_trail_vertices_evaluated = flat_vps.len();
    let flat_los_results = cast_camera_line_of_sight_batch(terrain, &flat_vps, &target_3d);

    let mut candidates: Vec<LandmarkViewpointHikeOption> = all_routes
        .par_iter()
        .enumerate()
        .filter_map(|(r_idx, route)| {
            let wd = walked_routes[r_idx].as_ref()?;
            let (s_start, s_end) = route_sample_ranges[r_idx];
            let los_slice = &flat_los_results[s_start..s_end];

            let mut best_vp_for_route: Option<(usize, LandmarkViewpointHikeOption)> = None;

            for (k, &(v_idx, lat1, lon1, z1, cum_dist_m, cum_gain_m, cum_time_min)) in
                wd.samples.iter().enumerate()
            {
                let los = &los_slice[k];
                if !los.is_visible {
                    continue;
                }

                let mut schedules = Vec::new();
                let mut best_light_bonus = 0.0_f64;
                let hike_mins = cum_time_min.round() as u32;

                for win in &gh_windows {
                    let cam_from_target_az =
                        (los.bearing_to_landmark_deg + 180.0).rem_euclid(360.0);
                    let angle_diff = ((cam_from_target_az - win.sun_azimuth_deg + 540.0)
                        .rem_euclid(360.0)
                        - 180.0)
                        .abs();

                    let (lighting_style, angle_bonus) = if angle_diff <= 35.0 {
                        ("Direct Front Alpenglow (Warm Full-Face Light)", 18.0)
                    } else if angle_diff <= 105.0 {
                        (
                            "3/4 Side-Relief Alpenglow (Dramatic 3D Granite Texture & Shadows)",
                            24.0,
                        )
                    } else if angle_diff <= 145.0 {
                        ("Grazing Edge / Rim Light", 14.0)
                    } else {
                        ("Backlit Silhouette / Sunburst Alignment", 12.0)
                    };

                    if angle_bonus > best_light_bonus {
                        best_light_bonus = angle_bonus;
                    }

                    let arrival_local = subtract_minutes_from_hhmm(&win.window_start_local, 20);
                    let depart_local = subtract_minutes_from_hhmm(
                        &win.window_start_local,
                        20 + hike_mins as i64,
                    );

                    schedules.push(ViewpointWindowSchedule {
                        phase: win.phase.clone(),
                        shooting_window_local: format!(
                            "{} – {}",
                            win.window_start_local, win.window_end_local
                        ),
                        peak_light_local: win.peak_time_local.clone(),
                        recommended_trailhead_departure_local: depart_local,
                        target_viewpoint_arrival_local: arrival_local,
                        sun_azimuth_deg: win.sun_azimuth_deg,
                        sun_elevation_deg: win.sun_elevation_deg,
                        camera_to_sun_angle_deg: (angle_diff * 10.0).round() / 10.0,
                        lighting_style: lighting_style.to_string(),
                        valley_shadow_contrast_pct: win.valley_shadow_contrast_pct,
                    });
                }

                let hike_miles = cum_dist_m / 1609.344;
                let hike_bonus = if hike_miles <= 5.5 { 6.0 } else { 2.0 };
                let overall_score =
                    ((los.composition_score + best_light_bonus + hike_bonus) * 10.0).round() / 10.0;

                if best_vp_for_route
                    .as_ref()
                    .map_or(true, |(_, b)| overall_score > b.overall_score)
                {
                    let vp_label = if let Some(first_lm) = route.landmarks.first() {
                        format!(
                            "{} Corridor ({} viewpoint)",
                            first_lm, los.compass_direction
                        )
                    } else {
                        format!("{} Viewpoint", route.name)
                    };

                    best_vp_for_route = Some((
                        v_idx,
                        LandmarkViewpointHikeOption {
                            viewpoint_rank: 0,
                            viewpoint_label: vp_label,
                            viewpoint_lat: (lat1 * 100_000.0).round() / 100_000.0,
                            viewpoint_lon: (lon1 * 100_000.0).round() / 100_000.0,
                            viewpoint_elevation_m: (z1 * 10.0).round() / 10.0,
                            viewpoint_elevation_ft: (z1 * 3.28084).round(),
                            route_id: route.id,
                            route_name: route.name.clone(),
                            route_type: route.route_type.clone(),
                            trailhead_name: route.trailhead_name.clone(),
                            trailhead_lat: route.trailhead_lat,
                            trailhead_lon: route.trailhead_lon,
                            hike_distance_to_vp_miles: (hike_miles * 100.0).round() / 100.0,
                            total_route_distance_miles: route.distance_miles,
                            elevation_gain_to_vp_feet: (cum_gain_m * 3.28084).round(),
                            hike_duration_to_vp_minutes: hike_mins,
                            sightline: los.clone(),
                            overall_score,
                            schedules,
                            trail_path_to_vp: Vec::new(),
                        },
                    ));
                }
            }

            let (win_v_idx, mut vp_opt) = best_vp_for_route?;
            vp_opt.trail_path_to_vp = wd.path_3d[..=win_v_idx].to_vec();
            Some(vp_opt)
        })
        .collect();

    // If no indexed recreation trails exist in this region (e.g. Mt. Tamalpais, Mount Diablo, Point Reyes,
    // Rockies, PNW), synthesize raycasted 3D topographic ridge viewpoints around the summit!
    if candidates.is_empty() {
        let m_per_lat = METERS_PER_DEG_LAT;
        let m_per_lon = meters_per_deg_lon(target_3d.summit_lat);
        let radii_m = [1450.0_f64, 2350.0_f64, 3450.0_f64];
        let mut ridge_pts: Vec<(f64, f64, f64, f64, usize)> = Vec::new();
        for (rad_idx, &dist_m) in radii_m.iter().enumerate() {
            for az_step in 0..36 {
                let az_deg = (az_step as f64) * 10.0;
                let az_rad = az_deg.to_radians();
                let d_north_m = dist_m * az_rad.cos();
                let d_east_m = dist_m * az_rad.sin();
                let v_lat = target_3d.summit_lat + d_north_m / m_per_lat;
                let v_lon = target_3d.summit_lon + d_east_m / m_per_lon;
                let z1 = terrain.elevation_m(v_lat, v_lon);
                if z1 > 8.0 {
                    ridge_pts.push((v_lat, v_lon, z1, az_deg, rad_idx));
                }
            }
        }
        let batch_coords: Vec<(f64, f64, f64)> =
            ridge_pts.iter().map(|&(la, lo, z, _, _)| (la, lo, z)).collect();
        total_trail_vertices_evaluated += batch_coords.len();
        let ridge_los = cast_camera_line_of_sight_batch(terrain, &batch_coords, &target_3d);

        for (idx, &(lat1, lon1, z1, az_deg, rad_idx)) in ridge_pts.iter().enumerate() {
            let los = &ridge_los[idx];
            if !los.is_visible {
                continue;
            }
            // Build a 680m local ridge approach segment outward from the summit
            let az_rad = az_deg.to_radians();
            let th_lat = lat1 + (680.0 * az_rad.cos()) / m_per_lat;
            let th_lon = lon1 + (680.0 * az_rad.sin()) / m_per_lon;
            let mut path_3d = Vec::with_capacity(5);
            let mut prev_z = terrain.elevation_m(th_lat, th_lon);
            let mut gain_m = 0.0_f64;
            for s in 0..=4 {
                let frac = (s as f64) / 4.0;
                let p_lat = th_lat + (lat1 - th_lat) * frac;
                let p_lon = th_lon + (lon1 - th_lon) * frac;
                let pz = terrain.elevation_m(p_lat, p_lon);
                if s > 0 && pz > prev_z {
                    gain_m += pz - prev_z;
                }
                prev_z = pz;
                path_3d.push([p_lon, p_lat, (pz * 10.0).round() / 10.0]);
            }

            let hike_miles = 0.42 + (rad_idx as f64) * 0.35;
            let hike_mins = ((hike_miles * 22.0) + (gain_m * 0.06)).round().max(12.0) as u32;
            let mut schedules = Vec::new();
            let mut best_light_bonus = 0.0_f64;

            for win in &gh_windows {
                let cam_from_target_az = (los.bearing_to_landmark_deg + 180.0).rem_euclid(360.0);
                let angle_diff = ((cam_from_target_az - win.sun_azimuth_deg + 540.0)
                    .rem_euclid(360.0)
                    - 180.0)
                    .abs();
                let (lighting_style, angle_bonus) = if angle_diff <= 35.0 {
                    ("Direct Front Alpenglow (Warm Full-Face Light)", 18.0)
                } else if angle_diff <= 105.0 {
                    (
                        "3/4 Side-Relief Alpenglow (Dramatic 3D Ridge Texture & Shadows)",
                        24.0,
                    )
                } else if angle_diff <= 145.0 {
                    ("Grazing Edge / Rim Light", 14.0)
                } else {
                    ("Backlit Silhouette / Sunburst Alignment", 12.0)
                };
                if angle_bonus > best_light_bonus {
                    best_light_bonus = angle_bonus;
                }
                let arrival_local = subtract_minutes_from_hhmm(&win.window_start_local, 20);
                let depart_local =
                    subtract_minutes_from_hhmm(&win.window_start_local, 20 + hike_mins as i64);
                schedules.push(ViewpointWindowSchedule {
                    phase: win.phase.clone(),
                    shooting_window_local: format!(
                        "{} – {}",
                        win.window_start_local, win.window_end_local
                    ),
                    peak_light_local: win.peak_time_local.clone(),
                    recommended_trailhead_departure_local: depart_local,
                    target_viewpoint_arrival_local: arrival_local,
                    sun_azimuth_deg: win.sun_azimuth_deg,
                    sun_elevation_deg: win.sun_elevation_deg,
                    camera_to_sun_angle_deg: (angle_diff * 10.0).round() / 10.0,
                    lighting_style: lighting_style.to_string(),
                    valley_shadow_contrast_pct: win.valley_shadow_contrast_pct,
                });
            }

            let elev_ratio = (z1 / target_3d.summit_elevation_m.max(100.0)).clamp(0.15, 1.0);
            let prominence_bonus = elev_ratio * 6.0;
            let overall_score =
                ((los.composition_score + best_light_bonus + prominence_bonus) * 10.0).round()
                    / 10.0;
            let sector = &los.compass_direction;
            candidates.push(LandmarkViewpointHikeOption {
                viewpoint_rank: 0,
                viewpoint_label: format!("{} Ridge Overlook (Looking {})", target_3d.name, sector),
                viewpoint_lat: (lat1 * 100_000.0).round() / 100_000.0,
                viewpoint_lon: (lon1 * 100_000.0).round() / 100_000.0,
                viewpoint_elevation_m: (z1 * 10.0).round() / 10.0,
                viewpoint_elevation_ft: (z1 * 3.28084).round(),
                route_id: 9000 + (idx as i64),
                route_name: format!("{} {} Ridge Approach", target_3d.name, sector),
                route_type: "Ridge Spur".to_string(),
                trailhead_name: format!("{} Ridge Spur Access", target_3d.name),
                trailhead_lat: (th_lat * 100_000.0).round() / 100_000.0,
                trailhead_lon: (th_lon * 100_000.0).round() / 100_000.0,
                hike_distance_to_vp_miles: (hike_miles * 100.0).round() / 100.0,
                total_route_distance_miles: (hike_miles * 2.0 * 100.0).round() / 100.0,
                elevation_gain_to_vp_feet: (gain_m * 3.28084).round(),
                hike_duration_to_vp_minutes: hike_mins,
                sightline: los.clone(),
                overall_score,
                schedules,
                trail_path_to_vp: path_3d,
            });
        }
    }

    // Sort candidates by overall photographic score descending
    candidates.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());

    // 5. Select Top K spatially diverse viewpoints (at least 1,350 meters / ~0.85 miles apart)
    // so VP1, VP2, VP3 represent distinct vantage points and angles around Landmark X!
    let mut top_viewpoints: Vec<LandmarkViewpointHikeOption> = Vec::new();
    for mut cand in candidates {
        let too_close = top_viewpoints.iter().any(|existing| {
            haversine_m(
                cand.viewpoint_lat,
                cand.viewpoint_lon,
                existing.viewpoint_lat,
                existing.viewpoint_lon,
            ) < 1350.0
        });
        if !too_close {
            cand.viewpoint_rank = top_viewpoints.len() + 1;
            top_viewpoints.push(cand);
            if top_viewpoints.len() >= top_k {
                break;
            }
        }
    }

    // 6. Compute overnight Astrophotography, Lunar Alpenglow & Terrain Moonrise/Moonset summary
    let astro_night_summary = compute_subject_astro_night_summary(terrain, &target_3d, date);

    // 7. Compute daily solar track (every 5 minutes from 04:30 to 20:30 local solar time) for interactive UI scrubbing
    let approx_tz_hours = (target_3d.summit_lon / 15.0).round() as i64;
    let base_utc = chrono::TimeZone::from_utc_datetime(
        &chrono::Utc,
        &date.and_hms_opt(0, 0, 0).unwrap(),
    );
    let mut daily_solar_track = Vec::new();
    for local_min in (270..=1230).step_by(5) {
        let dt_utc = base_utc + chrono::Duration::minutes(local_min - approx_tz_hours * 60);
        let pos = pp_solar::solar_position(target_3d.summit_lat, target_3d.summit_lon, dt_utc);
        daily_solar_track.push([
            local_min as f64,
            (pos.azimuth_deg * 100.0).round() / 100.0,
            (pos.apparent_elevation_deg * 100.0).round() / 100.0,
        ]);
    }

    Ok(LandmarkShootPlanReport {
        target_landmark: target_3d,
        reference_photo_url: ref_photo_url,
        date: date.format("%Y-%m-%d").to_string(),
        golden_hour_windows: gh_windows,
        astro_night_summary,
        total_routes_scanned,
        total_trail_vertices_evaluated,
        top_viewpoints,
        daily_solar_track,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SightlineProfileSample {
    pub distance_miles: f64,
    pub distance_m: f64,
    pub terrain_elevation_ft: f64,
    pub terrain_elevation_m: f64,
    pub sightline_elevation_ft: f64,
    pub sightline_elevation_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTripodAnalysisReport {
    pub tripod_lat: f64,
    pub tripod_lon: f64,
    pub tripod_elevation_m: f64,
    pub tripod_elevation_ft: f64,
    pub target_name: String,
    pub target_lat: f64,
    pub target_lon: f64,
    pub target_elevation_m: f64,
    pub target_elevation_ft: f64,
    pub sightline: LineOfSightResult,
    pub max_obstruction_above_ray_ft: f64,
    pub obstruction_distance_miles: Option<f64>,
    pub sightline_profile: Vec<SightlineProfileSample>,
    pub nearest_trail_option: Option<LandmarkViewpointHikeOption>,
    pub off_trail_distance_ft: f64,
    pub light_notches: Vec<LightNotch>,
    /// 360-degree horizon profile `[azimuth_deg, horizon_elevation_deg]` sampled every 2 degrees
    pub horizon_curve: Vec<[f64; 2]>,
}

/// Evaluates any user-clicked tripod coordinate `(tripod_lat, tripod_lon)` against `(target_lat, target_lon)`:
/// 1. Casts a 3D camera sightline & builds a 64-point terrain vs. sightline cross-section profile.
/// 2. Snaps to the nearest hikeable trail in `recreation/db.sqlite3` and computes the Tobler hike itinerary & 3D trail path.
/// 3. Computes the 360° mountain horizon profile and detects narrow keyhole Light Notches from that exact spot.
pub fn analyze_custom_tripod_point(
    db: &RecreationDbHarness,
    terrain: &HybridTerrain,
    tripod_lat: f64,
    tripod_lon: f64,
    target_lat: f64,
    target_lon: f64,
    target_name: &str,
    date: chrono::NaiveDate,
) -> Result<CustomTripodAnalysisReport> {
    let tripod_z = terrain.elevation_m(tripod_lat, tripod_lon);
    let target_3d = extract_landmark_3d_geometry(terrain, target_name, target_lat, target_lon);
    let los = cast_camera_line_of_sight(terrain, tripod_lat, tripod_lon, tripod_z, &target_3d);

    // 1. Build 64-step Sightline Terrain Cross-Section Profile from Tripod -> Target Upper Face
    let steps = 64usize;
    let total_d_m = haversine_m(
        tripod_lat,
        tripod_lon,
        target_3d.upper_face_lat,
        target_3d.upper_face_lon,
    )
    .max(10.0);
    let cam_eye_z = tripod_z + 1.75;
    let aim_z = target_3d.upper_face_z_m;
    let drop_target = (total_d_m * total_d_m) / (2.0 * 6_371_000.0) * 0.87;
    let eff_aim_z = aim_z - drop_target;

    let mut sightline_profile = Vec::with_capacity(steps + 1);
    let mut max_obstruction_m = 0.0_f64;
    let mut obstruction_dist_mi: Option<f64> = None;

    for s in 0..=steps {
        let frac = (s as f64) / (steps as f64);
        let d_m = frac * total_d_m;
        let d_mi = d_m / 1609.344;
        let slat = tripod_lat + frac * (target_3d.upper_face_lat - tripod_lat);
        let slon = tripod_lon + frac * (target_3d.upper_face_lon - tripod_lon);
        let tz = terrain.elevation_m(slat, slon);
        let curv_drop = (d_m * d_m) / (2.0 * 6_371_000.0) * 0.87;
        let eff_tz = tz - curv_drop;
        let ray_z = cam_eye_z + frac * (eff_aim_z - cam_eye_z);

        if s > 2 && s < steps - 2 {
            let diff = eff_tz - ray_z;
            if diff > max_obstruction_m {
                max_obstruction_m = diff;
                obstruction_dist_mi = Some((d_mi * 100.0).round() / 100.0);
            }
        }

        sightline_profile.push(SightlineProfileSample {
            distance_miles: (d_mi * 100.0).round() / 100.0,
            distance_m: d_m.round(),
            terrain_elevation_ft: (eff_tz * 3.28084).round(),
            terrain_elevation_m: (eff_tz * 10.0).round() / 10.0,
            sightline_elevation_ft: (ray_z * 3.28084).round(),
            sightline_elevation_m: (ray_z * 10.0).round() / 10.0,
        });
    }

    // 2. Snap to the nearest trail route in `recreation/db.sqlite3` and compute the hike up to the closest vertex
    let gh_windows = find_subject_golden_hour_windows(terrain, &target_3d, date);
    let nearby_routes = db.find_all_routes_near(tripod_lat, tripod_lon, 0.08, 24.0)?;
    let mut best_snap: Option<(f64, LandmarkViewpointHikeOption)> = None;

    for route in &nearby_routes {
        if route.geometry.len() < 2 {
            continue;
        }
        let scan_len = effective_scan_len(route);

        let mut cum_dist_m = 0.0_f64;
        let mut cum_gain_m = 0.0_f64;
        let mut cum_time_min = 0.0_f64;
        let mut prev_z = terrain.elevation_m(route.geometry[0][1], route.geometry[0][0]);
        let mut path_3d: Vec<[f64; 3]> = Vec::with_capacity(scan_len);
        path_3d.push([
            route.geometry[0][0],
            route.geometry[0][1],
            (prev_z * 10.0).round() / 10.0,
        ]);

        for i in 1..scan_len {
            let (lon0, lat0) = (route.geometry[i - 1][0], route.geometry[i - 1][1]);
            let (lon1, lat1) = (route.geometry[i][0], route.geometry[i][1]);
            let seg_d_m = haversine_m(lat0, lon0, lat1, lon1);
            let z1 = terrain.elevation_m(lat1, lon1);
            path_3d.push([lon1, lat1, (z1 * 10.0).round() / 10.0]);

            let dz = z1 - prev_z;
            prev_z = z1;
            let gain = dz.max(0.0);
            let slope = if seg_d_m > 1.0 { dz / seg_d_m } else { 0.0 };
            let dt_min = if seg_d_m > 0.1 {
                seg_d_m / tobler_speed_m_per_min(slope).max(10.0)
            } else {
                0.0
            };

            cum_dist_m += seg_d_m;
            cum_gain_m += gain;
            cum_time_min += dt_min;

            let off_trail_m = haversine_m(tripod_lat, tripod_lon, lat1, lon1);
            if best_snap.as_ref().map_or(true, |(best_d, _)| off_trail_m < *best_d) {
                // Include off-trail walk time if user clicked slightly off-trail
                let off_dz = (tripod_z - z1).max(0.0);
                let total_dist_m = cum_dist_m + off_trail_m;
                let total_gain_m = cum_gain_m + off_dz;
                let hike_mins = (cum_time_min + off_trail_m / 45.0).round() as u32;

                let mut trail_with_pin = path_3d.clone();
                if off_trail_m > 8.0 {
                    trail_with_pin.push([
                        tripod_lon,
                        tripod_lat,
                        (tripod_z * 10.0).round() / 10.0,
                    ]);
                }

                let mut schedules = Vec::new();
                for win in &gh_windows {
                    let arrival_local = subtract_minutes_from_hhmm(&win.window_start_local, 20);
                    let depart_local = subtract_minutes_from_hhmm(
                        &win.window_start_local,
                        20 + hike_mins as i64,
                    );
                    schedules.push(ViewpointWindowSchedule {
                        phase: win.phase.clone(),
                        shooting_window_local: format!(
                            "{} – {}",
                            win.window_start_local, win.window_end_local
                        ),
                        peak_light_local: win.peak_time_local.clone(),
                        recommended_trailhead_departure_local: depart_local,
                        target_viewpoint_arrival_local: arrival_local,
                        sun_azimuth_deg: win.sun_azimuth_deg,
                        sun_elevation_deg: win.sun_elevation_deg,
                        camera_to_sun_angle_deg: 65.0,
                        lighting_style: "Custom Tripod Vantage".to_string(),
                        valley_shadow_contrast_pct: win.valley_shadow_contrast_pct,
                    });
                }

                let hike_miles = total_dist_m / 1609.344;
                let opt = LandmarkViewpointHikeOption {
                    viewpoint_rank: 0,
                    viewpoint_label: format!("Custom Tripod via {}", route.name),
                    viewpoint_lat: (tripod_lat * 100_000.0).round() / 100_000.0,
                    viewpoint_lon: (tripod_lon * 100_000.0).round() / 100_000.0,
                    viewpoint_elevation_m: (tripod_z * 10.0).round() / 10.0,
                    viewpoint_elevation_ft: (tripod_z * 3.28084).round(),
                    route_id: route.id,
                    route_name: route.name.clone(),
                    route_type: route.route_type.clone(),
                    trailhead_name: route.trailhead_name.clone(),
                    trailhead_lat: route.trailhead_lat,
                    trailhead_lon: route.trailhead_lon,
                    hike_distance_to_vp_miles: (hike_miles * 100.0).round() / 100.0,
                    total_route_distance_miles: route.distance_miles,
                    elevation_gain_to_vp_feet: (total_gain_m * 3.28084).round(),
                    hike_duration_to_vp_minutes: hike_mins,
                    sightline: los.clone(),
                    overall_score: los.composition_score,
                    schedules,
                    trail_path_to_vp: trail_with_pin,
                };
                best_snap = Some((off_trail_m, opt));
            }
        }
    }

    let (off_trail_m, nearest_trail_option) = match best_snap {
        Some((d, opt)) => (d, Some(opt)),
        None => (0.0, None),
    };

    // 3. 360-degree horizon profile & Light Notches from the clicked tripod point
    let profile = compute_horizon_profile(
        terrain,
        tripod_lat,
        tripod_lon,
        1.75,
        0.0,
        359.5,
        0.5,
        32_000.0,
    );
    let year = chrono::Datelike::year(&date);
    let mut light_notches = detect_light_notches(&profile, year, 0.65, 4.5);
    light_notches.truncate(5);
    let horizon_curve = profile
        .samples
        .iter()
        .step_by(4)
        .map(|s| {
            [
                (s.azimuth_deg * 10.0).round() / 10.0,
                (s.horizon_elevation_deg * 100.0).round() / 100.0,
            ]
        })
        .collect();

    Ok(CustomTripodAnalysisReport {
        tripod_lat: (tripod_lat * 100_000.0).round() / 100_000.0,
        tripod_lon: (tripod_lon * 100_000.0).round() / 100_000.0,
        tripod_elevation_m: (tripod_z * 10.0).round() / 10.0,
        tripod_elevation_ft: (tripod_z * 3.28084).round(),
        target_name: target_3d.name,
        target_lat: target_3d.summit_lat,
        target_lon: target_3d.summit_lon,
        target_elevation_m: target_3d.summit_elevation_m,
        target_elevation_ft: target_3d.summit_elevation_ft,
        sightline: los,
        max_obstruction_above_ray_ft: (max_obstruction_m * 3.28084).round(),
        obstruction_distance_miles: obstruction_dist_mi,
        sightline_profile,
        nearest_trail_option,
        off_trail_distance_ft: (off_trail_m * 3.28084).round(),
        light_notches,
        horizon_curve,
    })
}

/// Task 3.1: A single resampled corridor point spaced every `150–250 m` along a trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailCorridorSample {
    pub milepost: f64,
    pub lat: f64,
    pub lon: f64,
    pub elevation_m: f64,
    pub elevation_ft: f64,
    pub cumulative_gain_ft: f64,
    pub tobler_minutes: u32,
    pub slope_pct: f64,
    pub horizon_openness_pct: f64,
    pub above_fog_p50: bool,
    pub above_fog_p90: bool,
}

/// A contiguous segment along a trail where hikers emerge into open skyline or above the fog lid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailBreakoutZone {
    pub zone_kind: String, // "horizon_opening" | "fog_breakout"
    pub start_milepost: f64,
    pub end_milepost: f64,
    pub peak_elevation_ft: f64,
    pub peak_openness_pct: f64,
    pub tobler_minutes_to_entry: u32,
    pub summary: String,
}

/// Task 3.2: Four non-prescriptive trail photography nudge categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrailNudgeCategory {
    HorizonOpening,
    LightNotchWindow,
    FogBreakout,
    HeadlampSafety,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailNudge {
    pub category: TrailNudgeCategory,
    pub milepost: f64,
    pub elevation_ft: f64,
    pub trigger_time_local: Option<String>,
    pub headline: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailCorridorScanReport {
    pub route_name: String,
    pub route_type: String,
    pub spacing_m: f64,
    pub scanned_length_miles: f64,
    pub samples: Vec<TrailCorridorSample>,
    pub breakout_zones: Vec<TrailBreakoutZone>,
    pub nudges: Vec<TrailNudge>,
}

/// Task 3.1: Resamples a 3D trail polyline (`[lon, lat, elev_m]`) at continuous `spacing_m`
/// intervals (clamped to `150–250 m`), computing cumulative Tobler hiking time, local horizon
/// openness proxy (from relative topographic prominence or supplied raycasted openness), and
/// contiguous Horizon-Opening & Fog-Breakout zones across the FULL loop or out-and-back corridor.
pub fn scan_trail_corridor_breakouts(
    route: &TrailRouteRecord,
    elevations_m: &[f64],
    spacing_m: f64,
    fog_top_p50_ft: Option<f64>,
    fog_top_p90_ft: Option<f64>,
    horizon_openness_override: Option<&[f64]>,
) -> TrailCorridorScanReport {
    let spacing = spacing_m.clamp(150.0, 250.0);
    let scan_len = effective_scan_len(route).min(elevations_m.len());
    if scan_len < 2 {
        return TrailCorridorScanReport {
            route_name: route.name.clone(),
            route_type: route.route_type.clone(),
            spacing_m: spacing,
            scanned_length_miles: 0.0,
            samples: Vec::new(),
            breakout_zones: Vec::new(),
            nudges: Vec::new(),
        };
    }

    let min_z = elevations_m[..scan_len]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_z = elevations_m[..scan_len]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let z_span = (max_z - min_z).max(1.0);

    let mut samples: Vec<TrailCorridorSample> = Vec::new();
    let mut cum_dist_m = 0.0_f64;
    let mut cum_gain_m = 0.0_f64;
    let mut cum_time_min = 0.0_f64;
    let mut next_sample_m = 0.0_f64;

    for i in 0..scan_len {
        let lon = route.geometry[i][0];
        let lat = route.geometry[i][1];
        let z = elevations_m[i];
        let mut slope = 0.0_f64;

        if i > 0 {
            let prev_lon = route.geometry[i - 1][0];
            let prev_lat = route.geometry[i - 1][1];
            let prev_z = elevations_m[i - 1];
            let seg_d = haversine_m(prev_lat, prev_lon, lat, lon);
            let dz = z - prev_z;
            slope = if seg_d > 1.0 { dz / seg_d } else { 0.0 };
            cum_dist_m += seg_d;
            cum_gain_m += dz.max(0.0);
            if seg_d > 0.1 {
                cum_time_min += seg_d / tobler_speed_m_per_min(slope).max(10.0);
            }
        }

        if cum_dist_m + 1e-6 >= next_sample_m || i == scan_len - 1 {
            let elev_ft = z * 3.28084;
            let rel_prom = ((z - min_z) / z_span).clamp(0.0, 1.0);
            let openness = horizon_openness_override
                .and_then(|arr| arr.get(i).copied())
                .unwrap_or_else(|| (25.0 + 70.0 * rel_prom).clamp(0.0, 100.0));
            let above_p50 = fog_top_p50_ft.map_or(false, |lid| elev_ft >= lid);
            let above_p90 = fog_top_p90_ft.map_or(false, |lid| elev_ft >= lid);

            samples.push(TrailCorridorSample {
                milepost: ((cum_dist_m / 1609.344) * 100.0).round() / 100.0,
                lat: (lat * 100_000.0).round() / 100_000.0,
                lon: (lon * 100_000.0).round() / 100_000.0,
                elevation_m: (z * 10.0).round() / 10.0,
                elevation_ft: elev_ft.round(),
                cumulative_gain_ft: (cum_gain_m * 3.28084).round(),
                tobler_minutes: cum_time_min.round() as u32,
                slope_pct: (slope * 1000.0).round() / 10.0,
                horizon_openness_pct: (openness * 10.0).round() / 10.0,
                above_fog_p50: above_p50,
                above_fog_p90: above_p90,
            });
            next_sample_m = cum_dist_m + spacing;
        }
    }

    let mut breakout_zones: Vec<TrailBreakoutZone> = Vec::new();

    // Detect contiguous Horizon Opening breakout zones (openness >= 65%)
    let mut in_open = false;
    let mut start_idx = 0usize;
    for (idx, s) in samples.iter().enumerate() {
        let is_open = s.horizon_openness_pct >= 65.0;
        if is_open && !in_open {
            in_open = true;
            start_idx = idx;
        } else if (!is_open || idx == samples.len() - 1) && in_open {
            let end_idx = if is_open { idx } else { idx - 1 };
            let slice = &samples[start_idx..=end_idx];
            let peak_el = slice.iter().map(|x| x.elevation_ft).fold(0.0_f64, f64::max);
            let peak_op = slice
                .iter()
                .map(|x| x.horizon_openness_pct)
                .fold(0.0_f64, f64::max);
            breakout_zones.push(TrailBreakoutZone {
                zone_kind: "horizon_opening".to_string(),
                start_milepost: slice[0].milepost,
                end_milepost: slice[slice.len() - 1].milepost,
                peak_elevation_ft: peak_el,
                peak_openness_pct: peak_op,
                tobler_minutes_to_entry: slice[0].tobler_minutes,
                summary: format!(
                    "Open skyline corridor from Mile {:.2}–{:.2} (up to {:.0} ft, {:.0}% horizon openness).",
                    slice[0].milepost,
                    slice[slice.len() - 1].milepost,
                    peak_el,
                    peak_op
                ),
            });
            in_open = false;
        }
    }

    // Detect contiguous Fog-Lid Breakout zones (above expected P50 lid)
    if let Some(lid_p50) = fog_top_p50_ft {
        let mut in_above = false;
        let mut f_start = 0usize;
        for (idx, s) in samples.iter().enumerate() {
            if s.above_fog_p50 && !in_above {
                in_above = true;
                f_start = idx;
            } else if (!s.above_fog_p50 || idx == samples.len() - 1) && in_above {
                let f_end = if s.above_fog_p50 { idx } else { idx - 1 };
                let slice = &samples[f_start..=f_end];
                let peak_el = slice.iter().map(|x| x.elevation_ft).fold(0.0_f64, f64::max);
                breakout_zones.push(TrailBreakoutZone {
                    zone_kind: "fog_breakout".to_string(),
                    start_milepost: slice[0].milepost,
                    end_milepost: slice[slice.len() - 1].milepost,
                    peak_elevation_ft: peak_el,
                    peak_openness_pct: slice
                        .iter()
                        .map(|x| x.horizon_openness_pct)
                        .fold(0.0_f64, f64::max),
                    tobler_minutes_to_entry: slice[0].tobler_minutes,
                    summary: format!(
                        "Above expected {:.0} ft fog lid from Mile {:.2}–{:.2} (peak {:.0} ft).",
                        lid_p50,
                        slice[0].milepost,
                        slice[slice.len() - 1].milepost,
                        peak_el
                    ),
                });
                in_above = false;
            }
        }
    }

    let scanned_miles = ((cum_dist_m / 1609.344) * 100.0).round() / 100.0;
    let nudges = generate_trail_nudges(&samples, &breakout_zones, None, None, fog_top_p50_ft, fog_top_p90_ft);

    TrailCorridorScanReport {
        route_name: route.name.clone(),
        route_type: route.route_type.clone(),
        spacing_m: spacing,
        scanned_length_miles: scanned_miles,
        samples,
        breakout_zones,
        nudges,
    }
}

/// Task 3.2: Generates gentle, non-prescriptive Trail Nudges across 4 field categories:
///   1. `HorizonOpening` — where the trail breaks out into open views
///   2. `LightNotchWindow` — optional detected topographic notch alignment window
///   3. `FogBreakout` — expected (P50) and high-confidence (P90) cloud-sea breakout mileposts
///   4. `HeadlampSafety` — time-reversed departure & twilight return headlamp reminder
pub fn generate_trail_nudges(
    samples: &[TrailCorridorSample],
    breakout_zones: &[TrailBreakoutZone],
    light_notch_window: Option<(f64, &str, u32)>, // (milepost, peak_local_hhmm, duration_min)
    target_golden_hour_hhmm: Option<&str>,
    fog_top_p50_ft: Option<f64>,
    fog_top_p90_ft: Option<f64>,
) -> Vec<TrailNudge> {
    let mut nudges: Vec<TrailNudge> = Vec::new();

    if let Some(hz) = breakout_zones.iter().find(|z| z.zone_kind == "horizon_opening") {
        nudges.push(TrailNudge {
            category: TrailNudgeCategory::HorizonOpening,
            milepost: hz.start_milepost,
            elevation_ft: hz.peak_elevation_ft,
            trigger_time_local: None,
            headline: format!("Open Horizon Begins at Mile {:.2}", hz.start_milepost),
            detail: hz.summary.clone(),
        });
    }

    if let Some((mp, peak_time, dur_min)) = light_notch_window {
        let elev_ft = samples
            .iter()
            .min_by(|a, b| (a.milepost - mp).abs().partial_cmp(&(b.milepost - mp).abs()).unwrap())
            .map_or(0.0, |s| s.elevation_ft);
        nudges.push(TrailNudge {
            category: TrailNudgeCategory::LightNotchWindow,
            milepost: mp,
            elevation_ft: elev_ft,
            trigger_time_local: Some(peak_time.to_string()),
            headline: format!("Light Notch Alignment (~{} min) at Mile {:.2}", dur_min, mp),
            detail: format!(
                "Topographic horizon notch channels grazing light around {} near Mile {:.2}.",
                peak_time, mp
            ),
        });
    }

    if let Some(lid_p50) = fog_top_p50_ft {
        if let Some(first_p50) = samples.iter().find(|s| s.above_fog_p50) {
            let p90_note = match (fog_top_p90_ft, samples.iter().find(|s| s.above_fog_p90)) {
                (Some(lid_p90), Some(s90)) => format!(
                    " >90% clear-sky odds past Mile {:.2} ({:.0} ft, P90 lid {:.0} ft).",
                    s90.milepost, s90.elevation_ft, lid_p90
                ),
                _ => String::new(),
            };
            nudges.push(TrailNudge {
                category: TrailNudgeCategory::FogBreakout,
                milepost: first_p50.milepost,
                elevation_ft: first_p50.elevation_ft,
                trigger_time_local: None,
                headline: format!("Cloud-Sea Breakout at Mile {:.2}", first_p50.milepost),
                detail: format!(
                    "Trail climbs above the expected {:.0} ft inversion lid around Mile {:.2} ({:.0} ft).{}",
                    lid_p50, first_p50.milepost, first_p50.elevation_ft, p90_note
                ),
            });
        }
    }

    if let Some(peak_s) = samples
        .iter()
        .max_by(|a, b| a.elevation_ft.partial_cmp(&b.elevation_ft).unwrap())
    {
        let depart_time = target_golden_hour_hhmm
            .map(|hhmm| subtract_minutes_from_hhmm(hhmm, peak_s.tobler_minutes as i64 + 20));
        let depart_clause = match &depart_time {
            Some(t) => format!("Leave the trailhead around {} ", t),
            None => "Budget ".to_string(),
        };
        nudges.push(TrailNudge {
            category: TrailNudgeCategory::HeadlampSafety,
            milepost: peak_s.milepost,
            elevation_ft: peak_s.elevation_ft,
            trigger_time_local: depart_time,
            headline: format!("Pacing & Headlamp Window ({} min climb)", peak_s.tobler_minutes),
            detail: format!(
                "{}({} min Tobler pace + 20 min setup to Mile {:.2}). Pack a headlamp for blue-hour approach or descent.",
                depart_clause, peak_s.tobler_minutes, peak_s.milepost
            ),
        });
    }

    nudges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_route(route_type: &str, geom: Vec<[f64; 2]>) -> TrailRouteRecord {
        TrailRouteRecord {
            id: 1,
            name: "Test Ridge".to_string(),
            route_type: route_type.to_string(),
            distance_miles: 4.0,
            elevation_gain_feet: 1500.0,
            trailhead_id: "th-1".to_string(),
            trailhead_name: "Test TH".to_string(),
            trailhead_lat: geom[0][1],
            trailhead_lon: geom[0][0],
            geometry: geom,
            landmarks: vec!["Ridge".to_string()],
        }
    }

    #[test]
    fn loop_trail_scans_100_percent_never_truncated() {
        // Closed square loop ~2 km perimeter
        let geom = vec![
            [-122.500, 37.560],
            [-122.495, 37.560],
            [-122.495, 37.565],
            [-122.500, 37.565],
            [-122.500, 37.560],
        ];
        let route = make_route("Loop", geom.clone());
        assert_eq!(effective_scan_len(&route), geom.len());
    }

    #[test]
    fn doubled_out_and_back_stops_at_true_apex() {
        // Outbound 4 vertices to apex at index 3, then return 3 vertices back to start
        let geom = vec![
            [-122.500, 37.560],
            [-122.496, 37.560],
            [-122.492, 37.560],
            [-122.488, 37.560], // apex (farthest)
            [-122.492, 37.560],
            [-122.496, 37.560],
            [-122.500, 37.560],
        ];
        let route = make_route("Out & Back", geom);
        assert_eq!(effective_scan_len(&route), 4);
    }

    #[test]
    fn single_leg_out_and_back_scans_full_leg() {
        let geom = vec![
            [-122.500, 37.560],
            [-122.495, 37.560],
            [-122.490, 37.560],
            [-122.485, 37.560],
        ];
        let route = make_route("Out & Back", geom.clone());
        assert_eq!(effective_scan_len(&route), geom.len());
    }

    #[test]
    fn continuous_scanner_and_four_category_nudges_detect_fog_breakout() {
        // 10-vertex ascending ridge loop ~2.7 km
        let mut geom = Vec::new();
        let mut elevs = Vec::new();
        for i in 0..10 {
            geom.push([-122.500 + (i as f64) * 0.003, 37.560]);
            elevs.push(200.0 + (i as f64) * 70.0); // 656 ft -> 2723 ft
        }
        let route = make_route("Loop", geom);
        let report = scan_trail_corridor_breakouts(
            &route,
            &elevs,
            200.0,
            Some(1500.0),
            Some(1950.0),
            None,
        );
        assert!(report.samples.len() >= 8);
        assert!(report.breakout_zones.iter().any(|z| z.zone_kind == "fog_breakout"));
        assert!(report.breakout_zones.iter().any(|z| z.zone_kind == "horizon_opening"));

        let nudges = generate_trail_nudges(
            &report.samples,
            &report.breakout_zones,
            Some((1.1, "19:22", 14)),
            Some("19:30"),
            Some(1500.0),
            Some(1950.0),
        );
        assert_eq!(nudges.len(), 4);
        assert!(nudges.iter().any(|n| n.category == TrailNudgeCategory::HorizonOpening));
        assert!(nudges.iter().any(|n| n.category == TrailNudgeCategory::LightNotchWindow));
        assert!(nudges.iter().any(|n| n.category == TrailNudgeCategory::FogBreakout));
        assert!(nudges.iter().any(|n| n.category == TrailNudgeCategory::HeadlampSafety));
    }
}



