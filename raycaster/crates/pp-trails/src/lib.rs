use anyhow::{Context, Result};
use pp_raycast::{
    cast_camera_line_of_sight, compute_horizon_profile, compute_subject_astro_night_summary,
    detect_light_notches, extract_landmark_3d_geometry, find_subject_golden_hour_windows,
    LandmarkTarget3D, LightNotch, LineOfSightResult, SubjectAstroNightSummary,
    SubjectGoldenHourWindow,
};
use pp_terrain::{ElevationSurface, HybridTerrain};
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
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open recreation DB at {}", db_path.display()))?;
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

    // 1. Walk the first half of the route geometry (up to turnaround / apex) to compute cumulative
    // distance, elevation gain, and Tobler hiking duration at every vertex.
    let half_len = (route.geometry.len() / 2 + 1).max(2).min(route.geometry.len());
    let mut cum_dist_m = vec![0.0_f64; half_len];
    let mut cum_gain_m = vec![0.0_f64; half_len];
    let mut cum_time_min = vec![0.0_f64; half_len];
    let mut vertex_z = vec![0.0_f64; half_len];

    vertex_z[0] = terrain.elevation_m(route.geometry[0][1], route.geometry[0][0]);

    for i in 1..half_len {
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

    // 2. Select `sample_count` spatially diverse candidate vertices along the trail
    // prioritizing the upper/scenic half and destination turnaround
    let mut candidate_indices = Vec::new();
    let step = (half_len / sample_count.max(1)).max(1);
    for idx in (step..half_len).step_by(step) {
        candidate_indices.push(idx);
    }
    if !candidate_indices.contains(&(half_len - 1)) {
        candidate_indices.push(half_len - 1);
    }

    // 3. Evaluate horizon profiles and Light Notches at each candidate trail point in parallel
    let ((sr_min, sr_max), (ss_min, ss_max)) =
        pp_solar::annual_solar_azimuth_windows(route.trailhead_lat);

    let mut vantage_plans: Vec<TrailVantageShootPlan> = candidate_indices
        .into_par_iter()
        .filter_map(|idx| {
            let lon = route.geometry[idx][0];
            let lat = route.geometry[idx][1];

            // Compute sunrise and sunset horizon windows at 0.10° resolution (separately to avoid azimuth jump)
            let sr_profile =
                compute_horizon_profile(terrain, lat, lon, 1.65, sr_min, sr_max, 0.10, 35_000.0);
            let ss_profile =
                compute_horizon_profile(terrain, lat, lon, 1.65, ss_min, ss_max, 0.10, 35_000.0);

            let mut notches = detect_light_notches(&sr_profile, year, 0.55, 8.0);
            notches.extend(detect_light_notches(&ss_profile, year, 0.55, 8.0));
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

    // 4. In parallel across all routes, walk the trail geometry up to its turnaround point,
    // cast 3D Line-of-Sight rays to Landmark X, and pick the best unobstructed viewpoint on each route.
    let route_evaluations: Vec<(usize, Option<LandmarkViewpointHikeOption>)> = all_routes
        .par_iter()
        .map(|route| {
            if route.geometry.len() < 2 {
                return (0, None);
            }
            let half_len = (route.geometry.len() / 2 + 1)
                .max(2)
                .min(route.geometry.len());

            let mut cum_dist_m = 0.0_f64;
            let mut cum_gain_m = 0.0_f64;
            let mut cum_time_min = 0.0_f64;
            let mut prev_z = terrain.elevation_m(route.geometry[0][1], route.geometry[0][0]);

            let mut best_vp_for_route: Option<LandmarkViewpointHikeOption> = None;
            let mut vertices_checked = 0usize;
            let mut path_3d: Vec<[f64; 3]> = Vec::with_capacity(half_len);
            path_3d.push([
                route.geometry[0][0],
                route.geometry[0][1],
                (prev_z * 10.0).round() / 10.0,
            ]);

            // Sample every ~4th vertex along the trail (plus the turnaround apex)
            for i in 1..half_len {
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

                if i % 3 != 0 && i != half_len - 1 {
                    continue;
                }
                vertices_checked += 1;

                let los = cast_camera_line_of_sight(terrain, lat1, lon1, z1, &target_3d);
                if !los.is_visible {
                    continue;
                }

                // Build Golden-Hour shooting schedules & evaluate lighting angle between camera sightline and sun ray
                let mut schedules = Vec::new();
                let mut best_light_bonus = 0.0_f64;
                let hike_mins = cum_time_min.round() as u32;

                for win in &gh_windows {
                    // Angle between camera->subject bearing and sun->subject ray
                    // Camera is at `los.bearing_to_landmark_deg + 180°` relative to Landmark X
                    let cam_from_target_az = (los.bearing_to_landmark_deg + 180.0).rem_euclid(360.0);
                    let angle_diff = ((cam_from_target_az - win.sun_azimuth_deg + 540.0)
                        .rem_euclid(360.0)
                        - 180.0)
                        .abs();

                    let (lighting_style, angle_bonus) = if angle_diff <= 35.0 {
                        ("Direct Front Alpenglow (Warm Full-Face Light)", 18.0)
                    } else if angle_diff <= 105.0 {
                        ("3/4 Side-Relief Alpenglow (Dramatic 3D Granite Texture & Shadows)", 24.0)
                    } else if angle_diff <= 145.0 {
                        ("Grazing Edge / Rim Light", 14.0)
                    } else {
                        ("Backlit Silhouette / Sunburst Alignment", 12.0)
                    };

                    if angle_bonus > best_light_bonus {
                        best_light_bonus = angle_bonus;
                    }

                    // Target arriving 20 minutes before the Golden Hour Window starts
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

                // Slight efficiency bonus for reasonable hike distance (1.0..6.0 miles to VP)
                let hike_miles = cum_dist_m / 1609.344;
                let hike_bonus = if hike_miles <= 5.5 { 6.0 } else { 2.0 };
                let overall_score =
                    ((los.composition_score + best_light_bonus + hike_bonus) * 10.0).round() / 10.0;

                if best_vp_for_route
                    .as_ref()
                    .map_or(true, |b| overall_score > b.overall_score)
                {
                    let vp_label = if let Some(first_lm) = route.landmarks.first() {
                        format!(
                            "{} Corridor ({} viewpoint)",
                            first_lm, los.compass_direction
                        )
                    } else {
                        format!("{} Viewpoint", route.name)
                    };

                    best_vp_for_route = Some(LandmarkViewpointHikeOption {
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
                        sightline: los,
                        overall_score,
                        schedules,
                        trail_path_to_vp: path_3d.clone(),
                    });
                }
            }

            (vertices_checked, best_vp_for_route)
        })
        .collect();

    let total_trail_vertices_evaluated: usize = route_evaluations.iter().map(|e| e.0).sum();
    let mut candidates: Vec<LandmarkViewpointHikeOption> = route_evaluations
        .into_iter()
        .filter_map(|e| e.1)
        .collect();

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
        let ray_z = cam_eye_z + frac * (aim_z - cam_eye_z);

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
        let half_len = (route.geometry.len() / 2 + 1)
            .max(2)
            .min(route.geometry.len());

        let mut cum_dist_m = 0.0_f64;
        let mut cum_gain_m = 0.0_f64;
        let mut cum_time_min = 0.0_f64;
        let mut prev_z = terrain.elevation_m(route.geometry[0][1], route.geometry[0][0]);
        let mut path_3d: Vec<[f64; 3]> = Vec::with_capacity(half_len);
        path_3d.push([
            route.geometry[0][0],
            route.geometry[0][1],
            (prev_z * 10.0).round() / 10.0,
        ]);

        for i in 1..half_len {
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


