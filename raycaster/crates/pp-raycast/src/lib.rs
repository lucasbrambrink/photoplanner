use anyhow::Result;
use image::{ImageBuffer, Rgb};
use pp_solar::{
    annual_solar_azimuth_windows, find_notch_solar_alignments, SolarAlignmentWindow, SolarPosition,
};
use pp_terrain::{
    earth_curvature_refraction_drop_m, meters_per_deg_lon, ElevationSurface, HybridTerrain,
    MaxMipmapGrid1m, METERS_PER_DEG_LAT,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HorizonSample {
    pub azimuth_deg: f64,
    pub horizon_elevation_deg: f64,
    pub occluder_distance_m: f64,
    pub occluder_height_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonProfile {
    pub origin_lat: f64,
    pub origin_lon: f64,
    pub origin_elevation_m: f64,
    pub eye_offset_m: f64,
    pub azimuth_step_deg: f64,
    pub samples: Vec<HorizonSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightNotch {
    /// Center azimuth of the notch floor in degrees [0, 360)
    pub azimuth_deg: f64,
    /// Horizon elevation angle at the bottom of the notch (degrees)
    pub floor_elevation_deg: f64,
    /// Minimum of left and right flanking wall elevations (degrees)
    pub wall_elevation_deg: f64,
    /// Left wall peak elevation (degrees) and azimuth
    pub left_wall_deg: f64,
    pub left_wall_az_deg: f64,
    /// Right wall peak elevation (degrees) and azimuth
    pub right_wall_deg: f64,
    pub right_wall_az_deg: f64,
    /// Vertical prominence of the notch in degrees: min(left_wall, right_wall) - floor
    pub prominence_deg: f64,
    /// Angular Full-Width at Half-Maximum (FWHM) in degrees
    pub angular_width_fwhm_deg: f64,
    /// Distance in meters to the terrain forming the notch walls
    pub notch_distance_m: f64,
    /// Physical width of the notch aperture in meters at `notch_distance_m`
    pub physical_aperture_width_m: f64,
    /// Composite score measuring how dramatic/collimated the light beam is
    pub notch_quality_score: f64,
    /// Event type ("Sunrise Notch" or "Sunset Notch")
    pub event_type: String,
    /// Predicted calendar dates and times when the sun shines directly through this notch
    pub alignments: Vec<SolarAlignmentWindow>,
}

/// Casts a single horizon ray from `(origin_lat, origin_lon, z0)` along `azimuth_deg`
/// using the Hierarchical Max-Mipmap in the 1m near-field and SRTM in the far-field.
pub fn cast_horizon_ray(
    terrain: &HybridTerrain,
    origin_lat: f64,
    origin_lon: f64,
    z0: f64,
    azimuth_deg: f64,
    max_distance_m: f64,
) -> HorizonSample {
    let az_rad = azimuth_deg.to_radians();
    let dir_east = az_rad.sin();
    let dir_north = az_rad.cos();

    let mut max_slope = f64::NEG_INFINITY;
    let mut best_dist_m = 10.0;
    let mut best_height_m = z0;

    let mut d = 4.0_f64; // Start 4 meters out from observer/surface patch

    // 1. Near-field traversal using Hierarchical Max-Mipmap 1m grid (if present)
    if let Some(ref grid) = terrain.near_1m {
        let (ox, oy) = grid.latlon_to_local_m(origin_lat, origin_lon);
        let global_max_z = grid.global_max_z as f64;

        while d <= max_distance_m {
            let rx = ox + d * dir_east;
            let ry = oy + d * dir_north;
            if !grid.contains_local_m(rx, ry) {
                break;
            }

            // Fast global ceiling check: if even the highest peak in the 1m grid cannot beat `max_slope`,
            // we can exit the 1m near-field immediately!
            let drop = earth_curvature_refraction_drop_m(d);
            if max_slope > 0.0 && (global_max_z - drop - z0) <= max_slope * d {
                // Jump straight to the edge of the 1m grid along the ray
                d += 32.0;
                continue;
            }

            // Check Level 4 (16m block) in the Max-Mipmap pyramid
            let l4 = &grid.levels[4]; // 16m blocks
            let cx16 = ((rx as usize) >> 4).min(l4.width - 1);
            let cy16 = ((ry as usize) >> 4).min(l4.height - 1);
            let block16_max_z = l4.max_z[cy16 * l4.width + cx16] as f64;

            if (block16_max_z - drop - z0) <= max_slope * d {
                // Skip across the 16m block
                d += 14.0;
                continue;
            }

            // Check Level 2 (4m block)
            let l2 = &grid.levels[2]; // 4m blocks
            let cx4 = ((rx as usize) >> 2).min(l2.width - 1);
            let cy4 = ((ry as usize) >> 2).min(l2.height - 1);
            let block4_max_z = l2.max_z[cy4 * l2.width + cx4] as f64;

            if (block4_max_z - drop - z0) <= max_slope * d {
                d += 3.5;
                continue;
            }

            // Fine 1-meter evaluation!
            let z = grid.sample_1m_bilinear(rx, ry);
            let eff_dz = z - drop - z0;
            let slope = eff_dz / d;
            if slope > max_slope {
                max_slope = slope;
                best_dist_m = d;
                best_height_m = z;
            }

            d += 1.0;
        }
    }

    // 2. Far-Field SRTM traversal out to `max_distance_m`
    let m_per_lat = METERS_PER_DEG_LAT;
    let m_per_lon = meters_per_deg_lon(origin_lat);

    while d <= max_distance_m {
        let lat = origin_lat + (d * dir_north) / m_per_lat;
        let lon = origin_lon + (d * dir_east) / m_per_lon;

        let z = terrain.far_srtm.elevation_m(lat, lon);
        let drop = earth_curvature_refraction_drop_m(d);
        let slope = (z - drop - z0) / d;

        if slope > max_slope {
            max_slope = slope;
            best_dist_m = d;
            best_height_m = z;
        }

        // Adaptive far-field stepping (15m up to 8km, 30m up to 25km, 60m beyond)
        if d < 8_000.0 {
            d += 15.0;
        } else if d < 25_000.0 {
            d += 30.0;
        } else {
            d += 60.0;
        }
    }

    let horizon_elevation_deg = max_slope.atan().to_degrees();

    HorizonSample {
        azimuth_deg: (azimuth_deg * 1000.0).round() / 1000.0,
        horizon_elevation_deg: (horizon_elevation_deg * 1000.0).round() / 1000.0,
        occluder_distance_m: best_dist_m.round(),
        occluder_height_m: (best_height_m * 10.0).round() / 10.0,
    }
}

/// Computes a full or windowed Horizon Profile `H(φ)` in parallel across CPU cores using `rayon`.
pub fn compute_horizon_profile(
    terrain: &HybridTerrain,
    origin_lat: f64,
    origin_lon: f64,
    eye_offset_m: f64,
    az_start_deg: f64,
    az_end_deg: f64,
    az_step_deg: f64,
    max_distance_m: f64,
) -> HorizonProfile {
    let ground_z = terrain.elevation_m(origin_lat, origin_lon);
    let z0 = ground_z + eye_offset_m;

    let num_steps = (((az_end_deg - az_start_deg) / az_step_deg).round() as usize) + 1;
    let azimuths: Vec<f64> = (0..num_steps)
        .map(|i| az_start_deg + (i as f64) * az_step_deg)
        .filter(|&az| az <= az_end_deg + 1e-6)
        .collect();

    let samples: Vec<HorizonSample> = azimuths
        .into_par_iter()
        .map(|az| cast_horizon_ray(terrain, origin_lat, origin_lon, z0, az, max_distance_m))
        .collect();

    HorizonProfile {
        origin_lat,
        origin_lon,
        origin_elevation_m: ground_z,
        eye_offset_m,
        azimuth_step_deg: az_step_deg,
        samples,
    }
}

/// Detects topographic Light Notches (narrow V-gaps, couloirs, saddles, and rock windows)
/// along the horizon profile and solves for exact solar alignment dates and times.
pub fn detect_light_notches(
    profile: &HorizonProfile,
    year: i32,
    min_prominence_deg: f64,
    max_fwhm_deg: f64,
) -> Vec<LightNotch> {
    let n = profile.samples.len();
    if n < 7 {
        return Vec::new();
    }

    let ((sr_min, sr_max), (ss_min, ss_max)) = annual_solar_azimuth_windows(profile.origin_lat);
    let step = profile.azimuth_step_deg;
    // Search window for flanking walls: ±5.5 degrees around the candidate notch floor
    let wall_window_steps = ((5.5 / step).round() as usize).max(3);
    // Minimum separation between distinct notches: 1.0 degree
    let dedup_steps = ((1.0 / step).round() as usize).max(2);

    let mut candidates: Vec<(usize, LightNotch)> = Vec::new();

    for i in 2..(n - 2) {
        let s_curr = profile.samples[i];
        let az = s_curr.azimuth_deg;

        // Only evaluate azimuths where the rising or setting sun can actually reach
        let in_sunrise = az >= sr_min && az <= sr_max;
        let in_sunset = az >= ss_min && az <= ss_max;
        if !in_sunrise && !in_sunset {
            continue;
        }

        let h_floor = s_curr.horizon_elevation_deg;
        // Must be a local minimum compared to immediate neighbors
        if h_floor > profile.samples[i - 1].horizon_elevation_deg
            || h_floor > profile.samples[i + 1].horizon_elevation_deg
        {
            continue;
        }

        // Find highest flanking wall to the left and right within `wall_window_steps`
        let left_start = i.saturating_sub(wall_window_steps);
        let right_end = (i + wall_window_steps).min(n - 1);

        let mut left_max = h_floor;
        let mut left_az = az;
        let mut left_dist = s_curr.occluder_distance_m;
        for j in left_start..i {
            if profile.samples[j].horizon_elevation_deg > left_max {
                left_max = profile.samples[j].horizon_elevation_deg;
                left_az = profile.samples[j].azimuth_deg;
                left_dist = profile.samples[j].occluder_distance_m;
            }
        }

        let mut right_max = h_floor;
        let mut right_az = az;
        let mut right_dist = s_curr.occluder_distance_m;
        for j in (i + 1)..=right_end {
            if profile.samples[j].horizon_elevation_deg > right_max {
                right_max = profile.samples[j].horizon_elevation_deg;
                right_az = profile.samples[j].azimuth_deg;
                right_dist = profile.samples[j].occluder_distance_m;
            }
        }

        let wall_elev = left_max.min(right_max);
        let prominence = wall_elev - h_floor;
        if prominence < min_prominence_deg {
            continue;
        }

        // Compute angular Full-Width at Half-Maximum (FWHM)
        let half_max_elev = h_floor + 0.5 * prominence;
        let mut left_half_az = az - step;
        for j in (left_start..i).rev() {
            if profile.samples[j].horizon_elevation_deg >= half_max_elev {
                left_half_az = profile.samples[j].azimuth_deg;
                break;
            }
        }
        let mut right_half_az = az + step;
        for j in (i + 1)..=right_end {
            if profile.samples[j].horizon_elevation_deg >= half_max_elev {
                right_half_az = profile.samples[j].azimuth_deg;
                break;
            }
        }

        let fwhm_deg = (right_half_az - left_half_az).max(step);
        if fwhm_deg > max_fwhm_deg {
            continue;
        }

        let notch_dist_m = 0.5 * (left_dist + right_dist);
        let physical_width_m = notch_dist_m * fwhm_deg.to_radians().tan();

        // Quality score rewards deep prominence, steep flanking walls (prominence / width),
        // and golden-hour elevation angles (1° to 18°)
        let steepness_ratio = prominence / fwhm_deg.max(0.2);
        let elevation_bonus = if (1.0..=16.0).contains(&h_floor) {
            1.3
        } else {
            0.85
        };
        let quality_score = (prominence * 12.0 + steepness_ratio * 18.0) * elevation_bonus;

        candidates.push((
            i,
            LightNotch {
                azimuth_deg: (az * 100.0).round() / 100.0,
                floor_elevation_deg: (h_floor * 100.0).round() / 100.0,
                wall_elevation_deg: (wall_elev * 100.0).round() / 100.0,
                left_wall_deg: (left_max * 100.0).round() / 100.0,
                left_wall_az_deg: (left_az * 100.0).round() / 100.0,
                right_wall_deg: (right_max * 100.0).round() / 100.0,
                right_wall_az_deg: (right_az * 100.0).round() / 100.0,
                prominence_deg: (prominence * 100.0).round() / 100.0,
                angular_width_fwhm_deg: (fwhm_deg * 100.0).round() / 100.0,
                notch_distance_m: notch_dist_m.round(),
                physical_aperture_width_m: (physical_width_m * 10.0).round() / 10.0,
                notch_quality_score: (quality_score * 10.0).round() / 10.0,
                event_type: if in_sunrise {
                    "Sunrise Notch".to_string()
                } else {
                    "Sunset Notch".to_string()
                },
                alignments: Vec::new(),
            },
        ));
    }

    // Non-maximum suppression: keep highest-quality notch within `dedup_steps`
    candidates.sort_by(|a, b| {
        b.1.notch_quality_score
            .partial_cmp(&a.1.notch_quality_score)
            .unwrap()
    });
    let mut kept_indices: Vec<usize> = Vec::new();
    let mut final_notches: Vec<LightNotch> = Vec::new();

    for (idx, mut notch) in candidates {
        if kept_indices
            .iter()
            .any(|&k| k.abs_diff(idx) <= dedup_steps)
        {
            continue;
        }
        // Solve for exact solar alignments during the year
        let alignments = find_notch_solar_alignments(
            profile.origin_lat,
            profile.origin_lon,
            year,
            notch.azimuth_deg,
            notch.angular_width_fwhm_deg,
            notch.floor_elevation_deg,
            notch.wall_elevation_deg,
            6,
        );
        if !alignments.is_empty() {
            notch.alignments = alignments;
            kept_indices.push(idx);
            final_notches.push(notch);
        }
    }

    final_notches
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowLightMapStats {
    pub width_px: usize,
    pub height_px: usize,
    pub step_m: usize,
    pub illuminated_fraction: f64,
    pub spotlight_shaft_pixels: usize,
    pub max_contrast_hotspot: Option<(f64, f64, f64)>, // (lat, lon, contrast_score)
}

/// Renders a 2D directional ray-cast shadow & light-shaft map across the 1m grid for a specific `SolarPosition`.
/// Returns statistics and optionally writes a visual PNG map highlighting terrain shadows and golden-hour light shafts.
pub fn compute_shadow_and_light_map(
    terrain: &HybridTerrain,
    grid: &MaxMipmapGrid1m,
    sun: SolarPosition,
    step_m: usize,
    output_png: Option<&Path>,
) -> Result<ShadowLightMapStats> {
    let step_m = step_m.max(1);
    let nx = grid.width_m / step_m;
    let ny = grid.height_m / step_m;

    let az_rad = sun.azimuth_deg.to_radians();
    let el_rad = sun.apparent_elevation_deg.to_radians();
    let sun_vec = [
        az_rad.sin() * el_rad.cos(),
        az_rad.cos() * el_rad.cos(),
        el_rad.sin(),
    ];
    let sun_slope = el_rad.tan();

    // Compute per-pixel illumination and direct irradiance in parallel across rows
    let pixels: Vec<(bool, f32, f32)> = (0..ny)
        .into_par_iter()
        .flat_map_iter(|iy| {
            let y_m = (iy * step_m) as f64;
            (0..nx).map(move |ix| {
                let x_m = (ix * step_m) as f64;
                let (lat, lon) = grid.local_m_to_latlon(x_m, y_m);
                let z0 = grid.sample_1m_bilinear(x_m, y_m);

                // Cast shadow ray toward the sun azimuth using hierarchical max-mipmap
                let horizon =
                    cast_horizon_ray(terrain, lat, lon, z0 + 0.5, sun.azimuth_deg, 30_000.0);
                let is_lit = sun.apparent_elevation_deg >= horizon.horizon_elevation_deg;

                let normal = terrain.surface_normal(lat, lon, step_m as f64);
                let cos_incidence = (normal[0] * sun_vec[0]
                    + normal[1] * sun_vec[1]
                    + normal[2] * sun_vec[2])
                    .max(0.0) as f32;

                let margin_deg =
                    (sun.apparent_elevation_deg - horizon.horizon_elevation_deg) as f32;
                (is_lit && sun_slope > -0.02, cos_incidence, margin_deg)
            })
        })
        .collect();

    let total_px = pixels.len().max(1);
    let lit_count = pixels.iter().filter(|p| p.0).count();

    // Detect "spotlight / light-shaft" pixels: illuminated pixels with low solar clearance (< 2.0° above
    // the upstream ridge/notch) and strong direct grazing illumination surrounded by broader valley shadow
    let mut spotlight_count = 0;
    let mut best_hotspot: Option<(f64, f64, f64)> = None;
    let mut spotlight_scores = vec![0.0_f32; pixels.len()];
    let r_check = 8_isize;

    for iy in (r_check as usize)..(ny.saturating_sub(r_check as usize)) {
        for ix in (r_check as usize)..(nx.saturating_sub(r_check as usize)) {
            let idx = iy * nx + ix;
            let (is_lit, cos_inc, margin_deg) = pixels[idx];
            if !is_lit || margin_deg > 1.8 || cos_inc < 0.25 {
                continue;
            }
            let mut shadowed_neighbors = 0;
            let mut total_neighbors = 0;
            for dy in (-r_check..=r_check).step_by(4) {
                for dx in (-r_check..=r_check).step_by(4) {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nidx = ((iy as isize + dy) as usize) * nx + ((ix as isize + dx) as usize);
                    total_neighbors += 1;
                    if !pixels[nidx].0 {
                        shadowed_neighbors += 1;
                    }
                }
            }
            let shadow_surround_ratio = shadowed_neighbors as f32 / total_neighbors.max(1) as f32;
            // High contrast spotlight when illuminated through a low-clearance horizon gap (< 1.8°)
            // while surrounding terrain is partially shadowed and surface faces the sun
            if (0.20..=0.80).contains(&shadow_surround_ratio) {
                spotlight_count += 1;
                let clearance_factor = (1.8 - margin_deg.max(0.0)) / 1.8;
                let score = shadow_surround_ratio * cos_inc * clearance_factor * 100.0;
                spotlight_scores[idx] = score;

                if best_hotspot.map_or(true, |b| (score as f64) > b.2) {
                    let (lat, lon) =
                        grid.local_m_to_latlon((ix * step_m) as f64, (iy * step_m) as f64);
                    best_hotspot = Some((lat, lon, score as f64));
                }
            }
        }
    }

    if let Some(path) = output_png {
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(nx as u32, ny as u32);
        for iy in 0..ny {
            let img_y = (ny - 1 - iy) as u32; // North at top of image
            for ix in 0..nx {
                let idx = iy * nx + ix;
                let (is_lit, cos_inc, _) = pixels[idx];
                let spot = spotlight_scores[idx];
                let z = grid.sample_1m_bilinear((ix * step_m) as f64, (iy * step_m) as f64) as f32;
                let rel_z = ((z - grid.global_min_z)
                    / (grid.global_max_z - grid.global_min_z).max(1.0))
                .clamp(0.0, 1.0);

                let rgb = if is_lit {
                    // Warm golden-hour direct sunlight modulated by grazing incidence + spotlight boost
                    let spot_boost = (spot / 35.0).clamp(0.0, 1.0);
                    let intensity = (0.30 + 0.70 * cos_inc).clamp(0.0, 1.0);
                    let r = (225.0 * intensity + 20.0 * rel_z + 30.0 * spot_boost).min(255.0) as u8;
                    let g = (160.0 * intensity + 22.0 * rel_z + 45.0 * spot_boost).min(255.0) as u8;
                    let b = (92.0 * intensity + 18.0 * rel_z - 25.0 * spot_boost).clamp(0.0, 255.0) as u8;
                    [r, g, b]
                } else {
                    // Deep terrain shadow (ambient twilight blue-slate modulated by elevation & slope)
                    let amb = 0.22 + 0.52 * rel_z;
                    [
                        (42.0 * amb) as u8,
                        (64.0 * amb) as u8,
                        (105.0 * amb) as u8,
                    ]
                };
                img.put_pixel(ix as u32, img_y, Rgb(rgb));
            }
        }
        img.save(path)?;
    }

    Ok(ShadowLightMapStats {
        width_px: nx,
        height_px: ny,
        step_m,
        illuminated_fraction: (lit_count as f64 / total_px as f64 * 1000.0).round() / 1000.0,
        spotlight_shaft_pixels: spotlight_count,
        max_contrast_hotspot: best_hotspot,
    })
}

/// 3D geometric model of a target photographic landmark (summit crown, sheer cliff face normal, and valley relief).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkTarget3D {
    pub name: String,
    pub summit_lat: f64,
    pub summit_lon: f64,
    pub summit_elevation_m: f64,
    pub summit_elevation_ft: f64,
    pub base_valley_elevation_m: f64,
    pub vertical_relief_m: f64,
    pub vertical_relief_ft: f64,
    /// Unit normal `[east, north, up]` of the primary steep cliff face near the summit
    pub primary_face_normal: [f64; 3],
    /// Compass azimuth (0..360°) that the primary cliff face looks out toward
    pub primary_face_aspect_deg: f64,
    /// Representative point `(lat, lon, z_m)` on the upper sheer cliff face for illumination checks
    pub upper_face_lat: f64,
    pub upper_face_lon: f64,
    pub upper_face_z_m: f64,
    /// Multi-aspect crown rim points `(lat, lon, z_m)` around the monolith perimeter so viewpoints from
    /// any compass quadrant (N, NW, W, SW, S, E) aim at the visible cliff rim facing them
    pub crown_rim_points: Vec<(f64, f64, f64)>,
}

/// Pinpoints the true topographic summit and dominant cliff face geometry within 450m of a landmark seed coordinate.
pub fn extract_landmark_3d_geometry(
    terrain: &HybridTerrain,
    name: &str,
    seed_lat: f64,
    seed_lon: f64,
) -> LandmarkTarget3D {
    let m_per_lat = METERS_PER_DEG_LAT;
    let m_per_lon = meters_per_deg_lon(seed_lat);

    let mut summit_lat = seed_lat;
    let mut summit_lon = seed_lon;
    let mut summit_z = terrain.elevation_m(seed_lat, seed_lon);
    let mut base_valley_z = summit_z;

    // 1. Scan 450m radius at 15m step to locate local topographic summit peak
    for dy_m in (-450..=450).step_by(15) {
        let lat = seed_lat + (dy_m as f64) / m_per_lat;
        for dx_m in (-450..=450).step_by(15) {
            if dx_m * dx_m + dy_m * dy_m > 450 * 450 {
                continue;
            }
            let lon = seed_lon + (dx_m as f64) / m_per_lon;
            let z = terrain.elevation_m(lat, lon);
            if z > summit_z {
                summit_z = z;
                summit_lat = lat;
                summit_lon = lon;
            }
        }
    }

    // 2. Build 8 directional crown rim crests around the summit (at 180m..360m out in each octant)
    let mut crown_rim_points = vec![(summit_lat, summit_lon, summit_z)];
    for oct in 0..8 {
        let ang = (oct as f64) * std::f64::consts::PI / 4.0;
        let mut best_pt = (summit_lat, summit_lon, summit_z);
        let mut best_prom = f64::NEG_INFINITY;
        for r_m in [160.0, 260.0, 360.0] {
            let rlat = summit_lat + (r_m * ang.cos()) / m_per_lat;
            let rlon = summit_lon + (r_m * ang.sin()) / m_per_lon;
            let rz = terrain.elevation_m(rlat, rlon);
            // Measure local cliff drop 200m further out in the same direction
            let out_lat = summit_lat + ((r_m + 220.0) * ang.cos()) / m_per_lat;
            let out_lon = summit_lon + ((r_m + 220.0) * ang.sin()) / m_per_lon;
            let out_z = terrain.elevation_m(out_lat, out_lon);
            let prom = rz + 0.6 * (rz - out_z);
            if prom > best_prom {
                best_prom = prom;
                best_pt = (rlat, rlon, rz);
            }
        }
        crown_rim_points.push(best_pt);
    }

    // 2. Scan 1800m radius around summit to find the surrounding valley floor elevation
    for dy_m in (-1800..=1800).step_by(60) {
        let lat = summit_lat + (dy_m as f64) / m_per_lat;
        for dx_m in (-1800..=1800).step_by(60) {
            let lon = summit_lon + (dx_m as f64) / m_per_lon;
            let z = terrain.elevation_m(lat, lon);
            if z < base_valley_z {
                base_valley_z = z;
            }
        }
    }

    // 3. Find the steepest upper cliff face within 320m of the summit to extract its 3D normal & aspect
    let mut best_slope_score = -1.0_f64;
    let mut face_normal = [0.0, 0.0, 1.0];
    let mut face_lat = summit_lat;
    let mut face_lon = summit_lon;
    let mut face_z = summit_z;

    for dy_m in (-320..=320).step_by(12) {
        let lat = summit_lat + (dy_m as f64) / m_per_lat;
        for dx_m in (-320..=320).step_by(12) {
            let r2 = dx_m * dx_m + dy_m * dy_m;
            if r2 < 25 * 25 || r2 > 320 * 320 {
                continue;
            }
            let lon = summit_lon + (dx_m as f64) / m_per_lon;
            let z = terrain.elevation_m(lat, lon);
            // Must be on the upper half of the landmark relief
            if z < base_valley_z + 0.55 * (summit_z - base_valley_z) {
                continue;
            }
            let n = terrain.surface_normal(lat, lon, 12.0);
            let horiz_mag = (n[0] * n[0] + n[1] * n[1]).sqrt();
            let elev_weight = (z - base_valley_z) / (summit_z - base_valley_z).max(1.0);
            let score = horiz_mag * (0.5 + 0.5 * elev_weight);
            if score > best_slope_score {
                best_slope_score = score;
                face_normal = n;
                face_lat = lat;
                face_lon = lon;
                face_z = z;
            }
        }
    }

    let aspect_deg = face_normal[0]
        .atan2(face_normal[1])
        .to_degrees()
        .rem_euclid(360.0);
    let relief_m = (summit_z - base_valley_z).max(10.0);

    LandmarkTarget3D {
        name: name.to_string(),
        summit_lat: (summit_lat * 100_000.0).round() / 100_000.0,
        summit_lon: (summit_lon * 100_000.0).round() / 100_000.0,
        summit_elevation_m: (summit_z * 10.0).round() / 10.0,
        summit_elevation_ft: (summit_z * 3.28084).round(),
        base_valley_elevation_m: (base_valley_z * 10.0).round() / 10.0,
        vertical_relief_m: (relief_m * 10.0).round() / 10.0,
        vertical_relief_ft: (relief_m * 3.28084).round(),
        primary_face_normal: face_normal,
        primary_face_aspect_deg: (aspect_deg * 10.0).round() / 10.0,
        upper_face_lat: face_lat,
        upper_face_lon: face_lon,
        upper_face_z_m: face_z,
        crown_rim_points,
    }
}

/// Represents a predicted golden-hour window when solar rays illuminate Landmark X with high topographic contrast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectGoldenHourWindow {
    pub phase: String, // "Sunset Golden Hour / Alpenglow" or "Sunrise Golden Hour"
    pub date: String,
    pub window_start_local: String,
    pub peak_time_local: String,
    pub window_end_local: String,
    pub peak_utc: String,
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    /// Fraction of surrounding valley/canyon already in deep shadow while the landmark crown is sunlit
    pub valley_shadow_contrast_pct: f64,
    /// Cosine of incidence angle between sun ray and the landmark's primary cliff face
    pub cliff_face_incidence: f64,
    /// Clearance of the sun ray above the upstream horizon ridge (smaller = more dramatic spotlight beam)
    pub horizon_clearance_deg: f64,
}

/// Traces solar rays onto Landmark X and its surrounding valley across Sunrise and Sunset on `date`
/// to identify the exact golden-hour windows where Landmark X is illuminated against shadowed terrain.
pub fn find_subject_golden_hour_windows(
    terrain: &HybridTerrain,
    target: &LandmarkTarget3D,
    date: chrono::NaiveDate,
) -> Vec<SubjectGoldenHourWindow> {
    use chrono::{TimeZone, Utc};

    let approx_tz_hours = (target.summit_lon / 15.0).round() as i64;
    let base_utc = Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap());

    // Sample 8 valley floor points at 1100m radius around the landmark to measure differential shadow contrast
    let m_per_lat = METERS_PER_DEG_LAT;
    let m_per_lon = meters_per_deg_lon(target.summit_lat);
    let mut valley_points = Vec::with_capacity(8);
    for k in 0..8 {
        let ang = (k as f64) * std::f64::consts::PI / 4.0;
        let vlat = target.summit_lat + (1100.0 * ang.cos()) / m_per_lat;
        let vlon = target.summit_lon + (1100.0 * ang.sin()) / m_per_lon;
        let vz = terrain.elevation_m(vlat, vlon);
        valley_points.push((vlat, vlon, vz));
    }

    let mut windows = Vec::new();

    for &(phase_name, start_local_min, end_local_min) in &[
        ("Sunset Golden Hour / Alpenglow", 15 * 60 + 30, 20 * 60 + 15),
        ("Sunrise Golden Hour", 5 * 60, 9 * 60 + 30),
    ] {
        let mut lit_samples = Vec::new();

        for local_min in (start_local_min..=end_local_min).step_by(3) {
            let dt_utc =
                base_utc + chrono::Duration::minutes(local_min - approx_tz_hours * 60);
            let sun = pp_solar::solar_position(target.summit_lat, target.summit_lon, dt_utc);

            // Golden hour / low-angle light filter: apparent sun elevation between -0.3° and +14.0°
            if !(-0.3..=14.0).contains(&sun.apparent_elevation_deg) {
                continue;
            }

            // Cast ray from Landmark X's upper cliff face/summit toward the sun
            let summit_horizon = cast_horizon_ray(
                terrain,
                target.summit_lat,
                target.summit_lon,
                target.summit_elevation_m + 1.0,
                sun.azimuth_deg,
                40_000.0,
            );
            let clearance_deg =
                sun.apparent_elevation_deg - summit_horizon.horizon_elevation_deg;
            if clearance_deg < -0.15 {
                // Landmark summit itself is in terrain shadow
                continue;
            }

            // Compute how many valley floor points around the landmark are in shadow at this exact minute
            let mut shadowed_valley = 0;
            for &(vlat, vlon, vz) in &valley_points {
                let vh = cast_horizon_ray(terrain, vlat, vlon, vz + 1.0, sun.azimuth_deg, 25_000.0);
                if sun.apparent_elevation_deg < vh.horizon_elevation_deg {
                    shadowed_valley += 1;
                }
            }
            let valley_shadow_ratio = shadowed_valley as f64 / valley_points.len() as f64;

            // Sun vector `[east, north, up]`
            let az_rad = sun.azimuth_deg.to_radians();
            let el_rad = sun.apparent_elevation_deg.to_radians();
            let sun_vec = [
                az_rad.sin() * el_rad.cos(),
                az_rad.cos() * el_rad.cos(),
                el_rad.sin(),
            ];
            let face_inc = (target.primary_face_normal[0] * sun_vec[0]
                + target.primary_face_normal[1] * sun_vec[1]
                + target.primary_face_normal[2] * sun_vec[2])
                .max(0.05);

            // Score peaks when sun is low (clearance < 6°), valley has strong shadow contrast, and face is grazing/lit
            let low_sun_bonus = ((14.0 - sun.apparent_elevation_deg) / 14.0).clamp(0.1, 1.0);
            let quality =
                (0.45 * valley_shadow_ratio + 0.35 * low_sun_bonus + 0.20 * face_inc) * 100.0;

            lit_samples.push((
                local_min,
                dt_utc,
                sun,
                clearance_deg,
                valley_shadow_ratio,
                face_inc,
                quality,
            ));
        }

        if lit_samples.is_empty() {
            continue;
        }

        // Find the peak quality timestamp and restrict the window to ±20 minutes around the best golden-hour light
        let best = lit_samples
            .iter()
            .max_by(|a, b| a.6.partial_cmp(&b.6).unwrap())
            .unwrap()
            .clone();

        let win_samples: Vec<_> = lit_samples
            .iter()
            .filter(|s| (s.0 - best.0).abs() <= 24)
            .collect();
        let start_min = win_samples.first().map(|s| s.0).unwrap_or(best.0);
        let end_min = win_samples.last().map(|s| s.0).unwrap_or(best.0);

        let fmt_time = |m: i64| -> String {
            let mm = m.rem_euclid(1440);
            format!("{:02}:{:02}", mm / 60, mm % 60)
        };

        windows.push(SubjectGoldenHourWindow {
            phase: phase_name.to_string(),
            date: date.format("%Y-%m-%d").to_string(),
            window_start_local: fmt_time(start_min),
            peak_time_local: fmt_time(best.0),
            window_end_local: fmt_time(end_min),
            peak_utc: best.1.to_rfc3339(),
            sun_azimuth_deg: (best.2.azimuth_deg * 100.0).round() / 100.0,
            sun_elevation_deg: (best.2.apparent_elevation_deg * 100.0).round() / 100.0,
            valley_shadow_contrast_pct: (best.4 * 1000.0).round() / 10.0,
            cliff_face_incidence: (best.5 * 100.0).round() / 100.0,
            horizon_clearance_deg: (best.3 * 100.0).round() / 100.0,
        });
    }

    windows
}

/// Result of casting a 3D Camera Line-of-Sight ray from a trail viewpoint `(vp_lat, vp_lon, vp_z)` to `LandmarkTarget3D`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineOfSightResult {
    pub is_visible: bool,
    pub distance_to_landmark_m: f64,
    pub distance_to_landmark_miles: f64,
    pub bearing_to_landmark_deg: f64,
    pub compass_direction: String,
    pub elevation_angle_to_summit_deg: f64,
    pub visible_vertical_relief_m: f64,
    pub visible_vertical_relief_ft: f64,
    pub visible_relief_fraction: f64,
    pub recommended_focal_length_mm: String,
    pub composition_score: f64,
}

fn bearing_to_compass(deg: f64) -> String {
    let dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW",
        "NW", "NNW",
    ];
    let idx = (((deg + 11.25).rem_euclid(360.0)) / 22.5).floor() as usize;
    dirs[idx.min(15)].to_string()
}

/// Casts a 3D Line-of-Sight ray from a candidate trail viewpoint `(vp_lat, vp_lon, vp_z)` to `target`,
/// checking terrain occlusion and computing visible vertical relief, viewing angle onto the cliff face, and focal length.
pub fn cast_camera_line_of_sight(
    terrain: &HybridTerrain,
    vp_lat: f64,
    vp_lon: f64,
    vp_z_m: f64,
    target: &LandmarkTarget3D,
) -> LineOfSightResult {
    let m_per_lat = METERS_PER_DEG_LAT;
    let m_per_lon = meters_per_deg_lon(vp_lat);

    let dy_m = (target.summit_lat - vp_lat) * m_per_lat;
    let dx_m = (target.summit_lon - vp_lon) * m_per_lon;
    let total_dist_m = (dx_m * dx_m + dy_m * dy_m).sqrt();

    if total_dist_m < 450.0 || total_dist_m > 18_000.0 {
        return LineOfSightResult {
            is_visible: false,
            distance_to_landmark_m: total_dist_m,
            distance_to_landmark_miles: total_dist_m / 1609.344,
            bearing_to_landmark_deg: 0.0,
            compass_direction: "N".to_string(),
            elevation_angle_to_summit_deg: 0.0,
            visible_vertical_relief_m: 0.0,
            visible_vertical_relief_ft: 0.0,
            visible_relief_fraction: 0.0,
            recommended_focal_length_mm: "N/A".to_string(),
            composition_score: 0.0,
        };
    }

    let bearing_deg = dx_m.atan2(dy_m).to_degrees().rem_euclid(360.0);
    let dir_east = dx_m / total_dist_m;
    let dir_north = dy_m / total_dist_m;

    let z_eye = vp_z_m + 1.65;

    // Evaluate visibility across the landmark's crown rim points (summit + directional cliff rims)
    // and pick the rim crest with the greatest unobstructed vertical relief from this viewpoint!
    let mut visible_relief_m = 0.0_f64;
    let mut summit_slope = 0.0_f64;

    for &(t_lat, t_lon, t_z) in &target.crown_rim_points {
        let c_dy = (t_lat - vp_lat) * m_per_lat;
        let c_dx = (t_lon - vp_lon) * m_per_lon;
        let c_dist = (c_dx * c_dx + c_dy * c_dy).sqrt();
        if c_dist < 400.0 {
            continue;
        }
        let c_dir_east = c_dx / c_dist;
        let c_dir_north = c_dy / c_dist;

        let drop_target = earth_curvature_refraction_drop_m(c_dist);
        let eff_tz = t_z - drop_target;
        let t_slope = (eff_tz - z_eye) / c_dist;

        // March from 280m out (past the viewpoint's own immediate dome/ledge shoulder)
        // to (c_dist - 400m) to check true intervening ridges across the valley
        let stop_d = (c_dist - 400.0).max(300.0);
        let mut max_int_slope = f64::NEG_INFINITY;
        let mut d = 280.0_f64;
        while d < stop_d {
            let lat = vp_lat + (d * c_dir_north) / m_per_lat;
            let lon = vp_lon + (d * c_dir_east) / m_per_lon;
            let z = terrain.elevation_m(lat, lon);
            let drop = earth_curvature_refraction_drop_m(d);
            let slope = (z - drop - z_eye) / d;
            if slope > max_int_slope {
                max_int_slope = slope;
            }
            d += if d < 2500.0 { 18.0 } else { 35.0 };
        }

        let occluded_z = if max_int_slope == f64::NEG_INFINITY {
            target.base_valley_elevation_m
        } else {
            (z_eye + max_int_slope * c_dist + drop_target).max(target.base_valley_elevation_m)
        };

        let rel_m = (t_z - occluded_z).max(0.0);
        if rel_m > visible_relief_m {
            visible_relief_m = rel_m;
            summit_slope = t_slope;
        }
    }

    let visible_fraction =
        (visible_relief_m / target.vertical_relief_m.max(1.0)).clamp(0.0, 1.0);

    // Require at least 25% of the landmark's vertical relief (and at least 80m) to be unobstructed
    let is_visible = visible_fraction >= 0.25 && visible_relief_m >= 80.0;

    // Evaluate how well this viewpoint looks at the landmark's sheer primary cliff face
    // Vector from landmark to viewpoint: `[-dir_east, -dir_north, ...]`
    let cam_dot_face = (-dir_east * target.primary_face_normal[0]
        + -dir_north * target.primary_face_normal[1])
        .clamp(-1.0, 1.0);
    let face_view_bonus = (0.55 + 0.45 * cam_dot_face).clamp(0.2, 1.0);

    // Distance framing sweet spot: 1.5 km to 7.5 km gives dramatic monumental scale
    let dist_bonus = if (1400.0..=7500.0).contains(&total_dist_m) {
        1.0
    } else if total_dist_m < 1400.0 {
        0.75
    } else {
        (7500.0 / total_dist_m).clamp(0.35, 0.95)
    };

    let composition_score = if is_visible {
        (visible_fraction * 65.0 + face_view_bonus * 25.0 + dist_bonus * 10.0) * dist_bonus
    } else {
        0.0
    };

    // Recommend 35mm full-frame focal length so the landmark fills ~35-50% of a horizontal/vertical frame
    let focal_str = if total_dist_m < 1800.0 {
        "16–24mm (Ultra-Wide)".to_string()
    } else if total_dist_m < 3500.0 {
        "24–50mm (Standard Wide)".to_string()
    } else if total_dist_m < 6500.0 {
        "50–85mm (Short Telephoto)".to_string()
    } else {
        "70–200mm (Telephoto Compression)".to_string()
    };

    LineOfSightResult {
        is_visible,
        distance_to_landmark_m: total_dist_m.round(),
        distance_to_landmark_miles: ((total_dist_m / 1609.344) * 100.0).round() / 100.0,
        bearing_to_landmark_deg: (bearing_deg * 10.0).round() / 10.0,
        compass_direction: bearing_to_compass(bearing_deg),
        elevation_angle_to_summit_deg: (summit_slope.atan().to_degrees() * 100.0).round() / 100.0,
        visible_vertical_relief_m: visible_relief_m.round(),
        visible_vertical_relief_ft: (visible_relief_m * 3.28084).round(),
        visible_relief_fraction: (visible_fraction * 1000.0).round() / 1000.0,
        recommended_focal_length_mm: focal_str,
        composition_score: (composition_score * 10.0).round() / 10.0,
    }
}

/// Represents a predicted night window where low-angle moonlight illuminates Landmark X's cliff face
/// above the upstream terrain horizon while the surrounding valley or sky remains dark enough for stars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LunarAlpenglowWindow {
    pub window_start_local: String,
    pub peak_time_local: String,
    pub window_end_local: String,
    pub moon_azimuth_deg: f64,
    pub moon_elevation_deg: f64,
    pub moon_illumination_pct: f64,
    pub horizon_clearance_deg: f64,
    pub cliff_face_incidence: f64,
    pub description: String,
}

/// Full overnight astrophotography & topographic lunar/Galactic Core analysis (18:00 to 06:00 local).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectAstroNightSummary {
    pub date: String,
    pub moon_phase_name: String,
    pub moon_illumination_pct: f64,
    pub moon_phase_angle_deg: f64,
    pub moon_distance_km: f64,
    pub astronomical_dusk_local: String,
    pub astronomical_dawn_local: String,
    pub flat_moonrise_local: Option<String>,
    pub flat_moonset_local: Option<String>,
    pub terrain_moonrise_local: Option<String>,
    pub terrain_moonset_local: Option<String>,
    pub terrain_moonset_az_deg: Option<f64>,
    pub terrain_moonset_horizon_deg: Option<f64>,
    /// Minutes earlier that the Moon drops behind the 3D terrain ridge compared to a flat 0° horizon
    pub terrain_moonset_lead_minutes: Option<i64>,
    /// Minutes later that the Moon clears the 3D terrain ridge compared to a flat 0° horizon
    pub terrain_moonrise_lag_minutes: Option<i64>,
    /// Window during astro darkness when the Moon is below the local 3D mountain horizon (or < 5% illuminated)
    pub shadowed_dark_sky_window_local: Option<String>,
    pub shadowed_dark_sky_duration_minutes: u32,
    pub galactic_core_visible: bool,
    pub galactic_core_window_local: Option<String>,
    pub galactic_core_peak_local: Option<String>,
    pub galactic_core_peak_az_deg: f64,
    pub galactic_core_peak_el_deg: f64,
    pub galactic_core_terrain_horizon_deg: f64,
    pub galactic_core_clearance_deg: f64,
    pub lunar_alpenglow_window: Option<LunarAlpenglowWindow>,
    pub astro_geometry_verdict: String,
    pub astro_geometry_summary: String,
    /// Nightly celestial track (18:00..30:00 local in 15-min steps):
    /// `[local_min, moon_az_deg, moon_el_deg, gc_az_deg, gc_el_deg, sun_el_deg]`
    pub nightly_celestial_track: Vec<[f64; 6]>,
}

fn interpolate_horizon_deg(profile: &HorizonProfile, az_deg: f64) -> f64 {
    if profile.samples.is_empty() {
        return 0.0;
    }
    let az = az_deg.rem_euclid(360.0);
    let idx = ((az / profile.azimuth_step_deg).round() as usize) % profile.samples.len();
    profile.samples[idx].horizon_elevation_deg
}

/// Computes overnight topographic moonrise/moonset, shadowed dark-sky windows, Galactic Core terrain
/// clearance, and "Lunar Alpenglow" windows from 18:00 local on `date` through 06:00 the next morning.
pub fn compute_subject_astro_night_summary(
    terrain: &HybridTerrain,
    target: &LandmarkTarget3D,
    date: chrono::NaiveDate,
) -> SubjectAstroNightSummary {
    use chrono::{TimeZone, Utc};

    let approx_tz_hours = (target.summit_lon / 15.0).round() as i64;
    let base_utc = Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap());

    // Reference valley observer coordinate ~1.4 km out in the direction of the primary cliff face aspect
    let m_per_lat = METERS_PER_DEG_LAT;
    let m_per_lon = meters_per_deg_lon(target.summit_lat);
    let aspect_rad = target.primary_face_aspect_deg.to_radians();
    let obs_lat = target.summit_lat + (1400.0 * aspect_rad.cos()) / m_per_lat;
    let obs_lon = target.summit_lon + (1400.0 * aspect_rad.sin()) / m_per_lon;

    // Precompute 360° horizon profiles at 1.0° resolution (parallel across cores via rayon)
    let obs_profile =
        compute_horizon_profile(terrain, obs_lat, obs_lon, 1.7, 0.0, 359.0, 1.0, 35_000.0);
    let summit_profile = compute_horizon_profile(
        terrain,
        target.summit_lat,
        target.summit_lon,
        1.0,
        0.0,
        359.0,
        1.0,
        35_000.0,
    );

    let fmt_time = |m: i64| -> String {
        let mm = m.rem_euclid(1440);
        format!("{:02}:{:02}", mm / 60, mm % 60)
    };

    // Sample midnight (24:00 local) for nominal lunar phase metadata
    let midnight_utc = base_utc + chrono::Duration::minutes(24 * 60 - approx_tz_hours * 60);
    let midnight_moon = pp_solar::lunar_position(target.summit_lat, target.summit_lon, midnight_utc);
    let moon_illum_pct = (midnight_moon.illuminated_fraction * 1000.0).round() / 10.0;

    let mut astro_dusk_min: Option<i64> = None;
    let mut astro_dawn_min: Option<i64> = None;

    let mut flat_moonrise_min: Option<i64> = None;
    let mut flat_moonset_min: Option<i64> = None;
    let mut terrain_moonrise_min: Option<i64> = None;
    let mut terrain_moonset_min: Option<i64> = None;
    let mut terrain_moonset_az: Option<f64> = None;
    let mut terrain_moonset_hor: Option<f64> = None;

    let mut prev_sun_el: Option<f64> = None;
    let mut prev_moon_flat_el: Option<f64> = None;
    let mut prev_moon_terrain_margin: Option<f64> = None;

    let mut dark_sky_mins: Vec<i64> = Vec::new();
    let mut gc_visible_samples: Vec<(i64, f64, f64, f64)> = Vec::new(); // (min, az, el, h_ter)
    let mut best_gc_any: (i64, f64, f64, f64) = (22 * 60, 180.0, -25.0, 8.0);
    let mut lunar_alpenglow_samples: Vec<(i64, f64, f64, f64, f64)> = Vec::new(); // (min, az, el, clearance, face_inc)
    let mut nightly_track: Vec<[f64; 6]> = Vec::new();

    // Sweep from 18:00 (1080 min) on `date` to 06:00 next morning (1800 min) in 3-minute increments
    for local_min in (1080..=1800).step_by(3) {
        let dt_utc = base_utc + chrono::Duration::minutes(local_min - approx_tz_hours * 60);
        let sun = pp_solar::solar_position(target.summit_lat, target.summit_lon, dt_utc);
        let moon = pp_solar::lunar_position(target.summit_lat, target.summit_lon, dt_utc);
        let gc = pp_solar::galactic_core_position(target.summit_lat, target.summit_lon, dt_utc);

        let h_obs_moon = interpolate_horizon_deg(&obs_profile, moon.azimuth_deg);
        let h_sum_moon = interpolate_horizon_deg(&summit_profile, moon.azimuth_deg);
        let h_obs_gc = interpolate_horizon_deg(&obs_profile, gc.azimuth_deg);

        if (local_min - 1080) % 15 == 0 {
            nightly_track.push([
                local_min as f64,
                moon.azimuth_deg,
                moon.apparent_elevation_deg,
                gc.azimuth_deg,
                gc.apparent_elevation_deg,
                (sun.apparent_elevation_deg * 100.0).round() / 100.0,
            ]);
        }

        // Detect astronomical dusk (-15° threshold for high mountain sky darkness) & dawn
        if let Some(ps) = prev_sun_el {
            if ps > -15.0 && sun.apparent_elevation_deg <= -15.0 && astro_dusk_min.is_none() {
                astro_dusk_min = Some(local_min);
            }
            if ps <= -15.0 && sun.apparent_elevation_deg > -15.0 && astro_dawn_min.is_none() {
                astro_dawn_min = Some(local_min);
            }
        }
        prev_sun_el = Some(sun.apparent_elevation_deg);

        // Detect flat-horizon (0.0°) moonrise and moonset
        if let Some(pm) = prev_moon_flat_el {
            if pm <= 0.0 && moon.apparent_elevation_deg > 0.0 && flat_moonrise_min.is_none() {
                flat_moonrise_min = Some(local_min);
            }
            if pm > 0.0 && moon.apparent_elevation_deg <= 0.0 && flat_moonset_min.is_none() {
                flat_moonset_min = Some(local_min);
            }
        }
        prev_moon_flat_el = Some(moon.apparent_elevation_deg);

        // Detect 3D terrain-occluded moonrise and moonset against observer horizon H(φ)
        let moon_margin = moon.apparent_elevation_deg - h_obs_moon;
        if let Some(ptm) = prev_moon_terrain_margin {
            if ptm <= 0.0 && moon_margin > 0.0 && terrain_moonrise_min.is_none() {
                terrain_moonrise_min = Some(local_min);
            }
            if ptm > 0.0 && moon_margin <= 0.0 && terrain_moonset_min.is_none() {
                terrain_moonset_min = Some(local_min);
                terrain_moonset_az = Some(moon.azimuth_deg);
                terrain_moonset_hor = Some((h_obs_moon * 10.0).round() / 10.0);
            }
        }
        prev_moon_terrain_margin = Some(moon_margin);

        // Track best Galactic Core elevation during darkness even if seasonal winter core is low
        if sun.apparent_elevation_deg <= -12.0 && gc.apparent_elevation_deg > best_gc_any.2 {
            best_gc_any = (
                local_min,
                gc.azimuth_deg,
                gc.apparent_elevation_deg,
                h_obs_gc,
            );
        }

        // Lunar Alpenglow during deep twilight / night (Sun <= -7.5°): Moon clears the summit's
        // upstream horizon (`h_sum_moon`), is above +0.3° apparent elevation, and lights the cliff face
        if sun.apparent_elevation_deg <= -7.5 {
            let sum_clearance = moon.apparent_elevation_deg - h_sum_moon;
            if moon.illuminated_fraction >= 0.08
                && (0.3..=32.0).contains(&moon.apparent_elevation_deg)
                && (0.2..=28.0).contains(&sum_clearance)
            {
                let az_rad = moon.azimuth_deg.to_radians();
                let el_rad = moon.apparent_elevation_deg.to_radians();
                let moon_vec = [
                    az_rad.sin() * el_rad.cos(),
                    az_rad.cos() * el_rad.cos(),
                    el_rad.sin(),
                ];
                let face_inc = (target.primary_face_normal[0] * moon_vec[0]
                    + target.primary_face_normal[1] * moon_vec[1]
                    + target.primary_face_normal[2] * moon_vec[2])
                    .max(0.0);

                if face_inc >= 0.15 {
                    lunar_alpenglow_samples.push((
                        local_min,
                        moon.azimuth_deg,
                        moon.apparent_elevation_deg,
                        sum_clearance,
                        face_inc,
                    ));
                }
            }
        }

        // Dark-sky & Galactic Core evaluation during nautical/astronomical night (Sun <= -12°)
        if sun.apparent_elevation_deg <= -12.0 {
            let moon_is_dark_or_occluded =
                moon.illuminated_fraction <= 0.08 || moon.apparent_elevation_deg <= h_obs_moon;
            if moon_is_dark_or_occluded {
                dark_sky_mins.push(local_min);
            }

            // Galactic Core visible above the 3D mountain horizon
            if gc.apparent_elevation_deg >= h_obs_gc + 0.5 {
                gc_visible_samples.push((
                    local_min,
                    gc.azimuth_deg,
                    gc.apparent_elevation_deg,
                    h_obs_gc,
                ));
            }
        }
    }

    let moonset_lead_min = match (flat_moonset_min, terrain_moonset_min) {
        (Some(flat_m), Some(terr_m)) if flat_m >= terr_m => Some(flat_m - terr_m),
        _ => None,
    };
    let moonrise_lag_min = match (flat_moonrise_min, terrain_moonrise_min) {
        (Some(flat_m), Some(terr_m)) if terr_m >= flat_m => Some(terr_m - flat_m),
        _ => None,
    };

    let shadowed_dark_sky_duration_minutes = (dark_sky_mins.len() as u32) * 3;
    let shadowed_dark_sky_window_local = match (dark_sky_mins.first(), dark_sky_mins.last()) {
        (Some(&s), Some(&e)) if e > s => Some(format!("{} – {}", fmt_time(s), fmt_time(e))),
        _ => None,
    };

    let (
        galactic_core_visible,
        galactic_core_window_local,
        galactic_core_peak_local,
        gc_peak_az,
        gc_peak_el,
        gc_hor_deg,
        gc_clear_deg,
    ) = if !gc_visible_samples.is_empty() {
        let start_m = gc_visible_samples.first().unwrap().0;
        let end_m = gc_visible_samples.last().unwrap().0;
        let best_gc = gc_visible_samples
            .iter()
            .max_by(|a, b| (a.2 - a.3).partial_cmp(&(b.2 - b.3)).unwrap())
            .unwrap();
        (
            true,
            Some(format!("{} – {}", fmt_time(start_m), fmt_time(end_m))),
            Some(fmt_time(best_gc.0)),
            (best_gc.1 * 10.0).round() / 10.0,
            (best_gc.2 * 10.0).round() / 10.0,
            (best_gc.3 * 10.0).round() / 10.0,
            ((best_gc.2 - best_gc.3) * 10.0).round() / 10.0,
        )
    } else {
        (
            false,
            None,
            Some(fmt_time(best_gc_any.0)),
            (best_gc_any.1 * 10.0).round() / 10.0,
            (best_gc_any.2 * 10.0).round() / 10.0,
            (best_gc_any.3 * 10.0).round() / 10.0,
            ((best_gc_any.2 - best_gc_any.3) * 10.0).round() / 10.0,
        )
    };

    let lunar_alpenglow_window = if !lunar_alpenglow_samples.is_empty() {
        let best_la = lunar_alpenglow_samples
            .iter()
            .max_by(|a, b| {
                let score_a = a.4 * (28.0 - a.3.min(25.0));
                let score_b = b.4 * (28.0 - b.3.min(25.0));
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap();
        let win_la: Vec<_> = lunar_alpenglow_samples
            .iter()
            .filter(|s| (s.0 - best_la.0).abs() <= 45)
            .collect();
        let la_start = win_la.first().map(|s| s.0).unwrap_or(best_la.0);
        let la_end = win_la.last().map(|s| s.0).unwrap_or(best_la.0);

        Some(LunarAlpenglowWindow {
            window_start_local: fmt_time(la_start),
            peak_time_local: fmt_time(best_la.0),
            window_end_local: fmt_time(la_end),
            moon_azimuth_deg: (best_la.1 * 10.0).round() / 10.0,
            moon_elevation_deg: (best_la.2 * 10.0).round() / 10.0,
            moon_illumination_pct: moon_illum_pct,
            horizon_clearance_deg: (best_la.3 * 10.0).round() / 10.0,
            cliff_face_incidence: (best_la.4 * 100.0).round() / 100.0,
            description: format!(
                "{}% {} side-lights {}'s upper cliff face from {} to {} (Az {:.1}°, El +{:.1}°)",
                moon_illum_pct.round(),
                midnight_moon.phase_name,
                target.name,
                fmt_time(la_start),
                fmt_time(la_end),
                best_la.1,
                best_la.2
            ),
        })
    } else {
        None
    };

    let (verdict, summary) = if shadowed_dark_sky_duration_minutes >= 210 && galactic_core_visible {
        (
            "Prime Dark-Sky Galactic Core + Terrain Shielding".to_string(),
            format!(
                "{:.1} hrs of moon-free darkness ({}) with the Galactic Core clearing the southern ridge by +{:.1}° at {}.",
                shadowed_dark_sky_duration_minutes as f64 / 60.0,
                shadowed_dark_sky_window_local
                    .as_deref()
                    .unwrap_or("All Night"),
                gc_clear_deg,
                galactic_core_peak_local.as_deref().unwrap_or("22:30")
            ),
        )
    } else if let Some(ref la) = lunar_alpenglow_window {
        (
            "Moonlit Granite Relief + Nightscape Window".to_string(),
            format!(
                "{}. Dark-sky window: {}.",
                la.description,
                shadowed_dark_sky_window_local
                    .as_deref()
                    .unwrap_or("Moonlit all night")
            ),
        )
    } else if shadowed_dark_sky_duration_minutes >= 180 {
        (
            "Deep Alpine Dark-Sky Starfield Window".to_string(),
            format!(
                "{:.1} hrs of unilluminated dark sky ({}) with {} ({:.0}% illuminated).",
                shadowed_dark_sky_duration_minutes as f64 / 60.0,
                shadowed_dark_sky_window_local
                    .as_deref()
                    .unwrap_or("All Night"),
                midnight_moon.phase_name,
                moon_illum_pct
            ),
        )
    } else {
        (
            "High-Illumination Moonlit Alpine Night".to_string(),
            format!(
                "{} ({:.0}% illumination) dominates the night sky; ideal for long-exposure moonlit granite landscapes and star trails.",
                midnight_moon.phase_name, moon_illum_pct
            ),
        )
    };

    SubjectAstroNightSummary {
        date: date.format("%Y-%m-%d").to_string(),
        moon_phase_name: midnight_moon.phase_name,
        moon_illumination_pct: moon_illum_pct,
        moon_phase_angle_deg: midnight_moon.phase_angle_deg,
        moon_distance_km: midnight_moon.distance_km,
        astronomical_dusk_local: fmt_time(astro_dusk_min.unwrap_or(19 * 60 + 50)),
        astronomical_dawn_local: fmt_time(astro_dawn_min.unwrap_or(29 * 60 + 15)),
        flat_moonrise_local: flat_moonrise_min.map(fmt_time),
        flat_moonset_local: flat_moonset_min.map(fmt_time),
        terrain_moonrise_local: terrain_moonrise_min.map(fmt_time),
        terrain_moonset_local: terrain_moonset_min.map(fmt_time),
        terrain_moonset_az_deg: terrain_moonset_az,
        terrain_moonset_horizon_deg: terrain_moonset_hor,
        terrain_moonset_lead_minutes: moonset_lead_min,
        terrain_moonrise_lag_minutes: moonrise_lag_min,
        shadowed_dark_sky_window_local,
        shadowed_dark_sky_duration_minutes,
        galactic_core_visible,
        galactic_core_window_local,
        galactic_core_peak_local,
        galactic_core_peak_az_deg: gc_peak_az,
        galactic_core_peak_el_deg: gc_peak_el,
        galactic_core_terrain_horizon_deg: gc_hor_deg,
        galactic_core_clearance_deg: gc_clear_deg,
        lunar_alpenglow_window,
        astro_geometry_verdict: verdict,
        astro_geometry_summary: summary,
        nightly_celestial_track: nightly_track,
    }
}
