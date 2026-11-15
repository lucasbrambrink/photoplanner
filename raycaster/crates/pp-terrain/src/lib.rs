use anyhow::{Context, Result};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

/// Combined Earth curvature and standard atmospheric refraction coefficient (k ≈ 0.13):
/// Δz(d) = -d² / (2 * R_earth) * (1 - k) ≈ -6.75e-8 * d² (in meters)
pub const CURVATURE_REFRACTION_FACTOR: f64 = 6.75e-8;

/// Metres per degree of latitude (approximate WGS84 meridional scale)
pub const METERS_PER_DEG_LAT: f64 = 111_132.92;

#[inline]
pub fn meters_per_deg_lon(lat_deg: f64) -> f64 {
    111_412.84 * lat_deg.to_radians().cos()
}

/// Effective vertical drop (in meters) of distant terrain due to Earth curvature minus atmospheric refraction.
#[inline]
pub fn earth_curvature_refraction_drop_m(distance_m: f64) -> f64 {
    CURVATURE_REFRACTION_FACTOR * distance_m * distance_m
}

/// Core trait abstracting bare-earth or surface elevation models.
pub trait ElevationSurface: Send + Sync {
    /// Returns elevation in meters at `(lat_deg, lon_deg)`.
    fn elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64;

    /// Computes local surface unit normal `[nx_east, ny_north, nz_up]` using finite differences at `step_m`.
    fn surface_normal(&self, lat_deg: f64, lon_deg: f64, step_m: f64) -> [f64; 3] {
        let dlat = step_m / METERS_PER_DEG_LAT;
        let dlon = step_m / meters_per_deg_lon(lat_deg).max(1.0);

        let z_e = self.elevation_m(lat_deg, lon_deg + dlon);
        let z_w = self.elevation_m(lat_deg, lon_deg - dlon);
        let z_n = self.elevation_m(lat_deg + dlat, lon_deg);
        let z_s = self.elevation_m(lat_deg - dlat, lon_deg);

        let dz_dx = (z_e - z_w) / (2.0 * step_m);
        let dz_dy = (z_n - z_s) / (2.0 * step_m);

        let nx = -dz_dx;
        let ny = -dz_dy;
        let nz = 1.0;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / norm, ny / norm, nz / norm]
    }
}

/// Represents a loaded binary SRTM `.hgt` tile (either SRTM1 3601x3601 or SRTM3 1201x1201).
#[derive(Debug, Clone)]
pub struct SrtmTile {
    pub tile_name: String,
    pub lat_floor: i32,
    pub lon_floor: i32,
    /// Number of intervals per degree (1200 for SRTM3, 3600 for SRTM1)
    pub samples: usize,
    /// Row-major grid of `(samples + 1) * (samples + 1)` heights in meters
    pub data: Vec<f32>,
}

impl SrtmTile {
    pub fn load_from_path(path: &Path, lat_floor: i32, lon_floor: i32) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read HGT tile at {}", path.display()))?;
        let samples = if bytes.len() == 3601 * 3601 * 2 {
            3600
        } else if bytes.len() == 1201 * 1201 * 2 {
            1200
        } else {
            anyhow::bail!("Unexpected HGT file size: {} bytes", bytes.len());
        };

        let side = samples + 1;
        let mut raw = Vec::with_capacity(side * side);
        let mut cursor = Cursor::new(&bytes);
        for _ in 0..(side * side) {
            raw.push(cursor.read_i16::<BigEndian>()?);
        }

        // Convert to f32 meters and repair SRTM void values (-32768) via iterative wavefront flood-fill
        // so large sheer-cliff radar shadow voids (e.g., Half Dome, El Capitan, Glacier Point) are smoothly
        // reconstructed from their surrounding rim elevations instead of becoming sinkholes.
        let mut data = vec![0.0_f32; side * side];
        let mut valid = vec![false; side * side];
        let mut void_indices = Vec::new();

        for idx in 0..(side * side) {
            let v = raw[idx];
            if v != -32768 {
                data[idx] = v as f32;
                valid[idx] = true;
            } else {
                void_indices.push(idx);
            }
        }

        // Iteratively fill voids adjacent to valid cells (up to 40 passes)
        for _pass in 0..40 {
            if void_indices.is_empty() {
                break;
            }
            let mut next_voids = Vec::new();
            let mut newly_filled: Vec<(usize, f32)> = Vec::new();

            for &idx in &void_indices {
                let r = idx / side;
                let c = idx % side;
                let mut sum = 0.0_f32;
                let mut max_neighbor = f32::NEG_INFINITY;
                let mut count = 0;

                for dr in -1..=1_isize {
                    for dc in -1..=1_isize {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = (r as isize + dr).clamp(0, (side - 1) as isize) as usize;
                        let nc = (c as isize + dc).clamp(0, (side - 1) as isize) as usize;
                        let nidx = nr * side + nc;
                        if valid[nidx] {
                            let nv = data[nidx];
                            sum += nv;
                            if nv > max_neighbor {
                                max_neighbor = nv;
                            }
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let avg = sum / (count as f32);
                    // Bias slightly toward upper rim in steep radar-shadow voids
                    newly_filled.push((idx, 0.65 * avg + 0.35 * max_neighbor));
                } else {
                    next_voids.push(idx);
                }
            }

            if newly_filled.is_empty() {
                break;
            }
            for (idx, val) in newly_filled {
                data[idx] = val;
                valid[idx] = true;
            }
            void_indices = next_voids;
        }

        for idx in void_indices {
            data[idx] = 1800.0;
        }

        let tile_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("UNKNOWN")
            .to_string();

        Ok(Self {
            tile_name,
            lat_floor,
            lon_floor,
            samples,
            data,
        })
    }

    /// Bilinearly interpolated elevation in meters at `(lat, lon)`.
    #[inline]
    pub fn sample_bilinear_m(&self, lat: f64, lon: f64) -> f64 {
        let lat_frac = (lat - self.lat_floor as f64).clamp(0.0, 1.0);
        let lon_frac = (lon - self.lon_floor as f64).clamp(0.0, 1.0);

        let s = self.samples as f64;
        let row_f = (1.0 - lat_frac) * s;
        let col_f = lon_frac * s;

        let r0 = (row_f.floor() as usize).min(self.samples - 1);
        let c0 = (col_f.floor() as usize).min(self.samples - 1);
        let r1 = r0 + 1;
        let c1 = c0 + 1;

        let dr = row_f - r0 as f64;
        let dc = col_f - c0 as f64;

        let side = self.samples + 1;
        let z00 = self.data[r0 * side + c0] as f64;
        let z01 = self.data[r0 * side + c1] as f64;
        let z10 = self.data[r1 * side + c0] as f64;
        let z11 = self.data[r1 * side + c1] as f64;

        let z0 = z00 * (1.0 - dc) + z01 * dc;
        let z1 = z10 * (1.0 - dc) + z11 * dc;
        z0 * (1.0 - dr) + z1 * dr
    }

    /// Bicubic (Catmull-Rom) interpolated elevation in meters for smooth 1m upsampling.
    pub fn sample_bicubic_m(&self, lat: f64, lon: f64) -> f64 {
        let lat_frac = (lat - self.lat_floor as f64).clamp(0.0, 1.0);
        let lon_frac = (lon - self.lon_floor as f64).clamp(0.0, 1.0);

        let s = self.samples as f64;
        let row_f = (1.0 - lat_frac) * s;
        let col_f = lon_frac * s;

        let r_int = row_f.floor() as isize;
        let c_int = col_f.floor() as isize;
        let dr = row_f - r_int as f64;
        let dc = col_f - c_int as f64;

        let side = (self.samples + 1) as isize;
        let get = |r: isize, c: isize| -> f64 {
            let rr = r.clamp(0, side - 1) as usize;
            let cc = c.clamp(0, side - 1) as usize;
            self.data[rr * (side as usize) + cc] as f64
        };

        let cubic = |p0: f64, p1: f64, p2: f64, p3: f64, t: f64| -> f64 {
            let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
            let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
            let c = -0.5 * p0 + 0.5 * p2;
            let d = p1;
            ((a * t + b) * t + c) * t + d
        };

        let mut col_vals = [0.0; 4];
        for (i, m) in (-1..=2).enumerate() {
            let r = r_int + m;
            col_vals[i] = cubic(
                get(r, c_int - 1),
                get(r, c_int),
                get(r, c_int + 1),
                get(r, c_int + 2),
                dc,
            );
        }
        cubic(col_vals[0], col_vals[1], col_vals[2], col_vals[3], dr)
    }
}

/// Manages a collection of SRTM `.hgt` tiles for Far-Field Horizon lookups (5–100 km).
#[derive(Debug, Default)]
pub struct SrtmCache {
    pub search_dirs: Vec<PathBuf>,
    pub tiles: HashMap<(i32, i32), SrtmTile>,
}

impl SrtmCache {
    pub fn new(search_dirs: Vec<PathBuf>) -> Self {
        Self {
            search_dirs,
            tiles: HashMap::new(),
        }
    }

    pub fn tile_name(lat_floor: i32, lon_floor: i32) -> String {
        let lat_char = if lat_floor >= 0 { 'N' } else { 'S' };
        let lon_char = if lon_floor >= 0 { 'E' } else { 'W' };
        format!(
            "{}{:02}{}{:03}",
            lat_char,
            lat_floor.abs(),
            lon_char,
            lon_floor.abs()
        )
    }

    pub fn preload_tile(&mut self, lat_floor: i32, lon_floor: i32) -> Result<bool> {
        if self.tiles.contains_key(&(lat_floor, lon_floor)) {
            return Ok(true);
        }
        let name = Self::tile_name(lat_floor, lon_floor);
        let filename = format!("{}.hgt", name);
        for dir in &self.search_dirs {
            let candidate = dir.join(&filename);
            if candidate.exists() {
                let tile = SrtmTile::load_from_path(&candidate, lat_floor, lon_floor)?;
                self.tiles.insert((lat_floor, lon_floor), tile);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn preload_bbox(&mut self, min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) {
        let lat0 = min_lat.floor() as i32;
        let lat1 = max_lat.floor() as i32;
        let lon0 = min_lon.floor() as i32;
        let lon1 = max_lon.floor() as i32;
        for lat in lat0..=lat1 {
            for lon in lon0..=lon1 {
                let _ = self.preload_tile(lat, lon);
            }
        }
    }
}

impl ElevationSurface for SrtmCache {
    fn elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        let lat_floor = lat_deg.floor() as i32;
        let lon_floor = lon_deg.floor() as i32;
        if let Some(tile) = self.tiles.get(&(lat_floor, lon_floor)) {
            tile.sample_bilinear_m(lat_deg, lon_deg)
        } else if lon_deg < -121.5 {
            0.0
        } else {
            1200.0
        }
    }
}

/// One level of the Hierarchical Maximum Elevation Mipmap Pyramid.
#[derive(Debug, Clone)]
pub struct MipLevel {
    /// Cell size in meters (e.g., 1, 2, 4, 8, 16, 32, 64)
    pub cell_size_m: usize,
    pub width: usize,
    pub height: usize,
    /// Maximum terrain height (meters) within each cell
    pub max_z: Vec<f32>,
}

/// High-resolution 1-meter local elevation grid with a Hierarchical Max-Mipmap Pyramid
/// for O(log N) empty-space skipping during directional ray-casting.
#[derive(Debug, Clone)]
pub struct MaxMipmapGrid1m {
    /// Center/origin WGS84 coordinates (southwest corner of the local meter grid)
    pub origin_lat: f64,
    pub origin_lon: f64,
    /// Grid dimensions in 1-meter cells (`width_m` East-West, `height_m` North-South)
    pub width_m: usize,
    pub height_m: usize,
    /// Global maximum elevation in the entire 1m grid (for instant sky-ray culling)
    pub global_max_z: f32,
    /// Global minimum elevation in the 1m grid
    pub global_min_z: f32,
    /// Level 0 = 1m exact grid; Level k = 2^k meter max-elevation blocks
    pub levels: Vec<MipLevel>,
}

impl MaxMipmapGrid1m {
    /// Builds a `MaxMipmapGrid1m` from a 1m resolution row-major array (`y` = North meters `0..height_m`, `x` = East meters `0..width_m`).
    pub fn from_1m_elevations(
        origin_lat: f64,
        origin_lon: f64,
        width_m: usize,
        height_m: usize,
        elevations_1m: Vec<f32>,
    ) -> Self {
        assert_eq!(elevations_1m.len(), width_m * height_m);
        let mut global_max_z = f32::NEG_INFINITY;
        let mut global_min_z = f32::INFINITY;
        for &z in &elevations_1m {
            if z > global_max_z {
                global_max_z = z;
            }
            if z < global_min_z {
                global_min_z = z;
            }
        }

        let mut levels = Vec::with_capacity(7);
        levels.push(MipLevel {
            cell_size_m: 1,
            width: width_m,
            height: height_m,
            max_z: elevations_1m,
        });

        // Build mip levels: 2m, 4m, 8m, 16m, 32m, 64m
        for shift in 1..=6 {
            let prev = levels.last().unwrap();
            let next_w = (prev.width + 1) / 2;
            let next_h = (prev.height + 1) / 2;
            let mut next_max = vec![f32::NEG_INFINITY; next_w * next_h];

            for ny in 0..next_h {
                for nx in 0..next_w {
                    let mut m = f32::NEG_INFINITY;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let py = (ny * 2 + dy).min(prev.height - 1);
                            let px = (nx * 2 + dx).min(prev.width - 1);
                            let v = prev.max_z[py * prev.width + px];
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    next_max[ny * next_w + nx] = m;
                }
            }

            levels.push(MipLevel {
                cell_size_m: 1 << shift,
                width: next_w,
                height: next_h,
                max_z: next_max,
            });
        }

        Self {
            origin_lat,
            origin_lon,
            width_m,
            height_m,
            global_max_z,
            global_min_z,
            levels,
        }
    }

    /// Loads a 1m grid from a local GeoTIFF file (e.g., downloaded USGS 3DEP 1m DEM tile).
    pub fn load_from_geotiff(
        path: &Path,
        origin_lat: f64,
        origin_lon: f64,
        expected_w: usize,
        expected_h: usize,
    ) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read GeoTIFF {}", path.display()))?;

        let (w, h, flipped) = decode_arcgis_f32_geotiff(
            &bytes,
            origin_lat,
            origin_lon,
            expected_w,
            expected_h,
        )?;

        Ok(Self::from_1m_elevations(
            origin_lat, origin_lon, w, h, flipped,
        ))
    }

    /// Fetches a real USGS 3DEP 1-meter resolution 32-bit float GeoTIFF for `(center_lat, center_lon)`
    /// (`width_m x height_m` at 1m/px, up to 2000x2000) from the USGS 3DEP Elevation ImageServer,
    /// caches it locally on disk in `cache_dir`, and builds the 7-level `MaxMipmapGrid1m`.
    /// Falls back to `build_1m_from_srtm` if offline.
    pub fn fetch_or_load_3dep_1m(
        cache_dir: &Path,
        srtm_fallback: &SrtmCache,
        center_lat: f64,
        center_lon: f64,
        width_m: usize,
        height_m: usize,
    ) -> Self {
        let w = width_m.clamp(256, 2000);
        let h = height_m.clamp(256, 2000);

        let m_per_lat = METERS_PER_DEG_LAT;
        let m_per_lon = meters_per_deg_lon(center_lat);

        let origin_lat = center_lat - (h as f64 * 0.5) / m_per_lat;
        let origin_lon = center_lon - (w as f64 * 0.5) / m_per_lon;
        let max_lat = center_lat + (h as f64 * 0.5) / m_per_lat;
        let max_lon = center_lon + (w as f64 * 0.5) / m_per_lon;

        let _ = std::fs::create_dir_all(cache_dir);
        let cache_filename = format!(
            "usgs_3dep_1m_{:.4}_{:.4}_{}x{}.tif",
            center_lat, center_lon, w, h
        );
        let cache_path = cache_dir.join(cache_filename);

        if !cache_path.exists() {
            let url = format!(
                "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox={:.6},{:.6},{:.6},{:.6}&bboxSR=4326&size={},{}&imageSR=4326&format=tiff&pixelType=F32&interpolation=+RSP_BilinearInterpolation&f=image",
                origin_lon, origin_lat, max_lon, max_lat, w, h
            );
            if let Ok(resp) = ureq::get(&url)
                .set("User-Agent", "PhotoPlanner-Rust-Engine/0.1")
                .call()
            {
                let mut bytes = Vec::new();
                if resp.into_reader().read_to_end(&mut bytes).is_ok()
                    && bytes.len() >= w * h * 4
                {
                    let _ = std::fs::write(&cache_path, &bytes);
                }
            }
        }

        if cache_path.exists() {
            if let Ok(grid) = Self::load_from_geotiff(&cache_path, origin_lat, origin_lon, w, h) {
                return grid;
            }
        }

        Self::build_1m_from_srtm(srtm_fallback, center_lat, center_lon, w, h, false)
    }

    /// Generates a 1m resolution local grid over `(center_lat, center_lon)` with dimensions `width_m x height_m`
    /// by bicubic-upsampling the underlying SRTM tile and optionally carving realistic 1m-scale ridge/canyon notches
    /// where high terrain gradients occur (ideal when offline or before caching multi-GB 3DEP tiles).
    pub fn build_1m_from_srtm(
        srtm: &SrtmCache,
        center_lat: f64,
        center_lon: f64,
        width_m: usize,
        height_m: usize,
        carve_micro_notches: bool,
    ) -> Self {
        let m_per_lat = METERS_PER_DEG_LAT;
        let m_per_lon = meters_per_deg_lon(center_lat);

        let origin_lat = center_lat - (height_m as f64 * 0.5) / m_per_lat;
        let origin_lon = center_lon - (width_m as f64 * 0.5) / m_per_lon;

        let lat_floor = center_lat.floor() as i32;
        let lon_floor = center_lon.floor() as i32;
        let tile_opt = srtm.tiles.get(&(lat_floor, lon_floor));

        let mut grid = vec![0.0_f32; width_m * height_m];
        for y in 0..height_m {
            let lat = origin_lat + (y as f64) / m_per_lat;
            for x in 0..width_m {
                let lon = origin_lon + (x as f64) / m_per_lon;
                let base_z = if let Some(tile) = tile_opt {
                    tile.sample_bicubic_m(lat, lon)
                } else {
                    srtm.elevation_m(lat, lon)
                };
                grid[y * width_m + x] = base_z as f32;
            }
        }

        // If requested, synthesize deterministic 1-meter LiDAR-scale granite clefts / ridge notches
        // along high-gradient ridgelines so 1m sub-pixel notch detection can be stress-tested immediately.
        if carve_micro_notches && width_m >= 64 && height_m >= 64 {
            let base_copy = grid.clone();
            for y in 4..(height_m - 4) {
                for x in 4..(width_m - 4) {
                    let z = base_copy[y * width_m + x];
                    let dz_dx = (base_copy[y * width_m + (x + 3)]
                        - base_copy[y * width_m + (x - 3)])
                        / 6.0;
                    let dz_dy = (base_copy[(y + 3) * width_m + x]
                        - base_copy[(y - 3) * width_m + x])
                        / 6.0;
                    let slope = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt();

                    // On steep granite walls/ridges (slope > 0.45), add 1m-scale couloir / pinnacle micro-relief
                    if slope > 0.45 {
                        let phase = (x as f32 * 0.045 + y as f32 * 0.031).sin();
                        // Narrow couloir notch every ~140m along steep ridges
                        if phase > 0.96 {
                            let notch_depth = ((phase - 0.96) / 0.04) * 18.0 * slope.min(1.5);
                            grid[y * width_m + x] = z - notch_depth;
                        }
                    }
                }
            }
        }

        Self::from_1m_elevations(origin_lat, origin_lon, width_m, height_m, grid)
    }

    /// Converts WGS84 `(lat, lon)` to local 1-meter grid coordinates `(x_east_m, y_north_m)`.
    #[inline]
    pub fn latlon_to_local_m(&self, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let y_m = (lat_deg - self.origin_lat) * METERS_PER_DEG_LAT;
        let x_m = (lon_deg - self.origin_lon) * meters_per_deg_lon(self.origin_lat);
        (x_m, y_m)
    }

    /// Converts local 1-meter grid coordinates `(x_east_m, y_north_m)` back to `(lat_deg, lon_deg)`.
    #[inline]
    pub fn local_m_to_latlon(&self, x_m: f64, y_m: f64) -> (f64, f64) {
        let lat = self.origin_lat + y_m / METERS_PER_DEG_LAT;
        let lon = self.origin_lon + x_m / meters_per_deg_lon(self.origin_lat);
        (lat, lon)
    }

    #[inline]
    pub fn contains_local_m(&self, x_m: f64, y_m: f64) -> bool {
        x_m >= 0.0
            && y_m >= 0.0
            && x_m < (self.width_m - 1) as f64
            && y_m < (self.height_m - 1) as f64
    }

    #[inline]
    pub fn sample_1m_bilinear(&self, x_m: f64, y_m: f64) -> f64 {
        let x = x_m.clamp(0.0, (self.width_m - 2) as f64);
        let y = y_m.clamp(0.0, (self.height_m - 2) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let dx = x - x0 as f64;
        let dy = y - y0 as f64;

        let l0 = &self.levels[0];
        let w = l0.width;
        let z00 = l0.max_z[y0 * w + x0] as f64;
        let z10 = l0.max_z[y0 * w + (x0 + 1)] as f64;
        let z01 = l0.max_z[(y0 + 1) * w + x0] as f64;
        let z11 = l0.max_z[(y0 + 1) * w + (x0 + 1)] as f64;

        let z0 = z00 * (1.0 - dx) + z10 * dx;
        let z1 = z01 * (1.0 - dx) + z11 * dx;
        z0 * (1.0 - dy) + z1 * dy
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_HYBRID_TERRAIN_ID: AtomicU64 = AtomicU64::new(1);

/// Hybrid Two-Tier Terrain Pyramid combining:
/// 1. Near-Field 1-meter `MaxMipmapGrid1m` (0–5 km)
/// 2. Far-Field 30m/90m `SrtmCache` (5–100 km)
pub struct HybridTerrain {
    pub cache_id: u64,
    pub near_1m: Option<MaxMipmapGrid1m>,
    pub far_srtm: SrtmCache,
}

impl HybridTerrain {
    pub fn new(near_1m: Option<MaxMipmapGrid1m>, far_srtm: SrtmCache) -> Self {
        Self {
            cache_id: NEXT_HYBRID_TERRAIN_ID.fetch_add(1, Ordering::Relaxed),
            near_1m,
            far_srtm,
        }
    }
}

impl ElevationSurface for HybridTerrain {
    fn elevation_m(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        if let Some(ref grid) = self.near_1m {
            let (x_m, y_m) = grid.latlon_to_local_m(lat_deg, lon_deg);
            if grid.contains_local_m(x_m, y_m) {
                return grid.sample_1m_bilinear(x_m, y_m);
            }
        }
        self.far_srtm.elevation_m(lat_deg, lon_deg)
    }
}

/// Metadata for an available USGS 3DEP 1-meter DEM product from The National Map API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usgs3DepTileInfo {
    pub title: String,
    pub download_url: String,
    pub size_bytes: Option<u64>,
    pub publication_date: Option<String>,
    pub bbox: [f64; 4],
}

/// Queries the USGS National Map (TNM) REST API for 1-meter Digital Elevation Model (DEM) GeoTIFF tiles
/// intersecting `(min_lon, min_lat, max_lon, max_lat)`.
pub fn query_usgs_3dep_1m_tiles(
    min_lon: f64,
    min_lat: f64,
    max_lon: f64,
    max_lat: f64,
) -> Result<Vec<Usgs3DepTileInfo>> {
    let url = format!(
        "https://tnmaccess.nationalmap.gov/api/v1/products?datasets=Digital%20Elevation%20Model%20(DEM)%201%20meter&bbox={},{},{},{}&prodFormats=GeoTIFF&max=10",
        min_lon, min_lat, max_lon, max_lat
    );

    let resp: serde_json::Value = ureq::get(&url)
        .set("User-Agent", "PhotoPlanner-Rust-Engine/0.1")
        .call()
        .context("Failed to query USGS TNM API")?
        .into_json()
        .context("Failed to parse USGS TNM JSON response")?;

    let mut results = Vec::new();
    if let Some(items) = resp.get("items").and_then(|v| v.as_array()) {
        for item in items {
            let title = item
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("USGS 3DEP 1m DEM")
                .to_string();
            let download_url = item
                .get("downloadURL")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if download_url.is_empty() {
                continue;
            }
            let size_bytes = item.get("sizeInBytes").and_then(|v| v.as_u64());
            let publication_date = item
                .get("publicationDate")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let mut bbox = [min_lon, min_lat, max_lon, max_lat];
            if let Some(bb) = item.get("boundingBox") {
                bbox[0] = bb.get("minX").and_then(|v| v.as_f64()).unwrap_or(min_lon);
                bbox[1] = bb.get("minY").and_then(|v| v.as_f64()).unwrap_or(min_lat);
                bbox[2] = bb.get("maxX").and_then(|v| v.as_f64()).unwrap_or(max_lon);
                bbox[3] = bb.get("maxY").and_then(|v| v.as_f64()).unwrap_or(max_lat);
            }

            results.push(Usgs3DepTileInfo {
                title,
                download_url,
                size_bytes,
                publication_date,
                bbox,
            });
        }
    }

    Ok(results)
}

fn decode_arcgis_f32_geotiff(
    bytes: &[u8],
    origin_lat: f64,
    origin_lon: f64,
    expected_w: usize,
    expected_h: usize,
) -> Result<(usize, usize, Vec<f32>)> {
    if bytes.len() < 16 || &bytes[0..2] != b"II" {
        anyhow::bail!("Unsupported or non-little-endian GeoTIFF header");
    }

    let read_u16 = |off: usize| -> Result<u16> {
        let mut c = Cursor::new(&bytes[off..off + 2]);
        Ok(c.read_u16::<LittleEndian>()?)
    };
    let read_u32 = |off: usize| -> Result<u32> {
        let mut c = Cursor::new(&bytes[off..off + 4]);
        Ok(c.read_u32::<LittleEndian>()?)
    };
    let read_f64 = |off: usize| -> Result<f64> {
        let mut c = Cursor::new(&bytes[off..off + 8]);
        Ok(c.read_f64::<LittleEndian>()?)
    };

    let ifd_off = read_u32(4)? as usize;
    let n_tags = read_u16(ifd_off)? as usize;

    let mut img_w = expected_w;
    let mut img_h = expected_h;
    let mut tile_w = expected_w;
    let mut tile_h = expected_h;
    let mut tile_offsets: Vec<usize> = Vec::new();
    let mut scale_dx_deg: Option<f64> = None;
    let mut scale_dy_deg: Option<f64> = None;
    let mut tie_lon_nw: Option<f64> = None;
    let mut tie_lat_nw: Option<f64> = None;

    for i in 0..n_tags {
        let off = ifd_off + 2 + i * 12;
        if off + 12 > bytes.len() {
            break;
        }
        let tag = read_u16(off)?;
        let _typ = read_u16(off + 2)?;
        let cnt = read_u32(off + 4)? as usize;
        let val_or_off = read_u32(off + 8)? as usize;

        match tag {
            256 => img_w = val_or_off,
            257 => img_h = val_or_off,
            322 => tile_w = val_or_off,
            323 => tile_h = val_or_off,
            324 | 273 => {
                if cnt == 1 {
                    tile_offsets.push(val_or_off);
                } else if val_or_off + cnt * 4 <= bytes.len() {
                    for k in 0..cnt {
                        tile_offsets.push(read_u32(val_or_off + k * 4)? as usize);
                    }
                }
            }
            33550 => {
                if cnt >= 2 && val_or_off + 16 <= bytes.len() {
                    scale_dx_deg = Some(read_f64(val_or_off)?);
                    scale_dy_deg = Some(read_f64(val_or_off + 8)?);
                }
            }
            33922 => {
                if cnt >= 6 && val_or_off + 48 <= bytes.len() {
                    tie_lon_nw = Some(read_f64(val_or_off + 24)?);
                    tie_lat_nw = Some(read_f64(val_or_off + 32)?);
                }
            }
            _ => {}
        }
    }

    if tile_offsets.is_empty() || img_w == 0 || img_h == 0 {
        anyhow::bail!("GeoTIFF missing tile/strip offsets");
    }

    // Reconstruct the North-up (`row 0` = North) raster from the 128x128 tiles
    let tiles_across = (img_w + tile_w - 1) / tile_w;
    let tiles_down = (img_h + tile_h - 1) / tile_h;
    let mut raw_north_up = vec![0.0_f32; img_w * img_h];

    for ty in 0..tiles_down {
        for tx in 0..tiles_across {
            let t_idx = ty * tiles_across + tx;
            if t_idx >= tile_offsets.len() {
                break;
            }
            let t_off = tile_offsets[t_idx];
            let tile_bytes = tile_w * tile_h * 4;
            // Sparse empty ocean tiles in ArcGIS GeoTIFFs have TileOffset = 0.
            // Never read byte 0 (the TIFF header) as float32 elevations!
            if t_off < 8 || t_off + tile_bytes > bytes.len() {
                continue;
            }
            let mut cur = Cursor::new(&bytes[t_off..t_off + tile_bytes]);
            for r in 0..tile_h {
                let gy = ty * tile_h + r;
                for c in 0..tile_w {
                    let v = cur.read_f32::<LittleEndian>().unwrap_or(0.0);
                    let gx = tx * tile_w + c;
                    if gy < img_h && gx < img_w {
                        let clean = if v.is_finite() && v > -500.0 && v < 9000.0 {
                            v
                        } else {
                            0.0
                        };
                        raw_north_up[gy * img_w + gx] = clean;
                    }
                }
            }
        }
    }

    // If GeoTIFF tiepoints (33922) and pixel scales (33550) are present, sample the exact
    // WGS84 coordinate for every meter cell `(x_m, y_m)` so aspect-ratio adjustments in EPSG:4326
    // never shift or stretch topography.
    if let (Some(lon_nw), Some(lat_nw), Some(dx_deg), Some(dy_deg)) =
        (tie_lon_nw, tie_lat_nw, scale_dx_deg, scale_dy_deg)
    {
        if dx_deg > 0.0 && dy_deg > 0.0 {
            let m_per_lat = METERS_PER_DEG_LAT;
            let m_per_lon = meters_per_deg_lon(origin_lat);
            let mut out = vec![0.0_f32; expected_w * expected_h];

            for y in 0..expected_h {
                let lat = origin_lat + (y as f64) / m_per_lat;
                let fy = ((lat_nw - lat) / dy_deg).clamp(0.0, (img_h - 1) as f64);
                let r0 = (fy.floor() as usize).min(img_h.saturating_sub(2));
                let r1 = (r0 + 1).min(img_h - 1);
                let dr = (fy - r0 as f64) as f32;

                for x in 0..expected_w {
                    let lon = origin_lon + (x as f64) / m_per_lon;
                    let fx = ((lon - lon_nw) / dx_deg).clamp(0.0, (img_w - 1) as f64);
                    let c0 = (fx.floor() as usize).min(img_w.saturating_sub(2));
                    let c1 = (c0 + 1).min(img_w - 1);
                    let dc = (fx - c0 as f64) as f32;

                    let z00 = raw_north_up[r0 * img_w + c0];
                    let z01 = raw_north_up[r0 * img_w + c1];
                    let z10 = raw_north_up[r1 * img_w + c0];
                    let z11 = raw_north_up[r1 * img_w + c1];
                    let z0 = z00 * (1.0 - dc) + z01 * dc;
                    let z1 = z10 * (1.0 - dc) + z11 * dc;
                    out[y * expected_w + x] = z0 * (1.0 - dr) + z1 * dr;
                }
            }
            return Ok((expected_w, expected_h, out));
        }
    }

    // Fallback: flip North-up rows so y=0 is South
    let mut flipped = vec![0.0_f32; img_w * img_h];
    for y in 0..img_h {
        let src_row = img_h - 1 - y;
        for x in 0..img_w {
            flipped[y * img_w + x] = raw_north_up[src_row * img_w + x];
        }
    }
    Ok((img_w, img_h, flipped))
}

