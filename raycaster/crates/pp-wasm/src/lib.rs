//! `pp-wasm` — Zero-glue WebAssembly (`wasm32-unknown-unknown`) bridge for:
//! 1. Real-time 3D Moist Atmospheric Fluid Dynamics & Upper-Tropospheric 3D Cirrus Microphysics (`pp-atmosphere`)
//! 2. Tileable `64³` Perlin-Worley + Curl-Filament 1-Meter Detail Texture synthesis
//! 3. Sub-millisecond client-side 3D Terrain Sightline & 360° Horizon Raycasting

use pp_atmosphere::{
    generate_detail_noise_volume_64, AtmosphericParams, FluidAtmosphere, DEFAULT_DOMAIN_XY_M,
    DEFAULT_Z_TOP_M, GRID_CELLS, GRID_NZ,
};
use std::cell::RefCell;

const METERS_PER_DEG_LAT: f64 = 111_132.92;

struct DemPatchState {
    elevations_m: Vec<f32>,
    width: usize,
    height: usize,
    patch_w_m: f64,
    patch_h_m: f64,
    center_lat: f64,
    center_lon: f64,
}

impl Default for DemPatchState {
    fn default() -> Self {
        Self {
            elevations_m: Vec::new(),
            width: 0,
            height: 0,
            patch_w_m: 9200.0,
            patch_h_m: 9200.0,
            center_lat: 37.59,
            center_lon: -122.51,
        }
    }
}

impl DemPatchState {
    #[inline]
    fn sample_elevation_local_xy(&self, x_east_m: f64, y_north_m: f64) -> Option<f64> {
        if self.width < 2 || self.height < 2 || self.elevations_m.len() < self.width * self.height {
            return None;
        }
        let u = (x_east_m + 0.5 * self.patch_w_m) / self.patch_w_m;
        let v = (y_north_m + 0.5 * self.patch_h_m) / self.patch_h_m;
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        let fx = u * (self.width - 1) as f64;
        let fy = v * (self.height - 1) as f64;
        let c0 = (fx.floor() as usize).min(self.width - 2);
        let r0 = (fy.floor() as usize).min(self.height - 2);
        let dc = (fx - c0 as f64) as f32;
        let dr = (fy - r0 as f64) as f32;

        let z00 = self.elevations_m[r0 * self.width + c0];
        let z10 = self.elevations_m[r0 * self.width + c0 + 1];
        let z01 = self.elevations_m[(r0 + 1) * self.width + c0];
        let z11 = self.elevations_m[(r0 + 1) * self.width + c0 + 1];
        let z0 = z00 * (1.0 - dc) + z10 * dc;
        let z1 = z01 * (1.0 - dc) + z11 * dc;
        Some((z0 * (1.0 - dr) + z1 * dr) as f64)
    }

    #[inline]
    fn sample_elevation_lat_lon(&self, lat: f64, lon: f64) -> f64 {
        let m_per_lon = METERS_PER_DEG_LAT * self.center_lat.to_radians().cos().max(0.2);
        let x_east = (lon - self.center_lon) * m_per_lon;
        let y_north = (lat - self.center_lat) * METERS_PER_DEG_LAT;
        self.sample_elevation_local_xy(x_east, y_north).unwrap_or(0.0)
    }
}

struct WasmEngineState {
    atmosphere: FluidAtmosphere,
    detail_noise: Vec<u8>,
    dem: DemPatchState,
    telemetry_buf: [f32; 16],
    /// Packed output for `wasm_raycast_tripod`:
    /// - `[0..8]`: header `[tripod_z_m, dist_m, bearing_deg, el_angle_deg, max_obs_m, obs_dist_m, is_visible, comp_score]`
    /// - `[8 .. 8 + 65*3]`: 65 sightline samples `[dist_m, eff_terrain_m, ray_z_m]` (`195` floats)
    /// - `[203 .. 203 + 180]`: 180 horizon elevation angles `horizon_el_deg` for azimuths `0, 2, 4, ..., 358°`
    raycast_out: Vec<f32>,
}

impl WasmEngineState {
    fn new(seed: u32) -> Self {
        let mut params = AtmosphericParams::default();
        params.seed = seed;
        let atmosphere = FluidAtmosphere::new(params);
        let detail_noise = generate_detail_noise_volume_64(seed.wrapping_add(17));
        let mut engine = Self {
            atmosphere,
            detail_noise,
            dem: DemPatchState::default(),
            telemetry_buf: [0.0; 16],
            raycast_out: vec![0.0; 8 + 65 * 3 + 180],
        };
        engine.refresh_telemetry();
        engine
    }

    fn refresh_telemetry(&mut self) {
        let t = self.atmosphere.compute_telemetry();
        self.telemetry_buf = [
            t.sim_time_s,
            t.lcl_altitude_m,
            t.freezing_level_m,
            t.glaciation_cirrus_base_m,
            t.max_updraft_ms,
            t.max_downdraft_ms,
            t.max_liquid_g_kg,
            t.max_ice_g_kg,
            t.low_cloud_coverage,
            t.mid_cloud_coverage,
            t.high_cirrus_coverage,
            t.froude_number,
            t.jet_shear_ms_km,
            self.atmosphere.params.z_base_m,
            self.atmosphere.params.z_top_m,
            self.atmosphere.params.domain_xy_m,
        ];
    }
}

thread_local! {
    static ENGINE: RefCell<Option<WasmEngineState>> = const { RefCell::new(None) };
}

fn with_engine<F, R>(f: F) -> R
where
    F: FnOnce(&mut WasmEngineState) -> R,
{
    ENGINE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(WasmEngineState::new(1337));
        }
        f(borrow.as_mut().unwrap())
    })
}

#[no_mangle]
pub extern "C" fn wasm_alloc_f32(len: usize) -> *mut f32 {
    let mut v = Vec::<f32>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[no_mangle]
pub extern "C" fn wasm_free_f32(ptr: *mut f32, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[no_mangle]
pub extern "C" fn wasm_init_atmosphere(seed: u32) {
    ENGINE.with(|cell| {
        *cell.borrow_mut() = Some(WasmEngineState::new(seed));
    });
}

#[no_mangle]
pub extern "C" fn wasm_detail_noise_ptr() -> *const u8 {
    with_engine(|eng| eng.detail_noise.as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_detail_noise_len() -> usize {
    with_engine(|eng| eng.detail_noise.len())
}

#[no_mangle]
pub extern "C" fn wasm_volume_ptr() -> *const u8 {
    with_engine(|eng| eng.atmosphere.rgba_volume.as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_volume_len() -> usize {
    GRID_CELLS * 4
}

#[no_mangle]
pub extern "C" fn wasm_wind_profile_ptr() -> *const f32 {
    with_engine(|eng| eng.atmosphere.wind_profile_flat.as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_wind_profile_len() -> usize {
    GRID_NZ * 4
}

#[no_mangle]
pub extern "C" fn wasm_telemetry_ptr() -> *const f32 {
    with_engine(|eng| eng.telemetry_buf.as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_telemetry_len() -> usize {
    16
}

/// Ingests the full `dem_w × dem_h` elevation grid (meters ASL) into WASM for both
/// 3D fluid boundary coupling and `< 0.3 ms` client-side tripod raycasting.
#[no_mangle]
pub extern "C" fn wasm_set_terrain_dem(
    ptr: *const f32,
    dem_w: usize,
    dem_h: usize,
    patch_w_m: f32,
    patch_h_m: f32,
    center_lat: f64,
    center_lon: f64,
) {
    if ptr.is_null() || dem_w < 2 || dem_h < 2 {
        return;
    }
    let count = dem_w * dem_h;
    let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
    with_engine(|eng| {
        eng.dem.elevations_m.clear();
        eng.dem.elevations_m.extend_from_slice(slice);
        eng.dem.width = dem_w;
        eng.dem.height = dem_h;
        eng.dem.patch_w_m = patch_w_m as f64;
        eng.dem.patch_h_m = patch_h_m as f64;
        eng.dem.center_lat = center_lat;
        eng.dem.center_lon = center_lon;

        eng.atmosphere
            .set_terrain_from_dem_patch(slice, dem_w, dem_h, patch_w_m, patch_h_m);
        eng.atmosphere.initialize_fields_and_spinup(8);
        eng.refresh_telemetry();
    });
}

/// Updates `AtmosphericParams` from a `22`-float buffer:
/// `[0] surface_temp_c`, `[1] surface_rh_pct`, `[2] low_cloud_pct`, `[3] mid_cloud_pct`, `[4] high_cloud_pct`,
/// `[5] inversion_base_m`, `[6] inversion_thickness_m`, `[7] inversion_strength_k`,
/// `[8] wind_sfc_u`, `[9] wind_sfc_v`, `[10] wind_850_u`, `[11] wind_850_v`,
/// `[12] wind_500_u`, `[13] wind_500_v`, `[14] wind_250_u`, `[15] wind_250_v`,
/// `[16] convective_cape`, `[17] orographic_wave_gain`, `[18] vorticity_confinement`,
/// `[19] cirrus_ice_boost`, `[20] solar_elevation_deg`, `[21] solar_azimuth_deg`, `[22] seed`.
#[no_mangle]
pub extern "C" fn wasm_set_params(params_ptr: *const f32, len: usize, reseed: u32) {
    if params_ptr.is_null() || len < 22 {
        return;
    }
    let s = unsafe { std::slice::from_raw_parts(params_ptr, len) };
    with_engine(|eng| {
        let p = AtmosphericParams {
            z_base_m: eng.atmosphere.params.z_base_m,
            z_top_m: eng.atmosphere.params.z_top_m.max(DEFAULT_Z_TOP_M),
            domain_xy_m: DEFAULT_DOMAIN_XY_M,
            surface_temp_c: s[0],
            surface_rh_pct: s[1],
            low_cloud_pct: s[2],
            mid_cloud_pct: s[3],
            high_cloud_pct: s[4],
            inversion_base_m: s[5],
            inversion_thickness_m: s[6],
            inversion_strength_k: s[7],
            wind_sfc_u: s[8],
            wind_sfc_v: s[9],
            wind_850_u: s[10],
            wind_850_v: s[11],
            wind_500_u: s[12],
            wind_500_v: s[13],
            wind_250_u: s[14],
            wind_250_v: s[15],
            convective_cape: s[16],
            orographic_wave_gain: s[17],
            vorticity_confinement: s[18],
            cirrus_ice_boost: s[19],
            solar_elevation_deg: s[20],
            solar_azimuth_deg: s[21],
            seed: if len >= 23 { s[22] as u32 } else { eng.atmosphere.params.seed },
        };
        eng.atmosphere.update_params(p, reseed != 0);
        eng.refresh_telemetry();
    });
}

#[no_mangle]
pub extern "C" fn wasm_step_atmosphere(dt_s: f32) {
    with_engine(|eng| {
        eng.atmosphere.step(dt_s);
        eng.refresh_telemetry();
    });
}

/// Casts an instant `< 0.3 ms` 3D camera sightline + 360° mountain horizon scan on the loaded DEM.
#[no_mangle]
pub extern "C" fn wasm_raycast_tripod(
    tripod_lat: f64,
    tripod_lon: f64,
    target_lat: f64,
    target_lon: f64,
    target_z_m: f64,
) -> *const f32 {
    with_engine(|eng| {
        let tripod_z = eng.dem.sample_elevation_lat_lon(tripod_lat, tripod_lon);
        let m_per_lon = METERS_PER_DEG_LAT * eng.dem.center_lat.to_radians().cos().max(0.2);

        let dx_m = (target_lon - tripod_lon) * m_per_lon;
        let dy_m = (target_lat - tripod_lat) * METERS_PER_DEG_LAT;
        let total_dist_m = (dx_m * dx_m + dy_m * dy_m).sqrt().max(10.0);
        let bearing_deg = dx_m.atan2(dy_m).to_degrees().rem_euclid(360.0);

        let cam_eye_z = tripod_z + 1.75;
        let aim_z = if target_z_m > 1.0 {
            target_z_m
        } else {
            eng.dem.sample_elevation_lat_lon(target_lat, target_lon)
        };
        let drop_target = (total_dist_m * total_dist_m) / (2.0 * 6_371_000.0) * 0.87;
        let eff_aim_z = aim_z - drop_target;
        let el_angle_deg = ((eff_aim_z - cam_eye_z) / total_dist_m)
            .atan()
            .to_degrees();

        let mut max_obs_m = 0.0_f64;
        let mut obs_dist_m = 0.0_f64;

        for s in 0..=64usize {
            let frac = s as f64 / 64.0;
            let d_m = frac * total_dist_m;
            let slat = tripod_lat + frac * (target_lat - tripod_lat);
            let slon = tripod_lon + frac * (target_lon - tripod_lon);
            let tz = eng.dem.sample_elevation_lat_lon(slat, slon);
            let curv = (d_m * d_m) / (2.0 * 6_371_000.0) * 0.87;
            let eff_tz = tz - curv;
            let ray_z = cam_eye_z + frac * (eff_aim_z - cam_eye_z);

            if (3..62).contains(&s) {
                let diff = eff_tz - ray_z;
                if diff > max_obs_m {
                    max_obs_m = diff;
                    obs_dist_m = d_m;
                }
            }

            let base = 8 + s * 3;
            eng.raycast_out[base] = d_m as f32;
            eng.raycast_out[base + 1] = eff_tz as f32;
            eng.raycast_out[base + 2] = ray_z as f32;
        }

        let is_visible = if max_obs_m <= 1.5 { 1.0_f32 } else { 0.0_f32 };
        let dist_bonus = if (1200.0..=7500.0).contains(&total_dist_m) {
            1.0
        } else {
            0.78
        };
        let comp_score = if is_visible > 0.5 {
            (84.0 * dist_bonus) as f32
        } else {
            (18.0_f64.max(55.0 - max_obs_m * 0.4)) as f32
        };

        // 360° horizon sweep at 2° intervals inside the high-res DEM patch
        let tp_x = (tripod_lon - eng.dem.center_lon) * m_per_lon;
        let tp_y = (tripod_lat - eng.dem.center_lat) * METERS_PER_DEG_LAT;
        for idx in 0..180usize {
            let az_rad = (idx as f64 * 2.0).to_radians();
            let dir_x = az_rad.sin();
            let dir_y = az_rad.cos();
            let mut max_slope = -0.05_f64; // slight ocean dip fallback
            let mut d = 45.0_f64;
            while d < 9000.0 {
                let sx = tp_x + d * dir_x;
                let sy = tp_y + d * dir_y;
                let Some(tz) = eng.dem.sample_elevation_local_xy(sx, sy) else {
                    break;
                };
                let curv = (d * d) / (2.0 * 6_371_000.0) * 0.87;
                let slope = (tz - curv - cam_eye_z) / d;
                if slope > max_slope {
                    max_slope = slope;
                }
                d += if d < 1200.0 { 22.0 } else { 55.0 };
            }
            eng.raycast_out[203 + idx] = max_slope.atan().to_degrees() as f32;
        }

        eng.raycast_out[0] = tripod_z as f32;
        eng.raycast_out[1] = total_dist_m as f32;
        eng.raycast_out[2] = bearing_deg as f32;
        eng.raycast_out[3] = el_angle_deg as f32;
        eng.raycast_out[4] = max_obs_m as f32;
        eng.raycast_out[5] = obs_dist_m as f32;
        eng.raycast_out[6] = is_visible;
        eng.raycast_out[7] = comp_score;
        eng.raycast_out.as_ptr()
    })
}

#[no_mangle]
pub extern "C" fn wasm_raycast_tripod_len() -> usize {
    8 + 65 * 3 + 180
}
