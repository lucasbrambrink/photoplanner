//! `pp-atmosphere` — 3D Moist Atmospheric Fluid Dynamics, Orographic Gravity-Wave,
//! Liquid + Upper-Tropospheric Ice (3D Cirrus) Microphysics, and Multi-Scale
//! Procedural Detail Volume Generator.
//!
//! Designed with zero OS-specific dependencies so the exact same solver compiles
//! to native `aarch64`/`x86_64` (`pp-cli`) and `wasm32-unknown-unknown` (`pp-wasm`).

use serde::{Deserialize, Serialize};

/// 3D Eulerian / Semi-Lagrangian atmospheric grid dimensions (`64 × 64 × 48 = 196,608` cells).
/// Covers an `18 km × 18 km` horizontal domain around the landmark and stretches vertically
/// from `z_base_m` (sea level / valley floor) to `z_top_m` (`12,200 m` ASL — mid-latitude tropopause).
pub const GRID_NX: usize = 64;
pub const GRID_NY: usize = 64;
pub const GRID_NZ: usize = 48;
pub const GRID_CELLS: usize = GRID_NX * GRID_NY * GRID_NZ;

pub const DEFAULT_DOMAIN_XY_M: f32 = 18_000.0;
pub const DEFAULT_Z_TOP_M: f32 = 12_200.0;

/// Physical constants (SI units).
const GRAVITY: f32 = 9.80665;
const RD: f32 = 287.058; // Dry air gas constant (J / kg / K)
const RV: f32 = 461.5; // Water vapor gas constant (J / kg / K)
const CP: f32 = 1004.0; // Specific heat of dry air at constant pressure (J / kg / K)
const LV: f32 = 2.501e6; // Latent heat of vaporization (J / kg)
const LS: f32 = 2.834e6; // Latent heat of sublimation for cirrus ice (J / kg)
const P0_PA: f32 = 101_325.0;

/// Stretched vertical coordinate parameters:
/// `z(s) = z_base + z_span * (STRETCH_A * s + STRETCH_B * s^2)` for `s in [0, 1]`.
/// Exact analytical inverse:
/// `s(z) = (-STRETCH_A + sqrt(STRETCH_A^2 + 4 * STRETCH_B * zeta)) / (2 * STRETCH_B)`.
pub const STRETCH_A: f32 = 0.26;
pub const STRETCH_B: f32 = 0.74;

#[inline]
pub fn s_to_altitude_m(s: f32, z_base_m: f32, z_top_m: f32) -> f32 {
    let sc = s.clamp(0.0, 1.0);
    let span = (z_top_m - z_base_m).max(1000.0);
    z_base_m + span * (STRETCH_A * sc + STRETCH_B * sc * sc)
}

#[inline]
pub fn altitude_m_to_s(z_m: f32, z_base_m: f32, z_top_m: f32) -> f32 {
    let span = (z_top_m - z_base_m).max(1000.0);
    let zeta = ((z_m - z_base_m) / span).clamp(0.0, 1.0);
    let disc = STRETCH_A * STRETCH_A + 4.0 * STRETCH_B * zeta;
    ((-STRETCH_A + disc.sqrt()) / (2.0 * STRETCH_B)).clamp(0.0, 1.0)
}

/// Meteorological & fluid-simulation boundary condition parameters.
/// Can be imputed directly from Open-Meteo multi-pressure-level forecasts (`1000..200 hPa`)
/// or driven interactively from UI atmospheric presets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AtmosphericParams {
    pub z_base_m: f32,
    pub z_top_m: f32,
    pub domain_xy_m: f32,
    pub surface_temp_c: f32,
    pub surface_rh_pct: f32,
    pub low_cloud_pct: f32,
    pub mid_cloud_pct: f32,
    pub high_cloud_pct: f32,
    /// Marine boundary layer / valley inversion base altitude (meters ASL).
    pub inversion_base_m: f32,
    /// Inversion transition depth (meters).
    pub inversion_thickness_m: f32,
    /// Potential temperature jump across the inversion (Kelvin). Strong coastal inversion ~ 6..10 K.
    pub inversion_strength_k: f32,
    /// 10m surface wind vector `(u_east, v_north)` in m/s.
    pub wind_sfc_u: f32,
    pub wind_sfc_v: f32,
    /// 850 hPa (~1,500 m ASL) wind vector in m/s.
    pub wind_850_u: f32,
    pub wind_850_v: f32,
    /// 500 hPa (~5,600 m ASL) mid-tropospheric wind vector in m/s.
    pub wind_500_u: f32,
    pub wind_500_v: f32,
    /// 250 hPa (~10,400 m ASL) upper-tropospheric jet-stream wind vector in m/s (shears 3D cirrus!).
    pub wind_250_u: f32,
    pub wind_250_v: f32,
    /// Convective Available Potential Energy proxy (`0..2500 J/kg`) driving cumulus updraft plumes.
    pub convective_cape: f32,
    /// Multiplier on Scorer mountain gravity-wave amplitude (`0.0..2.5`).
    pub orographic_wave_gain: f32,
    /// Vorticity confinement strength (`0.0..2.5`) controlling Kelvin-Helmholtz & billow roll energy.
    pub vorticity_confinement: f32,
    /// Upper-tropospheric (`7.2..11.8 km`) ice supersaturation boost (`0.0..2.0`) for 3D cirrus uncinus.
    pub cirrus_ice_boost: f32,
    /// Current solar geometry (drives sunlit anabatic slope thermals vs. nocturnal katabatic valley mist).
    pub solar_elevation_deg: f32,
    pub solar_azimuth_deg: f32,
    /// Deterministic seed for synoptic moisture perturbation phase.
    pub seed: u32,
}

impl Default for AtmosphericParams {
    fn default() -> Self {
        Self {
            z_base_m: 0.0,
            z_top_m: DEFAULT_Z_TOP_M,
            domain_xy_m: DEFAULT_DOMAIN_XY_M,
            surface_temp_c: 14.5,
            surface_rh_pct: 84.0,
            low_cloud_pct: 58.0,
            mid_cloud_pct: 24.0,
            high_cloud_pct: 42.0,
            inversion_base_m: 340.0,
            inversion_thickness_m: 210.0,
            inversion_strength_k: 6.5,
            wind_sfc_u: 5.2,
            wind_sfc_v: -3.4,
            wind_850_u: 8.5,
            wind_850_v: -2.0,
            wind_500_u: 18.0,
            wind_500_v: 4.5,
            wind_250_u: 36.0,
            wind_250_v: 14.0,
            convective_cape: 320.0,
            orographic_wave_gain: 1.15,
            vorticity_confinement: 1.15,
            cirrus_ice_boost: 1.0,
            solar_elevation_deg: 11.0,
            solar_azimuth_deg: 256.0,
            seed: 1337,
        }
    }
}

/// Live summary telemetry computed from the 3D fluid & microphysics state.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct AtmosphereTelemetry {
    pub sim_time_s: f32,
    pub lcl_altitude_m: f32,
    pub freezing_level_m: f32,
    pub glaciation_cirrus_base_m: f32,
    pub max_updraft_ms: f32,
    pub max_downdraft_ms: f32,
    pub max_liquid_g_kg: f32,
    pub max_ice_g_kg: f32,
    pub low_cloud_coverage: f32,
    pub mid_cloud_coverage: f32,
    pub high_cirrus_coverage: f32,
    pub froude_number: f32,
    pub jet_shear_ms_km: f32,
}

#[inline]
fn idx3(x: usize, y: usize, z: usize) -> usize {
    (z * GRID_NY + y) * GRID_NX + x
}

#[inline]
fn idx2(x: usize, y: usize) -> usize {
    y * GRID_NX + x
}

/// Fast deterministic integer hash -> `[0, 1]`.
#[inline]
fn hash_u32_f32(mut h: u32) -> f32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (h & 0x00ff_ffff) as f32 * (1.0 / 16_777_215.0)
}

#[inline]
fn smoothstep01(t: f32) -> f32 {
    let c = t.clamp(0.0, 1.0);
    c * c * (3.0 - 2.0 * c)
}

/// Tileable 3D value noise on period `(px, py, pz)`.
fn periodic_value_noise3(x: f32, y: f32, z: f32, px: i32, py: i32, pz: i32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let fx = smoothstep01(x - xi as f32);
    let fy = smoothstep01(y - yi as f32);
    let fz = smoothstep01(z - zi as f32);

    let x0 = xi.rem_euclid(px) as u32;
    let x1 = (xi + 1).rem_euclid(px) as u32;
    let y0 = yi.rem_euclid(py) as u32;
    let y1 = (yi + 1).rem_euclid(py) as u32;
    let z0 = zi.rem_euclid(pz) as u32;
    let z1 = (zi + 1).rem_euclid(pz) as u32;

    let sample = |ix: u32, iy: u32, iz: u32| -> f32 {
        let h = ix
            .wrapping_mul(374_761_393)
            .wrapping_add(iy.wrapping_mul(668_265_263))
            .wrapping_add(iz.wrapping_mul(2_147_483_647))
            .wrapping_add(seed.wrapping_mul(1_664_525));
        hash_u32_f32(h)
    };

    let c000 = sample(x0, y0, z0);
    let c100 = sample(x1, y0, z0);
    let c010 = sample(x0, y1, z0);
    let c110 = sample(x1, y1, z0);
    let c001 = sample(x0, y0, z1);
    let c101 = sample(x1, y0, z1);
    let c011 = sample(x0, y1, z1);
    let c111 = sample(x1, y1, z1);

    let cx00 = c000 + fx * (c100 - c000);
    let cx10 = c010 + fx * (c110 - c010);
    let cx01 = c001 + fx * (c101 - c001);
    let cx11 = c011 + fx * (c111 - c011);

    let cy0 = cx00 + fy * (cx10 - cx00);
    let cy1 = cx01 + fy * (cx11 - cx01);

    cy0 + fz * (cy1 - cy0)
}

fn periodic_fbm3(x: f32, y: f32, z: f32, base_p: i32, octaves: usize, seed: u32) -> f32 {
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    let mut amp = 0.5_f32;
    let mut freq = 1.0_f32;
    let mut p = base_p;
    for oct in 0..octaves {
        sum += amp
            * periodic_value_noise3(
                x * freq,
                y * freq,
                z * freq,
                p,
                p,
                p,
                seed.wrapping_add(oct as u32 * 977),
            );
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
        p *= 2;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.5
    }
}

/// Tileable 3D Worley cellular distance `F1` (`0` at cell feature point, `~1` at cell border)
/// on integer period `period`.
fn periodic_worley3(x: f32, y: f32, z: f32, period: i32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let fx = x - xi as f32;
    let fy = y - yi as f32;
    let fz = z - zi as f32;

    let mut min_d2 = 10.0_f32;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = (xi + dx).rem_euclid(period) as u32;
                let cy = (yi + dy).rem_euclid(period) as u32;
                let cz = (zi + dz).rem_euclid(period) as u32;
                let base_h = (cx.wrapping_mul(73_856_093)
                    ^ cy.wrapping_mul(19_349_663)
                    ^ cz.wrapping_mul(83_492_791))
                .wrapping_add(seed.wrapping_mul(2_654_435_761));
                let jx = hash_u32_f32(base_h);
                let jy = hash_u32_f32(base_h.wrapping_add(0x9e37_79b9));
                let jz = hash_u32_f32(base_h.wrapping_add(0x517c_c1b7));

                let rx = dx as f32 + jx - fx;
                let ry = dy as f32 + jy - fy;
                let rz = dz as f32 + jz - fz;
                let d2 = rx * rx + ry * ry + rz * rz;
                if d2 < min_d2 {
                    min_d2 = d2;
                }
            }
        }
    }
    min_d2.sqrt().min(1.0)
}

fn periodic_worley_fbm3(x: f32, y: f32, z: f32, base_p: i32, seed: u32) -> f32 {
    let w1 = 1.0 - periodic_worley3(x, y, z, base_p, seed);
    let w2 = 1.0 - periodic_worley3(x * 2.0, y * 2.0, z * 2.0, base_p * 2, seed + 101);
    let w3 = 1.0 - periodic_worley3(x * 4.0, y * 4.0, z * 4.0, base_p * 4, seed + 307);
    (0.625 * w1 + 0.25 * w2 + 0.125 * w3).clamp(0.0, 1.0)
}

/// Generates a seamless, tileable `64 × 64 × 64` RGBA8 (`1,048,576` bytes) 3D Procedural Cloud
/// Micro-Structure Volume (`uDetailNoise3D`) for GPU hardware trilinear sampling down to `1.0 m` scale:
/// - `R`: High-contrast Perlin-Worley cauliflower billow (`Horizon / Nubis` sculptural dome remap)
/// - `G`: Multi-octave 3D Worley cellular floret & cleft FBM (carves deep 3D canyons between turrets)
/// - `B`: Anisotropic curl-sheared silk & herringbone filament field (for 3D cirrus uncinus & virga)
/// - `A`: High-frequency Kolmogorov `-5/3` + cellular 1m micro-turbulence octave (ethereal wisps)
pub fn generate_detail_noise_volume_64(seed: u32) -> Vec<u8> {
    const N: usize = 64;
    let mut out = vec![0u8; N * N * N * 4];
    let inv_n = 1.0 / (N as f32);

    for z in 0..N {
        let u_z = z as f32 * inv_n;
        for y in 0..N {
            let u_y = y as f32 * inv_n;
            for x in 0..N {
                let u_x = x as f32 * inv_n;

                // 1. Sculptural Perlin-Worley cumulus billow base (base period = 4)
                let p_fbm = periodic_fbm3(u_x * 4.0, u_y * 4.0, u_z * 4.0, 4, 4, seed);
                let w_fbm = periodic_worley_fbm3(u_x * 4.0, u_y * 4.0, u_z * 4.0, 4, seed + 19);
                // Crisp Nubis billow remap with full [0..1] dynamic range for round 3D domes & deep clefts
                let denom = (1.0 - (1.0 - w_fbm) * 0.62).max(0.12);
                let pw_raw = ((p_fbm - (1.0 - w_fbm) * 0.46) / denom).clamp(0.0, 1.0);
                let perlin_worley = smoothstep01((pw_raw - 0.06) / 0.88);

                // 2. Higher-frequency cellular cauliflower florets & entrainment clefts (base period = 8)
                let w_hi1 = 1.0 - periodic_worley3(u_x * 8.0, u_y * 8.0, u_z * 8.0, 8, seed + 73);
                let w_hi2 =
                    1.0 - periodic_worley3(u_x * 16.0, u_y * 16.0, u_z * 16.0, 16, seed + 137);
                let w_hi3 =
                    1.0 - periodic_worley3(u_x * 32.0, u_y * 32.0, u_z * 32.0, 32, seed + 193);
                let cell_raw = (0.58 * w_hi1 + 0.28 * w_hi2 + 0.14 * w_hi3).clamp(0.0, 1.0);
                let cell_detail = cell_raw.powf(1.18).clamp(0.0, 1.0);

                // 3. Curl-warped anisotropic silk filaments + subtle gravity-wave herringbone (3D cirrus & virga)
                let warp_x =
                    periodic_value_noise3(u_x * 4.0, u_y * 4.0, u_z * 4.0, 4, 4, 4, seed + 211)
                        - 0.5;
                let warp_y =
                    periodic_value_noise3(u_x * 4.0, u_y * 4.0, u_z * 4.0, 4, 4, 4, seed + 223)
                        - 0.5;
                let fx = (u_x + 0.16 * warp_x + 0.20 * (u_z * std::f32::consts::TAU).sin())
                    .rem_euclid(1.0);
                let fy = (u_y + 0.12 * warp_y).rem_euclid(1.0);
                let streak_a =
                    periodic_value_noise3(fx * 5.0, fy * 20.0, u_z * 10.0, 5, 20, 10, seed + 401);
                let streak_b = periodic_value_noise3(
                    fx * 10.0,
                    fy * 36.0,
                    u_z * 18.0,
                    10,
                    36,
                    18,
                    seed + 419,
                );
                // Ridge-sharpened filament strands so thin ice hairs stand out against clear sky
                let strand_ridge = 1.0 - (streak_b * 2.0 - 1.0).abs();
                let fibrous =
                    smoothstep01((0.58 * streak_a + 0.42 * strand_ridge - 0.14) / 0.76);

                // 4. Kolmogorov 1-meter wispy micro-turbulence + fine scud tendrils (base period = 16)
                let micro_fbm =
                    periodic_fbm3(u_x * 16.0, u_y * 16.0, u_z * 16.0, 16, 2, seed + 887);
                let micro_cell =
                    1.0 - periodic_worley3(u_x * 16.0, u_y * 16.0, u_z * 16.0, 16, seed + 941);
                let micro = (0.56 * micro_fbm + 0.44 * micro_cell).clamp(0.0, 1.0);

                let dst = ((z * N + y) * N + x) * 4;
                out[dst] = (perlin_worley * 255.0).round() as u8;
                out[dst + 1] = (cell_detail * 255.0).round() as u8;
                out[dst + 2] = (fibrous * 255.0).round() as u8;
                out[dst + 3] = (micro * 255.0).round() as u8;
            }
        }
    }
    out
}

/// Saturation vapor pressure over liquid water `e_s,liq(T)` (Pa) — Bolton (1980) / Tetens.
#[inline]
pub fn saturation_vapor_pressure_liquid_pa(temp_k: f32) -> f32 {
    let tc = (temp_k - 273.15).clamp(-85.0, 55.0);
    611.2 * ((17.67 * tc) / (tc + 243.5)).exp()
}

/// Saturation vapor pressure over hexagonal ice `e_s,ice(T)` (Pa) — Murphy & Koop (2005) / Magnus-Tetens ice.
#[inline]
pub fn saturation_vapor_pressure_ice_pa(temp_k: f32) -> f32 {
    let tc = (temp_k - 273.15).clamp(-95.0, 0.0);
    611.15 * ((22.46 * tc) / (tc + 272.62)).exp()
}

#[inline]
pub fn saturation_mixing_ratio(e_sat_pa: f32, pressure_pa: f32) -> f32 {
    let eps = RD / RV; // 0.62198
    let denom = (pressure_pa - e_sat_pa).max(500.0);
    (eps * e_sat_pa / denom).clamp(1.0e-6, 0.065)
}

/// Full 3D Coupled Moist Atmospheric Fluid Dynamics & Cloud Microphysics Solver.
pub struct FluidAtmosphere {
    pub params: AtmosphericParams,
    pub sim_time_s: f32,

    /// Stretched vertical altitudes `z_k` (meters ASL) and Exner/hydrostatic background profiles.
    pub z_levels_m: [f32; GRID_NZ],
    pub dz_levels_m: [f32; GRID_NZ],
    pub p_bar_pa: [f32; GRID_NZ],
    pub exner_bar: [f32; GRID_NZ],
    pub temp_bar_k: [f32; GRID_NZ],
    pub theta_bar_k: [f32; GRID_NZ],
    pub rho_bar: [f32; GRID_NZ],
    pub qvs_liq_bar: [f32; GRID_NZ],
    pub qvs_ice_bar: [f32; GRID_NZ],
    pub haze_alt_bar: [f32; GRID_NZ],
    pub brunt_vaisala_n: [f32; GRID_NZ],
    pub u_bar: [f32; GRID_NZ],
    pub v_bar: [f32; GRID_NZ],
    pub qv_env: [f32; GRID_NZ],

    /// 2D terrain boundary (`GRID_NX × GRID_NY`), gradients `dh/dx`, `dh/dy`, and normal `[nx, ny, nz]`.
    pub terrain_h: Vec<f32>,
    pub dh_dx: Vec<f32>,
    pub dh_dy: Vec<f32>,
    pub peak_relief_m: f32,

    /// 3D prognostic fluid variables (`GRID_CELLS = 64 × 64 × 48`):
    /// Perturbation velocity `(u', v', w)` in m/s (total horizontal velocity is `u_bar[k] + u'`).
    pub u_pert: Vec<f32>,
    pub v_pert: Vec<f32>,
    pub w_vel: Vec<f32>,
    /// Potential temperature anomaly `theta'` (K).
    pub theta_pert: Vec<f32>,
    /// Total non-ice water mixing ratio `q_t = q_v + q_c` (kg/kg) and cloud liquid `q_c` (kg/kg).
    pub qt: Vec<f32>,
    pub qc: Vec<f32>,
    /// Upper-tropospheric ice-crystal mixing ratio `q_i` (kg/kg) + ice-supersaturated vapor anomaly `qv_ice`.
    pub qi: Vec<f32>,
    /// 3D generating-cell / synoptic moisture source field that advects across the upper/lower domain.
    pub vorticity_mag: Vec<f32>,

    /// Scratch buffers for zero-allocation fused stepping (`step()` allocates 0 heap bytes!).
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
    scratch_c: Vec<f32>,
    scratch_d: Vec<f32>,
    scratch_e: Vec<f32>,
    scratch_f: Vec<f32>,
    scratch_div: Vec<f32>,
    scratch_p: Vec<f32>,
    solid_mask: Vec<u8>,

    /// Precomputed periodic synoptic moisture (`64 × 64 × 4`) and 3D cirrus source (`64 × 64 × 2`) LUTs
    /// so `step()` performs O(1) bilinear lookups instead of per-voxel Worley evaluations!
    synoptic_lut_seed: u32,
    synoptic_lut_q: Vec<f32>,
    synoptic_lut_ci_band: Vec<f32>,
    synoptic_lut_ci_head: Vec<f32>,

    /// Packed RGBA8 GPU 3D volume texture (`64 × 64 × 48 × 4 = 786,432` bytes).
    pub rgba_volume: Vec<u8>,
    /// Packed RGBA8 wind profile (`48 × 4` bytes) or float array `[u, v, z, shear]` per level.
    pub wind_profile_flat: Vec<f32>,
}

impl FluidAtmosphere {
    pub fn new(params: AtmosphericParams) -> Self {
        let mut solver = Self {
            params,
            sim_time_s: 0.0,
            z_levels_m: [0.0; GRID_NZ],
            dz_levels_m: [100.0; GRID_NZ],
            p_bar_pa: [P0_PA; GRID_NZ],
            exner_bar: [1.0; GRID_NZ],
            temp_bar_k: [288.15; GRID_NZ],
            theta_bar_k: [288.15; GRID_NZ],
            rho_bar: [1.225; GRID_NZ],
            qvs_liq_bar: [0.01; GRID_NZ],
            qvs_ice_bar: [0.001; GRID_NZ],
            haze_alt_bar: [1.0; GRID_NZ],
            brunt_vaisala_n: [0.011; GRID_NZ],
            u_bar: [0.0; GRID_NZ],
            v_bar: [0.0; GRID_NZ],
            qv_env: [0.0; GRID_NZ],
            terrain_h: vec![0.0; GRID_NX * GRID_NY],
            dh_dx: vec![0.0; GRID_NX * GRID_NY],
            dh_dy: vec![0.0; GRID_NX * GRID_NY],
            peak_relief_m: 300.0,
            u_pert: vec![0.0; GRID_CELLS],
            v_pert: vec![0.0; GRID_CELLS],
            w_vel: vec![0.0; GRID_CELLS],
            theta_pert: vec![0.0; GRID_CELLS],
            qt: vec![0.0; GRID_CELLS],
            qc: vec![0.0; GRID_CELLS],
            qi: vec![0.0; GRID_CELLS],
            vorticity_mag: vec![0.0; GRID_CELLS],
            scratch_a: vec![0.0; GRID_CELLS],
            scratch_b: vec![0.0; GRID_CELLS],
            scratch_c: vec![0.0; GRID_CELLS],
            scratch_d: vec![0.0; GRID_CELLS],
            scratch_e: vec![0.0; GRID_CELLS],
            scratch_f: vec![0.0; GRID_CELLS],
            scratch_div: vec![0.0; GRID_CELLS],
            scratch_p: vec![0.0; GRID_CELLS],
            solid_mask: vec![0; GRID_CELLS],
            synoptic_lut_seed: u32::MAX,
            synoptic_lut_q: vec![0.0; GRID_NX * GRID_NY * 4],
            synoptic_lut_ci_band: vec![0.0; GRID_NX * GRID_NY],
            synoptic_lut_ci_head: vec![0.0; GRID_NX * GRID_NY],
            rgba_volume: vec![0; GRID_CELLS * 4],
            wind_profile_flat: vec![0.0; GRID_NZ * 4],
        };
        solver.rebuild_background_sounding();
        solver.initialize_fields_and_spinup(8);
        solver
    }

    /// Resamples an arbitrary high-resolution DEM patch (`dem_w × dem_h` covering `patch_span_m` meters
    /// centered at `(0, 0)`, south-up row order matching `server.py` / `pp-terrain`) onto the
    /// `18 km × 18 km` fluid domain with smooth sponge tapering outside the DEM patch.
    pub fn set_terrain_from_dem_patch(
        &mut self,
        dem_elevations_m: &[f32],
        dem_w: usize,
        dem_h: usize,
        patch_width_m: f32,
        patch_height_m: f32,
    ) {
        if dem_elevations_m.is_empty() || dem_w < 2 || dem_h < 2 {
            return;
        }
        let dom = self.params.domain_xy_m.max(4000.0);
        let half_dom = 0.5 * dom;
        let half_pw = 0.5 * patch_width_m.max(500.0);
        let half_ph = 0.5 * patch_height_m.max(500.0);

        let mut min_h = f32::INFINITY;
        let mut max_h = f32::NEG_INFINITY;

        // Sample DEM with edge clamping + gentle outer sponge relaxation toward perimeter elevation
        // so wind entering the 18km fluid domain doesn't hit an artificial cliff at the 9.2km patch edge!
        let mut edge_sum = 0.0_f32;
        let mut edge_cnt = 0usize;
        for c in 0..dem_w {
            edge_sum += dem_elevations_m[c] + dem_elevations_m[(dem_h - 1) * dem_w + c];
            edge_cnt += 2;
        }
        let perim_mean = if edge_cnt > 0 {
            (edge_sum / edge_cnt as f32).max(0.0)
        } else {
            0.0
        };

        for j in 0..GRID_NY {
            // Fluid y: -half_dom (South) .. +half_dom (North)
            let wy = -half_dom + ((j as f32 + 0.5) / GRID_NY as f32) * dom;
            for i in 0..GRID_NX {
                let wx = -half_dom + ((i as f32 + 0.5) / GRID_NX as f32) * dom;

                let u_dem = (wx + half_pw) / (2.0 * half_pw);
                let v_dem = (wy + half_ph) / (2.0 * half_ph);

                let uc = u_dem.clamp(0.0, 1.0) * (dem_w - 1) as f32;
                let vc = v_dem.clamp(0.0, 1.0) * (dem_h - 1) as f32;
                let c0 = (uc.floor() as usize).min(dem_w - 2);
                let r0 = (vc.floor() as usize).min(dem_h - 2);
                let dc = uc - c0 as f32;
                let dr = vc - r0 as f32;

                let z00 = dem_elevations_m[r0 * dem_w + c0];
                let z10 = dem_elevations_m[r0 * dem_w + c0 + 1];
                let z01 = dem_elevations_m[(r0 + 1) * dem_w + c0];
                let z11 = dem_elevations_m[(r0 + 1) * dem_w + c0 + 1];
                let clamped_z =
                    (z00 * (1.0 - dc) + z10 * dc) * (1.0 - dr) + (z01 * (1.0 - dc) + z11 * dc) * dr;

                // Distance outside the high-res patch normalized to sponge width (~3,500m)
                let over_x = (wx.abs() - half_pw).max(0.0);
                let over_y = (wy.abs() - half_ph).max(0.0);
                let over_d = (over_x * over_x + over_y * over_y).sqrt();
                let blend = smoothstep01(over_d / 3800.0);
                // Preserve west ocean (`wx < -half_pw` when clamped_z <= 3m) at 0m!
                let outer_target = if clamped_z < 5.0 { 0.0 } else { perim_mean * 0.65 + clamped_z * 0.35 };
                let h = clamped_z * (1.0 - blend) + outer_target * blend;

                min_h = min_h.min(h);
                max_h = max_h.max(h);
                self.terrain_h[idx2(i, j)] = h.max(0.0);
            }
        }

        self.peak_relief_m = (max_h - min_h).max(80.0);
        // Shift z_base_m cleanly so inland mountain valleys (e.g. Yosemite at 1,200m ASL) keep
        // high vertical resolution right above the valley floor while coastal scenes start at 0m ASL!
        self.params.z_base_m = (min_h - 80.0).max(0.0);
        self.params.z_top_m = DEFAULT_Z_TOP_M.max(max_h + 7500.0);

        let dx = dom / GRID_NX as f32;
        let dy = dom / GRID_NY as f32;
        for j in 0..GRID_NY {
            let jm = j.saturating_sub(1);
            let jp = (j + 1).min(GRID_NY - 1);
            for i in 0..GRID_NX {
                let im = i.saturating_sub(1);
                let ip = (i + 1).min(GRID_NX - 1);
                let dhx = (self.terrain_h[idx2(ip, j)] - self.terrain_h[idx2(im, j)])
                    / ((ip - im).max(1) as f32 * dx);
                let dhy = (self.terrain_h[idx2(i, jp)] - self.terrain_h[idx2(i, jm)])
                    / ((jp - jm).max(1) as f32 * dy);
                // Clamp extreme sub-grid cliff slopes to realistic fluid envelope slope <= 0.85
                self.dh_dx[idx2(i, j)] = dhx.clamp(-0.85, 0.85);
                self.dh_dy[idx2(i, j)] = dhy.clamp(-0.85, 0.85);
            }
        }

        self.rebuild_background_sounding();
        self.rebuild_solid_mask();
        self.initialize_fields_and_spinup(6);
    }

    pub fn update_params(&mut self, mut next: AtmosphericParams, reseed_fields: bool) {
        // Preserve terrain-derived vertical domain base unless explicitly overridden
        if next.z_top_m < 6000.0 {
            next.z_top_m = self.params.z_top_m;
        }
        next.z_base_m = self.params.z_base_m;
        self.params = next;
        self.rebuild_background_sounding();
        self.rebuild_solid_mask();
        if reseed_fields {
            self.initialize_fields_and_spinup(8);
        } else {
            // Gently nudge moisture background toward updated forecast hour without popping clouds!
            self.relax_toward_sounding(0.32);
            self.pack_rgba_volume();
        }
    }

    fn rebuild_solid_mask(&mut self) {
        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            for j in 0..GRID_NY {
                for i in 0..GRID_NX {
                    let h = self.terrain_h[idx2(i, j)];
                    self.solid_mask[idx3(i, j, k)] = if z < h { 1 } else { 0 };
                }
            }
        }
    }

    /// Builds hydrostatic background profiles `p_bar(z)`, `theta_bar(z)`, `rho_bar(z)`, `N(z)`,
    /// sheared multi-pressure-level wind `U_0(z)`, and environmental moisture `qv_env(z)` from `0..12.2 km`.
    pub fn rebuild_background_sounding(&mut self) {
        self.ensure_synoptic_luts();
        let z_base = self.params.z_base_m;
        let z_top = self.params.z_top_m;

        for k in 0..GRID_NZ {
            let s = k as f32 / (GRID_NZ - 1) as f32;
            self.z_levels_m[k] = s_to_altitude_m(s, z_base, z_top);
        }
        for k in 0..GRID_NZ {
            let z0 = if k == 0 {
                self.z_levels_m[0]
            } else {
                self.z_levels_m[k - 1]
            };
            let z1 = if k + 1 == GRID_NZ {
                self.z_levels_m[GRID_NZ - 1]
            } else {
                self.z_levels_m[k + 1]
            };
            let span = if k == 0 || k + 1 == GRID_NZ {
                (z1 - z0).max(35.0)
            } else {
                0.5 * (z1 - z0)
            };
            self.dz_levels_m[k] = span.max(35.0);
        }

        let t_sfc_k = self.params.surface_temp_c + 273.15;
        let inv_base = self.params.inversion_base_m.max(z_base + 90.0);
        let inv_thick = self.params.inversion_thickness_m.max(100.0);
        let inv_jump = self.params.inversion_strength_k.clamp(0.0, 16.0);

        // Integrate hydrostatic pressure upward from sea-level reference
        let mut p_curr = P0_PA * (1.0 - 0.0065 * z_base / 288.15).max(0.2).powf(5.255);

        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            let agl_ref = (z - z_base).max(0.0);

            // Standard moist-tropospheric lapse (~6.2 K/km) + boundary-layer marine/valley inversion jump!
            let inv_frac = smoothstep01((z - inv_base) / inv_thick);
            // Tropopause stabilization above 10.8 km
            let tropo_warm = ((z - 10_800.0).max(0.0)) * 0.0045;
            let temp_k = (t_sfc_k - 0.0063 * agl_ref + inv_jump * 0.72 * inv_frac + tropo_warm)
                .clamp(202.0, 320.0);

            if k > 0 {
                let dz = self.z_levels_m[k] - self.z_levels_m[k - 1];
                let t_mid = 0.5 * (temp_k + self.temp_bar_k[k - 1]).max(205.0);
                p_curr *= (-GRAVITY * dz / (RD * t_mid)).exp();
            }
            self.temp_bar_k[k] = temp_k;
            self.p_bar_pa[k] = p_curr;
            let exner = (p_curr / P0_PA).powf(RD / CP).max(0.15);
            self.exner_bar[k] = exner;
            self.theta_bar_k[k] = temp_k / exner;
            self.rho_bar[k] = (p_curr / (RD * temp_k)).clamp(0.18, 1.35);
            self.haze_alt_bar[k] = (-agl_ref / 2800.0).exp();

            // Multi-level wind profile with Ekman spiral near surface + upper-tropospheric jet shear (250 hPa)
            let (u_z, v_z) = self.interpolate_wind_at_z(z);
            self.u_bar[k] = u_z;
            self.v_bar[k] = v_z;

            // Precompute exact background liquid & ice saturation mixing ratios `qvs_bar(z)` once per level!
            let e_sat_liq = saturation_vapor_pressure_liquid_pa(temp_k);
            let qvs_liq = saturation_mixing_ratio(e_sat_liq, p_curr);
            let e_sat_ice = saturation_vapor_pressure_ice_pa(temp_k);
            let qvs_ice = saturation_mixing_ratio(e_sat_ice, p_curr);
            self.qvs_liq_bar[k] = qvs_liq;
            self.qvs_ice_bar[k] = qvs_ice;

            let sfc_rh = (self.params.surface_rh_pct * 0.01).clamp(0.15, 0.99);
            let low_cov = (self.params.low_cloud_pct * 0.01).clamp(0.0, 1.0);
            let mid_cov = (self.params.mid_cloud_pct * 0.01).clamp(0.0, 1.0);

            // Estimate physical Lifting Condensation Level `z_LCL` so cumulus & stratocumulus decks
            // form crisp, flat horizontal condensation bases with towering billowing crowns above!
            let tc_sfc = self.params.surface_temp_c;
            let gamma_lcl = sfc_rh.ln() + (17.67 * tc_sfc) / (243.5 + tc_sfc);
            let td_sfc = (243.5 * gamma_lcl) / (17.67 - gamma_lcl);
            let lcl_est = z_base + (125.0 * (tc_sfc - td_sfc)).clamp(80.0, 2600.0);

            // 1. Sub-inversion marine / valley boundary layer traps dense stratus right beneath `inv_base`
            let mbl_cap = 1.0 - smoothstep01((z - (inv_base + 0.55 * inv_thick)) / (inv_thick * 0.95));
            let mbl_base_ramp = smoothstep01((z - (inv_base - 210.0).max(z_base + 25.0)) / 95.0);
            let mbl_peak = (-(z - (inv_base - 35.0)).powi(2) / (2.0 * 175.0 * 175.0)).exp();
            let rh_low = if low_cov > 0.04 {
                let sub_cloud = (sfc_rh * 0.62 + 0.18).clamp(0.22, 0.88);
                let deck_core = (sfc_rh * 0.54 + 0.37 + low_cov * (0.21 + 0.15 * mbl_peak)).clamp(0.30, 1.06);
                (sub_cloud + (deck_core - sub_cloud) * mbl_base_ramp) * mbl_cap
            } else {
                (sfc_rh * 0.78).clamp(0.20, 0.82) * mbl_cap
            };

            // 2. Asymmetric Flat-Base LCL -> Towering Cumulus / Congestus Crown Layer!
            // Sharp unsaturated-to-saturated transition across `cu_base`, sustained buoyant tower body,
            // and clean convective detrainment lid so clouds have real 3D architectural silhouettes!
            let cu_base = lcl_est
                .max(inv_base + 0.35 * inv_thick)
                .min(z_base + self.peak_relief_m * 0.68 + 420.0);
            let cape_norm = (self.params.convective_cape / 1000.0).clamp(0.0, 1.35);
            let cu_depth = (850.0 + 920.0 * cape_norm).clamp(800.0, 2200.0);
            let above_cu_base = smoothstep01((z - (cu_base - 45.0)) / 115.0);
            let below_cu_top = 1.0 - smoothstep01((z - (cu_base + cu_depth * 0.68)) / (cu_depth * 0.52));
            let crown_boost = (-(z - (cu_base + cu_depth * 0.28)).powi(2)
                / (2.0 * (cu_depth * 0.42).powi(2)))
            .exp();
            let cu_profile = above_cu_base * below_cu_top * (0.84 + 0.16 * crown_boost);
            let rh_cu = if low_cov > 0.05 || cape_norm > 0.12 {
                (0.79 + 0.22 * low_cov + 0.06 * cape_norm).clamp(0.0, 1.02) * cu_profile
            } else {
                0.38 * cu_profile
            };

            // 3. Mid-level altocumulus / standing mountain-wave lenticular band (~3.5..5.5 km ASL)
            let mid_center = (z_base + self.peak_relief_m + 920.0).clamp(3500.0, 5400.0);
            let mid_bell = (-(z - mid_center).powi(2) / (2.0 * 740.0 * 740.0)).exp();
            let rh_mid = if mid_cov > 0.05 {
                (0.82 + 0.20 * mid_cov).clamp(0.0, 1.00) * mid_bell
            } else {
                0.30 * mid_bell
            };

            let baseline_rh = (0.28 * (-agl_ref / 5500.0).exp()).max(0.12);
            let eff_rh = rh_low.max(rh_cu).max(rh_mid).max(baseline_rh).clamp(0.10, 1.05);
            self.qv_env[k] = eff_rh * qvs_liq;
        }

        // Compute Brunt-Vaisala buoyancy frequency `N(z) = sqrt((g / theta) * dtheta/dz)`
        for k in 0..GRID_NZ {
            let km = k.saturating_sub(1);
            let kp = (k + 1).min(GRID_NZ - 1);
            let dz = (self.z_levels_m[kp] - self.z_levels_m[km]).max(40.0);
            let dtheta_dz = ((self.theta_bar_k[kp] - self.theta_bar_k[km]) / dz).max(1.0e-4);
            self.brunt_vaisala_n[k] = ((GRAVITY / self.theta_bar_k[k]) * dtheta_dz)
                .sqrt()
                .clamp(0.004, 0.038);

            let du = (self.u_bar[kp] - self.u_bar[km]) / dz;
            let dv = (self.v_bar[kp] - self.v_bar[km]) / dz;
            let shear = (du * du + dv * dv).sqrt();
            self.wind_profile_flat[k * 4] = self.u_bar[k];
            self.wind_profile_flat[k * 4 + 1] = self.v_bar[k];
            self.wind_profile_flat[k * 4 + 2] = self.z_levels_m[k];
            self.wind_profile_flat[k * 4 + 3] = shear;
        }
    }

    #[inline]
    fn interpolate_wind_at_z(&self, z_m: f32) -> (f32, f32) {
        let z_sfc = self.params.z_base_m + 15.0;
        let z_850 = (self.params.z_base_m + 1480.0).max(1480.0);
        let z_500 = 5600.0_f32;
        let z_250 = 10_400.0_f32;

        if z_m <= z_850 {
            let t = smoothstep01((z_m - z_sfc) / (z_850 - z_sfc).max(200.0));
            (
                self.params.wind_sfc_u + t * (self.params.wind_850_u - self.params.wind_sfc_u),
                self.params.wind_sfc_v + t * (self.params.wind_850_v - self.params.wind_sfc_v),
            )
        } else if z_m <= z_500 {
            let t = smoothstep01((z_m - z_850) / (z_500 - z_850));
            (
                self.params.wind_850_u + t * (self.params.wind_500_u - self.params.wind_850_u),
                self.params.wind_850_v + t * (self.params.wind_500_v - self.params.wind_850_v),
            )
        } else {
            let t = ((z_m - z_500) / (z_250 - z_500)).clamp(0.0, 1.25);
            // Jet core peaks near 10.4 km and tapers slightly approaching 12.2 km tropopause
            let jet_shape = if z_m <= z_250 {
                smoothstep01(t)
            } else {
                1.0 - 0.18 * ((z_m - z_250) / 2000.0).clamp(0.0, 1.0)
            };
            (
                self.params.wind_500_u
                    + jet_shape * (self.params.wind_250_u - self.params.wind_500_u),
                self.params.wind_500_v
                    + jet_shape * (self.params.wind_250_v - self.params.wind_500_v),
            )
        }
    }

    fn ensure_synoptic_luts(&mut self) {
        if self.synoptic_lut_seed == self.params.seed {
            return;
        }
        let seed = self.params.seed;
        self.synoptic_lut_seed = seed;

        let mut q_min = f32::INFINITY;
        let mut q_max = f32::NEG_INFINITY;
        for l in 0..4usize {
            let sz = l as f32 / 3.0;
            for j in 0..GRID_NY {
                let sy = j as f32 / GRID_NY as f32;
                for i in 0..GRID_NX {
                    let sx = i as f32 / GRID_NX as f32;
                    // Distinct mesoscale cloud clusters + cellular cumulus streets separated by true clear sky!
                    let cluster = periodic_fbm3(sx * 2.0, sy * 2.0, sz * 3.0, 2, 3, seed);
                    let billow = periodic_worley_fbm3(sx * 4.0, sy * 4.0, sz * 5.0, 4, seed + 53);
                    let turret = periodic_worley_fbm3(sx * 7.0, sy * 7.0, sz * 8.0, 7, seed + 149);
                    let raw = 0.48 * cluster + 0.34 * billow + 0.18 * turret;
                    q_min = q_min.min(raw);
                    q_max = q_max.max(raw);
                    self.synoptic_lut_q[(l * GRID_NY + j) * GRID_NX + i] = raw;
                }
            }
        }
        let q_span = (q_max - q_min).max(1.0e-4);
        for v in &mut self.synoptic_lut_q {
            // S-curve contrast separates dense 3D cloud cores (+1.0) cleanly from dry blue-sky lanes (-1.0)
            let norm01 = ((*v - q_min) / q_span).clamp(0.0, 1.0);
            let shaped = smoothstep01(norm01);
            *v = shaped * 2.04 - 1.02;
        }

        // Sculpted jet-aligned cirrus plumes + cellular uncinus generating tufts so wide twilight/cobalt
        // sky channels frame distinct 3D cirrus formations instead of smearing across 90% of the sky!
        let mut cb_min = f32::INFINITY;
        let mut cb_max = f32::NEG_INFINITY;
        let mut ch_min = f32::INFINITY;
        let mut ch_max = f32::NEG_INFINITY;
        for j in 0..GRID_NY {
            let cross = j as f32 / GRID_NY as f32;
            for i in 0..GRID_NX {
                let along = i as f32 / GRID_NX as f32;
                let streak = periodic_fbm3(along * 2.0, cross * 5.0, 0.78 * 5.0, 5, 3, seed + 701);
                let lane_mask = periodic_fbm3(along * 1.0, cross * 3.0, 0.35 * 3.0, 3, 2, seed + 811);
                // Subtle cross-jet gravity-wave vertebratus ribbing
                let rib = 0.5 + 0.5 * (along * 6.0 * std::f32::consts::TAU).cos();
                let b_val = streak * 0.54 + lane_mask * 0.38 + rib * 0.08;
                let h_val = periodic_worley_fbm3(along * 4.0, cross * 7.0, 0.85 * 7.0, 7, seed + 733);
                cb_min = cb_min.min(b_val);
                cb_max = cb_max.max(b_val);
                ch_min = ch_min.min(h_val);
                ch_max = ch_max.max(h_val);
                self.synoptic_lut_ci_band[j * GRID_NX + i] = b_val;
                self.synoptic_lut_ci_head[j * GRID_NX + i] = h_val;
            }
        }
        let cb_span = (cb_max - cb_min).max(1.0e-4);
        let ch_span = (ch_max - ch_min).max(1.0e-4);
        for v in &mut self.synoptic_lut_ci_band {
            let n = ((*v - cb_min) / cb_span).clamp(0.0, 1.0);
            *v = smoothstep01((n - 0.12) / 0.82);
        }
        for v in &mut self.synoptic_lut_ci_head {
            let n = ((*v - ch_min) / ch_span).clamp(0.0, 1.0);
            *v = n * n * (3.0 - 2.0 * n);
        }
    }

    #[inline]
    fn sample_lut2d_periodic(lut: &[f32], ux: f32, uy: f32) -> f32 {
        let fx = ux.rem_euclid(1.0) * GRID_NX as f32;
        let fy = uy.rem_euclid(1.0) * GRID_NY as f32;
        let i0 = fx.floor() as usize % GRID_NX;
        let i1 = (i0 + 1) % GRID_NX;
        let j0 = fy.floor() as usize % GRID_NY;
        let j1 = (j0 + 1) % GRID_NY;
        let tx = fx - fx.floor();
        let ty = fy - fy.floor();
        let c00 = lut[j0 * GRID_NX + i0];
        let c10 = lut[j0 * GRID_NX + i1];
        let c01 = lut[j1 * GRID_NX + i0];
        let c11 = lut[j1 * GRID_NX + i1];
        (c00 + tx * (c10 - c00)) * (1.0 - ty) + (c01 + tx * (c11 - c01)) * ty
    }

    /// Evaluates coherent synoptic 3D moisture & cirrus-generating-cell perturbations at `(x_norm, y_norm, z_m, time_s)`.
    #[inline]
    fn sample_synoptic_moisture_perturbation(
        &self,
        ux: f32,
        uy: f32,
        z_m: f32,
        k: usize,
        time_s: f32,
    ) -> (f32, f32) {
        let dom = self.params.domain_xy_m.max(4000.0);
        let shift_x = (self.u_bar[k] * time_s) / dom;
        let shift_y = (self.v_bar[k] * time_s) / dom;
        let sx = (ux - shift_x).rem_euclid(1.0);
        let sy = (uy - shift_y).rem_euclid(1.0);

        let sz_idx = ((z_m / 12_000.0).clamp(0.0, 1.0) * 3.0).min(2.999);
        let l0 = sz_idx.floor() as usize;
        let tl = sz_idx - l0 as f32;
        let slice_len = GRID_NX * GRID_NY;
        let q0 = Self::sample_lut2d_periodic(&self.synoptic_lut_q[l0 * slice_len..(l0 + 1) * slice_len], sx, sy);
        let q1 = Self::sample_lut2d_periodic(&self.synoptic_lut_q[(l0 + 1) * slice_len..(l0 + 2) * slice_len], sx, sy);
        let q_anom_frac = q0 + tl * (q1 - q0);

        let high_cov = (self.params.high_cloud_pct * 0.01).clamp(0.0, 1.0);
        let mut cirrus_src = 0.0_f32;
        if z_m >= 8200.0 && high_cov > 0.02 {
            let jet_u = self.params.wind_250_u;
            let jet_v = self.params.wind_250_v;
            let jet_spd = (jet_u * jet_u + jet_v * jet_v).sqrt().max(1.0);
            let jx = jet_u / jet_spd;
            let jy = jet_v / jet_spd;
            let along = sx * jx + sy * jy;
            let cross = -sx * jy + sy * jx;

            let band = Self::sample_lut2d_periodic(&self.synoptic_lut_ci_band, along, cross);
            let uncinus_head = Self::sample_lut2d_periodic(&self.synoptic_lut_ci_head, along, cross);
            // Fast rational bell curves centered at 10.35 km (generating tufts) and 9.4 km (fibratus)
            let dh = (z_m - 10_350.0) * (1.0 / 850.0);
            let alt_head = 1.0 / (1.0 + dh * dh);
            let ds = (z_m - 9_450.0) * (1.0 / 1100.0);
            let alt_sheet = 1.0 / (1.0 + ds * ds);

            // Threshold guarantees discrete cirrus bands & mares' tails with clear blue/twilight sky lanes!
            let thresh = (0.68 - 0.28 * high_cov).clamp(0.40, 0.72);
            let raw_ci = 0.52 * band * alt_sheet + 0.62 * uncinus_head * alt_head;
            if raw_ci > thresh {
                cirrus_src = ((raw_ci - thresh) / (1.0 - thresh).max(0.20)).clamp(0.0, 1.0)
                    * (0.38e-3 * self.params.cirrus_ice_boost.clamp(0.1, 2.2))
                    * (0.40 + 0.60 * high_cov);
            }
        }

        (q_anom_frac, cirrus_src)
    }

    /// Initializes the 3D velocity, potential temperature, liquid water, and upper-tropospheric 3D cirrus
    /// ice fields, then runs `spinup_steps` coupled Navier-Stokes + microphysics steps so wind-sheared
    /// cirrus uncinus fallstreaks and orographic wave clouds are physically established immediately.
    pub fn initialize_fields_and_spinup(&mut self, spinup_steps: usize) {
        self.sim_time_s = 0.0;
        self.rebuild_solid_mask();

        let low_cov = (self.params.low_cloud_pct * 0.01).clamp(0.0, 1.0);
        let mid_cov = (self.params.mid_cloud_pct * 0.01).clamp(0.0, 1.0);

        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            let qv0 = self.qv_env[k];
            for j in 0..GRID_NY {
                let uy = (j as f32 + 0.5) / GRID_NY as f32;
                for i in 0..GRID_NX {
                    let ux = (i as f32 + 0.5) / GRID_NX as f32;
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.u_pert[c] = 0.0;
                        self.v_pert[c] = 0.0;
                        self.w_vel[c] = 0.0;
                        self.theta_pert[c] = 0.0;
                        self.qt[c] = qv0 * 0.7;
                        self.qc[c] = 0.0;
                        self.qi[c] = 0.0;
                        continue;
                    }

                    let (q_anom, ci_src) =
                        self.sample_synoptic_moisture_perturbation(ux, uy, z, k, 0.0);
                    let amp = 0.16 + 0.12 * low_cov.max(mid_cov);
                    self.u_pert[c] = 0.0;
                    self.v_pert[c] = 0.0;
                    self.w_vel[c] = 0.0;
                    self.theta_pert[c] = -q_anom * 0.25;
                    self.qt[c] = (qv0 * (1.0 + amp * q_anom)).max(1.0e-6);
                    self.qc[c] = 0.0;
                    self.qi[c] = ci_src;
                }
            }
        }

        // Run physical fluid + microphysics spinup steps (e.g. 8 × 14s = ~112s of atmospheric adjustment
        // so falling ice crystals shear along the 250–350 hPa jet stream into curved 3D mares' tails!)
        let n = spinup_steps.max(1);
        for s in 0..n {
            self.step_internal(14.0, s + 1 == n);
        }
    }

    fn relax_toward_sounding(&mut self, alpha: f32) {
        let a = alpha.clamp(0.02, 1.0);
        let low_cov = (self.params.low_cloud_pct * 0.01).clamp(0.0, 1.0);
        let mid_cov = (self.params.mid_cloud_pct * 0.01).clamp(0.0, 1.0);
        let amp = 0.16 + 0.12 * low_cov.max(mid_cov);

        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            let qv0 = self.qv_env[k];
            for j in 0..GRID_NY {
                let uy = (j as f32 + 0.5) / GRID_NY as f32;
                for i in 0..GRID_NX {
                    let ux = (i as f32 + 0.5) / GRID_NX as f32;
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        continue;
                    }
                    let (q_anom, ci_src) =
                        self.sample_synoptic_moisture_perturbation(ux, uy, z, k, self.sim_time_s);
                    let target_qt = (qv0 * (1.0 + amp * q_anom)).max(1.0e-6);
                    self.qt[c] = self.qt[c] * (1.0 - a) + target_qt * a;
                    if z >= 6800.0 {
                        self.qi[c] = (self.qi[c] * (1.0 - a * 0.7) + ci_src * a * 0.7).max(0.0);
                    }
                }
            }
        }
        self.apply_saturation_adjustment_and_pack(4.0, false);
    }

    /// Advances the 3D Moist Navier-Stokes + Cloud Liquid/Ice Microphysics simulation by `dt_s` seconds
    /// and repacks `self.rgba_volume` (`786 KB` `RGBA8` buffer ready for `Data3DTexture`).
    pub fn step(&mut self, dt_s: f32) {
        let dt = dt_s.clamp(0.05, 18.0);
        self.step_internal(dt, true);
    }

    fn step_internal(&mut self, dt: f32, pack_output: bool) {
        self.sim_time_s += dt;

        // 1. Orographic Kinematic Uplift + Scorer Mountain Gravity Waves + Solar Slope Thermals + Buoyancy
        self.apply_orographic_and_buoyancy_forcing(dt);

        // 2. 3D Vorticity Confinement (`f_vc = eps * dx * (eta x omega)`) for billowing turrets & KH shear
        if self.params.vorticity_confinement > 0.01 {
            self.apply_vorticity_confinement(dt);
        }

        // 3. Terrain-Masked Anelastic Divergence-Free Pressure Projection (`div(rho_bar * u) = 0`)
        self.project_anelastic_pressure();

        // 4. Fused Semi-Lagrangian 3D Advection of `(u', v', w, theta', qt, qc)` + Wind-Sheared Ice Sedimentation (`qi`)
        self.advect_all_fields(dt);

        // 5. Fused Clausius-Clapeyron Liquid Condensation + Koop 3D Cirrus Ice Microphysics + RGBA8 Packing
        self.apply_saturation_adjustment_and_pack(dt, pack_output);
    }

    fn apply_orographic_and_buoyancy_forcing(&mut self, dt: f32) {
        let sun_el_rad = self.params.solar_elevation_deg.to_radians();
        let sun_az_rad = self.params.solar_azimuth_deg.to_radians();
        let sun_east = sun_az_rad.sin() * sun_el_rad.cos();
        let sun_north = sun_az_rad.cos() * sun_el_rad.cos();
        let sun_up = sun_el_rad.sin();
        let daytime_heat = sun_up.max(0.0);
        let cape_factor = (self.params.convective_cape / 800.0).clamp(0.0, 2.2);
        let wave_gain = self.params.orographic_wave_gain.clamp(0.0, 2.5);
        let inv_base = self.params.inversion_base_m;

        // Precompute 2D surface solar incidence + local summit prominence `(4,096 cells)` once!
        // Prominent peaks & sunlit ridge crests anchor persistent 3D orographic cumulus & banner clouds.
        let mean_h = {
            let sum_h: f32 = self.terrain_h.iter().copied().sum();
            sum_h / (GRID_NX * GRID_NY) as f32
        };
        let inv_relief = 1.0 / self.peak_relief_m.max(120.0);
        let mut sfc_thermal_col = [0.0_f32; GRID_NX * GRID_NY];
        let mut ridge_root_col = [0.0_f32; GRID_NX * GRID_NY];
        for ij in 0..(GRID_NX * GRID_NY) {
            let h = self.terrain_h[ij];
            if h > 15.0 {
                let dhx = self.dh_dx[ij];
                let dhy = self.dh_dy[ij];
                let inv_norm = 1.0 / (dhx * dhx + dhy * dhy + 1.0).sqrt();
                let cos_inc = ((-dhx * sun_east - dhy * sun_north + sun_up) * inv_norm).max(0.0);
                let prom = ((h - mean_h).max(0.0) * inv_relief).clamp(0.0, 1.0);
                ridge_root_col[ij] = prom * prom;
                if sun_up > 0.01 {
                    sfc_thermal_col[ij] =
                        (0.016 + 0.032 * cape_factor) * (0.35 + 0.65 * cos_inc) * (0.55 + 0.85 * prom)
                            * (0.35 + 0.65 * daytime_heat);
                } else if sun_up < -0.04 {
                    sfc_thermal_col[ij] = -0.006 * (1.0 - 0.5 * prom);
                }
            }
        }

        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            let u0 = self.u_bar[k];
            let v0 = self.v_bar[k];
            let n_bv = self.brunt_vaisala_n[k];
            let inv_theta0 = 1.0 / self.theta_bar_k[k];
            let qv0 = self.qv_env[k];
            let spd0 = (u0 * u0 + v0 * v0).sqrt().max(2.0);
            // Vertical Scorer wavenumber `m_z = sqrt(max(0, N^2 / U^2 - k_h^2))`
            let kh = 0.0009_f32; // characteristic ridge horizontal wavenumber (~7 km wavelength)
            let scorer_m2 = (n_bv * n_bv) / (spd0 * spd0) - kh * kh;
            let scorer_m = scorer_m2.max(0.0).sqrt();
            let inv_decay_h = if z < inv_base + 250.0 {
                1.0 / 1100.0
            } else {
                1.0 / 3400.0
            };
            let lid_damp = 1.0 - smoothstep01((z - (self.params.z_top_m - 900.0)) / 900.0);

            for j in 0..GRID_NY {
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.u_pert[c] = 0.0;
                        self.v_pert[c] = 0.0;
                        self.w_vel[c] = 0.0;
                        continue;
                    }

                    let ij = idx2(i, j);
                    let h = self.terrain_h[ij];
                    let agl = (z - h).max(0.0);
                    let dhx = self.dh_dx[ij];
                    let dhy = self.dh_dy[ij];

                    let u_tot = u0 + self.u_pert[c];
                    let v_tot = v0 + self.v_pert[c];

                    // Kinematic terrain slope forcing `w_oro = u * dh/dx + v * dh/dy`
                    let w_slope = u_tot * dhx + v_tot * dhy;

                    // Standing mountain lee waves (`cos(m_z * agl)`) when `scorer_m > 0`, with zero-`exp` Padé decay!
                    let x_dec = agl * inv_decay_h;
                    let env_decay = 1.0 / (1.0 + x_dec + 0.48 * x_dec * x_dec);
                    let wave_phase = if scorer_m > 1.0e-4 {
                        (scorer_m * agl * 0.72).cos()
                    } else {
                        let xe = agl * (1.0 / 900.0);
                        1.0 / (1.0 + xe + 0.48 * xe * xe)
                    };
                    let w_target_oro = wave_gain * w_slope * env_decay * wave_phase
                        + (0.38 + 0.45 * cape_factor) * ridge_root_col[ij] * env_decay;

                    let mut thermal_theta_rate = 0.0_f32;
                    if agl < 950.0 {
                        let xs = agl * (1.0 / 320.0);
                        let sfc_weight = 1.0 / (1.0 + xs + 0.48 * xs * xs);
                        thermal_theta_rate = sfc_thermal_col[ij] * sfc_weight;
                    }

                    self.theta_pert[c] =
                        (self.theta_pert[c] * 0.992 + dt * thermal_theta_rate).clamp(-6.0, 6.0);

                    // Moist Boussinesq buoyancy `b = g * (theta'/theta0 + 0.608*qv' - qc - qi)`
                    let qv = (self.qt[c] - self.qc[c]).max(0.0);
                    let qv_prime = qv - qv0;
                    let buoyancy = GRAVITY
                        * (self.theta_pert[c] * inv_theta0 + 0.608 * qv_prime
                            - (self.qc[c] + self.qi[c]));

                    let xr = agl * (1.0 / 1800.0);
                    let oro_relax = (0.40 / (1.0 + xr + 0.48 * xr * xr)).clamp(0.03, 0.40);
                    let w_next = (self.w_vel[c] + dt * buoyancy * 0.35) * (1.0 - oro_relax)
                        + w_target_oro * oro_relax;

                    self.w_vel[c] = (w_next * lid_damp).clamp(-14.0, 14.0);
                }
            }
        }
    }

    fn apply_vorticity_confinement(&mut self, dt: f32) {
        let dom = self.params.domain_xy_m.max(4000.0);
        let dx = dom / GRID_NX as f32;
        let inv_2dx = 0.5 / dx;

        // Compute vorticity magnitude `|omega| = |curl(u)|` into `self.vorticity_mag`
        for k in 1..(GRID_NZ - 1) {
            let inv_2dz = 1.0 / (self.z_levels_m[k + 1] - self.z_levels_m[k - 1]).max(60.0);
            for j in 0..GRID_NY {
                let jm = (j + GRID_NY - 1) % GRID_NY;
                let jp = (j + 1) % GRID_NY;
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.vorticity_mag[c] = 0.0;
                        continue;
                    }
                    let im = (i + GRID_NX - 1) % GRID_NX;
                    let ip = (i + 1) % GRID_NX;

                    let dw_dy = (self.w_vel[idx3(i, jp, k)] - self.w_vel[idx3(i, jm, k)]) * inv_2dx;
                    let dv_dz = ((self.v_bar[k + 1] + self.v_pert[idx3(i, j, k + 1)])
                        - (self.v_bar[k - 1] + self.v_pert[idx3(i, j, k - 1)]))
                        * inv_2dz;

                    let du_dz = ((self.u_bar[k + 1] + self.u_pert[idx3(i, j, k + 1)])
                        - (self.u_bar[k - 1] + self.u_pert[idx3(i, j, k - 1)]))
                        * inv_2dz;
                    let dw_dx = (self.w_vel[idx3(ip, j, k)] - self.w_vel[idx3(im, j, k)]) * inv_2dx;

                    let dv_dx =
                        (self.v_pert[idx3(ip, j, k)] - self.v_pert[idx3(im, j, k)]) * inv_2dx;
                    let du_dy =
                        (self.u_pert[idx3(i, jp, k)] - self.u_pert[idx3(i, jm, k)]) * inv_2dx;

                    let ox = dw_dy - dv_dz;
                    let oy = du_dz - dw_dx;
                    let oz = dv_dx - du_dy;
                    self.vorticity_mag[c] = (ox * ox + oy * oy + oz * oz).sqrt();
                }
            }
        }

        let eps_vc = 0.14 * self.params.vorticity_confinement.clamp(0.0, 2.5);
        for k in 2..(GRID_NZ - 2) {
            let inv_2dz = 1.0 / (self.z_levels_m[k + 1] - self.z_levels_m[k - 1]).max(60.0);
            for j in 0..GRID_NY {
                let jm = (j + GRID_NY - 1) % GRID_NY;
                let jp = (j + 1) % GRID_NY;
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        continue;
                    }
                    let im = (i + GRID_NX - 1) % GRID_NX;
                    let ip = (i + 1) % GRID_NX;

                    let gx = (self.vorticity_mag[idx3(ip, j, k)]
                        - self.vorticity_mag[idx3(im, j, k)])
                        * inv_2dx;
                    let gy = (self.vorticity_mag[idx3(i, jp, k)]
                        - self.vorticity_mag[idx3(i, jm, k)])
                        * inv_2dx;
                    let gz = (self.vorticity_mag[idx3(i, j, k + 1)]
                        - self.vorticity_mag[idx3(i, j, k - 1)])
                        * inv_2dz;
                    let glen = (gx * gx + gy * gy + gz * gz).sqrt() + 1.0e-5;
                    let nx = gx / glen;
                    let ny = gy / glen;
                    let nz = gz / glen;

                    // Local curl vector again (avoids storing 3 extra 3D scratch arrays!)
                    let ox = (self.w_vel[idx3(i, jp, k)] - self.w_vel[idx3(i, jm, k)]) * inv_2dx
                        - (self.v_pert[idx3(i, j, k + 1)] - self.v_pert[idx3(i, j, k - 1)])
                            * inv_2dz;
                    let oy = (self.u_pert[idx3(i, j, k + 1)] - self.u_pert[idx3(i, j, k - 1)])
                        * inv_2dz
                        - (self.w_vel[idx3(ip, j, k)] - self.w_vel[idx3(im, j, k)]) * inv_2dx;
                    let oz = (self.v_pert[idx3(ip, j, k)] - self.v_pert[idx3(im, j, k)]) * inv_2dx
                        - (self.u_pert[idx3(i, jp, k)] - self.u_pert[idx3(i, jm, k)]) * inv_2dx;

                    let scale = eps_vc * dx * dt;
                    self.u_pert[c] = (self.u_pert[c] + scale * (ny * oz - nz * oy)).clamp(-18.0, 18.0);
                    self.v_pert[c] = (self.v_pert[c] + scale * (nz * ox - nx * oz)).clamp(-18.0, 18.0);
                    self.w_vel[c] =
                        (self.w_vel[c] + 0.45 * scale * (nx * oy - ny * ox)).clamp(-14.0, 14.0);
                }
            }
        }
    }

    /// Terrain-masked 3D Red-Black SOR Pressure Projection (`div(u_pert, v_pert, w_anom) -> 0`)
    /// so wind splits laterally around peaks (Half Dome / Montara) and converges/diverges physically.
    fn project_anelastic_pressure(&mut self) {
        let dom = self.params.domain_xy_m.max(4000.0);
        let dx = dom / GRID_NX as f32;
        let inv_2dx = 0.5 / dx;

        for k in 1..(GRID_NZ - 1) {
            let dz = self.dz_levels_m[k];
            let inv_2dz = 0.5 / dz;
            let rho_k = self.rho_bar[k];
            let rho_up = self.rho_bar[k + 1];
            let rho_dn = self.rho_bar[k - 1];

            for j in 0..GRID_NY {
                let jm = (j + GRID_NY - 1) % GRID_NY;
                let jp = (j + 1) % GRID_NY;
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.scratch_div[c] = 0.0;
                        self.scratch_p[c] = 0.0;
                        continue;
                    }
                    let im = (i + GRID_NX - 1) % GRID_NX;
                    let ip = (i + 1) % GRID_NX;

                    let du_dx =
                        (self.u_pert[idx3(ip, j, k)] - self.u_pert[idx3(im, j, k)]) * inv_2dx;
                    let dv_dy =
                        (self.v_pert[idx3(i, jp, k)] - self.v_pert[idx3(i, jm, k)]) * inv_2dx;
                    let dw_dz = (rho_up * self.w_vel[idx3(i, j, k + 1)]
                        - rho_dn * self.w_vel[idx3(i, j, k - 1)])
                        * inv_2dz
                        / rho_k;
                    self.scratch_div[c] = du_dx + dv_dy + dw_dz;
                    // Warm-start pressure from previous step damped slightly for faster convergence!
                    self.scratch_p[c] *= 0.65;
                }
            }
        }

        // 5 Red-Black Gauss-Seidel SOR sweeps (`omega = 1.55`) accounting for anisotropic `dx` vs `dz`
        let dx2 = dx * dx;
        let omega = 1.55_f32;
        for _iter in 0..5 {
            for color in 0..2usize {
                for k in 1..(GRID_NZ - 1) {
                    let dz = self.dz_levels_m[k];
                    let ratio_z = (dx2 / (dz * dz)).clamp(0.25, 16.0);
                    let denom = 4.0 + 2.0 * ratio_z;
                    for j in 0..GRID_NY {
                        let jm = (j + GRID_NY - 1) % GRID_NY;
                        let jp = (j + 1) % GRID_NY;
                        for i in 0..GRID_NX {
                            if (i + j + k) & 1 != color {
                                continue;
                            }
                            let c = idx3(i, j, k);
                            if self.solid_mask[c] != 0 {
                                continue;
                            }
                            let im = (i + GRID_NX - 1) % GRID_NX;
                            let ip = (i + 1) % GRID_NX;

                            let pxm = self.scratch_p[idx3(im, j, k)];
                            let pxp = self.scratch_p[idx3(ip, j, k)];
                            let pym = self.scratch_p[idx3(i, jm, k)];
                            let pyp = self.scratch_p[idx3(i, jp, k)];
                            let pzm = self.scratch_p[idx3(i, j, k - 1)];
                            let pzp = self.scratch_p[idx3(i, j, k + 1)];

                            let p_gs = (pxm + pxp + pym + pyp + ratio_z * (pzm + pzp)
                                - dx2 * self.scratch_div[c])
                                / denom;
                            self.scratch_p[c] += omega * (p_gs - self.scratch_p[c]);
                        }
                    }
                }
            }
        }

        // Subtract pressure gradient from velocity perturbations
        for k in 1..(GRID_NZ - 1) {
            let inv_2dz = 0.5 / self.dz_levels_m[k];
            for j in 0..GRID_NY {
                let jm = (j + GRID_NY - 1) % GRID_NY;
                let jp = (j + 1) % GRID_NY;
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        continue;
                    }
                    let im = (i + GRID_NX - 1) % GRID_NX;
                    let ip = (i + 1) % GRID_NX;
                    let dpx = if self.solid_mask[idx3(ip, j, k)] == 0
                        && self.solid_mask[idx3(im, j, k)] == 0
                    {
                        (self.scratch_p[idx3(ip, j, k)] - self.scratch_p[idx3(im, j, k)]) * inv_2dx
                    } else {
                        0.0
                    };
                    let dpy = if self.solid_mask[idx3(i, jp, k)] == 0
                        && self.solid_mask[idx3(i, jm, k)] == 0
                    {
                        (self.scratch_p[idx3(i, jp, k)] - self.scratch_p[idx3(i, jm, k)]) * inv_2dx
                    } else {
                        0.0
                    };
                    let dpz = if self.solid_mask[idx3(i, j, k - 1)] == 0 {
                        (self.scratch_p[idx3(i, j, k + 1)] - self.scratch_p[idx3(i, j, k - 1)])
                            * inv_2dz
                    } else {
                        0.0
                    };

                    self.u_pert[c] = (self.u_pert[c] - 0.65 * dpx).clamp(-18.0, 18.0);
                    self.v_pert[c] = (self.v_pert[c] - 0.65 * dpy).clamp(-18.0, 18.0);
                    self.w_vel[c] = (self.w_vel[c] - 0.45 * dpz).clamp(-14.0, 14.0);
                }
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn bilerp8(
        field: &[f32],
        c000: usize,
        c100: usize,
        c010: usize,
        c110: usize,
        c001: usize,
        c101: usize,
        c011: usize,
        c111: usize,
        tx: f32,
        ty: f32,
        tz: f32,
    ) -> f32 {
        let v00 = field[c000] + tx * (field[c100] - field[c000]);
        let v10 = field[c010] + tx * (field[c110] - field[c010]);
        let v01 = field[c001] + tx * (field[c101] - field[c001]);
        let v11 = field[c011] + tx * (field[c111] - field[c011]);
        let v0 = v00 + ty * (v10 - v00);
        let v1 = v01 + ty * (v11 - v01);
        v0 + tz * (v1 - v0)
    }

    #[inline]
    fn sample_trilinear(field: &[f32], fx: f32, fy: f32, fk: f32) -> f32 {
        let xw = fx.rem_euclid(GRID_NX as f32);
        let yw = fy.rem_euclid(GRID_NY as f32);
        let kc = fk.clamp(0.0, (GRID_NZ - 1) as f32 - 1.0e-4);

        let xf = xw.floor();
        let yf = yw.floor();
        let i0 = xf as usize % GRID_NX;
        let i1 = (i0 + 1) % GRID_NX;
        let j0 = yf as usize % GRID_NY;
        let j1 = (j0 + 1) % GRID_NY;
        let k0 = (kc.floor() as usize).min(GRID_NZ - 2);
        let k1 = k0 + 1;

        Self::bilerp8(
            field,
            idx3(i0, j0, k0),
            idx3(i1, j0, k0),
            idx3(i0, j1, k0),
            idx3(i1, j1, k0),
            idx3(i0, j0, k1),
            idx3(i1, j0, k1),
            idx3(i0, j1, k1),
            idx3(i1, j1, k1),
            xw - xf,
            yw - yf,
            kc - k0 as f32,
        )
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn minmax8(
        field: &[f32],
        c000: usize,
        c100: usize,
        c010: usize,
        c110: usize,
        c001: usize,
        c101: usize,
        c011: usize,
        c111: usize,
    ) -> (f32, f32) {
        let a0 = field[c000].min(field[c100]);
        let a1 = field[c010].min(field[c110]);
        let a2 = field[c001].min(field[c101]);
        let a3 = field[c011].min(field[c111]);
        let b0 = field[c000].max(field[c100]);
        let b1 = field[c010].max(field[c110]);
        let b2 = field[c001].max(field[c101]);
        let b3 = field[c011].max(field[c111]);
        (a0.min(a1).min(a2.min(a3)), b0.max(b1).max(b2.max(b3)))
    }

    #[inline]
    fn sample_trilinear_with_bounds(field: &[f32], fx: f32, fy: f32, fk: f32) -> (f32, f32, f32) {
        let xw = fx.rem_euclid(GRID_NX as f32);
        let yw = fy.rem_euclid(GRID_NY as f32);
        let kc = fk.clamp(0.0, (GRID_NZ - 1) as f32 - 1.0e-4);

        let xf = xw.floor();
        let yf = yw.floor();
        let i0 = xf as usize % GRID_NX;
        let i1 = (i0 + 1) % GRID_NX;
        let j0 = yf as usize % GRID_NY;
        let j1 = (j0 + 1) % GRID_NY;
        let k0 = (kc.floor() as usize).min(GRID_NZ - 2);
        let k1 = k0 + 1;

        let c000 = idx3(i0, j0, k0);
        let c100 = idx3(i1, j0, k0);
        let c010 = idx3(i0, j1, k0);
        let c110 = idx3(i1, j1, k0);
        let c001 = idx3(i0, j0, k1);
        let c101 = idx3(i1, j0, k1);
        let c011 = idx3(i0, j1, k1);
        let c111 = idx3(i1, j1, k1);

        let val = Self::bilerp8(
            field, c000, c100, c010, c110, c001, c101, c011, c111, xw - xf, yw - yf, kc - k0 as f32,
        );
        let (mn, mx) = Self::minmax8(field, c000, c100, c010, c110, c001, c101, c011, c111);
        (val, mn, mx)
    }

    /// Fused semi-Lagrangian momentum advection + **Clamped 2nd-Order MacCormack (Predictor-Corrector)**
    /// scalar advection for `(qt, qc, qi)`!
    /// Eliminates 1st-order trilinear numerical diffusion so 3D cumulus towers, coastal fog decks,
    /// lenticular saucers, and 3D cirrus uncinus plumes maintain razor-sharp 3D permanence!
    fn advect_all_fields(&mut self, dt: f32) {
        let dom = self.params.domain_xy_m.max(4000.0);
        let inv_dx = GRID_NX as f32 / dom;
        let z_base = self.params.z_base_m;
        let z_top = self.params.z_top_m;
        let nz_scale = (GRID_NZ - 1) as f32;
        let low_cov = (self.params.low_cloud_pct * 0.01).clamp(0.0, 1.0);
        let mid_cov = (self.params.mid_cloud_pct * 0.01).clamp(0.0, 1.0);
        let q_amp = 0.18 + 0.14 * low_cov.max(mid_cov);

        // 1A. Semi-Lagrangian backtrace for momentum/buoyancy `(u', v', w, theta')`
        //     + MacCormack Predictor (`qt_hat -> scratch_e`, `qc_hat -> scratch_f`)!
        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            let u0 = self.u_bar[k];
            let v0 = self.v_bar[k];
            let qv0 = self.qv_env[k];

            for j in 0..GRID_NY {
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.scratch_a[c] = 0.0;
                        self.scratch_b[c] = 0.0;
                        self.scratch_c[c] = 0.0;
                        self.scratch_d[c] = 0.0;
                        self.scratch_e[c] = qv0 * 0.7;
                        self.scratch_f[c] = 0.0;
                        continue;
                    }

                    let u_tot = u0 + self.u_pert[c];
                    let v_tot = v0 + self.v_pert[c];
                    let w_tot = self.w_vel[c];

                    let back_x = i as f32 - dt * u_tot * inv_dx;
                    let back_y = j as f32 - dt * v_tot * inv_dx;
                    let back_z = (z - dt * w_tot).clamp(z_base, z_top);
                    let back_k = altitude_m_to_s(back_z, z_base, z_top) * nz_scale;

                    let xw = back_x.rem_euclid(GRID_NX as f32);
                    let yw = back_y.rem_euclid(GRID_NY as f32);
                    let kc = back_k.clamp(0.0, nz_scale - 1.0e-4);

                    let xf = xw.floor();
                    let yf = yw.floor();
                    let i0 = xf as usize % GRID_NX;
                    let i1 = (i0 + 1) % GRID_NX;
                    let j0 = yf as usize % GRID_NY;
                    let j1 = (j0 + 1) % GRID_NY;
                    let k0 = (kc.floor() as usize).min(GRID_NZ - 2);
                    let k1 = k0 + 1;

                    let tx = xw - xf;
                    let ty = yw - yf;
                    let tz = kc - k0 as f32;

                    let c000 = idx3(i0, j0, k0);
                    let c100 = idx3(i1, j0, k0);
                    let c010 = idx3(i0, j1, k0);
                    let c110 = idx3(i1, j1, k0);
                    let c001 = idx3(i0, j0, k1);
                    let c101 = idx3(i1, j0, k1);
                    let c011 = idx3(i0, j1, k1);
                    let c111 = idx3(i1, j1, k1);

                    self.scratch_a[c] = Self::bilerp8(
                        &self.u_pert, c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz,
                    );
                    self.scratch_b[c] = Self::bilerp8(
                        &self.v_pert, c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz,
                    );
                    self.scratch_c[c] = Self::bilerp8(
                        &self.w_vel, c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz,
                    );
                    self.scratch_d[c] = Self::bilerp8(
                        &self.theta_pert, c000, c100, c010, c110, c001, c101, c011, c111, tx, ty,
                        tz,
                    );
                    self.scratch_e[c] = Self::bilerp8(
                        &self.qt, c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz,
                    );
                    self.scratch_f[c] = Self::bilerp8(
                        &self.qc, c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz,
                    );
                }
            }
        }

        // Copy advected velocity/theta back so `scratch_a..d` are free for MacCormack corrector!
        self.u_pert.copy_from_slice(&self.scratch_a);
        self.v_pert.copy_from_slice(&self.scratch_b);
        self.w_vel.copy_from_slice(&self.scratch_c);
        self.theta_pert.copy_from_slice(&self.scratch_d);

        // 1B. Clamped 2nd-Order MacCormack Corrector on `qt` & `qc`!
        //     Forward-advect `qt_hat` (`scratch_e`) & `qc_hat` (`scratch_f`) by `+dt * u`, subtract half
        //     round-trip error, and clamp strictly to the 8-corner departure stencil `[min8, max8]`.
        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            let u0 = self.u_bar[k];
            let v0 = self.v_bar[k];
            let qv0 = self.qv_env[k];

            for j in 0..GRID_NY {
                let uy = (j as f32 + 0.5) * (1.0 / GRID_NY as f32);
                let edge_j = j.min(GRID_NY - 1 - j);
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.scratch_a[c] = qv0 * 0.7;
                        self.scratch_b[c] = 0.0;
                        continue;
                    }

                    let u_tot = u0 + self.u_pert[c];
                    let v_tot = v0 + self.v_pert[c];
                    let w_tot = self.w_vel[c];

                    let back_x = i as f32 - dt * u_tot * inv_dx;
                    let back_y = j as f32 - dt * v_tot * inv_dx;
                    let back_z = (z - dt * w_tot).clamp(z_base, z_top);
                    let back_k = altitude_m_to_s(back_z, z_base, z_top) * nz_scale;

                    let fwd_x = i as f32 + dt * u_tot * inv_dx;
                    let fwd_y = j as f32 + dt * v_tot * inv_dx;
                    let fwd_z = (z + dt * w_tot).clamp(z_base, z_top);
                    let fwd_k = altitude_m_to_s(fwd_z, z_base, z_top) * nz_scale;

                    let (_, qt_min, qt_max) =
                        Self::sample_trilinear_with_bounds(&self.qt, back_x, back_y, back_k);
                    let (_, qc_min, qc_max) =
                        Self::sample_trilinear_with_bounds(&self.qc, back_x, back_y, back_k);

                    let qt_check = Self::sample_trilinear(&self.scratch_e, fwd_x, fwd_y, fwd_k);
                    let qc_check = Self::sample_trilinear(&self.scratch_f, fwd_x, fwd_y, fwd_k);

                    let mut qt_mc = (self.scratch_e[c] + 0.5 * (self.qt[c] - qt_check))
                        .clamp(qt_min, qt_max);
                    let qc_mc = (self.scratch_f[c] + 0.5 * (self.qc[c] - qc_check))
                        .clamp(qc_min, qc_max)
                        .max(0.0);

                    // Lateral perimeter sponge relaxation (outermost 3 cells) feeds fresh synoptic cloud clusters
                    let edge_dist = i.min(GRID_NX - 1 - i).min(edge_j);
                    if edge_dist < 4 {
                        let ux = (i as f32 + 0.5) * (1.0 / GRID_NX as f32);
                        let (q_anom, _) = self.sample_synoptic_moisture_perturbation(
                            ux,
                            uy,
                            z,
                            k,
                            self.sim_time_s,
                        );
                        let inflow_qt = (qv0 * (1.0 + q_amp * q_anom)).max(1.0e-6);
                        let w_edge = (4 - edge_dist) as f32 * 0.14;
                        qt_mc = qt_mc * (1.0 - w_edge) + inflow_qt * w_edge;
                    }
                    self.scratch_a[c] = qt_mc;
                    self.scratch_b[c] = qc_mc;
                }
            }
        }
        self.qt.copy_from_slice(&self.scratch_a);
        self.qc.copy_from_slice(&self.scratch_b);

        // 2. Upper-tropospheric 3D Cirrus ice-crystal MacCormack sedimentation (`z >= 5,600 m` only!)
        let gen_rate = (dt * 0.022).clamp(0.0, 0.25);
        let jet_u = self.params.wind_250_u;
        let jet_v = self.params.wind_250_v;
        let jet_spd = (jet_u * jet_u + jet_v * jet_v).sqrt().max(1.0);
        let jx = jet_u / jet_spd;
        let jy = jet_v / jet_spd;

        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            let slice_start = k * GRID_NX * GRID_NY;
            if z < 5600.0 {
                self.scratch_a[slice_start..slice_start + GRID_NX * GRID_NY].fill(0.0);
                self.scratch_b[slice_start..slice_start + GRID_NX * GRID_NY].fill(0.0);
                self.scratch_c[slice_start..slice_start + GRID_NX * GRID_NY].fill(0.0);
                self.scratch_d[slice_start..slice_start + GRID_NX * GRID_NY].fill(0.0);
                continue;
            }
            let u0 = self.u_bar[k];
            let v0 = self.v_bar[k];
            let shift_x = (u0 * self.sim_time_s) / dom;
            let shift_y = (v0 * self.sim_time_s) / dom;

            for j in 0..GRID_NY {
                let uy = (j as f32 + 0.5) * (1.0 / GRID_NY as f32);
                let sy = (uy - shift_y).rem_euclid(1.0);
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.scratch_a[c] = 0.0;
                        self.scratch_b[c] = 0.0;
                        self.scratch_c[c] = 0.0;
                        self.scratch_d[c] = 0.0;
                        continue;
                    }
                    let ux = (i as f32 + 0.5) * (1.0 / GRID_NX as f32);
                    let sx = (ux - shift_x).rem_euclid(1.0);
                    let along = sx * jx + sy * jy;
                    let cross = -sx * jy + sy * jx;
                    // Store jet-streak moisture lane (`0..1`) in `scratch_b` so dry subsidence lanes sublimate ice!
                    self.scratch_b[c] =
                        Self::sample_lut2d_periodic(&self.synoptic_lut_ci_band, along, cross);

                    let qi_here = self.qi[c];
                    let v_fall = 0.62 + (qi_here * 2000.0).clamp(0.0, 0.80);
                    let u_tot = u0 + self.u_pert[c];
                    let v_tot = v0 + self.v_pert[c];
                    let w_tot = self.w_vel[c] - v_fall;

                    let back_x = i as f32 - dt * u_tot * inv_dx;
                    let back_y = j as f32 - dt * v_tot * inv_dx;
                    let back_z = (z - dt * w_tot).clamp(z_base, z_top);
                    let back_k = altitude_m_to_s(back_z, z_base, z_top) * nz_scale;

                    let (qi_hat, qi_min, qi_max) =
                        Self::sample_trilinear_with_bounds(&self.qi, back_x, back_y, back_k);
                    self.scratch_a[c] = qi_hat;
                    self.scratch_c[c] = qi_min;
                    self.scratch_d[c] = qi_max;
                }
            }
        }

        // Cirrus MacCormack Corrector + upper-tropospheric generating-tuft seeding
        for k in 0..GRID_NZ {
            let z = self.z_levels_m[k];
            let slice_start = k * GRID_NX * GRID_NY;
            if z < 5600.0 {
                self.scratch_e[slice_start..slice_start + GRID_NX * GRID_NY].fill(0.0);
                continue;
            }
            let u0 = self.u_bar[k];
            let v0 = self.v_bar[k];
            for j in 0..GRID_NY {
                let uy = (j as f32 + 0.5) * (1.0 / GRID_NY as f32);
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.scratch_e[c] = 0.0;
                        continue;
                    }
                    let qi_here = self.qi[c];
                    let v_fall = 0.62 + (qi_here * 2000.0).clamp(0.0, 0.80);
                    let u_tot = u0 + self.u_pert[c];
                    let v_tot = v0 + self.v_pert[c];
                    let w_tot = self.w_vel[c] - v_fall;

                    let fwd_x = i as f32 + dt * u_tot * inv_dx;
                    let fwd_y = j as f32 + dt * v_tot * inv_dx;
                    let fwd_z = (z + dt * w_tot).clamp(z_base, z_top);
                    let fwd_k = altitude_m_to_s(fwd_z, z_base, z_top) * nz_scale;

                    let qi_check = Self::sample_trilinear(&self.scratch_a, fwd_x, fwd_y, fwd_k);
                    let mut val = (self.scratch_a[c] + 0.5 * (qi_here - qi_check))
                        .clamp(self.scratch_c[c], self.scratch_d[c])
                        .max(0.0);

                    if z >= 8400.0 {
                        let ux = (i as f32 + 0.5) * (1.0 / GRID_NX as f32);
                        let (_, ci_src) = self.sample_synoptic_moisture_perturbation(
                            ux,
                            uy,
                            z,
                            k,
                            self.sim_time_s,
                        );
                        val = val * (1.0 - gen_rate * 0.52) + ci_src * gen_rate;
                    }
                    self.scratch_e[c] = val;
                }
            }
        }
        self.qi.copy_from_slice(&self.scratch_e);
    }

    /// Fused zero-`exp`/zero-`powf` Clausius-Clapeyron liquid saturation adjustment + Koop 3D Cirrus
    /// ice growth/sublimation + optional `RGBA8` GPU volume packing in a single cache-hot pass!
    fn apply_saturation_adjustment_and_pack(&mut self, dt: f32, pack_output: bool) {
        let high_cov = (self.params.high_cloud_pct * 0.01).clamp(0.0, 1.0);
        let ci_boost = self.params.cirrus_ice_boost.clamp(0.1, 2.2);

        // Sweep top-down (`k = GRID_NZ - 1 .. 0`) so terrain cells (`solid_mask != 0`) can copy
        // already-updated `self.qc[k + 1]` without needing a second pass!
        for k in (0..GRID_NZ).rev() {
            let z = self.z_levels_m[k];
            let exner = self.exner_bar[k];
            let t0 = self.temp_bar_k[k];
            let qvs_liq0 = self.qvs_liq_bar[k];
            let qvs_ice0 = self.qvs_ice_bar[k];
            let haze_alt = self.haze_alt_bar[k];

            // Precompute Clausius-Clapeyron logarithmic derivatives `d(ln q_vs)/dT = L / (R_v * T_0^2)`
            // and psychrometric latent-heat factor once per vertical level `k`!
            let alpha_liq = LV / (RV * t0 * t0);
            let alpha_ice = LS / (RV * t0 * t0);
            let inv_psychro0 = 1.0 / (1.0 + (LV * LV * qvs_liq0) / (CP * RV * t0 * t0));
            let latent_theta_gain = 0.22 * (LV / (CP * exner));

            for j in 0..GRID_NY {
                for i in 0..GRID_NX {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        self.qc[c] = 0.0;
                        self.qi[c] = 0.0;
                        if pack_output {
                            let dst = c * 4;
                            let k_up = (k + 1).min(GRID_NZ - 1);
                            let qc_up = self.qc[idx3(i, j, k_up)];
                            let qc_n = if qc_up > 0.0 {
                                (qc_up * (1.0 / 1.55e-3)).min(1.0).sqrt()
                            } else {
                                0.0
                            };
                            self.rgba_volume[dst] = (qc_n * 235.0) as u8;
                            self.rgba_volume[dst + 1] = 0;
                            self.rgba_volume[dst + 2] = 128;
                            self.rgba_volume[dst + 3] = (haze_alt * 96.0) as u8;
                        }
                        continue;
                    }

                    // Local parcel temperature anomaly including adiabatic updraft/wave-crest cooling!
                    let w_here = self.w_vel[c];
                    let lift_cool_k = (w_here * 0.48).clamp(-3.6, 3.6);
                    let dt_k = (self.theta_pert[c] * exner - lift_cool_k).clamp(-14.0, 14.0);
                    let temp_c = (t0 + dt_k) - 273.15;

                    let ad_liq = (alpha_liq * dt_k).clamp(-0.68, 0.90);
                    let qvs_liq =
                        (qvs_liq0 * (1.0 + ad_liq + 0.5 * ad_liq * ad_liq)).max(1.0e-6);

                    if dt > 0.0 {
                        // 1. Warm & Supercooled Liquid Cloud Condensation (`temp_c > -36°C`)
                        if temp_c > -36.0 {
                            let excess = (self.qt[c] - qvs_liq) * inv_psychro0;
                            let new_qc = excess.clamp(0.0, 4.5e-3);
                            let d_qc = new_qc - self.qc[c];
                            self.qc[c] = new_qc;
                            if d_qc.abs() > 1.0e-7 {
                                self.theta_pert[c] =
                                    (self.theta_pert[c] + latent_theta_gain * d_qc).clamp(-6.5, 6.5);
                            }
                        } else {
                            self.qi[c] = (self.qi[c] + self.qc[c] * 0.5).min(0.65e-3);
                            self.qc[c] = 0.0;
                        }

                        // 2. Upper-Tropospheric 3D Cirrus Heterogeneous Growth & Sublimation (`z >= 6.0 km`)
                        // Crucial physics: vapour deposition (`1.01 < RHi < 1.38`) grows ONLY inside narrow
                        // jet-streak filaments (`lane_cube`) or strong mountain-wave crests while dry lanes
                        // actively sublimate ice so distinct 3D cirrus plumes float in open sky (`~25-58%`)!
                        if z >= 6000.0 && temp_c <= -18.0 {
                            let ci_lane = self.scratch_b[c];
                            let lane_cube = ci_lane * ci_lane * (0.45 + 0.55 * ci_lane);
                            let rhi_env = if z >= 7900.0 {
                                0.52 + 0.78 * high_cov * lane_cube * ci_boost
                                    + 0.07 * w_here.clamp(-2.0, 2.6)
                            } else {
                                // Drier sub-cirrus shear zone (`6.0..7.9 km`) tapers falling mares' tails ethereally!
                                0.34 + 0.32 * high_cov * lane_cube
                            };

                            let has_ice = self.qi[c] > 4.5e-6;
                            let wave_nucleate =
                                z >= 8800.0 && w_here > 2.8 && rhi_env > 1.18 && high_cov > 0.15;

                            if (has_ice || wave_nucleate) && rhi_env > 1.02 {
                                let ad_ice = (alpha_ice * dt_k).clamp(-0.68, 0.90);
                                let qvs_ice = (qvs_ice0
                                    * (1.0 + ad_ice + 0.5 * ad_ice * ad_ice))
                                    .max(1.0e-7);
                                let dep = (rhi_env - 1.0) * qvs_ice * (dt * 0.022).min(0.22);
                                self.qi[c] = (self.qi[c] + dep).min(0.44e-3);
                            } else if self.qi[c] > 0.0 {
                                let sub_rate =
                                    ((1.08 - rhi_env).max(0.06) * dt * 0.065).clamp(0.0, 0.72);
                                let rem = self.qi[c] * (1.0 - sub_rate);
                                self.qi[c] = if rem < 4.5e-6 { 0.0 } else { rem };
                            }
                        } else {
                            self.qi[c] = 0.0;
                        }
                    }

                    if pack_output {
                        let dst = c * 4;
                        let qc_here = self.qc[c];
                        let qi_here = self.qi[c];
                        let qc_norm = if qc_here > 0.0 {
                            (qc_here * (1.0 / 1.50e-3)).min(1.0).sqrt()
                        } else {
                            0.0
                        };
                        let qi_norm = if qi_here > 0.0 {
                            (qi_here * (1.0 / 0.34e-3)).min(1.0).sqrt()
                        } else {
                            0.0
                        };

                        let vort_boost = (self.vorticity_mag[c] * 18.0).clamp(0.0, 0.35);
                        let w_encoded =
                            (0.5 + w_here * (1.0 / 14.0) + vort_boost * 0.5).clamp(0.0, 1.0);

                        let rh = (self.qt[c] / qvs_liq).clamp(0.0, 1.15);
                        let h0 = ((rh - 0.54) * (1.0 / 0.46)).clamp(0.0, 1.0);
                        let haze = if h0 > 0.0 {
                            (h0 * h0.sqrt() * haze_alt).min(1.0)
                        } else {
                            0.0
                        };

                        self.rgba_volume[dst] = (qc_norm * 255.0 + 0.5) as u8;
                        self.rgba_volume[dst + 1] = (qi_norm * 255.0 + 0.5) as u8;
                        self.rgba_volume[dst + 2] = (w_encoded * 255.0 + 0.5) as u8;
                        self.rgba_volume[dst + 3] = (haze * 255.0 + 0.5) as u8;
                    }
                }
            }
        }
    }

    /// Packs the 3D fluid & microphysics state into `self.rgba_volume` (`64 × 64 × 48 × 4` bytes).
    pub fn pack_rgba_volume(&mut self) {
        self.apply_saturation_adjustment_and_pack(0.0, true);
    }

    /// Computes live atmospheric physics telemetry for the UI diagnostics panel.
    pub fn compute_telemetry(&self) -> AtmosphereTelemetry {
        // Lifting Condensation Level (Espy's formula `z_LCL ~ z_base + 125 * (T - T_d)`)
        let rh_sfc = (self.params.surface_rh_pct * 0.01).clamp(0.10, 1.0);
        let tc = self.params.surface_temp_c;
        let gamma = (rh_sfc).ln() + (17.67 * tc) / (243.5 + tc);
        let td_c = (243.5 * gamma) / (17.67 - gamma);
        let lcl_m = self.params.z_base_m + (125.0 * (tc - td_c)).max(40.0);

        let mut freeze_m = self.params.z_top_m;
        let mut glac_m = 7500.0_f32;
        for k in 0..GRID_NZ {
            let tk = self.temp_bar_k[k];
            if tk <= 273.15 && freeze_m == self.params.z_top_m {
                freeze_m = self.z_levels_m[k];
            }
            if tk <= 238.15 {
                glac_m = self.z_levels_m[k];
                break;
            }
        }

        let mut max_up = 0.0_f32;
        let mut max_dn = 0.0_f32;
        let mut max_qc = 0.0_f32;
        let mut max_qi = 0.0_f32;
        for c in 0..GRID_CELLS {
            if self.solid_mask[c] != 0 {
                continue;
            }
            let w = self.w_vel[c];
            if w > max_up {
                max_up = w;
            }
            if w < max_dn {
                max_dn = w;
            }
            if self.qc[c] > max_qc {
                max_qc = self.qc[c];
            }
            if self.qi[c] > max_qi {
                max_qi = self.qi[c];
            }
        }

        let low_top_m = (self.params.z_base_m + 2400.0).max(2500.0);
        let mut low_col = 0usize;
        let mut mid_col = 0usize;
        let mut high_col = 0usize;
        for j in 0..GRID_NY {
            for i in 0..GRID_NX {
                let mut has_low = false;
                let mut has_mid = false;
                let mut has_high = false;
                for k in 0..GRID_NZ {
                    let c = idx3(i, j, k);
                    if self.solid_mask[c] != 0 {
                        continue;
                    }
                    let z = self.z_levels_m[k];
                    if z < low_top_m && self.qc[c] > 0.035e-3 {
                        has_low = true;
                    } else if (low_top_m..6800.0).contains(&z) && self.qc[c] > 0.035e-3 {
                        has_mid = true;
                    }
                    if z >= 6500.0 && self.qi[c] > 0.035e-3 {
                        has_high = true;
                    }
                }
                if has_low {
                    low_col += 1;
                }
                if has_mid {
                    mid_col += 1;
                }
                if has_high {
                    high_col += 1;
                }
            }
        }
        let cols = (GRID_NX * GRID_NY) as f32;

        let u_mbl = 0.5 * (self.params.wind_sfc_u + self.params.wind_850_u);
        let v_mbl = 0.5 * (self.params.wind_sfc_v + self.params.wind_850_v);
        let spd_mbl = (u_mbl * u_mbl + v_mbl * v_mbl).sqrt().max(0.5);
        let n_mbl = self.brunt_vaisala_n[6].max(0.004);
        let froude = spd_mbl / (n_mbl * self.peak_relief_m.max(100.0));

        let du_jet = self.params.wind_250_u - self.params.wind_500_u;
        let dv_jet = self.params.wind_250_v - self.params.wind_500_v;
        let jet_shear = (du_jet * du_jet + dv_jet * dv_jet).sqrt() / 4.8; // m/s per km between 5.6 and 10.4 km

        AtmosphereTelemetry {
            sim_time_s: self.sim_time_s,
            lcl_altitude_m: lcl_m.round(),
            freezing_level_m: freeze_m.round(),
            glaciation_cirrus_base_m: glac_m.round(),
            max_updraft_ms: (max_up * 100.0).round() / 100.0,
            max_downdraft_ms: (max_dn * 100.0).round() / 100.0,
            max_liquid_g_kg: (max_qc * 1000.0 * 100.0).round() / 100.0,
            max_ice_g_kg: (max_qi * 1000.0 * 100.0).round() / 100.0,
            low_cloud_coverage: ((low_col as f32 / cols) * 1000.0).round() / 10.0,
            mid_cloud_coverage: ((mid_col as f32 / cols) * 1000.0).round() / 10.0,
            high_cirrus_coverage: ((high_col as f32 / cols) * 1000.0).round() / 10.0,
            froude_number: (froude * 100.0).round() / 100.0,
            jet_shear_ms_km: (jet_shear * 100.0).round() / 10.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stretched_vertical_grid_exact_roundtrip() {
        let zb = 120.0;
        let zt = 12_200.0;
        for idx in 0..=100 {
            let s = idx as f32 / 100.0;
            let z = s_to_altitude_m(s, zb, zt);
            let s_back = altitude_m_to_s(z, zb, zt);
            assert!(
                (s - s_back).abs() < 1.0e-5,
                "stretched grid mismatch at s={s}: got {s_back}"
            );
        }
    }

    #[test]
    fn test_orographic_wave_and_3d_cirrus_evolution() {
        let mut params = AtmosphericParams::default();
        params.low_cloud_pct = 65.0;
        params.high_cloud_pct = 75.0;
        params.wind_250_u = 44.0;
        params.wind_250_v = 12.0;

        let mut atmos = FluidAtmosphere::new(params);

        // Synthesize a 64x64 coastal ridge DEM (0m ocean on west -> 750m ridge -> lee valley)
        let mut dem = vec![0.0_f32; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                let x = (c as f32 - 28.0) / 9.0;
                let ridge = 750.0 * (-x * x).exp();
                dem[r * 64 + c] = if c < 14 { 0.0 } else { ridge };
            }
        }
        atmos.set_terrain_from_dem_patch(&dem, 64, 64, 9200.0, 9200.0);

        for _ in 0..8 {
            atmos.step(8.0);
        }

        let tel = atmos.compute_telemetry();
        assert!(tel.max_updraft_ms > 0.25, "expected positive orographic updraft, got {}", tel.max_updraft_ms);
        assert!(tel.max_liquid_g_kg > 0.05, "expected marine/orographic liquid cloud water");
        assert!(tel.max_ice_g_kg > 0.01, "expected 3D upper-tropospheric cirrus ice crystals");
        assert!(
            tel.high_cirrus_coverage > 5.0 && tel.high_cirrus_coverage < 92.0,
            "expected structured 3D cirrus bands with clear sky lanes, got {}%",
            tel.high_cirrus_coverage
        );
        assert_eq!(atmos.rgba_volume.len(), GRID_CELLS * 4);
        assert!(atmos.w_vel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_alpine_half_dome_cumulus_and_lenticular_condensation() {
        let mut params = AtmosphericParams::default();
        params.surface_temp_c = 17.5;
        params.surface_rh_pct = 54.0;
        params.low_cloud_pct = 38.0;
        params.mid_cloud_pct = 58.0;
        params.high_cloud_pct = 68.0;
        params.inversion_base_m = 2450.0;
        params.inversion_strength_k = 2.0;
        params.convective_cape = 580.0;
        params.wind_sfc_u = 4.2;
        params.wind_850_u = 9.0;
        params.wind_500_u = 18.5;
        params.wind_250_u = 39.0;

        let mut atmos = FluidAtmosphere::new(params);
        let mut dem = vec![1210.0_f32; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                let dx = (c as f32 - 34.0) / 7.5;
                let dy = (r as f32 - 32.0) / 9.0;
                dem[r * 64 + c] = 1210.0 + 1485.0 * (-(dx * dx + dy * dy)).exp();
            }
        }
        atmos.set_terrain_from_dem_patch(&dem, 64, 64, 9200.0, 9200.0);
        for _ in 0..28 {
            atmos.step(8.0);
        }
        let tel = atmos.compute_telemetry();
        assert!(
            tel.max_liquid_g_kg > 0.08,
            "expected alpine cumulus/wave condensation over Half Dome DEM, got {} g/kg",
            tel.max_liquid_g_kg
        );
        assert!(
            tel.low_cloud_coverage + tel.mid_cloud_coverage > 4.0,
            "expected visible low/mid alpine clouds, got low={}%, mid={}%",
            tel.low_cloud_coverage,
            tel.mid_cloud_coverage
        );
        assert!(
            tel.high_cirrus_coverage > 8.0 && tel.high_cirrus_coverage < 90.0,
            "expected sculpted 3D cirrus mares' tails, got {}%",
            tel.high_cirrus_coverage
        );
    }

    #[test]
    fn test_detail_noise_volume_64_valid_and_non_degenerate() {
        let vol = generate_detail_noise_volume_64(42);
        assert_eq!(vol.len(), 64 * 64 * 64 * 4);
        let min_r = *vol.iter().step_by(4).min().unwrap();
        let max_r = *vol.iter().step_by(4).max().unwrap();
        assert!(max_r > min_r + 140, "expected wide dynamic range in Perlin-Worley R channel");
    }
}
