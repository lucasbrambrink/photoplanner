//! Full-Corridor (`0–380 km`) Curved-Earth Refractive, Spectral & 3D Stochastic Cloud Ray Integrator.
//!
//! Solves the radiative transfer and topographic + cloud occlusion problem along the full
//! solar line-of-sight from the upper troposphere / stratosphere (`~380 km` ground range)
//! down to a target landmark, viewpoint, or overhead cloud deck:
//!
//! 1. **Curved-Earth + Thermal-Inversion Refraction (`z_ray(s)`)**:
//!    Tracks true ray altitude above mean sea level and local slant angle `alpha_local(s)`
//!    accounting for Earth curvature (`R = 6371 km`) and lapse-rate/inversion-modulated
//!    horizontal refraction (`k = 0.08..0.50`).
//! 2. **Hybrid 1m LiDAR + Far-Field 30m SRTM Solar-Disk Horizon Gate (`0–220 km`)**:
//!    Evaluates finite `0.533°` solar-disk limb visibility across near-field notches and
//!    distant mountain ranges (plus analytical spherical ocean limb beyond `220 km`).
//! 3. **6-Band Spectral Molecular, Stratospheric Chappuis Ozone & CAMS Mie Aerosol Extinction**:
//!    Integrates `[450, 500, 550, 590, 630, 675] nm` optical depth through height-resolved
//!    Rayleigh air density, `14–32 km` Chappuis ozone absorption (`500–650 nm`, essential for
//!    true alpine rose/magenta twilight), and inversion-trapped boundary-layer aerosols.
//! 4. **Altitude-Exact 3D Stochastic Cloud-Chord & Grazing Side-Wall Projection**:
//!    Evaluates cloud bands *only* at the exact distances `s` where `z_ray(s)` intersects
//!    `[base_asl_m, top_asl_m]` (so a `2,693 m` Half Dome sunset ray passes cleanly above a
//!    `600 m` Central Valley Tule fog bank at `75 km`, while properly intersecting `9 km`
//!    Pacific cirrus at `280 km`). Computes Markov/Poisson clear-beam survival probability
//!    `P_clear`, Delta-Eddington cloudy-core transmission, bimodal `P10 / P50 / P90` quantiles,
//!    dominant blocker attribution, overhead cloud-canvas fire, and a compact 2D `(azimuth x altitude)`
//!    `CorridorRadiativeLut` (`180 x 32` RGBA8) for real-time WebGL2 terrain & volumetric rendering.

use crate::cast_horizon_ray;
use pp_terrain::{
    earth_curvature_refraction_drop_custom_m, meters_per_deg_lon, ElevationSurface, HybridTerrain,
    EARTH_RADIUS_M, METERS_PER_DEG_LAT,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// 6-band visible wavelengths in nanometers: Blue (450), Cyan (500), Green (550), Amber (590), Orange-Red (630), Deep Red (675).
pub const SPECTRAL_WAVELENGTHS_NM: [f64; 6] = [450.0, 500.0, 550.0, 590.0, 630.0, 675.0];

/// Sea-level (`1013.25 hPa`) vertical Rayleigh molecular optical depth at `SPECTRAL_WAVELENGTHS_NM`.
pub const TAU_RAYLEIGH_VERTICAL: [f64; 6] = [0.2210, 0.1435, 0.0970, 0.0730, 0.0558, 0.0422];

/// Vertical stratospheric ozone Chappuis-band optical depth per `300 DU` at `SPECTRAL_WAVELENGTHS_NM`.
pub const TAU_OZONE_300DU_VERTICAL: [f64; 6] = [0.0012, 0.0095, 0.0265, 0.0368, 0.0285, 0.0125];

/// Solar angular semi-diameter in degrees (`~16 arcminutes`).
pub const SOLAR_SEMIDIAMETER_DEG: f64 = 0.2665;

/// Default maximum ground range integrated along the solar corridor (`380 km`).
pub const DEFAULT_MAX_CORRIDOR_KM: f64 = 380.0;

/// Default start distance (`4,600 m`) for the WebGL2 far-corridor LUT so local `9 km x 9 km` (`±4.5 km`)
/// DEM & local volumetric fluid shadows are never double-counted.
pub const DEFAULT_LUT_START_DISTANCE_M: f64 = 4_600.0;

/// Maximum altitude encoded in `CorridorRadiativeLut` (`12,500 m` ASL).
pub const LUT_MAX_ALTITUDE_ASL_M: f64 = 12_500.0;

/// Non-linear altitude gamma `z = LUT_MAX_ALTITUDE_ASL_M * v^1.6` concentrating precision in `0–4,500 m`.
pub const LUT_ALTITUDE_GAMMA: f64 = 1.6;

/// Encodes far-field horizon elevation angle `[-6°, +26°]` into `[0.0, 1.0]` for LUT alpha channel.
pub const LUT_HORIZON_MIN_DEG: f64 = -6.0;
pub const LUT_HORIZON_SPAN_DEG: f64 = 32.0;

/// Altitude-resolved cloud layer at a corridor station.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudLayerBand {
    /// Layer identifier (`"fog"`, `"low"`, `"mid"`, `"high"`, `"convective"`).
    pub label: String,
    /// Cloud base altitude in meters above mean sea level (ASL).
    pub base_asl_m: f64,
    /// Cloud top altitude in meters above mean sea level (ASL).
    pub top_asl_m: f64,
    /// Fractional horizontal cloud cover in `[0.0, 1.0]`.
    pub cloud_fraction: f64,
    /// Mean vertical optical depth (`tau_perp`) averaged over the grid box (`0.0` = infer physically from thickness & phase).
    #[serde(default)]
    pub optical_depth_vertical: f64,
    /// 3D cellular vertical-to-horizontal aspect ratio `H / D` (`~0.06` stratus/fog, `~0.18` altocumulus, `~0.85` cumulus).
    #[serde(default = "default_aspect_ratio")]
    pub aspect_ratio_h_over_d: f64,
    /// Characteristic horizontal chord scale of individual cloud elements in km (`~1.8` cumulus, `~12.0` stratiform/cirrus).
    #[serde(default = "default_cell_km")]
    pub characteristic_cell_km: f64,
    /// Ice-phase fraction `[0.0 = liquid droplets, 1.0 = cirrus ice crystals]`.
    #[serde(default)]
    pub ice_fraction: f64,
}

fn default_aspect_ratio() -> f64 {
    0.22
}

fn default_cell_km() -> f64 {
    6.0
}

impl CloudLayerBand {
    /// Effective vertical optical depth inside cloudy cores (`tau_in_cloud`).
    #[inline]
    pub fn in_cloud_vertical_tau(&self) -> f64 {
        let c = self.cloud_fraction.clamp(0.0, 1.0);
        if c <= 1e-4 {
            return 0.0;
        }
        let dz_m = (self.top_asl_m - self.base_asl_m).max(60.0);
        if self.optical_depth_vertical > 0.0 {
            // Grid-mean optical depth divided by cloud fraction yields physical in-cloud core tau
            (self.optical_depth_vertical / c.max(0.12)).clamp(0.08, 120.0)
        } else {
            // Physical extinction coefficient (m^-1): dense liquid stratus/cumulus ~0.018/m, mixed ~0.006/m, ice cirrus ~0.0009/m
            let beta_vis = (1.0 - self.ice_fraction) * 0.014 + self.ice_fraction * 0.0011;
            (dz_m * beta_vis * (0.35 + 0.65 * c)).clamp(0.12, 95.0)
        }
    }
}

/// Atmospheric thermodynamic, aerosol, and cloud state at one distance station along a solar corridor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorridorStationProfile {
    /// Ground distance from the target landmark in km (`0.0 ..= 380.0`).
    pub distance_km: f64,
    /// Optional compass bearing (`0..360°`) of this station from origin (allows multi-sector sunrise + sunset states).
    #[serde(default)]
    pub azimuth_deg: Option<f64>,
    pub lat: f64,
    pub lon: f64,
    #[serde(default)]
    pub surface_elevation_m: f64,
    #[serde(default = "default_surface_temp_c")]
    pub surface_temp_c: f64,
    #[serde(default = "default_surface_pressure_hpa")]
    pub surface_pressure_hpa: f64,
    /// Free-tropospheric environmental lapse rate in `K / km` (standard `6.5`).
    #[serde(default = "default_lapse_rate")]
    pub lapse_rate_k_km: f64,
    /// Base of thermal inversion in meters ASL (`0.0` if none).
    #[serde(default)]
    pub inversion_base_asl_m: f64,
    /// Top of thermal inversion in meters ASL (`0.0` if none).
    #[serde(default)]
    pub inversion_top_asl_m: f64,
    /// Temperature jump `+Delta T` (Kelvin) across `[inversion_base_asl_m, inversion_top_asl_m]`.
    #[serde(default)]
    pub inversion_strength_k: f64,
    /// Planetary boundary layer top in meters ASL.
    #[serde(default = "default_pbl_top_m")]
    pub boundary_layer_top_asl_m: f64,
    /// Column Aerosol Optical Depth at `550 nm` (CAMS / HRRR smoke, haze, marine salt).
    #[serde(default = "default_aod_550")]
    pub aod_550nm: f64,
    /// Angstrom wavelength exponent (`~0.5` coarse sea salt/dust, `~1.4` fine urban/wildfire smoke).
    #[serde(default = "default_angstrom")]
    pub angstrom_exponent: f64,
    /// Aerosol exponential scale height in meters above surface (`~1,500 m`).
    #[serde(default = "default_aerosol_scale_m")]
    pub aerosol_scale_height_m: f64,
    /// Total ozone column in Dobson Units (`~300 DU`).
    #[serde(default = "default_ozone_du")]
    pub ozone_du: f64,
    /// Altitude-resolved cloud & fog bands at this station.
    #[serde(default)]
    pub cloud_bands: Vec<CloudLayerBand>,
}

fn default_surface_temp_c() -> f64 {
    14.0
}
fn default_surface_pressure_hpa() -> f64 {
    1013.25
}
fn default_lapse_rate() -> f64 {
    6.5
}
fn default_pbl_top_m() -> f64 {
    2_200.0
}
fn default_aod_550() -> f64 {
    0.08
}
fn default_angstrom() -> f64 {
    1.25
}
fn default_aerosol_scale_m() -> f64 {
    1_650.0
}
fn default_ozone_du() -> f64 {
    300.0
}

/// Complete `0–380 km` atmospheric corridor state around a target landmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphericCorridorState {
    /// Primary solar corridor azimuth (`0..360°`) for single-corridor profiles.
    pub azimuth_center_deg: f64,
    /// Maximum corridor integration range in km (`default = 380.0`).
    #[serde(default = "default_max_corridor_km")]
    pub max_corridor_km: f64,
    /// Ordered or multi-sector stations from `0 km` out to `320–380 km`.
    pub stations: Vec<CorridorStationProfile>,
}

fn default_max_corridor_km() -> f64 {
    DEFAULT_MAX_CORRIDOR_KM
}

#[derive(Debug, Clone)]
struct InterpolatedPointState {
    surface_elevation_m: f64,
    surface_temp_k: f64,
    surface_pressure_hpa: f64,
    lapse_rate_k_km: f64,
    inversion_base_asl_m: f64,
    inversion_top_asl_m: f64,
    inversion_strength_k: f64,
    boundary_layer_top_asl_m: f64,
    aod_550nm: f64,
    angstrom_exponent: f64,
    aerosol_scale_height_m: f64,
    ozone_du: f64,
}

impl AtmosphericCorridorState {
    /// Clean standard US-atmosphere clear-sky corridor for baseline testing or clear forecasts.
    pub fn clear_standard(origin_lat: f64, origin_lon: f64, origin_elev_m: f64, sun_az_deg: f64) -> Self {
        Self {
            azimuth_center_deg: sun_az_deg,
            max_corridor_km: DEFAULT_MAX_CORRIDOR_KM,
            stations: vec![CorridorStationProfile {
                distance_km: 0.0,
                azimuth_deg: Some(sun_az_deg),
                lat: origin_lat,
                lon: origin_lon,
                surface_elevation_m: origin_elev_m,
                surface_temp_c: 15.0 - 0.0065 * origin_elev_m,
                surface_pressure_hpa: 1013.25 * (1.0 - 2.25577e-5 * origin_elev_m).powf(5.25588),
                lapse_rate_k_km: 6.5,
                inversion_base_asl_m: 0.0,
                inversion_top_asl_m: 0.0,
                inversion_strength_k: 0.0,
                boundary_layer_top_asl_m: (origin_elev_m + 1_600.0).max(2_000.0),
                aod_550nm: 0.07,
                angstrom_exponent: 1.3,
                aerosol_scale_height_m: 1_600.0,
                ozone_du: 300.0,
                cloud_bands: Vec::new(),
            }],
        }
    }

    /// Computes normalized station weights for a query `(distance_km, ray_azimuth_deg)`.
    /// Supports both 1D single-azimuth corridors and multi-sector (e.g. sunrise + sunset) station sets.
    fn station_weights(&self, distance_km: f64, ray_azimuth_deg: f64, out: &mut [f64]) {
        let n = self.stations.len();
        if n == 0 {
            return;
        }
        if n == 1 {
            out[0] = 1.0;
            return;
        }

        // Angular alignment factor: at distance 0 km, all sectors share the local origin column;
        // further out along a ray, stations matching `ray_azimuth_deg` dominate smoothly.
        let dist_sector_blend = (distance_km / 25.0).clamp(0.0, 1.0);
        let mut sum_w = 0.0;

        for (i, st) in self.stations.iter().enumerate() {
            let st_az = st.azimuth_deg.unwrap_or(self.azimuth_center_deg);
            let mut d_az = (ray_azimuth_deg - st_az).rem_euclid(360.0);
            if d_az > 180.0 {
                d_az = 360.0 - d_az;
            }
            // Cosine-Gaussian directional kernel (sigma ~ 55 deg far out, isotropic near origin)
            let az_w = (1.0 - dist_sector_blend)
                + dist_sector_blend * (-(d_az * d_az) / (2.0 * 58.0 * 58.0)).exp().max(0.04);

            // Piecewise-tight distance Gaussian kernel adapting to log-spaced station spacing
            let sigma_km = (10.0 + 0.28 * distance_km).max(8.0);
            let dd = distance_km - st.distance_km;
            let dist_w = (-(dd * dd) / (2.0 * sigma_km * sigma_km)).exp() + 1e-6;

            let w = az_w * dist_w;
            out[i] = w;
            sum_w += w;
        }

        if sum_w > 0.0 {
            let inv = 1.0 / sum_w;
            for w in out.iter_mut().take(n) {
                *w *= inv;
            }
        } else {
            out[0] = 1.0;
        }
    }

    fn interpolate_continuous(&self, weights: &[f64]) -> InterpolatedPointState {
        if self.stations.is_empty() {
            return InterpolatedPointState {
                surface_elevation_m: 0.0,
                surface_temp_k: 288.15,
                surface_pressure_hpa: 1013.25,
                lapse_rate_k_km: 6.5,
                inversion_base_asl_m: 0.0,
                inversion_top_asl_m: 0.0,
                inversion_strength_k: 0.0,
                boundary_layer_top_asl_m: 2_000.0,
                aod_550nm: 0.08,
                angstrom_exponent: 1.25,
                aerosol_scale_height_m: 1_650.0,
                ozone_du: 300.0,
            };
        }

        let mut surf_z = 0.0;
        let mut temp_k = 0.0;
        let mut press = 0.0;
        let mut lapse = 0.0;
        let mut inv_base = 0.0;
        let mut inv_top = 0.0;
        let mut inv_str = 0.0;
        let mut pbl = 0.0;
        let mut aod = 0.0;
        let mut ang = 0.0;
        let mut h_aer = 0.0;
        let mut oz = 0.0;

        for (st, &w) in self.stations.iter().zip(weights.iter()) {
            if w <= 1e-6 {
                continue;
            }
            surf_z += w * st.surface_elevation_m;
            temp_k += w * (st.surface_temp_c + 273.15);
            press += w * st.surface_pressure_hpa;
            lapse += w * st.lapse_rate_k_km;
            inv_base += w * st.inversion_base_asl_m;
            inv_top += w * st.inversion_top_asl_m;
            inv_str += w * st.inversion_strength_k;
            pbl += w * st.boundary_layer_top_asl_m;
            aod += w * st.aod_550nm;
            ang += w * st.angstrom_exponent;
            h_aer += w * st.aerosol_scale_height_m;
            oz += w * st.ozone_du;
        }

        InterpolatedPointState {
            surface_elevation_m: surf_z,
            surface_temp_k: temp_k.max(210.0),
            surface_pressure_hpa: press.clamp(250.0, 1060.0),
            lapse_rate_k_km: lapse.clamp(1.0, 9.8),
            inversion_base_asl_m: inv_base,
            inversion_top_asl_m: inv_top,
            inversion_strength_k: inv_str.max(0.0),
            boundary_layer_top_asl_m: pbl.max(surf_z + 300.0),
            aod_550nm: aod.clamp(0.005, 4.0),
            angstrom_exponent: ang.clamp(0.1, 2.4),
            aerosol_scale_height_m: h_aer.clamp(400.0, 4500.0),
            ozone_du: oz.clamp(180.0, 480.0),
        }
    }

    /// Effective horizontal refraction coefficient `k` along the lower troposphere.
    #[inline]
    pub fn effective_refraction_k(&self, ray_azimuth_deg: f64, origin_asl_m: f64) -> f64 {
        if self.stations.is_empty() {
            return 0.135;
        }
        let mut weights = vec![0.0; self.stations.len()];
        self.station_weights(20.0, ray_azimuth_deg, &mut weights);
        let st = self.interpolate_continuous(&weights);

        // Standard dT/dz = -6.5 K/km -> k = 0.135.
        // If ray launches near or inside a thermal inversion, dT/dz > 0 -> stronger downward bending (ducting).
        let mut dtdz_k_km = -st.lapse_rate_k_km;
        if st.inversion_strength_k > 0.2 && st.inversion_top_asl_m > st.inversion_base_asl_m + 40.0 {
            let inv_dz_km = ((st.inversion_top_asl_m - st.inversion_base_asl_m) / 1000.0).max(0.08);
            let center_m = 0.5 * (st.inversion_base_asl_m + st.inversion_top_asl_m);
            let half_w = 0.75 * (st.inversion_top_asl_m - st.inversion_base_asl_m) + 350.0;
            let prox = (-((origin_asl_m - center_m) / half_w).powi(2)).exp();
            dtdz_k_km += prox * (st.inversion_strength_k / inv_dz_km + st.lapse_rate_k_km);
        }
        (0.135 + 0.028 * (dtdz_k_km + 6.5)).clamp(0.09, 0.46)
    }
}

/// Diagnostic summary of a range slice along the `0–380 km` corridor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorridorRaySegmentDiag {
    pub range_start_km: f64,
    pub range_end_km: f64,
    pub mean_ray_altitude_asl_m: f64,
    pub clear_beam_survival: f64,
    pub segment_cloud_hit_prob: f64,
    pub aerosol_optical_depth_550: f64,
    pub status_note: String,
}

/// Identifies the single physical feature responsible for the largest beam loss along the corridor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorridorBlockerAttribution {
    /// `"terrain"`, `"fog"`, `"low"`, `"mid"`, `"high"`, or `"aerosol_haze"`.
    pub blocker_type: String,
    pub distance_km: f64,
    pub altitude_asl_m: f64,
    pub beam_loss_pct: f64,
    pub summary: String,
}

/// Full radiative & stochastic diagnostic for a single Sun-to-Target ray (`0–380 km`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorridorRayResult {
    pub origin_lat: f64,
    pub origin_lon: f64,
    pub origin_asl_m: f64,
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    pub refraction_k: f64,
    pub terrain_horizon_deg: f64,
    pub far_field_horizon_deg: f64,
    pub terrain_occluder_km: f64,
    /// Fraction `[0.0, 1.0]` of the `0.533°` solar disk clearing the hybrid `1m + SRTM` horizon.
    pub terrain_disk_visibility: f64,
    /// Clear-sky (Rayleigh + Chappuis Ozone + CAMS Mie Aerosol) spectral RGB transmittance `[0..1]`.
    pub clear_sky_rgb: [f32; 3],
    /// Expected full-corridor spectral RGB transmittance `[0..1]` (including terrain disk & stochastic clouds).
    pub expected_beam_rgb: [f32; 3],
    /// Probability `[0.0, 1.0]` of an unobstructed direct beam through both terrain and 3D cloud gaps.
    pub direct_beam_probability: f64,
    /// Bimodal distribution quantiles of broadband direct solar transmittance `[0.0, 1.0]`.
    pub transmittance_p10: f64,
    pub transmittance_p50: f64,
    pub transmittance_p90: f64,
    pub transmittance_expected: f64,
    /// Correlated color temperature (Kelvin) of the surviving direct solar beam (`1850 K .. 5800 K`).
    pub beam_color_kelvin: f64,
    /// Chappuis ozone green-suppression index `[0.0, 1.0]` driving alpine rose/magenta twilight hues.
    pub chappuis_rose_index: f64,
    pub total_rayleigh_od_550: f64,
    pub total_ozone_od_590: f64,
    pub total_aerosol_od_550: f64,
    pub dominant_blocker: Option<CorridorBlockerAttribution>,
    pub segments: Vec<CorridorRaySegmentDiag>,
}

/// Illumination of local overhead cloud decks (`low`, `mid`, `high`) traced independently along the `0–380 km` corridor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadDeckGlow {
    pub band: String,
    pub deck_altitude_asl_m: f64,
    pub local_cloud_fraction: f64,
    pub corridor_direct_prob: f64,
    pub corridor_transmittance: f64,
    pub underlit_rgb: [f32; 3],
    pub color_kelvin: f64,
    pub canvas_glow_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadCanvasReport {
    pub decks: Vec<OverheadDeckGlow>,
    pub composite_canvas_score: f64,
    pub dominant_glow_band: String,
}

/// Compact 2D `(azimuth_bins x altitude_bins)` Look-Up Table (`RGBA8`) consumed by WebGL2 shaders:
/// - `R, G, B`: Far-corridor (`4.6 km -> 380 km`) expected spectral transmittance `(T_R, T_G, T_B)` at `(az, z_asl)`.
/// - `A`: Far-field SRTM + curved-Earth horizon elevation angle `H_far(az, z_asl)` encoded linearly from `[-6°, +26°]` into `[0, 255]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorridorRadiativeLut {
    pub az_bins: usize,
    pub alt_bins: usize,
    pub start_distance_m: f64,
    pub max_altitude_asl_m: f64,
    pub altitude_gamma: f64,
    pub horizon_min_deg: f64,
    pub horizon_span_deg: f64,
    pub sun_elevation_deg: f64,
    /// Row-major (`alt_idx * az_bins + az_idx`) packed `RGBA8` (`len = az_bins * alt_bins * 4`).
    pub rgba8: Vec<u8>,
}

impl CorridorRadiativeLut {
    #[inline]
    pub fn bin_to_altitude_m(alt_idx: usize, alt_bins: usize) -> f64 {
        if alt_bins <= 1 {
            return 0.0;
        }
        let v = (alt_idx as f64) / ((alt_bins - 1) as f64);
        LUT_MAX_ALTITUDE_ASL_M * v.powf(LUT_ALTITUDE_GAMMA)
    }

    #[inline]
    pub fn altitude_m_to_v(z_asl_m: f64) -> f64 {
        (z_asl_m / LUT_MAX_ALTITUDE_ASL_M)
            .clamp(0.0, 1.0)
            .powf(1.0 / LUT_ALTITUDE_GAMMA)
    }

    #[inline]
    pub fn encode_horizon_u8(horizon_deg: f64) -> u8 {
        let norm = ((horizon_deg - LUT_HORIZON_MIN_DEG) / LUT_HORIZON_SPAN_DEG).clamp(0.0, 1.0);
        (norm * 255.0).round() as u8
    }

    #[inline]
    pub fn decode_horizon_deg(byte_val: u8) -> f64 {
        LUT_HORIZON_MIN_DEG + (byte_val as f64 / 255.0) * LUT_HORIZON_SPAN_DEG
    }

    /// Bilinearly samples `(T_R, T_G, T_B, H_far_deg)` at `(azimuth_deg, z_asl_m)`.
    pub fn sample(&self, azimuth_deg: f64, z_asl_m: f64) -> ([f32; 3], f64) {
        let u = (azimuth_deg.rem_euclid(360.0) / 360.0) * (self.az_bins as f64);
        let u0 = (u.floor() as usize) % self.az_bins;
        let u1 = (u0 + 1) % self.az_bins;
        let du = (u - u.floor()) as f32;

        let v = Self::altitude_m_to_v(z_asl_m) * ((self.alt_bins - 1) as f64);
        let v0 = (v.floor() as usize).min(self.alt_bins - 1);
        let v1 = (v0 + 1).min(self.alt_bins - 1);
        let dv = (v - v0 as f64) as f32;

        let fetch = |iu: usize, iv: usize| -> [f32; 4] {
            let off = (iv * self.az_bins + iu) * 4;
            [
                self.rgba8[off] as f32 / 255.0,
                self.rgba8[off + 1] as f32 / 255.0,
                self.rgba8[off + 2] as f32 / 255.0,
                self.rgba8[off + 3] as f32 / 255.0,
            ]
        };

        let c00 = fetch(u0, v0);
        let c10 = fetch(u1, v0);
        let c01 = fetch(u0, v1);
        let c11 = fetch(u1, v1);

        let mut out = [0.0_f32; 4];
        for k in 0..4 {
            let a = c00[k] * (1.0 - du) + c10[k] * du;
            let b = c01[k] * (1.0 - du) + c11[k] * du;
            out[k] = a * (1.0 - dv) + b * dv;
        }
        let h_deg = LUT_HORIZON_MIN_DEG + (out[3] as f64) * LUT_HORIZON_SPAN_DEG;
        ([out[0], out[1], out[2]], h_deg)
    }

    /// Encodes the raw `RGBA8` buffer as standard Base64 for compact JSON API responses (`23 KB -> 30.7 KB`).
    pub fn to_base64(&self) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let bytes = &self.rgba8;
        let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
        let mut i = 0;
        while i + 3 <= bytes.len() {
            let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
            out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
            out.push(TABLE[(n & 0x3f) as usize] as char);
            i += 3;
        }
        let rem = bytes.len() - i;
        if rem == 1 {
            let n = (bytes[i] as u32) << 16;
            out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
            out.push('=');
            out.push('=');
        } else if rem == 2 {
            let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
            out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
            out.push('=');
        }
        out
    }
}

/// Smooth circular-segment visibility fraction `[0.0, 1.0]` of the `0.533°` solar disk above `horizon_deg`.
#[inline]
pub fn solar_disk_visibility_fraction(sun_center_el_deg: f64, horizon_deg: f64) -> f64 {
    let r = SOLAR_SEMIDIAMETER_DEG;
    let delta = sun_center_el_deg - horizon_deg;
    if delta <= -r {
        0.0
    } else if delta >= r {
        1.0
    } else {
        // Exact area fraction of a unit circle above signed offset u = -delta / r in [-1, 1]
        let u = (-delta / r).clamp(-1.0, 1.0);
        let area_above = std::f64::consts::PI - u.acos() + u * (1.0 - u * u).max(0.0).sqrt();
        (area_above / std::f64::consts::PI).clamp(0.0, 1.0)
    }
}

/// Converts 6-band spectral transmittance `[450, 500, 550, 590, 630, 675 nm]` to linear sRGB `[R, G, B]`,
/// broadband photopic transmittance, correlated color temperature (Kelvin), and Chappuis rose index.
fn spectral6_to_rgb_and_kelvin(trans: &[f64; 6]) -> ([f32; 3], f64, f64, f64) {
    // CIE-weighted conversion from narrow visible bands to linear sRGB direct sunlight tint
    let r = (0.18 * trans[3] + 0.47 * trans[4] + 0.35 * trans[5]).clamp(0.0, 1.0);
    let g = (0.20 * trans[1] + 0.58 * trans[2] + 0.22 * trans[3]).clamp(0.0, 1.0);
    let b = (0.76 * trans[0] + 0.24 * trans[1]).clamp(0.0, 1.0);

    let photopic = (0.06 * trans[0]
        + 0.14 * trans[1]
        + 0.36 * trans[2]
        + 0.24 * trans[3]
        + 0.13 * trans[4]
        + 0.07 * trans[5])
        .clamp(0.0, 1.0);

    // McCamy / Planckian chromaticity mapping from R/B spectral transmission ratio (5780 K exo-atmospheric sun)
    let kelvin = if r <= 1e-5 {
        2200.0
    } else {
        let rb_ratio = (b / r.max(1e-5)).clamp(0.02, 1.15);
        (1750.0 + 4030.0 * rb_ratio.powf(0.62)).clamp(1800.0, 5850.0)
    };

    // Chappuis rose index measures 675 nm crimson dominance over the 590 nm Chappuis ozone absorption trough
    let chappuis_rose = if trans[5] > 1.0e-4 && trans[3] > 1.0e-6 {
        let log_red_over_chappuis = (trans[5] / trans[3]).ln();
        ((log_red_over_chappuis - 0.16) / 1.05).clamp(0.0, 1.0)
    } else {
        0.0
    };

    ([r as f32, g as f32, b as f32], photopic, kelvin, chappuis_rose)
}

/// Quantile of a bimodal direct/cloudy-core radiative distribution with gap probability `p_clear`
/// and cloudy-core transmission `t_cloudy` (modeled with spatial sub-grid partial-limb variance).
fn bimodal_corridor_quantile(p_clear: f64, t_cloudy: f64, q: f64) -> f64 {
    let p_clear = p_clear.clamp(0.0, 1.0);
    let p_blocked = 1.0 - p_clear;
    let t_low = t_cloudy.clamp(0.0, 1.0);

    if p_clear >= 0.995 {
        return 1.0;
    }
    if p_clear <= 0.005 {
        return t_low;
    }

    // Sub-grid penumbra transition (~18% of probability mass sits on cloud-edge partial solar limbs)
    let edge_mass = (0.36 * p_clear * p_blocked).min(0.18);
    let core_blocked = (p_blocked - 0.5 * edge_mass).max(0.0);
    let core_clear_start = (core_blocked + edge_mass).min(1.0);

    if q <= core_blocked {
        let frac = if core_blocked > 1e-6 {
            q / core_blocked
        } else {
            0.5
        };
        t_low * (0.65 + 0.35 * frac)
    } else if q < core_clear_start {
        let frac = (q - core_blocked) / edge_mass.max(1e-6);
        t_low + (1.0 - t_low) * frac
    } else {
        let frac = (q - core_clear_start) / (1.0 - core_clear_start).max(1e-6);
        0.92 + 0.08 * frac
    }
}

/// Casts only the far-field (`start_dist_m .. max_dist_m`) SRTM curved-Earth horizon across all `alt_bins`
/// altitudes simultaneously in a single radial terrain walk along `azimuth_deg`.
pub fn cast_far_field_multi_alt_horizon(
    terrain: &HybridTerrain,
    origin_lat: f64,
    origin_lon: f64,
    azimuth_deg: f64,
    refraction_k: f64,
    start_dist_m: f64,
    max_dist_m: f64,
    altitudes_asl_m: &[f64],
    out_horizon_deg: &mut [f64],
) {
    assert_eq!(altitudes_asl_m.len(), out_horizon_deg.len());
    let n_alt = altitudes_asl_m.len();
    let k = refraction_k.clamp(0.05, 0.55);

    // Initialize with analytical sea-level curved-Earth limb depression so downward rays from
    // high clouds (e.g. 9,000 m ASL) never shine through the solid ocean sphere beyond max_dist_m.
    let mut max_slopes = vec![0.0_f64; n_alt];
    for i in 0..n_alt {
        let z0 = altitudes_asl_m[i].max(0.0);
        max_slopes[i] = -((2.0 * z0 * (1.0 - k)) / EARTH_RADIUS_M).sqrt();
    }

    let az_rad = azimuth_deg.to_radians();
    let dir_east = az_rad.sin();
    let dir_north = az_rad.cos();
    let m_per_lat = METERS_PER_DEG_LAT;
    let m_per_lon = meters_per_deg_lon(origin_lat).max(1.0);

    let mut d = start_dist_m.max(500.0);
    while d <= max_dist_m {
        let lat = origin_lat + (d * dir_north) / m_per_lat;
        let lon = origin_lon + (d * dir_east) / m_per_lon;
        let z_srtm = terrain.far_srtm.elevation_m(lat, lon);
        let drop = earth_curvature_refraction_drop_custom_m(d, k);
        let z_eff = z_srtm - drop;
        let inv_d = 1.0 / d;

        for i in 0..n_alt {
            let slope = (z_eff - altitudes_asl_m[i]) * inv_d;
            if slope > max_slopes[i] {
                max_slopes[i] = slope;
            }
        }

        if d < 12_000.0 {
            d += 25.0;
        } else if d < 40_000.0 {
            d += 50.0;
        } else if d < 100_000.0 {
            d += 90.0;
        } else {
            d += 150.0;
        }
    }

    for i in 0..n_alt {
        out_horizon_deg[i] = max_slopes[i].atan().to_degrees();
    }
}

/// Core internal integrator for atmospheric + cloud optical transport along `[start_dist_m, max_dist_m]`.
fn integrate_atmospheric_corridor_range(
    corridor: &AtmosphericCorridorState,
    origin_asl_m: f64,
    ray_azimuth_deg: f64,
    ray_elevation_deg: f64,
    refraction_k: f64,
    start_dist_m: f64,
    max_dist_m: f64,
    collect_diagnostics: bool,
) -> (
    [f64; 6], // clear_sky_trans
    [f64; 6], // expected_cloud_plus_sky_trans
    f64,      // p_clear_cloud
    f64,      // t_cloudy_core_550
    f64,      // rayleigh_od_550
    f64,      // ozone_od_590
    f64,      // aerosol_od_550
    Option<CorridorBlockerAttribution>,
    Vec<CorridorRaySegmentDiag>,
) {
    let n_stations = corridor.stations.len();
    let mut weights = vec![0.0_f64; n_stations.max(1)];

    let alpha0_rad = ray_elevation_deg.clamp(-5.0, 89.5).to_radians();
    let tan_alpha0 = alpha0_rad.tan();
    let r_eff = EARTH_RADIUS_M / (1.0 - refraction_k.clamp(0.05, 0.55));

    let mut tau_mol_aer = [0.0_f64; 6];
    let mut tau_rayleigh_550 = 0.0_f64;
    let mut tau_ozone_590 = 0.0_f64;
    let mut tau_aerosol_550 = 0.0_f64;

    let mut sum_cloud_hazard = 0.0_f64;
    let mut sum_uncond_cloudy_tau = 0.0_f64;

    let mut worst_loss_score = 0.0_f64;
    let mut worst_blocker: Option<(String, f64, f64, f64)> = None;

    // 6 diagnostic range buckets: [0-5, 5-25, 25-65, 65-130, 130-230, 230-380] km
    let seg_bounds_km: [(f64, f64); 6] = [
        (0.0, 5.0),
        (5.0, 25.0),
        (25.0, 65.0),
        (65.0, 130.0),
        (130.0, 230.0),
        (230.0, 380.0),
    ];
    let mut seg_alt_sum = [0.0_f64; 6];
    let mut seg_steps = [0usize; 6];
    let mut seg_hazard = [0.0_f64; 6];
    let mut seg_aer_od = [0.0_f64; 6];
    let mut seg_under_fog_margin_m = [f64::NEG_INFINITY; 6];

    // Extend molecular Rayleigh + stratospheric Chappuis ozone (15-42 km ASL) past the tropospheric
    // cloud corridor `max_dist_m` until the curved-Earth ray exits `z_ray_m > 45_000 m`.
    let atmo_limit_m = max_dist_m.max(820_000.0);
    let abs_tan0 = tan_alpha0.abs().max(0.01);

    let mut s_m = start_dist_m.max(0.0);
    while s_m < atmo_limit_m {
        let in_weather_corridor = s_m < max_dist_m;
        let base_ds_m: f64 = if !in_weather_corridor {
            8_000.0
        } else if s_m < 10_000.0 {
            250.0
        } else if s_m < 40_000.0 {
            600.0
        } else if s_m < 120_000.0 {
            1_250.0
        } else {
            2_500.0
        };
        let max_dz_m: f64 = if in_weather_corridor { 250.0 } else { 900.0 };
        let ds_m = base_ds_m.min(max_dz_m / abs_tan0).max(2.0);

        let step_end_m = if in_weather_corridor {
            (s_m + ds_m).min(max_dist_m)
        } else {
            (s_m + ds_m).min(atmo_limit_m)
        };
        let step_horiz_m = step_end_m - s_m;
        let s_mid_m = s_m + 0.5 * step_horiz_m;
        s_m = step_end_m;
        if step_horiz_m <= 0.0 {
            break;
        }

        // Exact curved-Earth + refracted ray altitude and local slope dz/ds at s_mid_m
        let drop_mid = (s_mid_m * s_mid_m) / (2.0 * r_eff);
        let z_ray_m = origin_asl_m + s_mid_m * tan_alpha0 + drop_mid;
        if z_ray_m > 45_000.0 {
            // Above stratospheric ozone layer; negligible remaining optical depth
            break;
        }
        if z_ray_m < -150.0 {
            // Below solid Earth/ocean surface
            break;
        }

        let dz_ds = tan_alpha0 + s_mid_m / r_eff;
        let dz_step_m = (step_horiz_m * dz_ds).abs();
        let ds_path_m = (step_horiz_m * step_horiz_m + dz_step_m * dz_step_m).sqrt();

        let s_km = s_mid_m * 0.001;
        corridor.station_weights(s_km.min(max_dist_m * 0.001), ray_azimuth_deg, &mut weights);
        let st = corridor.interpolate_continuous(&weights);

        // 1. Rayleigh molecular scattering (barometric density profile)
        let h_rayleigh = 8_434.0;
        let rho_rel = (st.surface_pressure_hpa / 1013.25)
            * (-z_ray_m.max(0.0) / h_rayleigh).exp()
            * (288.15 / (st.surface_temp_k - st.lapse_rate_k_km * 0.001 * (z_ray_m - st.surface_elevation_m).max(0.0)).max(212.0));
        let d_rayleigh_base = (ds_path_m / h_rayleigh) * rho_rel.max(0.0);

        // 2. Stratospheric Chappuis ozone absorption (12-36 km Gaussian centered at 22.5 km)
        let oz_z0 = 22_500.0;
        let oz_sigma = 5_200.0;
        let oz_gauss = (-(z_ray_m - oz_z0).powi(2) / (2.0 * oz_sigma * oz_sigma)).exp()
            / (2.50662827 * oz_sigma);
        let d_ozone_scale = ds_path_m * oz_gauss * (st.ozone_du / 300.0);

        // 3. CAMS / Boundary-Layer Mie aerosol extinction (trapped below inversion / PBL)
        let agl_m = (z_ray_m - st.surface_elevation_m).max(0.0);
        let mut inv_trap = 1.0;
        if st.inversion_strength_k > 0.5 && st.inversion_top_asl_m > st.surface_elevation_m {
            if z_ray_m <= st.inversion_top_asl_m {
                inv_trap = 1.0 + (0.22 * st.inversion_strength_k).min(1.4);
            } else {
                inv_trap = 0.22; // Pristine free alpine troposphere above inversion lid
            }
        } else if z_ray_m > st.boundary_layer_top_asl_m + 600.0 {
            inv_trap = 0.35;
        }
        let beta_aer_550 = (st.aod_550nm / st.aerosol_scale_height_m)
            * (-agl_m / st.aerosol_scale_height_m).exp()
            * inv_trap;
        let d_aer_550 = ds_path_m * beta_aer_550;

        tau_rayleigh_550 += TAU_RAYLEIGH_VERTICAL[2] * d_rayleigh_base;
        tau_ozone_590 += TAU_OZONE_300DU_VERTICAL[3] * d_ozone_scale;
        tau_aerosol_550 += d_aer_550;

        for b in 0..6 {
            let wl_ratio = SPECTRAL_WAVELENGTHS_NM[b] / 550.0;
            let d_aer_b = d_aer_550 * wl_ratio.powf(-st.angstrom_exponent);
            tau_mol_aer[b] += TAU_RAYLEIGH_VERTICAL[b] * d_rayleigh_base
                + TAU_OZONE_300DU_VERTICAL[b] * d_ozone_scale
                + d_aer_b;
        }

        if in_weather_corridor && d_aer_550 > 0.015 && d_aer_550 > worst_loss_score {
            worst_loss_score = d_aer_550;
            worst_blocker = Some(("aerosol_haze".to_string(), s_km, z_ray_m, d_aer_550));
        }

        if !in_weather_corridor {
            continue;
        }

        // 4. Altitude-exact 3D stochastic cloud-chord & grazing turret side-wall hazard
        let mut step_hazard = 0.0_f64;
        for (st_idx, station) in corridor.stations.iter().enumerate() {
            let w_st = weights[st_idx];
            if w_st < 0.02 {
                continue;
            }
            for band in &station.cloud_bands {
                let c = band.cloud_fraction.clamp(0.0, 0.992);
                if c <= 0.005 || band.top_asl_m <= band.base_asl_m + 10.0 {
                    continue;
                }
                if z_ray_m >= band.base_asl_m && z_ray_m <= band.top_asl_m {
                    let h_band = (band.top_asl_m - band.base_asl_m).max(80.0);
                    let cell_d_m = (band.characteristic_cell_km.max(0.25) * 1000.0)
                        .min((h_band / band.aspect_ratio_h_over_d.max(0.03)).max(250.0));

                    let neg_ln_clear = -(1.0 - c).max(0.012).ln();
                    let du_vert = (dz_step_m / h_band) * neg_ln_clear;
                    let du_grazing =
                        (step_horiz_m / cell_d_m) * neg_ln_clear * (1.0 - 0.50 * c);
                    let d_lambda = w_st * (du_vert * du_vert + du_grazing * du_grazing).sqrt();

                    // Delta-Eddington effective slant optical depth inside cloudy cores
                    // Forward scattering f = g^2 (~0.72 liquid, ~0.60 ice) transmits thin cirrus/mist glow
                    let g_asym = 0.85 * (1.0 - band.ice_fraction) + 0.77 * band.ice_fraction;
                    let delta_eddington_factor = 1.0 - g_asym * g_asym * 0.88;
                    let tau_core_vert = band.in_cloud_vertical_tau();
                    let d_uncond_tau = w_st
                        * c
                        * (tau_core_vert / h_band)
                        * ds_path_m.min(h_band * 12.0)
                        * delta_eddington_factor;

                    step_hazard += d_lambda;
                    sum_uncond_cloudy_tau += d_uncond_tau;

                    if d_lambda > worst_loss_score {
                        worst_loss_score = d_lambda;
                        worst_blocker = Some((band.label.clone(), s_km, z_ray_m, d_lambda));
                    }
                } else if collect_diagnostics
                    && (band.label == "fog" || band.label == "low")
                    && c > 0.25
                    && z_ray_m > band.top_asl_m
                {
                    let margin = z_ray_m - band.top_asl_m;
                    for (si, &(lo, hi)) in seg_bounds_km.iter().enumerate() {
                        if s_km >= lo && s_km < hi && margin > seg_under_fog_margin_m[si] {
                            seg_under_fog_margin_m[si] = margin;
                        }
                    }
                }
            }
        }

        sum_cloud_hazard += step_hazard;

        if collect_diagnostics {
            for (si, &(lo, hi)) in seg_bounds_km.iter().enumerate() {
                if s_km >= lo && s_km < hi {
                    seg_alt_sum[si] += z_ray_m;
                    seg_steps[si] += 1;
                    seg_hazard[si] += step_hazard;
                    seg_aer_od[si] += d_aer_550;
                    break;
                }
            }
        }
    }

    let p_clear_cloud = (-sum_cloud_hazard).exp().clamp(0.0, 1.0);
    let mean_cloudy_tau = if p_clear_cloud < 0.998 && sum_uncond_cloudy_tau > 0.0 {
        (sum_uncond_cloudy_tau / (1.0 - p_clear_cloud).max(0.01)).clamp(0.06, 40.0)
    } else {
        0.0
    };
    let t_cloudy_core = (-mean_cloudy_tau).exp().clamp(0.0, 1.0);
    let expected_cloud_trans = p_clear_cloud + (1.0 - p_clear_cloud) * t_cloudy_core;

    let mut clear_sky_trans = [0.0_f64; 6];
    let mut expected_total_trans = [0.0_f64; 6];
    for b in 0..6 {
        let ts = (-tau_mol_aer[b]).exp().clamp(0.0, 1.0);
        clear_sky_trans[b] = ts;
        // Thin clouds are nearly grey in visible (slight warm bias when very thin ice/mist)
        expected_total_trans[b] = ts * expected_cloud_trans;
    }

    let blocker_attr = worst_blocker.and_then(|(kind, dist_km, alt_m, _)| {
        let cloud_loss_pct = (1.0 - expected_cloud_trans) * 100.0;
        let aer_loss_pct = (1.0 - (-tau_aerosol_550).exp()) * 100.0;
        let loss_pct = if kind == "aerosol_haze" {
            aer_loss_pct
        } else {
            cloud_loss_pct
        };
        if loss_pct < 6.0 {
            None
        } else {
            let summary = if kind == "aerosol_haze" {
                format!(
                    "Boundary-layer haze/smoke around {:.0} km ({:.0} m ASL, -{:.0}% beam)",
                    dist_km, alt_m, loss_pct
                )
            } else {
                format!(
                    "{} cloud deck at {:.0} km ({:.0} m ASL, -{:.0}% beam)",
                    kind.to_uppercase(),
                    dist_km,
                    alt_m,
                    loss_pct
                )
            };
            Some(CorridorBlockerAttribution {
                blocker_type: kind,
                distance_km: (dist_km * 10.0).round() / 10.0,
                altitude_asl_m: alt_m.round(),
                beam_loss_pct: (loss_pct * 10.0).round() / 10.0,
                summary,
            })
        }
    });

    let mut segments = Vec::new();
    if collect_diagnostics {
        let mut cum_hazard = 0.0_f64;
        for (si, &(lo, hi)) in seg_bounds_km.iter().enumerate() {
            if seg_steps[si] == 0 {
                continue;
            }
            cum_hazard += seg_hazard[si];
            let seg_hit = (1.0 - (-seg_hazard[si]).exp()).clamp(0.0, 1.0);
            let mean_alt = seg_alt_sum[si] / (seg_steps[si] as f64);
            let status_note = if seg_hit > 0.35 {
                format!(
                    "Ray ({:.0} m ASL) intersects cloud band ({:.0}% chord hit)",
                    mean_alt,
                    seg_hit * 100.0
                )
            } else if seg_under_fog_margin_m[si] > 50.0 {
                format!(
                    "Ray ({:.0} m ASL) clears underlying fog/low deck (+{:.0} m margin)",
                    mean_alt, seg_under_fog_margin_m[si]
                )
            } else {
                format!("Clear corridor aloft ({:.0} m ASL)", mean_alt)
            };
            segments.push(CorridorRaySegmentDiag {
                range_start_km: lo,
                range_end_km: hi,
                mean_ray_altitude_asl_m: mean_alt.round(),
                clear_beam_survival: ((-cum_hazard).exp() * 1000.0).round() / 1000.0,
                segment_cloud_hit_prob: (seg_hit * 1000.0).round() / 1000.0,
                aerosol_optical_depth_550: (seg_aer_od[si] * 1000.0).round() / 1000.0,
                status_note,
            });
        }
    }

    (
        clear_sky_trans,
        expected_total_trans,
        p_clear_cloud,
        t_cloudy_core,
        tau_rayleigh_550,
        tau_ozone_590,
        tau_aerosol_550,
        blocker_attr,
        segments,
    )
}

/// Evaluates the full `0–380 km` Sun-to-Target corridor ray from `(origin_lat, origin_lon, origin_asl_m)`.
pub fn evaluate_corridor_ray(
    terrain: &HybridTerrain,
    corridor: &AtmosphericCorridorState,
    origin_lat: f64,
    origin_lon: f64,
    origin_asl_m: f64,
    sun_azimuth_deg: f64,
    sun_elevation_deg: f64,
) -> CorridorRayResult {
    let ground_asl_m = terrain.elevation_m(origin_lat, origin_lon);
    let origin_asl_m = origin_asl_m.max(ground_asl_m + 1.8);
    let refraction_k = corridor.effective_refraction_k(sun_azimuth_deg, origin_asl_m);
    let max_terrain_m = 200_000.0;

    // 1. Hybrid 1m + SRTM full horizon & far-field horizon
    let near_and_far = cast_horizon_ray(
        terrain,
        origin_lat,
        origin_lon,
        origin_asl_m,
        sun_azimuth_deg,
        max_terrain_m,
    );
    let mut far_h = [0.0_f64; 1];
    cast_far_field_multi_alt_horizon(
        terrain,
        origin_lat,
        origin_lon,
        sun_azimuth_deg,
        refraction_k,
        DEFAULT_LUT_START_DISTANCE_M,
        max_terrain_m,
        &[origin_asl_m],
        &mut far_h,
    );

    let effective_horizon_deg = near_and_far.horizon_elevation_deg.max(far_h[0]);
    let disk_vis = solar_disk_visibility_fraction(sun_elevation_deg, effective_horizon_deg);

    // 2. Integrate molecular + ozone + aerosol + 3D stochastic clouds out to 380 km
    let max_corridor_m = corridor.max_corridor_km.clamp(40.0, 420.0) * 1000.0;
    let (
        clear_sky_trans,
        expected_atmo_trans,
        p_clear_cloud,
        t_cloudy_core,
        rayleigh_od,
        ozone_od,
        aerosol_od,
        atmo_blocker,
        segments,
    ) = integrate_atmospheric_corridor_range(
        corridor,
        origin_asl_m,
        sun_azimuth_deg,
        sun_elevation_deg,
        refraction_k,
        0.0,
        max_corridor_m,
        true,
    );

    let (clear_rgb, clear_photopic, _, _) = spectral6_to_rgb_and_kelvin(&clear_sky_trans);
    let mut full_trans = [0.0_f64; 6];
    for b in 0..6 {
        full_trans[b] = expected_atmo_trans[b] * disk_vis;
    }
    let (expected_rgb, expected_photopic, kelvin, rose_idx) =
        spectral6_to_rgb_and_kelvin(&full_trans);

    let det_factor = disk_vis * clear_photopic;
    let t_p10 = det_factor * bimodal_corridor_quantile(p_clear_cloud, t_cloudy_core, 0.10);
    let t_p50 = det_factor * bimodal_corridor_quantile(p_clear_cloud, t_cloudy_core, 0.50);
    let t_p90 = det_factor * bimodal_corridor_quantile(p_clear_cloud, t_cloudy_core, 0.90);

    let dominant_blocker = if disk_vis < 0.5 {
        let dist_km = near_and_far.occluder_distance_m * 0.001;
        Some(CorridorBlockerAttribution {
            blocker_type: "terrain".to_string(),
            distance_km: (dist_km * 10.0).round() / 10.0,
            altitude_asl_m: near_and_far.occluder_height_m,
            beam_loss_pct: ((1.0 - disk_vis) * 1000.0).round() / 10.0,
            summary: format!(
                "Topographic horizon ({:.2}°) at {:.1} km ({:.0} m ASL) blocks {:.0}% of solar disk",
                effective_horizon_deg,
                dist_km,
                near_and_far.occluder_height_m,
                (1.0 - disk_vis) * 100.0
            ),
        })
    } else {
        atmo_blocker
    };

    CorridorRayResult {
        origin_lat,
        origin_lon,
        origin_asl_m: (origin_asl_m * 10.0).round() / 10.0,
        sun_azimuth_deg: (sun_azimuth_deg * 100.0).round() / 100.0,
        sun_elevation_deg: (sun_elevation_deg * 100.0).round() / 100.0,
        refraction_k: (refraction_k * 1000.0).round() / 1000.0,
        terrain_horizon_deg: (effective_horizon_deg * 1000.0).round() / 1000.0,
        far_field_horizon_deg: (far_h[0] * 1000.0).round() / 1000.0,
        terrain_occluder_km: (near_and_far.occluder_distance_m * 0.01).round() / 10.0,
        terrain_disk_visibility: (disk_vis * 1000.0).round() / 1000.0,
        clear_sky_rgb: clear_rgb,
        expected_beam_rgb: expected_rgb,
        direct_beam_probability: ((disk_vis * p_clear_cloud) * 1000.0).round() / 1000.0,
        transmittance_p10: (t_p10 * 1000.0).round() / 1000.0,
        transmittance_p50: (t_p50 * 1000.0).round() / 1000.0,
        transmittance_p90: (t_p90 * 1000.0).round() / 1000.0,
        transmittance_expected: (expected_photopic * 1000.0).round() / 1000.0,
        beam_color_kelvin: kelvin.round(),
        chappuis_rose_index: (rose_idx * 1000.0).round() / 1000.0,
        total_rayleigh_od_550: (rayleigh_od * 1000.0).round() / 1000.0,
        total_ozone_od_590: (ozone_od * 1000.0).round() / 1000.0,
        total_aerosol_od_550: (aerosol_od * 1000.0).round() / 1000.0,
        dominant_blocker,
        segments,
    }
}

/// Evaluates whether local overhead `low`, `mid`, and `high` cloud decks above `(origin_lat, origin_lon)`
/// receive direct underlit golden/crimson corridor sunlight from `[0..380 km]` (including post-sunset cloud fire!).
pub fn evaluate_overhead_cloud_canvas(
    terrain: &HybridTerrain,
    corridor: &AtmosphericCorridorState,
    origin_lat: f64,
    origin_lon: f64,
    sun_azimuth_deg: f64,
    sun_elevation_deg: f64,
) -> OverheadCanvasReport {
    let ground_z = terrain.elevation_m(origin_lat, origin_lon);
    let mut weights = vec![0.0_f64; corridor.stations.len().max(1)];
    corridor.station_weights(0.0, sun_azimuth_deg, &mut weights);

    // Extract representative local cloud bases & fractions for low / mid / high
    let targets = [
        ("low", (ground_z + 900.0).max(1_800.0)),
        ("mid", (ground_z + 2_400.0).max(4_400.0)),
        ("high", 8_400.0),
    ];

    let mut decks = Vec::with_capacity(3);
    let mut best_score = 0.0_f64;
    let mut best_band = "none".to_string();

    for (name, default_base_m) in targets {
        let mut frac_sum = 0.0_f64;
        let mut base_sum = 0.0_f64;
        let mut w_sum = 0.0_f64;

        for (i, st) in corridor.stations.iter().enumerate() {
            let w = weights[i];
            if w < 0.05 {
                continue;
            }
            for b in &st.cloud_bands {
                let matches = match name {
                    "low" => b.label == "low" || (b.base_asl_m < 3_400.0 && b.label != "fog"),
                    "mid" => b.label == "mid" || (b.base_asl_m >= 3_400.0 && b.base_asl_m < 6_800.0),
                    _ => b.label == "high" || b.base_asl_m >= 6_800.0,
                };
                if matches {
                    frac_sum += w * b.cloud_fraction.clamp(0.0, 1.0);
                    base_sum += w * (b.base_asl_m + 0.20 * (b.top_asl_m - b.base_asl_m).max(100.0));
                    w_sum += w;
                }
            }
        }

        let local_frac = if w_sum > 0.0 {
            (frac_sum / w_sum).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let deck_z = if w_sum > 0.0 {
            (base_sum / w_sum).max(ground_z + 150.0)
        } else {
            default_base_m
        };

        let ray = evaluate_corridor_ray(
            terrain,
            corridor,
            origin_lat,
            origin_lon,
            deck_z,
            sun_azimuth_deg,
            sun_elevation_deg,
        );

        // Ideal canvas score peaks when local cloud cover is 25%..75% (textured canvas without sealing sky)
        // AND the distant corridor delivers warm/crimson direct rays onto the underside of the deck!
        let coverage_bell = if local_frac < 0.03 {
            0.0
        } else if local_frac <= 0.55 {
            (local_frac / 0.55).powf(0.75)
        } else {
            (1.0 - (local_frac - 0.55) / 0.65).clamp(0.28, 1.0)
        };

        // Warmth bonus for golden/rose hues (< 3800 K or high Chappuis index)
        let warmth_factor = (1.0 + 0.35 * ((4200.0 - ray.beam_color_kelvin) / 2000.0).clamp(0.0, 1.0)
            + 0.22 * ray.chappuis_rose_index)
            .min(1.45);

        // Against a darkening civil-twilight sky (sun_el < +2°), red-channel crimson beam radiance
        // produces intense visual contrast on high cloud bases even when 550nm photopic lux is low.
        let red_warm_beam = (ray.expected_beam_rgb[0] as f64).max(ray.transmittance_expected);
        let twilight_sky_ref = 0.055
            + 0.42 * ((sun_elevation_deg + 2.5) / 8.5).clamp(0.0, 1.0).powi(2);
        let beam_contrast = (red_warm_beam / twilight_sky_ref).clamp(0.0, 1.0);

        let glow_score = (72.0 * coverage_bell * beam_contrast * warmth_factor).clamp(0.0, 100.0);

        if glow_score > best_score {
            best_score = glow_score;
            best_band = name.to_string();
        }

        decks.push(OverheadDeckGlow {
            band: name.to_string(),
            deck_altitude_asl_m: deck_z.round(),
            local_cloud_fraction: (local_frac * 1000.0).round() / 1000.0,
            corridor_direct_prob: ray.direct_beam_probability,
            corridor_transmittance: ray.transmittance_expected,
            underlit_rgb: ray.expected_beam_rgb,
            color_kelvin: ray.beam_color_kelvin,
            canvas_glow_score: (glow_score * 10.0).round() / 10.0,
        });
    }

    // Weighted combination of best deck + secondary textural layer
    let mut sorted_scores: Vec<f64> = decks.iter().map(|d| d.canvas_glow_score).collect();
    sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let composite = (sorted_scores.first().copied().unwrap_or(0.0) * 0.82
        + sorted_scores.get(1).copied().unwrap_or(0.0) * 0.18)
        .clamp(0.0, 100.0);

    OverheadCanvasReport {
        decks,
        composite_canvas_score: (composite * 10.0).round() / 10.0,
        dominant_glow_band: best_band,
    }
}

/// CPU (`rayon`) generator for the `az_bins x alt_bins` `CorridorRadiativeLut` (`4.6 km -> 380 km`).
pub fn compute_corridor_radiative_lut_cpu(
    terrain: &HybridTerrain,
    corridor: &AtmosphericCorridorState,
    origin_lat: f64,
    origin_lon: f64,
    sun_elevation_deg: f64,
    az_bins: usize,
    alt_bins: usize,
) -> CorridorRadiativeLut {
    let az_bins = az_bins.max(16);
    let alt_bins = alt_bins.max(8);
    let start_dist_m = DEFAULT_LUT_START_DISTANCE_M;
    let max_terrain_m = 200_000.0;
    let max_corridor_m = corridor.max_corridor_km.clamp(40.0, 420.0) * 1000.0;
    // Clamp radiative angle floor slightly (-2.6°) so upper-altitude cloud bins receive true twilight transmittance
    let lut_el_deg = sun_elevation_deg.clamp(-2.8, 89.0);

    let altitudes: Vec<f64> = (0..alt_bins)
        .map(|iv| CorridorRadiativeLut::bin_to_altitude_m(iv, alt_bins))
        .collect();

    // Parallel sweep over azimuth columns: each azimuth walks SRTM once for all `alt_bins` horizons,
    // then integrates `alt_bins` curved-Earth atmospheric+cloud rays from `4.6 km -> 380 km`.
    let cols: Vec<Vec<[u8; 4]>> = (0..az_bins)
        .into_par_iter()
        .map(|iu| {
            let az_deg = (iu as f64) * 360.0 / (az_bins as f64);
            let k_ref = corridor.effective_refraction_k(az_deg, 2_000.0);
            let mut far_horizons = vec![0.0_f64; alt_bins];
            cast_far_field_multi_alt_horizon(
                terrain,
                origin_lat,
                origin_lon,
                az_deg,
                k_ref,
                start_dist_m,
                max_terrain_m,
                &altitudes,
                &mut far_horizons,
            );

            let mut col = Vec::with_capacity(alt_bins);
            for iv in 0..alt_bins {
                let z_asl = altitudes[iv];
                let (_, expected_trans, _, _, _, _, _, _, _) = integrate_atmospheric_corridor_range(
                    corridor,
                    z_asl,
                    az_deg,
                    lut_el_deg,
                    k_ref,
                    start_dist_m,
                    max_corridor_m,
                    false,
                );
                let (rgb, _, _, _) = spectral6_to_rgb_and_kelvin(&expected_trans);
                col.push([
                    (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    CorridorRadiativeLut::encode_horizon_u8(far_horizons[iv]),
                ]);
            }
            col
        })
        .collect();

    let mut rgba8 = vec![0u8; az_bins * alt_bins * 4];
    for iu in 0..az_bins {
        for iv in 0..alt_bins {
            let off = (iv * az_bins + iu) * 4;
            rgba8[off..off + 4].copy_from_slice(&cols[iu][iv]);
        }
    }

    CorridorRadiativeLut {
        az_bins,
        alt_bins,
        start_distance_m: start_dist_m,
        max_altitude_asl_m: LUT_MAX_ALTITUDE_ASL_M,
        altitude_gamma: LUT_ALTITUDE_GAMMA,
        horizon_min_deg: LUT_HORIZON_MIN_DEG,
        horizon_span_deg: LUT_HORIZON_SPAN_DEG,
        sun_elevation_deg,
        rgba8,
    }
}

/// Computes the `CorridorRadiativeLut` (`180 x 32` RGBA8) on Apple Silicon Metal GPU when available,
/// falling back automatically to multi-core `rayon` CPU execution.
pub fn compute_corridor_radiative_lut(
    terrain: &HybridTerrain,
    corridor: &AtmosphericCorridorState,
    origin_lat: f64,
    origin_lon: f64,
    sun_elevation_deg: f64,
    az_bins: usize,
    alt_bins: usize,
) -> CorridorRadiativeLut {
    #[cfg(target_os = "macos")]
    {
        if let Some(lut) = crate::gpu_metal::dispatch_corridor_lut_gpu(
            terrain,
            corridor,
            origin_lat,
            origin_lon,
            sun_elevation_deg,
            az_bins,
            alt_bins,
        ) {
            return lut;
        }
    }
    compute_corridor_radiative_lut_cpu(
        terrain,
        corridor,
        origin_lat,
        origin_lon,
        sun_elevation_deg,
        az_bins,
        alt_bins,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pp_terrain::SrtmCache;

    fn flat_sea_terrain() -> HybridTerrain {
        HybridTerrain::new(None, SrtmCache::new(Vec::new()))
    }

    #[test]
    fn test_half_dome_ray_clears_central_valley_tule_fog() {
        // Half Dome summit (~2,693 m ASL) looking WSW (252°) at sunset (sun_el = +0.5°).
        // A dense Central Valley Tule fog bank (0..650 m ASL, cloud_fraction = 0.98, tau = 35)
        // sits 80 km away. Because of Half Dome's elevation + Earth curvature, z_ray(80 km) > 3,700 m ASL!
        // Meanwhile, Yosemite Valley floor or a low foothill at 350 m ASL looking at 0.0° into a nearby
        // 15 km fog bank DOES intersect the fog and gets heavily blocked.
        let terrain = flat_sea_terrain();
        let state = AtmosphericCorridorState {
            azimuth_center_deg: 252.0,
            max_corridor_km: 360.0,
            stations: vec![
                CorridorStationProfile {
                    distance_km: 0.0,
                    azimuth_deg: Some(252.0),
                    lat: 37.7459,
                    lon: -119.5332,
                    surface_elevation_m: 2000.0,
                    surface_temp_c: 10.0,
                    surface_pressure_hpa: 800.0,
                    lapse_rate_k_km: 6.5,
                    inversion_base_asl_m: 0.0,
                    inversion_top_asl_m: 0.0,
                    inversion_strength_k: 0.0,
                    boundary_layer_top_asl_m: 2800.0,
                    aod_550nm: 0.05,
                    angstrom_exponent: 1.3,
                    aerosol_scale_height_m: 1600.0,
                    ozone_du: 300.0,
                    cloud_bands: vec![],
                },
                CorridorStationProfile {
                    distance_km: 80.0,
                    azimuth_deg: Some(252.0),
                    lat: 37.50,
                    lon: -120.40,
                    surface_elevation_m: 60.0,
                    surface_temp_c: 8.0,
                    surface_pressure_hpa: 1016.0,
                    lapse_rate_k_km: 5.5,
                    inversion_base_asl_m: 100.0,
                    inversion_top_asl_m: 700.0,
                    inversion_strength_k: 6.0,
                    boundary_layer_top_asl_m: 750.0,
                    aod_550nm: 0.12,
                    angstrom_exponent: 1.3,
                    aerosol_scale_height_m: 1200.0,
                    ozone_du: 300.0,
                    cloud_bands: vec![CloudLayerBand {
                        label: "fog".to_string(),
                        base_asl_m: 50.0,
                        top_asl_m: 650.0,
                        cloud_fraction: 0.98,
                        optical_depth_vertical: 35.0,
                        aspect_ratio_h_over_d: 0.05,
                        characteristic_cell_km: 25.0,
                        ice_fraction: 0.0,
                    }],
                },
            ],
        };

        let hd_res = evaluate_corridor_ray(&terrain, &state, 37.7459, -119.5332, 2693.0, 252.0, 0.6);
        assert!(
            hd_res.direct_beam_probability > 0.95,
            "Half Dome summit ray must pass cleanly above 650m Central Valley Tule fog! Got prob={}",
            hd_res.direct_beam_probability
        );
        assert!(
            hd_res.segments.iter().any(|s| s.status_note.contains("clears underlying fog")),
            "Expected segment diagnostic confirming ray clears underlying fog: {:?}",
            hd_res.segments
        );
    }

    #[test]
    fn test_distant_pacific_cirrus_and_cumulus_grazing_aspect_ratio() {
        let terrain = flat_sea_terrain();

        // Case 1: Offshore Pacific high cirrus shield at 260 km (8,500..10,800 m ASL)
        // A 0.3° ray from Half Dome (2,693 m) reaches z_ray(260 km) ≈ 2693 + 1360 + 4560 = 8,613 m -> enters cirrus!
        let mut cirrus_state = AtmosphericCorridorState::clear_standard(37.7459, -119.5332, 2693.0, 252.0);
        cirrus_state.stations.push(CorridorStationProfile {
            distance_km: 260.0,
            azimuth_deg: Some(252.0),
            lat: 37.0,
            lon: -122.4,
            surface_elevation_m: 0.0,
            surface_temp_c: 14.0,
            surface_pressure_hpa: 1015.0,
            lapse_rate_k_km: 6.5,
            inversion_base_asl_m: 0.0,
            inversion_top_asl_m: 0.0,
            inversion_strength_k: 0.0,
            boundary_layer_top_asl_m: 1200.0,
            aod_550nm: 0.06,
            angstrom_exponent: 0.8,
            aerosol_scale_height_m: 1500.0,
            ozone_du: 310.0,
            cloud_bands: vec![CloudLayerBand {
                label: "high".to_string(),
                base_asl_m: 7_500.0,
                top_asl_m: 11_200.0,
                cloud_fraction: 0.85,
                optical_depth_vertical: 2.2,
                aspect_ratio_h_over_d: 0.12,
                characteristic_cell_km: 15.0,
                ice_fraction: 1.0,
            }],
        });

        let clear_res = evaluate_corridor_ray(
            &terrain,
            &AtmosphericCorridorState::clear_standard(37.7459, -119.5332, 2693.0, 252.0),
            37.7459,
            -119.5332,
            2693.0,
            252.0,
            0.35,
        );
        let cir_res = evaluate_corridor_ray(&terrain, &cirrus_state, 37.7459, -119.5332, 2693.0, 252.0, 0.35);

        assert!(
            cir_res.transmittance_expected < clear_res.transmittance_expected * 0.78,
            "260 km offshore cirrus must attenuate low-angle sunset ray: cirrus={} vs clear={}",
            cir_res.transmittance_expected,
            clear_res.transmittance_expected
        );
        // Chappuis + Rayleigh reddening at 0.35° must produce warm low-Kelvin alpenglow (< 3000 K)
        assert!(
            clear_res.beam_color_kelvin < 3000.0 && clear_res.chappuis_rose_index > 0.08,
            "Expected warm twilight Kelvin + Chappuis rose index, got K={} rose={}",
            clear_res.beam_color_kelvin,
            clear_res.chappuis_rose_index
        );

        // Case 2: Grazing 35% cumulus turrets (small cell_km = 1.2 km, tall aspect) block more direct rays
        // along a shallow chord than wide flat sheets of identical 35% cloud fraction (cell_km = 22 km).
        let make_mid_state = |aspect: f64, cell_km: f64| AtmosphericCorridorState {
            azimuth_center_deg: 252.0,
            max_corridor_km: 200.0,
            stations: vec![CorridorStationProfile {
                distance_km: 25.0,
                azimuth_deg: Some(252.0),
                lat: 37.7,
                lon: -119.8,
                surface_elevation_m: 1500.0,
                surface_temp_c: 12.0,
                surface_pressure_hpa: 850.0,
                lapse_rate_k_km: 6.5,
                inversion_base_asl_m: 0.0,
                inversion_top_asl_m: 0.0,
                inversion_strength_k: 0.0,
                boundary_layer_top_asl_m: 3200.0,
                aod_550nm: 0.06,
                angstrom_exponent: 1.2,
                aerosol_scale_height_m: 1500.0,
                ozone_du: 300.0,
                cloud_bands: vec![CloudLayerBand {
                    label: "mid".to_string(),
                    base_asl_m: 2_600.0,
                    top_asl_m: 4_200.0,
                    cloud_fraction: 0.35,
                    optical_depth_vertical: 12.0,
                    aspect_ratio_h_over_d: aspect,
                    characteristic_cell_km: cell_km,
                    ice_fraction: 0.2,
                }],
            }],
        };

        let cum_res = evaluate_corridor_ray(&terrain, &make_mid_state(1.1, 1.2), 37.7459, -119.5332, 2693.0, 252.0, 1.2);
        let strat_res = evaluate_corridor_ray(&terrain, &make_mid_state(0.06, 22.0), 37.7459, -119.5332, 2693.0, 252.0, 1.2);
        assert!(
            cum_res.direct_beam_probability < strat_res.direct_beam_probability,
            "Tall cellular turrets must have lower grazing gap survival than flat wide sheets: cum={} vs strat={}",
            cum_res.direct_beam_probability,
            strat_res.direct_beam_probability
        );
        // Bimodal spread check: P90 > P10 on broken 35% deck
        assert!(strat_res.transmittance_p90 > strat_res.transmittance_p10 * 2.0);
    }

    #[test]
    fn test_corridor_radiative_lut_and_post_sunset_cloud_canvas() {
        let terrain = flat_sea_terrain();
        let mut state = AtmosphericCorridorState::clear_standard(37.7459, -119.5332, 1200.0, 252.0);
        state.stations[0].cloud_bands.push(CloudLayerBand {
            label: "high".to_string(),
            base_asl_m: 8_200.0,
            top_asl_m: 9_800.0,
            cloud_fraction: 0.48,
            optical_depth_vertical: 1.5,
            aspect_ratio_h_over_d: 0.14,
            characteristic_cell_km: 10.0,
            ice_fraction: 1.0,
        });

        // At sun_elevation = -1.6° (below the -1.28° sea-limb upper-disk limit for 1,200 m ground!),
        // ground direct beam is 0, while the 8,500 m overhead cloud canvas (-2.70° limb) STILL receives crimson corridor sunlight!
        let ground_ray = evaluate_corridor_ray(&terrain, &state, 37.7459, -119.5332, 1200.0, 252.0, -1.6);
        let canvas = evaluate_overhead_cloud_canvas(&terrain, &state, 37.7459, -119.5332, 252.0, -1.6);
        assert_eq!(ground_ray.terrain_disk_visibility, 0.0);
        assert!(
            canvas.composite_canvas_score > 25.0 && canvas.dominant_glow_band == "high",
            "Post-sunset (-1.6°) 8.2km overhead cirrus must still catch underlit twilight fire: {:?}",
            canvas
        );

        let lut = compute_corridor_radiative_lut(&terrain, &state, 37.7459, -119.5332, 1.5, 90, 16);
        assert_eq!(lut.rgba8.len(), 90 * 16 * 4);
        let (rgb_low, h_low) = lut.sample(252.0, 500.0);
        let (rgb_high, h_high) = lut.sample(252.0, 9_000.0);
        assert!(rgb_high[0] >= rgb_low[0], "Higher altitude has less Rayleigh/aerosol attenuation");
        assert!(h_high < h_low, "Higher altitude has lower (more negative) sea-horizon depression angle");
    }
}
