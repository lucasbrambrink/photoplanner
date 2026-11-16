//! Native Apple Silicon Metal GPU Compute Backend (`Metal.framework` Unified Memory).
//!
//! Accelerates:
//! 1. Batch 1m Hierarchical Max-Mipmap + Far-Field SRTM curved-Earth horizon ray-casting
//!    (`compute_horizon_profile`, `compute_horizon_profiles_batch`, `find_subject_golden_hour_windows`)
//! 2. 2D Directional Ray-Cast Topographic Shadow & Golden-Hour Spotlight Rasterization + RGB8 PNG synthesis
//!    (`compute_shadow_and_light_map`)
//! 3. Batch Multi-Route Trail Viewpoint 3D Camera-to-Crown-Rim Line-of-Sight & Relief Scoring
//!    (`cast_camera_line_of_sight_batch`)
//! 4. Hybrid 1m 3DEP + SRTM 2D Terrain Patch Resampling (`compute_terrain_patch_elevations`)
//!
//! Uses `MTLResourceStorageModeShared` unified memory on Apple Silicon and caches uploaded
//! `HybridTerrain` pyramids by `terrain.cache_id` so multi-stage planners (`shoot-landmark`,
//! `trail-plan`, `notch-scan`) pay zero repeated upload overhead.

#![cfg(target_os = "macos")]

use crate::corridor::{
    AtmosphericCorridorState, CorridorRadiativeLut, DEFAULT_LUT_START_DISTANCE_M,
    LUT_ALTITUDE_GAMMA, LUT_HORIZON_MIN_DEG, LUT_HORIZON_SPAN_DEG, LUT_MAX_ALTITUDE_ASL_M,
};
use crate::{HorizonSample, LandmarkTarget3D, ShadowLightMapStats};
use anyhow::Result;
use image::{ImageBuffer, Rgb};
use pp_solar::SolarPosition;
use pp_terrain::{meters_per_deg_lon, HybridTerrain, MaxMipmapGrid1m};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

extern "C" {
    fn pp_metal_create(
        msl_source: *const c_char,
        device_name_out: *mut c_char,
        name_cap: usize,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn pp_metal_destroy(ctx_ptr: *mut c_void);
    fn pp_metal_create_pipeline(
        ctx_ptr: *mut c_void,
        fn_name: *const c_char,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn pp_metal_destroy_pipeline(pipe_ptr: *mut c_void);
    fn pp_metal_create_buffer(
        ctx_ptr: *mut c_void,
        initial_bytes: *const c_void,
        byte_len: usize,
    ) -> *mut c_void;
    fn pp_metal_buffer_contents(buf_ptr: *mut c_void) -> *mut c_void;
    fn pp_metal_destroy_buffer(buf_ptr: *mut c_void);
    fn pp_metal_dispatch_1d(
        ctx_ptr: *mut c_void,
        pipe_ptr: *mut c_void,
        buffers: *const *mut c_void,
        num_buffers: usize,
        num_threads: usize,
    ) -> c_int;
    fn pp_metal_dispatch_2pass_2d(
        ctx_ptr: *mut c_void,
        pipe1_ptr: *mut c_void,
        bufs1: *const *mut c_void,
        nbufs1: usize,
        pipe2_ptr: *mut c_void,
        bufs2: *const *mut c_void,
        nbufs2: usize,
        width: usize,
        height: usize,
    ) -> c_int;
}

const METAL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant float METERS_PER_DEG_LAT_F = 111132.92f;
constant float INV_METERS_PER_DEG_LAT_F = 1.0f / 111132.92f;
constant float CURVATURE_REFRACTION_F = 6.75e-8f;

struct GpuSceneMeta {
    uint has_near_1m;
    uint w0;
    uint h0;
    uint off_l0;
    uint w2;
    uint h2;
    uint off_l2;
    uint w4;
    uint h4;
    uint off_l4;
    float global_min_z;
    float global_max_z;
    float near_m_per_lon;
    int ref_lat_int;
    float ref_lat_frac;
    int ref_lon_int;
    float ref_lon_frac;
    uint num_srtm_tiles;
    uint _pad0;
    uint _pad1;
};

struct GpuSrtmTileMeta {
    int lat_floor;
    int lon_floor;
    uint samples;
    uint side;
    uint data_offset;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct GpuHorizonRayReq {
    float dlat_ref;
    float dlon_ref;
    float ox_near_m;
    float oy_near_m;
    float z0;
    float dir_east;
    float dir_north;
    float inv_m_per_lon;
    float max_distance_m;
    float _pad0;
    float _pad1;
    float _pad2;
};

struct GpuHorizonRayOut {
    float horizon_elevation_deg;
    float occluder_distance_m;
    float occluder_height_m;
    float _pad0;
};

struct GpuShadowParams {
    uint nx;
    uint ny;
    uint step_m;
    uint write_png;
    float sun_dir_east;
    float sun_dir_north;
    float sun_vec_e;
    float sun_vec_n;
    float sun_vec_u;
    float sun_el_deg;
    float sun_slope;
    float max_distance_m;
};

struct GpuShadowPixel {
    uint is_lit;
    float cos_inc;
    float margin_deg;
    float z0;
};

struct GpuCrownRimPt {
    float dlat_ref;
    float dlon_ref;
    float z_m;
    float _pad;
};

struct GpuLosTargetParams {
    uint num_crown_pts;
    uint num_viewpoints;
    float summit_dlat_ref;
    float summit_dlon_ref;
    float base_valley_z;
    float _pad0;
    float _pad1;
    float _pad2;
    GpuCrownRimPt crown_pts[12];
};

struct GpuLosVpReq {
    float dlat_ref;
    float dlon_ref;
    float z_m;
    float m_per_lon;
};

struct GpuLosVpOut {
    float visible_relief_m;
    float summit_slope;
    float _pad0;
    float _pad1;
};

struct GpuPatchParams {
    uint grid_size;
    float center_dlat_ref;
    float center_dlon_ref;
    float half_span_m;
    float step_m;
    float inv_m_per_lon;
    float _pad0;
    float _pad1;
};

inline bool contains_near_1m(constant GpuSceneMeta& meta, float x_m, float y_m) {
    return (meta.has_near_1m != 0u) &&
           (x_m >= 0.0f) &&
           (y_m >= 0.0f) &&
           (x_m < float(meta.w0 - 1u)) &&
           (y_m < float(meta.h0 - 1u));
}

inline float sample_1m_bilinear(
    constant GpuSceneMeta& meta,
    device const float* near_mip,
    float x_m,
    float y_m
) {
    float x = clamp(x_m, 0.0f, float(meta.w0 - 2u));
    float y = clamp(y_m, 0.0f, float(meta.h0 - 2u));
    uint x0 = uint(floor(x));
    uint y0 = uint(floor(y));
    float dx = x - float(x0);
    float dy = y - float(y0);

    uint w = meta.w0;
    uint base = meta.off_l0;
    float z00 = near_mip[base + y0 * w + x0];
    float z10 = near_mip[base + y0 * w + (x0 + 1u)];
    float z01 = near_mip[base + (y0 + 1u) * w + x0];
    float z11 = near_mip[base + (y0 + 1u) * w + (x0 + 1u)];

    float z0 = fma(z10 - z00, dx, z00);
    float z1 = fma(z11 - z01, dx, z01);
    return fma(z1 - z0, dy, z0);
}

inline float sample_srtm_m(
    constant GpuSceneMeta& meta,
    constant GpuSrtmTileMeta* srtm_tiles,
    device const float* srtm_data,
    float dlat_ref,
    float dlon_ref
) {
    float u_lat = meta.ref_lat_frac + dlat_ref;
    float u_lon = meta.ref_lon_frac + dlon_ref;
    float fl_lat = floor(u_lat);
    float fl_lon = floor(u_lon);
    int lat_floor = meta.ref_lat_int + int(fl_lat);
    int lon_floor = meta.ref_lon_int + int(fl_lon);
    float lat_frac = clamp(u_lat - fl_lat, 0.0f, 1.0f);
    float lon_frac = clamp(u_lon - fl_lon, 0.0f, 1.0f);

    for (uint t = 0u; t < meta.num_srtm_tiles; t++) {
        constant GpuSrtmTileMeta& tile = srtm_tiles[t];
        if (tile.lat_floor == lat_floor && tile.lon_floor == lon_floor) {
            float s = float(tile.samples);
            float row_f = (1.0f - lat_frac) * s;
            float col_f = lon_frac * s;

            uint r0 = min(uint(floor(row_f)), tile.samples - 1u);
            uint c0 = min(uint(floor(col_f)), tile.samples - 1u);
            uint r1 = r0 + 1u;
            uint c1 = c0 + 1u;

            float dr = row_f - float(r0);
            float dc = col_f - float(c0);

            uint side = tile.side;
            uint off = tile.data_offset;
            float z00 = srtm_data[off + r0 * side + c0];
            float z01 = srtm_data[off + r0 * side + c1];
            float z10 = srtm_data[off + r1 * side + c0];
            float z11 = srtm_data[off + r1 * side + c1];

            float z0 = fma(z01 - z00, dc, z00);
            float z1 = fma(z11 - z10, dc, z10);
            return fma(z1 - z0, dr, z0);
        }
    }

    return 0.0f;
}

inline float sample_hybrid_m(
    constant GpuSceneMeta& meta,
    device const float* near_mip,
    constant GpuSrtmTileMeta* srtm_tiles,
    device const float* srtm_data,
    float dlat_ref,
    float dlon_ref
) {
    if (meta.has_near_1m != 0u) {
        float x_m = dlon_ref * meta.near_m_per_lon;
        float y_m = dlat_ref * METERS_PER_DEG_LAT_F;
        if (contains_near_1m(meta, x_m, y_m)) {
            return sample_1m_bilinear(meta, near_mip, x_m, y_m);
        }
    }
    return sample_srtm_m(meta, srtm_tiles, srtm_data, dlat_ref, dlon_ref);
}

inline GpuHorizonRayOut trace_horizon_ray(
    constant GpuSceneMeta& meta,
    device const float* near_mip,
    constant GpuSrtmTileMeta* srtm_tiles,
    device const float* srtm_data,
    float dlat_ref,
    float dlon_ref,
    float ox,
    float oy,
    float z0,
    float dir_east,
    float dir_north,
    float inv_m_per_lon,
    float max_distance_m
) {
    float max_slope = -1.0e30f;
    float best_dist_m = 10.0f;
    float best_height_m = z0;
    float d = 4.0f;

    if (meta.has_near_1m != 0u) {
        float global_max_z = meta.global_max_z;
        while (d <= max_distance_m) {
            float rx = fma(d, dir_east, ox);
            float ry = fma(d, dir_north, oy);
            if (!contains_near_1m(meta, rx, ry)) {
                break;
            }

            float drop = CURVATURE_REFRACTION_F * d * d;
            if (max_slope > 0.0f && (global_max_z - drop - z0) <= max_slope * d) {
                d += 32.0f;
                continue;
            }

            uint cx16 = min(uint(rx) >> 4u, meta.w4 - 1u);
            uint cy16 = min(uint(ry) >> 4u, meta.h4 - 1u);
            float block16_max_z = near_mip[meta.off_l4 + cy16 * meta.w4 + cx16];
            if ((block16_max_z - drop - z0) <= max_slope * d) {
                d += 14.0f;
                continue;
            }

            uint cx4 = min(uint(rx) >> 2u, meta.w2 - 1u);
            uint cy4 = min(uint(ry) >> 2u, meta.h2 - 1u);
            float block4_max_z = near_mip[meta.off_l2 + cy4 * meta.w2 + cx4];
            if ((block4_max_z - drop - z0) <= max_slope * d) {
                d += 3.5f;
                continue;
            }

            float z = sample_1m_bilinear(meta, near_mip, rx, ry);
            float slope = (z - drop - z0) / d;
            if (slope > max_slope) {
                max_slope = slope;
                best_dist_m = d;
                best_height_m = z;
            }
            d += 1.0f;
        }
    }

    while (d <= max_distance_m) {
        float sample_dlat = fma(d * dir_north, INV_METERS_PER_DEG_LAT_F, dlat_ref);
        float sample_dlon = fma(d * dir_east, inv_m_per_lon, dlon_ref);
        float z = sample_srtm_m(meta, srtm_tiles, srtm_data, sample_dlat, sample_dlon);
        float drop = CURVATURE_REFRACTION_F * d * d;
        float slope = (z - drop - z0) / d;
        if (slope > max_slope) {
            max_slope = slope;
            best_dist_m = d;
            best_height_m = z;
        }
        if (d < 8000.0f) {
            d += 15.0f;
        } else if (d < 25000.0f) {
            d += 30.0f;
        } else {
            d += 60.0f;
        }
    }

    GpuHorizonRayOut out;
    out.horizon_elevation_deg = atan(max_slope) * 57.29577951308232f;
    out.occluder_distance_m = best_dist_m;
    out.occluder_height_m = best_height_m;
    out._pad0 = 0.0f;
    return out;
}

kernel void kernel_batch_horizon_rays(
    constant GpuSceneMeta& meta [[buffer(0)]],
    device const float* near_mip [[buffer(1)]],
    constant GpuSrtmTileMeta* srtm_tiles [[buffer(2)]],
    device const float* srtm_data [[buffer(3)]],
    device const GpuHorizonRayReq* reqs [[buffer(4)]],
    device GpuHorizonRayOut* outs [[buffer(5)]],
    constant uint& num_rays [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_rays) return;
    GpuHorizonRayReq r = reqs[gid];
    outs[gid] = trace_horizon_ray(
        meta,
        near_mip,
        srtm_tiles,
        srtm_data,
        r.dlat_ref,
        r.dlon_ref,
        r.ox_near_m,
        r.oy_near_m,
        r.z0,
        r.dir_east,
        r.dir_north,
        r.inv_m_per_lon,
        r.max_distance_m
    );
}

kernel void kernel_shadow_and_light_map(
    constant GpuSceneMeta& meta [[buffer(0)]],
    device const float* near_mip [[buffer(1)]],
    constant GpuSrtmTileMeta* srtm_tiles [[buffer(2)]],
    device const float* srtm_data [[buffer(3)]],
    constant GpuShadowParams& params [[buffer(4)]],
    device GpuShadowPixel* pixels [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.nx || gid.y >= params.ny) return;

    float step_f = float(params.step_m);
    float x_m = float(gid.x * params.step_m);
    float y_m = float(gid.y * params.step_m);

    float dlat_ref = y_m * INV_METERS_PER_DEG_LAT_F;
    float inv_near_lon = 1.0f / max(1.0f, meta.near_m_per_lon);
    float dlon_ref = x_m * inv_near_lon;

    // Account for local latitude cosine variation in ray stepping
    float lat_rad = (float(meta.ref_lat_int) + meta.ref_lat_frac + dlat_ref) * 0.017453292519943295f;
    float local_m_per_lon = max(1.0f, 111412.84f * cos(lat_rad));
    float inv_local_m_per_lon = 1.0f / local_m_per_lon;

    float z0 = sample_1m_bilinear(meta, near_mip, x_m, y_m);

    GpuHorizonRayOut hor = trace_horizon_ray(
        meta,
        near_mip,
        srtm_tiles,
        srtm_data,
        dlat_ref,
        dlon_ref,
        x_m,
        y_m,
        z0 + 0.5f,
        params.sun_dir_east,
        params.sun_dir_north,
        inv_local_m_per_lon,
        params.max_distance_m
    );

    bool is_lit = (params.sun_el_deg >= hor.horizon_elevation_deg) && (params.sun_slope > -0.02f);

    // Central-difference surface normal matching ElevationSurface::surface_normal
    float dlat_step = step_f * INV_METERS_PER_DEG_LAT_F;
    float dlon_step = step_f * inv_local_m_per_lon;

    float z_e = sample_hybrid_m(meta, near_mip, srtm_tiles, srtm_data, dlat_ref, dlon_ref + dlon_step);
    float z_w = sample_hybrid_m(meta, near_mip, srtm_tiles, srtm_data, dlat_ref, dlon_ref - dlon_step);
    float z_n = sample_hybrid_m(meta, near_mip, srtm_tiles, srtm_data, dlat_ref + dlat_step, dlon_ref);
    float z_s = sample_hybrid_m(meta, near_mip, srtm_tiles, srtm_data, dlat_ref - dlat_step, dlon_ref);

    float inv_2step = 1.0f / (2.0f * step_f);
    float nx = -(z_e - z_w) * inv_2step;
    float ny = -(z_n - z_s) * inv_2step;
    float nz = 1.0f;
    float inv_norm = rsqrt(nx * nx + ny * ny + 1.0f);
    nx *= inv_norm;
    ny *= inv_norm;
    nz *= inv_norm;

    float cos_inc = max(0.0f, nx * params.sun_vec_e + ny * params.sun_vec_n + nz * params.sun_vec_u);
    float margin_deg = params.sun_el_deg - hor.horizon_elevation_deg;

    uint idx = gid.y * params.nx + gid.x;
    GpuShadowPixel px;
    px.is_lit = is_lit ? 1u : 0u;
    px.cos_inc = cos_inc;
    px.margin_deg = margin_deg;
    px.z0 = z0;
    pixels[idx] = px;
}

kernel void kernel_spotlight_and_png(
    constant GpuSceneMeta& meta [[buffer(0)]],
    constant GpuShadowParams& params [[buffer(1)]],
    device const GpuShadowPixel* pixels [[buffer(2)]],
    device float* spotlight_scores [[buffer(3)]],
    device uchar* rgb_out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.nx || gid.y >= params.ny) return;

    uint idx = gid.y * params.nx + gid.x;
    GpuShadowPixel px = pixels[idx];
    float spot = 0.0f;

    if (gid.x >= 8u && gid.y >= 8u && (gid.x + 8u) < params.nx && (gid.y + 8u) < params.ny) {
        if (px.is_lit != 0u && px.margin_deg <= 1.8f && px.cos_inc >= 0.25f) {
            uint shadowed_neighbors = 0u;
            uint total_neighbors = 0u;
            for (int dy = -8; dy <= 8; dy += 4) {
                for (int dx = -8; dx <= 8; dx += 4) {
                    if (dx == 0 && dy == 0) continue;
                    uint nidx = uint(int(gid.y) + dy) * params.nx + uint(int(gid.x) + dx);
                    total_neighbors += 1u;
                    if (pixels[nidx].is_lit == 0u) {
                        shadowed_neighbors += 1u;
                    }
                }
            }
            float ratio = float(shadowed_neighbors) / float(max(1u, total_neighbors));
            if (ratio >= 0.20f && ratio <= 0.80f) {
                float clearance_factor = (1.8f - max(0.0f, px.margin_deg)) / 1.8f;
                spot = ratio * px.cos_inc * clearance_factor * 100.0f;
            }
        }
    }
    spotlight_scores[idx] = spot;

    if (params.write_png != 0u) {
        float denom = max(1.0f, meta.global_max_z - meta.global_min_z);
        float rel_z = clamp((px.z0 - meta.global_min_z) / denom, 0.0f, 1.0f);
        uchar r_u8, g_u8, b_u8;
        if (px.is_lit != 0u) {
            float spot_boost = clamp(spot / 35.0f, 0.0f, 1.0f);
            float intensity = clamp(0.30f + 0.70f * px.cos_inc, 0.0f, 1.0f);
            r_u8 = uchar(min(255.0f, 225.0f * intensity + 20.0f * rel_z + 30.0f * spot_boost));
            g_u8 = uchar(min(255.0f, 160.0f * intensity + 22.0f * rel_z + 45.0f * spot_boost));
            b_u8 = uchar(clamp(92.0f * intensity + 18.0f * rel_z - 25.0f * spot_boost, 0.0f, 255.0f));
        } else {
            float amb = 0.22f + 0.52f * rel_z;
            r_u8 = uchar(42.0f * amb);
            g_u8 = uchar(64.0f * amb);
            b_u8 = uchar(105.0f * amb);
        }
        uint img_y = params.ny - 1u - gid.y;
        uint out_base = (img_y * params.nx + gid.x) * 3u;
        rgb_out[out_base + 0u] = r_u8;
        rgb_out[out_base + 1u] = g_u8;
        rgb_out[out_base + 2u] = b_u8;
    }
}

kernel void kernel_batch_camera_los(
    constant GpuSceneMeta& meta [[buffer(0)]],
    device const float* near_mip [[buffer(1)]],
    constant GpuSrtmTileMeta* srtm_tiles [[buffer(2)]],
    device const float* srtm_data [[buffer(3)]],
    constant GpuLosTargetParams& target [[buffer(4)]],
    device const GpuLosVpReq* vp_reqs [[buffer(5)]],
    device GpuLosVpOut* vp_outs [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= target.num_viewpoints) return;

    GpuLosVpReq vp = vp_reqs[gid];
    float inv_m_per_lon = 1.0f / max(1.0f, vp.m_per_lon);
    float z_eye = vp.z_m + 1.65f;

    float visible_relief_m = 0.0f;
    float summit_slope = 0.0f;

    for (uint c = 0u; c < target.num_crown_pts; c++) {
        GpuCrownRimPt cp = target.crown_pts[c];
        float c_dy = (cp.dlat_ref - vp.dlat_ref) * METERS_PER_DEG_LAT_F;
        float c_dx = (cp.dlon_ref - vp.dlon_ref) * vp.m_per_lon;
        float c_dist = sqrt(c_dx * c_dx + c_dy * c_dy);
        if (c_dist < 400.0f) continue;

        float inv_cdist = 1.0f / c_dist;
        float c_dir_east = c_dx * inv_cdist;
        float c_dir_north = c_dy * inv_cdist;

        float drop_target = CURVATURE_REFRACTION_F * c_dist * c_dist;
        float eff_tz = cp.z_m - drop_target;
        float t_slope = (eff_tz - z_eye) * inv_cdist;

        float stop_d = max(300.0f, c_dist - 400.0f);
        float max_int_slope = -1.0e30f;
        float d = 280.0f;

        while (d < stop_d) {
            float s_dlat = fma(d * c_dir_north, INV_METERS_PER_DEG_LAT_F, vp.dlat_ref);
            float s_dlon = fma(d * c_dir_east, inv_m_per_lon, vp.dlon_ref);
            float z = sample_hybrid_m(meta, near_mip, srtm_tiles, srtm_data, s_dlat, s_dlon);
            float drop = CURVATURE_REFRACTION_F * d * d;
            float slope = (z - drop - z_eye) / d;
            if (slope > max_int_slope) {
                max_int_slope = slope;
            }
            d += (d < 2500.0f) ? 18.0f : 35.0f;
        }

        float occluded_z = (max_int_slope < -1.0e20f)
            ? target.base_valley_z
            : max(target.base_valley_z, z_eye + max_int_slope * c_dist + drop_target);

        float rel_m = max(0.0f, cp.z_m - occluded_z);
        if (rel_m > visible_relief_m) {
            visible_relief_m = rel_m;
            summit_slope = t_slope;
        }
    }

    GpuLosVpOut out;
    out.visible_relief_m = visible_relief_m;
    out.summit_slope = summit_slope;
    out._pad0 = 0.0f;
    out._pad1 = 0.0f;
    vp_outs[gid] = out;
}

kernel void kernel_resample_terrain_patch(
    constant GpuSceneMeta& meta [[buffer(0)]],
    device const float* near_mip [[buffer(1)]],
    constant GpuSrtmTileMeta* srtm_tiles [[buffer(2)]],
    device const float* srtm_data [[buffer(3)]],
    constant GpuPatchParams& params [[buffer(4)]],
    device float* out_elevations [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.grid_size || gid.y >= params.grid_size) return;

    float dy_m = -params.half_span_m + float(gid.y) * params.step_m;
    float dx_m = -params.half_span_m + float(gid.x) * params.step_m;

    float s_dlat = params.center_dlat_ref + dy_m * INV_METERS_PER_DEG_LAT_F;
    float s_dlon = params.center_dlon_ref + dx_m * params.inv_m_per_lon;

    float z = sample_hybrid_m(meta, near_mip, srtm_tiles, srtm_data, s_dlat, s_dlon);
    out_elevations[gid.y * params.grid_size + gid.x] = rint(z * 100.0f) * 0.01f;
}

struct GpuCorridorStation {
    float distance_km;
    float azimuth_deg;
    float surface_elev_m;
    float surface_temp_k;
    float surface_pressure_hpa;
    float lapse_rate_k_km;
    float inversion_top_asl_m;
    float inversion_strength_k;
    float pbl_top_asl_m;
    float aod_550nm;
    float angstrom_exp;
    float aerosol_scale_m;
    float ozone_du;
    uint band_offset;
    uint band_count;
    uint _pad0;
};

struct GpuCloudBand {
    float base_asl_m;
    float top_asl_m;
    float cloud_fraction;
    float tau_core_vert;
    float cell_d_m;
    float delta_edd_factor;
    float _pad0;
    float _pad1;
};

struct GpuCorridorLutParams {
    uint az_bins;
    uint alt_bins;
    uint num_stations;
    uint _pad0;
    float origin_dlat_ref;
    float origin_dlon_ref;
    float inv_m_per_lon;
    float lut_el_rad;
    float start_dist_m;
    float max_terrain_m;
    float max_corridor_m;
    float refraction_k;
};

constant float GPU_SPECTRAL_WL_NM[6] = {450.0f, 500.0f, 550.0f, 590.0f, 630.0f, 675.0f};
constant float GPU_TAU_RAYLEIGH[6]   = {0.2210f, 0.1435f, 0.0970f, 0.0730f, 0.0558f, 0.0422f};
constant float GPU_TAU_OZONE_300[6]  = {0.0012f, 0.0095f, 0.0265f, 0.0368f, 0.0285f, 0.0125f};

kernel void kernel_corridor_radiative_lut(
    constant GpuSceneMeta& meta [[buffer(0)]],
    constant GpuSrtmTileMeta* srtm_tiles [[buffer(1)]],
    device const float* srtm_data [[buffer(2)]],
    constant GpuCorridorLutParams& params [[buffer(3)]],
    constant GpuCorridorStation* stations [[buffer(4)]],
    constant GpuCloudBand* bands [[buffer(5)]],
    device uchar* out_rgba8 [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.az_bins || gid.y >= params.alt_bins) return;

    float az_deg = (float(gid.x) * 360.0f) / float(params.az_bins);
    float az_rad = az_deg * 0.01745329252f;
    float dir_east = sin(az_rad);
    float dir_north = cos(az_rad);

    float v_norm = (params.alt_bins > 1u) ? (float(gid.y) / float(params.alt_bins - 1u)) : 0.0f;
    float z_asl = 12500.0f * pow(max(v_norm, 0.0f), 1.6f);

    float k_ref = clamp(params.refraction_k, 0.05f, 0.55f);
    float earth_r = 6371008.8f;
    float r_eff = earth_r / (1.0f - k_ref);

    // 1. Far-field SRTM curved-Earth horizon from altitude z_asl (4.6 km -> max_terrain_m)
    float max_slope = -sqrt(max(0.0f, 2.0f * z_asl * (1.0f - k_ref) / earth_r));
    float d = max(500.0f, params.start_dist_m);
    while (d <= params.max_terrain_m) {
        float s_dlat = fma(d * dir_north, INV_METERS_PER_DEG_LAT_F, params.origin_dlat_ref);
        float s_dlon = fma(d * dir_east, params.inv_m_per_lon, params.origin_dlon_ref);
        float z_srtm = sample_srtm_m(meta, srtm_tiles, srtm_data, s_dlat, s_dlon);
        float drop = (d * d) / (2.0f * r_eff);
        float slope = (z_srtm - drop - z_asl) / d;
        if (slope > max_slope) {
            max_slope = slope;
        }
        if (d < 12000.0f) d += 25.0f;
        else if (d < 40000.0f) d += 50.0f;
        else if (d < 100000.0f) d += 90.0f;
        else d += 150.0f;
    }
    float far_h_deg = atan(max_slope) * 57.2957795f;

    // 2. Spectral Rayleigh + Chappuis Ozone + CAMS Aerosol + 3D Stochastic Cloud corridor integration
    float tan_alpha0 = tan(params.lut_el_rad);
    float abs_tan0 = max(0.01f, abs(tan_alpha0));
    float atmo_limit_m = max(params.max_corridor_m, 820000.0f);
    float tau_mol_aer[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sum_cloud_hazard = 0.0f;
    float sum_uncond_cloudy_tau = 0.0f;

    uint n_st = min(params.num_stations, 32u);
    float weights[32];

    float s_m = max(0.0f, params.start_dist_m);
    while (s_m < atmo_limit_m) {
        bool in_weather = (s_m < params.max_corridor_m);
        float base_ds_m = (!in_weather) ? 8000.0f : ((s_m < 10000.0f) ? 250.0f : ((s_m < 40000.0f) ? 600.0f : ((s_m < 120000.0f) ? 1250.0f : 2500.0f)));
        float max_dz_m = in_weather ? 250.0f : 900.0f;
        float ds_m = max(2.0f, min(base_ds_m, max_dz_m / abs_tan0));

        float step_end_m = in_weather ? min(s_m + ds_m, params.max_corridor_m) : min(s_m + ds_m, atmo_limit_m);
        float step_horiz_m = step_end_m - s_m;
        float s_mid_m = s_m + 0.5f * step_horiz_m;
        s_m = step_end_m;
        if (step_horiz_m <= 0.0f) break;

        float drop_mid = (s_mid_m * s_mid_m) / (2.0f * r_eff);
        float z_ray_m = z_asl + s_mid_m * tan_alpha0 + drop_mid;
        if (z_ray_m > 45000.0f || z_ray_m < -150.0f) break;

        float dz_ds = tan_alpha0 + s_mid_m / r_eff;
        float dz_step_m = abs(step_horiz_m * dz_ds);
        float ds_path_m = sqrt(step_horiz_m * step_horiz_m + dz_step_m * dz_step_m);
        float s_km = min(s_mid_m, params.max_corridor_m) * 0.001f;

        float dist_sector_blend = clamp(s_km / 25.0f, 0.0f, 1.0f);
        float sigma_km = max(8.0f, 10.0f + 0.28f * s_km);
        float inv_2sig2 = 1.0f / (2.0f * sigma_km * sigma_km);
        float sum_w = 0.0f;

        for (uint i = 0u; i < n_st; i++) {
            float d_az = fmod(az_deg - stations[i].azimuth_deg + 720.0f, 360.0f);
            if (d_az > 180.0f) d_az = 360.0f - d_az;
            float az_w = (1.0f - dist_sector_blend) + dist_sector_blend * max(0.04f, exp(-(d_az * d_az) / (2.0f * 58.0f * 58.0f)));
            float dd = s_km - stations[i].distance_km;
            float w = az_w * (exp(-(dd * dd) * inv_2sig2) + 1.0e-6f);
            weights[i] = w;
            sum_w += w;
        }

        float surf_z = 0.0f, temp_k = 0.0f, press = 0.0f, lapse = 0.0f;
        float inv_top = 0.0f, inv_str = 0.0f, pbl = 0.0f, aod = 0.0f, ang = 0.0f, h_aer = 0.0f, oz = 0.0f;

        if (n_st == 0u || sum_w <= 0.0f) {
            temp_k = 288.15f; press = 1013.25f; lapse = 6.5f; pbl = 2000.0f;
            aod = 0.08f; ang = 1.25f; h_aer = 1650.0f; oz = 300.0f;
        } else {
            float inv_sum = 1.0f / sum_w;
            for (uint i = 0u; i < n_st; i++) {
                float w = weights[i] * inv_sum;
                weights[i] = w;
                if (w <= 1.0e-5f) continue;
                surf_z  += w * stations[i].surface_elev_m;
                temp_k  += w * stations[i].surface_temp_k;
                press   += w * stations[i].surface_pressure_hpa;
                lapse   += w * stations[i].lapse_rate_k_km;
                inv_top += w * stations[i].inversion_top_asl_m;
                inv_str += w * stations[i].inversion_strength_k;
                pbl     += w * stations[i].pbl_top_asl_m;
                aod     += w * stations[i].aod_550nm;
                ang     += w * stations[i].angstrom_exp;
                h_aer   += w * stations[i].aerosol_scale_m;
                oz      += w * stations[i].ozone_du;
            }
            temp_k = max(210.0f, temp_k);
            press  = clamp(press, 250.0f, 1060.0f);
            h_aer  = clamp(h_aer, 400.0f, 4500.0f);
            aod    = clamp(aod, 0.005f, 4.0f);
            ang    = clamp(ang, 0.1f, 2.4f);
            oz     = clamp(oz, 180.0f, 480.0f);
        }

        float h_ray = 8434.0f;
        float t_local = max(212.0f, temp_k - lapse * 0.001f * max(0.0f, z_ray_m - surf_z));
        float rho_rel = (press / 1013.25f) * exp(-max(0.0f, z_ray_m) / h_ray) * (288.15f / t_local);
        float d_ray_base = (ds_path_m / h_ray) * max(0.0f, rho_rel);

        float oz_dz = (z_ray_m - 22500.0f) / 5200.0f;
        float d_oz_scale = ds_path_m * (exp(-0.5f * oz_dz * oz_dz) / (2.5066283f * 5200.0f)) * (oz / 300.0f);

        float agl_m = max(0.0f, z_ray_m - surf_z);
        float inv_trap = 1.0f;
        if (inv_str > 0.5f && inv_top > surf_z) {
            inv_trap = (z_ray_m <= inv_top) ? (1.0f + min(1.4f, 0.22f * inv_str)) : 0.22f;
        } else if (z_ray_m > max(surf_z + 300.0f, pbl) + 600.0f) {
            inv_trap = 0.35f;
        }
        float d_aer_550 = ds_path_m * (aod / h_aer) * exp(-agl_m / h_aer) * inv_trap;

        for (uint b = 0u; b < 6u; b++) {
            float wl_r = GPU_SPECTRAL_WL_NM[b] / 550.0f;
            tau_mol_aer[b] += GPU_TAU_RAYLEIGH[b] * d_ray_base
                            + GPU_TAU_OZONE_300[b] * d_oz_scale
                            + d_aer_550 * pow(max(wl_r, 0.1f), -ang);
        }

        if (!in_weather) continue;

        for (uint i = 0u; i < n_st; i++) {
            float w_st = weights[i];
            if (w_st < 0.02f) continue;
            uint b0 = stations[i].band_offset;
            uint bn = stations[i].band_count;
            for (uint bi = 0u; bi < bn; bi++) {
                constant GpuCloudBand& cb = bands[b0 + bi];
                if (z_ray_m >= cb.base_asl_m && z_ray_m <= cb.top_asl_m) {
                    float c = clamp(cb.cloud_fraction, 0.0f, 0.992f);
                    if (c <= 0.005f) continue;
                    float h_band = max(80.0f, cb.top_asl_m - cb.base_asl_m);
                    float neg_ln = -log(max(0.012f, 1.0f - c));
                    float du_v = (dz_step_m / h_band) * neg_ln;
                    float du_g = (step_horiz_m / max(250.0f, cb.cell_d_m)) * neg_ln * (1.0f - 0.50f * c);
                    sum_cloud_hazard += w_st * sqrt(du_v * du_v + du_g * du_g);
                    sum_uncond_cloudy_tau += w_st * c * (cb.tau_core_vert / h_band) * min(ds_path_m, h_band * 12.0f) * cb.delta_edd_factor;
                }
            }
        }
    }

    float p_clear = clamp(exp(-sum_cloud_hazard), 0.0f, 1.0f);
    float mean_cloudy_tau = (p_clear < 0.998f && sum_uncond_cloudy_tau > 0.0f)
        ? clamp(sum_uncond_cloudy_tau / max(0.01f, 1.0f - p_clear), 0.06f, 40.0f)
        : 0.0f;
    float t_cloudy = (mean_cloudy_tau > 0.0f) ? clamp(exp(-mean_cloudy_tau), 0.0f, 1.0f) : 1.0f;
    float exp_cloud = p_clear + (1.0f - p_clear) * t_cloudy;

    float tr[6];
    for (uint b = 0u; b < 6u; b++) {
        tr[b] = clamp(exp(-tau_mol_aer[b]) * exp_cloud, 0.0f, 1.0f);
    }

    float r = clamp(0.18f * tr[3] + 0.47f * tr[4] + 0.35f * tr[5], 0.0f, 1.0f);
    float g = clamp(0.20f * tr[1] + 0.58f * tr[2] + 0.22f * tr[3], 0.0f, 1.0f);
    float bl = clamp(0.76f * tr[0] + 0.24f * tr[1], 0.0f, 1.0f);
    float h_norm = clamp((far_h_deg - (-6.0f)) / 32.0f, 0.0f, 1.0f);

    uint out_off = (gid.y * params.az_bins + gid.x) * 4u;
    out_rgba8[out_off + 0u] = uchar(rint(r * 255.0f));
    out_rgba8[out_off + 1u] = uchar(rint(g * 255.0f));
    out_rgba8[out_off + 2u] = uchar(rint(bl * 255.0f));
    out_rgba8[out_off + 3u] = uchar(rint(h_norm * 255.0f));
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuSceneMeta {
    has_near_1m: u32,
    w0: u32,
    h0: u32,
    off_l0: u32,
    w2: u32,
    h2: u32,
    off_l2: u32,
    w4: u32,
    h4: u32,
    off_l4: u32,
    global_min_z: f32,
    global_max_z: f32,
    near_m_per_lon: f32,
    ref_lat_int: i32,
    ref_lat_frac: f32,
    ref_lon_int: i32,
    ref_lon_frac: f32,
    num_srtm_tiles: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuSrtmTileMeta {
    lat_floor: i32,
    lon_floor: i32,
    samples: u32,
    side: u32,
    data_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuHorizonRayReq {
    pub dlat_ref: f32,
    pub dlon_ref: f32,
    pub ox_near_m: f32,
    pub oy_near_m: f32,
    pub z0: f32,
    pub dir_east: f32,
    pub dir_north: f32,
    pub inv_m_per_lon: f32,
    pub max_distance_m: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuHorizonRayOut {
    pub horizon_elevation_deg: f32,
    pub occluder_distance_m: f32,
    pub occluder_height_m: f32,
    pub _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuShadowParams {
    nx: u32,
    ny: u32,
    step_m: u32,
    write_png: u32,
    sun_dir_east: f32,
    sun_dir_north: f32,
    sun_vec_e: f32,
    sun_vec_n: f32,
    sun_vec_u: f32,
    sun_el_deg: f32,
    sun_slope: f32,
    max_distance_m: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuShadowPixel {
    is_lit: u32,
    cos_inc: f32,
    margin_deg: f32,
    z0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuCrownRimPt {
    dlat_ref: f32,
    dlon_ref: f32,
    z_m: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuLosTargetParams {
    num_crown_pts: u32,
    num_viewpoints: u32,
    summit_dlat_ref: f32,
    summit_dlon_ref: f32,
    base_valley_z: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    crown_pts: [GpuCrownRimPt; 12],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuLosVpReq {
    dlat_ref: f32,
    dlon_ref: f32,
    z_m: f32,
    m_per_lon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuLosVpOut {
    pub visible_relief_m: f32,
    pub summit_slope: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuPatchParams {
    grid_size: u32,
    center_dlat_ref: f32,
    center_dlon_ref: f32,
    half_span_m: f32,
    step_m: f32,
    inv_m_per_lon: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuCorridorStation {
    distance_km: f32,
    azimuth_deg: f32,
    surface_elev_m: f32,
    surface_temp_k: f32,
    surface_pressure_hpa: f32,
    lapse_rate_k_km: f32,
    inversion_top_asl_m: f32,
    inversion_strength_k: f32,
    pbl_top_asl_m: f32,
    aod_550nm: f32,
    angstrom_exp: f32,
    aerosol_scale_m: f32,
    ozone_du: f32,
    band_offset: u32,
    band_count: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuCloudBand {
    base_asl_m: f32,
    top_asl_m: f32,
    cloud_fraction: f32,
    tau_core_vert: f32,
    cell_d_m: f32,
    delta_edd_factor: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuCorridorLutParams {
    az_bins: u32,
    alt_bins: u32,
    num_stations: u32,
    _pad0: u32,
    origin_dlat_ref: f32,
    origin_dlon_ref: f32,
    inv_m_per_lon: f32,
    lut_el_rad: f32,
    start_dist_m: f32,
    max_terrain_m: f32,
    max_corridor_m: f32,
    refraction_k: f32,
}

struct MetalBuffer {
    ptr: *mut c_void,
    len_bytes: usize,
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    fn from_slice<T: Copy>(ctx_ptr: *mut c_void, slice: &[T]) -> Option<Self> {
        let len_bytes = std::mem::size_of_val(slice);
        let ptr = unsafe {
            pp_metal_create_buffer(
                ctx_ptr,
                if len_bytes > 0 {
                    slice.as_ptr() as *const c_void
                } else {
                    std::ptr::null()
                },
                len_bytes.max(16),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                len_bytes: len_bytes.max(16),
            })
        }
    }

    fn uninit<T: Copy>(ctx_ptr: *mut c_void, count: usize) -> Option<Self> {
        let len_bytes = (count * std::mem::size_of::<T>()).max(16);
        let ptr = unsafe { pp_metal_create_buffer(ctx_ptr, std::ptr::null(), len_bytes) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, len_bytes })
        }
    }

    fn as_slice<T: Copy>(&self, count: usize) -> &[T] {
        assert!(count * std::mem::size_of::<T>() <= self.len_bytes);
        let raw = unsafe { pp_metal_buffer_contents(self.ptr) as *const T };
        unsafe { std::slice::from_raw_parts(raw, count) }
    }

    fn as_mut_slice<T: Copy>(&mut self, count: usize) -> &mut [T] {
        assert!(count * std::mem::size_of::<T>() <= self.len_bytes);
        let raw = unsafe { pp_metal_buffer_contents(self.ptr) as *mut T };
        unsafe { std::slice::from_raw_parts_mut(raw, count) }
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { pp_metal_destroy_buffer(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

struct CachedGpuTerrain {
    cache_id: u64,
    ref_lat: f64,
    ref_lon: f64,
    meta_buf: MetalBuffer,
    near_mip_buf: MetalBuffer,
    srtm_tiles_buf: MetalBuffer,
    srtm_data_buf: MetalBuffer,
}

struct MetalEngine {
    ctx_ptr: *mut c_void,
    device_name: String,
    pipe_horizon: *mut c_void,
    pipe_shadow: *mut c_void,
    pipe_spotlight: *mut c_void,
    pipe_camera_los: *mut c_void,
    pipe_patch: *mut c_void,
    pipe_corridor_lut: *mut c_void,
    cached_terrain: Mutex<Option<CachedGpuTerrain>>,
}

unsafe impl Send for MetalEngine {}
unsafe impl Sync for MetalEngine {}

impl Drop for MetalEngine {
    fn drop(&mut self) {
        unsafe {
            pp_metal_destroy_pipeline(self.pipe_horizon);
            pp_metal_destroy_pipeline(self.pipe_shadow);
            pp_metal_destroy_pipeline(self.pipe_spotlight);
            pp_metal_destroy_pipeline(self.pipe_camera_los);
            pp_metal_destroy_pipeline(self.pipe_patch);
            pp_metal_destroy_pipeline(self.pipe_corridor_lut);
            pp_metal_destroy(self.ctx_ptr);
        }
    }
}

static ENGINE: OnceLock<Option<MetalEngine>> = OnceLock::new();

fn get_engine() -> Option<&'static MetalEngine> {
    if std::env::var_os("PP_DISABLE_GPU").is_some() {
        return None;
    }
    ENGINE
        .get_or_init(|| {
            let src_c = CString::new(METAL_SHADER_SOURCE).ok()?;
            let mut name_buf = vec![0 as c_char; 128];
            let mut err_buf = vec![0 as c_char; 512];

            let ctx_ptr = unsafe {
                pp_metal_create(
                    src_c.as_ptr(),
                    name_buf.as_mut_ptr(),
                    name_buf.len(),
                    err_buf.as_mut_ptr(),
                    err_buf.len(),
                )
            };
            if ctx_ptr.is_null() {
                let err_msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy();
                eprintln!("[pp-raycast] Metal GPU unavailable ({}), using Rayon CPU.", err_msg);
                return None;
            }

            let device_name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();

            let mut load_pipe = |fn_name: &str| -> Option<*mut c_void> {
                let fn_c = CString::new(fn_name).ok()?;
                let p = unsafe {
                    pp_metal_create_pipeline(
                        ctx_ptr,
                        fn_c.as_ptr(),
                        err_buf.as_mut_ptr(),
                        err_buf.len(),
                    )
                };
                if p.is_null() {
                    let err_msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy();
                    eprintln!("[pp-raycast] Metal pipeline '{}' failed: {}", fn_name, err_msg);
                    None
                } else {
                    Some(p)
                }
            };

            let pipe_horizon = load_pipe("kernel_batch_horizon_rays")?;
            let pipe_shadow = load_pipe("kernel_shadow_and_light_map")?;
            let pipe_spotlight = load_pipe("kernel_spotlight_and_png")?;
            let pipe_camera_los = load_pipe("kernel_batch_camera_los")?;
            let pipe_patch = load_pipe("kernel_resample_terrain_patch")?;
            let pipe_corridor_lut = load_pipe("kernel_corridor_radiative_lut")?;

            Some(MetalEngine {
                ctx_ptr,
                device_name,
                pipe_horizon,
                pipe_shadow,
                pipe_spotlight,
                pipe_camera_los,
                pipe_patch,
                pipe_corridor_lut,
                cached_terrain: Mutex::new(None),
            })
        })
        .as_ref()
}

pub fn gpu_device_name() -> Option<String> {
    get_engine().map(|e| format!("{} (Metal Unified Memory GPU)", e.device_name))
}

impl MetalEngine {
    fn upload_terrain(&self, terrain: &HybridTerrain) -> Option<CachedGpuTerrain> {
        let (ref_lat, ref_lon) = if let Some(ref grid) = terrain.near_1m {
            (grid.origin_lat, grid.origin_lon)
        } else if let Some((&(lat_fl, lon_fl), _)) = terrain.far_srtm.tiles.iter().next() {
            (lat_fl as f64 + 0.5, lon_fl as f64 + 0.5)
        } else {
            (37.745, -119.533)
        };

        let ref_lat_int = ref_lat.floor() as i32;
        let ref_lat_frac = (ref_lat - ref_lat_int as f64) as f32;
        let ref_lon_int = ref_lon.floor() as i32;
        let ref_lon_frac = (ref_lon - ref_lon_int as f64) as f32;

        let mut meta = GpuSceneMeta {
            ref_lat_int,
            ref_lat_frac,
            ref_lon_int,
            ref_lon_frac,
            ..Default::default()
        };

        let near_mip_buf = if let Some(ref grid) = terrain.near_1m {
            let l0 = &grid.levels[0];
            let l2 = if grid.levels.len() > 2 {
                &grid.levels[2]
            } else {
                l0
            };
            let l4 = if grid.levels.len() > 4 {
                &grid.levels[4]
            } else {
                l2
            };

            let off_l0 = 0usize;
            let off_l2 = off_l0 + l0.max_z.len();
            let off_l4 = off_l2 + l2.max_z.len();
            let total_near = off_l4 + l4.max_z.len();

            meta.has_near_1m = 1;
            meta.w0 = l0.width as u32;
            meta.h0 = l0.height as u32;
            meta.off_l0 = off_l0 as u32;
            meta.w2 = l2.width as u32;
            meta.h2 = l2.height as u32;
            meta.off_l2 = off_l2 as u32;
            meta.w4 = l4.width as u32;
            meta.h4 = l4.height as u32;
            meta.off_l4 = off_l4 as u32;
            meta.global_min_z = grid.global_min_z;
            meta.global_max_z = grid.global_max_z;
            meta.near_m_per_lon = meters_per_deg_lon(grid.origin_lat) as f32;

            let mut buf = MetalBuffer::uninit::<f32>(self.ctx_ptr, total_near)?;
            let dst = buf.as_mut_slice::<f32>(total_near);
            dst[off_l0..off_l2].copy_from_slice(&l0.max_z);
            dst[off_l2..off_l4].copy_from_slice(&l2.max_z);
            dst[off_l4..total_near].copy_from_slice(&l4.max_z);
            buf
        } else {
            MetalBuffer::from_slice(self.ctx_ptr, &[0.0_f32; 4])?
        };

        // Write loaded SRTM tiles directly into shared Metal Unified Memory without intermediate Vec allocation
        let mut tile_keys: Vec<(i32, i32)> = terrain.far_srtm.tiles.keys().copied().collect();
        tile_keys.sort_unstable();

        let mut srtm_metas = Vec::with_capacity(tile_keys.len().max(1));
        let total_srtm_floats: usize = tile_keys
            .iter()
            .map(|k| terrain.far_srtm.tiles[k].data.len())
            .sum::<usize>()
            .max(4);

        let mut srtm_data_buf = MetalBuffer::uninit::<f32>(self.ctx_ptr, total_srtm_floats)?;
        {
            let dst = srtm_data_buf.as_mut_slice::<f32>(total_srtm_floats);
            let mut cursor = 0usize;
            for key in tile_keys {
                let tile = &terrain.far_srtm.tiles[&key];
                let len = tile.data.len();
                dst[cursor..cursor + len].copy_from_slice(&tile.data);
                srtm_metas.push(GpuSrtmTileMeta {
                    lat_floor: tile.lat_floor,
                    lon_floor: tile.lon_floor,
                    samples: tile.samples as u32,
                    side: (tile.samples + 1) as u32,
                    data_offset: cursor as u32,
                    ..Default::default()
                });
                cursor += len;
            }
        }

        meta.num_srtm_tiles = srtm_metas.len() as u32;
        if srtm_metas.is_empty() {
            srtm_metas.push(GpuSrtmTileMeta::default());
        }

        let meta_buf = MetalBuffer::from_slice(self.ctx_ptr, std::slice::from_ref(&meta))?;
        let srtm_tiles_buf = MetalBuffer::from_slice(self.ctx_ptr, &srtm_metas)?;

        Some(CachedGpuTerrain {
            cache_id: terrain.cache_id,
            ref_lat,
            ref_lon,
            meta_buf,
            near_mip_buf,
            srtm_tiles_buf,
            srtm_data_buf,
        })
    }

    fn with_terrain<R>(
        &self,
        terrain: &HybridTerrain,
        f: impl FnOnce(&CachedGpuTerrain) -> Option<R>,
    ) -> Option<R> {
        let mut guard = self.cached_terrain.lock().ok()?;
        let needs_upload = guard
            .as_ref()
            .map_or(true, |c| c.cache_id != terrain.cache_id);
        if needs_upload {
            *guard = Some(self.upload_terrain(terrain)?);
        }
        let cached = guard.as_ref()?;
        f(cached)
    }
}

/// Executes an arbitrary batch of curved-Earth 1m + SRTM horizon rays on the Apple Silicon Metal GPU.
/// Each tuple in `rays` is `(origin_lat, origin_lon, z0, azimuth_deg, max_distance_m)`.
pub fn dispatch_horizon_rays_gpu(
    terrain: &HybridTerrain,
    rays: &[(f64, f64, f64, f64, f64)],
) -> Option<Vec<HorizonSample>> {
    if rays.is_empty() {
        return Some(Vec::new());
    }
    let engine = get_engine()?;
    engine.with_terrain(terrain, |scene| {
        let mut reqs = Vec::with_capacity(rays.len());
        for &(olat, olon, z0, az_deg, max_dist_m) in rays {
            let az_rad = az_deg.to_radians();
            let (ox, oy) = if let Some(ref grid) = terrain.near_1m {
                let (x, y) = grid.latlon_to_local_m(olat, olon);
                (x as f32, y as f32)
            } else {
                (0.0, 0.0)
            };
            reqs.push(GpuHorizonRayReq {
                dlat_ref: (olat - scene.ref_lat) as f32,
                dlon_ref: (olon - scene.ref_lon) as f32,
                ox_near_m: ox,
                oy_near_m: oy,
                z0: z0 as f32,
                dir_east: az_rad.sin() as f32,
                dir_north: az_rad.cos() as f32,
                inv_m_per_lon: (1.0 / meters_per_deg_lon(olat).max(1.0)) as f32,
                max_distance_m: max_dist_m as f32,
                ..Default::default()
            });
        }

        let num_rays_u32 = rays.len() as u32;
        let req_buf = MetalBuffer::from_slice(engine.ctx_ptr, &reqs)?;
        let out_buf = MetalBuffer::uninit::<GpuHorizonRayOut>(engine.ctx_ptr, rays.len())?;
        let count_buf =
            MetalBuffer::from_slice(engine.ctx_ptr, std::slice::from_ref(&num_rays_u32))?;

        let bufs = [
            scene.meta_buf.ptr,
            scene.near_mip_buf.ptr,
            scene.srtm_tiles_buf.ptr,
            scene.srtm_data_buf.ptr,
            req_buf.ptr,
            out_buf.ptr,
            count_buf.ptr,
        ];

        let rc = unsafe {
            pp_metal_dispatch_1d(
                engine.ctx_ptr,
                engine.pipe_horizon,
                bufs.as_ptr(),
                bufs.len(),
                rays.len(),
            )
        };
        if rc != 0 {
            return None;
        }

        let gpu_outs = out_buf.as_slice::<GpuHorizonRayOut>(rays.len());
        let mut samples = Vec::with_capacity(rays.len());
        for (i, out) in gpu_outs.iter().enumerate() {
            let az_deg = rays[i].3;
            samples.push(HorizonSample {
                azimuth_deg: (az_deg * 1000.0).round() / 1000.0,
                horizon_elevation_deg: ((out.horizon_elevation_deg as f64) * 1000.0).round()
                    / 1000.0,
                occluder_distance_m: (out.occluder_distance_m as f64).round(),
                occluder_height_m: ((out.occluder_height_m as f64) * 10.0).round() / 10.0,
            });
        }
        Some(samples)
    })
}

/// Executes the 2-pass 2D Directional Ray-Cast Shadow + Golden-Hour Spotlight + RGB8 PNG synthesis on Metal GPU.
pub fn dispatch_shadow_and_light_map_gpu(
    terrain: &HybridTerrain,
    grid: &MaxMipmapGrid1m,
    sun: SolarPosition,
    step_m: usize,
    output_png: Option<&Path>,
) -> Option<Result<ShadowLightMapStats>> {
    // Ensure terrain.near_1m matches `grid` geometry
    let near_ref = terrain.near_1m.as_ref()?;
    if near_ref.width_m != grid.width_m || near_ref.height_m != grid.height_m {
        return None;
    }

    let step_m = step_m.max(1);
    let nx = grid.width_m / step_m;
    let ny = grid.height_m / step_m;
    if nx == 0 || ny == 0 {
        return None;
    }

    let engine = get_engine()?;
    engine.with_terrain(terrain, |scene| {
        let az_rad = sun.azimuth_deg.to_radians();
        let el_rad = sun.apparent_elevation_deg.to_radians();

        let params = GpuShadowParams {
            nx: nx as u32,
            ny: ny as u32,
            step_m: step_m as u32,
            write_png: if output_png.is_some() { 1 } else { 0 },
            sun_dir_east: az_rad.sin() as f32,
            sun_dir_north: az_rad.cos() as f32,
            sun_vec_e: (az_rad.sin() * el_rad.cos()) as f32,
            sun_vec_n: (az_rad.cos() * el_rad.cos()) as f32,
            sun_vec_u: el_rad.sin() as f32,
            sun_el_deg: sun.apparent_elevation_deg as f32,
            sun_slope: el_rad.tan() as f32,
            max_distance_m: 30_000.0,
        };

        let total_px = nx * ny;
        let params_buf = MetalBuffer::from_slice(engine.ctx_ptr, std::slice::from_ref(&params))?;
        let pixels_buf = MetalBuffer::uninit::<GpuShadowPixel>(engine.ctx_ptr, total_px)?;
        let spot_buf = MetalBuffer::uninit::<f32>(engine.ctx_ptr, total_px)?;
        let rgb_len = if output_png.is_some() {
            total_px * 3
        } else {
            16
        };
        let rgb_buf = MetalBuffer::uninit::<u8>(engine.ctx_ptr, rgb_len)?;

        let bufs1 = [
            scene.meta_buf.ptr,
            scene.near_mip_buf.ptr,
            scene.srtm_tiles_buf.ptr,
            scene.srtm_data_buf.ptr,
            params_buf.ptr,
            pixels_buf.ptr,
        ];
        let bufs2 = [
            scene.meta_buf.ptr,
            params_buf.ptr,
            pixels_buf.ptr,
            spot_buf.ptr,
            rgb_buf.ptr,
        ];

        let rc = unsafe {
            pp_metal_dispatch_2pass_2d(
                engine.ctx_ptr,
                engine.pipe_shadow,
                bufs1.as_ptr(),
                bufs1.len(),
                engine.pipe_spotlight,
                bufs2.as_ptr(),
                bufs2.len(),
                nx,
                ny,
            )
        };
        if rc != 0 {
            return None;
        }

        let px_slice = pixels_buf.as_slice::<GpuShadowPixel>(total_px);
        let spot_slice = spot_buf.as_slice::<f32>(total_px);

        let mut lit_count = 0usize;
        let mut spotlight_count = 0usize;
        let mut best_score = 0.0_f32;
        let mut best_idx: Option<usize> = None;

        for i in 0..total_px {
            if px_slice[i].is_lit != 0 {
                lit_count += 1;
            }
            let sc = spot_slice[i];
            if sc > 0.0 {
                spotlight_count += 1;
                if best_idx.is_none() || sc > best_score {
                    best_score = sc;
                    best_idx = Some(i);
                }
            }
        }

        let best_hotspot = best_idx.map(|idx| {
            let iy = idx / nx;
            let ix = idx % nx;
            let (lat, lon) = grid.local_m_to_latlon((ix * step_m) as f64, (iy * step_m) as f64);
            (lat, lon, best_score as f64)
        });

        let save_res: Result<()> = if let Some(path) = output_png {
            let rgb_bytes = rgb_buf.as_slice::<u8>(total_px * 3).to_vec();
            if let Some(img) = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(nx as u32, ny as u32, rgb_bytes) {
                img.save(path).map_err(Into::into)
            } else {
                Err(anyhow::anyhow!(
                    "Failed to construct RGB ImageBuffer from GPU output"
                ))
            }
        } else {
            Ok(())
        };

        Some(save_res.map(|()| ShadowLightMapStats {
            width_px: nx,
            height_px: ny,
            step_m,
            illuminated_fraction: (lit_count as f64 / total_px.max(1) as f64 * 1000.0).round()
                / 1000.0,
            spotlight_shaft_pixels: spotlight_count,
            max_contrast_hotspot: best_hotspot,
        }))
    })
}

/// Evaluates 3D Camera-to-Crown-Rim Line-of-Sight across a batch of trail viewpoints on Metal GPU.
pub fn dispatch_camera_los_batch_gpu(
    terrain: &HybridTerrain,
    viewpoints: &[(f64, f64, f64)],
    target: &LandmarkTarget3D,
) -> Option<Vec<GpuLosVpOut>> {
    if viewpoints.is_empty() {
        return Some(Vec::new());
    }
    let engine = get_engine()?;
    engine.with_terrain(terrain, |scene| {
        let mut crown_pts = [GpuCrownRimPt::default(); 12];
        let num_crown = target.crown_rim_points.len().min(12);
        for (i, &(clat, clon, cz)) in target.crown_rim_points.iter().take(12).enumerate() {
            crown_pts[i] = GpuCrownRimPt {
                dlat_ref: (clat - scene.ref_lat) as f32,
                dlon_ref: (clon - scene.ref_lon) as f32,
                z_m: cz as f32,
                _pad: 0.0,
            };
        }

        let target_params = GpuLosTargetParams {
            num_crown_pts: num_crown as u32,
            num_viewpoints: viewpoints.len() as u32,
            summit_dlat_ref: (target.summit_lat - scene.ref_lat) as f32,
            summit_dlon_ref: (target.summit_lon - scene.ref_lon) as f32,
            base_valley_z: target.base_valley_elevation_m as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            crown_pts,
        };

        let mut vp_reqs = Vec::with_capacity(viewpoints.len());
        for &(vlat, vlon, vz) in viewpoints {
            vp_reqs.push(GpuLosVpReq {
                dlat_ref: (vlat - scene.ref_lat) as f32,
                dlon_ref: (vlon - scene.ref_lon) as f32,
                z_m: vz as f32,
                m_per_lon: meters_per_deg_lon(vlat) as f32,
            });
        }

        let target_buf =
            MetalBuffer::from_slice(engine.ctx_ptr, std::slice::from_ref(&target_params))?;
        let req_buf = MetalBuffer::from_slice(engine.ctx_ptr, &vp_reqs)?;
        let out_buf = MetalBuffer::uninit::<GpuLosVpOut>(engine.ctx_ptr, viewpoints.len())?;

        let bufs = [
            scene.meta_buf.ptr,
            scene.near_mip_buf.ptr,
            scene.srtm_tiles_buf.ptr,
            scene.srtm_data_buf.ptr,
            target_buf.ptr,
            req_buf.ptr,
            out_buf.ptr,
        ];

        let rc = unsafe {
            pp_metal_dispatch_1d(
                engine.ctx_ptr,
                engine.pipe_camera_los,
                bufs.as_ptr(),
                bufs.len(),
                viewpoints.len(),
            )
        };
        if rc != 0 {
            return None;
        }

        Some(out_buf.as_slice::<GpuLosVpOut>(viewpoints.len()).to_vec())
    })
}

/// Resamples a 2D `grid_size x grid_size` hybrid 1m + SRTM `TerrainPatch` on Metal GPU.
pub fn dispatch_terrain_patch_gpu(
    terrain: &HybridTerrain,
    center_lat: f64,
    center_lon: f64,
    span_m: f64,
    grid_size: usize,
) -> Option<(f64, f64, Vec<f32>)> {
    if grid_size == 0 {
        return None;
    }
    let engine = get_engine()?;
    engine.with_terrain(terrain, |scene| {
        let half_span = span_m * 0.5;
        let step_m = span_m / ((grid_size - 1).max(1) as f64);
        let inv_m_per_lon = 1.0 / meters_per_deg_lon(center_lat).max(1.0);

        let params = GpuPatchParams {
            grid_size: grid_size as u32,
            center_dlat_ref: (center_lat - scene.ref_lat) as f32,
            center_dlon_ref: (center_lon - scene.ref_lon) as f32,
            half_span_m: half_span as f32,
            step_m: step_m as f32,
            inv_m_per_lon: inv_m_per_lon as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        };

        let total = grid_size * grid_size;
        let params_buf = MetalBuffer::from_slice(engine.ctx_ptr, std::slice::from_ref(&params))?;
        let out_buf = MetalBuffer::uninit::<f32>(engine.ctx_ptr, total)?;

        let bufs = [
            scene.meta_buf.ptr,
            scene.near_mip_buf.ptr,
            scene.srtm_tiles_buf.ptr,
            scene.srtm_data_buf.ptr,
            params_buf.ptr,
            out_buf.ptr,
        ];

        let rc = unsafe {
            pp_metal_dispatch_2pass_2d(
                engine.ctx_ptr,
                engine.pipe_patch,
                bufs.as_ptr(),
                bufs.len(),
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                grid_size,
                grid_size,
            )
        };
        if rc != 0 {
            return None;
        }

        let elevations = out_buf.as_slice::<f32>(total).to_vec();
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        for &z in &elevations {
            let zf = z as f64;
            if zf < min_z {
                min_z = zf;
            }
            if zf > max_z {
                max_z = zf;
            }
        }
        Some((min_z, max_z, elevations))
    })
}

/// Evaluates the 2D `(az_bins x alt_bins)` `CorridorRadiativeLut` (`4.6 km -> 380 km`) on Metal GPU.
pub fn dispatch_corridor_lut_gpu(
    terrain: &HybridTerrain,
    corridor: &AtmosphericCorridorState,
    origin_lat: f64,
    origin_lon: f64,
    sun_elevation_deg: f64,
    az_bins: usize,
    alt_bins: usize,
) -> Option<CorridorRadiativeLut> {
    let az_bins = az_bins.max(16);
    let alt_bins = alt_bins.max(8);
    if corridor.stations.len() > 32 {
        return None;
    }

    let engine = get_engine()?;
    engine.with_terrain(terrain, |scene| {
        let mut gpu_stations = Vec::with_capacity(corridor.stations.len().max(1));
        let mut gpu_bands = Vec::new();

        for st in &corridor.stations {
            let band_offset = gpu_bands.len() as u32;
            let mut band_count = 0u32;
            for b in &st.cloud_bands {
                let c = b.cloud_fraction.clamp(0.0, 0.992);
                if c <= 0.005 || b.top_asl_m <= b.base_asl_m + 10.0 {
                    continue;
                }
                let h_band = (b.top_asl_m - b.base_asl_m).max(80.0);
                let cell_d_m = (b.characteristic_cell_km.max(0.25) * 1000.0)
                    .min((h_band / b.aspect_ratio_h_over_d.max(0.03)).max(250.0));
                let g_asym = 0.85 * (1.0 - b.ice_fraction) + 0.77 * b.ice_fraction;
                let delta_edd = 1.0 - g_asym * g_asym * 0.88;
                gpu_bands.push(GpuCloudBand {
                    base_asl_m: b.base_asl_m as f32,
                    top_asl_m: b.top_asl_m as f32,
                    cloud_fraction: c as f32,
                    tau_core_vert: b.in_cloud_vertical_tau() as f32,
                    cell_d_m: cell_d_m as f32,
                    delta_edd_factor: delta_edd as f32,
                    _pad0: 0.0,
                    _pad1: 0.0,
                });
                band_count += 1;
            }

            gpu_stations.push(GpuCorridorStation {
                distance_km: st.distance_km as f32,
                azimuth_deg: st.azimuth_deg.unwrap_or(corridor.azimuth_center_deg) as f32,
                surface_elev_m: st.surface_elevation_m as f32,
                surface_temp_k: (st.surface_temp_c + 273.15) as f32,
                surface_pressure_hpa: st.surface_pressure_hpa as f32,
                lapse_rate_k_km: st.lapse_rate_k_km as f32,
                inversion_top_asl_m: st.inversion_top_asl_m as f32,
                inversion_strength_k: st.inversion_strength_k as f32,
                pbl_top_asl_m: st.boundary_layer_top_asl_m as f32,
                aod_550nm: st.aod_550nm as f32,
                angstrom_exp: st.angstrom_exponent as f32,
                aerosol_scale_m: st.aerosol_scale_height_m as f32,
                ozone_du: st.ozone_du as f32,
                band_offset,
                band_count,
                _pad0: 0,
            });
        }

        if gpu_stations.is_empty() {
            gpu_stations.push(GpuCorridorStation::default());
        }
        if gpu_bands.is_empty() {
            gpu_bands.push(GpuCloudBand::default());
        }

        let lut_el_deg = sun_elevation_deg.clamp(-2.8, 89.0);
        let k_ref = corridor.effective_refraction_k(corridor.azimuth_center_deg, 2_000.0);
        let params = GpuCorridorLutParams {
            az_bins: az_bins as u32,
            alt_bins: alt_bins as u32,
            num_stations: corridor.stations.len() as u32,
            _pad0: 0,
            origin_dlat_ref: (origin_lat - scene.ref_lat) as f32,
            origin_dlon_ref: (origin_lon - scene.ref_lon) as f32,
            inv_m_per_lon: (1.0 / meters_per_deg_lon(origin_lat).max(1.0)) as f32,
            lut_el_rad: lut_el_deg.to_radians() as f32,
            start_dist_m: DEFAULT_LUT_START_DISTANCE_M as f32,
            max_terrain_m: 200_000.0,
            max_corridor_m: (corridor.max_corridor_km.clamp(40.0, 420.0) * 1000.0) as f32,
            refraction_k: k_ref as f32,
        };

        let total_bytes = az_bins * alt_bins * 4;
        let params_buf = MetalBuffer::from_slice(engine.ctx_ptr, std::slice::from_ref(&params))?;
        let stations_buf = MetalBuffer::from_slice(engine.ctx_ptr, &gpu_stations)?;
        let bands_buf = MetalBuffer::from_slice(engine.ctx_ptr, &gpu_bands)?;
        let out_buf = MetalBuffer::uninit::<u8>(engine.ctx_ptr, total_bytes)?;

        let bufs = [
            scene.meta_buf.ptr,
            scene.srtm_tiles_buf.ptr,
            scene.srtm_data_buf.ptr,
            params_buf.ptr,
            stations_buf.ptr,
            bands_buf.ptr,
            out_buf.ptr,
        ];

        let rc = unsafe {
            pp_metal_dispatch_2pass_2d(
                engine.ctx_ptr,
                engine.pipe_corridor_lut,
                bufs.as_ptr(),
                bufs.len(),
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                az_bins,
                alt_bins,
            )
        };
        if rc != 0 {
            return None;
        }

        let rgba8 = out_buf.as_slice::<u8>(total_bytes).to_vec();
        Some(CorridorRadiativeLut {
            az_bins,
            alt_bins,
            start_distance_m: DEFAULT_LUT_START_DISTANCE_M,
            max_altitude_asl_m: LUT_MAX_ALTITUDE_ASL_M,
            altitude_gamma: LUT_ALTITUDE_GAMMA,
            horizon_min_deg: LUT_HORIZON_MIN_DEG,
            horizon_span_deg: LUT_HORIZON_SPAN_DEG,
            sun_elevation_deg,
            rgba8,
        })
    })
}
