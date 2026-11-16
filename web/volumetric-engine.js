// ============================================================================
// PhotoPlanner 3D Volumetric Atmosphere, 1m Cloud & Upper-Tropospheric Cirrus Engine
// Native WebGL2 GLSL3 GPU 3D Moist Navier-Stokes Solver (128x128x48 Tiled Atlas)
// + GPU Topographic Penumbra/Viewshed Baker + GPU 360° Horizon/Sightline Raycaster
// (with automatic fallback to Rust SIMD128 WASM `pp_wasm.wasm` in `sim-worker.js`)
// ============================================================================

import * as THREE from 'three';

export const WEATHER_REGIME_PRESETS = {
  live: {
    label: 'Live Forecast Imputation (HRRR / Sounding)',
    description: 'Couples Open-Meteo / HRRR pressure-level sounding directly into the GPU 128×128×48 Navier-Stokes & 3D ice solver.',
    overrides: null,
  },
  coastal_fog_sea: {
    label: 'Pacific Marine Fog Sea & Inversion',
    description: 'Strong 8.5 K marine inversion at 310m traps dense stratus against coastal ridges while 300hPa sheared cirrus glows overhead.',
    overrides: {
      surfaceTempC: 12.8,
      surfaceRhPct: 95,
      lowCloudPct: 84,
      midCloudPct: 12,
      highCloudPct: 48,
      inversionBaseM: 310,
      inversionThicknessM: 170,
      inversionStrengthK: 8.5,
      windSfcU: 6.4,
      windSfcV: -4.2,
      wind850U: 9.0,
      wind850V: -2.5,
      wind500U: 19.0,
      wind500V: 5.0,
      wind250U: 41.0,
      wind250V: 16.0,
      convectiveCape: 90,
      orographicWaveGain: 1.25,
      vorticityConfinement: 1.2,
      cirrusIceBoost: 1.15,
    },
  },
  crepuscular_godrays: {
    label: 'Broken Stratocumulus & God-Rays',
    description: 'Fenestrated coastal/valley stratocumulus cells with humid sub-cloud haze producing dramatic 3D crepuscular sunbeams.',
    overrides: {
      surfaceTempC: 14.5,
      surfaceRhPct: 89,
      lowCloudPct: 68,
      midCloudPct: 28,
      highCloudPct: 36,
      inversionBaseM: 540,
      inversionThicknessM: 240,
      inversionStrengthK: 5.4,
      windSfcU: 7.2,
      windSfcV: -2.8,
      wind850U: 11.5,
      wind850V: -1.5,
      wind500U: 22.0,
      wind500V: 6.0,
      wind250U: 35.0,
      wind250V: 12.0,
      convectiveCape: 380,
      orographicWaveGain: 1.15,
      vorticityConfinement: 1.45,
      cirrusIceBoost: 0.95,
    },
  },
  sierra_cumulus_cirrus: {
    label: 'Orographic Cumulus & High 3D Cirrus',
    description: 'Anabatic slope thermals build 1m-detailed cumulus turrets over summits beneath sweeping 10.5 km cirrus uncinus mares\' tails.',
    overrides: {
      surfaceTempC: 18.5,
      surfaceRhPct: 64,
      lowCloudPct: 44,
      midCloudPct: 38,
      highCloudPct: 76,
      inversionBaseM: 1950,
      inversionThicknessM: 380,
      inversionStrengthK: 2.2,
      windSfcU: 4.5,
      windSfcV: 2.1,
      wind850U: 8.8,
      wind850V: 3.4,
      wind500U: 21.0,
      wind500V: 8.5,
      wind250U: 48.0,
      wind250V: 21.0,
      convectiveCape: 1050,
      orographicWaveGain: 1.1,
      vorticityConfinement: 1.6,
      cirrusIceBoost: 1.45,
    },
  },
  lenticular_wave: {
    label: 'Sierra Lee-Wave Lenticular & Jet Cirrus',
    description: 'High-stability stratified cross-ridge flow excites standing 3D mountain gravity waves (altocumulus lenticularis) and jet-stream cirrus.',
    overrides: {
      surfaceTempC: 11.0,
      surfaceRhPct: 68,
      lowCloudPct: 26,
      midCloudPct: 66,
      highCloudPct: 68,
      inversionBaseM: 1450,
      inversionThicknessM: 280,
      inversionStrengthK: 6.8,
      windSfcU: 9.5,
      windSfcV: 1.8,
      wind850U: 16.5,
      wind850V: 3.0,
      wind500U: 29.0,
      wind500V: 7.0,
      wind250U: 54.0,
      wind250V: 15.0,
      convectiveCape: 140,
      orographicWaveGain: 2.15,
      vorticityConfinement: 0.95,
      cirrusIceBoost: 1.35,
    },
  },
  clear_crisp_cirrus: {
    label: 'Crystal Alpine Visibility & Wispy Cirrus',
    description: 'Clean dry boundary layer for razor-sharp mountain views with delicate high-tropospheric 3D cirrus filaments catching late alpenglow.',
    overrides: {
      surfaceTempC: 15.5,
      surfaceRhPct: 46,
      lowCloudPct: 8,
      midCloudPct: 10,
      highCloudPct: 62,
      inversionBaseM: 900,
      inversionThicknessM: 260,
      inversionStrengthK: 3.0,
      windSfcU: 3.2,
      windSfcV: -1.4,
      wind850U: 6.5,
      wind850V: 1.0,
      wind500U: 17.0,
      wind500V: 6.0,
      wind250U: 44.0,
      wind250V: 19.0,
      convectiveCape: 80,
      orographicWaveGain: 0.85,
      vorticityConfinement: 1.0,
      cirrusIceBoost: 1.35,
    },
  },
};

// GPU 3D Fluid Grid Dimensions: 128 x 128 x 48 (786,432 voxels — 4x WASM horizontal resolution!)
// State textures: 8 x 6 tiles of 128 x 128 = 1024 x 768
// Padded display atlas: 8 x 6 tiles of 130 x 130 (1-texel periodic XY ghost border per tile) = 1040 x 780
const GPU_NX = 128;
const GPU_NY = 128;
const GPU_NZ = 48;
const GPU_TILES_X = 8;
const GPU_TILES_Y = 6;
const STATE_ATLAS_W = GPU_NX * GPU_TILES_X; // 1024
const STATE_ATLAS_H = GPU_NY * GPU_TILES_Y; // 768
const PAD_TILE = GPU_NX + 2; // 130
const CLOUD_ATLAS_W = PAD_TILE * GPU_TILES_X; // 1040
const CLOUD_ATLAS_H = PAD_TILE * GPU_TILES_Y; // 780

const FULLSCREEN_VERTEX_GLSL3 = /* glsl */ `
  out vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

// ============================================================================
// GPU Pass 0: Tileable 64^3 Perlin-Worley + Curl-Filament + 1m Kolmogorov Generator
// Synthesizes all 262,144 voxels (512 x 512 atlas) on GPU in ~0.8 ms at startup!
// ============================================================================
const DETAIL_NOISE_GEN_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;
  out vec4 fragColor;

  uint hash3u(uvec3 p) {
    p = p * uvec3(1664525u, 1013904223u, 22695477u) + uvec3(1013904223u);
    p.x += p.y * p.z;
    p.y += p.z * p.x;
    p.z += p.x * p.y;
    p ^= p >> 16u;
    p.x += p.y * p.z;
    p.y += p.z * p.x;
    p.z += p.x * p.y;
    return p.x;
  }

  vec3 hash33Periodic(ivec3 cell, int period) {
    ivec3 w = ((cell % period) + period) % period;
    uint h = hash3u(uvec3(w));
    float rx = float(h & 1023u) / 1023.0;
    float ry = float((h >> 10u) & 1023u) / 1023.0;
    float rz = float((h >> 20u) & 1023u) / 1023.0;
    return vec3(rx, ry, rz);
  }

  float gradientNoisePeriodic(vec3 uvw, int period) {
    vec3 p = uvw * float(period);
    ivec3 i0 = ivec3(floor(p));
    vec3 f = p - vec3(i0);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float n000 = dot(hash33Periodic(i0 + ivec3(0, 0, 0), period) * 2.0 - 1.0, f - vec3(0.0, 0.0, 0.0));
    float n100 = dot(hash33Periodic(i0 + ivec3(1, 0, 0), period) * 2.0 - 1.0, f - vec3(1.0, 0.0, 0.0));
    float n010 = dot(hash33Periodic(i0 + ivec3(0, 1, 0), period) * 2.0 - 1.0, f - vec3(0.0, 1.0, 0.0));
    float n110 = dot(hash33Periodic(i0 + ivec3(1, 1, 0), period) * 2.0 - 1.0, f - vec3(1.0, 1.0, 0.0));
    float n001 = dot(hash33Periodic(i0 + ivec3(0, 0, 1), period) * 2.0 - 1.0, f - vec3(0.0, 0.0, 1.0));
    float n101 = dot(hash33Periodic(i0 + ivec3(1, 0, 1), period) * 2.0 - 1.0, f - vec3(1.0, 0.0, 1.0));
    float n011 = dot(hash33Periodic(i0 + ivec3(0, 1, 1), period) * 2.0 - 1.0, f - vec3(0.0, 1.0, 1.0));
    float n111 = dot(hash33Periodic(i0 + ivec3(1, 1, 1), period) * 2.0 - 1.0, f - vec3(1.0, 1.0, 1.0));

    return mix(
      mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
      mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
      u.z
    );
  }

  float worleyInvertedPeriodic(vec3 uvw, int period) {
    vec3 p = uvw * float(period);
    ivec3 base = ivec3(floor(p));
    float minD2 = 10.0;
    for (int dz = -1; dz <= 1; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          ivec3 c = base + ivec3(dx, dy, dz);
          vec3 feat = vec3(c) + hash33Periodic(c, period);
          vec3 d = feat - p;
          minD2 = min(minD2, dot(d, d));
        }
      }
    }
    return 1.0 - clamp(sqrt(minD2), 0.0, 1.0);
  }

  void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    int tx = px.x / 64;
    int ty = px.y / 64;
    int ix = px.x - tx * 64;
    int iy = px.y - ty * 64;
    int iz = ty * 8 + tx;
    vec3 uvw = (vec3(float(ix), float(iy), float(iz)) + 0.5) / 64.0;

    // R: High-contrast Perlin-Worley sculptural cauliflower domes (periods 4, 8, 16)
    float pFbm = 0.56 * gradientNoisePeriodic(uvw, 4)
               + 0.29 * gradientNoisePeriodic(uvw, 8)
               + 0.15 * gradientNoisePeriodic(uvw, 16);
    float perlin01 = clamp(pFbm * 0.94 + 0.5, 0.0, 1.0);
    float wFbmR = 0.60 * worleyInvertedPeriodic(uvw, 4)
                + 0.27 * worleyInvertedPeriodic(uvw, 8)
                + 0.13 * worleyInvertedPeriodic(uvw, 16);
    float pw = clamp(perlin01 * 0.44 + wFbmR * 0.68 - (1.0 - wFbmR) * (1.0 - perlin01) * 0.34, 0.0, 1.0);
    float chR = smoothstep(0.08, 0.94, pw);

    // G: Deep 3-octave inverted Worley cellular florets & 3D canyons (periods 8, 16, 32)
    float w8  = worleyInvertedPeriodic(uvw, 8);
    float w16 = worleyInvertedPeriodic(uvw, 16);
    float w32 = worleyInvertedPeriodic(uvw, 32);
    float chG = clamp(pow(clamp(0.56 * w8 + 0.29 * w16 + 0.15 * w32, 0.0, 1.0), 1.18), 0.0, 1.0);

    // B: Curl-warped silk / herringbone filamentary streaks for 3D cirrus mares' tails
    float warpX = gradientNoisePeriodic(uvw + vec3(0.31, 0.17, 0.53), 4) * 0.085;
    float warpY = gradientNoisePeriodic(uvw + vec3(0.73, 0.41, 0.19), 4) * 0.085;
    vec3 warped = fract(uvw + vec3(warpX, warpY, -warpX * 0.5));
    float r1 = 1.0 - abs(gradientNoisePeriodic(warped, 6) * 2.0);
    float r2 = 1.0 - abs(gradientNoisePeriodic(warped + vec3(0.19, 0.37, 0.11), 12) * 2.0);
    float r3 = 1.0 - abs(gradientNoisePeriodic(warped * vec3(1.0, 2.0, 1.0), 12) * 2.0);
    float chB = clamp(pow(clamp(0.50 * r1 + 0.32 * r2 + 0.18 * r3, 0.0, 1.0), 1.45), 0.0, 1.0);

    // A: Crisp 1-meter Kolmogorov (-5/3) gossamer wisps & micro-turbulence cascade (periods 16 & 32)
    float k16 = gradientNoisePeriodic(uvw, 16) * 0.5 + 0.5;
    float k32 = gradientNoisePeriodic(uvw + vec3(0.13, 0.29, 0.47), 32) * 0.5 + 0.5;
    float wFine = worleyInvertedPeriodic(uvw, 24);
    float chA = clamp(0.42 * k16 + 0.28 * k32 + 0.30 * wFine, 0.0, 1.0);

    fragColor = vec4(chR, chG, chB, chA);
  }
`;

// ============================================================================
// GPU Terrain DEM Downsampler + Sobel Slope & Ridge Prominence Extractor (128x128)
// ============================================================================
const TERRAIN_DOWNSAMPLE_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uHeightTex;
  uniform float uHasHeightTex;
  uniform float uSpanM;
  uniform float uDomainXyM;
  uniform float uMinElevM;

  float sampleDemWorld(vec2 eastNorthM) {
    if (uHasHeightTex < 0.5) return uMinElevM;
    float halfSpan = uSpanM * 0.5;
    vec2 uv = (eastNorthM + vec2(halfSpan)) / max(uSpanM, 1.0);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      vec2 clamped = clamp(uv, 0.0, 1.0);
      float edgeZ = texture(uHeightTex, clamped).r;
      vec2 overM = max(vec2(0.0), abs(eastNorthM) - vec2(halfSpan));
      float falloff = exp(-length(overM) / 1800.0);
      return mix(uMinElevM, edgeZ, falloff);
    }
    return texture(uHeightTex, uv).r;
  }

  void main() {
    vec2 en = (vUv - 0.5) * uDomainXyM;
    float dx = uDomainXyM / 128.0;
    float zC = sampleDemWorld(en);
    float zE = sampleDemWorld(en + vec2(dx, 0.0));
    float zW = sampleDemWorld(en - vec2(dx, 0.0));
    float zN = sampleDemWorld(en + vec2(0.0, dx));
    float zS = sampleDemWorld(en - vec2(0.0, dx));
    float zFar = 0.25 * (
      sampleDemWorld(en + vec2(dx * 4.0, 0.0)) +
      sampleDemWorld(en - vec2(dx * 4.0, 0.0)) +
      sampleDemWorld(en + vec2(0.0, dx * 4.0)) +
      sampleDemWorld(en - vec2(0.0, dx * 4.0))
    );
    float dhDx = clamp((zE - zW) / (2.0 * dx), -1.8, 1.8);
    float dhDy = clamp((zN - zS) / (2.0 * dx), -1.8, 1.8);
    float prominence = max(0.0, zC - zFar);
    fragColor = vec4(zC, dhDx, dhDy, prominence);
  }
`;

// Shared GLSL3 helper header for all 128x128x48 3D Moist Navier-Stokes GPU passes
const FLUID_COMMON_HEADER_GLSL3 = /* glsl */ `
  precision highp float;

  const int NX = 128;
  const int NY = 128;
  const int NZ = 48;
  const int TILES_X = 8;

  uniform float uDomainXyM;
  uniform float uZBaseM;
  uniform float uZTopM;
  uniform float uSurfaceTempC;
  uniform float uSurfaceRhPct;
  uniform float uLowCloudPct;
  uniform float uMidCloudPct;
  uniform float uHighCloudPct;
  uniform float uInversionBaseM;
  uniform float uInversionThicknessM;
  uniform float uInversionStrengthK;
  uniform vec2 uWindSfc;
  uniform vec2 uWind850;
  uniform vec2 uWind500;
  uniform vec2 uWind250;
  uniform float uConvectiveCape;
  uniform float uOrographicWaveGain;
  uniform float uVorticityConfinement;
  uniform float uCirrusIceBoost;
  uniform vec3 uSunDirWorld;
  uniform float uSunElDeg;
  uniform float uDt;
  uniform float uSimTime;

  ivec3 fragToVoxel(ivec2 px) {
    int tx = px.x / NX;
    int ty = px.y / NY;
    return ivec3(px.x - tx * NX, px.y - ty * NY, ty * TILES_X + tx);
  }

  ivec2 voxelToAtlas(int x, int y, int z) {
    int wx = (x + 1024) & 127;
    int wy = (y + 1024) & 127;
    int cz = clamp(z, 0, 47);
    int tx = cz & 7;
    int ty = cz >> 3;
    return ivec2(tx * 128 + wx, ty * 128 + wy);
  }

  float sToAltitudeM(float s) {
    float span = max(1000.0, uZTopM - uZBaseM);
    return uZBaseM + span * (0.26 * s + 0.74 * s * s);
  }

  float altitudeToS(float zAsl) {
    float span = max(1000.0, uZTopM - uZBaseM);
    float zeta = clamp((zAsl - uZBaseM) / span, 0.0, 1.0);
    return clamp((-0.26 + sqrt(0.0676 + 2.96 * zeta)) / 1.48, 0.0, 1.0);
  }

  float layerDzM(int iz) {
    float s0 = float(max(0, iz - 1)) / 47.0;
    float s1 = float(min(47, iz + 1)) / 47.0;
    return max(25.0, (sToAltitudeM(s1) - sToAltitudeM(s0)) / max(1.0, float(min(47, iz + 1) - max(0, iz - 1))));
  }

  void ambientSounding(float zAsl, out float tempK, out float pressPa, out float thetaAmb, out float qsatLiq, out float qsatIce) {
    float zRel = max(0.0, zAsl - uZBaseM);
    float tSfc = 273.15 + uSurfaceTempC;
    float invJump = uInversionStrengthK * smoothstep(
      uInversionBaseM,
      uInversionBaseM + max(80.0, uInversionThicknessM),
      zRel
    ) * exp(-max(0.0, zRel - uInversionBaseM - uInversionThicknessM) / 2200.0);
    tempK = max(214.0, tSfc - 0.0063 * zRel + invJump);
    pressPa = 101325.0 * pow(max(0.16, 1.0 - 0.0065 * max(0.0, zAsl) / 288.15), 5.25588);
    thetaAmb = tempK * pow(100000.0 / max(5000.0, pressPa), 0.2854);

    float tc = tempK - 273.15;
    float esLiq = 611.2 * exp(17.67 * tc / max(20.0, tc + 243.5));
    float esIce = 611.15 * exp(22.46 * tc / max(12.0, tc + 272.62));
    qsatLiq = 0.622 * esLiq / max(1200.0, pressPa - 0.378 * esLiq);
    qsatIce = 0.622 * esIce / max(1200.0, pressPa - 0.378 * esIce);
  }

  vec2 ambientWindAtZ(float zAsl) {
    float zRel = max(0.0, zAsl - uZBaseM);
    if (zRel < 1500.0) {
      return mix(uWindSfc, uWind850, zRel / 1500.0);
    } else if (zRel < 5600.0) {
      return mix(uWind850, uWind500, (zRel - 1500.0) / 4100.0);
    }
    return mix(uWind500, uWind250, clamp((zRel - 5600.0) / 4800.0, 0.0, 1.25));
  }

  float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
  }

  float valueNoisePeriodic2D(vec2 uv, float period) {
    vec2 p = uv * period;
    vec2 i = floor(p);
    vec2 f = p - i;
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 i00 = mod(i, period);
    vec2 i10 = mod(i + vec2(1.0, 0.0), period);
    vec2 i01 = mod(i + vec2(0.0, 1.0), period);
    vec2 i11 = mod(i + vec2(1.0, 1.0), period);
    float a = hash13(vec3(i00, period));
    float b = hash13(vec3(i10, period));
    float c = hash13(vec3(i01, period));
    float d = hash13(vec3(i11, period));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  }

  float sculptedCloudMacro2D(vec2 uv) {
    float wx = valueNoisePeriodic2D(uv + vec2(0.17, 0.43), 3.0) - 0.5;
    float wy = valueNoisePeriodic2D(uv + vec2(0.61, 0.29), 3.0) - 0.5;
    vec2 wUv = uv + vec2(wx, wy) * 0.14;
    float n4  = valueNoisePeriodic2D(wUv, 4.0);
    float n8  = valueNoisePeriodic2D(wUv + vec2(0.31, 0.19), 8.0);
    float n16 = valueNoisePeriodic2D(wUv + vec2(0.73, 0.53), 16.0);
    float ridge = 1.0 - abs(n8 * 2.0 - 1.0);
    float raw = 0.50 * n4 + 0.28 * n8 + 0.12 * n16 + 0.10 * ridge;
    return smoothstep(0.16, 0.84, raw);
  }

  vec4 sampleStateTrilinearWithBounds(
    sampler2D stateTex,
    vec2 xyIdx,
    float zIdx,
    out vec4 minB,
    out vec4 maxB
  ) {
    vec2 f0 = floor(xyIdx);
    vec2 tf = xyIdx - f0;
    int x0 = int(f0.x);
    int y0 = int(f0.y);
    int iz0 = clamp(int(floor(zIdx)), 0, 47);
    int iz1 = min(47, iz0 + 1);
    float tz = clamp(zIdx - float(iz0), 0.0, 1.0);

    vec4 c000 = texelFetch(stateTex, voxelToAtlas(x0,     y0,     iz0), 0);
    vec4 c100 = texelFetch(stateTex, voxelToAtlas(x0 + 1, y0,     iz0), 0);
    vec4 c010 = texelFetch(stateTex, voxelToAtlas(x0,     y0 + 1, iz0), 0);
    vec4 c110 = texelFetch(stateTex, voxelToAtlas(x0 + 1, y0 + 1, iz0), 0);
    vec4 slice0 = mix(mix(c000, c100, tf.x), mix(c010, c110, tf.x), tf.y);

    vec4 c001 = texelFetch(stateTex, voxelToAtlas(x0,     y0,     iz1), 0);
    vec4 c101 = texelFetch(stateTex, voxelToAtlas(x0 + 1, y0,     iz1), 0);
    vec4 c011 = texelFetch(stateTex, voxelToAtlas(x0,     y0 + 1, iz1), 0);
    vec4 c111 = texelFetch(stateTex, voxelToAtlas(x0 + 1, y0 + 1, iz1), 0);
    vec4 slice1 = mix(mix(c001, c101, tf.x), mix(c011, c111, tf.x), tf.y);

    minB = min(min(min(c000, c100), min(c010, c110)), min(min(c001, c101), min(c011, c111)));
    maxB = max(max(max(c000, c100), max(c010, c110)), max(max(c001, c101), max(c011, c111)));
    return mix(slice0, slice1, tz);
  }

  vec4 sampleStateTrilinear(sampler2D stateTex, vec2 xyIdx, float zIdx) {
    vec4 minB, maxB;
    return sampleStateTrilinearWithBounds(stateTex, xyIdx, zIdx, minB, maxB);
  }

  // Clamped 2nd-Order MacCormack (Predictor-Corrector) Semi-Lagrangian Advection
  // Preserves crisp 3D cloud edges & cirrus hooks without numerical trilinear smearing!
  vec4 sampleStateMacCormack(
    sampler2D stateTex,
    vec2 curXY,
    float curZAsl,
    vec2 velXY,
    float velZ,
    float dt,
    float dx
  ) {
    vec2 depXY = curXY - (velXY * dt) / dx;
    float depZAsl = clamp(curZAsl - velZ * dt, uZBaseM, uZTopM);
    float depZIdx = altitudeToS(depZAsl) * 47.0;

    vec4 minB, maxB;
    vec4 phiHat = sampleStateTrilinearWithBounds(stateTex, depXY, depZIdx, minB, maxB);

    // Reverse forward step from departure point back toward arrival voxel
    vec2 fwdXY = depXY + (velXY * dt) / dx;
    float fwdZAsl = clamp(depZAsl + velZ * dt, uZBaseM, uZTopM);
    float fwdZIdx = altitudeToS(fwdZAsl) * 47.0;
    vec4 phiTilde = sampleStateTrilinear(stateTex, fwdXY, fwdZIdx);

    vec4 phiOrigin = texelFetch(stateTex, voxelToAtlas(int(curXY.x), int(curXY.y), int(round(altitudeToS(curZAsl) * 47.0))), 0);
    vec4 corrected = phiHat + 0.5 * (phiOrigin - phiTilde);
    return clamp(corrected, minB, maxB);
  }
`;

// GPU Fluid Init / Weather Regime Seeding Pass (writes Velocity or Thermo depending on uInitMode)
const FLUID_INIT_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uTerrainGridTex;
  uniform float uInitMode; // 0.0 = velocity, 1.0 = thermodynamics
  out vec4 fragColor;

  void main() {
    ivec3 vox = fragToVoxel(ivec2(gl_FragCoord.xy));
    float s = float(vox.z) / 47.0;
    float zAsl = sToAltitudeM(s);
    float zRel = max(0.0, zAsl - uZBaseM);
    vec4 terr = texelFetch(uTerrainGridTex, vox.xy, 0);
    float elevM = terr.r;
    float promM = terr.a;

    if (uInitMode < 0.5) {
      if (zAsl <= elevM) {
        fragColor = vec4(0.0);
        return;
      }
      vec2 wAmb = ambientWindAtZ(zAsl);
      float agl = zAsl - elevM;
      float wOro = (wAmb.x * terr.g + wAmb.y * terr.b) * exp(-agl / 1500.0) * uOrographicWaveGain;
      fragColor = vec4(wAmb.x, wAmb.y, clamp(wOro, -6.0, 6.0), 0.0);
      return;
    }

    float tempK, pressPa, thetaAmb, qsatLiq, qsatIce;
    ambientSounding(zAsl, tempK, pressPa, thetaAmb, qsatLiq, qsatIce);

    vec2 uv = (vec2(vox.xy) + 0.5) / 128.0;
    float cellMacro = sculptedCloudMacro2D(uv);
    float midWave = 0.65 * valueNoisePeriodic2D(uv + vec2(0.31, 0.67), 4.0)
                  + 0.35 * valueNoisePeriodic2D(uv + vec2(0.12, 0.83), 8.0);

    // Physical Lifting Condensation Level (LCL) + sub-inversion marine stratus profile
    float lowCov = clamp(uLowCloudPct / 100.0, 0.0, 1.0);
    float lclRelM = max(130.0, (100.0 - clamp(uSurfaceRhPct, 15.0, 100.0)) * 25.0);
    float invTop = uInversionBaseM + max(90.0, uInversionThicknessM * 0.85);
    float stratusRegime = smoothstep(4.2, 7.2, uInversionStrengthK) * (1.0 - clamp(uConvectiveCape / 520.0, 0.0, 0.85));

    float stratusBand = smoothstep(max(20.0, uInversionBaseM - 220.0), uInversionBaseM - 15.0, zRel)
                      * (1.0 - smoothstep(invTop - 25.0, invTop + 115.0, zRel));

    // Summit prominence anchor: peaks/ridges seed persistent 3D orographic cumulus & banner clouds
    float summitRoot = clamp(promM / 280.0, 0.0, 1.0)
                     * smoothstep(40.0, 260.0, zAsl - elevM)
                     * exp(-max(0.0, zAsl - elevM - 240.0) / 1150.0);

    float lowField = clamp(cellMacro + summitRoot * 0.24, 0.0, 1.0);
    float lowThresh = 1.0 - lowCov * 0.78 - 0.05;
    float cellStrength = smoothstep(lowThresh, 0.98, lowField);
    float turretVar = valueNoisePeriodic2D(uv + vec2(0.47, 0.23), 8.0);

    // Tapered 3D pyramidal/cauliflower dome envelope: wide flat base at LCL, tapering inward with altitude
    // so only thermal plume centers reach high altitude (zero flat-topped cylinders!)
    float maxTowerSpan = 520.0 + clamp(uConvectiveCape, 0.0, 1400.0) * 1.15;
    float localCuTopRel = lclRelM + 160.0 + maxTowerSpan * pow(cellStrength, 1.35) * (0.58 + 0.60 * turretVar);
    float cumulusDome = smoothstep(lclRelM - 40.0, lclRelM + 75.0, zRel)
                      * smoothstep(localCuTopRel + 120.0, localCuTopRel - 320.0, zRel);

    float lowCellMask = smoothstep(lowThresh, lowThresh + 0.22 + lowCov * 0.28, lowField);
    float qcCumulus = lowCov * cellStrength * cumulusDome * 0.00195;
    float qcStratus = lowCov * lowCellMask * stratusBand * 0.00165;
    float qcLow = mix(qcCumulus, qcStratus, stratusRegime);

    // Mid-level altocumulus / standing lenticular wave lenses (tapered 3D lens thickness)
    float midCov = clamp(uMidCloudPct / 100.0, 0.0, 1.0);
    float leeCrest = clamp(promM / 320.0, 0.0, 1.0) * clamp((uOrographicWaveGain - 0.85) * 0.55, 0.0, 0.55);
    float midField = clamp(midWave + leeCrest * 0.30, 0.0, 1.0);
    float midStrength = smoothstep(1.0 - midCov * 0.76 - 0.05, 0.96, midField);
    float midCenterZ = 3350.0 + (turretVar - 0.5) * 650.0;
    float midHalfThick = 180.0 + 1150.0 * pow(midStrength, 1.15);
    float midLens = smoothstep(midCenterZ - midHalfThick - 220.0, midCenterZ - midHalfThick + 120.0, zRel)
                  * smoothstep(midCenterZ + midHalfThick + 220.0, midCenterZ + midHalfThick - 120.0, zRel);
    float qcMid = midCov * midStrength * midLens * 0.00135;

    float qc = (zAsl > elevM + 10.0) ? (qcLow + qcMid) : 0.0;

    // High 3D Cirrus Uncinus & Vertebratus ice crystal layer (6.7 - 11.8 km ASL) with clear sky channels
    float highCov = clamp(uHighCloudPct / 100.0, 0.0, 1.0);
    vec2 jetDir = normalize(uWind250 + vec2(1e-3));
    vec2 crossDir = vec2(-jetDir.y, jetDir.x);
    vec2 jetProj = vec2(dot(uv, jetDir), dot(uv, crossDir));
    float ciStreak = 0.62 * valueNoisePeriodic2D(vec2(jetProj.x * 0.5, jetProj.y * 2.0), 4.0)
                   + 0.38 * valueNoisePeriodic2D(vec2(jetProj.x * 1.0, jetProj.y * 3.0) + vec2(0.37, 0.19), 8.0);
    float ciRibs = 0.5 + 0.5 * sin(jetProj.x * 6.2831853 * 5.0 + ciStreak * 2.6);
    float cirrusPatch = clamp(ciStreak * (0.80 + 0.28 * ciRibs), 0.0, 1.0);
    float cirrusAlt = smoothstep(6700.0, 8500.0, zAsl) * (1.0 - smoothstep(10850.0, 11950.0, zAsl));
    float cirrusMask = smoothstep(1.0 - highCov * 0.72 - 0.04, 1.0 - highCov * 0.28 + 0.16, cirrusPatch);
    float qi = highCov * cirrusMask * cirrusAlt * 0.00048 * uCirrusIceBoost;

    float rhProfile = mix(uSurfaceRhPct / 100.0, 0.34 + 0.44 * midCov, smoothstep(0.0, 4200.0, zRel));
    float qv = qsatLiq * clamp(rhProfile * (0.80 + 0.34 * cellMacro), 0.12, 0.99);
    if (qc > 1e-5) {
      qv = max(qv, qsatLiq * 1.015);
    }
    float thetaPrime = (cellMacro - 0.5) * 0.52 * exp(-zRel / 2200.0);
    fragColor = vec4(thetaPrime, qv, qc, qi);
  }
`;

// GPU Fluid Pass 1: Orographic Uplift + 3D Scorer Lee Waves + Solar Slope Thermals + Moist Buoyancy + Curl
const FLUID_FORCING_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uVelTex;
  uniform sampler2D uThermoTex;
  uniform sampler2D uTerrainGridTex;
  out vec4 fragColor;

  void main() {
    ivec3 vox = fragToVoxel(ivec2(gl_FragCoord.xy));
    float s = float(vox.z) / 47.0;
    float zAsl = sToAltitudeM(s);
    vec4 terr = texelFetch(uTerrainGridTex, vox.xy, 0);
    float elevM = terr.r;

    if (zAsl <= elevM) {
      fragColor = vec4(0.0);
      return;
    }

    vec4 vel = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z), 0);
    vec4 th = texelFetch(uThermoTex, voxelToAtlas(vox.x, vox.y, vox.z), 0);

    float tempK, pressPa, thetaAmb, qsatLiq, qsatIce;
    ambientSounding(zAsl, tempK, pressPa, thetaAmb, qsatLiq, qsatIce);

    vec2 wAmb = ambientWindAtZ(zAsl);
    float agl = max(0.0, zAsl - elevM);

    // 1. Kinematic terrain-following slope uplift + standing 3D Scorer mountain gravity lee-waves
    float dhDx = terr.g;
    float dhDy = terr.b;
    float prom = terr.a;
    float wSlope = (vel.x * dhDx + vel.y * dhDy) * exp(-agl / 1550.0) * uOrographicWaveGain;

    vec2 windDir500 = normalize(uWind500 + vec2(1e-3));
    vec2 posM = ((vec2(vox.xy) + 0.5) / 128.0 - 0.5) * uDomainXyM;
    float downwindPhase = dot(posM, windDir500) * 0.00165 + zAsl * 0.00135;
    float leeEnv = smoothstep(900.0, 2200.0, zAsl) * (1.0 - smoothstep(6200.0, 8400.0, zAsl));
    float wScorer = sin(downwindPhase) * (0.28 + 0.0024 * prom) * (uOrographicWaveGain - 0.55) * leeEnv
                  * clamp(length(uWind500) / 18.0, 0.2, 1.8);

    // 2. Solar aspect-weighted anabatic slope thermals (drives cumulus over Half Dome / ridges!)
    vec2 sunHoriz = normalize(vec2(uSunDirWorld.x, -uSunDirWorld.z) + vec2(1e-4));
    float sunSlope = max(0.0, dot(normalize(vec2(-dhDx, -dhDy) + vec2(1e-4)), sunHoriz))
                   * length(vec2(dhDx, dhDy));
    float solarHeating = smoothstep(-2.0, 18.0, uSunElDeg) * (0.25 + 0.75 * clamp(uConvectiveCape / 850.0, 0.0, 1.4));
    float wThermal = sunSlope * solarHeating * 2.1 * exp(-agl / 850.0);

    // 3. Moist Boussinesq buoyancy (virtual potential temperature + hydrometeor drag)
    float qvAmb = qsatLiq * clamp(uSurfaceRhPct / 100.0, 0.2, 0.92) * exp(-max(0.0, zAsl - uZBaseM) / 3600.0);
    float buoy = 9.81 * (
      clamp(th.x / max(240.0, thetaAmb), -0.025, 0.035) +
      0.608 * clamp(th.y - qvAmb, -0.004, 0.006) -
      1.25 * (th.z + th.w)
    );

    float dt = clamp(uDt, 0.05, 6.0);
    // Gentle nudging toward synoptic wind profile + tropopause Rayleigh sponge layer
    float relaxRate = mix(0.035, 0.16, smoothstep(9800.0, 12000.0, zAsl));
    vec2 uvNext = mix(vel.xy, wAmb, clamp(relaxRate * dt, 0.0, 0.45));

    float wTargetOro = wSlope + max(-2.8, min(2.8, wScorer));
    float wNext = vel.z + dt * (buoy * 0.28 + (wTargetOro + wThermal - vel.z) * 0.22);
    // Damp vertical velocity smoothly at ground and tropopause lid
    float lidDamp = smoothstep(0.0, 90.0, agl) * (1.0 - smoothstep(11100.0, 12150.0, zAsl));
    wNext = clamp(wNext * lidDamp, -9.5, 11.5);

    // Compute 3D curl magnitude |nabla x u| for Pass 2 Vorticity Confinement
    float dx = uDomainXyM / 128.0;
    float dz = layerDzM(vox.z);
    vec3 vE = texelFetch(uVelTex, voxelToAtlas(vox.x + 1, vox.y, vox.z), 0).xyz;
    vec3 vW = texelFetch(uVelTex, voxelToAtlas(vox.x - 1, vox.y, vox.z), 0).xyz;
    vec3 vN = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y + 1, vox.z), 0).xyz;
    vec3 vS = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y - 1, vox.z), 0).xyz;
    vec3 vU = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z + 1), 0).xyz;
    vec3 vD = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z - 1), 0).xyz;

    vec3 omega = vec3(
      (vN.z - vS.z) / (2.0 * dx) - (vU.y - vD.y) / (2.0 * dz),
      (vU.x - vD.x) / (2.0 * dz) - (vE.z - vW.z) / (2.0 * dx),
      (vE.y - vW.y) / (2.0 * dx) - (vN.x - vS.x) / (2.0 * dx)
    );
    fragColor = vec4(uvNext, wNext, length(omega));
  }
`;

// GPU Fluid Pass 2a: 3D Vorticity Confinement (restores sub-grid Kelvin-Helmholtz & cumulus billow curls)
const FLUID_VORTICITY_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uVelTex;
  uniform sampler2D uTerrainGridTex;
  out vec4 fragColor;

  void main() {
    ivec3 vox = fragToVoxel(ivec2(gl_FragCoord.xy));
    float zAsl = sToAltitudeM(float(vox.z) / 47.0);
    float elevM = texelFetch(uTerrainGridTex, vox.xy, 0).r;
    if (zAsl <= elevM) {
      fragColor = vec4(0.0);
      return;
    }

    vec4 c = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z), 0);
    vec4 vE = texelFetch(uVelTex, voxelToAtlas(vox.x + 1, vox.y, vox.z), 0);
    vec4 vW = texelFetch(uVelTex, voxelToAtlas(vox.x - 1, vox.y, vox.z), 0);
    vec4 vN = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y + 1, vox.z), 0);
    vec4 vS = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y - 1, vox.z), 0);
    vec4 vU = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z + 1), 0);
    vec4 vD = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z - 1), 0);

    float dx = uDomainXyM / 128.0;
    float dz = layerDzM(vox.z);
    vec3 omega = vec3(
      (vN.z - vS.z) / (2.0 * dx) - (vU.y - vD.y) / (2.0 * dz),
      (vU.x - vD.x) / (2.0 * dz) - (vE.z - vW.z) / (2.0 * dx),
      (vE.y - vW.y) / (2.0 * dx) - (vN.x - vS.x) / (2.0 * dx)
    );
    vec3 gradMag = vec3(vE.w - vW.w, vN.w - vS.w, vU.w - vD.w);
    float gLen = length(gradMag);
    vec3 fConf = (gLen > 1e-5)
      ? cross(gradMag / gLen, omega) * (uVorticityConfinement * 16.0)
      : vec3(0.0);

    float dt = clamp(uDt, 0.05, 6.0);
    vec3 nextVel = clamp(c.xyz + dt * fConf * vec3(1.0, 1.0, 0.55), vec3(-75.0, -75.0, -9.5), vec3(75.0, 75.0, 11.5));
    fragColor = vec4(nextVel, c.w);
  }
`;

// GPU Fluid Pass 2b: Velocity Divergence + Warm-Started Pressure Seed + Solid Terrain Mask
const FLUID_DIVERGENCE_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uVelTex;
  uniform sampler2D uPrevDivPressTex;
  uniform sampler2D uTerrainGridTex;
  out vec4 fragColor;

  void main() {
    ivec3 vox = fragToVoxel(ivec2(gl_FragCoord.xy));
    float zAsl = sToAltitudeM(float(vox.z) / 47.0);
    float elevM = texelFetch(uTerrainGridTex, vox.xy, 0).r;
    if (zAsl <= elevM) {
      fragColor = vec4(0.0, 0.0, 1.0, 0.0);
      return;
    }

    float dx = uDomainXyM / 128.0;
    float dz = layerDzM(vox.z);
    vec3 vE = texelFetch(uVelTex, voxelToAtlas(vox.x + 1, vox.y, vox.z), 0).xyz;
    vec3 vW = texelFetch(uVelTex, voxelToAtlas(vox.x - 1, vox.y, vox.z), 0).xyz;
    vec3 vN = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y + 1, vox.z), 0).xyz;
    vec3 vS = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y - 1, vox.z), 0).xyz;
    vec3 vU = (vox.z < 47) ? texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z + 1), 0).xyz : vec3(0.0);
    vec3 vD = (vox.z > 0)  ? texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z - 1), 0).xyz : vec3(0.0);

    float div = (vE.x - vW.x) / (2.0 * dx) + (vN.y - vS.y) / (2.0 * dx) + (vU.z - vD.z) / (2.0 * dz);
    float warmP = texelFetch(uPrevDivPressTex, voxelToAtlas(vox.x, vox.y, vox.z), 0).y * 0.75;
    fragColor = vec4(div, warmP, 0.0, 0.0);
  }
`;

// GPU Fluid Pass 3: Anisotropic 3D Jacobi Pressure Poisson Solver with Neumann Terrain Boundaries
const FLUID_PRESSURE_JACOBI_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uDivPressTex;
  out vec4 fragColor;

  void main() {
    ivec3 vox = fragToVoxel(ivec2(gl_FragCoord.xy));
    vec4 center = texelFetch(uDivPressTex, voxelToAtlas(vox.x, vox.y, vox.z), 0);
    if (center.z > 0.5) {
      fragColor = center;
      return;
    }

    float pC = center.y;
    vec4 nE = texelFetch(uDivPressTex, voxelToAtlas(vox.x + 1, vox.y, vox.z), 0);
    vec4 nW = texelFetch(uDivPressTex, voxelToAtlas(vox.x - 1, vox.y, vox.z), 0);
    vec4 nN = texelFetch(uDivPressTex, voxelToAtlas(vox.x, vox.y + 1, vox.z), 0);
    vec4 nS = texelFetch(uDivPressTex, voxelToAtlas(vox.x, vox.y - 1, vox.z), 0);
    vec4 nU = (vox.z < 47) ? texelFetch(uDivPressTex, voxelToAtlas(vox.x, vox.y, vox.z + 1), 0) : vec4(0.0, 0.0, 0.0, 0.0);
    vec4 nD = (vox.z > 0)  ? texelFetch(uDivPressTex, voxelToAtlas(vox.x, vox.y, vox.z - 1), 0) : vec4(0.0, pC, 1.0, 0.0);

    float pE = nE.z > 0.5 ? pC : nE.y;
    float pW = nW.z > 0.5 ? pC : nW.y;
    float pN = nN.z > 0.5 ? pC : nN.y;
    float pS = nS.z > 0.5 ? pC : nS.y;
    float pU = (vox.z >= 47) ? 0.0 : (nU.z > 0.5 ? pC : nU.y);
    float pD = nD.z > 0.5 ? pC : nD.y;

    float dx = uDomainXyM / 128.0;
    float dz = layerDzM(vox.z);
    float invDx2 = 1.0 / (dx * dx);
    float invDz2 = 1.0 / (dz * dz);
    float denom = 4.0 * invDx2 + 2.0 * invDz2;
    float pNext = ((pE + pW + pN + pS) * invDx2 + (pU + pD) * invDz2 - center.x) / denom;

    fragColor = vec4(center.x, pNext, 0.0, 0.0);
  }
`;

// GPU Fluid Pass 4a: Divergence-Free Velocity Projection + Semi-Lagrangian 3D Velocity Advection
const FLUID_PROJECT_ADVECT_VEL_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uVelTex;
  uniform sampler2D uDivPressTex;
  uniform sampler2D uTerrainGridTex;
  out vec4 fragColor;

  vec3 readProjectedVel(ivec3 v) {
    vec4 c = texelFetch(uDivPressTex, voxelToAtlas(v.x, v.y, v.z), 0);
    if (c.z > 0.5) return vec3(0.0);
    vec3 uStar = texelFetch(uVelTex, voxelToAtlas(v.x, v.y, v.z), 0).xyz;
    float pC = c.y;
    vec4 nE = texelFetch(uDivPressTex, voxelToAtlas(v.x + 1, v.y, v.z), 0);
    vec4 nW = texelFetch(uDivPressTex, voxelToAtlas(v.x - 1, v.y, v.z), 0);
    vec4 nN = texelFetch(uDivPressTex, voxelToAtlas(v.x, v.y + 1, v.z), 0);
    vec4 nS = texelFetch(uDivPressTex, voxelToAtlas(v.x, v.y - 1, v.z), 0);
    vec4 nU = (v.z < 47) ? texelFetch(uDivPressTex, voxelToAtlas(v.x, v.y, v.z + 1), 0) : vec4(0.0);
    vec4 nD = (v.z > 0)  ? texelFetch(uDivPressTex, voxelToAtlas(v.x, v.y, v.z - 1), 0) : vec4(0.0, pC, 1.0, 0.0);

    float dx = uDomainXyM / 128.0;
    float dz = layerDzM(v.z);
    float dpdx = ((nE.z > 0.5 ? pC : nE.y) - (nW.z > 0.5 ? pC : nW.y)) / (2.0 * dx);
    float dpdy = ((nN.z > 0.5 ? pC : nN.y) - (nS.z > 0.5 ? pC : nS.y)) / (2.0 * dx);
    float dpdz = ((v.z >= 47 ? 0.0 : (nU.z > 0.5 ? pC : nU.y)) - (nD.z > 0.5 ? pC : nD.y)) / (2.0 * dz);
    return uStar - vec3(dpdx, dpdy, dpdz * 0.65);
  }

  void main() {
    ivec3 vox = fragToVoxel(ivec2(gl_FragCoord.xy));
    float s = float(vox.z) / 47.0;
    float zAsl = sToAltitudeM(s);
    float elevM = texelFetch(uTerrainGridTex, vox.xy, 0).r;
    if (zAsl <= elevM) {
      fragColor = vec4(0.0);
      return;
    }

    vec3 uProj = readProjectedVel(vox);
    float vort = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z), 0).w;

    // Semi-Lagrangian backtrace in physical meters -> stretched grid coordinates
    float dx = uDomainXyM / 128.0;
    float dt = clamp(uDt, 0.05, 6.0);
    vec2 depXY = vec2(vox.xy) - (uProj.xy * dt) / dx;
    float depZAsl = clamp(zAsl - uProj.z * dt, uZBaseM, uZTopM);
    float depZIdx = altitudeToS(depZAsl) * 47.0;

    vec4 adv = sampleStateTrilinear(uVelTex, depXY, depZIdx);
    vec3 blended = mix(adv.xyz, uProj, 0.42);
    fragColor = vec4(clamp(blended, vec3(-75.0, -75.0, -9.5), vec3(75.0, 75.0, 11.5)), vort);
  }
`;

// GPU Fluid Pass 4b: Clamped 2nd-Order MacCormack Scalar & Sheared Ice Advection + Microphysics
const FLUID_ADVECT_MICROPHYSICS_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uVelTex;
  uniform sampler2D uThermoTex;
  uniform sampler2D uTerrainGridTex;
  uniform sampler2D uMesoscaleCloudTex;
  uniform float uMesoscaleSpanM;
  uniform float uHasMesoscaleGrid;
  out vec4 fragColor;

  void main() {
    ivec3 vox = fragToVoxel(ivec2(gl_FragCoord.xy));
    float s = float(vox.z) / 47.0;
    float zAsl = sToAltitudeM(s);
    float zRel = max(0.0, zAsl - uZBaseM);
    vec4 terr = texelFetch(uTerrainGridTex, vox.xy, 0);
    float elevM = terr.r;
    float promM = terr.a;

    float tempK, pressPa, thetaAmb, qsatLiq, qsatIce;
    ambientSounding(zAsl, tempK, pressPa, thetaAmb, qsatLiq, qsatIce);

    if (zAsl <= elevM) {
      fragColor = vec4(0.0, qsatLiq * 0.65, 0.0, 0.0);
      return;
    }

    vec4 vel = texelFetch(uVelTex, voxelToAtlas(vox.x, vox.y, vox.z), 0);
    float dx = uDomainXyM / 128.0;
    float dt = clamp(uDt, 0.04, 4.0);

    // 1. Clamped 2nd-Order MacCormack backtrace for warm/liquid air parcel (zero numerical diffusion!)
    vec4 advParcel = sampleStateMacCormack(uThermoTex, vec2(vox.xy), zAsl, vel.xy, vel.z, dt, dx);

    // 2. Separate MacCormack ice-crystal backtrace with gravitational fall speed (0.88 m/s) across 250->500 hPa shear!
    float iceFallW = vel.z - 0.88;
    float qiAdv = sampleStateMacCormack(uThermoTex, vec2(vox.xy), zAsl, vel.xy, iceFallW, dt, dx).w;

    float thetaPrime = advParcel.x;
    float qv = max(0.0, advParcel.y);
    float qc = max(0.0, advParcel.z);
    float qi = max(0.0, qiAdv);

    // 3. 1:1 Lagrangian synoptic boundary-layer moisture & summit-anchored cloud root maintenance
    vec2 uv = (vec2(vox.xy) + 0.5) / 128.0;
    float invDomain = 1.0 / max(1000.0, uDomainXyM);
    vec2 driftLowUv = fract(uv - uWindSfc * (uSimTime * invDomain));
    float cellWave = sculptedCloudMacro2D(driftLowUv);

    float lowCov = clamp(uLowCloudPct / 100.0, 0.0, 1.0);
    float midCov = clamp(uMidCloudPct / 100.0, 0.0, 1.0);
    float highCov = clamp(uHighCloudPct / 100.0, 0.0, 1.0);

    // Lateral mesoscale boundary relaxation: nudge outer edges of the 18km box toward the 160km NWP/GOES grid
    if (uHasMesoscaleGrid > 0.5) {
      vec2 enLocal = (uv - 0.5) * uDomainXyM;
      vec2 mesoUv = clamp(enLocal / max(20000.0, uMesoscaleSpanM) + 0.5, 0.03, 0.97);
      vec4 mesoCov = texture(uMesoscaleCloudTex, mesoUv);
      float edgeBlend = smoothstep(0.30, 0.49, max(abs(uv.x - 0.5), abs(uv.y - 0.5)));
      lowCov = mix(lowCov, max(mesoCov.r, mesoCov.a * 0.85), edgeBlend * 0.72);
      midCov = mix(midCov, mesoCov.g, edgeBlend * 0.72);
      highCov = mix(highCov, mesoCov.b, edgeBlend * 0.72);
    }

    float lclRelM = max(130.0, (100.0 - clamp(uSurfaceRhPct, 15.0, 100.0)) * 25.0);
    float invTop = uInversionBaseM + max(90.0, uInversionThicknessM * 0.85);
    float stratusRegime = smoothstep(4.2, 7.2, uInversionStrengthK) * (1.0 - clamp(uConvectiveCape / 520.0, 0.0, 0.85));

    float stratusBand = smoothstep(max(20.0, uInversionBaseM - 220.0), uInversionBaseM - 15.0, zRel)
                      * (1.0 - smoothstep(invTop - 25.0, invTop + 115.0, zRel));

    float summitRoot = clamp(promM / 280.0, 0.0, 1.0)
                     * smoothstep(40.0, 260.0, zAsl - elevM)
                     * exp(-max(0.0, zAsl - elevM - 240.0) / 1150.0);

    float lowField = clamp(cellWave + summitRoot * 0.24, 0.0, 1.0);
    float lowThresh = 1.0 - lowCov * 0.78 - 0.05;
    float cellStrength = smoothstep(lowThresh, 0.98, lowField);
    float turretVar = valueNoisePeriodic2D(driftLowUv + vec2(0.47, 0.23), 8.0);
    float maxTowerSpan = 520.0 + clamp(uConvectiveCape, 0.0, 1400.0) * 1.15;
    float localCuTopRel = lclRelM + 160.0 + maxTowerSpan * pow(cellStrength, 1.35) * (0.58 + 0.60 * turretVar);
    float cumulusDome = smoothstep(lclRelM - 40.0, lclRelM + 75.0, zRel)
                      * smoothstep(localCuTopRel + 120.0, localCuTopRel - 320.0, zRel);
    float lowCellMask = smoothstep(lowThresh, lowThresh + 0.22 + lowCov * 0.28, lowField);
    float lowSupport = mix(cellStrength * cumulusDome, lowCellMask * stratusBand, stratusRegime);

    // Updraft adiabatic cooling drives genuine orographic & convective condensation!
    thetaPrime += (-0.0028 * vel.z) * dt - thetaPrime * 0.012 * dt;
    float targetRh = mix(uSurfaceRhPct / 100.0, 0.34 + 0.44 * midCov, smoothstep(0.0, 4200.0, zRel));
    if (lowCov > 0.05 && zRel < lclRelM + maxTowerSpan + 350.0) {
      targetRh = max(targetRh, mix(0.72, 1.085, lowSupport));
    }
    if (midCov > 0.05 && zRel > 1800.0 && zRel < 5800.0) {
      vec2 driftMidUv = fract(uv - uWind500 * (uSimTime * invDomain));
      float midCell = 0.65 * valueNoisePeriodic2D(driftMidUv + vec2(0.31, 0.67), 4.0)
                    + 0.35 * valueNoisePeriodic2D(driftMidUv + vec2(0.12, 0.83), 8.0);
      float leeCrest = clamp(promM / 320.0, 0.0, 1.0) * clamp((uOrographicWaveGain - 0.85) * 0.55, 0.0, 0.55);
      float midStrength = smoothstep(1.0 - midCov * 0.76 - 0.05, 0.96, clamp(midCell + leeCrest * 0.30, 0.0, 1.0));
      float midCenterZ = 3350.0 + (turretVar - 0.5) * 650.0;
      float midHalfThick = 180.0 + 1150.0 * pow(midStrength, 1.15);
      float midLens = smoothstep(midCenterZ - midHalfThick - 220.0, midCenterZ - midHalfThick + 120.0, zRel)
                    * smoothstep(midCenterZ + midHalfThick + 220.0, midCenterZ + midHalfThick - 120.0, zRel);
      targetRh = max(targetRh, mix(0.68, 1.070, midStrength * midLens));
    }
    // Orographic wave crest saturation boost + lee-trough drying for crisp lenticular lenses!
    if (zRel > 180.0 && zRel < 6500.0) {
      targetRh += clamp(vel.z * 0.036, -0.10, 0.15);
    }

    float qvTarget = qsatLiq * targetRh;
    qv = mix(qv, qvTarget, clamp(0.042 * dt, 0.0, 0.24));

    // 4. Clausius-Clapeyron Saturation Adjustment (qv <-> qc + latent heat release!)
    float superSat = qv - qsatLiq;
    if (superSat > 0.0 && tempK > 238.0) {
      float cond = superSat * 0.64;
      qv -= cond;
      qc += cond;
      thetaPrime += cond * 840.0; // Latent heat fuels persistent 3D cumulus towers
    } else if (superSat < 0.0 && qc > 0.0) {
      float evap = min(qc, -superSat * 0.52);
      qc -= evap;
      qv += evap;
      thetaPrime -= evap * 540.0;
    }
    qc = clamp(qc * (1.0 - 0.004 * dt), 0.0, 0.0034);

    // 5. Koop Upper-Tropospheric Cirrus Ice Nucleation & Wind-Sheared Fallstreak Sublimation
    vec2 jetDir = normalize(uWind250 + vec2(1e-3));
    vec2 crossDir = vec2(-jetDir.y, jetDir.x);
    vec2 jetUv = fract(uv - uWind250 * (uSimTime * invDomain));
    vec2 jetProj = vec2(dot(jetUv, jetDir), dot(jetUv, crossDir));
    float ciStreak = 0.62 * valueNoisePeriodic2D(vec2(jetProj.x * 0.5, jetProj.y * 2.0), 4.0)
                   + 0.38 * valueNoisePeriodic2D(vec2(jetProj.x * 1.0, jetProj.y * 3.0) + vec2(0.37, 0.19), 8.0);
    float ciRibs = 0.5 + 0.5 * sin(jetProj.x * 6.2831853 * 5.0 + ciStreak * 2.6);
    float ciWave = clamp(ciStreak * (0.80 + 0.28 * ciRibs), 0.0, 1.0);
    float ciGenBand = smoothstep(6700.0, 8600.0, zAsl) * (1.0 - smoothstep(10850.0, 11950.0, zAsl));
    float ciMask = smoothstep(1.0 - highCov * 0.72 - 0.04, 1.0 - highCov * 0.28 + 0.16, ciWave);
    float qiTarget = highCov * ciMask * ciGenBand * 0.00048 * uCirrusIceBoost
                   * (1.0 + 0.28 * clamp(vel.z, -0.8, 1.6));

    if (zAsl >= 6400.0) {
      qi = mix(qi, qiTarget, clamp(0.052 * dt, 0.0, 0.28));
      // Bergeron glaciation from supercooled mid/high liquid
      if (tempK < 253.0 && qc > 1e-5) {
        float freeze = qc * 0.16 * clamp(dt, 0.08, 2.0);
        qc -= freeze;
        qi += freeze * 0.65;
      }
    } else {
      // Ice crystals falling below 6.4 km sublimate smoothly into dry mid-tropospheric air (mares' tail tips!)
      qi *= exp(-0.20 * dt);
    }
    qi = clamp(qi, 0.0, 0.00090);

    fragColor = vec4(clamp(thetaPrime, -8.0, 8.0), qv, qc, qi);
  }
`;

// GPU Fluid Pass 5: Pack 128x128x48 Display Atlas (1040 x 780) with 1-Texel Periodic XY Ghost Ring per Tile!
// Enables 100% artifact-free hardware-bilinear sampling in Volumetric & Shadow passes with ZERO CPU transfer!
const FLUID_PACK_ATLAS_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uVelTex;
  uniform sampler2D uThermoTex;
  uniform sampler2D uTerrainGridTex;
  out vec4 fragColor;

  void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    int tx = px.x / 130;
    int ty = px.y / 130;
    int lx = px.x - tx * 130 - 1; // -1 .. 128 (wraps periodically to 0 .. 127!)
    int ly = px.y - ty * 130 - 1;
    int iz = clamp(ty * 8 + tx, 0, 47);

    int wx = (lx + 128) & 127;
    int wy = (ly + 128) & 127;
    ivec2 stCoord = voxelToAtlas(wx, wy, iz);

    vec4 vel = texelFetch(uVelTex, stCoord, 0);
    vec4 th  = texelFetch(uThermoTex, stCoord, 0);
    float zAsl = sToAltitudeM(float(iz) / 47.0);
    float elevM = texelFetch(uTerrainGridTex, ivec2(wx, wy), 0).r;

    if (zAsl <= elevM) {
      fragColor = vec4(0.0, 0.0, 0.5, 0.0);
      return;
    }

    float tempK, pressPa, thetaAmb, qsatLiq, qsatIce;
    ambientSounding(zAsl, tempK, pressPa, thetaAmb, qsatLiq, qsatIce);

    float qcNorm = sqrt(clamp(th.z / 0.0020, 0.0, 1.0));
    float qiNorm = sqrt(clamp(th.w / 0.00052, 0.0, 1.0));
    float wTurbNorm = clamp(0.5 + 0.5 * tanh(vel.z / 4.2 + vel.w * 0.18), 0.0, 1.0);
    float rhNorm = clamp(th.y / max(1e-5, qsatLiq), 0.0, 1.0);

    fragColor = vec4(qcNorm, qiNorm, wTurbNorm, rhNorm);
  }
`;

// GPU Pass 6: 4x4 Column Telemetry Reduction Pass (reads 64 bytes every ~0.45s for live UI telemetry!)
const FLUID_TELEMETRY_FRAGMENT_GLSL3 = /* glsl */ `
  ${FLUID_COMMON_HEADER_GLSL3}
  uniform sampler2D uVelTex;
  uniform sampler2D uThermoTex;
  out vec4 fragColor;

  void main() {
    ivec2 block = ivec2(gl_FragCoord.xy); // 0..3, 0..3
    float maxW = 0.0;
    float maxQc = 0.0;
    float maxQi = 0.0;
    float covAccum = 0.0;

    // Sample 16 representative columns in this quadrant across vertical layers
    for (int dy = 0; dy < 4; dy++) {
      for (int dx = 0; dx < 4; dx++) {
        int ix = block.x * 32 + dx * 8 + 4;
        int iy = block.y * 32 + dy * 8 + 4;
        float colLow = 0.0;
        float colMid = 0.0;
        float colHigh = 0.0;
        for (int iz = 1; iz < 47; iz += 2) {
          ivec2 c = voxelToAtlas(ix, iy, iz);
          vec4 v = texelFetch(uVelTex, c, 0);
          vec4 t = texelFetch(uThermoTex, c, 0);
          maxW = max(maxW, v.z);
          maxQc = max(maxQc, t.z * 1000.0);
          maxQi = max(maxQi, t.w * 1000.0);
          if (iz < 16 && t.z > 0.00012) colLow = 1.0;
          else if (iz < 30 && t.z > 0.00009) colMid = 1.0;
          if (iz >= 28 && t.w > 0.00004) colHigh = 1.0;
        }
        // Pack low/mid/high fractional indicator based on block.y row
        if (block.y == 0) covAccum += colLow;
        else if (block.y == 1) covAccum += colMid;
        else covAccum += colHigh;
      }
    }
    fragColor = vec4(maxW, maxQc, maxQi, (covAccum / 16.0) * 100.0);
  }
`;

// ============================================================================
// Shared 380 km Curved-Earth Atmospheric & Topographic Corridor LUT Sampler (180 x 32 RGBA8)
// RGB = Far-corridor (4.6 -> 380 km) spectral transmittance; A = Far SRTM horizon [-6°, +26°]
// ============================================================================
const CORRIDOR_LUT_HELPER_GLSL3 = /* glsl */ `
  uniform sampler2D uCorridorLutTex;
  uniform float uHasCorridorHorizon;
  uniform float uUseCorridorAtmo;
  uniform float uCorridorLutSunElDeg;

  vec4 sampleCorridorLut(float azimuthDeg, float zAslM) {
    float uAz = fract(azimuthDeg / 360.0) + (0.5 / 180.0);
    float vNorm = pow(clamp(max(0.0, zAslM) / 12500.0, 0.0, 1.0), 1.0 / 1.6);
    float vAlt = vNorm * (31.0 / 32.0) + (0.5 / 32.0);
    return texture(uCorridorLutTex, vec2(uAz, vAlt));
  }

  float decodeCorridorHorizonDeg(float alphaByteNorm) {
    return -6.0 + 32.0 * alphaByteNorm;
  }

  vec3 computeCorridorAtmoModulation(vec3 lutRgb, float zAslM) {
    float dipDeg = 1.015 * sqrt(max(0.0, zAslM * 0.001));
    float locEl = clamp(uCorridorLutSunElDeg + dipDeg * 0.52, -1.8, 89.0);
    float sinEl = sin(radians(max(0.0, locEl)));
    float amRay = exp(-max(0.0, zAslM) / 8434.0) / (sinEl + 0.15 * pow(max(0.25, locEl + 3.885), -1.253));
    float amAer = exp(-max(0.0, zAslM) / 1800.0) * 0.72 / (sinEl + 0.09 * pow(max(0.25, locEl + 2.2), -1.15));
    float o3Slant = 12.2 / sqrt(1.0 + 150.0 * sinEl * sinEl);
    vec3 tauRef = vec3(0.056, 0.110, 0.220) * amRay
                + vec3(0.036, 0.045, 0.056) * amAer
                + vec3(0.026, 0.020, 0.002) * o3Slant;
    vec3 clearRef = exp(-tauRef);
    float redMod   = clamp(lutRgb.r / max(0.032, clearRef.r), 0.0, 1.30);
    float greenMod = mix(redMod, clamp(lutRgb.g / max(0.025, clearRef.g), 0.0, 1.30), smoothstep(0.02, 0.08, clearRef.g));
    float blueMod  = mix(greenMod, clamp(lutRgb.b / max(0.025, clearRef.b), 0.0, 1.30), smoothstep(0.02, 0.08, clearRef.b));
    return vec3(redMod, greenMod, blueMod);
  }
`;

// ============================================================================
// GPU Pass: DEM-Space Topographic Soft-Penumbra Shadow & 360° Viewshed Baker
// Computed ONLY when sun angle, target subject, corridor LUT, or DEM changes — turns 242 raymarch
// texture taps per Retina pixel in `TERRAIN_FRAGMENT_SHADER` into 1 O(1) fetch!
// ============================================================================
const TERRAIN_BAKE_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uHeightTex;
  uniform float uGridSize;
  uniform float uSpanM;
  uniform vec3 uSunDirENU;
  uniform vec2 uTargetEN;
  uniform float uTargetUpperZ;
  uniform float uShowViewshed;
  ${CORRIDOR_LUT_HELPER_GLSL3}

  const float BAKE_EARTH_R_EFF = 7350000.0;

  float sampleDemSmooth(vec2 uv) {
    float n = max(uGridSize, 240.0);
    vec2 tcUv = clamp(uv, 0.0, 1.0) * ((n - 1.0) / n) + vec2(0.5 / n);
    return texture(uHeightTex, tcUv).r;
  }

  void main() {
    float z = sampleDemSmooth(vUv);
    float du = 1.0 / max(uGridSize - 1.0, 239.0);
    float stepWorld = uSpanM * du;

    float zNW = sampleDemSmooth(vUv + vec2(-du,  du));
    float zN  = sampleDemSmooth(vUv + vec2(0.0,  du));
    float zNE = sampleDemSmooth(vUv + vec2( du,  du));
    float zW  = sampleDemSmooth(vUv + vec2(-du, 0.0));
    float zE  = sampleDemSmooth(vUv + vec2( du, 0.0));
    float zSW = sampleDemSmooth(vUv + vec2(-du, -du));
    float zS  = sampleDemSmooth(vUv + vec2(0.0, -du));
    float zSE = sampleDemSmooth(vUv + vec2( du, -du));

    float dzdx = ((zNE + 2.0 * zE + zSE) - (zNW + 2.0 * zW + zSW)) / (8.0 * stepWorld);
    float dzdy = ((zNW + 2.0 * zN + zNE) - (zSW + 2.0 * zS + zSE)) / (8.0 * stepWorld);
    vec3 normalENU = normalize(vec3(-dzdx, -dzdy, 1.0));
    float ndotl = max(dot(normalENU, uSunDirENU), 0.0);

    float lit = 1.0;
    float minClearance = 400.0;
    float closestOccluderDist = 0.0;
    float minPenumbraRatio = 1e5;

    float sunElDeg = degrees(asin(clamp(uSunDirENU.z, -1.0, 1.0)));

    if (sunElDeg <= -2.2 || ndotl <= 0.002) {
      lit = 0.0;
    } else {
      vec2 horizDir = normalize(uSunDirENU.xy + vec2(1e-5));
      float tanEl = uSunDirENU.z / max(length(uSunDirENU.xy), 0.005);
      float marchStepM = uSpanM / 280.0;
      vec2 uvStep = (horizDir * marchStepM) / uSpanM;
      vec2 startUv = vUv + (normalENU.xy * stepWorld * 1.85) / uSpanM;
      float zOrigin = z + 0.85 + normalENU.z * 1.25;

      for (int i = 1; i <= 165; i++) {
        float fi = float(i);
        vec2 sampleUv = startUv + uvStep * fi;
        if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0) {
          break;
        }
        float distM = fi * marchStepM;
        float rayZ = zOrigin + distM * tanEl;
        float curvDrop = (distM * distM) / (2.0 * BAKE_EARTH_R_EFF);
        float terrainZ = sampleDemSmooth(sampleUv) - curvDrop;
        float clearance = rayZ - terrainZ;
        if (distM > 45.0 && clearance < minClearance) {
          minClearance = max(-50.0, clearance);
          closestOccluderDist = distM;
        }
        float penumbraHalf = max(1.4, distM * 0.0085);
        float ratio = clearance / penumbraHalf;
        minPenumbraRatio = min(minPenumbraRatio, ratio);
        if (ratio < -1.0) {
          lit = 0.0;
          break;
        }
      }
      if (lit > 0.0 && minPenumbraRatio < 1.0) {
        lit = smoothstep(-1.0, 1.0, minPenumbraRatio);
      }

      // Couple 4.6 km -> 200 km curved-Earth SRTM mountain & sea-limb horizon shadow!
      if (lit > 0.0 && uHasCorridorHorizon > 0.5) {
        float sunAzDeg = mod(degrees(atan(uSunDirENU.x, uSunDirENU.y)) + 360.0, 360.0);
        vec4 corr = sampleCorridorLut(sunAzDeg, zOrigin);
        float farHorizonDeg = decodeCorridorHorizonDeg(corr.a);
        float farMarginDeg = sunElDeg - farHorizonDeg;
        float farDiskVis = smoothstep(-0.2665, 0.2665, farMarginDeg);
        lit *= farDiskVis;
        if (farMarginDeg < 1.1) {
          float eqClearanceM = farMarginDeg * 140.0;
          if (eqClearanceM < minClearance) {
            minClearance = max(-50.0, eqClearanceM);
            closestOccluderDist = max(closestOccluderDist, 8500.0);
          }
        }
      } else if (uHasCorridorHorizon <= 0.5 && sunElDeg <= -0.25) {
        lit *= smoothstep(-0.55, 0.15, sunElDeg);
      }
    }

    float losClear = 1.0;
    if (uShowViewshed > 0.5) {
      vec2 posEN = (vUv - 0.5) * uSpanM;
      if (length(posEN - uTargetEN) > 260.0) {
        vec2 targetUv = (uTargetEN + vec2(uSpanM * 0.5)) / uSpanM;
        float startZ = z + 1.8;
        float endZ = uTargetUpperZ + 6.0;
        for (int j = 2; j <= 94; j++) {
          float frac = float(j) / 96.0;
          vec2 sUv = mix(vUv, targetUv, frac);
          float rayZ = mix(startZ, endZ, frac);
          if (sampleDemSmooth(sUv) > rayZ + 3.0) {
            losClear = 0.0;
            break;
          }
        }
      }
    }

    fragColor = vec4(lit, minClearance, closestOccluderDist, losClear);
  }
`;

// ============================================================================
// GPU Pass: 245x1 Parallel Curved-Earth Tripod Sightline + 360° DEM Horizon Scanner
// Pixels 0..64: 65 curved-Earth sightline samples; Pixels 65..244: 180 azimuth rays (2° step)
// ============================================================================
const TRIPOD_RAYCAST_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;
  out vec4 fragColor;

  uniform sampler2D uHeightTex;
  uniform float uGridSize;
  uniform float uSpanM;
  uniform vec2 uTripodEN;
  uniform vec2 uTargetEN;
  uniform float uTargetZM;
  ${CORRIDOR_LUT_HELPER_GLSL3}

  const float R_EFF = 7328000.0; // Curved Earth with k=0.13 standard refraction

  float sampleDemEN(vec2 en, out bool inside) {
    float halfSpan = uSpanM * 0.5;
    vec2 uv = (en + vec2(halfSpan)) / max(uSpanM, 1.0);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      inside = false;
      return 0.0;
    }
    inside = true;
    float n = max(uGridSize, 240.0);
    vec2 tcUv = uv * ((n - 1.0) / n) + vec2(0.5 / n);
    return texture(uHeightTex, tcUv).r;
  }

  void main() {
    int idx = int(gl_FragCoord.x);
    bool in0 = false;
    float tripodGroundZ = sampleDemEN(uTripodEN, in0);
    float eyeZ = tripodGroundZ + 1.75;

    if (idx <= 64) {
      float t = float(idx) / 64.0;
      vec2 delta = uTargetEN - uTripodEN;
      float totalDistM = max(10.0, length(delta));
      float d = totalDistM * t;
      vec2 pEN = uTripodEN + delta * t;
      bool inside = false;
      float terrZ = sampleDemEN(pEN, inside);
      float curvDrop = (d * d) / (2.0 * R_EFF);
      float effTz = terrZ - curvDrop;
      float aimZ = uTargetZM > 1.0 ? uTargetZM : sampleDemEN(uTargetEN, inside);
      float rayZ = eyeZ + t * (aimZ - eyeZ);
      float excessM = (idx >= 3 && idx < 62) ? (effTz - rayZ) : -999.0;
      fragColor = vec4(d, effTz, rayZ, excessM);
      return;
    }

    int azIdx = idx - 65; // 0..179
    float azDeg = float(azIdx) * 2.0;
    float azRad = radians(azDeg);
    vec2 dirEN = vec2(sin(azRad), cos(azRad));

    float maxSlope = -0.05;
    float bestDistM = uSpanM * 0.5;
    float stepM = max(12.0, uSpanM / 340.0);

    for (int s = 2; s <= 220; s++) {
      float d = float(s) * stepM;
      bool inside = false;
      float tz = sampleDemEN(uTripodEN + dirEN * d, inside);
      if (!inside) break;
      float curvDrop = (d * d) / (2.0 * R_EFF);
      float slope = ((tz - curvDrop) - eyeZ) / d;
      if (slope > maxSlope) {
        maxSlope = slope;
        bestDistM = d;
      }
    }
    float maxElDeg = degrees(atan(maxSlope));
    if (uHasCorridorHorizon > 0.5) {
      float farH = decodeCorridorHorizonDeg(sampleCorridorLut(azDeg, eyeZ).a);
      if (farH > maxElDeg) {
        maxElDeg = farH;
        bestDistM = max(bestDistM, uSpanM * 0.75);
      }
    }
    maxElDeg = max(0.0, maxElDeg);
    fragColor = vec4(azDeg, maxElDeg, bestDistM, tripodGroundZ);
  }
`;

// Dual-backend fluid volume sampler (samples GPU 1040x780 padded 128x128x48 atlas OR fallback 64x64x48 3D texture)
// Seamlessly blends beyond the 18 km local Navier-Stokes domain into the 160 km GOES/NWP mesoscale cloud field!
const SAMPLE_FLUID_VOLUME_HELPER_GLSL3 = /* glsl */ `
  uniform sampler2D uCloudAtlas2D;
  uniform sampler3D uCloudVolume3D;
  uniform float uUseGpuAtlas;
  uniform sampler2D uMesoscaleCloudTex;
  uniform float uMesoscaleSpanM;
  uniform float uHasMesoscaleGrid;

  vec4 sampleFluidMacro(vec2 uvWorld, float s) {
    if (uUseGpuAtlas > 0.5) {
      vec2 wUv = fract(uvWorld);
      // Smooth vertical slice interpolation across 48 levels (linear sf avoids C2 plateau rings!)
      float sTex = clamp(s, 0.0, 1.0) * 47.0;
      int iz0 = clamp(int(floor(sTex)), 0, 47);
      int iz1 = min(47, iz0 + 1);
      float sf = sTex - float(iz0);

      // Each 130x130 tile has a 1-texel periodic border around its 128x128 interior!
      vec2 localPx = wUv * 128.0 + 1.0;
      vec2 tile0 = vec2(float(iz0 & 7), float(iz0 >> 3)) * 130.0;
      vec2 tile1 = vec2(float(iz1 & 7), float(iz1 >> 3)) * 130.0;
      vec2 invAtlas = vec2(1.0 / 1040.0, 1.0 / 780.0);
      vec4 c0 = texture(uCloudAtlas2D, (tile0 + localPx) * invAtlas);
      vec4 c1 = texture(uCloudAtlas2D, (tile1 + localPx) * invAtlas);
      return mix(c0, c1, sf);
    }

    float sSmooth = clamp(s * (47.0 / 48.0) + (0.5 / 48.0), 0.5 / 48.0, 47.5 / 48.0);
    vec2 fTex = fract(uvWorld) * 64.0 - 0.5;
    vec2 fIdx = floor(fTex);
    vec2 fFrac = fTex - fIdx;
    vec2 fQuintic = fFrac * fFrac * (3.0 - 2.0 * fFrac);
    vec2 fluidUvSmooth = fract((fIdx + fQuintic + 0.5) * (1.0 / 64.0));
    return texture(uCloudVolume3D, vec3(fluidUvSmooth, sSmooth));
  }

  vec4 sampleFluidFieldNonRepeating(vec2 posWorldXZ, float domainXyM, float s) {
    vec2 en = vec2(posWorldXZ.x, -posWorldXZ.y);
    vec2 uvPrimary = en / max(1000.0, domainXyM) + 0.5;
    vec4 nearFl = sampleFluidMacro(uvPrimary, s);

    // Keep central landmark domain 100% faithful to local DEM fluid physics;
    // smoothly blend an incommensurately rotated (36.87 deg) 2.37x macro octave + 160 km mesoscale grid in the far vista!
    float radialDist = length(en);
    float farBlend = smoothstep(domainXyM * 0.43, domainXyM * 0.96, radialDist);
    if (farBlend <= 0.002) {
      return nearFl;
    }

    const mat2 rotIncomm = mat2(0.80, -0.60, 0.60, 0.80);
    vec2 uvFar = (rotIncomm * en) / (max(1000.0, domainXyM) * 2.37) + vec2(0.317, 0.713);
    vec4 farFl = sampleFluidMacro(uvFar, s);

    vec2 combinedCloud = mix(
      nearFl.rg,
      clamp(nearFl.rg * (0.52 + 0.92 * farFl.rg) + 0.22 * farFl.rg * farFl.rg, 0.0, 1.0),
      farBlend * 0.78
    );
    float outW = mix(nearFl.b, farFl.b, farBlend * 0.45);
    float outRh = mix(nearFl.a, farFl.a, farBlend * 0.55);

    if (uHasMesoscaleGrid > 0.5) {
      float mesoBlend = smoothstep(domainXyM * 0.48, domainXyM * 1.45, radialDist);
      if (mesoBlend > 0.005) {
        vec2 mesoUv = clamp(en / max(20000.0, uMesoscaleSpanM) + 0.5, 0.02, 0.98);
        vec4 meso = texture(uMesoscaleCloudTex, mesoUv); // R=low, G=mid, B=high, A=fog/inv
        float lowVert = (1.0 - smoothstep(0.32, 0.52, s)) * smoothstep(0.01, 0.08, s);
        float midVert = smoothstep(0.28, 0.44, s) * (1.0 - smoothstep(0.62, 0.78, s));
        float highVert = smoothstep(0.60, 0.76, s) * (1.0 - smoothstep(0.93, 1.0, s));
        float cellMod = clamp(0.45 + 0.95 * max(farFl.r, nearFl.r * 0.75), 0.35, 1.30);
        float iceMod  = clamp(0.45 + 0.95 * max(farFl.g, nearFl.g * 0.75), 0.35, 1.30);
        float mesoLiq = sqrt(clamp((max(meso.r, meso.a * 0.9) * lowVert + meso.g * midVert) * cellMod, 0.0, 1.0));
        float mesoIce = sqrt(clamp(meso.b * highVert * iceMod, 0.0, 1.0));
        combinedCloud.x = mix(combinedCloud.x, mesoLiq, mesoBlend * 0.85);
        combinedCloud.y = mix(combinedCloud.y, mesoIce, mesoBlend * 0.85);
        outRh = mix(outRh, clamp(0.25 + 0.70 * max(meso.r, meso.a), 0.0, 1.0), mesoBlend * 0.65);
      }
    }

    return vec4(combinedCloud, outW, outRh);
  }
`;

// Pass A: 256x256 GPU Cloud Shadow Transmittance LUT (shadowMapRT)
// Starts each shadow ray at the ACTUAL terrain surface elevation (uHeightTex) along true low-angle solar rays!
const CLOUD_SHADOW_MAP_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;
  precision highp sampler3D;

  in vec2 vUv;
  out vec4 fragColor;

  ${SAMPLE_FLUID_VOLUME_HELPER_GLSL3}
  uniform sampler2D uHeightTex;
  uniform float uHasHeightTex;
  uniform float uSpanM;
  uniform sampler3D uDetailNoise3D;
  uniform vec3 uSunDirWorld;
  uniform float uDomainXyM;
  uniform float uZBaseM;
  uniform float uZTopM;
  uniform float uMinElevM;
  uniform float uLclAltM;
  uniform float uInversionBaseM;
  uniform float uSimTime;
  uniform float uCloudDensityGain;
  uniform float uCirrusDensityGain;
  uniform vec2 uWindSfc;
  uniform vec2 uWind500;
  uniform vec2 uWindAdvectSfc;
  uniform vec2 uWindAdvectJet;

  float altitudeToS(float zAsl) {
    float span = max(1000.0, uZTopM - uZBaseM);
    float zeta = clamp((zAsl - uZBaseM) / span, 0.0, 1.0);
    return clamp((-0.26 + sqrt(0.0676 + 2.96 * zeta)) / 1.48, 0.0, 1.0);
  }

  float sampleShadowTerrainLocalY(vec2 worldXZ) {
    if (uHasHeightTex < 0.5) return 0.0;
    float halfSpan = uSpanM * 0.5;
    vec2 demUv = vec2((worldXZ.x + halfSpan) / max(uSpanM, 1.0), (-worldXZ.y + halfSpan) / max(uSpanM, 1.0));
    if (demUv.x >= 0.0 && demUv.x <= 1.0 && demUv.y >= 0.0 && demUv.y <= 1.0) {
      return max(0.0, texture(uHeightTex, demUv).r - uMinElevM);
    }
    return 0.0;
  }

  void main() {
    if (uSunDirWorld.y <= -0.03) {
      fragColor = vec4(0.07, 1.0, 0.0, 1.0);
      return;
    }
    float wx = (vUv.x - 0.5) * uDomainXyM;
    float wz = (0.5 - vUv.y) * uDomainXyM;
    float groundY = sampleShadowTerrainLocalY(vec2(wx, wz));
    vec3 origin = vec3(wx, groundY + 4.0, wz);
    vec3 lightDir = normalize(vec3(uSunDirWorld.x, max(0.016, uSunDirWorld.y), uSunDirWorld.z));
    float stepM = mix(440.0, 820.0, smoothstep(0.24, 0.02, lightDir.y));
    float dsKm = stepM * 0.001;

    bool convectiveBase = (uLclAltM > uZBaseM + 260.0) && (uInversionBaseM > 680.0);

    float tau = 0.0;
    const int STEPS = 20;
    for (int i = 0; i < STEPS; i++) {
      float t = (float(i) + 0.45) * stepM;
      vec3 p = origin + lightDir * t;
      float zAsl = p.y + uMinElevM;
      if (zAsl > uZTopM) break;
      float s = altitudeToS(zAsl);
      vec4 fl = sampleFluidFieldNonRepeating(p.xz, uDomainXyM, s);
      float baseGate = convectiveBase ? smoothstep(uLclAltM - 95.0, uLclAltM + 45.0, zAsl) : 1.0;
      float qc = fl.r * fl.r * uCloudDensityGain * baseGate;
      float qi = fl.g * fl.g * uCirrusDensityGain;
      if (qc + qi > 0.002) {
        vec2 wLoc = mix(uWindSfc, uWind500, smoothstep(700.0, 5200.0, zAsl));
        vec3 pAdv = p - vec3(wLoc.x, 0.0, -wLoc.y) * uSimTime;
        vec4 dMeso = texture(uDetailNoise3D, pAdv * (1.0 / 1380.0));
        float sculpted = max(0.0, qc * (0.55 + 0.90 * dMeso.r) - 0.16 * (1.0 - dMeso.g) * (1.0 - qc * 0.6));
        float ext = sculpted * 2.85 + qi * 0.48;
        tau += ext * dsKm;
      }
    }
    float trans = clamp(exp(-tau), 0.07, 1.0);
    fragColor = vec4(trans, clamp(tau * 0.25, 0.0, 1.0), 0.0, 1.0);
  }
`;

// Pass B: Full-Atmosphere (0 m .. 12,200 m ASL) Sculptural 3D Volumetric Cloud, Ethereal Gossamer Veil,
// 1m Kolmogorov Micro-Turbulence, Sheared 3D Cirrus Uncinus & Crepuscular God-Ray Raymarcher!
const VOLUMETRIC_RAYMARCH_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;
  precision highp sampler3D;

  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uSceneDepthTex;
  uniform sampler2D uHeightTex;
  uniform float uHasHeightTex;
  ${SAMPLE_FLUID_VOLUME_HELPER_GLSL3}
  ${CORRIDOR_LUT_HELPER_GLSL3}
  uniform sampler3D uDetailNoise3D;

  uniform mat4 uInvViewProj;
  uniform mat4 uViewProj;
  uniform vec3 uCameraPos;
  uniform vec2 uResolution;
  uniform float uFrameIndex;
  uniform float uSimTime;

  uniform float uSpanM;
  uniform float uMinElevM;
  uniform float uMaxLocalTerrainY;
  uniform float uDomainXyM;
  uniform float uZBaseM;
  uniform float uZTopM;
  uniform float uLclAltM;
  uniform float uInversionBaseM;
  uniform float uWaveRegime;

  uniform vec3 uSunDirWorld;
  uniform float uSunElDeg;
  uniform vec3 uMoonDirWorld;
  uniform float uMoonElDeg;
  uniform float uMoonIllum;

  uniform vec2 uWindSfc;
  uniform vec2 uWind500;
  uniform vec2 uWind250;

  uniform float uCloudDensityGain;
  uniform float uCirrusDensityGain;
  uniform float uMicroDetailGain;
  uniform float uGodRayGain;
  uniform float uQualitySteps;

  const float PI = 3.14159265359;
  const float EARTH_R_EFF = 7350000.0;

  float isotropicPixelHash(vec2 pix) {
    vec3 p3 = fract(vec3(pix.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  float hgPhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1e-3, pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
    return (1.0 - g2) / (4.0 * PI * denom);
  }

  // 3-lobe Mie droplet phase: intense silver-lining diffraction peak + broad forward corona
  // + opposition glory backscatter + subtle 39-deg primary fogbow!
  float liquidCloudPhase(float cosTheta) {
    float silverPeak = hgPhase(cosTheta, 0.86);
    float coronaGlow = hgPhase(cosTheta, 0.46);
    float gloryBack  = hgPhase(cosTheta, -0.36);
    float angleDeg   = degrees(acos(clamp(cosTheta, -1.0, 1.0)));
    float fogbow     = 0.036 * exp(-pow((angleDeg - 141.0) / 5.5, 2.0));
    return 0.42 * silverPeak + 0.38 * coronaGlow + 0.20 * gloryBack + fogbow;
  }

  // Hexagonal ice-crystal phase with crisp forward scattering + 22-deg halo + sundog glint
  float cirrusIcePhase(float cosTheta) {
    float forwardPeak = hgPhase(cosTheta, 0.88);
    float broadIce    = hgPhase(cosTheta, 0.42);
    float backGlint   = hgPhase(cosTheta, -0.40);
    float angleDeg    = degrees(acos(clamp(cosTheta, -1.0, 1.0)));
    float halo22      = 0.115 * exp(-pow((angleDeg - 22.0) / 2.8, 2.0));
    return 0.48 * forwardPeak + 0.34 * broadIce + 0.18 * backGlint + halo22;
  }

  float altitudeToS(float zAsl) {
    float span = max(1000.0, uZTopM - uZBaseM);
    float zeta = clamp((zAsl - uZBaseM) / span, 0.0, 1.0);
    return clamp((-0.26 + sqrt(0.0676 + 2.96 * zeta)) / 1.48, 0.0, 1.0);
  }

  float sampleTerrainLocalY(vec2 worldXZ, out float insidePatch) {
    float halfSpan = uSpanM * 0.5;
    vec2 demUv = vec2((worldXZ.x + halfSpan) / uSpanM, (-worldXZ.y + halfSpan) / uSpanM);
    if (uHasHeightTex > 0.5 && demUv.x >= 0.0 && demUv.x <= 1.0 && demUv.y >= 0.0 && demUv.y <= 1.0) {
      insidePatch = 1.0;
      return texture(uHeightTex, demUv).r - uMinElevM;
    }
    insidePatch = 0.0;
    return 0.0;
  }

  // detailLevel:
  //   0 = macro + non-repeating synoptic field (long-range shadow / coarse sky occlusion)
  //   1 = fast single-octave meso billow self-shadow (65m directional light tap)
  //   2 = full Kelvin-Helmholtz curls + 1.0m Kolmogorov gossamer wisps & virga streaks
  // stepFootprintM band-limits 3D mipmap LOD via Nyquist theorem so distant clouds stay creamy & grain-free!
  vec4 sampleAtmosphereExtinction(vec3 posWorld, float zAsl, float distToCam, float stepFootprintM, int detailLevel) {
    if (zAsl < uZBaseM - 40.0 || zAsl > uZTopM + 120.0) {
      return vec4(0.0);
    }

    float groundClear = 1.0;
    if (posWorld.y <= uMaxLocalTerrainY + 18.0) {
      float inDem = 0.0;
      float terrY = sampleTerrainLocalY(posWorld.xz, inDem);
      float agl = posWorld.y - terrY;
      if (inDem > 0.5 && agl < -5.0) {
        return vec4(0.0);
      }
      groundClear = inDem > 0.5 ? smoothstep(-5.0, 16.0, agl) : 1.0;
    }

    float s = altitudeToS(zAsl);
    vec4 fluid = sampleFluidFieldNonRepeating(posWorld.xz, uDomainXyM, s);

    float farHorizonFade = 1.0 - smoothstep(78000.0, 118000.0, distToCam);

    float qcMacro = fluid.r * fluid.r * uCloudDensityGain * groundClear * farHorizonFade;
    float qiMacro = fluid.g * fluid.g * uCirrusDensityGain * farHorizonFade;
    float wTurb = (fluid.b - 0.5) * 2.0;
    float rhHaze = fluid.a * groundClear * farHorizonFade;

    float sigmaLiq = 0.0;
    float gossamerFrac = 0.0;
    float nyquistM = max(stepFootprintM * 0.52, distToCam * 0.0015);

    if (qcMacro > 0.0006) {
      // Architectural flat condensation floor at LCL for convective cumulus regimes,
      // while preserving valley-hugging coastal marine stratus decks when inversion is low!
      bool convectiveBase = (uLclAltM > uZBaseM + 260.0) && (uInversionBaseM > 680.0);
      float flatBaseGate = convectiveBase
        ? smoothstep(uLclAltM - 75.0, uLclAltM + 42.0, zAsl)
        : 1.0;
      float heightAboveBase = max(0.0, zAsl - (convectiveBase ? uLclAltM : (uZBaseM + 120.0)));
      float hTower = clamp(heightAboveBase / 1650.0, 0.0, 1.0);

      if (detailLevel == 0) {
        sigmaLiq = qcMacro * flatBaseGate * 0.048;
      } else if (detailLevel == 1) {
        if (flatBaseGate > 0.002) {
          float midBlend = smoothstep(700.0, 5200.0, zAsl);
          vec2 windLocal = mix(uWindSfc, uWind500, midBlend);
          vec3 pAdv = posWorld - vec3(windLocal.x, 0.16 * wTurb, -windLocal.y) * uSimTime;
          vec4 dMeso = textureLod(uDetailNoise3D, pAdv * (1.0 / 3400.0), 1.5);
          float qcBase = qcMacro * flatBaseGate;
          float sculpt = max(0.0, qcBase * (0.58 + 0.94 * dMeso.r) - 0.20 * (1.0 - dMeso.g) * (1.0 - clamp(qcBase * 0.65, 0.0, 0.80)));
          sigmaLiq = sculpt * 0.064;
        }
      } else if (flatBaseGate > 0.002) {
        // 1:1 physical Lagrangian lockstep with the MacCormack fluid solver
        float midBlend = smoothstep(700.0, 5200.0, zAsl);
        vec2 windLocal = mix(uWindSfc, uWind500, midBlend);
        vec3 pAdv = posWorld - vec3(windLocal.x, 0.16 * wTurb, -windLocal.y) * uSimTime;

        // Regime morphing: sleek laminar airfoil lenses in high-wave regimes vs billowing cauliflower cumulus
        float waveLaminar = smoothstep(1.42, 2.05, uWaveRegime) * smoothstep(1400.0, 2600.0, zAsl);

        // True meteorological cumulus dome wavelengths (3400m -> 850m/425m/212m lobes; 1020m -> 255m/127m florets)
        // Band-limited with hardware trilinear 3D mipmaps (textureLod) to eliminate ray-step stippling!
        float mesoLambda = mix(3400.0, 4200.0, waveLaminar);
        vec3 mesoScale = mix(vec3(1.0 / 3400.0), vec3(1.0 / 4400.0, 1.0 / 950.0, 1.0 / 3600.0), waveLaminar);
        float mipMeso = clamp(log2(max(1.0, nyquistM / (mesoLambda / 64.0))), 0.0, 3.5);
        vec4 dMeso = textureLod(uDetailNoise3D, pAdv * mesoScale, mipMeso);

        float mipTurret = clamp(log2(max(1.0, nyquistM / (1020.0 / 64.0))), 0.0, 3.8);
        vec4 dTurret = textureLod(uDetailNoise3D, pAdv * (1.0 / 1020.0) + vec3(dMeso.b * 0.10, 0.23, dMeso.r * 0.08), mipTurret);

        // Smooth 3D Cauliflower Dome Displacement using ONLY broad macro lobes (zero high-freq stipple!)
        float lobeAmp = (1.0 - waveLaminar * 0.88) * smoothstep(0.0, 0.26, hTower);
        vec2 dispXZ = (vec2(dMeso.r, dMeso.g) - 0.5) * 420.0 * (0.40 + 0.60 * lobeAmp);
        float dispZ = (dMeso.r - 0.48) * 320.0 * lobeAmp;
        float sWarped = altitudeToS(zAsl - dispZ);
        vec4 warpedFl = sampleFluidFieldNonRepeating(posWorld.xz - dispXZ, uDomainXyM, sWarped);
        float qcWarped = warpedFl.r * warpedFl.r * uCloudDensityGain * groundClear * farHorizonFade * flatBaseGate;

        float updraftInflate = clamp(0.5 + 0.5 * wTurb, 0.0, 1.0);
        float domeShape = mix(0.64 * dMeso.r + 0.36 * dTurret.r, 0.74 * dMeso.r + 0.26 * dMeso.b, waveLaminar);

        // Creamy voluptuous 3D cauliflower billows & clefts vs sleek stacked lenticular airfoil laminae
        float billowWeight = clamp(95.0 / max(nyquistM, 18.0), 0.30, 1.0);
        float cauliflowerClefts = mix(1.0 - dMeso.g, (1.0 - dTurret.g) * 0.62 + (1.0 - dMeso.g) * 0.38, billowWeight);
        float airfoilStrata = 0.5 + 0.5 * sin(zAsl * 0.020 + dMeso.b * 2.0);
        float lenticularTrim = (1.0 - airfoilStrata) * 0.46 + (1.0 - dMeso.r) * 0.14;
        float erode = mix(cauliflowerClefts, lenticularTrim, waveLaminar);

        // 1. Voluptuous sculptural 3D cumulus core
        float rawCore = qcWarped * (0.58 + 0.98 * domeShape)
                      - (0.28 - 0.09 * updraftInflate) * erode * (1.0 - clamp(qcWarped * 0.68, 0.0, 0.82));
        float coreSculpt = max(0.0, rawCore);
        coreSculpt = coreSculpt * smoothstep(0.0, 0.042, coreSculpt) * 1.22;

        // 2. Ethereal gossamer perimeter veil, wind-sheared crests & trailing virga wisps!
        float qcHalo = max(qcMacro * flatBaseGate, qcWarped);
        float filamentWisp = smoothstep(0.38, 0.88, mix(dTurret.b, dMeso.b, 0.45));
        float rimZone = smoothstep(0.002, 0.095, qcHalo) * (1.0 - smoothstep(0.08, 0.36, coreSculpt));
        float virgaZone = (1.0 - smoothstep(0.0, 0.35, hTower)) * smoothstep(0.46, 0.88, dTurret.b) * (1.0 - waveLaminar * 0.75);
        float gossamer = qcHalo * (0.42 * rimZone * filamentWisp + 0.18 * virgaZone * dMeso.b);

        if ((coreSculpt + gossamer) > 0.0004 && distToCam < 4500.0) {
          // Near-field 15m -> 1m Kelvin-Helmholtz curls & Kolmogorov gossamer wisps (strictly Nyquist-gated!)
          float lodNear = exp(-distToCam / 1800.0) * clamp(28.0 / max(nyquistM, 4.0), 0.0, 1.0) * uMicroDetailGain * (1.0 - waveLaminar * 0.6);
          if (lodNear > 0.02) {
            float mipFine = clamp(log2(max(1.0, nyquistM / (192.0 / 64.0))), 0.0, 3.5);
            vec4 dFine = textureLod(uDetailNoise3D, pAdv * (1.0 / 192.0) + vec3(dTurret.b * 0.10), mipFine);
            coreSculpt = max(0.0, coreSculpt - (1.0 - dFine.r) * 0.12 * lodNear * (1.0 - clamp(coreSculpt * 0.65, 0.0, 0.80)));
            gossamer += qcHalo * smoothstep(0.50, 0.90, dFine.a) * 0.10 * lodNear * rimZone;
          }
        }

        float sigmaCore = coreSculpt * 0.068;
        float sigmaVeil = gossamer * 0.014;
        sigmaLiq = sigmaCore + sigmaVeil;
        gossamerFrac = sigmaVeil / max(1.0e-6, sigmaLiq);
      }
    }

    // Upper-Tropospheric 3D Cirrus Uncinus, Floccus Generating Heads & Vertebratus Wave Ribs (6.35 - 12.1 km)
    float sigmaIce = 0.0;
    if (qiMacro > 0.016 && zAsl >= 6300.0) {
      if (detailLevel < 2) {
        sigmaIce = qiMacro * smoothstep(0.06, 0.48, qiMacro) * 0.0038;
      } else {
        float fallDepthKm = clamp((10850.0 - zAsl) * 0.001, -1.1, 4.3);
        vec2 shearVec = (uWind250 - uWind500) * 52.0;
        vec2 hookOffset = shearVec * (fallDepthKm * 0.52 + 0.28 * fallDepthKm * fallDepthKm);

        vec2 jetDir = normalize(uWind250 + vec2(0.001));
        vec2 crossDir = vec2(-jetDir.y, jetDir.x);

        vec2 pHoriz = posWorld.xz + vec2(hookOffset.x, -hookOffset.y)
          - vec2(uWind250.x, -uWind250.y) * uSimTime;

        float alongJet = dot(vec2(pHoriz.x, -pHoriz.y), jetDir);
        float acrossJet = dot(vec2(pHoriz.x, -pHoriz.y), crossDir);
        vec3 ciCoord = vec3(alongJet / 5200.0, acrossJet / 920.0, zAsl / 1250.0);

        float mipCi = clamp(log2(max(1.0, nyquistM / 22.0)), 0.0, 3.0);
        vec4 ciTex = textureLod(uDetailNoise3D, ciCoord, mipCi);
        vec4 ciFine = textureLod(uDetailNoise3D, ciCoord * 3.2 + vec3(ciTex.b * 0.16, ciTex.r * 0.12, 0.0), min(3.5, mipCi + 1.0));

        float headWeight = smoothstep(9500.0, 10950.0, zAsl);
        float vertRib = 0.72 + 0.28 * sin(alongJet * (1.0 / 390.0) + ciTex.r * 3.2);
        float silkStrands = ciTex.b * (0.42 + 0.78 * ciFine.b) * vertRib;
        float tuftHeads = ciTex.g * (0.55 + 0.55 * ciFine.r);
        float filaments = mix(silkStrands, tuftHeads, 0.48 * headWeight);

        float coreStrand = smoothstep(0.32, 0.84, filaments);
        float haloStrand = smoothstep(0.14, 0.52, filaments) * 0.28;
        float macroGate = smoothstep(0.04, 0.50, qiMacro);
        float iceSculpt = qiMacro * macroGate * (0.14 + 1.48 * coreStrand + haloStrand);
        sigmaIce = iceSculpt * 0.0048;
        gossamerFrac = max(gossamerFrac, 0.68 * clamp(sigmaIce / max(1.0e-6, sigmaLiq + sigmaIce), 0.0, 1.0));
      }
    }

    float lowAirWeight = exp(-max(0.0, zAsl - uZBaseM) / 2250.0);
    float sigmaHaze = (0.0000042 + 0.000074 * rhHaze * lowAirWeight) * uGodRayGain * farHorizonFade;

    return vec4(sigmaLiq, sigmaIce, sigmaHaze, gossamerFrac);
  }

  float sampleTerrainSunVisibility(vec3 posWorld, vec3 lightDir) {
    if (uHasHeightTex < 0.5 || lightDir.y <= -0.04) return 1.0;
    if (posWorld.y > uMaxLocalTerrainY && lightDir.y >= 0.0) return 1.0;
    float halfSpan = uSpanM * 0.5;
    if (abs(posWorld.x) > halfSpan * 1.35 || abs(posWorld.z) > halfSpan * 1.35) return 1.0;

    const float dists[3] = float[3](210.0, 820.0, 2450.0);
    float vis = 1.0;
    for (int i = 0; i < 3; i++) {
      vec3 sp = posWorld + lightDir * dists[i];
      if (lightDir.y > 0.0 && sp.y > uMaxLocalTerrainY) break;
      float inside = 0.0;
      float ty = sampleTerrainLocalY(sp.xz, inside);
      if (inside > 0.5) {
        float margin = sp.y - ty;
        vis = min(vis, smoothstep(-28.0, 24.0, margin));
        if (vis < 0.02) return 0.0;
      }
    }
    return vis;
  }

  // 3-tap multi-scale directional light march:
  // Near tap (65m) resolves sculptural 3D cauliflower billows (detailLevel = 1)
  // while 240m & 900m macro taps capture deep cloud-mass optical depth with minimal texture load!
  float marchLightOpticalDepth(vec3 posWorld, float zAsl, vec3 lightDir) {
    const float stepsM[3]   = float[3](65.0, 240.0, 900.0);
    const float dWeights[3] = float[3](85.0, 230.0, 840.0);
    float tau = 0.0;
    for (int i = 0; i < 3; i++) {
      vec3 lp = posWorld + lightDir * stepsM[i];
      float lz = zAsl + lightDir.y * stepsM[i];
      int dLvl = (i == 0) ? 1 : 0;
      vec4 sig = sampleAtmosphereExtinction(lp, lz, 6000.0, stepsM[i] * 1.35, dLvl);
      tau += (sig.x + sig.y * 0.85) * dWeights[i];
    }
    return tau;
  }

  // Fast single-tap vertical sky-cone ambient occlusion: carves cool cerulean/lavender depth into 3D cloud canyons & bases!
  float sampleSkyConeOcclusion(vec3 posWorld, float zAsl) {
    float sUp = altitudeToS(zAsl + 165.0);
    vec4 flUp = sampleFluidFieldNonRepeating(posWorld.xz, uDomainXyM, sUp);
    float tauSky = (flUp.r * flUp.r * uCloudDensityGain * 0.052 + flUp.g * flUp.g * 0.005) * 240.0;
    return clamp(exp(-tauSky * 0.95), 0.24, 1.0);
  }

  vec3 computeDirectSolarIrradiance(float zAsl, out float localSunElDeg) {
    float dipDeg = 1.015 * sqrt(max(0.0, zAsl * 0.001));
    localSunElDeg = uSunElDeg + dipDeg;

    if (localSunElDeg <= -2.8) {
      return vec3(0.0);
    }

    float pressureRatio = exp(-max(0.0, zAsl) / 8200.0);
    float sinEl = sin(radians(max(localSunElDeg, 0.25)));
    float airMass = clamp(pressureRatio / (sinEl + 0.14 * pow(max(0.0, localSunElDeg + 3.4), -0.72)), 0.15, 11.5);

    // Balanced Rayleigh + Chappuis ozone attenuation: luminous warm champagne-gold / apricot-ivory
    // matching sunlit granite at sunset, transitioning to delicate coral-rose only at final horizon!
    vec3 betaExt = vec3(0.024, 0.066, 0.134);
    vec3 sunCol = exp(-betaExt * airMass) * 3.35;

    float roseHorizon = smoothstep(3.2, -1.2, localSunElDeg);
    sunCol *= mix(vec3(1.0), vec3(1.16, 0.92, 0.84), roseHorizon);

    float limbVis = smoothstep(-2.1, 0.45, localSunElDeg);
    if (uHasCorridorHorizon > 0.5 || uUseCorridorAtmo > 0.5) {
      float sunAzDeg = mod(degrees(atan(uSunDirWorld.x, -uSunDirWorld.z)) + 360.0, 360.0);
      vec4 corr = sampleCorridorLut(sunAzDeg, zAsl);
      if (uHasCorridorHorizon > 0.5) {
        float farHorizonDeg = decodeCorridorHorizonDeg(corr.a);
        limbVis = smoothstep(farHorizonDeg - 0.28, farHorizonDeg + 0.28, uSunElDeg);
      }
      if (uUseCorridorAtmo > 0.5) {
        sunCol *= computeCorridorAtmoModulation(corr.rgb, zAsl);
      }
    }
    return sunCol * limbVis;
  }

  void computeHemisphereIrradiance(float zAsl, out vec3 skyZenithIrr, out vec3 groundBounceIrr) {
    float dayT = smoothstep(-3.5, 12.0, uSunElDeg);
    float goldT = smoothstep(-6.0, -0.2, uSunElDeg) * (1.0 - smoothstep(3.5, 13.0, uSunElDeg));
    float nightT = 1.0 - smoothstep(-9.0, -2.2, uSunElDeg);

    vec3 daySky = vec3(0.38, 0.52, 0.74) * (0.88 + 0.22 * clamp(zAsl / 10000.0, 0.0, 1.0));
    vec3 duskSky = vec3(0.28, 0.32, 0.52) + vec3(0.16, 0.08, 0.14) * clamp(zAsl / 9500.0, 0.0, 1.15);
    vec3 nightSky = vec3(0.012, 0.020, 0.042) + vec3(0.026, 0.040, 0.070) * uMoonIllum * smoothstep(-2.0, 14.0, uMoonElDeg);

    skyZenithIrr = mix(duskSky, daySky, dayT);
    skyZenithIrr = mix(skyZenithIrr, vec3(0.32, 0.36, 0.56), goldT * 0.65);
    skyZenithIrr = mix(skyZenithIrr, nightSky, nightT);

    // Subtle warm sunlit granite / coastal alpenglow bounce onto lower cloud bases
    vec3 dayBounce = vec3(0.22, 0.21, 0.20);
    vec3 goldBounce = vec3(0.34, 0.24, 0.18);
    vec3 duskBounce = vec3(0.12, 0.09, 0.12);
    groundBounceIrr = mix(duskBounce, dayBounce, dayT);
    groundBounceIrr = mix(groundBounceIrr, goldBounce, goldT * 0.75 + smoothstep(8.0, 0.5, uSunElDeg) * dayT * 0.35);
    groundBounceIrr = mix(groundBounceIrr, nightSky * 0.35, nightT);
  }

  void main() {
    float rawDepth = texture(uSceneDepthTex, vUv).r;
    vec4 ndcFar = vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
    vec4 wFar = uInvViewProj * ndcFar;
    wFar /= wFar.w;

    vec3 rayOrigin = uCameraPos;
    vec3 rayDir = normalize(wFar.xyz - rayOrigin);

    bool hitTerrain = rawDepth < 0.99998;
    float maxDist = 116000.0;
    if (hitTerrain) {
      vec4 ndcHit = vec4(vUv * 2.0 - 1.0, rawDepth * 2.0 - 1.0, 1.0);
      vec4 wHit = uInvViewProj * ndcHit;
      wHit /= wHit.w;
      maxDist = length(wHit.xyz - rayOrigin);
    } else {
      float camZAsl = rayOrigin.y + uMinElevM;
      if (rayDir.y > 0.02 && camZAsl < uZTopM) {
        float distToTop = (uZTopM + 350.0 - camZAsl) / rayDir.y;
        maxDist = min(maxDist, distToTop * 1.12);
      } else if (rayDir.y < -0.12 && camZAsl < uZBaseM) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
      }
    }

    maxDist = clamp(maxDist, 1.0, 118000.0);

    bool useLunarLight = (uSunElDeg < -6.2) && (uMoonElDeg > 0.2) && (uMoonIllum > 0.05);
    vec3 primaryLightDir = useLunarLight ? normalize(uMoonDirWorld) : normalize(uSunDirWorld);
    float cosTheta = clamp(dot(rayDir, primaryLightDir), -1.0, 1.0);

    float phaseLiq = liquidCloudPhase(cosTheta);
    float phaseIce = cirrusIcePhase(cosTheta);
    float phaseHaze = mix(hgPhase(cosTheta, 0.68), 1.0 / (4.0 * PI), 0.35);

    // Golden-ratio temporal Halton sequence combined with isotropic 2D hash for 8x temporal supersampling!
    float isoHash = fract(isotropicPixelHash(gl_FragCoord.xy) + fract(uFrameIndex * 0.61803398875));

    int stepCount = int(clamp(uQualitySteps, 32.0, 108.0));
    float gamma = maxDist > 16000.0 ? 2.28 : 1.35;

    vec3 accumLight = vec3(0.0);
    float transmittance = 1.0;

    float prevT = 0.0;
    bool hasPrevMacro = false;
    vec4 prevMacro = vec4(0.0);
    int refinedShellSteps = 0;

    for (int i = 0; i < 108; i++) {
      if (i >= stepCount) break;
      float u1 = float(i + 1) / float(stepCount);
      float tStart = prevT;
      float tEnd = maxDist * pow(u1, gamma);
      prevT = tEnd;

      vec3 posStart = rayOrigin + rayDir * tStart;
      vec3 posEnd = rayOrigin + rayDir * tEnd;
      float hDist2A = dot(posStart.xz - rayOrigin.xz, posStart.xz - rayOrigin.xz);
      float hDist2B = dot(posEnd.xz - rayOrigin.xz, posEnd.xz - rayOrigin.xz);
      float zA = posStart.y + uMinElevM - hDist2A / (2.0 * EARTH_R_EFF);
      float zB = posEnd.y + uMinElevM - hDist2B / (2.0 * EARTH_R_EFF);

      if (min(zA, zB) > uZTopM + 160.0) {
        hasPrevMacro = false;
        if (rayDir.y > 0.05) break;
        continue;
      }
      if (max(zA, zB) < uZBaseM - 80.0) {
        hasPrevMacro = false;
        continue;
      }

      float coarseSpan = max(0.25, tEnd - tStart);
      vec4 macroA;
      if (hasPrevMacro) {
        macroA = prevMacro;
      } else {
        float tProbeA = mix(tStart, tEnd, 0.25);
        vec3 posProbeA = rayOrigin + rayDir * tProbeA;
        float hProbe2 = dot(posProbeA.xz - rayOrigin.xz, posProbeA.xz - rayOrigin.xz);
        float zProbeA = posProbeA.y + uMinElevM - hProbe2 / (2.0 * EARTH_R_EFF);
        macroA = sampleAtmosphereExtinction(posProbeA, zProbeA, tProbeA, coarseSpan, 0);
      }

      vec4 macroB = sampleAtmosphereExtinction(posEnd, zB, tEnd, coarseSpan, 0);
      prevMacro = macroB;
      hasPrevMacro = true;

      float liqShell = max(macroA.x, macroB.x);
      float iceShell = max(macroA.y, macroB.y);
      float shellDensity = liqShell + iceShell;
      float maxHaze = max(macroA.z, macroB.z);

      if (shellDensity <= 1.0e-5 && maxHaze <= 1.0e-7) {
        continue;
      }

      int subCount = 1;
      if (liqShell > 0.00014 && transmittance > 0.06 && tEnd < 24000.0 && refinedShellSteps < 14) {
        subCount = 3;
        refinedShellSteps++;
      } else if (iceShell > 0.00009 && transmittance > 0.12 && coarseSpan > 200.0 && refinedShellSteps < 18) {
        subCount = 2;
        refinedShellSteps++;
      }
      float subSpan = coarseSpan / float(subCount);

      // Altitude-orthogonal jitter: tight (0.22) below 5.6 km for velvety grain-free 3D cumulus domes;
      // full continuous [0..1] (0.98) above 6.4 km for band-free gossamer 3D cirrus mares' tails!
      float cirrusAltBlend = smoothstep(5600.0, 6450.0, max(zA, zB));
      float jitterSpan = mix(0.22, 0.98, cirrusAltBlend);
      float stepJitter = 0.5 + (isoHash - 0.5) * jitterSpan;

      bool shellIrrReady = false;
      float localSunEl = 0.0;
      vec3 directIrr = vec3(0.0);
      vec3 skyZenithIrr = vec3(0.0);
      vec3 groundBounceIrr = vec3(0.0);

      for (int sub = 0; sub < 3; sub++) {
        if (sub >= subCount) break;
        float dt = subSpan;
        float subJit = fract(isoHash + float(sub) * 0.381966);
        float localOff = (cirrusAltBlend > 0.5)
          ? (0.5 + (subJit - 0.5) * jitterSpan)
          : stepJitter;
        float tMid = tStart + (float(sub) + localOff) * subSpan;

        vec3 pos = rayOrigin + rayDir * tMid;
        float horizDist2 = dot(pos.xz - rayOrigin.xz, pos.xz - rayOrigin.xz);
        float curvDrop = horizDist2 / (2.0 * EARTH_R_EFF);
        float zAsl = pos.y + uMinElevM - curvDrop;

        vec4 sigmas = sampleAtmosphereExtinction(pos, zAsl, tMid, dt, 2);
        float sigmaCloud = sigmas.x + sigmas.y;
        float sigmaTotal = sigmaCloud + sigmas.z;
        float gossamerFrac = sigmas.w;

        if (sigmaTotal <= 1.0e-6) {
          continue;
        }

        // Lazily evaluate atmospheric & Corridor LUT irradiance once per active coarse shell
        if (!shellIrrReady) {
          shellIrrReady = true;
          float zShellMid = 0.5 * (zA + zB);
          directIrr = useLunarLight
            ? vec3(0.22, 0.28, 0.42) * uMoonIllum * smoothstep(0.0, 8.0, uMoonElDeg)
            : computeDirectSolarIrradiance(zShellMid, localSunEl);
          computeHemisphereIrradiance(zShellMid, skyZenithIrr, groundBounceIrr);
        }

        float terrainVis = 1.0;
        float cloudTauSun = 0.0;
        if (dot(directIrr, directIrr) > 1.0e-5 && transmittance > 0.04) {
          terrainVis = sampleTerrainSunVisibility(pos, primaryLightDir);
          if (terrainVis > 0.01 && sigmaCloud > 1.0e-5) {
            cloudTauSun = marchLightOpticalDepth(pos, zAsl, primaryLightDir);
          }
        }

        vec3 stepScatter = vec3(0.0);

        if (sigmaCloud > 1.0e-5) {
          float iceFrac = sigmas.y / max(1.0e-6, sigmaCloud);
          float phaseCloud = mix(phaseLiq, phaseIce, iceFrac);

          float twilightWarmth = smoothstep(12.0, -2.0, localSunEl);
          vec3 chromaKappa = mix(vec3(1.0), vec3(0.84, 0.98, 1.15), twilightWarmth * 0.55);

          // Dual-regime w0=0.999 droplet scattering:
          // ms0: crisp single-scatter 3D billow relief; msDiff: creamy alabaster-gold internal diffusion
          // so thick cumulus mounds glow like sunlit ivory/apricot marble rather than dark smoke!
          float isoPhase = 1.0 / (4.0 * PI);
          vec3 ms0 = exp(-cloudTauSun * chromaKappa) * phaseCloud;
          vec3 ms1 = 0.48 * exp(-cloudTauSun * chromaKappa * 0.52) * mix(phaseCloud, isoPhase, 0.55);
          vec3 msDiff = 0.34 * exp(-sqrt(max(0.0, cloudTauSun)) * chromaKappa * 0.86) * isoPhase;
          vec3 multiScatter = ms0 + ms1 + msDiff;

          float silverVeilBoost = 1.0 + 0.62 * gossamerFrac * exp(-cloudTauSun * 1.15) * smoothstep(-0.25, 0.92, cosTheta);
          float corePowder = mix(
            0.66 + 0.34 * (1.0 - exp(-sigmaCloud * dt * 2.2)),
            1.0,
            max(0.0, cosTheta) * 0.60 + gossamerFrac * 0.55 + iceFrac * 0.55
          );

          float skyVis = sampleSkyConeOcclusion(pos, zAsl);
          float bounceReach = exp(-max(0.0, zAsl - uZBaseM) / 3000.0);
          vec3 hemiAmbient = skyZenithIrr * mix(skyVis, 1.0, 0.38 * iceFrac) * 1.32
                           + groundBounceIrr * (0.28 + 0.72 * (1.0 - skyVis)) * bounceReach * 0.58;

          vec3 cloudLum = directIrr * terrainVis * multiScatter * corePowder * silverVeilBoost * 3.65
                        + hemiAmbient * mix(1.0, 1.28, iceFrac);

          stepScatter += cloudLum * sigmaCloud;
        }

        if (sigmas.z > 1.0e-7) {
          float shaftShadow = terrainVis * exp(-cloudTauSun * 0.94);
          vec3 shaftLum = directIrr * shaftShadow * phaseHaze * 4.8 + skyZenithIrr * 0.24;
          stepScatter += shaftLum * sigmas.z;
        }

        float stepTrans = exp(-sigmaTotal * dt);
        vec3 sint = stepScatter * ((1.0 - stepTrans) / max(1.0e-6, sigmaTotal));
        accumLight += transmittance * sint;
        transmittance *= stepTrans;

        if (transmittance < 0.010) {
          transmittance = 0.0;
          break;
        }
      }

      if (transmittance <= 0.0) {
        break;
      }
    }

    fragColor = vec4(accumLight, transmittance);
  }
`;

// Pass B2: Ping-Pong Volumetric Temporal Reprojection & Neighborhood Variance Clamp (TAA)
const VOLUMETRIC_TEMPORAL_RESOLVE_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;

  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uCurrVolTex;
  uniform sampler2D uHistVolTex;
  uniform sampler2D uSceneDepthTex;
  uniform mat4 uInvViewProj;
  uniform mat4 uPrevViewProj;
  uniform vec3 uCameraPos;
  uniform vec3 uWindWorldDelta;
  uniform vec2 uLowResTexel;
  uniform float uHistoryBlend;

  void main() {
    vec4 curr = texture(uCurrVolTex, vUv);
    if (uHistoryBlend <= 0.001) {
      fragColor = curr;
      return;
    }

    float rawDepth = texture(uSceneDepthTex, vUv).r;
    bool hitTerrain = rawDepth < 0.99998;

    vec4 ndcFar = vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
    vec4 wFar = uInvViewProj * ndcFar;
    wFar /= wFar.w;
    vec3 rayDir = normalize(wFar.xyz - uCameraPos);

    float repDist = 7200.0;
    if (hitTerrain) {
      vec4 ndcHit = vec4(vUv * 2.0 - 1.0, rawDepth * 2.0 - 1.0, 1.0);
      vec4 wHit = uInvViewProj * ndcHit;
      wHit /= wHit.w;
      float surfDist = length(wHit.xyz - uCameraPos);
      repDist = min(surfDist * 0.72, 4800.0);
    }

    vec3 wRep = uCameraPos + rayDir * repDist - uWindWorldDelta;
    vec4 prevClip = uPrevViewProj * vec4(wRep, 1.0);
    if (prevClip.w <= 0.001) {
      fragColor = curr;
      return;
    }

    vec2 prevUv = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
    if (prevUv.x < 0.002 || prevUv.x > 0.998 || prevUv.y < 0.002 || prevUv.y > 0.998) {
      fragColor = curr;
      return;
    }

    // 5-tap cross neighborhood variance & min-max AABB clamp prevents any granite silhouette ghosting!
    vec4 nN = texture(uCurrVolTex, clamp(vUv + vec2(0.0, uLowResTexel.y), vec2(0.001), vec2(0.999)));
    vec4 nS = texture(uCurrVolTex, clamp(vUv - vec2(0.0, uLowResTexel.y), vec2(0.001), vec2(0.999)));
    vec4 nE = texture(uCurrVolTex, clamp(vUv + vec2(uLowResTexel.x, 0.0), vec2(0.001), vec2(0.999)));
    vec4 nW = texture(uCurrVolTex, clamp(vUv - vec2(uLowResTexel.x, 0.0), vec2(0.001), vec2(0.999)));

    vec4 m1 = (curr + nN + nS + nE + nW) * 0.2;
    vec4 m2 = (curr * curr + nN * nN + nS * nS + nE * nE + nW * nW) * 0.2;
    vec4 sigma = sqrt(max(vec4(0.0), m2 - m1 * m1));

    vec4 boxMin = min(curr, min(min(nN, nS), min(nE, nW)));
    vec4 boxMax = max(curr, max(max(nN, nS), max(nE, nW)));

    float gammaSlack = hitTerrain ? 0.80 : 1.35;
    vec4 cMin = max(vec4(0.0), min(boxMin, m1 - sigma * gammaSlack));
    vec4 cMax = max(boxMax, m1 + sigma * gammaSlack);
    cMin.a = clamp(cMin.a, 0.0, 1.0);
    cMax.a = clamp(cMax.a, 0.0, 1.0);

    vec4 hist = texture(uHistVolTex, prevUv);
    vec4 clampedHist = clamp(hist, cMin, cMax);

    float uvShift = length(prevUv - vUv);
    float motionAtten = 1.0 - smoothstep(0.015, 0.14, uvShift);
    float blend = uHistoryBlend * mix(0.42, 1.0, motionAtten);

    fragColor = mix(curr, clampedHist, blend);
  }
`;

// Pass C: Full-Retina Depth-Aware Bilateral Upsampler + ACES Tone-Mapped Composite
const COMPOSITE_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;

  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uSceneColorTex;
  uniform sampler2D uSceneDepthTex;
  uniform sampler2D uVolumetricTex;
  uniform vec2 uLowResTexel;
  uniform float uShowAtmosphere;
  uniform float uNear;
  uniform float uFar;

  float linearizeDepth(float d) {
    float z = d * 2.0 - 1.0;
    return (2.0 * uNear * uFar) / max(1e-4, uFar + uNear - z * (uFar - uNear));
  }

  vec3 softHighlightCompress(vec3 col) {
    vec3 x = max(vec3(0.0), col);
    vec3 mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return mix(x, clamp(mapped, 0.0, 1.0), smoothstep(0.62, 1.35, max(x.r, max(x.g, x.b))));
  }

  void main() {
    vec4 sceneCol = texture(uSceneColorTex, vUv);
    float centerDepth = texture(uSceneDepthTex, vUv).r;
    gl_FragDepth = centerDepth;

    if (uShowAtmosphere < 0.5) {
      fragColor = sceneCol;
      return;
    }

    float centerLin = linearizeDepth(centerDepth);

    // Bilinear-tent 9-tap depth-aware bilateral upsampler:
    // Non-integer offsets leverage hardware linear filtering to blend a silky 4x4 neighborhood
    // across volumetric clouds while preserving razor-sharp 1-pixel granite silhouettes!
    const vec2 offsets[9] = vec2[9](
      vec2( 0.00,  0.00),
      vec2(-0.85,  0.00),
      vec2( 0.85,  0.00),
      vec2( 0.00, -0.85),
      vec2( 0.00,  0.85),
      vec2(-1.25, -1.25),
      vec2( 1.25, -1.25),
      vec2(-1.25,  1.25),
      vec2( 1.25,  1.25)
    );
    const float spatialW[9] = float[9](0.20, 0.13, 0.13, 0.13, 0.13, 0.07, 0.07, 0.07, 0.07);

    vec4 accumVol = vec4(0.0);
    float sumW = 0.0;

    for (int i = 0; i < 9; i++) {
      vec2 tapUv = clamp(vUv + offsets[i] * uLowResTexel, vec2(0.001), vec2(0.999));
      float tapDepth = texture(uSceneDepthTex, tapUv).r;
      float tapLin = linearizeDepth(tapDepth);
      float relDiff = abs(centerLin - tapLin) / max(22.0, centerLin * 0.05);
      float w = spatialW[i] * exp(-relDiff * relDiff * 4.2);
      accumVol += texture(uVolumetricTex, tapUv) * w;
      sumW += w;
    }

    vec4 vol = sumW > 1e-4 ? (accumVol / sumW) : texture(uVolumetricTex, vUv);
    vec3 inScatter = vol.rgb;
    float trans = clamp(vol.a, 0.0, 1.0);

    vec3 combined = sceneCol.rgb * trans + inScatter;
    fragColor = vec4(softHighlightCompress(combined), 1.0);
  }
`;

export class VolumetricAtmosphereEngine {
  constructor(renderer, camera) {
    this.renderer = renderer;
    this.camera = camera;

    this.enabled = true;
    this.evolveFluid = true;
    this.evolutionSpeed = 1.0;
    this.activePreset = 'live';
    this.qualityMode = 'ultra';
    this.resScale = 0.52;

    // Detect WebGL2 floating-point render-target support for native GPU 128x128x48 fluid solver
    const gl = renderer.getContext();
    const extColorFloat = renderer.capabilities.isWebGL2 ? gl.getExtension('EXT_color_buffer_float') : null;
    const extColorHalfFloat = renderer.capabilities.isWebGL2 ? gl.getExtension('EXT_color_buffer_half_float') : null;
    this.supportsFloatFBO = Boolean(renderer.capabilities.isWebGL2 && (extColorFloat || extColorHalfFloat));
    this._supportsAsyncFloatPbo = Boolean(renderer.capabilities.isWebGL2 && extColorFloat);
    this.useGpuFluidSolver = this.supportsFloatFBO;

    this.liveWeatherParams = null;
    this.userTuning = {
      cloudDensityGain: 1.0,
      cirrusDensityGain: 1.05,
      microDetailGain: 1.0,
      godRayGain: 1.0,
    };

    this.terrainMeta = {
      gridSize: 600,
      spanM: 9200,
      minElevM: 0,
      maxElevM: 1200,
      centerLat: 37.596,
      centerLon: -122.502,
      metersPerDegLat: 110990,
      metersPerDegLon: 88250,
      version: 0,
    };

    this.telemetry = {
      wasmReady: this.useGpuFluidSolver,
      gpuFluidActive: this.useGpuFluidSolver,
      stepMs: 0.28,
      fps: 60,
      frameMs: 16.6,
      lclAltitudeM: 380,
      freezingLevelM: 3200,
      glaciationCirrusBaseM: 7600,
      maxUpdraftMs: 1.4,
      maxLiquidGKg: 0.58,
      maxIceGKg: 0.36,
      lowCloudCoverage: 46,
      midCloudCoverage: 18,
      highCirrusCoverage: 52,
      froudeNumber: 0.64,
      jetShearMsKm: 4.5,
      zBaseM: 0,
      zTopM: 12200,
      domainXyM: 18000,
    };

    this.onTelemetryCallback = null;
    this.pendingRaycastCallbacks = new Map();
    this.nextRaycastReqId = 1;

    this.workerBusy = false;
    this.lastWorkerStepWallMs = 0;
    this.lastTelemetryWallMs = 0;
    this.simClockS = 0;
    this._prevRenderSimClockS = 0;
    this.frameIndex = 0;
    this._terrainBakeKey = '';
    this._needsTerrainBake = true;
    this._needsShadowBake = true;
    this._lastShadowSunDir = new THREE.Vector3(-0.85, 0.15, 0.2).normalize();
    this._lastShadowSimClockS = -999.0;
    this._volHistIdx = 0;
    this._historyValid = false;
    this._telemetryPbo = null;
    this._telemetryFence = null;

    this._init3DTextures();
    this._initRenderTargetsAndPasses();

    if (this.useGpuFluidSolver) {
      this._generateDetailNoise3DGpu();
      this._initGpuFluidSolver();
      this._reseedGpuFluid();
    } else {
      this._initWorker();
    }
  }

  _init3DTextures() {
    // Fallback 64x64x48 3D texture (only uploaded if WebGL2 Float FBOs are unavailable)
    const volData = new Uint8Array(64 * 64 * 48 * 4);
    this.cloudVolumeTex = new THREE.Data3DTexture(volData, 64, 64, 48);
    this.cloudVolumeTex.format = THREE.RGBAFormat;
    this.cloudVolumeTex.type = THREE.UnsignedByteType;
    this.cloudVolumeTex.minFilter = THREE.LinearFilter;
    this.cloudVolumeTex.magFilter = THREE.LinearFilter;
    this.cloudVolumeTex.wrapS = THREE.RepeatWrapping;
    this.cloudVolumeTex.wrapT = THREE.RepeatWrapping;
    this.cloudVolumeTex.wrapR = THREE.ClampToEdgeWrapping;
    this.cloudVolumeTex.unpackAlignment = 1;
    this.cloudVolumeTex.needsUpdate = true;

    // 64x64x64 RGBA8 Tileable 3D Perlin-Worley + Curl-Filament + 1m Kolmogorov Detail Volume
    // Hardware trilinear mipmaps (64->32->16->8->4) enable artifact-free Nyquist LOD via textureLod!
    const noiseData = new Uint8Array(64 * 64 * 64 * 4);
    this.detailNoiseTex = new THREE.Data3DTexture(noiseData, 64, 64, 64);
    this.detailNoiseTex.format = THREE.RGBAFormat;
    this.detailNoiseTex.type = THREE.UnsignedByteType;
    this.detailNoiseTex.minFilter = THREE.LinearMipmapLinearFilter;
    this.detailNoiseTex.magFilter = THREE.LinearFilter;
    this.detailNoiseTex.generateMipmaps = true;
    this.detailNoiseTex.wrapS = THREE.RepeatWrapping;
    this.detailNoiseTex.wrapT = THREE.RepeatWrapping;
    this.detailNoiseTex.wrapR = THREE.RepeatWrapping;
    this.detailNoiseTex.unpackAlignment = 1;
    this.detailNoiseTex.needsUpdate = true;

    this.dummyHeightTex = new THREE.DataTexture(new Float32Array([0]), 1, 1, THREE.RedFormat, THREE.FloatType);
    this.dummyHeightTex.needsUpdate = true;

    this.dummyAtlasTex = new THREE.DataTexture(new Uint8Array([0, 0, 128, 128]), 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
    this.dummyAtlasTex.needsUpdate = true;

    // 180x32 RGBA8 Curved-Earth 380 km Atmospheric & SRTM Horizon Corridor LUT
    const lutBytes = new Uint8Array(180 * 32 * 4);
    for (let i = 0; i < 180 * 32; i++) {
      lutBytes[i * 4 + 0] = 255;
      lutBytes[i * 4 + 1] = 255;
      lutBytes[i * 4 + 2] = 255;
      lutBytes[i * 4 + 3] = 48; // 0.0 deg horizon
    }
    this.corridorLutTex = new THREE.DataTexture(lutBytes, 180, 32, THREE.RGBAFormat, THREE.UnsignedByteType);
    this.corridorLutTex.minFilter = THREE.LinearFilter;
    this.corridorLutTex.magFilter = THREE.LinearFilter;
    this.corridorLutTex.wrapS = THREE.RepeatWrapping;
    this.corridorLutTex.wrapT = THREE.ClampToEdgeWrapping;
    this.corridorLutTex.flipY = false;
    this.corridorLutTex.unpackAlignment = 1;
    this.corridorLutTex.needsUpdate = true;

    // 16x16 RGBA8 160 km Mesoscale GOES/NWP Spatial Cloud Field (R=low, G=mid, B=high, A=fog)
    const mesoBytes = new Uint8Array(16 * 16 * 4);
    for (let i = 0; i < 16 * 16; i++) {
      mesoBytes[i * 4 + 0] = 96;
      mesoBytes[i * 4 + 1] = 48;
      mesoBytes[i * 4 + 2] = 96;
      mesoBytes[i * 4 + 3] = 32;
    }
    this.mesoscaleCloudTex = new THREE.DataTexture(mesoBytes, 16, 16, THREE.RGBAFormat, THREE.UnsignedByteType);
    this.mesoscaleCloudTex.minFilter = THREE.LinearFilter;
    this.mesoscaleCloudTex.magFilter = THREE.LinearFilter;
    this.mesoscaleCloudTex.wrapS = THREE.ClampToEdgeWrapping;
    this.mesoscaleCloudTex.wrapT = THREE.ClampToEdgeWrapping;
    this.mesoscaleCloudTex.flipY = false;
    this.mesoscaleCloudTex.unpackAlignment = 1;
    this.mesoscaleCloudTex.needsUpdate = true;

    this._hasLoadedCorridorLut = false;
    this._hasCorridorAtmoData = false;
    this._hasLoadedMesoscaleGrid = false;

    this.corridorUniforms = {
      uCorridorLutTex: { value: this.corridorLutTex },
      uHasCorridorHorizon: { value: 0.0 },
      uUseCorridorAtmo: { value: 0.0 },
      uCorridorLutSunElDeg: { value: 0.6 },
    };

    this.mesoscaleUniforms = {
      uMesoscaleCloudTex: { value: this.mesoscaleCloudTex },
      uMesoscaleSpanM: { value: 160000.0 },
      uHasMesoscaleGrid: { value: 0.0 },
    };
  }

  /**
   * Synthesizes all 262,144 voxels of the 64x64x64 tileable Perlin-Worley + Curl + 1m Kolmogorov
   * detail volume directly on the GPU in ~0.8 ms at startup (replacing ~65 ms CPU synthesis).
   */
  _generateDetailNoise3DGpu() {
    const rt = new THREE.WebGLRenderTarget(512, 512, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: false,
    });
    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERTEX_GLSL3,
      fragmentShader: DETAIL_NOISE_GEN_FRAGMENT_GLSL3,
      depthTest: false,
      depthWrite: false,
    });
    const sc = new THREE.Scene();
    const mesh = new THREE.Mesh(this._fsGeo, mat);
    sc.add(mesh);

    const prevRT = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(rt);
    this.renderer.render(sc, this.orthoCam);

    const atlasPixels = new Uint8Array(512 * 512 * 4);
    this.renderer.readRenderTargetPixels(rt, 0, 0, 512, 512, atlasPixels);
    this.renderer.setRenderTarget(prevRT);

    // Unpack 8x8 tile atlas -> contiguous 64x64x64 3D volume texture
    const dst = this.detailNoiseTex.image.data;
    const rowBytes = 64 * 4;
    for (let iz = 0; iz < 64; iz++) {
      const tx = iz & 7;
      const ty = iz >> 3;
      for (let iy = 0; iy < 64; iy++) {
        const srcOff = ((ty * 64 + iy) * 512 + tx * 64) * 4;
        const dstOff = ((iz * 64 + iy) * 64) * 4;
        dst.set(atlasPixels.subarray(srcOff, srcOff + rowBytes), dstOff);
      }
    }
    this.detailNoiseTex.needsUpdate = true;

    mat.dispose();
    rt.dispose();
  }

  _createComputeScene(fragmentShader, uniforms) {
    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERTEX_GLSL3,
      fragmentShader,
      uniforms,
      depthTest: false,
      depthWrite: false,
    });
    const sc = new THREE.Scene();
    sc.add(new THREE.Mesh(this._fsGeo, mat));
    return { mat, scene: sc };
  }

  _initGpuFluidSolver() {
    const stateOpts = {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
      stencilBuffer: false,
    };

    this.velRT = [
      new THREE.WebGLRenderTarget(STATE_ATLAS_W, STATE_ATLAS_H, stateOpts),
      new THREE.WebGLRenderTarget(STATE_ATLAS_W, STATE_ATLAS_H, stateOpts),
    ];
    this.thermoRT = [
      new THREE.WebGLRenderTarget(STATE_ATLAS_W, STATE_ATLAS_H, stateOpts),
      new THREE.WebGLRenderTarget(STATE_ATLAS_W, STATE_ATLAS_H, stateOpts),
    ];
    this.divPressRT = [
      new THREE.WebGLRenderTarget(STATE_ATLAS_W, STATE_ATLAS_H, stateOpts),
      new THREE.WebGLRenderTarget(STATE_ATLAS_W, STATE_ATLAS_H, stateOpts),
    ];
    this.velIdx = 0;
    this.thermoIdx = 0;
    this.divPressIdx = 0;

    // 128x128 GPU DEM Downsampled Terrain + Slope Grid
    this.terrainGridRT = new THREE.WebGLRenderTarget(GPU_NX, GPU_NY, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
    });

    // 1040x780 Padded Hardware-Bilinear Display Cloud & Cirrus Atlas (Zero CPU Readback!)
    this.cloudAtlasRT = new THREE.WebGLRenderTarget(CLOUD_ATLAS_W, CLOUD_ATLAS_H, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
    });

    // 4x4 Float Telemetry Reduction Target (256 bytes) + Async WebGL2 PIXEL_PACK_BUFFER (0.0 ms stall!)
    this.telemetryRT = new THREE.WebGLRenderTarget(4, 4, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.FloatType,
      depthBuffer: false,
    });
    this._telemetryBuf = new Float32Array(4 * 4 * 4);
    if (this._supportsAsyncFloatPbo) {
      const gl = this.renderer.getContext();
      this._telemetryPbo = gl.createBuffer();
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this._telemetryPbo);
      gl.bufferData(gl.PIXEL_PACK_BUFFER, this._telemetryBuf.byteLength, gl.STREAM_READ);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    }

    // Shared simulation parameter uniform objects (mutating .value updates all GPU passes at once!)
    this.simUniforms = {
      uDomainXyM: { value: 18000.0 },
      uZBaseM: { value: 0.0 },
      uZTopM: { value: 12200.0 },
      uSurfaceTempC: { value: 14.0 },
      uSurfaceRhPct: { value: 84.0 },
      uLowCloudPct: { value: 52.0 },
      uMidCloudPct: { value: 20.0 },
      uHighCloudPct: { value: 48.0 },
      uInversionBaseM: { value: 380.0 },
      uInversionThicknessM: { value: 210.0 },
      uInversionStrengthK: { value: 6.2 },
      uWindSfc: { value: new THREE.Vector2(5.2, -3.4) },
      uWind850: { value: new THREE.Vector2(8.0, -2.0) },
      uWind500: { value: new THREE.Vector2(18.0, 4.5) },
      uWind250: { value: new THREE.Vector2(38.0, 14.0) },
      uConvectiveCape: { value: 280.0 },
      uOrographicWaveGain: { value: 1.2 },
      uVorticityConfinement: { value: 1.2 },
      uCirrusIceBoost: { value: 1.05 },
      uSunDirWorld: { value: new THREE.Vector3(-0.85, 0.15, 0.2).normalize() },
      uSunElDeg: { value: 4.0 },
      uDt: { value: 2.2 },
      uSimTime: { value: 0.0 },
    };

    this.terrainDownPass = this._createComputeScene(TERRAIN_DOWNSAMPLE_FRAGMENT_GLSL3, {
      uHeightTex: { value: this.dummyHeightTex },
      uHasHeightTex: { value: 0.0 },
      uSpanM: { value: 9200.0 },
      uDomainXyM: this.simUniforms.uDomainXyM,
      uMinElevM: { value: 0.0 },
    });

    this.fluidInitPass = this._createComputeScene(FLUID_INIT_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      uTerrainGridTex: { value: this.terrainGridRT.texture },
      uInitMode: { value: 0.0 },
    });

    this.fluidForcingPass = this._createComputeScene(FLUID_FORCING_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      uVelTex: { value: this.velRT[0].texture },
      uThermoTex: { value: this.thermoRT[0].texture },
      uTerrainGridTex: { value: this.terrainGridRT.texture },
    });

    this.fluidVorticityPass = this._createComputeScene(FLUID_VORTICITY_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      uVelTex: { value: this.velRT[1].texture },
      uTerrainGridTex: { value: this.terrainGridRT.texture },
    });

    this.fluidDivergencePass = this._createComputeScene(FLUID_DIVERGENCE_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      uVelTex: { value: this.velRT[0].texture },
      uPrevDivPressTex: { value: this.divPressRT[0].texture },
      uTerrainGridTex: { value: this.terrainGridRT.texture },
    });

    this.fluidJacobiPass = this._createComputeScene(FLUID_PRESSURE_JACOBI_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      uDivPressTex: { value: this.divPressRT[0].texture },
    });

    this.fluidProjectAdvectVelPass = this._createComputeScene(FLUID_PROJECT_ADVECT_VEL_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      uVelTex: { value: this.velRT[0].texture },
      uDivPressTex: { value: this.divPressRT[0].texture },
      uTerrainGridTex: { value: this.terrainGridRT.texture },
    });

    this.fluidAdvectMicroPass = this._createComputeScene(FLUID_ADVECT_MICROPHYSICS_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      ...this.mesoscaleUniforms,
      uVelTex: { value: this.velRT[1].texture },
      uThermoTex: { value: this.thermoRT[0].texture },
      uTerrainGridTex: { value: this.terrainGridRT.texture },
    });

    this.fluidPackAtlasPass = this._createComputeScene(FLUID_PACK_ATLAS_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      uVelTex: { value: this.velRT[0].texture },
      uThermoTex: { value: this.thermoRT[0].texture },
      uTerrainGridTex: { value: this.terrainGridRT.texture },
    });

    this.fluidTelemetryPass = this._createComputeScene(FLUID_TELEMETRY_FRAGMENT_GLSL3, {
      ...this.simUniforms,
      uVelTex: { value: this.velRT[0].texture },
      uThermoTex: { value: this.thermoRT[0].texture },
    });

    // Bind GPU cloud atlas directly to Shadow & Volumetric Raymarcher shaders
    this.shadowMat.uniforms.uCloudAtlas2D.value = this.cloudAtlasRT.texture;
    this.shadowMat.uniforms.uUseGpuAtlas.value = 1.0;
    this.volumetricMat.uniforms.uCloudAtlas2D.value = this.cloudAtlasRT.texture;
    this.volumetricMat.uniforms.uUseGpuAtlas.value = 1.0;

    // Initialize flat terrain grid before first DEM arrives
    const prevRT = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.terrainGridRT);
    this.renderer.render(this.terrainDownPass.scene, this.orthoCam);
    this.renderer.setRenderTarget(prevRT);
  }

  _reseedGpuFluid() {
    if (!this.useGpuFluidSolver) return;
    const prevRT = this.renderer.getRenderTarget();
    this.simClockS = 0.0;
    this._prevRenderSimClockS = 0.0;
    this._gpuSimAccumS = 0.0;
    this._historyValid = false;
    this._needsShadowBake = true;

    this.fluidInitPass.mat.uniforms.uInitMode.value = 0.0;
    this.renderer.setRenderTarget(this.velRT[0]);
    this.renderer.render(this.fluidInitPass.scene, this.orthoCam);
    this.velIdx = 0;

    this.fluidInitPass.mat.uniforms.uInitMode.value = 1.0;
    this.renderer.setRenderTarget(this.thermoRT[0]);
    this.renderer.render(this.fluidInitPass.scene, this.orthoCam);
    this.thermoIdx = 0;

    this.renderer.setRenderTarget(this.divPressRT[0]);
    this.renderer.clearColor();
    this.divPressIdx = 0;

    // Run 3 warm-up projection + MacCormack microphysics steps so initial 3D cloud/lee-wave field is sculpted & divergence-free
    this._stepGpuFluidInternal(1.8);
    this._stepGpuFluidInternal(1.8);
    this._stepGpuFluidInternal(1.8);

    this.renderer.setRenderTarget(prevRT);
    this._updateGpuTelemetry(true);
  }

  _stepGpuFluidInternal(dtSim) {
    this.simUniforms.uDt.value = dtSim;
    this.simUniforms.uSimTime.value = this.simClockS;

    const vCur = this.velIdx;
    const vNext = 1 - vCur;
    const tCur = this.thermoIdx;
    const tNext = 1 - tCur;

    // 1. Orographic + Scorer Wave + Slope Thermal + Buoyancy Forcing (vel[vCur] -> vel[vNext])
    this.fluidForcingPass.mat.uniforms.uVelTex.value = this.velRT[vCur].texture;
    this.fluidForcingPass.mat.uniforms.uThermoTex.value = this.thermoRT[tCur].texture;
    this.renderer.setRenderTarget(this.velRT[vNext]);
    this.renderer.render(this.fluidForcingPass.scene, this.orthoCam);

    // 2a. 3D Vorticity Confinement (vel[vNext] -> vel[vCur])
    this.fluidVorticityPass.mat.uniforms.uVelTex.value = this.velRT[vNext].texture;
    this.renderer.setRenderTarget(this.velRT[vCur]);
    this.renderer.render(this.fluidVorticityPass.scene, this.orthoCam);

    // 2b. Anelastic Divergence + Warm Pressure Seed (vel[vCur] -> divPress[dNext])
    let dpCur = this.divPressIdx;
    let dpNext = 1 - dpCur;
    this.fluidDivergencePass.mat.uniforms.uVelTex.value = this.velRT[vCur].texture;
    this.fluidDivergencePass.mat.uniforms.uPrevDivPressTex.value = this.divPressRT[dpCur].texture;
    this.renderer.setRenderTarget(this.divPressRT[dpNext]);
    this.renderer.render(this.fluidDivergencePass.scene, this.orthoCam);
    dpCur = dpNext;

    // 3. 6-Sweep 3D Anisotropic Jacobi Pressure Projection
    for (let iter = 0; iter < 6; iter++) {
      dpNext = 1 - dpCur;
      this.fluidJacobiPass.mat.uniforms.uDivPressTex.value = this.divPressRT[dpCur].texture;
      this.renderer.setRenderTarget(this.divPressRT[dpNext]);
      this.renderer.render(this.fluidJacobiPass.scene, this.orthoCam);
      dpCur = dpNext;
    }
    this.divPressIdx = dpCur;

    // 4a. Pressure Gradient Subtraction + Semi-Lagrangian Velocity Advection (vel[vCur] -> vel[vNext])
    this.fluidProjectAdvectVelPass.mat.uniforms.uVelTex.value = this.velRT[vCur].texture;
    this.fluidProjectAdvectVelPass.mat.uniforms.uDivPressTex.value = this.divPressRT[dpCur].texture;
    this.renderer.setRenderTarget(this.velRT[vNext]);
    this.renderer.render(this.fluidProjectAdvectVelPass.scene, this.orthoCam);
    this.velIdx = vNext;

    // 4b. Semi-Lagrangian Scalar/Sheared-Ice Advection + Clausius-Clapeyron & Koop Microphysics
    this.fluidAdvectMicroPass.mat.uniforms.uVelTex.value = this.velRT[vNext].texture;
    this.fluidAdvectMicroPass.mat.uniforms.uThermoTex.value = this.thermoRT[tCur].texture;
    this.renderer.setRenderTarget(this.thermoRT[tNext]);
    this.renderer.render(this.fluidAdvectMicroPass.scene, this.orthoCam);
    this.thermoIdx = tNext;

    // 5. Pack Padded 1040x780 Hardware-Bilinear Cloud & Cirrus Atlas (Zero CPU Readback!)
    this.fluidPackAtlasPass.mat.uniforms.uVelTex.value = this.velRT[vNext].texture;
    this.fluidPackAtlasPass.mat.uniforms.uThermoTex.value = this.thermoRT[tNext].texture;
    this.renderer.setRenderTarget(this.cloudAtlasRT);
    this.renderer.render(this.fluidPackAtlasPass.scene, this.orthoCam);
    this._needsShadowBake = true;
  }

  _pollAsyncGpuTelemetry() {
    if (!this._telemetryFence || !this._telemetryPbo) return;
    const gl = this.renderer.getContext();
    const status = gl.clientWaitSync(this._telemetryFence, 0, 0);
    if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) {
      gl.deleteSync(this._telemetryFence);
      this._telemetryFence = null;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this._telemetryPbo);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this._telemetryBuf);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this._applyTelemetryBuffer();
    } else if (status === gl.WAIT_FAILED) {
      gl.deleteSync(this._telemetryFence);
      this._telemetryFence = null;
    }
  }

  _updateGpuTelemetry(force = false) {
    if (!this.useGpuFluidSolver) return;
    const nowMs = performance.now();
    if (!force && nowMs - this.lastTelemetryWallMs < 900) return;
    if (!force && this._telemetryFence) return;
    this.lastTelemetryWallMs = nowMs;

    const prevRT = this.renderer.getRenderTarget();
    this.fluidTelemetryPass.mat.uniforms.uVelTex.value = this.velRT[this.velIdx].texture;
    this.fluidTelemetryPass.mat.uniforms.uThermoTex.value = this.thermoRT[this.thermoIdx].texture;
    this.renderer.setRenderTarget(this.telemetryRT);
    this.renderer.render(this.fluidTelemetryPass.scene, this.orthoCam);

    if (!force && this._supportsAsyncFloatPbo && this._telemetryPbo) {
      const gl = this.renderer.getContext();
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE) {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this._telemetryPbo);
        gl.readPixels(0, 0, 4, 4, gl.RGBA, gl.FLOAT, 0);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        this._telemetryFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        this.renderer.setRenderTarget(prevRT);
        return;
      }
    }

    this.renderer.readRenderTargetPixels(this.telemetryRT, 0, 0, 4, 4, this._telemetryBuf);
    this.renderer.setRenderTarget(prevRT);
    this._applyTelemetryBuffer();
  }

  _applyTelemetryBuffer() {
    let maxW = 0.2;
    let maxQc = 0.05;
    let maxQi = 0.02;
    let sumLow = 0;
    let sumMid = 0;
    let sumHigh = 0;
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        const off = (y * 4 + x) * 4;
        maxW = Math.max(maxW, this._telemetryBuf[off]);
        maxQc = Math.max(maxQc, this._telemetryBuf[off + 1]);
        maxQi = Math.max(maxQi, this._telemetryBuf[off + 2]);
        const cov = this._telemetryBuf[off + 3];
        if (y === 0) sumLow += cov;
        else if (y === 1) sumMid += cov;
        else sumHigh += cov;
      }
    }

    const p = this.getEffectiveParams();
    const tC = p.surfaceTempC ?? 14.0;
    const rh = Math.max(15, Math.min(100, p.surfaceRhPct ?? 80));
    const dewSpreadC = ((100 - rh) / 5.0);
    const zBase = this.telemetry.zBaseM;
    const lclM = zBase + Math.max(80, dewSpreadC * 125.0);
    const freezeM = Math.max(zBase + 300, zBase + (tC / 6.3) * 1000.0 + (p.inversionStrengthK ?? 4) * 55.0);
    const wind850 = Math.hypot(p.wind850U ?? 8, p.wind850V ?? -2);
    const reliefM = Math.max(250, (this.terrainMeta.maxElevM || 1200) - zBase);
    const bvN = 0.011 + 0.0008 * (p.inversionStrengthK ?? 4);
    const froude = Math.min(2.8, Math.max(0.18, wind850 / Math.max(1.5, bvN * reliefM)));
    const shear250_500 = Math.hypot(
      (p.wind250U ?? 36) - (p.wind500U ?? 18),
      (p.wind250V ?? 14) - (p.wind500V ?? 5)
    ) / 4.8;

    this.telemetry.lclAltitudeM = lclM;
    this.telemetry.freezingLevelM = freezeM;
    this.telemetry.glaciationCirrusBaseM = Math.max(6800, freezeM + 4100);
    this.telemetry.maxUpdraftMs = maxW;
    this.telemetry.maxLiquidGKg = maxQc;
    this.telemetry.maxIceGKg = maxQi;
    this.telemetry.lowCloudCoverage = Math.min(100, Math.round(sumLow / 4));
    this.telemetry.midCloudCoverage = Math.min(100, Math.round(sumMid / 4));
    this.telemetry.highCirrusCoverage = Math.min(100, Math.round(sumHigh / 8));
    this.telemetry.froudeNumber = froude;
    this.telemetry.jetShearMsKm = shear250_500;

    if (this.volumetricMat) {
      this.volumetricMat.uniforms.uLclAltM.value = lclM;
    }
    if (this.shadowMat) {
      this.shadowMat.uniforms.uLclAltM.value = lclM;
    }

    if (this.onTelemetryCallback) {
      this.onTelemetryCallback(this.telemetry);
    }
  }

  _initRenderTargetsAndPasses() {
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    const w = Math.max(2, size.x);
    const h = Math.max(2, size.y);

    // 1. Opaque World RenderTarget (`skyDome` + `terrainMesh` + `trailsGroup`) with hardware DepthTexture
    this.sceneRT = new THREE.WebGLRenderTarget(w, h, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: true,
      stencilBuffer: false,
    });
    this.sceneRT.depthTexture = new THREE.DepthTexture(w, h, THREE.UnsignedIntType);
    this.sceneRT.depthTexture.minFilter = THREE.NearestFilter;
    this.sceneRT.depthTexture.magFilter = THREE.NearestFilter;

    // 2. HDR Volumetric Raymarch Raw RenderTarget + Ping-Pong Temporal History Targets
    const vw = Math.max(2, Math.round(w * this.resScale));
    const vh = Math.max(2, Math.round(h * this.resScale));
    const volOpts = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
    };
    this.volumetricRT = new THREE.WebGLRenderTarget(vw, vh, volOpts);
    this.volumetricHistRT = [
      new THREE.WebGLRenderTarget(vw, vh, volOpts),
      new THREE.WebGLRenderTarget(vw, vh, volOpts),
    ];

    // 3. 256x256 Ground Cloud-Shadow Transmittance LUT (`shadowMapRT` — dirty-gated!)
    this.shadowMapRT = new THREE.WebGLRenderTarget(256, 256, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: false,
    });

    // 4. 600x600 DEM-Space Topographic Penumbra Shadow, Grazing Clearance & 360° Viewshed Baker (`terrainBakeRT`)
    this.terrainBakeRT = new THREE.WebGLRenderTarget(600, 600, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
      format: THREE.RGBAFormat,
      type: this.supportsFloatFBO ? THREE.HalfFloatType : THREE.UnsignedByteType,
      depthBuffer: false,
    });

    // 5. 245x1 GPU Curved-Earth Tripod Sightline + 360° Horizon Raycaster Target
    if (this.supportsFloatFBO) {
      this.tripodRaycastRT = new THREE.WebGLRenderTarget(245, 1, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        type: THREE.FloatType,
        depthBuffer: false,
      });
      this._tripodReadBuf = new Float32Array(245 * 4);
    }

    this.orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const fsGeo = new THREE.PlaneGeometry(2, 2);
    this._fsGeo = fsGeo;

    this.terrainBakePass = this._createComputeScene(TERRAIN_BAKE_FRAGMENT_GLSL3, {
      uHeightTex: { value: this.dummyHeightTex },
      uGridSize: { value: 600.0 },
      uSpanM: { value: 9200.0 },
      uSunDirENU: { value: new THREE.Vector3(-0.85, -0.35, 0.08).normalize() },
      uTargetEN: { value: new THREE.Vector2(0, 0) },
      uTargetUpperZ: { value: 600.0 },
      uShowViewshed: { value: 0.0 },
      ...this.corridorUniforms,
    });

    if (this.supportsFloatFBO) {
      this.tripodRaycastPass = this._createComputeScene(TRIPOD_RAYCAST_FRAGMENT_GLSL3, {
        uHeightTex: { value: this.dummyHeightTex },
        uGridSize: { value: 600.0 },
        uSpanM: { value: 9200.0 },
        uTripodEN: { value: new THREE.Vector2(0, 0) },
        uTargetEN: { value: new THREE.Vector2(0, 0) },
        uTargetZM: { value: 600.0 },
        ...this.corridorUniforms,
      });
    }

    this.shadowMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERTEX_GLSL3,
      fragmentShader: CLOUD_SHADOW_MAP_FRAGMENT_GLSL3,
      uniforms: {
        uHeightTex: { value: this.dummyHeightTex },
        uHasHeightTex: { value: 0.0 },
        uSpanM: { value: 9200.0 },
        uCloudAtlas2D: { value: this.dummyAtlasTex },
        uCloudVolume3D: { value: this.cloudVolumeTex },
        uUseGpuAtlas: { value: 0.0 },
        ...this.mesoscaleUniforms,
        uDetailNoise3D: { value: this.detailNoiseTex },
        uSunDirWorld: { value: new THREE.Vector3(-0.85, 0.15, 0.2).normalize() },
        uDomainXyM: { value: 18000.0 },
        uZBaseM: { value: 0.0 },
        uZTopM: { value: 12200.0 },
        uMinElevM: { value: 0.0 },
        uLclAltM: { value: 380.0 },
        uInversionBaseM: { value: 380.0 },
        uSimTime: { value: 0.0 },
        uCloudDensityGain: { value: 1.0 },
        uCirrusDensityGain: { value: 1.05 },
        uWindSfc: { value: new THREE.Vector2(5.2, -3.4) },
        uWind500: { value: new THREE.Vector2(18.0, 4.5) },
        uWindAdvectSfc: { value: new THREE.Vector2(0, 0) },
        uWindAdvectJet: { value: new THREE.Vector2(0, 0) },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.shadowScene = new THREE.Scene();
    this.shadowScene.add(new THREE.Mesh(fsGeo, this.shadowMat));

    this.volumetricMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERTEX_GLSL3,
      fragmentShader: VOLUMETRIC_RAYMARCH_FRAGMENT_GLSL3,
      uniforms: {
        uSceneDepthTex: { value: this.sceneRT.depthTexture },
        uHeightTex: { value: this.dummyHeightTex },
        uHasHeightTex: { value: 0.0 },
        uCloudAtlas2D: { value: this.dummyAtlasTex },
        uCloudVolume3D: { value: this.cloudVolumeTex },
        uUseGpuAtlas: { value: 0.0 },
        ...this.mesoscaleUniforms,
        ...this.corridorUniforms,
        uDetailNoise3D: { value: this.detailNoiseTex },
        uInvViewProj: { value: new THREE.Matrix4() },
        uViewProj: { value: new THREE.Matrix4() },
        uCameraPos: { value: new THREE.Vector3() },
        uResolution: { value: new THREE.Vector2(vw, vh) },
        uFrameIndex: { value: 0.0 },
        uSimTime: { value: 0.0 },
        uSpanM: { value: 9200.0 },
        uMinElevM: { value: 0.0 },
        uMaxLocalTerrainY: { value: 1220.0 },
        uDomainXyM: { value: 18000.0 },
        uZBaseM: { value: 0.0 },
        uZTopM: { value: 12200.0 },
        uLclAltM: { value: 380.0 },
        uInversionBaseM: { value: 380.0 },
        uWaveRegime: { value: 1.15 },
        uSunDirWorld: { value: new THREE.Vector3(-0.85, 0.15, 0.2).normalize() },
        uSunElDeg: { value: 4.0 },
        uMoonDirWorld: { value: new THREE.Vector3(-0.6, 0.35, 0.4).normalize() },
        uMoonElDeg: { value: -10.0 },
        uMoonIllum: { value: 0.25 },
        uWindSfc: { value: new THREE.Vector2(5.2, -3.4) },
        uWind500: { value: new THREE.Vector2(18.0, 4.5) },
        uWind250: { value: new THREE.Vector2(36.0, 14.0) },
        uCloudDensityGain: { value: 1.0 },
        uCirrusDensityGain: { value: 1.05 },
        uMicroDetailGain: { value: 1.0 },
        uGodRayGain: { value: 1.0 },
        uQualitySteps: { value: 56.0 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.volumetricScene = new THREE.Scene();
    this.volumetricScene.add(new THREE.Mesh(fsGeo, this.volumetricMat));

    this.temporalResolveMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERTEX_GLSL3,
      fragmentShader: VOLUMETRIC_TEMPORAL_RESOLVE_FRAGMENT_GLSL3,
      uniforms: {
        uCurrVolTex: { value: this.volumetricRT.texture },
        uHistVolTex: { value: this.volumetricHistRT[0].texture },
        uSceneDepthTex: { value: this.sceneRT.depthTexture },
        uInvViewProj: { value: new THREE.Matrix4() },
        uPrevViewProj: { value: new THREE.Matrix4() },
        uCameraPos: { value: new THREE.Vector3() },
        uWindWorldDelta: { value: new THREE.Vector3() },
        uLowResTexel: { value: new THREE.Vector2(1.0 / vw, 1.0 / vh) },
        uHistoryBlend: { value: 0.0 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.temporalResolveScene = new THREE.Scene();
    this.temporalResolveScene.add(new THREE.Mesh(fsGeo, this.temporalResolveMat));

    this.compositeMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERTEX_GLSL3,
      fragmentShader: COMPOSITE_FRAGMENT_GLSL3,
      uniforms: {
        uSceneColorTex: { value: this.sceneRT.texture },
        uSceneDepthTex: { value: this.sceneRT.depthTexture },
        uVolumetricTex: { value: this.volumetricHistRT[0].texture },
        uLowResTexel: { value: new THREE.Vector2(1.0 / vw, 1.0 / vh) },
        uShowAtmosphere: { value: 1.0 },
        uNear: { value: this.camera.near },
        uFar: { value: this.camera.far },
      },
      depthTest: true,
      depthWrite: true,
      depthFunc: THREE.AlwaysDepth,
    });
    this.compositeScene = new THREE.Scene();
    this.compositeScene.add(new THREE.Mesh(fsGeo, this.compositeMat));

    this._tmpViewProj = new THREE.Matrix4();
    this._tmpInvViewProj = new THREE.Matrix4();
    this._prevViewProj = new THREE.Matrix4();
    this._tmpDrawSize = new THREE.Vector2();
  }

  _initWorker() {
    if (this.worker) return;
    try {
      this.worker = new Worker('./sim-worker.js?v=20261115a');
    } catch (err) {
      console.warn('Could not spawn fallback sim-worker.js:', err);
      return;
    }

    this.worker.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg) return;

      if (msg.type === 'ready') {
        this.telemetry.wasmReady = true;
        if (!this.useGpuFluidSolver) {
          const detailBytes = new Uint8Array(msg.detailNoiseBuffer);
          if (detailBytes.length === 64 * 64 * 64 * 4) {
            this.detailNoiseTex.image.data.set(detailBytes);
            this.detailNoiseTex.needsUpdate = true;
          }
          this._applyWorkerFrame(msg);
        }
        if (this.pendingTerrainMsg) {
          const tm = this.pendingTerrainMsg;
          this.pendingTerrainMsg = null;
          this.worker.postMessage(tm, [tm.elevationsBuffer]);
        }
        if (this.liveWeatherParams || this.activePreset !== 'live') {
          this.syncWeatherParamsToWorker(true);
        }
        return;
      }

      if (msg.type === 'frame') {
        this.workerBusy = false;
        if (!this.useGpuFluidSolver) {
          this._applyWorkerFrame(msg);
        }
        return;
      }

      if (msg.type === 'raycast_result') {
        const cb = this.pendingRaycastCallbacks.get(msg.reqId);
        if (cb) {
          this.pendingRaycastCallbacks.delete(msg.reqId);
          cb(new Float32Array(msg.dataBuffer), msg.elapsedMs);
        }
      }
    };

    this.worker.postMessage({ type: 'init', seed: 1337 });
  }

  _applyWorkerFrame(msg) {
    if (msg.volumeBuffer) {
      const volBytes = new Uint8Array(msg.volumeBuffer);
      if (volBytes.length === 64 * 64 * 48 * 4) {
        this.cloudVolumeTex.image.data.set(volBytes);
        this.cloudVolumeTex.needsUpdate = true;
        this._needsShadowBake = true;
      }
    }

    if (msg.telemetry) {
      Object.assign(this.telemetry, msg.telemetry);
      this.telemetry.stepMs = msg.stepMs ?? this.telemetry.stepMs;
      const u = this.volumetricMat.uniforms;
      u.uZBaseM.value = this.telemetry.zBaseM;
      u.uZTopM.value = this.telemetry.zTopM;
      u.uDomainXyM.value = this.telemetry.domainXyM;
      u.uLclAltM.value = this.telemetry.lclAltitudeM;
      this.shadowMat.uniforms.uZBaseM.value = this.telemetry.zBaseM;
      this.shadowMat.uniforms.uZTopM.value = this.telemetry.zTopM;
      this.shadowMat.uniforms.uDomainXyM.value = this.telemetry.domainXyM;
      this.shadowMat.uniforms.uLclAltM.value = this.telemetry.lclAltitudeM;
    }

    if (this.onTelemetryCallback) {
      this.onTelemetryCallback(this.telemetry);
    }
  }

  invalidateTemporalHistory() {
    this._historyValid = false;
  }

  getShadowMapTexture() {
    return this.shadowMapRT.texture;
  }

  getTerrainBakeTexture() {
    return this.terrainBakeRT.texture;
  }

  getCorridorLutTexture() {
    return this.corridorLutTex;
  }

  getCorridorUniforms() {
    return this.corridorUniforms;
  }

  setCorridorLut(lutPayload) {
    if (!lutPayload) {
      this._hasLoadedCorridorLut = false;
      this._hasCorridorAtmoData = false;
      this.corridorUniforms.uHasCorridorHorizon.value = 0.0;
      this.corridorUniforms.uUseCorridorAtmo.value = 0.0;
      this._needsTerrainBake = true;
      return;
    }

    const azBins = Math.max(16, Math.round(lutPayload.az_bins || 180));
    const altBins = Math.max(8, Math.round(lutPayload.alt_bins || 32));
    const expectedLen = azBins * altBins * 4;
    let bytes = null;

    if (typeof lutPayload.rgba8_b64 === 'string' && lutPayload.rgba8_b64.length > 0) {
      const binStr = atob(lutPayload.rgba8_b64);
      if (binStr.length === expectedLen) {
        bytes = new Uint8Array(expectedLen);
        for (let i = 0; i < expectedLen; i++) {
          bytes[i] = binStr.charCodeAt(i);
        }
      }
    } else if (lutPayload.rgba8 && lutPayload.rgba8.length === expectedLen) {
      bytes = lutPayload.rgba8 instanceof Uint8Array ? lutPayload.rgba8 : new Uint8Array(lutPayload.rgba8);
    }

    if (!bytes) return;

    if (this.corridorLutTex.image.width !== azBins || this.corridorLutTex.image.height !== altBins) {
      this.corridorLutTex.dispose();
      this.corridorLutTex = new THREE.DataTexture(bytes, azBins, altBins, THREE.RGBAFormat, THREE.UnsignedByteType);
      this.corridorLutTex.minFilter = THREE.LinearFilter;
      this.corridorLutTex.magFilter = THREE.LinearFilter;
      this.corridorLutTex.wrapS = THREE.RepeatWrapping;
      this.corridorLutTex.wrapT = THREE.ClampToEdgeWrapping;
      this.corridorLutTex.flipY = false;
      this.corridorLutTex.unpackAlignment = 1;
      this.corridorUniforms.uCorridorLutTex.value = this.corridorLutTex;
    } else {
      this.corridorLutTex.image.data.set(bytes);
    }
    this.corridorLutTex.needsUpdate = true;

    this._hasLoadedCorridorLut = true;
    this._hasCorridorAtmoData = lutPayload.has_weather !== false;
    this.corridorUniforms.uHasCorridorHorizon.value = 1.0;
    if (typeof lutPayload.sun_elevation_deg === 'number') {
      this.corridorUniforms.uCorridorLutSunElDeg.value = lutPayload.sun_elevation_deg;
    }
    this.corridorUniforms.uUseCorridorAtmo.value =
      this._hasCorridorAtmoData && this.activePreset === 'live' ? 1.0 : 0.0;
    this._needsTerrainBake = true;
  }

  setMesoscaleCloudGrid(gridPayload) {
    if (!gridPayload) {
      this._hasLoadedMesoscaleGrid = false;
      this.mesoscaleUniforms.uHasMesoscaleGrid.value = 0.0;
      return;
    }

    const n = Math.max(4, Math.round(gridPayload.grid_size || 16));
    const expectedPixels = n * n;
    let bytes = null;

    if (typeof gridPayload.rgba8_b64 === 'string' && gridPayload.rgba8_b64.length > 0) {
      const binStr = atob(gridPayload.rgba8_b64);
      if (binStr.length === expectedPixels * 4) {
        bytes = new Uint8Array(expectedPixels * 4);
        for (let i = 0; i < expectedPixels * 4; i++) {
          bytes[i] = binStr.charCodeAt(i);
        }
      }
    } else if (Array.isArray(gridPayload.cells) && gridPayload.cells.length === expectedPixels) {
      bytes = new Uint8Array(expectedPixels * 4);
      for (let i = 0; i < expectedPixels; i++) {
        const c = gridPayload.cells[i];
        bytes[i * 4 + 0] = Math.min(255, Math.max(0, Math.round(((c.low_pct ?? c.stratus_pct ?? 0) / 100) * 255)));
        bytes[i * 4 + 1] = Math.min(255, Math.max(0, Math.round(((c.mid_pct ?? c.altocumulus_pct ?? 0) / 100) * 255)));
        bytes[i * 4 + 2] = Math.min(255, Math.max(0, Math.round(((c.high_pct ?? c.cirrus_pct ?? 0) / 100) * 255)));
        bytes[i * 4 + 3] = Math.min(255, Math.max(0, Math.round(((c.fog_pct ?? 0) / 100) * 255)));
      }
    }

    if (!bytes) return;

    if (this.mesoscaleCloudTex.image.width !== n || this.mesoscaleCloudTex.image.height !== n) {
      this.mesoscaleCloudTex.dispose();
      this.mesoscaleCloudTex = new THREE.DataTexture(bytes, n, n, THREE.RGBAFormat, THREE.UnsignedByteType);
      this.mesoscaleCloudTex.minFilter = THREE.LinearFilter;
      this.mesoscaleCloudTex.magFilter = THREE.LinearFilter;
      this.mesoscaleCloudTex.wrapS = THREE.ClampToEdgeWrapping;
      this.mesoscaleCloudTex.wrapT = THREE.ClampToEdgeWrapping;
      this.mesoscaleCloudTex.flipY = false;
      this.mesoscaleCloudTex.unpackAlignment = 1;
      this.mesoscaleUniforms.uMesoscaleCloudTex.value = this.mesoscaleCloudTex;
    } else {
      this.mesoscaleCloudTex.image.data.set(bytes);
    }
    this.mesoscaleCloudTex.needsUpdate = true;
    this.mesoscaleUniforms.uMesoscaleSpanM.value = gridPayload.span_m || 160000.0;
    this._hasLoadedMesoscaleGrid = true;
    this.mesoscaleUniforms.uHasMesoscaleGrid.value = this.activePreset === 'live' ? 1.0 : 0.0;
  }

  setQualityMode(mode) {
    this.qualityMode = mode;
    if (mode === 'performance') {
      this.resScale = 0.38;
      this.volumetricMat.uniforms.uQualitySteps.value = 38.0;
    } else if (mode === 'balanced') {
      this.resScale = 0.45;
      this.volumetricMat.uniforms.uQualitySteps.value = 46.0;
    } else {
      this.resScale = 0.52;
      this.volumetricMat.uniforms.uQualitySteps.value = 56.0;
    }
    this.resize();
  }

  setPreset(presetKey) {
    if (!WEATHER_REGIME_PRESETS[presetKey]) return;
    this.activePreset = presetKey;
    this.syncWeatherParamsToWorker(true);
  }

  setUserTuning(partial) {
    Object.assign(this.userTuning, partial);
    const u = this.volumetricMat.uniforms;
    u.uCloudDensityGain.value = this.userTuning.cloudDensityGain;
    u.uCirrusDensityGain.value = this.userTuning.cirrusDensityGain;
    u.uMicroDetailGain.value = this.userTuning.microDetailGain;
    u.uGodRayGain.value = this.userTuning.godRayGain;
    this.shadowMat.uniforms.uCloudDensityGain.value = this.userTuning.cloudDensityGain;
    this.shadowMat.uniforms.uCirrusDensityGain.value = this.userTuning.cirrusDensityGain;
    this._needsShadowBake = true;
  }

  setTerrainDem(elevationsFloat32, gridSize, spanM, minElevM, centerLat, centerLon, heightTex, maxElevM = null) {
    const u = this.volumetricMat.uniforms;
    u.uSpanM.value = spanM;
    u.uMinElevM.value = minElevM;
    this.shadowMat.uniforms.uSpanM.value = spanM;
    this.shadowMat.uniforms.uMinElevM.value = minElevM;

    const computedMaxElev =
      maxElevM !== null
        ? maxElevM
        : elevationsFloat32.reduce((acc, v) => (v > acc ? v : acc), minElevM + 100);
    u.uMaxLocalTerrainY.value = Math.max(25.0, computedMaxElev - minElevM + 20.0);

    const latRad = ((centerLat || 37.6) * Math.PI) / 180.0;
    this.terrainMeta = {
      gridSize,
      spanM,
      minElevM,
      maxElevM: computedMaxElev,
      centerLat,
      centerLon,
      metersPerDegLat: 111132.92 - 559.82 * Math.cos(2 * latRad),
      metersPerDegLon: 111412.84 * Math.cos(latRad) - 93.5 * Math.cos(3 * latRad),
      version: (this.terrainMeta.version || 0) + 1,
    };

    const domainXyM = Math.max(18000.0, spanM * 1.95);
    const zBaseM = Math.max(0.0, minElevM);
    const zTopM = Math.max(12200.0, computedMaxElev + 8600.0);
    this.telemetry.domainXyM = domainXyM;
    this.telemetry.zBaseM = zBaseM;
    this.telemetry.zTopM = zTopM;

    u.uDomainXyM.value = domainXyM;
    u.uZBaseM.value = zBaseM;
    u.uZTopM.value = zTopM;
    this.shadowMat.uniforms.uDomainXyM.value = domainXyM;
    this.shadowMat.uniforms.uZBaseM.value = zBaseM;
    this.shadowMat.uniforms.uZTopM.value = zTopM;
    this._needsShadowBake = true;
    this._historyValid = false;

    if (heightTex) {
      u.uHeightTex.value = heightTex;
      u.uHasHeightTex.value = 1.0;
      this.shadowMat.uniforms.uHeightTex.value = heightTex;
      this.shadowMat.uniforms.uHasHeightTex.value = 1.0;

      // Resize & configure GPU DEM Bake target to match patch grid resolution
      if (this.terrainBakeRT.width !== gridSize || this.terrainBakeRT.height !== gridSize) {
        this.terrainBakeRT.setSize(gridSize, gridSize);
      }
      const bu = this.terrainBakePass.mat.uniforms;
      bu.uHeightTex.value = heightTex;
      bu.uGridSize.value = gridSize;
      bu.uSpanM.value = spanM;
      this._needsTerrainBake = true;

      if (this.tripodRaycastPass) {
        const tu = this.tripodRaycastPass.mat.uniforms;
        tu.uHeightTex.value = heightTex;
        tu.uGridSize.value = gridSize;
        tu.uSpanM.value = spanM;
      }
    }

    if (this.useGpuFluidSolver) {
      this.simUniforms.uDomainXyM.value = domainXyM;
      this.simUniforms.uZBaseM.value = zBaseM;
      this.simUniforms.uZTopM.value = zTopM;

      const tdu = this.terrainDownPass.mat.uniforms;
      tdu.uHeightTex.value = heightTex || this.dummyHeightTex;
      tdu.uHasHeightTex.value = heightTex ? 1.0 : 0.0;
      tdu.uSpanM.value = spanM;
      tdu.uMinElevM.value = minElevM;

      const prevRT = this.renderer.getRenderTarget();
      this.renderer.setRenderTarget(this.terrainGridRT);
      this.renderer.render(this.terrainDownPass.scene, this.orthoCam);
      this.renderer.setRenderTarget(prevRT);

      this._reseedGpuFluid();
      return;
    }

    const copyBuf = elevationsFloat32.slice().buffer;
    const msg = {
      type: 'set_terrain',
      elevationsBuffer: copyBuf,
      width: gridSize,
      height: gridSize,
      patchWidthM: spanM,
      patchHeightM: spanM,
      centerLat,
      centerLon,
    };

    if (this.worker && this.telemetry.wasmReady) {
      this.worker.postMessage(msg, [msg.elevationsBuffer]);
    } else {
      this.pendingTerrainMsg = msg;
    }
  }

  updateTerrainBakeUniforms({ sunDirENU, targetEN, targetUpperZ, showViewshed, isScrubbing = false }) {
    const bu = this.terrainBakePass.mat.uniforms;
    if (sunDirENU) bu.uSunDirENU.value.copy(sunDirENU);
    if (targetEN) bu.uTargetEN.value.copy(targetEN);
    if (typeof targetUpperZ === 'number') bu.uTargetUpperZ.value = targetUpperZ;
    if (typeof showViewshed === 'boolean' || typeof showViewshed === 'number') {
      bu.uShowViewshed.value = showViewshed ? 1.0 : 0.0;
    }

    const s = bu.uSunDirENU.value;
    const t = bu.uTargetEN.value;
    // Quantize slightly more coarsely (~0.35 deg) during rapid slider dragging, then settle to exact 0.005 deg on release
    const sunDigits = isScrubbing ? 2 : 4;
    const key = `${this.terrainMeta.version}|${s.x.toFixed(sunDigits)},${s.y.toFixed(sunDigits)},${s.z.toFixed(sunDigits)}|${t.x.toFixed(1)},${t.y.toFixed(1)},${bu.uTargetUpperZ.value.toFixed(1)}|${bu.uShowViewshed.value}|${this.corridorUniforms.uHasCorridorHorizon.value}`;
    if (key !== this._terrainBakeKey) {
      this._terrainBakeKey = key;
      this._needsTerrainBake = true;
    }
  }

  setLiveWeatherReport(weather) {
    if (!weather) return;
    if (weather.corridor_lut) {
      this.setCorridorLut(weather.corridor_lut);
    }
    if (weather.mesoscale_cloud_grid) {
      this.setMesoscaleCloudGrid(weather.mesoscale_cloud_grid);
    } else if (weather.spatial_cloud_field?.cells?.length) {
      const n = Math.round(Math.sqrt(weather.spatial_cloud_field.cells.length));
      if (n * n === weather.spatial_cloud_field.cells.length) {
        this.setMesoscaleCloudGrid({
          grid_size: n,
          span_m: weather.spatial_cloud_field.span_m || 160000.0,
          cells: weather.spatial_cloud_field.cells,
        });
      }
    }
    if (weather.fluid_atmosphere_params) {
      this.liveWeatherParams = { ...weather.fluid_atmosphere_params };
    } else {
      const spdMs = Math.max(1.0, (weather.wind_mph || 10) * 0.44704);
      const rad = ((weather.wind_dir_deg || 285) * Math.PI) / 180;
      const uSfc = -spdMs * Math.sin(rad);
      const vSfc = -spdMs * Math.cos(rad);
      this.liveWeatherParams = {
        surfaceTempC: ((weather.temp_sunset_f ?? 58) - 32) * (5 / 9),
        surfaceRhPct: 82,
        lowCloudPct: weather.cloud_low_pct ?? 45,
        midCloudPct: weather.cloud_mid_pct ?? 22,
        highCloudPct: weather.cloud_high_pct ?? 42,
        inversionBaseM: (weather.fog_top_ft || 1200) * 0.3048,
        inversionThicknessM: 220,
        inversionStrengthK: 6.0,
        windSfcU: uSfc,
        windSfcV: vSfc,
        wind850U: uSfc * 1.4,
        wind850V: vSfc * 1.2,
        wind500U: 18.0,
        wind500V: 5.0,
        wind250U: 38.0,
        wind250V: 14.0,
        convectiveCape: 280,
        orographicWaveGain: 1.15,
        vorticityConfinement: 1.15,
        cirrusIceBoost: 1.0,
        seed: 1337,
      };
    }
    if (this.activePreset === 'live') {
      this.syncWeatherParamsToWorker(true);
    }
  }

  getEffectiveParams() {
    const presetObj = WEATHER_REGIME_PRESETS[this.activePreset];
    const base = this.liveWeatherParams || WEATHER_REGIME_PRESETS.coastal_fog_sea.overrides;
    const merged = presetObj && presetObj.overrides ? { ...base, ...presetObj.overrides } : { ...base };
    merged.solarElevationDeg = this.volumetricMat.uniforms.uSunElDeg.value;
    const sd = this.volumetricMat.uniforms.uSunDirWorld.value;
    merged.solarAzimuthDeg = ((Math.atan2(sd.x, -sd.z) * 180) / Math.PI + 360) % 360;
    return merged;
  }

  syncWeatherParamsToWorker(reseed = false) {
    const isLive = this.activePreset === 'live';
    this.corridorUniforms.uUseCorridorAtmo.value =
      isLive && this._hasLoadedCorridorLut && this._hasCorridorAtmoData ? 1.0 : 0.0;
    this.mesoscaleUniforms.uHasMesoscaleGrid.value =
      isLive && this._hasLoadedMesoscaleGrid ? 1.0 : 0.0;

    const p = this.getEffectiveParams();
    const u = this.volumetricMat.uniforms;
    u.uWindSfc.value.set(p.windSfcU ?? 5.2, p.windSfcV ?? -3.4);
    u.uWind500.value.set(p.wind500U ?? 18.0, p.wind500V ?? 4.5);
    u.uWind250.value.set(p.wind250U ?? 36.0, p.wind250V ?? 14.0);
    u.uInversionBaseM.value = p.inversionBaseM ?? 380.0;
    u.uWaveRegime.value = p.orographicWaveGain ?? 1.15;

    this.shadowMat.uniforms.uWindSfc.value.set(p.windSfcU ?? 5.2, p.windSfcV ?? -3.4);
    this.shadowMat.uniforms.uWind500.value.set(p.wind500U ?? 18.0, p.wind500V ?? 4.5);
    this.shadowMat.uniforms.uInversionBaseM.value = p.inversionBaseM ?? 380.0;
    this._needsShadowBake = true;

    if (this.useGpuFluidSolver) {
      const su = this.simUniforms;
      su.uSurfaceTempC.value = p.surfaceTempC ?? 14.0;
      su.uSurfaceRhPct.value = p.surfaceRhPct ?? 82.0;
      su.uLowCloudPct.value = p.lowCloudPct ?? 48.0;
      su.uMidCloudPct.value = p.midCloudPct ?? 20.0;
      su.uHighCloudPct.value = p.highCloudPct ?? 48.0;
      su.uInversionBaseM.value = p.inversionBaseM ?? 380.0;
      su.uInversionThicknessM.value = p.inversionThicknessM ?? 210.0;
      su.uInversionStrengthK.value = p.inversionStrengthK ?? 6.0;
      su.uWindSfc.value.set(p.windSfcU ?? 5.2, p.windSfcV ?? -3.4);
      su.uWind850.value.set(p.wind850U ?? 8.0, p.wind850V ?? -2.0);
      su.uWind500.value.set(p.wind500U ?? 18.0, p.wind500V ?? 4.5);
      su.uWind250.value.set(p.wind250U ?? 36.0, p.wind250V ?? 14.0);
      su.uConvectiveCape.value = p.convectiveCape ?? 280.0;
      su.uOrographicWaveGain.value = p.orographicWaveGain ?? 1.2;
      su.uVorticityConfinement.value = p.vorticityConfinement ?? 1.2;
      su.uCirrusIceBoost.value = p.cirrusIceBoost ?? 1.05;
      if (reseed) {
        this._reseedGpuFluid();
      } else {
        this._updateGpuTelemetry(false);
      }
      return;
    }

    if (this.worker && this.telemetry.wasmReady) {
      this.worker.postMessage({
        type: 'set_params',
        params: p,
        reseed: Boolean(reseed),
      });
    }
  }

  updateCelestialUniforms({ sunDirWorld, sunElDeg, moonDirWorld, moonElDeg, moonIllum }) {
    const u = this.volumetricMat.uniforms;
    if (sunDirWorld) {
      if (sunDirWorld.distanceToSquared(this._lastShadowSunDir) > 2.5e-6) {
        this._lastShadowSunDir.copy(sunDirWorld);
        this._needsShadowBake = true;
      }
      u.uSunDirWorld.value.copy(sunDirWorld);
      this.shadowMat.uniforms.uSunDirWorld.value.copy(sunDirWorld);
      if (this.simUniforms) {
        this.simUniforms.uSunDirWorld.value.copy(sunDirWorld);
      }
    }
    if (typeof sunElDeg === 'number') {
      u.uSunElDeg.value = sunElDeg;
      if (this.simUniforms) {
        this.simUniforms.uSunElDeg.value = sunElDeg;
      }
    }
    if (moonDirWorld) {
      u.uMoonDirWorld.value.copy(moonDirWorld);
    }
    if (typeof moonElDeg === 'number') {
      u.uMoonElDeg.value = moonElDeg;
    }
    if (typeof moonIllum === 'number') {
      u.uMoonIllum.value = moonIllum;
    }
  }

  /**
   * Executes a single 245x1 WebGL2 Float FBO pass on GPU to compute both the 65-sample curved-Earth
   * sightline cross-section AND all 180 azimuths of the 360° DEM horizon profile in ~0.1 ms.
   */
  raycastTripodGpu(tripodLat, tripodLon, targetLat, targetLon, targetZM) {
    if (!this.supportsFloatFBO || !this.tripodRaycastPass) {
      return null;
    }
    const t0 = performance.now();
    const meta = this.terrainMeta;
    const tripodE = (tripodLon - meta.centerLon) * meta.metersPerDegLon;
    const tripodN = (tripodLat - meta.centerLat) * meta.metersPerDegLat;
    const targetE = (targetLon - meta.centerLon) * meta.metersPerDegLon;
    const targetN = (targetLat - meta.centerLat) * meta.metersPerDegLat;

    const tu = this.tripodRaycastPass.mat.uniforms;
    tu.uTripodEN.value.set(tripodE, tripodN);
    tu.uTargetEN.value.set(targetE, targetN);
    tu.uTargetZM.value = targetZM;

    const prevRT = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.tripodRaycastRT);
    this.renderer.render(this.tripodRaycastPass.scene, this.orthoCam);
    this.renderer.readRenderTargetPixels(this.tripodRaycastRT, 0, 0, 245, 1, this._tripodReadBuf);
    this.renderer.setRenderTarget(prevRT);

    const out = new Float32Array(383);
    const totalDistM = Math.max(10.0, Math.hypot(targetE - tripodE, targetN - tripodN));
    const bearingDeg = ((Math.atan2(targetE - tripodE, targetN - tripodN) * 180) / Math.PI + 360) % 360;
    const tripodGroundZ = this._tripodReadBuf[65 * 4 + 3];
    const camEyeZ = tripodGroundZ + 1.75;
    const aimZ = this._tripodReadBuf[64 * 4 + 2];
    const dropTarget = (totalDistM * totalDistM) / (2.0 * 7328000.0);
    const elAngleDeg = (Math.atan((aimZ - dropTarget - camEyeZ) / totalDistM) * 180) / Math.PI;

    let maxObsM = 0.0;
    let obsDistM = 0.0;
    for (let s = 0; s <= 64; s++) {
      const off = s * 4;
      const distM = this._tripodReadBuf[off];
      const effTz = this._tripodReadBuf[off + 1];
      const rayZ = this._tripodReadBuf[off + 2];
      const excessM = this._tripodReadBuf[off + 3];
      if (s >= 3 && s < 62 && excessM > maxObsM) {
        maxObsM = excessM;
        obsDistM = distM;
      }
      const base = 8 + s * 3;
      out[base] = distM;
      out[base + 1] = effTz;
      out[base + 2] = rayZ;
    }

    const isVisible = maxObsM <= 1.5 ? 1.0 : 0.0;
    const distBonus = totalDistM >= 1200.0 && totalDistM <= 7500.0 ? 1.0 : 0.78;
    const compScore = isVisible > 0.5 ? 84.0 * distBonus : Math.max(18.0, 55.0 - maxObsM * 0.4);

    out[0] = tripodGroundZ;
    out[1] = totalDistM;
    out[2] = bearingDeg;
    out[3] = elAngleDeg;
    out[4] = maxObsM;
    out[5] = obsDistM;
    out[6] = isVisible;
    out[7] = compScore;

    for (let a = 0; a < 180; a++) {
      const off = (65 + a) * 4;
      out[203 + a] = this._tripodReadBuf[off + 1];
    }

    return { data: out, elapsedMs: performance.now() - t0, backend: 'gpu' };
  }

  raycastTripodWasm(tripodLat, tripodLon, targetLat, targetLon, targetZM) {
    if (this.supportsFloatFBO) {
      const gpuHit = this.raycastTripodGpu(tripodLat, tripodLon, targetLat, targetLon, targetZM);
      if (gpuHit) return Promise.resolve(gpuHit);
    }

    if (!this.worker) {
      this._initWorker();
    }
    if (!this.worker || !this.telemetry.wasmReady) {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      const reqId = this.nextRaycastReqId++;
      const timer = setTimeout(() => {
        this.pendingRaycastCallbacks.delete(reqId);
        resolve(null);
      }, 250);
      this.pendingRaycastCallbacks.set(reqId, (floatArr, elapsedMs) => {
        clearTimeout(timer);
        resolve({ data: floatArr, elapsedMs, backend: 'wasm' });
      });
      this.worker.postMessage({
        type: 'raycast_tripod',
        reqId,
        tripodLat,
        tripodLon,
        targetLat,
        targetLon,
        targetZM,
      });
    });
  }

  resize() {
    this.renderer.getDrawingBufferSize(this._tmpDrawSize);
    const w = Math.max(2, this._tmpDrawSize.x);
    const h = Math.max(2, this._tmpDrawSize.y);
    this.sceneRT.setSize(w, h);
    if (this.sceneRT.depthTexture) {
      this.sceneRT.depthTexture.image.width = w;
      this.sceneRT.depthTexture.image.height = h;
      this.sceneRT.depthTexture.needsUpdate = true;
    }
    const vw = Math.max(2, Math.round(w * this.resScale));
    const vh = Math.max(2, Math.round(h * this.resScale));
    this.volumetricRT.setSize(vw, vh);
    if (this.volumetricHistRT) {
      this.volumetricHistRT[0].setSize(vw, vh);
      this.volumetricHistRT[1].setSize(vw, vh);
    }
    this._historyValid = false;
    this.volumetricMat.uniforms.uResolution.value.set(vw, vh);
    if (this.temporalResolveMat) {
      this.temporalResolveMat.uniforms.uLowResTexel.value.set(1.0 / vw, 1.0 / vh);
    }
    this.compositeMat.uniforms.uLowResTexel.value.set(1.0 / vw, 1.0 / vh);
  }

  /**
   * Executes the unified GPU Compute + Volumetric Rendering Pipeline:
   * 1. Steps native WebGL2 128x128x48 Moist Navier-Stokes + 3D Cirrus solver directly in VRAM (0 CPU-GPU copies)
   * 2. Updates DEM-space Topographic Penumbra Shadow & 360° Viewshed LUT (`terrainBakeRT`) ONLY when dirty
   * 3. Updates 256x256 3D Cloud Ground-Shadow LUT (`shadowMapRT`) ONLY when cloud field or sun changed
   * 4. Renders `worldScene` (`skyDome` + GPU-displaced `terrainGroup` + `trailsGroup`) -> `sceneRT`
   * 5. Renders `volumetricScene` -> `volumetricRT` + Temporal Reprojection Resolve -> `volumetricHistRT`
   * 6. Depth-Aware Bilateral Composites `sceneRT` + `volumetricHistRT` -> Screen (`gl_FragDepth`) + `overlayScene`
   */
  renderPipeline(worldScene, overlayScene, dtRealS, showAtmosphere) {
    this.frameIndex = (this.frameIndex + 1) % 4096;
    const clampedDt = Math.min(0.1, Math.max(0.001, dtRealS));

    if (this.useGpuFluidSolver) {
      this._pollAsyncGpuTelemetry();
    }

    if (this.evolveFluid && showAtmosphere && !document.hidden) {
      // Majestic, physically grounded 1.6x real-time cloud evolution in 1:1 Lagrangian lockstep!
      const dtSim = clampedDt * 1.6 * this.evolutionSpeed;
      this.simClockS += dtSim;

      if (this.useGpuFluidSolver) {
        this._gpuSimAccumS = (this._gpuSimAccumS || 0) + dtSim;
        const targetCadenceS = Math.max(0.14, Math.min(0.45, 0.18 * this.evolutionSpeed));
        if (this._gpuSimAccumS >= targetCadenceS) {
          const t0 = performance.now();
          const simStepDt = Math.min(2.2, this._gpuSimAccumS);
          this._gpuSimAccumS = 0.0;
          this._stepGpuFluidInternal(simStepDt);
          const stepElapsed = performance.now() - t0;
          this.telemetry.stepMs = this.telemetry.stepMs * 0.85 + stepElapsed * 0.15;
          this._updateGpuTelemetry(false);
        }
      } else {
        const nowMs = performance.now();
        if (
          this.worker &&
          this.telemetry.wasmReady &&
          !this.workerBusy &&
          nowMs - this.lastWorkerStepWallMs >= 110
        ) {
          this.workerBusy = true;
          this.lastWorkerStepWallMs = nowMs;
          this.worker.postMessage({
            type: 'step',
            dt: 0.18 * this.evolutionSpeed,
          });
        }
      }
    }

    // Ensure render targets match drawing buffer dimensions
    this.renderer.getDrawingBufferSize(this._tmpDrawSize);
    if (
      Math.abs(this._tmpDrawSize.x - this.sceneRT.width) > 1 ||
      Math.abs(this._tmpDrawSize.y - this.sceneRT.height) > 1
    ) {
      this.resize();
    }

    // Bake DEM-space topographic soft-penumbra shadows & viewshed ONLY when sun/target/DEM changed!
    if (this._needsTerrainBake) {
      this._needsTerrainBake = false;
      this.renderer.setRenderTarget(this.terrainBakeRT);
      this.renderer.clear();
      this.renderer.render(this.terrainBakePass.scene, this.orthoCam);
    }

    this.camera.updateMatrixWorld();
    this._tmpViewProj.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this._tmpInvViewProj.copy(this._tmpViewProj).invert();

    const u = this.volumetricMat.uniforms;
    u.uViewProj.value.copy(this._tmpViewProj);
    u.uInvViewProj.value.copy(this._tmpInvViewProj);
    u.uCameraPos.value.copy(this.camera.position);
    u.uFrameIndex.value = this.frameIndex;
    u.uSimTime.value = this.simClockS;

    // 0. Dirty-gated 256x256 3D Cloud Shadow Map (re-renders ONLY when fluid steps, sun moves, or ~0.25s drift)
    const simDeltaShadow = Math.abs(this.simClockS - this._lastShadowSimClockS);
    if (showAtmosphere && (this._needsShadowBake || simDeltaShadow >= 0.25)) {
      this._needsShadowBake = false;
      this._lastShadowSimClockS = this.simClockS;
      this.shadowMat.uniforms.uSimTime.value = this.simClockS;
      const sfcAdv = modSimOffset(u.uWindSfc.value, this.simClockS);
      this.shadowMat.uniforms.uWindAdvectSfc.value.set(sfcAdv.x, sfcAdv.y);
      this.renderer.setRenderTarget(this.shadowMapRT);
      this.renderer.clear();
      this.renderer.render(this.shadowScene, this.orthoCam);
    }

    // 1. Opaque World + Trails -> `sceneRT` (captures exact terrain `DepthTexture`)
    this.renderer.setRenderTarget(this.sceneRT);
    this.renderer.clear(true, true, false);
    this.renderer.render(worldScene, this.camera);

    // 2. Full-Atmosphere 3D Volumetric Raymarcher -> `volumetricRT` + Ping-Pong Temporal Reprojection -> `volumetricHistRT`
    if (showAtmosphere) {
      this.renderer.setRenderTarget(this.volumetricRT);
      this.renderer.clear(true, false, false);
      this.renderer.render(this.volumetricScene, this.orthoCam);

      const prevHistIdx = this._volHistIdx;
      const nextHistIdx = 1 - prevHistIdx;
      const dtSimFrame = Math.max(0.0, this.simClockS - this._prevRenderSimClockS);
      this._prevRenderSimClockS = this.simClockS;

      const tru = this.temporalResolveMat.uniforms;
      tru.uCurrVolTex.value = this.volumetricRT.texture;
      tru.uHistVolTex.value = this.volumetricHistRT[prevHistIdx].texture;
      tru.uInvViewProj.value.copy(this._tmpInvViewProj);
      tru.uPrevViewProj.value.copy(this._prevViewProj);
      tru.uCameraPos.value.copy(this.camera.position);
      const w500 = u.uWind500.value;
      tru.uWindWorldDelta.value.set(w500.x * dtSimFrame * 0.45, 0.0, -w500.y * dtSimFrame * 0.45);
      tru.uHistoryBlend.value = this._historyValid ? 0.86 : 0.0;

      this.renderer.setRenderTarget(this.volumetricHistRT[nextHistIdx]);
      this.renderer.render(this.temporalResolveScene, this.orthoCam);

      this.compositeMat.uniforms.uVolumetricTex.value = this.volumetricHistRT[nextHistIdx].texture;
      this._volHistIdx = nextHistIdx;
      this._prevViewProj.copy(this._tmpViewProj);
      this._historyValid = true;
    } else {
      this._historyValid = false;
    }

    // 3. Depth-Aware Bilateral Composite -> Screen (writes `gl_FragDepth = sceneDepth`!)
    this.compositeMat.uniforms.uShowAtmosphere.value = showAtmosphere ? 1.0 : 0.0;
    this.compositeMat.uniforms.uNear.value = this.camera.near;
    this.compositeMat.uniforms.uFar.value = this.camera.far;

    this.renderer.setRenderTarget(null);
    const prevAutoClear = this.renderer.autoClear;
    this.renderer.autoClear = true;
    this.renderer.render(this.compositeScene, this.orthoCam);

    // 4. Crisp 3D HUD Overlay (`sunBeamsGroup`, `customPinGroup`, `lightFieldGroup`) depth-tested against terrain!
    if (overlayScene) {
      this.renderer.autoClear = false;
      this.renderer.render(overlayScene, this.camera);
    }
    this.renderer.autoClear = prevAutoClear;
  }
}

function modSimOffset(windVec2, t) {
  const wrapped = t % 900.0;
  return { x: windVec2.x * wrapped, y: windVec2.y * wrapped };
}
