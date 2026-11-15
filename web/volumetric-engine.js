// ============================================================================
// PhotoPlanner 3D Volumetric Atmosphere, 1m Cloud & Upper-Tropospheric Cirrus Engine
// Powered by Rust WebAssembly (`pp_wasm.wasm` in `sim-worker.js`) + WebGL2 GLSL3
// ============================================================================

import * as THREE from 'three';

export const WEATHER_REGIME_PRESETS = {
  live: {
    label: 'Live Forecast Imputation (HRRR / Sounding)',
    description: 'Couples Open-Meteo / HRRR pressure-level sounding directly into the 3D Rust Navier-Stokes & ice solver.',
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

const FULLSCREEN_VERTEX_GLSL3 = /* glsl */ `
  out vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

// Pass A: 0.03 ms GPU 128x128 Cloud Shadow Transmittance LUT (`shadowMapRT`)
// Projects 3D liquid + cirrus cloud optical depth onto ground `(east_m, north_m)` along `uSunDirWorld`
// so `TERRAIN_FRAGMENT_SHADER` casts real drifting 3D cloud shadows across mountains and ocean!
const CLOUD_SHADOW_MAP_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;
  precision highp sampler3D;

  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler3D uCloudVolume3D;
  uniform sampler3D uDetailNoise3D;
  uniform vec3 uSunDirWorld;
  uniform float uDomainXyM;
  uniform float uZBaseM;
  uniform float uZTopM;
  uniform float uMinElevM;
  uniform float uCloudDensityGain;
  uniform float uCirrusDensityGain;
  uniform vec2 uWindAdvectSfc;
  uniform vec2 uWindAdvectJet;

  float altitudeToS(float zAsl) {
    float span = max(1000.0, uZTopM - uZBaseM);
    float zeta = clamp((zAsl - uZBaseM) / span, 0.0, 1.0);
    return clamp((-0.26 + sqrt(0.0676 + 2.96 * zeta)) / 1.48, 0.0, 1.0);
  }

  void main() {
    // Map vUv [0,1]^2 -> world (x = east, z = south = -north) over uDomainXyM
    float wx = (vUv.x - 0.5) * uDomainXyM;
    float wz = (0.5 - vUv.y) * uDomainXyM;
    vec3 origin = vec3(wx, max(0.0, 120.0), wz);
    vec3 lightDir = normalize(vec3(uSunDirWorld.x, max(0.08, abs(uSunDirWorld.y)), uSunDirWorld.z));

    float tau = 0.0;
    const int STEPS = 14;
    for (int i = 0; i < STEPS; i++) {
      float t = (float(i) + 0.5) * 680.0;
      vec3 p = origin + lightDir * t;
      float zAsl = p.y + uMinElevM;
      if (zAsl > uZTopM) break;
      float s = altitudeToS(zAsl);
      vec2 uvFluid = vec2(p.x / uDomainXyM + 0.5, (-p.z) / uDomainXyM + 0.5);
      vec4 fl = texture(uCloudVolume3D, vec3(fract(uvFluid), s));
      float qc = fl.r * fl.r * uCloudDensityGain;
      float qi = fl.g * fl.g * uCirrusDensityGain;
      if (qc + qi > 0.002) {
        vec3 dCoord = vec3(p.x - uWindAdvectSfc.x, p.y, p.z + uWindAdvectSfc.y) * 0.00042;
        float billow = texture(uDetailNoise3D, dCoord).r;
        float ext = max(0.0, qc * (0.45 + 0.85 * billow) * 2.4 + qi * 0.55);
        tau += ext * 0.68;
      }
    }
    float trans = clamp(exp(-tau), 0.08, 1.0);
    fragColor = vec4(trans, clamp(tau * 0.25, 0.0, 1.0), 0.0, 1.0);
  }
`;

// Pass B: Full-Atmosphere (0 m .. 12,200 m ASL) 3D Volumetric Cloud, 1m Micro-Turbulence,
// Sheared 3D Cirrus Uncinus & Terrain-Occluded Crepuscular God-Ray Raymarcher!
const VOLUMETRIC_RAYMARCH_FRAGMENT_GLSL3 = /* glsl */ `
  precision highp float;
  precision highp sampler3D;

  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uSceneDepthTex;
  uniform sampler2D uHeightTex;
  uniform float uHasHeightTex;
  uniform sampler3D uCloudVolume3D;
  uniform sampler3D uDetailNoise3D;

  uniform mat4 uInvViewProj;
  uniform mat4 uViewProj;
  uniform vec3 uCameraPos;
  uniform vec2 uResolution;
  uniform float uFrameIndex;
  uniform float uSimTime;

  uniform float uSpanM;
  uniform float uMinElevM;
  uniform float uDomainXyM;
  uniform float uZBaseM;
  uniform float uZTopM;

  uniform vec3 uSunDirWorld;
  uniform float uSunElDeg;
  uniform vec3 uMoonDirWorld;
  uniform float uMoonElDeg;
  uniform float uMoonIllum;

  uniform vec2 uWindSfc;     // [u_east, v_north] m/s
  uniform vec2 uWind500;     // [u_east, v_north] m/s
  uniform vec2 uWind250;     // [u_east, v_north] m/s (250 hPa Jet Stream for 3D cirrus fallstreak curvature!)

  uniform float uCloudDensityGain;
  uniform float uCirrusDensityGain;
  uniform float uMicroDetailGain;   // 1-meter Kolmogorov cascade intensity (0..2)
  uniform float uGodRayGain;        // Volumetric Tyndall / crepuscular shaft intensity
  uniform float uQualitySteps;      // Base step budget (e.g. 56..96)

  const float PI = 3.14159265359;
  const float EARTH_R_EFF = 7350000.0; // Effective refractive Earth radius (m)

  // Interleaved Gradient Noise (Jimenez 2014) for artifact-free sub-step blue-noise jitter
  float interleavedGradientNoise(vec2 pix) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pix, magic.xy)));
  }

  float hgPhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1e-3, pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
    return (1.0 - g2) / (4.0 * PI * denom);
  }

  // Dual-lobe liquid droplet Mie phase function (strong silver-lining forward peak + glory backscatter)
  float liquidCloudPhase(float cosTheta) {
    return mix(hgPhase(cosTheta, 0.78), hgPhase(cosTheta, -0.28), 0.22);
  }

  // Upper-tropospheric hexagonal ice crystal phase function (sharp forward peak + 22° ice halo ring!)
  float cirrusIcePhase(float cosTheta) {
    float base = mix(hgPhase(cosTheta, 0.85), hgPhase(cosTheta, -0.36), 0.18);
    float angleDeg = degrees(acos(clamp(cosTheta, -1.0, 1.0)));
    float halo22 = 0.085 * exp(-pow((angleDeg - 22.0) / 3.4, 2.0));
    return base + halo22;
  }

  // Exact analytical inverse of Rust s_to_altitude_m stretched vertical grid
  float altitudeToS(float zAsl) {
    float span = max(1000.0, uZTopM - uZBaseM);
    float zeta = clamp((zAsl - uZBaseM) / span, 0.0, 1.0);
    return clamp((-0.26 + sqrt(0.0676 + 2.96 * zeta)) / 1.48, 0.0, 1.0);
  }

  // Samples local terrain elevation worldY (relative to uMinElevM) inside the high-res 9.2km DEM patch
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

  // Evaluates coupled 3D Fluid Macro State + Multi-Scale 1-Meter Detail Cascade ([sigmaLiq, sigmaIce, sigmaHaze])
  // in units of m^-1 extinction coefficient!
  vec3 sampleAtmosphereExtinction(vec3 posWorld, float zAsl, float distToCam, bool highDetail) {
    if (zAsl < uZBaseM - 40.0 || zAsl > uZTopM + 100.0) {
      return vec3(0.0);
    }

    // Terrain boundary clearance over the 1m/15m DEM patch so fog wraps gullies without underground bleed
    float inDem = 0.0;
    float terrY = sampleTerrainLocalY(posWorld.xz, inDem);
    float agl = posWorld.y - terrY;
    if (inDem > 0.5 && agl < -6.0) {
      return vec3(0.0);
    }
    float groundClear = inDem > 0.5 ? smoothstep(-6.0, 14.0, agl) : 1.0;

    // Seamless domain coordinate with quintic C2 texel smoothing across the 48 vertical fluid slices
    // (eliminates linear C0 Mach banding along near-horizontal rays!)
    float s = altitudeToS(zAsl);
    float sTex = s * 47.0 + 0.5;
    float sIdx = floor(sTex);
    float sFrac = sTex - sIdx;
    float sQuintic = sFrac * sFrac * sFrac * (sFrac * (sFrac * 6.0 - 15.0) + 10.0);
    float sSmooth = clamp((sIdx + sQuintic) * (1.0 / 48.0), 0.5 / 48.0, 47.5 / 48.0);

    vec2 fluidUv = fract(vec2(posWorld.x / uDomainXyM + 0.5, (-posWorld.z) / uDomainXyM + 0.5));
    vec2 fTex = fluidUv * 64.0 - 0.5;
    vec2 fIdx = floor(fTex);
    vec2 fFrac = fTex - fIdx;
    vec2 fQuintic = fFrac * fFrac * (3.0 - 2.0 * fFrac);
    vec2 fluidUvSmooth = fract((fIdx + fQuintic + 0.5) * (1.0 / 64.0));

    vec4 fluid = texture(uCloudVolume3D, vec3(fluidUvSmooth, sSmooth));

    // Distance fade only beyond 65 km slant range toward the horizon
    float farHorizonFade = 1.0 - smoothstep(52000.0, 88000.0, distToCam);

    float qcMacro = fluid.r * fluid.r * uCloudDensityGain * groundClear * farHorizonFade;
    float qiMacro = fluid.g * fluid.g * uCirrusDensityGain * farHorizonFade;
    float wTurb = (fluid.b - 0.5) * 2.0; // [-1..+1] updraft & vorticity intensity from Rust Navier-Stokes
    float rhHaze = fluid.a * groundClear * farHorizonFade;

    float sigmaLiq = 0.0;
    if (qcMacro > 0.0015) {
      // Altitude-interpolated wind drift for continuous 60-120 FPS motion between WASM fluid frames
      float midBlend = smoothstep(700.0, 5200.0, zAsl);
      vec2 windLocal = mix(uWindSfc, uWind500, midBlend);
      vec3 pAdv = posWorld - vec3(windLocal.x, 0.25 * wTurb, -windLocal.y) * mod(uSimTime, 900.0) * 0.12;

      // Octave 1 (~380 m meso cauliflower billow) & Octave 2 (~72 m cumulus turret florets)
      vec4 dMeso = texture(uDetailNoise3D, pAdv * (1.0 / 1520.0));
      vec4 dTurret = texture(uDetailNoise3D, pAdv * (1.0 / 288.0) + vec3(0.17, 0.31, 0.09));

      // Updraft cores (wTurb > 0) inflate crisp billowy cauliflower domes; downdrafts shred edges
      float updraftInflate = clamp(0.5 + 0.5 * wTurb, 0.0, 1.0);
      float baseShape = mix(dMeso.b * 0.7 + dMeso.r * 0.3, dMeso.r, 0.45 + 0.50 * updraftInflate);
      float erode = mix(1.0 - dTurret.g, dTurret.b, 0.30 * (1.0 - updraftInflate));

      // Horizon / Nubis density remap: carves sculptural cauliflower clefts into macro fluid density!
      float sculpted = clamp((qcMacro * (0.64 + 0.84 * baseShape) - 0.22 * erode * (1.0 - qcMacro * 0.58)), 0.0, 1.6);

      if (highDetail && sculpted > 0.001) {
        // Octave 3 (~14 m Kelvin-Helmholtz shear curls)
        float lod14m = exp(-distToCam / 5500.0) * uMicroDetailGain;
        if (lod14m > 0.02) {
          vec4 dSub = texture(uDetailNoise3D, pAdv * (1.0 / 56.0) + vec3(dMeso.g * 0.08));
          sculpted = max(0.0, sculpted - (1.0 - dSub.r) * 0.14 * lod14m * (1.0 - sculpted * 0.45));
        }

        // Octave 4: TRUE 1.0-METER KOLMOGOROV (-5/3) MICRO-TURBULENCE CASCADE!
        // uDetailNoise3D has period 16 in channel A over 64 texels -> 1.0 / 16.0 world scale resolves
        // exact 1.0-meter turbulent wisps, sea-fog tendrils, and cumulus boundary-layer eddies!
        float lod1m = exp(-distToCam / 780.0) * uMicroDetailGain;
        if (lod1m > 0.025) {
          vec4 d1m = texture(uDetailNoise3D, pAdv * (1.0 / 16.0) + vec3(dTurret.a * 0.12));
          float wisp1m = mix(d1m.a, d1m.g, 0.5) - 0.48;
          sculpted = max(0.0, sculpted + wisp1m * 0.18 * lod1m * sqrt(sculpted));
        }
      }

      // Physical liquid cloud extinction sigma_ext (m^-1): dense marine fog / cumulus ~ 0.015 - 0.055 m^-1
      sigmaLiq = sculpted * 0.028;
    }

    // 2. Upper-Tropospheric 3D Cirrus Uncinus & Fibratus (6,600 m .. 12,200 m ASL)
    float sigmaIce = 0.0;
    if (qiMacro > 0.02 && zAsl >= 6300.0) {
      // Gravitational ice-crystal fallstreak curvature across 250hPa -> 500hPa vertical wind shear!
      // Generating tufts sit near 10.6 km; falling ice crystals trail downwind/upwind along the shear vector!
      float fallDepthKm = clamp((10800.0 - zAsl) * 0.001, -1.2, 4.2);
      vec2 shearVec = (uWind250 - uWind500) * 44.0; // meters of horizontal trail offset per km of fall
      // Parabolic comma hook (cirrus uncinus mares' tail geometry in 3D!)
      vec2 hookOffset = shearVec * (fallDepthKm * 0.55 + 0.24 * fallDepthKm * fallDepthKm);

      vec2 jetDir = normalize(uWind250 + vec2(0.001));
      vec2 crossDir = vec2(-jetDir.y, jetDir.x);

      vec2 pHoriz = posWorld.xz + vec2(hookOffset.x, -hookOffset.y)
        - vec2(uWind250.x, -uWind250.y) * mod(uSimTime, 1200.0) * 0.08;

      // Anisotropic jet-aligned coordinates: stretched 5.5x along the upper-level jet stream!
      float alongJet = dot(vec2(pHoriz.x, -pHoriz.y), jetDir);
      float acrossJet = dot(vec2(pHoriz.x, -pHoriz.y), crossDir);
      vec3 ciCoord = vec3(alongJet / 5600.0, acrossJet / 1080.0, zAsl / 1400.0);

      vec4 ciTex = texture(uDetailNoise3D, ciCoord);
      vec4 ciFine = texture(uDetailNoise3D, ciCoord * 3.9 + vec3(ciTex.r * 0.15));

      // Combine fibrous B channel + cellular generating-head tufts near 10.2-11.4 km
      float headWeight = smoothstep(9400.0, 10800.0, zAsl);
      float filaments = mix(ciTex.b * (0.50 + 0.72 * ciFine.b), ciTex.g, 0.42 * headWeight);

      // HighDetail 1-meter ice-crystal hair streaks when viewing high altitude / telephoto zoom
      if (highDetail) {
        float ciMicro = texture(uDetailNoise3D, vec3(alongJet / 95.0, acrossJet / 24.0, zAsl / 40.0)).b;
        filaments *= (0.76 + 0.48 * ciMicro);
      }

      // Crisp filament carving preserves deep cobalt/golden/twilight sky lanes between 3D mares' tails!
      float iceSculpt = qiMacro * smoothstep(0.05, 0.52, qiMacro) * (0.24 + 1.38 * smoothstep(0.28, 0.84, filaments));
      sigmaIce = iceSculpt * 0.0044;
    }

    // 3. Hygroscopic Sub-Cloud & Valley Mist Aerosol Extinction (drives 3D Crepuscular God-Rays!)
    float lowAirWeight = exp(-max(0.0, zAsl - uZBaseM) / 2300.0);
    float sigmaHaze = (0.0000045 + 0.000078 * rhHaze * lowAirWeight) * uGodRayGain * farHorizonFade;

    return vec3(sigmaLiq, sigmaIce, sigmaHaze);
  }

  // Evaluates 3D Terrain Shadowing along lightDir against uHeightTex (creates real mountain-notch light shafts!)
  float sampleTerrainSunVisibility(vec3 posWorld, vec3 lightDir) {
    if (uHasHeightTex < 0.5 || lightDir.y <= -0.04) return 1.0;
    float halfSpan = uSpanM * 0.5;
    if (abs(posWorld.x) > halfSpan * 1.35 || abs(posWorld.z) > halfSpan * 1.35) return 1.0;

    // 5 fast logarithmic shadow-march taps across the DEM toward the Sun/Moon
    const float dists[5] = float[5](140.0, 420.0, 1050.0, 2300.0, 4600.0);
    float vis = 1.0;
    for (int i = 0; i < 5; i++) {
      vec3 sp = posWorld + lightDir * dists[i];
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

  // 4-step volumetric self-shadowing optical depth toward lightDir
  float marchLightOpticalDepth(vec3 posWorld, float zAsl, vec3 lightDir) {
    const float stepsM[4] = float[4](95.0, 310.0, 880.0, 2400.0);
    const float dWeights[4] = float[4](130.0, 290.0, 780.0, 1800.0);
    float tau = 0.0;
    for (int i = 0; i < 4; i++) {
      vec3 lp = posWorld + lightDir * stepsM[i];
      float lz = zAsl + lightDir.y * stepsM[i];
      vec3 sig = sampleAtmosphereExtinction(lp, lz, 8000.0, false);
      tau += (sig.x + sig.y * 0.85) * dWeights[i];
    }
    return tau;
  }

  // Computes altitude-dependent direct solar color & intensity!
  // Crucial real physics: at z = 10.5 km (3D cirrus), the geometric horizon dip is +3.28 deg,
  // so high cirrus blazes crimson/coral for 20+ minutes AFTER low marine fog enters twilight shadow!
  vec3 computeDirectSolarIrradiance(float zAsl, out float localSunElDeg) {
    float dipDeg = 1.015 * sqrt(max(0.0, zAsl * 0.001));
    localSunElDeg = uSunElDeg + dipDeg;

    if (localSunElDeg <= -2.6) {
      return vec3(0.0);
    }

    // Atmospheric airmass extinction along the slant solar path at altitude zAsl
    float pressureRatio = exp(-max(0.0, zAsl) / 8200.0);
    float sinEl = sin(radians(max(localSunElDeg, 0.10)));
    float airMass = clamp(pressureRatio / (sinEl + 0.11 * pow(max(0.0, localSunElDeg + 3.2), -0.75)), 0.15, 28.0);

    // Wavelength-dependent Rayleigh + Chappuis ozone optical depth [R, G, B]
    vec3 betaExt = vec3(0.038, 0.128, 0.275);
    vec3 sunCol = exp(-betaExt * airMass) * 3.35;

    // Enrich crimson/coral Chappuis grazing glow when Sun grazes the curved Earth limb (-1.5 .. +3.5 deg)
    float crimsonGlow = smoothstep(5.5, -0.4, localSunElDeg);
    sunCol *= mix(vec3(1.0), vec3(1.32, 0.80, 0.65), crimsonGlow);

    // Smooth horizon cut right as the solar disk sets below the curved Earth limb
    float limbVis = smoothstep(-2.1, 0.45, localSunElDeg);
    return sunCol * limbVis;
  }

  vec3 computeAmbientSkyIrradiance(float zAsl) {
    float dayT = smoothstep(-4.0, 12.0, uSunElDeg);
    float goldT = smoothstep(-6.5, -0.5, uSunElDeg) * (1.0 - smoothstep(3.5, 11.0, uSunElDeg));
    float nightT = 1.0 - smoothstep(-9.0, -2.0, uSunElDeg);

    vec3 dayAmb = vec3(0.36, 0.47, 0.66) * (0.85 + 0.25 * clamp(zAsl / 10000.0, 0.0, 1.0));
    vec3 duskAmb = vec3(0.24, 0.20, 0.36) + vec3(0.22, 0.09, 0.10) * clamp(zAsl / 9000.0, 0.0, 1.2);
    vec3 nightAmb = vec3(0.012, 0.020, 0.042) + vec3(0.025, 0.038, 0.068) * uMoonIllum * smoothstep(-2.0, 14.0, uMoonElDeg);

    vec3 amb = mix(duskAmb, dayAmb, dayT);
    amb = mix(amb, vec3(0.28, 0.22, 0.38), goldT * 0.55);
    amb = mix(amb, nightAmb, nightT);
    return amb;
  }

  void main() {
    float rawDepth = texture(uSceneDepthTex, vUv).r;
    vec4 ndcNear = vec4(vUv * 2.0 - 1.0, -1.0, 1.0);
    vec4 ndcFar = vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
    vec4 wNear = uInvViewProj * ndcNear;
    vec4 wFar = uInvViewProj * ndcFar;
    wNear /= wNear.w;
    wFar /= wFar.w;

    vec3 rayOrigin = uCameraPos;
    vec3 rayDir = normalize(wFar.xyz - rayOrigin);

    bool hitTerrain = rawDepth < 0.99998;
    float maxDist = 86000.0;
    if (hitTerrain) {
      vec4 ndcHit = vec4(vUv * 2.0 - 1.0, rawDepth * 2.0 - 1.0, 1.0);
      vec4 wHit = uInvViewProj * ndcHit;
      wHit /= wHit.w;
      maxDist = length(wHit.xyz - rayOrigin);
    } else {
      // Sky ray: if looking steeply upward, exit cleanly at the tropopause uZTopM
      float camZAsl = rayOrigin.y + uMinElevM;
      if (rayDir.y > 0.02 && camZAsl < uZTopM) {
        float distToTop = (uZTopM + 350.0 - camZAsl) / rayDir.y;
        maxDist = min(maxDist, distToTop * 1.12);
      } else if (rayDir.y < -0.12 && camZAsl < uZBaseM) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
      }
    }

    maxDist = clamp(maxDist, 1.0, 88000.0);

    // Choose primary directional light (Sun during day/twilight, Moon during nightscape when Moon is up!)
    bool useLunarLight = (uSunElDeg < -6.2) && (uMoonElDeg > 0.2) && (uMoonIllum > 0.05);
    vec3 primaryLightDir = useLunarLight ? normalize(uMoonDirWorld) : normalize(uSunDirWorld);
    float cosTheta = clamp(dot(rayDir, primaryLightDir), -1.0, 1.0);

    float phaseLiq = liquidCloudPhase(cosTheta);
    float phaseIce = cirrusIcePhase(cosTheta);
    float phaseHaze = mix(hgPhase(cosTheta, 0.68), 1.0 / (4.0 * PI), 0.35);

    // Blue-noise / IGN sub-step temporal-spatial jitter for silky-smooth 1m clouds with zero banding
    float jitter = interleavedGradientNoise(gl_FragCoord.xy + vec2(mod(uFrameIndex, 16.0) * 5.588238));

    int stepCount = int(clamp(uQualitySteps, 36.0, 104.0));
    // Non-linear ray parameterization t(u) = maxDist * pow(u, gamma) concentrates sub-meter/meter
    // precision right near the camera / tripod / mountain ridge while reaching 85 km upper cirrus!
    float gamma = maxDist > 16000.0 ? 1.85 : 1.32;

    vec3 accumLight = vec3(0.0);
    float transmittance = 1.0;

    float prevT = 0.0;
    for (int i = 0; i < 104; i++) {
      if (i >= stepCount) break;
      float u0 = float(i) / float(stepCount);
      float u1 = float(i + 1) / float(stepCount);
      float um = mix(u0, u1, jitter);

      float tEnd = maxDist * pow(u1, gamma);
      float tMid = maxDist * pow(um, gamma);
      float dt = max(0.25, tEnd - prevT);
      prevT = tEnd;

      vec3 pos = rayOrigin + rayDir * tMid;
      float horizDist2 = dot(pos.xz - rayOrigin.xz, pos.xz - rayOrigin.xz);
      // Earth curvature drop makes distant 3D cirrus & marine stratus follow the planetary horizon!
      float curvDrop = horizDist2 / (2.0 * EARTH_R_EFF);
      float zAsl = pos.y + uMinElevM - curvDrop;

      if (zAsl > uZTopM + 150.0) {
        if (rayDir.y > 0.05) break;
        continue;
      }
      if (zAsl < uZBaseM - 80.0) {
        continue;
      }

      vec3 sigmas = sampleAtmosphereExtinction(pos, zAsl, tMid, true);
      float sigmaCloud = sigmas.x + sigmas.y;
      float sigmaTotal = sigmaCloud + sigmas.z;

      if (sigmaTotal <= 1.0e-6) {
        continue;
      }

      float localSunEl = 0.0;
      vec3 directIrr = useLunarLight
        ? vec3(0.22, 0.28, 0.42) * uMoonIllum * smoothstep(0.0, 8.0, uMoonElDeg)
        : computeDirectSolarIrradiance(zAsl, localSunEl);

      vec3 ambientIrr = computeAmbientSkyIrradiance(zAsl);

      // Evaluate terrain notch visibility & cloud self-shadowing when direct light is active
      float terrainVis = 1.0;
      float cloudTauSun = 0.0;
      if (dot(directIrr, directIrr) > 1.0e-5) {
        terrainVis = sampleTerrainSunVisibility(pos, primaryLightDir);
        if (terrainVis > 0.01) {
          cloudTauSun = marchLightOpticalDepth(pos, zAsl, primaryLightDir);
        }
      }

      vec3 stepScatter = vec3(0.0);

      // A) 3D Cloud & Cirrus Radiative Transfer (3-Octave Nubis Multi-Scattering + Beer-Powder)
      if (sigmaCloud > 1.0e-5) {
        float iceFrac = sigmas.y / max(1.0e-6, sigmaCloud);
        float phaseCloud = mix(phaseLiq, phaseIce, iceFrac);

        // 3-octave multi-scattering approximation (a = 0.52, b = 0.46) gives luminous cloud interiors!
        float ms0 = exp(-cloudTauSun) * phaseCloud;
        float ms1 = 0.54 * exp(-cloudTauSun * 0.44) * mix(phaseCloud, 1.0 / (4.0 * PI), 0.55);
        float ms2 = 0.29 * exp(-cloudTauSun * 0.18) * (1.0 / (4.0 * PI));
        float multiScatter = ms0 + ms1 + ms2;

        // Beer-Powder edge enhancement for crisp sunlit cumulus & stratocumulus rims
        float powder = mix(1.0 - 0.55 * exp(-sigmaCloud * dt * 2.6), 1.0, max(0.0, cosTheta) * 0.65 + iceFrac * 0.6);

        // Height-dependent ambient occlusion inside cloud decks
        float verticalBounce = 0.55 + 0.45 * clamp((zAsl - uZBaseM) / 1800.0, 0.0, 1.0);
        vec3 cloudLum = directIrr * terrainVis * multiScatter * powder * 3.4
          + ambientIrr * verticalBounce * mix(0.85, 1.22, iceFrac);

        stepScatter += cloudLum * sigmaCloud;
      }

      // B) Continuous 3D Volumetric Crepuscular God-Rays (Hygroscopic Haze + Mountain/Cloud Keyholes!)
      if (sigmas.z > 1.0e-7) {
        float shaftShadow = terrainVis * exp(-cloudTauSun * 0.92);
        vec3 shaftLum = directIrr * shaftShadow * phaseHaze * 4.8 + ambientIrr * 0.24;
        stepScatter += shaftLum * sigmas.z;
      }

      // Exact analytical energy-conserving step integration ((1 - exp(-sigma * dt)) / sigma)
      float stepTrans = exp(-sigmaTotal * dt);
      vec3 sint = stepScatter * ((1.0 - stepTrans) / max(1.0e-6, sigmaTotal));
      accumLight += transmittance * sint;
      transmittance *= stepTrans;

      if (transmittance < 0.012) {
        transmittance = 0.0;
        break;
      }
    }

    fragColor = vec4(accumLight, transmittance);
  }
`;

// Pass C: Full-Retina Depth-Aware Bilateral Upsampler + ACES Tone-Mapped Composite
// Preserves 1-pixel Retina mountain ridge edges against clouds & fog AND writes `gl_FragDepth`!
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

  // Soft ACES-inspired highlight compression so blazing golden-hour sun/cirrus never clips harshly
  vec3 softHighlightCompress(vec3 col) {
    vec3 x = max(vec3(0.0), col);
    // Blend linear below 0.75 into smooth filmic shoulder above 0.75
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

    // 9-tap 3x3 tent-bilateral depth-aware upsample from uVolumetricTex (melts half-res IGN grain cleanly)
    const vec2 offsets[9] = vec2[9](
      vec2( 0.0,  0.0),
      vec2(-1.0,  0.0),
      vec2( 1.0,  0.0),
      vec2( 0.0, -1.0),
      vec2( 0.0,  1.0),
      vec2(-1.0, -1.0),
      vec2( 1.0, -1.0),
      vec2(-1.0,  1.0),
      vec2( 1.0,  1.0)
    );
    const float spatialW[9] = float[9](0.25, 0.125, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625);

    vec4 accumVol = vec4(0.0);
    float sumW = 0.0;

    for (int i = 0; i < 9; i++) {
      vec2 tapUv = clamp(vUv + offsets[i] * uLowResTexel, vec2(0.001), vec2(0.999));
      float tapDepth = texture(uSceneDepthTex, tapUv).r;
      float tapLin = linearizeDepth(tapDepth);
      float relDiff = abs(centerLin - tapLin) / max(25.0, centerLin * 0.06);
      float w = spatialW[i] * exp(-relDiff * relDiff * 3.5);
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
    this.evolutionSpeed = 1.0; // 1.0 = smooth real-time speed, up to 8x time-lapse!
    this.activePreset = 'live';
    this.qualityMode = 'ultra'; // 'performance' | 'balanced' | 'ultra'
    this.resScale = 0.76;

    this.liveWeatherParams = null;
    this.userTuning = {
      cloudDensityGain: 1.0,
      cirrusDensityGain: 1.05,
      microDetailGain: 1.0,
      godRayGain: 1.0,
    };

    this.telemetry = {
      wasmReady: false,
      stepMs: 0,
      fps: 60,
      frameMs: 16.6,
      lclAltitudeM: 380,
      freezingLevelM: 3200,
      glaciationCirrusBaseM: 7600,
      maxUpdraftMs: 0.0,
      maxLiquidGKg: 0.0,
      maxIceGKg: 0.0,
      lowCloudCoverage: 0,
      midCloudCoverage: 0,
      highCirrusCoverage: 0,
      froudeNumber: 0.6,
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
    this.simClockS = 0;
    this.frameIndex = 0;

    this._init3DTextures();
    this._initRenderTargetsAndPasses();
    this._initWorker();
  }

  _init3DTextures() {
    // 1. 64 x 64 x 48 RGBA8 3D Fluid & Cirrus Volume Texture
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

    // 2. 64 x 64 x 64 RGBA8 Tileable 3D Perlin-Worley + Curl-Filament Detail Volume
    const noiseData = new Uint8Array(64 * 64 * 64 * 4);
    for (let i = 0; i < noiseData.length; i++) noiseData[i] = 128;
    this.detailNoiseTex = new THREE.Data3DTexture(noiseData, 64, 64, 64);
    this.detailNoiseTex.format = THREE.RGBAFormat;
    this.detailNoiseTex.type = THREE.UnsignedByteType;
    this.detailNoiseTex.minFilter = THREE.LinearFilter;
    this.detailNoiseTex.magFilter = THREE.LinearFilter;
    this.detailNoiseTex.wrapS = THREE.RepeatWrapping;
    this.detailNoiseTex.wrapT = THREE.RepeatWrapping;
    this.detailNoiseTex.wrapR = THREE.RepeatWrapping;
    this.detailNoiseTex.unpackAlignment = 1;
    this.detailNoiseTex.needsUpdate = true;

    this.dummyHeightTex = new THREE.DataTexture(new Float32Array([0]), 1, 1, THREE.RedFormat, THREE.FloatType);
    this.dummyHeightTex.needsUpdate = true;
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

    // 2. HDR Volumetric Raymarch RenderTarget
    const vw = Math.max(2, Math.round(w * this.resScale));
    const vh = Math.max(2, Math.round(h * this.resScale));
    this.volumetricRT = new THREE.WebGLRenderTarget(vw, vh, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
    });

    // 3. 128x128 Ground Cloud-Shadow Transmittance LUT (`shadowMapRT`)
    this.shadowMapRT = new THREE.WebGLRenderTarget(128, 128, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: false,
    });

    this.orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const fsGeo = new THREE.PlaneGeometry(2, 2);

    this.shadowMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERTEX_GLSL3,
      fragmentShader: CLOUD_SHADOW_MAP_FRAGMENT_GLSL3,
      uniforms: {
        uCloudVolume3D: { value: this.cloudVolumeTex },
        uDetailNoise3D: { value: this.detailNoiseTex },
        uSunDirWorld: { value: new THREE.Vector3(-0.85, 0.15, 0.2).normalize() },
        uDomainXyM: { value: 18000.0 },
        uZBaseM: { value: 0.0 },
        uZTopM: { value: 12200.0 },
        uMinElevM: { value: 0.0 },
        uCloudDensityGain: { value: 1.0 },
        uCirrusDensityGain: { value: 1.05 },
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
        uCloudVolume3D: { value: this.cloudVolumeTex },
        uDetailNoise3D: { value: this.detailNoiseTex },
        uInvViewProj: { value: new THREE.Matrix4() },
        uViewProj: { value: new THREE.Matrix4() },
        uCameraPos: { value: new THREE.Vector3() },
        uResolution: { value: new THREE.Vector2(vw, vh) },
        uFrameIndex: { value: 0.0 },
        uSimTime: { value: 0.0 },
        uSpanM: { value: 9200.0 },
        uMinElevM: { value: 0.0 },
        uDomainXyM: { value: 18000.0 },
        uZBaseM: { value: 0.0 },
        uZTopM: { value: 12200.0 },
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
        uQualitySteps: { value: 72.0 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.volumetricScene = new THREE.Scene();
    this.volumetricScene.add(new THREE.Mesh(fsGeo, this.volumetricMat));

    this.compositeMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERTEX_GLSL3,
      fragmentShader: COMPOSITE_FRAGMENT_GLSL3,
      uniforms: {
        uSceneColorTex: { value: this.sceneRT.texture },
        uSceneDepthTex: { value: this.sceneRT.depthTexture },
        uVolumetricTex: { value: this.volumetricRT.texture },
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
    this._tmpDrawSize = new THREE.Vector2();
  }

  _initWorker() {
    try {
      this.worker = new Worker('./sim-worker.js?v=20261114d');
    } catch (err) {
      console.warn('Could not spawn sim-worker.js:', err);
      return;
    }

    this.worker.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg) return;

      if (msg.type === 'ready') {
        this.telemetry.wasmReady = true;
        const detailBytes = new Uint8Array(msg.detailNoiseBuffer);
        if (detailBytes.length === 64 * 64 * 64 * 4) {
          this.detailNoiseTex.image.data.set(detailBytes);
          this.detailNoiseTex.needsUpdate = true;
        }
        this._applyWorkerFrame(msg);
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
        this._applyWorkerFrame(msg);
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
      }
    }

    if (msg.telemetry) {
      Object.assign(this.telemetry, msg.telemetry);
      this.telemetry.stepMs = msg.stepMs ?? this.telemetry.stepMs;
      const u = this.volumetricMat.uniforms;
      u.uZBaseM.value = this.telemetry.zBaseM;
      u.uZTopM.value = this.telemetry.zTopM;
      u.uDomainXyM.value = this.telemetry.domainXyM;
      this.shadowMat.uniforms.uZBaseM.value = this.telemetry.zBaseM;
      this.shadowMat.uniforms.uZTopM.value = this.telemetry.zTopM;
      this.shadowMat.uniforms.uDomainXyM.value = this.telemetry.domainXyM;
    }

    if (this.onTelemetryCallback) {
      this.onTelemetryCallback(this.telemetry);
    }
  }

  getShadowMapTexture() {
    return this.shadowMapRT.texture;
  }

  setQualityMode(mode) {
    this.qualityMode = mode;
    if (mode === 'performance') {
      this.resScale = 0.56;
      this.volumetricMat.uniforms.uQualitySteps.value = 46.0;
    } else if (mode === 'balanced') {
      this.resScale = 0.68;
      this.volumetricMat.uniforms.uQualitySteps.value = 62.0;
    } else {
      // 'ultra' (1m High-Precision Default)
      this.resScale = 0.82;
      this.volumetricMat.uniforms.uQualitySteps.value = 84.0;
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
  }

  setTerrainDem(elevationsFloat32, gridSize, spanM, minElevM, centerLat, centerLon, heightTex) {
    const u = this.volumetricMat.uniforms;
    u.uSpanM.value = spanM;
    u.uMinElevM.value = minElevM;
    this.shadowMat.uniforms.uMinElevM.value = minElevM;
    if (heightTex) {
      u.uHeightTex.value = heightTex;
      u.uHasHeightTex.value = 1.0;
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

  setLiveWeatherReport(weather) {
    if (!weather) return;
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
    const p = this.getEffectiveParams();
    const u = this.volumetricMat.uniforms;
    u.uWindSfc.value.set(p.windSfcU ?? 5.2, p.windSfcV ?? -3.4);
    u.uWind500.value.set(p.wind500U ?? 18.0, p.wind500V ?? 4.5);
    u.uWind250.value.set(p.wind250U ?? 36.0, p.wind250V ?? 14.0);

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
      u.uSunDirWorld.value.copy(sunDirWorld);
      this.shadowMat.uniforms.uSunDirWorld.value.copy(sunDirWorld);
    }
    if (typeof sunElDeg === 'number') {
      u.uSunElDeg.value = sunElDeg;
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

  raycastTripodWasm(tripodLat, tripodLon, targetLat, targetLon, targetZM) {
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
        resolve({ data: floatArr, elapsedMs });
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
    this.volumetricMat.uniforms.uResolution.value.set(vw, vh);
    this.compositeMat.uniforms.uLowResTexel.value.set(1.0 / vw, 1.0 / vh);
  }

  /**
   * Executes the 3-stage GPU volumetric atmosphere + terrain + overlay pipeline:
   * 1. Steps Rust WASM 3D Navier-Stokes & Cirrus solver asynchronously (~16 Hz, 0 ms main-thread cost)
   * 2. Renders `worldScene` (`skyDome` + `terrainGroup` + `trailsGroup`) -> `sceneRT` (+ `DepthTexture`)
   * 3. Renders `volumetricScene` (0..12.2 km 3D Cloud + 1m Detail + 3D Cirrus + God-Ray Raymarcher) -> `volumetricRT`
   * 4. Bilateral Depth-Aware Composites `sceneRT` + `volumetricRT` -> Default Screen Framebuffer (writing `gl_FragDepth`)
   * 5. Renders crisp 3D HUD Vectors & Markers (`overlayScene`) with hardware depth occlusion!
   */
  renderPipeline(worldScene, overlayScene, dtRealS, showAtmosphere) {
    this.frameIndex = (this.frameIndex + 1) % 4096;
    const clampedDt = Math.min(0.1, Math.max(0.001, dtRealS));

    if (this.evolveFluid && showAtmosphere && !document.hidden) {
      this.simClockS += clampedDt * 18.0 * this.evolutionSpeed;
      const nowMs = performance.now();
      // Request next Rust WASM 3D Navier-Stokes step at ~14 Hz when previous frame has arrived (zero queue lag!)
      if (
        this.worker &&
        this.telemetry.wasmReady &&
        !this.workerBusy &&
        nowMs - this.lastWorkerStepWallMs >= 68
      ) {
        this.workerBusy = true;
        this.lastWorkerStepWallMs = nowMs;
        this.worker.postMessage({
          type: 'step',
          dt: 3.2 * this.evolutionSpeed,
        });
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

    this.camera.updateMatrixWorld();
    this._tmpViewProj.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this._tmpInvViewProj.copy(this._tmpViewProj).invert();

    const u = this.volumetricMat.uniforms;
    u.uViewProj.value.copy(this._tmpViewProj);
    u.uInvViewProj.value.copy(this._tmpInvViewProj);
    u.uCameraPos.value.copy(this.camera.position);
    u.uFrameIndex.value = this.frameIndex;
    u.uSimTime.value = this.simClockS;

    const sfcAdv = modSimOffset(u.uWindSfc.value, this.simClockS * 0.12);
    this.shadowMat.uniforms.uWindAdvectSfc.value.set(sfcAdv.x, sfcAdv.y);

    // 0. Update 128x128 3D Cloud Shadow Map (`0.03 ms`) for terrain ground shadows
    if (showAtmosphere) {
      this.renderer.setRenderTarget(this.shadowMapRT);
      this.renderer.clear();
      this.renderer.render(this.shadowScene, this.orthoCam);
    }

    // 1. Opaque World + Trails -> `sceneRT` (captures exact 1m terrain `DepthTexture`)
    this.renderer.setRenderTarget(this.sceneRT);
    this.renderer.clear(true, true, false);
    this.renderer.render(worldScene, this.camera);

    // 2. Full-Atmosphere 3D Volumetric Raymarcher -> `volumetricRT`
    if (showAtmosphere) {
      this.renderer.setRenderTarget(this.volumetricRT);
      this.renderer.clear(true, false, false);
      this.renderer.render(this.volumetricScene, this.orthoCam);
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
