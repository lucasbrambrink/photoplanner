import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VolumetricAtmosphereEngine, WEATHER_REGIME_PRESETS } from "./volumetric-engine.js?v=20261114d";

// Conditional Popover API polyfill loader (content-first-noise-reduction)
if (!Object.hasOwn(HTMLElement.prototype, "popover")) {
  import("https://esm.sh/@oddbird/popover-polyfill").then((m) => m.apply?.());
}

// ============================================================================
// Application State (Default to SF Coast / Montara Mountain on Today's Sunset, persisted in localStorage)
// ============================================================================
const savedLandmark = localStorage.getItem("pp_landmark_v2") || "Pedro Point Headlands";
const savedDate = localStorage.getItem("pp_date_v2") || "2026-04-16";

const state = {
  landmark: savedLandmark,
  date: savedDate,
  maxMiles: 16.0,
  weatherEngine: "ensemble", // "ensemble" | "graphcast" | "hrrr" | "ecmwf" | "nowcast"
  shootReport: null,
  terrainPatch: null,
  weatherReport: null,
  customTripodReport: null,
  currentMinutes: 17 * 60 + 18,
  selectedVpIndex: 0, // 0..2 when a VP is expanded, -1 when collapsed or inspecting custom pin
  cameraMode: "orbit", // "orbit" | "vp-0" | "vp-1" | "vp-2" | "custom-tripod"
  clickMode: "tripod", // "tripod" | "target"
  surfaceStyle: 0, // 0 = Granite Relief, 1 = USGS Satellite, 2 = Topo Contours
  showViewshed: false,
  isPlayingSun: false,
  showSunBeams: false,
  showLightField: false,
  showAtmosphere: true,
  showSightlines: true,
  highlightLandingZone: false,
};

const VP_COLORS = [0x38bdf8, 0xa855f7, 0x22c55e];
const VP_HEX = ["#38bdf8", "#a855f7", "#22c55e"];

// ============================================================================
// DOM References & Slider Progress Sync (modern-web-guidance: slider)
// ============================================================================
const timeScrubber = document.getElementById("time-scrubber");
const timeOutput = document.getElementById("time-scrubber-output");
const loadingEl = document.getElementById("viewport-loading");
const loadingTextEl = document.getElementById("loading-text");
const vfOverlay = document.getElementById("viewfinder-overlay");
const vfHudReadout = document.getElementById("vf-hud-readout");
const cursorCoordsBar = document.getElementById("cursor-coords-bar");
const inspectorCard = document.getElementById("tripod-inspector-card");
const weatherDisclosure = document.getElementById("weather-disclosure");
const btnOrbitMode = document.getElementById("btn-orbit-mode");
const viewportStageEl = document.getElementById("viewport-stage");
const hudClickModeGroup = document.getElementById("hud-click-mode-group");

function syncSliderValue(input) {
  const field = input.closest(".slider-field");
  if (!field) return;
  field.style.setProperty("--attr-min", input.min || "0");
  field.style.setProperty("--attr-max", input.max || "100");
  field.style.setProperty("--control-value", input.value);
}

function formatMinutesToHHMM(totalMinutes) {
  const mins = Math.round(totalMinutes);
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function parseHHMMToMinutes(hhmm) {
  if (!hhmm || !hhmm.includes(":")) return 17 * 60 + 18;
  const [h, m] = hhmm.split(":").map((n) => parseInt(n, 10));
  return h * 60 + m;
}

function compassLabel(azDeg) {
  const dirs = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
  ];
  const idx = Math.round((((azDeg % 360) + 360) % 360) / 22.5) % 16;
  return dirs[idx];
}

// ============================================================================
// Three.js Dual-Scene GPU Volumetric Pipeline, Sky Dome, Camera & Controls
// ============================================================================
const container = document.getElementById("canvas-container");
const worldScene = new THREE.Scene();
const scene = worldScene;
worldScene.background = new THREE.Color(0x060a13);
worldScene.fog = new THREE.FogExp2(0x0b1930, 0.000009);

const overlayScene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  48,
  container.clientWidth / Math.max(container.clientHeight, 1),
  2.0,
  135000
);
camera.position.set(-3400, 2500, -2900);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
  stencil: false,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 2, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxDistance = 68000;
controls.minDistance = 2.5;
controls.target.set(0, 900, 0);

// World scene groups (rendered into `sceneRT` with hardware `DepthTexture` BEFORE 3D Volumetric Atmosphere)
const atmosphereGroup = new THREE.Group();
const terrainGroup = new THREE.Group();
const trailsGroup = new THREE.Group();
worldScene.add(atmosphereGroup);
worldScene.add(terrainGroup);
worldScene.add(trailsGroup);

// Crisp 3D HUD & analytical vector overlays (rendered AFTER depth-aware volumetric composite)
const lightFieldGroup = new THREE.Group();
const sunBeamsGroup = new THREE.Group();
const customPinGroup = new THREE.Group();
overlayScene.add(lightFieldGroup);
overlayScene.add(sunBeamsGroup);
overlayScene.add(customPinGroup);

// Initialize GPU WebGL2 Volumetric Atmosphere + Rust WASM 3D Navier-Stokes & 3D Cirrus Engine
const volumetricEngine = new VolumetricAtmosphereEngine(renderer, camera);
volumetricEngine.setQualityMode("ultra");

// ============================================================================
// Physical Atmospheric Sky Dome (with Limb-Darkened Solar & Lunar Disks)
// Rendered behind 3D clouds & terrain so clouds naturally attenuate & silver-line the Sun!
// ============================================================================
const SKY_DOME_VERTEX = /* glsl */ `
  varying vec3 vWorldDir;
  void main() {
    vWorldDir = normalize(position);
    vec4 wp = modelMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const SKY_DOME_FRAGMENT = /* glsl */ `
  precision highp float;
  varying vec3 vWorldDir;
  uniform vec3 uSunDirWorld;
  uniform float uSunElDeg;
  uniform vec3 uMoonDirWorld;
  uniform float uMoonElDeg;
  uniform float uMoonIllum;

  float starHash(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
  }

  void main() {
    vec3 dir = normalize(vWorldDir);
    float y = clamp(dir.y, -0.15, 1.0);
    float horizonFactor = pow(1.0 - clamp(y, 0.0, 1.0), 3.2);
    vec3 sunDir = normalize(uSunDirWorld);
    float sunCos = clamp(dot(dir, sunDir), -1.0, 1.0);
    float antiSunCos = clamp(dot(dir, -sunDir), 0.0, 1.0);

    float tBlueHour = smoothstep(-8.5, -1.5, uSunElDeg) * (1.0 - smoothstep(1.0, 7.5, uSunElDeg));
    float tGolden = smoothstep(-1.5, 3.0, uSunElDeg) * (1.0 - smoothstep(9.0, 18.0, uSunElDeg));
    float tDay = smoothstep(8.0, 20.0, uSunElDeg);
    float tNight = 1.0 - smoothstep(-8.0, -1.0, uSunElDeg);

    vec3 zenithNight = vec3(0.015, 0.030, 0.072);
    vec3 zenithBlueHour = vec3(0.045, 0.145, 0.35);
    vec3 zenithGolden = vec3(0.085, 0.24, 0.50);
    vec3 zenithDay = vec3(0.13, 0.36, 0.70);

    vec3 horizNight = vec3(0.04, 0.075, 0.15);
    vec3 horizBlueHour = vec3(0.15, 0.28, 0.52);
    vec3 horizGolden = vec3(0.34, 0.43, 0.60);
    vec3 horizDay = vec3(0.46, 0.65, 0.86);

    vec3 zenithCol = zenithNight * tNight + zenithBlueHour * tBlueHour + zenithGolden * tGolden + zenithDay * tDay;
    vec3 horizCol = horizNight * tNight + horizBlueHour * tBlueHour + horizGolden * tGolden + horizDay * tDay;
    float weightSum = max(tNight + tBlueHour + tGolden + tDay, 0.001);
    zenithCol /= weightSum;
    horizCol /= weightSum;

    vec3 skyCol = mix(zenithCol, horizCol, horizonFactor);

    // Belt of Venus & Earth's twilight shadow wedge
    float beltBand = exp(-pow((y - 0.09) / 0.085, 2.0)) * pow(antiSunCos, 1.4);
    skyCol = mix(skyCol, vec3(0.68, 0.42, 0.66), beltBand * (tBlueHour * 0.65 + tGolden * 0.35));

    float earthShadowBand = smoothstep(0.08, -0.02, y) * pow(antiSunCos, 1.2);
    skyCol = mix(skyCol, vec3(0.045, 0.095, 0.22), earthShadowBand * (tBlueHour * 0.72 + tGolden * 0.45));

    // Circumsolar Mie forward scattering & golden-crimson twilight glow
    float sunGlowVis = smoothstep(-6.5, 2.0, uSunElDeg);
    float mieCore = pow(max(sunCos, 0.0), 9.0) * exp(-abs(y) * 3.2);
    float mieBroad = pow(max(sunCos, 0.0), 2.3) * horizonFactor;
    vec3 dawnDuskGlow = mix(vec3(0.98, 0.34, 0.18), vec3(1.0, 0.74, 0.34), smoothstep(-3.0, 6.5, uSunElDeg));
    skyCol += dawnDuskGlow * (mieCore * 0.68 + mieBroad * 0.38) * sunGlowVis;

    // Physical limb-darkened Solar Disk & inner corona (occluded naturally by 3D clouds & terrain!)
    if (uSunElDeg > -1.4 && y > -0.04) {
      float angDeg = degrees(acos(clamp(sunCos, -1.0, 1.0)));
      float diskMask = 1.0 - smoothstep(0.24, 0.34, angDeg);
      float aureole = exp(-angDeg / 1.35) * 0.55 + exp(-angDeg / 4.2) * 0.18;
      float sinEl = sin(radians(max(uSunElDeg, 0.12)));
      float airMass = clamp(1.0 / (sinEl + 0.12 * pow(max(0.0, uSunElDeg + 2.8), -0.75)), 1.0, 28.0);
      vec3 sunExt = exp(-vec3(0.048, 0.122, 0.255) * airMass);
      float horizonCut = smoothstep(-1.2, 0.15, uSunElDeg);
      skyCol += sunExt * (diskMask * 4.2 + aureole * 1.15) * horizonCut;
    }

    // Physical Lunar Disk & soft nocturnal aureole
    if (uMoonElDeg > -0.8 && uMoonIllum > 0.03 && uSunElDeg < -2.0 && y > -0.02) {
      vec3 moonDir = normalize(uMoonDirWorld);
      float moonCos = clamp(dot(dir, moonDir), -1.0, 1.0);
      float mAng = degrees(acos(moonCos));
      float mDisk = 1.0 - smoothstep(0.24, 0.33, mAng);
      float mGlow = exp(-mAng / 1.8) * 0.25;
      skyCol += vec3(0.84, 0.90, 1.0) * (mDisk * 1.6 + mGlow) * uMoonIllum * smoothstep(-0.6, 1.0, uMoonElDeg);
    }

    if (y < 0.0) {
      skyCol = mix(skyCol, vec3(0.025, 0.05, 0.10), clamp(-y * 8.0, 0.0, 1.0));
    }

    // Procedural starfield + faint Milky Way band in deep twilight / night
    float starVis = smoothstep(-0.8, -6.8, uSunElDeg) * smoothstep(0.03, 0.32, y);
    if (starVis > 0.01) {
      vec3 starCell = floor(dir * 420.0);
      float h = starHash(starCell);
      if (h > 0.9962) {
        float brightness = (h - 0.9962) / 0.0038;
        skyCol += vec3(0.84, 0.91, 1.0) * brightness * starVis * 0.92;
      }
    }

    gl_FragColor = vec4(skyCol, 1.0);
  }
`;

const skyDomeMat = new THREE.ShaderMaterial({
  vertexShader: SKY_DOME_VERTEX,
  fragmentShader: SKY_DOME_FRAGMENT,
  uniforms: {
    uSunDirWorld: { value: new THREE.Vector3(-0.9, 0.05, 0.2).normalize() },
    uSunElDeg: { value: 0.5 },
    uMoonDirWorld: { value: new THREE.Vector3(-0.6, 0.35, 0.4).normalize() },
    uMoonElDeg: { value: -10.0 },
    uMoonIllum: { value: 0.25 },
  },
  side: THREE.BackSide,
  depthWrite: false,
});
const skyDomeMesh = new THREE.Mesh(new THREE.SphereGeometry(110000, 48, 32), skyDomeMat);
atmosphereGroup.add(skyDomeMesh);

// 1x1 fallback white texture before USGS orthoimagery or DEM loads
const dummyWhiteTex = new THREE.DataTexture(
  new Uint8Array([255, 255, 255, 255]),
  1,
  1,
  THREE.RGBAFormat
);
dummyWhiteTex.needsUpdate = true;

// 8x8 Positional Sounding Texture kept for optional analytical vector overlays & terrain gap alignment
const spatialCloudData = new Uint8Array(8 * 8 * 4);
for (let i = 0; i < 64; i++) {
  spatialCloudData[i * 4 + 0] = 64;
  spatialCloudData[i * 4 + 1] = 39;
  spatialCloudData[i * 4 + 2] = 72;
  spatialCloudData[i * 4 + 3] = 0;
}
const spatialCloudTex = new THREE.DataTexture(
  spatialCloudData,
  8,
  8,
  THREE.RGBAFormat,
  THREE.UnsignedByteType
);
spatialCloudTex.magFilter = THREE.LinearFilter;
spatialCloudTex.minFilter = THREE.LinearFilter;
spatialCloudTex.wrapS = THREE.ClampToEdgeWrapping;
spatialCloudTex.wrapT = THREE.ClampToEdgeWrapping;
spatialCloudTex.needsUpdate = true;

const CLOUD_SLANT_FACTOR = 1.65;
const createDefaultCloudGapsUniform = () =>
  Array.from({ length: 6 }, () => new THREE.Vector4(0, 0, 600, 0));
let activeWindDirENU = new THREE.Vector2(0.88, -0.46);
let activeFogTopLocalY = 360.0;

// Custom Volumetric Crepuscular God-Ray Cone Shader (Fresnel edge softness + vertical extinction + wind dust shimmer)
const GODRAY_VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vNormalWorld;
  varying vec3 vWorldPos;
  void main() {
    vUv = uv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    vNormalWorld = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const GODRAY_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  varying vec3 vNormalWorld;
  varying vec3 vWorldPos;
  uniform vec3 uBeamColor;
  uniform float uBaseOpacity;
  uniform float uCloudTime;
  uniform vec2 uWindDirENU;

  void main() {
    vec3 N = normalize(vNormalWorld);
    vec3 V = normalize(cameraPosition - vWorldPos);
    // Smooth cylindrical Fresnel falloff so the sunbeam has zero hard polygon edges
    float fresnel = pow(clamp(abs(dot(N, V)), 0.0, 1.0), 1.45);
    // Vertical atmospheric profile: intense near the 3D cloud hole (vUv.y -> 1), soft taper into sea-mist spotlight
    float vertProfile = smoothstep(0.0, 0.14, vUv.y) * mix(0.62, 1.0, vUv.y);
    // Subtle wind-advected Tyndall aerosol / marine mist striations drifting across the light shaft
    float drift = uCloudTime * 0.55;
    float striation = 0.84 + 0.16 * sin(vUv.x * 25.1327 + drift + vWorldPos.y * 0.012)
                           * cos(vWorldPos.x * 0.0045 - uWindDirENU.x * drift);
    float alpha = uBaseOpacity * fresnel * vertProfile * striation;
    if (alpha <= 0.003) discard;
    gl_FragColor = vec4(uBeamColor * (1.05 + 0.25 * fresnel), alpha);
  }
`;

let activeGodRayMaterials = [];

let terrainShaderMaterial = null;
let terrainMeshRef = null;
let vpMarkerMeshes = [];
const textureLoader = new THREE.TextureLoader();

const spriteTextureCache = new Map();

function disposeObjectTree(group) {
  if (!group) return;
  while (group.children.length > 0) {
    const obj = group.children.pop();
    if (obj.children && obj.children.length > 0) {
      disposeObjectTree(obj);
    }
    if (obj.geometry) {
      obj.geometry.dispose();
    }
    if (obj.material) {
      if (Array.isArray(obj.material)) {
        obj.material.forEach((m) => m?.dispose?.());
      } else {
        obj.material.dispose?.();
      }
    }
  }
}

// Custom GLSL Topographic Shader:
// - 9-tap Sobel Micro-Relief Normals & Laplacian Cavity Shading
// - 128-step Solar Ray-Marching for Topographic Shadows & Alpenglow Landing Zone
// - 8x8 Positional Cloud Shadow Projection & 15m Windward Topographic Fog Damming
// - 80-step Real-Time GPU Viewshed Ray-Marching (Pixel -> Target Upper Face)
// - 3 Surface Modes: Granite Relief (0), USGS Satellite Ortho (1), Topo Contours (2)
const TERRAIN_VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vWorldPos;
  void main() {
    vUv = uv;
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const TERRAIN_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  uniform sampler2D uHeightTex;
  uniform sampler2D uOrthoTex;
  uniform sampler2D uSpatialCloudTex;
  uniform sampler2D uCloudShadowTex;
  uniform float uCloudDomainM;
  uniform float uUseFluidShadows;
  uniform float uHasOrtho;
  uniform float uGridSize;
  uniform float uSpanM;
  uniform float uSpatialSpanM;
  uniform float uMinZ;
  uniform float uMaxZ;
  uniform vec3 uSunDirENU;       // [East, North, Up] unit vector toward sun
  uniform vec2 uWindDirENU;      // [East, North] unit vector wind is blowing toward
  uniform vec2 uTargetEN;        // [East_m, North_m] of target landmark summit
  uniform float uTargetUpperZ;   // Elevation (m) of target landmark upper face
  uniform float uHighlightZone;  // 1.0 = highlight golden-hour landing zone
  uniform float uShowViewshed;   // 1.0 = compute & overlay 360° GPU Viewshed to Target
  uniform float uSurfaceStyle;   // 0.0 = Granite, 1.0 = USGS Satellite, 2.0 = Topo Contours
  uniform float uFogBaseM;       // Elevation (m ASL) of marine layer / valley fog base
  uniform float uFogTopM;        // Elevation (m ASL) of marine layer / valley fog top
  uniform float uFogStrength;    // 0.0..1.0 inversion fog density from live weather forecast
  uniform vec4 uCloudGaps[6];    // [east_m, north_m, gapElevM, radius_m] 3D crepuscular cloud holes
  uniform float uNumCloudGaps;
  uniform float uMeanCloudTau;
  uniform float uCloudTime;

  varying vec2 vUv;
  varying vec3 vWorldPos;

  // Procedural rock micro-grain & multi-octave FBM for sub-meter cliff joint/ledge detail
  float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
  }

  float rockNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
      mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
      u.y
    );
  }

  float rockFbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.80, 0.60, -0.60, 0.80);
    for (int i = 0; i < 3; i++) {
      v += a * rockNoise(p);
      p = rot * p * 2.15;
      a *= 0.5;
    }
    return v;
  }

  // Exact vertex-aligned hardware bilinear DEM sampler (maps [0..1] PlaneGeometry UV -> texel centers
  // [(0.5/n) .. (1 - 0.5/n)] in 1 hardware-filtered fetch instead of 4 manual taps!)
  float sampleDemSmooth(vec2 uv) {
    float n = max(uGridSize, 240.0);
    vec2 tcUv = clamp(uv, 0.0, 1.0) * ((n - 1.0) / n) + vec2(0.5 / n);
    return texture2D(uHeightTex, tcUv).r;
  }

  void main() {
    float z = vWorldPos.y + uMinZ;
    float relZ = clamp((z - uMinZ) / max(uMaxZ - uMinZ, 1.0), 0.0, 1.0);

    // 9-tap Sobel filter for razor-sharp granite cliff normals + Laplacian curvature (cavity AO)
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
    float slope = 1.0 - normalENU.z;

    // Sub-meter procedural cliff-fracture & exfoliation micro-normal perturbation on steep slopes
    float cliffMask = smoothstep(0.12, 0.48, slope);
    vec2 microCoord = vec2(vWorldPos.x * 0.055 + z * 0.035, vWorldPos.z * 0.055 - z * 0.035);
    float f0 = rockFbm(microCoord);
    float fx = rockFbm(microCoord + vec2(0.35, 0.0));
    float fy = rockFbm(microCoord + vec2(0.0, 0.35));
    vec2 microGrad = vec2(fx - f0, fy - f0) * cliffMask * 0.42;
    normalENU = normalize(vec3(normalENU.x - microGrad.x, normalENU.y - microGrad.y, normalENU.z));

    // Curvature / Cavity AO: positive on exposed ridge crests, negative in couloirs/chutes
    float laplacian = (zN + zS + zE + zW - 4.0 * z) / max(stepWorld, 1.0);
    float cavityAO = clamp(1.0 + laplacian * 0.48, 0.54, 1.32);

    // Multi-scale procedural granite/bluff striation on steep cliff walls
    float rockGrain = ((f0 - 0.5) * 0.08 + (hash21(vWorldPos.xz * 0.22) - 0.5) * 0.04) * cliffMask;

    float ndotl = max(dot(normalENU, uSunDirENU), 0.0);

    // 1. Hardware-bilinear DEM shadow march with 3D surface-normal origin offset (eliminates grazing cliff-facet
    //    sawteeth on sheer 80-deg granite walls like Half Dome!) + physical 0.53-deg solar disk penumbra
    float lit = 1.0;
    float minClearance = 1e5;
    float closestOccluderDist = 0.0;
    float minPenumbraRatio = 1e5;
    if (uSunDirENU.z <= -0.01) {
      lit = 0.0;
    } else if (ndotl <= 0.002) {
      // Back-facing cliff slopes turned away from the Sun are naturally in self-shadow!
      lit = 0.0;
    } else {
      vec2 horizDir = normalize(uSunDirENU.xy + vec2(1e-5));
      float tanEl = uSunDirENU.z / max(length(uSunDirENU.xy), 0.005);
      float marchStepM = uSpanM / 280.0;
      vec2 uvStep = (horizDir * marchStepM) / uSpanM;
      // Offset ray origin outward along the 3D cliff normal by ~1.8 DEM cells so grazing rays along sheer walls
      // never scrape sideways through adjacent 15m raster columns!
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
        float terrainZ = sampleDemSmooth(sampleUv);
        float clearance = rayZ - terrainZ;
        if (distM > 45.0 && clearance < minClearance) {
          minClearance = clearance;
          closestOccluderDist = distM;
        }
        float penumbraHalf = max(1.4, distM * 0.0085);
        float ratio = clearance / penumbraHalf;
        if (ratio < minPenumbraRatio) {
          minPenumbraRatio = ratio;
        }
        if (ratio < -1.0) {
          lit = 0.0;
          break;
        }
      }
      if (lit > 0.0 && minPenumbraRatio < 1.0) {
        lit = smoothstep(-1.0, 1.0, minPenumbraRatio);
      }
    }
    float topoLit = lit;

    // 2. Base Surface Material according to selected uSurfaceStyle
    vec3 valleyForest = vec3(0.10, 0.17, 0.15);
    vec3 subAlpine    = vec3(0.22, 0.27, 0.25);
    vec3 midGranite   = vec3(0.42, 0.45, 0.50);
    vec3 highGranite  = vec3(0.68, 0.71, 0.76);

    vec3 rockBase = mix(valleyForest, subAlpine, smoothstep(0.06, 0.30, relZ));
    rockBase = mix(rockBase, midGranite, smoothstep(0.26, 0.55, relZ + slope * 0.45));
    rockBase = mix(rockBase, highGranite, smoothstep(0.52, 0.88, relZ + slope * 0.55));
    rockBase = (rockBase + rockGrain) * cavityAO;

    if (uSurfaceStyle > 0.5 && uSurfaceStyle < 1.5 && uHasOrtho > 0.5) {
      // 2.4K USGS National Map Orthoimagery blended with LiDAR micro-relief cavity AO
      vec3 orthoRgb = texture2D(uOrthoTex, vec2(vUv.x, vUv.y)).rgb;
      rockBase = mix(rockBase, (orthoRgb + rockGrain * 0.45) * cavityAO * 1.14, 0.85);
    }

    // Coastal & Pacific Ocean Water Surface Shader (for sea-level terrain where uMinZ < 5.0 and z <= 2.2m)
    float isOcean = (uMinZ < 5.0) ? (1.0 - smoothstep(0.3, 2.2, z)) : 0.0;
    float isSurfZone = (uMinZ < 5.0) ? (smoothstep(0.1, 0.7, z) * (1.0 - smoothstep(0.7, 2.8, z))) : 0.0;

    // Slope-compensated anti-aliased topographic contour lines
    float slopeComp = clamp(1.0 + slope * 3.2, 1.0, 2.8);
    float contour50  = (1.0 - smoothstep(0.0, 0.024 * slopeComp, abs(fract(z / 50.0) - 0.5))) * (1.0 - isOcean);
    float contour200 = (1.0 - smoothstep(0.0, 0.018 * slopeComp, abs(fract(z / 200.0) - 0.5))) * (1.0 - isOcean);
    float contour20  = (1.0 - smoothstep(0.0, 0.032 * slopeComp, abs(fract(z / 20.0) - 0.5))) * (1.0 - isOcean);

    // Distance from Target Landmark summit in [East, North] meters
    vec2 posEN = vec2(vWorldPos.x, -vWorldPos.z);
    float distToTarget = length(posEN - uTargetEN);
    float targetMask = 1.0 - smoothstep(220.0, 920.0, distToTarget);

    // 3D Multi-Altitude Beer-Lambert Volumetric Cloud Ray-Marching & Coaxial Crepuscular Cloud-Gap Spotlights
    // Uses the shared coaxial slant factor (CLOUD_SLANT = 1.65) so every 3D hole in the cloud deck casts a
    // diagonal 3D sunbeam shaft that lands at the EXACT same ground/ocean pixel as this shader's spotlight!
    vec2 spUv = clamp(posEN / max(uSpatialSpanM, 1000.0) + 0.5, 0.02, 0.98);
    vec4 scHere = texture2D(uSpatialCloudTex, spUv);
    vec2 horizSun = normalize(uSunDirENU.xy + vec2(1e-5));
    const float CLOUD_SLANT = 1.65;

    // 1. Ray intercept through Marine Stratus / Valley Fog slab (uFogBaseM .. localFogTop)
    float localFogTopEst = scHere.g > 0.01 ? mix(uFogTopM, scHere.g * 2500.0, 0.78) : uFogTopM;
    float stratusMidZ = max(z + 25.0, (uFogBaseM + localFogTopEst) * 0.5);
    vec2 windDriftEN = uWindDirENU * (uCloudTime * 18.0);
    vec2 stratusHitEN = posEN + horizSun * ((stratusMidZ - z) * CLOUD_SLANT) - windDriftEN;
    vec2 spStratusUv = clamp(stratusHitEN / max(uSpatialSpanM, 1000.0) + 0.5, 0.02, 0.98);
    vec4 scStratus = texture2D(uSpatialCloudTex, spStratusUv);
    float stratusRayCov = scStratus.r * step(z, localFogTopEst + 80.0);
    float stratusMacro = rockFbm(stratusHitEN * 0.00075 + vec2(1.4, -0.9));
    float stratusFine1m = rockFbm(stratusHitEN * 0.032 + vec2(stratusMacro * 3.2, -1.7));
    float stratusCellNoise = clamp(stratusMacro + (stratusFine1m - 0.48) * 0.34, 0.0, 1.0);
    float tauStratus = max(0.0, stratusRayCov - 0.14) * (0.42 + 0.58 * smoothstep(0.22, 0.70, stratusCellNoise)) * 1.48;

    // 2. Ray intercept through Main 850 hPa Cumulus / Stratocumulus Deck (matches 3D gap height)
    float cumLayerZ = uMinZ + (uMinZ < 200.0 ? 640.0 : 580.0);
    vec2 cumHitEN = posEN + horizSun * (max(60.0, cumLayerZ - z) * CLOUD_SLANT) - windDriftEN * 1.25;
    vec2 spCumUv = clamp(cumHitEN / max(uSpatialSpanM, 1000.0) + 0.5, 0.02, 0.98);
    vec4 scCum = texture2D(uSpatialCloudTex, spCumUv);
    float cumRayCov = scCum.b;
    float cumMacro = rockFbm(cumHitEN * 0.00085 + vec2(2.3, -1.4));
    float cumTurret = 1.0 - abs(rockFbm(cumHitEN * 0.0042 + vec2(cumMacro * 2.5)) * 2.0 - 1.0);
    float cumWisp1m = rockFbm(cumHitEN * 0.055 - vec2(cumTurret * 2.2));
    float cumBillowNoise = clamp(mix(cumMacro, cumTurret, 0.55) + (cumWisp1m - 0.48) * 0.28, 0.0, 1.0);
    float tauCumulus = max(0.0, cumRayCov - 0.14) * (0.45 + 0.55 * smoothstep(0.24, 0.72, cumBillowNoise)) * 1.68;

    // 3. Direct 12-level vertical optical depth tau encoded in uSpatialCloudTex.a (0..6.5)
    float encodedTau = scCum.a * 6.5 * (0.68 + 0.44 * cumBillowNoise);
    float totalCloudTau = max(tauStratus + tauCumulus, encodedTau);

    // 4. Coaxial 3D Crepuscular Cloud-Gap Fenestrations with ragged wind-torn 1m fractal spotlight edges!
    float gapSpotlight = 0.0;
    float gapCoreBeam = 0.0;
    for (int g = 0; g < 6; g++) {
      if (float(g) >= uNumCloudGaps) break;
      vec4 gap = uCloudGaps[g];
      if (gap.w > 10.0) {
        float gapZ = max(z + 60.0, gap.z);
        vec2 rayAtGapEN = posEN + horizSun * ((gapZ - z) * CLOUD_SLANT);
        vec2 deltaG = rayAtGapEN - gap.xy;
        float alongW = dot(deltaG, uWindDirENU);
        float crossW = dot(deltaG, vec2(-uWindDirENU.y, uWindDirENU.x));
        float anisoDist = length(vec2(alongW * 0.80, crossW * 1.15));
        float raggedWarp = (cumTurret - 0.48) * gap.w * 0.45 + (cumWisp1m - 0.5) * 52.0;
        float dGap = anisoDist + raggedWarp;
        float spot = 1.0 - smoothstep(gap.w * 0.24, gap.w * 0.98, dGap);
        float core = 1.0 - smoothstep(0.0, gap.w * 0.50, dGap);
        gapSpotlight = max(gapSpotlight, spot);
        gapCoreBeam = max(gapCoreBeam, core);
      }
    }

    // Beer-Lambert Transmittance T = exp(-0.72 * tau):
    // Under a thick overcast deck (tau ~ 3.5-5.5), transmittance drops to ~0.04-0.08 (deep storm-cloud shadow),
    // EXCEPT directly inside the coaxial 3D crepuscular sunbeam footprints where sunlight punches through!
    totalCloudTau *= (1.0 - gapSpotlight * 0.94);
    float cloudTransmittance = mix(exp(-clamp(totalCloudTau * 0.72, 0.0, 4.0)), 0.98, gapSpotlight);
    if (uUseFluidShadows > 0.5) {
      vec2 fluidShadowUv = fract(posEN / max(uCloudDomainM, 1000.0) + 0.5);
      float fluidShadowTrans = texture2D(uCloudShadowTex, fluidShadowUv).r;
      cloudTransmittance = mix(cloudTransmittance, fluidShadowTrans, 0.78);
    }
    float isThickOvercast = smoothstep(1.25, 2.6, uMeanCloudTau);
    float ambientFloor = mix(0.38, 0.16, isThickOvercast);
    float effectiveCloudLight = ambientFloor + (1.0 - ambientFloor) * cloudTransmittance;

    // Low-sun Golden Hour factor
    float sunElDeg = degrees(asin(clamp(uSunDirENU.z, -1.0, 1.0)));
    float goldenFactor = smoothstep(-0.5, 1.5, sunElDeg) * (1.0 - smoothstep(9.5, 18.0, sunElDeg));

    // High-contrast Shadowed Umbra vs. Sunlit Area coloring for a razor-sharp light/dark border
    vec3 shadowColor = rockBase * vec3(0.12, 0.19, 0.35) * (0.34 + 0.50 * relZ);
    vec3 warmSunColor = mix(vec3(1.0, 0.92, 0.78), vec3(1.0, 0.58, 0.22), goldenFactor);
    vec3 sunlitColor = rockBase * (0.46 + 0.88 * ndotl) * warmSunColor * effectiveCloudLight;

    // Dramatic molten-gold crepuscular spotlight patch where a 3D cloud-gap sunbeam strikes the bluffs/ridges
    vec3 crepuscularGold = mix(vec3(1.0, 0.88, 0.42), vec3(1.0, 0.62, 0.16), goldenFactor);
    sunlitColor = mix(
      sunlitColor,
      sunlitColor * 1.45 + crepuscularGold * 0.42,
      clamp(gapSpotlight * 0.78 + gapCoreBeam * 0.35, 0.0, 0.96)
    );

    // Illuminate EVERY shadow-producing grazing ray by area (and trace a razor-sharp golden terminator border
    // right where the sunlit grazing ray meets the cast shadow of any ridge/cliff/headland), modulated by cloud transmittance
    float isGrazingShadowRay = step(40.0, closestOccluderDist) * step(0.01, topoLit);
    float grazingRayAreaBand = isGrazingShadowRay * (1.0 - smoothstep(0.0, 58.0, minClearance)) * cloudTransmittance;
    float terminatorKnifeEdge = isGrazingShadowRay * (1.0 - smoothstep(0.0, 4.5, minClearance)) * (0.25 + 0.75 * cloudTransmittance);
    vec3 grazingAreaCol = mix(vec3(1.0, 0.88, 0.44), vec3(1.0, 0.62, 0.16), goldenFactor);
    sunlitColor = mix(
      sunlitColor,
      sunlitColor * 1.28 + grazingAreaCol * 0.34,
      clamp(grazingRayAreaBand * 0.68 + terminatorKnifeEdge * 0.85, 0.0, 0.94)
    );

    // Highlight Golden-Hour Sun Landing Zone on the target landmark's lit face
    if (uHighlightZone > 0.5 && topoLit > 0.02 && isOcean < 0.5) {
      float cliffLanding = targetMask * smoothstep(0.32, 0.76, relZ) * (0.35 + 0.65 * ndotl) * (0.28 + 0.72 * cloudTransmittance);
      vec3 alpenglowCopper = vec3(1.0, 0.46, 0.11);
      vec3 alpenglowGold   = vec3(1.0, 0.78, 0.26);
      vec3 landingGlow = mix(alpenglowGold, alpenglowCopper, goldenFactor);
      sunlitColor = mix(sunlitColor, landingGlow, clamp(cliffLanding * 0.74, 0.0, 0.86));

      float rimRing = smoothstep(570.0, 610.0, distToTarget) * (1.0 - smoothstep(610.0, 650.0, distToTarget));
      sunlitColor += vec3(1.0, 0.74, 0.22) * rimRing * 0.48;
    }

    vec3 finalColor = mix(shadowColor, sunlitColor, topoLit);

    // Blend Pacific Ocean water surface, live wind-advected swell ripples, coastal surf foam & crepuscular cloud-gap spotlights on water
    if (isOcean > 0.01) {
      float wavePhase = uCloudTime * 0.85;
      float wave1 = sin(vWorldPos.x * 0.042 + vWorldPos.z * 0.026 - wavePhase) * 0.5 + 0.5;
      float wave2 = cos(vWorldPos.x * 0.021 - vWorldPos.z * 0.048 + wavePhase * 0.72) * 0.5 + 0.5;
      float swell = rockNoise(vWorldPos.xz * 0.015 - uWindDirENU * (uCloudTime * 0.12));
      float waveRipple = mix(mix(wave1, wave2, 0.5), swell, 0.42);
      vec3 oceanShadow = vec3(0.02, 0.06, 0.14);
      vec3 oceanLitBase = mix(vec3(0.05, 0.18, 0.34), vec3(0.11, 0.34, 0.48), clamp((sunElDeg + 6.0) / 20.0, 0.0, 1.0));
      vec3 oceanLit = mix(oceanLitBase, vec3(0.14, 0.40, 0.52), isSurfZone * 0.65 + waveRipple * 0.16) * effectiveCloudLight;

      // Glowing golden-amber crepuscular sunbeam spotlight patch on the dark ocean where a 3D cloud gap punches through!
      vec3 waterSpotCol = mix(vec3(1.0, 0.78, 0.32), vec3(1.0, 0.54, 0.14), goldenFactor) * (0.55 + 0.45 * waveRipple);
      oceanLit = mix(
        oceanLit,
        oceanLit * 1.55 + waterSpotCol * 0.68,
        clamp(gapSpotlight * 0.88 + gapCoreBeam * 0.45, 0.0, 0.96)
      );

      // Illuminate shadow-producing grazing ray area & sharp terminator across the ocean surface
      oceanLit = mix(
        oceanLit,
        oceanLit * 1.25 + grazingAreaCol * 0.30,
        clamp(grazingRayAreaBand * 0.62 + terminatorKnifeEdge * 0.82, 0.0, 0.92)
      );
      vec3 oceanCol = mix(oceanShadow, oceanLit, topoLit);

      // Solar specular glitter path across the sunlit Pacific toward uSunDirENU (occluded under thick clouds except in gaps)
      vec2 sunHoriz = normalize(uSunDirENU.xy + vec2(1e-5));
      vec2 pixHoriz = normalize(-posEN + vec2(1e-5));
      float sunAlign = max(dot(sunHoriz, -pixHoriz), 0.0);
      float glint = pow(sunAlign, 14.0) * (0.35 + 0.65 * waveRipple) * smoothstep(-2.0, 3.0, sunElDeg) * (0.05 + 0.95 * topoLit) * cloudTransmittance;
      vec3 glintCol = mix(vec3(1.0, 0.58, 0.24), vec3(1.0, 0.88, 0.55), clamp(sunElDeg / 10.0, 0.0, 1.0));
      oceanCol += glintCol * glint * 0.92;

      // Whitewater coastal surf line right where waves meet Pedro Point / Devil's Slide bluffs
      oceanCol = mix(oceanCol, vec3(0.84, 0.92, 0.98) * (0.28 + 0.72 * topoLit * effectiveCloudLight), isSurfZone * (0.38 + 0.38 * wave1));
      finalColor = mix(finalColor, oceanCol, isOcean);
    }

    // 2b. Positional Volumetric Marine Layer / Valley Inversion Fog Immersion + 15m Windward Topographic Damming
    if (uFogStrength > 0.05 && uFogTopM > uMinZ + 15.0) {
      float localFogTop = scHere.g > 0.01 ? mix(uFogTopM, scHere.g * 2500.0, 0.78) : uFogTopM;
      float localFogStr = mix(uFogStrength, scHere.r, 0.80);

      // March 4 steps upwind (-uWindDirENU) on 15m DEM to detect mountain ridge fog damming (e.g. Montara Mountain)
      float maxUpwindZ = 0.0;
      for (int k = 1; k <= 4; k++) {
        float stepDist = float(k) * 420.0;
        vec2 upUv = vUv - (uWindDirENU * stepDist) / max(uSpanM, 1000.0);
        if (upUv.x >= 0.01 && upUv.x <= 0.99 && upUv.y >= 0.01 && upUv.y <= 0.99) {
          maxUpwindZ = max(maxUpwindZ, texture2D(uHeightTex, upUv).r);
        }
      }
      if (maxUpwindZ > localFogTop + 16.0) {
        float damBlock = clamp((maxUpwindZ - localFogTop) / 140.0, 0.0, 1.0);
        localFogStr *= (1.0 - 0.82 * damBlock);
        localFogTop -= 60.0 * damBlock;
      }

      // Windward orographic fog pile-up against coastal bluffs
      float windwardLift = max(0.0, dot(normalENU.xy, -uWindDirENU));
      localFogTop += windwardLift * 24.0;

      float belowTop = smoothstep(localFogTop + 28.0, localFogTop - 48.0, z);
      float aboveBase = smoothstep(uFogBaseM - 120.0, uFogBaseM + 20.0, z);
      float fogNoise = rockFbm(vWorldPos.xz * 0.0018 + vec2(1.7, -2.3));
      float fogVeil = belowTop * aboveBase * clamp(localFogStr * (0.72 + 0.45 * fogNoise), 0.0, 0.86);
      vec3 fogLitCol = mix(vec3(0.62, 0.72, 0.86), vec3(0.98, 0.80, 0.62), goldenFactor * 0.75);
      // Keep subtle ground mist dampen here while 3D volumetric raymarcher renders real 1m fog billows above!
      finalColor = mix(finalColor, fogLitCol, fogVeil * 0.18);

      float topShore = exp(-pow((z - localFogTop) / 22.0, 2.0)) * (1.0 - isOcean);
      finalColor = mix(finalColor, fogLitCol * 1.06, topShore * localFogStr * 0.22);
    }

    // 3. Real-Time GPU 360° Viewshed Heatmap Overlay (80-step Line-of-Sight from every terrain pixel to Target Upper Face)
    if (uShowViewshed > 0.5 && distToTarget > 260.0) {
      vec2 targetUv = (uTargetEN + vec2(uSpanM * 0.5)) / uSpanM;
      float startZ = z + 1.8;
      float endZ = uTargetUpperZ + 6.0;
      float losClear = 1.0;

      for (int j = 2; j <= 78; j++) {
        float frac = float(j) / 80.0;
        vec2 sampleUv = mix(vUv, targetUv, frac);
        float rayZ = mix(startZ, endZ, frac);
        float tZ = texture2D(uHeightTex, sampleUv).r;
        if (tZ > rayZ + 3.0) {
          losClear = 0.0;
          break;
        }
      }

      if (losClear > 0.5) {
        // Unobstructed line of sight to Landmark
        vec3 viewshedCyan = vec3(0.06, 0.82, 0.62);
        float pulse = 0.36 + 0.18 * smoothstep(0.2, 0.8, relZ);
        finalColor = mix(finalColor, viewshedCyan, pulse);
      } else {
        // Occluded by intervening ridge
        finalColor *= 0.68;
      }
    }

    // Apply contour overlays
    if (uSurfaceStyle > 1.5) {
      // High-contrast Topo Contours mode
      finalColor = mix(finalColor, vec3(0.95, 0.76, 0.32), contour20 * 0.26);
      finalColor = mix(finalColor, vec3(1.0, 0.86, 0.45), contour200 * 0.65);
    } else {
      finalColor = mix(finalColor, finalColor * 0.76, contour50 * 0.25);
      finalColor = mix(finalColor, vec3(0.95, 0.82, 0.55), contour200 * 0.18);
    }

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

// ============================================================================
// Coordinate Conversion & Mesh Elevation Sampling Helpers
// ============================================================================
function latLonElevToWorld(lat, lon, elevM) {
  const patch = state.terrainPatch;
  if (!patch) return new THREE.Vector3(0, 0, 0);
  const eastM = (lon - patch.center_lon) * patch.meters_per_deg_lon;
  const northM = (lat - patch.center_lat) * patch.meters_per_deg_lat;
  const yM = elevM - patch.min_elevation_m;
  return new THREE.Vector3(eastM, yM, -northM);
}

function worldToLatLonElev(worldVec) {
  const patch = state.terrainPatch;
  if (!patch) return { lat: 37.74554, lon: -119.53315, elevM: 2000 };
  const eastM = worldVec.x;
  const northM = -worldVec.z;
  const lat = patch.center_lat + northM / patch.meters_per_deg_lat;
  const lon = patch.center_lon + eastM / patch.meters_per_deg_lon;
  const elevM = worldVec.y + patch.min_elevation_m;
  return { lat, lon, elevM };
}

function sampleLocalTerrainY(eastM, northM) {
  const patch = state.terrainPatch;
  if (!patch) return 0;
  const half = patch.span_m * 0.5;
  const u = (eastM + half) / patch.span_m;
  const v = (northM + half) / patch.span_m;
  const n = patch.grid_size;
  const fx = Math.max(0, Math.min(n - 1, u * (n - 1)));
  const fy = Math.max(0, Math.min(n - 1, v * (n - 1)));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(n - 1, x0 + 1);
  const y1 = Math.min(n - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const z00 = patch.elevations[y0 * n + x0];
  const z10 = patch.elevations[y0 * n + x1];
  const z01 = patch.elevations[y1 * n + x0];
  const z11 = patch.elevations[y1 * n + x1];
  const z0 = z00 + (z10 - z00) * tx;
  const z1 = z01 + (z11 - z01) * tx;
  const z = z0 + (z1 - z0) * ty;
  return z - patch.min_elevation_m;
}

function makeTextSprite(text, bgColor = "#0f172a", borderColor = "#f59e0b", textColor = "#f8fafc") {
  const cacheKey = `${text}|${bgColor}|${borderColor}|${textColor}`;
  let tex = spriteTextureCache.get(cacheKey);
  if (!tex) {
    if (spriteTextureCache.size > 96) {
      const oldestKey = spriteTextureCache.keys().next().value;
      const oldTex = spriteTextureCache.get(oldestKey);
      oldTex?.dispose?.();
      spriteTextureCache.delete(oldestKey);
    }
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 420;
    canvas.height = 88;

    ctx.fillStyle = bgColor;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(8, 8, 404, 72, 18);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = textColor;
    ctx.font = "700 26px -apple-system, BlinkMacSystemFont, 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 210, 45);

    tex = new THREE.CanvasTexture(canvas);
    spriteTextureCache.set(cacheKey, tex);
  }

  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(470, 98, 1);
  sprite.renderOrder = 999;
  return sprite;
}

// ============================================================================
// Build High-Density 600x600 3D Terrain Mesh + Sync DEM to Rust WASM Fluid Solver
// ============================================================================
function buildTerrainMesh() {
  if (terrainShaderMaterial?.uniforms?.uHeightTex?.value) {
    terrainShaderMaterial.uniforms.uHeightTex.value.dispose?.();
  }
  disposeObjectTree(terrainGroup);
  terrainMeshRef = null;

  const patch = state.terrainPatch;
  if (!patch) return;

  const n = patch.grid_size;
  const spanM = patch.span_m;
  const minZ = patch.min_elevation_m;
  const maxZ = patch.max_elevation_m;

  const floatData =
    patch.elevations instanceof Float32Array
      ? patch.elevations
      : new Float32Array(patch.elevations);
  patch.elevations = floatData;

  const heightTex = new THREE.DataTexture(
    floatData,
    n,
    n,
    THREE.RedFormat,
    THREE.FloatType
  );
  heightTex.magFilter = THREE.LinearFilter;
  heightTex.minFilter = THREE.LinearFilter;
  heightTex.needsUpdate = true;

  const geom = new THREE.PlaneGeometry(spanM, spanM, n - 1, n - 1);
  geom.rotateX(-Math.PI / 2);

  const posAttr = geom.attributes.position;
  const uvAttr = geom.attributes.uv;

  for (let i = 0; i < posAttr.count; i++) {
    const u = uvAttr.getX(i);
    const v = uvAttr.getY(i);
    const ix = Math.min(n - 1, Math.max(0, Math.round(u * (n - 1))));
    const iy = Math.min(n - 1, Math.max(0, Math.round(v * (n - 1))));
    const zM = floatData[iy * n + ix];
    posAttr.setY(i, zM - minZ);
  }
  geom.computeVertexNormals();

  const lm = state.shootReport?.target_landmark;
  const upperZ = lm ? lm.upper_face_z_m : maxZ * 0.9;

  terrainShaderMaterial = new THREE.ShaderMaterial({
    vertexShader: TERRAIN_VERTEX_SHADER,
    fragmentShader: TERRAIN_FRAGMENT_SHADER,
    uniforms: {
      uHeightTex: { value: heightTex },
      uOrthoTex: { value: dummyWhiteTex },
      uSpatialCloudTex: { value: spatialCloudTex },
      uCloudShadowTex: { value: volumetricEngine.getShadowMapTexture() },
      uCloudDomainM: { value: volumetricEngine.telemetry.domainXyM || 18000.0 },
      uUseFluidShadows: { value: state.showAtmosphere ? 1.0 : 0.0 },
      uHasOrtho: { value: 0.0 },
      uGridSize: { value: n },
      uSpanM: { value: spanM },
      uSpatialSpanM: { value: 18400.0 },
      uMinZ: { value: minZ },
      uMaxZ: { value: maxZ },
      uSunDirENU: { value: new THREE.Vector3(-0.85, -0.35, 0.08).normalize() },
      uWindDirENU: { value: new THREE.Vector2(0.88, -0.46) },
      uTargetEN: { value: new THREE.Vector2(0, 0) },
      uTargetUpperZ: { value: upperZ },
      uHighlightZone: { value: state.highlightLandingZone ? 1.0 : 0.0 },
      uShowViewshed: { value: state.showViewshed ? 1.0 : 0.0 },
      uSurfaceStyle: { value: state.surfaceStyle },
      uFogBaseM: { value: 90.0 },
      uFogTopM: { value: 380.0 },
      uFogStrength: { value: 0.25 },
      uCloudGaps: { value: createDefaultCloudGapsUniform() },
      uNumCloudGaps: { value: 0.0 },
      uMeanCloudTau: { value: 0.35 },
      uCloudTime: { value: 0.0 },
    },
  });

  // Couple the high-res USGS 3DEP DEM into the Rust WASM 3D Navier-Stokes & Cirrus solver + GPU raymarcher!
  volumetricEngine.setTerrainDem(
    floatData,
    n,
    spanM,
    minZ,
    patch.center_lat,
    patch.center_lon,
    heightTex
  );

  const mesh = new THREE.Mesh(geom, terrainShaderMaterial);
  terrainMeshRef = mesh;
  terrainGroup.add(mesh);

  // Asynchronously load USGS National Map Orthoimagery texture for this patch
  const orthoUrl = `/api/usgs-ortho?lat=${patch.center_lat}&lon=${patch.center_lon}&span_m=${patch.span_m}`;
  textureLoader.load(
    orthoUrl,
    (orthoTex) => {
      orthoTex.colorSpace = THREE.SRGBColorSpace;
      orthoTex.minFilter = THREE.LinearMipmapLinearFilter;
      orthoTex.magFilter = THREE.LinearFilter;
      orthoTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
      if (terrainShaderMaterial) {
        terrainShaderMaterial.uniforms.uOrthoTex.value = orthoTex;
        terrainShaderMaterial.uniforms.uHasOrtho.value = 1.0;
      }
    },
    undefined,
    () => {
      // Ortho fetch optional fallback
    }
  );
}

// ============================================================================
// Build 3D Trails, Viewpoint Beacons (VP1..VP3) & Camera Sightlines
// Only the actively selected viewpoint shows its floating label & bright sightline
// ============================================================================
function buildTrailsAndSightlines() {
  disposeObjectTree(trailsGroup);
  vpMarkerMeshes = [];

  const report = state.shootReport;
  const patch = state.terrainPatch;
  if (!report || !patch) return;

  const lm = report.target_landmark;
  const summitPos = latLonElevToWorld(lm.summit_lat, lm.summit_lon, lm.summit_elevation_m);
  const upperFacePos = latLonElevToWorld(lm.upper_face_lat, lm.upper_face_lon, lm.upper_face_z_m);

  if (terrainShaderMaterial) {
    terrainShaderMaterial.uniforms.uTargetEN.value.set(summitPos.x, -summitPos.z);
    terrainShaderMaterial.uniforms.uTargetUpperZ.value = lm.upper_face_z_m;
  }

  // Target Landmark Crown Beacon + Label
  const crownGeo = new THREE.OctahedronGeometry(28, 1);
  const crownMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
  const crownMesh = new THREE.Mesh(crownGeo, crownMat);
  crownMesh.position.copy(summitPos).add(new THREE.Vector3(0, 42, 0));
  trailsGroup.add(crownMesh);

  const lmSprite = makeTextSprite(`★ ${lm.name.toUpperCase()}`, "#0f172a", "#fbbf24", "#fde68a");
  lmSprite.position.copy(summitPos).add(new THREE.Vector3(0, 195, 0));
  trailsGroup.add(lmSprite);

  // Each Top Viewpoint (VP1, VP2, VP3)
  report.top_viewpoints.forEach((vp, idx) => {
    const colorInt = VP_COLORS[idx % VP_COLORS.length];
    const colorHex = VP_HEX[idx % VP_HEX.length];
    const isSelected = idx === state.selectedVpIndex;

    // 1. 3D Hiking Trail Polyline from Trailhead -> Viewpoint
    if (vp.trail_path_to_vp && vp.trail_path_to_vp.length >= 2) {
      const rawPoints = vp.trail_path_to_vp.map(([lon, lat, elev]) => {
        const p = latLonElevToWorld(lat, lon, elev);
        p.y += 14;
        return p;
      });
      const trailPoints = [];
      for (const pt of rawPoints) {
        if (
          trailPoints.length === 0 ||
          trailPoints[trailPoints.length - 1].distanceTo(pt) > 2.0
        ) {
          trailPoints.push(pt);
        }
      }
      if (trailPoints.length >= 2) {
        const curve = new THREE.CatmullRomCurve3(trailPoints);
        const tubeGeo = new THREE.TubeGeometry(
          curve,
          Math.min(trailPoints.length * 2, 160),
          isSelected ? 15 : 7,
          8,
          false
        );
        const tubeMat = new THREE.MeshBasicMaterial({
          color: colorInt,
          transparent: true,
          opacity: isSelected ? 0.95 : 0.28,
        });
        const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
        tubeMesh.userData = { vpIndex: idx };
        trailsGroup.add(tubeMesh);
      }
    }

    // 2. Clickable Viewpoint Marker Pillar
    const vpPos = latLonElevToWorld(
      vp.viewpoint_lat,
      vp.viewpoint_lon,
      vp.viewpoint_elevation_m
    );

    const pillarGeo = new THREE.CylinderGeometry(isSelected ? 22 : 15, 5, isSelected ? 120 : 85, 16);
    const pillarMat = new THREE.MeshBasicMaterial({
      color: colorInt,
      transparent: true,
      opacity: isSelected ? 1.0 : 0.65,
    });
    const pillar = new THREE.Mesh(pillarGeo, pillarMat);
    pillar.position.copy(vpPos).add(new THREE.Vector3(0, isSelected ? 60 : 42, 0));
    pillar.userData = { vpIndex: idx };
    trailsGroup.add(pillar);
    vpMarkerMeshes.push(pillar);

    // Only highlight the floating 3D billboard & sightline beam for the selected viewpoint
    if (isSelected) {
      const vpLabel = makeTextSprite(
        `VP${vp.viewpoint_rank} • ${vp.viewpoint_label} (${vp.hike_distance_to_vp_miles} mi)`,
        "#0f172a",
        colorHex,
        "#ffffff"
      );
      vpLabel.position.copy(vpPos).add(new THREE.Vector3(0, 175, 0));
      vpLabel.userData = { vpIndex: idx };
      trailsGroup.add(vpLabel);

      const sightStart = vpPos.clone().add(new THREE.Vector3(0, 24, 0));
      const sightEnd = upperFacePos.clone().add(new THREE.Vector3(0, 18, 0));
      const sightGeo = new THREE.BufferGeometry().setFromPoints([sightStart, sightEnd]);
      const sightMat = new THREE.LineDashedMaterial({
        color: colorInt,
        dashSize: 85,
        gapSize: 42,
        transparent: true,
        opacity: 0.92,
      });
      const sightLine = new THREE.Line(sightGeo, sightMat);
      sightLine.computeLineDistances();
      trailsGroup.add(sightLine);
    }
  });

  trailsGroup.visible = state.showSightlines && state.cameraMode === "orbit";
}

// ============================================================================
// Instant Client-Side Sightline & Cross-Section Profiler (< 1 ms on Click)
// ============================================================================
function buildInstantClientTripodReport(lat, lon, elevM) {
  const lm = state.shootReport?.target_landmark;
  const patch = state.terrainPatch;
  if (!lm || !patch) return null;

  const tripodWorld = latLonElevToWorld(lat, lon, elevM);
  const targetWorld = latLonElevToWorld(lm.upper_face_lat, lm.upper_face_lon, lm.upper_face_z_m);

  const dx = targetWorld.x - tripodWorld.x;
  const dz = targetWorld.z - tripodWorld.z; // Note: North is -z
  const dNorth = -dz;
  const distM = Math.hypot(dx, dNorth);
  const distMiles = Number((distM / 1609.344).toFixed(2));
  const bearingDeg = Number(((((Math.atan2(dx, dNorth) * 180) / Math.PI) + 360) % 360).toFixed(1));
  const compassDir = compassLabel(bearingDeg);

  let focalMm = "24–70mm";
  if (distMiles < 1.1) focalMm = "16–35mm";
  else if (distMiles < 2.6) focalMm = "24–70mm";
  else if (distMiles < 5.0) focalMm = "70–200mm";
  else focalMm = "100–400mm";

  const numSamples = 64;
  const profile = [];
  let maxObstructionFt = 0;
  let obstructionDistMi = null;
  const startElevM = elevM + 1.8; // 1.8m tripod height
  const endElevM = lm.upper_face_z_m;

  for (let i = 0; i < numSamples; i++) {
    const t = i / (numSamples - 1);
    const sx = tripodWorld.x + dx * t;
    const sz = tripodWorld.z + dz * t;
    const rayLocalY = tripodWorld.y + 1.8 + (targetWorld.y - (tripodWorld.y + 1.8)) * t;
    const terrainLocalY =
      i === 0
        ? tripodWorld.y
        : i === numSamples - 1
          ? targetWorld.y
          : sampleLocalTerrainY(sx, -sz);

    const terrainElevM = (patch.min_elevation_m ?? 0) + terrainLocalY;
    const rayElevM = startElevM + (endElevM - startElevM) * t;
    const terrainFt = terrainElevM * 3.28084;
    const rayFt = rayElevM * 3.28084;

    if (i > 1 && i < numSamples - 2) {
      const excessFt = terrainFt - rayFt;
      if (excessFt > maxObstructionFt) {
        maxObstructionFt = excessFt;
        obstructionDistMi = Number((distMiles * t).toFixed(2));
      }
    }

    profile.push({
      distance_miles: Number((distMiles * t).toFixed(3)),
      terrain_elevation_ft: Number(terrainFt.toFixed(1)),
      sightline_elevation_ft: Number(rayFt.toFixed(1)),
    });
  }

  const isClear = maxObstructionFt <= 15.0;
  return {
    tripod_lat: lat,
    tripod_lon: lon,
    tripod_elevation_m: elevM,
    tripod_elevation_ft: elevM * 3.28084,
    target_name: lm.name,
    target_lat: lm.upper_face_lat,
    target_lon: lm.upper_face_lon,
    target_elevation_m: lm.upper_face_z_m,
    max_obstruction_above_ray_ft: Number(maxObstructionFt.toFixed(1)),
    obstruction_distance_miles: obstructionDistMi,
    sightline: {
      is_visible: isClear,
      visible_relief_fraction: isClear ? 0.92 : 0.0,
      distance_to_landmark_miles: distMiles,
      bearing_to_landmark_deg: bearingDeg,
      compass_direction: compassDir,
      recommended_focal_length_mm: focalMm,
    },
    sightline_profile: profile,
    nearest_trail_option: null,
    light_notches: [],
    _isComputingBackend: true,
  };
}

// ============================================================================
// Render Clicked Custom Tripod Pin, Sightline & Cross-Section Chart
// ============================================================================
function renderCustomTripodPinAndCrossSection(report) {
  disposeObjectTree(customPinGroup);
  if (!report || !state.terrainPatch) return;

  const isClear = report.sightline.is_visible && report.max_obstruction_above_ray_ft <= 15;
  const pinColorInt = isClear ? 0x10b981 : 0xf43f5e;
  const pinColorHex = isClear ? "#10b981" : "#f43f5e";

  const tripodWorld = latLonElevToWorld(
    report.tripod_lat,
    report.tripod_lon,
    report.tripod_elevation_m
  );
  const targetWorld = latLonElevToWorld(
    report.target_lat,
    report.target_lon,
    report.target_elevation_m
  );

  // Pin Beacon
  const pinGeo = new THREE.CylinderGeometry(22, 5, 135, 18);
  const pinMat = new THREE.MeshBasicMaterial({ color: pinColorInt });
  const pinMesh = new THREE.Mesh(pinGeo, pinMat);
  pinMesh.position.copy(tripodWorld).add(new THREE.Vector3(0, 68, 0));
  customPinGroup.add(pinMesh);

  const statusTag = isClear ? "CLEAR LOS" : "BLOCKED";
  const pinSprite = makeTextSprite(
    `📍 TRIPOD (${statusTag} • ${report.sightline.distance_to_landmark_miles} mi)`,
    "#0f172a",
    pinColorHex,
    "#ffffff"
  );
  pinSprite.position.copy(tripodWorld).add(new THREE.Vector3(0, 195, 0));
  customPinGroup.add(pinSprite);

  // Sightline beam from Custom Tripod -> Target
  const startPt = tripodWorld.clone().add(new THREE.Vector3(0, 26, 0));
  const endPt = targetWorld.clone().add(new THREE.Vector3(0, 20, 0));
  const lineCurve = new THREE.LineCurve3(startPt, endPt);
  const tubeGeo = new THREE.TubeGeometry(lineCurve, 24, 9, 8, false);
  const tubeMat = new THREE.MeshBasicMaterial({
    color: pinColorInt,
    transparent: true,
    opacity: 0.85,
  });
  customPinGroup.add(new THREE.Mesh(tubeGeo, tubeMat));

  // Snapped nearest trail path
  const trailOpt = report.nearest_trail_option;
  if (trailOpt?.trail_path_to_vp && trailOpt.trail_path_to_vp.length >= 2) {
    const rawPts = trailOpt.trail_path_to_vp.map(([lon, lat, elev]) => {
      const p = latLonElevToWorld(lat, lon, elev);
      p.y += 16;
      return p;
    });
    const dedup = [];
    for (const pt of rawPts) {
      if (dedup.length === 0 || dedup[dedup.length - 1].distanceTo(pt) > 2.0) {
        dedup.push(pt);
      }
    }
    if (dedup.length >= 2) {
      const curve = new THREE.CatmullRomCurve3(dedup);
      const tGeo = new THREE.TubeGeometry(curve, Math.min(dedup.length * 2, 120), 11, 8, false);
      const tMat = new THREE.MeshBasicMaterial({
        color: 0x34d399,
        transparent: true,
        opacity: 0.78,
      });
      customPinGroup.add(new THREE.Mesh(tGeo, tMat));
    }
  }

  // Populate the Custom Tripod & Sightline Cross-Section Inspector Card
  inspectorCard.hidden = false;
  document.getElementById("inspector-title").textContent =
    `Clicked Vantage (${report.tripod_lat.toFixed(4)}, ${report.tripod_lon.toFixed(4)} • ${Math.round(report.tripod_elevation_ft).toLocaleString()} ft)`;

  const statusRow = document.getElementById("inspector-status-row");
  if (isClear) {
    statusRow.innerHTML = `
      <span class="los-pill-clear">✓ Unobstructed Sightline to ${report.target_name}</span>
      <span><strong>${Math.round(report.sightline.visible_relief_fraction * 100)}%</strong> face visible • <strong>${report.sightline.recommended_focal_length_mm}</strong></span>
    `;
  } else {
    const obMi = report.obstruction_distance_miles ?? (report.sightline.distance_to_landmark_miles * 0.5).toFixed(2);
    statusRow.innerHTML = `
      <span class="los-pill-blocked">✕ Occluded by Ridge (+${Math.max(20, Math.round(report.max_obstruction_above_ray_ft))} ft at ${obMi} mi)</span>
      <span>Bearing ${report.sightline.bearing_to_landmark_deg}° (${report.sightline.compass_direction})</span>
    `;
  }

  document.getElementById("cross-section-dist-label").textContent =
    `${report.sightline.distance_to_landmark_miles} mi (${report.sightline.compass_direction})`;

  // Render 64-point SVG Sightline vs. Terrain Elevation Cross-Section Chart
  const svg = document.getElementById("cross-section-svg");
  const samples = report.sightline_profile || [];
  if (samples.length >= 2) {
    const w = 420;
    const h = 105;
    const padX = 8;
    const padY = 10;
    let minFt = Infinity;
    let maxFt = -Infinity;
    for (const s of samples) {
      minFt = Math.min(minFt, s.terrain_elevation_ft, s.sightline_elevation_ft);
      maxFt = Math.max(maxFt, s.terrain_elevation_ft, s.sightline_elevation_ft);
    }
    const spanFt = Math.max(maxFt - minFt, 200);
    const toX = (i) => padX + (i / (samples.length - 1)) * (w - 2 * padX);
    const toY = (ft) => h - padY - ((ft - minFt) / spanFt) * (h - 2 * padY);

    let areaPath = `M ${toX(0).toFixed(1)} ${(h - padY).toFixed(1)} `;
    let terrainLine = "";
    for (let i = 0; i < samples.length; i++) {
      const x = toX(i).toFixed(1);
      const y = toY(samples[i].terrain_elevation_ft).toFixed(1);
      areaPath += `L ${x} ${y} `;
      terrainLine += `${i === 0 ? "M" : "L"} ${x} ${y} `;
    }
    areaPath += `L ${toX(samples.length - 1).toFixed(1)} ${(h - padY).toFixed(1)} Z`;

    const rayX0 = toX(0).toFixed(1);
    const rayY0 = toY(samples[0].sightline_elevation_ft).toFixed(1);
    const rayX1 = toX(samples.length - 1).toFixed(1);
    const rayY1 = toY(samples[samples.length - 1].sightline_elevation_ft).toFixed(1);
    const rayStroke = isClear ? "#34d399" : "#fb7185";

    svg.innerHTML = `
      <defs>
        <linearGradient id="terrainGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#64748b" stop-opacity="0.65" />
          <stop offset="100%" stop-color="#0f172a" stop-opacity="0.9" />
        </linearGradient>
      </defs>
      <path d="${areaPath}" fill="url(#terrainGrad)" />
      <path d="${terrainLine}" fill="none" stroke="#94a3b8" stroke-width="2" />
      <line x1="${rayX0}" y1="${rayY0}" x2="${rayX1}" y2="${rayY1}" stroke="${rayStroke}" stroke-width="2.4" stroke-dasharray="5,3" />
      <circle cx="${rayX0}" cy="${rayY0}" r="4" fill="${rayStroke}" />
      <circle cx="${rayX1}" cy="${rayY1}" r="4.5" fill="#fbbf24" />
    `;
  }

  // Nearest Trailhead & Detected 360° Keyhole Light Notches
  const metaGrid = document.getElementById("inspector-meta-grid");
  const topNotch = report.light_notches?.[0];
  const notchStr = topNotch
    ? `Az ${topNotch.azimuth_deg.toFixed(1)}° (${topNotch.prominence_deg.toFixed(1)}° deep)`
    : "Open Alpine Horizon";

  if (trailOpt) {
    const hrs = Math.floor(trailOpt.hike_duration_to_vp_minutes / 60);
    const mins = trailOpt.hike_duration_to_vp_minutes % 60;
    const durStr = hrs > 0 ? `${hrs}h ${mins}m` : `${mins}m`;
    const depStr = trailOpt.schedules?.[0]?.recommended_trailhead_departure_local || "14:30";
    metaGrid.innerHTML = `
      <div class="metric-tile"><span class="metric-tile-label">Nearest Trailhead</span><strong class="metric-tile-val">${trailOpt.trailhead_name}</strong></div>
      <div class="metric-tile"><span class="metric-tile-label">Hike to Pin</span><strong class="metric-tile-val">${trailOpt.hike_distance_to_vp_miles} mi (${durStr})</strong></div>
      <div class="metric-tile"><span class="metric-tile-label">Gain · Off-Trail</span><strong class="metric-tile-val">+${Math.round(trailOpt.elevation_gain_to_vp_feet).toLocaleString()} ft · ${Math.round(report.off_trail_distance_ft)} ft</strong></div>
      <div class="metric-tile"><span class="metric-tile-label">Keyhole · Depart</span><strong class="metric-tile-val">${notchStr} · ${depStr}</strong></div>
    `;
  } else if (report._isComputingBackend) {
    metaGrid.innerHTML = `
      <div class="metric-tile"><span class="metric-tile-label">Tripod Elevation</span><strong class="metric-tile-val">${Math.round(report.tripod_elevation_ft).toLocaleString()} ft ASL</strong></div>
      <div class="metric-tile"><span class="metric-tile-label">360° Horizon &amp; Trail</span><strong class="metric-tile-val">Solving WASM route…</strong></div>
    `;
  } else {
    metaGrid.innerHTML = `
      <div class="metric-tile"><span class="metric-tile-label">Tripod Elevation</span><strong class="metric-tile-val">${Math.round(report.tripod_elevation_ft).toLocaleString()} ft ASL</strong></div>
      <div class="metric-tile"><span class="metric-tile-label">Best Keyhole Notch</span><strong class="metric-tile-val">${notchStr}</strong></div>
    `;
  }
}

// ============================================================================
// Unified 3D Cloud Extinction Voxel Sampler & Beer-Lambert Ray-Marcher
// ============================================================================
let activeCloudEllipsoids = [];
let activeCloudGaps = [];
let activeMeanCloudTau = 0.35;
let activeStratusBand = { baseLocalY: 40, topLocalY: 360, mistCov: 0.24, minElevM: 0, windEast: 0.88, windNorth: -0.46 };

function sampleBilinearSpatialCloud(eastM, northM) {
  const span = state.weatherReport?.spatial_cloud_field?.span_m || 18400.0;
  const u = Math.max(0.0, Math.min(0.999, eastM / span + 0.5));
  const v = Math.max(0.0, Math.min(0.999, northM / span + 0.5));
  const fx = u * 7.0;
  const fy = v * 7.0;
  const x0 = Math.min(6, Math.floor(fx));
  const y0 = Math.min(6, Math.floor(fy));
  const tx = fx - x0;
  const ty = fy - y0;

  const idx00 = (y0 * 8 + x0) * 4;
  const idx10 = (y0 * 8 + x0 + 1) * 4;
  const idx01 = ((y0 + 1) * 8 + x0) * 4;
  const idx11 = ((y0 + 1) * 8 + x0 + 1) * 4;

  const bilerpCh = (ch) => {
    const v00 = spatialCloudData[idx00 + ch];
    const v10 = spatialCloudData[idx10 + ch];
    const v01 = spatialCloudData[idx01 + ch];
    const v11 = spatialCloudData[idx11 + ch];
    return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
  };

  return {
    stratusCov: bilerpCh(0) / 255.0,
    fogTopM: (bilerpCh(1) / 255.0) * 2500.0,
    midHighCov: bilerpCh(2) / 255.0,
    tauNorm: bilerpCh(3) / 255.0,
  };
}

// Evaluates 3D optical extinction coefficient sigma_ext(x, y, z) in km^-1 across:
// 1. Topographically dammed 3D Marine Stratus / Valley Fog slab
// 2. Exact 3D Cumulus (850 hPa) & Altocumulus (700 hPa) volumetric billow ellipsoids
// 3. Carved 3D Crepuscular Cloud Gaps (where extinction drops to ~0 so sunbeams punch through)
function sample3DCloudExtinction(worldX, worldY, worldZ) {
  if (!state.showAtmosphere) return 0.0;
  const eastM = worldX;
  const northM = -worldZ;
  let sigmaKm = 0.0;

  // Check if (worldX, worldZ) lies inside any carved 3D crepuscular cloud gap fenestration
  let gapClearance = 0.0;
  for (let g = 0; g < activeCloudGaps.length; g++) {
    const gap = activeCloudGaps[g];
    const dGap = Math.hypot(worldX - gap.wx, worldZ - gap.wz);
    if (dGap < gap.radiusM) {
      const hole = 1.0 - Math.min(1.0, Math.max(0.0, (dGap - gap.radiusM * 0.35) / (gap.radiusM * 0.65)));
      gapClearance = Math.max(gapClearance, hole);
    }
  }

  // 1. Marine Stratus / Valley Inversion Slab (with 15m windward ridge damming)
  const sb = activeStratusBand;
  const sc = sampleBilinearSpatialCloud(eastM, northM);
  if (worldY >= sb.baseLocalY - 20 && worldY <= sb.topLocalY + 240) {
    let localTopY = Math.max(25, sc.fogTopM - sb.minElevM);
    let stratusDensity = Math.max(sb.mistCov * 0.65, sc.stratusCov, sc.tauNorm * 0.85);

    const upX = worldX - sb.windEast * 1100;
    const upZ = worldZ + sb.windNorth * 1100;
    const upTerrainY = sampleLocalTerrainY(upX, -upZ);
    if (upTerrainY > localTopY + 18) {
      const dam = Math.min(1.0, (upTerrainY - localTopY) / 140.0);
      stratusDensity *= 1.0 - 0.82 * dam;
      localTopY -= 55 * dam;
    }

    if (worldY <= localTopY + 120 && stratusDensity > 0.12) {
      const cellWave =
        0.5 +
        0.5 *
          Math.sin(worldX * 0.0014 + worldZ * 0.0011) *
          Math.cos(worldX * 0.0009 - worldZ * 0.0015);
      if (cellWave > 1.0 - stratusDensity) {
        sigmaKm += stratusDensity * (0.65 + 0.55 * cellWave) * 2.1;
      }
    }
  }

  // 2. Exact 3D Cumulus & Altocumulus Billow Ellipsoids in cloudClustersGroup
  for (let i = 0; i < activeCloudEllipsoids.length; i++) {
    const c = activeCloudEllipsoids[i];
    const dx = (worldX - c.x) / c.rx;
    const dy = (worldY - c.y) / c.ry;
    const dz = (worldZ - c.z) / c.rz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < 1.0) {
      const coreWeight = 1.0 - d2;
      sigmaKm += c.sigmaKm * coreWeight;
    }
  }

  return sigmaKm * (1.0 - gapClearance * 0.94);
}

// Integrates Beer-Lambert transmittance T = exp(-int sigma_ext ds) from a 3D point toward the Sun,
// and checks coaxial alignment with any 3D crepuscular cloud hole along CLOUD_SLANT_FACTOR = 1.65!
function integrateSolarCloudTransmittance(worldX, worldY, worldZ, sunDirWorld, checkTerrain = false) {
  if (!state.showAtmosphere) return 1.0;

  // Coaxial check: does the ray from (worldX, worldY, worldZ) pass up through any 3D crepuscular cloud hole?
  const horizLen = Math.max(0.01, Math.hypot(sunDirWorld.x, sunDirWorld.z));
  const ux = sunDirWorld.x / horizLen;
  const uz = sunDirWorld.z / horizLen;
  let maxGapPass = 0.0;
  for (let g = 0; g < activeCloudGaps.length; g++) {
    const gap = activeCloudGaps[g];
    if (worldY < gap.wy + 120) {
      const dy = Math.max(40.0, gap.wy - worldY);
      const hitX = worldX + ux * dy * CLOUD_SLANT_FACTOR;
      const hitZ = worldZ + uz * dy * CLOUD_SLANT_FACTOR;
      const dGap = Math.hypot(hitX - gap.wx, hitZ - gap.wz);
      if (dGap < gap.radiusM * 1.05) {
        const spot = 1.0 - Math.min(1.0, Math.max(0.0, (dGap - gap.radiusM * 0.25) / (gap.radiusM * 0.80)));
        maxGapPass = Math.max(maxGapPass, spot);
      }
    }
  }

  let tau = 0.0;
  const stepM = 240.0;
  const stepKm = stepM / 1000.0;
  const halfSpan = (state.terrainPatch?.span_m || 9200) * 0.485;

  for (let k = 1; k <= 12; k++) {
    const dist = k * stepM;
    const sx = worldX + sunDirWorld.x * dist;
    const sy = worldY + sunDirWorld.y * dist;
    const sz = worldZ + sunDirWorld.z * dist;

    if (checkTerrain && Math.abs(sx) <= halfSpan && Math.abs(sz) <= halfSpan) {
      const ty = sampleLocalTerrainY(sx, -sz);
      if (sy <= ty + 4.0) {
        return 0.04; // Occluded by uprange mountain peak!
      }
    }
    const sigma = sample3DCloudExtinction(sx, sy, sz);
    tau += sigma * stepKm;
    if (tau > 3.6) break;
  }

  // Also include local 12-level vertical optical depth when below the main cloud deck
  if (worldY < 720) {
    const hitX = worldX + ux * Math.max(60, 640 - worldY) * CLOUD_SLANT_FACTOR;
    const hitZ = worldZ + uz * Math.max(60, 640 - worldY) * CLOUD_SLANT_FACTOR;
    const scHit = sampleBilinearSpatialCloud(hitX, -hitZ);
    tau = Math.max(tau, scHit.tauNorm * 4.2 * (1.0 - maxGapPass * 0.92));
  }

  const beerTrans = Math.exp(-tau * 0.82);
  const blended = beerTrans * (1.0 - maxGapPass) + 0.96 * maxGapPass;
  return Math.max(0.04, Math.min(1.0, blended));
}

// ============================================================================
// Valley-Wide 3D Solar Light Field (Full-View Terrain + 3D Cloud Beer-Lambert Lattice)
// ============================================================================
function updateValleyLightField(sunDirWorld, elevationDeg) {
  activeGodRayMaterials = [];
  disposeObjectTree(lightFieldGroup);

  if (!state.terrainPatch || !state.showLightField || elevationDeg <= -2.5) {
    lightFieldGroup.visible = false;
    return;
  }
  lightFieldGroup.visible = true;

  const propDir = sunDirWorld.clone().negate(); // Sun -> Valley propagation vector
  const rightVec = new THREE.Vector3(-sunDirWorld.z, 0, sunDirWorld.x).normalize();
  const halfSpan = state.terrainPatch.span_m * 0.475;
  const isOvercastWithGaps = state.showAtmosphere && activeMeanCloudTau >= 1.35;

  // Color palette shifts smoothly from Dawn/Dusk Blue-Hour Cyan-Amber -> Golden Hour Gold -> Daylight Warm White
  const isBlueHour = elevationDeg < 1.2;
  const isGolden = elevationDeg <= 13.0;
  const baseR = isBlueHour ? 0.56 : isGolden ? 0.99 : 0.98;
  const baseG = isBlueHour ? 0.74 : isGolden ? 0.76 : 0.93;
  const baseB = isBlueHour ? 0.98 : isGolden ? 0.30 : 0.68;

  const grazeR = isBlueHour ? 0.96 : 1.0;
  const grazeG = isBlueHour ? 0.58 : 0.80;
  const grazeB = isBlueHour ? 0.36 : 0.32;

  const positions = [];
  const colors = [];
  const areaPositions = [];
  const areaColors = [];

  const pushSegment = (ax, ay, az, ar, ag, ab, aa, bx, by, bz, br, bg, bb, ba) => {
    positions.push(ax, ay, az, bx, by, bz);
    colors.push(ar * aa, ag * aa, ab * aa, br * ba, bg * ba, bb * ba);
  };

  const pushAreaQuad = (
    ax, ay, az, ar, ag, ab, aa,
    bx, by, bz, br, bg, bb, ba,
    cx, cy, cz, cr, cg, cb, ca,
    dx, dy, dz, dr, dg, db, da
  ) => {
    areaPositions.push(
      ax, ay, az, bx, by, bz, cx, cy, cz,
      ax, ay, az, cx, cy, cz, dx, dy, dz
    );
    areaColors.push(
      ar * aa, ag * aa, ab * aa,
      br * ba, bg * ba, bb * ba,
      cr * ca, cg * ca, cb * ca,
      ar * aa, ag * aa, ab * aa,
      cr * ca, cg * ca, cb * ca,
      dr * da, dg * da, db * da
    );
  };

  let prevLaneCrests = [];

  // 35 high-density lateral lanes spanning the entire 8.2 km width of the terrain patch (-4080m to +4080m)
  for (let latOff = -4080; latOff <= 4080; latOff += 240) {
    const laneCenter = rightVec.clone().multiplyScalar(latOff);
    const currLaneCrests = [];

    // Part 1: Detect EVERY shadow-producing ridge/bluff crest along this solar lane across the ENTIRE DEM
    // and light up every shadow-producing ray BY AREA (modulated by 3D cloud transmittance so solid clouds
    // physically occlude rays except where a crepuscular cloud hole illuminates the ridge!)
    const numSamples = 76;
    const stepD = (halfSpan * 1.92) / numSamples;
    const dStart = -halfSpan * 0.96;
    const heights = [];
    const relScores = [];

    for (let i = 0; i <= numSamples; i++) {
      const d = dStart + i * stepD;
      const sx = laneCenter.x + propDir.x * d;
      const sz = laneCenter.z + propDir.z * d;
      const ty = sampleLocalTerrainY(sx, -sz);
      heights.push({ d, sx, sz, ty });
      relScores.push(ty - d * propDir.y);
    }

    let runningMaxRel = -Infinity;
    for (let i = 2; i < numSamples - 2; i++) {
      const r = relScores[i];
      const isLocalCrest =
        r > relScores[i - 1] &&
        r >= relScores[i - 2] &&
        r >= relScores[i + 1] &&
        r > relScores[i + 2];
      const clearShadowDropBehind = heights[i].ty - heights[i + 2].ty > 26;

      if (isLocalCrest && clearShadowDropBehind && r > runningMaxRel - 18) {
        runningMaxRel = Math.max(runningMaxRel, r);
        const crest = heights[i];
        const originY = crest.ty + 5.0;

        // Incoming Beer-Lambert transmittance through 3D clouds to this ridge crest
        const crestCloudTrans = integrateSolarCloudTransmittance(
          crest.sx,
          originY,
          crest.sz,
          sunDirWorld,
          false
        );

        // Under thick overcast, if a ridge crest is completely in deep cloud shadow (< 0.16),
        // the sun ray was physically stopped by the 3D cloud deck above!
        if (isOvercastWithGaps && crestCloudTrans < 0.16) {
          continue;
        }

        const upDist = 850;
        const upX = crest.sx - propDir.x * upDist;
        const upY = originY - propDir.y * upDist;
        const upZ = crest.sz - propDir.z * upDist;
        pushSegment(
          upX, upY, upZ, baseR, baseG, baseB, 0.05 * crestCloudTrans,
          crest.sx, originY, crest.sz, grazeR, grazeG, grazeB, 0.44 * crestCloudTrans
        );

        let prevX = crest.sx;
        let prevY = originY;
        let prevZ = crest.sz;
        let prevAlpha = 0.44 * crestCloudTrans;
        let rayCloudTau = 0.0;
        let endX = crest.sx;
        let endY = originY;
        let endZ = crest.sz;
        const ribbonHeightM = 68.0;

        for (let marchD = 140; marchD <= 3600; marchD += 140) {
          const nx = crest.sx + propDir.x * marchD;
          const ny = originY + propDir.y * marchD;
          const nz = crest.sz + propDir.z * marchD;
          if (Math.abs(nx) > halfSpan || Math.abs(nz) > halfSpan) break;

          const localSigma = sample3DCloudExtinction(nx, ny, nz);
          rayCloudTau += localSigma * 0.14;
          const beerLambert = Math.exp(-rayCloudTau);
          const mieBoost = localSigma > 0.15 && rayCloudTau < 0.85 ? 1.32 : 1.0;

          const groundY = sampleLocalTerrainY(nx, -nz);
          const clearance = ny - groundY;
          if (clearance <= 2.5) {
            endX = nx;
            endY = groundY + 2.5;
            endZ = nz;
            const hitAlpha = 0.36 * crestCloudTrans * beerLambert;
            pushSegment(
              prevX, prevY, prevZ, grazeR, grazeG, grazeB, prevAlpha,
              endX, endY, endZ, grazeR, grazeG, grazeB, hitAlpha
            );
            pushAreaQuad(
              prevX, prevY, prevZ, grazeR, grazeG, grazeB, prevAlpha * 0.36,
              endX, endY, endZ, grazeR, grazeG, grazeB, hitAlpha * 0.36,
              endX, endY + ribbonHeightM * 0.45, endZ, baseR, baseG, baseB, 0.01,
              prevX, prevY + ribbonHeightM, prevZ, baseR, baseG, baseB, 0.02
            );
            break;
          }

          endX = nx;
          endY = ny;
          endZ = nz;
          const fade = Math.max(0.04, 0.38 * (1.0 - marchD / 3800) * crestCloudTrans * beerLambert * mieBoost);
          pushSegment(
            prevX, prevY, prevZ, grazeR, grazeG, grazeB, prevAlpha,
            nx, ny, nz, grazeR, grazeG, grazeB, fade
          );
          pushAreaQuad(
            prevX, prevY, prevZ, grazeR, grazeG, grazeB, prevAlpha * 0.34,
            nx, ny, nz, grazeR, grazeG, grazeB, fade * 0.34,
            nx, ny + ribbonHeightM, nz, baseR, baseG, baseB, 0.015,
            prevX, prevY + ribbonHeightM, prevZ, baseR, baseG, baseB, 0.015
          );

          prevX = nx;
          prevY = ny;
          prevZ = nz;
          prevAlpha = fade;
          if (beerLambert < 0.08) break;
        }

        const cRec = {
          d: crest.d,
          sx: crest.sx,
          sy: originY,
          sz: crest.sz,
          ex: endX,
          ey: endY,
          ez: endZ,
          trans: crestCloudTrans,
        };
        currLaneCrests.push(cRec);

        for (const pRec of prevLaneCrests) {
          if (Math.abs(pRec.d - cRec.d) < 560 && Math.abs(pRec.sy - cRec.sy) < 185) {
            const sheetAlpha = 0.16 * Math.min(pRec.trans, cRec.trans);
            const endAlpha = 0.08 * Math.min(pRec.trans, cRec.trans);
            pushAreaQuad(
              pRec.sx, pRec.sy, pRec.sz, grazeR, grazeG, grazeB, sheetAlpha,
              cRec.sx, cRec.sy, cRec.sz, grazeR, grazeG, grazeB, sheetAlpha,
              cRec.ex, cRec.ey, cRec.ez, baseR, baseG, baseB, endAlpha,
              pRec.ex, pRec.ey, pRec.ez, baseR, baseG, baseB, endAlpha
            );
            pushSegment(
              pRec.sx, pRec.sy, pRec.sz, grazeR, grazeG, grazeB, 0.52 * pRec.trans,
              cRec.sx, cRec.sy, cRec.sz, grazeR, grazeG, grazeB, 0.52 * cRec.trans
            );
            pushSegment(
              pRec.ex, pRec.ey, pRec.ez, grazeR, grazeG, grazeB, 0.44 * pRec.trans,
              cRec.ex, cRec.ey, cRec.ez, grazeR, grazeG, grazeB, 0.44 * cRec.trans
            );
            break;
          }
        }
      }
    }
    prevLaneCrests = currLaneCrests;

    // Part 2: Upper-atmosphere solar rays hitting the TOP of the 3D cloud deck and being occluded by cloud opacity!
    if (Math.abs(latOff) % 720 === 0) {
      const startY = 1350;
      let prevX = laneCenter.x + propDir.x * dStart;
      let prevZ = laneCenter.z + propDir.z * dStart;
      let prevY = startY - propDir.y * dStart * 0.35;
      let streamTau = 0.0;

      for (let i = 1; i <= numSamples; i++) {
        const d = dStart + i * stepD;
        const nx = laneCenter.x + propDir.x * d;
        const nz = laneCenter.z + propDir.z * d;
        const ny = prevY + propDir.y * stepD;
        if (Math.abs(nx) > halfSpan || Math.abs(nz) > halfSpan) break;

        const sigma = sample3DCloudExtinction(nx, ny, nz);
        const enteringCloud = sigma > 0.32 && streamTau < 0.22;
        streamTau += sigma * (stepD / 1000.0) * (isOvercastWithGaps ? 2.4 : 1.0);
        const trans = Math.exp(-streamTau);

        const ty = sampleLocalTerrainY(nx, -nz);
        if (ny <= ty + 3.0) {
          pushSegment(
            prevX, prevY, prevZ, baseR, baseG, baseB, 0.06 * trans,
            nx, ty + 3.0, nz, grazeR, grazeG, grazeB, 0.14 * trans
          );
          break;
        }

        const segAlpha = (enteringCloud ? 0.24 : 0.045) * trans;
        const segR = enteringCloud ? grazeR : baseR;
        const segG = enteringCloud ? grazeG : baseG;
        const segB = enteringCloud ? grazeB : baseB;
        pushSegment(
          prevX, prevY, prevZ, segR, segG, segB, segAlpha,
          nx, ny, nz, segR, segG, segB, segAlpha
        );
        prevX = nx;
        prevY = ny;
        prevZ = nz;
        if (trans < 0.05) break; // Ray is physically occluded inside the 3D cloud volume!
      }
    }
  }

  // Part 3: True 3D Crepuscular "God Rays" Peeking Through Carved 3D Cloud Holes (activeCloudGaps)
  // Every 3D hole in the thick cloud deck emits a focused bundle of volumetric sunbeam cones, area-lit curtains,
  // sharp boundary rays, and a glowing golden spotlight ring on the ocean/terrain at the EXACT coaxial slant angle!
  if (state.showAtmosphere && activeCloudGaps.length > 0 && elevationDeg > -1.5) {
    const horizLen = Math.max(0.01, Math.hypot(sunDirWorld.x, sunDirWorld.z));
    const slantDx = (-sunDirWorld.x / horizLen) * CLOUD_SLANT_FACTOR;
    const slantDz = (-sunDirWorld.z / horizLen) * CLOUD_SLANT_FACTOR;
    const perpX = -slantDz / CLOUD_SLANT_FACTOR;
    const perpZ = slantDx / CLOUD_SLANT_FACTOR;

    activeCloudGaps.forEach((gap, gIdx) => {
      const gapOrigin = new THREE.Vector3(gap.wx, gap.wy, gap.wz);
      const estGroundY = sampleLocalTerrainY(
        gap.wx + slantDx * gap.wy,
        -(gap.wz + slantDz * gap.wy)
      );
      const dropY = Math.max(80, gap.wy - (estGroundY + 3.0));
      const landX = gap.wx + slantDx * dropY;
      const landZ = gap.wz + slantDz * dropY;
      const landY = sampleLocalTerrainY(landX, -landZ) + 3.0;
      const landPt = new THREE.Vector3(landX, landY, landZ);

      // 1. Incoming upper solar feeder ray from the Sun striking into the top of the 3D cloud hole
      const upperEntry = gapOrigin.clone().addScaledVector(sunDirWorld, 2400);
      pushSegment(
        upperEntry.x, upperEntry.y, upperEntry.z, baseR, baseG, baseB, 0.14,
        gapOrigin.x, gapOrigin.y, gapOrigin.z, grazeR, grazeG, grazeB, 0.72
      );

      // 2. Volumetric 3D Crepuscular Sunbeam Cone punching down from the 3D cloud hole to the ocean/headland spotlight
      const beamVec = landPt.clone().sub(gapOrigin);
      const beamLen = beamVec.length();
      if (beamLen > 60) {
        const topR = Math.max(120, gap.radiusM * 0.34);
        const botR = Math.max(240, gap.radiusM * 0.76);
        const nowSec = performance.now() * 0.001;
        const windVec = activeWindDirENU;

        const coneGeo = new THREE.CylinderGeometry(topR, botR, beamLen, 24, 1, true);
        const coneMat = new THREE.ShaderMaterial({
          vertexShader: GODRAY_VERTEX_SHADER,
          fragmentShader: GODRAY_FRAGMENT_SHADER,
          uniforms: {
            uBeamColor: { value: new THREE.Color(isGolden ? 0xfbbf24 : 0xfde047) },
            uBaseOpacity: { value: isOvercastWithGaps ? 0.30 : 0.18 },
            uCloudTime: { value: nowSec },
            uWindDirENU: { value: windVec.clone() },
          },
          transparent: true,
          side: THREE.DoubleSide,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        activeGodRayMaterials.push(coneMat);
        const coneMesh = new THREE.Mesh(coneGeo, coneMat);
        coneMesh.position.copy(gapOrigin).addScaledVector(beamVec, 0.5);
        coneMesh.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, -1, 0),
          beamVec.clone().normalize()
        );
        lightFieldGroup.add(coneMesh);

        // Inner high-intensity core shaft inside the crepuscular cloud-gap cone
        const coreGeo = new THREE.CylinderGeometry(topR * 0.42, botR * 0.46, beamLen, 20, 1, true);
        const coreMat = new THREE.ShaderMaterial({
          vertexShader: GODRAY_VERTEX_SHADER,
          fragmentShader: GODRAY_FRAGMENT_SHADER,
          uniforms: {
            uBeamColor: { value: new THREE.Color(0xfef08a) },
            uBaseOpacity: { value: isOvercastWithGaps ? 0.38 : 0.22 },
            uCloudTime: { value: nowSec },
            uWindDirENU: { value: windVec.clone() },
          },
          transparent: true,
          side: THREE.DoubleSide,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        activeGodRayMaterials.push(coreMat);
        const coreMesh = new THREE.Mesh(coreGeo, coreMat);
        coreMesh.position.copy(coneMesh.position);
        coreMesh.quaternion.copy(coneMesh.quaternion);
        lightFieldGroup.add(coreMesh);
      }

      // 3. Fan of 9 terrain-checked crepuscular rays + lit area curtains across the cloud hole diameter
      let prevRayTop = null;
      let prevRayBot = null;
      for (let s = -4; s <= 4; s++) {
        const frac = s / 4.0;
        const topOff = frac * gap.radiusM * 0.42;
        const botOff = frac * gap.radiusM * 0.78;
        const rTop = new THREE.Vector3(
          gap.wx + perpX * topOff,
          gap.wy,
          gap.wz + perpZ * topOff
        );
        let rBot = new THREE.Vector3(
          landX + perpX * botOff,
          landY,
          landZ + perpZ * botOff
        );

        // Check if this individual ray in the crepuscular fan hits an intervening bluff/ridge on the way down
        for (let step = 1; step <= 18; step++) {
          const t = step / 18.0;
          const sx = rTop.x + (rBot.x - rTop.x) * t;
          const sy = rTop.y + (rBot.y - rTop.y) * t;
          const sz = rTop.z + (rBot.z - rTop.z) * t;
          if (Math.abs(sx) <= halfSpan && Math.abs(sz) <= halfSpan) {
            const ty = sampleLocalTerrainY(sx, -sz);
            if (sy <= ty + 3.0) {
              rBot = new THREE.Vector3(sx, ty + 3.0, sz);
              break;
            }
          }
        }

        const edgeBoost = Math.abs(s) === 4 || s === 0 ? 0.86 : 0.54;
        pushSegment(
          rTop.x, rTop.y, rTop.z, grazeR, grazeG, grazeB, edgeBoost,
          rBot.x, rBot.y, rBot.z, grazeR, grazeG, grazeB, edgeBoost * 0.88
        );

        if (prevRayTop && prevRayBot) {
          pushAreaQuad(
            prevRayTop.x, prevRayTop.y, prevRayTop.z, grazeR, grazeG, grazeB, 0.26,
            rTop.x, rTop.y, rTop.z, grazeR, grazeG, grazeB, 0.26,
            rBot.x, rBot.y, rBot.z, baseR, baseG, baseB, 0.18,
            prevRayBot.x, prevRayBot.y, prevRayBot.z, baseR, baseG, baseB, 0.18
          );
        }
        prevRayTop = rTop;
        prevRayBot = rBot;
      }

      // 4. Glowing golden spotlight perimeter ring on the ocean/terrain surface + 3D Cloud-Hole Rim Ring
      const ringPts = [];
      const holePts = [];
      const spotR = Math.max(240, gap.radiusM * 0.76);
      const holeR = Math.max(150, gap.radiusM * 0.45);
      for (let a = 0; a <= 28; a++) {
        const ang = (a / 28) * Math.PI * 2;
        const cx = landX + Math.cos(ang) * spotR;
        const cz = landZ + Math.sin(ang) * spotR;
        const cy = sampleLocalTerrainY(cx, -cz) + 4.5;
        ringPts.push(new THREE.Vector3(cx, cy, cz));
        holePts.push(
          new THREE.Vector3(
            gap.wx + Math.cos(ang) * holeR,
            gap.wy - 12,
            gap.wz + Math.sin(ang) * holeR
          )
        );
      }
      for (let a = 0; a < 28; a++) {
        pushSegment(
          ringPts[a].x, ringPts[a].y, ringPts[a].z, grazeR, grazeG, grazeB, 0.78,
          ringPts[a + 1].x, ringPts[a + 1].y, ringPts[a + 1].z, grazeR, grazeG, grazeB, 0.78
        );
        pushSegment(
          holePts[a].x, holePts[a].y, holePts[a].z, 1.0, 0.85, 0.38, 0.82,
          holePts[a + 1].x, holePts[a + 1].y, holePts[a + 1].z, 1.0, 0.85, 0.38, 0.82
        );
      }

      if (gIdx < 2) {
        const gapBadge = makeTextSprite(
          `☀️ CLOUD BREAK #${gIdx + 1} (${gap.transmittancePct}% Ray Pass • τ=${gap.tau})`,
          "#0f172a",
          "#fbbf24",
          "#fef3c7"
        );
        gapBadge.scale.set(440, 88, 1);
        gapBadge.position.set(gap.wx, gap.wy + 170, gap.wz);
        lightFieldGroup.add(gapBadge);
      }
    });
  }

  if (areaPositions.length > 0) {
    const areaGeo = new THREE.BufferGeometry();
    areaGeo.setAttribute("position", new THREE.Float32BufferAttribute(areaPositions, 3));
    areaGeo.setAttribute("color", new THREE.Float32BufferAttribute(areaColors, 3));
    const areaMat = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    lightFieldGroup.add(new THREE.Mesh(areaGeo, areaMat));
  }

  if (positions.length > 0) {
    const fieldGeo = new THREE.BufferGeometry();
    fieldGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    fieldGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const fieldMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    lightFieldGroup.add(new THREE.LineSegments(fieldGeo, fieldMat));
  }
}

// ============================================================================
// Interpolate Solar Position & Update Volumetric Sun-Ray Shafts + Sky Glow
// ============================================================================
function getSolarAnglesAtMinutes(localMinutes) {
  const track = state.shootReport?.daily_solar_track;
  if (!track || track.length === 0) {
    return { azimuthDeg: 259.1, elevationDeg: 0.1 };
  }
  if (localMinutes <= track[0][0]) {
    return { azimuthDeg: track[0][1], elevationDeg: track[0][2] };
  }
  const last = track[track.length - 1];
  if (localMinutes >= last[0]) {
    return { azimuthDeg: last[1], elevationDeg: last[2] };
  }
  for (let i = 0; i < track.length - 1; i++) {
    const [m0, az0, el0] = track[i];
    const [m1, az1, el1] = track[i + 1];
    if (localMinutes >= m0 && localMinutes <= m1) {
      const t = (localMinutes - m0) / Math.max(m1 - m0, 1e-5);
      return {
        azimuthDeg: az0 + t * (az1 - az0),
        elevationDeg: el0 + t * (el1 - el0),
      };
    }
  }
  return { azimuthDeg: 259.1, elevationDeg: 0.1 };
}

let _lastSpatialCloudKey = null;

function update3DPositionalClouds(sunDirWorld, elevationDeg, minElevM, fogBaseM, fogTopM, mistCov) {
  const wr = state.weatherReport;
  const scf = wr?.spatial_cloud_field;
  const windFromDeg = scf?.wind_dir_deg ?? 295.0;
  const windToRad = ((windFromDeg + 180.0) * Math.PI) / 180.0;
  const windEast = Math.sin(windToRad);
  const windNorth = Math.cos(windToRad);

  activeWindDirENU.set(windEast, windNorth);
  const slabTopM = Math.max(fogBaseM + 90.0, fogTopM + 35.0);
  activeFogTopLocalY = Math.max(30.0, fogTopM - minElevM);

  activeMeanCloudTau = scf?.mean_optical_depth_tau ?? 0.35;
  const rawGaps = scf?.crepuscular_gaps || [];
  const cumBaseAglM = minElevM < 200 ? 640 : 580;

  activeCloudGaps = rawGaps.slice(0, 6).map((g) => ({
    eastM: g.east_m,
    northM: g.north_m,
    wx: g.east_m,
    wz: -g.north_m,
    wy: cumBaseAglM,
    elevM: minElevM + cumBaseAglM,
    radiusM: g.radius_m || 950,
    tau: g.optical_depth_tau ?? 0.6,
    transmittancePct: g.transmittance_pct ?? 68,
  }));

  activeStratusBand = {
    baseLocalY: Math.max(10.0, fogBaseM - minElevM),
    topLocalY: Math.max(30.0, slabTopM - minElevM),
    mistCov,
    minElevM,
    windEast,
    windNorth,
  };

  if (terrainShaderMaterial) {
    const u = terrainShaderMaterial.uniforms;
    u.uWindDirENU.value.set(windEast, windNorth);
    u.uSpatialSpanM.value = scf?.span_m || 18400.0;
    u.uMeanCloudTau.value = activeMeanCloudTau;
    u.uNumCloudGaps.value = activeCloudGaps.length;
    u.uUseFluidShadows.value = state.showAtmosphere ? 1.0 : 0.0;
    u.uCloudDomainM.value = volumetricEngine.telemetry.domainXyM || 18000.0;
    if (u.uCloudGaps?.value) {
      for (let i = 0; i < 6; i++) {
        if (i < activeCloudGaps.length) {
          const g = activeCloudGaps[i];
          u.uCloudGaps.value[i].set(g.eastM, g.northM, g.elevM, g.radiusM);
        } else {
          u.uCloudGaps.value[i].set(0, 0, 600, 0);
        }
      }
    }
  }

  // Update 8x8 Sounding Texture used by optional analytical light-field diagnostics
  const cloudKey = scf
    ? `${wr.date}_${wr.phase}_${scf.cells?.length}_${scf.cells?.[0]?.stratus_pct}_${scf.mean_optical_depth_tau}_${minElevM.toFixed(0)}`
    : `fallback_${minElevM.toFixed(0)}`;

  if (cloudKey !== _lastSpatialCloudKey) {
    _lastSpatialCloudKey = cloudKey;
    activeCloudEllipsoids = [];
    if (scf?.cells && scf.cells.length === 64) {
      for (let i = 0; i < 64; i++) {
        const c = scf.cells[i];
        const midHigh = Math.max(
          c.cumulus_pct || 0,
          (c.altocumulus_pct || 0) * 0.85,
          (c.cirrus_pct || 0) * 0.45
        );
        const cellTau = c.optical_depth_tau ?? ((c.stratus_pct || 0) * 0.032 + midHigh * 0.024);
        spatialCloudData[i * 4 + 0] = Math.min(255, Math.max(0, Math.round(((c.stratus_pct || 0) / 100) * 255)));
        spatialCloudData[i * 4 + 1] = Math.min(255, Math.max(1, Math.round(((c.fog_top_m || 380) / 2500) * 255)));
        spatialCloudData[i * 4 + 2] = Math.min(255, Math.max(0, Math.round((midHigh / 100) * 255)));
        spatialCloudData[i * 4 + 3] = Math.min(255, Math.max(0, Math.round((Math.min(6.5, cellTau) / 6.5) * 255)));
      }
      spatialCloudTex.needsUpdate = true;
    }
  }
}

function updateSunRaysAndShader() {
  const { azimuthDeg, elevationDeg } = getSolarAnglesAtMinutes(state.currentMinutes);
  const azRad = (azimuthDeg * Math.PI) / 180;
  const elRad = (elevationDeg * Math.PI) / 180;

  const sunEast = Math.sin(azRad) * Math.cos(elRad);
  const sunNorth = Math.cos(azRad) * Math.cos(elRad);
  const sunUp = Math.sin(elRad);
  const sunDirWorld = new THREE.Vector3(sunEast, sunUp, -sunNorth).normalize();

  if (terrainShaderMaterial) {
    terrainShaderMaterial.uniforms.uSunDirENU.value.set(sunEast, sunNorth, sunUp).normalize();
    terrainShaderMaterial.uniforms.uHighlightZone.value = state.highlightLandingZone ? 1.0 : 0.0;
    terrainShaderMaterial.uniforms.uShowViewshed.value = state.showViewshed ? 1.0 : 0.0;
    terrainShaderMaterial.uniforms.uSurfaceStyle.value = state.surfaceStyle;
  }

  // Compute Lunar Ephemeris Vector for Nocturnal Cloud Scattering & Sky Dome Lunar Disk
  const astroSummary = state.shootReport?.astro_night_summary;
  const celTrack = astroSummary?.nightly_celestial_track || [];
  let moonAz = 240.0;
  let moonEl = -12.0;
  let moonIllum = (astroSummary?.moon_illumination_pct ?? 20.0) / 100.0;
  if (celTrack.length > 0) {
    let bestRow = celTrack[0];
    for (const row of celTrack) {
      if (Math.abs(row[0] - state.currentMinutes) < Math.abs(bestRow[0] - state.currentMinutes)) {
        bestRow = row;
      }
    }
    moonAz = bestRow[1] ?? 240.0;
    moonEl = bestRow[2] ?? -12.0;
  }
  const mAzRad = (moonAz * Math.PI) / 180;
  const mElRad = (moonEl * Math.PI) / 180;
  const moonDirWorld = new THREE.Vector3(
    Math.sin(mAzRad) * Math.cos(mElRad),
    Math.sin(mElRad),
    -Math.cos(mAzRad) * Math.cos(mElRad)
  ).normalize();

  // Update Physical Sky Dome (keep sky dome visible even if Volumetric Clouds toggle is off)
  atmosphereGroup.visible = true;
  skyDomeMat.uniforms.uSunDirWorld.value.copy(sunDirWorld);
  skyDomeMat.uniforms.uSunElDeg.value = elevationDeg;
  skyDomeMat.uniforms.uMoonDirWorld.value.copy(moonDirWorld);
  skyDomeMat.uniforms.uMoonElDeg.value = moonEl;
  skyDomeMat.uniforms.uMoonIllum.value = moonIllum;

  // Sync Solar & Lunar Vectors to the GPU WebGL2 Volumetric Raymarcher (0–12.2 km column + 3D Cirrus)
  volumetricEngine.updateCelestialUniforms({
    sunDirWorld,
    sunElDeg: elevationDeg,
    moonDirWorld,
    moonElDeg: moonEl,
    moonIllum,
  });

  const wr = state.weatherReport;
  const inv = wr?.inversion_model;
  const mistCov = wr
    ? Math.max(
        0.14,
        ((wr.valley_inversion_fog_pct || 25) * 0.8 +
          (wr.cloud_low_pct || 15) * 0.3 +
          (inv?.cloud_sea_score || 0) * 0.25) /
          110
      )
    : 0.24;

  if (state.terrainPatch) {
    const minElevM = state.terrainPatch.min_elevation_m ?? 0.0;
    const fogTopFt = inv?.fog_top_ft ?? (wr?.lcl_cloud_base_ft ? wr.lcl_cloud_base_ft - 600 : 1250);
    const fogBaseFt = inv?.fog_base_ft ?? Math.max(100, fogTopFt - 650);
    const fogTopM = fogTopFt / 3.28084;
    const fogBaseM = fogBaseFt / 3.28084;

    update3DPositionalClouds(sunDirWorld, elevationDeg, minElevM, fogBaseM, fogTopM, mistCov);

    if (terrainShaderMaterial) {
      terrainShaderMaterial.uniforms.uFogBaseM.value = fogBaseM;
      terrainShaderMaterial.uniforms.uFogTopM.value = fogTopM;
      terrainShaderMaterial.uniforms.uFogStrength.value = state.showAtmosphere
        ? Math.min(0.92, mistCov * 1.1)
        : 0.0;
    }
  }

  // Match scene background & far-field fog to the physical sky horizon phase (Night / Blue Hour / Golden Hour / Day)
  // and modulate volumetric Mie scattering density by live CAMS Aerosol Optical Depth (AOD 550nm)
  const aodVal = wr?.astro_atmosphere?.aod_550nm ?? 0.06;
  scene.fog.density = 0.000007 + Math.min(0.25, aodVal) * 0.000032;

  if (elevationDeg < -7.0) {
    scene.background.setHex(0x04070e);
    scene.fog.color.setHex(aodVal > 0.16 ? 0x161422 : 0x070d1b);
  } else if (elevationDeg <= 1.5) {
    scene.background.setHex(0x0d2147);
    scene.fog.color.setHex(0x1b3668);
  } else if (elevationDeg <= 12.0) {
    scene.background.setHex(0x152c55);
    scene.fog.color.setHex(0x2d466e);
  } else {
    scene.background.setHex(0x214c86);
    scene.fog.color.setHex(0x4a72a8);
  }

  // Rebuild Valley-Wide 3D Solar Light Field when enabled
  updateValleyLightField(sunDirWorld, elevationDeg);

  // Rebuild optional analytical sun-ray shafts & sky corridor gates (only when visible!)
  disposeObjectTree(sunBeamsGroup);

  const astroDisclosureEl = document.getElementById("astro-disclosure");
  const shouldBuildSunBeams =
    state.showSunBeams || Boolean(weatherDisclosure?.open) || Boolean(astroDisclosureEl?.open);

  if (shouldBuildSunBeams && state.shootReport && state.terrainPatch && elevationDeg > -1.0) {
    const lm = state.shootReport.target_landmark;
    const targetWorld = latLonElevToWorld(lm.upper_face_lat, lm.upper_face_lon, lm.upper_face_z_m);
    const SUN_DIST_M = 34000;
    const isGolden = elevationDeg <= 12.5;

    // Terrain-Occluded Crepuscular Ray-Caster (analytical diagnostic overlay)
    const propDir = sunDirWorld.clone().negate();
    const rightVec = new THREE.Vector3(-sunDirWorld.z, 0, sunDirWorld.x).normalize();
    const halfSpan = state.terrainPatch.span_m * 0.485;
    const beamColorHex = isGolden ? 0xfbbf24 : 0xfde047;
    const edgeColorHex = isGolden ? 0xf59e0b : 0xfef08a;

    const isOvercast = state.showAtmosphere && activeMeanCloudTau >= 1.35;

    // A. Blocked rays that originate at the distant 34 km Sun:
    //    When thick 3D clouds are present, these rays physically terminate on the TOP surface of the 3D cloud deck!
    const blockedLateralOffsets = [-160, 0, 160];
    blockedLateralOffsets.forEach((latOff) => {
      const aimPt = targetWorld
        .clone()
        .addScaledVector(rightVec, latOff)
        .add(new THREE.Vector3(0, isOvercast ? 680 : -145, 0));
      const srcPt = aimPt.clone().addScaledVector(sunDirWorld, SUN_DIST_M);
      let impactPt = aimPt.clone();
      if (!isOvercast) {
        for (let d = -1200; d <= 200; d += 25) {
          const p = aimPt.clone().addScaledVector(propDir, d);
          const ty = sampleLocalTerrainY(p.x, -p.z);
          if (p.y <= ty + 3.0) {
            impactPt = new THREE.Vector3(p.x, ty + 3.0, p.z);
            break;
          }
        }
      }
      const blockedCurve = new THREE.LineCurve3(srcPt, impactPt);
      const blockedGeo = new THREE.TubeGeometry(blockedCurve, 8, 9, 8, false);
      const blockedMat = new THREE.MeshBasicMaterial({
        color: beamColorHex,
        transparent: true,
        opacity: isOvercast ? 0.11 : 0.16,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      sunBeamsGroup.add(new THREE.Mesh(blockedGeo, blockedMat));
    });

    // B. Sample 37 high-density lateral slices across the monolith & flanking ridges to find the exact 3D grazing silhouette
    const sliceOffsets = [];
    for (let s = -1260; s <= 1260; s += 70) {
      sliceOffsets.push(s);
    }

    const grazingSlices = sliceOffsets.map((latOff) => {
      const sliceBase = targetWorld.clone().addScaledVector(rightVec, latOff);
      let bestRelativeScore = -Infinity;
      let crestPt = sliceBase.clone();

      // Search along the solar ray path near the monolith (-650m to +650m) for the highest horizon-occluding edge
      for (let d = -650; d <= 650; d += 32) {
        const samplePos = sliceBase.clone().addScaledVector(propDir, d);
        if (Math.abs(samplePos.x) > halfSpan || Math.abs(samplePos.z) > halfSpan) continue;
        const ty = sampleLocalTerrainY(samplePos.x, -samplePos.z);
        // Score relative to the sloping solar ray plane so we find the true tangency/grazing crest
        const relScore = ty - d * propDir.y;
        if (relScore > bestRelativeScore) {
          bestRelativeScore = relScore;
          crestPt = new THREE.Vector3(samplePos.x, ty, samplePos.z);
        }
      }

      // Start the grazing ray 7m above the silhouette edge, originating from the distant 34 km Sun, and march 4,300m PAST the interfering object
      const grazeOrigin = crestPt.clone().add(new THREE.Vector3(0, 7, 0));
      const uprangeSource = grazeOrigin.clone().addScaledVector(sunDirWorld, SUN_DIST_M);
      let leewardEnd = grazeOrigin.clone().addScaledVector(propDir, 4200);
      let touchedLeewardTerrain = false;

      for (let d = 60; d <= 4200; d += 40) {
        const p = grazeOrigin.clone().addScaledVector(propDir, d);
        if (Math.abs(p.x) > halfSpan || Math.abs(p.z) > halfSpan) {
          leewardEnd = p;
          break;
        }
        const ty = sampleLocalTerrainY(p.x, -p.z);
        if (p.y <= ty + 2.5) {
          leewardEnd = new THREE.Vector3(p.x, ty + 3.0, p.z);
          touchedLeewardTerrain = true;
          break;
        }
      }

      const cloudTrans = integrateSolarCloudTransmittance(
        grazeOrigin.x,
        grazeOrigin.y,
        grazeOrigin.z,
        sunDirWorld,
        false
      );

      return {
        latOff,
        crestPt: grazeOrigin,
        uprangeSource,
        leewardEnd,
        touchedLeewardTerrain,
        cloudTrans,
      };
    });

    // C. Render sharp post-occlusion crepuscular lines, vertical & horizontal lit area sheets, and terminator border
    const ribbonPositions = [];
    const verticalCurtainPositions = [];
    const crestLinePts = [];
    const terminatorLinePts = [];
    let sumCloudTrans = 0.0;

    grazingSlices.forEach((slice, idx) => {
      crestLinePts.push(slice.crestPt);
      terminatorLinePts.push(slice.leewardEnd);
      const rawTrans = slice.cloudTrans ?? 1.0;
      sumCloudTrans += rawTrans;
      const ct = isOvercast ? rawTrans : (0.25 + 0.75 * rawTrans);

      // 1. Sharp crepuscular shadow-boundary line cast from the grazing rock edge into the leeward valley
      const edgeLineGeo = new THREE.BufferGeometry().setFromPoints([
        slice.crestPt,
        slice.leewardEnd,
      ]);
      const isCenterOrFlank = idx % 2 === 0 || Math.abs(slice.latOff) <= 280;
      const edgeLineMat = new THREE.LineBasicMaterial({
        color: isCenterOrFlank ? edgeColorHex : beamColorHex,
        transparent: true,
        opacity: (isCenterOrFlank ? 0.82 : 0.48) * ct,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      sunBeamsGroup.add(new THREE.Line(edgeLineGeo, edgeLineMat));

      // 2. Vertical lit-area curtain along the shadow-producing ray (highlights the sunlit plane directly above the shadow boundary)
      if (idx % 2 === 0 && ct > 0.14) {
        const a = slice.crestPt;
        const b = slice.leewardEnd;
        const hTop = 85;
        verticalCurtainPositions.push(
          a.x, a.y, a.z,
          b.x, b.y, b.z,
          b.x, b.y + hTop * 0.55, b.z,
          a.x, a.y, a.z,
          b.x, b.y + hTop * 0.55, b.z,
          a.x, a.y + hTop, a.z
        );
      }

      // 3. Full continuous volumetric shaft (only when not blocked by solid 3D overcast deck)
      if (idx % 3 === 0 && ct > 0.22) {
        const fullCurve = new THREE.LineCurve3(slice.uprangeSource, slice.leewardEnd);
        const radius = slice.latOff === 0 ? 15 : 9;
        const shaftGeo = new THREE.TubeGeometry(fullCurve, 16, radius, 8, false);
        const shaftMat = new THREE.MeshBasicMaterial({
          color: beamColorHex,
          transparent: true,
          opacity: (slice.latOff === 0 ? 0.28 : 0.14) * ct,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        sunBeamsGroup.add(new THREE.Mesh(shaftGeo, shaftMat));
      }

      // 4. Build continuous crepuscular light-boundary area sheet behind the interfering object between adjacent slices
      if (idx > 0 && ct > 0.14) {
        const prev = grazingSlices[idx - 1];
        const a = prev.crestPt;
        const b = slice.crestPt;
        const c = slice.leewardEnd;
        const d = prev.leewardEnd;
        ribbonPositions.push(
          a.x, a.y, a.z,
          b.x, b.y, b.z,
          c.x, c.y, c.z,
          a.x, a.y, a.z,
          c.x, c.y, c.z,
          d.x, d.y, d.z
        );
      }
    });

    const meanSliceCt = sumCloudTrans / Math.max(1, grazingSlices.length);
    const sheetCtScale = isOvercast ? Math.max(0.18, meanSliceCt) : 1.0;

    if (ribbonPositions.length > 0) {
      const ribbonGeo = new THREE.BufferGeometry();
      ribbonGeo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(ribbonPositions, 3)
      );
      const ribbonMat = new THREE.MeshBasicMaterial({
        color: beamColorHex,
        transparent: true,
        opacity: 0.26 * sheetCtScale,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      sunBeamsGroup.add(new THREE.Mesh(ribbonGeo, ribbonMat));
    }

    if (verticalCurtainPositions.length > 0) {
      const vcGeo = new THREE.BufferGeometry();
      vcGeo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(verticalCurtainPositions, 3)
      );
      const vcMat = new THREE.MeshBasicMaterial({
        color: edgeColorHex,
        transparent: true,
        opacity: 0.18 * sheetCtScale,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      sunBeamsGroup.add(new THREE.Mesh(vcGeo, vcMat));
    }

    // D. Glowing silhouette rim line along the occluding ridge/cliff crest + sharp ground shadow-terminator line
    if (crestLinePts.length >= 2) {
      const rimGeo = new THREE.BufferGeometry().setFromPoints(crestLinePts);
      const rimMat = new THREE.LineBasicMaterial({
        color: 0xfbbf24,
        transparent: true,
        opacity: 0.92,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      sunBeamsGroup.add(new THREE.Line(rimGeo, rimMat));
    }

    if (terminatorLinePts.length >= 2) {
      const termGeo = new THREE.BufferGeometry().setFromPoints(terminatorLinePts);
      const termMat = new THREE.LineBasicMaterial({
        color: 0xf59e0b,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      sunBeamsGroup.add(new THREE.Line(termGeo, termMat));
    }

    // E. If the user clicked a custom tripod point on the terrain, trace the exact solar ray
    //    passing over any uprange interfering object (e.g. Half Dome) toward that clicked point
    if (!inspectorCard.hidden && state.customTripodReport) {
      const rep = state.customTripodReport;
      const pinWorld = latLonElevToWorld(rep.tripod_lat, rep.tripod_lon, rep.tripod_elevation_m);
      let maxObstructionScore = -Infinity;
      let occluderCrest = null;

      // March uprange from the clicked pin toward the sun to find the highest interfering ridge/edge
      for (let d = 80; d <= 3800; d += 30) {
        const uPt = pinWorld.clone().addScaledVector(sunDirWorld, d);
        if (Math.abs(uPt.x) > halfSpan || Math.abs(uPt.z) > halfSpan) break;
        const ty = sampleLocalTerrainY(uPt.x, -uPt.z);
        const rayHeightAtD = pinWorld.y + d * sunDirWorld.y;
        const excess = ty - rayHeightAtD;
        if (excess > maxObstructionScore) {
          maxObstructionScore = excess;
          occluderCrest = new THREE.Vector3(uPt.x, ty + 6, uPt.z);
        }
      }

      if (occluderCrest) {
        const grazeSrc = occluderCrest.clone().addScaledVector(sunDirWorld, SUN_DIST_M);
        const grazePastPin = occluderCrest.clone().addScaledVector(propDir, 3600);
        const customRayGeo = new THREE.BufferGeometry().setFromPoints([
          grazeSrc,
          occluderCrest,
          grazePastPin,
        ]);
        const customRayMat = new THREE.LineBasicMaterial({
          color: 0x34d399,
          transparent: true,
          opacity: 0.9,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        sunBeamsGroup.add(new THREE.Line(customRayGeo, customRayMat));
      }
    }

    // F. Render 3D Weather Corridor Cloud Gates (75 km Entry, 18 km Keyhole, 0 km Fog/Inversion Deck)
    //    along the 34 km solar vector whenever Atmosphere or the Weather Forecast drawer is active
    const stations = state.weatherReport?.ray_path_stations;
    if ((state.showAtmosphere || weatherDisclosure?.open) && stations && stations.length === 3) {
      const st75 = stations[0];
      const st18 = stations[1];
      const st0 = stations[2];

      // 18 km Keyhole Gate placed at 10.5 km along the 34 km solar corridor
      const gate18Pos = targetWorld.clone().addScaledVector(sunDirWorld, 10500);
      const gate18RingGeo = new THREE.RingGeometry(420, 780, 32);
      const gate18RingMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const gate18Ring = new THREE.Mesh(gate18RingGeo, gate18RingMat);
      gate18Ring.position.copy(gate18Pos);
      gate18Ring.lookAt(targetWorld);
      sunBeamsGroup.add(gate18Ring);

      const gate18Sprite = makeTextSprite(
        `18km GATE: ${st18.cloud_low_pct}%L/${st18.cloud_mid_pct}%M/${st18.cloud_high_pct}%H (${st18.gate_transmittance_pct}% PASS)`,
        "#0f172a",
        "#38bdf8",
        "#e0f2fe"
      );
      gate18Sprite.scale.set(1150, 235, 1);
      gate18Sprite.position.copy(gate18Pos).add(new THREE.Vector3(0, 620, 0));
      sunBeamsGroup.add(gate18Sprite);

      // 75 km Horizon Entry Gate placed at 22.5 km along the 34 km solar corridor
      const gate75Pos = targetWorld.clone().addScaledVector(sunDirWorld, 22500);
      const gate75RingGeo = new THREE.RingGeometry(750, 1380, 32);
      const gate75RingMat = new THREE.MeshBasicMaterial({
        color: 0xf59e0b,
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const gate75Ring = new THREE.Mesh(gate75RingGeo, gate75RingMat);
      gate75Ring.position.copy(gate75Pos);
      gate75Ring.lookAt(targetWorld);
      sunBeamsGroup.add(gate75Ring);

      const gate75Sprite = makeTextSprite(
        `75km ENTRY: ${st75.cloud_low_pct}%L/${st75.cloud_mid_pct}%M/${st75.cloud_high_pct}%H (${st75.gate_transmittance_pct}% PASS)`,
        "#0f172a",
        "#f59e0b",
        "#fef3c7"
      );
      gate75Sprite.scale.set(1950, 400, 1);
      gate75Sprite.position.copy(gate75Pos).add(new THREE.Vector3(0, 1100, 0));
      sunBeamsGroup.add(gate75Sprite);

      // 0 km Marine Inversion / Fog Top 3D Altitude Marker
      if (inv) {
        const sign = inv.inversion_delta_t_f >= 0 ? "+" : "";
        const fogTopLocalY = activeFogTopLocalY;
        const fogBadgePos = new THREE.Vector3(
          targetWorld.x + sunDirWorld.x * 1350,
          fogTopLocalY + 95,
          targetWorld.z + sunDirWorld.z * 1350
        );
        const fogSprite = makeTextSprite(
          `FOG TOP ${inv.fog_top_ft.toLocaleString()} ft • ΔT ${sign}${inv.inversion_delta_t_f}°F • ${wr.temp_sunset_f}°F (${st0.cloud_high_pct}%H/${st0.cloud_low_pct}%L)`,
          "#0f172a",
          "#34d399",
          "#d1fae5"
        );
        fogSprite.scale.set(540, 110, 1);
        fogSprite.position.copy(fogBadgePos);
        sunBeamsGroup.add(fogSprite);
      }
    }
  }

  // Render 3D Upper-Atmosphere Jet Stream Streamlines, Aerosol Layer & Lunar/Galactic Vectors
  // when the user opens the Astrophotography disclosure card or enables Sun Beams at night
  const astroAtm = state.weatherReport?.astro_atmosphere;
  const astroGeo = state.shootReport?.astro_night_summary;
  if (
    shouldBuildSunBeams &&
    state.shootReport &&
    state.terrainPatch &&
    (astroDisclosureEl?.open || elevationDeg <= -0.5) &&
    astroAtm
  ) {
    const lm = state.shootReport.target_landmark;
    const targetWorld = latLonElevToWorld(lm.upper_face_lat, lm.upper_face_lon, lm.upper_face_z_m);

    // 1. Upper-Troposphere 250hPa Jet Stream Current Streamlines (y = targetWorld.y + 1850m)
    const jetY = targetWorld.y + 1850;
    const shearAmp = Math.min(180, Math.max(22, astroAtm.vertical_wind_shear_mph * 2.2));
    for (let lane = -900; lane <= 900; lane += 450) {
      const streamPts = [];
      for (let x = -3400; x <= 3400; x += 220) {
        const waveY = jetY + Math.sin(x * 0.0022 + lane * 0.01) * shearAmp;
        const waveZ = targetWorld.z + lane + Math.cos(x * 0.0018) * (shearAmp * 0.65);
        streamPts.push(new THREE.Vector3(targetWorld.x + x, waveY, waveZ));
      }
      const streamGeo = new THREE.BufferGeometry().setFromPoints(streamPts);
      const streamMat = new THREE.LineBasicMaterial({
        color: 0x818cf8,
        transparent: true,
        opacity: 0.55,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      sunBeamsGroup.add(new THREE.Line(streamGeo, streamMat));
    }

    const jetBadge = makeTextSprite(
      `250hPa JET STREAM (${astroAtm.jet_stream_250hpa_mph} MPH • SEEING ${astroAtm.seeing_score}/100)`,
      "#0f172a",
      "#818cf8",
      "#e0e7ff"
    );
    jetBadge.scale.set(460, 88, 1);
    jetBadge.position.set(targetWorld.x - 450, jetY + 150, targetWorld.z - 250);
    sunBeamsGroup.add(jetBadge);

    // 2. Crest-Level CAMS Aerosol Optical Depth (AOD 550nm) & Transparency Badge
    const hazeColor = astroAtm.aod_550nm < 0.12 ? "#34d399" : "#f59e0b";
    const hazeBadge = makeTextSprite(
      `UPPER HAZE / AOD ${astroAtm.aod_550nm} (${astroAtm.transparency_score}% TRANSPARENCY)`,
      "#0f172a",
      hazeColor,
      "#ecfdf5"
    );
    hazeBadge.scale.set(450, 86, 1);
    hazeBadge.position.set(targetWorld.x + 650, targetWorld.y + 820, targetWorld.z + 320);
    sunBeamsGroup.add(hazeBadge);

    // 3. 3D Galactic Core & Lunar Alpenglow / Moon Vectors at Night
    if (astroGeo) {
      const gcPeakAz = astroGeo.galactic_core_peak_az_deg ?? 205;
      const gcPeakEl = astroGeo.galactic_core_peak_el_deg ?? 14;
      const gcClearDeg = astroGeo.galactic_core_clearance_deg ?? 0;
      const gcAzRad = (gcPeakAz * Math.PI) / 180;
      const gcElRad = (Math.max(6, gcPeakEl) * Math.PI) / 180;
      const gcDirWorld = new THREE.Vector3(
        Math.sin(gcAzRad) * Math.cos(gcElRad),
        Math.sin(gcElRad),
        -Math.cos(gcAzRad) * Math.cos(gcElRad)
      ).normalize();
      const gcEnd = targetWorld.clone().addScaledVector(gcDirWorld, 3400);
      const gcCurve = new THREE.LineCurve3(targetWorld, gcEnd);
      const gcTube = new THREE.TubeGeometry(gcCurve, 12, 10, 8, false);
      const gcMat = new THREE.MeshBasicMaterial({
        color: 0xc084fc,
        transparent: true,
        opacity: 0.42,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      sunBeamsGroup.add(new THREE.Mesh(gcTube, gcMat));

      const gcSprite = makeTextSprite(
        `🌌 GALACTIC CORE (AZ ${gcPeakAz}° • +${gcClearDeg}° CLEAR)`,
        "#0f172a",
        "#c084fc",
        "#f3e8ff"
      );
      gcSprite.scale.set(450, 86, 1);
      gcSprite.position.copy(gcEnd).add(new THREE.Vector3(0, 90, 0));
      sunBeamsGroup.add(gcSprite);
    }
  }

  sunBeamsGroup.visible = shouldBuildSunBeams && state.cameraMode === "orbit";

  // Update unified bottom Scrubber & telemetry readouts
  const hhmm = formatMinutesToHHMM(state.currentMinutes);
  const comp = compassLabel(azimuthDeg);
  const elSign = elevationDeg >= 0 ? "+" : "";

  let phaseLabel = "Daylight High Sun";
  let cliffLitLabel = "82% Direct Light";
  if (elevationDeg < -0.5) {
    const astro = state.shootReport?.astro_night_summary;
    const track = astro?.nightly_celestial_track || [];
    let bestCel = null;
    for (const row of track) {
      if (!bestCel || Math.abs(row[0] - state.currentMinutes) < Math.abs(bestCel[0] - state.currentMinutes)) {
        bestCel = row;
      }
    }
    if (bestCel && astro) {
      const [, moonAz, moonEl, gcAz, gcEl] = bestCel;
      if (astro.galactic_core_visible && gcEl > 2.0) {
        phaseLabel = `Nightscape • Galactic Core Az ${gcAz.toFixed(1)}° (El +${gcEl.toFixed(1)}°)`;
      } else if (moonEl > 0.0) {
        phaseLabel = `Nightscape • ${astro.moon_phase_name} (${astro.moon_illumination_pct}%) Az ${moonAz.toFixed(1)}°`;
      } else {
        phaseLabel = "Deep Alpine Dark-Sky Window";
      }
      cliffLitLabel = moonEl > 2.0 ? `${astro.moon_illumination_pct}% Moonlit` : "Dark Sky (Stars)";
    } else {
      phaseLabel = "Blue Hour / Night Shadow";
      cliffLitLabel = "0% (Below Horizon)";
    }
  } else if (elevationDeg <= 13.5 && state.currentMinutes >= 900) {
    phaseLabel = "Sunset Golden Hour / Alpenglow";
    cliffLitLabel = "96% Alpenglow";
  } else if (elevationDeg <= 13.5 && state.currentMinutes < 720) {
    phaseLabel = "Sunrise Golden Hour";
    cliffLitLabel = "88% Rim Light";
  }

  document.getElementById("hud-local-time").textContent = `${hhmm} Local`;
  document.getElementById("hud-phase-badge").textContent = phaseLabel;
  document.getElementById("hud-sun-az").textContent = `${azimuthDeg.toFixed(1)}° ${comp}`;
  document.getElementById("hud-sun-el").textContent = `${elSign}${elevationDeg.toFixed(1)}°`;
  document.getElementById("hud-cliff-lit").textContent = cliffLitLabel;

  timeOutput.textContent = `${hhmm} • ${phaseLabel}`;
}

// ============================================================================
// Camera Mode Switching (3D Valley Orbit vs. VP1..VP3 or Custom Tripod Viewfinder)
// ============================================================================
function setCameraMode(mode) {
  state.cameraMode = mode;

  if (mode === "orbit") {
    if (btnOrbitMode) btnOrbitMode.hidden = true;
    if (hudClickModeGroup) hudClickModeGroup.hidden = false;
    viewportStageEl?.classList.remove("is-viewfinder");
    vfOverlay.hidden = true;
    trailsGroup.visible = state.showSightlines;
    customPinGroup.visible = true;
    camera.fov = 48;
    camera.updateProjectionMatrix();

    if (state.shootReport && state.terrainPatch) {
      const lm = state.shootReport.target_landmark;
      const targetPos = latLonElevToWorld(lm.summit_lat, lm.summit_lon, lm.summit_elevation_m);
      controls.target.copy(targetPos).add(new THREE.Vector3(0, 260, 0));
      camera.position.copy(targetPos).add(new THREE.Vector3(-3400, 1520, -3100));
      controls.update();
    }
    updateSunRaysAndShader();
    return;
  }

  if (btnOrbitMode) btnOrbitMode.hidden = false;
  if (hudClickModeGroup) hudClickModeGroup.hidden = true;
  viewportStageEl?.classList.add("is-viewfinder");
  trailsGroup.visible = false;
  customPinGroup.visible = false;
  sunBeamsGroup.visible = false;

  const lm = state.shootReport?.target_landmark;
  if (!lm) return;

  if (mode === "custom-tripod" && state.customTripodReport) {
    const rep = state.customTripodReport;
    const vpWorld = latLonElevToWorld(rep.tripod_lat, rep.tripod_lon, rep.tripod_elevation_m);
    const meshY = sampleLocalTerrainY(vpWorld.x, -vpWorld.z);
    vpWorld.y = Math.max(vpWorld.y, meshY) + 14;

    const targetWorld = latLonElevToWorld(lm.upper_face_lat, lm.upper_face_lon, lm.upper_face_z_m);
    camera.fov = 34;
    camera.updateProjectionMatrix();
    camera.position.copy(vpWorld);
    controls.target.copy(targetWorld);
    controls.update();

    const focalShort = (rep.sightline.recommended_focal_length_mm || "").split("(")[0].trim();
    vfOverlay.hidden = false;
    vfHudReadout.textContent = `CUSTOM TRIPOD · ${focalShort} · ${rep.sightline.bearing_to_landmark_deg}° ${rep.sightline.compass_direction} · ${rep.sightline.distance_to_landmark_miles} mi`;
    return;
  }

  // First-person tripod mode from VP1, VP2, or VP3
  const vpIdx = parseInt(mode.replace("vp-", ""), 10);
  state.selectedVpIndex = vpIdx;
  highlightSelectedVpCard();

  const vp = state.shootReport?.top_viewpoints?.[vpIdx];
  if (!vp) return;

  const vpWorld = latLonElevToWorld(
    vp.viewpoint_lat,
    vp.viewpoint_lon,
    vp.viewpoint_elevation_m
  );
  const meshY = sampleLocalTerrainY(vpWorld.x, -vpWorld.z);
  vpWorld.y = Math.max(vpWorld.y, meshY) + 14;

  const targetWorld = latLonElevToWorld(
    lm.upper_face_lat,
    lm.upper_face_lon,
    lm.upper_face_z_m
  );

  camera.fov = 34;
  camera.updateProjectionMatrix();
  camera.position.copy(vpWorld);
  controls.target.copy(targetWorld);
  controls.update();

  const vpShort = (vp.viewpoint_label || "").split("(")[0].trim();
  const focalShort = (vp.sightline.recommended_focal_length_mm || "").split("(")[0].trim();
  vfOverlay.hidden = false;
  vfHudReadout.textContent = `VP${vp.viewpoint_rank} · ${vpShort} · ${focalShort} · ${vp.sightline.bearing_to_landmark_deg}° ${vp.sightline.compass_direction} · ${vp.sightline.distance_to_landmark_miles} mi`;
  buildTrailsAndSightlines();
}

function selectViewpoint(idx, focusOrbit = true) {
  state.selectedVpIndex = idx;
  inspectorCard.hidden = true;
  disposeObjectTree(customPinGroup);
  highlightSelectedVpCard();
  buildTrailsAndSightlines();

  const vp = state.shootReport?.top_viewpoints?.[idx];
  const lm = state.shootReport?.target_landmark;
  if (focusOrbit && vp && lm) {
    state.cameraMode = "orbit";
    if (btnOrbitMode) btnOrbitMode.hidden = true;
    if (hudClickModeGroup) hudClickModeGroup.hidden = false;
    viewportStageEl?.classList.remove("is-viewfinder");
    vfOverlay.hidden = true;
    trailsGroup.visible = state.showSightlines;
    customPinGroup.visible = true;
    camera.fov = 48;
    camera.updateProjectionMatrix();

    const vpWorld = latLonElevToWorld(
      vp.viewpoint_lat,
      vp.viewpoint_lon,
      vp.viewpoint_elevation_m
    );
    const lmWorld = latLonElevToWorld(lm.summit_lat, lm.summit_lon, lm.summit_elevation_m);
    const midPoint = vpWorld.clone().lerp(lmWorld, 0.55);
    controls.target.copy(midPoint);
    camera.position.copy(vpWorld).add(new THREE.Vector3(-900, 1450, 1200));
    controls.update();
    updateSunRaysAndShader();
  }
}

// ============================================================================
// Render Sidebar Summary, Weather Disclosure & Collapsible VP1..VP3 Cards
// ============================================================================
function highlightSelectedVpCard() {
  document.querySelectorAll(".vp-card").forEach((el, idx) => {
    const isSel = idx === state.selectedVpIndex;
    el.classList.toggle("is-selected", isSel);
    el.setAttribute("aria-expanded", String(isSel));
  });
}

function renderWeatherCard(w) {
  if (!w) return;
  document.getElementById("weather-verdict").textContent = w.verdict;
  document.getElementById("weather-score-badge").textContent = `${w.alpenglow_score} / 100`;
  document.getElementById("weather-recommendation").textContent = w.recommendation;

  const summarySub = document.getElementById("weather-summary-sub");
  if (summarySub) {
    summarySub.textContent = `${w.ray_transmittance_pct}% Beam • ${w.temp_sunset_f}°F • Click to inspect 75 km ray corridor`;
  }

  document.getElementById("raypath-azimuth-label").textContent =
    `${w.phase} Ray Corridor: Az ${w.sun_azimuth_deg}° ${w.corridor_direction} (El +${w.sun_elevation_deg}°)`;
  document.getElementById("raypath-trans-pill").textContent =
    `End-to-End Beam: ${w.ray_transmittance_pct}%`;

  // Render SVG 3-Station Atmospheric Ray-Path Cross-Section (75 km -> 18 km -> 0 km)
  const svg = document.getElementById("raypath-svg");
  const stations = w.ray_path_stations || [];
  if (svg && stations.length === 3) {
    const [st75, st18, st0] = stations;
    svg.innerHTML = `
      <defs>
        <linearGradient id="beamGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#fde047" stop-opacity="0.95" />
          <stop offset="55%" stop-color="#fbbf24" stop-opacity="0.9" />
          <stop offset="100%" stop-color="#f97316" stop-opacity="0.95" />
        </linearGradient>
        <linearGradient id="cirrusGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#fbbf24" stop-opacity="0.22" />
          <stop offset="100%" stop-color="#ea580c" stop-opacity="0.58" />
        </linearGradient>
      </defs>

      <!-- High Cirrus Alpenglow Reflector Canvas (>20,000 ft) -->
      <rect x="210" y="7" width="138" height="12" rx="6" fill="url(#cirrusGrad)" stroke="#f59e0b" stroke-width="0.8" stroke-dasharray="3,2" />
      <text x="279" y="16" fill="#fde68a" font-size="7.5" font-weight="700" text-anchor="middle">
        0km High Cirrus Canvas (${st0.cloud_high_pct}%)
      </text>

      <!-- Mid/Low Cloud Decks at 75 km Entry & 18 km Ridge -->
      <rect x="14" y="24" width="76" height="11" rx="5" fill="rgba(148,163,184,0.25)" stroke="#64748b" stroke-width="0.8" />
      <text x="52" y="32" fill="#cbd5e1" font-size="7" text-anchor="middle">75km Deck (${st75.cloud_low_pct}%L/${st75.cloud_mid_pct}%M)</text>

      <rect x="128" y="31" width="74" height="11" rx="5" fill="rgba(56,189,248,0.18)" stroke="#38bdf8" stroke-width="0.8" />
      <text x="165" y="39" fill="#bae6fd" font-size="7" text-anchor="middle">18km Cap (${st18.cloud_low_pct}%L/${st18.cloud_mid_pct}%M)</text>

      <!-- Terrain Silhouette: 75 km Foothills -> 18 km Keyhole Ridge -> 0 km Monolith -->
      <path d="M 8 84 L 8 72 L 65 70 L 125 66 L 158 49 L 172 56 L 188 47 L 235 74 L 296 74 L 316 34 L 335 34 L 348 84 Z"
            fill="rgba(30,41,59,0.88)" stroke="#64748b" stroke-width="1.2" />

      <!-- Incoming Solar Ray Shaft from 75 km -> 18 km Keyhole -> 0 km Monolith Face -->
      <line x1="16" y1="42" x2="172" y2="52" stroke="url(#beamGrad)" stroke-width="3.2" />
      <line x1="172" y1="52" x2="316" y2="41" stroke="url(#beamGrad)" stroke-width="3.2" />
      <!-- Reflected Alpenglow Rays up onto Overhead High Cirrus Canvas -->
      <line x1="316" y1="41" x2="275" y2="19" stroke="#fb923c" stroke-width="1.3" stroke-dasharray="2,2" />

      <!-- Station Markers -->
      <circle cx="24" cy="42" r="4" fill="#fde047" />
      <circle cx="172" cy="52" r="4" fill="#38bdf8" />
      <circle cx="316" cy="41" r="4.5" fill="#f97316" />

      <text x="26" y="81" fill="#94a3b8" font-size="7.2" font-weight="700">75 km Entry</text>
      <text x="145" y="81" fill="#94a3b8" font-size="7.2" font-weight="700">18 km Keyhole</text>
      <text x="292" y="81" fill="#fde68a" font-size="7.2" font-weight="700">0 km Monolith</text>
    `;

    // Render 3-Column Station Cards
    const gridEl = document.getElementById("raypath-stations-grid");
    gridEl.innerHTML = stations
      .map((st, idx) => {
        const badgeText =
          idx === 2 ? `Canvas ${st.canvas_score || 90}%` : `${st.gate_transmittance_pct}% Pass`;
        return `
          <div class="station-mini-card">
            <div class="station-mini-top">
              <span>${st.distance_km} km</span>
              <span class="station-mini-pass">${badgeText}</span>
            </div>
            <div class="station-mini-sub" title="${st.region}">${st.name}</div>
            <div class="station-cloud-pills">
              <div><span>Ray Alt:</span> <strong>${st.ray_altitude_ft.toLocaleString()} ft</strong></div>
              <div><span>High/Mid:</span> <strong>${st.cloud_high_pct}% / ${st.cloud_mid_pct}%</strong></div>
              <div><span>Low Cloud:</span> <strong>${st.cloud_low_pct}%</strong></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  document.getElementById("val-lcl-base").textContent =
    `${(w.lcl_cloud_base_ft || 6850).toLocaleString()} ft ASL`;
  document.getElementById("val-visibility").textContent = `${w.visibility_miles} mi`;
  document.getElementById("val-inversion").textContent = `${w.valley_inversion_fog_pct}%`;
  document.getElementById("val-temp-wind").textContent = `${w.temp_sunset_f}°F • ${w.wind_mph} mph`;

  const inv = w.inversion_model;
  if (inv) {
    const hdrEl = document.getElementById("inversion-header-label");
    const clrEl = document.getElementById("inversion-clearance-pill");
    const bandEl = document.getElementById("val-fog-band");
    const dtEl = document.getElementById("val-inv-deltat");
    const csEl = document.getElementById("val-cloud-sea-score");
    const udEl = document.getElementById("val-under-deck-score");

    if (hdrEl) {
      const modeTag = inv.is_historical ? "Historical Archive" : "Live Sounding";
      hdrEl.textContent = inv.is_coastal
        ? `Marine Layer Inversion (${modeTag})`
        : `Valley Radiation Inversion (${modeTag})`;
    }
    if (clrEl) {
      const sign = inv.summit_clearance_ft >= 0 ? "+" : "";
      clrEl.textContent =
        inv.summit_clearance_ft >= 0
          ? `Peak +${inv.summit_clearance_ft.toLocaleString()} ft Above Fog`
          : `Peak ${sign}${inv.summit_clearance_ft.toLocaleString()} ft Inside Deck`;
    }
    if (bandEl) {
      bandEl.textContent = `${inv.fog_base_ft.toLocaleString()} – ${inv.fog_top_ft.toLocaleString()} ft ASL`;
    }
    if (dtEl) {
      const sign = inv.inversion_delta_t_f >= 0 ? "+" : "";
      dtEl.textContent = `${sign}${inv.inversion_delta_t_f}°F (${inv.temp_2m_f}°→${inv.temp_925hpa_f}°F)`;
    }
    if (csEl) {
      csEl.textContent = `${inv.cloud_sea_score} / 100`;
    }
    if (udEl) {
      udEl.textContent = `${inv.under_deck_sunset_score} / 100`;
    }

    const vpWeatherPill = document.getElementById("viewport-weather-pill");
    if (vpWeatherPill) {
      const sign = inv.inversion_delta_t_f >= 0 ? "+" : "";
      const scf = w.spatial_cloud_field;
      const gapCount = scf?.crepuscular_gaps?.length || 0;
      const tauVal = scf?.mean_optical_depth_tau;
      const aiEng = w.ai_ensemble?.selected_engine || w.ai_ensemble?.active_engine || state.weatherEngine;
      const engLabelMap = {
        ensemble: "⚡ Hybrid",
        graphcast: "🧠 GraphCast",
        hrrr: "🛰️ HRRR",
        ecmwf: "🌍 ECMWF",
        nowcast: "📡 Nowcast",
      };
      const engTag = engLabelMap[aiEng] ? `${engLabelMap[aiEng]} · ` : "";
      const tauTag =
        tauVal !== undefined
          ? ` • 3D τ=${tauVal} (${gapCount} Gap${gapCount === 1 ? "" : "s"})`
          : "";
      vpWeatherPill.textContent = `${engTag}🌤️ ${w.temp_sunset_f}°F · 💨 ${w.wind_mph} mph · ☀️ ${w.alpenglow_score}/100`;
      vpWeatherPill.title = `${engTag}${w.temp_sunset_f}°F • Wind ${w.wind_mph} mph • Fog ${inv.fog_base_ft.toLocaleString()}–${inv.fog_top_ft.toLocaleString()} ft (ΔT ${sign}${inv.inversion_delta_t_f}°F)${tauTag} — Click to inspect 75 km ray corridor`;
    }
  }

  // Render Multi-Model AI Weather Engine & 3D Atmospheric Feel Synthesizer (#ai-ensemble-box)
  const ai = w.ai_ensemble;
  if (ai) {
    const activeEng = ai.selected_engine || ai.active_engine || state.weatherEngine || "ensemble";
    const agreeEl = document.getElementById("ai-ensemble-agreement-pill");
    const archEl = document.getElementById("ai-feel-archetype");
    const noteEl = document.getElementById("ai-feel-note");
    const modelsGrid = document.getElementById("ai-models-comparison-grid");

    if (agreeEl && ai.ensemble_agreement_pct !== undefined) {
      agreeEl.textContent = `${ai.ensemble_agreement_pct}% Agreement`;
    }
    if (archEl && ai.atmospheric_feel_archetype) {
      archEl.textContent = `✨ ${ai.atmospheric_feel_archetype}`;
    }
    if (noteEl && ai.synthesis_note) {
      noteEl.textContent = ai.synthesis_note;
    }

    document.querySelectorAll("#ai-engine-selector-bar [data-weather-engine]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.weatherEngine === activeEng);
    });

    if (modelsGrid && Array.isArray(ai.models)) {
      modelsGrid.innerHTML = ai.models
        .slice(0, 4)
        .map((m) => {
          const isSel = m.id === activeEng;
          const selClass = isSel ? " is-selected-model" : "";
          const lowP = m.low ?? m.cloud_low_pct ?? 0;
          const midP = m.mid ?? m.cloud_mid_pct ?? 0;
          const highP = m.high ?? m.cloud_high_pct ?? 0;
          const shortTitle = m.short_name || m.name;
          const resShort = (m.resolution_3d || m.resolution || "").split("×")[0].trim();
          return `
            <div class="station-mini-card${selClass}" data-select-engine="${m.id}" style="padding: 6px; cursor: pointer;" title="${m.provider || ""} — Click to simulate this model in 3D">
              <div class="station-mini-top" style="font-size: 0.65rem;">
                <span>${shortTitle}</span>
              </div>
              <div class="station-mini-sub" style="font-size: 0.62rem;">${resShort}</div>
              <div class="station-cloud-pills" style="font-size: 0.63rem;">
                <div><span>3D τ</span> <strong>${m.optical_depth_tau}</strong></div>
                <div><span>L/M/H</span> <strong>${lowP}/${midP}/${highP}%</strong></div>
                <div><span>Score</span> <strong>${m.alpenglow_score}/100</strong></div>
              </div>
            </div>
          `;
        })
        .join("");
    }
  }

  // Render 5-Station West -> East Positional 3D Cloud & Fog Transect
  const transectGrid = document.getElementById("spatial-transect-grid");
  const transectPill = document.getElementById("spatial-transect-pill");
  const scf = w.spatial_cloud_field;
  if (transectGrid && scf?.transect_west_to_east) {
    if (transectPill) {
      const gapCount = scf?.crepuscular_gaps?.length || 0;
      const tauStr = scf?.mean_optical_depth_tau !== undefined ? ` • τ=${scf.mean_optical_depth_tau} (${gapCount} Gaps)` : "";
      transectPill.textContent = `Wind ${Math.round(scf.wind_dir_deg || 295)}° • 8×8 Voxel${tauStr}`;
    }
    transectGrid.innerHTML = scf.transect_west_to_east
      .map((pt) => {
        const shortLoc = pt.label.replace("11km W ", "").replace("5.5km W ", "").replace("0km ", "").replace("5.5km E ", "").replace("11km E ", "");
        const distTag = pt.east_km === 0 ? "0km" : pt.east_km < 0 ? `${pt.east_km}km W` : `+${pt.east_km}km E`;
        const lowPct = pt.low_cloud_pct ?? pt.cloud_low_pct ?? 0;
        const midPct = pt.mid_cloud_pct ?? pt.cloud_mid_pct ?? 0;
        const hiPct = pt.high_cloud_pct ?? pt.cloud_high_pct ?? 0;
        return `
          <div class="station-mini-card" style="padding: 6px 6px;">
            <div class="station-mini-top" style="font-size: 0.66rem;">
              <span>${distTag}</span>
              <span class="station-mini-pass">${lowPct}%Fog</span>
            </div>
            <div class="station-mini-sub" style="font-size: 0.65rem;">${shortLoc}</div>
            <div class="station-cloud-pills" style="font-size: 0.64rem;">
              <div><span>Top:</span> <strong>${pt.fog_top_ft.toLocaleString()}ft</strong></div>
              <div><span>RH975:</span> <strong>${pt.rh_975_pct}%</strong></div>
              <div><span>Mid/Hi:</span> <strong>${midPct}/${hiPct}%</strong></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  renderAstroCard(w, state.shootReport);
}

function renderAstroCard(w, shootReport) {
  const astroAtm = w?.astro_atmosphere;
  const astroGeo = shootReport?.astro_night_summary;
  if (!astroAtm && !astroGeo) return;

  const atmScore = astroAtm ? astroAtm.astro_atmosphere_score : 84;
  const darkBonus =
    astroGeo && astroGeo.shadowed_dark_sky_duration_minutes >= 180
      ? 6
      : astroGeo?.lunar_alpenglow_window
        ? 4
        : 0;
  const compositeScore = Math.min(98, Math.round(atmScore * 0.88 + 8 + darkBonus));

  const verdictEl = document.getElementById("astro-verdict");
  const badgeEl = document.getElementById("astro-score-badge");
  const subEl = document.getElementById("astro-summary-sub");
  const recEl = document.getElementById("astro-recommendation");
  const gridEl = document.getElementById("astro-pillars-grid");

  if (verdictEl) {
    verdictEl.textContent = astroGeo?.astro_geometry_verdict || "Alpine Dark-Sky & Seeing Forecast";
  }
  if (badgeEl) {
    badgeEl.textContent = `${compositeScore} / 100`;
  }
  if (subEl && astroAtm) {
    const moonStr = astroGeo
      ? `${astroGeo.moon_phase_name} (${Math.round(astroGeo.moon_illumination_pct)}%)`
      : "Moon Ephemeris";
    subEl.textContent = `250hPa Jet ${astroAtm.jet_stream_250hpa_mph} mph • AOD ${astroAtm.aod_550nm} • ${moonStr}`;
  }
  if (recEl) {
    const geoText = astroGeo ? astroGeo.astro_geometry_summary : "";
    const seeText = astroAtm ? `${astroAtm.seeing_note} ${astroAtm.haze_note}` : "";
    recEl.textContent = `${geoText} ${seeText}`.trim();
  }

  // Render Vertical Atmospheric Column & 3D Lunar/Galactic Horizon SVG (#astro-column-svg)
  const colSvg = document.getElementById("astro-column-svg");
  const colLabel = document.getElementById("astro-column-label");
  const seePill = document.getElementById("astro-seeing-pill");
  if (colLabel && astroAtm) {
    colLabel.textContent = `250hPa Jet: ${astroAtm.jet_stream_250hpa_mph} mph (Δv ${astroAtm.vertical_wind_shear_mph} mph) • AOD: ${astroAtm.aod_550nm}`;
  }
  if (seePill && astroAtm) {
    seePill.textContent = `Seeing ${astroAtm.seeing_score}% • Haze ${astroAtm.transparency_score}%`;
  }
  if (colSvg && astroAtm) {
    const isCalmJet = astroAtm.seeing_score >= 75;
    const jetWavePath = isCalmJet
      ? "M 10 18 Q 95 14, 180 18 T 350 18"
      : "M 10 18 Q 52 8, 95 22 T 180 14 T 265 23 T 350 15";
    const hazeFill =
      astroAtm.aod_550nm < 0.09
        ? "rgba(52, 211, 153, 0.13)"
        : astroAtm.aod_550nm < 0.18
          ? "rgba(251, 191, 36, 0.20)"
          : "rgba(244, 63, 94, 0.26)";
    const hazeStroke =
      astroAtm.aod_550nm < 0.09 ? "#34d399" : astroAtm.aod_550nm < 0.18 ? "#f59e0b" : "#fb7185";
    const moonsetStr = astroGeo?.terrain_moonset_local || "21:42";
    const leadStr =
      astroGeo?.terrain_moonset_lead_minutes != null
        ? `-${astroGeo.terrain_moonset_lead_minutes}m vs flat`
        : "DEM Shielded";

    colSvg.innerHTML = `
      <!-- Band 1: 30,000-35,000 ft (250hPa) Jet Stream & Seeing Shear Layer -->
      <rect x="8" y="6" width="344" height="23" rx="5" fill="rgba(99,102,241,0.14)" stroke="rgba(129,140,248,0.45)" stroke-width="0.8" />
      <path d="${jetWavePath}" fill="none" stroke="#818cf8" stroke-width="1.8" stroke-dasharray="4,2" />
      <text x="16" y="16" fill="#c7d2fe" font-size="7.2" font-weight="700">
        34k ft (250hPa) Jet Stream: ${astroAtm.jet_stream_250hpa_mph} mph (Shear Δv ${astroAtm.vertical_wind_shear_mph} mph)
      </text>

      <!-- Band 2: 8,000-16,000 ft (700hPa) Free Troposphere Aerosol / Smoke Plume (AOD 550nm) -->
      <rect x="8" y="34" width="344" height="22" rx="5" fill="${hazeFill}" stroke="${hazeStroke}" stroke-width="0.8" stroke-dasharray="3,2" />
      <text x="16" y="47" fill="#e2e8f0" font-size="7.2" font-weight="700">
        10k ft Crest Aerosol Layer: AOD ${astroAtm.aod_550nm} • PM2.5 ${astroAtm.pm25_ug_m3} µg/m³ (${astroAtm.transparency_score}% Transmittance)
      </text>

      <!-- Terrain Silhouette: West Rim Canyon Wall (Blocks Moon Early) -> Valley Floor -> Half Dome Monolith -->
      <path d="M 8 98 L 8 64 L 64 58 L 106 84 L 228 84 L 288 46 L 315 46 L 348 98 Z"
            fill="rgba(15,23,42,0.92)" stroke="#64748b" stroke-width="1.2" />

      <!-- Terrain-Blocked Moon Ray (Blocked at Canyon Wall) -->
      <line x1="14" y1="46" x2="64" y2="58" stroke="#fbbf24" stroke-width="2.0" stroke-dasharray="3,2" />
      <circle cx="14" cy="46" r="3.5" fill="#fde047" />
      <text x="72" y="69" fill="#fde68a" font-size="6.8" font-weight="700">
        3D Moonset ${moonsetStr} (${leadStr})
      </text>

      <!-- Unobstructed Galactic Core / Starlight Beam passing through 250hPa + AOD Column onto Valley/Summit -->
      <line x1="215" y1="6" x2="176" y2="82" stroke="#c084fc" stroke-width="2.2" />
      <line x1="265" y1="6" x2="288" y2="48" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="2,2" />
      <circle cx="176" cy="82" r="3.5" fill="#c084fc" />
      <circle cx="288" cy="48" r="3.5" fill="#38bdf8" />

      <text x="122" y="95" fill="#e9d5ff" font-size="7" font-weight="700">Dark-Sky Tripod</text>
      <text x="272" y="95" fill="#bae6fd" font-size="7" font-weight="700">Summit Crown</text>
    `;
  }

  if (gridEl && astroAtm) {
    const moonsetStr = astroGeo?.terrain_moonset_local || astroGeo?.flat_moonset_local || "N/A";
    const leadMin = astroGeo?.terrain_moonset_lead_minutes;
    const leadTag = leadMin != null && leadMin > 0 ? `-${leadMin}m early` : "3D DEM";
    const gcClear =
      astroGeo && astroGeo.galactic_core_visible
        ? `+${astroGeo.galactic_core_clearance_deg}° clear`
        : "Below S Ridge";

    gridEl.innerHTML = `
      <div class="station-mini-card">
        <div class="station-mini-top">
          <span>250hPa Seeing</span>
          <span class="station-mini-pass">${astroAtm.seeing_score}/100</span>
        </div>
        <div class="station-mini-sub" title="${astroAtm.seeing_grade}">${astroAtm.seeing_grade}</div>
        <div class="station-cloud-pills">
          <div><span>Jet (34k ft):</span> <strong>${astroAtm.jet_stream_250hpa_mph} mph</strong></div>
          <div><span>Crest (10k):</span> <strong>${astroAtm.crest_wind_700hpa_mph} mph</strong></div>
          <div><span>Wind Shear:</span> <strong>${astroAtm.vertical_wind_shear_mph} mph</strong></div>
        </div>
      </div>

      <div class="station-mini-card">
        <div class="station-mini-top">
          <span>Upper Haze</span>
          <span class="station-mini-pass">${astroAtm.transparency_score}/100</span>
        </div>
        <div class="station-mini-sub" title="${astroAtm.haze_grade}">${astroAtm.haze_grade}</div>
        <div class="station-cloud-pills">
          <div><span>AOD 550nm:</span> <strong>${astroAtm.aod_550nm}</strong></div>
          <div><span>PM2.5 / Smoke:</span> <strong>${astroAtm.pm25_ug_m3} µg</strong></div>
          <div><span>Night Cloud:</span> <strong>${astroAtm.night_cloud_high_pct}%H/${astroAtm.night_cloud_low_pct}%L</strong></div>
        </div>
      </div>

      <div class="station-mini-card">
        <div class="station-mini-top">
          <span>3D Horizon</span>
          <span class="station-mini-pass">${leadTag}</span>
        </div>
        <div class="station-mini-sub">${astroGeo?.moon_phase_name || "Lunar Ephemeris"}</div>
        <div class="station-cloud-pills">
          <div><span>3D Moonset:</span> <strong>${moonsetStr}</strong></div>
          <div><span>Flat Horizon:</span> <strong>${astroGeo?.flat_moonset_local || "N/A"}</strong></div>
          <div><span>Milky Way:</span> <strong>${gcClear}</strong></div>
        </div>
      </div>
    `;
  }

  if (astroGeo) {
    const phaseEl = document.getElementById("val-moon-phase");
    if (phaseEl) {
      phaseEl.textContent = `${astroGeo.moon_phase_name} (${Math.round(astroGeo.moon_illumination_pct)}%)`;
    }
    const msEl = document.getElementById("val-terrain-moonset");
    if (msEl) {
      if (astroGeo.terrain_moonset_local) {
        const diff =
          astroGeo.terrain_moonset_lead_minutes != null
            ? ` (-${astroGeo.terrain_moonset_lead_minutes}m vs flat)`
            : "";
        msEl.textContent = `${astroGeo.terrain_moonset_local}${diff}`;
      } else if (astroGeo.terrain_moonrise_local) {
        const diff =
          astroGeo.terrain_moonrise_lag_minutes != null
            ? ` (+${astroGeo.terrain_moonrise_lag_minutes}m vs flat)`
            : "";
        msEl.textContent = `Rises ${astroGeo.terrain_moonrise_local}${diff}`;
      } else {
        msEl.textContent = "Below / Circumpolar";
      }
    }
    const dwEl = document.getElementById("val-dark-window");
    if (dwEl) {
      dwEl.textContent =
        astroGeo.shadowed_dark_sky_window_local ||
        (astroGeo.lunar_alpenglow_window
          ? `Lunar Glow ${astroGeo.lunar_alpenglow_window.window_start_local}–${astroGeo.lunar_alpenglow_window.window_end_local}`
          : "Moonlit Night");
    }
    const gcEl = document.getElementById("val-galactic-core");
    if (gcEl) {
      if (astroGeo.galactic_core_visible && astroGeo.galactic_core_window_local) {
        gcEl.textContent = `${astroGeo.galactic_core_window_local} (+${astroGeo.galactic_core_clearance_deg}° clear)`;
      } else if (astroGeo.lunar_alpenglow_window) {
        gcEl.textContent = `Lunar Glow @ ${astroGeo.lunar_alpenglow_window.peak_time_local}`;
      } else {
        gcEl.textContent = `Below Ridge (${astroGeo.galactic_core_peak_el_deg ?? 14}°)`;
      }
    }
  }
}

async function fetchWeatherForCorridor(phase = "sunset") {
  const lm = state.shootReport?.target_landmark;
  if (!lm) return;

  const win = state.shootReport?.golden_hour_windows?.find((w) =>
    w.phase.toLowerCase().includes(phase.toLowerCase())
  );
  const sunAz = win ? win.sun_azimuth_deg : phase === "sunrise" ? 102.5 : 259.1;
  const sunEl = win ? win.sun_elevation_deg : 0.4;

  document.getElementById("tab-corridor-sunset")?.classList.toggle("is-active", phase === "sunset");
  document.getElementById("tab-corridor-sunrise")?.classList.toggle("is-active", phase === "sunrise");

  const url = `/api/weather-forecast?lat=${lm.summit_lat}&lon=${lm.summit_lon}&date=${encodeURIComponent(state.date)}&phase=${phase}&sun_az=${sunAz}&sun_el=${sunEl}&elev_m=${lm.summit_elevation_m}&engine=${encodeURIComponent(state.weatherEngine)}`;
  const res = await fetch(url);
  if (res.ok) {
    state.weatherReport = await res.json();
    volumetricEngine.setLiveWeatherReport(state.weatherReport);
    renderWeatherCard(state.weatherReport);
    updateSunRaysAndShader();
  }
}

function renderSidebar() {
  const report = state.shootReport;
  if (!report) return;

  const lm = report.target_landmark;
  document.getElementById("summary-title").textContent = lm.name;
  document.getElementById("summary-elevation").textContent = `${Math.round(lm.summit_elevation_ft).toLocaleString()} ft`;
  document.getElementById("summary-subtitle").textContent =
    `${Math.round(lm.vertical_relief_ft).toLocaleString()} ft vertical relief • ${lm.primary_face_aspect_deg}° face aspect • ${report.total_routes_scanned} trails analyzed`;

  // Top 3 Viewpoint & Hike Cards (Progressive Disclosure Accordion)
  const vpContainer = document.getElementById("vp-cards-container");
  vpContainer.innerHTML = "";

  report.top_viewpoints.forEach((vp, idx) => {
    const isSelected = idx === state.selectedVpIndex;
    const card = document.createElement("article");
    card.className = `vp-card ${isSelected ? "is-selected" : ""}`;
    card.dataset.rank = String(vp.viewpoint_rank);
    card.setAttribute("aria-expanded", String(isSelected));

    const primarySched = vp.schedules[0] || {
      phase: "Sunset Golden Hour",
      shooting_window_local: "16:30 – 18:15",
      peak_light_local: "17:18",
      recommended_trailhead_departure_local: "14:40",
      target_viewpoint_arrival_local: "16:10",
      lighting_style: "3/4 Side-Relief Alpenglow",
    };

    const hrs = Math.floor(vp.hike_duration_to_vp_minutes / 60);
    const mins = vp.hike_duration_to_vp_minutes % 60;
    const durationStr = hrs > 0 ? `${hrs}h ${mins}m` : `${mins}m`;

    card.innerHTML = `
      <div class="vp-card-top">
        <h4 class="vp-title">
          <span class="vp-badge">VP${vp.viewpoint_rank}</span>${vp.viewpoint_label}
        </h4>
        <div class="vp-score-cluster">
          <span class="vp-score">${vp.overall_score}</span>
          <span class="vp-chevron" aria-hidden="true">▾</span>
        </div>
      </div>

      <div class="vp-compact-summary">
        ${vp.hike_distance_to_vp_miles} mi hike (${durationStr}) • Depart ${primarySched.recommended_trailhead_departure_local} • ${vp.sightline.distance_to_landmark_miles} mi sightline
      </div>

      <div class="vp-card-details">
        <div class="vp-trail-meta">
          <div><strong>Trailhead:</strong> ${vp.trailhead_name}</div>
          <div><strong>Gain:</strong> +${Math.round(vp.elevation_gain_to_vp_feet).toLocaleString()} ft</div>
          <div><strong>Bearing:</strong> ${vp.sightline.bearing_to_landmark_deg}° (${vp.sightline.compass_direction})</div>
          <div><strong>Optics:</strong> ${vp.sightline.recommended_focal_length_mm}</div>
        </div>

        <div class="vp-schedule-box">
          <div class="vp-schedule-row">
            <span>🥾 Depart <strong>${vp.trailhead_name}</strong>:</span>
            <strong>${primarySched.recommended_trailhead_departure_local}</strong>
          </div>
          <div class="vp-schedule-row">
            <span>📍 Arrive <strong>VP${vp.viewpoint_rank}</strong>:</span>
            <strong>${primarySched.target_viewpoint_arrival_local}</strong>
          </div>
          <div class="vp-schedule-row">
            <span>☀️ Peak Window (${primarySched.phase.split(" ")[0]}):</span>
            <strong>${primarySched.shooting_window_local}</strong>
          </div>
        </div>

        <div class="vp-actions">
          <button type="button" class="vp-btn" data-action="orbit-vp" data-idx="${idx}">
            Focus in 3D Map
          </button>
          <button type="button" class="vp-btn" data-action="tripod-vp" data-idx="${idx}">
            📷 Enter Viewfinder
          </button>
        </div>
      </div>
    `;

    card.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-action]");
      if (btn) {
        const action = btn.dataset.action;
        if (action === "tripod-vp") {
          selectViewpoint(idx, false);
          setCameraMode(`vp-${idx}`);
        } else {
          selectViewpoint(idx, true);
        }
        return;
      }

      // Clicking card body toggles selection / highlights this viewpoint
      if (state.selectedVpIndex === idx && ! e.target.closest(".vp-card-details")) {
        state.selectedVpIndex = -1;
        highlightSelectedVpCard();
        buildTrailsAndSightlines();
      } else {
        selectViewpoint(idx, true);
      }
    });

    vpContainer.appendChild(card);
  });
}

// ============================================================================
// Load Landmark Shoot Plan, 340x340 Terrain Patch & Weather Forecast
// ============================================================================
async function fetchBinaryOrJsonTerrainPatch(lat, lon, spanM = 9200, gridSize = 600) {
  try {
    const binRes = await fetch(
      `/api/terrain-patch.bin?lat=${lat}&lon=${lon}&span_m=${spanM}&grid_size=${gridSize}`
    );
    if (binRes.ok) {
      const buf = await binRes.arrayBuffer();
      const headerLen = new DataView(buf, 0, 4).getUint32(0, true);
      const headerStr = new TextDecoder().decode(new Uint8Array(buf, 4, headerLen));
      const meta = JSON.parse(headerStr);
      meta.elevations = new Float32Array(buf, 4 + headerLen);
      return meta;
    }
  } catch (err) {
    console.warn("Binary terrain stream fallback to JSON:", err);
  }
  const jsonRes = await fetch(
    `/api/terrain-patch?lat=${lat}&lon=${lon}&span_m=${spanM}&grid_size=${gridSize}`
  );
  if (!jsonRes.ok) throw new Error(`Terrain API error ${jsonRes.status}`);
  return jsonRes.json();
}

async function loadLandmarkScenario(customCoord = null) {
  loadingEl.hidden = false;
  inspectorCard.hidden = true;
  window.getSelection?.()?.removeAllRanges?.();
  disposeObjectTree(customPinGroup);

  const lmSel = document.getElementById("landmark-select");
  const dtInp = document.getElementById("shoot-date");
  if (lmSel && !customCoord) lmSel.value = state.landmark;
  if (dtInp) dtInp.value = state.date;
  try {
    if (!customCoord) localStorage.setItem("pp_landmark_v2", state.landmark);
    localStorage.setItem("pp_date_v2", state.date);
  } catch (_) {}

  const label = customCoord ? customCoord.name : state.landmark;
  loadingTextEl.textContent = `Ray-casting 1m USGS 3DEP LiDAR & trail sightlines for ${label}…`;

  try {
    let planUrl;
    if (customCoord) {
      planUrl = `/api/shoot-coord?lat=${customCoord.lat}&lon=${customCoord.lon}&name=${encodeURIComponent(customCoord.name)}&date=${encodeURIComponent(state.date)}&max_miles=${state.maxMiles}&top_k=3`;
    } else {
      planUrl = `/api/shoot-plan?landmark=${encodeURIComponent(state.landmark)}&date=${encodeURIComponent(state.date)}&max_miles=${state.maxMiles}&top_k=3`;
    }

    const planRes = await fetch(planUrl);
    if (!planRes.ok) throw new Error(`API error ${planRes.status}`);
    const report = await planRes.json();
    state.shootReport = report;

    const lm = report.target_landmark;
    const sunsetWin = report.golden_hour_windows?.[0];
    const initAz = sunsetWin ? sunsetWin.sun_azimuth_deg : 259.1;
    const initEl = sunsetWin ? sunsetWin.sun_elevation_deg : 0.4;

    const [terrainPatchData, weatherRes] = await Promise.all([
      fetchBinaryOrJsonTerrainPatch(lm.summit_lat, lm.summit_lon, 9200, 600),
      fetch(`/api/weather-forecast?lat=${lm.summit_lat}&lon=${lm.summit_lon}&date=${encodeURIComponent(state.date)}&phase=sunset&sun_az=${initAz}&sun_el=${initEl}&elev_m=${lm.summit_elevation_m}&engine=${encodeURIComponent(state.weatherEngine)}`),
    ]);

    state.terrainPatch = terrainPatchData;

    if (weatherRes.ok) {
      state.weatherReport = await weatherRes.json();
      volumetricEngine.setLiveWeatherReport(state.weatherReport);
      renderWeatherCard(state.weatherReport);
    }

    if (report.golden_hour_windows && report.golden_hour_windows.length > 0) {
      state.currentMinutes = parseHHMMToMinutes(report.golden_hour_windows[0].peak_time_local);
      timeScrubber.value = String(state.currentMinutes);
      syncSliderValue(timeScrubber);
    }

    state.selectedVpIndex = 0;
    buildTerrainMesh();
    buildTrailsAndSightlines();
    renderSidebar();
    setCameraMode("orbit");
    updateSunRaysAndShader();
  } catch (err) {
    console.error("Failed to load scenario:", err);
    document.getElementById("summary-subtitle").textContent = `Error loading scenario: ${err.message}`;
  } finally {
    loadingEl.hidden = true;
  }
}

// ============================================================================
// Interactive 3D Terrain Raycaster (Hover Readout + Click-to-Highlight)
// ============================================================================
const raycaster = new THREE.Raycaster();
const pointerNDC = new THREE.Vector2();
let pointerDownPos = { x: 0, y: 0 };

function updatePointerNDC(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointerNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointerNDC.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

renderer.domElement.addEventListener("pointermove", (e) => {
  if (!terrainMeshRef || !state.terrainPatch) return;
  updatePointerNDC(e);
  raycaster.setFromCamera(pointerNDC, camera);

  // Check if hovering a 3D Viewpoint marker first
  const vpHits = raycaster.intersectObjects(vpMarkerMeshes, false);
  if (vpHits.length > 0 && vpHits[0].object.userData?.vpIndex !== undefined) {
    const vpIdx = vpHits[0].object.userData.vpIndex;
    const vp = state.shootReport?.top_viewpoints?.[vpIdx];
    if (vp) {
      cursorCoordsBar.textContent = `VP${vp.viewpoint_rank}: ${vp.viewpoint_label} (${vp.hike_distance_to_vp_miles} mi) • Click to highlight`;
      return;
    }
  }

  const hits = raycaster.intersectObject(terrainMeshRef, false);
  if (hits.length > 0) {
    const { lat, lon, elevM } = worldToLatLonElev(hits[0].point);
    const elevFt = Math.round(elevM * 3.28084);
    const actionLabel =
      state.clickMode === "tripod" ? "Click to inspect sightline" : "Click to set Target";
    cursorCoordsBar.textContent = `${lat.toFixed(4)}, ${lon.toFixed(4)} • ${elevFt.toLocaleString()} ft • ${actionLabel}`;
  }
});

let tripodRequestSeq = 0;

renderer.domElement.addEventListener("selectstart", (e) => e.preventDefault());

renderer.domElement.addEventListener("pointerdown", (e) => {
  window.getSelection?.()?.removeAllRanges?.();
  pointerDownPos = { x: e.clientX, y: e.clientY };
});

renderer.domElement.addEventListener("pointerup", async (e) => {
  if (e.button !== 0) return;
  const moveDist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
  if (moveDist > 14 || !terrainMeshRef || !state.terrainPatch || !state.shootReport) return;

  updatePointerNDC(e);
  raycaster.setFromCamera(pointerNDC, camera);

  const hits = raycaster.intersectObject(terrainMeshRef, false);

  // 1. Check if the user clicked directly on a 3D Viewpoint Marker Pillar (VP1..VP3)
  const vpHits = raycaster.intersectObjects(vpMarkerMeshes, false);
  if (
    vpHits.length > 0 &&
    vpHits[0].object.userData?.vpIndex !== undefined &&
    (hits.length === 0 || vpHits[0].distance <= hits[0].distance + 45)
  ) {
    const vpIdx = vpHits[0].object.userData.vpIndex;
    selectViewpoint(vpIdx, true);
    return;
  }

  // 2. Otherwise intersect any point on the 3D terrain mesh
  if (hits.length === 0) return;

  const { lat, lon, elevM } = worldToLatLonElev(hits[0].point);

  if (state.clickMode === "target") {
    // Mode B: Set Clicked Peak/Cliff as the New Target Subject & Enable Viewshed Heatmap
    state.showViewshed = true;
    document.getElementById("toggle-viewshed").checked = true;
    const customName = `Custom Peak (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
    await loadLandmarkScenario({ lat, lon, name: customName });
    return;
  }

  // Mode A (Default): Collapse VP cards & Immediately Render Instant Client-Side Pin + Sightline (<16ms)
  state.selectedVpIndex = -1;
  highlightSelectedVpCard();
  buildTrailsAndSightlines();

  const reqId = ++tripodRequestSeq;
  const instantReport = buildInstantClientTripodReport(lat, lon, elevM);
  if (instantReport) {
    state.customTripodReport = instantReport;
    renderCustomTripodPinAndCrossSection(instantReport);
    updateSunRaysAndShader();
  }

  // Sub-millisecond (<0.3 ms) Rust WASM curved-Earth sightline & 360° DEM horizon refinement
  const lmTarget = state.shootReport.target_landmark;
  volumetricEngine
    .raycastTripodWasm(
      lat,
      lon,
      lmTarget.upper_face_lat,
      lmTarget.upper_face_lon,
      lmTarget.upper_face_z_m
    )
    .then((wasmHit) => {
      if (!wasmHit || reqId !== tripodRequestSeq || !state.customTripodReport?._isComputingBackend) {
        return;
      }
      const d = wasmHit.data;
      if (d && d.length >= 203) {
        const wasmProfile = [];
        for (let s = 0; s <= 64; s++) {
          const base = 8 + s * 3;
          wasmProfile.push([
            Math.round(d[base]),
            Math.round(d[base + 1] * 10) / 10,
            Math.round(d[base + 2] * 10) / 10,
          ]);
        }
        const cur = state.customTripodReport;
        cur.tripod_elevation_m = Math.round(d[0] * 10) / 10;
        cur.tripod_elevation_ft = Math.round(d[0] * 3.28084);
        cur.sightline.is_visible = d[6] > 0.5;
        cur.sightline.max_obstruction_m = Math.round(d[4] * 10) / 10;
        cur.sightline.obstruction_distance_m = Math.round(d[5]);
        cur.sightline.composition_score = Math.round(d[7] * 10) / 10;
        cur.sightline_profile_m = wasmProfile;
        renderCustomTripodPinAndCrossSection(cur);
      }
    });

  // Enrich asynchronously with backend trailhead routing & 360° keyhole light notches
  try {
    const lm = state.shootReport.target_landmark;
    const url = `/api/tripod-los?tripod_lat=${lat.toFixed(5)}&tripod_lon=${lon.toFixed(5)}&target_lat=${lm.summit_lat}&target_lon=${lm.summit_lon}&target_name=${encodeURIComponent(lm.name)}&date=${encodeURIComponent(state.date)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Tripod LOS error ${res.status}`);
    const tripodRep = await res.json();
    if (reqId !== tripodRequestSeq) return;
    state.customTripodReport = tripodRep;
    renderCustomTripodPinAndCrossSection(tripodRep);
    updateSunRaysAndShader();
  } catch (err) {
    console.error("Tripod LOS enrichment failed:", err);
  }
});

// ============================================================================
// UI Event Listeners
// ============================================================================
syncSliderValue(timeScrubber);

timeScrubber.addEventListener("input", () => {
  state.currentMinutes = parseFloat(timeScrubber.value);
  syncSliderValue(timeScrubber);
  updateSunRaysAndShader();
});

document.getElementById("query-form").addEventListener("submit", (e) => {
  e.preventDefault();
  state.landmark = document.getElementById("landmark-select").value;
  state.date = document.getElementById("shoot-date").value;
  state.maxMiles = parseFloat(document.getElementById("max-miles").value);
  loadLandmarkScenario();
});

document.getElementById("landmark-select").addEventListener("change", (e) => {
  state.landmark = e.target.value;
  loadLandmarkScenario();
});

document.getElementById("shoot-date").addEventListener("change", (e) => {
  state.date = e.target.value;
  loadLandmarkScenario();
});

document.getElementById("max-miles").addEventListener("change", (e) => {
  state.maxMiles = parseFloat(e.target.value);
  loadLandmarkScenario();
});

if (btnOrbitMode) {
  btnOrbitMode.addEventListener("click", () => setCameraMode("orbit"));
}

document.getElementById("viewport-weather-pill")?.addEventListener("click", () => {
  if (weatherDisclosure) {
    weatherDisclosure.open = true;
    weatherDisclosure.scrollIntoView({ behavior: "smooth", block: "nearest" });
    updateSunRaysAndShader();
  }
});

// Toggle 3D sky station billboards when the user opens/closes the Weather Ray-Path or Astro disclosure
weatherDisclosure?.addEventListener("toggle", () => {
  updateSunRaysAndShader();
});

document.getElementById("astro-disclosure")?.addEventListener("toggle", () => {
  updateSunRaysAndShader();
});

// Click Mode Switcher (Inspect Point vs. Set Target)
const btnClickTripod = document.getElementById("btn-click-tripod");
const btnClickTarget = document.getElementById("btn-click-target");

btnClickTripod.addEventListener("click", () => {
  state.clickMode = "tripod";
  btnClickTripod.classList.add("is-active-emerald");
  btnClickTarget.classList.remove("is-active");
});

btnClickTarget.addEventListener("click", () => {
  state.clickMode = "target";
  btnClickTarget.classList.add("is-active");
  btnClickTripod.classList.remove("is-active-emerald");
});

// Surface Style Switcher (Granite Relief, USGS Satellite, Topo Contours)
const surfBtns = [
  document.getElementById("btn-surf-granite"),
  document.getElementById("btn-surf-aerial"),
  document.getElementById("btn-surf-topo"),
];

surfBtns.forEach((btn, idx) => {
  btn.addEventListener("click", () => {
    state.surfaceStyle = idx;
    surfBtns.forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    updateSunRaysAndShader();
  });
});

document.getElementById("toggle-viewshed").addEventListener("change", (e) => {
  state.showViewshed = e.target.checked;
  updateSunRaysAndShader();
});

document.getElementById("toggle-sun-beams").addEventListener("change", (e) => {
  state.showSunBeams = e.target.checked;
  updateSunRaysAndShader();
});

document.getElementById("toggle-light-field")?.addEventListener("change", (e) => {
  state.showLightField = e.target.checked;
  updateSunRaysAndShader();
});

document.getElementById("toggle-atmosphere")?.addEventListener("change", (e) => {
  state.showAtmosphere = e.target.checked;
  updateSunRaysAndShader();
});

document.getElementById("toggle-sightlines").addEventListener("change", (e) => {
  state.showSightlines = e.target.checked;
  trailsGroup.visible = state.showSightlines && state.cameraMode === "orbit";
});

document.getElementById("toggle-alpenglow-zone").addEventListener("change", (e) => {
  state.highlightLandingZone = e.target.checked;
  updateSunRaysAndShader();
});

document.getElementById("btn-enter-custom-vf").addEventListener("click", () => {
  if (state.customTripodReport) {
    setCameraMode("custom-tripod");
  }
});

document.getElementById("btn-close-inspector").addEventListener("click", () => {
  inspectorCard.hidden = true;
  disposeObjectTree(customPinGroup);
  if (state.selectedVpIndex === -1) {
    state.selectedVpIndex = 0;
    highlightSelectedVpCard();
    buildTrailsAndSightlines();
  }
});

const btnSunrise = document.getElementById("btn-jump-sunrise");
const btnSunset = document.getElementById("btn-jump-sunset");
const btnNight = document.getElementById("btn-jump-night");

btnSunrise.addEventListener("click", () => {
  btnSunrise.classList.add("is-active");
  btnSunset.classList.remove("is-active");
  btnNight?.classList.remove("is-active");
  const sunriseWin = state.shootReport?.golden_hour_windows?.find((w) =>
    w.phase.toLowerCase().includes("sunrise")
  );
  const mins = sunriseWin ? parseHHMMToMinutes(sunriseWin.peak_time_local) : 6 * 60 + 21;
  state.currentMinutes = mins;
  timeScrubber.value = String(mins);
  syncSliderValue(timeScrubber);
  fetchWeatherForCorridor("sunrise");
  updateSunRaysAndShader();
});

btnSunset.addEventListener("click", () => {
  btnSunset.classList.add("is-active");
  btnSunrise.classList.remove("is-active");
  btnNight?.classList.remove("is-active");
  const sunsetWin = state.shootReport?.golden_hour_windows?.find((w) =>
    w.phase.toLowerCase().includes("sunset")
  );
  const mins = sunsetWin ? parseHHMMToMinutes(sunsetWin.peak_time_local) : 17 * 60 + 18;
  state.currentMinutes = mins;
  timeScrubber.value = String(mins);
  syncSliderValue(timeScrubber);
  fetchWeatherForCorridor("sunset");
  updateSunRaysAndShader();
});

btnNight?.addEventListener("click", () => {
  btnNight.classList.add("is-active");
  btnSunrise.classList.remove("is-active");
  btnSunset.classList.remove("is-active");
  const astro = state.shootReport?.astro_night_summary;
  const peakStr =
    astro?.galactic_core_peak_local ||
    astro?.lunar_alpenglow_window?.peak_time_local ||
    astro?.terrain_moonset_local ||
    "21:45";
  let mins = parseHHMMToMinutes(peakStr);
  if (mins < 720) mins = 22 * 60 + 15;
  state.currentMinutes = mins;
  timeScrubber.value = String(mins);
  syncSliderValue(timeScrubber);
  const astroDetails = document.getElementById("astro-disclosure");
  if (astroDetails) astroDetails.open = true;
  updateSunRaysAndShader();
});

document.getElementById("tab-corridor-sunset")?.addEventListener("click", () => {
  btnSunset.click();
});

document.getElementById("tab-corridor-sunrise")?.addEventListener("click", () => {
  btnSunrise.click();
});

function selectWeatherPredictionEngine(engineId) {
  if (!engineId) return;
  state.weatherEngine = engineId;
  document.querySelectorAll("#ai-engine-selector-bar [data-weather-engine]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.weatherEngine === engineId);
  });
  const isSunriseActive = document.getElementById("tab-corridor-sunrise")?.classList.contains("is-active");
  fetchWeatherForCorridor(isSunriseActive ? "sunrise" : "sunset");
}

document.getElementById("ai-engine-selector-bar")?.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-weather-engine]");
  if (btn) {
    selectWeatherPredictionEngine(btn.dataset.weatherEngine);
  }
});

document.getElementById("ai-models-comparison-grid")?.addEventListener("click", (e) => {
  const card = e.target.closest("[data-select-engine]");
  if (card) {
    selectWeatherPredictionEngine(card.dataset.selectEngine);
  }
});

function applyQuickScenario(landmark, date, fluidPreset = null) {
  document.getElementById("scenarios-popover")?.hidePopover?.();
  state.landmark = landmark;
  state.date = date;
  const lmSel = document.getElementById("landmark-select");
  const dtInp = document.getElementById("shoot-date");
  if (lmSel) lmSel.value = landmark;
  if (dtInp) dtInp.value = date;
  if (fluidPreset) {
    const fSel = document.getElementById("select-fluid-preset");
    if (fSel) fSel.value = fluidPreset;
    volumetricEngine.setPreset(fluidPreset);
    const info = WEATHER_REGIME_PRESETS[fluidPreset];
    const descEl = document.getElementById("fluid-preset-desc");
    if (descEl && info) descEl.textContent = info.description;
  }
  loadLandmarkScenario();
}

document.getElementById("btn-yesterday-pedro")?.addEventListener("click", () => {
  applyQuickScenario("Pedro Point Headlands", "2026-04-16", "live");
});

document.getElementById("btn-vp-yesterday-pedro")?.addEventListener("click", () => {
  applyQuickScenario("Pedro Point Headlands", "2026-04-16", "live");
});

document.getElementById("btn-live-today-sf")?.addEventListener("click", () => {
  applyQuickScenario("Montara Mountain", "2026-04-17", "live");
});

document.getElementById("btn-vp-today-sf")?.addEventListener("click", () => {
  applyQuickScenario("Montara Mountain", "2026-04-17", "live");
});

document.getElementById("btn-hist-fog-sea")?.addEventListener("click", () => {
  applyQuickScenario("Montara Mountain", "2025-08-15", "coastal_fog_sea");
});

document.getElementById("btn-vp-fog-sea")?.addEventListener("click", () => {
  applyQuickScenario("Montara Mountain", "2025-08-15", "coastal_fog_sea");
});

document.getElementById("btn-hist-clear-slot")?.addEventListener("click", () => {
  applyQuickScenario("Pedro Point Headlands", "2025-10-22", "crepuscular_godrays");
});

document.getElementById("btn-vp-clear-slot")?.addEventListener("click", () => {
  applyQuickScenario("Pedro Point Headlands", "2025-10-22", "crepuscular_godrays");
});

document.getElementById("btn-vp-halfdome-alpenglow")?.addEventListener("click", () => {
  applyQuickScenario("Half Dome", "2026-04-16", "sierra_cumulus_cirrus");
});

document.getElementById("btn-vp-clouds-rest-wave")?.addEventListener("click", () => {
  applyQuickScenario("Clouds Rest", "2026-04-16", "lenticular_wave");
});

const playBtn = document.getElementById("btn-play-sun");
playBtn.addEventListener("click", () => {
  state.isPlayingSun = !state.isPlayingSun;
  playBtn.textContent = state.isPlayingSun ? "⏸ Pause" : "▶ Animate";
  playBtn.classList.toggle("is-active", state.isPlayingSun);
});

function syncThemeToggleButton(theme) {
  const iconEl = document.getElementById("theme-toggle-icon");
  const labelEl = document.getElementById("theme-toggle-label");
  const btn = document.getElementById("theme-toggle");
  if (iconEl) iconEl.textContent = theme === "dark" ? "☀️" : "🌙";
  if (labelEl) labelEl.textContent = theme === "dark" ? "Light" : "Dark";
  btn?.setAttribute("aria-pressed", String(theme === "light"));
}

syncThemeToggleButton(document.documentElement.getAttribute("data-theme") || "dark");

document.getElementById("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  syncThemeToggleButton(next);
});

// Escape key exits First-Person Tripod Viewfinder back to 3D Orbit when no popover is open
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.cameraMode !== "orbit") {
    const anyPopoverOpen = document.querySelector("[popover]:popover-open");
    if (!anyPopoverOpen) {
      setCameraMode("orbit");
    }
  }
});

// Annual Keyhole Light-Notch Alignment Finder Popover
document.getElementById("btn-open-calendar").addEventListener("click", async () => {
  const listEl = document.getElementById("calendar-alignments-list");
  const vpNameEl = document.getElementById("calendar-vp-name");
  const activeIdx = state.selectedVpIndex >= 0 ? state.selectedVpIndex : 0;
  const vp = state.shootReport?.top_viewpoints?.[activeIdx];
  const lm = state.shootReport?.target_landmark;
  const lat = vp ? vp.viewpoint_lat : lm?.summit_lat || 37.74554;
  const lon = vp ? vp.viewpoint_lon : lm?.summit_lon || -119.53315;
  const label = vp ? `VP${vp.viewpoint_rank} (${vp.viewpoint_label})` : lm?.name || "Half Dome";
  vpNameEl.textContent = label;
  listEl.innerHTML = `<div class="calendar-empty">Scanning 365-day solar ephemeris against 1m LiDAR + SRTM horizon profile from ${label}…</div>`;

  try {
    const year = parseInt(state.date.split("-")[0], 10) || 2026;
    const res = await fetch(`/api/annual-calendar?lat=${lat}&lon=${lon}&year=${year}`);
    if (!res.ok) throw new Error(`Status ${res.status}`);
    const data = await res.json();
    const notches = data.notches || [];

    const rows = [];
    for (const n of notches.slice(0, 6)) {
      for (const al of (n.alignments || []).slice(0, 2)) {
        rows.push({ notch: n, alignment: al });
      }
    }

    if (rows.length === 0) {
      listEl.innerHTML = `<div class="calendar-empty">No narrow V-notches within the solar azimuth band at this vantage point; open alpine horizon all year.</div>`;
      return;
    }

    listEl.innerHTML = "";
    rows.slice(0, 8).forEach(({ notch, alignment }) => {
      const row = document.createElement("div");
      row.className = "calendar-notch-row";
      row.innerHTML = `
        <div class="calendar-notch-meta">
          <div class="calendar-notch-title">
            ${alignment.date} @ ${alignment.peak_local_solar_time} (${alignment.phase}) — Keyhole Az ${notch.azimuth_deg.toFixed(1)}°
          </div>
          <div class="calendar-notch-sub">
            Notch Prominence: ${(notch.prominence_deg ?? 0).toFixed(2)}° • Width (FWHM): ${(notch.angular_width_fwhm_deg ?? notch.fwhm_deg ?? 0).toFixed(2)}° • Ridge Dist: ${((notch.notch_distance_m ?? notch.distance_m ?? 0) / 1000).toFixed(2)} km • Score ${(notch.notch_quality_score ?? notch.quality_score ?? 0).toFixed(1)}
          </div>
        </div>
        <button type="button" class="calendar-load-btn">Load Date &amp; Sun</button>
      `;
      row.querySelector("button").addEventListener("click", async () => {
        state.date = alignment.date;
        document.getElementById("shoot-date").value = alignment.date;
        document.getElementById("annual-calendar-popover")?.hidePopover?.();
        await loadLandmarkScenario();
        const mins = parseHHMMToMinutes(alignment.peak_local_solar_time);
        state.currentMinutes = mins;
        timeScrubber.value = String(mins);
        syncSliderValue(timeScrubber);
        updateSunRaysAndShader();
      });
      listEl.appendChild(row);
    });
  } catch (err) {
    listEl.innerHTML = `<div class="calendar-empty">Error loading annual alignments: ${err.message}</div>`;
  }
});

// ============================================================================
// 3D Fluid Dynamics & Upper-Atmosphere Cirrus Controls (Rust WASM + WebGL2)
// ============================================================================
const fluidDisclosure = document.getElementById("fluid-physics-disclosure");
const selectFluidPreset = document.getElementById("select-fluid-preset");
const fluidPresetDesc = document.getElementById("fluid-preset-desc");
const btnToggleFluidEvolve = document.getElementById("btn-toggle-fluid-evolve");
const sliderFluidSpeed = document.getElementById("slider-fluid-speed");
const sliderMicroDetail = document.getElementById("slider-micro-detail");
const sliderCloudDensity = document.getElementById("slider-cloud-density");
const sliderCirrusDensity = document.getElementById("slider-cirrus-density");
const fluidPerfBadge = document.getElementById("fluid-perf-badge");
const fluidSummarySub = document.getElementById("fluid-summary-sub");
const viewportGpuPill = document.getElementById("viewport-gpu-pill");

[sliderFluidSpeed, sliderMicroDetail, sliderCloudDensity, sliderCirrusDensity].forEach((inp) => {
  if (inp) syncSliderValue(inp);
});

viewportGpuPill?.addEventListener("click", () => {
  if (fluidDisclosure) {
    fluidDisclosure.open = true;
    fluidDisclosure.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});

selectFluidPreset?.addEventListener("change", (e) => {
  const key = e.target.value;
  volumetricEngine.setPreset(key);
  const info = WEATHER_REGIME_PRESETS[key];
  if (fluidPresetDesc && info) {
    fluidPresetDesc.textContent = info.description;
  }
  updateSunRaysAndShader();
});

document.querySelectorAll("[data-fluid-quality]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const q = btn.dataset.fluidQuality;
    volumetricEngine.setQualityMode(q);
    document.querySelectorAll("[data-fluid-quality]").forEach((b) => {
      b.classList.toggle("is-active", b === btn);
    });
  });
});

btnToggleFluidEvolve?.addEventListener("click", () => {
  volumetricEngine.evolveFluid = !volumetricEngine.evolveFluid;
  btnToggleFluidEvolve.classList.toggle("is-active", volumetricEngine.evolveFluid);
  btnToggleFluidEvolve.textContent = volumetricEngine.evolveFluid
    ? "⏸ Pause"
    : "▶ Resume";
});

sliderFluidSpeed?.addEventListener("input", () => {
  syncSliderValue(sliderFluidSpeed);
  const v = parseFloat(sliderFluidSpeed.value);
  volumetricEngine.evolutionSpeed = v;
  const out = document.getElementById("out-fluid-speed");
  if (out) out.textContent = `${v.toFixed(1)}×`;
});

sliderMicroDetail?.addEventListener("input", () => {
  syncSliderValue(sliderMicroDetail);
  const pct = parseFloat(sliderMicroDetail.value);
  volumetricEngine.setUserTuning({ microDetailGain: pct / 100.0 });
  const out = document.getElementById("out-micro-detail");
  if (out) out.textContent = `${Math.round(pct)}%`;
});

sliderCloudDensity?.addEventListener("input", () => {
  syncSliderValue(sliderCloudDensity);
  const pct = parseFloat(sliderCloudDensity.value);
  volumetricEngine.setUserTuning({ cloudDensityGain: pct / 100.0 });
  const out = document.getElementById("out-cloud-density");
  if (out) out.textContent = `${Math.round(pct)}%`;
});

sliderCirrusDensity?.addEventListener("input", () => {
  syncSliderValue(sliderCirrusDensity);
  const pct = parseFloat(sliderCirrusDensity.value);
  volumetricEngine.setUserTuning({ cirrusDensityGain: pct / 100.0 });
  const out = document.getElementById("out-cirrus-density");
  if (out) out.textContent = `${Math.round(pct)}%`;
});

let rollingFps = 60;

volumetricEngine.onTelemetryCallback = (tel) => {
  const lclEl = document.getElementById("val-fluid-lcl");
  const cirBaseEl = document.getElementById("val-fluid-cirrus-base");
  const updraftEl = document.getElementById("val-fluid-updraft");
  const frShearEl = document.getElementById("val-fluid-froude-shear");
  const lowMidEl = document.getElementById("val-fluid-low-mid");
  const cirCovEl = document.getElementById("val-fluid-cirrus-cov");

  const lclFt = Math.round((tel.lclAltitudeM || 0) * 3.28084);
  const frzFt = Math.round((tel.freezingLevelM || 0) * 3.28084);
  const cirKm = ((tel.glaciationCirrusBaseM || 8400) / 1000).toFixed(1);
  const topKm = ((tel.zTopM || 12200) / 1000).toFixed(1);

  if (lclEl) lclEl.textContent = `${lclFt.toLocaleString()} / ${frzFt.toLocaleString()} ft`;
  if (cirBaseEl) cirBaseEl.textContent = `${cirKm}–${topKm} km`;
  if (updraftEl) {
    updraftEl.textContent = `+${(tel.maxUpdraftMs || 0).toFixed(1)}m/s · ${(tel.maxLiquidGKg || 0).toFixed(2)}g/kg`;
  }
  if (frShearEl) {
    frShearEl.textContent = `Fr ${(tel.froudeNumber || 0).toFixed(2)} · ${(tel.jetShearMsKm || 0).toFixed(1)}/km`;
  }
  const lowPct = Math.round(tel.lowCloudCoverage || 0);
  const midPct = Math.round(tel.midCloudCoverage || 0);
  const highPct = Math.round(tel.highCirrusCoverage || 0);
  if (lowMidEl) {
    lowMidEl.textContent = `${lowPct}% / ${midPct}%`;
  }
  if (cirCovEl) {
    cirCovEl.textContent = `${highPct}% · ${(tel.maxIceGKg || 0).toFixed(2)} g/kg`;
  }

  const stepStr = (tel.stepMs || 1.2).toFixed(1);
  if (fluidPerfBadge) {
    fluidPerfBadge.textContent = `${Math.round(rollingFps)} FPS`;
    fluidPerfBadge.title = `WASM Fluid Step: ${stepStr} ms (Worker Thread)`;
  }
  if (viewportGpuPill) {
    viewportGpuPill.textContent = `⚡ ${Math.round(rollingFps)} FPS · ${stepStr}ms`;
  }
  if (fluidSummarySub) {
    fluidSummarySub.textContent = `1.0m Detail • 0–${topKm} km 3D Cirrus • Low ${lowPct}% / High ${highPct}%`;
  }
};

window.addEventListener("resize", () => {
  const w = container.clientWidth;
  const h = Math.max(container.clientHeight, 1);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  volumetricEngine.resize();
});

// ============================================================================
// Main GPU Volumetric + Rust WASM Render Loop
// ============================================================================
let lastFrameTimeMs = performance.now();
let fpsAccumFrames = 0;
let fpsAccumMs = 0;

function animate(nowMs = performance.now()) {
  requestAnimationFrame(animate);

  const dtMs = Math.min(100, Math.max(0.5, nowMs - lastFrameTimeMs));
  lastFrameTimeMs = nowMs;
  const dtRealS = dtMs * 0.001;

  fpsAccumFrames += 1;
  fpsAccumMs += dtMs;
  if (fpsAccumMs >= 450) {
    rollingFps = (fpsAccumFrames * 1000) / fpsAccumMs;
    fpsAccumFrames = 0;
    fpsAccumMs = 0;
    const stepStr = (volumetricEngine.telemetry.stepMs || 1.2).toFixed(1);
    if (fluidPerfBadge) {
      fluidPerfBadge.textContent = `${Math.round(rollingFps)} FPS`;
      fluidPerfBadge.title = `WASM Fluid Step: ${stepStr} ms (Worker Thread)`;
    }
    if (viewportGpuPill) {
      viewportGpuPill.textContent = `⚡ ${Math.round(rollingFps)} FPS · ${stepStr}ms`;
    }
  }

  const nowSec = nowMs * 0.001;
  if (terrainShaderMaterial?.uniforms?.uCloudTime) {
    terrainShaderMaterial.uniforms.uCloudTime.value = nowSec;
  }
  for (let i = 0; i < activeGodRayMaterials.length; i++) {
    activeGodRayMaterials[i].uniforms.uCloudTime.value = nowSec;
  }

  if (state.isPlayingSun) {
    state.currentMinutes += 0.65;
    if (state.currentMinutes > 1410) {
      state.currentMinutes = 330;
    }
    timeScrubber.value = String(Math.round(state.currentMinutes));
    syncSliderValue(timeScrubber);
    updateSunRaysAndShader();
  }

  controls.update();
  volumetricEngine.renderPipeline(worldScene, overlayScene, dtRealS, state.showAtmosphere);
}

animate();
loadLandmarkScenario();
