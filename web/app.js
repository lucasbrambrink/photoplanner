import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VolumetricAtmosphereEngine } from "./volumetric-engine.js?v=20261117c";

// Conditional Popover API polyfill loader (content-first-noise-reduction)
if (!Object.hasOwn(HTMLElement.prototype, "popover")) {
  import("https://esm.sh/@oddbird/popover-polyfill").then((m) => m.apply?.());
}

// ============================================================================
// Application State (Default to SF Coast / Pedro Point on Today's Live Sunset, persisted in localStorage)
// ============================================================================
function formatLocalDateISO(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

const nowLocalDate = new Date();
const todayIso = formatLocalDateISO(nowLocalDate);
const yesterdayDateObj = new Date(nowLocalDate.getTime() - 86400 * 1000);
const yesterdayIso = formatLocalDateISO(yesterdayDateObj);

const savedLandmark = localStorage.getItem("pp_landmark_v2") || "Pedro Point Headlands";
const savedDate = localStorage.getItem("pp_date_v3") || todayIso;

const state = {
  landmark: savedLandmark,
  date: savedDate,
  maxMiles: 16.0,
  weatherEngine: "ensemble", // "ensemble" | "graphcast" | "hrrr" | "ecmwf" | "nowcast"
  replayMode: "truth", // "truth" | "nowcast_1h" | "hindcast_24h" | "hindcast_72h" (for past dates)
  cachedTruthReport: null,
  cachedTruthKey: "",
  shootReport: null,
  terrainPatch: null,
  weatherReport: null,
  activeAtmosphericLayers: null,
  hoveredCloudLayerId: null,
  customTripodReport: null,
  currentMinutes: 17 * 60 + 18,
  selectedVpIndex: 0, // 0..2 when a VP is expanded, -1 when collapsed or inspecting custom pin
  cameraMode: "orbit", // "orbit" | "vp-0" | "vp-1" | "vp-2" | "custom-tripod" | "webcam-snap"
  clickMode: "tripod", // "tripod" | "target"
  surfaceStyle: 0, // 0 = Granite Relief, 1 = USGS Satellite, 2 = Topo Contours
  showViewshed: false,
  isPlayingSun: false,
  showSunBeams: false,
  showLightField: false,
  showAtmosphere: true,
  showSightlines: true,
  highlightLandingZone: false,
  webcamReport: null,
  selectedWebcamIdx: 0,
  selectedWebcamFrameIdx: -1,
  showWebcamCard: false,
  webcamGhostEnabled: false,
  webcamGhostOpacity: 0.38,
};

function isPastShootDate(dateStr = state.date) {
  return Boolean(dateStr && dateStr < todayIso);
}

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
const webcamTruthCard = document.getElementById("webcam-truth-card");
const btnToggleWebcamDrawer = document.getElementById("btn-toggle-webcam-drawer");
const btnCloseWebcamCard = document.getElementById("btn-close-webcam-card");
const btnSidebarOpenWebcam = document.getElementById("btn-sidebar-open-webcam");
const btnSnapWebcamCamera = document.getElementById("btn-snap-webcam-camera");
const btnResetWebcamSnap = document.getElementById("btn-reset-webcam-snap");
const webcamGhostOverlayEl = document.getElementById("webcam-ghost-overlay");
const toggleWebcamGhostEl = document.getElementById("toggle-webcam-ghost");
const webcamGhostOpacityEl = document.getElementById("webcam-ghost-opacity");
const valWebcamGhostOpacityEl = document.getElementById("val-webcam-ghost-opacity");
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
  antialias: false,
  powerPreference: "high-performance",
  stencil: false,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
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

function computeDynamicCloudSlant(sunDirY, horizLen = 1.0) {
  return Math.min(8.2, Math.max(0.35, horizLen / Math.max(0.11, sunDirY)));
}

const CORRIDOR_LUT_GLSL1 = /* glsl */ `
  uniform sampler2D uCorridorLutTex;
  uniform float uHasCorridorHorizon;
  uniform float uUseCorridorAtmo;
  uniform float uCorridorLutSunElDeg;

  vec4 sampleCorridorLut(float azimuthDeg, float zAslM) {
    float uAz = fract(azimuthDeg / 360.0) + (0.5 / 180.0);
    float vNorm = pow(clamp(max(0.0, zAslM) / 12500.0, 0.0, 1.0), 1.0 / 1.6);
    float vAlt = vNorm * (31.0 / 32.0) + (0.5 / 32.0);
    return texture2D(uCorridorLutTex, vec2(uAz, vAlt));
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
// - GPU Vertex-Shader Elevation Displacement directly from `uHeightTex` (zero CPU vertex loops)
// - 9-tap Sobel Micro-Relief Normals & Laplacian Cavity Shading
// - O(1) Fetch from GPU-Baked Topographic Soft-Penumbra Shadow & 96-Step Viewshed LUT (`uTerrainBakeTex`)
// - GPU 256x256 Volumetric Cloud Ground-Shadow Projection (`uCloudShadowTex`) & 15m Topographic Fog Damming
// - 3 Surface Modes: Granite Relief (0), USGS Satellite Ortho (1), Topo Contours (2)
const TERRAIN_VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uHeightTex;
  uniform float uGridSize;
  uniform float uMinZ;

  varying vec2 vUv;
  varying vec3 vWorldPos;

  void main() {
    vUv = uv;
    float n = max(uGridSize, 240.0);
    vec2 tcUv = clamp(uv, 0.0, 1.0) * ((n - 1.0) / n) + vec2(0.5 / n);
    float zM = texture2D(uHeightTex, tcUv).r;
    vec3 displaced = vec3(position.x, zM - uMinZ, position.z);
    vec4 wp = modelMatrix * vec4(displaced, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const TERRAIN_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  uniform sampler2D uHeightTex;
  uniform sampler2D uTerrainBakeTex;
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
  ${CORRIDOR_LUT_GLSL1}

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

    // 1. O(1) Lookup from GPU-Baked Topographic Soft-Penumbra Shadow & Grazing Clearance LUT (uTerrainBakeTex)
    //    Re-baked ONLY when solar angle, target coordinate, or DEM changes (replaces 165 ray steps per Retina pixel!)
    vec4 bakedTopo = texture2D(uTerrainBakeTex, vUv);
    float topoLit = (uSunDirENU.z <= -0.01 || ndotl <= 0.002) ? 0.0 : bakedTopo.r;
    float minClearance = bakedTopo.g;
    float closestOccluderDist = bakedTopo.b;
    float bakedViewshedLos = bakedTopo.a;

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

    // 3D Volumetric Cloud Shadow Projection & Coaxial Crepuscular Cloud-Gap Spotlights
    vec2 spUv = clamp(posEN / max(uSpatialSpanM, 1000.0) + 0.5, 0.02, 0.98);
    vec4 scHere = texture2D(uSpatialCloudTex, spUv);
    vec2 horizSun = normalize(uSunDirENU.xy + vec2(1e-5));
    float CLOUD_SLANT = clamp(length(uSunDirENU.xy) / max(0.11, uSunDirENU.z), 0.35, 8.2);

    float gapSpotlight = 0.0;
    float gapCoreBeam = 0.0;
    if (uNumCloudGaps > 0.5) {
      vec2 windDriftEN = uWindDirENU * (uCloudTime * 18.0);
      float cumLayerZ = uMinZ + (uMinZ < 200.0 ? 640.0 : 580.0);
      vec2 cumHitEN = posEN + horizSun * (max(60.0, cumLayerZ - z) * CLOUD_SLANT) - windDriftEN * 1.25;
      float cumMacro = rockFbm(cumHitEN * 0.00085 + vec2(2.3, -1.4));
      float cumTurret = 1.0 - abs(rockFbm(cumHitEN * 0.0042 + vec2(cumMacro * 2.5)) * 2.0 - 1.0);
      float cumWisp1m = rockFbm(cumHitEN * 0.055 - vec2(cumTurret * 2.2));

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
    }

    float cloudTransmittance = 1.0;
    if (uUseFluidShadows > 0.5) {
      // Direct O(1) lookup from the 256x256 GPU 3D Navier-Stokes cloud shadow map (uCloudShadowTex)
      vec2 fluidShadowUv = fract(posEN / max(uCloudDomainM, 1000.0) + 0.5);
      float fluidShadowTrans = texture2D(uCloudShadowTex, fluidShadowUv).r;
      cloudTransmittance = mix(fluidShadowTrans, 0.98, gapSpotlight);
    } else {
      float localFogTopEst = scHere.g > 0.01 ? mix(uFogTopM, scHere.g * 2500.0, 0.78) : uFogTopM;
      float stratusMidZ = max(z + 25.0, (uFogBaseM + localFogTopEst) * 0.5);
      vec2 windDriftEN = uWindDirENU * (uCloudTime * 18.0);
      vec2 stratusHitEN = posEN + horizSun * ((stratusMidZ - z) * CLOUD_SLANT) - windDriftEN;
      vec2 spStratusUv = clamp(stratusHitEN / max(uSpatialSpanM, 1000.0) + 0.5, 0.02, 0.98);
      vec4 scStratus = texture2D(uSpatialCloudTex, spStratusUv);
      float stratusRayCov = scStratus.r * step(z, localFogTopEst + 80.0);
      float stratusMacro = rockFbm(stratusHitEN * 0.00075 + vec2(1.4, -0.9));
      float tauStratus = max(0.0, stratusRayCov - 0.14) * (0.42 + 0.58 * stratusMacro) * 1.48;
      float totalCloudTau = max(tauStratus, scHere.a * 4.5) * (1.0 - gapSpotlight * 0.94);
      cloudTransmittance = mix(exp(-clamp(totalCloudTau * 0.72, 0.0, 4.0)), 0.98, gapSpotlight);
    }

    vec3 corrChromaTint = vec3(1.0);
    if (uUseCorridorAtmo > 0.5 && uUseFluidShadows > 0.5) {
      float sunAzDeg = mod(degrees(atan(uSunDirENU.x, uSunDirENU.y)) + 360.0, 360.0);
      vec4 corr = sampleCorridorLut(sunAzDeg, z);
      vec3 corrMod = computeCorridorAtmoModulation(corr.rgb, z);
      float corrBeamStrength = clamp(dot(corrMod, vec3(0.55, 0.33, 0.12)), 0.0, 1.25);
      cloudTransmittance *= clamp(corrBeamStrength, 0.03, 1.18);
      corrChromaTint = mix(vec3(1.0), clamp(corrMod / max(0.08, corrBeamStrength), 0.35, 1.45), 0.85);
    }

    float isThickOvercast = smoothstep(1.25, 2.6, uMeanCloudTau);
    float ambientFloor = mix(0.38, 0.16, isThickOvercast);
    float effectiveCloudLight = ambientFloor + (1.0 - ambientFloor) * cloudTransmittance;

    // Low-sun Golden Hour factor
    float sunElDeg = degrees(asin(clamp(uSunDirENU.z, -1.0, 1.0)));
    float goldenFactor = smoothstep(-0.5, 1.5, sunElDeg) * (1.0 - smoothstep(9.5, 18.0, sunElDeg));

    // High-contrast Shadowed Umbra vs. Sunlit Area coloring for a razor-sharp light/dark border
    vec3 shadowColor = rockBase * vec3(0.12, 0.19, 0.35) * (0.34 + 0.50 * relZ);
    vec3 warmSunColor = mix(vec3(1.0, 0.92, 0.78), vec3(1.0, 0.58, 0.22), goldenFactor) * corrChromaTint;
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

    // 3. GPU-Baked 360-deg Viewshed Heatmap Overlay (bakedViewshedLos from uTerrainBakeTex — 1 fetch vs 77 steps!)
    if (uShowViewshed > 0.5 && distToTarget > 260.0) {
      if (bakedViewshedLos > 0.5) {
        vec3 viewshedCyan = vec3(0.06, 0.82, 0.62);
        float pulse = 0.36 + 0.18 * smoothstep(0.2, 0.8, relZ);
        finalColor = mix(finalColor, viewshedCyan, pulse);
      } else {
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

const _terrainBox = new THREE.Box3();
const _rayEntryPt = new THREE.Vector3();
const _rayExitPt = new THREE.Vector3();

/**
 * Sub-0.015 ms analytical ray-heightfield intersection (replaces 720,000-triangle CPU
 * `THREE.Raycaster.intersectObject` so `terrainMesh` stays 100% GPU-displaced!).
 */
function intersectTerrainHeightfield(ray) {
  const patch = state.terrainPatch;
  if (!patch) return null;
  const half = patch.span_m * 0.5;
  const maxLocalY = Math.max(20.0, patch.max_elevation_m - patch.min_elevation_m + 15.0);
  _terrainBox.min.set(-half, -10.0, -half);
  _terrainBox.max.set(half, maxLocalY, half);

  let tMin = 0.0;
  if (!_terrainBox.containsPoint(ray.origin)) {
    const hitEntry = ray.intersectBox(_terrainBox, _rayEntryPt);
    if (!hitEntry) return null;
    tMin = ray.origin.distanceTo(_rayEntryPt);
  }

  // Compute exit distance along ray inside bounding box
  let tMax = tMin + patch.span_m * 1.6;
  const invDx = Math.abs(ray.direction.x) > 1e-6 ? 1.0 / ray.direction.x : 1e9;
  const invDy = Math.abs(ray.direction.y) > 1e-6 ? 1.0 / ray.direction.y : 1e9;
  const invDz = Math.abs(ray.direction.z) > 1e-6 ? 1.0 / ray.direction.z : 1e9;
  const tx1 = (-half - ray.origin.x) * invDx;
  const tx2 = (half - ray.origin.x) * invDx;
  const ty1 = (-10.0 - ray.origin.y) * invDy;
  const ty2 = (maxLocalY - ray.origin.y) * invDy;
  const tz1 = (-half - ray.origin.z) * invDz;
  const tz2 = (half - ray.origin.z) * invDz;
  const boxExit = Math.min(Math.max(tx1, tx2), Math.max(ty1, ty2), Math.max(tz1, tz2));
  if (boxExit > tMin) tMax = Math.min(tMax, boxExit);

  const STEPS = 140;
  const dt = (tMax - tMin) / STEPS;
  let prevT = tMin;
  for (let i = 0; i <= STEPS; i++) {
    const t = tMin + i * dt;
    const px = ray.origin.x + ray.direction.x * t;
    const py = ray.origin.y + ray.direction.y * t;
    const pz = ray.origin.z + ray.direction.z * t;
    if (Math.abs(px) <= half && Math.abs(pz) <= half) {
      const terrY = sampleLocalTerrainY(px, -pz);
      if (py <= terrY) {
        let lo = prevT;
        let hi = t;
        for (let b = 0; b < 7; b++) {
          const mid = 0.5 * (lo + hi);
          const mx = ray.origin.x + ray.direction.x * mid;
          const my = ray.origin.y + ray.direction.y * mid;
          const mz = ray.origin.z + ray.direction.z * mid;
          if (my <= sampleLocalTerrainY(mx, -mz)) hi = mid;
          else lo = mid;
        }
        const hitT = 0.5 * (lo + hi);
        const hx = ray.origin.x + ray.direction.x * hitT;
        const hz = ray.origin.z + ray.direction.z * hitT;
        const hy = sampleLocalTerrainY(hx, -hz);
        _rayExitPt.set(hx, hy, hz);
        return { point: _rayExitPt.clone(), distance: hitT };
      }
    }
    prevT = t;
  }
  return null;
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

// Cached flat PlaneGeometry reused across scenarios — displaced 100% in `TERRAIN_VERTEX_SHADER` on GPU!
let cachedTerrainPlaneKey = "";
let cachedTerrainPlaneGeo = null;

// ============================================================================
// Build High-Density 600x600 GPU-Displaced Terrain Mesh + Sync DEM to GPU Fluid Solver
// ============================================================================
function buildTerrainMesh() {
  if (terrainShaderMaterial) {
    terrainShaderMaterial.uniforms?.uHeightTex?.value?.dispose?.();
    if (
      terrainShaderMaterial.uniforms?.uOrthoTex?.value &&
      terrainShaderMaterial.uniforms.uOrthoTex.value !== dummyWhiteTex
    ) {
      terrainShaderMaterial.uniforms.uOrthoTex.value.dispose?.();
    }
    terrainShaderMaterial.dispose();
    terrainShaderMaterial = null;
  }
  terrainGroup.clear();
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

  const meshSegs = Math.min(299, Math.max(1, n - 1));
  const geoKey = `${meshSegs}_${spanM}`;
  if (cachedTerrainPlaneKey !== geoKey || !cachedTerrainPlaneGeo) {
    cachedTerrainPlaneGeo?.dispose?.();
    cachedTerrainPlaneGeo = new THREE.PlaneGeometry(spanM, spanM, meshSegs, meshSegs);
    cachedTerrainPlaneGeo.rotateX(-Math.PI / 2);
    cachedTerrainPlaneGeo.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(0, 1500, 0),
      spanM * 0.9
    );
    cachedTerrainPlaneKey = geoKey;
  }

  const lm = state.shootReport?.target_landmark;
  const upperZ = lm ? lm.upper_face_z_m : maxZ * 0.9;

  terrainShaderMaterial = new THREE.ShaderMaterial({
    vertexShader: TERRAIN_VERTEX_SHADER,
    fragmentShader: TERRAIN_FRAGMENT_SHADER,
    uniforms: {
      uHeightTex: { value: heightTex },
      uTerrainBakeTex: { value: volumetricEngine.getTerrainBakeTexture() },
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
      ...volumetricEngine.getCorridorUniforms(),
    },
  });

  // Couple the high-res USGS 3DEP DEM directly into the GPU 128x128x48 Navier-Stokes & Cirrus solver + DEM Baker!
  volumetricEngine.setTerrainDem(
    floatData,
    n,
    spanM,
    minZ,
    patch.center_lat,
    patch.center_lon,
    heightTex,
    maxZ
  );
  volumetricEngine.updateTerrainBakeUniforms({
    sunDirENU: terrainShaderMaterial.uniforms.uSunDirENU.value,
    targetEN: terrainShaderMaterial.uniforms.uTargetEN.value,
    targetUpperZ: upperZ,
    showViewshed: state.showViewshed,
  });

  const mesh = new THREE.Mesh(cachedTerrainPlaneGeo, terrainShaderMaterial);
  mesh.frustumCulled = false;
  terrainMeshRef = mesh;
  terrainGroup.add(mesh);

  // Sync new heightTex to GPU Light-Field & GPU Monolith Grazing Sunbeam shaders if already initialized
  syncGpuLightOverlaysDem(heightTex, n, spanM, minZ);

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
        wakeRenderLoop(false);
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
    volumetricEngine.updateTerrainBakeUniforms({
      targetEN: terrainShaderMaterial.uniforms.uTargetEN.value,
      targetUpperZ: lm.upper_face_z_m,
    });
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
  const cloudSlant = computeDynamicCloudSlant(sunDirWorld.y, horizLen);
  const ux = sunDirWorld.x / horizLen;
  const uz = sunDirWorld.z / horizLen;
  let maxGapPass = 0.0;
  for (let g = 0; g < activeCloudGaps.length; g++) {
    const gap = activeCloudGaps[g];
    if (worldY < gap.wy + 120) {
      const dy = Math.max(40.0, gap.wy - worldY);
      const hitX = worldX + ux * dy * cloudSlant;
      const hitZ = worldZ + uz * dy * cloudSlant;
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
    const hitX = worldX + ux * Math.max(60, 640 - worldY) * cloudSlant;
    const hitZ = worldZ + uz * Math.max(60, 640 - worldY) * cloudSlant;
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
// ============================================================================
// Persistent GPU Vertex-Shader Raymarched Valley Light-Field & Monolith Grazing Sunbeams
// Eliminates thousands of per-frame CPU raymarch steps & BufferGeometry GC churn!
// ============================================================================
const GPU_VALLEY_LIGHT_VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uHeightTex;
  uniform sampler2D uCloudShadowTex;
  uniform float uGridSize;
  uniform float uSpanM;
  uniform float uMinZ;
  uniform float uCloudDomainM;
  uniform float uUseFluidShadows;
  uniform vec3 uSunDirWorld;
  uniform vec3 uBaseColor;
  uniform vec3 uGrazeColor;
  uniform float uIsOvercast;
  ${CORRIDOR_LUT_GLSL1}

  varying vec4 vColor;

  float sampleLocalY(vec2 xz, out bool inside) {
    float halfSpan = uSpanM * 0.5;
    vec2 uv = vec2((xz.x + halfSpan) / uSpanM, (-xz.y + halfSpan) / uSpanM);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      inside = false;
      return 0.0;
    }
    inside = true;
    float n = max(uGridSize, 240.0);
    vec2 tcUv = uv * ((n - 1.0) / n) + vec2(0.5 / n);
    return texture2D(uHeightTex, tcUv).r - uMinZ;
  }

  void main() {
    // position.x = latOff (-4080..+4080), position.y = dFrac (-0.92..+0.92), position.z = role (0..5)
    float latOff = position.x;
    float dFrac = position.y;
    float role = position.z;

    vec3 propDir = normalize(-uSunDirWorld);
    vec3 rightVec = normalize(vec3(-uSunDirWorld.z, 0.0, uSunDirWorld.x) + vec3(1e-5, 0.0, 0.0));
    float halfSpan = uSpanM * 0.475;
    float stepD = (halfSpan * 1.88) / 36.0;
    float d = dFrac * halfSpan;

    vec3 laneCenter = rightVec * latOff;
    bool inC = false, inM1 = false, inM2 = false, inP1 = false, inP2 = false;
    vec3 pC  = laneCenter + propDir * d;
    vec3 pM1 = laneCenter + propDir * (d - stepD);
    vec3 pM2 = laneCenter + propDir * (d - 2.0 * stepD);
    vec3 pP1 = laneCenter + propDir * (d + stepD);
    vec3 pP2 = laneCenter + propDir * (d + 2.0 * stepD);

    float yC  = sampleLocalY(pC.xz, inC);
    float yM1 = sampleLocalY(pM1.xz, inM1);
    float yM2 = sampleLocalY(pM2.xz, inM2);
    float yP1 = sampleLocalY(pP1.xz, inP1);
    float yP2 = sampleLocalY(pP2.xz, inP2);

    float rC  = yC  - d * propDir.y;
    float rM1 = yM1 - (d - stepD) * propDir.y;
    float rM2 = yM2 - (d - 2.0 * stepD) * propDir.y;
    float rP1 = yP1 - (d + stepD) * propDir.y;
    float rP2 = yP2 - (d + 2.0 * stepD) * propDir.y;

    bool isCrest = inC && inP2 && (rC > rM1) && (rC >= rM2) && (rC >= rP1) && (rC > rP2) && ((yC - yP2) > 24.0);
    if (!isCrest) {
      vColor = vec4(0.0);
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      return;
    }

    vec2 crestEN = vec2(pC.x, -pC.z);
    float zCrestAsl = yC + 5.0 + uMinZ;
    float cloudTrans = 1.0;
    if (uUseFluidShadows > 0.5) {
      vec2 shadowUv = fract(crestEN / max(uCloudDomainM, 1000.0) + 0.5);
      cloudTrans = texture2D(uCloudShadowTex, shadowUv).r;
    }
    if (uHasCorridorHorizon > 0.5 || uUseCorridorAtmo > 0.5) {
      float sunAzDeg = mod(degrees(atan(uSunDirWorld.x, -uSunDirWorld.z)) + 360.0, 360.0);
      float sunElDeg = degrees(asin(clamp(uSunDirWorld.y, -1.0, 1.0)));
      vec4 corr = sampleCorridorLut(sunAzDeg, zCrestAsl);
      if (uHasCorridorHorizon > 0.5) {
        float farH = decodeCorridorHorizonDeg(corr.a);
        cloudTrans *= smoothstep(farH - 0.28, farH + 0.28, sunElDeg);
      }
      if (uUseCorridorAtmo > 0.5 && uUseFluidShadows > 0.5) {
        vec3 cMod = computeCorridorAtmoModulation(corr.rgb, zCrestAsl);
        cloudTrans *= clamp(dot(cMod, vec3(0.55, 0.33, 0.12)), 0.0, 1.15);
      }
    }
    if (cloudTrans < 0.02 || (uIsOvercast > 0.5 && cloudTrans < 0.16)) {
      vColor = vec4(0.0);
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      return;
    }

    vec3 crestOrigin = vec3(pC.x, yC + 5.0, pC.z);
    vec3 endPos = crestOrigin + propDir * 3200.0;
    float endDist = 3200.0;
    for (int m = 1; m <= 16; m++) {
      float md = float(m) * 200.0;
      vec3 mp = crestOrigin + propDir * md;
      bool inM = false;
      float gy = sampleLocalY(mp.xz, inM);
      if (!inM) {
        endPos = mp;
        endDist = md;
        break;
      }
      if (mp.y <= gy + 2.5) {
        endPos = vec3(mp.x, gy + 2.8, mp.z);
        endDist = md;
        break;
      }
    }

    float ribbonH = 68.0;
    vec3 worldPt = crestOrigin;
    vec3 col = uGrazeColor;
    float alpha = 0.0;

    if (role < 0.5) {
      // 0: Sunward feeder start
      worldPt = crestOrigin - propDir * 850.0;
      col = uBaseColor;
      alpha = 0.05 * cloudTrans;
    } else if (role < 1.5) {
      // 1: Crest origin (line)
      worldPt = crestOrigin;
      col = uGrazeColor;
      alpha = 0.44 * cloudTrans;
    } else if (role < 2.5) {
      // 2: Leeward end (line)
      worldPt = endPos;
      col = uGrazeColor;
      alpha = 0.30 * (1.0 - endDist / 4200.0) * cloudTrans;
    } else if (role < 3.5) {
      // 3: Quad bottom-crest
      worldPt = crestOrigin;
      col = uGrazeColor;
      alpha = 0.16 * cloudTrans;
    } else if (role < 4.5) {
      // 4: Quad bottom-leeward
      worldPt = endPos;
      col = uGrazeColor;
      alpha = 0.11 * cloudTrans;
    } else if (role < 5.5) {
      // 5: Quad top-leeward
      worldPt = endPos + vec3(0.0, ribbonH * 0.48, 0.0);
      col = uBaseColor;
      alpha = 0.012 * cloudTrans;
    } else {
      // 6: Quad top-crest
      worldPt = crestOrigin + vec3(0.0, ribbonH, 0.0);
      col = uBaseColor;
      alpha = 0.02 * cloudTrans;
    }

    vColor = vec4(col * alpha, alpha);
    gl_Position = projectionMatrix * viewMatrix * vec4(worldPt, 1.0);
  }
`;

const GPU_OVERLAY_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec4 vColor;
  void main() {
    if (vColor.a <= 0.001) discard;
    gl_FragColor = vec4(vColor.rgb, 1.0);
  }
`;

const GPU_MONOLITH_GRAZING_VERTEX_SHADER = /* glsl */ `
  uniform sampler2D uHeightTex;
  uniform sampler2D uCloudShadowTex;
  uniform float uGridSize;
  uniform float uSpanM;
  uniform float uMinZ;
  uniform float uCloudDomainM;
  uniform float uUseFluidShadows;
  uniform vec3 uSunDirWorld;
  uniform vec3 uTargetWorld;
  uniform vec3 uBeamColor;
  uniform vec3 uEdgeColor;
  uniform float uIsOvercast;
  ${CORRIDOR_LUT_GLSL1}

  varying vec4 vColor;

  float sampleLocalY(vec2 xz, out bool inside) {
    float halfSpan = uSpanM * 0.5;
    vec2 uv = vec2((xz.x + halfSpan) / uSpanM, (-xz.y + halfSpan) / uSpanM);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      inside = false;
      return 0.0;
    }
    inside = true;
    float n = max(uGridSize, 240.0);
    vec2 tcUv = uv * ((n - 1.0) / n) + vec2(0.5 / n);
    return texture2D(uHeightTex, tcUv).r - uMinZ;
  }

  void solveSlice(float latOff, out vec3 grazeOrigin, out vec3 leewardEnd, out float cloudTrans) {
    vec3 propDir = normalize(-uSunDirWorld);
    vec3 rightVec = normalize(vec3(-uSunDirWorld.z, 0.0, uSunDirWorld.x) + vec3(1e-5, 0.0, 0.0));
    vec3 sliceBase = uTargetWorld + rightVec * latOff;
    float bestScore = -1e9;
    vec3 crestPt = sliceBase;

    for (int i = -16; i <= 16; i++) {
      float d = float(i) * 40.0;
      vec3 sp = sliceBase + propDir * d;
      bool inS = false;
      float ty = sampleLocalY(sp.xz, inS);
      if (inS) {
        float rel = ty - d * propDir.y;
        if (rel > bestScore) {
          bestScore = rel;
          crestPt = vec3(sp.x, ty, sp.z);
        }
      }
    }

    grazeOrigin = crestPt + vec3(0.0, 7.0, 0.0);
    leewardEnd = grazeOrigin + propDir * 4200.0;
    for (int m = 1; m <= 35; m++) {
      float md = float(m) * 120.0;
      vec3 mp = grazeOrigin + propDir * md;
      bool inM = false;
      float gy = sampleLocalY(mp.xz, inM);
      if (!inM) {
        leewardEnd = mp;
        break;
      }
      if (mp.y <= gy + 2.5) {
        leewardEnd = vec3(mp.x, gy + 3.0, mp.z);
        break;
      }
    }

    cloudTrans = 1.0;
    if (uUseFluidShadows > 0.5) {
      vec2 en = vec2(grazeOrigin.x, -grazeOrigin.z);
      vec2 suv = fract(en / max(uCloudDomainM, 1000.0) + 0.5);
      cloudTrans = texture2D(uCloudShadowTex, suv).r;
    }
    if (uHasCorridorHorizon > 0.5 || uUseCorridorAtmo > 0.5) {
      float zGrazeAsl = grazeOrigin.y + uMinZ;
      float sunAzDeg = mod(degrees(atan(uSunDirWorld.x, -uSunDirWorld.z)) + 360.0, 360.0);
      float sunElDeg = degrees(asin(clamp(uSunDirWorld.y, -1.0, 1.0)));
      vec4 corr = sampleCorridorLut(sunAzDeg, zGrazeAsl);
      if (uHasCorridorHorizon > 0.5) {
        float farH = decodeCorridorHorizonDeg(corr.a);
        cloudTrans *= smoothstep(farH - 0.28, farH + 0.28, sunElDeg);
      }
      if (uUseCorridorAtmo > 0.5 && uUseFluidShadows > 0.5) {
        vec3 cMod = computeCorridorAtmoModulation(corr.rgb, zGrazeAsl);
        cloudTrans *= clamp(dot(cMod, vec3(0.55, 0.33, 0.12)), 0.0, 1.15);
      }
    }
  }

  void main() {
    // position.x = sliceLatOff, position.y = role, position.z = auxParam
    float latOff = position.x;
    float role = position.y;
    float aux = position.z;

    vec3 crest, leeEnd;
    float rawCt;
    solveSlice(latOff, crest, leeEnd, rawCt);
    float ct = uIsOvercast > 0.5 ? rawCt : (0.25 + 0.75 * rawCt);

    vec3 worldPt = crest;
    vec3 col = uBeamColor;
    float alpha = 0.0;

    if (role < 0.5) {
      // 0: Edge line start (crest)
      worldPt = crest;
      col = aux > 0.5 ? uEdgeColor : uBeamColor;
      alpha = (aux > 0.5 ? 0.82 : 0.48) * ct;
    } else if (role < 1.5) {
      // 1: Edge line end (leewardEnd)
      worldPt = leeEnd;
      col = aux > 0.5 ? uEdgeColor : uBeamColor;
      alpha = (aux > 0.5 ? 0.72 : 0.40) * ct;
    } else if (role < 2.5) {
      // 2: Silhouette rim vertex
      worldPt = crest;
      col = vec3(0.984, 0.749, 0.141);
      alpha = 0.90;
    } else if (role < 3.5) {
      // 3: Ground terminator vertex
      worldPt = leeEnd;
      col = vec3(0.961, 0.620, 0.043);
      alpha = 0.82;
    } else if (role < 4.5) {
      // 4: 34km Sunward feeder shaft start
      worldPt = crest + normalize(uSunDirWorld) * 34000.0;
      col = uBeamColor;
      alpha = 0.18 * ct;
    } else if (role < 5.5) {
      // 5: Vertical curtain bottom (crest or leeEnd via aux: 0=crest, 1=leeEnd)
      worldPt = aux < 0.5 ? crest : leeEnd;
      col = uEdgeColor;
      alpha = 0.19 * ct;
    } else if (role < 6.5) {
      // 6: Vertical curtain top (aux: 0=crest+85m, 1=leeEnd+46m)
      worldPt = (aux < 0.5 ? crest + vec3(0.0, 85.0, 0.0) : leeEnd + vec3(0.0, 46.0, 0.0));
      col = uBeamColor;
      alpha = 0.03 * ct;
    } else {
      // 7: Ribbon sheet vertex (aux: 0=crest, 1=leeEnd)
      worldPt = aux < 0.5 ? crest : leeEnd;
      col = uBeamColor;
      alpha = (aux < 0.5 ? 0.25 : 0.16) * ct;
    }

    vColor = vec4(col * alpha, alpha);
    gl_Position = projectionMatrix * viewMatrix * vec4(worldPt, 1.0);
  }
`;

let gpuValleyLightMat = null;
let gpuValleyPersistentGroup = null;
let gpuValleyGapsGroup = null;
let _lastValleyGapsKey = "";

let gpuMonolithGrazingMat = null;
let gpuMonolithPersistentGroup = null;
let sunBeamsDynamicExtrasGroup = null;
let _lastSunBeamsExtrasKey = "";

function ensureGpuValleyLightField() {
  if (gpuValleyPersistentGroup) return;
  gpuValleyPersistentGroup = new THREE.Group();
  gpuValleyGapsGroup = new THREE.Group();
  lightFieldGroup.add(gpuValleyPersistentGroup);
  lightFieldGroup.add(gpuValleyGapsGroup);

  gpuValleyLightMat = new THREE.ShaderMaterial({
    vertexShader: GPU_VALLEY_LIGHT_VERTEX_SHADER,
    fragmentShader: GPU_OVERLAY_FRAGMENT_SHADER,
    uniforms: {
      uHeightTex: { value: terrainShaderMaterial?.uniforms?.uHeightTex?.value || dummyWhiteTex },
      uCloudShadowTex: { value: volumetricEngine.getShadowMapTexture() },
      uGridSize: { value: state.terrainPatch?.grid_size || 600 },
      uSpanM: { value: state.terrainPatch?.span_m || 9200 },
      uMinZ: { value: state.terrainPatch?.min_elevation_m || 0 },
      uCloudDomainM: { value: volumetricEngine.telemetry.domainXyM || 18000 },
      uUseFluidShadows: { value: 1.0 },
      uSunDirWorld: { value: new THREE.Vector3(-0.85, 0.15, 0.2).normalize() },
      uBaseColor: { value: new THREE.Color(0.99, 0.76, 0.30) },
      uGrazeColor: { value: new THREE.Color(1.0, 0.80, 0.32) },
      uIsOvercast: { value: 0.0 },
      ...volumetricEngine.getCorridorUniforms(),
    },
    transparent: true,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const lineVerts = [];
  const quadVerts = [];
  for (let latOff = -4080; latOff <= 4080; latOff += 240) {
    for (let s = 2; s <= 34; s++) {
      const dFrac = -0.94 + (s / 36.0) * 1.88;
      // Feeder line (0 -> 1) + Leeward ray line (1 -> 2)
      lineVerts.push(latOff, dFrac, 0, latOff, dFrac, 1);
      lineVerts.push(latOff, dFrac, 1, latOff, dFrac, 2);
      // Vertical sunlit area quad (3, 4, 5) + (3, 5, 6)
      quadVerts.push(
        latOff, dFrac, 3,
        latOff, dFrac, 4,
        latOff, dFrac, 5,
        latOff, dFrac, 3,
        latOff, dFrac, 5,
        latOff, dFrac, 6
      );
    }
  }

  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
  const lineMesh = new THREE.LineSegments(lineGeo, gpuValleyLightMat);
  lineMesh.frustumCulled = false;
  gpuValleyPersistentGroup.add(lineMesh);

  const quadGeo = new THREE.BufferGeometry();
  quadGeo.setAttribute("position", new THREE.Float32BufferAttribute(quadVerts, 3));
  const quadMesh = new THREE.Mesh(quadGeo, gpuValleyLightMat);
  quadMesh.frustumCulled = false;
  gpuValleyPersistentGroup.add(quadMesh);
}

function ensureGpuMonolithGrazing() {
  if (gpuMonolithPersistentGroup) return;
  gpuMonolithPersistentGroup = new THREE.Group();
  sunBeamsDynamicExtrasGroup = new THREE.Group();
  sunBeamsGroup.add(gpuMonolithPersistentGroup);
  sunBeamsGroup.add(sunBeamsDynamicExtrasGroup);

  gpuMonolithGrazingMat = new THREE.ShaderMaterial({
    vertexShader: GPU_MONOLITH_GRAZING_VERTEX_SHADER,
    fragmentShader: GPU_OVERLAY_FRAGMENT_SHADER,
    uniforms: {
      uHeightTex: { value: terrainShaderMaterial?.uniforms?.uHeightTex?.value || dummyWhiteTex },
      uCloudShadowTex: { value: volumetricEngine.getShadowMapTexture() },
      uGridSize: { value: state.terrainPatch?.grid_size || 600 },
      uSpanM: { value: state.terrainPatch?.span_m || 9200 },
      uMinZ: { value: state.terrainPatch?.min_elevation_m || 0 },
      uCloudDomainM: { value: volumetricEngine.telemetry.domainXyM || 18000 },
      uUseFluidShadows: { value: 1.0 },
      uSunDirWorld: { value: new THREE.Vector3(-0.85, 0.15, 0.2).normalize() },
      uTargetWorld: { value: new THREE.Vector3(0, 400, 0) },
      uBeamColor: { value: new THREE.Color(0xfbbf24) },
      uEdgeColor: { value: new THREE.Color(0xf59e0b) },
      uIsOvercast: { value: 0.0 },
      ...volumetricEngine.getCorridorUniforms(),
    },
    transparent: true,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const slices = [];
  for (let s = -1260; s <= 1260; s += 70) slices.push(s);

  const lineVerts = [];
  const quadVerts = [];
  slices.forEach((latOff, idx) => {
    const isCenterOrFlank = idx % 2 === 0 || Math.abs(latOff) <= 280 ? 1 : 0;
    // Sharp post-occlusion edge line
    lineVerts.push(latOff, 0, isCenterOrFlank, latOff, 1, isCenterOrFlank);
    // 34 km sunward shaft every 3rd slice
    if (idx % 3 === 0) {
      lineVerts.push(latOff, 4, 0, latOff, 0, 1);
    }
    // Vertical sunlit curtain every 2nd slice
    if (idx % 2 === 0) {
      quadVerts.push(
        latOff, 5, 0,
        latOff, 5, 1,
        latOff, 6, 1,
        latOff, 5, 0,
        latOff, 6, 1,
        latOff, 6, 0
      );
    }
    // Continuous inter-slice ribbon sheet + silhouette rim + ground shadow terminator
    if (idx > 0) {
      const prevOff = slices[idx - 1];
      lineVerts.push(prevOff, 2, 0, latOff, 2, 0);
      lineVerts.push(prevOff, 3, 0, latOff, 3, 0);
      quadVerts.push(
        prevOff, 7, 0,
        latOff, 7, 0,
        latOff, 7, 1,
        prevOff, 7, 0,
        latOff, 7, 1,
        prevOff, 7, 1
      );
    }
  });

  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
  const lineMesh = new THREE.LineSegments(lineGeo, gpuMonolithGrazingMat);
  lineMesh.frustumCulled = false;
  gpuMonolithPersistentGroup.add(lineMesh);

  const quadGeo = new THREE.BufferGeometry();
  quadGeo.setAttribute("position", new THREE.Float32BufferAttribute(quadVerts, 3));
  const quadMesh = new THREE.Mesh(quadGeo, gpuMonolithGrazingMat);
  quadMesh.frustumCulled = false;
  gpuMonolithPersistentGroup.add(quadMesh);
}

let isScrubbingTime = false;
let corridorGatesSubGroup = null;
let nightAstroSubGroup = null;
let customPinRaySubGroup = null;
let _cachedCorridorGates = null;
let _lastCorridorGatesStaticKey = "";
let _lastNightAstroStaticKey = "";
let _lastCustomPinRayKey = "";

function syncGpuLightOverlaysDem(heightTex, gridSize, spanM, minZ) {
  if (gpuValleyLightMat) {
    gpuValleyLightMat.uniforms.uHeightTex.value = heightTex;
    gpuValleyLightMat.uniforms.uGridSize.value = gridSize;
    gpuValleyLightMat.uniforms.uSpanM.value = spanM;
    gpuValleyLightMat.uniforms.uMinZ.value = minZ;
  }
  if (gpuMonolithGrazingMat) {
    gpuMonolithGrazingMat.uniforms.uHeightTex.value = heightTex;
    gpuMonolithGrazingMat.uniforms.uGridSize.value = gridSize;
    gpuMonolithGrazingMat.uniforms.uSpanM.value = spanM;
    gpuMonolithGrazingMat.uniforms.uMinZ.value = minZ;
  }
  _lastValleyGapsKey = "";
  _lastSunBeamsExtrasKey = "";
  _lastCorridorGatesStaticKey = "";
  _lastNightAstroStaticKey = "";
  _lastCustomPinRayKey = "";
}

function updateValleyLightField(sunDirWorld, elevationDeg) {
  if (!state.terrainPatch || !state.showLightField || elevationDeg <= -2.5) {
    lightFieldGroup.visible = false;
    return;
  }
  lightFieldGroup.visible = true;
  ensureGpuValleyLightField();

  const isOvercastWithGaps = state.showAtmosphere && activeMeanCloudTau >= 1.35;
  const isBlueHour = elevationDeg < 1.2;
  const isGolden = elevationDeg <= 13.0;

  // Update GPU Valley Light-Field uniforms in O(1) (0.002 ms CPU, zero geometry rebuild!)
  const u = gpuValleyLightMat.uniforms;
  if (terrainShaderMaterial?.uniforms?.uHeightTex?.value) {
    u.uHeightTex.value = terrainShaderMaterial.uniforms.uHeightTex.value;
  }
  u.uGridSize.value = state.terrainPatch.grid_size;
  u.uSpanM.value = state.terrainPatch.span_m;
  u.uMinZ.value = state.terrainPatch.min_elevation_m;
  u.uCloudDomainM.value = volumetricEngine.telemetry.domainXyM || 18000.0;
  u.uUseFluidShadows.value = state.showAtmosphere ? 1.0 : 0.0;
  u.uSunDirWorld.value.copy(sunDirWorld);
  u.uBaseColor.value.setRGB(
    isBlueHour ? 0.56 : isGolden ? 0.99 : 0.98,
    isBlueHour ? 0.74 : isGolden ? 0.76 : 0.93,
    isBlueHour ? 0.98 : isGolden ? 0.30 : 0.68
  );
  u.uGrazeColor.value.setRGB(
    isBlueHour ? 0.96 : 1.0,
    isBlueHour ? 0.58 : 0.80,
    isBlueHour ? 0.36 : 0.32
  );
  u.uIsOvercast.value = isOvercastWithGaps ? 1.0 : 0.0;

  // Rebuild discrete 3D Crepuscular Cloud-Gap cones ONLY when gaps/azimuth quantized key changes
  const showGaps = state.showAtmosphere && activeCloudGaps.length > 0 && elevationDeg > -1.5;
  if (!showGaps) {
    gpuValleyGapsGroup.visible = false;
    return;
  }
  gpuValleyGapsGroup.visible = true;

  const gapsKey = `${activeCloudGaps.length}_${activeCloudGaps[0]?.eastM}_${sunDirWorld.x.toFixed(2)}_${sunDirWorld.z.toFixed(2)}_${isGolden ? 1 : 0}_${isOvercastWithGaps ? 1 : 0}`;
  if (gapsKey === _lastValleyGapsKey) {
    return;
  }
  _lastValleyGapsKey = gapsKey;

  activeGodRayMaterials = [];
  disposeObjectTree(gpuValleyGapsGroup);

  const horizLen = Math.max(0.01, Math.hypot(sunDirWorld.x, sunDirWorld.z));
  const cloudSlant = computeDynamicCloudSlant(sunDirWorld.y, horizLen);
  const slantDx = (-sunDirWorld.x / horizLen) * cloudSlant;
  const slantDz = (-sunDirWorld.z / horizLen) * cloudSlant;
  const nowSec = performance.now() * 0.001;

  activeCloudGaps.forEach((gap, gIdx) => {
    const gapOrigin = new THREE.Vector3(gap.wx, gap.wy, gap.wz);
    const estGroundY = sampleLocalTerrainY(gap.wx + slantDx * gap.wy, -(gap.wz + slantDz * gap.wy));
    const dropY = Math.max(80, gap.wy - (estGroundY + 3.0));
    const landX = gap.wx + slantDx * dropY;
    const landZ = gap.wz + slantDz * dropY;
    const landY = sampleLocalTerrainY(landX, -landZ) + 3.0;
    const landPt = new THREE.Vector3(landX, landY, landZ);
    const beamVec = landPt.clone().sub(gapOrigin);
    const beamLen = beamVec.length();

    if (beamLen > 60) {
      const topR = Math.max(120, gap.radiusM * 0.34);
      const botR = Math.max(240, gap.radiusM * 0.76);
      const coneGeo = new THREE.CylinderGeometry(topR, botR, beamLen, 24, 1, true);
      const coneMat = new THREE.ShaderMaterial({
        vertexShader: GODRAY_VERTEX_SHADER,
        fragmentShader: GODRAY_FRAGMENT_SHADER,
        uniforms: {
          uBeamColor: { value: new THREE.Color(isGolden ? 0xfbbf24 : 0xfde047) },
          uBaseOpacity: { value: isOvercastWithGaps ? 0.30 : 0.18 },
          uCloudTime: { value: nowSec },
          uWindDirENU: { value: activeWindDirENU.clone() },
        },
        transparent: true,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      activeGodRayMaterials.push(coneMat);
      const coneMesh = new THREE.Mesh(coneGeo, coneMat);
      coneMesh.position.copy(gapOrigin).addScaledVector(beamVec, 0.5);
      coneMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), beamVec.clone().normalize());
      gpuValleyGapsGroup.add(coneMesh);
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
      gpuValleyGapsGroup.add(gapBadge);
    }
  });
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

let _activeCorridorLutSector = null;
let _lastSyncedFluidMinutes = -999;

function applyHourlyWeatherAtMinutes(minutes) {
  const wr = state.weatherReport;
  if (!wr) return null;

  // Automatically swap between precomputed 380 km Sunrise (East) & Sunset (West) Corridor LUTs across solar noon
  const wantSector = minutes < 720 ? "sunrise" : "sunset";
  if (_activeCorridorLutSector !== wantSector) {
    const sectorLut =
      wantSector === "sunrise"
        ? wr.sunrise_corridor_lut || wr.corridor_lut
        : wr.sunset_corridor_lut || wr.corridor_lut;
    if (sectorLut) {
      volumetricEngine.setCorridorLut(sectorLut);
      _activeCorridorLutSector = wantSector;
    }
  }

  const timeline = wr.hourly_timeline;
  if (!Array.isArray(timeline) || timeline.length < 2) return null;

  // Each entry `h` in `hourly_timeline` is centered at `h * 60 + 30` minutes local
  const clampedMin = Math.max(0, Math.min(1439, minutes));
  const fHour = (clampedMin - 30) / 60.0;
  const idx0 = Math.max(0, Math.min(timeline.length - 1, Math.floor(fHour)));
  const idx1 = Math.max(0, Math.min(timeline.length - 1, idx0 + 1));
  const alpha = idx0 === idx1 ? 0.0 : Math.max(0.0, Math.min(1.0, fHour - idx0));

  const row0 = timeline[idx0];
  const row1 = timeline[idx1];
  const lerpNum = (a, b, fallback = 0) => {
    const va = typeof a === "number" ? a : fallback;
    const vb = typeof b === "number" ? b : va;
    return va + (vb - va) * alpha;
  };

  if (
    row0?.fluid_atmosphere_params &&
    row1?.fluid_atmosphere_params &&
    Math.abs(clampedMin - _lastSyncedFluidMinutes) >= 3
  ) {
    _lastSyncedFluidMinutes = clampedMin;
    const p0 = row0.fluid_atmosphere_params;
    const p1 = row1.fluid_atmosphere_params;
    const interpParams = { ...(volumetricEngine.liveWeatherParams || p0) };
    const numericKeys = [
      "surfaceTempC",
      "surfaceRhPct",
      "fogCloudPct",
      "fogBaseAslM",
      "fogTopAslM",
      "cumulusCloudPct",
      "lowCloudPct",
      "lowBaseAslM",
      "lowTopAslM",
      "midCloudPct",
      "midBaseAslM",
      "midTopAslM",
      "midIceFrac",
      "highCloudPct",
      "highBaseAslM",
      "highTopAslM",
      "highOpticalDepth",
      "inversionBaseM",
      "inversionThicknessM",
      "inversionStrengthK",
      "windSfcU",
      "windSfcV",
      "wind850U",
      "wind850V",
      "wind500U",
      "wind500V",
      "wind250U",
      "wind250V",
      "convectiveCape",
      "orographicWaveGain",
      "froudeNumber",
      "vorticityConfinement",
      "cirrusIceBoost",
      "jet250Mph",
      "jet300Mph",
    ];
    for (const k of numericKeys) {
      if (typeof p0[k] === "number" || typeof p1[k] === "number") {
        interpParams[k] = lerpNum(p0[k], p1[k], interpParams[k] ?? 0);
      }
    }
    volumetricEngine.liveWeatherParams = interpParams;
    volumetricEngine.syncWeatherParamsToWorker(false);
  }

  const pickRow = alpha < 0.5 ? row0 : row1;
  const nextAtmoLayers = pickRow?.atmospheric_layers || wr.atmospheric_layers || null;
  if (nextAtmoLayers && nextAtmoLayers !== state.activeAtmosphericLayers) {
    state.activeAtmosphericLayers = nextAtmoLayers;
    renderUnifiedAtmosphereStack(nextAtmoLayers);
  }
  return {
    expBeamPct: Math.round(lerpNum(row0.expected_transmittance_pct, row1.expected_transmittance_pct, wr.ray_transmittance_pct ?? 50)),
    directBeamPct: Math.round(lerpNum(row0.direct_beam_pct, row1.direct_beam_pct, wr.direct_beam_probability_pct ?? 50)),
    p10Pct: Math.round(lerpNum(row0.p10_pct, row1.p10_pct, wr.transmittance_p10_pct ?? 0)),
    p50Pct: Math.round(lerpNum(row0.p50_pct, row1.p50_pct, wr.transmittance_p50_pct ?? 50)),
    p90Pct: Math.round(lerpNum(row0.p90_pct, row1.p90_pct, wr.transmittance_p90_pct ?? 80)),
    kelvin: Math.round(lerpNum(row0.kelvin, row1.kelvin, wr.color_temperature_kelvin ?? 2400)),
    canvasGlow: Math.round(lerpNum(row0.canvas_glow_score, row1.canvas_glow_score, wr.canvas_score ?? 50)),
    fogTopFt: Math.round(lerpNum(row0.fog_top_ft, row1.fog_top_ft, wr.fog_top_ft ?? 1250)),
    cloudLowPct: Math.round(lerpNum(row0.cloud_low_pct, row1.cloud_low_pct, wr.cloud_low_pct ?? 15)),
    aod550: lerpNum(row0.aod_550nm, row1.aod_550nm, wr.astro_atmosphere?.aod_550nm ?? 0.06),
    blockerLabel: pickRow?.blocker_label || "clear",
    blockerDistKm: pickRow?.blocker_distance_km || 0,
  };
}

const _hudTextCache = {
  localTime: "",
  phaseBadge: "",
  sunAz: "",
  sunEl: "",
  cliffLit: "",
  scrubberOut: "",
};

function setTextIfChanged(el, cacheField, nextVal) {
  if (el && _hudTextCache[cacheField] !== nextVal) {
    _hudTextCache[cacheField] = nextVal;
    el.textContent = nextVal;
  }
}

function updateSunRaysAndShader() {
  const hourlyMet = applyHourlyWeatherAtMinutes(state.currentMinutes);
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
    volumetricEngine.updateTerrainBakeUniforms({
      sunDirENU: terrainShaderMaterial.uniforms.uSunDirENU.value,
      showViewshed: state.showViewshed,
      isScrubbing: isScrubbingTime || state.isPlayingSun,
    });
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
  const activeLowPct = hourlyMet ? hourlyMet.cloudLowPct : (wr?.cloud_low_pct || 15);
  const mistCov = wr
    ? Math.max(
        0.14,
        ((wr.valley_inversion_fog_pct || 25) * 0.8 +
          activeLowPct * 0.3 +
          (inv?.cloud_sea_score || 0) * 0.25) /
          110
      )
    : 0.24;

  if (state.terrainPatch) {
    const minElevM = state.terrainPatch.min_elevation_m ?? 0.0;
    const fogTopFt = hourlyMet?.fogTopFt ?? inv?.fog_top_ft ?? (wr?.lcl_cloud_base_ft ? wr.lcl_cloud_base_ft - 600 : 1250);
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
  const aodVal = hourlyMet?.aod550 ?? wr?.astro_atmosphere?.aod_550nm ?? 0.06;
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

  // Update Valley-Wide 3D Solar Light Field on GPU when enabled
  updateValleyLightField(sunDirWorld, elevationDeg);

  // Update GPU-driven Monolith Grazing Crepuscular Silhouettes & Sky Corridor Gates
  ensureGpuMonolithGrazing();
  if (!corridorGatesSubGroup) {
    corridorGatesSubGroup = new THREE.Group();
    nightAstroSubGroup = new THREE.Group();
    customPinRaySubGroup = new THREE.Group();
    sunBeamsDynamicExtrasGroup.add(corridorGatesSubGroup);
    sunBeamsDynamicExtrasGroup.add(nightAstroSubGroup);
    sunBeamsDynamicExtrasGroup.add(customPinRaySubGroup);
  }

  const astroDisclosureEl = document.getElementById("astro-disclosure");
  const shouldBuildSunBeams =
    state.showSunBeams || Boolean(weatherDisclosure?.open) || Boolean(astroDisclosureEl?.open);

  const hasDaySunBeams = Boolean(
    shouldBuildSunBeams && state.shootReport && state.terrainPatch && elevationDeg > -1.0
  );
  gpuMonolithPersistentGroup.visible = hasDaySunBeams;

  const lm = state.shootReport?.target_landmark;
  const targetWorld =
    lm && state.terrainPatch
      ? latLonElevToWorld(lm.upper_face_lat, lm.upper_face_lon, lm.upper_face_z_m)
      : null;

  if (hasDaySunBeams && targetWorld) {
    const isGolden = elevationDeg <= 12.5;
    const isOvercast = state.showAtmosphere && activeMeanCloudTau >= 1.35;

    const mu = gpuMonolithGrazingMat.uniforms;
    if (terrainShaderMaterial?.uniforms?.uHeightTex?.value) {
      mu.uHeightTex.value = terrainShaderMaterial.uniforms.uHeightTex.value;
    }
    mu.uCloudShadowTex.value = volumetricEngine.getShadowMapTexture();
    mu.uGridSize.value = state.terrainPatch.grid_size;
    mu.uSpanM.value = state.terrainPatch.span_m;
    mu.uMinZ.value = state.terrainPatch.min_elevation_m;
    mu.uCloudDomainM.value = volumetricEngine.telemetry.domainXyM || 18000.0;
    mu.uUseFluidShadows.value = state.showAtmosphere ? 1.0 : 0.0;
    mu.uSunDirWorld.value.copy(sunDirWorld);
    mu.uTargetWorld.value.copy(targetWorld);
    mu.uBeamColor.value.setHex(isGolden ? 0xfbbf24 : 0xfde047);
    mu.uEdgeColor.value.setHex(isGolden ? 0xf59e0b : 0xfef08a);
    mu.uIsOvercast.value = isOvercast ? 1.0 : 0.0;
  }

  // Zero-allocation 3D Corridor Gates, Night Astro Streamlines & Custom Tripod Ray
  const astroAtm = state.weatherReport?.astro_atmosphere;
  const astroGeo = state.shootReport?.astro_night_summary;
  const showCorridorGates = Boolean(
    hasDaySunBeams &&
      targetWorld &&
      (state.showAtmosphere || weatherDisclosure?.open) &&
      state.weatherReport?.ray_path_stations?.length === 3
  );
  const showNightAstro = Boolean(
    shouldBuildSunBeams &&
      targetWorld &&
      state.terrainPatch &&
      (astroDisclosureEl?.open || elevationDeg <= -0.5) &&
      astroAtm
  );
  const hasCustomPinRay = Boolean(
    hasDaySunBeams && targetWorld && !inspectorCard.hidden && state.customTripodReport
  );

  if (!showCorridorGates && !showNightAstro && !hasCustomPinRay) {
    sunBeamsDynamicExtrasGroup.visible = false;
  } else {
    sunBeamsDynamicExtrasGroup.visible = true;
    const lmName = lm?.name || "";

    // F. 3D Weather Corridor Cloud Gates — build meshes ONCE per forecast, update transforms in O(1) while scrubbing!
    corridorGatesSubGroup.visible = showCorridorGates;
    if (showCorridorGates) {
      const stations = state.weatherReport.ray_path_stations;
      const stFar = stations[0];
      const stMid = stations[1];
      const st0 = stations[2];
      const corridorStaticKey = `${lmName}_${stFar.distance_km}_${stFar.gate_transmittance_pct}_${stMid.distance_km}_${stMid.gate_transmittance_pct}_${st0.cloud_high_pct}_${inv?.fog_top_ft || 0}`;

      if (corridorStaticKey !== _lastCorridorGatesStaticKey || !_cachedCorridorGates) {
        _lastCorridorGatesStaticKey = corridorStaticKey;
        disposeObjectTree(corridorGatesSubGroup);

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
        corridorGatesSubGroup.add(gate18Ring);

        const gate18Sprite = makeTextSprite(
          `${stMid.distance_km}km GATE: ${stMid.cloud_low_pct}%L/${stMid.cloud_mid_pct}%M/${stMid.cloud_high_pct}%H (${stMid.gate_transmittance_pct}% PASS)`,
          "#0f172a",
          "#38bdf8",
          "#e0f2fe"
        );
        gate18Sprite.scale.set(1150, 235, 1);
        corridorGatesSubGroup.add(gate18Sprite);

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
        corridorGatesSubGroup.add(gate75Ring);

        const gate75Sprite = makeTextSprite(
          `${stFar.distance_km}km ENTRY: ${stFar.cloud_low_pct}%L/${stFar.cloud_mid_pct}%M/${stFar.cloud_high_pct}%H (${stFar.gate_transmittance_pct}% PASS)`,
          "#0f172a",
          "#f59e0b",
          "#fef3c7"
        );
        gate75Sprite.scale.set(1950, 400, 1);
        corridorGatesSubGroup.add(gate75Sprite);

        let fogSprite = null;
        if (inv) {
          const sign = inv.inversion_delta_t_f >= 0 ? "+" : "";
          fogSprite = makeTextSprite(
            `FOG TOP ${inv.fog_top_ft.toLocaleString()} ft • ΔT ${sign}${inv.inversion_delta_t_f}°F • ${wr.temp_sunset_f}°F (${st0.cloud_high_pct}%H/${st0.cloud_low_pct}%L)`,
            "#0f172a",
            "#34d399",
            "#d1fae5"
          );
          fogSprite.scale.set(540, 110, 1);
          corridorGatesSubGroup.add(fogSprite);
        }

        _cachedCorridorGates = {
          gate18Ring,
          gate18Sprite,
          gate75Ring,
          gate75Sprite,
          fogSprite,
        };
      }

      // Fast O(1) transform update along live solar azimuth/elevation (zero allocations!)
      const cg = _cachedCorridorGates;
      cg.gate18Ring.position.copy(targetWorld).addScaledVector(sunDirWorld, 10500);
      cg.gate18Ring.lookAt(targetWorld);
      cg.gate18Sprite.position.copy(cg.gate18Ring.position);
      cg.gate18Sprite.position.y += 620;

      cg.gate75Ring.position.copy(targetWorld).addScaledVector(sunDirWorld, 22500);
      cg.gate75Ring.lookAt(targetWorld);
      cg.gate75Sprite.position.copy(cg.gate75Ring.position);
      cg.gate75Sprite.position.y += 1100;

      if (cg.fogSprite) {
        cg.fogSprite.position.set(
          targetWorld.x + sunDirWorld.x * 1350,
          activeFogTopLocalY + 95,
          targetWorld.z + sunDirWorld.z * 1350
        );
      }
    }

    // Night Astrophotography Jet Stream Streamlines, AOD Badge & Galactic Core Vector (independent of sun scrubber!)
    nightAstroSubGroup.visible = showNightAstro;
    if (showNightAstro) {
      const nightStaticKey = `${lmName}_${astroAtm.jet_stream_250hpa_mph}_${astroAtm.aod_550nm}_${astroGeo?.galactic_core_peak_az_deg ?? 205}`;
      if (nightStaticKey !== _lastNightAstroStaticKey) {
        _lastNightAstroStaticKey = nightStaticKey;
        disposeObjectTree(nightAstroSubGroup);

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
          nightAstroSubGroup.add(new THREE.Line(streamGeo, streamMat));
        }

        const jetBadge = makeTextSprite(
          `250hPa JET STREAM (${astroAtm.jet_stream_250hpa_mph} MPH • SEEING ${astroAtm.seeing_score}/100)`,
          "#0f172a",
          "#818cf8",
          "#e0e7ff"
        );
        jetBadge.scale.set(460, 88, 1);
        jetBadge.position.set(targetWorld.x - 450, jetY + 150, targetWorld.z - 250);
        nightAstroSubGroup.add(jetBadge);

        const hazeColor = astroAtm.aod_550nm < 0.12 ? "#34d399" : "#f59e0b";
        const hazeBadge = makeTextSprite(
          `UPPER HAZE / AOD ${astroAtm.aod_550nm} (${astroAtm.transparency_score}% TRANSPARENCY)`,
          "#0f172a",
          hazeColor,
          "#ecfdf5"
        );
        hazeBadge.scale.set(450, 86, 1);
        hazeBadge.position.set(targetWorld.x + 650, targetWorld.y + 820, targetWorld.z + 320);
        nightAstroSubGroup.add(hazeBadge);

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
          nightAstroSubGroup.add(new THREE.Mesh(gcTube, gcMat));

          const gcSprite = makeTextSprite(
            `🌌 GALACTIC CORE (AZ ${gcPeakAz}° • +${gcClearDeg}° CLEAR)`,
            "#0f172a",
            "#c084fc",
            "#f3e8ff"
          );
          gcSprite.scale.set(450, 86, 1);
          gcSprite.position.copy(gcEnd).add(new THREE.Vector3(0, 90, 0));
          nightAstroSubGroup.add(gcSprite);
        }
      }
    }

    // E. Custom tripod uprange grazing ray (only active when custom pin inspector is open)
    customPinRaySubGroup.visible = hasCustomPinRay;
    if (hasCustomPinRay) {
      const rep = state.customTripodReport;
      const pinRayKey = `${rep.tripod_lat.toFixed(4)}_${rep.tripod_lon.toFixed(4)}_${sunDirWorld.x.toFixed(2)}_${sunDirWorld.y.toFixed(2)}_${sunDirWorld.z.toFixed(2)}`;
      if (pinRayKey !== _lastCustomPinRayKey) {
        _lastCustomPinRayKey = pinRayKey;
        disposeObjectTree(customPinRaySubGroup);

        const SUN_DIST_M = 34000;
        const propDir = sunDirWorld.clone().negate();
        const halfSpan = state.terrainPatch.span_m * 0.485;
        const pinWorld = latLonElevToWorld(rep.tripod_lat, rep.tripod_lon, rep.tripod_elevation_m);
        let maxObstructionScore = -Infinity;
        let occluderCrest = null;

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
          customPinRaySubGroup.add(new THREE.Line(customRayGeo, customRayMat));
        }
      }
    }
  }

  sunBeamsGroup.visible = shouldBuildSunBeams && state.cameraMode === "orbit";

  // Update unified bottom Scrubber & telemetry readouts (diff-guarded against unnecessary DOM layout)
  const hhmm = formatMinutesToHHMM(state.currentMinutes);
  const comp = compassLabel(azimuthDeg);
  const elSign = elevationDeg >= 0 ? "+" : "";

  let phaseLabel = "Daylight High Sun";
  let cliffLitLabel = hourlyMet
    ? `${hourlyMet.expBeamPct}% Beam · P90 ${hourlyMet.p90Pct}% (${hourlyMet.kelvin}K)`
    : "82% Direct Light";
  if (elevationDeg < -0.5) {
    const astro = state.shootReport?.astro_night_summary;
    const track = astro?.nightly_celestial_track || [];
    let bestCel = null;
    for (const row of track) {
      if (!bestCel || Math.abs(row[0] - state.currentMinutes) < Math.abs(bestCel[0] - state.currentMinutes)) {
        bestCel = row;
      }
    }
    if (elevationDeg >= -5.5 && hourlyMet) {
      phaseLabel = state.currentMinutes >= 720 ? "Civil Twilight / Cirrus Afterglow" : "Pre-Dawn Civil Twilight";
      cliffLitLabel = `Afterglow ${hourlyMet.canvasGlow}/100`;
    } else if (bestCel && astro) {
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
    if (!hourlyMet) cliffLitLabel = "96% Alpenglow";
  } else if (elevationDeg <= 13.5 && state.currentMinutes < 720) {
    phaseLabel = "Sunrise Golden Hour";
    if (!hourlyMet) cliffLitLabel = "88% Rim Light";
  }

  setTextIfChanged(document.getElementById("hud-local-time"), "localTime", `${hhmm} Local`);
  setTextIfChanged(document.getElementById("hud-phase-badge"), "phaseBadge", phaseLabel);
  setTextIfChanged(document.getElementById("hud-sun-az"), "sunAz", `${azimuthDeg.toFixed(1)}° ${comp}`);
  setTextIfChanged(document.getElementById("hud-sun-el"), "sunEl", `${elSign}${elevationDeg.toFixed(1)}°`);
  setTextIfChanged(document.getElementById("hud-cliff-lit"), "cliffLit", cliffLitLabel);
  setTextIfChanged(timeOutput, "scrubberOut", `${hhmm} • ${phaseLabel}`);
  wakeRenderLoop(false);
}

function syncWebcamGhostOverlay() {
  if (!webcamGhostOverlayEl) return;
  const cam =
    state.webcamReport?.cameras?.[state.selectedWebcamIdx] ||
    state.webcamReport?.primary_camera;
  const frames = cam?.timeline_frames || [];
  const activeFrame =
    (state.selectedWebcamFrameIdx >= 0 && frames[state.selectedWebcamFrameIdx]) ||
    cam?.primary_frame;
  const shouldShow =
    Boolean(state.webcamGhostEnabled) &&
    state.cameraMode === "webcam-snap" &&
    Boolean(activeFrame?.proxy_url);

  if (!shouldShow) {
    webcamGhostOverlayEl.hidden = true;
    return;
  }
  if (webcamGhostOverlayEl.dataset.activeSrc !== activeFrame.proxy_url) {
    webcamGhostOverlayEl.dataset.activeSrc = activeFrame.proxy_url;
    webcamGhostOverlayEl.src = activeFrame.proxy_url;
  }
  webcamGhostOverlayEl.style.opacity = String(state.webcamGhostOpacity);
  webcamGhostOverlayEl.hidden = false;
}

// ============================================================================
// Camera Mode Switching (3D Valley Orbit vs. VP1..VP3 or Custom Tripod Viewfinder)
// ============================================================================
function setCameraMode(mode) {
  state.cameraMode = mode;
  volumetricEngine.invalidateTemporalHistory();
  if (btnResetWebcamSnap) {
    btnResetWebcamSnap.hidden = mode !== "webcam-snap";
  }
  syncWebcamGhostOverlay();

  if (mode === "orbit") {
    if (btnOrbitMode) btnOrbitMode.hidden = true;
    if (hudClickModeGroup) hudClickModeGroup.hidden = false;
    viewportStageEl?.classList.remove("is-viewfinder", "is-webcam-snap");
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
    wakeRenderLoop(true);
    return;
  }

  if (btnOrbitMode) btnOrbitMode.hidden = false;
  if (hudClickModeGroup) hudClickModeGroup.hidden = true;
  viewportStageEl?.classList.add("is-viewfinder");
  viewportStageEl?.classList.toggle("is-webcam-snap", mode === "webcam-snap");
  trailsGroup.visible = false;
  customPinGroup.visible = false;
  sunBeamsGroup.visible = false;

  const lm = state.shootReport?.target_landmark;
  if (!lm) return;

  if (mode === "webcam-snap") {
    const cam =
      state.webcamReport?.cameras?.[state.selectedWebcamIdx] ||
      state.webcamReport?.primary_camera;
    if (!cam || !state.terrainPatch) return;

    const patch = state.terrainPatch;
    const hfovDeg = Math.max(24.0, Math.min(92.0, Number(cam.hfov_deg || 62.0)));
    const hfovRad = (hfovDeg * Math.PI) / 180.0;
    const vfovDeg =
      (2.0 * Math.atan(Math.tan(hfovRad * 0.5) / Math.max(0.45, camera.aspect)) * 180.0) /
      Math.PI;
    camera.fov = Math.max(20.0, Math.min(76.0, vfovDeg));
    camera.updateProjectionMatrix();

    const candWorld = latLonElevToWorld(cam.lat, cam.lon, cam.elev_m);
    const halfSpan = patch.span_m * 0.5;
    const insidePatch =
      Math.abs(candWorld.x) <= halfSpan * 1.15 && Math.abs(candWorld.z) <= halfSpan * 1.15;

    let camWorld;
    if (insidePatch) {
      const clampedX = Math.max(-halfSpan * 0.98, Math.min(halfSpan * 0.98, candWorld.x));
      const clampedZ = Math.max(-halfSpan * 0.98, Math.min(halfSpan * 0.98, candWorld.z));
      const terrY = sampleLocalTerrainY(clampedX, -clampedZ);
      camWorld = new THREE.Vector3(clampedX, Math.max(candWorld.y, terrY + 12.0), clampedZ);
    } else {
      const vp0 =
        state.shootReport?.top_viewpoints?.[Math.max(0, state.selectedVpIndex)] ||
        state.shootReport?.top_viewpoints?.[0];
      if (vp0) {
        camWorld = latLonElevToWorld(vp0.viewpoint_lat, vp0.viewpoint_lon, vp0.viewpoint_elevation_m);
      } else {
        camWorld = latLonElevToWorld(lm.summit_lat, lm.summit_lon, lm.summit_elevation_m);
      }
      const terrY = sampleLocalTerrainY(camWorld.x, -camWorld.z);
      camWorld.y = Math.max(camWorld.y, terrY) + 14.0;
    }

    const lookAzDeg = Number(cam.azimuth_deg ?? state.weatherReport?.sun_azimuth_deg ?? 259.0);
    const tiltDeg = Math.max(-12.0, Math.min(8.0, Number(cam.tilt_deg ?? -1.2)));
    const azRad = (lookAzDeg * Math.PI) / 180.0;
    const tiltRad = (tiltDeg * Math.PI) / 180.0;
    const cosT = Math.cos(tiltRad);
    const dirWorld = new THREE.Vector3(
      Math.sin(azRad) * cosT,
      Math.sin(tiltRad),
      -Math.cos(azRad) * cosT
    ).normalize();

    camera.position.copy(camWorld);
    controls.target.copy(camWorld).addScaledVector(dirWorld, 2600.0);
    controls.update();

    // Sync solar timeline to the active webcam frame timestamp on historical replays
    const frames = cam.timeline_frames || [];
    const activeFrame =
      (state.selectedWebcamFrameIdx >= 0 && frames[state.selectedWebcamFrameIdx]) ||
      cam.primary_frame;
    if (isPastShootDate(state.date) && activeFrame?.minutes_local != null) {
      state.currentMinutes = Math.max(330, Math.min(1410, Number(activeFrame.minutes_local)));
      timeScrubber.value = String(Math.round(state.currentMinutes));
      syncSliderValue(timeScrubber);
    }
    updateSunRaysAndShader();

    const posTag = insidePatch ? "EXACT WEBCAM GPS" : "MATCHED AZIMUTH/FOV";
    vfOverlay.hidden = false;
    vfHudReadout.textContent = `WEBCAM SNAP (${posTag}) · ${cam.short_name || cam.camera_id} · Az ${lookAzDeg.toFixed(1)}° (${compassLabel(lookAzDeg)}) · HFOV ${Math.round(hfovDeg)}°`;
    wakeRenderLoop(true);
    return;
  }

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
    volumetricEngine.invalidateTemporalHistory();
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
    summarySub.textContent = `${w.ray_transmittance_pct}% Beam • ${w.temp_sunset_f}°F • Click to inspect 380 km ray corridor`;
  }

  document.getElementById("raypath-azimuth-label").textContent =
    `${w.phase} Ray Corridor: Az ${w.sun_azimuth_deg}° ${w.corridor_direction} (El +${w.sun_elevation_deg}°)`;
  document.getElementById("raypath-trans-pill").textContent =
    `End-to-End Beam: ${w.ray_transmittance_pct}%`;

  // Populate 380 km Curved-Earth Bimodal & Spectral Telemetry (#corridor-bimodal-box)
  const nowcastPillEl = document.getElementById("corridor-nowcast-pill");
  if (nowcastPillEl) {
    if (w.historical_replay?.is_observed_truth && w.nowcast_status?.active) {
      const stId = w.nowcast_status.metar_station?.id || "ABI";
      const raobId = w.nowcast_status.raob_station?.id ? `+${w.nowcast_status.raob_station.id}` : "";
      nowcastPillEl.textContent = `📡 Observed Truth (${stId}${raobId})`;
    } else if ((w.historical_replay?.hindcast_lead_days || 0) > 0) {
      nowcastPillEl.textContent = `⏪ +${w.historical_replay.hindcast_lead_days * 24}h Hindcast`;
    } else if (w.nowcast_status?.active) {
      const stId = w.nowcast_status.metar_station?.id || "ABI";
      const wtPct = Math.round((w.nowcast_status.nowcast_blend_weight || 0) * 100);
      nowcastPillEl.textContent = `📡 GOES-18 + ${stId} (${wtPct}%)`;
    } else if (w.inversion_model?.is_historical) {
      nowcastPillEl.textContent = "📚 Historical Archive";
    } else {
      nowcastPillEl.textContent = "🛰️ 6-Station 380 km";
    }
  }
  const directProbEl = document.getElementById("val-direct-beam-prob");
  if (directProbEl) {
    directProbEl.textContent = `${w.direct_beam_probability_pct ?? w.ray_transmittance_pct}%`;
  }
  const quantEl = document.getElementById("val-bimodal-quantiles");
  if (quantEl) {
    const p10 = w.transmittance_p10_pct ?? 0;
    const p50 = w.transmittance_p50_pct ?? w.ray_transmittance_pct;
    const p90 = w.transmittance_p90_pct ?? w.ray_transmittance_pct;
    quantEl.textContent = `${p10} / ${p50} / ${p90}%`;
  }
  const kelvEl = document.getElementById("val-kelvin-canvas");
  if (kelvEl) {
    kelvEl.textContent = `${w.color_temperature_kelvin ?? 2400} K · ${w.canvas_score ?? 50}/100`;
  }
  const blockerEl = document.getElementById("val-dominant-blocker");
  if (blockerEl) {
    const blkLbl = w.corridor_raycast?.dominant_blocker_label || "";
    const blkDist = Math.round(w.corridor_raycast?.dominant_blocker_distance_km || 0);
    if ((w.ray_transmittance_pct ?? 0) >= 65 && (!blkLbl || blkLbl === "clear_corridor")) {
      blockerEl.textContent = "Clear 380 km";
    } else if (!blkLbl || blkLbl === "clear_corridor") {
      blockerEl.textContent = "Clear / Haze";
    } else {
      blockerEl.textContent = `${blkLbl.toUpperCase()} @ ${blkDist} km`;
    }
  }

  // Render SVG 3-Station Atmospheric Ray-Path Cross-Section (Far Entry -> Mid Keyhole -> 0 km)
  const svg = document.getElementById("raypath-svg");
  const stations = w.ray_path_stations || [];
  if (svg && stations.length === 3) {
    const [stFar, stMid, st0] = stations;
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

      <!-- Mid/Low Cloud Decks at Far Entry & Mid Keyhole Ridge -->
      <rect x="14" y="24" width="76" height="11" rx="5" fill="rgba(148,163,184,0.25)" stroke="#64748b" stroke-width="0.8" />
      <text x="52" y="32" fill="#cbd5e1" font-size="7" text-anchor="middle">${stFar.distance_km}km Deck (${stFar.cloud_low_pct}%L/${stFar.cloud_mid_pct}%M)</text>

      <rect x="128" y="31" width="74" height="11" rx="5" fill="rgba(56,189,248,0.18)" stroke="#38bdf8" stroke-width="0.8" />
      <text x="165" y="39" fill="#bae6fd" font-size="7" text-anchor="middle">${stMid.distance_km}km Cap (${stMid.cloud_low_pct}%L/${stMid.cloud_mid_pct}%M)</text>

      <!-- Terrain Silhouette: Far Foothills -> Mid Keyhole Ridge -> 0 km Monolith -->
      <path d="M 8 84 L 8 72 L 65 70 L 125 66 L 158 49 L 172 56 L 188 47 L 235 74 L 296 74 L 316 34 L 335 34 L 348 84 Z"
            fill="rgba(30,41,59,0.88)" stroke="#64748b" stroke-width="1.2" />

      <!-- Incoming Solar Ray Shaft from Far Entry -> Mid Keyhole -> 0 km Monolith Face -->
      <line x1="16" y1="42" x2="172" y2="52" stroke="url(#beamGrad)" stroke-width="3.2" />
      <line x1="172" y1="52" x2="316" y2="41" stroke="url(#beamGrad)" stroke-width="3.2" />
      <!-- Reflected Alpenglow Rays up onto Overhead High Cirrus Canvas -->
      <line x1="316" y1="41" x2="275" y2="19" stroke="#fb923c" stroke-width="1.3" stroke-dasharray="2,2" />

      <!-- Station Markers -->
      <circle cx="24" cy="42" r="4" fill="#fde047" />
      <circle cx="172" cy="52" r="4" fill="#38bdf8" />
      <circle cx="316" cy="41" r="4.5" fill="#f97316" />

      <text x="20" y="81" fill="#94a3b8" font-size="7.2" font-weight="700">${stFar.distance_km} km Entry</text>
      <text x="142" y="81" fill="#94a3b8" font-size="7.2" font-weight="700">${stMid.distance_km} km Keyhole</text>
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
        graphcast: "🧠 AI Forecast",
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
      vpWeatherPill.title = `${engTag}${w.temp_sunset_f}°F • Wind ${w.wind_mph} mph • Fog ${inv.fog_base_ft.toLocaleString()}–${inv.fog_top_ft.toLocaleString()} ft (ΔT ${sign}${inv.inversion_delta_t_f}°F)${tauTag} — Click to inspect 380 km ray corridor`;
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
  const mcg = w.mesoscale_cloud_grid;
  if (transectGrid && scf?.transect_west_to_east) {
    if (transectPill) {
      const gapCount = scf?.crepuscular_gaps?.length || 0;
      const gSize = mcg?.grid_size || 64;
      const tauStr = scf?.mean_optical_depth_tau !== undefined ? ` • τ=${scf.mean_optical_depth_tau} (${gapCount} Gaps)` : "";
      transectPill.textContent = `Wind ${Math.round(scf.wind_dir_deg || 295)}° • ${gSize}×${gSize} (1.75 km)${tauStr}`;
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

  state.activeAtmosphericLayers = w.atmospheric_layers || null;
  renderUnifiedAtmosphereStack(state.activeAtmosphericLayers);
  renderHistoricalReplayPanel(w);
  renderAstroCard(w, state.shootReport);
}

function formatSignedInt(val, suffix = "") {
  const n = Math.round(val || 0);
  return `${n >= 0 ? "+" : ""}${n.toLocaleString()}${suffix}`;
}

function renderHistoricalReplayPanel(w) {
  const panelEl = document.getElementById("historical-replay-panel");
  if (!panelEl) return;
  if (!isPastShootDate(state.date)) {
    panelEl.hidden = true;
    return;
  }
  panelEl.hidden = false;

  const activeMode = state.replayMode || "truth";
  panelEl.querySelectorAll("[data-replay-mode]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.replayMode === activeMode);
  });

  const badgeEl = document.getElementById("historical-replay-badge");
  if (badgeEl) {
    if (activeMode === "truth") {
      badgeEl.textContent = "Observed Truth";
      badgeEl.className = "conf-pill conf-high";
    } else if (activeMode === "nowcast_1h") {
      badgeEl.textContent = "+1h vs. Truth";
      badgeEl.className = "conf-pill conf-high";
    } else if (activeMode === "hindcast_24h") {
      badgeEl.textContent = "+24h vs. Truth";
      badgeEl.className = "conf-pill conf-moderate";
    } else {
      badgeEl.textContent = "+72h vs. Truth";
      badgeEl.className = "conf-pill conf-low";
    }
  }

  const deltaEl = document.getElementById("historical-delta-strip");
  if (!deltaEl) return;
  const truth = state.cachedTruthReport;
  if (activeMode === "truth" || !w || !truth) {
    deltaEl.hidden = true;
    return;
  }

  const dLow = (w.cloud_low_pct ?? 0) - (truth.cloud_low_pct ?? 0);
  const dHigh = (w.cloud_high_pct ?? 0) - (truth.cloud_high_pct ?? 0);
  const wLid = w.inversion_model?.fog_top_ft ?? w.fog_top_ft ?? 0;
  const tLid = truth.inversion_model?.fog_top_ft ?? truth.fog_top_ft ?? 0;
  const dLid = wLid - tLid;
  const dBeam = (w.ray_transmittance_pct ?? 0) - (truth.ray_transmittance_pct ?? 0);
  const dGlow = (w.alpenglow_score ?? 0) - (truth.alpenglow_score ?? 0);

  const clsPct = (err) => (Math.abs(err) <= 12 ? "delta-good" : Math.abs(err) >= 28 ? "delta-warn" : "");
  const clsFt = (err) => (Math.abs(err) <= 350 ? "delta-good" : Math.abs(err) >= 900 ? "delta-warn" : "");

  deltaEl.hidden = false;
  deltaEl.innerHTML = `
    <div class="hist-delta-cell" title="Forecast Low/Fog coverage minus Observed Truth (${truth.cloud_low_pct ?? 0}%)">
      <span>Low/Fog Δ</span>
      <strong class="${clsPct(dLow)}">${formatSignedInt(dLow, "pp")}</strong>
    </div>
    <div class="hist-delta-cell" title="Forecast High Cirrus coverage minus Observed Truth (${truth.cloud_high_pct ?? 0}%)">
      <span>Cirrus Δ</span>
      <strong class="${clsPct(dHigh)}">${formatSignedInt(dHigh, "pp")}</strong>
    </div>
    <div class="hist-delta-cell" title="Forecast Fog/Inversion Lid minus Balloon+ASOS Observed Lid (${tLid.toLocaleString()} ft)">
      <span>Fog Lid Δ</span>
      <strong class="${clsFt(dLid)}">${formatSignedInt(dLid, " ft")}</strong>
    </div>
    <div class="hist-delta-cell" title="Forecast Beam Transmittance &amp; Alpenglow Score minus Observed Truth">
      <span>Beam / Glow Δ</span>
      <strong class="${clsPct(dBeam)}">${formatSignedInt(dBeam, "%")} · ${formatSignedInt(dGlow)}</strong>
    </div>
  `;
}

function escapeHtmlText(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setHoveredCloudLayerHighlight(layerId) {
  if (state.hoveredCloudLayerId === layerId) return;
  state.hoveredCloudLayerId = layerId || null;
  const stackEl = document.getElementById("unified-layer-stack");
  if (!stackEl) return;
  stackEl.querySelectorAll(".atmo-layer-row[data-layer-id]").forEach((row) => {
    row.classList.toggle("is-hovered-layer", row.dataset.layerId === state.hoveredCloudLayerId);
  });
}

function renderUnifiedAtmosphereStack(atmoLayers) {
  if (!atmoLayers || !Array.isArray(atmoLayers.layers)) return;

  const pipelineEl = document.getElementById("atmo-pipeline-label");
  const overallPillEl = document.getElementById("atmo-overall-conf-pill");
  const simDiscEl = document.getElementById("atmo-sim-disclosure");
  const regimeTitleEl = document.getElementById("fluid-regime-title");
  const stackEl = document.getElementById("unified-layer-stack");

  if (pipelineEl && atmoLayers.primary_pipeline_label) {
    pipelineEl.textContent = atmoLayers.primary_pipeline_label;
  }
  if (overallPillEl && typeof atmoLayers.overall_confidence_pct === "number") {
    const tier = atmoLayers.overall_confidence_tier || "moderate";
    overallPillEl.textContent = `${Math.round(atmoLayers.overall_confidence_pct)}% Conf`;
    overallPillEl.className = `conf-pill conf-${tier}`;
  }
  if (simDiscEl && atmoLayers.simulation_note) {
    simDiscEl.textContent = atmoLayers.simulation_note;
  }

  if (regimeTitleEl) {
    const activeDecks = atmoLayers.layers.filter((l) => (l.coverage_pct ?? 0) >= 8);
    if (activeDecks.length === 0) {
      regimeTitleEl.textContent = `Clear Column (${Math.round(atmoLayers.overall_confidence_pct ?? 80)}% Conf)`;
    } else if (activeDecks.length === 1) {
      regimeTitleEl.textContent = `${activeDecks[0].short_name} (${Math.round(activeDecks[0].coverage_pct)}%)`;
    } else if (activeDecks.length === 2) {
      regimeTitleEl.textContent = `${activeDecks[0].short_name} + ${activeDecks[1].short_name}`;
    } else {
      regimeTitleEl.textContent = `${activeDecks.length}-Layer Cloud Column (${activeDecks[0].short_name} + ${activeDecks[activeDecks.length - 1].short_name})`;
    }
  }

  if (stackEl) {
    stackEl.innerHTML = atmoLayers.layers
      .map((layer) => {
        const cov = Math.round(layer.coverage_pct ?? 0);
        const conf = Math.round(layer.confidence_pct ?? 60);
        const tier = layer.confidence_tier || (conf >= 75 ? "high" : conf >= 52 ? "moderate" : "low");
        const baseFt = Math.round(layer.base_ft ?? (layer.base_asl_m || 0) * 3.28084).toLocaleString();
        const topFt = Math.round(layer.top_ft ?? (layer.top_asl_m || 0) * 3.28084).toLocaleString();
        const baseKm = ((layer.base_asl_m || 0) / 1000).toFixed(1);
        const topKm = ((layer.top_asl_m || 0) / 1000).toFixed(1);
        const primarySrc = layer.sources?.[0] || "";
        const isTrace = cov <= 3;
        const isHovered = state.hoveredCloudLayerId === layer.id;
        const rowClasses = [
          "atmo-layer-row",
          isTrace ? "is-trace" : "",
          isHovered ? "is-hovered-layer" : "",
        ]
          .filter(Boolean)
          .join(" ");

        return `
          <div class="${rowClasses}" role="listitem" data-layer-id="${escapeHtmlText(layer.id)}" title="${escapeHtmlText(layer.hover_disclosure || "")}">
            <div class="atmo-layer-main">
              <span class="atmo-layer-name">${escapeHtmlText(layer.short_name || layer.name)}</span>
              <span class="atmo-layer-alt">${baseFt}–${topFt} ft ASL (${baseKm}–${topKm} km)</span>
              ${primarySrc ? `<span class="atmo-layer-source">${escapeHtmlText(primarySrc)}</span>` : ""}
            </div>
            <div class="atmo-layer-metrics">
              <span class="atmo-layer-cov">${cov}%</span>
              <span class="conf-pill conf-${escapeHtmlText(tier)}" title="Layer data confidence: ${conf}%">${conf}%</span>
            </div>
          </div>
        `;
      })
      .join("");
  }
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

function buildTruthCacheKey(lat, lon, dateStr, phase) {
  return `${Number(lat).toFixed(4)}:${Number(lon).toFixed(4)}:${dateStr}:${phase.toLowerCase()}`;
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

  const isPast = isPastShootDate(state.date);
  const effectiveReplay = isPast ? (state.replayMode || "truth") : "auto";
  const truthKey = buildTruthCacheKey(lm.summit_lat, lm.summit_lon, state.date, phase);
  const baseUrl = `/api/weather-forecast?lat=${lm.summit_lat}&lon=${lm.summit_lon}&date=${encodeURIComponent(state.date)}&phase=${phase}&sun_az=${sunAz}&sun_el=${sunEl}&elev_m=${lm.summit_elevation_m}&engine=${encodeURIComponent(state.weatherEngine)}`;

  const needBgTruth = isPast && effectiveReplay !== "truth" && state.cachedTruthKey !== truthKey;
  const [res, bgTruthRes] = await Promise.all([
    fetch(`${baseUrl}&replay_mode=${encodeURIComponent(effectiveReplay)}`),
    needBgTruth ? fetch(`${baseUrl}&replay_mode=truth`) : Promise.resolve(null),
  ]);

  if (bgTruthRes && bgTruthRes.ok) {
    state.cachedTruthReport = await bgTruthRes.json();
    state.cachedTruthKey = truthKey;
  }

  if (res.ok) {
    state.weatherReport = await res.json();
    if (isPast && effectiveReplay === "truth") {
      state.cachedTruthReport = state.weatherReport;
      state.cachedTruthKey = truthKey;
    }
    _activeCorridorLutSector = null;
    _lastSyncedFluidMinutes = -999;
    volumetricEngine.setLiveWeatherReport(state.weatherReport);
    renderWeatherCard(state.weatherReport);
    renderWebcamChromaComparison();
    updateSunRaysAndShader();
  }
  fetchWebcamTruthForScenario(phase);
}

// ============================================================================
// Phase 2 (v4.2): Mountain Webcam Ground-Truth Verification & Sky Chromaticity
// ============================================================================
let _webcamCanvas = null;
let _webcamCtx = null;
let _activeWebcamReqId = 0;

function computeClientSkyChromaticity(imgEl) {
  if (!imgEl || !imgEl.naturalWidth || !imgEl.naturalHeight) return null;
  try {
    if (!_webcamCanvas) {
      _webcamCanvas = document.createElement("canvas");
      _webcamCanvas.width = 160;
      _webcamCanvas.height = 120;
      _webcamCtx = _webcamCanvas.getContext("2d", { willReadFrequently: true });
    }
    _webcamCtx.drawImage(imgEl, 0, 0, 160, 120);
    const y0 = Math.round(120 * 0.04);
    const y1 = Math.round(120 * 0.20);
    const x0 = Math.round(160 * 0.04);
    const x1 = Math.round(160 * 0.96);
    const imgData = _webcamCtx.getImageData(x0, y0, x1 - x0, y1 - y0).data;

    const rbs = [];
    const rgs = [];
    const sats = [];
    const warmSats = [];
    const lums = [];
    let blueCnt = 0;
    let warmCnt = 0;
    let greyCnt = 0;
    let validCnt = 0;
    let rSum = 0;
    let gSum = 0;
    let bSum = 0;

    for (let i = 0; i < imgData.length; i += 4) {
      const r = imgData[i] / 255.0;
      const g = imgData[i + 1] / 255.0;
      const b = imgData[i + 2] / 255.0;
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      if (lum <= 0.085 || lum >= 0.975) continue;
      const cMax = Math.max(r, g, b);
      const cMin = Math.min(r, g, b);
      const sat = (cMax - cMin) / Math.max(cMax, 1e-4);
      const rb = (r + 0.018) / (b + 0.018);
      const rg = (r + 0.018) / (g + 0.018);
      const wFactor = Math.max(0.0, Math.min(1.0, (rb - 0.96) / 0.30));
      rbs.push(rb);
      rgs.push(rg);
      sats.push(sat);
      warmSats.push(sat * wFactor);
      lums.push(lum);
      if (rb < 0.84 && sat > 0.17) blueCnt++;
      if (rb > 1.14 && lum > 0.16) warmCnt++;
      if (rb >= 0.86 && rb <= 1.13 && sat < 0.165) greyCnt++;
      validCnt++;
      rSum += r;
      gSum += g;
      bSum += b;
    }
    if (validCnt < 30) return null;
    rbs.sort((a, b) => a - b);
    rgs.sort((a, b) => a - b);
    sats.sort((a, b) => a - b);
    warmSats.sort((a, b) => a - b);
    const rbP90 = rbs[Math.min(rbs.length - 1, Math.floor(rbs.length * 0.9))];
    const rgP90 = rgs[Math.min(rgs.length - 1, Math.floor(rgs.length * 0.9))];
    const satP85 = sats[Math.min(sats.length - 1, Math.floor(sats.length * 0.85))];
    const warmSatP85 = warmSats[Math.min(warmSats.length - 1, Math.floor(warmSats.length * 0.85))];
    const lumMean = lums.reduce((s, v) => s + v, 0) / validCnt;
    const lumStd = Math.sqrt(lums.reduce((s, v) => s + (v - lumMean) ** 2, 0) / validCnt);
    const blueFrac = blueCnt / validCnt;
    const warmFrac = warmCnt / validCnt;
    const greyFrac = greyCnt / validCnt;

    const warmIdx = Math.max(0.0, Math.min(1.35, (rbP90 - 0.92) * 0.85 + Math.max(0.0, rgP90 - 1.02) * 0.65));
    const estKelvin =
      greyFrac > 0.55 && rbP90 < 1.18
        ? Math.round(5550 + Math.min(750, greyFrac * 700))
        : Math.round(Math.max(2050, Math.min(6300, 5200 - warmIdx * 2450)));

    const warmPts = Math.min(56, Math.max(0, (rbP90 - 0.98) * 52 + Math.max(0, rgP90 - 1.02) * 34 + warmFrac * 22));
    const satPts = Math.min(22, warmSatP85 * 54);
    const clarPts = Math.min(14, blueFrac * 12 + Math.min(6, lumStd * 36));
    const empScore = Math.round(Math.max(12, Math.min(97, 16 + warmPts + satPts + clarPts - greyFrac * 18)));

    return {
      available: true,
      rb_warmth_p90: Number(rbP90.toFixed(2)),
      saturation_p85: Number(satP85.toFixed(2)),
      estimated_color_temp_kelvin: estKelvin,
      empirical_alpenglow_score: empScore,
      horizon_swatch_rgb: [
        Math.round((rSum / validCnt) * 255),
        Math.round((gSum / validCnt) * 255),
        Math.round((bSum / validCnt) * 255),
      ],
    };
  } catch (_) {
    return null;
  }
}

function renderWebcamChromaComparison(overrideChroma = null) {
  const cam =
    state.webcamReport?.cameras?.[state.selectedWebcamIdx] ||
    state.webcamReport?.primary_camera;
  const ch = overrideChroma || cam?.sky_chromaticity || {};
  const w = state.weatherReport;

  const rbEl = document.getElementById("val-wc-rb");
  const kelvEl = document.getElementById("val-wc-kelvin");
  const empEl = document.getElementById("val-wc-emp-score");
  const simEl = document.getElementById("val-wc-sim-score");
  const sbSwatch = document.getElementById("webcam-sidebar-swatch");
  const sbTitle = document.getElementById("webcam-sidebar-title");
  const sbSub = document.getElementById("webcam-sidebar-sub");

  if (!cam) {
    if (sbTitle) sbTitle.textContent = "No Nearby Mountain Webcam Found";
    if (sbSub) sbSub.textContent = "Try selecting a California Coast or Sierra landmark";
    return;
  }

  const swatch = ch.horizon_swatch_rgb;
  if (sbSwatch && Array.isArray(swatch) && swatch.length === 3) {
    sbSwatch.style.background = `rgb(${swatch[0]}, ${swatch[1]}, ${swatch[2]})`;
  }

  const empScore = ch.available && ch.empirical_alpenglow_score > 0 ? ch.empirical_alpenglow_score : null;
  const estK = ch.available && ch.estimated_color_temp_kelvin ? ch.estimated_color_temp_kelvin : null;
  const simScore = w?.alpenglow_score ?? null;
  const simK = w?.color_temperature_kelvin ?? null;

  if (rbEl) {
    rbEl.textContent = ch.available
      ? `${Number(ch.rb_warmth_p90 || 1.0).toFixed(2)}× · Sat ${Math.round((ch.saturation_p85 || 0) * 100)}%`
      : "Loading RGB…";
  }
  if (kelvEl) {
    kelvEl.textContent = `${estK ? `${estK}K` : "--"} / ${simK ? `${simK}K` : "--"}`;
  }
  if (empEl) {
    empEl.textContent = empScore != null ? `${empScore} / 100` : "NIR / Night";
  }
  if (simEl) {
    if (simScore != null && empScore != null) {
      const diff = simScore - empScore;
      const sign = diff > 0 ? `+${diff}` : `${diff}`;
      simEl.textContent = `${simScore} / 100 (${sign})`;
    } else {
      simEl.textContent = simScore != null ? `${simScore} / 100` : "-- / 100";
    }
  }

  if (sbTitle) {
    const tag = cam.source === "alertca" ? "Live HD" : "Archive";
    sbTitle.textContent = `${cam.short_name || cam.camera_id} (${tag} • ${cam.distance_miles} mi)`;
  }
  if (sbSub) {
    if (empScore != null && simScore != null) {
      sbSub.textContent = `Webcam Glow ${empScore}/100 (${estK}K) vs 3D Engine ${simScore}/100`;
    } else {
      sbSub.textContent = `Az ${cam.azimuth_deg}° ${cam.azimuth_cardinal} • ${cam.elev_ft.toLocaleString()} ft ASL`;
    }
  }
}

function renderWebcamTruthUI() {
  const rep = state.webcamReport;
  const cams = rep?.cameras || [];
  const cam = cams[state.selectedWebcamIdx] || rep?.primary_camera;

  const eyebrowEl = document.getElementById("webcam-card-eyebrow");
  const titleEl = document.getElementById("webcam-card-title");
  const tabsEl = document.getElementById("webcam-camera-tabs");
  const imgEl = document.getElementById("webcam-frame-img");
  const tsPill = document.getElementById("webcam-ts-pill");
  const azPill = document.getElementById("webcam-az-pill");
  const chipsEl = document.getElementById("webcam-timeline-chips");
  const snapNoteEl = document.getElementById("webcam-snap-note");

  if (!cam) {
    renderWebcamChromaComparison();
    return;
  }

  if (eyebrowEl) {
    eyebrowEl.textContent =
      cam.source === "alertca"
        ? "ALERTCALIFORNIA LIVE HD • MOUNTAIN GROUND TRUTH"
        : "PHENOCAM ARCHIVE • GOLDEN-HOUR SKY CHROMATICITY";
  }
  if (titleEl) {
    titleEl.textContent = `${cam.short_name || cam.camera_id} • ${cam.elev_ft.toLocaleString()} ft • ${cam.distance_miles} mi`;
  }

  if (tabsEl) {
    if (cams.length > 1) {
      tabsEl.hidden = false;
      tabsEl.innerHTML = cams
        .map((c, idx) => {
          const badge = c.source === "alertca" ? "📡 Live" : "📚 Science";
          return `<button type="button" class="corridor-tab ${idx === state.selectedWebcamIdx ? "is-active" : ""}" data-wc-idx="${idx}">${badge} ${c.short_name || c.camera_id}</button>`;
        })
        .join("");
    } else {
      tabsEl.hidden = true;
      tabsEl.innerHTML = "";
    }
  }

  const frames = cam.timeline_frames || [];
  let fIdx = state.selectedWebcamFrameIdx;
  if (fIdx < 0 || fIdx >= frames.length) {
    fIdx = Math.max(
      0,
      frames.findIndex((f) => f.is_primary)
    );
    state.selectedWebcamFrameIdx = fIdx;
  }
  const activeFrame = frames[fIdx] || cam.primary_frame;

  if (imgEl && activeFrame?.proxy_url) {
    if (imgEl.dataset.activeSrc !== activeFrame.proxy_url) {
      imgEl.dataset.activeSrc = activeFrame.proxy_url;
      imgEl.onload = () => {
        const liveCh = computeClientSkyChromaticity(imgEl);
        renderWebcamChromaComparison(liveCh || cam.sky_chromaticity);
      };
      imgEl.src = activeFrame.proxy_url;
    }
  }

  if (tsPill && activeFrame) {
    const offMin = activeFrame.offset_vs_sunset_min;
    const offStr =
      cam.source === "alertca"
        ? "Live Frame"
        : offMin === 0
          ? "At Sunset"
          : `${offMin > 0 ? "+" : ""}${offMin}m vs Sunset`;
    tsPill.textContent = `${activeFrame.timestamp_local} • ${offStr}`;
  }
  if (azPill) {
    azPill.textContent = `Az ${cam.azimuth_deg}° ${cam.azimuth_cardinal} • FOV ${Math.round(cam.hfov_deg || 62)}°`;
  }

  if (chipsEl) {
    if (frames.length > 1) {
      chipsEl.innerHTML = frames
        .map(
          (f, idx) =>
            `<button type="button" class="webcam-time-chip ${idx === fIdx ? "is-active" : ""}" data-wc-frame="${idx}">${f.timestamp_local} (${f.chip_label || ""})</button>`
        )
        .join("");
    } else {
      chipsEl.innerHTML = "";
    }
  }

  if (snapNoteEl) {
    snapNoteEl.textContent = cam.is_inside_terrain_patch
      ? `Exact GPS vantage inside 1m DEM (${cam.elev_ft.toLocaleString()} ft ASL) — snaps 3D camera to exact webcam coordinates, Az ${cam.azimuth_deg}°, and ${Math.round(cam.hfov_deg || 62)}° HFOV.`
      : `Regional ${cam.provider_label} (${cam.distance_miles} mi away) — snaps 3D camera at active summit/viewpoint to matched Az ${cam.azimuth_deg}° (${cam.azimuth_cardinal}) & ${Math.round(cam.hfov_deg || 62)}° HFOV.`;
  }

  renderWebcamChromaComparison(cam.sky_chromaticity);
}

function toggleWebcamDrawer(forceShow = null) {
  const next = forceShow !== null ? Boolean(forceShow) : !state.showWebcamCard;
  state.showWebcamCard = next;
  if (webcamTruthCard) webcamTruthCard.hidden = !next;
  if (btnToggleWebcamDrawer) {
    btnToggleWebcamDrawer.classList.toggle("is-active", next);
    btnToggleWebcamDrawer.setAttribute("aria-expanded", String(next));
  }
  if (next) {
    inspectorCard.hidden = true;
    renderWebcamTruthUI();
  }
}

async function fetchWebcamTruthForScenario(phase = "sunset") {
  const lm = state.shootReport?.target_landmark;
  if (!lm) return;
  const reqId = ++_activeWebcamReqId;
  const win = state.shootReport?.golden_hour_windows?.find((w) =>
    w.phase.toLowerCase().includes(phase.toLowerCase())
  );
  const sunAz = win ? win.sun_azimuth_deg : phase === "sunrise" ? 102.5 : 259.1;
  try {
    const res = await fetch(
      `/api/webcam-truth?lat=${lm.summit_lat}&lon=${lm.summit_lon}&date=${encodeURIComponent(state.date)}&phase=${encodeURIComponent(phase)}&elev_m=${lm.summit_elevation_m}&sun_az=${sunAz}`
    );
    if (!res.ok || reqId !== _activeWebcamReqId) return;
    state.webcamReport = await res.json();
    state.selectedWebcamIdx = 0;
    state.selectedWebcamFrameIdx = -1;
    renderWebcamTruthUI();
  } catch (err) {
    console.warn("Webcam truth fetch skipped:", err);
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

  state.replayMode = "truth";
  state.cachedTruthReport = null;
  state.cachedTruthKey = "";

  const lmSel = document.getElementById("landmark-select");
  const dtInp = document.getElementById("shoot-date");
  if (lmSel && !customCoord) lmSel.value = state.landmark;
  if (dtInp) dtInp.value = state.date;
  try {
    if (!customCoord) localStorage.setItem("pp_landmark_v2", state.landmark);
    localStorage.setItem("pp_date_v3", state.date);
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
    const isPast = isPastShootDate(state.date);
    const initReplay = isPast ? "truth" : "auto";
    const truthKey = buildTruthCacheKey(lm.summit_lat, lm.summit_lon, state.date, "sunset");

    const [terrainPatchData, weatherRes] = await Promise.all([
      fetchBinaryOrJsonTerrainPatch(lm.summit_lat, lm.summit_lon, 9200, 600),
      fetch(`/api/weather-forecast?lat=${lm.summit_lat}&lon=${lm.summit_lon}&date=${encodeURIComponent(state.date)}&phase=sunset&sun_az=${initAz}&sun_el=${initEl}&elev_m=${lm.summit_elevation_m}&engine=${encodeURIComponent(state.weatherEngine)}&replay_mode=${initReplay}`),
    ]);

    state.terrainPatch = terrainPatchData;

    if (weatherRes.ok) {
      state.weatherReport = await weatherRes.json();
      if (isPast) {
        state.cachedTruthReport = state.weatherReport;
        state.cachedTruthKey = truthKey;
      }
      _activeCorridorLutSector = null;
      _lastSyncedFluidMinutes = -999;
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
    fetchWebcamTruthForScenario("sunset");
  } catch (err) {
    console.error("Failed to load scenario:", err);
    document.getElementById("summary-subtitle").textContent = `Error loading scenario: ${err.message}`;
  } finally {
    loadingEl.hidden = true;
  }
}

// Wire Mountain Webcam Split-View Drawer, 3D Azimuth/FOV Snap & Ghost Overlay Controls
btnToggleWebcamDrawer?.addEventListener("click", () => toggleWebcamDrawer());
btnCloseWebcamCard?.addEventListener("click", () => toggleWebcamDrawer(false));
btnSidebarOpenWebcam?.addEventListener("click", () => toggleWebcamDrawer(true));
btnSnapWebcamCamera?.addEventListener("click", () => {
  toggleWebcamDrawer(true);
  setCameraMode("webcam-snap");
});
btnResetWebcamSnap?.addEventListener("click", () => setCameraMode("orbit"));

toggleWebcamGhostEl?.addEventListener("change", (e) => {
  state.webcamGhostEnabled = Boolean(e.target.checked);
  if (state.webcamGhostEnabled && state.cameraMode !== "webcam-snap") {
    setCameraMode("webcam-snap");
  } else {
    syncWebcamGhostOverlay();
  }
});

webcamGhostOpacityEl?.addEventListener("input", (e) => {
  const pct = Math.max(10, Math.min(85, parseInt(e.target.value, 10) || 38));
  state.webcamGhostOpacity = pct / 100.0;
  if (valWebcamGhostOpacityEl) {
    valWebcamGhostOpacityEl.textContent = `${pct}%`;
  }
  if (!state.webcamGhostEnabled && toggleWebcamGhostEl) {
    state.webcamGhostEnabled = true;
    toggleWebcamGhostEl.checked = true;
    if (state.cameraMode !== "webcam-snap") {
      setCameraMode("webcam-snap");
      return;
    }
  }
  syncWebcamGhostOverlay();
});

document.getElementById("webcam-camera-tabs")?.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-wc-idx]");
  if (!btn) return;
  state.selectedWebcamIdx = parseInt(btn.dataset.wcIdx, 10) || 0;
  state.selectedWebcamFrameIdx = -1;
  renderWebcamTruthUI();
  if (state.cameraMode === "webcam-snap") {
    setCameraMode("webcam-snap");
  }
});

document.getElementById("webcam-timeline-chips")?.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-wc-frame]");
  if (!btn) return;
  const idx = parseInt(btn.dataset.wcFrame, 10);
  if (Number.isNaN(idx)) return;
  state.selectedWebcamFrameIdx = idx;
  renderWebcamTruthUI();
  const cam =
    state.webcamReport?.cameras?.[state.selectedWebcamIdx] ||
    state.webcamReport?.primary_camera;
  const frame = cam?.timeline_frames?.[idx];
  if (frame?.minutes_local != null && isPastShootDate(state.date)) {
    state.currentMinutes = Math.max(330, Math.min(1410, Number(frame.minutes_local)));
    timeScrubber.value = String(Math.round(state.currentMinutes));
    syncSliderValue(timeScrubber);
    updateSunRaysAndShader();
    wakeRenderLoop(true);
  }
});

// ============================================================================
// Interactive 3D Terrain & Volumetric Cloud Raycaster (Hover Provenance + Click)
// ============================================================================
const raycaster = new THREE.Raycaster();
const pointerNDC = new THREE.Vector2();
let pointerDownPos = { x: 0, y: 0 };
let pointerHoverRafPending = false;
let latestPointerClientX = 0;
let latestPointerClientY = 0;
let latestPointerButtons = 0;
let lastCloudProbeReqMs = 0;
let _lastTooltipHtmlKey = "";

const cloudHoverTooltipEl = document.getElementById("cloud-hover-tooltip");

const FALLBACK_LAYER_META = {
  high: {
    id: "high",
    short_name: "High Ice Cirrus",
    confidence_pct: 74,
    sources: ["Upper-Troposphere Sounding (300–200 hPa)", "GOES-18 ABI L2 Cirrus"],
    hover_disclosure: "Upper-tropospheric ice band altitude & wind shear from NWP/satellite; 3D mares' tails simulated.",
  },
  mid: {
    id: "mid",
    short_name: "Mid Altocumulus / Wave",
    confidence_pct: 68,
    sources: ["Mid-Level Sounding (700–500 hPa)", "64×64 (1.75 km) Spatial Grid"],
    hover_disclosure: "700–500 hPa moisture & Froude wave envelope from NWP sounding; 3D wave structure simulated.",
  },
  low: {
    id: "low",
    short_name: "Low Cumulus / Sc",
    confidence_pct: 66,
    sources: ["Boundary-Layer LCL Sounding", "METAR Ceilometer"],
    hover_disclosure: "LCL condensation base & convective depth from NWP/METAR; individual 1m cumulus billows simulated.",
  },
  fog: {
    id: "fog",
    short_name: "Marine Stratus / Fog",
    confidence_pct: 76,
    sources: ["GOES-18 Marine Stratus L2", "METAR Ceilometer + 925 hPa Inversion"],
    hover_disclosure: "Sub-inversion marine stratus top & inland penetration from GOES/METAR/HRRR; 1m cloud-top eddies simulated.",
  },
};

function hideCloudHoverTooltip(cancelPending = true) {
  if (cloudHoverTooltipEl && !cloudHoverTooltipEl.hidden) {
    cloudHoverTooltipEl.hidden = true;
  }
  _lastTooltipHtmlKey = "";
  setHoveredCloudLayerHighlight(null);
  if (cancelPending) {
    volumetricEngine.cancelCloudProbe();
  }
}

function positionCloudHoverTooltip(clientX, clientY) {
  if (!cloudHoverTooltipEl || !viewportStageEl) return;
  const stageRect = viewportStageEl.getBoundingClientRect();
  const cardW = cloudHoverTooltipEl.offsetWidth || 292;
  const cardH = cloudHoverTooltipEl.offsetHeight || 128;
  const pad = 12;
  let x = clientX - stageRect.left + 16;
  let y = clientY - stageRect.top + 16;
  if (x + cardW + pad > stageRect.width) {
    x = Math.max(pad, clientX - stageRect.left - cardW - 14);
  }
  if (y + cardH + pad > stageRect.height) {
    y = Math.max(pad, clientY - stageRect.top - cardH - 14);
  }
  x = Math.max(pad, x);
  y = Math.max(pad, y);
  cloudHoverTooltipEl.style.transform = `translate3d(${Math.round(x)}px, ${Math.round(y)}px, 0)`;
}

volumetricEngine.onCloudProbeCallback = (probe, meta) => {
  if (
    !cloudHoverTooltipEl ||
    !state.showAtmosphere ||
    !meta ||
    latestPointerButtons !== 0 ||
    !probe ||
    probe.cloudAlpha < 0.038
  ) {
    hideCloudHoverTooltip(false);
    return;
  }

  const domId = probe.dominantLayer || "low";
  const layersArr = state.activeAtmosphericLayers?.layers;
  const layerMeta =
    (Array.isArray(layersArr) && layersArr.find((l) => l.id === domId)) ||
    FALLBACK_LAYER_META[domId] ||
    FALLBACK_LAYER_META.low;

  const baseConf = typeof layerMeta.confidence_pct === "number" ? layerMeta.confidence_pct : 68;
  const distM = Math.max(0, probe.meanDistM || 0);
  const has64Grid = (state.weatherReport?.mesoscale_cloud_grid?.grid_size || 0) >= 64;
  const fineRadiusM = has64Grid ? 50000 : 25000;
  const distPenalty = distM > fineRadiusM ? Math.min(8, Math.round((distM - fineRadiusM) / 12000)) : 0;
  // Thin parcel wisps are governed more by simulated sub-grid turbulence than dense cloud cores
  const densityAdj = probe.cloudAlpha < 0.14 ? -4 : probe.cloudAlpha > 0.55 ? 2 : 0;
  const confPct = Math.max(18, Math.min(95, Math.round(baseConf - distPenalty + densityAdj)));
  const confTier = confPct >= 74 ? "high" : confPct >= 52 ? "moderate" : "low";

  const altM = Math.max(0, probe.meanAltAslM || 0);
  const altFtRounded = Math.round((altM * 3.28084) / 50) * 50;
  const altKmStr = (altM / 1000).toFixed(1);
  const distKm = distM / 1000;
  const distMi = distKm * 0.621371;
  const distStr = distMi < 10 ? `${distMi.toFixed(1)} mi` : `${Math.round(distMi)} mi`;
  const opacityPct = Math.max(4, Math.min(100, Math.round(probe.cloudAlpha * 100)));

  // Check if ray intersects a secondary overlapping cloud deck (>=22% optical depth share)
  let secondaryOverlapLabel = "";
  if (probe.layerShares) {
    let bestSecId = null;
    let bestSecShare = 0.22;
    for (const [lid, share] of Object.entries(probe.layerShares)) {
      if (lid !== domId && share >= bestSecShare) {
        bestSecShare = share;
        bestSecId = lid;
      }
    }
    if (bestSecId) {
      const secMeta =
        (Array.isArray(layersArr) && layersArr.find((l) => l.id === bestSecId)) ||
        FALLBACK_LAYER_META[bestSecId];
      if (secMeta) {
        secondaryOverlapLabel = `+ ${secMeta.short_name}`;
      }
    }
  }

  const sources = Array.isArray(layerMeta.sources) ? layerMeta.sources.slice(0, 3) : [];
  const disclosure = layerMeta.hover_disclosure || "";
  const htmlKey = `${domId}|${confPct}|${altFtRounded}|${distStr}|${opacityPct}|${secondaryOverlapLabel}|${sources.join(";")}|${disclosure}`;

  if (htmlKey !== _lastTooltipHtmlKey) {
    _lastTooltipHtmlKey = htmlKey;
    const chipsHtml = sources
      .map((s) => `<span class="cht-source-chip">${escapeHtmlText(s)}</span>`)
      .join("");

    cloudHoverTooltipEl.innerHTML = `
      <div class="cht-top-row">
        <span class="cht-layer-title">${escapeHtmlText(layerMeta.short_name || layerMeta.name)}</span>
        <span class="conf-pill conf-${confTier}">${confPct}% Confidence</span>
      </div>
      <div class="cht-meta-row">
        <span>${altFtRounded.toLocaleString()} ft ASL (${altKmStr} km)</span>
        <span>•</span>
        <span>${distStr} away</span>
        <span>•</span>
        <span>${opacityPct}% opacity</span>
        ${secondaryOverlapLabel ? `<span>• ${escapeHtmlText(secondaryOverlapLabel)}</span>` : ""}
      </div>
      ${
        chipsHtml
          ? `<div class="cht-divider"></div><div class="cht-sources-row">${chipsHtml}</div>`
          : ""
      }
      ${disclosure ? `<p class="cht-disclosure">${escapeHtmlText(disclosure)}</p>` : ""}
    `;
  }

  cloudHoverTooltipEl.hidden = false;
  positionCloudHoverTooltip(meta.clientX, meta.clientY);
  setHoveredCloudLayerHighlight(domId);
};

function updatePointerNDCFromCoords(clientX, clientY) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointerNDC.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointerNDC.y = -((clientY - rect.top) / rect.height) * 2 + 1;
}

function updatePointerNDC(event) {
  updatePointerNDCFromCoords(event.clientX, event.clientY);
}

renderer.domElement.addEventListener(
  "pointerleave",
  () => {
    hideCloudHoverTooltip(true);
  },
  { passive: true }
);

renderer.domElement.addEventListener(
  "pointermove",
  (e) => {
    if (!terrainMeshRef || !state.terrainPatch) return;
    latestPointerClientX = e.clientX;
    latestPointerClientY = e.clientY;
    latestPointerButtons = e.buttons;
    if (e.buttons !== 0) {
      hideCloudHoverTooltip(true);
    } else if (cloudHoverTooltipEl && !cloudHoverTooltipEl.hidden) {
      positionCloudHoverTooltip(e.clientX, e.clientY);
    }
    if (pointerHoverRafPending) return;
    pointerHoverRafPending = true;
    requestAnimationFrame(() => {
      pointerHoverRafPending = false;
      if (!terrainMeshRef || !state.terrainPatch) return;
      updatePointerNDCFromCoords(latestPointerClientX, latestPointerClientY);
      raycaster.setFromCamera(pointerNDC, camera);

      // Check if hovering a 3D Viewpoint marker first
      const vpHits = raycaster.intersectObjects(vpMarkerMeshes, false);
      if (vpHits.length > 0 && vpHits[0].object.userData?.vpIndex !== undefined) {
        const vpIdx = vpHits[0].object.userData.vpIndex;
        const vp = state.shootReport?.top_viewpoints?.[vpIdx];
        if (vp) {
          hideCloudHoverTooltip(true);
          cursorCoordsBar.textContent = `VP${vp.viewpoint_rank}: ${vp.viewpoint_label} (${vp.hike_distance_to_vp_miles} mi) • Click to highlight`;
          return;
        }
      }

      const terrHit = intersectTerrainHeightfield(raycaster.ray);
      if (terrHit) {
        const { lat, lon, elevM } = worldToLatLonElev(terrHit.point);
        const elevFt = Math.round(elevM * 3.28084);
        const actionLabel =
          state.clickMode === "tripod" ? "Click to inspect sightline" : "Click to set Target";
        cursorCoordsBar.textContent = `${lat.toFixed(4)}, ${lon.toFixed(4)} • ${elevFt.toLocaleString()} ft • ${actionLabel}`;
      } else {
        const d = raycaster.ray.direction;
        // Local scene coordinates: +X = East, -Z = North, +Y = Up
        const azDeg = ((Math.atan2(d.x, -d.z) * 180) / Math.PI + 360) % 360;
        const elDeg = (Math.asin(Math.max(-1, Math.min(1, d.y))) * 180) / Math.PI;
        cursorCoordsBar.textContent = `Sky Azimuth ${azDeg.toFixed(1)}° (${compassLabel(azDeg)}) • Elev ${elDeg >= 0 ? "+" : ""}${elDeg.toFixed(1)}° • Hover cloud for confidence`;
      }

      if (state.showAtmosphere && latestPointerButtons === 0) {
        const nowPerf = performance.now();
        if (nowPerf - lastCloudProbeReqMs >= 38) {
          lastCloudProbeReqMs = nowPerf;
          const maxProbeDist = terrHit ? Math.max(15.0, terrHit.distance) : 116000.0;
          volumetricEngine.requestCloudProbe(
            raycaster.ray.origin,
            raycaster.ray.direction,
            maxProbeDist,
            { clientX: latestPointerClientX, clientY: latestPointerClientY }
          );
          wakeRenderLoop(false);
        }
      } else {
        hideCloudHoverTooltip(true);
      }
    });
  },
  { passive: true }
);

let tripodRequestSeq = 0;

renderer.domElement.addEventListener("selectstart", (e) => e.preventDefault());

renderer.domElement.addEventListener("pointerdown", (e) => {
  window.getSelection?.()?.removeAllRanges?.();
  pointerDownPos = { x: e.clientX, y: e.clientY };
  latestPointerButtons = e.buttons;
  hideCloudHoverTooltip(true);
});

renderer.domElement.addEventListener("pointerup", async (e) => {
  if (e.button !== 0) return;
  const moveDist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
  if (moveDist > 14 || !terrainMeshRef || !state.terrainPatch || !state.shootReport) return;

  updatePointerNDC(e);
  raycaster.setFromCamera(pointerNDC, camera);

  const terrHit = intersectTerrainHeightfield(raycaster.ray);
  const hits = terrHit ? [terrHit] : [];

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

let scrubberRafPending = false;
let scrubberSettleTimer = 0;

function scheduleScrubberSettleBake() {
  if (scrubberSettleTimer) clearTimeout(scrubberSettleTimer);
  scrubberSettleTimer = setTimeout(() => {
    scrubberSettleTimer = 0;
    isScrubbingTime = false;
    if (terrainShaderMaterial) {
      volumetricEngine.updateTerrainBakeUniforms({
        sunDirENU: terrainShaderMaterial.uniforms.uSunDirENU.value,
        showViewshed: state.showViewshed,
        isScrubbing: false,
      });
    }
    wakeRenderLoop(false);
  }, 95);
}

timeScrubber.addEventListener("input", () => {
  isScrubbingTime = true;
  wakeRenderLoop(true);
  state.currentMinutes = parseFloat(timeScrubber.value);
  syncSliderValue(timeScrubber);
  scheduleScrubberSettleBake();
  if (scrubberRafPending) return;
  scrubberRafPending = true;
  requestAnimationFrame(() => {
    scrubberRafPending = false;
    updateSunRaysAndShader();
  });
});

timeScrubber.addEventListener("change", () => {
  scheduleScrubberSettleBake();
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
  if (!state.showAtmosphere) {
    hideCloudHoverTooltip(true);
  }
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

function selectHistoricalReplayMode(mode) {
  if (!mode || !isPastShootDate(state.date)) return;
  state.replayMode = mode;
  document.querySelectorAll("#historical-replay-btns [data-replay-mode]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.replayMode === mode);
  });
  hideCloudHoverTooltip(true);
  const isSunriseActive = document.getElementById("tab-corridor-sunrise")?.classList.contains("is-active");
  fetchWeatherForCorridor(isSunriseActive ? "sunrise" : "sunset");
}

document.getElementById("historical-replay-btns")?.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-replay-mode]");
  if (btn) {
    selectHistoricalReplayMode(btn.dataset.replayMode);
  }
});

function applyQuickScenario(landmark, date) {
  document.getElementById("scenarios-popover")?.hidePopover?.();
  hideCloudHoverTooltip(true);
  state.landmark = landmark;
  state.date = date;
  const lmSel = document.getElementById("landmark-select");
  const dtInp = document.getElementById("shoot-date");
  if (lmSel) lmSel.value = landmark;
  if (dtInp) dtInp.value = date;
  volumetricEngine.setPreset("live");
  loadLandmarkScenario();
}

document.getElementById("btn-yesterday-pedro")?.addEventListener("click", () => {
  applyQuickScenario("Pedro Point Headlands", yesterdayIso);
});

document.getElementById("btn-vp-yesterday-pedro")?.addEventListener("click", () => {
  applyQuickScenario("Pedro Point Headlands", yesterdayIso);
});

document.getElementById("btn-live-today-sf")?.addEventListener("click", () => {
  applyQuickScenario("Montara Mountain", todayIso);
});

document.getElementById("btn-vp-today-sf")?.addEventListener("click", () => {
  applyQuickScenario("Montara Mountain", todayIso);
});

document.getElementById("btn-hist-fog-sea")?.addEventListener("click", () => {
  applyQuickScenario("Montara Mountain", "2025-06-05");
});

document.getElementById("btn-vp-fog-sea")?.addEventListener("click", () => {
  applyQuickScenario("Montara Mountain", "2025-06-05");
});

document.getElementById("btn-hist-clear-slot")?.addEventListener("click", () => {
  applyQuickScenario("Pedro Point Headlands", "2025-05-16");
});

document.getElementById("btn-vp-clear-slot")?.addEventListener("click", () => {
  applyQuickScenario("Pedro Point Headlands", "2025-05-16");
});

document.getElementById("btn-vp-halfdome-alpenglow")?.addEventListener("click", () => {
  applyQuickScenario("Half Dome", "2025-05-25");
});

document.getElementById("btn-vp-clouds-rest-wave")?.addEventListener("click", () => {
  applyQuickScenario("Clouds Rest", "2025-06-07");
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
// Unified 4-Layer Physical Atmosphere Controls & Fine-Tuning Drawer (WebGL2)
// ============================================================================
const fluidDisclosure = document.getElementById("fluid-physics-disclosure");
const btnToggleFluidEvolve = document.getElementById("btn-toggle-fluid-evolve");
const btnResetFluidTuning = document.getElementById("btn-reset-fluid-tuning");
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

btnResetFluidTuning?.addEventListener("click", () => {
  if (sliderFluidSpeed) {
    sliderFluidSpeed.value = "1.0";
    syncSliderValue(sliderFluidSpeed);
    const out = document.getElementById("out-fluid-speed");
    if (out) out.textContent = "1.0×";
  }
  if (sliderMicroDetail) {
    sliderMicroDetail.value = "100";
    syncSliderValue(sliderMicroDetail);
    const out = document.getElementById("out-micro-detail");
    if (out) out.textContent = "100%";
  }
  if (sliderCloudDensity) {
    sliderCloudDensity.value = "100";
    syncSliderValue(sliderCloudDensity);
    const out = document.getElementById("out-cloud-density");
    if (out) out.textContent = "100%";
  }
  if (sliderCirrusDensity) {
    sliderCirrusDensity.value = "105";
    syncSliderValue(sliderCirrusDensity);
    const out = document.getElementById("out-cirrus-density");
    if (out) out.textContent = "105%";
  }
  volumetricEngine.evolutionSpeed = 1.0;
  volumetricEngine.setUserTuning({
    cloudDensityGain: 1.0,
    cirrusDensityGain: 1.05,
    microDetailGain: 1.0,
    godRayGain: 1.0,
  });
  wakeRenderLoop(true);
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

  const stepStr = (tel.stepMs || 0.28).toFixed(2);
  const solverTag = tel.gpuFluidActive ? "GPU 128×128×48 Fluid" : "WASM Fluid (Worker)";
  if (!isRenderLoopSleeping) {
    if (fluidPerfBadge) {
      fluidPerfBadge.textContent = `${Math.round(rollingFps)} FPS`;
      fluidPerfBadge.title = `${solverTag} Step: ${stepStr} ms`;
    }
    if (viewportGpuPill) {
      viewportGpuPill.textContent = `⚡ ${Math.round(rollingFps)} FPS · ${stepStr}ms`;
    }
  }
  if (fluidSummarySub) {
    const overallConf = state.activeAtmosphericLayers?.overall_confidence_pct;
    const confTag = typeof overallConf === "number" ? `${Math.round(overallConf)}% Conf` : "Hover clouds";
    fluidSummarySub.textContent = `0–${topKm} km Column • Low ${lowPct}% / High ${highPct}% • ${confTag}`;
  }
};

window.addEventListener("resize", () => {
  const w = container.clientWidth;
  const h = Math.max(container.clientHeight, 1);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  volumetricEngine.resize();
  wakeRenderLoop(true);
});

// ============================================================================
// Adaptive 3-Tier GPU Render Loop:
//   Tier 1 (Active):       60 FPS while orbiting, scrubbing, or animating sun
//   Tier 2 (Gentle Drift): 20 FPS low-power cloud evolution when hands-off (<8s)
//   Tier 3 (Idle Sleep):    0 FPS (0W GPU) after 8s idle or when hidden; <16ms wake!
// ============================================================================
let lastRenderedFrameMs = performance.now();
let lastUserActivityMs = performance.now();
let camSettledAtMs = performance.now();
let fpsAccumFrames = 0;
let fpsAccumMs = 0;
let activeAnimRafId = 0;
let stageIntersecting = true;
let isRenderLoopSleeping = false;
const _lastCamPos = camera.position.clone();
const _lastCamQuat = camera.quaternion.clone();

const ACTIVE_BOOST_WINDOW_MS = 950;
const CAM_SETTLE_TAIL_MS = 340;
const GENTLE_DRIFT_WINDOW_MS = 8000;
const GENTLE_DRIFT_MIN_INTERVAL_MS = 47.5; // ~20 FPS

function shouldRunRenderLoop() {
  return !document.hidden && stageIntersecting;
}

function ensureRenderLoopRunning() {
  if (activeAnimRafId === 0 && shouldRunRenderLoop()) {
    isRenderLoopSleeping = false;
    lastRenderedFrameMs = performance.now() - 16.7;
    activeAnimRafId = requestAnimationFrame(animate);
  }
}

function wakeRenderLoop(markFullActive = true) {
  const now = performance.now();
  if (markFullActive) {
    lastUserActivityMs = now;
    camSettledAtMs = now;
  } else {
    // Programmatic/hover wake: ensure at least a crisp convergence burst without pinning 60 FPS indefinitely
    camSettledAtMs = Math.max(camSettledAtMs, now);
    if (now - lastUserActivityMs >= GENTLE_DRIFT_WINDOW_MS) {
      lastUserActivityMs = now - (ACTIVE_BOOST_WINDOW_MS + 50);
    }
  }
  ensureRenderLoopRunning();
}

// Instant (<16ms) wakeup on any pointer, wheel, touch, keyboard, or UI control interaction
controls.addEventListener("start", () => wakeRenderLoop(true));
controls.addEventListener("change", () => {
  camSettledAtMs = performance.now();
  ensureRenderLoopRunning();
});

["pointerdown", "wheel", "touchstart", "keydown", "input", "change", "click"].forEach((evt) => {
  window.addEventListener(evt, () => wakeRenderLoop(true), { passive: true });
});

container?.addEventListener(
  "pointermove",
  (e) => {
    if (e.buttons !== 0) {
      wakeRenderLoop(true);
    } else if (isRenderLoopSleeping) {
      wakeRenderLoop(false);
    }
  },
  { passive: true }
);

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    if (activeAnimRafId !== 0) {
      cancelAnimationFrame(activeAnimRafId);
      activeAnimRafId = 0;
    }
  } else {
    wakeRenderLoop(true);
  }
});

if (typeof IntersectionObserver !== "undefined" && container) {
  const stageObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        stageIntersecting = entry.isIntersecting;
      }
      if (!stageIntersecting && activeAnimRafId !== 0) {
        cancelAnimationFrame(activeAnimRafId);
        activeAnimRafId = 0;
      } else if (stageIntersecting) {
        wakeRenderLoop(false);
      }
    },
    { threshold: 0.01 }
  );
  stageObserver.observe(container);
}

function animate(nowMs = performance.now()) {
  if (!shouldRunRenderLoop()) {
    activeAnimRafId = 0;
    return;
  }

  const idleElapsedMs = nowMs - lastUserActivityMs;
  const camSettleElapsedMs = nowMs - camSettledAtMs;
  const isHighRateActive =
    state.isPlayingSun ||
    isScrubbingTime ||
    volumetricEngine.hasPendingCloudProbe() ||
    idleElapsedMs < ACTIVE_BOOST_WINDOW_MS ||
    camSettleElapsedMs < CAM_SETTLE_TAIL_MS;

  const canGentleDrift =
    volumetricEngine.evolveFluid &&
    state.showAtmosphere &&
    idleElapsedMs < GENTLE_DRIFT_WINDOW_MS;

  // Tier 3: Deep 0-FPS Idle Sleep after 8s hands-off (0 GPU watts, instant <16ms wake on input!)
  if (!isHighRateActive && !canGentleDrift) {
    activeAnimRafId = 0;
    isRenderLoopSleeping = true;
    rollingFps = 0;
    fpsAccumFrames = 0;
    fpsAccumMs = 0;
    if (fluidPerfBadge) {
      fluidPerfBadge.textContent = "Idle (0W)";
    }
    if (viewportGpuPill) {
      viewportGpuPill.textContent = "⚡ Idle · 0 GPU";
    }
    return;
  }

  activeAnimRafId = requestAnimationFrame(animate);

  // Tier 2: Gentle 20 FPS Cloud Drift when camera is stationary & hands-off (saves ~65% GPU power)
  const sinceLastRenderMs = nowMs - lastRenderedFrameMs;
  if (!isHighRateActive && sinceLastRenderMs < GENTLE_DRIFT_MIN_INTERVAL_MS) {
    return;
  }

  isRenderLoopSleeping = false;
  const dtMs = Math.min(100, Math.max(0.5, sinceLastRenderMs));
  lastRenderedFrameMs = nowMs;
  const dtRealS = dtMs * 0.001;

  fpsAccumFrames += 1;
  fpsAccumMs += dtMs;
  if (fpsAccumMs >= 450) {
    rollingFps = (fpsAccumFrames * 1000) / fpsAccumMs;
    fpsAccumFrames = 0;
    fpsAccumMs = 0;
    const stepStr = (volumetricEngine.telemetry.stepMs || 0.28).toFixed(2);
    const solverTag = volumetricEngine.telemetry.gpuFluidActive
      ? "GPU 128×128×48 Fluid"
      : "WASM Fluid (Worker)";
    const tierSuffix = isHighRateActive ? "" : " (Eco)";
    if (fluidPerfBadge) {
      fluidPerfBadge.textContent = `${Math.round(rollingFps)} FPS${tierSuffix}`;
      fluidPerfBadge.title = `${solverTag} Step: ${stepStr} ms`;
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
  if (
    camera.position.distanceToSquared(_lastCamPos) > 1e-4 ||
    1.0 - Math.abs(camera.quaternion.dot(_lastCamQuat)) > 1e-6
  ) {
    _lastCamPos.copy(camera.position);
    _lastCamQuat.copy(camera.quaternion);
    camSettledAtMs = nowMs;
  }

  volumetricEngine.renderPipeline(worldScene, overlayScene, dtRealS, state.showAtmosphere);
}

ensureRenderLoopRunning();
loadLandmarkScenario();
