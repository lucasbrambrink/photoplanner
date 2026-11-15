// Dedicated Background Web Worker for Rust WASM (`pp_wasm.wasm`)
// Runs 3D Moist Navier-Stokes + Orographic Gravity Waves + 3D Cirrus Ice Microphysics
// and sub-millisecond 1m Terrain Sightline / 360° Horizon Raycasting off the main UI thread.

let wasm = null;
let paramsPtr = 0;
const PARAMS_LEN = 23;

async function loadWasmModule(seed = 1337) {
  const t0 = performance.now();
  const url = new URL('./pkg/pp_wasm.wasm?v=20261115a', self.location.href);
  let instance;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} loading pp_wasm.wasm`);
    if (WebAssembly.instantiateStreaming) {
      try {
        const streamed = await WebAssembly.instantiateStreaming(res.clone(), {});
        instance = streamed.instance;
      } catch (_) {
        const bytes = await res.arrayBuffer();
        const instantiated = await WebAssembly.instantiate(bytes, {});
        instance = instantiated.instance;
      }
    } else {
      const bytes = await res.arrayBuffer();
      const instantiated = await WebAssembly.instantiate(bytes, {});
      instance = instantiated.instance;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err) });
    return;
  }

  wasm = instance.exports;
  wasm.wasm_init_atmosphere(seed >>> 0);
  paramsPtr = wasm.wasm_alloc_f32(PARAMS_LEN);

  const detailPtr = wasm.wasm_detail_noise_ptr();
  const detailLen = wasm.wasm_detail_noise_len();
  const detailCopy = new Uint8Array(wasm.memory.buffer, detailPtr, detailLen).slice();

  const snapshot = readAtmosphereSnapshot(performance.now() - t0);
  self.postMessage(
    {
      type: 'ready',
      detailNoiseBuffer: detailCopy.buffer,
      volumeBuffer: snapshot.volumeBuffer,
      telemetry: snapshot.telemetry,
      windProfile: snapshot.windProfile,
      stepMs: snapshot.stepMs,
    },
    [detailCopy.buffer, snapshot.volumeBuffer]
  );
}

function readAtmosphereSnapshot(stepMs) {
  const memBuf = wasm.memory.buffer;
  const volPtr = wasm.wasm_volume_ptr();
  const volLen = wasm.wasm_volume_len();
  const volCopy = new Uint8Array(memBuf, volPtr, volLen).slice();

  const telPtr = wasm.wasm_telemetry_ptr();
  const telLen = wasm.wasm_telemetry_len();
  const telArr = Array.from(new Float32Array(memBuf, telPtr, telLen));

  const windPtr = wasm.wasm_wind_profile_ptr();
  const windLen = wasm.wasm_wind_profile_len();
  const windArr = Array.from(new Float32Array(memBuf, windPtr, windLen));

  return {
    volumeBuffer: volCopy.buffer,
    telemetry: {
      simTimeS: telArr[0],
      lclAltitudeM: telArr[1],
      freezingLevelM: telArr[2],
      glaciationCirrusBaseM: telArr[3],
      maxUpdraftMs: telArr[4],
      maxDowndraftMs: telArr[5],
      maxLiquidGKg: telArr[6],
      maxIceGKg: telArr[7],
      lowCloudCoverage: telArr[8],
      midCloudCoverage: telArr[9],
      highCirrusCoverage: telArr[10],
      froudeNumber: telArr[11],
      jetShearMsKm: telArr[12],
      zBaseM: telArr[13],
      zTopM: telArr[14],
      domainXyM: telArr[15],
    },
    windProfile: windArr,
    stepMs: Math.round(stepMs * 100) / 100,
  };
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  if (!msg || !msg.type) return;

  if (msg.type === 'init') {
    await loadWasmModule(msg.seed || 1337);
    return;
  }

  if (!wasm) return;

  if (msg.type === 'set_terrain') {
    const t0 = performance.now();
    const elevations = new Float32Array(msg.elevationsBuffer);
    const len = elevations.length;
    const ptr = wasm.wasm_alloc_f32(len);
    new Float32Array(wasm.memory.buffer, ptr, len).set(elevations);
    wasm.wasm_set_terrain_dem(
      ptr,
      msg.width >>> 0,
      msg.height >>> 0,
      Number(msg.patchWidthM) || 9200,
      Number(msg.patchHeightM) || 9200,
      Number(msg.centerLat) || 37.59,
      Number(msg.centerLon) || -122.51
    );
    wasm.wasm_free_f32(ptr, len);
    const snap = readAtmosphereSnapshot(performance.now() - t0);
    self.postMessage({ type: 'frame', ...snap }, [snap.volumeBuffer]);
    return;
  }

  if (msg.type === 'set_params') {
    const t0 = performance.now();
    const p = msg.params || {};
    const arr = new Float32Array(wasm.memory.buffer, paramsPtr, PARAMS_LEN);
    arr[0] = p.surfaceTempC ?? 14.5;
    arr[1] = p.surfaceRhPct ?? 84.0;
    arr[2] = p.lowCloudPct ?? 58.0;
    arr[3] = p.midCloudPct ?? 24.0;
    arr[4] = p.highCloudPct ?? 42.0;
    arr[5] = p.inversionBaseM ?? 340.0;
    arr[6] = p.inversionThicknessM ?? 210.0;
    arr[7] = p.inversionStrengthK ?? 6.5;
    arr[8] = p.windSfcU ?? 5.2;
    arr[9] = p.windSfcV ?? -3.4;
    arr[10] = p.wind850U ?? 8.5;
    arr[11] = p.wind850V ?? -2.0;
    arr[12] = p.wind500U ?? 18.0;
    arr[13] = p.wind500V ?? 4.5;
    arr[14] = p.wind250U ?? 36.0;
    arr[15] = p.wind250V ?? 14.0;
    arr[16] = p.convectiveCape ?? 320.0;
    arr[17] = p.orographicWaveGain ?? 1.15;
    arr[18] = p.vorticityConfinement ?? 1.15;
    arr[19] = p.cirrusIceBoost ?? 1.0;
    arr[20] = p.solarElevationDeg ?? 11.0;
    arr[21] = p.solarAzimuthDeg ?? 256.0;
    arr[22] = p.seed ?? 1337;

    wasm.wasm_set_params(paramsPtr, PARAMS_LEN, msg.reseed ? 1 : 0);
    const snap = readAtmosphereSnapshot(performance.now() - t0);
    self.postMessage({ type: 'frame', ...snap }, [snap.volumeBuffer]);
    return;
  }

  if (msg.type === 'step') {
    const t0 = performance.now();
    wasm.wasm_step_atmosphere(Number(msg.dt) || 5.0);
    const snap = readAtmosphereSnapshot(performance.now() - t0);
    self.postMessage({ type: 'frame', ...snap }, [snap.volumeBuffer]);
    return;
  }

  if (msg.type === 'raycast_tripod') {
    const t0 = performance.now();
    const outPtr = wasm.wasm_raycast_tripod(
      Number(msg.tripodLat),
      Number(msg.tripodLon),
      Number(msg.targetLat),
      Number(msg.targetLon),
      Number(msg.targetZM) || 0.0
    );
    const outLen = wasm.wasm_raycast_tripod_len();
    const outCopy = new Float32Array(wasm.memory.buffer, outPtr, outLen).slice();
    const elapsedMs = Math.round((performance.now() - t0) * 1000) / 1000;
    self.postMessage(
      {
        type: 'raycast_result',
        reqId: msg.reqId,
        dataBuffer: outCopy.buffer,
        elapsedMs,
      },
      [outCopy.buffer]
    );
  }
};
