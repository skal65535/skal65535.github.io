// ba_gpu_bench.js — What ba_gpu_bench.e2e.js runs in the page: the GPU bundle adjustment (src/sfm/ba_gpu.js) and ba.js on
// synthetic windows shaped like the mapper's, each from the same start, timed.
import { bundleAdjust, linearizeObservation } from '../src/sfm/ba.js';
import { GpuBundleAdjuster } from '../src/sfm/ba_gpu.js';
import { expSO3, mul3, rotationAngle } from '../src/sfm/linalg.js';
import { poseFromCenter, poseCenter } from '../src/sfm/camera.js';
import { makeRng } from '../src/math.js';
import { gaussian } from '../src/sfm/synth.js';

// A window: nFree free poses after `fixed` fixed ones along a line, nPts points each seen by about `track` consecutive
// frames, observations 0.5 px off, the free poses and the points perturbed as a window's start is
function makeWindow({ nFree, nPts, track, fixed = 2, seed = 1 }) {
  const rng = makeRng(seed), n = nFree + fixed;
  const camera = { model: 'OPENCV', width: 1920, height: 1080, params: [1000, 1000, 960, 540, 0.02, 0, 0, 0] };
  const truth = Array.from({ length: n }, (_, k) => poseFromCenter(expSO3([0.02 * gaussian(rng), 0.02 * gaussian(rng), 0]), [0.5 * k, 0, 0]));
  const points = new Float64Array(3 * nPts), obsCam = [], obsPoint = [], obsUV = [];
  for (let p = 0; p < nPts; p++) {
    const l = Math.max(2, Math.min(n, Math.round(track + 2 * (rng() - 0.5)))), s = Math.floor(rng() * (n - l + 1));
    const X = [0.5 * (s + (l - 1) / 2) + 6 * (rng() - 0.5), 4 * (rng() - 0.5), 4 + 8 * rng()];
    for (let k = s; k < s + l; k++) {
      const { r } = linearizeObservation(camera, truth[k], X, [0, 0]);
      obsCam.push(k); obsPoint.push(p); obsUV.push(r[0] + 0.5 * gaussian(rng), r[1] + 0.5 * gaussian(rng));
    }
    for (let i = 0; i < 3; i++) points[3 * p + i] = X[i] + 0.02 * gaussian(rng);
  }
  const poses = truth.map((T, k) => k < fixed ? T : poseFromCenter(mul3(expSO3([0, 1, 2].map(() => 0.002 * gaussian(rng))), T.R),
                                                                     poseCenter(T).map(x => x + 0.02 * gaussian(rng))));
  return { camera, poses, free: Uint8Array.from(poses, (_, k) => +(k >= fixed)), points,
           obsCam: Int32Array.from(obsCam), obsPoint: Int32Array.from(obsPoint), obsUV: Float64Array.from(obsUV) };
}

const copy = (p) => ({ ...p, camera: { ...p.camera, params: [...p.camera.params] }, poses: p.poses.map(x => ({ R: [...x.R], t: [...x.t] })), points: p.points.slice() });
const median = (v) => [...v].sort((a, b) => a - b)[v.length >> 1];

// The floor of a round trip: a 64 KB copy submitted and mapped back, median of n (ms)
async function roundTrip(device, n = 30, bytes = 65536) {
  const src = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), ts = [];
  for (let k = 0; k < n; k++) {
    const t = performance.now(), enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, dst, 0, bytes);
    device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    dst.unmap();
    ts.push(performance.now() - t);
  }
  src.destroy(); dst.destroy();
  return median(ts);
}

// Each of `windows` ({ name, nFree, nPts, track, reps }) solved by ba.js and on the GPU from the same start, reps times
// after a warm-up: { roundTrip (ms), windows: [{ ...window, obs, cpu, gpu: { ms, iterations, solves, cost, ... } }] }
export async function run(windows, { maxIterations = 10 } = {}) {
  const gpu = await GpuBundleAdjuster.create(), out = { roundTrip: await roundTrip(gpu.device), windows: [] };
  for (const w of windows) {
    const prob = makeWindow(w), cpu = [], hyb = [];
    let a, b, ra, rb;
    for (let r = 0; r <= (w.reps ?? 3); r++) {
      a = copy(prob);
      let t = performance.now();
      ra = bundleAdjust(a, { maxIterations });
      if (r) cpu.push(performance.now() - t);
      b = copy(prob);
      t = performance.now();
      rb = await gpu.bundleAdjust(b, { maxIterations });
      if (r) hyb.push(performance.now() - t);
    }
    out.windows.push({ ...w, obs: prob.obsCam.length,
      cpu: { ms: median(cpu), iterations: ra.iterations, solves: ra.profile.solves, cost: ra.cost },
      gpu: { ms: median(hyb), iterations: rb.iterations, solves: rb.profile.solves, cost: rb.cost, ...rb.profile },
      rotApartDeg: Math.max(...a.poses.map((x, k) => rotationAngle(x.R, b.poses[k].R) * 180 / Math.PI)) });
  }
  return out;
}
