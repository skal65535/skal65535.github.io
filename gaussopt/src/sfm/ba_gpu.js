// ba_gpu.js — Bundle adjustment on the GPU, its reduced camera system solved on the CPU: the GPU linearizes, inverts the
// points' blocks and sums the reduced system's terms (ba_gpu.wgsl: a workgroup per chunk of terms, no atomics), the CPU
// reads it back and solves it by Cholesky in f64, the GPU back-substitutes the points and sums the candidate's cost, read
// back too: two round trips per LM try. The problem, options and result of ba.js bundleAdjust (async here), with the
// window's conditioning (cond); the window solved centred on its points' median and scaled by their median distance
// (normalizeFrame). The points and the GPU's sums are f32, the poses, intrinsics and the reduced system f64.
"use strict";

import { expSO3, mul3, mulVec3, choleskyFactor } from './linalg.js';
import { ReducedSystem, DENSE_MAX, CG_TOL } from './rcs.js';
import { intrinsics, distortionParams, paramIndex, withFocal } from './camera.js';
import { INTRINSICS, normalizeFrame } from './ba.js';
import { computePipelines } from '../gpu.js';
import { initGPU, writeInPieces } from '../../../common/webgpu.js';

const ENTRIES = ['linearize', 'points', 'pointPred', 'system', 'gradient', 'back', 'cost'];
const CHUNK = 2048;  // terms per workgroup
const U = globalThis.GPUBufferUsage;
// bytes rounded up to 16, at least 16 (arithmetic: & ~15 wraps from 2^31 bytes on)
export const pad16 = (n) => Math.max(16, Math.ceil(n / 16) * 16);
let shared = null;

export class GpuBundleAdjuster {
  // One per realm (a worker, a test run), made on first use, again once its device is lost or its making failed; rejects
  // where WebGPU is not there
  static shared() {
    if (!shared) {
      const made = shared = GpuBundleAdjuster.create();
      const forget = () => { if (shared === made) shared = null; };
      made.then(g => g.device.lost.then(forget), forget);
    }
    return shared;
  }

  // The realm's solver idle (idle), once it is made: a worker between its actions
  static async releaseShared() {
    const g = await shared?.catch(() => null);
    g?.idle();
  }

  static async create(device = null) {
    device ??= (await initGPU(null)).device;
    return new GpuBundleAdjuster(device, await computePipelines(device, new URL('./ba_gpu.wgsl', import.meta.url), ENTRIES));
  }

  constructor(device, pipelines) {
    this.device = device; this.pipelines = pipelines;
    this.lost = false;  // the device lost: every solve fails from then on
    device.lost?.then(() => { this.lost = true; });
    // the buffers kept from one solve to the next, by role, grown when a window needs more (they were 26 made and
    // destroyed a solve, hundreds of solves a mapping); solving: the solve that has them (its token; another one at the
    // same time, a stalled one's successor, makes its own)
    this.pool = new Map();
    this.solving = null;
  }

  // The kept buffers destroyed (the next solve makes them again)
  release() {
    for (const b of this.pool.values()) b.destroy();
    this.pool.clear();
  }

  // Between its caller's actions: the kept buffers destroyed, the next solve's kept again (a solve on now is one its
  // caller's watchdog gave up: it fails on its destroyed buffers)
  idle() {
    this.release();
    this.solving = null;
  }

  // problem, opts: ba.js bundleAdjust's. Returns { initialCost, cost, iterations, rmse,
  // profile, cond }; profile: the problem's size, setup (the indexes and uploads), build (the GPU's passes up to the
  // reduced system read back), cholesky (its assembly and solve), eval (the step's back-substitution and cost read back),
  // in ms, and the counts (solves, evals, terms, chunks)
  async bundleAdjust(problem, opts = {}) {
    const { refine = [], refineCameras = null, lossScale = 2, maxIterations = 30, fixPoints = false, fixedPoints = null, loss = 'huber',
            denseMax = DENSE_MAX, cgTol = CG_TOL, alive = null } = opts;
    if (fixedPoints || loss !== 'huber') throw new Error('the GPU solver has the Huber loss only and no fixedPoints: ba.js has them');
    const { device, pipelines } = this, now = () => performance.now();
    let t = now();
    const { poses, free, points } = problem;
    let { obsCam, obsPoint, obsUV } = problem;
    const single = !problem.cameras, numPoses = poses.length, poseCam = problem.poseCam ?? new Int32Array(numPoses);
    let cams = single ? [problem.camera] : [...problem.cameras];
    const model = cams[0].model;
    if (cams.some(c => c.model !== model)) throw new Error('GpuBundleAdjuster: the camera models must be of one type');
    const slot = (name) => (name === 'f' ? Math.max(paramIndex(model, 'f'), paramIndex(model, 'fx')) : paramIndex(model, name));
    const kIdx = INTRINSICS.map((name, k) => [name, k]).filter(([name]) => refine.includes(name) && slot(name) >= 0).map(([, k]) => k);
    const ni = kIdx.length, numObs = obsCam.length, numPoints = points.length / 3;
    const profile = { obs: numObs, freeObs: 0, points: numPoints, free: 0, solves: 0, evals: 0, terms: 0, chunks: 0,
                      setup: 0, build: 0, cholesky: 0, eval: 0 };

    // ── host: the frame, the blocks (free poses, then refined lenses), the halves' blocks ──
    const { c: c0, s: sc } = normalizeFrame(points);
    let P = poses.map(({ R, t: tt }) => { const Rc = mulVec3(R, c0); return { R: [...R], t: [0, 1, 2].map(j => (tt[j] + Rc[j]) / sc) }; });
    const blockOf = new Int32Array(numPoses).fill(-1);
    let nb = 0;
    for (let k = 0; k < numPoses; k++) if (free[k]) blockOf[k] = nb++;
    profile.free = nb;
    const used = new Set(poseCam), intrOf = new Int32Array(cams.length).fill(-1);
    let nm = 0;
    if (ni) cams.forEach((_, k) => { if (used.has(k) && (!refineCameras || refineCameras[k])) intrOf[k] = nm++; });
    const nB = nb + nm, iOff = 6 * nb, m = iOff + ni * nm;
    const size = (k) => (k < nb ? 6 : ni), offset = (k) => (k < nb ? 6 * k : iOff + ni * (k - nb));
    // the observations of a free pose or a refined lens first, nA of them: theirs alone have pose and lens rows on the
    // GPU (JH, Y), the others their residual and point rows (JP) only (a large window's most: 3.7M of 4.15M)
    const active = (o) => blockOf[obsCam[o]] >= 0 || intrOf[poseCam[obsCam[o]]] >= 0;
    let nA = 0;
    for (let o = 0; o < numObs; o++) if (active(o)) nA++;
    if (nA < numObs) {
      const order = new Uint32Array(numObs);
      for (let o = 0, a = 0, f = nA; o < numObs; o++) order[active(o) ? a++ : f++] = o;
      const cam = new Int32Array(numObs), pt = new Int32Array(numObs), uv = new Float64Array(2 * numObs);
      order.forEach((o, k) => { cam[k] = obsCam[o]; pt[k] = obsPoint[o]; uv[2 * k] = obsUV[2 * o]; uv[2 * k + 1] = obsUV[2 * o + 1]; });
      obsCam = cam; obsPoint = pt; obsUV = uv;
    }
    const halfBlock = new Int32Array(2 * numObs);
    for (let o = 0; o < numObs; o++) {
      const k = obsCam[o], io = intrOf[poseCam[k]];
      halfBlock[2 * o] = blockOf[k];
      halfBlock[2 * o + 1] = io >= 0 ? nb + io : -1;
      if (blockOf[k] >= 0) profile.freeObs++;
    }

    // ── the indexes: the points' observations; per block pair (a >= b) the half pairs of each point, chunked ──
    const ptStart = new Uint32Array(numPoints + 1);
    for (let o = 0; o < numObs; o++) ptStart[obsPoint[o] + 1]++;
    for (let p = 0; p < numPoints; p++) ptStart[p + 1] += ptStart[p];
    const ptObs = new Uint32Array(numObs);
    { const fill = ptStart.slice(0, numPoints); for (let o = 0; o < numObs; o++) ptObs[fill[obsPoint[o]]++] = o; }
    const pairOf = (a, b) => a * (a + 1) / 2 + b, nPair = nB * (nB + 1) / 2;
    // each point's half pairs, counted then placed by block pair (fill = null: counted)
    const pairCount = new Uint32Array(nPair + 1);
    const eachPair = (fill, terms) => {
      for (let p = 0; p < numPoints; p++) {
        for (let i = ptStart[p]; i < ptStart[p + 1]; i++) for (let s = 0; s < 2; s++) {
          const ha = 2 * ptObs[i] + s, a = halfBlock[ha];
          if (a < 0) continue;
          for (let j = ptStart[p]; j < ptStart[p + 1]; j++) for (let u = 0; u < 2; u++) {
            const hb = 2 * ptObs[j] + u, b = halfBlock[hb];
            if (b < 0 || b > a) continue;
            const q = pairOf(a, b);
            if (!fill) { pairCount[q + 1]++; continue; }
            const d = fill[q]++;
            terms[2 * d] = ha; terms[2 * d + 1] = hb;
          }
        }
      }
    };
    eachPair(null);
    for (let q = 0; q < nPair; q++) pairCount[q + 1] += pairCount[q];
    const nTerms = pairCount[nPair], terms = new Uint32Array(2 * Math.max(1, nTerms));
    eachPair(pairCount.slice(0, nPair), terms);
    // chunks of one block pair each: [start, end), their pairs' blocks for the assembly
    const chunked = (start, n) => {
      const ranges = [], owner = [];
      for (let q = 0; q < n; q++) for (let s = start[q]; s < start[q + 1]; s += CHUNK) { ranges.push(s, Math.min(start[q + 1], s + CHUNK)); owner.push(q); }
      return { ranges: Uint32Array.from(ranges.length ? ranges : [0, 0]), owner, n: owner.length };
    };
    const sys = chunked(pairCount, nPair);
    const blocksOf = (q) => { let a = Math.floor((Math.sqrt(8 * q + 1) - 1) / 2); if (pairOf(a + 1, 0) <= q) a++; else if (pairOf(a, 0) > q) a--; return [a, q - pairOf(a, 0)]; };
    const sizes = Array.from({ length: nB }, (_, k) => size(k));
    // per block, its halves (the gradient's terms)
    const gStart = new Uint32Array(nB + 1);
    for (let h = 0; h < 2 * numObs; h++) if (halfBlock[h] >= 0) gStart[halfBlock[h] + 1]++;
    for (let k = 0; k < nB; k++) gStart[k + 1] += gStart[k];
    const gTerms = new Uint32Array(Math.max(1, gStart[nB]));
    { const fill = gStart.slice(0, nB); for (let h = 0; h < 2 * numObs; h++) if (halfBlock[h] >= 0) gTerms[fill[halfBlock[h]]++] = h; }
    const grad = chunked(gStart, nB);
    profile.terms = nTerms; profile.chunks = sys.n;

    // a pass of n workgroups dispatched as x by y, x at most maxComputeWorkgroupsPerDimension (a window of 4.5M observations
    // has 101k chunks), y too
    const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
    // a kept buffer's growth: within a binding's limit (none given: a test's limits)
    const maxPooled = Math.floor((device.limits.maxStorageBufferBindingSize ?? Infinity) / 16) * 16;
    if (Math.max(Math.ceil(numObs / 64), sys.n, grad.n) > maxGroups * maxGroups)
      throw new Error(`GpuBundleAdjuster: ${numObs} observations, ${sys.n} chunks: over ${maxGroups}^2 workgroups a pass`);

    // ── the buffers ── (a validation error from here to the start's cost: thrown, not a solve of garbage; the buffers
    // destroyed and the error scope popped whatever happens)
    device.pushErrorScope('validation');
    let scoped = true;
    const token = {}, pooled = !this.solving, buffers = [];
    let failed = false;
    if (pooled) this.solving = token;
    try {
      // the role's kept buffer when it is large enough, else one a quarter larger than asked (fewer regrowths as the
      // windows grow); a solve beside another: its own, destroyed at its end
      const make = (name, bytes, usage) => {
        const size = pad16(bytes), kept = pooled ? this.pool.get(name) : null;
        if (kept && kept.size >= size && kept.usage === usage) return kept;
        kept?.destroy();
        const b = device.createBuffer({ size: pooled ? Math.max(size, Math.min(pad16(1.25 * size), maxPooled)) : size, usage });
        if (pooled) this.pool.set(name, b); else buffers.push(b);
        return b;
      };
      const buf = (name, bytes, usage = U.STORAGE) => make(name, bytes, usage | U.COPY_DST | U.COPY_SRC);
      const upload = (name, data, usage) => { const b = buf(name, data.byteLength, usage); writeInPieces(device.queue, b, 0, data); return b; };
      const staging = (name, bytes) => make(name, bytes, U.MAP_READ | U.COPY_DST);
      const poseData = (Ps) => {
        const a = new Float32Array(16 * numPoses), bits = new Uint32Array(a.buffer);
        Ps.forEach(({ R, t: tt }, k) => {
          for (let i = 0; i < 3; i++) { a.set(R.slice(3 * i, 3 * i + 3), 16 * k + 4 * i); a[16 * k + 4 * i + 3] = tt[i]; }
          bits[16 * k + 12] = poseCam[k];
        });
        return a;
      };
      const lensData = (C) => Float32Array.from(C.flatMap(cam => [...intrinsics(cam), ...distortionParams(cam)]));
      const X = new Float32Array(4 * numPoints);
      for (let p = 0; p < numPoints; p++) for (let i = 0; i < 3; i++) X[4 * p + i] = (points[3 * p + i] - c0[i]) / sc;
      const obsData = new Uint32Array(4 * numObs), uvBits = new Float32Array(obsData.buffer);
      for (let o = 0; o < numObs; o++) {
        obsData[4 * o] = obsCam[o]; obsData[4 * o + 1] = obsPoint[o]; uvBits[4 * o + 2] = obsUV[2 * o]; uvBits[4 * o + 3] = obsUV[2 * o + 1];
      }
      const params = new ArrayBuffer(64), pf = new Float32Array(params), pu = new Uint32Array(params);
      pf[0] = lossScale; pu[2] = numObs; pu[3] = numPoints; pu[4] = ni; pu[5] = fixPoints ? 1 : 0; pu[6] = sys.n; pu[7] = grad.n; pu[13] = nA;
      kIdx.forEach((k, j) => { pu[j < 5 ? 8 + j : 14] = k; });  // k, k4, then k5
      const costBytes = 16 * Math.ceil(numObs / 256), ppN = Math.ceil(numPoints / 256);
      const pose0 = poseData(P), lens0 = lensData(cams);
      const B = {
        params: upload('params', params, U.UNIFORM), lens: upload('lens', lens0), poses: upload('poses', pose0), pts: upload('pts', X),
        obs: upload('obs', obsData), JP: buf('JP', 36 * numObs), JH: buf('JH', 96 * nA), Y: buf('Y', 144 * nA),
        ptStart: upload('ptStart', ptStart), ptObs: upload('ptObs', ptObs), halfBlock: upload('halfBlock', halfBlock.subarray(0, 2 * nA)),
        Vi: buf('Vi', 64 * numPoints),
        chunks: upload('chunks', sys.ranges), terms: upload('terms', terms), S: buf('S', 168 * sys.n), gChunks: upload('gChunks', grad.ranges),
        gTerms: upload('gTerms', gTerms), G: buf('G', 24 * grad.n), dX: buf('dX', 24 * nB), ptsCand: upload('ptsCand', X),
        posesCand: upload('posesCand', pose0), lensCand: upload('lensCand', lens0), C: buf('C', costBytes), PP: buf('PP', 4 * ppN),
      };
      const sysBytes = 168 * sys.n + 24 * grad.n + 4 * ppN, sysRead = staging('sysRead', sysBytes), costRead = staging('costRead', costBytes);
      const group = (name, list) => device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0),
        entries: list.map(([binding, b]) => ({ binding, resource: { buffer: B[b] } })) });
      const groups = {
        linearize: group('linearize', [[0, 'params'], [1, 'lens'], [2, 'poses'], [3, 'pts'], [4, 'obs'], [5, 'JP'], [23, 'JH']]),
        points: group('points', [[0, 'params'], [5, 'JP'], [6, 'Y'], [7, 'ptStart'], [8, 'ptObs'], [9, 'halfBlock'], [10, 'Vi'], [23, 'JH']]),
        pointPred: group('pointPred', [[0, 'params'], [10, 'Vi'], [22, 'PP']]),
        system: group('system', [[0, 'params'], [5, 'JP'], [6, 'Y'], [11, 'chunks'], [12, 'terms'], [13, 'S'], [23, 'JH']]),
        gradient: group('gradient', [[0, 'params'], [4, 'obs'], [5, 'JP'], [6, 'Y'], [10, 'Vi'], [14, 'gChunks'], [15, 'gTerms'], [16, 'G'], [23, 'JH']]),
        back: group('back', [[0, 'params'], [3, 'pts'], [6, 'Y'], [7, 'ptStart'], [8, 'ptObs'], [9, 'halfBlock'], [10, 'Vi'], [17, 'dX'], [18, 'ptsCand']]),
        cost: group('cost', [[0, 'params'], [4, 'obs'], [18, 'ptsCand'], [19, 'posesCand'], [20, 'lensCand'], [21, 'C']]),
      };
      const pass = (enc, name, n) => {
        if (!n) return;
        const p = enc.beginComputePass();
        p.setPipeline(pipelines[name]); p.setBindGroup(0, groups[name]);
        const x = Math.min(n, maxGroups);
        p.dispatchWorkgroups(x, Math.ceil(n / x));
        p.end();
      };
      const read = async (dst, bytes) => {
        await dst.mapAsync(GPUMapMode.READ, 0, pad16(bytes));
        alive?.();  // the GPU answered (the caller's watchdog: a long solve is not a stalled one)
        const out = new Float32Array(dst.getMappedRange(0, pad16(bytes)).slice(0, bytes));
        dst.unmap();
        return out;
      };

      // the candidate's cost (ptsCand, posesCand, lensCand), its sums read back and added in f64
      const evalCost = async (enc) => {
        pass(enc, 'cost', Math.ceil(numObs / 256));
        enc.copyBufferToBuffer(B.C, 0, costRead, 0, costBytes);
        device.queue.submit([enc.finish()]);
        const c = await read(costRead, costBytes);
        let cost = 0, sq = 0, n = 0;
        for (let k = 0; k < c.length; k += 4) { cost += c[k]; sq += c[k + 1]; n += c[k + 2]; }
        profile.evals++;
        return { cost, rmse: Math.sqrt(sq / Math.max(1, n)) };
      };
      // the reduced system at lambda (the points' blocks redone), read back: { S (m x m, its lower triangle), g, ud, pp: the
      // decrease the points alone predict, sum of gp^T V^-1 gp } in f64
      const build = async (lambda, relinearize) => {
        pf[1] = lambda;
        device.queue.writeBuffer(B.params, 0, params);
        const enc = device.createCommandEncoder();
        if (relinearize) pass(enc, 'linearize', Math.ceil(numObs / 64));
        pass(enc, 'points', Math.ceil(numPoints / 64));
        pass(enc, 'pointPred', ppN);
        pass(enc, 'system', sys.n);
        pass(enc, 'gradient', grad.n);
        enc.copyBufferToBuffer(B.S, 0, sysRead, 0, 168 * sys.n);
        enc.copyBufferToBuffer(B.G, 0, sysRead, 168 * sys.n, 24 * grad.n);
        enc.copyBufferToBuffer(B.PP, 0, sysRead, 168 * sys.n + 24 * grad.n, 4 * ppN);
        device.queue.submit([enc.finish()]);
        const raw = await read(sysRead, sysBytes);
        const S = new ReducedSystem(sizes, { denseMax, tol: cgTol }), g = new Float64Array(m), ud = new Float64Array(m);
        sys.owner.forEach((q, c) => {
          const [a, b] = blocksOf(q), oa = offset(a), na = size(a), nbw = size(b), [A, base, stride] = S.at(a, b);
          for (let i = 0; i < na; i++) {
            for (let j = 0, nj = a === b ? i + 1 : nbw; j < nj; j++) A[base + i * stride + j] += raw[42 * c + 6 * i + j];
            if (a === b) ud[oa + i] += raw[42 * c + 36 + i];
          }
        });
        const g0 = 42 * sys.n;
        grad.owner.forEach((k, c) => { for (let i = 0; i < size(k); i++) g[offset(k) + i] += raw[g0 + 6 * c + i]; });
        let pp = 0;
        for (let k = 0; k < ppN; k++) pp += raw[g0 + 6 * grad.n + k];
        // an f32 overflow (a NaN, an infinity) thrown: it would fail every Cholesky, the solve then taking no step, unseen
        if (!Number.isFinite(S.sum() + g.reduce((a, x) => a + x, 0) + pp)) throw new Error('GpuBundleAdjuster: the reduced system is not finite');
        return { S, g, ud, pp };
      };

      await device.queue.onSubmittedWorkDone();
      profile.setup = now() - t;

      let { cost, rmse } = await evalCost(device.createCommandEncoder());  // the candidates hold the start
      const error = await device.popErrorScope();
      scoped = false;
      if (error) throw new Error(`GpuBundleAdjuster: ${error.message}`);
      const initialCost = cost, initialRmse = rmse;
      let lambda = 1e-4, it = 0;
      for (; it < maxIterations; it++) {
        let accepted = false, relinearize = true;
        for (let tries = 0; tries < 10 && !accepted; tries++) {
          t = now();
          const { S, g, ud, pp } = await build(lambda, relinearize);
          relinearize = false;
          profile.build += now() - t; t = now();
          profile.solves++;
          const D = Float64Array.from(ud, x => lambda * Math.max(x, 1e-6));
          S.addDiagonal(D);
          const dx = S.solve(g.map(x => -x));
          alive?.();  // a long solve on the CPU is no stall either
          profile.cholesky += now() - t;
          if (!S.dense) { profile.cg = (profile.cg ?? 0) + S.iterations; if (!S.converged) profile.cgCut = (profile.cgCut ?? 0) + 1; }
          if (!dx) { lambda *= 10; continue; }
          // the decrease the model predicts, the points' alone and the reduced system's -g^T dx + dx^T D dx (f64; the points'
          // damping left out): under ba.js's 1e-7 of the cost, converged (the f32 cost itself resolves about 1e-6: its sums,
          // the points' rounding)
          let pred = pp;
          for (let i = 0; i < m; i++) pred += dx[i] * (D[i] * dx[i] - g[i]);
          if (pred < 1e-7 * cost) break;
          t = now();
          // the candidate: q' = exp(dw) q + dt, the intrinsics stepped; dX per block, padded to 6
          const Pc = P.map((pose, k) => {
            const b = blockOf[k];
            if (b < 0) return pose;
            const dR = expSO3([dx[6 * b], dx[6 * b + 1], dx[6 * b + 2]]), t2 = mulVec3(dR, pose.t);
            return { R: mul3(dR, pose.R), t: [0, 1, 2].map(j => t2[j] + dx[6 * b + 3 + j]) };
          });
          const Cc = cams.map((cam, c) => {
            if (intrOf[c] < 0) return cam;
            kIdx.forEach((kk, j) => {
              const step = dx[iOff + ni * intrOf[c] + j];
              if (kk === 0) cam = withFocal(cam, cam.params[0] + step);
              else { const p = [...cam.params]; p[slot(INTRINSICS[kk])] += step; cam = { ...cam, params: p }; }
            });
            return cam;
          });
          const d6 = new Float32Array(6 * nB);
          for (let k = 0; k < nB; k++) for (let i = 0; i < size(k); i++) d6[6 * k + i] = dx[offset(k) + i];
          if (nB) device.queue.writeBuffer(B.dX, 0, d6);
          device.queue.writeBuffer(B.posesCand, 0, poseData(Pc));
          device.queue.writeBuffer(B.lensCand, 0, lensData(Cc));
          const enc = device.createCommandEncoder();
          pass(enc, 'back', Math.ceil(numPoints / 64));
          const e = await evalCost(enc);
          profile.eval += now() - t;
          if (!(e.cost < cost)) { lambda *= 5; continue; }
          P = Pc; cams = Cc;
          const ec = device.createCommandEncoder();
          ec.copyBufferToBuffer(B.ptsCand, 0, B.pts, 0, 16 * numPoints);
          ec.copyBufferToBuffer(B.posesCand, 0, B.poses, 0, 64 * numPoses);
          ec.copyBufferToBuffer(B.lensCand, 0, B.lens, 0, 32 * cams.length);
          device.queue.submit([ec.finish()]);
          cost = e.cost; rmse = e.rmse;
          lambda = Math.max(1e-10, lambda / 3);
          accepted = true;
        }
        if (!accepted) break;  // converged, or no decrease even with heavy damping
      }

      // The window's conditioning at the solution, undamped: per free pose, the smallest Cholesky pivot of its
      // block of the reduced system Jacobi-scaled; minPivot their least (null when gaugeFree: fewer than 2 fixed poses, the
      // scale free, the pivots saying nothing)
      const gaugeFree = numPoses - nb < 2;
      const S = nb && !gaugeFree ? (await build(0, true)).S : null;
      let minPivot = gaugeFree ? null : 1;
      const Ms = new Float64Array(36), Ls = new Float64Array(36);
      for (let b = 0; S && b < nb; b++) {
        const [A, base, stride] = S.at(b, b), d = Array.from({ length: 6 }, (_, i) => Math.sqrt(A[base + i * stride + i]));
        let piv = 0;
        if (d.every(x => x > 0)) {
          for (let i = 0; i < 6; i++) for (let j = 0; j <= i; j++) Ms[6 * i + j] = A[base + i * stride + j] / (d[i] * d[j]);
          if (choleskyFactor(Ms, 0, Ls, 0, 6)) { piv = Infinity; for (let i = 0; i < 6; i++) piv = Math.min(piv, Ls[7 * i] ** 2); }
        }
        minPivot = Math.min(minPivot, piv);
      }
      const cond = { minPivot, gaugeFree, rmse0: initialRmse, rmse };

      // ── back to the problem's frame (fixed points as they were, not through f32) ──
      if (!fixPoints) {
        const back = staging('back', 16 * numPoints), enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(B.pts, 0, back, 0, 16 * numPoints);
        device.queue.submit([enc.finish()]);
        const Xs = await read(back, 16 * numPoints);
        for (let p = 0; p < numPoints; p++) for (let i = 0; i < 3; i++) points[3 * p + i] = sc * Xs[4 * p + i] + c0[i];
      }
      for (let k = 0; k < numPoses; k++) {
        if (!free[k]) continue;
        const { R, t: tt } = P[k], Rc = mulVec3(R, c0);
        poses[k] = { R, t: [0, 1, 2].map(j => sc * tt[j] - Rc[j]) };
      }
      if (single) problem.camera = cams[0]; else problem.cameras = cams;
      return { initialCost, cost, iterations: it, rmse, profile, cond };
    } catch (e) {
      failed = true;
      throw e;
    } finally {
      if (scoped) await device.popErrorScope().catch(() => {});
      for (const b of buffers) b.destroy();
      // a failed solve's buffers not trusted again; the flag left to a newer solve when releaseShared took them meanwhile
      if (pooled && this.solving === token) { if (failed) this.release(); this.solving = null; }
    }
  }
}
