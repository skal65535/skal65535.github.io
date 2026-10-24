// match.js — Descriptor matching (DESIGN-sfm §5.3): GPU brute force (match.wgsl) plus COLMAP's
// filters, and a CPU reference with identical semantics for the tests. Descriptors are COLMAP's
// SIFT uint8 (unit L2 x 512, 128 per keypoint). A match passes when its angle
// acos(dot / 512^2) is below maxDistance and below maxRatio x the second best, and (crossCheck)
// when it is also the best match in the other direction.
"use strict";

import { initGPU, createGPUBuffer, createUniformBuffer, createStorageBuffer } from '../../../common/webgpu.js';
import { computePipelines, dispatch } from '../gpu.js';

const NORM = 1 / (512 * 512);
const angle = (dot) => Math.acos(Math.min(1, dot * NORM));
export const MATCH_DEFAULTS = { maxRatio: 0.8, maxDistance: 0.7, crossCheck: true };

// best: Uint32Array(4 n) = [idx, bestDot, secondDot, 0] per query -> index of the accepted match or -1
function accept(best, n, { maxRatio, maxDistance }) {
  const out = new Int32Array(n).fill(-1);
  for (let q = 0; q < n; q++) {
    const idx = best[4 * q], d1 = angle(best[4 * q + 1]), d2 = angle(best[4 * q + 2]);
    if (idx === 0xffffffff || d1 >= maxDistance || d1 >= maxRatio * d2) continue;
    out[q] = idx;
  }
  return out;
}

// Accepted (a, b) pairs from both directions' nearest neighbours
function combine(ab, ba, nA, nB, options) {
  const o = { ...MATCH_DEFAULTS, ...options };
  const fwd = accept(ab, nA, o), bwd = o.crossCheck ? accept(ba, nB, o) : null;
  const pairs = [];
  for (let a = 0; a < nA; a++) {
    const b = fwd[a];
    if (b < 0 || (bwd && bwd[b] !== a)) continue;
    pairs.push(a, b);
  }
  return Uint32Array.from(pairs);
}

// CPU reference: nearest neighbours by exhaustive dot products
function nearestCPU(query, nQ, refs, nR) {
  const out = new Uint32Array(4 * nQ);
  for (let q = 0; q < nQ; q++) {
    let best = 0, second = 0, idx = 0xffffffff;
    for (let r = 0; r < nR; r++) {
      let s = 0;
      for (let k = 0; k < 128; k++) s += query[128 * q + k] * refs[128 * r + k];
      if (s > best) { second = best; best = s; idx = r; } else if (s > second) second = s;
    }
    out.set([idx, best, second, 0], 4 * q);
  }
  return out;
}

export function matchCPU(descA, descB, options = {}) {
  const nA = descA.length / 128, nB = descB.length / 128;
  return combine(nearestCPU(descA, nA, descB, nB), nearestCPU(descB, nB, descA, nA), nA, nB, options);
}

export class GpuMatcher {
  static async create(device = null) {
    device ??= (await initGPU(null)).device;
    const packedDot = navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product');
    const { main } = await computePipelines(device, new URL('./match.wgsl', import.meta.url), ['main'], code =>
      packedDot ? code : code.replace('return dot4U8Packed(a, b);', 'return dot(unpack4xU8(a), unpack4xU8(b));'));
    return new GpuMatcher(device, main);
  }

  constructor(device, pipeline) {
    this.device = device;
    this.pipeline = pipeline;
    this.cache = new Map();  // key -> { buf, n }
  }

  // Descriptors on the GPU, cached by key (e.g. the image index) when one is given. No descriptor:
  // a 4-byte buffer (bindings cannot be empty), n = 0.
  upload(desc, key = null) {
    if (key !== null && this.cache.has(key)) return this.cache.get(key);
    const u32 = desc.length ? new Uint32Array(desc.buffer, desc.byteOffset, desc.byteLength / 4) : new Uint32Array(1);
    const entry = { buf: createGPUBuffer(this.device, u32), n: desc.length / 128 };
    if (key !== null) this.cache.set(key, entry);
    return entry;
  }

  release(key) {
    const e = this.cache.get(key);
    if (e) { e.buf.destroy(); this.cache.delete(key); }
  }

  clear() { for (const key of [...this.cache.keys()]) this.release(key); }

  // Matches between two descriptor sets (Uint8Array, or upload() entries): Uint32Array [a, b, ...]
  async match(descA, descB, options = {}, keys = [null, null]) {
    const A = descA.buf ? descA : this.upload(descA, keys[0]);
    const B = descB.buf ? descB : this.upload(descB, keys[1]);
    const [m] = await this.matchBatch([[A, B]], options);
    if (keys[0] === null && !descA.buf) A.buf.destroy();
    if (keys[1] === null && !descB.buf) B.buf.destroy();
    return m;
  }

  // Several pairs [[A, B]] of upload() entries in one submit and one readback (both directions
  // per pair): per-dispatch latency, not arithmetic, dominates single-pair matching.
  async matchBatch(pairs, options = {}) {
    const o = { ...MATCH_DEFAULTS, ...options };
    const jobs = pairs.flatMap(([A, B]) => o.crossCheck ? [[A, B], [B, A]] : [[A, B]]);
    const near = await this.nearestBatch(jobs);
    return pairs.map(([A, B], p) => o.crossCheck
      ? combine(near[2 * p], near[2 * p + 1], A.n, B.n, o)
      : combine(near[p], null, A.n, B.n, o));
  }

  // Raw nearest neighbours for [[query, refs]] jobs (upload() entries): per job a Uint32Array of
  // [best index, best dot, second dot, 0] per query, one submit and one readback for all.
  async nearestBatch(jobs) {
    const { device } = this;
    // slices start on 256-byte boundaries (minStorageBufferOffsetAlignment): 16 results of 16 bytes
    let total = 0;
    const offsets = jobs.map(([Q]) => { const off = total; total += Math.ceil(Math.max(1, Q.n) / 16) * 16; return off; });
    const out = createStorageBuffer(device, 16 * Math.max(1, total), GPUBufferUsage.COPY_SRC);
    const read = device.createBuffer({ size: 16 * Math.max(1, total), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const temp = [out, read];
    const enc = device.createCommandEncoder();
    jobs.forEach(([Q, R], k) => {
      const uni = createUniformBuffer(device, 16);
      device.queue.writeBuffer(uni, 0, Uint32Array.of(Q.n, R.n, 0, 0));
      temp.push(uni);
      // each job writes its own slice of `out`; 64 queries per workgroup
      dispatch(device, enc, this.pipeline, [uni, Q.buf, R.buf, { buffer: out, offset: 16 * offsets[k], size: 16 * Math.max(1, Q.n) }],
               Math.max(1, Math.ceil(Q.n / 64)));
    });
    enc.copyBufferToBuffer(out, 0, read, 0, 16 * Math.max(1, total));
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const res = new Uint32Array(read.getMappedRange()).slice(0, 4 * total);
    read.unmap();
    for (const b of temp) b.destroy();
    return jobs.map(([Q], k) => res.subarray(4 * offsets[k], 4 * (offsets[k] + Q.n)));
  }
}
