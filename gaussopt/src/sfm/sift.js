// sift.js — Host side of the GPU SIFT (sift.wgsl): builds the Gaussian / DoG pyramid octave by
// octave, runs detection, orientation and description on the GPU, and reads back once per image.
// Output follows COLMAP: keypoints [x, y, scale, orientation] with pixel centers at +0.5, and
// uint8 descriptors (128 per keypoint), the strongest maxFeatures by DoG response.
"use strict";

import { initGPU, createGPUBuffer, createUniformBuffer, createStorageBuffer } from '../../../common/webgpu.js';
import { computePipelines, dispatch } from '../gpu.js';

export const SIFT_DEFAULTS = {
  maxFeatures: 4096, numOctaves: 4, levels: 3, sigma0: 1.6, inputSigma: 0.5,
  peak: 0.02 / 3, edge: 10, maxKeypoints: 65536,
};
const ENTRIES = ['to_float', 'blur_h', 'blur_v', 'downsample', 'diff', 'snapshot', 'extrema', 'orient', 'describe'];

function gaussianWeights(sigma) {
  const r = Math.max(1, Math.ceil(4 * sigma));
  const w = Float32Array.from({ length: 2 * r + 1 }, (_, k) => Math.exp(-((k - r) ** 2) / (2 * sigma * sigma)));
  const s = w.reduce((a, b) => a + b, 0);
  return { r, w: w.map(x => x / s) };
}

export class GpuSift {
  static async create(device = null) {
    device ??= (await initGPU(null)).device;
    return new GpuSift(device, await computePipelines(device, new URL('./sift.wgsl', import.meta.url), ENTRIES));
  }

  constructor(device, pipelines) { this.device = device; this.pipelines = pipelines; }

  // gray: Uint8Array (w h). Returns { keypoints: Float32Array(4 n), descriptors: Uint8Array(128 n), capped }:
  // capped when candidates or oriented keypoints exceeded maxKeypoints (the GPU kept the first ones
  // it appended, so which ones varies between runs)
  async extract(gray, width, height, options = {}) {
    const o = { ...SIFT_DEFAULTS, ...options };
    const { device, pipelines } = this;
    const S = o.levels, L = S + 3;
    const temp = [];  // buffers to free after the submit
    const params = (p) => {
      const buf = createUniformBuffer(device, 64);
      const u = new Uint32Array(16), f = new Float32Array(u.buffer);
      u[0] = p.w; u[1] = p.h; u[2] = p.radius ?? 0; u[3] = p.src_off ?? 0;
      u[4] = p.dst_off ?? 0; u[5] = p.level ?? 0; u[6] = p.octave ?? 0; u[7] = S;
      f[8] = o.sigma0; f[9] = o.peak; f[10] = o.edge; f[11] = p.scale ?? 1;
      u[12] = o.maxKeypoints;
      device.queue.writeBuffer(buf, 0, u);
      temp.push(buf);
      return buf;
    };
    const enc = device.createCommandEncoder();
    const run = (entry, resources, groups) => dispatch(device, enc, pipelines[entry], resources, ...groups);
    const grid = (w, h) => [Math.ceil(w / 16), Math.ceil(h / 16)];

    const padded = new Uint8Array(Math.ceil(gray.length / 4) * 4);
    padded.set(gray);
    const grayBuf = createGPUBuffer(device, new Uint32Array(padded.buffer));
    const kp = createStorageBuffer(device, 32 * o.maxKeypoints);
    const okp = createStorageBuffer(device, 32 * o.maxKeypoints, GPUBufferUsage.COPY_SRC);
    const desc = createStorageBuffer(device, 128 * o.maxKeypoints, GPUBufferUsage.COPY_SRC);
    const counters = createGPUBuffer(device, new Uint32Array(2 + 2 * o.numOctaves), GPUBufferUsage.COPY_SRC);
    temp.push(grayBuf, kp, okp, desc, counters);

    const blur = (stack, tmp, w, h, src, dst, sigma) => {
      const { r, w: wts } = gaussianWeights(sigma);
      const wb = createGPUBuffer(device, wts);
      temp.push(wb);
      run('blur_h', { 0: params({ w, h, radius: r, src_off: src * w * h }), 3: stack, 4: wb, 5: tmp }, grid(w, h));
      run('blur_v', { 0: params({ w, h, radius: r, dst_off: dst * w * h }), 3: stack, 4: wb, 5: tmp }, grid(w, h));
    };

    let prev = null, w = width, h = height;
    for (let oct = 0; oct < o.numOctaves && Math.min(w, h) >= 32; oct++) {
      const n = w * h, scale = 2 ** oct;
      const gauss = createStorageBuffer(device, 4 * L * n), dog = createStorageBuffer(device, 4 * (L - 1) * n), tmp = createStorageBuffer(device, 4 * n);
      temp.push(gauss, dog, tmp);
      if (oct === 0) {
        run('to_float', { 0: params({ w, h }), 1: grayBuf, 2: gauss }, [Math.ceil(n / 256)]);
        blur(gauss, tmp, w, h, 0, 0, Math.sqrt(o.sigma0 ** 2 - o.inputSigma ** 2));
      } else {
        // level S of the previous octave has sigma 2 sigma0 = sigma0 at this octave's resolution
        run('downsample', { 0: params({ w, h, radius: prev.w, src_off: S * prev.w * prev.h }), 3: gauss, 6: prev.gauss }, grid(w, h));
      }
      for (let k = 1; k < L; k++) blur(gauss, tmp, w, h, k - 1, k, o.sigma0 * Math.sqrt(2 ** (2 * k / S) - 2 ** (2 * (k - 1) / S)));
      for (let k = 0; k < L - 1; k++) run('diff', { 0: params({ w, h, level: k }), 6: gauss, 7: dog }, [Math.ceil(n / 256)]);
      run('snapshot', { 0: params({ w, h, octave: oct }), 9: counters }, [1]);
      for (let k = 1; k <= S; k++) {
        run('extrema', { 0: params({ w, h, level: k, octave: oct, scale }), 6: dog, 8: kp, 9: counters }, grid(w, h));
      }
      run('orient', { 0: params({ w, h, octave: oct, scale }), 6: gauss, 8: kp, 9: counters, 10: okp }, [Math.ceil(o.maxKeypoints / 64)]);
      run('describe', { 0: params({ w, h, octave: oct, scale }), 6: gauss, 9: counters, 10: okp, 11: desc }, [Math.ceil(o.maxKeypoints / 64)]);
      prev = { gauss, w, h };
      w = Math.floor(w / 2); h = Math.floor(h / 2);
    }

    const readCounters = device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readKp = device.createBuffer({ size: 32 * o.maxKeypoints, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readDesc = device.createBuffer({ size: 128 * o.maxKeypoints, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(counters, 0, readCounters, 0, 8);
    enc.copyBufferToBuffer(okp, 0, readKp, 0, 32 * o.maxKeypoints);
    enc.copyBufferToBuffer(desc, 0, readDesc, 0, 128 * o.maxKeypoints);
    device.queue.submit([enc.finish()]);
    await Promise.all([readCounters, readKp, readDesc].map(b => b.mapAsync(GPUMapMode.READ)));
    const [candidates, oriented] = new Uint32Array(readCounters.getMappedRange());
    const count = Math.min(oriented, o.maxKeypoints), capped = Math.max(candidates, oriented) > o.maxKeypoints;
    const rec = new Float32Array(readKp.getMappedRange()).slice(0, 8 * count);
    const allDesc = new Uint8Array(readDesc.getMappedRange()).slice(0, 128 * count);
    for (const b of [readCounters, readKp, readDesc]) { b.unmap(); b.destroy(); }
    for (const b of temp) b.destroy();

    // strongest maxFeatures by DoG response; image coordinates with COLMAP's +0.5 pixel centers.
    // The GPU appends in a varying order: ties are broken by octave, position, orientation, so
    // the same image always gives the same keypoints in the same order (DESIGN-sfm §5.7)
    const key = [5, 3, 1, 0, 6], sign = [-1, 1, 1, 1, 1];
    const order = Array.from({ length: count }, (_, i) => i).sort((a, b) => {
      for (let k = 0; k < key.length; k++) {
        const d = rec[8 * a + key[k]] - rec[8 * b + key[k]];
        if (d) return sign[k] * d;
      }
      return 0;
    }).slice(0, o.maxFeatures);
    const keypoints = new Float32Array(4 * order.length), descriptors = new Uint8Array(128 * order.length);
    order.forEach((i, k) => {
      const scale = 2 ** rec[8 * i + 3];
      keypoints.set([rec[8 * i] * scale + 0.5, rec[8 * i + 1] * scale + 0.5, rec[8 * i + 4], rec[8 * i + 6]], 4 * k);
      descriptors.set(allDesc.subarray(128 * i, 128 * i + 128), 128 * k);
    });
    return { keypoints, descriptors, capped };
  }
}
