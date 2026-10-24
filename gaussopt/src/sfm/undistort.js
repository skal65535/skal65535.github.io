// undistort.js — S5 hand-over (DESIGN-sfm §4): GPU remap of the keyframes from the SfM camera to
// the PINHOLE camera of the exported model (undistort.wgsl, as tools/undistort.py with
// cv2.undistort), and the point colors as the median over each point's observations.
"use strict";

import { initGPU } from '../../../common/webgpu.js';
import { computePipelines, dispatch } from '../gpu.js';
import { intrinsics, distortionParams } from './camera.js';

export class GpuUndistort {
  static async create(device = null) {
    device ??= (await initGPU(null)).device;
    const { main } = await computePipelines(device, new URL('./undistort.wgsl', import.meta.url));
    return new GpuUndistort(device, main);
  }

  constructor(device, pipeline) { this.device = device; this.pipeline = pipeline; this.size = 0; }

  _buffers(n) {
    if (this.size === n) return;
    for (const b of [this.src, this.dst, this.read]) b?.destroy();
    const d = this.device, bytes = 4 * n;
    this.src = d.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.dst = d.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.read = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.uniform ??= d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.size = n;
  }

  // rgba: Uint8Array | Uint8ClampedArray (w h 4) seen by `camera`; returns the image seen by
  // `pinhole` (same size), as a Uint8ClampedArray ready for ImageData
  async run(rgba, camera, pinhole) {
    const { device } = this, w = camera.width, h = camera.height;
    this._buffers(w * h);
    const u = new Float32Array(16), [fx, fy, cx, cy] = intrinsics(pinhole);
    new Uint32Array(u.buffer).set([w, h]);
    u.set([fx, fy, cx, cy, ...intrinsics(camera), ...distortionParams(camera)], 4);
    device.queue.writeBuffer(this.uniform, 0, u);
    device.queue.writeBuffer(this.src, 0, rgba.buffer, rgba.byteOffset, 4 * w * h);
    const enc = device.createCommandEncoder();
    dispatch(device, enc, this.pipeline, [this.uniform, this.src, this.dst], Math.ceil(w / 16), Math.ceil(h / 16));
    enc.copyBufferToBuffer(this.dst, 0, this.read, 0, 4 * w * h);
    device.queue.submit([enc.finish()]);
    await this.read.mapAsync(GPUMapMode.READ);
    const out = new Uint8ClampedArray(this.read.getMappedRange().slice(0));
    this.read.unmap();
    return out;
  }

  destroy() { for (const b of [this.src, this.dst, this.read, this.uniform]) b?.destroy(); this.size = 0; }
}

// Point colors of a datasetFromMapper() result, image by image: add(image, rgba) with the
// undistorted pixels of each image, then finish() writes the per-channel median of every point's
// observations into points[].rgb (tools/undistort.py's rule: the pixel containing the
// observation, i.e. floor with COLMAP's centers at +0.5; median, truncated)
export class PointColors {
  constructor({ camera, points }) {
    this.camera = camera;
    this.points = points;
    this.offset = new Map();  // point id -> first sample slot
    let n = 0;
    for (const p of points) { this.offset.set(p.id, n); n += p.track.length; }
    this.samples = new Uint8Array(3 * n);
    this.fill = new Uint32Array(points.length);
    this.index = new Map(points.map((p, k) => [p.id, k]));
  }

  add(image, rgba) {
    const { width: w, height: h } = this.camera;
    for (const [x, y, pid] of image.points2D) {
      const k = this.index.get(pid);
      if (k === undefined) continue;
      const u = Math.min(Math.max(Math.floor(x), 0), w - 1), v = Math.min(Math.max(Math.floor(y), 0), h - 1);
      const o = 4 * (v * w + u), s = 3 * (this.offset.get(pid) + this.fill[k]++);
      this.samples[s] = rgba[o]; this.samples[s + 1] = rgba[o + 1]; this.samples[s + 2] = rgba[o + 2];
    }
  }

  finish() {
    this.points.forEach((p, k) => {
      const n = this.fill[k], s = 3 * this.offset.get(p.id);
      if (!n) return;
      p.rgb = [0, 1, 2].map(c => {
        const v = Array.from({ length: n }, (_, j) => this.samples[s + 3 * j + c]).sort((a, b) => a - b);
        return n % 2 ? v[n >> 1] : Math.trunc((v[(n >> 1) - 1] + v[n >> 1]) / 2);
      });
    });
  }
}
