// stipple-frame.js — image/video source → grayscale GPU texture.
//
// FrameSource owns the source→gray pipeline (luma or edge-detect) and a
// running image-average atomic. Each captureFrame() submits one compute pass
// and (optionally) reads back per-pixel grayscale for the CPU iterator.
//
// The returned `frame` object is a thin handle:
//   { W, H, texture, pixels?, get average(), averageReady }
// `texture` is reused across captures, so iterators that bind it at
// construction time keep seeing fresh content without rebuilding bind groups.
// `average` is a live getter on the FrameSource so callers always read the
// most recent sum-readback value, even one frame behind the GPU.
"use strict";

import { loadShaders } from './stipple-shaders.js';

export class FrameSource {
  constructor(device) {
    this._device = device;
    this._W = 0;
    this._H = 0;
    this._tex_src    = null;
    this._tex_gray   = null;
    this._sampler    = null;
    this._ep_buf     = null;
    this._pipe_edge  = null;
    this._pipe_luma  = null;
    this._sum_buf    = null;
    this._sum_read   = null;
    this._read_buf   = null;
    this._readBytesPerRow = 0;
    this._scratch    = null;
    this._scratchCtx = null;
    this._average    = 0;
    this._sumPending = false;
    this._averageReadyRes = null;
    this._averageReady    = new Promise(r => { this._averageReadyRes = r; });
  }

  get average() { return this._average; }

  async _ensureSize(W, H) {
    if (this._tex_src && this._W === W && this._H === H) return;
    const device = this._device;
    const mods   = await loadShaders(device);

    this._tex_src?.destroy?.();
    this._tex_gray?.destroy?.();
    this._read_buf?.destroy?.();
    this._read_buf = null;

    this._W = W; this._H = H;
    this._tex_src = device.createTexture({
      size: { width: W, height: H }, format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
             GPUTextureUsage.RENDER_ATTACHMENT,
      label: 'frame_src',
    });
    this._tex_gray = device.createTexture({
      size: { width: W, height: H }, format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING |
             GPUTextureUsage.COPY_SRC,
      label: 'frame_gray',
    });
    if (!this._sampler) {
      this._sampler = device.createSampler({
        magFilter: 'linear', minFilter: 'linear',
        addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      });
      this._ep_buf = device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: 'edge_params',
      });
      this._pipe_edge = device.createComputePipeline({
        layout: 'auto', compute: { module: mods.edge_detect, entryPoint: 'main' },
      });
      this._pipe_luma = device.createComputePipeline({
        layout: 'auto', compute: { module: mods.luma, entryPoint: 'main' },
      });
      this._sum_buf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'image_sum',
      });
      this._sum_read = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: 'image_sum_read',
      });
    }
    if (!this._scratch) {
      this._scratch    = document.createElement('canvas');
      this._scratchCtx = this._scratch.getContext('2d');
    }
    this._scratch.width  = W;
    this._scratch.height = H;
  }

  // Capture src into the gray texture. Returns a frame handle.
  // opts = { W, H, useEdges, alpha, strength, needPixels }
  async captureFrame(src, opts) {
    const W = opts.W, H = opts.H;
    const useEdges = opts.useEdges !== false;
    await this._ensureSize(W, H);
    const device = this._device;

    // copyExternalImageToTexture crops to copySize, so blit src onto a fixed
    // W×H scratch canvas first to handle arbitrary source resolutions.
    this._scratchCtx.clearRect(0, 0, W, H);
    this._scratchCtx.drawImage(src, 0, 0, W, H);
    device.queue.copyExternalImageToTexture(
      { source: this._scratch }, { texture: this._tex_src }, { width: W, height: H },
    );

    const ep = new ArrayBuffer(32);
    const u  = new Uint32Array(ep);   const f = new Float32Array(ep);
    u[0] = W; u[1] = H;
    f[2] = opts.alpha;
    f[3] = 0.25 * (1.0 - opts.alpha);
    f[4] = opts.strength;
    device.queue.writeBuffer(this._ep_buf, 0, ep);
    device.queue.writeBuffer(this._sum_buf, 0, new Uint32Array([0]));

    const enc  = device.createCommandEncoder({ label: 'capture' });
    const pipe = useEdges ? this._pipe_edge : this._pipe_luma;
    const bg = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._ep_buf } },
        { binding: 1, resource: this._tex_src.createView() },
        { binding: 2, resource: this._sampler },
        { binding: 3, resource: this._tex_gray.createView() },
        { binding: 4, resource: { buffer: this._sum_buf } },
      ],
    });
    const pass = enc.beginComputePass({ label: 'edge' });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    pass.end();
    // Gate the sum copy on the readback buffer being available. While
    // sum_read is mapped (or pending map) WebGPU forbids any copy into it;
    // skipping the copy when the prior map hasn't resolved leaves the
    // stale-by-1 cached average in place, which is what we want anyway.
    const willReadSum = !this._sumPending;
    if (willReadSum) enc.copyBufferToBuffer(this._sum_buf, 0, this._sum_read, 0, 4);

    let pixelsP = null;
    if (opts.needPixels) {
      if (!this._read_buf) {
        this._readBytesPerRow = Math.ceil(W * 4 / 256) * 256;
        this._read_buf = device.createBuffer({
          size: this._readBytesPerRow * H,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          label: 'gray_read',
        });
      }
      enc.copyTextureToBuffer(
        { texture: this._tex_gray },
        { buffer: this._read_buf, bytesPerRow: this._readBytesPerRow, rowsPerImage: H },
        { width: W, height: H },
      );
    }
    device.queue.submit([enc.finish()]);

    if (opts.needPixels) {
      // Capture the buffer ref locally so a concurrent destroy() (e.g. user
      // drops a new source mid-capture) reaches an unmap on the buffer that
      // was actually mapped, and the catch swallows the AbortError instead of
      // surfacing it as a "frame capture failed".
      const buf = this._read_buf;
      pixelsP = buf.mapAsync(GPUMapMode.READ).then(() => {
        const mapped = new Uint8Array(buf.getMappedRange());
        const px = new Uint8ClampedArray(W * H);
        for (let y = 0; y < H; y++) {
          const rowOff = y * this._readBytesPerRow;
          for (let x = 0; x < W; x++) px[x + y * W] = mapped[rowOff + x * 4];
        }
        buf.unmap();
        return px;
      }).catch(() => null);
    }

    // Sum readback: kick off async, don't block. The cached value is updated
    // when mapAsync resolves. Both the copy above and this mapAsync are
    // gated on the same willReadSum check so we never trample a buffer
    // that's still mapped from the previous frame.
    if (willReadSum) {
      this._sumPending = true;
      // Same race as pixels: destroy() may invalidate _sum_read before the
      // map resolves; swallow that and clear the pending flag.
      const buf = this._sum_read;
      buf.mapAsync(GPUMapMode.READ).then(() => {
        const v = new Uint32Array(buf.getMappedRange(0, 4).slice(0))[0];
        buf.unmap();
        this._average    = v;
        this._sumPending = false;
        const r = this._averageReadyRes;
        this._averageReady = new Promise(rr => { this._averageReadyRes = rr; });
        r();
      }).catch(() => { this._sumPending = false; });
    }

    const pixels = pixelsP ? await pixelsP : null;
    const fs = this;
    return {
      W, H,
      texture: this._tex_gray,
      pixels,
      averageReady: this._averageReady,
      get average() { return fs._average; },
    };
  }

  destroy() {
    for (const k of ['_tex_src', '_tex_gray',
                     '_ep_buf', '_sum_buf', '_sum_read', '_read_buf']) {
      this[k]?.destroy?.();
      this[k] = null;
    }
  }
}
