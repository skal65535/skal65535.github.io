// stipple-gpu.js — WebGPU stippling pipeline (compute + render).
//
// Exports:
//   initGPU(canvas)             → { device, context, format }
//   GPUStipplingIterator        — compute iterator (JFA Voronoi + Lloyd)
//   GPURenderer                 — points / target / voronoi rendering
//   grabFrameGPU(...)           — source → grayscale texture (+ CPU pixels)
//
// Phase 1: scaffold + JFA Voronoi + render path. Lloyd update runs on CPU
// (the existing stipple.js implementation) until Phase 3.
"use strict";

import { Point, StipplingIterator, computeVoronoi } from './stipple.js';

const log = (...a) => globalThis.STIPPLE_DEBUG && console.log(...a);

// 10 u32 params + 2 u32 pad → 48 bytes (mirrors WGSL Params struct).
const PARAMS_BYTES = 12 * 4;
// Per-cell atomic accumulator slots: [acc, r, rx, ry, rxx, rxy, ryy].
const CELL_SLOTS = 7;

// ── Device / context init ─────────────────────────────────────────────────────
export async function initGPU(canvas) {
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter');
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Could not get WebGPU canvas context');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });
  return { device, context, format };
}

// ── Shader loading ────────────────────────────────────────────────────────────
function splitShaderSource(src) {
  const out = {};
  const re = /\/\/\s*===\s*([a-zA-Z0-9_]+)\s*===\s*\n/g;
  const parts = src.split(re);
  for (let i = 1; i < parts.length; i += 2) out[parts[i]] = parts[i + 1];
  return out;
}

let _shaderCache = null;
async function loadShaders(device) {
  if (_shaderCache && _shaderCache.device === device) return _shaderCache.mods;
  const url = new URL('./stipple-shaders.wgsl', import.meta.url);
  const src = await (await fetch(url)).text();
  const map = splitShaderSource(src);
  const mods = {};
  for (const [name, code] of Object.entries(map)) {
    mods[name] = device.createShaderModule({ code, label: name });
  }
  _shaderCache = { device, mods };
  return mods;
}

// ── Params buffer helpers ─────────────────────────────────────────────────────
function makeParamsBuffer(device, bytes = PARAMS_BYTES, label = 'params') {
  return device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label,
  });
}

function writeIterParams(device, buf, p) {
  const a = new Uint32Array(12);
  a[0] = p.W; a[1] = p.H; a[2] = p.full_W; a[3] = p.full_H;
  a[4] = p.N; a[5] = p.N_max;
  a[6] = p.jfa_step; a[7] = p.jfa_min_step;
  a[8] = p.pyramid_lvl; a[9] = p.full_moments;
  device.queue.writeBuffer(buf, 0, a.buffer);
}

// ── grabFrameGPU ──────────────────────────────────────────────────────────────
// Captures a frame from <img>/<video>/HTMLCanvasElement into a grayscale
// rgba8unorm texture via copyExternalImageToTexture + edge-detect compute.
// The edge-detect (or luma) kernel also reduces the grayscale sum into a
// single u32 atomic; that 4-byte value is read back asynchronously and
// exposed via grays.average (live getter). No per-pixel CPU readback.
//
// Returns { W, H, texture, get average(), ready }. `ready` resolves the
// next time the image-sum readback completes (use it on first frame so
// split/merge sees a real average rather than 0).
let _grab           = null;
let _grabAverage    = 0;
let _grabMapPending = false;
let _grabReadyRes   = null;
let _grabReady      = new Promise(r => { _grabReadyRes = r; });

export async function grabFrameGPU(device, src, opts) {
  const W = opts.W, H = opts.H;
  const useEdges = opts.use_edges !== false;

  if (!_grab || _grab.device !== device || _grab.W !== W || _grab.H !== H) {
    const mods = await loadShaders(device);
    const tex_src = device.createTexture({
      size: { width: W, height: H }, format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
             GPUTextureUsage.RENDER_ATTACHMENT,
      label: 'src',
    });
    const tex_gray = device.createTexture({
      size: { width: W, height: H }, format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING |
             GPUTextureUsage.COPY_SRC,   // COPY_SRC for the CPU-mode readback
      label: 'gray',
    });
    const sampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });
    const ep_buf = device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'edge_params',
    });
    const pipe_edge = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.edge_detect, entryPoint: 'main' },
    });
    const pipe_luma = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.luma, entryPoint: 'main' },
    });
    const sum_buf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
             GPUBufferUsage.COPY_DST,
      label: 'image_sum',
    });
    const sum_read = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'image_sum_read',
    });
    // copyExternalImageToTexture does NOT scale — it crops to copySize. To
    // fit any-sized source into the W×H working texture we first draw it onto
    // a 2D scratch canvas at exactly W×H. Reused across frames.
    const scratch = document.createElement('canvas');
    scratch.width = W; scratch.height = H;
    const scratchCtx = scratch.getContext('2d');
    _grab = {
      device, W, H, tex_src, tex_gray, sampler, ep_buf,
      pipe_edge, pipe_luma, sum_buf, sum_read, scratch, scratchCtx,
    };
    // Cache rebuilt → reset average tracking.
    _grabAverage    = 0;
    _grabMapPending = false;
    _grabReady      = new Promise(r => { _grabReadyRes = r; });
  }
  const G = _grab;

  G.scratchCtx.clearRect(0, 0, W, H);
  G.scratchCtx.drawImage(src, 0, 0, W, H);
  device.queue.copyExternalImageToTexture(
    { source: G.scratch }, { texture: G.tex_src }, { width: W, height: H },
  );

  const ep = new ArrayBuffer(32);
  const u = new Uint32Array(ep);   const f = new Float32Array(ep);
  u[0] = W; u[1] = H;
  f[2] = opts.alpha; f[3] = 0.25 * (1.0 - opts.alpha);
  f[4] = opts.strength;
  device.queue.writeBuffer(G.ep_buf, 0, ep);

  // Zero the image-sum atomic before this kernel run.
  device.queue.writeBuffer(G.sum_buf, 0, new Uint32Array([0]));

  const enc = device.createCommandEncoder({ label: 'grab' });
  const pipe = useEdges ? G.pipe_edge : G.pipe_luma;
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: G.ep_buf } },
      { binding: 1, resource: G.tex_src.createView() },
      { binding: 2, resource: G.sampler },
      { binding: 3, resource: G.tex_gray.createView() },
      { binding: 4, resource: { buffer: G.sum_buf } },
    ],
  });
  const pass = enc.beginComputePass({ label: 'edge' });
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
  pass.end();
  enc.copyBufferToBuffer(G.sum_buf, 0, G.sum_read, 0, 4);

  // Optional pixel readback for the CPU iterator path. Lazily allocate the
  // staging buffer; only blocks when needPixels is set.
  let pixelsP = null;
  if (opts.needPixels) {
    if (!G.read_buf) {
      G.bytesPerRow = Math.ceil(W * 4 / 256) * 256;
      G.read_buf = device.createBuffer({
        size: G.bytesPerRow * H,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: 'gray_read',
      });
    }
    enc.copyTextureToBuffer(
      { texture: G.tex_gray },
      { buffer: G.read_buf, bytesPerRow: G.bytesPerRow, rowsPerImage: H },
      { width: W, height: H },
    );
  }
  device.queue.submit([enc.finish()]);

  if (opts.needPixels) {
    pixelsP = G.read_buf.mapAsync(GPUMapMode.READ).then(() => {
      const mapped = new Uint8Array(G.read_buf.getMappedRange());
      const px = new Uint8ClampedArray(W * H);
      for (let y = 0; y < H; y++) {
        const rowOff = y * G.bytesPerRow;
        for (let x = 0; x < W; x++) px[x + y * W] = mapped[rowOff + x * 4];
      }
      G.read_buf.unmap();
      return px;
    });
  }

  // Kick off a non-blocking 4-byte sum readback. Skip if previous is still
  // in flight (the previous-frame average is fine for next-frame split/merge).
  if (!_grabMapPending) {
    _grabMapPending = true;
    G.sum_read.mapAsync(GPUMapMode.READ).then(() => {
      const v = new Uint32Array(G.sum_read.getMappedRange(0, 4).slice(0))[0];
      G.sum_read.unmap();
      _grabAverage = v;
      _grabMapPending = false;
      const r = _grabReadyRes;
      _grabReady = new Promise(rr => { _grabReadyRes = rr; });
      r();
    });
  }

  // Resolve pixels (when requested) before returning. The 1MB readback +
  // 1M-iter unpack add ~30 ms; only the CPU iterator path needs them.
  const pixels = pixelsP ? await pixelsP : null;

  return {
    W, H,
    texture: G.tex_gray,
    pixels,
    get average() { return _grabAverage; },
    ready: _grabReady,
  };
}

// ── Iterator ──────────────────────────────────────────────────────────────────
export class GPUStipplingIterator {
  static async create(device, grays, params, existingPoints = null) {
    const it = new GPUStipplingIterator(device, grays, params, existingPoints);
    await it._pendingInit;
    return it;
  }

  constructor(device, grays, params, existingPoints) {
    this._device  = device;
    this._grays   = grays;
    this._params  = params;
    this._max_ops = params.num_iters * params.num_Lloyd_iters;
    this._ops     = 0;
    this._ready   = false;
    this._pendingInit = this._init(existingPoints);
  }

  get points()      { return this._points; }
  // Continuous mode: never "done" intrinsically. The tick loop applies an
  // iter cap externally for static images so they don't burn rAF cycles
  // after convergence.
  get done()        { return false; }
  get ops()         { return this._ops; }
  get maxOps()      { return this._max_ops; }
  get siteIdsTex()  { return this._tex_jfa_a; }
  get sitesBuffer() { return this._buf_sites_a; }
  get N()           { return this._N; }
  get pixelsTex()   { return this._grays.texture; }

  async _init(existingPoints) {
    const { _device: device, _grays: g, _params: p } = this;
    const { W, H } = g;

    let seed = p.seed ?? 91651088029;
    const rand = () => {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
    const N0 = existingPoints ? existingPoints.length : (p.num_points >> 1);
    const N_max = Math.max(p.num_points * 2, 1024);
    this._N = N0;
    this._N_max = N_max;
    // grays.pixels is only populated when the CPU iterator is also in play
    // (see grabFrameGPU's needPixels). When available we seed initial point
    // colors from the source image so an immediate show_voronoi render isn't
    // all-black before the first Lloyd pass fills colors in.
    const pix = g.pixels;
    this._points = existingPoints
      ? existingPoints.map(pt => new Point(pt.x, pt.y, pt.c))
      : Array.from({length: N0}, () => {
          const x = rand(), y = rand();
          const c = pix
            ? pix[Math.min(W-1, (x*W)|0) + Math.min(H-1, (y*H)|0) * W]
            : 0;
          return new Point(x, y, c);
        });

    const mods = await loadShaders(device);
    this._pipe_clear = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.clear_jfa, entryPoint: 'main' },
    });
    this._pipe_seed = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.seed_jfa, entryPoint: 'main' },
    });
    this._pipe_jfa = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.jfa_step, entryPoint: 'main' },
    });
    this._pipe_cclear = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.clear_cells, entryPoint: 'main' },
    });
    this._pipe_accum = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.accumulate, entryPoint: 'main' },
    });
    this._pipe_lloyd = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.lloyd_update, entryPoint: 'main' },
    });

    this._buf_params  = makeParamsBuffer(device);
    // JFA step sequence is (W,H)-determined; pre-compute and pre-allocate a
    // params buffer per pass (each holds a distinct jfa_step).
    this._jfa_steps = (() => {
      const a = [];
      let s = Math.max(1, Math.floor(Math.max(W, H) / 2));
      while (true) { a.push(s); if (s === 1) break; s = Math.max(1, s >> 1); }
      a.push(1);  // JFA+1 boundary cleanup pass.
      return a;
    })();
    this._buf_jfa_params = this._jfa_steps.map((step, i) => {
      const buf = makeParamsBuffer(device, PARAMS_BYTES, `jfa_params_${i}`);
      writeIterParams(device, buf, {
        W, H, full_W: W, full_H: H,
        N: this._N, N_max,
        jfa_step: step, jfa_min_step: 1,
        pyramid_lvl: 0, full_moments: 0,
      });
      return buf;
    });
    // After all passes, the result lives in jfa_a iff steps.length is even
    // (we start src=a→dst=b). Odd → final write went to jfa_b; copy back.
    this._jfa_last_in_a = (this._jfa_steps.length & 1) === 0;

    const cellsBytes = N_max * CELL_SLOTS * 4;
    this._buf_cells   = device.createBuffer({
      size: cellsBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
             GPUBufferUsage.COPY_SRC,
      label: 'cells',
    });
    this._buf_cells_read = device.createBuffer({
      size: cellsBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'cells_read',
    });
    const sitesBytes  = N_max * 2 * 4;
    this._buf_sites_a = device.createBuffer({
      size: sitesBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
             GPUBufferUsage.COPY_SRC,
      label: 'sites_a',
    });
    const texDesc = {
      size: { width: W, height: H }, format: 'r32uint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    };
    this._tex_jfa_a = device.createTexture({ ...texDesc, label: 'jfa_a' });
    this._tex_jfa_b = device.createTexture({ ...texDesc, label: 'jfa_b' });

    // Bind groups are stable across iters: buffer/texture handles don't
    // change (only buffer *contents* do, which doesn't invalidate the BG).
    // Building once cuts ~14 BG allocations per iter.
    this._buildBindGroups();

    this._pendingSplit = null;  // deferred split/merge readback (1-frame lag).

    this._uploadSites();
    this._ready = true;
  }

  _buildBindGroups() {
    const device = this._device;
    const view_a = this._tex_jfa_a.createView();
    const view_b = this._tex_jfa_b.createView();
    const view_pixels = this._grays.texture.createView();

    this._bg_clear = device.createBindGroup({
      layout: this._pipe_clear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buf_params } },
        { binding: 1, resource: view_a },
      ],
    });
    this._bg_seed = device.createBindGroup({
      layout: this._pipe_seed.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buf_params } },
        { binding: 1, resource: { buffer: this._buf_sites_a } },
        { binding: 2, resource: view_a },
      ],
    });
    this._bg_jfa = this._jfa_steps.map((_, i) => {
      // Even passes ping a→b; odd b→a.
      const src = (i & 1) ? view_b : view_a;
      const dst = (i & 1) ? view_a : view_b;
      return device.createBindGroup({
        layout: this._pipe_jfa.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buf_jfa_params[i] } },
          { binding: 1, resource: { buffer: this._buf_sites_a } },
          { binding: 2, resource: src },
          { binding: 3, resource: dst },
        ],
      });
    });
    this._bg_cclear = device.createBindGroup({
      layout: this._pipe_cclear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buf_params } },
        { binding: 1, resource: { buffer: this._buf_cells } },
      ],
    });
    this._bg_accum = device.createBindGroup({
      layout: this._pipe_accum.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buf_params } },
        { binding: 1, resource: view_pixels },
        { binding: 2, resource: view_a },
        { binding: 3, resource: { buffer: this._buf_cells } },
      ],
    });
    this._bg_lloyd = device.createBindGroup({
      layout: this._pipe_lloyd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buf_params } },
        { binding: 1, resource: { buffer: this._buf_cells } },
        { binding: 2, resource: { buffer: this._buf_sites_a } },
      ],
    });
  }

  _uploadSites() {
    const arr = new Float32Array(this._N * 2);
    for (let i = 0; i < this._N; i++) {
      arr[2*i] = this._points[i].x;
      arr[2*i+1] = this._points[i].y;
    }
    this._device.queue.writeBuffer(this._buf_sites_a, 0,
                                   arr.buffer, 0, arr.byteLength);
  }

  _setParams(extra = {}) {
    const { _grays: g, _N: N, _N_max: N_max } = this;
    writeIterParams(this._device, this._buf_params, {
      W: g.W, H: g.H, full_W: g.W, full_H: g.H,
      N, N_max, jfa_step: 0, jfa_min_step: 1,
      pyramid_lvl: 0, full_moments: 0, ...extra,
    });
  }

  runJFA(enc) {
    const { W, H } = this._grays;
    {
      const pass = enc.beginComputePass({ label: 'jfa_clear' });
      pass.setPipeline(this._pipe_clear);
      pass.setBindGroup(0, this._bg_clear);
      pass.dispatchWorkgroups(Math.ceil(W * H / 64));
      pass.end();
    }
    {
      const pass = enc.beginComputePass({ label: 'jfa_seed' });
      pass.setPipeline(this._pipe_seed);
      pass.setBindGroup(0, this._bg_seed);
      pass.dispatchWorkgroups(Math.ceil(this._N / 64));
      pass.end();
    }
    // Ping-pong jfa_a ↔ jfa_b through the pre-baked step sequence
    // (W/2, W/4, …, 2, 1, 1). The trailing extra step=1 is "JFA+1" — it
    // cleans up the boundary errors that vanilla JFA leaves at cell borders;
    // under weighted Lloyd those errors compound and drift centroids.
    for (let i = 0; i < this._jfa_steps.length; i++) {
      const pass = enc.beginComputePass({ label: `jfa_${this._jfa_steps[i]}` });
      pass.setPipeline(this._pipe_jfa);
      pass.setBindGroup(0, this._bg_jfa[i]);
      pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
      pass.end();
    }
    if (!this._jfa_last_in_a) {
      enc.copyTextureToTexture(
        { texture: this._tex_jfa_b }, { texture: this._tex_jfa_a },
        { width: W, height: H });
    }
  }

  // Re-run JFA on the current sites_a. Cheap (~one iter's worth of JFA work)
  // and useful for display: after step(), jfa_a corresponds to the PRE-Lloyd
  // positions, and split/merge may have shuffled IDs. Calling this before
  // renderVoronoi makes jfa_a's site IDs match the current _points / sites_a.
  refreshJFA() {
    if (!this._ready) return;
    this._setParams();
    const enc = this._device.createCommandEncoder({ label: 'refresh_jfa' });
    this.runJFA(enc);
    this._device.queue.submit([enc.finish()]);
  }

  // Whether a deferred split/merge readback is still in flight. Callers (the
  // HTML tick loop) use this to keep scheduling rAFs past the iter budget so
  // the final pending split actually lands on screen.
  get hasPendingSplit() { return this._pendingSplit != null; }

  async step() {
    if (!this._ready) await this._pendingInit;

    // Apply the prior iter's pending split/merge (one-frame lag), so the GPU
    // could keep working while mapAsync resolved.
    if (this._pendingSplit) {
      const { arr, N } = await this._pendingSplit;
      this._pendingSplit = null;
      this._applySplitMerge(arr, N);
    }

    const device = this._device;
    const p = this._params;
    // Split/merge runs every num_Lloyd_iters iters. Continuous mode keeps
    // calling it indefinitely; the hysteresis ramp saturates so steady-state
    // churn is bounded.
    const willSplit = (this._ops + 1) % p.num_Lloyd_iters === 0;
    this._setParams({ full_moments: willSplit ? 1 : 0 });

    const enc = device.createCommandEncoder({ label: `iter_${this._ops}` });
    const { W, H } = this._grays;

    // 1. JFA Voronoi → site_ids in _tex_jfa_a.
    this.runJFA(enc);

    // 2. Clear per-cell accumulators.
    {
      const pass = enc.beginComputePass({ label: 'clear_cells' });
      pass.setPipeline(this._pipe_cclear);
      pass.setBindGroup(0, this._bg_cclear);
      pass.dispatchWorkgroups(Math.ceil(this._N * CELL_SLOTS / 64));
      pass.end();
    }

    // 3. Accumulate per-pixel into cells via f32-via-u32 CAS-loop.
    {
      const pass = enc.beginComputePass({ label: 'accumulate' });
      pass.setPipeline(this._pipe_accum);
      pass.setBindGroup(0, this._bg_accum);
      pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
      pass.end();
    }

    // 4. Lloyd update: sites_a[i] ← (rx, ry) / r.
    {
      const pass = enc.beginComputePass({ label: 'lloyd_update' });
      pass.setPipeline(this._pipe_lloyd);
      pass.setBindGroup(0, this._bg_lloyd);
      pass.dispatchWorkgroups(Math.ceil(this._N / 64));
      pass.end();
    }

    // 5. If split/merge frame, copy cells to readback buffer.
    if (willSplit) {
      enc.copyBufferToBuffer(this._buf_cells, 0,
                             this._buf_cells_read, 0,
                             this._N * CELL_SLOTS * 4);
    }

    device.queue.submit([enc.finish()]);
    this._ops++;

    // Kick the readback without awaiting — the result is consumed at the
    // top of the *next* step(). Frees the GPU to start the next iter while
    // mapAsync resolves on the JS side.
    if (willSplit) {
      const N = this._N;
      const bytes = N * CELL_SLOTS * 4;
      this._pendingSplit = (async () => {
        await this._buf_cells_read.mapAsync(GPUMapMode.READ, 0, bytes);
        const arr = new Float32Array(
          this._buf_cells_read.getMappedRange(0, bytes).slice(0));
        this._buf_cells_read.unmap();
        return { arr, N };
      })();
    }
    return false;
  }

  // Applies a readback to produce new _points + sites_a upload.
  // `g.average` may be stale by one frame in video mode (the sum readback is
  // also deferred — see grabFrameGPU); this is acceptable since it only
  // shifts the split/merge thresholds slightly.
  _applySplitMerge(arr, N) {
    const p = this._params, g = this._grays;
    const { W, H } = g;
    const avg_rho = Math.floor(p.rho * 255 + g.average / p.num_points);
    // Hysteresis ramps up over the first _max_ops iters, then saturates.
    // The ramp gives initial convergence room; saturation bounds steady-state
    // churn in continuous mode.
    const hyst    = Math.min(0.61, 0.01 + 0.6 * this._ops / this._max_ops);
    const Tu      = Math.ceil((1 + hyst) * avg_rho);
    const Tl      = Math.floor((1 - hyst) * avg_rho);
    const eps_W   = 1 / W, eps_H = 1 / H;
    const inUnit  = v => v >= 0 && v <= 1;
    const Nmax    = this._N_max;
    const new_pts = [];

    for (let i = 0; i < N; i++) {
      const base  = i * CELL_SLOTS;
      const acc   = arr[base + 0];
      if (acc < 1) continue;
      const r_acc = arr[base + 1];
      if (r_acc <= 0) continue;

      const inv_r = 1 / r_acc;
      const xf    = arr[base + 2] * inv_r;    // r-weighted centroid x
      const yf    = arr[base + 3] * inv_r;
      const rxx   = arr[base + 4] * inv_r;    // <r·x²>/<r>
      const rxy   = arr[base + 5] * inv_r;
      const ryy   = arr[base + 6] * inv_r;

      const rho   = Math.round(r_acc);
      if (rho < Tl) continue;
      const color = r_acc / acc;

      if (acc > 1 && rho > Tu && new_pts.length + 2 <= Nmax) {
        // r-weighted variance/covariance — all moments are r-weighted, so
        // subtracting (rx/r)² (NOT an unweighted mean) is the coherent form.
        const exx = rxx - xf * xf;
        const eyy = ryy - yf * yf;
        const num = rxy - xf * yf;
        const den = exx - eyy;
        const t   = (num*num + den*den > 0) ? 0.5 * Math.atan2(2*num, den) : 0;
        const radius = 0.5 * Math.sqrt(acc / Math.PI);
        const dx = radius * Math.cos(t);
        const dy = radius * Math.sin(t);
        const x1 = xf - dx, y1 = yf - dy;
        const x2 = xf + dx, y2 = yf + dy;
        if ((Math.abs(dx) > eps_W || Math.abs(dy) > eps_H) &&
            inUnit(x1) && inUnit(y1) && inUnit(x2) && inUnit(y2)) {
          new_pts.push(new Point(x1, y1, color));
          new_pts.push(new Point(x2, y2, color));
          continue;
        }
      }
      if (new_pts.length < Nmax) new_pts.push(new Point(xf, yf, color));
    }
    log(`iter #${this._ops}/${this._max_ops}: T=[${Tl},${Tu}] pts:${N}=>${new_pts.length}`);

    this._points = new_pts;
    this._N      = new_pts.length;
    this._uploadSites();
  }

  destroy() {
    for (const k of ['_buf_params','_buf_sites_a','_buf_cells','_buf_cells_read'])
      this[k]?.destroy?.();
    for (const k of ['_tex_jfa_a','_tex_jfa_b'])
      this[k]?.destroy?.();
    for (const b of this._buf_jfa_params ?? []) b.destroy();
  }
}

// ── CPU adapter ───────────────────────────────────────────────────────────────
// Wraps stipple.js's StipplingIterator (CPU compute) but uploads positions to
// a GPU buffer + site_ids texture so the WebGPU renderer can display them.
// Exposes the same surface as GPUStipplingIterator. Used for side-by-side
// CPU/GPU comparison.
export class CPUStipplingIteratorAdapter {
  static async create(device, grays, params, existingPoints = null) {
    if (!grays.pixels) {
      throw new Error('CPU iterator needs grays.pixels; pass needPixels:true to grabFrameGPU');
    }
    const it = new CPUStipplingIteratorAdapter();
    it._device  = device;
    it._params  = params;
    it._grays   = grays;
    it._iter    = new StipplingIterator(grays, params, existingPoints);
    it._N_max   = Math.max(params.num_points * 2, 1024);
    it._buf_sites = device.createBuffer({
      size: it._N_max * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'cpu_sites',
    });
    it._tex_jfa = device.createTexture({
      size: { width: grays.W, height: grays.H }, format: 'r32uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      label: 'cpu_site_ids',
    });
    it._uploadSites();
    return it;
  }

  _uploadSites() {
    const pts = this._iter.points;
    if (pts.length === 0) return;
    const arr = new Float32Array(pts.length * 2);
    for (let i = 0; i < pts.length; i++) {
      arr[2*i] = pts[i].x;
      arr[2*i+1] = pts[i].y;
    }
    this._device.queue.writeBuffer(this._buf_sites, 0,
                                   arr.buffer, 0, arr.byteLength);
  }

  async step() {
    if (this._iter.done) return false;
    this._iter.step();
    this._uploadSites();
    return false;
  }

  // For show_voronoi: recompute the CPU Voronoi and upload to a r32uint
  // texture so the WebGPU voronoi_render shader can read it. Slow (~tens of
  // ms at 1024²) but only runs when the user toggles show_voronoi.
  refreshJFA() {
    const { W, H } = this._grays;
    const out = computeVoronoi(W, H, this._iter.points);
    this._device.queue.writeTexture(
      { texture: this._tex_jfa },
      out.buffer, { bytesPerRow: W * 4 },
      { width: W, height: H },
    );
  }

  get points()           { return this._iter.points; }
  get done()             { return false; }
  get sitesBuffer()      { return this._buf_sites; }
  get N()                { return this._iter.points.length; }
  get siteIdsTex()       { return this._tex_jfa; }
  get ops()              { return this._iter._ops; }
  get maxOps()           { return this._iter._max_ops; }
  get hasPendingSplit()  { return false; }

  destroy() {
    this._buf_sites?.destroy?.();
    this._tex_jfa?.destroy?.();
  }
}

// ── Renderer ──────────────────────────────────────────────────────────────────
export class GPURenderer {
  static async create(device, context, format) {
    const r = new GPURenderer();
    await r._init(device, context, format);
    return r;
  }

  async _init(device, context, format) {
    this._device  = device;
    this._context = context;
    this._format  = format;
    const mods = await loadShaders(device);

    this._pipe_points = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.points_render, entryPoint: 'vs' },
      fragment: { module: mods.points_render, entryPoint: 'fs',
                  targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
    });
    this._pipe_target = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.target_render, entryPoint: 'vs' },
      fragment: { module: mods.target_render, entryPoint: 'fs',
                  targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
    });
    this._pipe_voro = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.voronoi_render, entryPoint: 'vs' },
      fragment: { module: mods.voronoi_render, entryPoint: 'fs',
                  targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
    });

    this._pp_buf = device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'point_params',
    });
    this._sampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });
    this._vp_buf = device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'voro_params',
    });
    this._color_buf = null;
    this._color_capacity = 0;
  }

  _bgColor(invert) { return invert ? { r:0,g:0,b:0,a:1 } : { r:1,g:1,b:1,a:1 }; }

  renderClear({ invert }) {
    const device = this._device;
    const enc = device.createCommandEncoder({ label: 'render_clear' });
    const view = this._context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: this._bgColor(invert),
      }],
    });
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  renderPoints({ sitesBuf, N, canvas, radius, invert }) {
    if (!sitesBuf || N <= 0) { this.renderClear({ invert }); return; }
    const device = this._device;
    const pp = new ArrayBuffer(32);
    const f = new Float32Array(pp);
    f[0] = radius;
    f[1] = canvas.width;
    f[2] = canvas.height;
    f[3] = invert ? 1 : 0;
    device.queue.writeBuffer(this._pp_buf, 0, pp);

    const bg = device.createBindGroup({
      layout: this._pipe_points.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._pp_buf } },
        { binding: 1, resource: { buffer: sitesBuf } },
      ],
    });
    const enc = device.createCommandEncoder({ label: 'render_points' });
    const view = this._context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: this._bgColor(invert),
      }],
    });
    pass.setPipeline(this._pipe_points);
    pass.setBindGroup(0, bg);
    pass.draw(4, N);  // 4 verts per point, N instances
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  renderTarget({ grayTex }) {
    const device = this._device;
    const bg = device.createBindGroup({
      layout: this._pipe_target.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: grayTex.createView() },
        { binding: 1, resource: this._sampler },
      ],
    });
    const enc = device.createCommandEncoder({ label: 'render_target' });
    const view = this._context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: { r:0, g:0, b:0, a:1 },
      }],
    });
    pass.setPipeline(this._pipe_target);
    pass.setBindGroup(0, bg);
    pass.draw(4);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  renderVoronoi({ siteIdsTex, points, scale, W, H }) {
    const device = this._device;
    if (this._color_capacity < points.length) {
      this._color_buf?.destroy?.();
      this._color_buf = device.createBuffer({
        size: Math.max(1024, points.length * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'site_color',
      });
      this._color_capacity = points.length;
    }
    const colors = new Float32Array(points.length);
    for (let i = 0; i < points.length; i++) colors[i] = (points[i].c ?? 0) / 255;
    device.queue.writeBuffer(this._color_buf, 0, colors.buffer, 0,
                             colors.byteLength);
    const vp = new ArrayBuffer(16);
    const u = new Uint32Array(vp); const f = new Float32Array(vp);
    u[0] = W; u[1] = H; f[2] = scale;
    device.queue.writeBuffer(this._vp_buf, 0, vp);

    const bg = device.createBindGroup({
      layout: this._pipe_voro.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: siteIdsTex.createView() },
        { binding: 1, resource: { buffer: this._color_buf } },
        { binding: 2, resource: { buffer: this._vp_buf } },
      ],
    });
    const enc = device.createCommandEncoder({ label: 'render_voronoi' });
    const view = this._context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: { r:0, g:0, b:0, a:1 },
      }],
    });
    pass.setPipeline(this._pipe_voro);
    pass.setBindGroup(0, bg);
    pass.draw(4);
    pass.end();
    device.queue.submit([enc.finish()]);
  }
}
