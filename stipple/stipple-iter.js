// stipple-iter.js — GPU + CPU iterators, behind a shared interface.
//
// Both iterators expose:
//   await iter.step()             — one Lloyd iter (+ optional split/merge).
//   iter.refreshSiteIds()         — make siteIdsTex reflect current points.
//   iter.destroy()                — release GPU resources.
// And these getters: points, N, sitesBuffer, siteIdsTex, ops, maxOps,
// hasPendingWork. siteIdsTex is always a TEXTURE_BINDING r32uint texture
// suitable for the voronoi_render pipeline.
"use strict";

import { Point, seedPoints, StipplingIterator, computeVoronoi } from './stipple.js';
import { loadShaders } from './stipple-shaders.js';

const log = (...a) => globalThis.STIPPLE_DEBUG && console.log(...a);

const PARAMS_BYTES = 12 * 4;
const CELL_SLOTS   = 7;   // [acc, r, rx, ry, rxx, rxy, ryy]

function makeParamsBuffer(device, label = 'params') {
  return device.createBuffer({
    size: PARAMS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label,
  });
}

function writeParams(device, buf, p) {
  const a = new Uint32Array(12);
  a[0] = p.W; a[1] = p.H; a[2] = p.full_W; a[3] = p.full_H;
  a[4] = p.N; a[5] = p.N_max;
  a[6] = p.jfa_step; a[7] = p.jfa_min_step;
  a[8] = p.pyramid_lvl; a[9] = p.full_moments;
  device.queue.writeBuffer(buf, 0, a.buffer);
}

// Factory entrypoint used by the app.
export async function createIterator({ useCPU, device, frame, params }) {
  if (useCPU) return CPUStipplingIterator.create(device, frame, params);
  return       GPUStipplingIterator.create(device, frame, params);
}

// ── GPU iterator ─────────────────────────────────────────────────────────────
export class GPUStipplingIterator {
  static async create(device, frame, params) {
    const it = new GPUStipplingIterator(device, frame, params);
    await it._init();
    return it;
  }

  constructor(device, frame, params) {
    this._device  = device;
    this._frame   = frame;
    this._params  = params;
    this._max_ops = params.num_iters * params.num_Lloyd_iters;
    this._ops     = 0;
    this._N       = 0;
    this._N_max   = 0;
    this._points  = [];
    this._pendingSplit = null;
  }

  get points()         { return this._points; }
  get N()              { return this._N; }
  get sitesBuffer()    { return this._buf_sites; }
  get siteIdsTex()     { return this._tex_jfa_a; }
  get ops()            { return this._ops; }
  get maxOps()         { return this._max_ops; }
  get hasPendingWork() { return this._pendingSplit != null; }

  // Adopt a refreshed frame (same texture handle, same W/H — only the
  // texture *content* and pixel-sum value change). Bind groups stay valid.
  setFrame(frame) { this._frame = frame; }

  async _init() {
    const { _device: device, _frame: g, _params: p } = this;
    const { W, H }  = g;
    const N0        = p.num_points >> 1;
    const N_max     = Math.max(p.num_points * 2, 1024);
    this._N         = N0;
    this._N_max     = N_max;
    this._points    = seedPoints(N0, p.seed, g);

    const mods = await loadShaders(device);
    this._pipe_clear  = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.clear_jfa, entryPoint: 'main' },
    });
    this._pipe_seed   = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.seed_jfa, entryPoint: 'main' },
    });
    this._pipe_jfa    = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.jfa_step, entryPoint: 'main' },
    });
    this._pipe_cclear = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.clear_cells, entryPoint: 'main' },
    });
    this._pipe_accum  = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.accumulate, entryPoint: 'main' },
    });
    this._pipe_lloyd  = device.createComputePipeline({
      layout: 'auto', compute: { module: mods.lloyd_update, entryPoint: 'main' },
    });

    this._buf_params = makeParamsBuffer(device);

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
      const buf = makeParamsBuffer(device, `jfa_params_${i}`);
      writeParams(device, buf, {
        W, H, full_W: W, full_H: H,
        N: N0, N_max, jfa_step: step, jfa_min_step: 1,
        pyramid_lvl: 0, full_moments: 0,
      });
      return buf;
    });
    // Final result lives in jfa_a iff steps.length is even (we ping a→b).
    this._jfa_last_in_a = (this._jfa_steps.length & 1) === 0;

    const cellsBytes = N_max * CELL_SLOTS * 4;
    this._buf_cells      = device.createBuffer({
      size: cellsBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'cells',
    });
    this._buf_cells_read = device.createBuffer({
      size: cellsBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'cells_read',
    });
    const sitesBytes = N_max * 2 * 4;
    this._buf_sites  = device.createBuffer({
      size: sitesBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'sites',
    });
    const texDesc = {
      size: { width: W, height: H }, format: 'r32uint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    };
    this._tex_jfa_a = device.createTexture({ ...texDesc, label: 'jfa_a' });
    this._tex_jfa_b = device.createTexture({ ...texDesc, label: 'jfa_b' });

    this._buildBindGroups();
    this._uploadSites();
  }

  _buildBindGroups() {
    const device = this._device;
    const view_a = this._tex_jfa_a.createView();
    const view_b = this._tex_jfa_b.createView();
    const view_pixels = this._frame.texture.createView();

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
        { binding: 1, resource: { buffer: this._buf_sites } },
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
          { binding: 1, resource: { buffer: this._buf_sites } },
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
        { binding: 2, resource: { buffer: this._buf_sites } },
      ],
    });
  }

  _uploadSites() {
    if (this._N === 0) return;
    const arr = new Float32Array(this._N * 2);
    for (let i = 0; i < this._N; i++) {
      arr[2*i]   = this._points[i].x;
      arr[2*i+1] = this._points[i].y;
    }
    this._device.queue.writeBuffer(this._buf_sites, 0,
                                   arr.buffer, 0, arr.byteLength);
  }

  _setParams(extra = {}) {
    const { _frame: g } = this;
    writeParams(this._device, this._buf_params, {
      W: g.W, H: g.H, full_W: g.W, full_H: g.H,
      N: this._N, N_max: this._N_max,
      jfa_step: 0, jfa_min_step: 1,
      pyramid_lvl: 0, full_moments: 0, ...extra,
    });
  }

  _runJFA(enc) {
    const { W, H } = this._frame;
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

  // Re-run JFA on current sites_a. Cheap (~one iter's worth of JFA work).
  // After step(), jfa_a corresponds to the PRE-Lloyd positions and split/merge
  // may have shuffled IDs; calling this before voronoi rendering makes jfa_a
  // match the current _points / sites_a.
  refreshSiteIds() {
    this._setParams();
    const enc = this._device.createCommandEncoder({ label: 'refresh_jfa' });
    this._runJFA(enc);
    this._device.queue.submit([enc.finish()]);
  }

  async step() {
    // Apply the prior iter's pending split/merge (one-frame lag), so the GPU
    // can keep working while mapAsync resolves. The promise resolves to null
    // if the buffer was destroyed (i.e. iter was torn down mid-map).
    if (this._pendingSplit) {
      const r = await this._pendingSplit;
      this._pendingSplit = null;
      if (r) this._applySplitMerge(r.arr, r.N);
    }

    const device = this._device;
    const p      = this._params;
    const willSplit = (this._ops + 1) % p.num_Lloyd_iters === 0;
    this._setParams({ full_moments: willSplit ? 1 : 0 });

    const enc = device.createCommandEncoder({ label: `iter_${this._ops}` });
    const { W, H } = this._frame;

    this._runJFA(enc);

    {
      const pass = enc.beginComputePass({ label: 'clear_cells' });
      pass.setPipeline(this._pipe_cclear);
      pass.setBindGroup(0, this._bg_cclear);
      pass.dispatchWorkgroups(Math.ceil(this._N * CELL_SLOTS / 64));
      pass.end();
    }
    {
      const pass = enc.beginComputePass({ label: 'accumulate' });
      pass.setPipeline(this._pipe_accum);
      pass.setBindGroup(0, this._bg_accum);
      pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
      pass.end();
    }
    {
      const pass = enc.beginComputePass({ label: 'lloyd_update' });
      pass.setPipeline(this._pipe_lloyd);
      pass.setBindGroup(0, this._bg_lloyd);
      pass.dispatchWorkgroups(Math.ceil(this._N / 64));
      pass.end();
    }
    if (willSplit) {
      enc.copyBufferToBuffer(this._buf_cells, 0,
                             this._buf_cells_read, 0,
                             this._N * CELL_SLOTS * 4);
    }
    device.queue.submit([enc.finish()]);
    this._ops++;

    if (willSplit) {
      const N = this._N;
      const bytes = N * CELL_SLOTS * 4;
      const buf = this._buf_cells_read;
      this._pendingSplit = (async () => {
        try {
          await buf.mapAsync(GPUMapMode.READ, 0, bytes);
          const arr = new Float32Array(buf.getMappedRange(0, bytes).slice(0));
          buf.unmap();
          return { arr, N };
        } catch {
          // Iter was destroyed mid-map; consumer will treat null as no-op.
          return null;
        }
      })();
    }
    return false;
  }

  // `g.average` may be stale by one frame in video mode (the sum readback is
  // also deferred); this is acceptable since it only shifts split/merge
  // thresholds slightly.
  _applySplitMerge(arr, N) {
    const p = this._params, g = this._frame;
    const { W, H } = g;
    const avg_rho = Math.floor(p.rho * 255 + g.average / p.num_points);
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
      const xf    = arr[base + 2] * inv_r;
      const yf    = arr[base + 3] * inv_r;
      const rxx   = arr[base + 4] * inv_r;
      const rxy   = arr[base + 5] * inv_r;
      const ryy   = arr[base + 6] * inv_r;

      const rho   = Math.round(r_acc);
      if (rho < Tl) continue;
      const color = r_acc / acc;

      if (acc > 1 && rho > Tu && new_pts.length + 2 <= Nmax) {
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
    for (const k of ['_buf_params','_buf_sites','_buf_cells','_buf_cells_read'])
      this[k]?.destroy?.();
    for (const k of ['_tex_jfa_a','_tex_jfa_b'])
      this[k]?.destroy?.();
    for (const b of this._buf_jfa_params ?? []) b.destroy();
  }
}

// ── CPU iterator (drives stipple.js, uploads results for the GPU renderer) ──
// Same interface as GPUStipplingIterator. Used for A/B comparison.
export class CPUStipplingIterator {
  static async create(device, frame, params) {
    if (!frame.pixels) {
      throw new Error('CPU iterator needs a frame captured with needPixels:true');
    }
    const it = new CPUStipplingIterator();
    it._device = device;
    it._frame  = frame;
    it._params = params;
    it._N_max  = Math.max(params.num_points * 2, 1024);

    const seeded = seedPoints(params.num_points >> 1, params.seed, frame);
    it._inner    = new StipplingIterator(frame, params, seeded);

    it._buf_sites = device.createBuffer({
      size: it._N_max * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'cpu_sites',
    });
    it._tex_jfa = device.createTexture({
      size: { width: frame.W, height: frame.H }, format: 'r32uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      label: 'cpu_site_ids',
    });
    it._uploadSites();
    return it;
  }

  _uploadSites() {
    const pts = this._inner.points;
    if (pts.length === 0) return;
    const arr = new Float32Array(pts.length * 2);
    for (let i = 0; i < pts.length; i++) {
      arr[2*i]   = pts[i].x;
      arr[2*i+1] = pts[i].y;
    }
    this._device.queue.writeBuffer(this._buf_sites, 0,
                                   arr.buffer, 0, arr.byteLength);
  }

  async step() {
    // Inner caps at `done` after num_iters*num_Lloyd_iters ops — right for
    // image mode, wrong for video where App keeps asking for steps. Rewind
    // one Lloyd cycle so we keep adapting to new frames; matches GPU's
    // unbounded loop. The 4-step cadence (Lloyd, Lloyd, Lloyd, Lloyd+split)
    // is preserved because rewinding to max_ops - num_Lloyd_iters lines the
    // next split/merge up on the same ops % num_Lloyd_iters == 0 boundary.
    const inner = this._inner;
    if (inner.done) {
      inner._ops = inner._max_ops - inner._params.num_Lloyd_iters;
    }
    inner.step();
    this._uploadSites();
    return false;
  }

  refreshSiteIds() {
    const { W, H } = this._frame;
    const out = computeVoronoi(W, H, this._inner.points);
    this._device.queue.writeTexture(
      { texture: this._tex_jfa },
      out.buffer, { bytesPerRow: W * 4, rowsPerImage: H },
      { width: W, height: H },
    );
  }

  // Same contract as GPUStipplingIterator.setFrame, but also forwards the
  // refreshed pixel array into the inner CPU iterator so the next step()
  // sees the new frame.
  setFrame(frame) {
    this._frame = frame;
    this._inner._frame = frame;
  }

  get points()         { return this._inner.points; }
  get N()              { return this._inner.points.length; }
  get sitesBuffer()    { return this._buf_sites; }
  get siteIdsTex()     { return this._tex_jfa; }
  get ops()            { return this._inner.ops; }
  get maxOps()         { return this._inner.maxOps; }
  get hasPendingWork() { return false; }

  destroy() {
    this._buf_sites?.destroy?.();
    this._tex_jfa?.destroy?.();
  }
}
