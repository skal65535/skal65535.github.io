// trainer.js — Hybrid GPU/CPU 3DGS optimizer: GPU inner loop (project -> tile forward ->
// tile backward + WLBG moments -> Adam / Monte-Carlo / Simulated Annealing update), CPU outer
// loop every `rebalance_every` steps (prune / grow / floaters / octree rebuild, see octree.js).
"use strict";

import { SPLAT_FLOATS } from './colmap.js';
import { MOMENT_FLOATS } from './octree.js';
import { loadShaders } from './render.js';
import { dispatch } from './gpu.js';
import { annealStep, remapAnnealState } from './anneal.js';
import { residualDensity, stippleSites, buildViewDepthGrid, makeSeedSplats } from './stipple.js';
import { makeRegionGrid, regionStats, RegionTracker } from './regions.js';
import { rgbPsnr, rgbPsnrDown, rgbSsim } from './metrics.js';
import { fuseTrainer, snapToSurface } from './mesh.js';

const UNIFORM_BYTES = 176;     // TrainUniforms (shaders.wgsl): 44 x f32
const HISTORY_LEN   = 300;     // training-PSNR history kept for the telemetry plot
const OPT_METHODS   = { adam: 0, monte_carlo: 1, sim_anneal: 2 };
const BUF_RW   = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const BUF_READ = () => GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

// Defaults of the params a training step reads (the app's params start from these)
export const TRAIN_DEFAULTS = {
  // Optimizer extras: Monte-Carlo (3DGS-MCMC: noise, L1 terms, relocation), simulated annealing T0, cooling and restarts
  mc_noise:        1e6,    // lambda_noise: SGLD noise on the nearly transparent splats' centers, x lr_pos x covariance (3DGS-MCMC: 5e5;
                           // 1e6 the best of 5e5 / 1e6 / 2e6 here, DESIGN.md §5.4)
  mc_opacity_reg:  0.01,   // lambda_o: mean |opacity| added to the loss
  mc_scale_reg:    0.01,   // lambda_s: mean |scale| (per axis)
  mc_relocate:     'opacity',  // the live splats the dead ones move onto, drawn ~ 'opacity' or 'error'
  mc_relocate_mode: 'copy',   // ... as copies in place (3DGS-MCMC) or 'shape': each one drawn splits within its shape (brush)
  sa_temp:         0.15,
  sa_cooling:      0.96,
  sa_share:        0.25,   // the share of the visible splats given a new jump at each rebalance
  sa_size:         1,      // a jump's standard deviation, in the splat's own scales, times T
  sa_reheat:       false,  // back to sa_temp every 10 K steps (at least 1000)
  // Learning rates
  lr_pos:          0.00005,  // x the scene radius (3DGS: 1.6e-4; 5e-5 with split_mode shape: better PSNR and depth on jardin5 / 7, DESIGN.md E9)
  lr_pos_final:    0.01,   // lr_pos decays exponentially to lr_pos * lr_pos_final ...
  lr_decay_steps:  30000,  // ... over this many steps
  lr_scale:        0.005,
  lr_quat:         0.002,
  lr_color:        0.0025,
  lr_opacity:      0.01,   // brush's; 0.05 floated splats under the dodecahedron's floor and streaked it (DESIGN.md E9)
  ssim:            0,      // D-SSIM loss weight (0 = off: 0.8 L1 + 0.2 L2), see tile_backward
  drop_rate:       0,      // DropGaussian: share of splats skipped in each training render (0 = off)
  split_min_views: 0,      // growth splits only splats seen from at least this many views (N_eff; 0 = any)
  split_min_err:   0,      // ... and whose mean error under their footprint, R / W, is above this (0 = any; DESIGN.md E9)
  grow_rate:       0,      // at most this share of new splats per rebalance, the highest error first (0 = no limit; E9)
  split_in_plane:  false,  // split / clone offsets keep off the splat's thinnest axis (E9)
  grow_by:         'error', // what growth splits by: 'error' (R > Tu, §5.1) or 'grad' (mean |positional gradient| per view > grow_grad, as brush; E9)
  grow_grad:       3e-5,    // grow_by 'grad''s threshold (scores at 1k steps on the dodecahedron: p50 ~1e-6, p90 ~1.5e-5, p99 ~1e-4)
  split_mode:      'shape', // how a splat splits: 'shape' (within its own shape, as brush; DESIGN.md E9) or 'error' (along its error moments, §5.1)
  mesh_snap:       0,      // mesh in the loop (DESIGN.md §9 E5): move splats this share of the way onto the fused surface (0 = off) ...
  mesh_snap_every: 1000,   // ... at the first rebalance every this many steps ...
  mesh_snap_views: 3,      // ... where every corner of the grid cell was seen by at least this many training views
  opacity_reset:   3000,   // clamp all opacities to 0.01 every N steps (0 = off)
  // Shape limits (applied every Adam step): huge splats & needles are almost always wrong
  max_splat_angle: 0.03,   // max axis scale / depth, in the training view
  max_splat_scale: 0.1,    // max axis scale, as a fraction of the scene radius
  max_anisotropy:  10,     // max ratio of the largest axis to the median one
};

// Whether step `step` rebalances, the last rebalance at step `last`: rebalance_every steps
// after it, rebalance_every_cap once `count` is at the splat cap (0: rebalance_every), and at
// every opacity reset. While the lease's budget falls (consolidating, until consolidationEnd): rebalance_every
// steps, and one at its end, so the count is max_splats from there
export function rebalanceDue(P, step, last, count) {
  if (consolidating(P, step)) {
    const c1 = consolidationEnd(P);
    if (last < c1 && step >= c1) return true;
    if (step < c1) return step - last >= P.rebalance_every;
  }
  const atCap = P.rebalance_every_cap > 0 && count >= 0.99 * splatBudget(P, step);
  return step - last >= (atCap ? P.rebalance_every_cap : P.rebalance_every) ||
         (P.opacity_reset > 0 && step % P.opacity_reset === 0);
}

// The splat budget at `step` (DESIGN.md §9 E7): max_splats, or with a planned length (train_steps > 0) and a
// lease (cap_lease > 0), max_splats x (1 + cap_lease) until the consolidation, the last consolidate_frac of the
// steps: there it falls linearly back to max_splats over the first 3/4, the rest trains at max_splats
export const consolidationEnd = (P) => P.train_steps * (1 - 0.25 * P.consolidate_frac);
export function splatBudget(P, step) {
  const T = P.train_steps, lease = P.cap_lease;
  if (!(T > 0) || !(lease > 0)) return P.max_splats;
  const c0 = T * (1 - P.consolidate_frac), c1 = consolidationEnd(P);
  const f = step <= c0 ? 1 : step >= c1 ? 0 : (c1 - step) / (c1 - c0);
  return Math.round(P.max_splats * (1 + lease * f));
}
// The consolidation's keep score: each splat's W (visibility mass) per window, averaged over the windows it
// lived with a decay of 0.7 a window (a video's window does not visit every view). vis: { sum, norm, score } or null
export function visibilityScore(vis, moments, count) {
  const sum = new Float32Array(count), norm = new Float32Array(count), score = new Float32Array(count);
  const ok = vis?.sum.length === count;
  for (let i = 0; i < count; i++) {
    sum[i] = (ok ? 0.7 * vis.sum[i] : 0) + moments[i * MOMENT_FLOATS];
    norm[i] = (ok ? 0.7 * vis.norm[i] : 0) + 1;
    score[i] = sum[i] / norm[i];
  }
  return { sum, norm, score };
}
// through a rebalance: src[i] = the pre-rebalance index of splat i (-1: new, no history)
export function remapVisibility(vis, src) {
  const sum = new Float32Array(src.length), norm = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) if (src[i] >= 0) { sum[i] = vis.sum[src[i]]; norm[i] = vis.norm[src[i]]; }
  return { sum, norm, score: null };
}
// the last consolidate_frac of a planned length: the lease's consolidation, and with polish Monte-Carlo's noise and L1 terms off
export const endPhase = (P, step) => P.train_steps > 0 && step > P.train_steps * (1 - P.consolidate_frac);
export const consolidating = (P, step) => P.cap_lease > 0 && endPhase(P, step);

// Fixed-point (x65536) 64-bit squared-error sum [lo, hi] of loss_acc
const lossFixed = (u32) => u32[1] * 4294967296 + u32[0];

// Mean depth spread (pix_state.w) over the pixels with T_final < 0.25 (pix_state: 4 floats / pixel)
function meanDepthSpread(pix) {
  let sum = 0, n = 0;
  for (let p = 0; p < pix.length; p += 4) if (pix[p] < 0.25) { sum += pix[p + 3]; n++; }
  return n > 0 ? sum / n : 0;
}

// Cache key of a view matrix: its 12 non-constant entries
function viewKey(view) {
  let key = '';
  for (const i of [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]) key += `,${view[i].toFixed(4)}`;
  return key;
}

// out[i] = src[idx[i]] (rows of `stride` floats); idx[i] < 0 leaves row i zeroed.
function gatherRows(src, idx, stride) {
  const out = new Float32Array(idx.length * stride);
  for (let i = 0; i < idx.length; i++) {
    const j = idx[i];
    if (j >= 0) out.set(src.subarray(j * stride, (j + 1) * stride), i * stride);
  }
  return out;
}

export class GaussOptTrainer {
  // params: kept (the app changes them while training), missing TRAIN_DEFAULTS filled in
  static async create(device, octree, initialSplats, images, params) {
    for (const [k, v] of Object.entries(TRAIN_DEFAULTS)) params[k] ??= v;
    const t = new GaussOptTrainer(device, octree, initialSplats, images, params);
    await t._init();
    return t;
  }

  constructor(device, octree, initialSplats, images, params) {
    this.device   = device;
    this.octree   = octree;
    this.images   = images || [];
    this.params   = params;
    this.count    = initialSplats.count;
    this.cpuParams  = new Float32Array(initialSplats.params);
    this.numVisible     = 0;  // training-camera sort (sortedIdxBuf)
    this.refineLeft     = 0;  // display: GPU refine passes still owed after the view stopped
    this.numVisibleView = 0;  // display-view sort (viewIdxBuf)
    this.sortGen    = 0;      // bumped whenever cpuParams / octree change: invalidates the CPU sorts

    this.stepCount     = 0;
    this.lastRebalanceStep = 0;  // step of the last rebalance (rebalanceDue)
    this.adamStepCount = 0;   // steps since the Adam state was last reset (bias correction)
    this.stepInFlight  = false;  // step lock, see _withStepLock()
    this.avgR          = 0.01;
    this.lastPsnr      = 0;
    this.currentTemp   = params.sa_temp;
    this.history       = [];  // { step, psnr, count, temp } training-loss PSNR
    this.valHistory    = [];  // { step, psnr, count, views } from evaluateValidation()
    this.splitVectors  = null;
    this.lastOpacityReset = -1;
    this.trainViews    = [];
    this.valViews      = [];
    this.gtTextures    = new Map(); // image index -> GPUTexture
    this.gtLoading     = new Map(); // image index -> in-flight Promise<GPUTexture>
    this.pendingLossRead = null;
    this.points3D      = null;  // COLMAP points3D (set by the app): depth for stipple seeds
    this.seedCursor    = 0;     // round-robin over trainViews for stipple seeding
    this.lastSeedInfo  = null;
    // Monte-Carlo's noise moves the faint splats with no bound: out of 1.25 times the starting octree's root (the box of
    // the initial splats; Adam's gradients never took a gallery scene's out of it), one is dead (octree.js
    // reconcileAndGrow `bounds`). Unbounded, jardin7's splats drifted 20000 units out, and its saved octree had a
    // leaf of 82k splats (the gallery's sort: 39 ms a frame).
    const root = octree.root;
    this.sceneBox = root ? { center: [root.cx, root.cy, root.cz], halfSize: 1.25 * root.halfSize } : null;
    this._invalidateCaches();
  }

  // Forget the cached training sort, display sort key and hatTex render
  _invalidateCaches() {
    this._lastSortKey = this._lastViewSortKey = this._lastHatKey = '';
  }

  // Training / validation view indices; with no training view, view 0 trains
  _splitViews() {
    this.trainViews = [];
    this.valViews = [];
    this.images.forEach((im, i) => (im.isValidation ? this.valViews : this.trainViews).push(i));
    if (this.trainViews.length === 0 && this.images.length > 0) this.trainViews = [0];
  }

  // Free the GPU buffers and cached GT textures, after any step in flight. The trainer is
  // unusable afterwards (step / renders return null).
  async destroy() {
    return this._withStepLock(async () => {
      while (this.gtLoading.size) await Promise.allSettled([...this.gtLoading.values()]);
      for (const tex of this.gtTextures.values()) tex.destroy();
      this.gtTextures.clear();
      for (const v of [...Object.values(this), ...(this.refineUniformBufs || [])]) {
        if (v instanceof GPUBuffer || v instanceof GPUTexture) v.destroy();
      }
      this.destroyed = true;
      this.images = [];
      this.count = 0;
      this.trainViews = [];
      this.valViews = [];
    });
  }

  async _init() {
    const mods = await loadShaders(this.device);
    const pipe = (name) => this.device.createComputePipeline({
      layout: 'auto', compute: { module: mods[name], entryPoint: 'main' },
    });
    this.pipeProject  = pipe('project_splats');
    this.pipeForward  = pipe('tile_forward');
    this.pipeBackward = pipe('tile_backward');
    this.pipeAdam     = pipe('adam_step');
    this.pipeSsimCoef = pipe('ssim_coef');
    this.pipeSsimGrad = pipe('ssim_grad');
    this.pipeTileScan = pipe('tile_scan');
    this.pipeTileFill = pipe('tile_fill');
    this.pipeTileSort = pipe('tile_sort');
    this.pipeSortRefine = pipe('sort_refine');
    this.refineUniformBufs = [0, 1].map(k =>
      this._buffer(`sort_refine_${k}`, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));

    this.trainUniformBuf = this._buffer('train_uniforms', UNIFORM_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.lossBuf     = this._buffer('loss_acc', 16, BUF_RW());
    this.lossReadBuf = this._buffer('loss_read', 16, BUF_READ());

    // Training resolution: first camera downscaled by params.downsample
    const cam0 = this.images[0]?.camera || { width: 512, height: 512 };
    const ds = this.params.downsample || 4;
    this.trainW  = Math.max(64, Math.floor(cam0.width  / ds));
    this.trainH  = Math.max(64, Math.floor(cam0.height / ds));
    this.scaleFx = this.trainW / cam0.width;
    this.scaleFy = this.trainH / cam0.height;
    this.rowPitch = Math.ceil(this.trainW * 4 / 256) * 256;  // bytesPerRow for texture readback

    this.hatTex = this.device.createTexture({
      size: [this.trainW, this.trainH], format: 'rgba8unorm', label: 'hat_tex',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    // D-SSIM: the partials per pixel (3 vec4f, ssim_coef), the gradient (ssim_grad; bound even when off)
    this.ssimCoefBuf = this._buffer('ssim_coef', this.trainW * this.trainH * 48, BUF_RW());
    this.ssimGradTex = this.device.createTexture({
      size: [this.trainW, this.trainH], format: 'rgba32float', label: 'ssim_grad_tex',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    // a vec4f per pixel: [T, z_med, last_k, spread] (tile_forward)
    this.pixStateBuf = this._buffer('pix_state', this.trainW * this.trainH * 16, BUF_RW());
    // Per-tile splat lists, built on the GPU each render (see _runPasses)
    this.tilesX = Math.ceil(this.trainW / 16);
    this.tilesY = Math.ceil(this.trainH / 16);
    const nTiles = this.tilesX * this.tilesY;
    this.tileCountsBuf = this._buffer('tile_counts', nTiles * 4, BUF_RW());
    this.tileCursorBuf = this._buffer('tile_cursor', nTiles * 4, BUF_RW());
    this.tileRangesBuf = this._buffer('tile_ranges', nTiles * 8, BUF_RW());
    this.binUniformBuf = this._buffer('bin_uniforms', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.binStatsBuf     = this._buffer('bin_stats', 16, BUF_RW());
    this.binStatsReadBuf = this._buffer('bin_stats_read', 16, BUF_READ());
    this._allocateTileLists(1 << 20);
    this._allocateSplatBuffers(Math.max(this.count * 2, 8192));
    this._uploadSplats();

    this._splitViews();
  }

  _buffer(label, size, usage) {
    return this.device.createBuffer({ label, size: Math.max(4, size), usage });
  }

  _allocateSplatBuffers(capacity) {
    this.capacity = capacity;
    for (const k of ['splatsBuf', 'momentsBuf', 'grads2dBuf', 'growAccBuf', 'adamMBuf', 'adamVBuf',
                     'sortedIdxBuf', 'viewIdxBuf', 'projSplatsBuf', 'projTilesBuf']) {
      this[k]?.destroy?.();
    }
    const splatBytes  = capacity * SPLAT_FLOATS * 4;
    const momentBytes = capacity * MOMENT_FLOATS * 4;
    const IDX = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.splatsBuf     = this._buffer('splats',   splatBytes,  BUF_RW());
    this.momentsBuf    = this._buffer('moments',  momentBytes, BUF_RW());
    this.grads2dBuf    = this._buffer('grads_2d', momentBytes, BUF_RW());
    this.growAccBuf    = this._buffer('grow_acc', capacity * 8, BUF_RW());  // adam_step's (|positional grad|, views)
    this.adamMBuf      = this._buffer('adam_m',   splatBytes,  BUF_RW());
    this.adamVBuf      = this._buffer('adam_v',   splatBytes,  BUF_RW());
    this.sortedIdxBuf  = this._buffer('sorted_idx',      capacity * 4, IDX);
    this.viewIdxBuf    = this._buffer('view_sorted_idx', capacity * 4, IDX);
    this.projSplatsBuf = this._buffer('proj_splats',     capacity * 32, BUF_RW());  // 2 vec4f / visible splat
    this.projTilesBuf  = this._buffer('proj_tiles',      capacity * 8, BUF_RW());  // tile rect / visible splat
    this.sortedIdx = new Uint32Array(capacity);
    this.viewIdx   = new Uint32Array(capacity);
  }

  // Upload cpuParams[0..count) and zero the moments / 2D grads.
  // adam: { adamM, adamV } keeps the optimizer state (rebalance); omitted = reset it.
  _uploadSplats(adam = null) {
    if (this.count > this.capacity) {
      this._allocateSplatBuffers(Math.max(this.count * 2, this.capacity * 2));
    }
    const q = this.device.queue;
    const splatFloats = this.count * SPLAT_FLOATS;
    q.writeBuffer(this.splatsBuf, 0, this.cpuParams, 0, splatFloats);
    const zeroMoments = new Uint32Array(this.count * MOMENT_FLOATS);
    q.writeBuffer(this.momentsBuf, 0, zeroMoments);
    q.writeBuffer(this.grads2dBuf, 0, zeroMoments);
    q.writeBuffer(this.growAccBuf, 0, new Uint32Array(this.count * 2));
    if (adam) {
      q.writeBuffer(this.adamMBuf, 0, adam.adamM, 0, splatFloats);
      q.writeBuffer(this.adamVBuf, 0, adam.adamV, 0, splatFloats);
    } else {
      const zeroAdam = new Uint32Array(splatFloats);
      q.writeBuffer(this.adamMBuf, 0, zeroAdam);
      q.writeBuffer(this.adamVBuf, 0, zeroAdam);
      this.adamStepCount = 0;
    }
    this.octree.updateExtents(this.cpuParams);
    this.sortGen++;
    this._invalidateCaches();
  }

  // Load and cache a ground-truth image texture at (trainW, trainH).
  // Concurrent loads (prefetch, frustum images, compare view) are deduped.
  async ensureGtTexture(imgIdx) {
    if (this.gtTextures.has(imgIdx)) return this.gtTextures.get(imgIdx);
    let p = this.gtLoading.get(imgIdx);
    if (!p) {
      p = this._loadGtTexture(imgIdx).finally(() => this.gtLoading.delete(imgIdx));
      this.gtLoading.set(imgIdx, p);
    }
    return p;
  }

  // (overridden by the headless harness)
  async _loadGtTexture(imgIdx) {
    const imgInfo = this.images[imgIdx];
    if (!imgInfo) return null;
    let blob = imgInfo.fileBlob;
    if (!blob && imgInfo.imageUrl) {
      const resp = await fetch(imgInfo.imageUrl);
      if (!resp.ok) throw new Error(`Failed to fetch ${imgInfo.imageUrl}`);
      blob = await resp.blob();
    }
    if (!blob) return null;

    const size = [this.trainW, this.trainH];
    const bmp = await createImageBitmap(blob, { resizeWidth: size[0], resizeHeight: size[1], resizeQuality: 'medium' });
    const tex = this.device.createTexture({
      size, format: 'rgba8unorm', label: `gt_${imgInfo.name}`,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC |
             GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.device.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, size);
    bmp.close();
    this.gtTextures.set(imgIdx, tex);
    return tex;
  }

  // Intrinsics of image `img` at training resolution (+ NDC-normalized focals for the sort).
  _intrinsics(img) {
    const c = img.camera;
    const fx = c.fx * this.scaleFx, fy = c.fy * this.scaleFy;
    return { fx, fy, cx: c.cx * this.scaleFx, cy: c.cy * this.scaleFy,
             fxNorm: 2 * fx / this.trainW, fyNorm: 2 * fy / this.trainH };
  }

  // Write TrainUniforms: view | intrinsics | dims | lrs | opt_params | limits | loss.
  // `extra` fills f32[24..39] (lrs, opt_params, limits, loss); forward-only renders leave them at 0.
  _writeUniforms(img, k, extra = null) {
    const buf = new ArrayBuffer(UNIFORM_BYTES);
    const f32 = new Float32Array(buf), u32 = new Uint32Array(buf);
    f32.set(img.view, 0);
    f32[16] = k.fx; f32[17] = k.fy; f32[18] = k.cx; f32[19] = k.cy;
    u32[20] = this.trainW; u32[21] = this.trainH; u32[22] = this.numVisible; u32[23] = this.count;
    if (extra) f32.set(extra, 24);
    this.device.queue.writeBuffer(this.trainUniformBuf, 0, buf);
  }

  // Compute pass with bind group 0 = `resources` (GPUBuffer or texture view, binding = index).
  _pass(enc, pipeline, resources, wx, wy = 1) { dispatch(this.device, enc, pipeline, resources, wx, wy); }

  // tile_list / tile_scratch with room for `capacity` (tile, splat) entries
  _allocateTileLists(capacity) {
    this.tileListCapacity = capacity;
    this.tileListBuf?.destroy();
    this.tileScratchBuf?.destroy();
    this.tileListBuf    = this._buffer('tile_list',    capacity * 4, BUF_RW());
    this.tileScratchBuf = this._buffer('tile_scratch', capacity * 4, BUF_RW());
  }

  // Render the current sortedIdxBuf in one submission: project (+ per-tile counts), build the
  // per-tile lists (scan, unordered fill, per-tile sort back to depth order), tile forward
  // (into hatTex / pixStateBuf). gtTex != null also runs tile backward (2D grads, WLBG
  // moments, loss; after ssim_coef / ssim_grad when params.ssim > 0) and the param update. extra(enc) appends commands.
  // The list capacity follows the entry count read back asynchronously: an overflow
  // (entries dropped for that render) grows it for the next one.
  _runPasses(gtTex = null, extra = null) {
    const n = this.numVisible, U = this.trainUniformBuf, B = this.binUniformBuf;
    const nTiles = this.tilesX * this.tilesY;
    this.device.queue.writeBuffer(B, 0, new Uint32Array([n, this.tilesX, nTiles, this.tileListCapacity]));
    const enc = this.device.createCommandEncoder({ label: gtTex ? 'train' : 'forward' });
    enc.clearBuffer(this.tileCountsBuf);
    this._pass(enc, this.pipeProject, [U, this.splatsBuf, this.sortedIdxBuf, this.projSplatsBuf, this.projTilesBuf,
                                       this.tileCountsBuf], Math.ceil(n / 64));
    this._pass(enc, this.pipeTileScan, [B, this.tileCountsBuf, this.tileRangesBuf, this.tileCursorBuf, this.binStatsBuf], 1);
    this._pass(enc, this.pipeTileFill, [B, this.projTilesBuf, this.tileCursorBuf, this.tileListBuf], Math.ceil(n / 64));
    this._pass(enc, this.pipeTileSort, [B, this.tileRangesBuf, this.tileListBuf, this.tileScratchBuf],
               this.tilesX, this.tilesY);

    const lists = [this.tileRangesBuf, this.tileListBuf];
    this._pass(enc, this.pipeForward, [U, this.splatsBuf, this.projSplatsBuf, this.hatTex.createView(), this.pixStateBuf,
                                       ...lists], this.tilesX, this.tilesY);
    if (gtTex) {
      if (this.params.ssim > 0) {
        const wx = Math.ceil(this.trainW / 16), wy = Math.ceil(this.trainH / 16);
        this._pass(enc, this.pipeSsimCoef, [U, gtTex.createView(), this.hatTex.createView(), this.ssimCoefBuf], wx, wy);
        this._pass(enc, this.pipeSsimGrad, [U, gtTex.createView(), this.hatTex.createView(), this.ssimCoefBuf,
                                            this.ssimGradTex.createView()], wx, wy);
      }
      this._pass(enc, this.pipeBackward, [U, this.splatsBuf, this.projSplatsBuf, gtTex.createView(), this.hatTex.createView(),
                                          this.pixStateBuf, this.grads2dBuf, this.momentsBuf, this.lossBuf, ...lists,
                                          this.ssimGradTex.createView()],
                 this.tilesX, this.tilesY);
      // per-splat 2D->3D chain rule + Adam / MC / SA update
      this._pass(enc, this.pipeAdam, [U, this.splatsBuf, this.grads2dBuf, this.adamMBuf, this.adamVBuf, this.momentsBuf, this.growAccBuf],
                 Math.ceil(this.count / 64));
    }
    extra?.(enc);
    const pollStats = !this.binStatsPending;
    if (pollStats) enc.copyBufferToBuffer(this.binStatsBuf, 0, this.binStatsReadBuf, 0, 16);
    this.device.queue.submit([enc.finish()]);
    if (pollStats) this._pollTileEntries();
  }

  // Non-blocking read of the last render's list entry count; grows the lists on overflow.
  _pollTileEntries() {
    this.binStatsPending = this.binStatsReadBuf.mapAsync(GPUMapMode.READ, 0, 16).then(() => {
      const total = new Uint32Array(this.binStatsReadBuf.getMappedRange(0, 16))[0];
      this.binStatsReadBuf.unmap();
      if (total > 0.8 * this.tileListCapacity) {
        if (total > this.tileListCapacity) console.warn(`[Gauss-Opt] tile lists overflowed (${total} > ${this.tileListCapacity}): growing`);
        this._allocateTileLists(Math.ceil(total * 2));
      }
    }).catch(() => {}).finally(() => { this.binStatsPending = null; });
  }

  // Copy GPU resources into one staging buffer and return CPU copies, one per source.
  // src: { buf, bytes } (-> Float32Array) or { tex } (rgba8, -> Uint8Array with rowPitch).
  async _readback(label, srcs) {
    const texBytes = this.rowPitch * this.trainH;
    let total = 0;
    const offs = srcs.map(s => { const o = total; total += s.tex ? texBytes : s.bytes; return o; });
    const staging = this._buffer(label, total, BUF_READ());
    try {
      const enc = this.device.createCommandEncoder({ label });
      srcs.forEach((s, i) => {
        if (s.tex) {
          enc.copyTextureToBuffer({ texture: s.tex }, { buffer: staging, offset: offs[i], bytesPerRow: this.rowPitch },
                                  [this.trainW, this.trainH]);
        } else if (s.bytes > 0) {
          enc.copyBufferToBuffer(s.buf, 0, staging, offs[i], s.bytes);
        }
      });
      this.device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const data = staging.getMappedRange().slice(0);
      staging.unmap();
      return srcs.map((s, i) => s.tex ? new Uint8Array(data, offs[i], texBytes)
                                      : new Float32Array(data, offs[i], s.bytes / 4));
    } finally {
      staging.destroy();
    }
  }

  // Training-loss PSNR from the fixed-point (x65536) 64-bit squared-error sum [lo, hi] in lossBuf.
  _recordLoss(mseFixed, step, count, temp) {
    const mse = (mseFixed / 65536.0) / (this.trainW * this.trainH);
    if (mse <= 0) return;
    this.lastPsnr = -10 * Math.log10(Math.max(1e-8, mse));
    this.history.push({ step, psnr: this.lastPsnr, count, temp });
    if (this.history.length > HISTORY_LEN) this.history.shift();
  }

  async _drainLossRead() {
    if (this.pendingLossRead) await this.pendingLossRead;
  }

  // Sort the visible splats for the training camera into sortedIdxBuf, cached per pose and
  // sortGen: the sort reads cpuParams, which only change with it.
  updateSortingForView(camCenter, view, fxNorm, fyNorm) {
    const sortKey = `${this.sortGen}:${this.count}:${fxNorm.toFixed(3)},${fyNorm.toFixed(3)}${viewKey(view)}`;
    if (this._lastSortKey === sortKey && this.numVisible > 0) return this.numVisible;
    this._lastSortKey = sortKey;
    this.numVisible = this.octree.sortOctreeFrontToBack(
      this.cpuParams, camCenter, view, fxNorm, fyNorm, this.sortedIdx).numVisible;
    if (this.numVisible > 0) this.device.queue.writeBuffer(this.sortedIdxBuf, 0, this.sortedIdx, 0, this.numVisible);
    return this.numVisible;
  }

  // Sort for the on-screen view into viewIdxBuf, independently of the training sort.
  // The exact CPU sort (octree cull + radix) only runs when its culled set may be stale:
  // splats re-synced (sortGen), intrinsics changed, the view turned > 5 deg or moved > 5% of
  // the scene radius since, or every REFRESH moving frames. In between, a moving view keeps
  // the last list and refines its order on the GPU against the live splat positions
  // (sort_refine, as in splats/); the CPU culls with a 25% wider frustum to cover the drift.
  // A still view costs nothing.
  sortForDisplay(camCenter, view, fxNorm, fyNorm) {
    if (this.destroyed) return 0;
    const REFRESH = 30, MARGIN = 0.8, MAX_COS = Math.cos(5 * Math.PI / 180), PASSES = 4;
    const baseKey = `${this.sortGen}:${this.count}:${fxNorm.toFixed(3)},${fyNorm.toFixed(3)}`;
    const key = baseKey + viewKey(view);
    const moved = this._lastViewSortKey !== key;
    if (!moved && this.refineLeft <= 0) return this.numVisibleView;
    this._lastViewSortKey = key;

    const fwd = [view[2], view[6], view[10]], vs = this.viewSort;
    let full = !vs || vs.baseKey !== baseKey || vs.frames >= REFRESH;
    if (!full) {
      const cosA = fwd[0] * vs.fwd[0] + fwd[1] * vs.fwd[1] + fwd[2] * vs.fwd[2];
      const dist = Math.hypot(camCenter[0] - vs.center[0], camCenter[1] - vs.center[1], camCenter[2] - vs.center[2]);
      full = cosA < MAX_COS || dist > 0.05 * this.octree.sceneRadius;
    }
    if (full) {
      this.numVisibleView = this.octree.sortOctreeFrontToBack(
        this.cpuParams, camCenter, view, fxNorm * MARGIN, fyNorm * MARGIN, this.viewIdx).numVisible;
      if (this.numVisibleView > 0) this.device.queue.writeBuffer(this.viewIdxBuf, 0, this.viewIdx, 0, this.numVisibleView);
      this.viewSort = { baseKey, fwd, center: [...camCenter], frames: 0 };
    } else {
      vs.frames++;
    }
    // keep refining for a few frames after the view stops, so the order settles exactly
    this.refineLeft = moved ? 8 : this.refineLeft - 1;
    if (this.numVisibleView > 1) this._refineDisplaySort(view, PASSES);
    return this.numVisibleView;
  }

  // `passes` chunked insertion-sort passes over viewIdxBuf, alternating chunk offsets.
  _refineDisplaySort(view, passes) {
    const n = this.numVisibleView, CHUNK = 32;
    const enc = this.device.createCommandEncoder({ label: 'sort_refine' });
    for (let k = 0; k < 2; k++) {
      const b = new ArrayBuffer(32), f = new Float32Array(b), u = new Uint32Array(b);
      f[0] = view[2]; f[1] = view[6]; f[2] = view[10]; f[3] = view[14];  // z row of the view
      u[4] = n; u[5] = k * (CHUNK / 2);
      this.device.queue.writeBuffer(this.refineUniformBufs[k], 0, b);
    }
    for (let p = 0; p < passes; p++) {
      const k = p & 1, chunks = Math.ceil((n - k * (CHUNK / 2)) / CHUNK);
      if (chunks > 0) {
        this._pass(enc, this.pipeSortRefine, [this.refineUniformBufs[k], this.splatsBuf, this.viewIdxBuf],
                   Math.ceil(chunks / 64));
      }
    }
    this.device.queue.submit([enc.finish()]);
  }

  // Forward-render COLMAP view `viewIdx` into hatTex without touching the parameters
  // (compare view). Backs off (null) while a step / snapshot holds the step lock.
  async renderViewForward(viewIdx) {
    if (this.stepInFlight) return null;
    return this._renderViewForward(viewIdx);
  }

  // Lock-free version, for callers already holding the step lock. Cached per (view, step, count).
  async _renderViewForward(viewIdx) {
    if (this.images.length === 0 || this.count === 0) return null;
    const idx = Math.min(Math.max(0, viewIdx | 0), this.images.length - 1);
    const gtTex = await this.ensureGtTexture(idx);
    if (!gtTex || this.destroyed) return null;  // destroy() may have run meanwhile (no step lock here)
    const out = { viewIdx: idx, gtTex, hatTex: this.hatTex };
    const hatKey = `${idx}:${this.stepCount}:${this.count}`;
    if (this._lastHatKey === hatKey) return out;

    const img = this.images[idx];
    const k = this._intrinsics(img);
    if (this.updateSortingForView(img.center, img.view, k.fxNorm, k.fyNorm) === 0) return out;
    this._writeUniforms(img, k);
    this._runPasses();
    this._lastHatKey = hatKey;
    return out;
  }

  // One GPU training iteration (next training camera, or `forcedCamIdx` if it is a training view).
  // Returns { viewIdx, gtTex, hatTex, psnr }; hatTex is null when a rebalance left another
  // view's render in it (stipple seeding).
  async step(forcedCamIdx = null) {
    if (this.stepInFlight || this.images.length === 0 || this.count === 0) return null;
    this.stepInFlight = true;
    try {
      this._lastHatKey = '';
      return await this._stepImpl(forcedCamIdx);
    } finally {
      this.stepInFlight = false;
    }
  }

  async _stepImpl(forcedCamIdx) {
    const nViews = this.trainViews.length;
    // Round-robin cursor, separate from stepCount: a view with no visible splat (or no GT)
    // is skipped instead of being retried forever
    this.viewCursor ??= this.stepCount;
    if (forcedCamIdx !== null && !this.trainViews.includes(forcedCamIdx)) forcedCamIdx = null;  // never train on held-out views
    const cur = (forcedCamIdx === null) ? this.viewCursor++ : this.viewCursor;
    const viewIdx = forcedCamIdx ?? this.trainViews[cur % nViews];
    if (forcedCamIdx === null && nViews > 1) {  // prefetch the next 3 GT images
      for (let p = 1; p <= 3; p++) {
        const next = this.trainViews[(cur + p) % nViews];
        if (!this.gtTextures.has(next)) this.ensureGtTexture(next).catch(() => {});
      }
    }
    const gtTex = await this.ensureGtTexture(viewIdx);
    if (!gtTex) return null;

    const img = this.images[viewIdx];
    const k = this._intrinsics(img);
    if (this.updateSortingForView(img.center, img.view, k.fxNorm, k.fyNorm) === 0) return null;

    this.stepCount++;
    this.adamStepCount++;

    const P = this.params, R = this.octree.sceneRadius;
    const optMethod = OPT_METHODS[P.opt_method] ?? 0;
    // SA temperature T = T0 * cooling^(t/50), t optionally restarting every max(1000, 10 K) steps
    const cycleSteps = P.sa_reheat ? (this.stepCount % Math.max(1000, P.rebalance_every * 10)) : this.stepCount;
    this.currentTemp = P.sa_temp * Math.pow(P.sa_cooling, cycleSteps / 50.0);
    // 50-step lr warmup (matters when resuming), exponential lr_pos decay to lr_pos * lr_pos_final
    const warmup = Math.min(1.0, this.stepCount / 50.0);
    const decayT = Math.min(1, this.stepCount / Math.max(1, P.lr_decay_steps));
    const posDecay = Math.pow(P.lr_pos_final, decayT);
    const mcTerms = optMethod === 1 && !(P.polish && endPhase(P, this.stepCount));
    this._writeUniforms(img, k, [
      P.lr_pos * R * warmup * posDecay, P.lr_scale * warmup, P.lr_quat * warmup, P.lr_color,  // lrs
      optMethod, mcTerms ? P.mc_noise : 0, Math.random(), this.adamStepCount,  // opt_params
      P.max_splat_angle, P.max_splat_scale * R,                                                 // limits
      Math.log(Math.max(1, P.max_anisotropy)), P.lr_opacity,
      P.ssim, P.drop_rate, P.drop_rate > 0 ? 1 / (1 - P.drop_rate) : 0, 0,                     // loss
      ...(mcTerms ? [P.mc_opacity_reg / this.count, P.mc_scale_reg / (3 * this.count)] : [0, 0]), 0, 0,  // reg
    ]);
    this.device.queue.writeBuffer(this.lossBuf, 0, new Uint32Array(4));

    const needRebalance = rebalanceDue(P, this.stepCount, this.lastRebalanceStep, this.count);
    const needLossPoll  = !needRebalance && (this.stepCount % 5 === 0) && !this.pendingLossRead;
    this._runPasses(gtTex, needLossPoll ? enc => enc.copyBufferToBuffer(this.lossBuf, 0, this.lossReadBuf, 0, 16) : null);

    let hatTex = this.hatTex;
    if (needRebalance) {
      if (await this._rebalance()) hatTex = null;
    } else if (needLossPoll) {
      const step = this.stepCount, count = this.count, temp = this.currentTemp;
      this.pendingLossRead = this.lossReadBuf.mapAsync(GPUMapMode.READ, 0, 16).then(() => {
        const mseFixed = lossFixed(new Uint32Array(this.lossReadBuf.getMappedRange(0, 16)));
        this.lossReadBuf.unmap();
        this._recordLoss(mseFixed, step, count, temp);
      }).catch(() => {}).finally(() => { this.pendingLossRead = null; });
    }
    return { viewIdx, gtTex, hatTex, psnr: this.lastPsnr };
  }

  // Copy the GPU splats into cpuParams (e.g. before a .ply export).
  async syncSplatsFromGpu() {
    return this._withStepLock(async () => {
      await this._drainLossRead();
      const [params] = await this._readback('sync_splats', [{ buf: this.splatsBuf, bytes: this.count * SPLAT_FLOATS * 4 }]);
      this.cpuParams.set(params);
      this.octree.updateExtents(this.cpuParams);
      this.sortGen++;
    });
  }

  // New cameras for the same splats (an SfM repair, DESIGN-sfm §6.3): the images are matched by
  // name; a view keeps its ground-truth texture when its photo is the same (keepTextures, i.e.
  // unchanged intrinsics), and the octree's camera statistics are refreshed. The splats, their
  // Adam state and the step count stay. Returns { kept, added, dropped } view counts.
  async replaceImages(images, { keepTextures = true } = {}) {
    const c0 = this.images[0]?.camera, c1 = images[0]?.camera;
    if (c0 && c1 && (c0.width !== c1.width || c0.height !== c1.height)) throw new Error('replaceImages: the image size changed');
    return this._withStepLock(async () => {
      while (this.gtLoading.size) await Promise.allSettled([...this.gtLoading.values()]);
      const index = new Map(images.map((im, i) => [im.name, i]));
      const textures = new Map();
      for (const [i, tex] of this.gtTextures) {
        const k = index.get(this.images[i]?.name);
        if (keepTextures && k !== undefined) textures.set(k, tex);
        else tex.destroy();
      }
      const kept = images.filter(im => this.images.some(o => o.name === im.name)).length;
      const dropped = this.images.length - kept;
      this.images = images;
      this.gtTextures = textures;
      this._splitViews();
      this.octree.updateCameraAndNeighborStats(images);
      this._invalidateCaches();
      this.sortGen++;
      return { kept, added: images.length - kept, dropped };
    });
  }

  // Run `fn` holding the step lock: step() and renderViewForward() back off meanwhile, so they
  // can't overwrite the shared sort / projection / hat buffers or change the splats while `fn`
  // awaits a GPU readback.
  async _withStepLock(fn) {
    while (this.stepInFlight) await new Promise(r => setTimeout(r, 1));
    this.stepInFlight = true;
    try {
      return await fn();
    } finally {
      this.stepInFlight = false;
    }
  }

  // Full optimizer state (splats, WLBG moments, Adam m/v) for a .gosnap snapshot.
  async readbackTrainingState() {
    return this._withStepLock(async () => {
      await this._drainLossRead();
      const splatBytes = this.count * SPLAT_FLOATS * 4;
      const [params, moments, adam_m, adam_v] = await this._readback('snapshot_read', [
        { buf: this.splatsBuf,  bytes: splatBytes },
        { buf: this.momentsBuf, bytes: this.count * MOMENT_FLOATS * 4 },
        { buf: this.adamMBuf,   bytes: splatBytes },
        { buf: this.adamVBuf,   bytes: splatBytes },
      ]);
      this.cpuParams.set(params);
      this.octree.updateExtents(this.cpuParams);
      this.sortGen++;
      return {
        count:         this.count,
        stepCount:     this.stepCount,
        lastRebalanceStep: this.lastRebalanceStep,
        adamStepCount: this.adamStepCount,
        currentTemp:   this.currentTemp,
        avgR:          this.avgR,
        lastPsnr:      this.lastPsnr,
        history:       this.history,
        valHistory:    this.valHistory,
        blobs:         { params, moments, adam_m, adam_v },
      };
    });
  }

  // Restore optimizer state saved by readbackTrainingState() (call right after create()).
  applyTrainingState(state, blobs) {
    const check = (name, floats) => {
      const b = blobs[name];
      if (b && b.length !== this.count * floats) {
        throw new Error(`snapshot blob "${name}" has ${b.length} floats, expected ${this.count * floats}`);
      }
      return b || null;
    };
    const moments = check('moments', MOMENT_FLOATS);
    const adamM   = check('adam_m',  SPLAT_FLOATS);
    const adamV   = check('adam_v',  SPLAT_FLOATS);
    if (moments) this.device.queue.writeBuffer(this.momentsBuf, 0, moments);
    if (adamM && adamV) {
      this.device.queue.writeBuffer(this.adamMBuf, 0, adamM);
      this.device.queue.writeBuffer(this.adamVBuf, 0, adamV);
      this.adamStepCount = state.adamStepCount | 0;
    }
    this.stepCount   = state.stepCount | 0;
    // older snapshots: the last multiple of rebalance_every (their schedule)
    this.lastRebalanceStep = state.lastRebalanceStep ?? this.stepCount - this.stepCount % this.params.rebalance_every;
    this.currentTemp = state.currentTemp ?? this.currentTemp;
    this.avgR        = state.avgR ?? this.avgR;
    this.lastPsnr    = state.lastPsnr ?? 0;
    this.history     = Array.isArray(state.history) ? state.history.slice(-HISTORY_LEN) : [];
    this.valHistory  = Array.isArray(state.valHistory) ? state.valHistory : [];
  }

  // Held-out metrics: forward-render every validation camera (never trained on) at training
  // resolution vs its GT; means of the per-view PSNR, SSIM and 1/4-resolution PSNR (psnr4), as
  // 3DGS averages per-view PSNRs, and of the depth spread (pix_state.w, see tile_forward) over
  // the pixels where it is measured (T_final < 0.25).
  async evaluateValidation() {
    if (this.valViews.length === 0 || this.count === 0) return null;
    return this._withStepLock(async () => {
      let sumPsnr = 0, sumSsim = 0, sumPsnr4 = 0, sumSpread = 0, n = 0;
      const perView = [];  // [name, dB]: the SfM panel scores keyframes that train badly (DESIGN-sfm §6.1)
      const W = this.trainW, H = this.trainH, bpr = this.rowPitch;
      try {
        for (const v of this.valViews) {
          this._lastHatKey = '';
          const out = await this._renderViewForward(v);
          if (!out) continue;
          const [gt, hat, pix] = await this._readback('val_readback', [
            { tex: out.gtTex }, { tex: this.hatTex }, { buf: this.pixStateBuf, bytes: W * H * 16 },
          ]);
          const psnr = rgbPsnr(gt, hat, W, H, bpr);
          perView.push([this.images[v].name, psnr]);
          sumPsnr += psnr;
          sumSsim += rgbSsim(gt, hat, W, H, bpr);
          sumPsnr4 += rgbPsnrDown(gt, hat, W, H, bpr, 4);
          sumSpread += meanDepthSpread(pix);
          n++;
        }
      } finally {
        this._invalidateCaches();  // shared buffers now hold the last val view
      }
      if (n === 0) return null;
      const res = { step: this.stepCount, psnr: sumPsnr / n, ssim: sumSsim / n, psnr4: sumPsnr4 / n,
                    spread: sumSpread / n, count: this.count, views: n };
      this.valHistory.push(res);
      return { ...res, perView };
    });
  }

  // Stipple seeding: render the next training view, read back GT / hat / pix_state and turn
  // the under-fit residual into up to `budget` new splats (see stipple.js). Needs the step lock.
  async _stippleSeeds(budget) {
    if (budget <= 0 || this.trainViews.length === 0) return null;
    const t0 = performance.now();
    const viewIdx = this.trainViews[this.seedCursor++ % this.trainViews.length];
    this._lastHatKey = '';
    const out = await this._renderViewForward(viewIdx);
    if (!out || this.numVisible === 0) return null;

    const W = this.trainW, H = this.trainH, bpr = this.rowPitch;
    const [gt, hat, pix] = await this._readback('stipple_readback', [
      { tex: out.gtTex }, { tex: this.hatTex }, { buf: this.pixStateBuf, bytes: W * H * 16 },
    ]);
    const img = this.images[viewIdx];
    const { fx, fy, cx, cy } = this._intrinsics(img);
    const density = residualDensity(gt, hat, W, H, bpr);
    const sites = stippleSites(density, W, H, budget, { seed: 1234 + this.stepCount });
    const depthGrid = this.points3D
      ? buildViewDepthGrid(img, this.points3D, this.scaleFx, this.scaleFy, W, H)
      : { grid: new Float32Array(1), gw: 1, gh: 1, cell: Math.max(W, H), median: 0.5 * (img.zMin + img.zMax) };
    const seeds = makeSeedSplats({ img, sites, pix, gt, bytesPerRow: bpr, W, H, fx, fy, cx, cy, depthGrid });
    this.lastSeedInfo = {
      viewIdx, count: seeds.count, covered: seeds.covered,
      fromPoints: seeds.fromPoints, fromMedian: seeds.fromMedian, ms: performance.now() - t0,
    };
    return seeds.params;
  }

  // CPU outer loop (inside a step, holding the lock): read back loss / splats / moments / Adam,
  // prune-grow-rebuild (or just rebuild) the octree, carry Adam state over, re-upload.
  // Returns true when stipple seeding rendered another view into hatTex.
  // Everything here runs on the main thread, so the page stops drawing meanwhile (300k splats:
  // 160-200 ms, mostly reconcileAndGrow, then the readback, gather and upload; ~33% of the
  // training time at one rebalance per 100 steps, hence 400 at the cap). No worker: DESIGN.md §5.3.
  async _rebalance() {
    const ms = {}, t = [performance.now()], lap = (k) => { t.push(performance.now()); ms[k] = t.at(-1) - t.at(-2); };
    await this._drainLossRead();  // keep history in step order
    const P = this.params;
    const splatBytes = this.count * SPLAT_FLOATS * 4;
    const byGrad = P.grow_by === 'grad';
    const [loss, params, moments, adamM0, adamV0, growAcc] = await this._readback('rebalance_read', [
      { buf: this.lossBuf,    bytes: 16 },
      { buf: this.splatsBuf,  bytes: splatBytes },
      { buf: this.momentsBuf, bytes: this.count * MOMENT_FLOATS * 4 },
      { buf: this.adamMBuf,   bytes: splatBytes },
      { buf: this.adamVBuf,   bytes: splatBytes },
      ...(byGrad ? [{ buf: this.growAccBuf, bytes: this.count * 8 }] : []),
    ]);
    // grow_by 'grad' (E9, as brush): a splat's score, its summed |positional gradient| per view it was seen from
    let growScore = null;
    if (byGrad) {
      growScore = new Float32Array(this.count);
      for (let i = 0; i < this.count; i++) growScore[i] = growAcc[2 * i + 1] > 0 ? growAcc[2 * i] / growAcc[2 * i + 1] : 0;
      const seen = growScore.filter(g => g > 0).sort(), q = (f) => seen.length ? seen[Math.floor(f * (seen.length - 1))] : 0;
      this.lastGrowInfo = { step: this.stepCount, p50: q(0.5), p90: q(0.9), p99: q(0.99), over: seen.filter(g => g > P.grow_grad).length };
    }
    this._recordLoss(lossFixed(new Uint32Array(loss.buffer, loss.byteOffset, 2)), this.stepCount, this.count, this.currentTemp);
    this.cpuParams.set(params);
    this.sortGen++;
    lap('read');

    // mean error mass R_i over active splats (diagnostic heatmap scale)
    let sumR = 0, active = 0;
    for (let i = 0; i < this.count; i++) {
      const r = moments[i * MOMENT_FLOATS + 2];
      if (r > 0) { sumR += r; active++; }
    }
    if (active > 0) this.avgR = sumR / active;

    // E7: the error of fixed cells of the scene over this window (regions.js), when region_log is on
    if (P.region_log) {
      this.regionGrid ??= makeRegionGrid(this.cpuParams, this.count, P.region_res);
      (this.regionLog ??= []).push({ step: this.stepCount, count: this.count, cells: regionStats(this.regionGrid, this.cpuParams, this.count, moments) });
    }

    // mesh in the loop: the training views' rendered depth fused (the splats as they are on the GPU,
    // = cpuParams), then the centers moved onto that surface before the octree sees them
    let src, hatUsed = false;
    const snapEvery = Math.max(1, P.mesh_snap_every);
    if (P.mesh_snap > 0 && Math.floor(this.stepCount / snapEvery) > Math.floor(this.lastRebalanceStep / snapEvery)) {
      const grid = await fuseTrainer(this, this.trainViews);
      this.lastSnapInfo = { step: this.stepCount, voxel: grid.voxel,
                            ...snapToSurface(this.cpuParams, this.count, grid, { amount: P.mesh_snap, minViews: P.mesh_snap_views }) };
      hatUsed = true;
      lap('snap');
    }

    // simulated annealing: last window's jumps kept or undone, new ones (anneal.js)
    if (P.opt_method === 'sim_anneal') {
      const r = annealStep(this.cpuParams, this.count, moments, this.saState, { temp: this.currentTemp, share: P.sa_share, size: P.sa_size });
      this.saState = r.state;
      this.lastAnnealInfo = { accepted: r.accepted, rejected: r.rejected, proposed: r.proposed, temp: this.currentTemp };
      lap('anneal');
    } else {
      this.saState = null;
    }

    // the cap lease (splatBudget): what the consolidation keeps, each splat's visibility W per window averaged
    // over the windows it lived (decay 0.7 a window: a video's window does not visit every view)
    const budget = splatBudget(P, this.stepCount), lease = P.cap_lease > 0 && P.train_steps > 0;
    if (lease) this.vis = visibilityScore(this.vis, moments, this.count);
    // E7's repair: the stuck regions (regions.js RegionTracker) split more easily and draw more relocations
    let repairMask = null;
    if (P.region_repair > 0) {
      this.regionTracker ??= new RegionTracker(P.region_res);
      const stuck = this.regionTracker.update(this.cpuParams, this.count, moments);
      repairMask = this.regionTracker.splatMask(this.cpuParams, this.count);
      this.lastRepairInfo = stuck && { cells: stuck.reduce((a, b) => a + b, 0), splats: repairMask.reduce((a, b) => a + b, 0) };
    }

    // src[i] = pre-rebalance index of new splat i (-1 = new child, Adam starts from zero)
    if (P.enable_rebalance) {
      this.lastSeedInfo = null;
      let seedSplats = null;
      if (P.stipple_seed && !consolidating(P, this.stepCount)) {
        // at the cap too: the seeds take the room the prunes free (octree.js reconcileAndGrow)
        const nSeeds = Math.min(Number(P.stipple_count) || 0, P.max_splats);
        hatUsed = nSeeds > 0;
        seedSplats = await this._stippleSeeds(nSeeds);
        lap('seed');
      }
      const res = this.octree.reconcileAndGrow(this.cpuParams, this.count, moments, this.images, {
        maxSplats:      budget,
        keepScore:      lease ? this.vis.score : null,
        repairMask,
        repairBoost:    P.region_repair,
        pruneOpacity:   P.prune_opacity,
        hysteresis:     P.hysteresis,
        enableFloaters: P.remove_floaters,
        floaterDepth:   P.floater_depth,
        floaterSpare:   P.floater_spare > 0 && this.points3D && !this.points3D.fallback
                          ? { positions: this.points3D.positions, count: this.points3D.count, radius: P.floater_spare * this.octree.sceneRadius }
                          : null,
        splitMinViews:  P.split_min_views,
        splitMinErr:    P.split_min_err,
        growRate:       P.grow_rate,
        splitInPlane:   P.split_in_plane,
        growScore,
        growThreshold:  P.grow_grad,
        splitMode:      P.split_mode,
        visitsPerView:  Math.max(1, (this.stepCount - this.lastRebalanceStep) / Math.max(1, this.trainViews.length)),
        enableSplit:    P.enable_split,
        enableMerge:    P.enable_merge,
        optMethod:      P.opt_method,
        relocateBy:     P.mc_relocate,
        relocateMode:   P.mc_relocate_mode,
        bounds:         P.opt_method === 'monte_carlo' ? this.sceneBox : null,
        saTemp:         this.currentTemp,
        seedSplats,
      });
      // res.srcIdx is in pre-build order; the final octree build then permuted by lastPerm
      const perm = this.octree.lastPerm;
      src = new Int32Array(res.count);
      for (let i = 0; i < res.count; i++) src[i] = res.srcIdx[perm[i]];
      this.count        = res.count;
      this.cpuParams    = res.params;
      this.splitVectors = res.splitVectors;
      lap('reconcile');
    } else {
      this.octree.build(this.cpuParams, this.count, this.images);  // re-bin drifted splats only
      src = this.octree.lastPerm;
      lap('build');
    }
    const adamM = gatherRows(adamM0, src, SPLAT_FLOATS);
    const adamV = gatherRows(adamV0, src, SPLAT_FLOATS);
    this.saState = remapAnnealState(this.saState, src);
    if (lease) this.vis = remapVisibility(this.vis, src);
    lap('gather');

    // Periodic opacity reset (3DGS): cap alpha at 0.01, so splats that don't earn their
    // opacity back before the next prune get removed. Not with monte_carlo: its L1 term on the opacity does that
    const every = P.opt_method === 'monte_carlo' ? 0 : P.opacity_reset;
    if (every > 0 && Math.floor(this.stepCount / every) > Math.floor(this.lastRebalanceStep / every)) {
      const cap = Math.log(0.01 / 0.99);
      for (let b = 3; b < this.count * SPLAT_FLOATS; b += SPLAT_FLOATS) {
        if (this.cpuParams[b] > cap) {
          this.cpuParams[b] = cap;
          adamM[b] = 0; adamV[b] = 0;
        }
      }
      this.lastOpacityReset = this.stepCount;
    }
    this._uploadSplats({ adamM, adamV });
    lap('upload');
    ms.total = t.at(-1) - t[0];
    this.lastRebalanceMs = ms;  // phase times (ms) of the last rebalance, for the log
    this.lastRebalanceStep = this.stepCount;
    return hatUsed;
  }
}
