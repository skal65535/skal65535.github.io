// trainer.js — Hybrid GPU/CPU 3DGS optimizer: GPU inner loop (project -> tile forward ->
// tile backward + WLBG moments -> Adam / Monte-Carlo update), CPU outer
// loop every `rebalance_every` steps (prune / grow / floaters / octree rebuild, see octree.js).
"use strict";

import { SPLAT_FLOATS } from './colmap.js';
import { MOMENT_FLOATS } from './octree.js';
import { loadShaders } from './render.js';
import { dispatch } from './gpu.js';
import { ENV_N, ENV_TEXELS, envFromBytes, envToBytes } from './env.js';
import { buildViewDepthGrid, seedsFromView } from './stipple.js';
import { makeRegionGrid, regionStats, RegionTracker } from './regions.js';
import { rgbPsnr, rgbPsnrDown, ssimMean, rgbPsnrRegions, rgbPsnrAffine, COVERED_T } from './metrics.js';
import { Exposure, EXPO_FLOATS } from './exposure.js';
import { Poses } from './poses.js';
import { setPose } from './colmap.js';

const UNIFORM_BYTES = 224;     // TrainUniforms (shaders.wgsl): 56 x f32
const EXPO_OFF  = 16 + ENV_TEXELS * 16;  // loss_acc's bytes of the camera's gradient: the exposure's (tile_backward EXPO_AT), then
const CAMG_FLOATS = EXPO_FLOATS + 12;    // the pose's and the lens's (cam_reduce CAM_AT: rho, phi, log f, cx, cy, k1, 2 pad)
const EXPO_RING = 16;          // the camera gradients of this many steps read back at once (_expoRecord)
const HISTORY_LEN   = 300;     // training-PSNR history kept for the telemetry plot
export const SEED_LEAD = 16;  // steps between the read of the stipple seeds' view and their rebalance (_prepareSeeds)
const OPT_METHODS   = { adam: 0, monte_carlo: 1 };
const BUF_RW   = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const BUF_READ = () => GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

// Defaults of the params a training step reads (the app's params start from these)
export const TRAIN_DEFAULTS = {
  // Optimizer extras: Monte-Carlo (3DGS-MCMC: noise, L1 terms, relocation)
  mc_noise:        1e6,    // lambda_noise: SGLD noise on the nearly transparent splats' centers, x lr_pos x covariance (3DGS-MCMC: 5e5;
                           // 1e6 the best of 5e5 / 1e6 / 2e6 here, DESIGN.md §5.4)
  mc_opacity_reg:  0.01,   // lambda_o: mean |opacity| added to the loss
  mc_scale_reg:    0.01,   // lambda_s: mean |scale| (per axis)
  mc_relocate:     'opacity',  // the live splats the dead ones move onto, drawn ~ 'opacity' or 'error'
  mc_relocate_mode: 'copy',   // ... as copies in place (3DGS-MCMC) or 'shape': each one drawn splits within its shape (brush)
  // Learning rates
  lr_pos:          0.00002,  // x the scene radius, decaying to 1e-6 x it as brush's lr_mean (x its bounds' median side there;
                             // 3DGS: 1.6e-4; 5e-5 before DESIGN.md E11, whose sweep has this level on held-out PSNR, the depth better)
  lr_pos_final:    0.05,   // lr_pos decays exponentially to lr_pos * lr_pos_final ...
  lr_decay_steps:  10000,  // ... over the planned length (train_steps) or, open-ended, this many steps
  lr_scale:        0.005,
  lr_quat:         0.002,
  lr_color:        0.0025,
  lr_opacity:      0.01,   // brush's; 0.05 floated splats under the dodecahedron's floor and streaked it (DESIGN.md E9)
  ssim:            0,      // D-SSIM loss weight (0 = off: 0.8 L1 + 0.2 L2), see tile_backward
  bg_lr:           0,      // the background's learning rate (env.js, DESIGN.md E13; 0 = off: black, as before)
  exposure_lr:     0,      // each training view's exposure, its learning rate (exposure.js, DESIGN.md E19; 0 = off)
  exposure_from:   1000,   // ... learned from this step on (the splats formed first)
  pose_lr:         0,      // each training view's pose (poses.js, DESIGN.md E19; rad a visit, the translation x the scene radius; 0 = off)
  lens_lr:         0,      // each lens's log focal (the principal point x the image's width; 0 = off)
  pose_from:       1000,   // both from this step on
  pose_prior:      0,      // their pull to the SfM's: the increments held within about this (degrees; 0 = none)
  pose_refit:      0,      // every N steps (with pose_lr), each training camera refitted with the splats frozen (0 = never)
  pose_refit_iters: 20,    // ... its steps a view
  drop_rate:       0,      // DropGaussian: share of splats skipped in each training render (0 = off)
  split_min_views: 0,      // growth splits only splats seen from at least this many views (N_eff; 0 = any)
  split_mode:      'shape', // how a splat splits: 'shape' (within its own shape, as brush; DESIGN.md E9) or 'error' (along its error moments, §5.1)
  opacity_reset:   3000,   // clamp all opacities to 0.01 every N steps (0 = off)
  opacity_reset_until: 15000,  // ... up to this step (3DGS: densify_until_iter), and half the planned length (train_steps)
  // Shape limits (applied every Adam step; 0: none). Off by default, as brush: on, they shrank the long thin
  // splats of grass and branches (DESIGN.md E11: +0.4 / +0.6 dB held out on jardin5 / 7, +1.6 on cuisine's training views)
  max_splat_angle: 0,      // max axis scale / depth, in the training view (0.03 before E11)
  max_splat_scale: 0,      // max axis scale, as a fraction of the scene radius (0.1)
  max_anisotropy:  0,      // max ratio of the largest axis to the median one (10)
};

// lr_pos's factor at `step`: down exponentially to lr_pos_final over the planned length (train_steps), or
// lr_decay_steps when open-ended, then flat
export const posLrDecay = (P, step) =>
  Math.pow(P.lr_pos_final, Math.min(1, step / Math.max(1, P.train_steps > 0 ? P.train_steps : P.lr_decay_steps)));

// An Adam state starting at p (poses.js, exposure.js)
const fresh = (p) => ({ p: Float64Array.from(p), m: new Float64Array(p.length), v: new Float64Array(p.length), t: 0 });
// The angle in degrees between two rotations (row-major 3 x 3)
const turnDeg = (A, B) => {
  let tr = 0;
  for (let i = 0; i < 9; i++) tr += A[i] * B[i];
  return Math.acos(Math.min(1, (tr - 1) / 2)) * 180 / Math.PI;
};

// The shape limits as TrainUniforms.limits.xyz: max scale / depth, max world scale, log(max anisotropy); 0 is none
const NO_LIMIT = 1e6;
export const shapeLimits = (P, R) => [
  P.max_splat_angle > 0 ? P.max_splat_angle : NO_LIMIT,
  (P.max_splat_scale > 0 ? P.max_splat_scale : NO_LIMIT) * R,
  Math.log(P.max_anisotropy > 0 ? Math.max(1, P.max_anisotropy) : NO_LIMIT),
];

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

// The opacity reset (3DGS; not with monte_carlo, whose L1 term on the opacity prunes instead) at a rebalance
// at `step`, the last one at `last`: when a multiple of opacity_reset is in (last, step], up to
// opacity_reset_until and half the planned length (train_steps): the splats need the steps after the last one
// to earn their opacity back (canyon2.mp4, Adam, 30000 steps: reset at 30000 too, 9.7 dB held out against 19.4 before it)
export function opacityResetDue(P, step, last) {
  const every = resetPeriod(P);
  if (!(every > 0)) return false;
  const m = Math.floor(step / every) * every;
  return m > last && m <= resetUntil(P);
}
const resetPeriod = (P) => P.opt_method === 'monte_carlo' ? 0 : P.opacity_reset;
const resetUntil = (P) => P.train_steps > 0 ? Math.min(P.opacity_reset_until, P.train_steps / 2) : P.opacity_reset_until;

// The first step after `step` that may reset the opacities: the next multiple of opacity_reset, which rebalances
// (rebalanceDue; while consolidating, the reset waits for the next rebalance), Infinity when none is left
export function nextOpacityReset(P, step) {
  const every = resetPeriod(P);
  if (!(every > 0)) return Infinity;
  const m = (Math.floor(step / every) + 1) * every;
  return m <= resetUntil(P) ? m : Infinity;
}

// Whether the training loop (app.js) stops before its next step, at `step`: 'stop' stop_after steps after training
// started (at `start`), 'val' right before an opacity reset when not validated at this step yet (at `lastVal`), so
// validation reads the splats before the reset turns them nearly transparent, not just after (canyon2.mp4, Adam:
// 8.14 dB held out at step 3002, two steps after the reset at 3000)
export function haltBefore(P, step, lastVal, start) {
  if (P.stop_after > 0 && step - start >= P.stop_after) return 'stop';
  if (P.val_every > 0 && lastVal < step && nextOpacityReset(P, step) === step + 1) return 'val';
  return null;
}

// The splat budget at `step` (DESIGN.md §9 E7): max_splats, or with a planned length (train_steps > 0) and a
// lease (cap_lease > 0), max_splats x (1 + cap_lease) until the consolidation, the last consolidate_frac of the
// steps: there it falls linearly back to max_splats over the first 3/4, the rest trains at max_splats
export const consolidationEnd = (P) => P.train_steps * (1 - 0.25 * P.consolidate_frac);
// Whether step `step` (no rebalance), the last rebalance at `last`, starts the stipple seeds of the next one
// (GaussOptTrainer._prepareSeeds): it is due SEED_LEAD steps on, seeds it, and this window has none yet (`prepStep`,
// the step of the last ones made, or undefined). A rebalance that comes earlier seeds in the page.
export function seedPrepDue(P, step, last, count, prepStep) {
  const ahead = step + SEED_LEAD;
  if (!P.enable_rebalance || !P.stipple_seed || !(Number(P.stipple_count) > 0) || prepStep > last) return false;
  return rebalanceDue(P, ahead, last, count) && !consolidating(P, ahead);
}

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
  for (let p = 0; p < pix.length; p += 4) if (pix[p] < COVERED_T) { sum += pix[p + 3]; n++; }
  return n > 0 ? sum / n : 0;
}

// Per part, the mean over the views that have it (rgbPsnrRegions; null when none does)
export function meanRegions(perView) {
  return Object.fromEntries(Object.keys(perView[0] ?? {}).map(k => {
    const v = perView.map(r => r[k]).filter(x => x !== null);
    return [k, v.length ? v.reduce((x, y) => x + y, 0) / v.length : null];
  }));
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

// b with a's values where a's alpha is 0 (the masked pixels scored as exact)
function withMasked(b, a) {
  const out = b.slice();
  for (let o = 0; o < a.length; o += 4) if (a[o + 3] === 0) { out[o] = a[o]; out[o + 1] = a[o + 1]; out[o + 2] = a[o + 2]; }
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
    this.history       = [];  // { step, psnr, count } training-loss PSNR
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
    this.seedPrep      = null;  // { step, job }: the next rebalance's stipple seeds, in the making (_prepareSeeds)
    this.seedWorker    = null;  // stipple_worker.js, once used
    // Monte-Carlo's noise moves the faint splats with no bound: out of 1.25 times the starting octree's root (the box of
    // the initial splats; Adam's gradients never took a gallery scene's out of it), one is dead (octree.js
    // reconcileAndGrow `bounds`). Unbounded, jardin7's splats drifted 20000 units out, and its saved octree had a
    // leaf of 82k splats (the gallery's sort: 39 ms a frame).
    const root = octree.root;
    this.sceneBox = root ? { center: [root.cx, root.cy, root.cz], halfSize: 1.25 * root.halfSize } : null;
    this.exposure   = new Exposure();  // the training views' (exposure.js), by image name
    this.poses      = new Poses(octree.sceneRadius || 1);  // their poses and the lenses (poses.js)
    this.expoRing   = [];     // the names of the views whose gradients are in expoRingBuf, in its order
    this.pendingExpoRead = null;
    this._invalidateCaches();
  }

  // The training resolution (the first camera downscaled by params.downsample) and what is sized by it: the render,
  // the D-SSIM terms, the pixels' state, the tiles
  _allocResolution() {
    const cam0 = this.images[0]?.camera || { width: 512, height: 512 };
    const ds = this.ds = this.params.downsample || 4;
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
    this.ssimSumsBytes = Math.ceil(this.trainW / 16) * Math.ceil(this.trainH / 16) * 4;  // held-out SSIM, per workgroup (ssim_map)
    this.ssimSumsBuf = this._buffer('ssim_sums', this.ssimSumsBytes, BUF_RW());
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
  }

  // Train on from here at another resolution (a second pass on larger images): the splats, their Adam state and the
  // step count stay; the per-pixel buffers are made again and the ground truths reloaded at the new size
  async setDownsample(ds) {
    return this._withStepLock(async () => {
      while (this.gtLoading.size) await Promise.allSettled([...this.gtLoading.values()]);
      for (const tex of this.gtTextures.values()) tex.destroy();
      this.gtTextures.clear();
      for (const v of [this.hatTex, this.ssimCoefBuf, this.ssimSumsBuf, this.ssimGradTex, this.pixStateBuf,
                       this.tileCountsBuf, this.tileCursorBuf, this.tileRangesBuf]) v.destroy();
      this.params.downsample = ds;
      this._allocResolution();
      this._invalidateCaches();
      this.seedPrep = null;  // its view was read at the old size
      this.sortGen++;
    });
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
      this.seedWorker?.terminate();
      this.seedPrep = this.seedWorker = null;
      this.destroyed = true;
      this.images = [];
      this.count = 0;
      this.trainViews = [];
      this.valViews = [];
    });
  }

  async _init() {
    const mods = await loadShaders(this.device);
    const pipe = (name, constants = {}) => this.device.createComputePipeline({
      layout: 'auto', compute: { module: mods[name], entryPoint: 'main', constants },
    });
    this.pipeProject  = pipe('project_splats');
    this.pipeForward  = pipe('tile_forward');
    this.pipeBackward = pipe('tile_backward');
    this.pipeBackwardDir = pipe('tile_backward', { DIR_MOMS: 1 });  // + the directional error moments (split_mode 'error')
    this.pipeAdam     = pipe('adam_step');
    this.pipeCamReduce = pipe('cam_reduce');
    this.pipeEnvAdam  = pipe('env_adam');
    this.pipeSsimCoef = pipe('ssim_coef');
    this.pipeSsimGrad = pipe('ssim_grad');
    this.pipeSsimMap  = pipe('ssim_map');
    this.pipeTileScan = pipe('tile_scan');
    this.pipeTileFill = pipe('tile_fill');
    this.pipeTileSort = pipe('tile_sort');
    this.pipeSortRefine = pipe('sort_refine');
    this.refineUniformBufs = [0, 1].map(k =>
      this._buffer(`sort_refine_${k}`, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));

    this.trainUniformBuf = this._buffer('train_uniforms', UNIFORM_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.lossBuf     = this._buffer('loss_acc', EXPO_OFF + CAMG_FLOATS * 4, BUF_RW());  // + the background's and the camera's gradients
    this.expoRingBuf = this._buffer('expo_ring', EXPO_RING * CAMG_FLOATS * 4, BUF_RW());
    this.lossReadBuf = this._buffer('loss_read', 16, BUF_READ());
    // the background (env.js): rgb + pad a texel (env_adam's), Adam's moments, and the texture the tile passes read
    // (a storage buffer more there would be over WebGPU's default 8 a stage)
    this.envBuf     = this._buffer('env', ENV_TEXELS * 16, BUF_RW());
    this.envMBuf    = this._buffer('env_m', ENV_TEXELS * 16, BUF_RW());
    this.envVBuf    = this._buffer('env_v', ENV_TEXELS * 16, BUF_RW());
    this.envTex     = this.device.createTexture({ size: [ENV_N, 6 * ENV_N], format: 'rgba32float', label: 'env_tex',
                                                  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });

    this._allocResolution();
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
    for (const k of ['splatsBuf', 'momentsBuf', 'grads2dBuf', 'adamMBuf', 'adamVBuf',
                     'sortedIdxBuf', 'viewIdxBuf', 'projSplatsBuf', 'projTilesBuf', 'camPartBuf']) {
      this[k]?.destroy?.();
    }
    const splatBytes  = capacity * SPLAT_FLOATS * 4;
    const momentBytes = capacity * MOMENT_FLOATS * 4;
    const IDX = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.splatsBuf     = this._buffer('splats',   splatBytes,  BUF_RW());
    this.momentsBuf    = this._buffer('moments',  momentBytes, BUF_RW());
    this.grads2dBuf    = this._buffer('grads_2d', momentBytes, BUF_RW());
    this.adamMBuf      = this._buffer('adam_m',   splatBytes,  BUF_RW());
    this.adamVBuf      = this._buffer('adam_v',   splatBytes,  BUF_RW());
    this.sortedIdxBuf  = this._buffer('sorted_idx',      capacity * 4, IDX);
    this.viewIdxBuf    = this._buffer('view_sorted_idx', capacity * 4, IDX);
    this.projSplatsBuf = this._buffer('proj_splats',     capacity * 32, BUF_RW());  // 2 vec4f / visible splat
    this.projTilesBuf  = this._buffer('proj_tiles',      capacity * 8, BUF_RW());  // tile rect / visible splat
    this.camPartBuf    = this._buffer('cam_part', Math.ceil(capacity / 64) * 48, BUF_RW());  // adam_step's camera sums a workgroup
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
    const mask = imgInfo.camera?.mask;
    if (mask) {  // the camera's on-screen display (overlay.js) as alpha 0: tile_backward leaves those pixels out
      const [W, H] = size, ctx = new OffscreenCanvas(W, H).getContext('2d', { willReadFrequently: true });
      ctx.drawImage(bmp, 0, 0);
      const px = ctx.getImageData(0, 0, W, H).data;
      for (let y = 0; y < H; y++) {
        const my = Math.min(mask.h - 1, Math.floor((y + 0.5) * mask.h / H)) * mask.w;
        for (let x = 0; x < W; x++) if (mask.data[my + Math.min(mask.w - 1, Math.floor((x + 0.5) * mask.w / W))]) px[4 * (y * W + x) + 3] = 0;
      }
      this.device.queue.writeTexture({ texture: tex }, px, { bytesPerRow: 4 * W }, size);
    } else {
      this.device.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, size);
    }
    bmp.close();
    this.gtTextures.set(imgIdx, tex);
    return tex;
  }

  // Intrinsics of image `img` at training resolution (+ NDC-normalized focals for the sort).
  _intrinsics(img) {
    const c = img.camera;
    const fx = c.fx * this.scaleFx, fy = c.fy * this.scaleFy;
    return { fx, fy, cx: c.cx * this.scaleFx, cy: c.cy * this.scaleFy, k1: c.k1Residual ?? 0,  // poses.js
             fxNorm: 2 * fx / this.trainW, fyNorm: 2 * fy / this.trainH };
  }

  // Write TrainUniforms: view | intrinsics | dims | lrs | opt_params | limits | loss | reg | expo_g | expo_b.
  // `extra` fills f32[24..43] (lrs, opt_params, limits, loss, reg); forward-only renders leave them at 0. The view's
  // exposure always (identity for a view with none); `learnExpo`: tile_backward sums its gradient; `learnCam`: adam_step
  // sums the pose's and the lens's (cam_reduce); `camOnly`: adam_step leaves the splats (a held-out camera fitted).
  _writeUniforms(img, k, extra = null, learnExpo = false, learnCam = false, camOnly = false) {
    const buf = new ArrayBuffer(UNIFORM_BYTES);
    const f32 = new Float32Array(buf), u32 = new Uint32Array(buf);
    f32.set(img.view, 0);
    f32[16] = k.fx; f32[17] = k.fy; f32[18] = k.cx; f32[19] = k.cy;
    u32[20] = this.trainW; u32[21] = this.trainH; u32[22] = this.numVisible; u32[23] = this.count;
    if (extra) f32.set(extra, 24);
    f32.set(this.exposure.uniforms(img.name), 44);
    f32[47] = learnExpo ? 1 : 0;
    f32[52] = learnCam ? 1 : 0;
    f32[53] = camOnly ? 1 : 0;
    f32[54] = k.k1 ?? 0;
    this._learnCam = learnCam;
    this._camOnly = camOnly;
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
                                       ...lists, this.envTex.createView()], this.tilesX, this.tilesY);
    if (gtTex) {
      if (this.params.ssim > 0) {
        const wx = Math.ceil(this.trainW / 16), wy = Math.ceil(this.trainH / 16);
        this._pass(enc, this.pipeSsimCoef, [U, gtTex.createView(), this.hatTex.createView(), this.ssimCoefBuf], wx, wy);
        this._pass(enc, this.pipeSsimGrad, [U, gtTex.createView(), this.hatTex.createView(), this.ssimCoefBuf,
                                            this.ssimGradTex.createView()], wx, wy);
      }
      const backward = this.params.split_mode === 'error' ? this.pipeBackwardDir : this.pipeBackward;
      this._pass(enc, backward, [U, this.splatsBuf, this.projSplatsBuf, gtTex.createView(), this.hatTex.createView(),
                                          this.pixStateBuf, this.grads2dBuf, this.momentsBuf, this.lossBuf, ...lists,
                                          this.ssimGradTex.createView(), this.envTex.createView()],
                 this.tilesX, this.tilesY);
      if (this.params.bg_lr > 0 && !this.domeN && !this._camOnly) {
        this._pass(enc, this.pipeEnvAdam, [U, this.envBuf, this.lossBuf, this.envMBuf, this.envVBuf], Math.ceil(ENV_TEXELS / 64));
        enc.copyBufferToTexture({ buffer: this.envBuf, bytesPerRow: ENV_N * 16 }, { texture: this.envTex }, [ENV_N, 6 * ENV_N]);
      } else {
        enc.clearBuffer(this.lossBuf, 16, ENV_TEXELS * 16);  // the background's part (the exposure's is read after: _expoRecord)
      }
      // per-splat 2D->3D chain rule + Adam / MC update, over the visible splats only (the ones with grads)
      this._pass(enc, this.pipeAdam, [U, this.splatsBuf, this.grads2dBuf, this.adamMBuf, this.adamVBuf, this.momentsBuf,
                                      this.sortedIdxBuf, this.camPartBuf], Math.ceil(n / 64));
      if (this._learnCam) this._pass(enc, this.pipeCamReduce, [U, this.camPartBuf, this.lossBuf], 1);
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
  _recordLoss(mseFixed, step, count) {
    const mse = (mseFixed / 65536.0) / (this.trainW * this.trainH);
    if (mse <= 0) return;
    this.lastPsnr = -10 * Math.log10(Math.max(1e-8, mse));
    this.history.push({ step, psnr: this.lastPsnr, count });
    if (this.history.length > HISTORY_LEN) this.history.shift();
  }

  async _drainLossRead() {
    if (this.pendingLossRead) await this.pendingLossRead;
    if (this.pendingExpoRead) await this.pendingExpoRead;
  }

  // The camera's gradient of this step (the exposure's, tile_backward; the pose's and the lens's, cam_reduce; of view
  // `img`, what it learned: `what` { expo, cam }) into the ring; when it is full, the ring's copy to read back (a buffer
  // of its own: the next steps write the ring meanwhile), else null. In the step's encoder.
  _expoRecord(enc, img, what) {
    const B = CAMG_FLOATS * 4;
    enc.copyBufferToBuffer(this.lossBuf, EXPO_OFF, this.expoRingBuf, this.expoRing.length * B, B);
    this.expoRing.push({ img, ...what });
    if (this.expoRing.length < EXPO_RING) return null;
    const staging = this._buffer('expo_read', EXPO_RING * B, BUF_READ());
    enc.copyBufferToBuffer(this.expoRingBuf, 0, staging, 0, EXPO_RING * B);
    const names = this.expoRing;
    this.expoRing = [];
    return { staging, names };
  }

  // Its read, after the step's submit: one Adam step a view, then the gauge (exposure.js). The reads resolve in order.
  // The rate decays as lr_pos's (posLrDecay): at a fixed one, the gains stayed within 2% of the known ones only just
  // (exposure.e2e.js: 1.4-2.2%, lr 0.02)
  _expoRead({ staging, names }) {
    const prev = this.pendingExpoRead ?? Promise.resolve();
    const read = staging.mapAsync(GPUMapMode.READ).then(async () => {
      const g = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      await prev;
      const P = this.params, decay = posLrDecay(P, this.stepCount);  // decaying as lr_pos: Adam's steps keep their length
      const lens = new Map();  // camera -> its views' summed dL/dlog f, cx, cy (full resolution), k1
      const sigma = P.pose_prior * Math.PI / 180;  // the pull to the SfM's (poses.js)
      names.forEach((e, i) => {
        const at = i * CAMG_FLOATS, c = g.subarray(at + EXPO_FLOATS, at + CAMG_FLOATS);
        if (e.expo) this.exposure.update(e.img.name, g.subarray(at, at + EXPO_FLOATS), P.exposure_lr * decay);
        if (e.cam && P.pose_lr > 0) this.poses.update(e.img, c, P.pose_lr * decay, sigma);
        if (e.cam && P.lens_lr > 0) {
          const L = lens.get(e.img.camera) ?? [0, 0, 0, 0];
          lens.set(e.img.camera, [L[0] + c[6], L[1] + c[7] * this.scaleFx, L[2] + c[8] * this.scaleFy, L[3] + c[9]]);
        }
      });
      for (const [cam, gl] of lens) this.poses.updateLens(cam, gl, P.lens_lr * decay, sigma);
      this.exposure.center();
      if (names.some(e => e.cam)) {
        this.poses.center(new Map(this.images.map(im => [im.name, im])));
        this.poses.apply(this.images);
      }
    }).catch(() => {}).finally(() => {
      staging.destroy();
      if (this.pendingExpoRead === read) this.pendingExpoRead = null;
    });
    this.pendingExpoRead = read;
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
      const P = this.params;
      if (P.pose_refit > 0 && P.pose_lr > 0 && this.stepCount >= P.pose_from &&
          this.stepCount - (this.lastRefitStep ?? P.pose_from) >= P.pose_refit) {  // the cameras refitted (_refitCameras)
        this.lastRefitStep = this.stepCount;
        this.lastRefit = await this._refitCameras(P.pose_refit_iters);
      }
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
    // 50-step lr warmup (matters when resuming), exponential lr_pos decay to lr_pos * lr_pos_final
    const warmup = Math.min(1.0, this.stepCount / 50.0);
    const posDecay = posLrDecay(P, this.stepCount);
    const mcTerms = optMethod === 1 && !(P.polish && endPhase(P, this.stepCount));
    const learnExpo = P.exposure_lr > 0 && this.stepCount >= P.exposure_from;
    const learnCam = (P.pose_lr > 0 || P.lens_lr > 0) && this.stepCount >= P.pose_from;
    this._writeUniforms(img, k, [
      P.lr_pos * R * warmup * posDecay, P.lr_scale * warmup, P.lr_quat * warmup, P.lr_color,  // lrs
      optMethod, mcTerms ? P.mc_noise : 0, Math.random(), this.adamStepCount,  // opt_params
      ...shapeLimits(P, R), P.lr_opacity,                                                       // limits
      P.ssim, P.drop_rate, P.drop_rate > 0 ? 1 / (1 - P.drop_rate) : 0, 0,                     // loss
      ...(mcTerms ? [P.mc_opacity_reg / this.count, P.mc_scale_reg / (3 * this.count)] : [0, 0]), P.bg_lr || 0, 0,  // reg (z: env_adam's lr)
    ], learnExpo, learnCam);
    this.device.queue.writeBuffer(this.lossBuf, 0, new Uint32Array(4));
    if (learnExpo || learnCam) this.device.queue.writeBuffer(this.lossBuf, EXPO_OFF, new Uint32Array(CAMG_FLOATS));

    const needRebalance = rebalanceDue(P, this.stepCount, this.lastRebalanceStep, this.count);
    const needLossPoll  = !needRebalance && (this.stepCount % 5 === 0) && !this.pendingLossRead;
    let expoRead = null;
    this._runPasses(gtTex, enc => {
      if (needLossPoll) enc.copyBufferToBuffer(this.lossBuf, 0, this.lossReadBuf, 0, 16);
      if (learnExpo || learnCam) expoRead = this._expoRecord(enc, img, { expo: learnExpo, cam: learnCam });
    });
    if (expoRead) this._expoRead(expoRead);

    let hatTex = this.hatTex;
    if (needRebalance) {
      if (await this._rebalance()) hatTex = null;
    } else if (this.trainViews.length && seedPrepDue(P, this.stepCount, this.lastRebalanceStep, this.count, this.seedPrep?.step) &&
               await this._prepareSeeds(Math.min(Number(P.stipple_count) || 0, P.max_splats))) {
      hatTex = null;  // the seeding view's render
    }
    if (!needRebalance && needLossPoll) {
      const step = this.stepCount, count = this.count;
      this.pendingLossRead = this.lossReadBuf.mapAsync(GPUMapMode.READ, 0, 16).then(() => {
        const mseFixed = lossFixed(new Uint32Array(this.lossReadBuf.getMappedRange(0, 16)));
        this.lossReadBuf.unmap();
        this._recordLoss(mseFixed, step, count);
      }).catch(() => {}).finally(() => { this.pendingLossRead = null; });
    }
    return { viewIdx, gtTex, hatTex, psnr: this.lastPsnr };
  }

  // The background (env.js) as rgb bytes (ENV_TEXELS x 3), and set from them
  async readEnv() {
    return this._withStepLock(async () => envToBytes((await this._readback('env_read', [{ buf: this.envBuf, bytes: ENV_TEXELS * 16 }]))[0]));
  }
  // A dome (photo_runs/tools/dome.py: env.js's layout at n texels a face, rgb bytes, n x 6n x 3) in place of the learned
  // background, fixed: the tile passes read it at its own size (env_taps_tex), tile_backward sends it no gradient, and
  // bg_lr is not applied while it is on. null: back to the learned background.
  setDome(rgb, n) {
    this.envTex.destroy();
    this.domeN = rgb ? n : 0;
    const m = this.domeN || ENV_N;
    this.envTex = this.device.createTexture({ size: [m, 6 * m], format: 'rgba32float', label: this.domeN ? 'dome_tex' : 'env_tex',
                                              usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC });
    if (!rgb) {  // the learned background as it was
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToTexture({ buffer: this.envBuf, bytesPerRow: ENV_N * 16 }, { texture: this.envTex }, [ENV_N, 6 * ENV_N]);
      this.device.queue.submit([enc.finish()]);
      return;
    }
    const f32 = new Float32Array(4 * 6 * m * m);
    for (let t = 0; t < 6 * m * m; t++) for (let c = 0; c < 3; c++) f32[4 * t + c] = rgb[3 * t + c] / 255;
    this.device.queue.writeTexture({ texture: this.envTex }, f32, { bytesPerRow: m * 16 }, [m, 6 * m]);
  }
  setEnv(rgb) {
    if (this.domeN) throw new Error('setEnv: a dome is on (setDome(null) first)');
    const f32 = envFromBytes(rgb);
    this.device.queue.writeBuffer(this.envBuf, 0, f32);
    this.device.queue.writeTexture({ texture: this.envTex }, f32, { bytesPerRow: ENV_N * 16 }, [ENV_N, 6 * ENV_N]);
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
      if (this.poses.views.size) this.poses.apply(images);  // by name: the refined poses kept (the lenses: new cameras, the SfM's)
      this.octree.updateCameraAndNeighborStats(images);
      this._invalidateCaches();
      this.seedPrep = null;  // of a view of the old images
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
        avgR:          this.avgR,
        lastPsnr:      this.lastPsnr,
        history:       this.history,
        valHistory:    this.valHistory,
        exposure:      this.exposure.toJSON(),
        poses:         this.poses.toJSON(this.images),
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
    this.seedPrep = null;
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
    this.avgR        = state.avgR ?? this.avgR;
    this.lastPsnr    = state.lastPsnr ?? 0;
    this.history     = Array.isArray(state.history) ? state.history.slice(-HISTORY_LEN) : [];
    this.valHistory  = Array.isArray(state.valHistory) ? state.valHistory : [];
    this.exposure    = Exposure.fromJSON(state.exposure);
    this.poses       = Poses.fromJSON(state.poses, this.images, this.octree.sceneRadius || 1);
    this.poses.apply(this.images);
    this.expoRing    = [];
  }

  // The next step that may reset the opacities (nextOpacityReset; train.mjs checkpoints just before it)
  nextOpacityReset() { return nextOpacityReset(this.params, this.stepCount); }

  // Held-out metrics: forward-render every validation camera (never trained on) at training
  // resolution vs its GT; means of the per-view PSNR, of the PSNR after the best gain and offset per channel (psnrFit:
  // the exposure no step learned for it, metrics.js rgbPsnrAffine), SSIM and 1/4-resolution PSNR (psnr4), as
  // 3DGS averages per-view PSNRs, and of the depth spread (pix_state.w, see tile_forward) over
  // the pixels where it is measured (T_final < 0.25); regions: the means of rgbPsnrRegions' parts
  // over the views that have them (perViewRegions: each view's, in perView's order).
  async evaluateValidation() {
    if (this.valViews.length === 0 || this.count === 0) return null;
    return this._withStepLock(async () => {
      let sumPsnr = 0, sumSsim = 0, sumPsnr4 = 0, sumSpread = 0, sumFit = 0, n = 0;
      const perView = [];  // [name, dB]: the SfM panel scores keyframes that train badly (DESIGN-sfm §6.1)
      const perViewRegions = [];
      const W = this.trainW, H = this.trainH, bpr = this.rowPitch;
      try {
        for (const v of this.valViews) {
          this._lastHatKey = '';
          const out = await this._renderViewForward(v);
          if (!out) continue;
          const enc = this.device.createCommandEncoder({ label: 'val_ssim' });
          this._pass(enc, this.pipeSsimMap, [this.trainUniformBuf, out.gtTex.createView(), this.hatTex.createView(), this.ssimSumsBuf],
                     Math.ceil(W / 16), Math.ceil(H / 16));
          this.device.queue.submit([enc.finish()]);
          const [gt, hat, pix, ssimSums] = await this._readback('val_readback', [
            { tex: out.gtTex }, { tex: this.hatTex }, { buf: this.pixStateBuf, bytes: W * H * 16 }, { buf: this.ssimSumsBuf, bytes: this.ssimSumsBytes },
          ]);
          // the camera's on-screen display (alpha 0 in the ground truth) left out of the PSNRs; the 1/4-resolution
          // one sees the photo's own values there, SSIM (on the GPU) the whole frame
          const masked = !!this.images[v].camera?.mask;
          const psnr = rgbPsnr(gt, hat, W, H, bpr, { masked });
          perView.push([this.images[v].name, psnr]);
          sumPsnr += psnr;
          sumFit += rgbPsnrAffine(gt, hat, W, H, bpr, { masked }).psnr;
          sumSsim += ssimMean(ssimSums, W, H);
          sumPsnr4 += rgbPsnrDown(gt, masked ? withMasked(hat, gt) : hat, W, H, bpr, 4);
          sumSpread += meanDepthSpread(pix);
          perViewRegions.push(rgbPsnrRegions(gt, hat, W, H, bpr, pix, undefined, { masked }));
          n++;
        }
      } finally {
        this._invalidateCaches();  // shared buffers now hold the last val view
      }
      if (n === 0) return null;
      const res = { step: this.stepCount, psnr: sumPsnr / n, psnrFit: sumFit / n, ssim: sumSsim / n, psnr4: sumPsnr4 / n,
                    spread: sumSpread / n, regions: meanRegions(perViewRegions), count: this.count, views: n };
      this.valHistory.push(res);
      return { ...res, perView, perViewRegions };
    });
  }

  // Runs fn(extra) with adam_step in its camera-only mode (the splats, their Adam state and the background left as
  // they are): the moments, which tile_backward adds to, saved and put back after; `extra` the uniforms' f32[24..43]
  // for a fit (no learning rates). Needs the step lock.
  async _cameraOnly(fn) {
    await this._drainLossRead();
    const mBytes = this.count * MOMENT_FLOATS * 4, P = this.params;
    const saved = this._buffer('moments_saved', mBytes, BUF_RW());
    let enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.momentsBuf, 0, saved, 0, mBytes);
    this.device.queue.submit([enc.finish()]);
    const extra = [0, 0, 0, 0, 0, 0, 0, 1, ...shapeLimits(P, this.octree.sceneRadius), 0, P.ssim, 0, 0, 0, 0, 0, 0, 0];
    try {
      return await fn(extra);
    } finally {
      enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(saved, 0, this.momentsBuf, 0, mBytes);
      this.device.queue.submit([enc.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      saved.destroy();
      this._writeUniforms(this.images[0], this._intrinsics(this.images[0]));  // camOnly off
      this._invalidateCaches();
    }
  }

  // `iters` steps of view `img`'s pose (in `pose`, a Poses with its state) and exposure (in `expo`, the same) on its
  // photo, inside _cameraOnly: Adam at rates falling from 1e-3 rad (the move x the scene radius) and 0.01 to a tenth,
  // the pose pulled to the SfM's by sigma (poses.js); the image left at the fitted pose
  async _fitCamera(img, gtTex, pose, expo, iters, extra, sigma = 0) {
    pose.apply([img]);
    for (let i = 0; i < iters; i++) {
      const k = this._intrinsics(img), f = Math.pow(0.1, i / Math.max(1, iters - 1));
      if (this.updateSortingForView(img.center, img.view, k.fxNorm, k.fyNorm) === 0) break;
      this._writeUniforms(img, k, extra, true, true, true);
      this.device.queue.writeBuffer(this.lossBuf, 0, new Uint32Array(4));
      this.device.queue.writeBuffer(this.lossBuf, EXPO_OFF, new Uint32Array(CAMG_FLOATS));
      this._runPasses(gtTex);
      const [l] = await this._readback('fit_read', [{ buf: this.lossBuf, bytes: EXPO_OFF + CAMG_FLOATS * 4 }]);
      const g = l.subarray(EXPO_OFF / 4);
      expo.update(img.name, g.subarray(0, EXPO_FLOATS), 0.01 * f);
      pose.update(img, g.subarray(EXPO_FLOATS, EXPO_FLOATS + 6), 1e-3 * f, sigma);
      pose.apply([img]);
    }
  }

  // The held-out views scored after their camera is fitted to their photo with the splats frozen (DESIGN.md E19's
  // held-out protocol): from its neighbours' pose and exposure (poses.js, exposure.js), `iters` steps of its pose and
  // exposure only (_fitCamera), then its PSNR (masked as evaluateValidation's). Each view's pose put back after.
  // { psnr: the mean, perView: [name, dB, its pose's turn in degrees and move x the radius from where it started] }
  async evaluateHeldOutFitted(iters = 40) {
    if (this.valViews.length === 0 || this.count === 0) return null;
    return this._withStepLock(() => this._cameraOnly(async (extra) => {
      const W = this.trainW, H = this.trainH, bpr = this.rowPitch, perView = [];
      for (const v of this.valViews) {
        const img = this.images[v], gtTex = await this.ensureGtTexture(v);
        if (!gtTex) continue;
        const { R, t } = img, c0 = img.center.slice(), expoHad = this.exposure.views.get(img.name);
        const pose = new Poses(this.poses.radius), expo = this.exposure;
        pose.views.set(img.name, fresh(this.poses._between(img) ?? new Float64Array(6)));
        expo.views.set(img.name, fresh(expoHad?.p ?? expo._between(img.name) ?? new Float64Array(6)));
        await this._fitCamera(img, gtTex, pose, expo, iters, extra);
        this._lastHatKey = '';
        const out = await this._renderViewForward(v), [gt, hat] = await this._readback('fit_val', [{ tex: out.gtTex }, { tex: this.hatTex }]);
        perView.push([img.name, rgbPsnr(gt, hat, W, H, bpr, { masked: !!img.camera?.mask }), turnDeg(img.R, R),
                      Math.hypot(...img.center.map((x, k) => x - c0[k])) / this.poses.radius]);
        setPose(img, R, t);
        if (expoHad) expo.views.set(img.name, expoHad); else expo.views.delete(img.name);
      }
      if (!perView.length) return null;
      return { psnr: perView.reduce((a, x) => a + x[1], 0) / perView.length, perView, iters };
    }));
  }

  // The training views' cameras refitted with the splats frozen (pose_refit, DESIGN.md E19 2): in training, a view's
  // pose takes one Adam step a visit (a pass over the views: ~120 in a 20000-step run of the classroom's 164 views,
  // its pose still 0.08 degree off the true one); here each takes `iters` (_fitCamera, from its current pose and
  // exposure, the pull pose_prior's), kept: the pose's increment and, when the exposures learn, the exposure, into the
  // trainer's states (their Adam moments kept); then the gauge. Needs the step lock. { views, turn, move (means from
  // where they were, degrees and x the radius), ms }
  async _refitCameras(iters) {
    const t0 = performance.now(), P = this.params, learnExpo = P.exposure_lr > 0 && this.stepCount >= P.exposure_from;
    const sigma = P.pose_prior * Math.PI / 180;
    const moved = await this._cameraOnly(async (extra) => {
      const out = [];
      for (const v of this.trainViews) {
        const img = this.images[v], gtTex = await this.ensureGtTexture(v);
        if (!gtTex) continue;
        const R0 = img.R.slice(), c0 = img.center.slice(), expoHad = this.exposure.views.get(img.name);
        const pose = new Poses(this.poses.radius), expo = this.exposure;
        pose.views.set(img.name, fresh(this.poses.views.get(img.name)?.p ?? new Float64Array(6)));
        expo.views.set(img.name, fresh(expoHad?.p ?? new Float64Array(6)));
        await this._fitCamera(img, gtTex, pose, expo, iters, extra, sigma);
        if (!this.poses.views.has(img.name)) this.poses.update(img, [0, 0, 0, 0, 0, 0], 0);  // a state for it
        this.poses.views.get(img.name).p.set(pose.views.get(img.name).p);
        const fitted = expo.views.get(img.name);
        if (expoHad) expo.views.set(img.name, expoHad); else expo.views.delete(img.name);
        if (learnExpo) { if (expoHad) expoHad.p.set(fitted.p); else expo.views.set(img.name, fitted); }
        out.push([turnDeg(img.R, R0), Math.hypot(...img.center.map((x, k) => x - c0[k])) / this.poses.radius]);
      }
      return out;
    });
    this.poses.center(new Map(this.images.map(im => [im.name, im])));
    this.poses.apply(this.images);
    this.exposure.center();
    const mean = (j) => moved.reduce((a, x) => a + x[j], 0) / Math.max(1, moved.length);
    return { step: this.stepCount, views: moved.length, iters, turn: mean(0), move: mean(1), ms: performance.now() - t0 };
  }

  // Stipple seeding: render the next training view, read back GT / hat / pix_state and turn
  // the under-fit residual into up to `budget` new splats (see stipple.js). Needs the step lock.
  // In the page, when _prepareSeeds did not make them ahead.
  async _stippleSeeds(budget) {
    if (budget <= 0 || this.trainViews.length === 0) return null;
    const t0 = performance.now(), r = await this._seedRead();
    if (!r) return null;
    const [gt, hat, pix] = await r.read;
    const seeds = seedsFromView({ gt, hat, pix, budget, seed: 1234 + this.stepCount, ...r.args });
    this.lastSeedInfo = { viewIdx: r.viewIdx, count: seeds.count, covered: seeds.covered,
                          fromPoints: seeds.fromPoints, fromMedian: seeds.fromMedian, ms: performance.now() - t0 };
    return seeds.params;
  }

  // Renders the next seeding view into hatTex and queues the readback of its GT, hat and pix_state (queued now: the
  // next step's renders come after it). { viewIdx, read: promise of [gt, hat, pix], args: the rest of seedsFromView's }
  async _seedRead() {
    const viewIdx = this.trainViews[this.seedCursor++ % this.trainViews.length];
    this._lastHatKey = '';
    const out = await this._renderViewForward(viewIdx);
    if (!out || this.numVisible === 0) return null;
    const W = this.trainW, H = this.trainH, img = this.images[viewIdx];
    const read = this._readback('stipple_readback', [
      { tex: out.gtTex }, { tex: this.hatTex }, { buf: this.pixStateBuf, bytes: W * H * 16 },
    ]);
    const { fx, fy, cx, cy } = this._intrinsics(img);
    const depthGrid = this.points3D
      ? buildViewDepthGrid(img, this.points3D, this.scaleFx, this.scaleFy, W, H)
      : { grid: new Float32Array(1), gw: 1, gh: 1, cell: Math.max(W, H), median: 0.5 * (img.zMin + img.zMax) };
    return { viewIdx, read, args: { W, H, bytesPerRow: this.rowPitch, img: { R: img.R, center: img.center }, fx, fy, cx, cy, depthGrid, masked: !!img.camera?.mask } };
  }

  // The seeds of the rebalance due in SEED_LEAD steps, made off the page's thread (stipple_worker.js) while the
  // steps in between train: the view is read back now, the rebalance takes them (this.seedPrep) unless one came
  // first. Their residual is that many steps older. Needs the step lock; true when hatTex holds the seeding view.
  async _prepareSeeds(budget) {
    const t0 = performance.now(), step = this.stepCount, r = await this._seedRead();
    if (!r) return false;
    const job = r.read.then(([gt, hat, pix]) => this._seedJob({ gt, hat, pix, budget, seed: 1234 + step, ...r.args }))
      .then(s => ({ ...s, info: { viewIdx: r.viewIdx, count: s.count, covered: s.covered, fromPoints: s.fromPoints,
                                  fromMedian: s.fromMedian, ms: performance.now() - t0, ahead: step } }));
    job.catch(() => {});  // a failed job: the rebalance seeds in the page
    this.seedPrep = { step, job };
    return true;
  }

  // seedsFromView in the worker (in the page without workers); each answer goes to its job by id
  _seedJob(args) {
    if (typeof Worker === 'undefined') return seedsFromView(args);
    if (!this.seedWorker) {
      this.seedWorker = new Worker(new URL('./stipple_worker.js', import.meta.url), { type: 'module' });
      this.seedJobs = new Map();
      this.seedWorker.onmessage = ({ data }) => {
        const j = this.seedJobs.get(data.id);
        this.seedJobs.delete(data.id);
        if (data.error) j?.reject(new Error(data.error)); else j?.resolve(data);
      };
      this.seedWorker.onerror = (e) => { for (const j of this.seedJobs.values()) j.reject(new Error(e.message)); this.seedJobs.clear(); };
    }
    const id = (this.seedJobId = (this.seedJobId ?? 0) + 1);
    return new Promise((resolve, reject) => {
      this.seedJobs.set(id, { resolve, reject });
      this.seedWorker.postMessage({ ...args, id }, [args.gt.buffer]);  // gt, hat and pix share it
    });
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
    const [loss, params, moments, adamM0, adamV0] = await this._readback('rebalance_read', [
      { buf: this.lossBuf,    bytes: 16 },
      { buf: this.splatsBuf,  bytes: splatBytes },
      { buf: this.momentsBuf, bytes: this.count * MOMENT_FLOATS * 4 },
      { buf: this.adamMBuf,   bytes: splatBytes },
      { buf: this.adamVBuf,   bytes: splatBytes },
    ]);
    this._recordLoss(lossFixed(new Uint32Array(loss.buffer, loss.byteOffset, 2)), this.stepCount, this.count);
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

    let src, hatUsed = false;

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
        const prep = this.seedPrep?.step > this.lastRebalanceStep ? await this.seedPrep.job.catch(() => null) : null;
        this.seedPrep = null;
        if (prep) {
          seedSplats = prep.params;
          this.lastSeedInfo = prep.info;
        } else {
          hatUsed = nSeeds > 0;
          seedSplats = await this._stippleSeeds(nSeeds);
        }
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
        splitMode:      P.split_mode,
        visitsPerView:  Math.max(1, (this.stepCount - this.lastRebalanceStep) / Math.max(1, this.trainViews.length)),
        enableSplit:    P.enable_split,
        enableMerge:    P.enable_merge,
        optMethod:      P.opt_method,
        relocateBy:     P.mc_relocate,
        relocateMode:   P.mc_relocate_mode,
        bounds:         P.opt_method === 'monte_carlo' ? this.sceneBox : null,
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
    if (lease) this.vis = remapVisibility(this.vis, src);
    lap('gather');

    // Periodic opacity reset (3DGS): cap alpha at 0.01, so splats that don't earn their
    // opacity back before the next prune get removed (opacityResetDue)
    if (opacityResetDue(P, this.stepCount, this.lastRebalanceStep)) {
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
