// trainer.js — Hybrid GPU/CPU 3DGS optimizer: GPU inner loop (project -> tile forward ->
// tile backward + WLBG moments -> Adam / Monte-Carlo / Simulated Annealing update), CPU outer
// loop every `rebalance_every` steps (prune / grow / floaters / octree rebuild, see octree.js).
"use strict";

import { SPLAT_FLOATS } from './colmap.js';
import { MOMENT_FLOATS } from './octree.js';
import { loadShaders } from './render.js';
import { residualDensity, stippleSites, buildViewDepthGrid, makeSeedSplats } from './stipple.js';

const UNIFORM_BYTES = 160;     // TrainUniforms (gaussopt-shaders.wgsl): 40 x f32
const HISTORY_LEN   = 300;     // training-PSNR history kept for the telemetry plot
const OPT_METHODS   = { adam: 0, monte_carlo: 1, sim_anneal: 2 };
const BUF_RW   = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const BUF_READ = () => GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

// PSNR over the RGB channels of two rgba8 images with row pitch `bpr`.
function rgbPsnr(a, b, W, H, bpr) {
  let sumSq = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0, o = y * bpr; x < W; x++, o += 4) {
      for (let c = 0; c < 3; c++) {
        const d = (a[o + c] - b[o + c]) / 255;
        sumSq += d * d;
      }
    }
  }
  const mse = sumSq / (W * H * 3);
  return (mse < 1e-10) ? 99.9 : -10 * Math.log10(mse);
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
  static async create(device, octree, initialSplats, images, params) {
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
    this.numVisibleView = 0;  // display-view sort (viewIdxBuf)
    this.sortGen    = 0;      // bumped whenever cpuParams / octree change: invalidates display sort

    this.stepCount     = 0;
    this.adamStepCount = 0;   // steps since the Adam state was last reset (bias correction)
    this.stepInFlight  = false;  // step lock, see _withStepLock()
    this.avgR          = 0.01;
    this.lastPsnr      = 0;
    this.currentTemp   = params.sa_temp || 0.15;
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
    this._lastSortKey = this._lastViewSortKey = this._lastHatKey = '';
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
    this.pipeTileScan = pipe('tile_scan');
    this.pipeTileFill = pipe('tile_fill');
    this.pipeTileSort = pipe('tile_sort');

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
    this.tileEntries = 0;
    this._allocateTileLists(1 << 20);
    this._allocateSplatBuffers(Math.max(this.count * 2, 8192));
    this._uploadSplats();

    this.images.forEach((im, i) => (im.isValidation ? this.valViews : this.trainViews).push(i));
    if (this.trainViews.length === 0 && this.images.length > 0) this.trainViews = [0];
  }

  _buffer(label, size, usage) {
    return this.device.createBuffer({ label, size: Math.max(4, size), usage });
  }

  _allocateSplatBuffers(capacity) {
    this.capacity = capacity;
    for (const k of ['splatsBuf', 'momentsBuf', 'grads2dBuf', 'adamMBuf', 'adamVBuf',
                     'sortedIdxBuf', 'viewIdxBuf', 'projSplatsBuf', 'projTilesBuf']) {
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
    this.projSplatsBuf = this._buffer('proj_splats',     capacity * 32, IDX);  // 2 vec4f / visible splat
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
    if (adam) {
      q.writeBuffer(this.adamMBuf, 0, adam.adamM, 0, splatFloats);
      q.writeBuffer(this.adamVBuf, 0, adam.adamV, 0, splatFloats);
    } else {
      const zeroAdam = new Uint32Array(splatFloats);
      q.writeBuffer(this.adamMBuf, 0, zeroAdam);
      q.writeBuffer(this.adamVBuf, 0, zeroAdam);
      this.adamStepCount = 0;
    }
    this.sortGen++;
    this._lastSortKey = this._lastViewSortKey = this._lastHatKey = '';
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

  // Write TrainUniforms: view | cam_pos (w = adam step) | intrinsics | dims | lrs | opt_params | limits.
  // `extra` fills f32[28..39] (lrs, opt_params, limits); forward-only renders leave them at 0.
  _writeUniforms(img, k, extra = null) {
    const buf = new ArrayBuffer(UNIFORM_BYTES);
    const f32 = new Float32Array(buf), u32 = new Uint32Array(buf);
    f32.set(img.view, 0);
    f32.set(img.center.slice(0, 3), 16);
    f32[19] = this.adamStepCount;
    f32[20] = k.fx; f32[21] = k.fy; f32[22] = k.cx; f32[23] = k.cy;
    u32[24] = this.trainW; u32[25] = this.trainH; u32[26] = this.numVisible; u32[27] = this.count;
    if (extra) f32.set(extra, 28);
    this.device.queue.writeBuffer(this.trainUniformBuf, 0, buf);
  }

  // Compute pass with bind group 0 = `resources` (GPUBuffer or texture view, binding = index).
  _pass(enc, pipeline, resources, wx, wy = 1) {
    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((r, i) => ({ binding: i, resource: (r instanceof GPUBuffer) ? { buffer: r } : r })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wx, wy);
    pass.end();
  }

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
  // moments, loss) and the param update. extra(enc) appends commands.
  // The list capacity follows the entry count read back asynchronously: an overflow
  // (entries dropped for that render) grows it for the next one.
  async _runPasses(gtTex = null, extra = null) {
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
      this._pass(enc, this.pipeBackward, [U, this.splatsBuf, this.projSplatsBuf, gtTex.createView(), this.hatTex.createView(),
                                          this.pixStateBuf, this.grads2dBuf, this.momentsBuf, this.lossBuf, ...lists],
                 this.tilesX, this.tilesY);
      // per-splat 2D->3D chain rule + Adam / MC / SA update
      this._pass(enc, this.pipeAdam, [U, this.splatsBuf, this.grads2dBuf, this.adamMBuf, this.adamVBuf, this.momentsBuf],
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
      this.tileEntries = total;
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

  // Sort the visible splats for the training camera into sortedIdxBuf (cached per step & pose).
  updateSortingForView(camCenter, view, fxNorm, fyNorm) {
    const sortKey = `${this.count}:${this.stepCount}:` +
      `${view[2].toFixed(4)},${view[6].toFixed(4)},${view[10].toFixed(4)},${view[14].toFixed(3)},` +
      `${view[0].toFixed(4)},${view[5].toFixed(4)},${fxNorm.toFixed(3)},${fyNorm.toFixed(3)}`;
    if (this._lastSortKey === sortKey && this.numVisible > 0) return this.numVisible;
    this._lastSortKey = sortKey;
    this.numVisible = this.octree.sortOctreeFrontToBack(
      this.cpuParams, camCenter, view, fxNorm, fyNorm, this.sortedIdx).numVisible;
    if (this.numVisible > 0) this.device.queue.writeBuffer(this.sortedIdxBuf, 0, this.sortedIdx, 0, this.numVisible);
    return this.numVisible;
  }

  // Sort for the on-screen view into viewIdxBuf, cached independently of the training sort.
  // cpuParams only change on readback (sortGen bump), so a still view costs no CPU while training.
  sortForDisplay(camCenter, view, fxNorm, fyNorm) {
    let key = `${this.sortGen}:${this.count}:${fxNorm.toFixed(3)},${fyNorm.toFixed(3)}`;
    for (const i of [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]) key += `,${view[i].toFixed(4)}`;
    if (this._lastViewSortKey === key) return this.numVisibleView;
    this._lastViewSortKey = key;
    this.numVisibleView = this.octree.sortOctreeFrontToBack(
      this.cpuParams, camCenter, view, fxNorm, fyNorm, this.viewIdx).numVisible;
    if (this.numVisibleView > 0) this.device.queue.writeBuffer(this.viewIdxBuf, 0, this.viewIdx, 0, this.numVisibleView);
    return this.numVisibleView;
  }

  // Forward-render COLMAP view `viewIdx` into hatTex without touching the parameters
  // (compare view). Backs off (null) while a step / E1 / snapshot holds the step lock.
  async renderViewForward(viewIdx) {
    if (this.stepInFlight) return null;
    return this._renderViewForward(viewIdx);
  }

  // Lock-free version, for callers already holding the step lock. Cached per (view, step, count).
  async _renderViewForward(viewIdx) {
    if (this.images.length === 0 || this.count === 0) return null;
    const idx = Math.min(Math.max(0, viewIdx | 0), this.images.length - 1);
    const gtTex = await this.ensureGtTexture(idx);
    if (!gtTex) return null;
    const out = { viewIdx: idx, gtTex, hatTex: this.hatTex };
    const hatKey = `${idx}:${this.stepCount}:${this.count}`;
    if (this._lastHatKey === hatKey) return out;

    const img = this.images[idx];
    const k = this._intrinsics(img);
    if (this.updateSortingForView(img.center, img.view, k.fxNorm, k.fyNorm) === 0) return out;
    this._writeUniforms(img, k);
    await this._runPasses();
    this._lastHatKey = hatKey;
    return out;
  }

  // One GPU training iteration (next training camera, or `forcedCamIdx`).
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
    // SA temperature T = T0 * cooling^(t/50), t optionally restarting every 2 rebalances
    const cycleSteps = P.sa_reheat ? (this.stepCount % Math.max(100, P.rebalance_every * 2)) : this.stepCount;
    this.currentTemp = (P.sa_temp || 0.15) * Math.pow(P.sa_cooling || 0.96, cycleSteps / 50.0);
    // 50-step lr warmup (matters when resuming), exponential lr_pos decay to lr_pos * lr_pos_final
    const warmup = Math.min(1.0, this.stepCount / 50.0);
    const decayT = Math.min(1, this.stepCount / Math.max(1, P.lr_decay_steps || 30000));
    const posDecay = Math.pow(P.lr_pos_final ?? 0.01, decayT);
    this._writeUniforms(img, k, [
      P.lr_pos * R * warmup * posDecay, P.lr_scale * warmup, P.lr_quat * warmup, P.lr_color,  // lrs
      optMethod, (optMethod === 1) ? (P.mc_noise || 0.08) : this.currentTemp, Math.random(), R,  // opt_params
      P.max_splat_angle ?? 0.03, (P.max_splat_scale ?? 0.1) * R,                              // limits
      Math.log(Math.max(1, P.max_anisotropy ?? 10)), P.lr_opacity ?? 0.05,
    ]);
    this.device.queue.writeBuffer(this.lossBuf, 0, new Uint32Array(4));

    const needRebalance = (this.stepCount % P.rebalance_every === 0);
    const needLossPoll  = !needRebalance && (this.stepCount % 5 === 0) && !this.pendingLossRead;
    await this._runPasses(gtTex, needLossPoll ? enc => enc.copyBufferToBuffer(this.lossBuf, 0, this.lossReadBuf, 0, 16) : null);

    if (needRebalance) {
      await this._rebalance();
    } else if (needLossPoll) {
      const step = this.stepCount, count = this.count, temp = this.currentTemp;
      this.pendingLossRead = this.lossReadBuf.mapAsync(GPUMapMode.READ, 0, 16).then(() => {
        const [lo, hi] = new Uint32Array(this.lossReadBuf.getMappedRange(0, 16));
        const mseFixed = hi * 4294967296 + lo;
        this.lossReadBuf.unmap();
        this._recordLoss(mseFixed, step, count, temp);
      }).catch(() => {}).finally(() => { this.pendingLossRead = null; });
    }
    return { viewIdx, gtTex, hatTex: this.hatTex, psnr: this.lastPsnr };
  }

  // Copy the GPU splats into cpuParams (e.g. before a .ply export).
  async syncSplatsFromGpu() {
    return this._withStepLock(async () => {
      await this._drainLossRead();
      const [params] = await this._readback('sync_splats', [{ buf: this.splatsBuf, bytes: this.count * SPLAT_FLOATS * 4 }]);
      this.cpuParams.set(params);
      this.sortGen++;
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
      this.sortGen++;
      return {
        count:         this.count,
        stepCount:     this.stepCount,
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
    this.currentTemp = state.currentTemp ?? this.currentTemp;
    this.avgR        = state.avgR ?? this.avgR;
    this.lastPsnr    = state.lastPsnr ?? 0;
    this.history     = Array.isArray(state.history) ? state.history.slice(-HISTORY_LEN) : [];
    this.valHistory  = Array.isArray(state.valHistory) ? state.valHistory : [];
  }

  // Held-out PSNR: forward-render every validation camera (never trained on) at training
  // resolution vs its GT; mean of per-view PSNRs (3DGS convention).
  async evaluateValidation() {
    if (this.valViews.length === 0 || this.count === 0) return null;
    return this._withStepLock(async () => {
      let sumPsnr = 0, n = 0;
      try {
        for (const v of this.valViews) {
          this._lastHatKey = '';
          const out = await this._renderViewForward(v);
          if (!out) continue;
          const [gt, hat] = await this._readback('val_readback', [{ tex: out.gtTex }, { tex: this.hatTex }]);
          sumPsnr += rgbPsnr(gt, hat, this.trainW, this.trainH, this.rowPitch);
          n++;
        }
      } finally {
        this._lastSortKey = this._lastHatKey = '';  // shared buffers now hold the last val view
      }
      if (n === 0) return null;
      const res = { step: this.stepCount, psnr: sumPsnr / n, count: this.count, views: n };
      this.valHistory.push(res);
      return res;
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
  async _rebalance() {
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
    const [lo, hi] = new Uint32Array(loss.buffer, loss.byteOffset, 2);
    this._recordLoss(hi * 4294967296 + lo, this.stepCount, this.count, this.currentTemp);
    this.cpuParams.set(params);

    // mean error mass R_i over active splats (diagnostic heatmap scale)
    let sumR = 0, active = 0;
    for (let i = 0; i < this.count; i++) {
      const r = moments[i * MOMENT_FLOATS + 2];
      if (r > 0) { sumR += r; active++; }
    }
    if (active > 0) this.avgR = sumR / active;

    // src[i] = pre-rebalance index of new splat i (-1 = new child, Adam starts from zero)
    let src;
    if (P.enable_rebalance) {
      this.lastSeedInfo = null;
      let seedSplats = null;
      if (P.stipple_seed) {
        const budget = Math.min(Number(P.stipple_count) || 0, Math.max(0, P.max_splats - this.count));
        seedSplats = await this._stippleSeeds(budget);
        if (budget === 0) this.lastSeedInfo = { count: 0, capped: true };
      }
      const res = this.octree.reconcileAndGrow(this.cpuParams, this.count, moments, this.images, {
        maxSplats:      P.max_splats,
        pruneOpacity:   P.prune_opacity,
        hysteresis:     P.hysteresis,
        enableFloaters: P.remove_floaters,
        enableSplit:    P.enable_split,
        enableMerge:    P.enable_merge,
        optMethod:      P.opt_method,
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
    } else {
      this.octree.build(this.cpuParams, this.count, this.images);  // re-bin drifted splats only
      src = this.octree.lastPerm;
    }
    const adamM = gatherRows(adamM0, src, SPLAT_FLOATS);
    const adamV = gatherRows(adamV0, src, SPLAT_FLOATS);

    // Periodic opacity reset (3DGS): cap alpha at 0.01, so splats that don't earn their
    // opacity back before the next prune get removed
    const every = P.opacity_reset || 0;
    if (every > 0 && Math.floor(this.stepCount / every) > Math.floor((this.stepCount - P.rebalance_every) / every)) {
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
  }

  // Experiment E1 sanity check (DESIGN.md §9): octree cull + radix sort vs exhaustive exact
  // z-sort over the same splats; both should render identically (99.9 dB).
  async runExperimentE1(camIdx = 0) {
    if (this.images.length === 0) return 'No cameras loaded';
    try {
      return await this._withStepLock(() => this._runExperimentE1(camIdx));
    } finally {
      this._lastSortKey = this._lastHatKey = '';  // shared buffers now hold E1's sort / render
    }
  }

  async _runExperimentE1(camIdx) {
    const img = this.images[camIdx];
    const k = this._intrinsics(img);
    // sortFn fills this.sortedIdx and returns the visible count
    const render = async (sortFn) => {
      const t0 = performance.now();
      this.numVisible = sortFn();
      const sortMs = performance.now() - t0;
      this.device.queue.writeBuffer(this.sortedIdxBuf, 0, this.sortedIdx, 0, this.numVisible);
      this._writeUniforms(img, k);
      await this._runPasses();
      const [pixels] = await this._readback('e1_read', [{ tex: this.hatTex }]);
      return { pixels, sortMs, n: this.numVisible };
    };
    const ref = await render(() =>
      this.octree.sortGlobalExact(this.cpuParams, this.count, img.view, k.fxNorm, k.fyNorm, this.sortedIdx));
    const oct = await render(() =>
      this.octree.sortOctreeFrontToBack(this.cpuParams, img.center, img.view, k.fxNorm, k.fyNorm, this.sortedIdx).numVisible);
    const psnr = rgbPsnr(ref.pixels, oct.pixels, this.trainW, this.trainH, this.rowPitch);
    const summary = `E1 cam #${camIdx}: exhaustive z-sort ${ref.n} vis (${ref.sortMs.toFixed(1)}ms) vs ` +
      `octree cull + radix ${oct.n} vis (${oct.sortMs.toFixed(1)}ms) -> ${psnr.toFixed(2)} dB`;
    console.log('[Experiment E1]', summary);
    return summary;
  }
}
