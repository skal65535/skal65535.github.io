// gaussopt-trainer.js — Hybrid GPU/CPU 3DGS Optimizer with Loose Octree, WLBG Moments,
//                       Monte-Carlo (SGLD/MCMC), and Simulated Annealing (SA).
"use strict";

import { SPLAT_FLOATS } from './gaussopt-colmap.js';
import { MOMENT_FLOATS } from './gaussopt-octree.js';
import { loadShaders } from './gaussopt-render.js';

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
    this.cpuMoments = new Float32Array(Math.max(1, this.count) * MOMENT_FLOATS);
    this.sortedIdx  = new Uint32Array(Math.max(1024, this.count * 2));
    this.numVisible = this.count;

    this.stepCount     = 0;
    this.adamStepCount = 0;
    this.stepInFlight  = false;
    this.avgR          = 0.01;
    this.lastPsnr      = 0;
    this.currentTemp   = params.sa_temp || 0.15;
    this.history       = [];
    this.splitVectors  = null;
    this.trainViews    = [];
    this.valViews      = [];
    this.gtTextures    = new Map(); // image index -> GPUTexture
    this.pendingRead   = null;
  }

  async _init() {
    const mods = await loadShaders(this.device);

    this.pipeProject = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: mods.project_splats, entryPoint: 'main' },
    });
    this.pipeForward = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: mods.tile_forward, entryPoint: 'main' },
    });
    this.pipeBackward = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: mods.tile_backward, entryPoint: 'main' },
    });
    this.pipeAdam = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: mods.adam_step, entryPoint: 'main' },
    });

    this.trainUniformBuf = this.device.createBuffer({
      size: 160,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'train_uniforms',
    });
    this.lossBuf = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'loss_acc',
    });
    this.lossReadBuf = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'loss_read',
    });

    // Determine training resolution (downscaled by params.downsample for interactive speed)
    const firstCam = this.images[0]?.camera || { width: 512, height: 512, fx: 400, fy: 400, cx: 256, cy: 256 };
    const ds = this.params.downsample || 4;
    this.trainW  = Math.max(64, Math.floor(firstCam.width  / ds));
    this.trainH  = Math.max(64, Math.floor(firstCam.height / ds));
    this.scaleFx = this.trainW / firstCam.width;
    this.scaleFy = this.trainH / firstCam.height;

    this._allocateRenderTargets(this.trainW, this.trainH);
    this._allocateSplatBuffers(Math.max(this.count * 2, 8192));
    this._uploadSplats();

    this.trainViews = [];
    this.valViews   = [];
    for (let i = 0; i < this.images.length; i++) {
      if (this.images[i].isValidation) this.valViews.push(i);
      else                             this.trainViews.push(i);
    }
    if (this.trainViews.length === 0 && this.images.length > 0) {
      this.trainViews = [0];
    }
  }

  _allocateRenderTargets(W, H) {
    this.hatTex?.destroy?.();
    this.pixStateBuf?.destroy?.();

    this.hatTex = this.device.createTexture({
      size: [W, H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      label: 'hat_tex',
    });
    this.pixStateBuf = this.device.createBuffer({
      size: W * H * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'pix_state',
    });
  }

  _allocateSplatBuffers(capacity) {
    this.capacity = capacity;
    for (const k of ['splatsBuf', 'splatsReadBuf', 'momentsBuf', 'momentsReadBuf',
                     'grads2dBuf', 'adamMBuf', 'adamVBuf', 'sortedIdxBuf', 'projSplatsBuf']) {
      this[k]?.destroy?.();
    }

    const splatBytes  = capacity * SPLAT_FLOATS * 4;
    const momentBytes = capacity * MOMENT_FLOATS * 4;

    this.splatsBuf = this.device.createBuffer({
      size: splatBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'splats',
    });
    this.splatsReadBuf = this.device.createBuffer({
      size: splatBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'splats_read',
    });
    this.momentsBuf = this.device.createBuffer({
      size: momentBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'moments',
    });
    this.momentsReadBuf = this.device.createBuffer({
      size: momentBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'moments_read',
    });
    this.grads2dBuf = this.device.createBuffer({
      size: momentBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'grads_2d',
    });
    this.adamMBuf = this.device.createBuffer({
      size: splatBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'adam_m',
    });
    this.adamVBuf = this.device.createBuffer({
      size: splatBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'adam_v',
    });
    this.sortedIdxBuf = this.device.createBuffer({
      size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'sorted_idx',
    });
    this.projSplatsBuf = this.device.createBuffer({
      size: capacity * 8 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'proj_splats',
    });

    this.sortedIdx  = new Uint32Array(capacity);
    this.cpuMoments = new Float32Array(capacity * MOMENT_FLOATS);
  }

  _uploadSplats() {
    if (this.count > this.capacity) {
      this._allocateSplatBuffers(Math.max(this.count * 2, this.capacity * 2));
    }
    this.device.queue.writeBuffer(
      this.splatsBuf, 0,
      this.cpuParams.buffer, this.cpuParams.byteOffset, this.count * SPLAT_FLOATS * 4
    );
    const zeroMoments = new Uint32Array(this.count * MOMENT_FLOATS);
    const zeroAdam    = new Uint32Array(this.count * SPLAT_FLOATS);
    this.device.queue.writeBuffer(this.momentsBuf, 0, zeroMoments);
    this.device.queue.writeBuffer(this.grads2dBuf, 0, zeroMoments);
    this.device.queue.writeBuffer(this.adamMBuf,   0, zeroAdam);
    this.device.queue.writeBuffer(this.adamVBuf,   0, zeroAdam);
    this.adamStepCount = 0;
  }

  // Load and cache a ground-truth image texture at (trainW, trainH)
  async ensureGtTexture(imgIdx) {
    if (this.gtTextures.has(imgIdx)) return this.gtTextures.get(imgIdx);
    const imgInfo = this.images[imgIdx];
    if (!imgInfo) return null;

    let blob;
    if (imgInfo.fileBlob) {
      blob = imgInfo.fileBlob;
    } else if (imgInfo.imageUrl) {
      const resp = await fetch(imgInfo.imageUrl);
      if (!resp.ok) throw new Error(`Failed to fetch ${imgInfo.imageUrl}`);
      blob = await resp.blob();
    } else {
      return null;
    }

    const bmp = await createImageBitmap(blob, {
      resizeWidth: this.trainW,
      resizeHeight: this.trainH,
      resizeQuality: 'medium',
    });
    const tex = this.device.createTexture({
      size: [this.trainW, this.trainH],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      label: `gt_${imgInfo.name}`,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: bmp },
      { texture: tex },
      [this.trainW, this.trainH]
    );
    bmp.close();
    this.gtTextures.set(imgIdx, tex);
    return tex;
  }

  // Update `sortedIdxBuf` for an arbitrary camera (Free-View or COLMAP view)
  updateSortingForView(camCenter, view, fxNorm, fyNorm, { useGlobalSort = false, slidingWindow = 0 } = {}) {
    if (useGlobalSort) {
      this.numVisible = this.octree.sortGlobalExact(
        this.cpuParams, this.count, view, fxNorm, fyNorm, this.sortedIdx
      );
    } else {
      const res = this.octree.sortOctreeFrontToBack(
        this.cpuParams, camCenter, view, fxNorm, fyNorm, this.sortedIdx, { slidingWindow }
      );
      this.numVisible = res.numVisible;
    }
    if (this.numVisible > 0) {
      this.device.queue.writeBuffer(
        this.sortedIdxBuf, 0,
        this.sortedIdx.buffer, 0, this.numVisible * 4
      );
    }
    return this.numVisible;
  }

  // Execute one GPU training iteration on a training camera
  async step(forcedCamIdx = null) {
    if (this.stepInFlight || this.images.length === 0 || this.count === 0) return null;
    this.stepInFlight = true;
    try {
      return await this._stepImpl(forcedCamIdx);
    } finally {
      this.stepInFlight = false;
    }
  }

  async _stepImpl(forcedCamIdx = null) {
    // Pick next training camera
    const viewIdx = (forcedCamIdx !== null)
      ? forcedCamIdx
      : this.trainViews[this.stepCount % this.trainViews.length];
    const gtTex = await this.ensureGtTexture(viewIdx);
    if (!gtTex) return null;

    const img = this.images[viewIdx];
    const fx = img.camera.fx * this.scaleFx;
    const fy = img.camera.fy * this.scaleFy;
    const cx = img.camera.cx * this.scaleFx;
    const cy = img.camera.cy * this.scaleFy;

    // Sort visible splats via front-to-back Loose Octree traversal
    this.updateSortingForView(
      img.center, img.view,
      2 * fx / this.trainW, 2 * fy / this.trainH,
      { useGlobalSort: this.params.sort_mode === 'global_exact', slidingWindow: this.params.sliding_window || 0 }
    );
    if (this.numVisible === 0) return null;

    this.stepCount++;
    this.adamStepCount++;

    // Optimizer method & Simulated Annealing cooling schedule
    const optMap = { 'adam': 0, 'monte_carlo': 1, 'sim_anneal': 2 };
    const optMethodId = optMap[this.params.opt_method] ?? 0;
    const cooling = this.params.sa_cooling || 0.96;
    const cycleSteps = this.params.sa_reheat
      ? (this.stepCount % Math.max(100, this.params.rebalance_every * 2))
      : this.stepCount;
    this.currentTemp = (this.params.sa_temp || 0.15) * Math.pow(cooling, cycleSteps / 50.0);
    const mcOrSaParam = (optMethodId === 1)
      ? (this.params.mc_noise || 0.08)
      : this.currentTemp;

    // Warmup ramp for the first 50 steps (especially important when resuming from .ply)
    const warmup = Math.min(1.0, this.stepCount / 50.0);
    const uBuf = new ArrayBuffer(160);
    const f32  = new Float32Array(uBuf);
    const u32  = new Uint32Array(uBuf);
    f32.set(img.view, 0);
    f32[16] = img.center[0]; f32[17] = img.center[1]; f32[18] = img.center[2]; f32[19] = this.adamStepCount;
    f32[20] = fx; f32[21] = fy; f32[22] = cx; f32[23] = cy;
    u32[24] = this.trainW; u32[25] = this.trainH; u32[26] = this.numVisible; u32[27] = this.count;
    f32[28] = this.params.lr_pos   * this.octree.sceneRadius * warmup;
    f32[29] = this.params.lr_scale * warmup;
    f32[30] = this.params.lr_quat  * warmup;
    f32[31] = this.params.lr_color;
    f32[32] = optMethodId;
    f32[33] = mcOrSaParam;
    f32[34] = Math.random();
    f32[35] = this.octree.sceneRadius;
    this.device.queue.writeBuffer(this.trainUniformBuf, 0, uBuf);
    this.device.queue.writeBuffer(this.lossBuf, 0, new Uint32Array([0, 0, 0, 0]));

    const enc = this.device.createCommandEncoder({ label: `train_step_${this.stepCount}` });

    // 1. Project visible splats to 2D
    {
      const bg = this.device.createBindGroup({
        layout: this.pipeProject.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainUniformBuf } },
          { binding: 1, resource: { buffer: this.splatsBuf } },
          { binding: 2, resource: { buffer: this.sortedIdxBuf } },
          { binding: 3, resource: { buffer: this.projSplatsBuf } },
        ],
      });
      const pass = enc.beginComputePass({ label: 'project_splats' });
      pass.setPipeline(this.pipeProject);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.numVisible / 64));
      pass.end();
    }

    // 2. Tile Forward Rasterizer (16x16 tiles)
    const tilesX = Math.ceil(this.trainW / 16);
    const tilesY = Math.ceil(this.trainH / 16);
    {
      const bg = this.device.createBindGroup({
        layout: this.pipeForward.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainUniformBuf } },
          { binding: 1, resource: { buffer: this.splatsBuf } },
          { binding: 2, resource: { buffer: this.projSplatsBuf } },
          { binding: 3, resource: this.hatTex.createView() },
          { binding: 4, resource: { buffer: this.pixStateBuf } },
        ],
      });
      const pass = enc.beginComputePass({ label: 'tile_forward' });
      pass.setPipeline(this.pipeForward);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(tilesX, tilesY);
      pass.end();
    }

    // 3. Tile Backward + 3D WLBG Moments Pass
    {
      const bg = this.device.createBindGroup({
        layout: this.pipeBackward.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainUniformBuf } },
          { binding: 1, resource: { buffer: this.splatsBuf } },
          { binding: 2, resource: { buffer: this.projSplatsBuf } },
          { binding: 3, resource: gtTex.createView() },
          { binding: 4, resource: this.hatTex.createView() },
          { binding: 5, resource: { buffer: this.pixStateBuf } },
          { binding: 6, resource: { buffer: this.grads2dBuf } },
          { binding: 7, resource: { buffer: this.momentsBuf } },
          { binding: 8, resource: { buffer: this.lossBuf } },
        ],
      });
      const pass = enc.beginComputePass({ label: 'tile_backward' });
      pass.setPipeline(this.pipeBackward);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(tilesX, tilesY);
      pass.end();
    }

    // 4. Per-Splat 2D->3D Chain Rule + Adam / Monte-Carlo / Simulated Annealing Update
    {
      const bg = this.device.createBindGroup({
        layout: this.pipeAdam.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.trainUniformBuf } },
          { binding: 1, resource: { buffer: this.splatsBuf } },
          { binding: 2, resource: { buffer: this.grads2dBuf } },
          { binding: 3, resource: { buffer: this.adamMBuf } },
          { binding: 4, resource: { buffer: this.adamVBuf } },
          { binding: 5, resource: { buffer: this.momentsBuf } },
        ],
      });
      const pass = enc.beginComputePass({ label: 'adam_step' });
      pass.setPipeline(this.pipeAdam);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.count / 64));
      pass.end();
    }

    const needReadback = (this.stepCount % 5 === 0) ||
                         (this.stepCount % this.params.rebalance_every === 0);
    const needRebalance = (this.stepCount % this.params.rebalance_every === 0);

    if (needReadback && !this.pendingRead) {
      enc.copyBufferToBuffer(this.lossBuf, 0, this.lossReadBuf, 0, 16);
      enc.copyBufferToBuffer(this.splatsBuf, 0, this.splatsReadBuf, 0, this.count * SPLAT_FLOATS * 4);
      if (needRebalance) {
        enc.copyBufferToBuffer(this.momentsBuf, 0, this.momentsReadBuf, 0, this.count * MOMENT_FLOATS * 4);
      }
    }

    this.device.queue.submit([enc.finish()]);

    if (needReadback && !this.pendingRead) {
      this.pendingRead = this._handleReadback(needRebalance);
      await this.pendingRead;
      this.pendingRead = null;
    }

    return { viewIdx, gtTex, hatTex: this.hatTex, psnr: this.lastPsnr };
  }

  async _handleReadback(doRebalance) {
    const splatBytes  = this.count * SPLAT_FLOATS * 4;
    const momentBytes = this.count * MOMENT_FLOATS * 4;

    const maps = [
      this.lossReadBuf.mapAsync(GPUMapMode.READ, 0, 16),
      this.splatsReadBuf.mapAsync(GPUMapMode.READ, 0, splatBytes),
    ];
    if (doRebalance) {
      maps.push(this.momentsReadBuf.mapAsync(GPUMapMode.READ, 0, momentBytes));
    }
    await Promise.all(maps);

    const mseFixed = new Uint32Array(this.lossReadBuf.getMappedRange(0, 16))[0];
    this.lossReadBuf.unmap();
    const mse = (mseFixed / 65536.0) / (this.trainW * this.trainH);
    if (mse > 0) {
      this.lastPsnr = -10 * Math.log10(Math.max(1e-8, mse));
      this.history.push({ step: this.stepCount, psnr: this.lastPsnr, count: this.count, temp: this.currentTemp });
      if (this.history.length > 300) this.history.shift();
    }

    this.cpuParams.set(new Float32Array(this.splatsReadBuf.getMappedRange(0, splatBytes)));
    this.splatsReadBuf.unmap();

    if (doRebalance) {
      this.cpuMoments.set(new Float32Array(this.momentsReadBuf.getMappedRange(0, momentBytes)));
      this.momentsReadBuf.unmap();

      // Compute average R_i for diagnostic heatmap & run WLBG growth + floater removal + octree rebalance
      let sumR = 0, active = 0;
      for (let i = 0; i < this.count; i++) {
        const r = this.cpuMoments[i * MOMENT_FLOATS + 2];
        if (r > 0) { sumR += r; active++; }
      }
      if (active > 0) this.avgR = sumR / active;

      if (this.params.enable_rebalance) {
        const res = this.octree.reconcileAndGrow(
          this.cpuParams, this.count, this.cpuMoments, this.images,
          {
            maxSplats:      this.params.max_splats,
            pruneOpacity:   this.params.prune_opacity,
            hysteresis:     this.params.hysteresis,
            enableFloaters: this.params.remove_floaters,
            enableSplit:    this.params.enable_split,
            enableMerge:    this.params.enable_merge,
            optMethod:      this.params.opt_method,
            saTemp:         this.currentTemp,
          }
        );
        this.count        = res.count;
        this.cpuParams    = res.params;
        this.splitVectors = res.splitVectors;
        this._uploadSplats();
      } else {
        // Just rebuild octree for drifted positions without changing splat count
        this.octree.build(this.cpuParams, this.count, this.images);
        this._uploadSplats();
      }
    }
  }

  // ── Experiment E1: Compare Exact Global Radix Sort vs. Octree Front-to-Back Sort ──
  async runExperimentE1(camIdx = 0) {
    if (this.images.length === 0) return 'No cameras loaded';
    const img = this.images[camIdx];
    const fx = img.camera.fx * this.scaleFx;
    const fy = img.camera.fy * this.scaleFy;
    const cx = img.camera.cx * this.scaleFx;
    const cy = img.camera.cy * this.scaleFy;
    const fxNorm = 2 * fx / this.trainW;
    const fyNorm = 2 * fy / this.trainH;

    const renderToPixels = async (useGlobal, slidingWindow) => {
      const t0 = performance.now();
      this.updateSortingForView(img.center, img.view, fxNorm, fyNorm, {
        useGlobalSort: useGlobal,
        slidingWindow,
      });
      const sortMs = performance.now() - t0;
      const uBuf = new ArrayBuffer(160);
      const f32  = new Float32Array(uBuf);
      const u32  = new Uint32Array(uBuf);
      f32.set(img.view, 0);
      f32[16] = img.center[0]; f32[17] = img.center[1]; f32[18] = img.center[2]; f32[19] = 1;
      f32[20] = fx; f32[21] = fy; f32[22] = cx; f32[23] = cy;
      u32[24] = this.trainW; u32[25] = this.trainH; u32[26] = this.numVisible; u32[27] = this.count;
      this.device.queue.writeBuffer(this.trainUniformBuf, 0, uBuf);

      const bytesPerRow = Math.ceil((this.trainW * 4) / 256) * 256;
      const readBuf = this.device.createBuffer({
        size: bytesPerRow * this.trainH,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const enc = this.device.createCommandEncoder();
      {
        const bg = this.device.createBindGroup({
          layout: this.pipeProject.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.trainUniformBuf } },
            { binding: 1, resource: { buffer: this.splatsBuf } },
            { binding: 2, resource: { buffer: this.sortedIdxBuf } },
            { binding: 3, resource: { buffer: this.projSplatsBuf } },
          ],
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipeProject);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.numVisible / 64));
        pass.end();
      }
      {
        const bg = this.device.createBindGroup({
          layout: this.pipeForward.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.trainUniformBuf } },
            { binding: 1, resource: { buffer: this.splatsBuf } },
            { binding: 2, resource: { buffer: this.projSplatsBuf } },
            { binding: 3, resource: this.hatTex.createView() },
            { binding: 4, resource: { buffer: this.pixStateBuf } },
          ],
        });
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipeForward);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.trainW / 16), Math.ceil(this.trainH / 16));
        pass.end();
      }
      enc.copyTextureToBuffer(
        { texture: this.hatTex },
        { buffer: readBuf, bytesPerRow },
        [this.trainW, this.trainH]
      );
      this.device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const out = new Uint8Array(readBuf.getMappedRange()).slice();
      readBuf.unmap();
      readBuf.destroy();
      return { pixels: out, bytesPerRow, sortMs };
    };

    const ref  = await renderToPixels(true, 0);
    const oct0 = await renderToPixels(false, 0);
    const oct8 = await renderToPixels(false, 8);

    const calcPsnr = (a, b, bpr) => {
      let sumSq = 0, count = 0;
      for (let y = 0; y < this.trainH; y++) {
        for (let x = 0; x < this.trainW; x++) {
          const o = y * bpr + x * 4;
          for (let c = 0; c < 3; c++) {
            const d = (a[o + c] - b[o + c]) / 255.0;
            sumSq += d * d;
            count++;
          }
        }
      }
      const mse = sumSq / count;
      return mse < 1e-10 ? 99.9 : -10 * Math.log10(mse);
    };

    const psnr0 = calcPsnr(ref.pixels, oct0.pixels, ref.bytesPerRow);
    const psnr8 = calcPsnr(ref.pixels, oct8.pixels, ref.bytesPerRow);
    const summary = `E1 (vs Global ${ref.sortMs.toFixed(1)}ms): Octree W=0 (${oct0.sortMs.toFixed(1)}ms) -> ${psnr0.toFixed(2)} dB | W=8 -> ${psnr8.toFixed(2)} dB`;
    console.log('[Experiment E1]', summary);
    return summary;
  }
}
