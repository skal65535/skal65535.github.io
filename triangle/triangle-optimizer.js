'use strict';

// Phase 4: Greedy optimizer with linearly-decaying tolerance.
// Matches libwebp2 preview_opt.cc logic: step=1 mutations, multiple independent
// probabilities per iteration, accept if score <= best + tolerance*(rem/total).
// Depends on: triangle-core.js, triangle-ans-enc.js, triangle-gpu.js.

function clonePreview(p) {
  return { ...p, qpts: p.qpts.map(v => ({ ...v })) };
}

function clampI(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function sortColors(preview, color_data) {
  const order = color_data.map((_, i) => i).sort((a, b) => {
    const ca = color_data[a], cb = color_data[b];
    return ca.cg !== cb.cg ? ca.cg - cb.cg :
           ca.co !== cb.co ? ca.co - cb.co :
           ca.y  !== cb.y  ? ca.y  - cb.y  : ca.a - cb.a;
  });
  if (order.every((o, i) => o === i)) return { preview, color_data };
  const sorted = order.map(i => color_data[i]);
  const remap = new Int32Array(color_data.length);
  order.forEach((o, ni) => { remap[o] = ni; });
  const p = clonePreview(preview);
  for (const v of p.qpts) v.idx = remap[v.idx];
  return { preview: p, color_data: sorted };
}

function buildDel(preview) {
  return new Delaunay(preview.grid_x, preview.grid_y, preview.qpts);
}
function buildPalRGB(color_data) {
  return color_data.map(c => YCoCg_to_RGB(c.y, c.co, c.cg, c.a));
}

function encodeSize(preview, color_data) {
  return Math.ceil(encodePreview(preview, color_data).length * 3 / 4);
}

// ---------------------------------------------------------------------------
// Mutation helpers — all step sizes = 1, matching C++ distance=1.

function applyMoveVertex(preview, color_data, rng, amplitude = 1, border_escape = 5) {
  const nb = preview.qpts.length - 4;
  if (nb <= 0) return null;
  const { grid_x: gx, grid_y: gy } = preview;
  const p = clonePreview(preview);
  const v = p.qpts[4 + Math.floor(rng() * nb)];
  const delta = () => (Math.floor(rng() * amplitude) + 1) * (rng() < 0.5 ? -1 : 1);
  const onBorder = v.x === 0 || v.x === gx-1 || v.y === 0 || v.y === gy-1;
  if (onBorder && rng() * 100 >= border_escape) {
    // Slide along current edge, avoiding corners.
    if (v.x === 0 || v.x === gx-1) v.y = clampI(v.y + delta(), 1, gy-2);
    else                             v.x = clampI(v.x + delta(), 1, gx-2);
  } else {
    // Free move; allow reaching border but reject landing on a corner.
    if (rng() < 0.5) v.x = clampI(v.x + delta(), 0, gx-1);
    else              v.y = clampI(v.y + delta(), 0, gy-1);
    if ((v.x === 0 || v.x === gx-1) && (v.y === 0 || v.y === gy-1)) return null;
  }
  return { preview: p, color_data };
}

function applyAddVertex(preview, color_data, rng) {
  if (preview.nb_pts >= kPreviewMaxNumVertices) return null;
  const gx = preview.grid_x, gy = preview.grid_y;
  const x = Math.floor(rng() * gx);
  const y = Math.floor(rng() * gy);
  if ((x === 0 || x === gx - 1) && (y === 0 || y === gy - 1)) return null;
  const p = clonePreview(preview);
  if (p.qpts.some(v => v.x === x && v.y === y)) return null;
  const idx = Math.floor(rng() * preview.nb_colors);
  const pos = p.qpts.findIndex((v, i) => i >= 4 && (v.y > y || (v.y === y && v.x > x)));
  p.qpts.splice(pos < 0 ? p.qpts.length : pos, 0, { x, y, idx });
  p.nb_pts = p.qpts.length - 4;
  return { preview: p, color_data };
}

function applyRemoveVertex(preview, color_data, rng) {
  const nb = preview.qpts.length - 4;
  if (nb <= 0) return null;
  const p = clonePreview(preview);
  const k = 4 + Math.floor(rng() * nb);
  p.qpts.splice(k, 1);
  p.nb_pts = p.qpts.length - 4;
  return { preview: p, color_data };
}

function applyMoveColorIndex(preview, color_data, rng) {
  // includes corners (indices -4..-1) as in C++
  const total = preview.qpts.length + 4;
  const k = Math.floor(rng() * total) - 4;  // -4 .. qpts.length-1
  const newIdx = Math.floor(rng() * preview.nb_colors);
  const p = clonePreview(preview);
  if (k < 0) {
    // corner: k=-4→TL, -3→TR, -2→BL, -1→BR
    p.qpts[k + 4].idx = newIdx;
  } else {
    p.qpts[k].idx = newIdx;
  }
  return { preview: p, color_data };
}

function applyMoveColor(preview, color_data, rng) {
  const k = Math.floor(rng() * preview.nb_colors);
  const c = color_data[k];
  const ch = Math.floor(rng() * 3);
  const d = rng() < 0.5 ? 1 : -1;
  const nc = { ...c,
    y:  ch === 0 ? clampI(c.y  + d, 0, kYCoCgMax) : c.y,
    co: ch === 1 ? clampI(c.co + d, 0, kYCoCgMax) : c.co,
    cg: ch === 2 ? clampI(c.cg + d, 0, kYCoCgMax) : c.cg,
  };
  if (color_data.some((x, i) => i !== k && x.y === nc.y && x.co === nc.co && x.cg === nc.cg && x.a === nc.a))
    return null;
  return sortColors(preview, color_data.map((x, i) => i === k ? nc : x));
}

function applyAddColor(preview, color_data, rng) {
  if (preview.nb_colors >= kPreviewMaxNumColors) return null;
  const c = {
    y:  Math.floor(rng() * (kYCoCgMax + 1)),
    co: Math.floor(rng() * (kYCoCgMax + 1)),
    cg: Math.floor(rng() * (kYCoCgMax + 1)),
    a: 1,
  };
  if (color_data.some(x => x.y === c.y && x.co === c.co && x.cg === c.cg && x.a === c.a))
    return null;
  const p = clonePreview(preview);
  const nc = [...color_data, c];
  p.nb_colors = nc.length;
  return sortColors(p, nc);
}

function applyFlipAlpha(preview, color_data, rng) {
  if (!preview.has_alpha) return null;
  const k = Math.floor(rng() * preview.nb_colors);
  return sortColors(preview, color_data.map((c, i) => i === k ? { ...c, a: 1 - c.a } : c));
}

function applyRemoveColor(preview, color_data, rng) {
  if (preview.nb_colors <= kPreviewMinNumColors) return null;
  // Remove the color with the least vertex references (like C++ RemoveColor).
  const counts = new Array(preview.nb_colors).fill(0);
  for (const v of preview.qpts) counts[v.idx]++;
  let minK = 0;
  for (let i = 1; i < preview.nb_colors; i++) if (counts[i] < counts[minK]) minK = i;
  const removed = color_data[minK];
  // Find closest replacement
  let bestD = Infinity, repIdx = 0;
  for (let i = 0; i < color_data.length; i++) {
    if (i === minK) continue;
    const c = color_data[i];
    const d = (c.y-removed.y)**2 + (c.co-removed.co)**2 + (c.cg-removed.cg)**2;
    if (d < bestD) { bestD = d; repIdx = i; }
  }
  // remap: minK→repIdx (adjusted for removal), last→minK
  const nc = color_data.filter((_, i) => i !== minK);
  const p = clonePreview(preview);
  p.nb_colors = nc.length;
  const adjRepIdx = repIdx < minK ? repIdx : repIdx - 1;
  for (const v of p.qpts) {
    if (v.idx === minK)    v.idx = adjRepIdx;
    else if (v.idx > minK) v.idx -= 1;
  }
  return { preview: p, color_data: nc };
}

// ---------------------------------------------------------------------------

// Convert preview+color_data to typed arrays for TriangleOptGPU.
// Vertices indexed by qpts order (unique, nv = qpts.length).
// Triangle indices remapped from del.vtx (which has duplicates) back to qpts indices.
function makeGPUState(preview, color_data) {
  const qpts = preview.qpts;
  const nv   = qpts.length;
  const nc   = color_data.length;

  const posI32  = new Int32Array(nv * 2);
  const cidxU32 = new Uint32Array(nv);
  for (let i = 0; i < nv; ++i) {
    posI32[i*2]   = qpts[i].x;
    posI32[i*2+1] = qpts[i].y;
    cidxU32[i]    = qpts[i].idx;
  }

  // Build Delaunay to get triangle topology, then remap duplicate vtx → qpts index.
  const del = buildDel(preview);
  const { positions: fullPos, indices: fullIdx } = del.getFlatBuffers();
  const nFull = fullPos.length / 2;

  const qptsMap = new Map();
  for (let i = 0; i < nv; ++i) {
    qptsMap.set(`${qpts[i].x},${qpts[i].y}`, i);
  }
  const fullToQpts = new Int32Array(nFull);
  for (let i = 0; i < nFull; ++i) {
    fullToQpts[i] = qptsMap.get(`${fullPos[i*2] | 0},${fullPos[i*2+1] | 0}`) ?? 0;
  }

  const nt     = fullIdx.length / 3;
  const triU32 = new Uint32Array(nt * 3);
  for (let i = 0; i < fullIdx.length; ++i) triU32[i] = fullToQpts[fullIdx[i]];

  const palU32 = new Uint32Array(nc * 4);
  for (let i = 0; i < nc; ++i) {
    const c = color_data[i];
    palU32[i*4]   = c.y;
    palU32[i*4+1] = c.co;
    palU32[i*4+2] = c.cg;
    palU32[i*4+3] = c.a;
  }

  return { posI32, cidxU32, triU32, palU32, nv, nt, nc };
}

// Apply GPU state back into preview.qpts and color_data.
// posI32: Int32Array [nv*2], cidxU32: Uint32Array [nv], palU32: Uint32Array [nc*4]
// The topology (triangle indices) is unchanged, only vertex positions/colors mutated.
function applyGPUState(preview, color_data, posI32, cidxU32, palU32) {
  const nv = posI32.length / 2;
  for (let i = 0; i < nv; ++i) {
    preview.qpts[i].x   = posI32[i*2];
    preview.qpts[i].y   = posI32[i*2+1];
    preview.qpts[i].idx = cidxU32[i];
  }
  const nc = palU32.length / 4;
  const nc_old = color_data.length;
  if (nc !== nc_old) return;   // shouldn't happen in GPU batch
  for (let i = 0; i < nc; ++i) {
    color_data[i].y  = palU32[i*4];
    color_data[i].co = palU32[i*4+1];
    color_data[i].cg = palU32[i*4+2];
    color_data[i].a  = palU32[i*4+3];
  }
  // re-sort interior qpts by (y,x) as required by the codec
  const corners = preview.qpts.slice(0, 4);
  const inner   = preview.qpts.slice(4).sort((a, b) => a.y !== b.y ? a.y - b.y : a.x - b.x);
  preview.qpts  = [...corners, ...inner];
}

// ---------------------------------------------------------------------------

class TriangleOptimizer {
  constructor(gpu, preview, color_data, {
    lambda                    = 0.0001,
    seed                      = 0x12345678,
    score_tolerance           = 0.0002,
    num_mutations_per_iter    = 1,
    proba_vertex_move         = 50,
    proba_vertex_add          = 20,
    proba_vertex_sub          = 25,
    proba_color_index_move    = 25,
    proba_color_move          = 20,
    proba_color_add           = 1,
    proba_color_sub           = 3,
    proba_flip_alpha          = 25,
    border_escape_prob        = 5,
    gpu_batch_size            = 1000,  // iterations per GPU dispatch
    opt_gpu                   = null,  // TriangleOptGPU instance (optional)
    vertex_amplitude          = 1,     // max pixel delta per vertex move
    topo_cadence              = 10,    // topology CPU step every N GPU batches
  } = {}) {
    this.gpu = gpu;
    this.optGpu     = opt_gpu;
    this.preview    = clonePreview(preview);
    this.color_data = color_data.map(c => ({ ...c }));
    this.lambda     = lambda;
    this.score_tolerance        = score_tolerance;
    this.num_mutations_per_iter = num_mutations_per_iter;
    this.proba_vertex_move      = proba_vertex_move;
    this.proba_vertex_add       = proba_vertex_add;
    this.proba_vertex_sub       = proba_vertex_sub;
    this.proba_color_index_move = proba_color_index_move;
    this.proba_color_move       = proba_color_move;
    this.proba_color_add        = proba_color_add;
    this.proba_color_sub        = proba_color_sub;
    this.proba_flip_alpha       = proba_flip_alpha;
    this.border_escape_prob     = border_escape_prob;
    this.gpu_batch_size    = gpu_batch_size;
    this.vertex_amplitude  = vertex_amplitude;
    this.topo_cadence      = topo_cadence;
    this._refGrid = null;   // Float32Array [gx*gy*4] set by caller
    this._refDirty = true;  // force ref upload on first GPU batch
    this.iter     = 0;
    this.bestLoss = Infinity;
    this.bestPreview   = null;
    this.bestColorData = null;
    let s = seed >>> 0;
    this._rng = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0x100000000; };
    this._rngSeed = seed >>> 0;
  }

  async _score(preview, color_data) {
    const del = buildDel(preview);
    const palRGB = buildPalRGB(color_data);
    const distortion = await this.gpu.computeLoss(del, palRGB);
    const size = encodeSize(preview, color_data);
    return distortion + this.lambda * size;
  }

  async init() {
    this.currentScore = await this._score(this.preview, this.color_data);
    this.bestLoss      = this.currentScore;
    this.bestPreview   = clonePreview(this.preview);
    this.bestColorData = this.color_data.map(c => ({ ...c }));
    return this.currentScore;
  }

  async step(iter, maxIter) {
    const rng = this._rng;
    const chance = p => rng() * 100 < p;

    let np = clonePreview(this.preview);
    let nc = this.color_data.map(c => ({ ...c }));

    for (let m = 0; m < this.num_mutations_per_iter; ++m) {
      let r;
      if (np.qpts.length > 4 && chance(this.proba_vertex_move)) {
        r = applyMoveVertex(np, nc, rng, this.vertex_amplitude, this.border_escape_prob);
        if (r) { np = r.preview; nc = r.color_data; }
      }
      if (chance(this.proba_vertex_add)) {
        r = applyAddVertex(np, nc, rng);
        if (r) { np = r.preview; nc = r.color_data; }
      }
      if (np.qpts.length > 4 && chance(this.proba_vertex_sub)) {
        r = applyRemoveVertex(np, nc, rng);
        if (r) { np = r.preview; nc = r.color_data; }
      }
      if (chance(this.proba_color_index_move)) {
        r = applyMoveColorIndex(np, nc, rng);
        if (r) { np = r.preview; nc = r.color_data; }
      }
      if (chance(this.proba_color_move)) {
        r = applyMoveColor(np, nc, rng);
        if (r) { np = r.preview; nc = r.color_data; }
      }
      if (chance(this.proba_color_add)) {
        r = applyAddColor(np, nc, rng);
        if (r) { np = r.preview; nc = r.color_data; }
      }
      if (np.nb_colors > kPreviewMinNumColors && chance(this.proba_color_sub)) {
        r = applyRemoveColor(np, nc, rng);
        if (r) { np = r.preview; nc = r.color_data; }
      }
      if (np.has_alpha && chance(this.proba_flip_alpha)) {
        r = applyFlipAlpha(np, nc, rng);
        if (r) { np = r.preview; nc = r.color_data; }
      }
    }

    const newScore = await this._score(np, nc);
    const tolerance = this.score_tolerance * (maxIter - iter) / maxIter;
    const accept = newScore <= this.currentScore + tolerance;

    if (accept) {
      this.preview      = np;
      this.color_data   = nc;
      this.currentScore = newScore;
      if (newScore < this.bestLoss) {
        this.bestLoss      = newScore;
        this.bestPreview   = clonePreview(np);
        this.bestColorData = nc.map(c => ({ ...c }));
      }
    }

    this.iter++;
    return { accepted: accept, score: this.currentScore };
  }

  stop() { this._stop = true; }

  // Run one GPU batch of gpu_batch_size iterations.
  // Returns { accepted, score } where accepted = true if any improvement found.
  async stepGPUBatch(iterOffset, maxIter) {
    const { optGpu } = this;
    const bsz = Math.min(this.gpu_batch_size, maxIter - iterOffset);

    // Build GPU state from current preview / color_data
    const state = makeGPUState(this.preview, this.color_data);

    // Upload state; re-upload ref only when topology changed (flagged by _refDirty).
    if (!this._refDirty) {
      optGpu.uploadStateNoRef(state.posI32, state.cidxU32, state.palU32, state.triU32);
    } else {
      optGpu.uploadState(state.posI32, state.cidxU32, state.palU32, state.triU32, this._refGrid);
      this._refDirty = false;
    }

    const opts = {
      p_vm:  this.proba_vertex_move,
      p_ci:  this.proba_color_index_move,
      p_cm:  this.proba_color_move,
      p_fa:  this.preview.has_alpha ? 25 : 0,
      score_tolerance: this.score_tolerance,
      has_alpha: this.preview.has_alpha ? 1 : 0,
      vm_amp: this.vertex_amplitude,
      vm_border_escape: this.border_escape_prob,
      rng_seed: this._rngSeed,
    };

    const distortion = await optGpu.runBatch(bsz, iterOffset, maxIter, opts);

    // After first batch the RNG state lives on GPU; don't re-seed
    this._rngSeed = undefined;

    // Read back updated state
    const readback = await optGpu.readState();
    if (!readback) return { score: this.currentScore, accepted: false };
    const { posI32, cidxU32, palU32 } = readback;
    applyGPUState(this.preview, this.color_data, posI32, cidxU32, palU32);

    const size  = encodeSize(this.preview, this.color_data);
    const score = distortion + this.lambda * size;

    const accepted = score < this.currentScore;
    if (accepted) {
      this.currentScore  = score;
    }
    if (score < this.bestLoss) {
      this.bestLoss      = score;
      this.bestPreview   = clonePreview(this.preview);
      this.bestColorData = this.color_data.map(c => ({ ...c }));
    }
    this.iter += bsz;
    return { accepted, score: this.currentScore, distortion, size };
  }

  async run(maxIter, onProgress) {
    this._stop = false;

    if (this.optGpu) {
      // GPU-accelerated inner loop: gpu_batch_size iterations per dispatch.
      // CPU handles topology mutations between batches.
      let _lastDist = 1, _lastSize = 1;
      for (let i = 0; i < maxIter && !this._stop; i += this.gpu_batch_size) {
        const r = await this.stepGPUBatch(i, maxIter);
        if (r.distortion !== undefined) { _lastDist = r.distortion; _lastSize = r.size; }
        if (onProgress) await onProgress(Math.min(i + this.gpu_batch_size, maxIter), this.bestLoss, r);

        // Topology mutations on CPU; frequency adapts to how much size dominates score
        if ((i / this.gpu_batch_size) % this.topo_cadence === this.topo_cadence - 1) {
          const sizeFrac = (this.lambda * _lastSize) / Math.max(_lastDist, 1e-8);
          const nTrials  = Math.max(1, Math.min(20, Math.ceil(sizeFrac)));
          for (let t = 0; t < nTrials && !this._stop; ++t)
            await this._cpuTopologyStep(i, maxIter);
        }
      }
    } else {
      for (let i = 0; i < maxIter && !this._stop; ++i) {
        const r = await this.step(i, maxIter);
        if (onProgress) await onProgress(i + 1, this.bestLoss, r);
      }
    }

    return { preview: this.bestPreview, color_data: this.bestColorData, loss: this.bestLoss };
  }

  // A few CPU-side topology mutations (add/remove vertex/color).
  async _cpuTopologyStep(iter, maxIter) {
    const rng     = this._rng;
    const chance  = p => rng() * 100 < p;
    let np = clonePreview(this.preview);
    let nc = this.color_data.map(c => ({ ...c }));
    let r;
    if (chance(this.proba_vertex_add))  { r = applyAddVertex(np, nc, rng);    if (r) { np = r.preview; nc = r.color_data; } }
    if (np.qpts.length > 4 && chance(this.proba_vertex_sub)) { r = applyRemoveVertex(np, nc, rng); if (r) { np = r.preview; nc = r.color_data; } }
    if (chance(this.proba_color_add))   { r = applyAddColor(np, nc, rng);     if (r) { np = r.preview; nc = r.color_data; } }
    if (np.nb_colors > kPreviewMinNumColors && chance(this.proba_color_sub)) { r = applyRemoveColor(np, nc, rng); if (r) { np = r.preview; nc = r.color_data; } }

    const newScore = await this._score(np, nc);
    const tolerance = this.score_tolerance * (maxIter - iter) / maxIter;
    if (newScore <= this.currentScore + tolerance) {
      this.preview      = np;
      this.color_data   = nc;
      this.currentScore = newScore;
      this._refDirty = true;  // topology may have changed; force re-upload
      if (newScore < this.bestLoss) {
        this.bestLoss      = newScore;
        this.bestPreview   = clonePreview(np);
        this.bestColorData = nc.map(c => ({ ...c }));
      }
    }
  }
}

// ---------------------------------------------------------------------------

if (typeof module !== 'undefined') {
  module.exports = { TriangleOptimizer, clonePreview, encodeSize };
}
