'use strict';

// Greedy optimizer — CPU scoring via computeCPULoss, GPU used only for display.
// Depends on: triangle-core.js, triangle-ans-enc.js, triangle-cpu.js.

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

function encodeSizeComponents(preview, color_data) {
  const total      = encodeSize(preview, color_data);
  const noVtx      = { ...preview, nb_pts: 0, qpts: preview.qpts.slice(0, 4) };
  const color_bytes = encodeSize(noVtx, color_data);
  return { total, color_bytes, vtx_bytes: total - color_bytes };
}

// ---------------------------------------------------------------------------
// Mutation helpers

function applyMoveVertex(preview, color_data, rng, amplitude = 1, border_escape = 5) {
  const nb = preview.qpts.length - 4;
  if (nb <= 0) return null;
  const { grid_x: gx, grid_y: gy } = preview;
  const p = clonePreview(preview);
  const v = p.qpts[4 + Math.floor(rng() * nb)];
  const delta = () => (Math.floor(rng() * amplitude) + 1) * (rng() < 0.5 ? -1 : 1);
  const onBorder = v.x === 0 || v.x === gx-1 || v.y === 0 || v.y === gy-1;
  if (onBorder && rng() * 100 >= border_escape) {
    if (v.x === 0 || v.x === gx-1) v.y = clampI(v.y + delta(), 1, gy-2);
    else                             v.x = clampI(v.x + delta(), 1, gx-2);
  } else {
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
  const total = preview.qpts.length + 4;
  const k = Math.floor(rng() * total) - 4;
  const newIdx = Math.floor(rng() * preview.nb_colors);
  const p = clonePreview(preview);
  if (k < 0) p.qpts[k + 4].idx = newIdx;
  else        p.qpts[k].idx = newIdx;
  return { preview: p, color_data };
}

function colorEq(a, b) {
  if (a.a === 0 && b.a === 0) return true;
  return a.a === b.a && a.y === b.y && a.co === b.co && a.cg === b.cg;
}

function applyMoveColor(preview, color_data, rng) {
  const k = Math.floor(rng() * preview.nb_colors);
  const c = color_data[k];
  if (c.a === 0) return null;
  const ch = Math.floor(rng() * 3);
  const d = rng() < 0.5 ? 1 : -1;
  const nc = { ...c,
    y:  ch === 0 ? clampI(c.y  + d, 0, kYCoCgMax) : c.y,
    co: ch === 1 ? clampI(c.co + d, 0, kYCoCgMax) : c.co,
    cg: ch === 2 ? clampI(c.cg + d, 0, kYCoCgMax) : c.cg,
  };
  if (color_data.some((x, i) => i !== k && colorEq(x, nc))) return null;
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
  if (color_data.some(x => colorEq(x, c))) return null;
  const p = clonePreview(preview);
  const nc = [...color_data, c];
  p.nb_colors = nc.length;
  return sortColors(p, nc);
}

function applyFlipAlpha(preview, color_data, rng) {
  if (!preview.has_alpha) return null;
  const k = Math.floor(rng() * preview.nb_colors);
  const flipped = { ...color_data[k], a: 1 - color_data[k].a };
  if (color_data.some((x, i) => i !== k && colorEq(x, flipped))) return null;
  return sortColors(preview, color_data.map((c, i) => i === k ? flipped : c));
}

function applyRemoveColor(preview, color_data, rng) {
  if (preview.nb_colors <= kPreviewMinNumColors) return null;
  const counts = new Array(preview.nb_colors).fill(0);
  for (const v of preview.qpts) counts[v.idx]++;
  let minK = 0;
  for (let i = 1; i < preview.nb_colors; i++) if (counts[i] < counts[minK]) minK = i;
  const removed = color_data[minK];
  let bestD = Infinity, repIdx = 0;
  for (let i = 0; i < color_data.length; i++) {
    if (i === minK) continue;
    const c = color_data[i];
    const d = (c.y-removed.y)**2 + (c.co-removed.co)**2 + (c.cg-removed.cg)**2;
    if (d < bestD) { bestD = d; repIdx = i; }
  }
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

class TriangleOptimizer {
  constructor(preview, color_data, {
    refGrid                   = null,   // Float32Array zGrid (gx*zoom × gy*zoom, RGBA [0,255])
    zoom                      = 1,
    lambda_vtx                = 0.0001,
    lambda_color              = 0.0001,
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
    batch_size                = 500,    // iterations between UI-yield / onProgress calls
    vertex_amplitude          = 1,
  } = {}) {
    this.preview    = clonePreview(preview);
    this.color_data = color_data.map(c => ({ ...c }));
    this.refGrid    = refGrid;
    this.zoom       = zoom;
    this.lambda_vtx   = lambda_vtx;
    this.lambda_color = lambda_color;
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
    this.batch_size        = batch_size;
    this.vertex_amplitude  = vertex_amplitude;
    this.iter     = 0;
    this.bestLoss = Infinity;
    this.bestPreview   = null;
    this.bestColorData = null;
    let s = seed >>> 0;
    this._rng = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0x100000000; };
  }

  _score(preview, color_data) {
    const del    = buildDel(preview);
    const palRGB = buildPalRGB(color_data);
    const distortion = computeCPULoss(preview.grid_x, preview.grid_y, del, palRGB, this.refGrid, this.zoom);
    const { total, vtx_bytes, color_bytes } = encodeSizeComponents(preview, color_data);
    const score = distortion + this.lambda_vtx * vtx_bytes + this.lambda_color * color_bytes;
    return { distortion, score, total, vtx_bytes, color_bytes };
  }

  init() {
    const { distortion, score } = this._score(this.preview, this.color_data);
    this.currentScore       = score;
    this.currentDistortion  = distortion;
    this.bestLoss           = score;
    this.bestPreview   = clonePreview(this.preview);
    this.bestColorData = this.color_data.map(c => ({ ...c }));
    return score;
  }

  step(iter, maxIter) {
    const rng = this._rng;

    let np = clonePreview(this.preview);
    let nc = this.color_data.map(c => ({ ...c }));

    let dbgLabel = null;
    for (let m = 0; m < this.num_mutations_per_iter; ++m) {
      const mutations = [
        ['v=', this.proba_vertex_move, () => np.qpts.length > 4 ? applyMoveVertex(np, nc, rng, this.vertex_amplitude, this.border_escape_prob) : null],
        ['v+', this.proba_vertex_add,  () => applyAddVertex(np, nc, rng)],
        ['v-', this.proba_vertex_sub,  () => np.qpts.length > 4 ? applyRemoveVertex(np, nc, rng) : null],
        ['ci', this.proba_color_index_move, () => applyMoveColorIndex(np, nc, rng)],
        ['c=', this.proba_color_move,  () => applyMoveColor(np, nc, rng)],
        ['c+', this.proba_color_add,   () => applyAddColor(np, nc, rng)],
        ['c-', this.proba_color_sub,   () => np.nb_colors > kPreviewMinNumColors ? applyRemoveColor(np, nc, rng) : null],
        ['a=', this.proba_flip_alpha,  () => np.has_alpha ? applyFlipAlpha(np, nc, rng) : null],
      ];
      const total = mutations.reduce((s, [, p]) => s + p, 0);
      if (total <= 0) break;
      let pick = rng() * total;
      for (const [lbl, p, fn] of mutations) {
        pick -= p;
        if (pick < 0) {
          const r = fn();
          if (this.debug) dbgLabel = lbl + (r ? '' : '?');
          if (r) { np = r.preview; nc = r.color_data; }
          break;
        }
      }
    }

    const { distortion, score: newScore } = this._score(np, nc);
    const tolerance = this.score_tolerance * (maxIter - iter) / maxIter;
    const accept = newScore <= this.bestLoss + tolerance;

    if (this.debug && dbgLabel !== null) {
      console.log(`[${iter}] ${dbgLabel}${accept ? ' ACC' : ' N'} dist=${distortion.toFixed(6)} score=${newScore.toFixed(6)} cur=${this.currentScore.toFixed(6)} tol=${tolerance.toFixed(6)}`);
    }

    if (accept) {
      this.preview            = np;
      this.color_data         = nc;
      this.currentScore       = newScore;
      this.currentDistortion  = distortion;
      if (newScore < this.bestLoss) {
        this.bestLoss      = newScore;
        this.bestPreview   = clonePreview(np);
        this.bestColorData = nc.map(c => ({ ...c }));
      }
    }

    this.iter++;
    return { accepted: accept, score: this.currentScore, distortion };
  }

  stop() { this._stop = true; }

  async run(maxIter, onProgress) {
    this._stop = false;
    for (let i = 0; i < maxIter && !this._stop; i += this.batch_size) {
      const bsz = Math.min(this.batch_size, maxIter - i);
      let r = { accepted: false, score: this.currentScore, distortion: 0 };
      for (let j = 0; j < bsz && !this._stop; ++j) {
        r = this.step(i + j, maxIter);
      }
      await new Promise(res => setTimeout(res, 0));  // yield to event loop
      if (onProgress) await onProgress(Math.min(i + bsz, maxIter), this.bestLoss, r);
    }
    return { preview: this.bestPreview, color_data: this.bestColorData, loss: this.bestLoss };
  }
}

// ---------------------------------------------------------------------------

if (typeof module !== 'undefined') {
  module.exports = { TriangleOptimizer, clonePreview, encodeSize };
}
