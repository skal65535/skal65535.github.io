'use strict';

// Greedy optimizer — CPU scoring via computeCPULoss, GPU used only for display.
// Depends on: triangle-core.js, triangle-ans-enc.js, triangle-cpu.js.

function clonePreview(p) {
  return { ...p, qpts: p.qpts.slice() };
}

function clampI(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function sortColors(preview, color_data) {
  const order = color_data.map((_, i) => i).sort((a, b) => comparePaletteEntries(color_data[a], color_data[b]));
  if (order.every((o, i) => o === i)) return { preview, color_data };
  const sorted = order.map(i => color_data[i]);
  const remap = new Int32Array(color_data.length);
  order.forEach((o, ni) => { remap[o] = ni; });
  const p = clonePreview(preview);
  for (let i = 2, n = p.qpts.length; i < n; i += 3) p.qpts[i] = remap[p.qpts[i]];
  return { preview: p, color_data: sorted };
}

function buildDelaunay(preview) {
  return new Delaunay(preview.grid_x, preview.grid_y, preview.qpts);
}
function buildPalRGB(color_data) {
  return color_data.map(c => YCoCg_to_RGB(c.y, c.co, c.cg, c.a));
}

function encodeSize(preview, color_data) {
  const enc = new ANSEncCount();
  _fillEncoder(enc, preview, color_data);
  return enc.byteSize();
}

// ---------------------------------------------------------------------------
// Mutation helpers

function applyMoveVertex(preview, color_data, rng, amplitude = 1, border_escape = 5) {
  const nb = preview.qpts.length / 3 - 4;
  if (nb <= 0) return null;
  const { grid_x: gx, grid_y: gy } = preview;
  const p = clonePreview(preview);
  const vi = 4 + Math.floor(rng() * nb);
  const pq = p.qpts;
  const delta = () => (Math.floor(rng() * amplitude) + 1) * (rng() < 0.5 ? -1 : 1);
  const vx = pq[vi*3], vy = pq[vi*3+1];
  const onBorder = vx === 0 || vx === gx-1 || vy === 0 || vy === gy-1;
  if (onBorder && rng() * 100 >= border_escape) {
    if (vx === 0 || vx === gx-1) pq[vi*3+1] = clampI(vy + delta(), 1, gy-2);
    else                          pq[vi*3]   = clampI(vx + delta(), 1, gx-2);
  } else {
    if (rng() < 0.5) pq[vi*3]   = clampI(vx + delta(), 0, gx-1);
    else              pq[vi*3+1] = clampI(vy + delta(), 0, gy-1);
    if ((pq[vi*3] === 0 || pq[vi*3] === gx-1) && (pq[vi*3+1] === 0 || pq[vi*3+1] === gy-1)) return null;
  }
  return { preview: p, color_data };
}

function applyLaplaceSmooth(preview, color_data, rng) {
  const n = preview.qpts.length / 3;
  const nb = n - 4;
  if (nb <= 0) return null;

  // 1. Pick a random non-corner vertex
  const p = clonePreview(preview);
  const pq = p.qpts;
  const vi = 4 + Math.floor(rng() * nb);
  const vx = pq[vi*3], vy = pq[vi*3+1];

  // 2. Find its neighbors using Delaunay triangulation
  const del = buildDelaunay(p);
  const v_idx_in_del = del.vtx.findIndex(dv => dv.x === vx && dv.y === vy);
  if (v_idx_in_del === -1) return null;

  const neighbors = new Set();
  for (const t of del.getTriangles()) {
    if (t.vtx.includes(v_idx_in_del)) {
      for (const neighbor_vtx_idx of t.vtx) {
        if (neighbor_vtx_idx !== v_idx_in_del) {
          neighbors.add(del.vtx[neighbor_vtx_idx]);
        }
      }
    }
  }

  if (neighbors.size === 0) return null;

  // 3. Calculate barycenter
  let sum_x = 0, sum_y = 0;
  for (const neighbor of neighbors) {
    sum_x += neighbor.x;
    sum_y += neighbor.y;
  }
  const new_x = Math.round(sum_x / neighbors.size);
  const new_y = Math.round(sum_y / neighbors.size);

  const { grid_x: gx, grid_y: gy } = preview;

  // 4. Validate new position
  if (new_x === vx && new_y === vy) return null;
  if (new_x < 0 || new_x >= gx || new_y < 0 || new_y >= gy) return null;
  if ((new_x === 0 || new_x === gx - 1) && (new_y === 0 || new_y === gy - 1)) return null;
  // collision check (flat loop)
  for (let i = 0; i < n; ++i) {
    if (i !== vi && pq[i*3] === new_x && pq[i*3+1] === new_y) return null;
  }

  // 5. Apply move and re-sort (remove vi, find insert position, insert)
  const vidx = pq[vi*3+2];
  pq.copyWithin(vi*3, vi*3+3, n*3);  // remove slot vi

  // find insert position in the now-(n-1)-length array (corners stay at 0..3)
  let ins = n - 1;  // default: append at end
  for (let i = 4; i < n - 1; ++i) {
    if (pq[i*3+1] > new_y || (pq[i*3+1] === new_y && pq[i*3] > new_x)) { ins = i; break; }
  }
  pq.copyWithin((ins+1)*3, ins*3, (n-1)*3);  // shift right from ins
  pq[ins*3] = new_x; pq[ins*3+1] = new_y; pq[ins*3+2] = vidx;

  return { preview: p, color_data };
}

function applyAddVertex(preview, color_data, rng) {
  if (preview.nb_pts >= kPreviewMaxNumVertices) return null;
  const gx = preview.grid_x, gy = preview.grid_y;
  const x = Math.floor(rng() * gx);
  const y = Math.floor(rng() * gy);
  if ((x === 0 || x === gx - 1) && (y === 0 || y === gy - 1)) return null;
  const q = preview.qpts;
  const n = q.length / 3;
  // collision check
  for (let i = 0; i < n; ++i) { if (q[i*3] === x && q[i*3+1] === y) return null; }
  const idx = Math.floor(rng() * preview.nb_colors);
  // find insert position
  let ins = n;
  for (let i = 4; i < n; ++i) {
    if (q[i*3+1] > y || (q[i*3+1] === y && q[i*3] > x)) { ins = i; break; }
  }
  const newQ = new Int16Array(q.length + 3);
  newQ.set(q.subarray(0, ins*3));
  newQ[ins*3] = x; newQ[ins*3+1] = y; newQ[ins*3+2] = idx;
  newQ.set(q.subarray(ins*3), ins*3+3);
  return { preview: { ...preview, qpts: newQ, nb_pts: newQ.length / 3 - 4 }, color_data };
}

function applyRemoveVertex(preview, color_data, rng) {
  const q = preview.qpts;
  const n = q.length / 3;
  const nb = n - 4;
  if (nb <= 0) return null;
  const k = 4 + Math.floor(rng() * nb);
  const newQ = new Int16Array(q.length - 3);
  newQ.set(q.subarray(0, k*3));
  newQ.set(q.subarray(k*3+3), k*3);
  return { preview: { ...preview, qpts: newQ, nb_pts: newQ.length / 3 - 4 }, color_data };
}

function applyMoveColorIndex(preview, color_data, rng) {
  const nv = preview.qpts.length / 3;
  const vi = Math.floor(rng() * nv);
  const newIdx = Math.floor(rng() * preview.nb_colors);
  const p = clonePreview(preview);
  p.qpts[vi*3+2] = newIdx;
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
  const q = preview.qpts;
  const counts = new Array(preview.nb_colors).fill(0);
  for (let i = 2, n = q.length; i < n; i += 3) counts[q[i]]++;
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
  for (let i = 2, n = p.qpts.length; i < n; i += 3) {
    if (p.qpts[i] === minK)    p.qpts[i] = adjRepIdx;
    else if (p.qpts[i] > minK) p.qpts[i] -= 1;
  }
  return { preview: p, color_data: nc };
}

// ---------------------------------------------------------------------------

class TriangleOptimizer {
  constructor(preview, color_data, {
    refGrid                   = null,   // Float32Array zGrid (gx*zoom × gy*zoom, RGBA [0,255])
    zoom                      = 1,
    lambda                    = 0.0001,
    seed                      = 0x12345678,
    score_tolerance           = 0.0002,
    color_change_penalty      = 0,
    num_mutations_per_iter    = 1,
    proba_vertex_move         = 50,
    proba_laplace_smooth      = 0,
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
    maskWeights               = null,   // Float32Array per-pixel importance weights, or null
    roiStrength               = 5,
  } = {}) {
    this.preview    = clonePreview(preview);
    this.color_data = color_data.map(c => ({ y: c.y, co: c.co, cg: c.cg, a: c.a }));
    this.refGrid    = refGrid;
    this.zoom       = zoom;
    this.lambda                 = lambda;
    this.score_tolerance        = score_tolerance;
    this.color_change_penalty   = color_change_penalty;
    this.num_mutations_per_iter = num_mutations_per_iter;
    this.proba_vertex_move      = proba_vertex_move;
    this.proba_laplace_smooth   = proba_laplace_smooth;
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
    this.maskWeights  = maskWeights;
    this.roiStrength  = roiStrength;
    this.iter     = 0;
    this.bestLoss = Infinity;
    this.bestPreview   = null;
    this.bestColorData = null;
    let s = seed >>> 0;
    this._rng = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0x100000000; };
    this._pos2idx = new Int16Array(256 * 256);
  }

  _syncDelIdx(qpts) {
    const p = this._pos2idx;
    for (let i = 0; i < qpts.length; i += 3) p[qpts[i] + qpts[i+1] * 256] = qpts[i+2];
    for (const v of this._cachedDel.vtx) v.idx = p[v.x + v.y * 256];
    for (let i = 0; i < qpts.length; i += 3) p[qpts[i] + qpts[i+1] * 256] = 0;
  }

  _score(preview, color_data, del) {
    const palRGB = buildPalRGB(color_data);
    const distortion = computeCPULoss(preview.grid_x, preview.grid_y, del, palRGB, this.refGrid, this.zoom, this.maskWeights, this.roiStrength);
    const size = encodeSize(preview, color_data);
    return { distortion, score: distortion + this.lambda * size, size };
  }

  init() {
    this._cachedDel = buildDelaunay(this.preview);
    const { distortion, score } = this._score(this.preview, this.color_data, this._cachedDel);
    this.currentScore       = score;
    this.currentDistortion  = distortion;
    this.bestLoss           = score;
    this.vtxAccepted        = 0;
    this.colorAccepted      = 0;
    this.bestPreview   = clonePreview(this.preview);
    this.bestColorData = this.color_data.map(c => ({ y: c.y, co: c.co, cg: c.cg, a: c.a }));
    return score;
  }

  _buildMutationTable() {
    const probas = [
      this.proba_vertex_move, this.proba_laplace_smooth,
      this.proba_vertex_add,  this.proba_vertex_sub,
      this.proba_color_index_move, this.proba_color_move,
      this.proba_color_add,   this.proba_color_sub,
      this.proba_flip_alpha,
    ];
    let total = 0;
    for (let i = 0; i < probas.length; i++) total += probas[i];
    const cdf = new Float64Array(probas.length);
    let acc = 0;
    for (let i = 0; i < probas.length; i++) { acc += probas[i]; cdf[i] = acc / total; }
    return { total, cdf };
  }

  step(iter, maxIter, mutTable) {
    const rng = this._rng;
    const { total, cdf } = mutTable;

    let np = this.preview;
    let nc = this.color_data;

    let isColorChange = false;
    let topologyChanged = false;
    for (let m = 0; m < this.num_mutations_per_iter; ++m) {
      if (total <= 0) break;
      const pick = rng();
      let idx = 0;
      while (idx < cdf.length - 1 && pick > cdf[idx]) idx++;
      let r = null;
      if      (idx === 0) r = np.qpts.length > 12 ? applyMoveVertex(np, nc, rng, this.vertex_amplitude, this.border_escape_prob) : null;
      else if (idx === 1) r = np.qpts.length > 12 ? applyLaplaceSmooth(np, nc, rng) : null;
      else if (idx === 2) r = applyAddVertex(np, nc, rng);
      else if (idx === 3) r = np.qpts.length > 12 ? applyRemoveVertex(np, nc, rng) : null;
      else if (idx === 4) r = applyMoveColorIndex(np, nc, rng);
      else if (idx === 5) r = applyMoveColor(np, nc, rng);
      else if (idx === 6) r = applyAddColor(np, nc, rng);
      else if (idx === 7) r = np.nb_colors > kPreviewMinNumColors ? applyRemoveColor(np, nc, rng) : null;
      else if (idx === 8) r = np.has_alpha ? applyFlipAlpha(np, nc, rng) : null;
      if (r) {
        isColorChange = (idx === 6 || idx === 7);
        if (idx < 4) topologyChanged = true;
        np = r.preview; nc = r.color_data;
      }
    }

    const del = topologyChanged ? buildDelaunay(np) : this._cachedDel;
    if (!topologyChanged) this._syncDelIdx(np.qpts);
    const { distortion, score: newScore, size } = this._score(np, nc, del);
    const tolerance = this.score_tolerance * (maxIter - iter) / maxIter;
    const threshold = isColorChange ? -this.color_change_penalty * this.bestLoss : tolerance;
    const accept = newScore <= this.bestLoss + threshold;

    if (accept) {
      this.preview            = np;
      this.color_data         = nc;
      if (topologyChanged) this._cachedDel = del;
      // else: _cachedDel.vtx.idx already synced to np.qpts (now this.preview.qpts)
      this.currentScore       = newScore;
      this.currentDistortion  = distortion;
      if (!isColorChange) this.vtxAccepted++;
      else                this.colorAccepted++;
      if (newScore < this.bestLoss) {
        this.bestLoss      = newScore;
        this.bestPreview   = clonePreview(np);
        this.bestColorData = nc.map(c => ({ y: c.y, co: c.co, cg: c.cg, a: c.a }));
      }
    } else if (!topologyChanged) {
      this._syncDelIdx(this.preview.qpts);  // restore cached del to accepted state
    }

    this.iter++;
    return { score: this.currentScore, distortion };
  }

  stop() { this._stop = true; }

  async run(maxIter, onProgress) {
    this._stop = false;
    for (let i = 0; i < maxIter && !this._stop; i += this.batch_size) {
      const bsz = Math.min(this.batch_size, maxIter - i);
      let r = { score: this.currentScore, distortion: 0 };
      const mutTable = this._buildMutationTable();
      for (let j = 0; j < bsz && !this._stop; ++j) {
        r = this.step(i + j, maxIter, mutTable);
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
