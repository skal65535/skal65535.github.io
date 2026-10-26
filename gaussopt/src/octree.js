// octree.js — 3D Loose Octree: build / compaction, JSON & PLY-node serialization,
// octree-KNN splat initialization, culling + depth sort, and the outer-loop
// reconcileAndGrow (WLBG prune/split/clone, floater removal, MCMC teleport,
// SA Boltzmann split, overlap merge, stipple seeds).
"use strict";

import { SPLAT_FLOATS, rgbToSh0 } from './colmap.js';
import { clamp, logit, sigmoid, quatToMat3, mat3ToQuat, symEigen3, projectPinhole } from './math.js';

// LooseOctree constructor defaults (also for a .ply / JSON octree that doesn't record them)
export const OCTREE_DEFAULTS = { maxLeafSize: 256, maxDepth: 10, beta: 0.20 };

// Floats per splat in the GPU WLBG moments buffer:
// [0]: W   visibility mass (sum T * alpha)
// [1]: S   sum of squared per-view visibility (N_eff = W^2 / S)
// [2]: R   error mass (sum w * |I_gt - I_hat|)
// [3..5]:  R1 first error moment of delta = X - mu
// [6..11]: R2 second error moment: xx, yy, zz, xy, xz, yz
export const MOMENT_FLOATS = 12;

// Clamp a world-space offset o from the center of splat p (params slice) to the box of
// +-c * scale along each of the splat's own axes, so a child moved by o stays on the splat:
// a flat disc's children stay in its plane, a needle's along its length.
function clampToSplat(p, o, c) {
  const Rm = quatToMat3([p[8], p[9], p[10], p[11]]);  // row-major; axis k = column k
  const out = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const lim = c * Math.exp(p[4 + k]);
    const ok = clamp(Rm[k] * o[0] + Rm[3 + k] * o[1] + Rm[6 + k] * o[2], -lim, lim);
    out[0] += Rm[k] * ok; out[1] += Rm[3 + k] * ok; out[2] += Rm[6 + k] * ok;
  }
  return out;
}

// Root box: exact bounds of all finite points. Scene center / radius (orbit framing):
// per-axis median and 0.85 x the largest 10th-90th percentile span (8192-point sample).
function sceneBounds(arr, count, stride) {
  if (count === 0) return { center: [0, 0, 0], halfSize: 2.0, rootCenter: [0, 0, 0], rootHalfSize: 2.0 };
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) {
    const b = i * stride;
    if (!Number.isFinite(arr[b] + arr[b + 1] + arr[b + 2])) continue;
    for (let a = 0; a < 3; a++) {
      const v = arr[b + a];
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  }
  if (!Number.isFinite(lo[0])) { lo.fill(-1); hi.fill(1); }
  const rootCenter = [0, 1, 2].map(a => 0.5 * (lo[a] + hi[a]));
  const rootHalfSize = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 0.5) * 0.505;

  const n = Math.min(count, 8192), step = Math.max(1, Math.floor(count / n));
  const axes = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  for (let i = 0; i < n; i++) {
    const b = Math.min(count - 1, i * step) * stride;
    for (let a = 0; a < 3; a++) axes[a][i] = arr[b + a];
  }
  for (const v of axes) v.sort();
  const iMed = n >> 1, iLow = Math.floor(n * 0.10), iHigh = Math.min(n - 1, Math.ceil(n * 0.90));
  const span = Math.max(...axes.map(v => v[iHigh] - v[iLow]), 0.5);
  return { center: axes.map(v => v[iMed]), halfSize: span * 0.85, rootCenter, rootHalfSize };
}

class OctreeNode {
  constructor(cx, cy, cz, halfSize, depth) {
    this.cx = cx;
    this.cy = cy;
    this.cz = cz;
    this.halfSize = halfSize;
    this.depth = depth;
    this.children = null;  // Array(8), or null for a leaf
    this.indices = [];     // splat indices during build (null afterwards)
    this.leafId = -1;
    this.start = 0;        // leaf slice [start, start + count) in the compacted order
    this.count = 0;
    this.numCameras = 0;   // M_cell: training frustums intersecting the loose box
    this.state = 0;        // 0 normal, 2 near split, 3 isolated floater (1: unused, see STATE_COLORS)
  }

  get isLeaf() { return this.children === null; }

  octantOf(x, y, z) {
    return (x >= this.cx ? 1 : 0) | (y >= this.cy ? 2 : 0) | (z >= this.cz ? 4 : 0);
  }
}

// Stable LSD radix sort (3 passes of 11 bits) of ids[0..n) by float32 keys > 0, given as their
// uint32 bit patterns (which order like the floats). Writes the sorted ids to `out`.
const _radixHist = new Uint32Array(2048);
function radixSortByPositiveKey(keys, ids, n, tmpKeys, tmpIds, out) {
  let srcK = keys, srcI = ids, dstK = tmpKeys, dstI = tmpIds;
  for (let shift = 0; shift < 32; shift += 11) {
    const h = _radixHist;
    h.fill(0);
    for (let i = 0; i < n; i++) h[(srcK[i] >>> shift) & 2047]++;
    let sum = 0;
    for (let d = 0; d < 2048; d++) { const c = h[d]; h[d] = sum; sum += c; }
    for (let i = 0; i < n; i++) {
      const p = h[(srcK[i] >>> shift) & 2047]++;
      dstK[p] = srcK[i];
      dstI[p] = srcI[i];
    }
    [srcK, dstK] = [dstK, srcK];
    [srcI, dstI] = [dstI, srcI];
  }
  out.set(srcI.subarray(0, n));
}

// 3-sigma world extent of splat b (largest axis, log-scale clamped as in the shaders)
const splatExtent = (params, b) => 3 * Math.exp(Math.min(3, Math.max(params[b + 4], params[b + 5], params[b + 6])));

// Culling: near plane, frustum widening (node and splat tests must agree, so that the octree
// path keeps what the exhaustive one does) and the opacity logit below which a splat is skipped
const CULL_ZNEAR = 0.05, CULL_FRUSTUM = 1.25, CULL_MIN_LOGIT = -5.5;

// Per-splat cull of the octree and exhaustive sorts, over splats [start, end) of `params`
// (`view`: column-major world->camera, +Z forward): appends the view z / index of the kept ones
// to keys / ids from n, returns the new n. Culled: near-transparent, behind the near plane, or
// outside the |x| fx, |y| fy <= 1.25 z cone with its 3-sigma extent too (large splats centered
// off-screen still cover the frame edges). fx, fy <= 0 disables the xy test.
function cullSplats(params, start, end, view, fx, fy, keys, ids, n) {
  const zx = view[2], zy = view[6], zz = view[10], zw = view[14];
  const xx = view[0], xy = view[4], xz = view[8],  xw = view[12];
  const yx = view[1], yy = view[5], yz = view[9],  yw = view[13];
  const cullXY = fx > 0 && fy > 0;
  for (let idx = start; idx < end; idx++) {
    const b = idx * SPLAT_FLOATS;
    if (params[b + 3] < CULL_MIN_LOGIT) continue;
    const X = params[b], Y = params[b + 1], Z = params[b + 2];
    const z = zx * X + zy * Y + zz * Z + zw;
    if (z <= CULL_ZNEAR) continue;
    if (cullXY) {
      const ax = Math.abs(xx * X + xy * Y + xz * Z + xw) * fx, ay = Math.abs(yx * X + yy * Y + yz * Z + yw) * fy;
      const zf = z * CULL_FRUSTUM;
      if (ax > zf || ay > zf) {
        const e = splatExtent(params, b);
        if (ax > zf + e * fx || ay > zf + e * fy) continue;
      }
    }
    keys[n] = z;
    ids[n++] = idx;
  }
  return n;
}

// Node table entry (PLY `octree_node` element / JSON): leaves store their slice, internal nodes
// their first child's index (children are 8 contiguous entries) and depth.
const nodeEntry = (n) => ({
  cx: n.cx, cy: n.cy, cz: n.cz, halfSize: n.halfSize,
  isLeaf: n.isLeaf ? 1 : 0,
  firstChildOrStart: n.isLeaf ? n.start : -1,
  countOrDepth: n.isLeaf ? n.count : n.depth,
  state: n.state | 0,
});

// Leaf box wireframe: 12 edges as (corner0, corner1) in [-1, 1]^3
const BOX_EDGES = [
  [-1,-1,-1,  1,-1,-1], [ 1,-1,-1,  1, 1,-1], [ 1, 1,-1, -1, 1,-1], [-1, 1,-1, -1,-1,-1],
  [-1,-1, 1,  1,-1, 1], [ 1,-1, 1,  1, 1, 1], [ 1, 1, 1, -1, 1, 1], [-1, 1, 1, -1,-1, 1],
  [-1,-1,-1, -1,-1, 1], [ 1,-1,-1,  1,-1, 1], [ 1, 1,-1,  1, 1, 1], [-1, 1,-1, -1, 1, 1],
];
// ABGR by leaf state: normal cyan, active green (state 1, DESIGN.md Mode B: never assigned yet),
// near split orange, isolated red
const STATE_COLORS = [0x88ffcc44, 0xff44ff44, 0xff2299ff, 0xff3333ff];

export class LooseOctree {
  constructor({ maxLeafSize = OCTREE_DEFAULTS.maxLeafSize, maxDepth = OCTREE_DEFAULTS.maxDepth, beta = OCTREE_DEFAULTS.beta } = {}) {
    this.maxLeafSize = maxLeafSize;
    this.maxDepth    = maxDepth;
    this.beta        = beta;  // looseness: a node's loose box has side (1 + 2 beta) L
    this.root        = null;
    this.leaves      = [];
    this.leafIndices = null;  // index-only mode: leaf slot -> original splat index
    this.lastPerm    = null;  // after build(): new splat index -> old index
    this.sceneCenter = [0, 0, 0];
    this.sceneRadius = 2.0;
    this.lastStats   = { split: 0, cloned: 0, pruned: 0, floaters: 0, teleported: 0, merged: 0, seeded: 0 };
  }

  // Build the tree over xyz at arr[i * stride] and number the leaves; leaf.indices are kept
  // for the caller to lay out.
  _buildTree(arr, count, stride) {
    const { center, halfSize, rootCenter, rootHalfSize } = sceneBounds(arr, count, stride);
    this.sceneCenter = center;
    this.sceneRadius = halfSize;
    this.root = new OctreeNode(rootCenter[0], rootCenter[1], rootCenter[2], rootHalfSize, 0);
    this.root.indices = Array.from({ length: count }, (_, i) => i);
    this._splitRecursive(this.root, arr, stride);
    this.leaves = [];
    this._collectLeaves(this.root);
    let cursor = 0;
    for (let lid = 0; lid < this.leaves.length; lid++) {
      const leaf = this.leaves[lid];
      leaf.leafId = lid;
      leaf.start  = cursor;
      leaf.count  = leaf.indices.length;
      cursor += leaf.count;
    }
  }

  // Build from `params` (count * SPLAT_FLOATS) and reorder it in place so that each leaf is a
  // contiguous slice; slot 7 gets the leaf id. Sets lastPerm (for state kept outside params, e.g. Adam).
  build(params, count, images = null) {
    this._buildTree(params, count, SPLAT_FLOATS);
    const compacted = new Float32Array(count * SPLAT_FLOATS);
    this.lastPerm = new Int32Array(count);
    for (const leaf of this.leaves) {
      if (leaf.count > this.maxLeafSize * 0.8) leaf.state = 2;
      for (let k = 0; k < leaf.count; k++) {
        const src = leaf.indices[k], dst = leaf.start + k;
        compacted.set(params.subarray(src * SPLAT_FLOATS, (src + 1) * SPLAT_FLOATS), dst * SPLAT_FLOATS);
        compacted[dst * SPLAT_FLOATS + 7] = leaf.leafId;
        this.lastPerm[dst] = src;
      }
      leaf.indices = null;
    }
    params.set(compacted);
    if (images && images.length > 0) this.updateCameraAndNeighborStats(images);
    return this;
  }

  // Build an index (leafIndices) without reordering the splats: for viewers whose buffers are
  // already on the GPU in file order (`splats/index.html`).
  buildIndexOnly(positions, count, stride = 3) {
    this._buildTree(positions, count, stride);
    this.leafIndices = new Uint32Array(count);
    for (const leaf of this.leaves) {
      this.leafIndices.set(leaf.indices, leaf.start);
      leaf.indices = null;
    }
    return this;
  }

  _splitRecursive(node, arr, stride) {
    if (node.indices.length <= this.maxLeafSize || node.depth >= this.maxDepth) return;
    const h = node.halfSize * 0.5;
    node.children = new Array(8);
    for (let oct = 0; oct < 8; oct++) {
      node.children[oct] = new OctreeNode(
        node.cx + ((oct & 1) ? h : -h), node.cy + ((oct & 2) ? h : -h), node.cz + ((oct & 4) ? h : -h),
        h, node.depth + 1);
    }
    for (const idx of node.indices) {
      const b = idx * stride;
      node.children[node.octantOf(arr[b], arr[b + 1], arr[b + 2])].indices.push(idx);
    }
    node.indices = [];
    for (const c of node.children) this._splitRecursive(c, arr, stride);
  }

  // node.maxExtent = largest 3-sigma splat extent in the subtree, so frustum culling of a
  // node also keeps large splats poking out of its loose bounds. Call when params change.
  updateExtents(params) {
    const visit = (node) => {
      let m = 0;
      if (node.isLeaf) {
        for (let i = node.start, end = node.start + node.count; i < end; i++) m = Math.max(m, splatExtent(params, i * SPLAT_FLOATS));
      } else {
        for (const c of node.children) if (c) m = Math.max(m, visit(c));
      }
      node.maxExtent = m;
      return m;
    };
    if (this.root) visit(this.root);
  }

  // Non-empty leaves in depth-first octant order
  _collectLeaves(node) {
    if (node.isLeaf) {
      if ((node.indices && node.indices.length > 0) || node.count > 0) this.leaves.push(node);
      return;
    }
    for (const c of node.children) if (c) this._collectLeaves(c);
  }

  // ── Serialization (.ply `octree_node` element / .octree.json) ──

  // Flat node table: root at 0; each internal node's 8 children are contiguous, and subtrees
  // are appended depth-first after their siblings.
  serializeNodes() {
    if (!this.root) return [];
    const nodes = [nodeEntry(this.root)];
    const expand = (node, entry) => {
      entry.firstChildOrStart = nodes.length;
      const base = nodes.length;
      for (const c of node.children) nodes.push(nodeEntry(c));
      node.children.forEach((c, k) => { if (!c.isLeaf) expand(c, nodes[base + k]); });
    };
    if (!this.root.isLeaf) expand(this.root, nodes[0]);
    return nodes;
  }

  static fromSerializedNodes(meta, nodes, leafIndices = null) {
    const tree = new LooseOctree({
      maxLeafSize: meta.maxLeafSize || undefined,
      maxDepth:    meta.maxDepth    || undefined,
      beta:        meta.beta        ?? undefined,
    });
    tree.sceneCenter = meta.sceneCenter || [0, 0, 0];
    tree.sceneRadius = meta.sceneRadius || 2.0;
    tree.leafIndices = leafIndices ? new Uint32Array(leafIndices) : null;
    if (!nodes || nodes.length === 0) return tree;

    const buildNode = (idx, depth) => {
      const d = nodes[idx];
      const n = new OctreeNode(d.cx, d.cy, d.cz, d.halfSize, depth);
      n.indices = null;
      n.state   = d.state || 0;
      if (d.isLeaf) {
        n.start = d.firstChildOrStart;
        n.count = d.countOrDepth;
      } else {
        n.children = Array.from({ length: 8 }, (_, oct) => buildNode(d.firstChildOrStart + oct, depth + 1));
      }
      return n;
    };
    tree.root = buildNode(0, 0);
    tree._collectLeaves(tree.root);
    tree.leaves.forEach((leaf, i) => { leaf.leafId = i; });
    return tree;
  }

  toJSON() {
    return {
      format:      'gaussopt_octree_v1',
      maxLeafSize: this.maxLeafSize,
      maxDepth:    this.maxDepth,
      beta:        this.beta,
      sceneCenter: this.sceneCenter,
      sceneRadius: this.sceneRadius,
      nodes:       this.serializeNodes(),
      leafIndices: this.leafIndices ? Array.from(this.leafIndices) : null,
    };
  }

  static fromJSON(obj) {
    if (typeof obj === 'string') obj = JSON.parse(obj);
    return LooseOctree.fromSerializedNodes(obj, obj.nodes, obj.leafIndices);
  }

  // Splats from COLMAP points3D (opacity 0.5, point color), built into the octree, with scales
  // from neighbors found in the point's leaf (+-16 slots, i.e. into adjacent leaves):
  // isotropic 0.8 x mean 3-NN distance; with `aniso`, oriented along the PCA of the 8 nearest
  // (flat discs on surfaces, thin along the normal).
  initSplatsFromPoints3D(points3D, images = null, { aniso = true } = {}) {
    const { count, positions, colors } = points3D;
    const params = new Float32Array(count * SPLAT_FLOATS);
    for (let i = 0; i < count; i++) {
      const b = i * SPLAT_FLOATS;
      for (let c = 0; c < 3; c++) params[b + c] = positions[3 * i + c];
      params[b + 3] = logit(0.5);
      params[b + 8] = 1.0;  // identity quaternion
      for (let c = 0; c < 3; c++) params[b + 12 + c] = rgbToSh0(colors[3 * i + c] / 255);
    }
    this.build(params, count, images);

    const minScale = this.sceneRadius * 1e-4, maxScale = this.sceneRadius * 0.15;
    const K_NN = 8, LOG2 = Math.LN2, LOG8 = Math.log(8), LOG10 = Math.log(10);
    const nbD = new Float64Array(512), nbJ = new Int32Array(512);

    for (const leaf of this.leaves) {
      const s0 = leaf.start, s1 = s0 + leaf.count;
      const searchStart = Math.max(0, s0 - 16), searchEnd = Math.min(count, s1 + 16);
      for (let i = s0; i < s1; i++) {
        const bi = i * SPLAT_FLOATS;
        const xi = params[bi], yi = params[bi + 1], zi = params[bi + 2];
        let d1 = Infinity, d2 = Infinity, d3 = Infinity;  // 3 smallest squared distances
        let nbN = 0;
        for (let j = searchStart; j < searchEnd; j++) {
          if (i === j) continue;
          const bj = j * SPLAT_FLOATS;
          const dx = xi - params[bj], dy = yi - params[bj + 1], dz = zi - params[bj + 2];
          const d = dx * dx + dy * dy + dz * dz;
          if (d < d1)      { d3 = d2; d2 = d1; d1 = d; }
          else if (d < d2) { d3 = d2; d2 = d; }
          else if (d < d3) { d3 = d; }
          if (aniso && nbN < 512) { nbD[nbN] = d; nbJ[nbN++] = j; }
        }
        const meanDist = (d3 < Infinity) ? (Math.sqrt(d1) + Math.sqrt(d2) + Math.sqrt(d3)) / 3
                       : (d1 < Infinity) ? Math.sqrt(d1) : leaf.halfSize * 0.25;
        const s = clamp(meanDist * 0.8, minScale, maxScale);
        params[bi + 4] = params[bi + 5] = params[bi + 6] = Math.log(s);
        if (!aniso || nbN < K_NN) continue;

        // K nearest (partial selection sort), then PCA of the point + its K neighbors
        for (let a = 0; a < K_NN; a++) {
          let best = a;
          for (let c = a + 1; c < nbN; c++) if (nbD[c] < nbD[best]) best = c;
          [nbD[a], nbD[best]] = [nbD[best], nbD[a]];
          [nbJ[a], nbJ[best]] = [nbJ[best], nbJ[a]];
        }
        let mx = xi, my = yi, mz = zi;
        for (let a = 0; a < K_NN; a++) {
          const bj = nbJ[a] * SPLAT_FLOATS;
          mx += params[bj]; my += params[bj + 1]; mz += params[bj + 2];
        }
        mx /= K_NN + 1; my /= K_NN + 1; mz /= K_NN + 1;
        let cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
        for (let a = -1; a < K_NN; a++) {
          const bj = (a < 0 ? i : nbJ[a]) * SPLAT_FLOATS;
          const dx = params[bj] - mx, dy = params[bj + 1] - my, dz = params[bj + 2] - mz;
          cxx += dx * dx; cxy += dx * dy; cxz += dx * dz; cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
        }
        const { vals, vecs: v } = symEigen3(cxx, cxy, cxz, cyy, cyz, czz);
        const lmax = Math.max(vals[0], vals[1], vals[2]);
        if (!(lmax > 0)) continue;
        // With 9 noisy points, ratios up to ~2:1 are mostly sampling noise: per axis, the
        // log-ratio a = log(s_max / s_axis) (capped at log 10) becomes a * smoothstep(log 2, log 8, a),
        // so mild anisotropy -> round, strong (planes, lines) -> kept.
        let wMax = 0;
        const la = [0, 0, 0];
        for (let a = 0; a < 3; a++) {
          const lr = Math.min(LOG10, -Math.log(Math.max(1e-6, Math.sqrt(Math.max(0, vals[a]) / lmax))));
          const t = clamp((lr - LOG2) / (LOG8 - LOG2), 0, 1);
          const w = t * t * (3 - 2 * t);
          la[a] = lr * w;
          wMax = Math.max(wMax, w);
        }
        if (wMax === 0) continue;  // stays round
        const sMax = Math.min(maxScale, s * (1 + 0.2 * wMax));
        for (let a = 0; a < 3; a++) params[bi + 4 + a] = Math.log(Math.max(minScale, sMax * Math.exp(-la[a])));
        // eigenvectors are the columns of v: make the frame right-handed
        const det = v[0] * (v[4] * v[8] - v[5] * v[7])
                  - v[1] * (v[3] * v[8] - v[5] * v[6])
                  + v[2] * (v[3] * v[7] - v[4] * v[6]);
        if (det < 0) { v[2] = -v[2]; v[5] = -v[5]; v[8] = -v[8]; }
        params.set(mat3ToQuat(v), bi + 8);
      }
    }
    return { count, params };
  }

  // Per leaf: numCameras = training frustums hitting its loose bounding sphere (M_cell); leaves
  // with <= 3 splats and no other leaf within 1.25 x (sum of half sizes) get state 3 (isolated).
  updateCameraAndNeighborStats(images) {
    const trainImgs = images.filter(im => !im.isValidation);
    const uv = [0, 0];
    for (const leaf of this.leaves) {
      const looseR = leaf.halfSize * (1 + 2 * this.beta) * 1.732;
      let camHits = 0;
      for (const { R, t, camera } of trainImgs) {
        const zc = projectPinhole(R, t, camera, leaf.cx, leaf.cy, leaf.cz, uv, 0.1);
        if (zc + looseR < 0.05) continue;
        const [u, v] = uv, rPx = camera.fx * (looseR / Math.max(0.1, zc));
        if (u + rPx >= 0 && u - rPx <= camera.width && v + rPx >= 0 && v - rPx <= camera.height) camHits++;
      }
      leaf.numCameras = camHits;
    }
    for (const a of this.leaves) {
      if (a.count > 3) continue;
      const hasNeighbor = this.leaves.some(b => {
        const reach = (a.halfSize + b.halfSize) * 1.25;
        return b !== a && Math.abs(a.cx - b.cx) <= reach && Math.abs(a.cy - b.cy) <= reach && Math.abs(a.cz - b.cz) <= reach;
      });
      if (!hasNeighbor) a.state = 3;
    }
  }

  // Visible non-empty leaves in a front-to-back visibility order for the camera at `camCenter`.
  // `view` (optional): column-major world->camera matrix for frustum culling of loose bounding
  // spheres; zSign = -1 for -Z-forward cameras (`splats/index.html`); fx = fy = 0 skips the xy test.
  // Children go in a valid visibility order (a child precedes any child flipped on a superset of
  // its axes relative to the camera octant), chosen to approximate view-z order: flipping the
  // most ray-aligned axis adds the most depth, so it goes last (DESIGN.md, E1).
  traverseFrontToBack(camCenter, view = null, fx = 1.0, fy = 1.0, zSign = 1.0) {
    const out = [];
    if (!this.root) return out;
    const betaScale = 1 + 2 * this.beta;

    const visit = (node) => {
      if (view) {
        const xc = view[0] * node.cx + view[4] * node.cy + view[8]  * node.cz + view[12];
        const yc = view[1] * node.cx + view[5] * node.cy + view[9]  * node.cz + view[13];
        const zc = (view[2] * node.cx + view[6] * node.cy + view[10] * node.cz + view[14]) * zSign;
        const r  = node.halfSize * betaScale * 1.73205 + (node.maxExtent || 0);
        if (zc + r < CULL_ZNEAR) return;  // completely behind the camera
        if (zc - r > CULL_ZNEAR && fx > 0 && fy > 0) {  // the splat cull's cone, widened by r
          if (Math.abs(xc) * fx > (zc + r) * CULL_FRUSTUM + r * fx) return;
          if (Math.abs(yc) * fy > (zc + r) * CULL_FRUSTUM + r * fy) return;
        }
      }
      if (node.isLeaf) {
        if (node.count > 0) out.push(node);
        return;
      }
      // Camera octant relative to `node`: child = flips ^ mask
      const mask = (camCenter[0] >= node.cx ? 1 : 0) |
                   (camCenter[1] >= node.cy ? 2 : 0) |
                   (camCenter[2] >= node.cz ? 4 : 0);
      // Axis bits sorted by |camera - node| (a: least ray-aligned .. c: most). Flip-set order:
      // 0 < a < b < {ab, c} < ac < bc < abc; only ab vs c depends on the actual values.
      const w = [Math.abs(node.cx - camCenter[0]), Math.abs(node.cy - camCenter[1]), Math.abs(node.cz - camCenter[2])];
      let a = 0, b = 1, c = 2, t;
      if (w[a] > w[b]) { t = a; a = b; b = t; }
      if (w[b] > w[c]) { t = b; b = c; c = t; }
      if (w[a] > w[b]) { t = a; a = b; b = t; }
      const A = 1 << a, B = 1 << b, C = 1 << c;
      const abFirst = (w[a] + w[b] <= w[c]);
      const order = [0, A, B, abFirst ? (A | B) : C, abFirst ? C : (A | B), A | C, B | C, 7];
      for (let k = 0; k < 8; k++) visit(node.children[order[k] ^ mask]);
    };
    visit(this.root);
    return out;
  }

  // Octree-culled visible splats, exactly sorted front-to-back by view z -> outIdx[0..numVisible).
  // The octree only culls: leaf order is a per-ray order for points, not for extended splats (E1).
  sortOctreeFrontToBack(params, camCenter, view, fx, fy, outIdx) {
    const visibleLeaves = this.traverseFrontToBack(camCenter, view, fx, fy, 1.0);
    let total = 0;
    for (const leaf of visibleLeaves) total += leaf.count;
    if (!this._sortKeys || this._sortKeys.length < total) {
      const cap = Math.max(1024, total * 2);
      this._sortKeys = new Float32Array(cap);
      this._sortIds  = new Uint32Array(cap);
      this._tmpKeys  = new Uint32Array(cap);
      this._tmpIds   = new Uint32Array(cap);
    }
    const keys = this._sortKeys, ids = this._sortIds;
    let n = 0;
    for (const leaf of visibleLeaves) n = cullSplats(params, leaf.start, leaf.start + leaf.count, view, fx, fy, keys, ids, n);
    radixSortByPositiveKey(new Uint32Array(keys.buffer, 0, n), ids, n, this._tmpKeys, this._tmpIds, outIdx);
    return { numVisible: n };
  }

  // Viewer sort (index-only octree, `splats/index.html`): visible leaves front-to-back, each
  // insertion-sorted by depth; no per-splat xy cull. Returns the number written to outIdx.
  sortIndexFrontToBack(positions, stride, camCenter, view, fx, fy, outIdx, zSign = -1.0) {
    const visibleLeaves = this.traverseFrontToBack(camCenter, view, fx, fy, zSign);
    const vx = view[2] * zSign, vy = view[6] * zSign, vz = view[10] * zSign, vw = view[14] * zSign;
    let maxN = this.maxLeafSize;
    for (const leaf of visibleLeaves) maxN = Math.max(maxN, leaf.count);
    if (!this._leafIds || this._leafIds.length < maxN) {
      this._leafIds    = new Uint32Array(maxN);
      this._leafDepths = new Float32Array(maxN);
    }
    const ids = this._leafIds, depths = this._leafDepths, indexMap = this.leafIndices;
    let writePos = 0;
    for (const leaf of visibleLeaves) {
      // a leaf over maxLeafSize (maxDepth reached: a few far splats can make the root huge) gets a sort in
      // O(k log k), the insertion sort being O(k^2) (the gallery's jardin7-latest, a leaf of 82k: 39 ms a frame, 6 after)
      const big = leaf.count > this.maxLeafSize;
      let valid = 0;
      for (let k = leaf.start, end = leaf.start + leaf.count; k < end; k++) {
        const idx = indexMap ? indexMap[k] : k;
        const b = idx * stride;
        const zv = vx * positions[b] + vy * positions[b + 1] + vz * positions[b + 2] + vw;
        if (zv <= 0.05) continue;
        // stable insertion sort of the f32 depth into ids/depths[0..valid]
        const z = Math.fround(zv);
        let j = valid++;
        if (!big) for (; j > 0 && depths[j - 1] > z; j--) { ids[j] = ids[j - 1]; depths[j] = depths[j - 1]; }
        ids[j] = idx;
        depths[j] = z;
      }
      if (big) {  // stable too: ties in file order
        const order = Uint32Array.from({ length: valid }, (_, k) => k).sort((a, b) => depths[a] - depths[b] || a - b);
        for (let k = 0; k < valid; k++) outIdx[writePos++] = ids[order[k]];
      } else {
        for (let k = 0; k < valid; k++) outIdx[writePos++] = ids[k];
      }
    }
    return writePos;
  }

  // Reference for tests/ply_octree_test.js (E1): exhaustive cull (same test as sortOctreeFrontToBack) + exact z-sort
  sortGlobalExact(params, count, view, fx, fy, outIdx) {
    const keys = new Float32Array(count), ids = new Uint32Array(count);
    const n = cullSplats(params, 0, count, view, fx, fy, keys, ids, 0);
    const pairs = Array.from({ length: n }, (_, k) => ({ id: ids[k], z: keys[k] }));
    pairs.sort((a, b) => a.z - b.z);
    for (let i = 0; i < pairs.length; i++) outIdx[i] = pairs[i].id;
    return pairs.length;
  }

  // Leaf box line list for the 3D overlay: per vertex [x, y, z, ABGR color as u32]
  buildWireframeVertices({ showLoose = false } = {}) {
    const scale = showLoose ? (1 + 2 * this.beta) : 1.0;
    const verts = new Float32Array(this.leaves.length * BOX_EDGES.length * 8);
    const u32   = new Uint32Array(verts.buffer);
    let ptr = 0;
    for (const leaf of this.leaves) {
      const h = leaf.halfSize * scale;
      const col = STATE_COLORS[leaf.state] ?? STATE_COLORS[0];
      for (const e of BOX_EDGES) {
        for (let v = 0; v < 2; v++, ptr += 4) {
          verts[ptr + 0] = leaf.cx + e[3 * v + 0] * h;
          verts[ptr + 1] = leaf.cy + e[3 * v + 1] * h;
          verts[ptr + 2] = leaf.cz + e[3 * v + 2] * h;
          u32[ptr + 3]   = col;
        }
      }
    }
    return { buffer: verts, vertexCount: this.leaves.length * BOX_EDGES.length * 2 };
  }

  // ── Outer loop: prune / floaters / split / merge / seeds, then rebuild ──
  // Returns { count, params (new, compacted), srcIdx, splitVectors, stats }. srcIdx[k] is the
  // input splat whose optimizer state new splat k inherits (-1 = fresh), in the order BEFORE the
  // final build(); compose with this.lastPerm. `params` is also modified in place (opacity of
  // MCMC teleport parents and of decayed occluders), so the caller should switch to the result.
  reconcileAndGrow(params, count, moments, images, opts = {}) {
    const {
      maxSplats      = 100000,
      pruneOpacity   = 0.008,
      hysteresis     = 0.35,
      enableFloaters = true,
      floaterDepth   = 'observed',  // nearCameraCone's depth
      floaterSpare   = null,        // nearCameraCone's spare: { positions, count, radius } (the SfM points)
      splitMinViews  = 0,           // > 0: no split / clone of a splat seen from fewer views (N_eff / visitsPerView)
      visitsPerView  = 1,           // steps per training view since the moments were zeroed: N_eff = W^2 / S counts
                                    // view visits (S grows each step), so a photo set's one view counts K / n times
      splitMinErr    = 0,           // > 0: no split / clone of a splat whose mean error under its footprint, R / W, is below (E9)
      growRate       = 0,           // > 0: at most this share of `count` new splats, the highest R first (E9)
      splitInPlane   = false,       // split / clone offsets off the splat's thinnest axis (E9)
      growScore      = null,        // Float32Array(count): splits / clones go by this score > growThreshold instead of
      growThreshold  = 0,           // R > Tu (E9's grow_by 'grad': the mean positional gradient per view, as brush)
      splitMode      = 'error',     // 'error': along the error moments (splitSplat); 'shape': within the splat (splitByShape)
      enableSplit    = true,
      enableMerge    = true,
      optMethod      = 'adam',  // 'adam' | 'monte_carlo' | 'sim_anneal'
      relocateBy     = 'opacity',  // monte_carlo: the live splats the dead ones move onto, drawn ~ 'opacity' or 'error' (R)
      relocateMode   = 'copy',  // ... as copies in place (3DGS-MCMC) or 'shape': split within their shape (brush, relocate)
      saTemp         = 0.15,    // current SA temperature T(t)
      seedSplats     = null,    // Float32Array of splats to append (stipple seeding)
      repairMask     = null,    // Uint8Array(count), 1: in a stuck region (regions.js RegionTracker, E7): its splits
      repairBoost    = 0,       // need R > Tu / (1 + repairBoost), and monte_carlo relocates onto it (1 + repairBoost) as often
      keepScore      = null,    // Float32Array(count): over maxSplats (the cap lease's consolidation, trainer.js
                                // splatBudget), the rows of the lowest scores go (default: the window's W)
      bounds         = null,    // { center, halfSize }: a center out of this cube is dead, as a transparent splat
                                // (monte_carlo: its noise has no bound, trainer.js sceneBox)
    } = opts;
    const err = errorStats(moments, count, hysteresis);
    const growCut = growRate > 0 ? growthCut(moments, count, growScore ?? err.Tu, growThreshold, splitMinErr, Math.ceil(growRate * count)) : 0;
    // stipple seeds come before splits: the room they need is kept free (at the cap, the room the prunes free)
    const nSeeds = seedSplats ? seedSplats.length / SPLAT_FLOATS : 0;
    const out = new KeptSplats(params, count);
    const stats = { split: 0, cloned: 0, pruned: 0, floaters: 0, floaterRules: [0, 0, 0], merged: 0, teleported: 0, seeded: 0 };
    // monte_carlo: the dead splats (pruned, floaters) are not dropped but moved onto live ones, after the pass
    let dead = 0;
    const teleport = () => optMethod === 'monte_carlo' && ++dead > 0;
    const nearCamera = nearCameraCone(images, { depth: floaterDepth, spare: floaterSpare });

    for (let i = 0; i < count; i++) {
      const b = i * SPLAT_FLOATS, m = i * MOMENT_FLOATS;
      const alpha = sigmoid(params[b + 3]);
      const maxS = Math.exp(Math.max(params[b + 4], params[b + 5], params[b + 6]));
      const leaf = this.leaves[clamp(params[b + 7] | 0, 0, this.leaves.length - 1)];
      const W = moments[m], S = moments[m + 1], R = moments[m + 2];

      // A. Prune transparent / oversized splats, and those out of `bounds`
      const away = bounds && Math.max(Math.abs(params[b] - bounds.center[0]), Math.abs(params[b + 1] - bounds.center[1]),
                                      Math.abs(params[b + 2] - bounds.center[2])) > bounds.halfSize;
      if (alpha < pruneOpacity || maxS > this.sceneRadius * 0.5 || away) {
        if (!teleport()) stats.pruned++;
        continue;
      }
      // B. Floaters (removed, or teleported); high-error occluders lose opacity
      if (enableFloaters) {
        const rule = isFloater(params, b, alpha, leaf, W, S, R, nearCamera);
        if (rule) {
          stats.floaterRules[rule - 1]++;
          if (!teleport()) stats.floaters++;
          continue;
        }
        if (W > 1.0 && (R / W) > 0.22 && alpha > 0.15) params[b + 3] = logit(alpha * 0.65);
      }
      // C. Split when R > Tu (sim_anneal: also Tl < R <= Tu, with Boltzmann probability).
      const Tu = repairMask?.[i] ? err.Tu / (1 + repairBoost) : err.Tu;
      const G = growScore ? growScore[i] : R;
      let shouldSplit = growScore ? G > growThreshold : (R > Tu && err.Tu > 1e-4);
      if (!growScore && !shouldSplit && optMethod === 'sim_anneal' && R > err.Tl && err.Tu > 1e-4) {
        const boltzmann = Math.exp(-(err.Tu - R) / Math.max(1e-6, saTemp * err.avgR));
        if (Math.random() < boltzmann * 0.35) shouldSplit = true;
      }
      if (shouldSplit && splitMinViews > 0 && !(S > 1e-8 && (W * W) / S / visitsPerView >= splitMinViews)) {
        shouldSplit = false;
        stats.splitTooFewViews = (stats.splitTooFewViews ?? 0) + 1;
      }
      if (shouldSplit && splitMinErr > 0 && !(W > 1e-8 && R / W > splitMinErr)) {
        shouldSplit = false;
        stats.splitLowError = (stats.splitLowError ?? 0) + 1;
      }
      if (shouldSplit && G < growCut) {
        shouldSplit = false;
        stats.splitOverRate = (stats.splitOverRate ?? 0) + 1;
      }
      // budget: splats kept so far + the ones still to visit + this one's extra child + the seeds, or the
      // dead ones relocated (monte_carlo) when there are more of them
      if (enableSplit && shouldSplit && out.n + (count - i) + 1 + Math.max(nSeeds, dead) <= maxSplats) {
        const how = splitMode === 'shape' ? splitByShape(out, i) : splitSplat(out, i, moments, maxS, this.sceneRadius, splitInPlane);
        if (how) { stats[how]++; continue; }
      }
      out.add(i, i);
    }

    // monte_carlo: the dead become the seeds first, the others copies of live splats, within the cap
    const toCopy = Math.max(0, Math.min(dead - nSeeds, maxSplats - out.n - nSeeds));
    if (toCopy > 0) stats.teleported = relocate(out, toCopy, relocateBy === 'error' ? moments : null, repairMask, repairBoost, relocateMode);
    stats.pruned += Math.max(0, dead - nSeeds) - stats.teleported;
    if (enableMerge) stats.merged = mergeOverlaps(out);
    stats.trimmed = trimToCap(out, maxSplats, keepScore ?? moments, keepScore ? 1 : MOMENT_FLOATS);
    const res = compactWithSeeds(out, seedSplats, maxSplats);
    stats.seeded = res.nSeeds;
    this.lastStats = stats;
    this.build(res.params, res.count, images);
    return { count: res.count, params: res.params, srcIdx: res.srcIdx, splitVectors: new Float32Array(out.splitVectors), stats };
  }
}

// ── reconcileAndGrow() phases ────────────────────────────────────────────────

// growRate's cut: the score of the `room`-th largest among the splats that pass the split's gates (R > Tu, or with
// a growScore array its score > threshold; R / W > minErr); 0 when fewer pass (no cut)
function growthCut(moments, count, TuOrScore, threshold, minErr, room) {
  const rs = [], score = typeof TuOrScore === 'number' ? null : TuOrScore;
  for (let i = 0; i < count; i++) {
    const W = moments[i * MOMENT_FLOATS], R = moments[i * MOMENT_FLOATS + 2], G = score ? score[i] : R;
    if ((score ? G > threshold : R > TuOrScore) && !(minErr > 0 && !(W > 1e-8 && R / W > minErr))) rs.push(G);
  }
  if (rs.length <= room) return 0;
  return Float64Array.from(rs).sort().reverse()[room - 1];
}

// Mean error mass over visible splats (W > 1e-3) -> hysteresis thresholds Tu / Tl
function errorStats(moments, count, hysteresis) {
  let totalR = 0, activeCount = 0;
  for (let i = 0; i < count; i++) {
    const W = moments[i * MOMENT_FLOATS], R = moments[i * MOMENT_FLOATS + 2];
    if (!(W > 1e-3)) continue;
    totalR += R;
    activeCount++;
  }
  const avgR = activeCount > 0 ? totalR / activeCount : 0;
  return { avgR, Tu: (1 + hysteresis) * avgR, Tl: (1 - hysteresis) * avgR };
}

// Output splats of the pass, in one growable buffer (a small array per splat cost ~100 ms per
// rebalance at 300k splats): row k of `buf`, its source input splat src[k] (-1 = new) and
// alive[k] (0 = merged away); keptOf maps an input splat to its row.
class KeptSplats {
  constructor(params, count) {
    this.params = params;
    this.n = 0;
    this._grow(count + (count >> 2) + 64);
    this.keptOf = new Int32Array(count).fill(-1);
    this.splitVectors = [];  // split axes, for the 3D overlay
  }
  _grow(cap) {
    const buf = new Float32Array(cap * SPLAT_FLOATS), src = new Int32Array(cap);
    if (this.buf) { buf.set(this.buf.subarray(0, this.n * SPLAT_FLOATS)); src.set(this.src.subarray(0, this.n)); }
    this.buf = buf; this.src = src; this.cap = cap;
  }
  // a copy of input splat i as a new row (src = i: its kept copy, -1: a new splat); returns the row
  add(i, src) {
    if (this.n === this.cap) this._grow(2 * this.cap);
    const k = this.n++;
    this.buf.set(this.params.subarray(i * SPLAT_FLOATS, (i + 1) * SPLAT_FLOATS), k * SPLAT_FLOATS);
    this.src[k] = src;
    if (src >= 0) this.keptOf[src] = k;
    return k;
  }
  row(k) { return this.buf.subarray(k * SPLAT_FLOATS, (k + 1) * SPLAT_FLOATS); }
}

// monte_carlo (3DGS-MCMC, Kheradmand et al. 2024, §4.2): `n` dead splats moved onto live rows of `out`,
// drawn with probability ~ their opacity (moments null) or their error mass R (the moments; new rows: 0).
// A row drawn j times becomes N = j + 1 copies at its place whose opacity and scale keep the look:
// o' = 1 - (1 - o)^(1/N), s' = s o / sum_{i=1..N} sum_{k=0..i-1} C(i-1, k) (-1)^k o'^(k+1) / sqrt(k+1).
// The copies differ in their sort order, so their gradients split them. Returns how many moved.
// mask (by input splat, E7's stuck regions): those rows drawn (1 + boost) times as often.
// mode 'shape' (brush): each of n distinct rows drawn splits into two halves within its shape (splitRowByShape);
// a row is drawn again only when there are fewer rows than n.
export function relocate(out, n, moments = null, mask = null, boost = 0, mode = 'copy') {
  const rows = [], cdf = [];
  let acc = 0;
  for (let k = 0; k < out.n; k++) {
    let w = moments ? (out.src[k] >= 0 ? moments[out.src[k] * MOMENT_FLOATS + 2] : 0) : sigmoid(out.buf[k * SPLAT_FLOATS + 3]);
    if (mask && out.src[k] >= 0 && mask[out.src[k]]) w *= 1 + boost;
    if (!(w > 0)) continue;
    acc += w; rows.push(k); cdf.push(acc);
  }
  if (!(acc > 0)) return 0;
  const draw = () => {
    const u = Math.random() * acc;
    let lo = 0, hi = cdf.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    return rows[lo];
  };
  if (mode === 'shape') {
    const picked = new Set();
    for (let tries = 0; picked.size < Math.min(n, rows.length) && tries < 20 * n; tries++) picked.add(draw());
    const order = [...picked];
    for (let j = 0; j < n; j++) splitRowByShape(out, order[j % order.length]);
    return n;
  }
  const picks = new Map();
  for (let j = 0; j < n; j++) { const k = draw(); picks.set(k, (picks.get(k) ?? 0) + 1); }
  for (const [k, j] of picks) {
    const N = j + 1, row = out.row(k), [o, ds] = relocatedOpacityScale(sigmoid(row[3]), N);
    row[3] = logit(o);
    for (let c = 4; c < 7; c++) row[c] += Math.log(ds);
    for (let c = 0; c < j; c++) {
      if (out.n === out.cap) out._grow(2 * out.cap);
      const r = out.n++;
      out.buf.copyWithin(r * SPLAT_FLOATS, k * SPLAT_FLOATS, (k + 1) * SPLAT_FLOATS);
      out.src[r] = -1;
    }
  }
  return n;
}

// relocate's opacity and scale factor for one splat of opacity o split into N
export function relocatedOpacityScale(o, N) {
  const on = 1 - Math.pow(1 - o, 1 / N);
  let sum = 0;
  for (let i = 1; i <= N; i++) {
    let binom = 1;  // C(i-1, k)
    for (let k = 0; k < i; k++) {
      sum += binom * (k % 2 ? -1 : 1) * Math.pow(on, k + 1) / Math.sqrt(k + 1);
      binom = binom * (i - 1 - k) / (k + 1);
    }
  }
  return [Math.max(on, 1e-4), o / sum];
}

// Rule 1 of the floater removal: (x, y, z) -> in front of a training camera, closer than 0.65 z(c).
// depth: 'observed', z = zMin (of the points the camera observed: a camera walking past things,
// e.g. gare-st-lazare, carves the ones other cameras saw near it, half the SfM points there), or
// 'view', z = zView (of all the points in its view, colmap.js computeCameraDepthStats).
// spare: { positions, count, radius }: a splat within `radius` of one of these points (the SfM's) is not in
// any cone. 'observed' counts only the points a camera itself saw: on gare-st-lazare the cameras walking past
// the suitcase statue saw none of its points and carved it, all 5271 points of it nearer than 4 units being
// in some camera's cone.
export function nearCameraCone(images, { depth = 'observed', spare = null } = {}) {
  const trainImgs = images ? images.filter(im => !im.isValidation) : [];
  const uv = [0, 0];
  const near = depth === 'view' ? (im) => im.zView ?? im.zMin : (im) => im.zMin;
  const inCone = (im, x, y, z) => {
    const { R, t, camera } = im, zc = projectPinhole(R, t, camera, x, y, z, uv);
    return zc > 0 && zc < 0.65 * near(im) && uv[0] >= 0 && uv[0] <= camera.width && uv[1] >= 0 && uv[1] <= camera.height;
  };
  // Testing every camera for every splat cost ~1 s per rebalance at 300k splats, 370 cameras.
  // Instead each camera goes into the grid cells its cone (a pyramid of depth D = 0.65 z)
  // can reach, so a splat only tests the few cameras of its cell. A cell is skipped when it lies
  // entirely outside one of the pyramid's planes n.(R X + t) >= 0 (camera frame), a conservative
  // test; the pyramid's bounding box bounds the cells visited.
  const grid = new Map(), cones = [];
  for (const im of trainImgs) {
    const { R, t, camera: c } = im, D = 0.65 * near(im);
    if (!(D > 0)) continue;
    const planes = [[0, 0, 1, 0], [0, 0, -1, D], [c.fx, 0, c.cx, 0], [-c.fx, 0, c.width - c.cx, 0],
                    [0, c.fy, c.cy, 0], [0, -c.fy, c.height - c.cy, 0]].map(([x, y, z, d]) => {
      const w = [0, 1, 2].map(k => R[k] * x + R[3 + k] * y + R[6 + k] * z);  // R^T n: the plane in the world
      return [w[0], w[1], w[2], x * t[0] + y * t[1] + z * t[2] + d, Math.abs(w[0]) + Math.abs(w[1]) + Math.abs(w[2])];
    });
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const [u, v, z] of [[0, 0, 0], [0, 0, 1], [c.width, 0, 1], [0, c.height, 1], [c.width, c.height, 1]]) {
      const pc = [(u - c.cx) / c.fx * D * z, (v - c.cy) / c.fy * D * z, D * z];  // apex, base corners
      for (let k = 0; k < 3; k++) {
        const X = R[k] * (pc[0] - t[0]) + R[3 + k] * (pc[1] - t[1]) + R[6 + k] * (pc[2] - t[2]);
        lo[k] = Math.min(lo[k], X); hi[k] = Math.max(hi[k], X);
      }
    }
    cones.push({ im, planes, lo, hi, size: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) });
  }
  if (!cones.length) return () => false;
  // cells of half the median cone: on jardin5 (425 cameras, 300k splats) 22 candidate cameras
  // per splat and 75 ms in all; smaller cells cost more to fill than they save
  const sizes = cones.map(s => s.size).sort((a, b) => a - b), cell = sizes[sizes.length >> 1] / 2, h = 0.5 * cell;
  const key = (i, j, k) => (i * 73856093) ^ (j * 19349663) ^ (k * 83492791);  // collisions only add candidates
  for (const { im, planes, lo, hi } of cones) {
    const a = lo.map(v => Math.floor(v / cell)), b = hi.map(v => Math.floor(v / cell));
    for (let i = a[0]; i <= b[0]; i++) for (let j = a[1]; j <= b[1]; j++) for (let k = a[2]; k <= b[2]; k++) {
      const q = [(i + 0.5) * cell, (j + 0.5) * cell, (k + 0.5) * cell];
      if (planes.some(([x, y, z, d, l1]) => x * q[0] + y * q[1] + z * q[2] + d + h * l1 < 0)) continue;
      const hk = key(i, j, k);
      if (!grid.has(hk)) grid.set(hk, []);
      grid.get(hk).push(im);
    }
  }
  const spared = spare && spare.radius > 0 ? nearPoints(spare.positions, spare.count, spare.radius) : () => false;
  return (x, y, z) => {
    const list = grid.get(key(Math.floor(x / cell), Math.floor(y / cell), Math.floor(z / cell)));
    return !!list && list.some(im => inCone(im, x, y, z)) && !spared(x, y, z);
  };
}

// (x, y, z) -> whether one of the `count` points (xyz at positions[3 i]) is within r: a hash of cells of side r,
// the 27 around (x, y, z) tested
function nearPoints(positions, count, r) {
  const cells = new Map(), key = (i, j, k) => `${i},${j},${k}`, r2 = r * r;
  for (let p = 0; p < count; p++) {
    const k = key(Math.floor(positions[3 * p] / r), Math.floor(positions[3 * p + 1] / r), Math.floor(positions[3 * p + 2] / r));
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(p);
  }
  return (x, y, z) => {
    const i = Math.floor(x / r), j = Math.floor(y / r), k = Math.floor(z / r);
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) {
      const list = cells.get(key(i + di, j + dj, k + dk));
      if (list?.some(p => (positions[3 * p] - x) ** 2 + (positions[3 * p + 1] - y) ** 2 + (positions[3 * p + 2] - z) ** 2 <= r2)) return true;
    }
    return false;
  };
}

// Floaters: near-camera carve; single-view billboards (N_eff = W^2 / S < 1.45) in well-seen
// cells; faint splats in isolated leaves.
// The rule that removes the splat: 1 in a camera's near cone, 2 seen from too few directions for
// its error, 3 faint in an empty leaf; 0 for none
function isFloater(params, b, alpha, leaf, W, S, R, nearCamera) {
  const nEff = S > 1e-8 ? (W * W) / S : 1.0;
  if (nearCamera(params[b], params[b + 1], params[b + 2])) return 1;
  if (leaf && leaf.numCameras >= 4 && W > 0.5 && nEff < 1.45 && (R / W) > 0.12) return 2;
  if (leaf && leaf.state === 3 && alpha < 0.25) return 3;
  return 0;
}

// The offset o without its component along splat p's thinnest axis (E9's splitInPlane): the error
// offsets are taken in each view's image plane, which a surface seen at a grazing angle is far from
function inSplatPlane(p, o) {
  const Rm = quatToMat3([p[8], p[9], p[10], p[11]]);  // row-major; axis k = column k
  const k = p[4] <= p[5] && p[4] <= p[6] ? 0 : p[5] <= p[6] ? 1 : 2;
  const d = Rm[k] * o[0] + Rm[3 + k] * o[1] + Rm[6 + k] * o[2];
  return [o[0] - d * Rm[k], o[1] - d * Rm[3 + k], o[2] - d * Rm[6 + k]];
}

// Split splat i along its error covariance: a large one into two children, a small one keeps
// itself and clones at the error centroid. Returns 'split' | 'cloned', or null (nothing pushed).
// inPlane: the offsets keep off the splat's thinnest axis (inSplatPlane)
function splitSplat(out, i, moments, maxS, sceneRadius, inPlane = false) {
  const b = i * SPLAT_FLOATS, m = i * MOMENT_FLOATS, params = out.params;
  const x = params[b], y = params[b + 1], z = params[b + 2], R = moments[m + 2];
  // error centroid offset d and covariance E (regularized diagonal)
  const invR = 1 / Math.max(1e-8, R);
  const dx = moments[m + 3] * invR, dy = moments[m + 4] * invR, dz = moments[m + 5] * invR;
  const epsIso = 1e-4 * maxS * maxS;
  const exx = Math.max(epsIso, moments[m + 6] * invR - dx * dx);
  const eyy = Math.max(epsIso, moments[m + 7] * invR - dy * dy);
  const ezz = Math.max(epsIso, moments[m + 8] * invR - dz * dz);
  const exy = moments[m + 9]  * invR - dx * dy;
  const exz = moments[m + 10] * invR - dx * dz;
  const eyz = moments[m + 11] * invR - dy * dz;
  // principal axis v by 4 power iterations
  let vx = exx >= eyy ? 1 : 0, vy = exx < eyy ? 1 : 0, vz = 0.5;
  for (let it = 0; it < 4; it++) {
    const nx = exx * vx + exy * vy + exz * vz;
    const ny = exy * vx + eyy * vy + eyz * vz;
    const nz = exz * vx + eyz * vy + ezz * vz;
    const len = Math.hypot(nx, ny, nz) || 1;
    vx = nx / len; vy = ny / len; vz = nz / len;
  }
  const lambdaMax = Math.max(0,
    vx * (exx * vx + exy * vy + exz * vz) +
    vy * (exy * vx + eyy * vy + eyz * vz) +
    vz * (exz * vx + eyz * vy + ezz * vz));
  const errStd = clamp(Math.sqrt(lambdaMax), maxS * 0.35, maxS * 1.2);
  const shiftNorm = Math.min(maxS, Math.hypot(dx, dy, dz));

  if (maxS > sceneRadius * 0.006) {
    // large: two children +-0.65 errStd along v (kept within 0.65 sigma on each splat
    // axis, so v's component off a flat splat's plane doesn't lift them off it), scales / 1.55
    const p = params.subarray(b, b + SPLAT_FLOATS);
    let o = [vx * errStd * 0.65, vy * errStd * 0.65, vz * errStd * 0.65];
    if (inPlane) {
      const q = inSplatPlane(p, o);
      // v (nearly) along the thinnest axis: the children apart along the longest axis instead
      if (Math.hypot(...q) < 0.25 * errStd * 0.65) {
        const Rm = quatToMat3([p[8], p[9], p[10], p[11]]), k = p[4] >= p[5] && p[4] >= p[6] ? 0 : p[5] >= p[6] ? 1 : 2;
        o = [Rm[k] * errStd * 0.65, Rm[3 + k] * errStd * 0.65, Rm[6 + k] * errStd * 0.65];
      } else o = q;
    }
    const [sx, sy, sz] = clampToSplat(p, o, 0.65);
    const logShrink = Math.log(1 / 1.55);
    for (const sign of [-1, 1]) {
      const c = out.row(out.add(i, -1));
      c[0] = x + sign * sx; c[1] = y + sign * sy; c[2] = z + sign * sz;
      c[4] += logShrink; c[5] += logShrink; c[6] += logShrink;
    }
    out.splitVectors.push(x - sx, y - sy, z - sz, x + sx, y + sy, z + sz);
    return 'split';
  }
  if (shiftNorm > 1e-5) {
    // small: keep the parent, clone at the error centroid (within 1 sigma on each axis)
    out.add(i, i);
    const child = out.row(out.add(i, -1));
    const [ox, oy, oz] = clampToSplat(child, inPlane ? inSplatPlane(child, [dx, dy, dz]) : [dx, dy, dz], 1.0);
    child[0] = x + ox; child[1] = y + oy; child[2] = z + oz;
    return 'cloned';
  }
  return null;
}

// Split splat i as brush (v0.3, train.rs refine_splats): the parent and a fresh child at -o / +o, o drawn from the
// splat's own shape (N(0, 0.5) along each axis x its scale), both scales / sqrt(2), both opacities
// 1 - sqrt(1 - o) (together they cover what the parent did). Returns 'split'.
function splitByShape(out, i) {
  return splitRowByShape(out, out.add(i, i));
}

// row k of `out` split as splitByShape, the second half a new row (src -1); also monte_carlo's relocation with
// relocateMode 'shape' (relocate)
function splitRowByShape(out, k) {
  if (out.n === out.cap) out._grow(2 * out.cap);
  const kc = out.n++;
  out.src[kc] = -1;
  const p = out.row(k), c = out.row(kc);  // rows after the growth
  const Rm = quatToMat3([p[8], p[9], p[10], p[11]]);  // row-major; axis k = column k
  const e = [0, 1, 2].map(a => 0.5 * gaussRand() * Math.exp(p[4 + a]));
  const o = [0, 1, 2].map(r => Rm[3 * r] * e[0] + Rm[3 * r + 1] * e[1] + Rm[3 * r + 2] * e[2]);
  const a = 1 - Math.sqrt(1 - sigmoid(p[3]));
  for (let c = 4; c < 7; c++) p[c] -= 0.5 * Math.LN2;
  p[3] = logit(Math.max(a, 1e-4));
  c.set(p);
  for (let r = 0; r < 3; r++) { c[r] = p[r] + o[r]; p[r] -= o[r]; }
  out.splitVectors?.push(p[0], p[1], p[2], c[0], c[1], c[2]);
  return 'split';
}

function gaussRand() {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

// D. Merge consecutive (same leaf neighborhood) overlapping splats of similar color:
//    opacity-weighted position / color, combined opacity 1 - (1 - a)(1 - b). Returns the count.
function mergeOverlaps(out) {
  const { buf, src, n } = out, F = SPLAT_FLOATS;
  out.alive = new Uint8Array(n).fill(1);
  if (n <= 16) return 0;
  let nMerged = 0;
  for (let i = 0; i + 1 < n; i++) {
    if (src[i] < 0 || src[i + 1] < 0) continue;  // don't undo this pass's new splats
    const a = i * F, b = a + F;
    const dx = buf[a] - buf[b], dy = buf[a + 1] - buf[b + 1], dz = buf[a + 2] - buf[b + 2];
    const sa = Math.exp(Math.max(buf[a + 4], buf[a + 5], buf[a + 6]));
    const sb = Math.exp(Math.max(buf[b + 4], buf[b + 5], buf[b + 6]));
    const colDiff = Math.abs(buf[a + 12] - buf[b + 12]) + Math.abs(buf[a + 13] - buf[b + 13]) + Math.abs(buf[a + 14] - buf[b + 14]);
    if (dx * dx + dy * dy + dz * dz < 0.08 * (sa * sa + sb * sb) && colDiff < 0.25) {
      const wa = sigmoid(buf[a + 3]), wb = sigmoid(buf[b + 3]);
      const wSum = wa + wb + 1e-6;
      for (const k of [0, 1, 2, 12, 13, 14]) buf[a + k] = (wa * buf[a + k] + wb * buf[b + k]) / wSum;
      buf[a + 3] = logit(1 - (1 - wa) * (1 - wb));
      out.alive[i + 1] = 0;
      nMerged++;
      i++;
    }
  }
  return nMerged;
}

// Over the cap (the count above maxSplats: the lease's consolidation, or max_splats lowered), the rows of the lowest score go (alive = 0).
// score[src * stride]: the input splat's; new rows of this pass (src -1: children, copies) stay. Returns how many went.
function trimToCap(out, maxSplats, score, stride) {
  out.alive ??= new Uint8Array(out.n).fill(1);
  const rows = [];
  for (let k = 0; k < out.n; k++) if (out.alive[k] && out.src[k] >= 0) rows.push(k);
  let alive = 0;
  for (let k = 0; k < out.n; k++) alive += out.alive[k];
  const excess = Math.min(rows.length, alive - maxSplats);
  if (!(excess > 0)) return 0;
  const key = Float32Array.from(rows, k => score[out.src[k] * stride]);
  const order = Uint32Array.from(rows.keys()).sort((a, b) => key[a] - key[b]);
  for (let j = 0; j < excess; j++) out.alive[rows[order[j]]] = 0;
  return excess;
}

// E. Compact the kept splats and append stipple seeds within the maxSplats budget.
// Returns { count, params, srcIdx, nSeeds }.
function compactWithSeeds({ buf, src, n, alive }, seedSplats, maxSplats) {
  const F = SPLAT_FLOATS;
  let nKept = 0;
  for (let k = 0; k < n; k++) nKept += alive ? alive[k] : 1;
  const nSeeds = seedSplats ? Math.max(0, Math.min(seedSplats.length / F, maxSplats - nKept)) : 0;
  const count = nKept + nSeeds;
  const params = new Float32Array(count * F), srcIdx = new Int32Array(count).fill(-1);
  for (let k = 0, j = 0; k < n; k++) {
    if (alive && !alive[k]) continue;
    params.set(buf.subarray(k * F, (k + 1) * F), j * F);
    srcIdx[j++] = src[k];
  }
  if (nSeeds > 0) params.set(seedSplats.subarray(0, nSeeds * F), nKept * F);
  return { count, params, srcIdx, nSeeds };
}
