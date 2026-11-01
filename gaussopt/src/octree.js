// octree.js — 3D Loose Octree: build / compaction, JSON & PLY-node serialization,
// octree-KNN splat initialization, culling + depth sort (the outer loop that grows the splats: densify.js).
"use strict";

import { SPLAT_FLOATS, rgbToSh0 } from './colmap.js';
import { clamp, logit, quatToMat3, mat3ToQuat, symEigen3, projectPinhole } from './math.js';

// LooseOctree constructor defaults (also for a .ply / JSON octree that doesn't record them)
const OCTREE_DEFAULTS = { maxLeafSize: 256, maxDepth: 10, beta: 0.20 };

// Floats per splat in the GPU WLBG moments buffer:
// [0]: W   visibility mass (sum T * alpha)
// [1]: S   sum of squared per-view visibility (N_eff = W^2 / S)
// [2]: R   error mass (sum w * |I_gt - I_hat|)
// [3..5]:  R1 first error moment of delta = X - mu
// [6..11]: R2 second error moment: xx, yy, zz, xy, xz, yz
export const MOMENT_FLOATS = 12;

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

// Per-splat 3-sigma extents (splatExtent) for the viewer's sort (sortIndexFrontToBack), -1 for a splat
// whose opacity is under the cull's (CULL_MIN_LOGIT)
export function splatExtents(params, count, out = new Float32Array(count)) {
  for (let i = 0; i < count; i++) out[i] = params[i * SPLAT_FLOATS + 3] < CULL_MIN_LOGIT ? -1 : splatExtent(params, i * SPLAT_FLOATS);
  return out;
}

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

// The sorts' grow-only buffers, `bufs` or new ones `grow` x larger than n: the keys (view z), the ids and the radix
// sort's two scratch arrays
function sortBuffers(bufs, n, grow) {
  if (bufs && bufs.keys.length >= n) return bufs;
  const cap = Math.max(1024, Math.ceil(n * grow));
  return { keys: new Float32Array(cap), ids: new Uint32Array(cap), tmpKeys: new Uint32Array(cap), tmpIds: new Uint32Array(cap) };
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
    this._setMaxExtents((i) => splatExtent(params, i * SPLAT_FLOATS));
  }

  // node.maxExtent of every node: the max of extentAt(k) over its leaves' slices k
  _setMaxExtents(extentAt) {
    const visit = (node) => {
      let m = 0;
      if (node.isLeaf) {
        for (let k = node.start, end = node.start + node.count; k < end; k++) m = Math.max(m, extentAt(k));
      } else {
        for (const c of node.children) if (c) m = Math.max(m, visit(c));
      }
      return (node.maxExtent = m);
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
    const { keys, ids, tmpKeys, tmpIds } = this._sortBufs = sortBuffers(this._sortBufs, total, 2);
    let n = 0;
    for (const leaf of visibleLeaves) n = cullSplats(params, leaf.start, leaf.start + leaf.count, view, fx, fy, keys, ids, n);
    radixSortByPositiveKey(new Uint32Array(keys.buffer, 0, n), ids, n, tmpKeys, tmpIds, outIdx);
    return { numVisible: n };
  }

  // node.maxExtent from per-splat extents (splatExtents) in the viewer's index order (leafIndices, or
  // the leaf slices): a tree read from a .ply has none, and without it the leaf cull drops the large
  // splats of an off-screen leaf that reach into the frame
  setExtents(extents) {
    const indexMap = this.leafIndices;
    this._setMaxExtents((k) => extents[indexMap ? indexMap[k] : k]);
  }

  // Viewer sort (index-only octree, the gallery): the splats of the visible leaves sorted exactly by
  // view z (radix, as sortOctreeFrontToBack: E1; ties in leaf order). With `extents` (splatExtents;
  // < 0: never drawn), splats outside the frame are culled as cullSplats does. Returns the count in outIdx.
  sortIndexFrontToBack(positions, stride, camCenter, view, fx, fy, outIdx, zSign = -1.0, extents = null) {
    const visibleLeaves = this.traverseFrontToBack(camCenter, view, fx, fy, zSign);
    const vx = view[2] * zSign, vy = view[6] * zSign, vz = view[10] * zSign, vw = view[14] * zSign;
    const xx = view[0], xy = view[4], xz = view[8], xw = view[12], yx = view[1], yy = view[5], yz = view[9], yw = view[13];
    const cullXY = extents && fx > 0 && fy > 0;
    let total = 0;
    for (const leaf of visibleLeaves) total += leaf.count;
    const { keys, ids, tmpKeys, tmpIds } = this._viewBufs = sortBuffers(this._viewBufs, total, 1.5), indexMap = this.leafIndices;
    let n = 0;
    for (const leaf of visibleLeaves) {
      for (let k = leaf.start, end = leaf.start + leaf.count; k < end; k++) {
        const idx = indexMap ? indexMap[k] : k;
        const b = idx * stride, X = positions[b], Y = positions[b + 1], Z = positions[b + 2];
        const zv = vx * X + vy * Y + vz * Z + vw;
        if (zv <= CULL_ZNEAR) continue;
        if (extents) {
          const e = extents[idx];
          if (e < 0) continue;
          if (cullXY) {
            const zf = zv * CULL_FRUSTUM;
            if (Math.abs(xx * X + xy * Y + xz * Z + xw) * fx > zf + e * fx || Math.abs(yx * X + yy * Y + yz * Z + yw) * fy > zf + e * fy) continue;
          }
        }
        keys[n] = zv;
        ids[n++] = idx;
      }
    }
    radixSortByPositiveKey(new Uint32Array(keys.buffer, 0, n), ids, n, tmpKeys, tmpIds, outIdx);
    return n;
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
}
