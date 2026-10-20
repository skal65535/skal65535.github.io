// gaussopt-octree.js — 3D Loose Octree spatial partitioning, octree-KNN scale
//                      initialization, front-to-back camera traversal, WLBG
//                      error-moment split/prune, MCMC teleportation, Simulated
//                      Annealing Boltzmann acceptance, octree floater removal,
//                      and JSON / PLY octree serialization.
"use strict";

import { SPLAT_FLOATS, logit, sigmoid, rgbToSh0, quatToMat3 } from './gaussopt-colmap.js';

// Number of floats per splat in the GPU WLBG moments buffer:
// [0]: W_i (visibility mass = sum T_i * alpha_i)
// [1]: S_i (sum of squared per-view visibility weights, for N_eff = W_i^2 / S_i)
// [2]: R_i (error mass = sum w_i * |I_gt - I_hat|)
// [3..5]: R1_x, R1_y, R1_z (1st error moment in world offset delta = X - mu)
// [6..11]: R2_xx, R2_yy, R2_zz, R2_xy, R2_xz, R2_yz (2nd error moment upper triangle)
export const MOMENT_FLOATS = 12;

function percentileBoundsFromStrided(arr, count, stride, offset = 0) {
  if (count === 0) return { center: [0, 0, 0], halfSize: 2.0 };
  const xs = new Float32Array(count);
  const ys = new Float32Array(count);
  const zs = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const b = i * stride + offset;
    xs[i] = arr[b + 0];
    ys[i] = arr[b + 1];
    zs[i] = arr[b + 2];
  }
  xs.sort(); ys.sort(); zs.sort();
  const iLow  = Math.floor(count * 0.02);
  const iHigh = Math.min(count - 1, Math.ceil(count * 0.98));

  const cx = 0.5 * (xs[iLow] + xs[iHigh]);
  const cy = 0.5 * (ys[iLow] + ys[iHigh]);
  const cz = 0.5 * (zs[iLow] + zs[iHigh]);
  const span = Math.max(
    xs[iHigh] - xs[iLow],
    ys[iHigh] - ys[iLow],
    zs[iHigh] - zs[iLow],
    0.5
  );
  // Expand by 1.5x so 98%+ of scene sits comfortably inside the root cube
  return { center: [cx, cy, cz], halfSize: span * 0.75 };
}

function randn() {
  const u1 = Math.max(1e-7, Math.random());
  const u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

class OctreeNode {
  constructor(cx, cy, cz, halfSize, depth, morton = 0) {
    this.cx = cx;
    this.cy = cy;
    this.cz = cz;
    this.halfSize = halfSize;
    this.depth = depth;
    this.morton = morton;
    this.children = null; // Array(8) of OctreeNode or null
    this.indices = [];    // temporary splat indices during build (or kept in index-only mode)
    this.leafId = -1;
    this.start = 0;       // contiguous start index in compacted splat buffer
    this.count = 0;       // number of splats in this leaf
    this.numCameras = 0;  // M_cell: number of training camera frustums intersecting leaf
    this.neighbors = 0;   // number of populated face/corner neighbor leaves
    this.state = 0;       // 0=normal, 1=active_subtask, 2=near_split, 3=isolated_floater
    this.errorSum = 0;    // sum of R_i in this leaf
  }

  get isLeaf() { return this.children === null; }

  octantOf(x, y, z) {
    return (x >= this.cx ? 1 : 0) |
           (y >= this.cy ? 2 : 0) |
           (z >= this.cz ? 4 : 0);
  }
}

export class LooseOctree {
  constructor({ maxLeafSize = 256, minLeafSize = 64, maxDepth = 8, beta = 0.20 } = {}) {
    this.maxLeafSize = maxLeafSize;
    this.minLeafSize = minLeafSize;
    this.maxDepth    = maxDepth;
    this.beta        = beta; // looseness overlap factor: expanded box = (1 + 2*beta) * L
    this.root        = null;
    this.leaves      = [];
    this.leafIndices = null; // Uint32Array of splat indices (used in non-compacting visualizer mode)
    this.sceneCenter = [0, 0, 0];
    this.sceneRadius = 2.0;
    this.lastStats   = { split: 0, cloned: 0, pruned: 0, floaters: 0, merged: 0, teleported: 0 };
  }

  // Build Loose Octree from a flat `params` Float32Array (count * SPLAT_FLOATS)
  // and reorder `params` in-place so each leaf's splats are contiguous.
  build(params, count, images = null) {
    const { center, halfSize } = percentileBoundsFromStrided(params, count, SPLAT_FLOATS, 0);
    this.sceneCenter = center;
    this.sceneRadius = halfSize;
    this.root = new OctreeNode(center[0], center[1], center[2], halfSize, 0, 0);

    for (let i = 0; i < count; i++) {
      this.root.indices.push(i);
    }

    this._splitRecursive(this.root, params, SPLAT_FLOATS);

    this.leaves = [];
    this._collectLeaves(this.root);

    // Compact splats in leaf order so each leaf is a contiguous slice [start, start+count)
    const compacted = new Float32Array(count * SPLAT_FLOATS);
    let cursor = 0;
    for (let lid = 0; lid < this.leaves.length; lid++) {
      const leaf = this.leaves[lid];
      leaf.leafId = lid;
      leaf.start  = cursor;
      leaf.count  = leaf.indices.length;
      if (leaf.count > this.maxLeafSize * 0.8) leaf.state = 2; // near-split indicator
      for (let k = 0; k < leaf.count; k++) {
        const oldIdx = leaf.indices[k];
        const srcOff = oldIdx * SPLAT_FLOATS;
        const dstOff = cursor * SPLAT_FLOATS;
        compacted.set(params.subarray(srcOff, srcOff + SPLAT_FLOATS), dstOff);
        compacted[dstOff + 7] = lid; // store leaf_id in slot 7
        cursor++;
      }
      leaf.indices = null; // free temporary array
    }
    params.set(compacted);

    if (images && images.length > 0) {
      this.updateCameraAndNeighborStats(images);
    }
    return this;
  }

  // Build Loose Octree index WITHOUT reordering the underlying splat buffers
  // (ideal for standalone visualizers like `splats/index.html` where `positions`,
  // `sigmas`, and `colors` are already uploaded to GPU in original order).
  buildIndexOnly(positions, count, stride = 3) {
    const { center, halfSize } = percentileBoundsFromStrided(positions, count, stride, 0);
    this.sceneCenter = center;
    this.sceneRadius = halfSize;
    this.root = new OctreeNode(center[0], center[1], center[2], halfSize, 0, 0);

    for (let i = 0; i < count; i++) {
      this.root.indices.push(i);
    }
    this._splitRecursive(this.root, positions, stride);

    this.leaves = [];
    this._collectLeaves(this.root);
    this.leafIndices = new Uint32Array(count);

    let cursor = 0;
    for (let lid = 0; lid < this.leaves.length; lid++) {
      const leaf = this.leaves[lid];
      leaf.leafId = lid;
      leaf.start  = cursor;
      leaf.count  = leaf.indices.length;
      for (let k = 0; k < leaf.count; k++) {
        this.leafIndices[cursor++] = leaf.indices[k];
      }
      leaf.indices = null;
    }
    return this;
  }

  _splitRecursive(node, arr, stride) {
    if (node.indices.length <= this.maxLeafSize || node.depth >= this.maxDepth) {
      return;
    }
    const h = node.halfSize * 0.5;
    node.children = new Array(8);
    for (let oct = 0; oct < 8; oct++) {
      const ox = (oct & 1) ? h : -h;
      const oy = (oct & 2) ? h : -h;
      const oz = (oct & 4) ? h : -h;
      node.children[oct] = new OctreeNode(
        node.cx + ox, node.cy + oy, node.cz + oz,
        h, node.depth + 1, (node.morton << 3) | oct
      );
    }
    for (let k = 0; k < node.indices.length; k++) {
      const idx = node.indices[k];
      const b = idx * stride;
      const oct = node.octantOf(arr[b], arr[b + 1], arr[b + 2]);
      node.children[oct].indices.push(idx);
    }
    node.indices = [];
    for (let oct = 0; oct < 8; oct++) {
      this._splitRecursive(node.children[oct], arr, stride);
    }
  }

  _collectLeaves(node) {
    if (node.isLeaf) {
      if (node.indices && node.indices.length > 0) this.leaves.push(node);
      else if (node.count > 0) this.leaves.push(node);
      return;
    }
    for (let oct = 0; oct < 8; oct++) {
      if (node.children[oct]) this._collectLeaves(node.children[oct]);
    }
  }

  // ── Serialization: Save & Restore Loose Octree (.ply element or .octree.json) ──

  // Flatten the octree hierarchy into a compact node table for PLY or JSON export.
  // Each node has: { cx, cy, cz, halfSize, isLeaf, firstChildOrStart, countOrDepth, state }
  serializeNodes() {
    const nodes = [];
    if (!this.root) return nodes;

    const visit = (node) => {
      const myIdx = nodes.length;
      const entry = {
        cx: node.cx,
        cy: node.cy,
        cz: node.cz,
        halfSize: node.halfSize,
        isLeaf: node.isLeaf ? 1 : 0,
        firstChildOrStart: node.isLeaf ? node.start : -1,
        countOrDepth: node.isLeaf ? node.count : node.depth,
        state: node.state | 0,
      };
      nodes.push(entry);
      if (!node.isLeaf) {
        const childBase = nodes.length;
        entry.firstChildOrStart = childBase;
        // Reserve 8 contiguous slots for the 8 children
        for (let oct = 0; oct < 8; oct++) {
          nodes.push(null);
        }
        for (let oct = 0; oct < 8; oct++) {
          const c = node.children[oct];
          nodes[childBase + oct] = {
            cx: c.cx,
            cy: c.cy,
            cz: c.cz,
            halfSize: c.halfSize,
            isLeaf: c.isLeaf ? 1 : 0,
            firstChildOrStart: c.isLeaf ? c.start : -1,
            countOrDepth: c.isLeaf ? c.count : c.depth,
            state: c.state | 0,
            _ref: c,
          };
        }
        for (let oct = 0; oct < 8; oct++) {
          const slot = nodes[childBase + oct];
          const c = slot._ref;
          delete slot._ref;
          if (!c.isLeaf) {
            visitChildren(c, slot);
          }
        }
      }
      return myIdx;
    };

    const visitChildren = (node, entry) => {
      const childBase = nodes.length;
      entry.firstChildOrStart = childBase;
      for (let oct = 0; oct < 8; oct++) nodes.push(null);
      for (let oct = 0; oct < 8; oct++) {
        const c = node.children[oct];
        nodes[childBase + oct] = {
          cx: c.cx,
          cy: c.cy,
          cz: c.cz,
          halfSize: c.halfSize,
          isLeaf: c.isLeaf ? 1 : 0,
          firstChildOrStart: c.isLeaf ? c.start : -1,
          countOrDepth: c.isLeaf ? c.count : c.depth,
          state: c.state | 0,
          _ref: c,
        };
      }
      for (let oct = 0; oct < 8; oct++) {
        const slot = nodes[childBase + oct];
        const c = slot._ref;
        delete slot._ref;
        if (!c.isLeaf) visitChildren(c, slot);
      }
    };

    visit(this.root);
    return nodes;
  }

  static fromSerializedNodes(meta, nodes, leafIndices = null) {
    const tree = new LooseOctree({
      maxLeafSize: meta.maxLeafSize || 256,
      maxDepth:    meta.maxDepth    || 8,
      beta:        meta.beta        ?? 0.20,
    });
    tree.sceneCenter = meta.sceneCenter || [0, 0, 0];
    tree.sceneRadius = meta.sceneRadius || 2.0;
    tree.leafIndices = leafIndices ? new Uint32Array(leafIndices) : null;
    if (!nodes || nodes.length === 0) return tree;

    const buildNode = (idx, depth, morton) => {
      const d = nodes[idx];
      const n = new OctreeNode(d.cx, d.cy, d.cz, d.halfSize, depth, morton);
      n.indices = null;
      n.state   = d.state || 0;
      if (d.isLeaf) {
        n.start = d.firstChildOrStart;
        n.count = d.countOrDepth;
      } else {
        n.children = new Array(8);
        const base = d.firstChildOrStart;
        for (let oct = 0; oct < 8; oct++) {
          n.children[oct] = buildNode(base + oct, depth + 1, (morton << 3) | oct);
        }
      }
      return n;
    };

    tree.root = buildNode(0, 0, 0);
    tree.leaves = [];
    tree._collectLeaves(tree.root);
    for (let i = 0; i < tree.leaves.length; i++) {
      tree.leaves[i].leafId = i;
    }
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

  // Initialize splats from COLMAP `points3D.bin` using Octree-accelerated 3-NN scales
  initSplatsFromPoints3D(points3D, images = null) {
    const count = points3D.count;
    const params = new Float32Array(count * SPLAT_FLOATS);
    const { positions, colors } = points3D;

    for (let i = 0; i < count; i++) {
      const b = i * SPLAT_FLOATS;
      params[b + 0]  = positions[3 * i + 0];
      params[b + 1]  = positions[3 * i + 1];
      params[b + 2]  = positions[3 * i + 2];
      params[b + 3]  = logit(0.5); // initial opacity = 0.5
      params[b + 4]  = -3.5;       // placeholder scale, refined by octree 3-NN below
      params[b + 5]  = -3.5;
      params[b + 6]  = -3.5;
      params[b + 7]  = 0;
      params[b + 8]  = 1.0;        // identity quaternion
      params[b + 9]  = 0.0;
      params[b + 10] = 0.0;
      params[b + 11] = 0.0;
      params[b + 12] = rgbToSh0(colors[3 * i + 0] / 255);
      params[b + 13] = rgbToSh0(colors[3 * i + 1] / 255);
      params[b + 14] = rgbToSh0(colors[3 * i + 2] / 255);
      params[b + 15] = 0;
    }

    // Build Loose Octree first so points are grouped spatially in leaves of <= 256
    this.build(params, count, images);

    // Compute 3-NN mean distance inside each leaf (Section 3.4: < 5ms in JS)
    const minScale = this.sceneRadius * 1e-4;
    const maxScale = this.sceneRadius * 0.15;

    for (const leaf of this.leaves) {
      const s0 = leaf.start;
      const n  = leaf.count;
      // If a leaf has < 4 points, expand search window slightly to adjacent indices
      const searchStart = Math.max(0, s0 - 16);
      const searchEnd   = Math.min(count, s0 + n + 16);

      for (let i = s0; i < s0 + n; i++) {
        const bi = i * SPLAT_FLOATS;
        const xi = params[bi], yi = params[bi + 1], zi = params[bi + 2];
        let d1 = Infinity, d2 = Infinity, d3 = Infinity;

        for (let j = searchStart; j < searchEnd; j++) {
          if (i === j) continue;
          const bj = j * SPLAT_FLOATS;
          const dx = xi - params[bj];
          const dy = yi - params[bj + 1];
          const dz = zi - params[bj + 2];
          const d2v = dx * dx + dy * dy + dz * dz;
          if (d2v < d1)      { d3 = d2; d2 = d1; d1 = d2v; }
          else if (d2v < d2) { d3 = d2; d2 = d2v; }
          else if (d2v < d3) { d3 = d2v; }
        }
        const meanDist = ( d3 < Infinity )
          ? (Math.sqrt(d1) + Math.sqrt(d2) + Math.sqrt(d3)) / 3
          : ( d1 < Infinity ? Math.sqrt(d1) : leaf.halfSize * 0.25 );
        const s = Math.min(maxScale, Math.max(minScale, meanDist * 0.8));
        const logS = Math.log(s);
        params[bi + 4] = logS;
        params[bi + 5] = logS;
        params[bi + 6] = logS;
      }
    }

    return { count, params };
  }

  // Compute how many camera frustums see each leaf (M_cell) and how many
  // 3D neighbor leaves each leaf has (for Section 5.2 floater detection).
  updateCameraAndNeighborStats(images) {
    const trainImgs = images.filter(im => !im.isValidation);
    for (const leaf of this.leaves) {
      const looseR = leaf.halfSize * (1 + 2 * this.beta) * 1.732;
      let camHits = 0;
      for (const img of trainImgs) {
        const { R, t, camera } = img;
        const xc = R[0] * leaf.cx + R[1] * leaf.cy + R[2] * leaf.cz + t[0];
        const yc = R[3] * leaf.cx + R[4] * leaf.cy + R[5] * leaf.cz + t[1];
        const zc = R[6] * leaf.cx + R[7] * leaf.cy + R[8] * leaf.cz + t[2];
        if (zc + looseR < 0.05) continue;
        const zClamped = Math.max(0.1, zc);
        const u = camera.fx * (xc / zClamped) + camera.cx;
        const v = camera.fy * (yc / zClamped) + camera.cy;
        const rPx = camera.fx * (looseR / zClamped);
        if (u + rPx >= 0 && u - rPx <= camera.width &&
            v + rPx >= 0 && v - rPx <= camera.height) {
          camHits++;
        }
      }
      leaf.numCameras = camHits;
    }

    // 3D neighbor support check
    for (let i = 0; i < this.leaves.length; i++) {
      const a = this.leaves[i];
      let neigh = 0;
      for (let j = 0; j < this.leaves.length; j++) {
        if (i === j) continue;
        const b = this.leaves[j];
        const reach = (a.halfSize + b.halfSize) * 1.25;
        if (Math.abs(a.cx - b.cx) <= reach &&
            Math.abs(a.cy - b.cy) <= reach &&
            Math.abs(a.cz - b.cz) <= reach) {
          neigh++;
        }
      }
      a.neighbors = neigh;
      if (a.count <= 3 && neigh === 0) a.state = 3; // isolated floater leaf candidate
    }
  }

  // Exact front-to-back octree traversal for camera at `camCenter = [cx, cy, cz]`.
  // Supports both +Z forward (OpenCV/Gauss-Opt) and -Z forward (`splats/index.html` with zSign = -1).
  traverseFrontToBack(camCenter, view = null, fx = 1.0, fy = 1.0, zSign = 1.0) {
    const out = [];
    if (!this.root) return out;
    const betaScale = 1 + 2 * this.beta;

    const visit = (node) => {
      if (view) {
        const xc = view[0] * node.cx + view[4] * node.cy + view[8]  * node.cz + view[12];
        const yc = view[1] * node.cx + view[5] * node.cy + view[9]  * node.cz + view[13];
        const zc = (view[2] * node.cx + view[6] * node.cy + view[10] * node.cz + view[14]) * zSign;
        const r  = node.halfSize * betaScale * 1.73205;
        if (zc + r < 0.05) return; // completely behind camera
        if (zc - r > 0.05) {
          if (Math.abs(xc) * fx > (zc + r) * 1.25 + r * fx) return;
          if (Math.abs(yc) * fy > (zc + r) * 1.25 + r * fy) return;
        }
      }
      if (node.isLeaf) {
        if (node.count > 0) out.push(node);
        return;
      }
      // 3-bit mask of which octant the camera sits in relative to `node`:
      // Visiting children in `rank ^ mask` order guarantees strict front-to-back order!
      const mask = (camCenter[0] >= node.cx ? 1 : 0) |
                   (camCenter[1] >= node.cy ? 2 : 0) |
                   (camCenter[2] >= node.cz ? 4 : 0);
      for (let rank = 0; rank < 8; rank++) {
        visit(node.children[rank ^ mask]);
      }
    };

    visit(this.root);
    return out;
  }

  // Sort splats using front-to-back octree leaf traversal + intra-leaf depth sort
  // (+ optional cross-leaf sliding window for Experiment E1).
  sortOctreeFrontToBack(params, camCenter, view, fx, fy, outIdx, { slidingWindow = 0 } = {}) {
    const visibleLeaves = this.traverseFrontToBack(camCenter, view, fx, fy, 1.0);
    const vx = view[2], vy = view[6], vz = view[10], vw = view[14];

    const leafIds    = new Uint32Array(this.maxLeafSize);
    const leafDepths = new Float32Array(this.maxLeafSize);

    let writePos = 0;
    for (let l = 0; l < visibleLeaves.length; l++) {
      const leaf = visibleLeaves[l];
      const n = leaf.count;
      const start = leaf.start;
      for (let k = 0; k < n; k++) {
        const idx = start + k;
        const b = idx * SPLAT_FLOATS;
        const z = vx * params[b] + vy * params[b + 1] + vz * params[b + 2] + vw;
        leafIds[k]    = idx;
        leafDepths[k] = z;
      }
      // Insertion sort within leaf (identical result to workgroup Bitonic sort on <=256 elements)
      for (let i = 1; i < n; i++) {
        const idA = leafIds[i];
        const zA  = leafDepths[i];
        let j = i;
        while (j > 0 && leafDepths[j - 1] > zA) {
          leafIds[j]    = leafIds[j - 1];
          leafDepths[j] = leafDepths[j - 1];
          j--;
        }
        leafIds[j]    = idA;
        leafDepths[j] = zA;
      }
      for (let k = 0; k < n; k++) {
        outIdx[writePos++] = leafIds[k];
      }
    }

    // Optional cross-leaf boundary sliding window insertion sort (Experiment E1, Q1b)
    if (slidingWindow > 0 && writePos > 1) {
      for (let i = 1; i < writePos; i++) {
        const idA = outIdx[i];
        const bA  = idA * SPLAT_FLOATS;
        const zA  = vx * params[bA] + vy * params[bA + 1] + vz * params[bA + 2] + vw;
        let j = i;
        const stop = Math.max(0, i - slidingWindow);
        while (j > stop) {
          const idB = outIdx[j - 1];
          const bB  = idB * SPLAT_FLOATS;
          const zB  = vx * params[bB] + vy * params[bB + 1] + vz * params[bB + 2] + vw;
          if (zB > zA) {
            outIdx[j] = idB;
            j--;
          } else {
            break;
          }
        }
        outIdx[j] = idA;
      }
    }

    return { numVisible: writePos, numVisibleLeaves: visibleLeaves.length, visibleLeaves };
  }

  // Sort splats for non-compacting visualizers (like `splats/index.html`, where
  // `this.leafIndices` maps leaf slots to original splat indices and `zSign = -1`
  // for `-Z` forward cameras).
  sortIndexFrontToBack(positions, stride, camCenter, view, fx, fy, outIdx, zSign = -1.0) {
    const visibleLeaves = this.traverseFrontToBack(camCenter, view, fx, fy, zSign);
    const vx = view[2] * zSign, vy = view[6] * zSign, vz = view[10] * zSign, vw = view[14] * zSign;

    const leafIds    = new Uint32Array(this.maxLeafSize);
    const leafDepths = new Float32Array(this.maxLeafSize);
    const indexMap   = this.leafIndices;

    let writePos = 0;
    for (let l = 0; l < visibleLeaves.length; l++) {
      const leaf = visibleLeaves[l];
      const n = leaf.count;
      const start = leaf.start;
      for (let k = 0; k < n; k++) {
        const idx = indexMap ? indexMap[start + k] : (start + k);
        const b = idx * stride;
        const z = vx * positions[b] + vy * positions[b + 1] + vz * positions[b + 2] + vw;
        leafIds[k]    = idx;
        leafDepths[k] = z;
      }
      for (let i = 1; i < n; i++) {
        const idA = leafIds[i];
        const zA  = leafDepths[i];
        let j = i;
        while (j > 0 && leafDepths[j - 1] > zA) {
          leafIds[j]    = leafIds[j - 1];
          leafDepths[j] = leafDepths[j - 1];
          j--;
        }
        leafIds[j]    = idA;
        leafDepths[j] = zA;
      }
      for (let k = 0; k < n; k++) {
        outIdx[writePos++] = leafIds[k];
      }
    }
    return writePos;
  }

  // Exact global Radix/Array Sort across all visible splats (Baseline for Experiment E1)
  sortGlobalExact(params, count, view, fx, fy, outIdx) {
    const vx = view[2], vy = view[6], vz = view[10], vw = view[14];
    const xx = view[0], xy = view[4], xz = view[8],  xw = view[12];
    const yx = view[1], yy = view[5], yz = view[9],  yw = view[13];

    const pairs = [];
    for (let i = 0; i < count; i++) {
      const b = i * SPLAT_FLOATS;
      const X = params[b], Y = params[b + 1], Z = params[b + 2];
      const z = vx * X + vy * Y + vz * Z + vw;
      if (z <= 0.05) continue;
      const x = xx * X + xy * Y + xz * Z + xw;
      const y = yx * X + yy * Y + yz * Z + yw;
      if (Math.abs(x) * fx > z * 1.3 || Math.abs(y) * fy > z * 1.3) continue;
      pairs.push({ id: i, z });
    }
    pairs.sort((a, b) => a.z - b.z);
    for (let i = 0; i < pairs.length; i++) outIdx[i] = pairs[i].id;
    return pairs.length;
  }

  // Generate wireframe line segments (vec4f pairs: [p0.xyz, color, p1.xyz, color])
  // for 3D octree visualization (`draw_octree` in Section 6.3).
  buildWireframeVertices({ showLoose = false } = {}) {
    const edges = [
      [-1,-1,-1,  1,-1,-1], [ 1,-1,-1,  1, 1,-1], [ 1, 1,-1, -1, 1,-1], [-1, 1,-1, -1,-1,-1],
      [-1,-1, 1,  1,-1, 1], [ 1,-1, 1,  1, 1, 1], [ 1, 1, 1, -1, 1, 1], [-1, 1, 1, -1,-1, 1],
      [-1,-1,-1, -1,-1, 1], [ 1,-1,-1,  1,-1, 1], [ 1, 1,-1,  1, 1, 1], [-1, 1,-1, -1, 1, 1],
    ];
    const scale = showLoose ? (1 + 2 * this.beta) : 1.0;
    const verts = new Float32Array(this.leaves.length * 12 * 2 * 4);
    const u32   = new Uint32Array(verts.buffer);
    let ptr = 0;

    for (const leaf of this.leaves) {
      const h = leaf.halfSize * scale;
      // Color by leaf state (ABGR u32):
      // 1 (active) = green, 2 (near split) = orange, 3 (isolated floater) = red, 0 = cyan/gray
      let col = 0x88ffcc44;
      if (leaf.state === 1)      col = 0xff44ff44;
      else if (leaf.state === 2) col = 0xff2299ff;
      else if (leaf.state === 3) col = 0xff3333ff;

      for (const e of edges) {
        verts[ptr + 0] = leaf.cx + e[0] * h;
        verts[ptr + 1] = leaf.cy + e[1] * h;
        verts[ptr + 2] = leaf.cz + e[2] * h;
        u32[ptr + 3]   = col;
        verts[ptr + 4] = leaf.cx + e[3] * h;
        verts[ptr + 5] = leaf.cy + e[4] * h;
        verts[ptr + 6] = leaf.cz + e[5] * h;
        u32[ptr + 7]   = col;
        ptr += 8;
      }
    }
    return { buffer: verts, vertexCount: this.leaves.length * 24 };
  }

  // ── Outer-Loop WLBG Split/Prune + MCMC Teleport + SA Acceptance + Reconciliation ──
  reconcileAndGrow(params, count, moments, images, opts = {}) {
    const {
      maxSplats      = 100000,
      pruneOpacity   = 0.008,
      hysteresis     = 0.35,
      enableFloaters = true,
      enableSplit    = true,
      enableMerge    = true,
      optMethod      = 'adam', // 'adam' | 'monte_carlo' | 'sim_anneal'
      saTemp         = 0.15,   // current Simulated Annealing temperature T(t)
    } = opts;

    // 1. Compute average WLBG error mass R_bar across visible splats
    let totalR = 0, activeCount = 0;
    const highErrorCDF = [];
    let cdfSum = 0;

    for (let i = 0; i < count; i++) {
      const m = i * MOMENT_FLOATS;
      const W = moments[m + 0];
      const R = moments[m + 2];
      if (W > 1e-3) {
        totalR += R;
        activeCount++;
        if (R > 0) {
          cdfSum += R;
          highErrorCDF.push({ idx: i, cdf: cdfSum });
        }
      }
    }
    const avgR = activeCount > 0 ? totalR / activeCount : 0;
    const Tu   = (1 + hysteresis) * avgR;
    const Tl   = (1 - hysteresis) * avgR;

    const kept = []; // array of Float32Array(SPLAT_FLOATS)
    const splitVectors = []; // 3D diagnostic lines for `draw_stipple_3d`
    let nSplit = 0, nCloned = 0, nPruned = 0, nFloaters = 0, nMerged = 0, nTeleported = 0;

    // Helper for Monte-Carlo (`monte_carlo` / 3DGS-MCMC) dead-splat teleportation:
    // Samples an active high-error parent splat with probability proportional to R_j
    // and spawns a perturbed clone while halving the parent's effective opacity.
    const tryMcmcTeleport = () => {
      if (optMethod !== 'monte_carlo' || highErrorCDF.length === 0 || cdfSum <= 0) return false;
      const u = Math.random() * cdfSum;
      let lo = 0, hi = highErrorCDF.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (highErrorCDF[mid].cdf < u) lo = mid + 1;
        else hi = mid;
      }
      const pIdx = highErrorCDF[lo].idx;
      const pb = pIdx * SPLAT_FLOATS;
      const alphaP = sigmoid(params[pb + 3]);
      if (alphaP < pruneOpacity * 2) return false;

      // Preserve total transmittance 1 - (1 - a_new)^2 = a_old  =>  a_new = 1 - sqrt(1 - a_old)
      const alphaNew = Math.max(0.05, 1.0 - Math.sqrt(Math.max(0.01, 1.0 - alphaP)));
      params[pb + 3] = logit(alphaNew);

      const child = new Float32Array(params.subarray(pb, pb + SPLAT_FLOATS));
      child[3] = logit(alphaNew);
      const Rmat = quatToMat3([child[8], child[9], child[10], child[11]]);
      const sx = Math.exp(child[4]), sy = Math.exp(child[5]), sz = Math.exp(child[6]);
      const jx = randn() * sx * 0.5, jy = randn() * sy * 0.5, jz = randn() * sz * 0.5;
      child[0] += Rmat[0] * jx + Rmat[3] * jy + Rmat[6] * jz;
      child[1] += Rmat[1] * jx + Rmat[4] * jy + Rmat[7] * jz;
      child[2] += Rmat[2] * jx + Rmat[5] * jy + Rmat[8] * jz;
      kept.push(child);
      nTeleported++;
      return true;
    };

    const trainImgs = images ? images.filter(im => !im.isValidation) : [];

    for (let i = 0; i < count; i++) {
      const b = i * SPLAT_FLOATS;
      const m = i * MOMENT_FLOATS;
      const x = params[b + 0], y = params[b + 1], z = params[b + 2];
      const alpha = sigmoid(params[b + 3]);
      const sx = Math.exp(params[b + 4]);
      const sy = Math.exp(params[b + 5]);
      const sz = Math.exp(params[b + 6]);
      const maxS = Math.max(sx, sy, sz);
      const leafId = Math.min(this.leaves.length - 1, Math.max(0, params[b + 7] | 0));
      const leaf = this.leaves[leafId];

      const W = moments[m + 0];
      const S = moments[m + 1];
      const R = moments[m + 2];

      // ── Step A: WLBG Low-Opacity / Dead Splat Pruning (or MCMC Teleportation) ──
      if (alpha < pruneOpacity || maxS > this.sceneRadius * 0.5) {
        if (!tryMcmcTeleport()) nPruned++;
        continue;
      }

      // ── Step B: Octree-Guided Floater Removal (Section 5.2) ──
      if (enableFloaters) {
        // Rule 1: Near-camera free-space carve (z < 0.65 * z_min(c))
        let inNearCameraCone = false;
        for (const img of trainImgs) {
          const { R: rot, t, camera, zMin } = img;
          const zc = rot[6] * x + rot[7] * y + rot[8] * z + t[2];
          if (zc > 0 && zc < 0.65 * zMin) {
            const xc = rot[0] * x + rot[1] * y + rot[2] * z + t[0];
            const yc = rot[3] * x + rot[4] * y + rot[5] * z + t[1];
            const u = camera.fx * (xc / zc) + camera.cx;
            const v = camera.fy * (yc / zc) + camera.cy;
            if (u >= 0 && u <= camera.width && v >= 0 && v <= camera.height) {
              inNearCameraCone = true;
              break;
            }
          }
        }
        if (inNearCameraCone) {
          if (!tryMcmcTeleport()) nFloaters++;
          continue;
        }

        // Rule 2: Multi-view participation ratio N_eff = W^2 / S < 1.45 in well-seen cells
        const nEff = S > 1e-8 ? (W * W) / S : 1.0;
        if (leaf && leaf.numCameras >= 4 && W > 0.5 && nEff < 1.45 && (R / W) > 0.12) {
          if (!tryMcmcTeleport()) nFloaters++;
          continue;
        }

        // Rule 3: Topological isolation in 3D octree
        if (leaf && leaf.state === 3 && alpha < 0.25) {
          if (!tryMcmcTeleport()) nFloaters++;
          continue;
        }

        // Rule 4: Selective sightline opacity decay for high-error occluders
        if (W > 1.0 && (R / W) > 0.22 && alpha > 0.15) {
          params[b + 3] = logit(alpha * 0.65);
        }
      }

      // ── Step C: WLBG 2nd-Moment Error-Covariance Split / SA Boltzmann Split ──
      let shouldSplit = (R > Tu && Tu > 1e-4);
      if (!shouldSplit && optMethod === 'sim_anneal' && R > Tl && Tu > 1e-4) {
        // Simulated Annealing: accept borderline splits with Boltzmann probability exp(-(Tu - R) / (T * avgR))
        const boltzmann = Math.exp(-(Tu - R) / Math.max(1e-6, saTemp * avgR));
        if (Math.random() < boltzmann * 0.35) shouldSplit = true;
      }

      if (enableSplit && shouldSplit && kept.length + 2 <= maxSplats) {
        const invR = 1 / Math.max(1e-8, R);
        const dx = moments[m + 3] * invR;
        const dy = moments[m + 4] * invR;
        const dz = moments[m + 5] * invR;
        const epsIso = 1e-4 * maxS * maxS;
        const exx = Math.max(epsIso, moments[m + 6] * invR - dx * dx);
        const eyy = Math.max(epsIso, moments[m + 7] * invR - dy * dy);
        const ezz = Math.max(epsIso, moments[m + 8] * invR - dz * dz);
        const exy = moments[m + 9]  * invR - dx * dy;
        const exz = moments[m + 10] * invR - dx * dz;
        const eyz = moments[m + 11] * invR - dy * dz;

        // Power iteration (4 steps) on 3x3 error covariance E_i to get principal axis v_max
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
          vz * (exz * vx + eyz * vy + ezz * vz)
        );
        const errStd = Math.min(maxS * 1.2, Math.max(maxS * 0.35, Math.sqrt(lambdaMax)));
        const shiftNorm = Math.min(maxS, Math.hypot(dx, dy, dz));

        if (maxS > this.sceneRadius * 0.006) {
          // Over-reconstructed: split into 2 children along error covariance axis
          const c1 = new Float32Array(params.subarray(b, b + SPLAT_FLOATS));
          const c2 = new Float32Array(params.subarray(b, b + SPLAT_FLOATS));
          const stepX = vx * errStd * 0.65;
          const stepY = vy * errStd * 0.65;
          const stepZ = vz * errStd * 0.65;
          c1[0] = x - stepX; c1[1] = y - stepY; c1[2] = z - stepZ;
          c2[0] = x + stepX; c2[1] = y + stepY; c2[2] = z + stepZ;
          const logShrink = Math.log(1 / 1.55);
          c1[4] += logShrink; c1[5] += logShrink; c1[6] += logShrink;
          c2[4] += logShrink; c2[5] += logShrink; c2[6] += logShrink;
          kept.push(c1, c2);
          splitVectors.push(x - stepX, y - stepY, z - stepZ, x + stepX, y + stepY, z + stepZ);
          nSplit++;
          continue;
        } else if (shiftNorm > 1e-5) {
          // Under-reconstructed: asymmetric Lloyd clone at error centroid offset
          const parent = new Float32Array(params.subarray(b, b + SPLAT_FLOATS));
          const child  = new Float32Array(params.subarray(b, b + SPLAT_FLOATS));
          const scaleJump = Math.min(maxS, shiftNorm) / shiftNorm;
          child[0] = x + dx * scaleJump;
          child[1] = y + dy * scaleJump;
          child[2] = z + dz * scaleJump;
          kept.push(parent, child);
          nCloned++;
          continue;
        }
      }

      kept.push(new Float32Array(params.subarray(b, b + SPLAT_FLOATS)));
    }

    // ── Step D: Overlap Moment-Matching Merge (Section 5.3) ──
    if (enableMerge && kept.length > 16) {
      for (let i = 0; i < kept.length - 1; i++) {
        const a = kept[i];
        if (!a) continue;
        const b = kept[i + 1];
        if (!b) continue;
        const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        const dist2 = dx * dx + dy * dy + dz * dz;
        const sa = Math.exp(Math.max(a[4], a[5], a[6]));
        const sb = Math.exp(Math.max(b[4], b[5], b[6]));
        const colDiff = Math.abs(a[12] - b[12]) + Math.abs(a[13] - b[13]) + Math.abs(a[14] - b[14]);
        if (dist2 < 0.08 * (sa * sa + sb * sb) && colDiff < 0.25) {
          const wa = sigmoid(a[3]), wb = sigmoid(b[3]);
          const wSum = wa + wb + 1e-6;
          a[0] = (wa * a[0] + wb * b[0]) / wSum;
          a[1] = (wa * a[1] + wb * b[1]) / wSum;
          a[2] = (wa * a[2] + wb * b[2]) / wSum;
          a[3] = logit(1 - (1 - wa) * (1 - wb));
          a[12] = (wa * a[12] + wb * b[12]) / wSum;
          a[13] = (wa * a[13] + wb * b[13]) / wSum;
          a[14] = (wa * a[14] + wb * b[14]) / wSum;
          kept[i + 1] = null;
          nMerged++;
          i++;
        }
      }
    }

    const finalSplats = kept.filter(Boolean);
    const newCount = finalSplats.length;
    const newParams = new Float32Array(newCount * SPLAT_FLOATS);
    for (let i = 0; i < newCount; i++) {
      newParams.set(finalSplats[i], i * SPLAT_FLOATS);
    }

    this.lastStats = {
      split: nSplit,
      cloned: nCloned,
      pruned: nPruned,
      floaters: nFloaters,
      merged: nMerged,
      teleported: nTeleported,
    };
    this.build(newParams, newCount, images);
    return { count: newCount, params: newParams, splitVectors: new Float32Array(splitVectors), stats: this.lastStats };
  }
}
