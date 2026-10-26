// path.js — The camera path as data (DESIGN-sfm §13.3 P7), DOM-free: an order of the dataset's
// images (indices into the name-sorted images), name (= capture) order by default; and the
// shortest open path through every camera, offered for photo sets, whose capture order goes back
// and forth. A step costs the distance between the centers plus rotWeight x the turn between the
// viewing directions (radians) x the median nearest-neighbour distance, so a short step that
// turns the camera around is not free.
"use strict";

import { clamp, dot, sub } from './math.js';

const EXACT_MAX = 16;  // Held-Karp up to here (n 2^n states); beyond, 2-opt from several starts

const norm = (v) => Math.hypot(v[0], v[1], v[2]);
const angle = (u, v) => Math.acos(clamp(dot(u, v) / (norm(u) * norm(v)), -1, 1));

// views: [{ center: [3], forward: [3] }]. Returns the step cost matrix (Float64Array n x n) and the
// distance scale (median nearest-neighbour distance)
export function stepCosts(views, { rotWeight = 1 } = {}) {
  const n = views.length, D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i * n + j] = i === j ? 0 : norm(sub(views[i].center, views[j].center));
  const nearest = views.map((_, i) => Math.min(...views.map((_, j) => (j === i ? Infinity : D[i * n + j]))));
  const s = [...nearest].sort((a, b) => a - b)[n >> 1] ?? 0;
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) C[i * n + j] = i === j ? 0 : D[i * n + j] + rotWeight * s * angle(views[i].forward, views[j].forward);
  return { C, D, scale: s };
}

// Cost of an order (open path) under the matrix M (n x n)
export function orderCost(order, M, n) {
  let c = 0;
  for (let k = 1; k < order.length; k++) c += M[order[k - 1] * n + order[k]];
  return c;
}

// The open path of least cost through every view (either end free). Exact (Held-Karp) for
// n <= EXACT_MAX, else the best 2-opt local optimum from nearest-neighbour starts at every view.
export function shortestPath(views, opts = {}) {
  const n = views.length;
  if (n <= 2) return [...Array(n).keys()];
  const { C } = stepCosts(views, opts);
  return n <= EXACT_MAX ? heldKarp(C, n) : twoOpt(C, n);
}

function heldKarp(C, n) {
  const full = 1 << n, best = new Float64Array(full * n).fill(Infinity), prev = new Int8Array(full * n).fill(-1);
  for (let i = 0; i < n; i++) best[(1 << i) * n + i] = 0;  // a path may start anywhere
  for (let S = 1; S < full; S++) {
    for (let j = 0; j < n; j++) {
      const v = best[S * n + j];
      if (!(S & (1 << j)) || v === Infinity) continue;
      for (let k = 0; k < n; k++) {
        if (S & (1 << k)) continue;
        const T = S | (1 << k), c = v + C[j * n + k];
        if (c < best[T * n + k]) { best[T * n + k] = c; prev[T * n + k] = j; }
      }
    }
  }
  let end = 0;
  for (let j = 1; j < n; j++) if (best[(full - 1) * n + j] < best[(full - 1) * n + end]) end = j;
  const order = [];
  for (let S = full - 1, j = end; j >= 0;) { order.push(j); const p = prev[S * n + j]; S &= ~(1 << j); j = p; }
  return order.reverse();
}

function twoOpt(C, n) {
  let bestOrder = null, bestCost = Infinity;
  for (let start = 0; start < n; start++) {
    const order = [start], used = new Set(order);  // nearest neighbour from `start`
    while (order.length < n) {
      const last = order.at(-1);
      let next = -1;
      for (let k = 0; k < n; k++) if (!used.has(k) && (next < 0 || C[last * n + k] < C[last * n + next])) next = k;
      order.push(next);
      used.add(next);
    }
    for (let improved = true; improved;) {  // reverse order[i..j] while it lowers the open path's cost
      improved = false;
      for (let i = 0; i < n - 1; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = i > 0 ? order[i - 1] : -1, b = order[i], c = order[j], d = j < n - 1 ? order[j + 1] : -1;
          const before = (a >= 0 ? C[a * n + b] : 0) + (d >= 0 ? C[c * n + d] : 0);
          const after = (a >= 0 ? C[a * n + c] : 0) + (d >= 0 ? C[b * n + d] : 0);
          if (after < before - 1e-12) {
            for (let p = i, q = j; p < q; p++, q--) [order[p], order[q]] = [order[q], order[p]];
            improved = true;
          }
        }
      }
    }
    const cost = orderCost(order, C, n);
    if (cost < bestCost) { bestCost = cost; bestOrder = order; }
  }
  return bestOrder;
}

// A path { order, length (sum of center distances), maxTurnDeg (largest turn of one step) } of views
export function pathStats(order, views) {
  let length = 0, maxTurn = 0;
  for (let k = 1; k < order.length; k++) {
    const a = views[order[k - 1]], b = views[order[k]];
    length += norm(sub(a.center, b.center));
    maxTurn = Math.max(maxTurn, angle(a.forward, b.forward));
  }
  return { order, length, maxTurnDeg: maxTurn * 180 / Math.PI };
}
