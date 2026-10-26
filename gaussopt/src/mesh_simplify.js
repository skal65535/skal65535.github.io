// mesh_simplify.js — Quadric error edge collapse (Garland & Heckbert 1997) for src/mesh.js's meshes
// (DESIGN.md §9 E5): surface nets puts ~2 triangles in every voxel, flat walls included.
// Passes: every edge's cost (the summed plane quadrics of its two ends, at the best of the two ends
// and the midpoint), then the cheapest collapses below `maxError`, each vertex at most once per pass,
// skipping those that would flip a face, turn one away from the original surface there (its normal
// against the merged area-weighted normal of the original faces, below `minFacing`: on the TSDF's
// bumpy surfaces the per-pass turns otherwise add up to folds, 42% of the area on the corner photos
// mesh, 18% with it, 6% before simplifying) or pinch the surface (the link condition: edges shared by
// more than two faces, ends with other common neighbours, the two opposite vertices joined). Open borders get extra planes along them, so holes keep their
// outline. Colors follow the kept vertex (averaged at a midpoint).
"use strict";

// mesh: { positions: Float32Array 3V, faces: Uint32Array 3F, colors?: Uint8Array 3V }
// target: the face count to stop at; maxError: the largest distance error allowed (world units)
// Returns a new mesh (unused vertices dropped) and { passes, error } (the largest cost accepted, as a distance).
export function simplify(mesh, { target = 0, maxError = Infinity, maxPasses = 40, borderWeight = 10, minFacing = 0.3 } = {}) {
  const nv = mesh.positions.length / 3;
  const P = Float64Array.from(mesh.positions), C = mesh.colors ? Float64Array.from(mesh.colors) : null;
  let F = Uint32Array.from(mesh.faces);
  const Q = new Float64Array(10 * nv);
  const N = new Float64Array(3 * nv);  // per vertex: the area-weighted normal of the original faces around it (merged by collapses)  // per vertex: a2 ab ac ad b2 bc bd c2 cd d2 of sum (n.x + d)^2
  const addPlane = (v, n0, n1, n2, d, w) => {
    const q = 10 * v;
    Q[q] += w * n0 * n0; Q[q + 1] += w * n0 * n1; Q[q + 2] += w * n0 * n2; Q[q + 3] += w * n0 * d;
    Q[q + 4] += w * n1 * n1; Q[q + 5] += w * n1 * n2; Q[q + 6] += w * n1 * d;
    Q[q + 7] += w * n2 * n2; Q[q + 8] += w * n2 * d; Q[q + 9] += w * d * d;
  };
  const faceNormal = (a, b, c, out) => {
    const ux = P[3 * b] - P[3 * a], uy = P[3 * b + 1] - P[3 * a + 1], uz = P[3 * b + 2] - P[3 * a + 2];
    const vx = P[3 * c] - P[3 * a], vy = P[3 * c + 1] - P[3 * a + 1], vz = P[3 * c + 2] - P[3 * a + 2];
    out[0] = uy * vz - uz * vy; out[1] = uz * vx - ux * vz; out[2] = ux * vy - uy * vx;
    return Math.hypot(out[0], out[1], out[2]);
  };
  const n = [0, 0, 0];
  // area-weighted face planes; border edges (in one face only) also get the plane through the edge
  // perpendicular to the face
  const edgeCount = new Map(), ekey = (a, b) => a < b ? a * nv + b : b * nv + a;
  for (let f = 0; f < F.length; f += 3) for (let k = 0; k < 3; k++) {
    const key = ekey(F[f + k], F[f + (k + 1) % 3]);
    edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
  }
  for (let f = 0; f < F.length; f += 3) {
    const a = F[f], b = F[f + 1], c = F[f + 2], len = faceNormal(a, b, c, n);
    if (!(len > 0)) continue;
    const n0 = n[0] / len, n1 = n[1] / len, n2 = n[2] / len, d = -(n0 * P[3 * a] + n1 * P[3 * a + 1] + n2 * P[3 * a + 2]);
    for (const v of [a, b, c]) { addPlane(v, n0, n1, n2, d, len / 2); N[3 * v] += n[0]; N[3 * v + 1] += n[1]; N[3 * v + 2] += n[2]; }
    for (let k = 0; k < 3; k++) {
      const u = F[f + k], v = F[f + (k + 1) % 3];
      if (edgeCount.get(ekey(u, v)) !== 1) continue;
      const ex = P[3 * v] - P[3 * u], ey = P[3 * v + 1] - P[3 * u + 1], ez = P[3 * v + 2] - P[3 * u + 2], el = Math.hypot(ex, ey, ez);
      if (!(el > 0)) continue;
      let m0 = ey * n2 - ez * n1, m1 = ez * n0 - ex * n2, m2 = ex * n1 - ey * n0;  // edge x normal
      const ml = Math.hypot(m0, m1, m2); m0 /= ml; m1 /= ml; m2 /= ml;
      const dm = -(m0 * P[3 * u] + m1 * P[3 * u + 1] + m2 * P[3 * u + 2]);
      addPlane(u, m0, m1, m2, dm, borderWeight * el * el); addPlane(v, m0, m1, m2, dm, borderWeight * el * el);
    }
  }
  const qerr = (q0, q1, x, y, z) => {  // (x y z 1) (Q0 + Q1) (x y z 1)^T
    const s = (i) => Q[q0 + i] + Q[q1 + i];
    return s(0) * x * x + 2 * s(1) * x * y + 2 * s(2) * x * z + 2 * s(3) * x + s(4) * y * y + 2 * s(5) * y * z + 2 * s(6) * y +
           s(7) * z * z + 2 * s(8) * z + s(9);
  };
  const maxCost = maxError * maxError;
  const remap = new Uint32Array(nv);
  let passes = 0, worst = 0;
  while (passes < maxPasses && F.length / 3 > target) {
    passes++;
    // vertex -> faces
    const start = new Uint32Array(nv + 1);
    for (let i = 0; i < F.length; i++) start[F[i] + 1]++;
    for (let v = 0; v < nv; v++) start[v + 1] += start[v];
    const vf = new Uint32Array(F.length), fill = start.slice(0, nv);
    for (let i = 0; i < F.length; i++) vf[fill[F[i]]++] = (i / 3) | 0;
    // candidate edges (each once), with their cost and where the kept vertex goes (0: a, 1: b, 2: midpoint)
    const ea = [], eb = [], cost = [], where = [];
    const seen = new Set();
    for (let f = 0; f < F.length; f += 3) for (let k = 0; k < 3; k++) {
      const a = F[f + k], b = F[f + (k + 1) % 3], key = ekey(a, b);
      if (seen.has(key)) continue;
      seen.add(key);
      // the vertex quadrics are area-weighted: divide by the summed area of the two ends' faces to
      // get a squared distance
      let w = 0;
      for (const v of [a, b]) for (let i = start[v]; i < start[v + 1]; i++) w += faceNormal(F[3 * vf[i]], F[3 * vf[i] + 1], F[3 * vf[i] + 2], n) / 2;
      w = Math.max(w, 1e-30);
      const mx = (P[3 * a] + P[3 * b]) / 2, my = (P[3 * a + 1] + P[3 * b + 1]) / 2, mz = (P[3 * a + 2] + P[3 * b + 2]) / 2;
      const c0 = qerr(10 * a, 10 * b, P[3 * a], P[3 * a + 1], P[3 * a + 2]), c1 = qerr(10 * a, 10 * b, P[3 * b], P[3 * b + 1], P[3 * b + 2]);
      const c2 = qerr(10 * a, 10 * b, mx, my, mz);
      const best = Math.min(c0, c1, c2);
      if (best / w > maxCost) continue;
      ea.push(a); eb.push(b); cost.push(best / w); where.push(best === c0 ? 0 : best === c1 ? 1 : 2);
    }
    if (!ea.length) break;
    const order = Array.from(cost.keys()).sort((i, j) => cost[i] - cost[j]);
    const locked = new Uint8Array(nv);
    for (let v = 0; v < nv; v++) remap[v] = v;
    let budget = F.length / 3 - target, collapsed = 0;
    const nb = new Set(), opp = new Set();
    for (const e of order) {
      if (budget <= 0) break;
      const a = ea[e], b = eb[e];
      if (locked[a] || locked[b]) continue;
      // manifold: the edge in at most 2 faces, and the only common neighbours of a and b the vertices
      // opposite the edge in those faces (link condition)
      let shared = 0;
      nb.clear(); opp.clear();
      for (let i = start[a]; i < start[a + 1]; i++) {
        const f = 3 * vf[i], hasB = F[f] === b || F[f + 1] === b || F[f + 2] === b;
        if (hasB) shared++;
        for (let k = 0; k < 3; k++) {
          const x = F[f + k];
          if (x === a || x === b) continue;
          nb.add(x);
          if (hasB) opp.add(x);
        }
      }
      if (shared > 2) continue;
      let pinch = false;
      for (let i = start[b]; i < start[b + 1] && !pinch; i++) {
        const f = 3 * vf[i];
        for (let k = 0; k < 3; k++) { const x = F[f + k]; if (x !== a && x !== b && nb.has(x) && !opp.has(x)) pinch = true; }
      }
      // ... nor joined by an edge themselves (a tetrahedron would fold into a double sheet)
      if (!pinch && opp.size === 2) {
        const [c, d] = opp;
        for (let i = start[c]; i < start[c + 1] && !pinch; i++) { const f = 3 * vf[i]; if (F[f] === d || F[f + 1] === d || F[f + 2] === d) pinch = true; }
      }
      if (pinch) continue;
      // the new position, and no face around a or b may flip (or become a sliver turned > 60 degrees)
      const w = where[e];
      const nx = w === 0 ? P[3 * a] : w === 1 ? P[3 * b] : (P[3 * a] + P[3 * b]) / 2;
      const ny = w === 0 ? P[3 * a + 1] : w === 1 ? P[3 * b + 1] : (P[3 * a + 1] + P[3 * b + 1]) / 2;
      const nz = w === 0 ? P[3 * a + 2] : w === 1 ? P[3 * b + 2] : (P[3 * a + 2] + P[3 * b + 2]) / 2;
      let flips = false;
      const gx = N[3 * a] + N[3 * b], gy = N[3 * a + 1] + N[3 * b + 1], gz = N[3 * a + 2] + N[3 * b + 2], gl = Math.hypot(gx, gy, gz);
      for (const v of [a, b]) for (let i = start[v]; i < start[v + 1] && !flips; i++) {
        const f = 3 * vf[i], tri = [F[f], F[f + 1], F[f + 2]];
        if (tri.includes(a) && tri.includes(b)) continue;  // removed by the collapse
        const before = faceNormal(tri[0], tri[1], tri[2], n), b0 = n[0], b1 = n[1], b2 = n[2];
        const sx = P[3 * v], sy = P[3 * v + 1], sz = P[3 * v + 2];
        P[3 * v] = nx; P[3 * v + 1] = ny; P[3 * v + 2] = nz;
        const after = faceNormal(tri[0], tri[1], tri[2], n);
        P[3 * v] = sx; P[3 * v + 1] = sy; P[3 * v + 2] = sz;
        if (!(after > 0) || (b0 * n[0] + b1 * n[1] + b2 * n[2]) < 0.5 * before * after) flips = true;
        // ... nor face away from the original surface there (turns that add up over the passes fold noisy surfaces)
        else if (gl > 0 && gx * n[0] + gy * n[1] + gz * n[2] < minFacing * after * gl) flips = true;
      }
      if (flips) continue;
      // collapse b into a
      P[3 * a] = nx; P[3 * a + 1] = ny; P[3 * a + 2] = nz;
      if (C && w !== 0) for (let c = 0; c < 3; c++) C[3 * a + c] = w === 1 ? C[3 * b + c] : (C[3 * a + c] + C[3 * b + c]) / 2;
      for (let i = 0; i < 10; i++) Q[10 * a + i] += Q[10 * b + i];
      for (let k = 0; k < 3; k++) N[3 * a + k] += N[3 * b + k];
      remap[b] = a;
      // the neighbours of both are locked for this pass (their faces change)
      for (const v of [a, b]) for (let i = start[v]; i < start[v + 1]; i++) { const f = 3 * vf[i]; locked[F[f]] = locked[F[f + 1]] = locked[F[f + 2]] = 1; }
      worst = Math.max(worst, cost[e]);
      budget -= shared;
      collapsed++;
    }
    if (!collapsed) break;
    // new faces: remapped, degenerate ones dropped
    const out = new Uint32Array(F.length);
    let o = 0;
    for (let f = 0; f < F.length; f += 3) {
      const a = remap[F[f]], b = remap[F[f + 1]], c = remap[F[f + 2]];
      if (a === b || b === c || a === c) continue;
      out[o++] = a; out[o++] = b; out[o++] = c;
    }
    F = out.slice(0, o);
  }
  // compact: the vertices still used, in first-use order
  const newIdx = new Int32Array(nv).fill(-1);
  let m = 0;
  for (let i = 0; i < F.length; i++) if (newIdx[F[i]] < 0) newIdx[F[i]] = m++;
  const positions = new Float32Array(3 * m), colors = C ? new Uint8Array(3 * m) : null, faces = new Uint32Array(F.length);
  for (let v = 0; v < nv; v++) {
    const j = newIdx[v];
    if (j < 0) continue;
    for (let k = 0; k < 3; k++) {
      positions[3 * j + k] = P[3 * v + k];
      if (colors) colors[3 * j + k] = Math.round(C[3 * v + k]);
    }
  }
  for (let i = 0; i < F.length; i++) faces[i] = newIdx[F[i]];
  return { mesh: { positions, faces, ...(colors && { colors }) }, info: { passes, error: Math.sqrt(worst) } };
}
