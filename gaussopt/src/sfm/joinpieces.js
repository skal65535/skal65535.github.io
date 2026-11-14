// joinpieces.js — The pieces of a map that loop.js cannot merge by their 3D points joined by a global positioning of
// the pieces, as GLOMAP's. In a photo set the matches across two pieces can fall on points neither triangulated,
// through frames neither registered.
//   1. The nodes: the segments (rigid, a Sim(3) each) and the unregistered frames that have pairs (a pose each).
//   2. The pairs across two nodes (minMatches F inliers or more, not a homography): each gives by its essential matrix
//      the rotation between them.
//   3. The rotations to start: from the largest segment, the node most pairs agree on next (maxRotDeg), averaged
//      over those.
//   4. The poses by robust nonlinear least squares on directions:
//        - each observation of a track across the nodes (the pairs' inlier matches, joined): the unit vector from its
//          camera's centre to the track's point against the camera's ray;
//        - each pair: the unit vector between the centres against its baseline direction;
//        - a track's point a segment has: the track's point at it, relative to the track's distance from its cameras.
//      Directions, because no piece fits them better shrunk to a point; a linear form, (X - C) x r = 0, collapses the
//      pieces whose cameras see few points of the largest segment.
//      Unknowns: a node's rotation (a left perturbation, folded in after each step), a segment's log scale and shift,
//      a frame's centre, the tracks' points (eliminated, Schur). Started from the nodes placed along the strongest
//      pairs (a typical spacing between centres, a segment scaled to it), solved by Levenberg-Marquardt with a
//      Cauchy loss (cauchyDeg).
//   5. The nodes that join the largest segment: the standard errors of their scale and centroid within maxScaleErr
//      and maxPosErr (of the spacing), and minSupport observations or pairs within maxObsDeg. Their frames are
//      triangulated anew.
//   6. The join undone when it moves the largest segment's own frames: refined (a global BA, intrinsics free) with
//      it and without it, they differ (after a Sim(3)) by more than maxShift of the spacing or maxTurnDeg.
"use strict";

import { dot, cross, sub, norm, normalize, unionFind } from '../math.js';
import { relativePose, ransacHomography, triangulate, depthIn } from './geometry.js';
import { poseCenter } from './camera.js';
import { mul3, transpose3, mulVec3, rotationAngle, projectToSO3, svd, inverse3, choleskySolve, expSO3 } from './linalg.js';
import { median, angleBetween, deg } from './util.js';
import { umeyama, mean } from './align.js';

export const JOIN_DEFAULTS = { minMatches: 50, maxRotDeg: 3, maxObsDeg: 1, cauchyDeg: 1, minSupport: 10, iterations: 60,
                               maxScaleErr: 0.05, maxPosErr: 0.25, maxShift: 0.1, maxTurnDeg: 0.5 };

const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1], AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
// a Jacobian as sparse columns [[unknown, d (3)]]: summed by unknown, scaled, turned by a 3x3
const sumCols = (cols) => { const m = new Map(); for (const [k, c] of cols) m.set(k, m.has(k) ? m.get(k).map((v, r) => v + c[r]) : c); return [...m]; };
const scaleCols = (cols, s) => cols.map(([k, c]) => [k, c.map(v => v * s)]);
const mulCols = (J, cols) => cols.map(([k, c]) => [k, mulVec3(J, c)]);
// a unit-vector residual normalize(u) - w and its Jacobian by u: (I - n n^T) / |u|
const unitRes = (u, w) => {
  const l = norm(u), n = u.map(c => c / l);
  return { e: sub(n, w), J: [0, 1, 2].flatMap(a => [0, 1, 2].map(b => ((a === b ? 1 : 0) - n[a] * n[b]) / l)) };
};

// Returns { joined: nodes joined, frames: frames registered, segments: segments merged } or null when none joins
export async function joinPieces(mapper, { rng, log = () => {}, ...options } = {}) {
  const o = { ...JOIN_DEFAULTS, ...options };
  const g = pieceGraph(mapper, rng, o);
  if (!g) return null;
  const pl = new Placement(mapper, g);
  if (!pl.startRotations(o.maxRotDeg)) return null;
  const kept = g.edges.filter(ed => pl.Q[g.nodeOf[ed.i]] && pl.Q[g.nodeOf[ed.j]] &&
                                    deg(rotationAngle(mul3(ed.R, pl.Rw0(ed.i)), pl.Rw0(ed.j))) <= o.maxRotDeg);
  if (!kept.length || !pl.allocate()) return null;
  const tracks = buildTracks(pl, kept);
  // a pair's baseline direction in node i's map
  const dirs = kept.map(ed => ({ i: ed.i, j: ed.j, d: normalize(mulVec3(transpose3(pl.Rloc(ed.i)), mulVec3(transpose3(ed.R), ed.t)).map(v => -v)) }));
  const b0 = pl.spacing(g.root) ?? 1;
  pl.startPlaces(kept, dirs, b0);
  startPoints(pl, tracks, b0);
  const fit = solve(pl, tracks, dirs, o);
  const { joined, verdict } = decide(pl, tracks, dirs, fit, b0, o);
  log(`join pieces: ${g.nodes.length - 1} nodes besides the largest segment, ${kept.length}/${g.edges.length} pairs, ${tracks.length} tracks; ` +
      `cost ${fit.cost0.toFixed(1)} -> ${fit.cost.toFixed(1)}; ${verdict.join('; ')}`);
  if (!joined.length) return null;
  return apply(pl, joined, kept, b0, o, log);
}

// ── 1, 2. the nodes and the pairs across them ──
function pieceGraph(mapper, rng, o) {
  const nodes = [], nodeOf = new Int32Array(mapper.n).fill(-1), bySeg = new Map();
  for (let k = 0; k < mapper.n; k++) {
    if (!mapper.poses[k]) continue;
    const s = mapper.segment[k];
    if (!bySeg.has(s)) { bySeg.set(s, nodes.length); nodes.push({ seg: s, frames: [] }); }
    nodeOf[k] = bySeg.get(s);
    nodes[nodeOf[k]].frames.push(k);
  }
  if (!nodes.length) return null;
  const root = bySeg.get(mapper.largestSegment());
  const edges = [];
  for (let i = 0; i < mapper.n; i++) {
    for (const e of mapper.adj[i]) {
      const j = e.other;
      if (j <= i || e.a.length < o.minMatches) continue;
      if (!mapper.focalKnown[mapper.camOf[i]] || !mapper.focalKnown[mapper.camOf[j]]) continue;
      if (mapper.poses[i] && mapper.poses[j] && nodeOf[i] === nodeOf[j]) continue;
      const [xi, xj] = mapper.pairPixels(i, e);
      const rel = relativePose(xi, xj, mapper.cam(i), { rng, threshold: 2, minInliers: o.minMatches, camera2: mapper.cam(j) });
      if (!rel) continue;
      const H = ransacHomography(xi, xj, { rng, threshold: 4 });
      if (H && H.numInliers > 0.8 * rel.F.numInliers) continue;  // rotation only / planar: no baseline direction
      edges.push({ i, j, R: rel.R, t: rel.t, a: rel.idx.map(m => e.a[m]), b: rel.idx.map(m => e.b[m]) });
    }
  }
  for (const { i, j } of edges) for (const k of [i, j]) if (nodeOf[k] < 0) { nodeOf[k] = nodes.length; nodes.push({ seg: -1, frames: [k] }); }
  if (nodes.length < 2 || !edges.length) return null;
  return { nodes, nodeOf, root, edges };
}

// The nodes' placement in the root's map. Q per node: its map -> the root's (R_k = Rloc(k) Q^T); the unknowns p, per
// node but the root, [w (3), log s, T (3)] (a segment) or [w (3), C (3)] (a frame) at off[n]
class Placement {
  constructor(mapper, { nodes, nodeOf, root, edges }) {
    Object.assign(this, { mapper, nodes, nodeOf, root, edges });
    this.Q = new Array(nodes.length).fill(null);
    this.Q[root] = I3;
    this.off = new Int32Array(nodes.length).fill(-1);
    this.placed = new Set([root]);
  }
  isSeg(n) { return this.nodes[n].seg >= 0; }
  live(n) { return this.placed.has(n); }
  Rloc(k) { const P = this.mapper.poses[k]; return P ? P.R : I3; }
  cloc(k) { const P = this.mapper.poses[k]; return P ? poseCenter(P) : [0, 0, 0]; }
  Rw0(k) { return mul3(this.Rloc(k), transpose3(this.Q[this.nodeOf[k]])); }

  // ── 3. the rotations to start: a pair x_j = R x_i + t ──
  startRotations(maxRotDeg) {
    const { nodes, nodeOf, edges, Q } = this;
    for (;;) {
      let pick = null;
      for (let n = 0; n < nodes.length; n++) {
        if (Q[n]) continue;
        const cand = [];
        for (const ed of edges) {
          const ni = nodeOf[ed.i], nj = nodeOf[ed.j];
          if (nj === n && Q[ni]) cand.push(mul3(Q[ni], mul3(transpose3(this.Rloc(ed.i)), mul3(transpose3(ed.R), this.Rloc(ed.j)))));
          else if (ni === n && Q[nj]) cand.push(mul3(Q[nj], mul3(transpose3(this.Rloc(ed.j)), mul3(ed.R, this.Rloc(ed.i)))));
        }
        let agree = [];
        for (const c of cand) {
          const these = cand.filter(d => deg(rotationAngle(c, d)) <= maxRotDeg);
          if (these.length > agree.length) agree = these;
        }
        if (agree.length && (!pick || agree.length > pick.agree.length)) pick = { n, agree };
      }
      if (!pick) break;
      Q[pick.n] = projectToSO3(pick.agree.reduce((s, q) => s.map((x, k) => x + q[k]), new Array(9).fill(0)));
    }
    return Q.some((q, n) => q && n !== this.root);
  }

  // the unknowns of the nodes with a rotation; false when none has
  allocate() {
    let np = 0;
    for (let n = 0; n < this.nodes.length; n++) if (n !== this.root && this.Q[n]) { this.off[n] = np; np += this.isSeg(n) ? 7 : 6; }
    this.np = np;
    this.p = new Float64Array(np);
    return np > 0;
  }

  rot(n) {
    const { p, off } = this;
    return n === this.root ? I3 : mul3(expSO3([p[off[n]], p[off[n] + 1], p[off[n] + 2]]), this.Q[n]);
  }
  // a vector y turned by node n's rotation: its derivative by the perturbation w, w x y
  rotCols(n, y) { return n === this.root ? [] : AXES.map((a, r) => [this.off[n] + r, cross(a, y)]); }
  // node n's map point x in the root's, and its derivative by the unknowns
  place(n, x) {
    if (n === this.root) return { v: x, J: [] };
    const { p } = this, k = this.off[n];
    if (!this.isSeg(n)) return { v: [p[k + 3], p[k + 4], p[k + 5]], J: AXES.map((a, r) => [k + 3 + r, a]) };
    const a = mulVec3(this.rot(n), x).map(c => Math.exp(p[k + 3]) * c);
    return { v: [a[0] + p[k + 4], a[1] + p[k + 5], a[2] + p[k + 6]], J: [...this.rotCols(n, a), [k + 3, a], ...AXES.map((e, r) => [k + 4 + r, e])] };
  }
  centre(k) { return this.place(this.nodeOf[k], this.cloc(k)); }
  worldPose(k) { const R = mul3(this.Rloc(k), transpose3(this.rot(this.nodeOf[k]))); return { R, t: mulVec3(R, this.centre(k).v).map(v => -v) }; }
  // the rotations' perturbations folded into Q
  fold() {
    for (let n = 0; n < this.nodes.length; n++) if (this.off[n] >= 0) { this.Q[n] = this.rot(n); this.p.fill(0, this.off[n], this.off[n] + 3); }
  }

  // node n's typical spacing: the median distance of its frames' centres to their nearest; null under 2 frames
  spacing(n) {
    const C = this.nodes[n].frames.map(k => this.cloc(k));
    if (C.length < 2) return null;
    return median(C.map((c, a) => {
      let best = Infinity;
      for (let b = 0; b < C.length; b++) if (b !== a) best = Math.min(best, norm(sub(c, C[b])));
      return best;
    }));
  }

  // the start: the nodes placed from the root along the strongest pairs, b0 apart, a segment scaled to that spacing
  startPlaces(kept, dirs, b0) {
    const { nodeOf, placed, p } = this;
    const strongest = kept.map((ed, q) => ({ ed, dir: dirs[q] })).sort((a, b) => b.ed.a.length - a.ed.a.length);
    for (let grew = true; grew;) {
      grew = false;
      for (const { ed, dir } of strongest) {
        const ni = nodeOf[ed.i], nj = nodeOf[ed.j];
        if (placed.has(ni) === placed.has(nj)) continue;
        const d = mulVec3(this.rot(ni), dir.d), [from, to, n, sgn] = placed.has(ni) ? [ed.i, ed.j, nj, 1] : [ed.j, ed.i, ni, -1];
        const C = this.centre(from).v.map((c, r) => c + sgn * b0 * d[r]), k = this.off[n];
        if (!this.isSeg(n)) p.set(C, k + 3);
        else {
          const s = b0 / (this.spacing(n) ?? b0), a = mulVec3(this.Q[n], this.cloc(to)).map(c => s * c);
          p[k + 3] = Math.log(s);
          p.set(sub(C, a), k + 4);
        }
        placed.add(n);
        grew = true;
        break;
      }
    }
  }
}

// Packed (frame, keypoint) keys
const KP = 1 << 20;

// The tracks of the kept pairs' inlier matches (one keypoint a frame, across two nodes or more), with the segments'
// own points as anchors; each observation's ray in its node's map (rot(n) turns it into the root's)
function buildTracks(pl, kept) {
  const { mapper, nodeOf } = pl, index = new Map(), keys = [];
  const id = (k, kp) => {
    if (kp >= KP) throw new Error(`joinPieces: keypoint ${kp} over ${KP}`);
    const u = k * KP + kp;
    if (!index.has(u)) { index.set(u, keys.length); keys.push(u); }
    return index.get(u);
  };
  const pairs = [];
  for (const ed of kept) for (let m = 0; m < ed.a.length; m++) pairs.push([id(ed.i, ed.a[m]), id(ed.j, ed.b[m])]);
  const { find, union } = unionFind(keys.length);
  for (const [u, v] of pairs) union(u, v);
  const groups = new Map();
  for (let u = 0; u < keys.length; u++) { const r = find(u); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(keys[u]); }
  const tracks = [];
  for (const g of groups.values()) {
    const obs = g.map(u => [Math.floor(u / KP), u % KP]);
    if (new Set(obs.map(([k]) => k)).size !== obs.length) continue;  // two keypoints of one frame: a wrong match in it
    if (new Set(obs.map(([k]) => nodeOf[k])).size < 2) continue;
    const anchors = new Map();
    for (const [k, kp] of obs) {
      const pid = mapper.poses[k] ? mapper.kpPoint[k][kp] : -1;
      if (pid >= 0 && mapper.points[pid] && !anchors.has(nodeOf[k])) anchors.set(nodeOf[k], mapper.points[pid].X);
    }
    tracks.push({ obs: obs.map(([k, kp]) => { const xy = mapper.ray(k, kp); return { k, xy, r: normalize(mulVec3(transpose3(pl.Rloc(k)), [...xy, 1])) }; }),
                  anchors: [...anchors], X: null, D: 1 });
  }
  return tracks;
}

// The tracks' points to start: an anchor's, else triangulated, else a point far along the first ray
function startPoints(pl, tracks, b0) {
  const { nodeOf } = pl;
  for (const tr of tracks) {
    const on = tr.obs.filter(ob => pl.live(nodeOf[ob.k])), a = tr.anchors.find(([n]) => pl.live(n));
    let X = a ? pl.place(a[0], a[1]).v : null;
    if (!X && on.length >= 2) {
      const poses = on.map(ob => pl.worldPose(ob.k));
      X = triangulate(poses, on.map(ob => ob.xy));
      if (X && poses.some(P => !(depthIn(P, X) > 0))) X = null;
    }
    const ob0 = on[0] ?? tr.obs[0];
    tr.X = X ?? pl.centre(ob0.k).v.map((c, r) => c + 5 * b0 * mulVec3(pl.rot(nodeOf[ob0.k]), ob0.r)[r]);
    tr.D = median(on.map(ob => norm(sub(tr.X, pl.centre(ob.k).v)))) || b0;
  }
}

// ── 4. the residuals: each { tr (null: a pair's), e, JX (3x3 flat by the track's point, or null), Jp: columns by the unknowns } ──
function* residuals(pl, tracks, dirs) {
  const { nodeOf } = pl;
  for (const tr of tracks) {
    for (const ob of tr.obs) {
      const n = nodeOf[ob.k];
      if (!pl.live(n)) continue;
      const C = pl.centre(ob.k), w = mulVec3(pl.rot(n), ob.r), { e, J } = unitRes(sub(tr.X, C.v), w);
      yield { tr, e, JX: J, Jp: sumCols([...scaleCols(mulCols(J, C.J), -1), ...scaleCols(pl.rotCols(n, w), -1)]) };
    }
    for (const [n, x] of tr.anchors) {
      if (!pl.live(n)) continue;
      const P = pl.place(n, x), s = 1 / tr.D;
      yield { tr, e: sub(tr.X, P.v).map(v => v * s), JX: [s, 0, 0, 0, s, 0, 0, 0, s], Jp: scaleCols(P.J, -s) };
    }
  }
  for (const dr of dirs) {
    const ni = nodeOf[dr.i];
    if (!pl.live(ni) || !pl.live(nodeOf[dr.j])) continue;
    const Ci = pl.centre(dr.i), Cj = pl.centre(dr.j), d = mulVec3(pl.rot(ni), dr.d), { e, J } = unitRes(sub(Cj.v, Ci.v), d);
    yield { tr: null, e, JX: null, Jp: sumCols([...mulCols(J, Cj.J), ...scaleCols(mulCols(J, Ci.J), -1), ...scaleCols(pl.rotCols(ni, d), -1)]) };
  }
}

// The Cauchy cost and the number of residuals in it
function costOf(pl, tracks, dirs, c2) {
  let cost = 0, count = 0;
  for (const { e } of residuals(pl, tracks, dirs)) { cost += Math.log1p(dot(e, e) / c2); count++; }
  return { cost, count };
}

// The normal equations (Cauchy weights): H, g over the unknowns, and per track its 3x3 block, coupling and gradient
function normals(pl, tracks, dirs, c2) {
  const np = pl.np, H = new Float64Array(np * np), g = new Float64Array(np), blocks = new Map();
  for (const { tr, e, JX, Jp } of residuals(pl, tracks, dirs)) {
    const w = 1 / (1 + dot(e, e) / c2);
    for (const [k, ck] of Jp) { g[k] += w * dot(ck, e); for (const [l, cl] of Jp) H[np * k + l] += w * dot(ck, cl); }
    if (!JX) continue;
    if (!blocks.has(tr)) blocks.set(tr, { tr, Hxx: new Array(9).fill(0), Hxp: new Map(), gx: [0, 0, 0] });
    const b = blocks.get(tr), JXt = transpose3(JX);
    const JtJ = mul3(JXt, JX), Jte = mulVec3(JXt, e);
    for (let u = 0; u < 9; u++) b.Hxx[u] += w * JtJ[u];
    for (let u = 0; u < 3; u++) b.gx[u] += w * Jte[u];
    for (const [k, ck] of Jp) { const h = mulVec3(JXt, ck); b.Hxp.set(k, (b.Hxp.get(k) ?? [0, 0, 0]).map((v, r) => v + w * h[r])); }
  }
  return { H, g, blocks: [...blocks.values()] };
}

// The reduced system H - Hpx Hxx^-1 Hxp (damped by lambda), its right side, and each block's inverse
function reduce(np, { H, g, blocks }, lambda) {
  const S = Float64Array.from(H), r = Float64Array.from(g);
  for (const b of blocks) {
    b.inv = inverse3(b.Hxx.map((v, u) => (u % 4 === 0 ? v * (1 + lambda) + 1e-12 : v)));
    if (!b.inv) continue;
    const ks = [...b.Hxp], G = ks.map(([, h]) => mulVec3(b.inv, h)), gi = mulVec3(b.inv, b.gx);
    ks.forEach(([k, h], a) => { r[k] -= dot(h, gi); ks.forEach(([l], c) => { S[np * k + l] -= dot(h, G[c]); }); });
  }
  for (let k = 0; k < np; k++) S[np * k + k] = S[np * k + k] * (1 + lambda) + 1e-12;
  return { S, r };
}

// Levenberg-Marquardt, the tracks' points eliminated. Returns { c2, cost0, cost, count }
function solve(pl, tracks, dirs, o) {
  const { np, p } = pl, c2 = Math.sin(o.cauchyDeg * Math.PI / 180) ** 2;
  let lambda = 1e-3, cur = costOf(pl, tracks, dirs, c2);
  const cost0 = cur.cost;
  for (let it = 0; it < o.iterations; it++) {
    const ne = normals(pl, tracks, dirs, c2), { S, r } = reduce(np, ne, lambda), x = choleskySolve(S, r, np);
    if (!x) { lambda *= 4; continue; }
    const p0 = Float64Array.from(p), X0 = tracks.map(tr => tr.X);
    for (let k = 0; k < np; k++) p[k] -= x[k];
    for (const b of ne.blocks) {
      if (!b.inv) continue;
      const h = [...b.gx];
      for (const [k, c] of b.Hxp) for (let u = 0; u < 3; u++) h[u] -= c[u] * x[k];
      b.tr.X = sub(b.tr.X, mulVec3(b.inv, h));
    }
    const next = costOf(pl, tracks, dirs, c2);
    if (next.cost < cur.cost) {
      pl.fold();
      const done = cur.cost - next.cost < 1e-6 * cur.cost;
      cur = next; lambda = Math.max(1e-9, lambda / 3);
      if (done) break;
    } else {
      p.set(p0); tracks.forEach((tr, t) => { tr.X = X0[t]; });
      if ((lambda *= 4) > 1e8) break;
    }
  }
  return { c2, cost0, cost: cur.cost, count: cur.count };
}

// The unknowns' covariance: the pseudo-inverse of the reduced normal equations, an undetermined direction (a singular
// value under 1e-12 of the largest) infinite; the products skip zeros so that 0 x Infinity stays out
function covariance(pl, tracks, dirs, c2) {
  const np = pl.np, { S } = reduce(np, normals(pl, tracks, dirs, c2), 0), { S: sv, V } = svd(S, np, np), cov = new Float64Array(np * np);
  for (let j = 0; j < np; j++) {
    const inv = sv[j] > 1e-12 * sv[0] ? 1 / sv[j] : Infinity;
    for (let k = 0; k < np; k++) for (let l = 0; l < np; l++) if (V[np * k + j] && V[np * l + j]) cov[np * k + l] += V[np * k + j] * V[np * l + j] * inv;
  }
  return cov;
}

// ── 5. the nodes that join: standard errors, support within maxObsDeg. Returns { joined, verdict } ──
function decide(pl, tracks, dirs, { c2, cost, count }, b0, o) {
  const { np, p, nodeOf } = pl, cov = covariance(pl, tracks, dirs, c2);
  const sigma2 = c2 * Math.max(1, cost) / Math.max(1, count);  // the Cauchy cost is ~ |e|^2 / c2 a residual
  // the largest of the three axes' variances of a point with Jacobian J: diag(J cov J^T)
  const axisVar = (J, r) => J.reduce((s, [k, d]) => s + J.reduce((t, [l, e]) => t + d[r] * e[r] * cov[np * k + l], 0), 0);
  const placeErr = (J) => Math.sqrt(sigma2 * Math.max(...[0, 1, 2].map(r => axisVar(J, r))));
  const within = (u, w) => deg(angleBetween(u, w)) <= o.maxObsDeg;
  const support = new Array(pl.nodes.length).fill(0);
  for (const tr of tracks) {
    const on = tr.obs.filter(ob => pl.live(nodeOf[ob.k]) && within(sub(tr.X, pl.centre(ob.k).v), mulVec3(pl.rot(nodeOf[ob.k]), ob.r)));
    if (new Set(on.map(ob => nodeOf[ob.k])).size >= 2) for (const ob of on) support[nodeOf[ob.k]]++;
  }
  for (const dr of dirs) {
    const ni = nodeOf[dr.i], nj = nodeOf[dr.j];
    if (pl.live(ni) && pl.live(nj) && within(sub(pl.centre(dr.j).v, pl.centre(dr.i).v), mulVec3(pl.rot(ni), dr.d))) { support[ni]++; support[nj]++; }
  }
  const joined = [], verdict = [];
  for (let n = 0; n < pl.nodes.length; n++) {
    if (n === pl.root || pl.off[n] < 0 || !pl.live(n)) continue;
    const k = pl.off[n], seg = pl.isSeg(n), fr = pl.nodes[n].frames, mid = mean(fr.map(f => pl.cloc(f)));
    const scaleErr = seg ? Math.sqrt(sigma2 * cov[np * (k + 3) + k + 3]) : 0, posErr = placeErr(pl.place(n, mid).J) / b0;
    const rotErr = deg(Math.sqrt(sigma2 * Math.max(...[0, 1, 2].map(r => cov[np * (k + r) + k + r]))));  // for the log only
    const ok = scaleErr <= o.maxScaleErr && posErr <= o.maxPosErr && support[n] >= o.minSupport;
    verdict.push(`${seg ? `segment ${pl.nodes[n].seg}` : `frame ${fr[0]}`}: ${ok ? 'joined' : 'not'} (${seg ? `scale ${Math.exp(p[k + 3]).toFixed(3)} +-${scaleErr.toFixed(3)}, ` : ''}` +
                 `position +-${posErr.toFixed(2)} of the spacing, rotation +-${rotErr.toFixed(2)} deg, ${support[n]} within ${o.maxObsDeg} deg)`);
    if (ok) joined.push(n);
  }
  return { joined, verdict };
}

// ── 5, 6. the joined nodes into the root's segment, undone when they move its own frames ──
async function apply(pl, joined, kept, b0, o, log) {
  const { mapper, nodes, nodeOf, root, p, off } = pl;
  const rootSeg = nodes[root].seg, segments = [], frames = [], own = nodes[root].frames;
  // the root's own frames after a refinement of it (intrinsics free)
  const refined = async () => {
    await mapper.globalBA(rootSeg, own[0], true);
    return own.map(k => ({ R: mapper.poses[k].R, C: poseCenter(mapper.poses[k]) }));
  };
  // without the join first; restore() consumes its snapshot, hence a second one for the undo
  const alone = mapper.snapshot(), before = await refined();
  mapper.restore(alone);
  const snap = mapper.snapshot();
  for (const n of joined) {
    const k = off[n];
    if (pl.isSeg(n)) {
      mapper.transformSegment(nodes[n].seg, { s: Math.exp(p[k + 3]), R: pl.rot(n), t: [p[k + 4], p[k + 5], p[k + 6]] });
      mapper.relabelSegment(nodes[n].seg, rootSeg);
      segments.push(nodes[n].seg);
    } else {
      const f = nodes[n].frames[0];
      mapper.place(f, pl.worldPose(f), rootSeg);
      frames.push(f);
    }
  }
  const touched = new Set();
  for (const ed of kept) if (joined.includes(nodeOf[ed.i]) || joined.includes(nodeOf[ed.j])) { touched.add(ed.i); touched.add(ed.j); }
  const made = mapper.triangulateAndFilter(touched);
  const after = await refined();
  const G = own.length >= 3 ? umeyama(before.map(b => b.C), after.map(a => a.C)) : { s: 1, R: I3, t: [0, 0, 0] }, Gt = transpose3(G.R);
  const shift = Math.max(0, ...own.map((_, q) => norm(sub(after[q].C, mulVec3(G.R, before[q].C).map((v, r) => G.s * v + G.t[r]))))) / (G.s * b0);
  const turn = Math.max(0, ...own.map((_, q) => deg(rotationAngle(after[q].R, mul3(before[q].R, Gt)))));
  const moved = `the largest segment's frames moved by up to ${shift.toFixed(3)} of the spacing, ${turn.toFixed(2)} deg`;
  if (!(shift <= o.maxShift && turn <= o.maxTurnDeg)) {
    mapper.restore(snap);
    log(`join pieces: undone, ${moved}`);
    return null;
  }
  log(`join pieces: ${segments.length} segments and ${frames.length} frames into segment ${rootSeg}, ${made} points made, ${moved}`);
  mapper.trace('join', { segments, frames });
  return { joined: joined.length, segments, frames };
}
