// mapper.js — Sequential incremental mapping with window bundle adjustment (DESIGN-sfm §5.3, S2).
// Input is the `input` shape of synth.js / dump.js: { camera, images: [{ name, keypoints }],
// pairs: [{ i, j, type, matches }] } with images in time order; or { cameras: [camera], images:
// [{ name, keypoints, camera (index) }] }, a camera model per lens and zoom (DESIGN-sfm §13 P8). Only pairs of `pairTypes` are
// used ('seq' by default: loop pairs are S3's job). Steps:
//   1. focal self-calibration from fundamental matrices of sequential pairs (focal.js)
//   2. initial pair with enough parallax that a homography does not explain
//   3. repeatedly: register the frame with the most 2D-3D matches (P3P RANSAC + pose refinement),
//      triangulate its new tracks, run a window BA (intrinsics free for the first frames),
//      and drop observations that no longer reproject.
"use strict";

import { makeRng } from '../math.js';
import { normalizedFromImage, project, poseCenter, meanFocal, withFocal } from './camera.js';
import { ransacFundamental, ransacHomography, relativePose, triangulate, triangulationAngle, depthIn, ransacPnP } from './geometry.js';
import { estimateFocal } from './focal.js';
import { bundleAdjust } from './ba.js';
import { median, deg, matchedPixels } from './util.js';
import { mul3, mulVec3, rotationAngle } from './linalg.js';

const DEFAULTS = {
  seed: 1,
  pairTypes: ['seq'],
  window: 10,              // window BA: the last `window` registered frames around the new one are free
  focal: 'auto',           // 'auto' | pixels | 'prior' (keep the input camera)
  hfovDeg: null,           // horizontal field of view instead of `focal` (DESIGN-sfm §5.3: hfov=)
  minFocalSharpness: 1.3,  // below: the focal is unobservable from F, use the fallback field of view
  weakFocalSharpness: 3,   // below: weakly constrained, warn
  fallbackHfovDeg: 76,     // typical drone camera (DJI: f = 0.64 w); override with hfovDeg / focal
  maxFocalGap: 8,          // focal calibration: widest frame gap tried per frame
  maxFocalPairs: 200,      // and number of frames sampled
  refineIntrinsicsUntil: 50,
  refine: ['f', 'k1'],
  minCameraFrames: 5,      // several cameras: while mapping, one's intrinsics are refined once it has this many registered frames (P8.3)
  minCameraFramesSweeps: 2, // and in the final sweeps (refine()) from this many: a lens with a few photos is fitted to the
                            // whole map instead of keeping its EXIF ratio (7% off on the Pixel 10 ultrawide, DESIGN-sfm §13 P8)
  minInliers: 30,          // PnP inliers to register a frame
  relativeRegistration: false,  // when PnP has no frame left: registerRelative (photo sets)
  relMinMatches: 50,       // registerRelative: F inliers with the registered neighbour
  relMinScaleInliers: 8,   // and 2D-3D correspondences agreeing on the baseline length (6: a false one on cuisine)
  relScaleTol: 0.1,        // relative, for that agreement
  relMaxTurnDeg: 10,       // and the most the pose refinement may turn the pair's rotation (cuisine: right ones 0.4-4.6, a wrong one 35.7)
  regrowSegments: false,   // after mapping: offer the frames of smaller segments to the largest (photo sets)
  pnpThreshold: 4,         // px
  maxReproj: 4,            // px, triangulation and filtering
  minAngleDeg: 1.5,        // triangulation angle of new points
  initMinAngleDeg: 3,      // median triangulation angle of the initial pair
  initOrder: 'time',       // initial pair: 'time' (first frames first) | 'strongest' (most matches first)
  baIterations: 10,
  log: () => {},
  progress: () => {},       // (stage, done, total): 'register' while growing, 'refine' per window
};

export class SequentialMapper {
  constructor(input, options = {}) {
    this.o = { ...DEFAULTS, ...options };
    this.rng = makeRng(this.o.seed);
    this.cameras = (input.cameras ?? [input.camera]).map(c => ({ ...c, params: [...c.params] }));
    this.priors = this.cameras.map(c => meanFocal(c));  // the input focals: their ratios place the other cameras (calibrateFocal)
    this.images = input.images;
    this.n = input.images.length;
    this.camOf = Int32Array.from(input.images, im => im.camera ?? 0);
    // adjacency: per image, [{ other, a (my keypoint indices), b (other's) }]
    this.adj = Array.from({ length: this.n }, () => []);
    this.addPairs(input.pairs.filter(p => this.o.pairTypes.includes(p.type)));
    this.kpPoint = input.images.map(im => new Int32Array(im.keypoints.length / 2).fill(-1));
    this.poses = new Array(this.n).fill(null);
    this.points = [];  // { X: [3], obs: Map(image -> keypoint) } | null when deleted
    this.segment = new Int32Array(this.n).fill(-1);  // connected map id per frame (S3 merges them)
    this.nextSegmentId = 0;  // ids ever created; merges leave gaps (the live ones are in this.segment)
    this.stats = { registered: 0, focalInit: NaN, segments: [] };  // registered: frames with a pose now
  }

  // camera 0 (the only one, or the reference of several: the worker puts the largest group first)
  get camera() { return this.cameras[0]; }
  set camera(c) { this.cameras[0] = c; }
  cam(img) { return this.cameras[this.camOf[img]]; }

  pixel(img, kp) { const k = this.images[img].keypoints; return [k[2 * kp], k[2 * kp + 1]]; }
  ray(img, kp) { return normalizedFromImage(this.cam(img), ...this.pixel(img, kp)); }
  pairPixels(img, e) { return matchedPixels(this.images[img].keypoints, this.images[e.other].keypoints, e); }

  // ── 1. focal ──────────────────────────────────────────────────────────────
  calibrateFocal() {
    const { focal, maxFocalGap, maxFocalPairs } = this.o;
    if (focal === 'prior') { this.stats.focalSource = 'prior'; return (this.stats.focalInit = meanFocal(this.camera)); }
    const fromHfov = (deg) => (this.camera.width / 2) / Math.tan(deg * Math.PI / 360);
    let f = this.o.hfovDeg ? fromHfov(this.o.hfovDeg) : focal, source = this.o.hfovDeg ? 'hfov' : 'given';
    if (!this.o.hfovDeg && focal === 'auto') {
      // per frame (sampled over the whole sequence), the widest-gap pair that still has enough
      // matches: parallax makes F informative about the focal
      const Fs = [], stride = Math.max(1, Math.floor(this.n / maxFocalPairs));  // floor (up to 2x maxFocalPairs): ceil loses jardin5's late merge
      for (let i = 0; i < this.n; i += stride) {
        if (this.camOf[i] !== 0) continue;
        const e = this.adj[i].filter(e => e.other > i && e.other - i <= maxFocalGap && e.a.length >= 80 && this.camOf[e.other] === 0)
          .sort((p, q) => q.other - p.other)[0];
        if (!e) continue;
        const [x1, x2] = this.pairPixels(i, e);
        const res = ransacFundamental(x1, x2, { rng: this.rng, threshold: 2 });
        if (res && res.numInliers >= 50) Fs.push(res.model);
      }
      const est = Fs.length >= 5 ? estimateFocal(Fs, this.camera.width, this.camera.height) : null;
      this.stats.focalSharpness = est ? est.sharpness : 0;
      if (est && est.sharpness >= this.o.minFocalSharpness) {
        f = est.focal; source = 'calibrated';
        const weak = est.sharpness < this.o.weakFocalSharpness ? ' (weakly constrained: refined by BA)' : '';
        this.o.log(`focal: ${f.toFixed(1)} px from ${Fs.length} pairs, sharpness ${est.sharpness.toFixed(2)}${weak} (prior ${meanFocal(this.camera).toFixed(1)})`);
      } else {
        f = fromHfov(this.o.fallbackHfovDeg); source = 'fallback';
        this.o.log(`focal: not observable from ${Fs.length} pairs (sharpness ${est ? est.sharpness.toFixed(2) : '-'}, ` +
                   `e.g. translation without rotation): using ${this.o.fallbackHfovDeg} deg HFOV = ${f.toFixed(1)} px; set hfov= or focal= if known`);
      }
    }
    this.stats.focalSource = source;
    // camera 0 gets f; the others keep their input focal's ratio to it (EXIF: f35 x zoom)
    this.cameras = this.cameras.map((c, k) => withFocal(c, k === 0 ? f : f * this.priors[k] / this.priors[0]));
    if (this.cameras.length > 1) this.o.log(`focal of the other cameras, by their prior's ratio to camera 0: ${this.cameras.slice(1).map(c => meanFocal(c).toFixed(1)).join(', ')} px`);
    this.stats.focalInit = f;
    return f;
  }

  // ── 2. initial pair ───────────────────────────────────────────────────────
  // First pair (both unregistered, both in `allowed` when given) with enough parallax, trying
  // looser angles when nothing qualifies. Starts a new segment.
  initialize(allowed = null) {
    for (const minAngle of [this.o.initMinAngleDeg, this.o.initMinAngleDeg / 2, this.o.minAngleDeg]) {
      if (this.initializeWith(minAngle, allowed)) return true;
    }
    return false;
  }

  // Candidate pairs [i, e] (i < e.other) in the order initializeWith tries them: time order (the
  // first frame, then its nearest partner), or the most matches first ('strongest', photo sets)
  initialPairs(ok) {
    const pairs = [];
    for (let i = 0; i < this.n; i++) {
      if (!ok(i)) continue;
      for (const e of [...this.adj[i]].filter(e => e.other > i && ok(e.other)).sort((p, q) => p.other - q.other)) pairs.push([i, e]);
    }
    return this.o.initOrder === 'strongest' ? pairs.sort((p, q) => q[1].a.length - p[1].a.length) : pairs;
  }

  initializeWith(minAngleDeg, allowed = null) {
    const ok = (k) => !this.poses[k] && (!allowed || allowed.has(k));
    for (const [i, e] of this.initialPairs(ok)) {
      if (e.a.length < 100) continue;
      const [x1, x2] = this.pairPixels(i, e);
      const rel = relativePose(x1, x2, this.cam(i), { rng: this.rng, threshold: 2, minInliers: 100, camera2: this.cam(e.other) });
      if (!rel) continue;
      const H = ransacHomography(x1, x2, { rng: this.rng, threshold: 4 });
      if (H && H.numInliers > 0.8 * rel.F.numInliers) continue;  // rotation only / planar: no reliable depth
      const P1 = { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] }, P2 = { R: rel.R, t: rel.t };
      const C2 = poseCenter(P2);
      const cand = [];
      rel.idx.forEach((m, k) => {
        if (!rel.front[k]) return;
        const X = triangulate([P1, P2], [rel.rays1[k], rel.rays2[k]]);
        if (X) cand.push({ m, X, angle: triangulationAngle([0, 0, 0], C2, X) });
      });
      const med = deg(median(cand.map(c => c.angle)));
      if (cand.length < 100 || med < minAngleDeg) continue;
      // the points within maxReproj, as addPoint checks them: a pair left with few would start a
      // segment that cannot grow, holding frames a better pair needs (gare-st-lazare: 0 and 36
      // points, then merged by a weak Sim(3) and kinked the path)
      this.poses[i] = P1; this.poses[e.other] = P2;
      const kept = cand.filter(({ m, X }) => this.reprojErr(i, e.a[m], X) <= this.o.maxReproj && this.reprojErr(e.other, e.b[m], X) <= this.o.maxReproj);
      if (kept.length < 100) { this.poses[i] = this.poses[e.other] = null; continue; }
      const seg = this.nextSegmentId++;
      this.segment[i] = this.segment[e.other] = seg;
      let created = 0;
      for (const { m, X } of kept) created += this.addPoint(X, [[i, e.a[m]], [e.other, e.b[m]]]) >= 0;
      this.o.log(`segment ${seg}: init frames ${i}-${e.other}, ${rel.F.numInliers} F inliers, ${created} points, median angle ${med.toFixed(1)} deg`);
      this.localBA(e.other, { refineIntrinsics: false, iterations: 20 });
      return true;
    }
    return false;
  }

  addPoint(X, obs, checkReproj = false) {
    if (checkReproj && obs.some(([img, kp]) => this.reprojErr(img, kp, X) > this.o.maxReproj)) return -1;
    const id = this.points.length;
    this.points.push({ X, obs: new Map(obs) });
    for (const [img, kp] of obs) this.kpPoint[img][kp] = id;
    return id;
  }

  // 2D-3D correspondences of an unregistered frame through its registered neighbours
  correspondences(img, seg = this.nextSegmentId - 1) {
    const byKp = new Map();
    for (const e of this.adj[img]) {
      if (!this.poses[e.other] || this.segment[e.other] !== seg) continue;
      const kp = this.kpPoint[e.other];
      for (let m = 0; m < e.a.length; m++) {
        const pid = kp[e.b[m]];
        if (pid >= 0 && !byKp.has(e.a[m])) byKp.set(e.a[m], pid);
      }
    }
    return byKp;
  }

  // ── 3. registration ───────────────────────────────────────────────────────
  register(img, seg = this.nextSegmentId - 1) {
    const corr = [...this.correspondences(img, seg)];
    if (corr.length < this.o.minInliers) return false;
    const pixels = corr.map(([kp]) => this.pixel(img, kp));
    const rays = corr.map(([kp]) => this.ray(img, kp));
    const pts = corr.map(([, pid]) => this.points[pid].X);
    const res = ransacPnP(rays, pixels, pts, this.cam(img), { rng: this.rng, threshold: this.o.pnpThreshold });
    if (!res || res.numInliers < this.o.minInliers) return false;
    const inl = corr.filter((_, k) => res.inliers[k]);
    this.poses[img] = this.refinePose(img, res.model, inl);
    this.segment[img] = seg;
    for (const [kp, pid] of inl) {
      const P = this.points[pid];
      if (P && !P.obs.has(img) && this.reprojErr(img, kp, P.X) <= this.o.maxReproj) {
        P.obs.set(img, kp);
        this.kpPoint[img][kp] = pid;
      }
    }
    return true;
  }

  // Registration from one registered neighbour: rotation and baseline direction from the pair's
  // essential matrix, the baseline length (one unknown) from the frame's few 2D-3D
  // correspondences by 1-point RANSAC. With nbr's pose (Rn, tn) and the pair's x_img = Rr x_nbr +
  // s tr, the pose is R = Rr Rn, t = Rr tn + s tr, linear in s. Tries the unregistered frames by
  // their largest match count with a registered frame of `seg`; returns the frame registered, or -1.
  registerRelative(seg, candidates = null, failed = new Map()) {
    const { relMinMatches, relMinScaleInliers, pnpThreshold } = this.o;
    const tries = [];
    for (let k = 0; k < this.n; k++) {
      if (this.poses[k] || (failed.get(k) ?? 0) > 1 || (candidates && !candidates.has(k))) continue;
      for (const e of this.adj[k]) {
        if (this.poses[e.other] && this.segment[e.other] === seg && e.a.length >= relMinMatches) tries.push([k, e]);
      }
    }
    tries.sort((p, q) => q[1].a.length - p[1].a.length);
    for (const [img, e] of tries) {
      const corr = [...this.correspondences(img, seg)];
      if (corr.length < relMinScaleInliers) continue;
      const [xImg, xNbr] = this.pairPixels(img, e);
      const rel = relativePose(xNbr, xImg, this.cam(e.other), { rng: this.rng, threshold: 2, minInliers: relMinMatches, camera2: this.cam(img) });
      if (!rel) continue;
      const H = ransacHomography(xNbr, xImg, { rng: this.rng, threshold: 4 });
      if (H && H.numInliers > 0.8 * rel.F.numInliers) continue;  // rotation only / planar: no baseline direction
      const Pn = this.poses[e.other], Rr = rel.R, tr = rel.t;
      const R = mul3(Rr, Pn.R), t0 = mulVec3(Rr, Pn.t);
      const pose = (s) => ({ R, t: [t0[0] + s * tr[0], t0[1] + s * tr[1], t0[2] + s * tr[2]] });
      // per correspondence, the s that best fits its normalized ray (least squares on the two
      // cross-multiplied projection equations)
      const scales = corr.map(([kp, pid]) => {
        const [u, v] = this.ray(img, kp), X = this.points[pid].X;
        const q = mulVec3(R, X).map((x, i) => x + t0[i]);
        const a = [q[0] - u * q[2], q[1] - v * q[2]], b = [tr[0] - u * tr[2], tr[1] - v * tr[2]];
        const bb = b[0] * b[0] + b[1] * b[1];
        return bb > 1e-12 ? -(a[0] * b[0] + a[1] * b[1]) / bb : NaN;
      });
      // 1-point RANSAC on s: the correspondences within relScaleTol of a candidate agree on it
      let bestInl = [];
      for (const s of scales) {
        if (!(s > 0)) continue;
        const inl = corr.filter((_, k) => Math.abs(scales[k] / s - 1) <= this.o.relScaleTol);
        if (inl.length > bestInl.length) bestInl = inl;
      }
      if (bestInl.length < relMinScaleInliers) { failed.set(img, (failed.get(img) ?? 0) + 1); continue; }
      // the pair's rotation and direction are a few pixels off (a 2-view estimate): refine the
      // pose on the agreeing correspondences, as register() does after PnP, then check them
      const s = median(bestInl.map(c => scales[corr.indexOf(c)]));
      const P = this.refinePose(img, pose(s), bestInl), good = bestInl.filter(([kp, pid]) => this.reprojWith(P, img, kp, this.points[pid].X) <= pnpThreshold);
      // the refinement may only correct the pair's rotation, not replace it: a few correspondences
      // can turn the camera far away (cuisine: 8 of them, a frame 40 deg off)
      const turn = deg(rotationAngle(P.R, R));
      if (good.length < relMinScaleInliers || turn > this.o.relMaxTurnDeg) {
        this.o.log(`frame ${img}: not registered from frame ${e.other}: ${good.length}/${corr.length} 2D-3D after refinement, which turned it by ${turn.toFixed(1)} deg`);
        failed.set(img, (failed.get(img) ?? 0) + 1);
        continue;
      }
      this.poses[img] = P;
      this.segment[img] = seg;
      for (const [kp, pid] of good) {
        const X = this.points[pid];
        if (X && !X.obs.has(img)) { X.obs.set(img, kp); this.kpPoint[img][kp] = pid; }
      }
      this.o.log(`frame ${img}: registered from frame ${e.other} (${rel.F.numInliers} F inliers), scale from ${good.length}/${corr.length} 2D-3D, refinement turned it by ${turn.toFixed(2)} deg`);
      return img;
    }
    return -1;
  }

  // Pose-only refinement of `pose` on the 2D-3D correspondences corr: [[kp, pid]] (points fixed)
  refinePose(img, pose, corr) {
    const prob = {
      camera: this.cam(img), poses: [pose], free: Uint8Array.of(1),
      points: Float64Array.from(corr.flatMap(([, pid]) => this.points[pid].X)),
      obsCam: new Int32Array(corr.length), obsPoint: Int32Array.from(corr.keys()),
      obsUV: Float64Array.from(corr.flatMap(([kp]) => this.pixel(img, kp))),
    };
    bundleAdjust(prob, { fixPoints: true, maxIterations: 10 });
    return prob.poses[0];
  }

  reprojWith(pose, img, kp, X) {
    const [u, v, z] = project(this.cam(img), pose, X), [pu, pv] = this.pixel(img, kp);
    return z > 0 ? Math.hypot(u - pu, v - pv) : Infinity;
  }

  reprojErr(img, kp, X) { return this.reprojWith(this.poses[img], img, kp, X); }

  // New tracks of `img` with its registered neighbours, and extension of existing ones
  triangulateNew(img) {
    const C = poseCenter(this.poses[img]);
    const minAngle = this.o.minAngleDeg * Math.PI / 180;
    const nbrs = this.adj[img].filter(e => this.poses[e.other] && this.segment[e.other] === this.segment[img])
      .map(e => ({ e, base: Math.hypot(...poseCenter(this.poses[e.other]).map((x, k) => x - C[k])) }))
      .sort((p, q) => q.base - p.base);
    let created = 0;
    for (const { e } of nbrs) {
      const Co = poseCenter(this.poses[e.other]);
      for (let m = 0; m < e.a.length; m++) {
        const a = e.a[m], b = e.b[m];
        const pa = this.kpPoint[img][a], pb = this.kpPoint[e.other][b];
        if (pa >= 0 && pb < 0) {  // extend the track to the neighbour
          const P = this.points[pa];
          if (!P.obs.has(e.other) && this.reprojErr(e.other, b, P.X) <= this.o.maxReproj) { P.obs.set(e.other, b); this.kpPoint[e.other][b] = pa; }
        } else if (pa < 0 && pb >= 0) {
          const P = this.points[pb];
          if (!P.obs.has(img) && this.reprojErr(img, a, P.X) <= this.o.maxReproj) { P.obs.set(img, a); this.kpPoint[img][a] = pb; }
        } else if (pa < 0 && pb < 0) {
          const X = triangulate([this.poses[img], this.poses[e.other]], [this.ray(img, a), this.ray(e.other, b)]);
          if (!X || depthIn(this.poses[img], X) <= 0 || depthIn(this.poses[e.other], X) <= 0) continue;
          if (triangulationAngle(C, Co, X) < minAngle) continue;
          if (this.addPoint(X, [[img, a], [e.other, b]], true) >= 0) created++;
        }
      }
    }
    return created;
  }

  // ── window BA ─────────────────────────────────────────────────────────────
  // Free: the `window` registered frames nearest in time to `center`; fixed: other observers.
  localBA(center, { refineIntrinsics = false, iterations = this.o.baIterations, frames = null, minCameraFrames = this.o.minCameraFrames } = {}) {
    const reg = [];
    for (let k = 0; k < this.n; k++) if (this.poses[k] && this.segment[k] === this.segment[center]) reg.push(k);
    const freeSet = new Set(frames ?? reg.sort((a, b) => Math.abs(a - center) - Math.abs(b - center)).slice(0, this.o.window));
    const pids = new Set();
    for (const img of freeSet) for (const pid of this.kpPoint[img]) if (pid >= 0) pids.add(pid);
    const camIndex = new Map(), camList = [];
    const ptList = [...pids].filter(p => this.points[p] && this.points[p].obs.size >= 2);
    const obsCam = [], obsPoint = [], obsUV = [];
    ptList.forEach((pid, li) => {
      for (const [img, kp] of this.points[pid].obs) {
        if (!camIndex.has(img)) { camIndex.set(img, camList.length); camList.push(img); }
        obsCam.push(camIndex.get(img)); obsPoint.push(li); obsUV.push(...this.pixel(img, kp));
      }
    });
    if (!camList.length) return null;
    // gauge: with no fixed observer (first window), the earliest free frame is held fixed
    const free = Uint8Array.from(camList, img => freeSet.has(img) ? 1 : 0);
    if (free.every(Boolean)) free[camList.indexOf(Math.min(...camList))] = 0;
    const prob = {
      cameras: this.cameras, poseCam: Int32Array.from(camList, img => this.camOf[img]), poses: camList.map(img => this.poses[img]), free,
      points: Float64Array.from(ptList.flatMap(pid => this.points[pid].X)),
      obsCam: Int32Array.from(obsCam), obsPoint: Int32Array.from(obsPoint), obsUV: Float64Array.from(obsUV),
    };
    const out = bundleAdjust(prob, { refine: refineIntrinsics ? this.o.refine : [], refineCameras: this.refinable(minCameraFrames), maxIterations: iterations });
    this.cameras = prob.cameras;  // a camera for all its frames, whatever this window made of it
    camList.forEach((img, c) => { if (free[c]) this.poses[img] = prob.poses[c]; });
    ptList.forEach((pid, li) => { this.points[pid].X = Array.from(prob.points.subarray(3 * li, 3 * li + 3)); });
    this.filter(ptList);
    return out;
  }

  // Per camera, whether its intrinsics may be refined: one camera always; several, once a camera
  // has `min` registered frames (fewer keep their prior, P8.3)
  refinable(min = this.o.minCameraFrames) {
    if (this.cameras.length === 1) return null;
    const n = new Array(this.cameras.length).fill(0);
    for (let k = 0; k < this.n; k++) if (this.poses[k]) n[this.camOf[k]]++;
    return n.map(c => c >= min);
  }

  // Drop observations that no longer reproject, and points left with < 2 observations
  filter(pids) {
    for (const pid of pids) {
      const P = this.points[pid];
      if (!P) continue;
      for (const [img, kp] of [...P.obs]) {
        if (this.reprojErr(img, kp, P.X) > this.o.maxReproj) { P.obs.delete(img); this.kpPoint[img][kp] = -1; }
      }
      if (P.obs.size < 2) {
        for (const [img, kp] of P.obs) this.kpPoint[img][kp] = -1;
        this.points[pid] = null;
      }
    }
  }

  run() {
    this.calibrateFocal();
    while (this.initialize()) {
      const seg = this.nextSegmentId - 1;
      this.stats.registered += 2;
      const size = 2 + this.grow(seg);
      const frames = [];
      for (let k = 0; k < this.n; k++) if (this.segment[k] === seg) frames.push(k);
      this.stats.segments.push({ id: seg, size, first: frames[0], last: frames.at(-1) });
      this.o.log(`segment ${seg}: ${size} frames (${frames[0]}..${frames.at(-1)})`);
    }
    if (this.o.regrowSegments) this.regrowSegments();
    return this.model();
  }

  // Photo sets: which segment grows first decides what can join it (cuisine: from the pair 3-9,
  // frames 3, 9, 17, 10 make a segment the rest cannot join; from 2-5, those four join through
  // frame 10). So each smaller segment, the larger first, is taken apart and its frames (and the
  // unregistered ones) offered to the largest segment; kept when that loses no frame, undone otherwise.
  regrowSegments() {
    for (let merged = true; merged;) {
      merged = false;
      const sizes = new Map();
      for (const s of this.segment) if (s >= 0) sizes.set(s, (sizes.get(s) ?? 0) + 1);
      if (sizes.size < 2) break;  // none, or one: nothing to regrow
      const big = this.largestSegment(), rest = [...sizes].filter(([s]) => s !== big).sort((a, b) => b[1] - a[1] || a[0] - b[0]);
      for (const [seg, size] of rest) {
        const frames = [...this.segment.keys()].filter(k => this.segment[k] === seg);
        const snap = this.snapshot();
        this.unregister(frames);
        const free = new Set([...this.segment.keys()].filter(k => !this.poses[k]));
        const added = this.grow(big, free);
        if (added >= size) {
          this.o.log(`segment ${seg} (${size} frames) regrown into segment ${big}: ${added} frames added`);
          merged = true;
          break;
        }
        this.restore(snap);
      }
    }
    // and the largest segment, now complete, tries the frames still unregistered once more: a frame
    // registered by a regrow that was undone is not lost (outside-mixed: frame 22, 19 2D-3D)
    const big = this.largestSegment(), free = new Set([...this.segment.keys()].filter(k => !this.poses[k]));
    if (big >= 0 && free.size) {
      const added = this.grow(big, free);
      if (added) this.o.log(`segment ${big}: ${added} unregistered frames added after the regrow`);
    }
  }

  // Register frames into segment `seg`, the one with the most 2D-3D matches first, each followed
  // by triangulation and a window BA (intrinsics free while few frames are registered in all).
  // candidates: Set of frame indices (default: every frame). Returns the number registered.
  grow(seg, candidates = null) {
    const failed = new Map(), relFailed = new Map();  // per frame: PnP failures, registerRelative failures
    let added = 0;
    for (;;) {
      let best = -1, bestCount = 0;
      for (let k = 0; k < this.n; k++) {
        if (this.poses[k] || (failed.get(k) ?? 0) > 1 || (candidates && !candidates.has(k))) continue;
        const c = this.correspondences(k, seg).size;  // a full rescan costs ~0.2 s of jardin5's 125 s
        if (c > bestCount) { best = k; bestCount = c; }
      }
      if (best < 0 || bestCount < this.o.minInliers) {
        // no frame for PnP: one linked to the map by its matches with a registered frame, with
        // too few 2D-3D correspondences of its own (photo sets: links of one pair)
        best = this.o.relativeRegistration ? this.registerRelative(seg, candidates, relFailed) : -1;
        if (best < 0) break;
      } else if (!this.register(best, seg)) { failed.set(best, (failed.get(best) ?? 0) + 1); continue; }
      const registered = ++this.stats.registered;
      added++;
      this.o.progress('register', registered, this.n);
      this.triangulateNew(best);
      this.localBA(best, { refineIntrinsics: registered >= 6 && registered <= this.o.refineIntrinsicsUntil });
      if (registered % 50 === 0) this.o.log(`registered ${registered}/${this.n}, ${this.numPoints()} points, f ${this.camera.params[0].toFixed(1)}`);
    }
    return added;
  }

  // Pairs usable for registration, track extension and triangulation (the constructor's, then
  // e.g. loop pairs once segments are merged); a pair already known is skipped
  addPairs(pairs) {
    const have = new Set(this.adj.flatMap((list, i) => list.map(e => `${i},${e.other}`)));
    for (const { i, j, matches } of pairs) {
      if (have.has(`${i},${j}`)) continue;
      const a = new Int32Array(matches.length / 2), b = new Int32Array(matches.length / 2);
      for (let m = 0; m < a.length; m++) { a[m] = matches[2 * m]; b[m] = matches[2 * m + 1]; }
      this.adj[i].push({ other: j, a, b });
      this.adj[j].push({ other: i, a: b, b: a });
      have.add(`${i},${j}`).add(`${j},${i}`);
    }
  }

  // S4 (DESIGN-sfm §5.5): window BAs gliding over each segment, a window being `window` consecutive
  // registered frames plus its most covisible frames elsewhere (loop partners). Successive sweeps
  // shift the windows by half a window, so their boundaries do not stay fixed. Free intrinsics are
  // refined by each window BA in turn, starting from the previous window's: the final f, k1 are
  // the last window's estimate, fitted to its frames. One more sweep with them fixed then fits
  // every other window to that camera (without it, a window keeps the poses and points it made for
  // its own estimate: gare-st-lazare 1.97 -> 0.79 px median reprojection, jardin5 p90 1.62 -> 0.73).
  refine({ sweeps = 2, refineIntrinsics = true, extra = 5 } = {}) {
    const W = this.o.window, half = Math.max(1, W >> 1);
    const segs = [...new Set(Array.from(this.segment).filter(s => s >= 0))];
    const sizes = segs.map(seg => this.poses.filter((P, k) => P && this.segment[k] === seg).length);
    const all = sweeps && refineIntrinsics ? sweeps + 1 : sweeps;  // + the sweep with the intrinsics fixed
    let done = 0, total = 0;  // windows, for progress
    for (let sw = 0; sw < all; sw++) for (const n of sizes) total += Math.max(0, Math.ceil((n - (sw % 2) * half) / W));
    for (let sw = 0; sw < all; sw++) {
      for (const seg of segs) {
        const reg = [];
        for (let k = 0; k < this.n; k++) if (this.poses[k] && this.segment[k] === seg) reg.push(k);
        for (let start = (sw % 2) * half; start < reg.length; start += W) {
          const core = reg.slice(start, start + W);
          if (!core.length) continue;
          this.localBA(core[0], { frames: [...core, ...this.partners(core, { extra, minGap: W })], refineIntrinsics: refineIntrinsics && sw < sweeps, iterations: 10,
                                  minCameraFrames: this.o.minCameraFramesSweeps });
          this.o.progress('refine', ++done, total);
        }
      }
    }
  }

  // The frames outside `core`, more than minGap from its first frame, sharing the most points with
  // it: at most `extra`, each with at least minShared points (loop partners of a window). Measured
  // from the first frame only, so the frames just after a refine() window qualify too: overlap with
  // the next window that jardin5 needs (without it, one COLMAP piece: 0.28 -> 0.51 deg median rotation).
  partners(core, { extra = 5, minShared = 50, minGap = 0 } = {}) {
    const inCore = new Set(core), shared = new Map();
    for (const k of core) {
      for (const pid of this.kpPoint[k]) {
        if (pid < 0 || !this.points[pid]) continue;
        for (const img of this.points[pid].obs.keys()) if (!inCore.has(img) && Math.abs(img - core[0]) > minGap) shared.set(img, (shared.get(img) ?? 0) + 1);
      }
    }
    return [...shared].filter(([, c]) => c >= minShared).sort((a, b) => b[1] - a[1]).slice(0, extra).map(([img]) => img);
  }

  // End of S4 (DESIGN-sfm §5.5, final retriangulation, mapping side): with the final poses, extend
  // tracks and triangulate new ones over all usable pairs, then drop observations that no longer
  // reproject. Existing points keep their BA positions; the trainer's stricter point filter is
  // export.js datasetFromMapper's.
  extendTracks() {
    for (let k = 0; k < this.n; k++) if (this.poses[k]) this.triangulateNew(k);
    this.filter(this.points.keys());
  }

  // ── repair primitives (DESIGN-sfm §6.3) ─────────────────────────────────────

  // Pairs whose matches changed (e.g. re-matched with looser settings): replaced, or added
  replacePairs(pairs) {
    const keys = new Set(pairs.flatMap(({ i, j }) => [`${i},${j}`, `${j},${i}`]));
    this.adj = this.adj.map((list, i) => list.filter(e => !keys.has(`${i},${e.other}`)));
    this.addPairs(pairs);
  }

  // Remove frames from the map: poses, their observations, and points left with < 2 observations
  unregister(frames) {
    for (const k of frames) {
      const kp = this.kpPoint[k];
      for (let a = 0; a < kp.length; a++) {
        const pid = kp[a];
        if (pid < 0) continue;
        kp[a] = -1;
        const P = this.points[pid];
        if (!P) continue;
        P.obs.delete(k);
        if (P.obs.size < 2) {
          for (const [img, b] of P.obs) this.kpPoint[img][b] = -1;
          this.points[pid] = null;
        }
      }
      if (this.poses[k]) this.stats.registered--;
      this.poses[k] = null;
      this.segment[k] = -1;
    }
  }

  // New frame list: oldIndex[k] = the previous index of new frame k, or -1 for an inserted frame
  // (no pose, no matches yet); frames absent from oldIndex are dropped with their observations.
  // Time order is the caller's: names sort like timestamps (DESIGN-sfm §3).
  reindex(images, oldIndex) {
    const toNew = new Int32Array(this.n).fill(-1);
    oldIndex.forEach((o, k) => { if (o >= 0) toNew[o] = k; });
    this.unregister([...toNew.keys()].filter(o => toNew[o] < 0));
    const n = images.length, at = (arr, fill) => oldIndex.map(o => (o >= 0 ? arr[o] : fill()));
    this.poses = at(this.poses, () => null);
    this.segment = Int32Array.from(at(this.segment, () => -1));
    this.kpPoint = oldIndex.map((o, k) => (o >= 0 ? this.kpPoint[o] : new Int32Array(images[k].keypoints.length / 2).fill(-1)));
    this.adj = oldIndex.map((o) => (o >= 0 ? this.adj[o].filter(e => toNew[e.other] >= 0).map(e => ({ ...e, other: toNew[e.other] })) : []));
    for (const P of this.points) {
      if (P) P.obs = new Map([...P.obs].map(([img, kp]) => [toNew[img], kp]));
    }
    this.images = images;
    this.n = n;
    this.camOf = Int32Array.from(images, im => im.camera ?? 0);
  }

  // Copy of the mutable state, for undo (features and pair matches are shared, never mutated)
  snapshot() {
    return {
      cameras: this.cameras.map(c => ({ ...c, params: [...c.params] })), camOf: this.camOf.slice(),
      images: this.images, n: this.n, nextSegmentId: this.nextSegmentId,
      poses: this.poses.map(p => p && { R: [...p.R], t: [...p.t] }),
      segment: this.segment.slice(),
      points: this.points.map(P => P && { X: [...P.X], obs: new Map(P.obs) }),
      kpPoint: this.kpPoint.map(a => a.slice()),
      adj: this.adj.map(list => list.slice()),
      stats: structuredClone(this.stats), o: { ...this.o, refine: [...this.o.refine] },
    };
  }

  // Back to a snapshot (which is consumed: its arrays become the live state)
  restore(s) { Object.assign(this, s); }

  // Compact state for saving (project.js): what the pairs and the points do not imply
  toState() {
    const live = this.points.filter(Boolean);
    const obs = live.flatMap(P => [...P.obs].flat());
    return {
      json: { camera: this.camera, cameras: this.cameras, nextSegmentId: this.nextSegmentId, stats: this.stats, o: { ...this.o, log: undefined, progress: undefined } },
      poses: Float64Array.from(this.poses.flatMap(p => p ? [...p.R, ...p.t] : new Array(12).fill(NaN))),
      segment: this.segment.slice(),
      pointX: Float64Array.from(live.flatMap(P => P.X)),
      pointObsCount: Uint32Array.from(live.map(P => P.obs.size)),
      pointObs: Int32Array.from(obs),
    };
  }

  // Inverse of toState(), on a mapper built from the same keyframes and pairs (all of them)
  setState({ json, poses, segment, pointX, pointObsCount, pointObs }) {
    Object.assign(this, { cameras: json.cameras ?? [json.camera], nextSegmentId: json.nextSegmentId ?? json.numSegments, stats: json.stats, o: { ...this.o, ...json.o, log: this.o.log, progress: this.o.progress } });
    this.poses = Array.from({ length: this.n }, (_, k) => Number.isNaN(poses[12 * k]) ? null
      : { R: Array.from(poses.subarray(12 * k, 12 * k + 9)), t: Array.from(poses.subarray(12 * k + 9, 12 * k + 12)) });
    this.segment = Int32Array.from(segment);
    this.kpPoint = this.images.map(im => new Int32Array(im.keypoints.length / 2).fill(-1));
    this.points = [];
    for (let p = 0, o = 0; p < pointObsCount.length; p++) {
      const obs = [];
      for (let m = 0; m < pointObsCount[p]; m++, o += 2) obs.push([pointObs[o], pointObs[o + 1]]);
      this.addPoint(Array.from(pointX.subarray(3 * p, 3 * p + 3)), obs);
    }
  }

  // The segment with the most registered frames (ties: the one met first in frame order), -1 for none
  largestSegment() {
    const sizes = new Map();
    for (let k = 0; k < this.n; k++) if (this.poses[k]) sizes.set(this.segment[k], (sizes.get(this.segment[k]) ?? 0) + 1);
    return [...sizes].sort((a, b) => b[1] - a[1])[0]?.[0] ?? -1;
  }

  numPoints() { return this.points.reduce((s, p) => s + (p ? 1 : 0), 0); }

  // { camera (camera 0), cameras, images: [{ name, index, segment, camera (index), R, t }], points: [{ X, obs }], focal (camera 0's),
  // numPoints } (compare.js model).
  // Segments have independent gauges until loop closure merges them.
  model() {
    const images = [];
    this.poses.forEach((p, k) => { if (p) images.push({ name: this.images[k].name, index: k, segment: this.segment[k], camera: this.camOf[k], R: p.R, t: p.t }); });
    const points = this.points.filter(Boolean);
    return { camera: this.camera, cameras: this.cameras, images, points, focal: meanFocal(this.camera), numPoints: points.length };
  }
}

