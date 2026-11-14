// mapper.js — Sequential incremental mapping with local and global bundle adjustment (DESIGN-sfm §5.3, S2).
// Input is the `input` shape of synth.js / dump.js: { camera, images: [{ name, keypoints }],
// pairs: [{ i, j, type, matches }] } with images in time order; or { cameras: [camera], images:
// [{ name, keypoints, camera (index) }] }, a camera model per lens and zoom (DESIGN-sfm §13 P8). Only pairs of `pairTypes` are
// used ('seq' by default; reconstruct.js maps with the loop pairs too, as COLMAP). Steps:
//   1. focal self-calibration from fundamental matrices of sequential pairs (focal.js)
//   2. initial pair with enough parallax that a homography does not explain
//   3. repeatedly: register the frame whose 2D-3D matches cover the image best (nextImage; P3P RANSAC + pose
//      refinement), triangulate its new tracks, run a local BA (localBA; intrinsics free for the first frames),
//      complete and merge the tracks, and drop observations that no longer reproject (up to localRefinements times);
//      a global BA each time the segment grows by globalBARatio.
"use strict";

import { makeRng, dist } from '../math.js';
import { normalizedFromImage, project, poseCenter, meanFocal, withFocal, focalFromHfov } from './camera.js';
import { ransacFundamental, ransacHomography, relativePose, triangulate, triangulationAngle, depthIn, ransacPnP } from './geometry.js';
import { estimateFocal } from './focal.js';
import { bundleAdjust } from './ba.js';
import { GpuBundleAdjuster } from './ba_gpu.js';
import { median, deg, matchedPixels } from './util.js';
import { mul3, mulVec3, rotationAngle } from './linalg.js';
import { compose, inverse, apply, fromPose, toPose } from './sim3.js';
import { WINDOW } from './defaults.js';

// run(beat)'s promise, or STALLED thrown once ms have passed since it began or last called beat() (the GPU's solves:
// one that never ends falls back as an error does; one that goes on, its GPU answering, is not stalled however long:
// autumn-garden's global BA of 677 frames passed 120 s in its Cholesky and turned the GPU off for every BA after it)
export const STALLED = new Error('stalled');
const TRACK_PARTS = 50;  // the retriangulation's parts (extendTracks)
export function watchdog(run, ms) {
  let timer, fail, over = false;
  const stall = new Promise((_, reject) => { fail = reject; });
  // beats after the race (a stalled solve that goes on) arm nothing
  const beat = () => { if (over) return; clearTimeout(timer); timer = setTimeout(() => fail(STALLED), ms); };
  beat();
  return Promise.race([run(beat), stall]).finally(() => { over = true; clearTimeout(timer); });
}

const DEFAULTS = {
  seed: 1,
  pairTypes: ['seq'],
  window: WINDOW,          // localBA 'window' and the sweeps: the `window` registered frames nearest the new one are free
  focal: 'auto',           // 'auto' | pixels | 'prior' (keep the input camera)
  hfovDeg: null,           // horizontal field of view instead of `focal` (DESIGN-sfm §5.3: hfov=)
  minFocalSharpness: 1.3,  // below: the focal is unobservable from F, use the fallback field of view
  weakFocalSharpness: 3,   // below: weakly constrained, warn
  fallbackHfovDeg: 76,     // typical drone camera (DJI: f = 0.64 w), across the long side: a portrait video's height
  fallbackFocal: null,     // pixels, instead of fallbackHfovDeg (a photo set's EXIF focal); override both with hfovDeg / focal
  maxFocalGap: 8,          // focal calibration: widest frame gap tried per frame
  maxFocalPairs: 200,      // and number of frames sampled
  // the intrinsics refined in each local BA while mapping (frames 6..refineIntrinsicsUntil registered, null: all), and
  // their terms: on gare2 (COLMAP's matches) the lens fixed from its first 50 frames left the path 25% off COLMAP's at
  // the ends, f and k1 alone as far off, p1, p2 too without k2 6.5% at p90: all of them, in every window, 0.35% median,
  // 1.2% p90. refineUnobserved: those of them refined when F does not observe the focal (it is given, the fallback or the
  // prior): over flat ground seen from above, k2, p1, p2 moved it 3.65% off (mapper_test's nadir pass; 0.23% with f, k1)
  refineIntrinsicsUntil: null,
  refine: ['f', 'k1', 'k2', 'p1', 'p2', 'fy'],  // fy apart from fx: COLMAP's OPENCV
  refineUnobserved: ['f', 'k1', 'fy'],
  minCameraFrames: 5,      // several cameras: while mapping, one's intrinsics are refined once it has this many registered frames (P8.3)
  minCameraFramesSweeps: 2, // and in the final sweeps (refine()) from this many: a lens with a few photos is fitted to the
                            // whole map instead of keeping its EXIF ratio (DESIGN-sfm §13 P8)
  minInliers: 30,          // PnP inliers to register a frame
  relativeRegistration: false,  // when PnP has no frame left: registerRelative (photo sets)
  relMinMatches: 50,       // registerRelative: F inliers with the registered neighbour
  relMinScaleInliers: 8,   // and 2D-3D correspondences agreeing on the baseline length
  relScaleTol: 0.1,        // relative, for that agreement
  relMaxTurnDeg: 10,       // and the most the pose refinement may turn the pair's rotation
  regrowSegments: false,   // after mapping: offer the frames of smaller segments to the largest (photo sets)
  joinPieces: false,       // the segments loop.js cannot merge by their 3D points placed by joinpieces.js (photo sets)
  scanFocal: false,        // several cameras: those but camera 0 get their focal from their first frame to register (scanFocalOf: a video's zooms)
  scanMinCorr: 60,         // 2D-3D correspondences that frame needs
  mergeInliers: 0.97,      // a scanned camera joins a known one that its frame fits nearly as well (this fraction of the inliers)
  mergeMaxRatio: 1.1,      // at a focal this close
  pnpThreshold: 12,        // px (COLMAP's abs_pose_max_error)
  maxReproj: 4,            // px, triangulation and filtering
  minAngleDeg: 1.5,        // triangulation angle of new points
  initMinAngleDeg: 3,      // median triangulation angle of the initial pair
  initOrder: 'time',       // initial pair: 'time' (first frames first) | 'strongest' (most matches first)
  baIterations: 10,
  // a global BA (every frame of the segment free, the intrinsics as the window's) each time the segment has grown by
  // this ratio since the last one (COLMAP's ba_global_frames_ratio), from globalBAMinFrames frames on, then the
  // segment retriangulated with the poses it fixed; null: none. Window BAs alone let the scale drift where little
  // parallax holds it (gare2's pan at 34-38 s: the steps after it 1/11 of COLMAP's)
  globalBARatio: 1.1,
  globalBAMinFrames: 20,
  globalBAIterations: 10,
  // a match too narrow for its pair (minAngleDeg) triangulated from its track instead (triangulateTrack): while the
  // camera turns, each pair of frames is too narrow for the near points, the track across the turn is not (gare2's
  // pavement through the pan: without them, a step sideways taken for a turn, 30 deg off COLMAP 8 s later)
  trackTriangulation: true,
  // COLMAP's local BA after each registration ('covisible'): the new frame and its localBAImages - 1 most covisible frames
  // (those localBAMinAngleDeg apart first, as FindLocalBundle), its points of tracks up to localBAMaxTrack and the points
  // seen only by those frames free, the others held; localBAIterations, the loss localBALoss at localBALossScale px on the
  // first pass, least squares on the repeats (on ba.js: the GPU solver has neither held points nor those losses).
  // 'window': the window frames nearest in time, all their points free
  localBA: 'covisible',
  localBAImages: 6,
  localBAMinAngleDeg: 6,
  localBAMaxTrack: 15,
  localBAIterations: 25,
  localBALoss: 'softl1',
  localBALossScale: 1,
  triMaxReproj: null,      // px, creating and continuing a point; null: triMaxDeg at the focal, else maxReproj
  triMaxDeg: 2,            // COLMAP's create/continue_max_angle_error (33 px at f 950)
  completeMerge: true,     // COLMAP's CompleteAndMergeTracks after each local BA (the bundle's points) and global BA (all)
  completeHops: 5,         // COLMAP's complete_max_transitivity
  localRefinements: 2,     // COLMAP's ba_local_max_refinements: the local BA, completion, merge and filter again while they change >= 0.001 of the observations
  globalRefinements: 5,    // COLMAP's ba_global_max_refinements, at 0.0005; > 1 also completes, merges and retriangulates (reMaxDeg) before the BA
  reMaxDeg: 5,             // COLMAP's re_max_angle_error
  // that retriangulation skips a frame pair whose matches are this much one point already (COLMAP's re_min_ratio, its 0.2);
  // null: none skipped. Autumn-garden's first 450 keyframes: 396 -> 291 s, 1 segment as before (at 0.2: 2), its end's
  // poses nearer COLMAP's (its 220-frame model: 1.43 -> 0.53% RMS)
  reMinRatio: 0.05,
  ignoreTwoView: true,     // COLMAP's ignore_two_view_tracks: no point made from a keypoint matched in one other frame only
  nextImage: 'pyramid',    // 'pyramid': COLMAP's MIN_UNCERTAINTY (a 6-level visibility pyramid of the correspondences); 'count': the most
  filterMinAngle: true,    // COLMAP's filter_min_tri_angle: a point whose widest pair of views is under minAngleDeg removed
  maxTrack: 40,            // the frames of a track at most
  // the BAs without held points (localBA): the GPU solver (ba_gpu.js) for one worth it (gpuFriendly), ba.js otherwise, and
  // ba.js from the start where the GPU solve fails (illReason, an error) or there is no WebGPU
  baIllPivot: 1e-4,        // a pose pivot under it (the gauge fixed) fails the GPU solve
  baIllRmse: true,         // and so does its RMSE up
  baGpuMinObs: 2000,       // fewer observations to ba.js (gpuFriendly)
  baGpu: null,             // the GPU solver (a GpuBundleAdjuster or its promise; null: GpuBundleAdjuster.shared())
  baDenseMax: undefined,   // the most reduced-system unknowns solved by dense Cholesky (rcs.js DENSE_MAX), conjugate gradients past it
  baCgTol: undefined,      // and their stop (rcs.js CG_TOL)
  baGpuTimeoutMs: 120000,  // a GPU solve silent that long (no readback), or the solver's making not done by then: stalled, the GPU off from there
  log: () => {},
  progress: () => {},       // (stage, done, total): 'init' over the candidate pairs, 'register' while growing, 'refine' per window
  // async (mapper) after a registration once checkpointMs have passed since the mapping began or the last one: the state
  // saved (toState) for run({ resume }) to go on from (a long video maps for hours: a crash, a timeout, the machine
  // asleep lose them); null: none
  checkpoint: null,
  checkpointMs: 15 * 60000,
};

// The keypoint of e.other matched to keypoint k of the edge's frame (n keypoints), -1: none. The edge's inverse
// index is made on first use (an Int32Array of n: on gare2, 16 KB an edge)
function matchOf(e, k, n) {
  if (!e.inv) { e.inv = new Int32Array(n).fill(-1); for (let m = 0; m < e.a.length; m++) e.inv[e.a[m]] = e.b[m]; }
  return e.inv[k];
}

// Copy of a mapper's mutable state, or of a snapshot of it (features and pair matches are shared,
// never mutated)
export function copySnapshot(s) {
  return {
    cameras: s.cameras.map(c => ({ ...c, params: [...c.params] })), camOf: s.camOf.slice(),
    merged: s.merged.slice(), focalKnown: s.focalKnown.slice(),
    images: s.images, n: s.n, nextSegmentId: s.nextSegmentId,
    poses: s.poses.map(p => p && { R: [...p.R], t: [...p.t] }),
    segment: s.segment.slice(), byFlow: s.byFlow.slice(),
    points: s.points.map(P => P && { X: [...P.X], obs: new Map(P.obs) }),
    kpPoint: s.kpPoint.map(a => a.slice()),
    adj: s.adj.map(list => list.slice()),
    stats: structuredClone(s.stats), o: { ...s.o, refine: [...s.o.refine], refineUnobserved: [...s.o.refineUnobserved] },
  };
}

export class SequentialMapper {
  constructor(input, options = {}) {
    this.o = { ...DEFAULTS, ...options };
    this.rng = makeRng(this.o.seed);
    this.cameras = (input.cameras ?? [input.camera]).map(c => ({ ...c, params: [...c.params] }));
    this.priors = this.cameras.map(c => meanFocal(c));  // the input focals: their ratios place the other cameras (calibrateFocal)
    this.images = input.images;
    this.n = input.images.length;
    this.merged = Int32Array.from(this.cameras, (_, k) => k);  // camera -> the camera it joined (scanFocalOf), itself by default
    this.focalKnown = Uint8Array.from(this.cameras, (_, k) => (k === 0 || !this.o.scanFocal ? 1 : 0));
    this.scanNeeds = new Map();  // camera -> the correspondences its next focal scan needs (scanFocalOf)
    this.camOf = this.camerasOf(input.images);
    // adjacency: per image, [{ other, a (my keypoint indices), b (other's) }]
    this.adj = Array.from({ length: this.n }, () => []);
    this.addPairs(input.pairs.filter(p => this.o.pairTypes.includes(p.type)));
    this.kpPoint = input.images.map(im => new Int32Array(im.keypoints.length / 2).fill(-1));
    this.poses = new Array(this.n).fill(null);
    this.byFlow = new Uint8Array(this.n);  // 1: posed by the flow across a turn in place (flowjoin.js), not by points
    this.points = [];  // { X: [3], obs: Map(image -> keypoint) } | null when deleted
    this.segment = new Int32Array(this.n).fill(-1);  // connected map id per frame (S3 merges them)
    this.nextSegmentId = 0;  // ids ever created; merges leave gaps (the live ones are in this.segment)
    this.stats = { registered: 0, focalInit: NaN, segments: [], initTried: 0 };  // registered: frames with a pose now; initTried: initial pairs' relative poses
    this.resetTime();
    // pairs no initial pair comes from, whatever the angle asked or the frames mapped: no relative pose, or a homography
    // explains it (rotation only, a plane); initializeWith skips them (a pair added or reindexed is a new one)
    this.noInit = new WeakSet();
  }

  // an event for the SfM theatre (trace.js; the `trace` option, none by default)
  trace(kind, info = {}) { this.o.trace?.(this, kind, info); }

  // camera 0 (the only one, or the reference of several: the worker puts the largest group first)
  get camera() { return this.cameras[0]; }
  set camera(c) { this.cameras[0] = c; }
  cam(img) { return this.cameras[this.camOf[img]]; }
  camerasOf(images) { return Int32Array.from(images, im => this.merged[im.camera ?? 0]); }

  pixel(img, kp) { const k = this.images[img].keypoints; return [k[2 * kp], k[2 * kp + 1]]; }
  ray(img, kp) { return normalizedFromImage(this.cam(img), ...this.pixel(img, kp)); }
  pairPixels(img, e) { return matchedPixels(this.images[img].keypoints, this.images[e.other].keypoints, e); }

  // ── 1. focal ──────────────────────────────────────────────────────────────
  calibrateFocal() {
    const { focal, maxFocalGap, maxFocalPairs } = this.o;
    if (focal === 'prior') { this.stats.focalSource = 'prior'; return (this.stats.focalInit = meanFocal(this.camera)); }
    let f = this.o.hfovDeg ? focalFromHfov(this.camera.width, this.o.hfovDeg) : focal, source = this.o.hfovDeg ? 'hfov' : 'given';
    if (!this.o.hfovDeg && focal === 'auto') {
      // per frame (sampled over the whole sequence), the widest-gap pair that still has enough
      // matches: parallax makes F informative about the focal
      const Fs = [], stride = Math.max(1, Math.floor(this.n / maxFocalPairs));  // floor: up to 2x maxFocalPairs, the denser sample
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
        // a camera's field of view is across its sensor's long side, however the frame is held (a portrait
        // video's width would give far too short a focal)
        const long = Math.max(this.camera.width, this.camera.height);
        f = this.o.fallbackFocal ?? focalFromHfov(long, this.o.fallbackHfovDeg); source = 'fallback';
        const from = this.o.fallbackFocal ? 'the prior' : `${this.o.fallbackHfovDeg} deg across the long side`;
        this.o.log(`focal: not observable from ${Fs.length} pairs (sharpness ${est ? est.sharpness.toFixed(2) : '-'}, ` +
                   `e.g. translation without rotation): using ${from} = ${f.toFixed(1)} px; set hfov= or focal= if known`);
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
  // Frames of a camera whose focal is still unknown (scanFocal) start one only when no other pair can.
  async initialize(allowed = null) {
    const withKnown = new Set();
    let unknown = false;
    for (let k = 0; k < this.n; k++) {
      if (allowed && !allowed.has(k)) continue;
      if (this.focalKnown[this.camOf[k]]) withKnown.add(k); else unknown = true;
    }
    for (const set of unknown ? [withKnown, allowed] : [allowed]) {
      for (const minAngle of this.initAngles()) {
        if (await this.initializeWith(minAngle, set)) return true;
      }
    }
    return false;
  }

  // The median triangulation angles an initial pair needs, tried in turn (each once: the defaults' half is minAngleDeg)
  initAngles() { return [...new Set([this.o.initMinAngleDeg, this.o.initMinAngleDeg / 2, this.o.minAngleDeg])]; }

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

  async initializeWith(minAngleDeg, allowed = null) {
    const ok = (k) => !this.poses[k] && (!allowed || allowed.has(k));
    const pairs = this.initialPairs(ok);
    for (let q = 0; q < pairs.length; q++) {
      const [i, e] = pairs[q];
      if (q % 64 === 0) this.o.progress('init', q, pairs.length);
      if (e.a.length < 100 || this.noInit.has(e)) continue;
      const [x1, x2] = this.pairPixels(i, e);
      this.stats.initTried++;
      const rel = relativePose(x1, x2, this.cam(i), { rng: this.rng, threshold: 2, minInliers: 100, camera2: this.cam(e.other) });
      if (!rel) { this.noInit.add(e); continue; }
      const H = ransacHomography(x1, x2, { rng: this.rng, threshold: 4 });
      if (H && H.numInliers > 0.8 * rel.F.numInliers) { this.noInit.add(e); continue; }  // rotation only / planar: no reliable depth
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
      // segment that cannot grow, holding frames a better pair needs
      this.poses[i] = P1; this.poses[e.other] = P2;
      const kept = cand.filter(({ m, X }) => this.reprojErr(i, e.a[m], X) <= this.o.maxReproj && this.reprojErr(e.other, e.b[m], X) <= this.o.maxReproj);
      if (kept.length < 100) { this.poses[i] = this.poses[e.other] = null; continue; }
      const seg = this.nextSegmentId++;
      this.segment[i] = this.segment[e.other] = seg;
      let created = 0;
      for (const { m, X } of kept) created += this.addPoint(X, [[i, e.a[m]], [e.other, e.b[m]]]) >= 0;
      this.o.log(`segment ${seg}: init frames ${i}-${e.other}, ${rel.F.numInliers} F inliers, ${created} points, median angle ${med.toFixed(1)} deg`);
      this.trace('init', { segment: seg, i, j: e.other, points: created, angleDeg: med });
      await this.localBA(e.other, { refineIntrinsics: false, iterations: 20 });
      return true;
    }
    return false;
  }

  addPoint(X, obs, checkReproj = false) {
    if (checkReproj && obs.some(([img, kp]) => this.reprojErr(img, kp, X) > this.triMax())) return -1;
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

  // COLMAP's VisibilityPyramid score: levels of 2x2 .. 64x64 cells over the image, each cell holding a correspondence adds its level's cell count
  pyramidScore(img, corr) {
    const { width, height } = this.camera, seen = new Set();
    let score = 0;
    for (const kp of corr.keys()) {
      const [x, y] = this.pixel(img, kp);
      let cx = Math.min(63, Math.max(0, Math.floor(64 * x / width))), cy = Math.min(63, Math.max(0, Math.floor(64 * y / height)));
      for (let l = 5; l >= 0; l--, cx >>= 1, cy >>= 1) {
        const key = (l << 12) | (cy << 6) | cx;
        if (!seen.has(key)) { seen.add(key); score += 4 ** (l + 1); }
      }
    }
    return score;
  }

  // ── 3. registration ───────────────────────────────────────────────────────
  register(img, seg = this.nextSegmentId - 1) {
    const corr = [...this.correspondences(img, seg)];
    if (corr.length < this.o.minInliers) return false;
    if (!this.focalKnown[this.camOf[img]] && !this.scanFocalOf(img, corr)) return false;
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
      if (P && !P.obs.has(img) && this.reprojErr(img, kp, P.X) <= this.triMax()) {
        P.obs.set(img, kp);
        this.kpPoint[img][kp] = pid;
      }
    }
    return true;
  }

  // The focal of img's camera, still unknown (scanFocal: a video's stretch after a zoom, worker.js),
  // from img's 2D-3D correspondences corr: PnP at focals from 1/8 to 8 times the camera's prior (8
  // per octave), then finer around the best (64 per octave); the focal whose refined pose reprojects
  // the most correspondences within pnpThreshold wins (the least error among ties). A known camera
  // that img fits nearly as well (mergeInliers) at a focal within mergeMaxRatio takes this one's
  // frames instead: the same lens and zoom again (zoomed in, then back out).
  // False when img has fewer than scanMinCorr correspondences or no focal gives minInliers; the
  // camera's next scan then needs 1.5 times as many (a scan is 66 PnP RANSACs, each with a pose refinement).
  scanFocalOf(img, corr) {
    const c = this.camOf[img], base = this.cameras[c], t = this.o.pnpThreshold;
    if (corr.length < Math.max(this.o.scanMinCorr, this.scanNeeds.get(c) ?? 0)) return false;
    const pixels = corr.map(([kp]) => this.pixel(img, kp)), pts = corr.map(([, pid]) => this.points[pid].X);
    const score = (cam) => {
      const rays = pixels.map(([u, v]) => normalizedFromImage(cam, u, v));
      const res = ransacPnP(rays, pixels, pts, cam, { rng: this.rng, threshold: t });
      if (!res) return { n: 0, err: Infinity, cam };
      const pose = this.refinePose(img, res.model, corr.filter((_, k) => res.inliers[k]), cam);
      let n = 0, err = 0;
      for (const [kp, pid] of corr) {
        const e = this.reprojWith(pose, img, kp, this.points[pid].X, cam);
        if (e <= t) { n++; err += e; }
      }
      return { n, err, cam };
    };
    const better = (a, b) => !b || a.n > b.n || (a.n === b.n && a.err < b.err);
    let best = null;
    const scan = (center, perOctave, steps) => {
      for (let k = -steps; k <= steps; k++) {
        const s = score(withFocal(base, center * 2 ** (k / perOctave)));
        if (better(s, best)) best = s;
      }
    };
    scan(meanFocal(base), 8, 24);
    scan(meanFocal(best.cam), 64, 8);
    if (best.n < this.o.minInliers) { this.scanNeeds.set(c, Math.ceil(1.5 * corr.length)); return false; }
    const f = meanFocal(best.cam);
    let into = -1, intoScore = null;
    for (let d = 0; d < this.cameras.length; d++) {
      if (d === c || this.merged[d] !== d || !this.focalKnown[d]) continue;
      const r = meanFocal(this.cameras[d]) / f;
      if (Math.max(r, 1 / r) > this.o.mergeMaxRatio) continue;
      const s = score(this.cameras[d]);
      if (s.n >= this.o.mergeInliers * best.n && better(s, intoScore)) { into = d; intoScore = s; }
    }
    this.focalKnown[c] = 1;
    const ratio = (x) => (x / meanFocal(this.camera)).toFixed(3);
    if (into >= 0) {
      for (let k = 0; k < this.merged.length; k++) if (this.merged[k] === c) this.merged[k] = into;
      this.camOf = this.camerasOf(this.images);
      this.o.log(`camera ${c}: focal ${f.toFixed(1)} px (x${ratio(f)}) from frame ${img}, ${best.n}/${corr.length} 2D-3D; ` +
                 `camera ${into} (${meanFocal(this.cameras[into]).toFixed(1)} px) fits ${intoScore.n}: its frames join camera ${into}`);
    } else {
      this.cameras = this.cameras.map((x, k) => (k === c ? best.cam : x));
      this.o.log(`camera ${c}: focal ${f.toFixed(1)} px (x${ratio(f)}, prior ${meanFocal(base).toFixed(1)}) from frame ${img}, ${best.n}/${corr.length} 2D-3D`);
    }
    (this.stats.focalScans ??= []).push({ camera: c, frame: img, focal: f, inliers: best.n, of: corr.length, into });
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
      if (this.poses[k] || (failed.get(k) ?? 0) > 1 || (candidates && !candidates.has(k)) || !this.focalKnown[this.camOf[k]]) continue;
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
      // can turn the camera far away
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
  refinePose(img, pose, corr, camera = this.cam(img)) {
    const prob = {
      camera, poses: [pose], free: Uint8Array.of(1),
      points: Float64Array.from(corr.flatMap(([, pid]) => this.points[pid].X)),
      obsCam: new Int32Array(corr.length), obsPoint: Int32Array.from(corr.keys()),
      obsUV: Float64Array.from(corr.flatMap(([kp]) => this.pixel(img, kp))),
    };
    bundleAdjust(prob, { fixPoints: true, maxIterations: 10 });
    return prob.poses[0];
  }

  reprojWith(pose, img, kp, X, camera = this.cam(img)) {
    const [u, v, z] = project(camera, pose, X), [pu, pv] = this.pixel(img, kp);
    return z > 0 ? Math.hypot(u - pu, v - pv) : Infinity;
  }

  reprojErr(img, kp, X) { return this.reprojWith(this.poses[img], img, kp, X); }

  // New tracks of `img` with its registered neighbours, and extension of existing ones. seen (a Set, optional): the
  // frame pairs done in a pass over many frames, each done once (from its first frame's side, not again from the other's:
  // its matches are the same both ways). minRatio: a pair whose matches are that much one point already skipped (COLMAP's
  // re_min_ratio)
  triangulateNew(img, seen = null, minRatio = null) {
    const C = poseCenter(this.poses[img]);
    const minAngle = this.o.minAngleDeg * Math.PI / 180;
    const pairKey = (e) => img < e.other ? img * this.n + e.other : e.other * this.n + img;
    const done = (e) => {
      let n = 0;
      for (let m = 0; m < e.a.length; m++) { const p = this.kpPoint[img][e.a[m]]; if (p >= 0 && p === this.kpPoint[e.other][e.b[m]]) n++; }
      return n >= minRatio * e.a.length;
    };
    const nbrs = this.adj[img].filter(e => this.poses[e.other] && this.segment[e.other] === this.segment[img] && !seen?.has(pairKey(e)) &&
                                           !(minRatio != null && done(e)))
      .map(e => ({ e, base: dist(poseCenter(this.poses[e.other]), C) }))
      .sort((p, q) => q.base - p.base);
    // img's keypoints whose track failed (triangulateTrack): walked once, not again from each narrow neighbour (on
    // autumn-garden's first 450 keyframes, 19.0M of 24.5M walks, the mapping 430 -> 266 s). The track can change
    // meanwhile (the keypoints the loop gives points to leave it), so a later walk could succeed where this skips it:
    // there the same 449/450 registered, 1 segment
    const failed = new Set();
    let created = 0;
    for (const { e } of nbrs) seen?.add(pairKey(e));
    for (const { e } of nbrs) {
      const Co = poseCenter(this.poses[e.other]);
      for (let m = 0; m < e.a.length; m++) {
        const a = e.a[m], b = e.b[m];
        const pa = this.kpPoint[img][a], pb = this.kpPoint[e.other][b];
        if (pa >= 0 && pb < 0) {  // extend the track to the neighbour
          const P = this.points[pa];
          if (!P.obs.has(e.other) && this.reprojErr(e.other, b, P.X) <= this.triMax()) { P.obs.set(e.other, b); this.kpPoint[e.other][b] = pa; }
        } else if (pa < 0 && pb >= 0) {
          const P = this.points[pb];
          if (!P.obs.has(img) && this.reprojErr(img, a, P.X) <= this.triMax()) { P.obs.set(img, a); this.kpPoint[img][a] = pb; }
        } else if (pa < 0 && pb < 0) {
          if (this.o.ignoreTwoView && this.matchCount(img)[a] === 1) continue;  // a two-view track
          const X = triangulate([this.poses[img], this.poses[e.other]], [this.ray(img, a), this.ray(e.other, b)]);
          if (!X || depthIn(this.poses[img], X) <= 0 || depthIn(this.poses[e.other], X) <= 0) continue;
          if (triangulationAngle(C, Co, X) < minAngle) {
            if (this.o.trackTriangulation && !failed.has(a)) {
              if (this.triangulateTrack(img, a, minAngle) >= 0) created++;
              else failed.add(a);
            }
            continue;
          }
          if (this.addPoint(X, [[img, a], [e.other, b]], true) >= 0) created++;
        }
      }
    }
    return created;
  }

  // The track of keypoint `a` of `img`: its matches, theirs, and so on (breadth first, at most maxTrack), in registered
  // frames of img's segment with no point yet. Triangulated from all of them; the views it reprojects in (maxReproj, in
  // front) kept, at least 2 with their widest pair minAngle apart. A frame twice in it: ambiguous, none. Returns the
  // point's id or -1.
  triangulateTrack(img, a, minAngle) {
    const seg = this.segment[img], frames = new Set([img]), track = [[img, a]];
    for (let q = 0; q < track.length && track.length < this.o.maxTrack; q++) {
      const [f, k] = track[q];
      for (const e of this.adj[f]) {
        if (!this.poses[e.other] || this.segment[e.other] !== seg) continue;
        const b = matchOf(e, k, this.images[f].keypoints.length >> 1);
        if (b < 0 || this.kpPoint[e.other][b] >= 0) continue;
        if (frames.has(e.other)) {
          if (!track.some(([g, c]) => g === e.other && c === b)) return -1;
          continue;
        }
        frames.add(e.other); track.push([e.other, b]);
        if (track.length >= this.o.maxTrack) break;
      }
    }
    if (track.length < 3) return -1;
    const X = triangulate(track.map(([f]) => this.poses[f]), track.map(([f, k]) => this.ray(f, k)));
    if (!X) return -1;
    const ok = track.filter(([f, k]) => depthIn(this.poses[f], X) > 0 && this.reprojErr(f, k, X) <= this.triMax());
    const Cs = ok.map(([f]) => poseCenter(this.poses[f]));
    let widest = 0;
    for (let i = 0; i < Cs.length; i++) for (let j = i + 1; j < Cs.length; j++) widest = Math.max(widest, triangulationAngle(Cs[i], Cs[j], X));
    return ok.length >= 2 && widest >= minAngle ? this.addPoint(X, ok) : -1;
  }

  // COLMAP's CompleteTracks then MergeTracks on points `pids` (maxReproj: its complete/merge_max_reproj_error 4 px).
  // Complete: each point's observations followed through the matches (completeHops) to registered frames of its segment
  // whose keypoint has no point, added where it reprojects; merge: a match between two points' keypoints, the two fused
  // at their track-weighted mean when it reprojects in every view (and they share no frame). Returns [added, merged,
  // the observations moved by the merges].
  completeAndMerge(pids) {
    let added = 0, merged = 0, mergedObs = 0;
    const n = (f) => this.images[f].keypoints.length >> 1, list = [...new Set(pids)];
    for (const pid of list) {
      const P = this.points[pid];
      if (!P) continue;
      const seg = this.segment[P.obs.keys().next().value];
      let front = [...P.obs];
      for (let hop = 0; hop < this.o.completeHops && front.length; hop++) {
        const next = [];
        for (const [f, k] of front) for (const e of this.adj[f]) {
          const b = matchOf(e, k, n(f));
          if (b < 0 || P.obs.has(e.other) || !this.poses[e.other] || this.segment[e.other] !== seg || this.kpPoint[e.other][b] >= 0) continue;
          if (depthIn(this.poses[e.other], P.X) > 0 && this.reprojErr(e.other, b, P.X) <= this.o.maxReproj) {
            P.obs.set(e.other, b); this.kpPoint[e.other][b] = pid; added++; next.push([e.other, b]);
          }
        }
        front = next;
      }
    }
    const tried = new Set();
    for (const pid0 of list) {
      let pid = pid0;
      for (let again = true; again;) {
        again = false;
        const P = this.points[pid];
        if (!P) break;
        for (const [f, k] of P.obs) {
          for (const e of this.adj[f]) {
            if (!this.poses[e.other]) continue;
            const b = matchOf(e, k, n(f)), q = b >= 0 ? this.kpPoint[e.other][b] : -1;
            if (q < 0 || q === pid || !this.points[q]) continue;
            const key = pid < q ? `${pid},${q}` : `${q},${pid}`;
            if (tried.has(key)) continue;
            tried.add(key);
            const Q = this.points[q];
            if ([...P.obs.keys()].some(g => Q.obs.has(g))) continue;
            const wp = P.obs.size, wq = Q.obs.size, X = P.X.map((x, i) => (wp * x + wq * Q.X[i]) / (wp + wq));
            const fits = (R) => [...R.obs].every(([g, c]) => depthIn(this.poses[g], X) > 0 && this.reprojErr(g, c, X) <= this.o.maxReproj);
            if (!fits(P) || !fits(Q)) continue;
            Q.X = X;
            const np = P.obs.size;
            if (this.fuseTracks(pid, q)) { merged++; mergedObs += np; pid = q; again = true; break; }
          }
          if (again) break;
        }
      }
    }
    return [added, merged, mergedObs];
  }

  // A global BA of segment `seg` (globalBARatio), then its frames retriangulated and its points filtered
  async globalBA(seg, center, refineIntrinsics) {
    const frames = [...this.segment.keys()].filter(k => this.poses[k] && this.segment[k] === seg);
    const t = performance.now();
    let created = 0, r = null, its = 1;
    if (this.o.globalRefinements > 1) {  // COLMAP's IterativeGlobalRefinement: complete, merge, retriangulate, then BA, complete, merge, filter while they change
      const cm = () => { const [a, m, mo] = this.completeAndMerge(this.points.keys()); this.stats.cm = [(this.stats.cm?.[0] ?? 0) + a, (this.stats.cm?.[1] ?? 0) + m]; return a + mo; };
      cm();
      const tri = this.o.triMaxReproj;
      this.o.triMaxReproj = this.o.reMaxDeg ? this.camera.params[0] * Math.tan(this.o.reMaxDeg * Math.PI / 180) : tri;
      const seen = new Set();
      for (const k of frames) created += this.triangulateNew(k, seen, this.o.reMinRatio);
      this.o.triMaxReproj = tri;
      for (let it = 0; it < this.o.globalRefinements; it++) {
        its = it + 1;
        let nObs = 0;
        for (const P of this.points) if (P) nObs += P.obs.size;
        r = await this.localBA(center, { frames, refineIntrinsics, iterations: this.o.globalBAIterations });
        const changed = (cm() + this.filter(this.points.keys()) + (r?.filtered ?? 0)) / Math.max(1, nObs);
        if (changed < 0.0005) break;
      }
    } else {
      r = await this.localBA(center, { frames, refineIntrinsics, iterations: this.o.globalBAIterations });
      const seen = new Set();
      for (const k of frames) created += this.triangulateNew(k, seen);
      if (this.o.completeMerge) { const [a, m] = this.completeAndMerge(this.points.keys()); this.stats.cm = [(this.stats.cm?.[0] ?? 0) + a, (this.stats.cm?.[1] ?? 0) + m]; }
      this.filter(this.points.keys());
    }
    (this.stats.globalBAs ??= []).push({ segment: seg, frames: frames.length, rmse: r?.rmse ?? NaN, created, its });
    this.o.log(`global BA of segment ${seg}, ${frames.length} frames: rmse ${r?.rmse?.toFixed(3)} px, f ${this.camera.params[0].toFixed(1)}, ` +
               `${created} points retriangulated, ${((performance.now() - t) / 1e3).toFixed(1)} s`);
  }

  // ── local BA ──────────────────────────────────────────────────────────────
  // Free: `frames`, else the local bundle of `center` (covisible), else the `window` registered frames nearest in time to
  // it; fixed: the other observers.
  async localBA(center, { refineIntrinsics = false, iterations = this.o.baIterations, frames = null, minCameraFrames = this.o.minCameraFrames, covisible = false, loss = null } = {}) {
    const t0 = performance.now();
    const reg = [];
    for (let k = 0; k < this.n; k++) if (this.poses[k] && this.segment[k] === this.segment[center]) reg.push(k);
    const freeSet = new Set(covisible ? this.localBundle(center) : frames ?? reg.sort((a, b) => Math.abs(a - center) - Math.abs(b - center)).slice(0, this.o.window));
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
    // COLMAP's local BA: a point held unless it is the new frame's with a short track, or seen by the free frames only
    const fixedPoints = covisible ? Uint8Array.from(ptList, pid => {
      const obs = this.points[pid].obs;
      if (obs.has(center) && obs.size <= this.o.localBAMaxTrack) return 0;
      for (const img of obs.keys()) if (!freeSet.has(img)) return 1;
      return 0;
    }) : null;
    if (free.every(Boolean)) free[camList.indexOf(Math.min(...camList))] = 0;
    const prob = {
      cameras: this.cameras, poseCam: Int32Array.from(camList, img => this.camOf[img]), poses: camList.map(img => this.poses[img]), free,
      points: Float64Array.from(ptList.flatMap(pid => this.points[pid].X)),
      obsCam: Int32Array.from(obsCam), obsPoint: Int32Array.from(obsPoint), obsUV: Float64Array.from(obsUV),
    };
    const opts = { refine: refineIntrinsics ? this.refineTerms() : [], refineCameras: this.refinable(minCameraFrames), maxIterations: iterations, fixedPoints,
                   ...(this.o.baDenseMax !== undefined ? { denseMax: this.o.baDenseMax } : {}), ...(this.o.baCgTol !== undefined ? { cgTol: this.o.baCgTol } : {}), ...(covisible ? { loss: loss ?? this.o.localBALoss, lossScale: this.o.localBALossScale } : {}) };
    const solver = !fixedPoints && this.gpuFriendly(prob) ? await this.gpuSolver() : null;
    this.time.route[solver ? 'gpu' : 'cpu']++;
    const t1 = performance.now();
    const start = solver && { cameras: prob.cameras, poses: prob.poses.slice(), points: prob.points.slice() };
    let error = '', stalled = false;
    let out = solver ? await watchdog((alive) => solver.bundleAdjust(prob, { ...opts, alive }), this.o.baGpuTimeoutMs).catch((e) => { error = e.message; stalled = e === STALLED; return null; })
                     : bundleAdjust(prob, opts);
    const reason = error ? 'GPU error' : solver ? this.illReason(out.cond) : '';
    if (reason) {  // solved again by ba.js, from the start (a stalled solve, should it end, writes to the arrays prob had)
      Object.assign(prob, start);
      const off = solver.lost || stalled;
      if (error) this.o.log(`window BA: the GPU solve failed (${error}), solved by ba.js${off ? `; ${stalled ? 'stalled' : 'its device lost'}, the GPU off from here` : ''}`);
      if (off) this._gpu = null;
      out = bundleAdjust(prob, opts);
      this.time.fallback.push(reason);
    }
    const ms = performance.now() - t1;
    this.time.solve += ms;
    this.time.windows.push({ ...out.profile, ms });
    this.cameras = prob.cameras;  // a camera for all its frames, whatever this window made of it
    camList.forEach((img, c) => { if (free[c]) this.poses[img] = prob.poses[c]; });
    ptList.forEach((pid, li) => { this.points[pid].X = Array.from(prob.points.subarray(3 * li, 3 * li + 3)); });
    out.filtered = this.filter(ptList);
    this.time.baCalls++;
    this.time.ba += performance.now() - t0;
    this.trace('ba', { center, free: camList.filter((_, c) => free[c]), observers: camList.length, points: ptList.length });
    return out;
  }

  // COLMAP's FindLocalBundle: img and the registered frames of its segment sharing the most of its points, those whose
  // shared points' triangulation angle (75th percentile) reaches localBAMinAngleDeg / [1, 1.5, ... 6] with at least
  // [0.6, ... 0.1] of its points shared first, then by count
  localBundle(img) {
    const shared = new Map(), mine = new Set();
    for (const pid of this.kpPoint[img]) {
      if (pid < 0 || !this.points[pid]) continue;
      mine.add(pid);
      for (const k of this.points[pid].obs.keys()) if (k !== img && this.poses[k] && this.segment[k] === this.segment[img]) shared.set(k, (shared.get(k) ?? 0) + 1);
    }
    const over = [...shared].sort((a, b) => b[1] - a[1]), want = Math.min(this.o.localBAImages - 1, over.length), out = [];
    if (over.length > want) {
      const C = poseCenter(this.poses[img]), angle = new Map();
      const tri = (k) => {
        if (!angle.has(k)) {
          const Ck = poseCenter(this.poses[k]), a = [];
          for (const pid of this.kpPoint[k]) if (pid >= 0 && mine.has(pid)) a.push(triangulationAngle(C, Ck, this.points[pid].X));
          a.sort((x, y) => x - y);
          angle.set(k, a.length ? a[Math.min(a.length - 1, Math.floor(0.75 * a.length))] : 0);
        }
        return angle.get(k);
      };
      const min = this.o.localBAMinAngleDeg * Math.PI / 180, used = new Set();
      for (const [div, frac] of [[1, 0.6], [1.5, 0.6], [2, 0.5], [2.5, 0.4], [3, 0.3], [4, 0.2], [5, 0.1], [6, 0.1]]) {
        for (const [k, c] of over) {
          if (c < frac * mine.size) break;
          if (used.has(k)) continue;
          if (tri(k) >= min / div) { out.push(k); used.add(k); if (out.length >= want) break; }
        }
        if (out.length >= want) break;
      }
      for (const [k] of over) { if (out.length >= want) break; if (!used.has(k)) { out.push(k); used.add(k); } }
    } else out.push(...over.map(([k]) => k));
    return [img, ...out];
  }

  // ms: ba, the window BAs (localBA) in all, solve their bundleAdjust, baCalls their count; focal, init: calibrateFocal,
  // initialize outside their window BAs, focalBA, initBA within them; windows: each window BA's bundleAdjust profile
  // with its ms; route: the window BAs solved on the GPU and on ba.js (a fallback counted on the GPU); fallback: why
  // each GPU solve redone by ba.js was (illReason, 'GPU error')
  resetTime() {
    this.time = { ba: 0, solve: 0, baCalls: 0, focal: 0, init: 0, focalBA: 0, initBA: 0, windows: [], route: { gpu: 0, cpu: 0 }, fallback: [] };
  }

  // Why the GPU's solve of a window fails, from its conditioning (cond), or '': f32 lost it (a result not finite), a direction
  // the window leaves free (a pivot under baIllPivot, the gauge fixed), or its RMSE up (baIllRmse)
  illReason(cond) {
    if (!cond) return '';
    if (!Number.isFinite(cond.rmse)) return 'not finite';
    if (!cond.gaugeFree && cond.minPivot < this.o.baIllPivot) return 'pivot';
    if (this.o.baIllRmse && cond.rmse > cond.rmse0) return 'RMSE up';
    return '';
  }

  // The intrinsics a window BA refines (refine, refineUnobserved)
  refineTerms() {
    const { refine, refineUnobserved } = this.o;
    return this.stats.focalSource === 'calibrated' ? refine : refine.filter(t => refineUnobserved.includes(t));
  }

  // Whether a window BA is worth the GPU solver: baGpuMinObs observations (fewer, the round trips make it slower than
  // ba.js: in Chrome on an M-series GPU, e2e/ba_gpu_bench.e2e.js, 743 observations 1.32x ba.js's time, 3231 0.81x)
  gpuFriendly(prob) { return prob.obsCam.length >= this.o.baGpuMinObs; }

  // The GPU solver (baGpu, or the realm's), made once; null where there is none (its windows then on ba.js, said once)
  async gpuSolver() {
    if (this._gpu === undefined) {
      this._gpu = await watchdog(() => Promise.resolve(this.o.baGpu ?? GpuBundleAdjuster.shared()), this.o.baGpuTimeoutMs).catch((e) => {
        this.o.log(`no GPU solver (${e?.message ?? e}): the window BAs on ba.js`);
        return null;
      });
    }
    return this._gpu;
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
    let removed = 0;
    for (const pid of pids) {
      const P = this.points[pid];
      if (!P) continue;
      for (const [img, kp] of [...P.obs]) {
        if (this.reprojErr(img, kp, P.X) > this.o.maxReproj) { P.obs.delete(img); this.kpPoint[img][kp] = -1; removed++; }
      }
      if (P.obs.size < 2 || (this.o.filterMinAngle && !this.wideEnough(P))) {
        for (const [img, kp] of P.obs) this.kpPoint[img][kp] = -1;
        removed += P.obs.size;
        this.points[pid] = null;
      }
    }
    return removed;
  }

  // triangulateNew on the registered of `frames`, then their points filtered: triangulateNew creates and extends at
  // triMax, wider than maxReproj, which only a BA's filter applies otherwise. Returns the points made
  triangulateAndFilter(frames) {
    let made = 0;
    const pids = new Set();
    for (const k of frames) if (this.poses[k]) made += this.triangulateNew(k);
    for (const k of frames) if (this.poses[k]) for (const pid of this.kpPoint[k]) if (pid >= 0) pids.add(pid);
    this.filter(pids);
    return made;
  }

  // Whether some two of P's views are minAngleDeg apart (COLMAP's HasPoint3DSufficientTriangulationAngle)
  wideEnough(P) {
    const min = this.o.minAngleDeg * Math.PI / 180, C = [...P.obs.keys()].map(img => poseCenter(this.poses[img]));
    for (let a = 0; a < C.length; a++) for (let b = a + 1; b < C.length; b++) if (triangulationAngle(C[a], C[b], P.X) >= min) return true;
    return false;
  }

  // resume: go on from a checkpoint's state (setState): the focal as it was, the segment it was growing grown on, then
  // the next ones
  async run({ resume = false } = {}) {
    // key's time outside its window BAs, key + 'BA' within them
    const timed = async (key, fn) => {
      const t = performance.now(), ba = this.time.ba, r = await fn(), inBA = this.time.ba - ba;
      this.time[key] += performance.now() - t - inBA;
      this.time[`${key}BA`] += inBA;
      return r;
    };
    this.lastCheckpoint = performance.now();
    const ended = (seg) => {
      let size = 0;
      const frames = [];
      for (let k = 0; k < this.n; k++) if (this.segment[k] === seg) { frames.push(k); if (this.poses[k]) size++; }
      this.stats.segments.push({ id: seg, size, first: frames[0], last: frames.at(-1) });
      this.o.log(`segment ${seg}: ${size} frames (${frames[0]}..${frames.at(-1)})`);
      this.trace('segment', { segment: seg, size, first: frames[0], last: frames.at(-1) });
    };
    const growing = this.nextSegmentId - 1;
    if (!resume) await timed('focal', () => this.calibrateFocal());
    else if (growing >= 0 && !this.stats.segments.some(s => s.id === growing)) {
      this.o.log(`mapping resumed: ${this.stats.registered}/${this.n} registered, segment ${growing} growing`);
      await this.grow(growing);
      ended(growing);
    }
    while (await timed('init', () => this.initialize())) {
      const seg = this.nextSegmentId - 1;
      this.stats.registered += 2;
      await this.grow(seg);
      ended(seg);
    }
    if (this.o.regrowSegments) await this.regrowSegments();
    return this.model();
  }

  // Photo sets: which segment grows first decides what can join it (a small segment can hold
  // frames through which the others would have joined). So each smaller segment, the larger first, is taken apart and its frames (and the
  // unregistered ones) offered to the largest segment; kept when that loses no frame, undone otherwise.
  async regrowSegments() {
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
        const added = await this.grow(big, free);
        if (added >= size) {
          this.o.log(`segment ${seg} (${size} frames) regrown into segment ${big}: ${added} frames added`);
          merged = true;
          break;
        }
        this.restore(snap);
      }
    }
    // and the largest segment, now complete, tries the frames still unregistered once more: a frame
    // registered by a regrow that was undone is not lost
    const big = this.largestSegment(), free = new Set([...this.segment.keys()].filter(k => !this.poses[k]));
    if (big >= 0 && free.size) {
      const added = await this.grow(big, free);
      if (added) this.o.log(`segment ${big}: ${added} unregistered frames added after the regrow`);
    }
  }

  // Register frames into segment `seg`, the next one chosen by nextImage, each followed by triangulation and the local
  // refinement (intrinsics free while few frames are registered in all), then a global BA when the segment has grown.
  // candidates: Set of frame indices (default: every frame). intrinsics false: the cameras held (a repair's grow: a
  // window's BA would refit the camera every frame shares). Returns the number registered.
  async grow(seg, candidates = null, { intrinsics = true } = {}) {
    const failed = new Map(), relFailed = new Map();  // per frame: PnP failures, registerRelative failures
    let added = 0;
    for (;;) {
      let best = -1, bestCount = 0;
      for (let k = 0; k < this.n; k++) {
        if (this.poses[k] || (failed.get(k) ?? 0) > 1 || (candidates && !candidates.has(k))) continue;
        const corr = this.correspondences(k, seg);  // a full rescan: cheap next to the BAs
        if (corr.size < this.o.minInliers) continue;
        const c = this.o.nextImage === 'pyramid' ? this.pyramidScore(k, corr) : corr.size;
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
      this.trace('register', { img: best, segment: seg });
      this.o.progress('register', registered, this.n);
      this.triangulateNew(best);
      const refineIntrinsics = intrinsics && registered >= 6 && registered <= (this.o.refineIntrinsicsUntil ?? Infinity);
      for (let it = 0; it < this.o.localRefinements; it++) {  // COLMAP's IterativeLocalRefinement: the robust loss on the first only
        (this.stats.localIts ??= [0, 0])[it ? 1 : 0]++;  // the first passes, the repeats
        const r = this.o.localBA === 'covisible' ? await this.localBA(best, { refineIntrinsics, covisible: true, iterations: this.o.localBAIterations, loss: it ? 'trivial' : null })
                                                 : await this.localBA(best, { refineIntrinsics });
        if (!this.o.completeMerge) break;
        const pids = new Set();  // COLMAP's, on the points of the local bundle's frames, then those filtered
        for (const k of this.localBundle(best)) for (const pid of this.kpPoint[k]) if (pid >= 0 && this.points[pid]) pids.add(pid);
        let nObs = 0;
        for (const pid of pids) nObs += this.points[pid].obs.size;
        const [a, m, mo] = this.completeAndMerge(pids);
        this.stats.cm = [(this.stats.cm?.[0] ?? 0) + a, (this.stats.cm?.[1] ?? 0) + m];
        const changed = (a + mo + this.filter(pids) + (r?.filtered ?? 0)) / Math.max(1, nObs);
        if (changed < 0.001) break;
      }
      if (this.o.globalBARatio) {
        let size = 0;
        for (let k = 0; k < this.n; k++) if (this.poses[k] && this.segment[k] === seg) size++;
        const last = (this.stats.globalAt ??= {})[seg] ?? 0;  // the segment's size at its last global BA (a retry's grow goes on from it)
        if (size >= this.o.globalBAMinFrames && size >= last * this.o.globalBARatio) { await this.globalBA(seg, best, refineIntrinsics); this.stats.globalAt[seg] = size; }
      }
      if (registered % 50 === 0) this.o.log(`registered ${registered}/${this.n}, ${this.numPoints()} points, f ${this.camera.params[0].toFixed(1)}`);
      // the mapping's grows only: a regrow's trial (candidates, regrowSegments) is undone from a snapshot a checkpoint
      // does not hold
      if (this.o.checkpoint && !candidates && performance.now() - (this.lastCheckpoint ?? 0) >= this.o.checkpointMs) {
        await this.o.checkpoint(this);
        this.lastCheckpoint = performance.now();
      }
    }
    return added;
  }

  // px for creating and continuing a point (triMaxReproj, else triMaxDeg at the focal, else maxReproj)
  triMax() {
    return this.o.triMaxReproj ?? (this.o.triMaxDeg ? this.camera.params[0] * Math.tan(this.o.triMaxDeg * Math.PI / 180) : this.o.maxReproj);
  }

  // Per keypoint of img, the frames it is matched in (cached, reset by addPairs)
  matchCount(img) {
    const c = (this._matchCount ??= [])[img];
    if (c) return c;
    const n = new Uint16Array(this.images[img].keypoints.length >> 1);
    for (const e of this.adj[img]) for (const a of e.a) n[a]++;
    return (this._matchCount[img] = n);
  }

  // Pairs usable for registration, track extension and triangulation (the constructor's, then
  // e.g. loop pairs once segments are merged); a pair already known is skipped
  addPairs(pairs) {
    this._matchCount = null;
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
  // its own estimate).
  async refine({ sweeps = 2, refineIntrinsics = true, extra = 5 } = {}) {
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
          await this.localBA(core[0], { frames: [...core, ...this.partners(core, { extra, minGap: W })], iterations: 10 });
          this.o.progress('refine', ++done, total);
        }
      }
      if (refineIntrinsics && sw < sweeps) await this.fitIntrinsics({ minCameraFrames: this.o.minCameraFramesSweeps });
    }
  }

  // The intrinsics fitted to the whole map at once: a window BA (localBA) whose free frames are up to
  // maxFrames per camera of the largest segment, evenly spaced in time, with the intrinsics; the other
  // frames observe as fixed (the sweep after it moves them to the new intrinsics). Refined window by
  // window instead, the lens would be whatever the last windows made of it.
  async fitIntrinsics({ minCameraFrames = this.o.minCameraFrames, maxFrames = 60, iterations = 10 } = {}) {
    const seg = this.largestSegment();
    const byCam = this.cameras.map(() => []);
    for (let k = 0; k < this.n; k++) if (this.poses[k] && this.segment[k] === seg) byCam[this.camOf[k]].push(k);
    const frames = byCam.flatMap(ks => ks.length <= maxFrames ? ks : Array.from({ length: maxFrames }, (_, i) => ks[Math.round(i * (ks.length - 1) / (maxFrames - 1))]));
    if (!frames.length) return null;
    return this.localBA(frames[0], { frames, refineIntrinsics: true, iterations, minCameraFrames });
  }

  // The frames outside `core`, more than minGap from its first frame, sharing the most points with
  // it: at most `extra`, each with at least minShared points (loop partners of a window). Measured
  // from the first frame only, so the frames just after a refine() window qualify too: overlap with
  // the next window.
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
  // export.js datasetFromMapper's. In TRACK_PARTS parts of the keyframes (minutes on a long video): each says how far
  // it is (progress), records its new points for the theatre (a 'tracks' event) and lets the worker post them
  async extendTracks() {
    const step = Math.max(1, Math.ceil(this.n / TRACK_PARTS)), seen = new Set();
    for (let first = 0; first < this.n; first += step) {
      const last = Math.min(this.n, first + step) - 1;
      let created = 0;
      for (let k = first; k <= last; k++) if (this.poses[k]) created += this.triangulateNew(k, seen);
      this.o.progress('retriangulate', last + 1, this.n);
      this.trace('tracks', { first, last, created });
      await new Promise(r => setTimeout(r, 0));
    }
    this.o.progress('filterPoints', 0, 0);
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
      this.byFlow[k] = 0;
      this.segment[k] = -1;
    }
    this.trace('unregister', { frames: [...frames] });
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
    this.byFlow = Uint8Array.from(at(this.byFlow, () => 0));
    this.kpPoint = oldIndex.map((o, k) => (o >= 0 ? this.kpPoint[o] : new Int32Array(images[k].keypoints.length / 2).fill(-1)));
    this.adj = oldIndex.map((o) => (o >= 0 ? this.adj[o].filter(e => toNew[e.other] >= 0).map(e => ({ ...e, other: toNew[e.other] })) : []));
    for (const P of this.points) {
      if (P) P.obs = new Map([...P.obs].map(([img, kp]) => [toNew[img], kp]));
    }
    this.images = images;
    this.n = n;
    this.camOf = this.camerasOf(images);
  }

  // Move a whole segment by the Sim(3) G (old world -> new world)
  transformSegment(seg, G) {
    const Ginv = inverse(G);
    const moved = new Set();
    for (let k = 0; k < this.n; k++) {
      if (this.segment[k] !== seg) continue;
      this.poses[k] = toPose(compose(fromPose(this.poses[k]), Ginv));
      for (const pid of this.kpPoint[k]) {
        if (pid < 0 || moved.has(pid) || !this.points[pid]) continue;
        moved.add(pid);
        this.points[pid].X = apply(G, this.points[pid].X);
      }
    }
  }

  // An unregistered frame placed at `pose` in segment `seg`, without points
  place(k, pose, seg) {
    this.poses[k] = pose;
    this.segment[k] = seg;
    this.stats.registered++;
  }

  // Frames of segment `from` into segment `to`
  relabelSegment(from, to) {
    for (let k = 0; k < this.n; k++) if (this.segment[k] === from) this.segment[k] = to;
  }

  // Merge track pa into pb when they never share an image
  fuseTracks(pa, pb) {
    if (pa === pb) return false;
    const A = this.points[pa], B = this.points[pb];
    if (!A || !B) return false;
    for (const img of A.obs.keys()) if (B.obs.has(img)) return false;
    for (const [img, kp] of A.obs) { B.obs.set(img, kp); this.kpPoint[img][kp] = pb; }
    this.points[pa] = null;
    return true;
  }

  // Copy of the mutable state, for undo (copySnapshot)
  snapshot() { return copySnapshot(this); }

  // Back to a snapshot (which is consumed: its arrays become the live state)
  restore(s) { Object.assign(this, s); }

  // Compact state for saving (project.js): what the pairs and the points do not imply
  toState() {
    const live = this.points.filter(Boolean);
    const obs = live.flatMap(P => [...P.obs].flat());
    return {
      json: { camera: this.camera, cameras: this.cameras, nextSegmentId: this.nextSegmentId, stats: this.stats, o: { ...this.o, log: undefined, progress: undefined, trace: undefined, checkpoint: undefined },
              merged: Array.from(this.merged), focalKnown: Array.from(this.focalKnown), byFlow: [...this.byFlow.keys()].filter(k => this.byFlow[k]) },
      poses: Float64Array.from(this.poses.flatMap(p => p ? [...p.R, ...p.t] : new Array(12).fill(NaN))),
      segment: this.segment.slice(),
      pointX: Float64Array.from(live.flatMap(P => P.X)),
      pointObsCount: Uint32Array.from(live.map(P => P.obs.size)),
      pointObs: Int32Array.from(obs),
    };
  }

  // Inverse of toState(), on a mapper built from the same keyframes and pairs (all of them)
  setState({ json, poses, segment, pointX, pointObsCount, pointObs }) {
    Object.assign(this, { cameras: json.cameras ?? [json.camera], nextSegmentId: json.nextSegmentId ?? json.numSegments, stats: json.stats, o: { ...this.o, ...json.o, log: this.o.log, progress: this.o.progress, trace: this.o.trace } });
    if (json.merged) {  // absent when saved before scanFocal: every camera its own, every focal known
      this.merged = Int32Array.from(json.merged);
      this.focalKnown = Uint8Array.from(json.focalKnown);
      this.camOf = this.camerasOf(this.images);
    }
    this.poses = Array.from({ length: this.n }, (_, k) => Number.isNaN(poses[12 * k]) ? null
      : { R: Array.from(poses.subarray(12 * k, 12 * k + 9)), t: Array.from(poses.subarray(12 * k + 9, 12 * k + 12)) });
    this.segment = Int32Array.from(segment);
    this.byFlow = new Uint8Array(this.n);
    for (const k of json.byFlow ?? []) this.byFlow[k] = 1;
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

