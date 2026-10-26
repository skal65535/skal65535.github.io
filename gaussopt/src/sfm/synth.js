// synth.js — Seeded synthetic scenes with exact ground truth, for the SfM tests (DESIGN-sfm §8):
// a drone-like keyframe path over terrain + boxes + a far ring, projected through a distorted
// camera into keypoints (detector budget, pixel noise) and pairwise matches (sequential window +
// revisit loops, optional outliers).
//
// Output mirrors a COLMAP dump (dump.js), so solvers take either:
//   input: { camera (prior, COLMAP default focal), images: [{ name, keypoints }], pairs: [{ i, j, type, matches }] }
//   truth: { camera, images: [{ name, R, t, pointIds }], points, model }  (model feeds compare.js)
// World is Z-up, units are meters; cameras follow camera.js (OpenCV axes, world -> camera poses).
"use strict";

import { makeRng, cross, normalize } from '../math.js';
import { makeCamera, defaultCamera, project, inFieldOfView, poseFromCenter, meanFocal, withFocal } from './camera.js';

export function gaussian(rng) {
  const u1 = Math.max(1e-12, rng()), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const DEFAULTS = {
  seed: 1,
  width: 1920, height: 1080,
  model: 'OPENCV', params: [1230, 1230, 960, 540, 0.005, -0.002, 0.001, 0.0002],  // jardin5's COLMAP fit
  path: 'orbit',        // 'orbit' (revisits after one turn) | 'line' (no revisit)
  numFrames: 120,
  turns: 1.25,          // orbit: > 1 gives loops
  radius: 12, altitude: 7, pitchDeg: 30,
  length: 60,           // line: flight length
  spin: null,           // { at: 0..1, frames, deg }: yaw in place (no parallax) inserted along the path
  numGround: 12000, numBoxes: 12, pointsPerBox: 600, numFar: 3000,
  maxFeatures: 4096, noisePx: 0.5, maxDepth: 120,
  overlap: 10,          // sequential pairs: |i - j| <= overlap
  minLoopMatches: 40,   // revisit pairs (|i - j| > overlap) kept when sharing at least this many points
  outlierFrac: 0,       // fraction of each pair's matches replaced by random keypoint pairs
  cut: null,            // 0..1: no sequential pair crosses that frame (only revisits link both sides)
  fps: 30, keyframeEvery: 6,  // timestamps: keyframe k at k * keyframeEvery / fps seconds
  // several lenses (DESIGN-sfm §13 P8): params of each, frame k's lens cameraOf(k), and the focal
  // priors the input cameras start from (else COLMAP's default focal); null: one camera, `params`
  cameras: null, cameraOf: null, priorFocals: null,
};

function makePoints(o, rng) {
  const pts = [];
  const span = o.path === 'line' ? o.length / 2 + 15 : o.radius * 1.6;
  const ground = (x, y) => 0.4 * Math.sin(0.21 * x) * Math.cos(0.17 * y) + 0.15 * Math.sin(0.9 * x + 0.4 * y);
  for (let i = 0; i < o.numGround; i++) {
    const x = (2 * rng() - 1) * span, y = (2 * rng() - 1) * (o.path === 'line' ? 18 : span);
    pts.push([x, y, ground(x, y)]);
  }
  for (let b = 0; b < o.numBoxes; b++) {
    const cx = (2 * rng() - 1) * span * 0.7, cy = (2 * rng() - 1) * span * 0.5;
    const sx = 1 + 2 * rng(), sy = 1 + 2 * rng(), h = 1 + 4 * rng(), z0 = ground(cx, cy);
    const area = [sx * h, sx * h, sy * h, sy * h, sx * sy];  // faces: -y, +y, -x, +x, top
    const total = area.reduce((s, a) => s + a, 0);
    for (let i = 0; i < o.pointsPerBox; i++) {
      let r = rng() * total, f = 0;
      while (r > area[f]) r -= area[f++];
      const u = rng() - 0.5, v = rng();
      const p = [
        [cx + u * sx, cy - sy / 2, z0 + v * h], [cx + u * sx, cy + sy / 2, z0 + v * h],
        [cx - sx / 2, cy + u * sy, z0 + v * h], [cx + sx / 2, cy + u * sy, z0 + v * h],
        [cx + u * sx, cy + (v - 0.5) * sy, z0 + h],
      ][f];
      pts.push(p);
    }
  }
  for (let i = 0; i < o.numFar; i++) {
    const a = 2 * Math.PI * rng(), d = 40 + 40 * rng();
    pts.push([d * Math.cos(a), d * Math.sin(a), 15 * rng() * rng()]);
  }
  return pts;
}

// Rotation (world -> camera) looking along `forward` with world up +Z
function lookRotation(forward) {
  const f = normalize(forward);
  const r = normalize(cross(f, [0, 0, 1]));
  const d = cross(f, r);
  return [...r, ...d, ...f];
}

function makePath(o) {
  const frames = [];
  const pitch = o.pitchDeg * Math.PI / 180;
  for (let k = 0; k < o.numFrames; k++) {
    const u = k / Math.max(1, o.numFrames - 1);
    let C, yaw;
    if (o.path === 'orbit') {
      const a = 2 * Math.PI * o.turns * u;
      const r = o.radius * (1 + 0.08 * Math.sin(3 * a));
      C = [r * Math.cos(a), r * Math.sin(a), o.altitude + 0.8 * Math.sin(2 * a)];
      yaw = Math.atan2(-C[1], -C[0]) + 0.15 * Math.sin(5 * a);  // toward the center, with some sway
    } else {
      const x = (u - 0.5) * o.length;
      C = [x, 3 * Math.sin(0.08 * x), o.altitude + 0.5 * Math.sin(0.2 * x)];
      yaw = Math.atan(0.24 * Math.cos(0.08 * x));
    }
    frames.push({ C, yaw });
  }
  if (o.spin) {
    const at = Math.round(o.spin.at * (frames.length - 1));
    const { C, yaw } = frames[at];
    const spin = Array.from({ length: o.spin.frames }, (_, k) =>
      ({ C: [...C], yaw: yaw + (o.spin.deg * Math.PI / 180) * (k + 1) / o.spin.frames }));
    const back = spin.at(-1).yaw - yaw;  // keep turning: the rest of the path is yawed by the spin
    frames.splice(at + 1, 0, ...spin, ...frames.splice(at + 1).map(f => ({ ...f, yaw: f.yaw + back })));
  }
  return frames.map(({ C, yaw }) => {
    const fwd = [Math.cos(yaw) * Math.cos(pitch), Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch)];
    return poseFromCenter(lookRotation(fwd), C);
  });
}

export function makeSyntheticScene(options = {}) {
  const o = { ...DEFAULTS, ...options };
  const rng = makeRng(o.seed);
  const lenses = (o.cameras ?? [o.params]).map(p => makeCamera(o.model, o.width, o.height, p));
  const lensOf = (k) => (o.cameras && o.cameraOf ? o.cameraOf(k) : 0);
  const camera = lenses[0];
  const points = makePoints(o, rng);
  const strength = points.map(() => rng());  // detector response: the same points win in every frame
  const poses = makePath(o);

  const images = poses.map((pose, k) => {
    const camera = lenses[lensOf(k)];
    const visible = [];
    for (let p = 0; p < points.length; p++) {
      if (!inFieldOfView(camera, pose, points[p])) continue;
      const [u, v, z] = project(camera, pose, points[p]);
      if (z < 0.5 || z > o.maxDepth || u < 4 || v < 4 || u > o.width - 4 || v > o.height - 4) continue;
      visible.push({ p, u, v, score: strength[p] + 0.1 * rng() });
    }
    visible.sort((a, b) => b.score - a.score);
    const kept = visible.slice(0, o.maxFeatures);
    for (let i = kept.length - 1; i > 0; i--) {  // shuffle: keypoint index says nothing about the point
      const j = Math.floor(rng() * (i + 1));
      [kept[i], kept[j]] = [kept[j], kept[i]];
    }
    const keypoints = new Float32Array(2 * kept.length);
    const pointIds = new Int32Array(kept.length);
    kept.forEach(({ p, u, v }, i) => {
      keypoints[2 * i] = u + o.noisePx * gaussian(rng);
      keypoints[2 * i + 1] = v + o.noisePx * gaussian(rng);
      pointIds[i] = p;
    });
    const ms = Math.round(1000 * k * o.keyframeEvery / o.fps);
    return { name: `t${String(ms).padStart(7, '0')}.jpg`, keypoints, pointIds, R: pose.R, t: pose.t };
  });

  const pairs = [];
  const cutAt = o.cut === null ? null : Math.round(o.cut * (images.length - 1));
  const index = images.map(im => new Map(Array.from(im.pointIds, (p, i) => [p, i])));
  for (let i = 0; i < images.length; i++) {
    for (let j = i + 1; j < images.length; j++) {
      const m = [];
      for (const [p, a] of index[i]) { const b = index[j].get(p); if (b !== undefined) m.push(a, b); }
      const type = j - i <= o.overlap ? 'seq' : 'loop';
      if (m.length === 0 || (type === 'loop' && m.length / 2 < o.minLoopMatches)) continue;
      if (cutAt !== null && type === 'seq' && i < cutAt && j >= cutAt) continue;
      const numOut = Math.round(o.outlierFrac * m.length / 2);
      for (let k = 0; k < numOut; k++) {
        const slot = Math.floor(rng() * (m.length / 2));
        m[2 * slot] = Math.floor(rng() * images[i].pointIds.length);
        m[2 * slot + 1] = Math.floor(rng() * images[j].pointIds.length);
      }
      pairs.push({ i, j, type, matches: Uint32Array.from(m) });
    }
  }

  const truthImages = images.map(({ name, R, t, pointIds }) => ({ name, R, t, pointIds }));
  const prior = (k) => (o.priorFocals ? withFocal(defaultCamera(o.model, o.width, o.height), o.priorFocals[k]) : defaultCamera(o.model, o.width, o.height));
  return {
    options: o,
    input: {
      camera: prior(0),
      ...(o.cameras ? { cameras: lenses.map((_, k) => prior(k)) } : {}),
      images: images.map(({ name, keypoints }, k) => ({ name, keypoints, ...(o.cameras ? { camera: lensOf(k) } : {}) })),
      pairs,
    },
    truth: {
      camera, ...(o.cameras ? { cameras: lenses } : {}), images: truthImages, points,
      model: { images: truthImages, focal: meanFocal(camera), numPoints: points.length },
    },
  };
}

