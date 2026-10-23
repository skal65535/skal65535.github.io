import { realRoots, fundamental8, sampsonError, homographyDLT, transferError, ransacFundamental,
         essentialFromFundamental, poseFromEssential, triangulate, p3p, ransacPnP, pnpLinear } from '../src/sfm/geometry.js';
import { makeCamera, project, normalizedFromImage, poseFromCenter, poseCenter } from '../src/sfm/camera.js';
import { rotationAngle, mul3, transpose3, mulVec3, expSO3 } from '../src/sfm/linalg.js';
import { gaussian } from '../src/sfm/synth.js';
import { assert, assertClose, rng, randomRotation } from './util.js';

const CAM = makeCamera('PINHOLE', 1920, 1080, [1230, 1230, 960, 540]);
const K = [1230, 1230, 960, 540];

// Two cameras looking at a random cloud; returns pixels, rays, points and the poses
function twoViews(r, { n = 200, baseline = 1, depth = 10, planar = false, noise = 0 } = {}) {
  const pose1 = poseFromCenter([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);
  const R2 = expSO3([0.03 * gaussian(r), 0.03 * gaussian(r), 0.03 * gaussian(r)]);
  const yaw = 0.1 * (r() - 0.5);
  const Rz = [Math.cos(yaw), 0, Math.sin(yaw), 0, 1, 0, -Math.sin(yaw), 0, Math.cos(yaw)];
  const pose2 = poseFromCenter(mul3(Rz, R2), [baseline, 0.2 * baseline * (r() - 0.5), 0.1 * baseline]);
  const pts = [], x1 = [], x2 = [];
  while (pts.length < n) {
    const X = [(r() - 0.5) * depth, (r() - 0.5) * depth * 0.5, planar ? depth : depth * (0.7 + 0.6 * r())];
    const a = project(CAM, pose1, X), b = project(CAM, pose2, X);
    if (a[2] <= 0 || b[2] <= 0 || a[0] < 0 || a[0] > 1920 || b[0] < 0 || b[0] > 1920 || a[1] < 0 || a[1] > 1080 || b[1] < 0 || b[1] > 1080) continue;
    pts.push(X);
    x1.push([a[0] + noise * gaussian(r), a[1] + noise * gaussian(r)]);
    x2.push([b[0] + noise * gaussian(r), b[1] + noise * gaussian(r)]);
  }
  return { pose1, pose2, pts, x1, x2 };
}

Deno.test('realRoots finds the real roots of quartics', () => {
  const r = rng(20);
  for (let trial = 0; trial < 200; trial++) {
    const roots = [0, 1, 2, 3].map(() => 10 * (r() - 0.5)).sort((a, b) => a - b);
    let poly = [1];
    for (const x of roots) poly = [...poly, 0].map((c, i) => c - x * (i > 0 ? poly[i - 1] : 0));  // * (t - x)
    const found = realRoots(poly).sort((a, b) => a - b);
    if (roots.some((x, i) => i && x - roots[i - 1] < 1e-3)) continue;  // near-double roots: skip
    assertClose(found, roots, 1e-7, `roots of ${poly}`);
  }
  assertClose(realRoots([1, 0, 1]), [], 0, 'no real roots');
  assertClose(realRoots([1, 0, 0, 0, -16]).sort((a, b) => a - b), [-2, 2], 1e-9, 'x^4 = 16');
});

Deno.test('fundamental8 and the homography DLT fit exact data', () => {
  const r = rng(21);
  const { x1, x2 } = twoViews(r);
  const F = fundamental8(x1, x2);
  for (let i = 0; i < x1.length; i++) assert(sampsonError(F, x1[i], x2[i]) < 1e-12, 'epipolar');
  const p = twoViews(r, { planar: true });
  const H = homographyDLT(p.x1, p.x2);
  for (let i = 0; i < p.x1.length; i++) assert(transferError(H, p.x1[i], p.x2[i]) < 1e-12, 'transfer');
});

Deno.test('RANSAC rejects outliers and recovers the relative pose from E', () => {
  const r = rng(22);
  const { pose2, x1, x2 } = twoViews(r, { n: 300, noise: 0.5 });
  const truth = Array(300).fill(1);
  const outliers = new Set();  // 30%, distinct
  while (outliers.size < 90) outliers.add(Math.floor(r() * 300));
  for (const k of outliers) {
    x2[k] = [1920 * r(), 1080 * r()];
    truth[k] = 0;
  }
  const res = ransacFundamental(x1, x2, { rng: r, threshold: 2 });
  let agree = 0;
  for (let i = 0; i < 300; i++) agree += res.inliers[i] === truth[i];
  assert(agree / 300 > 0.97, `inlier classification ${agree}/300`);
  const E = essentialFromFundamental(res.model, K);
  const idx = [...res.inliers.keys()].filter(i => res.inliers[i]);
  const rays1 = idx.map(i => normalizedFromImage(CAM, ...x1[i])), rays2 = idx.map(i => normalizedFromImage(CAM, ...x2[i]));
  const rel = poseFromEssential(E, rays1, rays2);
  assert(rel.numFront > 0.95 * idx.length, 'cheirality');
  assert(rotationAngle(rel.R, pose2.R) < 0.01, `rotation ${rotationAngle(rel.R, pose2.R)}`);
  const tTrue = pose2.t.map(x => x / Math.hypot(...pose2.t));
  assertClose(rel.t, tTrue, 0.03, 'translation direction');
});

Deno.test('triangulate recovers points from several views', () => {
  const r = rng(23);
  const poses = [0, 1, 2, 3].map(k => poseFromCenter(expSO3([0.05 * k, -0.03 * k, 0.02]), [k, 0.3 * k, 0]));
  for (let trial = 0; trial < 50; trial++) {
    const X = [r() * 4, r() * 2, 8 + r() * 4];
    const rays = poses.map(p => { const c = mulVec3(p.R, X).map((v, k) => v + p.t[k]); return [c[0] / c[2], c[1] / c[2]]; });
    assertClose(triangulate(poses, rays), X, 1e-9, 'point');
  }
});

Deno.test('p3p returns the true pose among its solutions', () => {
  const r = rng(24);
  let fails = 0;
  for (let trial = 0; trial < 500; trial++) {
    const pose = poseFromCenter(randomRotation(r), [10 * (r() - 0.5), 10 * (r() - 0.5), 10 * (r() - 0.5)]);
    const C = poseCenter(pose), Rt = transpose3(pose.R);
    const b = [], X = [];
    for (let k = 0; k < 3; k++) {
      const d = [r() - 0.5, r() - 0.5, 1].map((x, i) => (i < 2 ? x * 1.2 : x));
      const n = Math.hypot(...d), bc = d.map(x => x / n), depth = 2 + 20 * r();
      b.push(bc);
      X.push(mulVec3(Rt, bc.map(x => x * depth)).map((x, i) => x + C[i]));
    }
    const sols = p3p(b, X);
    const ok = sols.some(s => rotationAngle(s.R, pose.R) < 1e-6 && Math.hypot(...s.t.map((x, i) => x - pose.t[i])) < 1e-6 * (1 + Math.hypot(...pose.t)));
    fails += !ok;
  }
  assert(fails <= 2, `${fails}/500 P3P cases missed the true pose`);  // near-degenerate configurations
});

Deno.test('ransacPnP locates a camera with outliers; pnpLinear on clean data', () => {
  const r = rng(25);
  const pose = poseFromCenter(expSO3([0.1, -0.2, 0.05]), [1, 2, -3]);
  const pts = [], px = [], rays = [];
  while (pts.length < 300) {
    const X = [10 * (r() - 0.5), 6 * (r() - 0.5), 10 + 10 * r()].map((x, i) => x + [1, 2, -3][i]);
    const [u, v, z] = project(CAM, pose, X);
    if (z <= 0 || u < 0 || u > 1920 || v < 0 || v > 1080) continue;
    pts.push(X);
    const outlier = pts.length % 4 === 0;
    const p = outlier ? [1920 * r(), 1080 * r()] : [u + 0.5 * gaussian(r), v + 0.5 * gaussian(r)];
    px.push(p);
    rays.push(normalizedFromImage(CAM, ...p));
  }
  const res = ransacPnP(rays, px, pts, CAM, { rng: r, threshold: 4 });
  assert(res.numInliers >= 220 && res.numInliers <= 230, `${res.numInliers} inliers (225 expected)`);
  assert(rotationAngle(res.model.R, pose.R) < 2e-3, 'rotation');
  assert(Math.hypot(...poseCenter(res.model).map((x, i) => x - [1, 2, -3][i])) < 0.05, 'center');
  const clean = [...Array(300).keys()].filter(i => i % 4 !== 3);
  const lin = pnpLinear(clean.map(i => rays[i]), clean.map(i => pts[i]));
  assert(rotationAngle(lin.R, pose.R) < 5e-3, 'linear rotation');
});
