// GPU SIFT (sift.wgsl): invariance on synthetic texture, determinism, and (slow) matching
// performance on real frames against COLMAP's verified matches.
import { GpuSift } from '../src/sfm/sift.js';
import { GpuMatcher } from '../src/sfm/match.js';
import { ransacFundamental } from '../src/sfm/geometry.js';
import { loadDump, denoReader } from '../src/sfm/dump.js';
import { readPGM } from '../tools/pgm.js';
import { makeRng } from '../src/math.js';
import { assert, SLOW, env, hasGPU } from './util.js';


// Smooth random texture: random blobs of several sizes on a gradient
function texture(w, h, seed) {
  const r = makeRng(seed), f = new Float32Array(w * h);
  for (let b = 0; b < 600; b++) {
    const cx = r() * w, cy = r() * h, s = 2 + 12 * r() ** 2, a = r() - 0.5;
    const R = Math.ceil(3 * s);
    for (let y = Math.max(0, Math.floor(cy - R)); y < Math.min(h, cy + R); y++) {
      for (let x = Math.max(0, Math.floor(cx - R)); x < Math.min(w, cx + R); x++) {
        f[y * w + x] += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
      }
    }
  }
  return Uint8Array.from(f, (v, i) => Math.max(0, Math.min(255, Math.round(128 + 90 * v + 20 * (i % w) / w))));
}

// 90 degrees clockwise: pixel (c, r) -> (h - 1 - r, c) in an h x w image
const rotate90 = (img, w, h) => {
  const out = new Uint8Array(w * h);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) out[c * h + (h - 1 - r)] = img[r * w + c];
  return out;
};

Deno.test({ name: 'SIFT is deterministic and rotation invariant (90 deg, synthetic texture)', ignore: !hasGPU, fn: async () => {
  const w = 640, h = 480, img = texture(w, h, 7);
  const sift = await GpuSift.create();
  const a = await sift.extract(img, w, h), a2 = await sift.extract(img, w, h);
  assert(a.keypoints.length > 4 * 300, `${a.keypoints.length / 4} keypoints`);
  assert(a.keypoints.every((x, i) => x === a2.keypoints[i]) && a.descriptors.every((x, i) => x === a2.descriptors[i]), 'deterministic');
  const b = await sift.extract(rotate90(img, w, h), h, w);
  const matcher = await GpuMatcher.create(sift.device);
  const m = await matcher.match(a.descriptors, b.descriptors);
  let good = 0;
  for (let k = 0; k < m.length; k += 2) {
    const [xa, ya] = [a.keypoints[4 * m[k]], a.keypoints[4 * m[k] + 1]], [xb, yb] = [b.keypoints[4 * m[k + 1]], b.keypoints[4 * m[k + 1] + 1]];
    good += Math.hypot(xb - (h - ya), yb - xa) < 2;  // continuous coordinates: (x, y) -> (h - y, x)
  }
  assert(m.length / 2 > 150 && good / (m.length / 2) > 0.95, `${m.length / 2} matches, ${good} at the rotated position`);
} });

const PGM = env('SFM_PGM'), DUMP = env('SFM_DESC_DUMP');

// Same consecutive pairs: our features + matcher + F RANSAC against COLMAP's verified matches
Deno.test({ name: 'SIFT on jardin5 frames: verified matches vs COLMAP (SFM_PGM, SFM_DESC_DUMP)', ignore: !SLOW || !PGM || !DUMP || !hasGPU, fn: async () => {
  const dump = await loadDump(denoReader(DUMP));
  const sift = await GpuSift.create(), matcher = await GpuMatcher.create(sift.device), rng = makeRng(1);
  const feats = new Map();
  const frame = (k) => readPGM(`${PGM}/${dump.images[k].name.replace(/\.[^.]+$/, '.pgm')}`);
  const get = async (k) => {
    if (!feats.has(k)) {
      const img = await frame(k);
      feats.set(k, await sift.extract(img.data, img.width, img.height, { maxFeatures: 1e9 }));
    }
    return feats.get(k);
  };
  let ours = 0, theirs = 0;
  for (const p of dump.pairs.filter(p => p.j - p.i <= 3 && p.j < 16)) {
    const A = await get(p.i), B = await get(p.j);
    const m = await matcher.match(A.descriptors, B.descriptors);
    const x1 = [], x2 = [];
    for (let k = 0; k < m.length; k += 2) { x1.push([A.keypoints[4 * m[k]], A.keypoints[4 * m[k] + 1]]); x2.push([B.keypoints[4 * m[k + 1]], B.keypoints[4 * m[k + 1] + 1]]); }
    const F = ransacFundamental(x1, x2, { rng, threshold: 4 });
    ours += F ? F.numInliers : 0;
    theirs += p.matches.length / 2;
  }
  console.log(`  verified matches over the pairs COLMAP verified: ours ${ours}, COLMAP ${theirs} (${(ours / theirs).toFixed(2)}x)`);
  assert(ours >= 0.9 * theirs, 'at least 90% of COLMAP');
  // real frames have DoG response ties, which the GPU appends in a varying order
  for (const k of [...feats.keys()].slice(0, 8)) {
    const img = await frame(k);
    const again = await sift.extract(img.data, img.width, img.height, { maxFeatures: 1e9 }), first = feats.get(k);
    assert(again.keypoints.every((x, i) => x === first.keypoints[i]) && again.descriptors.every((x, i) => x === first.descriptors[i]),
           `frame ${k}: same keypoints in the same order on a re-run`);
  }
} });
