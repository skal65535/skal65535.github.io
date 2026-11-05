// frontend.js matchPairs: GPU descriptor lifetime (one upload per frame, all released at the end,
// none reused from an earlier call), with a matcher that only counts.
import { matchPairs, trimFeatures, linkStrength, adaptFeatures } from '../src/sfm/frontend.js';
import { makeRng } from '../src/math.js';
import { assert } from './util.js';

function countingMatcher() {
  const cache = new Map(), stats = { uploads: 0 };
  return {
    stats, cache,
    upload(desc, key) {
      if (cache.has(key)) return cache.get(key);
      stats.uploads++;
      const e = { desc, n: desc.length / 128 };
      cache.set(key, e);
      return e;
    },
    release(key) { cache.delete(key); },
    clear() { cache.clear(); },
    async matchBatch(entries) { return entries.map(() => new Uint32Array(0)); },
  };
}

Deno.test('matchPairs: a window plus loops uploads each frame once and releases all', async () => {
  const n = 60, W = 10, todo = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j <= Math.min(n - 1, i + W); j++) todo.push([i, j, 'seq']);
  todo.push([5, 40, 'loop'], [2, 50, 'loop'], [30, 55, 'loop']);
  const features = Array.from({ length: n }, () => ({ keypoints: new Float32Array(0), descriptors: new Uint8Array(128) }));
  const m = countingMatcher();
  await matchPairs(features, m, todo, { rng: Math.random });
  assert(m.stats.uploads === n, `${m.stats.uploads} uploads for ${n} frames`);
  assert(m.cache.size === 0, `still on the GPU: ${[...m.cache.keys()]}`);
});

Deno.test('matchPairs: descriptors left by an interrupted call are not reused', async () => {
  const features = Array.from({ length: 3 }, () => ({ keypoints: new Float32Array(0), descriptors: new Uint8Array(128) }));
  const m = countingMatcher(), stale = { desc: null, n: -1 };
  m.cache.set(1, stale);
  let seen = null;
  m.matchBatch = async (entries) => { seen = entries; return entries.map(() => new Uint32Array(0)); };
  await matchPairs(features, m, [[0, 1, 'seq']], { rng: Math.random });
  assert(seen[0][1] !== stale && seen[0][1].n === 1, 'stale descriptors of frame 1 used');
});

// Ten views of 300 points, a third of each pair's matches wrong: the pairs verified on the worker
// pool are those verified inline, inliers and F alike (each pair's RANSAC draws from its own seed)
Deno.test('matchPairs: the verification pool gives the inline results', async () => {
  const rng = makeRng(7), n = 10, N = 300;
  const points = Array.from({ length: N }, () => [4 * rng() - 2, 3 * rng() - 1.5, 4 + 3 * rng()]);
  const features = Array.from({ length: n }, (_, c) => {
    const kp = new Float32Array(4 * N), yaw = 0.04 * c;
    points.forEach(([x, y, z], k) => {
      const X = Math.cos(yaw) * x - Math.sin(yaw) * z + 0.15 * c, Z = Math.sin(yaw) * x + Math.cos(yaw) * z;
      kp[4 * k] = 500 + 800 * X / Z; kp[4 * k + 1] = 400 + 800 * y / Z;
    });
    return { keypoints: kp, descriptors: new Uint8Array(128 * N) };
  });
  const todo = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) todo.push([i, j, 'seq']);
  // per pair: k -> k, a third of them to a random point; the same raw matches for both runs
  const fixed = todo.map(() => {
    const r = new Uint32Array(2 * N);
    for (let k = 0; k < N; k++) { r[2 * k] = k; r[2 * k + 1] = rng() < 0.33 ? Math.floor(rng() * N) : k; }
    return r;
  });
  const matcher = (calls) => ({ ...countingMatcher(), async matchBatch(entries) { return entries.map(() => fixed[calls.k++].slice()); } });
  const inline = await matchPairs(features, matcher({ k: 0 }), todo, { rng: makeRng(1), workers: 0 });
  const pooled = await matchPairs(features, matcher({ k: 0 }), todo, { rng: makeRng(1), workers: 4 });
  assert(pooled.stats.workers === 4 && inline.stats.workers === 0, `workers ${pooled.stats.workers}, ${inline.stats.workers}`);
  assert(inline.pairs.length === todo.length, `${inline.pairs.length} of ${todo.length} verified inline`);
  const key = (p) => `${p.i},${p.j}:${p.matches.join()}:${p.F.join()}`;
  inline.pairs.forEach((p, k) => assert(key(p) === key(pooled.pairs[k]), `pair ${p.i},${p.j} differs on the pool`));
});

// trimFeatures: the strongest maxFeatures of response >= peak (responses sorted, the strongest first)
Deno.test('trimFeatures: by the cap, by the peak, nothing to trim', () => {
  const n = 6, f = { keypoints: Float32Array.from({ length: 4 * n }, (_, i) => i), descriptors: Uint8Array.from({ length: 128 * n }, (_, i) => i >> 7),
                     responses: Float32Array.from([0.9, 0.5, 0.2, 0.05, 0.01, 0.001]) };
  const t = (m, p) => { const r = trimFeatures(f, m, p); return [r.keypoints.length / 4, r.descriptors.length / 128, r.responses.length, r.descriptors.at(-1)]; };
  assert(t(2, 0) + '' === '2,2,2,1' && t(10, 0.05) + '' === '4,4,4,3' && t(3, 0.05) + '' === '3,3,3,2', `${t(2, 0)} ${t(10, 0.05)} ${t(3, 0.05)}`);
  assert(trimFeatures(f, 10, 0) === f, 'nothing trimmed: the same features');
});

// linkStrength: per boundary the inliers of the sequential pairs crossing it, within the window
Deno.test('linkStrength: the sequential pairs crossing each boundary', () => {
  const p = (i, j, n, type = 'seq') => ({ i, j, type, matches: new Uint32Array(2 * n) });
  const s = linkStrength([p(0, 1, 10), p(0, 2, 5), p(1, 3, 7), p(0, 3, 100, 'loop'), p(0, 4, 50)], 5, 3);
  assert([...s] + '' === '15,12,7,0', `${[...s]}`);
});

// adaptFeatures: 30 views of 600 points, the first 300 of each keyframe its trimmed features (the others under the peak);
// the matcher matches a point to itself, but across boundary 10 | 11 only the points of the dense features. Boundary 10
// is weak (no pair across it verified), keyframes 8..13 get their dense features, their pairs matched again; the rest
// trimmed, their pairs as matched first
Deno.test('adaptFeatures: a weak link found, the keyframes about it dense, their pairs matched again', async () => {
  const rng = makeRng(3), n = 30, N = 600, half = 300;
  const points = Array.from({ length: N }, () => [6 * rng() - 3, 4 * rng() - 2, 6 + 4 * rng()]);
  const dense = Array.from({ length: n }, (_, c) => {
    const kp = new Float32Array(4 * N), de = new Uint8Array(128 * N);
    points.forEach(([x, y, z], k) => {
      const X = x + 0.05 * c, Z = z;
      kp[4 * k] = 960 + 1000 * X / Z; kp[4 * k + 1] = 540 + 1000 * y / Z;
      de[128 * k] = k & 255; de[128 * k + 1] = k >> 8;
    });
    return { name: `f${c}`, keypoints: kp, descriptors: de, responses: Float32Array.from({ length: N }, (_, k) => (k < half ? 1 : 0.001)) };
  });
  const matcher = countingMatcher();
  matcher.matchBatch = async (entries) => entries.map(([A, B]) => {
    const id = (e, q) => e.desc[128 * q] | (e.desc[128 * q + 1] << 8), across = A.key <= 10 && B.key > 10;
    const inB = new Map(Array.from({ length: B.n }, (_, q) => [id(B, q), q])), m = [];
    for (let q = 0; q < A.n; q++) { const k = id(A, q), r = inB.get(k); if (r !== undefined && (!across || k >= half)) m.push(q, r); }
    return Uint32Array.from(m);
  });
  const up = matcher.upload;
  matcher.upload = (desc, key) => Object.assign(up(desc, key), { key });
  const a = await adaptFeatures(dense, matcher, { rng: makeRng(1), maxFeatures: 4096, peak: 0.02 / 3, window: 10, minLink: 0.1, workers: 0 });
  assert(a.weak + '' === '10' && a.denseFrames + '' === '8,9,10,11,12,13', `weak ${a.weak}, dense ${a.denseFrames}`);
  assert(a.feats.every((f, k) => f.keypoints.length / 4 === (k >= 8 && k <= 13 ? N : half)), a.feats.map(f => f.keypoints.length / 4).join(' '));
  const pair = (i, j) => (a.seq.pairs.find(p => p.i === i && p.j === j)?.matches.length ?? 0) / 2;
  assert(pair(9, 12) === half && pair(8, 9) === N && pair(2, 3) === half && pair(5, 12) === 0, `${pair(9, 12)} ${pair(8, 9)} ${pair(2, 3)} ${pair(5, 12)}`);
  assert(linkStrength(a.seq.pairs, n, 10)[10] > 0 && a.seq.pairs.length === new Set(a.seq.pairs.map(p => `${p.i},${p.j}`)).size, 'each pair once, the link restored');
});
