// frontend.js matchPairs: GPU descriptor lifetime (one upload per frame, all released at the end,
// none reused from an earlier call), with a matcher that only counts.
import { matchPairs } from '../src/sfm/frontend.js';
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
