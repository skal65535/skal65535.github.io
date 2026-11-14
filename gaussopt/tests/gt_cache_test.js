// gt_cache_test.js — The trainer's ground-truth textures (trainer.js ensureGtTexture, _keepGt): at most gtCacheBytes of
// them, the first kept pinned but for GT_CACHE_MIN going round the others least recently used first.
import { GaussOptTrainer, GT_CACHE_MIN } from '../src/trainer.js';
import { assert } from './util.js';

// a trainer with its ground-truth cache only (fake textures, loaded at once), of n textures' bytes at 960 x 540
function cacheOf(n) {
  const tr = Object.create(GaussOptTrainer.prototype), destroyed = [];
  Object.assign(tr, { gtTextures: new Map(), gtPinned: new Set(), gtLoading: new Map(), trainW: 960, trainH: 540, loads: 0 });
  tr.gtCacheBytes = n * 4 * 960 * 540;
  tr._loadGtTexture = async (i) => { tr.loads++; const tex = { i, destroy: () => destroyed.push(i) }; tr._keepGt(i, tex); return tex; };
  return { tr, destroyed };
}

Deno.test('gt cache: the first pinned, the others round least recently used, a hit made the most recent', async () => {
  const n = GT_CACHE_MIN + 6, { tr, destroyed } = cacheOf(n);  // 6 pinned: 0..5
  for (let i = 0; i < n; i++) await tr.ensureGtTexture(i);
  assert(tr.gtTextures.size === n && !destroyed.length && tr.gtCacheFull(), `${tr.gtTextures.size} kept, ${destroyed.length} dropped`);
  assert([...tr.gtPinned].join() === '0,1,2,3,4,5', `pinned ${[...tr.gtPinned]}`);
  await tr.ensureGtTexture(6);  // a hit: kept, not loaded again, now the most recent
  assert(tr.loads === n, `${tr.loads} loads`);
  await tr.ensureGtTexture(n);
  await tr.ensureGtTexture(n + 1);
  assert(tr.gtTextures.size === n, `${tr.gtTextures.size} kept`);
  assert(destroyed.join() === '7,8' && tr.gtTextures.has(6) && tr.gtTextures.has(0), `dropped ${destroyed}`);
  const t = await tr.ensureGtTexture(7);  // dropped: loaded again
  assert(t.i === 7 && tr.loads === n + 3, `${tr.loads} loads`);
});

// the training's order: the views in turn. Least recently used alone: every visit a load; pinned, the pinned hit
Deno.test('gt cache: views in turn, the pinned ones hit every round', async () => {
  const views = 300, cap = 100, { tr } = cacheOf(cap);
  for (let round = 0; round < 3; round++) for (let v = 0; v < views; v++) await tr.ensureGtTexture(v);
  const pinned = cap - GT_CACHE_MIN;
  assert(tr.loads === views + 2 * (views - pinned), `${tr.loads} loads in 3 rounds of ${views} (${pinned} pinned)`);
  assert(tr.gtTextures.size === cap, `${tr.gtTextures.size} kept`);
});

Deno.test('gt cache: GT_CACHE_MIN kept under a smaller budget', async () => {
  const { tr, destroyed } = cacheOf(1);
  for (let i = 0; i < GT_CACHE_MIN + 10; i++) await tr.ensureGtTexture(i);
  assert(tr.gtTextures.size === GT_CACHE_MIN && destroyed.length === 10 && tr.gtCacheFull() && !tr.gtPinned.size,
         `${tr.gtTextures.size} kept, ${destroyed.length} dropped, ${tr.gtPinned.size} pinned`);
});
