// Repair actions that change the keyframes (session.js, DESIGN-sfm §6.3): drop, refine, insert (also
// of photos into a photo set, §13.3 P4).
import { assert, ORBIT } from './util.js';
import { accuracy, session, tablePairSource } from './session_util.js';
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SfmSession } from '../src/sfm/session.js';
import { PHOTO_MAPPING } from '../src/sfm/defaults.js';

Deno.test('session: drop removes keyframes and keeps the rest accurate; refine is harmless', async () => {
  const { scene, s } = await session();
  const gone = s.images.slice(18, 21).map(im => im.name);
  const res = await s.drop(gone);
  assert(s.images.length === 27 && s.model().images.length === 27 && !s.images.some(im => gone.includes(im.name)), `drop: ${s.images.length} keyframes`);
  assert(s.mapper.points.every(P => !P || [...P.obs.keys()].every(k => k >= 0 && k < 27)), 'observations re-indexed');
  let acc = accuracy(s, scene);
  assert(acc.fails.length === 0 && res.after.red === 0, `after drop: ${acc.tag}: ${acc.fails.join(', ')}`);
  await s.refine(s.images.slice(5, 15).map(im => im.name), { intrinsics: false });  // as the panel's Refine (sfm_panel.js)
  acc = accuracy(s, scene);
  assert(acc.fails.length === 0, `after refine: ${acc.tag}`);
  assert(s.undo() === 'refine' && s.undo() === 'drop' && s.images.length === 30 && s.model().images.length === 30, 'undo both');
});

Deno.test('session: insert registers keyframes added between existing ones (densify)', async () => {
  const keep = [...Array(30).keys()].filter(k => k < 10 || k > 20 || k % 2 === 0);
  const { scene, s } = await session({}, keep);
  const extra = [11, 13, 15, 17, 19].map(k => scene.input.images[k]);
  const res = await s.insert(extra);
  const acc = accuracy(s, scene);
  assert(s.images.length === 30 && res.registered === 5, `insert: ${s.images.length} keyframes, ${res.registered} registered`);
  assert(s.images.every((im, k) => k === 0 || im.name > s.images[k - 1].name), 'time order kept');
  assert(acc.fails.length === 0 && res.after.red === 0, `after insert: ${acc.tag}: ${acc.fails.join(', ')}`);
});

// Add photos (DESIGN-sfm §13.3 P4): a set in two pieces, frames 0-5 and 7-11, no pair between
// them; the photo taken between them (6) is inserted, matched against every photo, and with
// PHOTO_MAPPING the smaller piece is regrown into the larger: one segment
Deno.test('session: a photo inserted between two pieces of a photo set joins them', async () => {
  const scene = makeSyntheticScene({ ...ORBIT, numFrames: 12, turns: 0.5, overlap: 11 });
  const pairs = scene.input.pairs.filter(p => p.i === 6 || p.j === 6 || (p.i < 6) === (p.j < 6));
  const keep = [...Array(12).keys()].filter(k => k !== 6), at = new Map(keep.map((k, n) => [k, n]));
  const input = { ...scene.input, images: keep.map(k => scene.input.images[k]),
                  pairs: pairs.filter(p => at.has(p.i) && at.has(p.j)).map(p => ({ ...p, i: at.get(p.i), j: at.get(p.j) })) };
  const s = new SfmSession(input, { pairSource: tablePairSource({ ...scene.input, pairs }), mapping: PHOTO_MAPPING, window: 11 });
  await s.reconstruct();
  const segments = () => new Set(s.mapper.segment.filter(x => x >= 0)).size;
  assert(segments() === 2, `two pieces first: ${[...s.mapper.segment]}`);
  s.window = 11;  // every pair (the worker spans the set)
  const res = await s.insert([scene.input.images[6]]);
  const acc = accuracy(s, scene);
  assert(res.registered === 1 && segments() === 1 && s.model().images.length === 12, `after insert: ${segments()} segments, ${acc.tag}`);
  assert(acc.fails.length === 0, `after insert: ${acc.tag}: ${acc.fails.join(', ')}`);
  assert(s.undo() === 'insert' && s.images.length === 11 && segments() === 2, 'undo');
});
