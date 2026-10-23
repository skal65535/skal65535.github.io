// Repair actions that change the keyframes (session.js, DESIGN-sfm §6.3): drop, refine, insert.
import { assert } from './util.js';
import { accuracy, session } from './session_util.js';

Deno.test('session: drop removes keyframes and keeps the rest accurate; refine is harmless', async () => {
  const { scene, s } = await session();
  const gone = s.images.slice(30, 33).map(im => im.name);
  const res = await s.drop(gone);
  assert(s.images.length === 47 && s.model().images.length === 47 && !s.images.some(im => gone.includes(im.name)), `drop: ${s.images.length} keyframes`);
  assert(s.mapper.points.every(P => !P || [...P.obs.keys()].every(k => k >= 0 && k < 47)), 'observations re-indexed');
  let acc = accuracy(s, scene);
  assert(acc.fails.length === 0 && res.after.red === 0, `after drop: ${acc.tag}: ${acc.fails.join(', ')}`);
  await s.refine(s.images.slice(10, 20).map(im => im.name), { intrinsics: true });
  acc = accuracy(s, scene);
  assert(acc.fails.length === 0, `after refine: ${acc.tag}`);
  assert(s.undo() === 'refine' && s.undo() === 'drop' && s.images.length === 50 && s.model().images.length === 50, 'undo both');
});

Deno.test('session: insert registers keyframes added between existing ones (densify)', async () => {
  const keep = [...Array(50).keys()].filter(k => k < 20 || k > 30 || k % 2 === 0);
  const { scene, s } = await session({}, keep);
  const extra = [21, 23, 25, 27, 29].map(k => scene.input.images[k]);
  const res = await s.insert(extra);
  const acc = accuracy(s, scene);
  assert(s.images.length === 50 && res.registered === 5, `insert: ${s.images.length} keyframes, ${res.registered} registered`);
  assert(s.images.every((im, k) => k === 0 || im.name > s.images[k - 1].name), 'time order kept');
  assert(acc.fails.length === 0 && res.after.red === 0, `after insert: ${acc.tag}: ${acc.fails.join(', ')}`);
});
