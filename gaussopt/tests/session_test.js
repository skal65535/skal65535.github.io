// Repair actions on synthetic scenes (session.js, DESIGN-sfm §6.3): retry, repair and redo of a damaged
// stretch back to the `truth` thresholds; undo restores the exact state.
import { assert } from './util.js';
import { accuracy, session, damage, posesOf } from './session_util.js';

Deno.test('session: quality flags a damaged stretch; retry repairs it; undo restores the exact state', async () => {
  const { scene, s } = await session();
  assert(accuracy(s, scene).fails.length === 0, 'clean reconstruction');
  const clean = s.quality();
  assert(clean.counts.red === 0, `clean: ${JSON.stringify(clean.counts)}`);
  const range = s.images.slice(20, 25).map(im => im.name);
  damage(s, range);
  const hurt = s.quality();
  assert(range.every(n => hurt.frames.find(f => f.name === n).score === 'red'), 'damaged frames are red');
  const before = posesOf(s);
  const res = await s.retry(range);
  const acc = accuracy(s, scene);
  assert(res.after.red === 0 && res.registered === range.length, `retry: ${JSON.stringify(res.before)} -> ${JSON.stringify(res.after)}, ${res.registered} registered`);
  assert(acc.fails.length === 0, `after retry: ${acc.tag}: ${acc.fails.join(', ')}`);
  assert(s.undo() === 'retry' && posesOf(s) === before, 'undo restores the damaged poses');
  assert(s.undo() === null, 'nothing more to undo');
});

Deno.test('session: repair retries the damaged stretch, keeps it, and undoes as one action', async () => {
  const { scene, s } = await session();
  const range = s.images.slice(20, 25).map(im => im.name);
  damage(s, range);
  const before = posesOf(s);
  const res = await s.repair();
  const acc = accuracy(s, scene);
  assert(res.kept >= 1 && res.after.red === 0 && res.quality.counts.red === 0, `repair: ${res.kept}/${res.tried} kept, ${JSON.stringify(res.quality.counts)}`);
  assert(range.every(n => res.frames.includes(n)), 'the repaired frames are reported');
  assert(acc.fails.length === 0, `after repair: ${acc.tag}: ${acc.fails.join(', ')}`);
  assert(s.ops.length === 1 && s.undo() === 'repair' && posesOf(s) === before, 'one undo restores the damaged poses');
});

Deno.test('session: redo maps a damaged stretch as a new segment and merges it back', async () => {
  const { scene, s } = await session();
  const range = s.images.slice(15, 27).map(im => im.name);
  damage(s, range, 4);
  const res = await s.redo(range);
  const acc = accuracy(s, scene);
  assert(res.merges >= 1 && new Set(s.model().images.map(im => im.segment)).size === 1, `redo: ${res.merges} merges`);
  assert(acc.fails.length === 0 && res.after.red === 0, `after redo: ${acc.tag}: ${acc.fails.join(', ')}; ${JSON.stringify(res.after)}`);
});
