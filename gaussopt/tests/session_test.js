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

Deno.test('session: repair holds no snapshot of a kept retry while it goes on', async () => {
  const { s } = await session();
  damage(s, s.images.slice(5, 8).map(im => im.name));
  damage(s, s.images.slice(20, 23).map(im => im.name));
  const retry = s.retry.bind(s), held = [];
  s.retry = (names) => (held.push(s.ops.length), retry(names));
  const res = await s.repair();
  assert(res.kept >= 2, `repair: ${res.kept}/${res.tried} kept`);
  assert(held.every(n => n === 0), `snapshots held at each retry: ${held}`);
  assert(s.ops.length === 1 && s.undo() === 'repair', 'one undo');
});

// the intrinsics refined in every local BA while mapping (a video's default: refineIntrinsicsUntil null)
const VIDEO_INTRINSICS = { mapping: { refineIntrinsicsUntil: null, refine: ['f', 'k1'] } };
const cameraOf = (s) => JSON.stringify(s.mapper.cameras.map(c => c.params));

Deno.test('session: retry and redo hold the cameras (one window refits no camera every frame shares)', async () => {
  for (const [op, at, deg] of [['retry', [20, 25], 2], ['redo', [15, 27], 4]]) {
    const { s } = await session({}, null, VIDEO_INTRINSICS);
    const range = s.images.slice(...at).map(im => im.name);
    damage(s, range, deg);
    const before = cameraOf(s);
    await s[op](range);
    assert(cameraOf(s) === before, `${op}: camera ${before} -> ${cameraOf(s)}`);
  }
});

Deno.test('session: retry leaves no observation of its frames over maxReproj', async () => {
  const { s } = await session({ outlierFrac: 0.3 }, null, { mapping: { triMaxDeg: 2, maxReproj: 4 } });
  const range = s.images.slice(20, 25).map(im => im.name);
  damage(s, range);
  await s.retry(range);
  const m = s.mapper, ix = s._indices(range);
  let over = 0, n = 0;
  for (const k of ix) for (const pid of m.kpPoint[k]) {
    if (pid < 0) continue;
    for (const [img, kp] of m.points[pid].obs) { n++; if (m.reprojErr(img, kp, m.points[pid].X) > m.o.maxReproj) over++; }
  }
  assert(n > 0 && over === 0, `${over} of ${n} observations over ${m.o.maxReproj} px`);
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
