// session.js rejectLoops replays from the sequential map: an undone replay must leave that map
// as it was, so the next replay gives what it gives on a fresh session; the loop is gone, the map accurate.
import { assert } from './util.js';
import { session, key, accuracy } from './session_util.js';

const poses = (s) => JSON.stringify(s.mapper.poses);

Deno.test('session: reject loop X, undo, reject loop Y = reject loop Y', async () => {
  const { s } = await session();
  const { scene, s: t } = await session();
  const [x, y] = [...s.loops].sort((a, b) => b.inliers - a.inliers);
  assert(x && y, `${s.loops.length} loops`);
  await s.rejectLoops([[x.a, x.b]]);
  assert(s.undo() === 'rejectLoops', 'undo');
  await s.rejectLoops([[y.a, y.b]]);
  await t.rejectLoops([[y.a, y.b]]);
  assert(poses(s) === poses(t), 'the replay started from another map');
  assert(!t.loops.some(l => key(l.a, l.b) === key(y.a, y.b)) && t.rejected.has(key(y.a, y.b)), 'loop gone');
  assert(accuracy(t, scene).fails.length === 0, `after reject loop: ${accuracy(t, scene).tag}`);
});
