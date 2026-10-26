// Repair actions on loops and the focal (session.js, DESIGN-sfm §6.3): force loop, set focal (reject loop:
// session_replay_test.js).
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SfmSession } from '../src/sfm/session.js';
import { assert, ORBIT } from './util.js';
import { tablePairSource, accuracy, session } from './session_util.js';

Deno.test('session: force loop joins two segments that loop detection missed', async () => {
  // a cut, and only the sequential pairs: two segments
  const scene = makeSyntheticScene({ ...ORBIT, cut: 0.4 });
  const input = { ...scene.input, pairs: scene.input.pairs.filter(p => p.type === 'seq') };
  const s = new SfmSession(input, { pairSource: tablePairSource(scene.input) });
  await s.reconstruct();
  assert(new Set(s.model().images.map(im => im.segment)).size === 2, 'two segments without loops');
  const names = scene.input.images.map(im => im.name);
  const revisit = scene.input.pairs.filter(p => p.type === 'loop').sort((a, b) => b.matches.length - a.matches.length)[0];
  await s.forceLoop(names[revisit.i], names[revisit.j]);
  assert(new Set(s.model().images.map(im => im.segment)).size === 1, 'one segment after the forced loop');
  const acc = accuracy(s, scene);
  assert(acc.fails.length === 0, `after force loop: ${acc.tag}: ${acc.fails.join(', ')}`);
});

Deno.test('session: set focal (fixed) then re-estimate it', async () => {
  const { scene, s } = await session();
  const f = scene.truth.camera.params[0];
  await s.setFocal(1.05 * f, { fix: true });
  assert(Math.abs(s.mapper.camera.params[0] / (1.05 * f) - 1) < 1e-9, 'fixed focal kept through the refinement');
  await s.setFocal('auto', { fix: false });
  const acc = accuracy(s, scene);
  assert(acc.fails.length === 0, `after re-estimating: ${acc.tag}, focal ${(100 * acc.m.focalErr).toFixed(2)}%: ${acc.fails.join(', ')}`);
  assert(s.undo() === 'setFocal' && s.mapping.focal === 1.05 * f && s.mapper.camera.params[0] === 1.05 * f, 'undo restores the fixed focal');
});
