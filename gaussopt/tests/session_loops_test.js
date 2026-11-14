// Repair actions on loops and the focal (session.js, DESIGN-sfm §6.3): force loop, set focal (reject loop:
// session_replay_test.js).
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SfmSession } from '../src/sfm/session.js';
import { geometricCandidates } from '../src/sfm/loopdetect.js';
import { assert, ORBIT, rng } from './util.js';
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

Deno.test('session: geometric loop candidates again after the merges, across the segments they joined', async () => {
  // a cut, the sequential pairs and one revisit pair: it merges the two sides; the revisit's other pairs, proposed
  // before the merge only within a segment, are found after it
  const scene = makeSyntheticScene({ ...ORBIT, cut: 0.4 });
  const names = scene.input.images.map(im => im.name), cutAt = Math.round(0.4 * names.length);
  const across = (p) => (p.i < cutAt) !== (p.j < cutAt);
  const revisits = scene.input.pairs.filter(p => p.type === 'loop' && across(p)).sort((a, b) => b.matches.length - a.matches.length);
  const input = { ...scene.input, pairs: [...scene.input.pairs.filter(p => p.type === 'seq'), revisits[0]] };
  const table = tablePairSource(scene.input), calls = [];
  const proposeLoops = async (mapper, before) => {
    calls.push(before ? 'after' : 'first');
    const geo = geometricCandidates(mapper, { minGap: 10, perFrame: 8, before, rng: rng(5) });
    const found = await table(geo.map(([i, j]) => [names[i], names[j], 'loop']));
    const ix = new Map(names.map((n, k) => [n, k]));
    return found.map(p => ({ i: ix.get(p.a), j: ix.get(p.b), type: p.type, matches: p.matches }));
  };
  // mapped on the sequential pairs (the loop pair would join the sides while mapping): two segments, merged by closeLoops;
  // every frame pair retriangulated (reMinRatio null): the pairs proposed after the merge hang on it (1 point: 139 -> 134)
  const s = new SfmSession(input, { pairSource: table, mapping: { pairTypes: ['seq'], reMinRatio: null } });
  const res = await s.reconstruct({ proposeLoops });
  const ix = new Map(names.map((n, k) => [n, k]));
  const edges = s.loops.map(l => ({ i: ix.get(l.a), j: ix.get(l.b) })).filter(across);
  assert(res.stats.numSegments === 1 && res.stats.loops.merges === 1, `${res.stats.numSegments} segments, ${res.stats.loops.merges} merges`);
  assert(res.stats.loops.proposed > 5 && edges.length > 5, `${res.stats.loops.proposed} pairs after the merge, ${edges.length} loop edges across the cut`);
  assert(calls.join() === 'first,after', `proposals ${calls.join()}`);
  // `before`: only the pairs then apart (at a short gap, pairs within a side overlap too)
  const side = Int32Array.from(names, (_, k) => (k < cutAt ? 0 : 1));
  const all = geometricCandidates(s.mapper, { minGap: 2, rng: rng(5) }), apart = geometricCandidates(s.mapper, { minGap: 2, before: side, rng: rng(5) });
  assert(all.some(([i, j]) => side[i] === side[j]) && apart.length > 5 && apart.every(([i, j]) => side[i] !== side[j]),
         `${all.length} pairs, ${apart.length} with before, ${apart.filter(([i, j]) => side[i] === side[j]).length} of them within a side`);
  assert(s.pairs.filter(p => across({ i: ix.get(p.a), j: ix.get(p.b) })).length > revisits.length / 4, 'the pairs kept in the session');
  const acc = accuracy(s, scene);
  assert(acc.fails.length === 0, `${acc.tag}: ${acc.fails.join(', ')}`);
});
