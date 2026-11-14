// Mapping checkpoints (mapper.js checkpoint, session.js reconstruct checkpoint / resume): a session's state saved during
// the mapping, a session made from it goes on with the mapping from there, not from the start; the hook is the
// mapping's only.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SfmSession } from '../src/sfm/session.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { assert } from './util.js';
import { tablePairSource, accuracy, damage } from './session_util.js';
import { ORBIT } from './util.js';

// regrowSegments' trial grows (candidates given) are undone from a snapshot of their own: no checkpoint during them
Deno.test('checkpoint: none during a grow of given candidates, one per registration in the mapping\'s', async () => {
  const scene = makeSyntheticScene({ ...ORBIT, numFrames: 30 });
  let calls = 0;
  const mapper = new SequentialMapper(scene.input, { checkpointMs: 0, checkpoint: async () => { calls++; } });
  await mapper.run();
  const mapped = calls, frames = [24, 25, 26, 27, 28, 29];
  assert(mapped > 0, `${mapped} checkpoints in the mapping`);
  mapper.unregister(frames);
  const added = await mapper.grow(mapper.segment[0], new Set(frames));
  assert(added === frames.length && calls === mapped, `a grow of ${frames.length} candidates: ${added} added, ${calls - mapped} checkpoints`);
  mapper.unregister(frames);
  await mapper.grow(mapper.segment[0]);
  assert(calls - mapped === frames.length, `a grow without candidates: ${calls - mapped} checkpoints`);
});

Deno.test('checkpoint: a session from a state saved mid-mapping goes on from it, to the same accuracy', async () => {
  const scene = makeSyntheticScene({ ...ORBIT, numFrames: 30 });
  const n = scene.input.images.length;
  let saved = null, calls = 0;
  const s = new SfmSession(scene.input, { pairSource: tablePairSource(scene.input) });
  // every registration (checkpointMs 0); the state at 15 registered kept, as a project would read it back
  await s.reconstruct({ checkpointMs: 0, checkpoint: async (session) => {
    calls++;
    if (!saved && session.mapper.stats.registered >= 15) saved = structuredClone(session.toState());
  } });
  assert(saved && calls >= n - 15, `${calls} checkpoints, one at ${saved?.json.mapper.stats.registered} registered`);
  const done = saved.json.mapper.stats.registered, posed = Array.from(saved.arrays.poses).filter((_, k) => k % 12 === 0 && !Number.isNaN(saved.arrays.poses[k])).length;
  assert(done >= 15 && done < n && posed === done, `the checkpoint: ${done} registered, ${posed} poses`);

  const byName = new Map(scene.input.images.map(im => [im.name, im.keypoints]));
  const registers = [];
  const r = SfmSession.fromState(saved, (name) => byName.get(name), { pairSource: tablePairSource(scene.input),
    progress: (stage, d) => { if (stage === 'register') registers.push(d); } });
  let more = 0;
  await r.reconstruct({ resume: true, checkpointMs: 0, checkpoint: async () => { more++; } });
  assert(registers[0] === done + 1, `resumed: the first registration is number ${registers[0]}, the checkpoint had ${done}`);
  assert(more > 0 && more <= n - done, `${more} checkpoints after the resume`);
  const acc = accuracy(r, scene);
  assert(acc.m.numEst === n && acc.fails.length === 0, `resumed: ${acc.tag}: ${acc.fails.join(', ')}`);

  // a repair grows the map too: no checkpoint then
  const before = more, range = r.images.slice(10, 14).map(im => im.name);
  damage(r, range);
  await r.retry(range);
  assert(more === before, `${more - before} checkpoints during a retry`);
});
