// .gosfm containers (project.js) and saved sessions (session.js toState / fromState): a restored
// session has the same model and scores, and repairs it the same way as the original.
import { packProject, unpackProject, mergeProjects, projectPart } from '../src/sfm/project.js';
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SfmSession } from '../src/sfm/session.js';
import { assert, ORBIT } from './util.js';
import { tablePairSource } from './session_util.js';

Deno.test('project container: JSON, typed arrays and blobs round trip, merged and split', async () => {
  const parts = { a: Float64Array.of(1.5, NaN, -2), b: Uint8Array.of(1, 2, 3), c: new Blob(['jpeg bytes'], { type: 'image/jpeg' }), d: Int32Array.of(-7) };
  const one = packProject({ hello: 'world', n: [1, 2] }, parts);
  const p = await unpackProject(one);
  assert(p.json.hello === 'world' && p.keys.join() === 'a,b,c,d', 'header');
  const a = await p.get('a'), b = await p.get('b'), c = await p.get('c'), d = await p.get('d');
  assert(a instanceof Float64Array && a[0] === 1.5 && Number.isNaN(a[1]) && a[2] === -2, 'Float64Array');
  assert(b instanceof Uint8Array && b.join() === '1,2,3' && d[0] === -7, 'integer arrays');
  assert(c.type === 'image/jpeg' && await c.text() === 'jpeg bytes', 'blob');
  const merged = await mergeProjects({ features: one, model: packProject({ v: 2 }, { x: Uint32Array.of(9) }) });
  const f = await projectPart(merged, 'features'), m = await projectPart(merged, 'model');
  assert(f.json.hello === 'world' && (await f.get('a'))[2] === -2 && await (await f.get('c')).text() === 'jpeg bytes', 'features part');
  assert(m.json.v === 2 && (await m.get('x'))[0] === 9 && !(await projectPart(merged, 'other')), 'model part');
  let threw = false;
  try { await unpackProject(new Blob(['not a project'])); } catch { threw = true; }
  assert(threw, 'rejects other files');
});

Deno.test('saved session: restored model, scores and repairs match the original', async () => {
  const scene = makeSyntheticScene({ ...ORBIT, numFrames: 40 });
  const names = scene.input.images.map(im => im.name), pairSource = tablePairSource(scene.input);
  const s = new SfmSession(scene.input, { pairSource });
  await s.reconstruct();
  await s.drop([names[5]]);  // a saved state after a repair

  const { json, arrays } = s.toState();
  const p = await unpackProject(packProject(json, arrays));
  const loaded = {};
  for (const k of p.keys) loaded[k] = await p.get(k);
  const keypoints = new Map(scene.input.images.map(im => [im.name, im.keypoints]));
  const r = SfmSession.fromState({ json: p.json, arrays: loaded }, (n) => keypoints.get(n), { pairSource });

  const poses = (x) => JSON.stringify(x.model().images.map(im => [im.name, im.R, im.t]));
  assert(poses(r) === poses(s), 'same poses');
  assert(r.model().numPoints === s.model().numPoints && r.images.length === 39, `points ${r.model().numPoints} vs ${s.model().numPoints}`);
  assert(JSON.stringify(r.quality().counts) === JSON.stringify(s.quality().counts), 'same scores');
  // the same next action on both: the RNG is seeded by the action's position in the log, which
  // restarts on a restored session, so align the logs first
  s.ops = [];
  const range = names.slice(20, 24);
  await s.retry(range);
  await r.retry(range);
  assert(poses(r) === poses(s), 'same result of a retry');
});
