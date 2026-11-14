// Several videos of one place in a project (clips): the keyframes one after the other, each with its `clip`; no
// sequential pair across two videos (they share no time), the pairs between them from the loop candidates, which
// keep some for the other videos; the map of each joined by the loop stage's merge.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
import { SfmSession } from '../src/sfm/session.js';
import { matchFeatures, adaptFeatures, linkStrength, seqTodo } from '../src/sfm/frontend.js';
import { loopCandidates } from '../src/sfm/loopdetect.js';
import { assert, ORBIT } from './util.js';
import { tablePairSource, accuracy } from './session_util.js';
import { joinTelemetry, positionAt } from '../src/sfm/flight.js';

// two orbits of the same place, distinct: another radius, height, pitch and start
const CLIPS = { ...ORBIT, clips: [{ numFrames: 36, turns: 1 }, { numFrames: 36, turns: 1, radius: 16, altitude: 10, pitchDeg: 35, phase: 1.5 }] };

// a matcher keyed by frame index (the upload key); match(i, j): the pair's raw matches
function keyedMatcher(match) {
  const seen = [];
  return {
    seen,
    upload: (desc, key) => ({ key, n: desc.length / 128 }),
    release() {}, clear() {},
    async matchBatch(entries) { return entries.map(([a, b]) => { seen.push([a.key, b.key]); return match(a.key, b.key); }); },
  };
}

Deno.test('clips: no sequential pair across two videos; a loop candidate at the break matched', async () => {
  const clip = (k) => (k < 10 ? 0 : 1), n = 20;
  assert(seqTodo(n, 5, Array.from({ length: n }, (_, k) => clip(k))).every(([i, j]) => clip(i) === clip(j)), 'seqTodo across the break');
  const features = Array.from({ length: n }, (_, k) => ({ name: `f${k}`, keypoints: new Float32Array(0), descriptors: new Uint8Array(128), clip: clip(k) }));
  const m = keyedMatcher(() => new Uint32Array(0));
  await matchFeatures(features, m, { rng: Math.random, window: 5, loopCandidates: [[9, 10], [3, 15]], workers: 0 });
  const across = m.seen.filter(([i, j]) => clip(i) !== clip(j)).map(p => p.join('-'));
  assert(across.sort().join() === '3-15,9-10', `pairs matched across the videos: ${across}`);
});

// two videos of the same views, none of their matches across (as two flights' frames next to each other in the list):
// the break is no weak link, its keyframes get no dense features
Deno.test('clips: the break between two videos is not a weak link', async () => {
  const n = 20, N = 200, clip = (k) => (k < 10 ? 0 : 1);
  const dense = Array.from({ length: n }, (_, c) => {
    const kp = new Float32Array(4 * N), de = new Uint8Array(128 * N);
    for (let k = 0; k < N; k++) { kp[4 * k] = 10 + k + c; kp[4 * k + 1] = 10 + (7 * k) % 100; de[128 * k] = k; }
    return { name: `f${c}`, keypoints: kp, descriptors: de, responses: new Float32Array(N).fill(1), clip: clip(c) };
  });
  const m = keyedMatcher((i, j) => (clip(i) === clip(j) ? Uint32Array.from({ length: 2 * N }, (_, q) => q >> 1) : new Uint32Array(0)));
  const a = await adaptFeatures(dense, m, { rng: Math.random, maxFeatures: 4096, peak: 0, window: 3, minLink: 0.1, workers: 0 });
  assert(a.weak.length === 0 && a.denseFrames.length === 0, `weak ${a.weak}, dense ${a.denseFrames}`);
  assert(Number.isNaN(linkStrength(a.seq.pairs, n, 3, dense.map(f => f.clip))[9]), 'the break has a link strength');
});

// each video flies its circle again and again (the same 5 places): its own revisits are more alike than the other
// video's views of a place, and would fill every frame's candidates
Deno.test('clips: loop candidates keep some in the other video', () => {
  const n = 160, clip = (k) => (k < 80 ? 0 : 1), place = (k) => k % 5, D = 2 + 5 + 10;
  const vectors = Array.from({ length: n }, (_, k) => {
    const v = new Float32Array(D);
    v[clip(k)] = 0.3; v[2 + place(k)] = 0.8; v[7 + 5 * clip(k) + place(k)] = Math.sqrt(1 - 0.09 - 0.64);
    return v;
  });
  const pairs = loopCandidates(vectors, { minGap: 20, clips: vectors.map((_, k) => clip(k)) });
  const across = new Set(pairs.filter(([i, j]) => clip(i) !== clip(j)).flat());
  const none = Array.from({ length: n }, (_, k) => k).filter(k => !across.has(k));
  assert(!none.length, `${none.length} frames with no candidate in the other video (${none.slice(0, 8)}...)`);
  assert(pairs.filter(([i, j]) => clip(i) !== clip(j)).every(([i, j]) => place(i) === place(j)), 'a candidate across of another place');
  // within a video as before: the same pairs without the other video
  const alone = loopCandidates(vectors.slice(0, 80), { minGap: 20 });
  assert(alone.join(';') === pairs.filter(([i, j]) => j < 80).join(';'), 'the first video\'s own candidates changed');
});

Deno.test('clips: two flights of one place mapped as one, the session\'s repairs within a video', async () => {
  const scene = makeSyntheticScene(CLIPS), { input } = scene;
  assert(input.images.every((im, k) => (im.clip ?? 0) === (k < 36 ? 0 : 1)), 'the keyframes\' clips');
  assert(!input.pairs.some(p => p.type === 'seq' && input.images[p.i].clip !== input.images[p.j].clip), 'a sequential pair across');
  const s = new SfmSession(input, { pairSource: tablePairSource(input) });
  const { stats } = await s.reconstruct();
  const { fails, tag } = accuracy(s, scene);
  assert(stats.numSegments === 1 && !fails.length, `${stats.numSegments} segments, ${tag}: ${fails.join('; ')}`);
  // the neighbours a retry matches: never in the other video
  const todo = s._neighbourTodo([35, 36], 10), clipOf = new Map(s.images.map(im => [im.name, im.clip ?? 0]));
  assert(todo.length > 0 && todo.every(([a, b]) => clipOf.get(a) === clipOf.get(b)), `neighbours across the break: ${todo.filter(([a, b]) => clipOf.get(a) !== clipOf.get(b)).length}`);
  // the clips kept in the saved state
  const { json } = s.toState();
  assert(json.images.filter(im => im.clip === 1).length === 36, 'the saved keyframes\' clips');
});

// two flights' GPS tracks as one: the second's times after its offset, its positions from the first's origin, no
// position between the two
Deno.test('clips: two videos\' GPS tracks joined', () => {
  const track = (lat0, n) => ({ source: 'djmd', t: Array.from({ length: n }, (_, k) => k), lat: Array.from({ length: n }, (_, k) => lat0 + 1e-5 * k),
                                lon: new Array(n).fill(7), alt: new Array(n).fill(100), attitude: new Array(n).fill([0, 0, 0]), gimbal: new Array(n).fill([0, 0, 0]) });
  const tel = joinTelemetry([{ tel: track(45, 10), offset: 0 }, { tel: null, offset: 12 }, { tel: track(45.0001, 5), offset: 30 }]);
  assert(tel.t.length === 15 && tel.t[10] === 30 && tel.breaks.join() === '10', `${tel.t}, breaks ${tel.breaks}`);
  const north = positionAt(tel, 31)[1];  // 1e-4 + 1e-5 degrees north of the first fix
  assert(Math.abs(north - 6378137 * 1.1e-4 * Math.PI / 180) < 0.01, `${north} m north`);
  assert(positionAt(tel, 20) === null && positionAt(tel, 5) !== null, 'a position between the two flights');
  assert(joinTelemetry([{ tel: null, offset: 0 }]) === null, 'no track');
});
