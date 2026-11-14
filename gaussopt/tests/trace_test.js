// The SfM theatre's record (trace.js): its events replayed give the mapper's cameras, its stages in order, and the
// mapping itself the same with it as without it.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
import { SfmTrace, stateAt, encodeTrace, decodeTrace } from '../src/sfm/trace.js';
import { eventSays, stageText } from '../src/ui/theatre.js';
import { assert, ORBIT } from './util.js';

const CUT = { ...ORBIT, cut: 0.4 };  // two segments, merged by the revisits
const SEQ = { pairTypes: ['seq'] };  // the revisits for loop closing only: mapped with them, the orbit is one segment

Deno.test('trace: replayed, the mapper\'s cameras and segments at the end; the stages in order', async () => {
  const trace = new SfmTrace({ maxPoints: 500, pointsEvery: 10 });
  const { mapper, stats } = await reconstruct(makeSyntheticScene(CUT).input, { ...SEQ, trace: trace.hook });
  const ev = trace.events, { poses, segments } = stateAt(ev);
  let posed = 0;
  for (let k = 0; k < mapper.n; k++) {
    const P = mapper.poses[k];
    assert(segments[k] === (P ? mapper.segment[k] : -1), `frame ${k}: segment ${segments[k]}, the mapper's ${mapper.segment[k]}`);
    if (!P) { assert(Number.isNaN(poses[12 * k]), `frame ${k}: a pose, the mapper has none`); continue; }
    posed++;
    for (let c = 0; c < 9; c++) assert(poses[12 * k + c] === Math.fround(P.R[c]), `frame ${k}: R[${c}] ${poses[12 * k + c]}, the mapper's ${P.R[c]}`);
    const C = [0, 1, 2].map(i => -(P.R[i] * P.t[0] + P.R[3 + i] * P.t[1] + P.R[6 + i] * P.t[2]));
    for (let i = 0; i < 3; i++) assert(poses[12 * k + 9 + i] === Math.fround(C[i]), `frame ${k}: centre ${i}`);
  }
  assert(posed === mapper.poses.filter(Boolean).length && posed > 0, `${posed} posed`);
  const kinds = (k) => ev.filter(e => e.kind === k);
  assert(kinds('init').length === mapper.stats.segments.length, `${kinds('init').length} inits`);
  assert(kinds('init').length >= 2 && kinds('merge').length >= 1 && kinds('posegraph').length >= 1, ev.map(e => e.kind).join());
  assert(kinds('register').length + 2 * kinds('init').length === mapper.stats.segments.reduce((a, s) => a + s.size, 0), 'a register event a frame');
  assert(kinds('register').every(e => e.frames.includes(e.info.img)), 'a register event moves its frame');
  assert(kinds('ba').length === mapper.time.baCalls, `${kinds('ba').length} BA events, ${mapper.time.baCalls} window BAs`);
  const stages = kinds('stage').map(e => e.info.name);
  assert(stages.join() === ['map', 'loops', 'flowJoin', 'refine', 'retriangulate', 'lateMerge'].join(), stages.join());
  assert(ev.at(-1).kind === 'end' && ev.at(-1).points.X.length > 0 && ev.at(-1).points.X.length <= 3 * 500, `${ev.at(-1).points.X.length / 3} points`);
  assert(ev.every((e, i) => !i || e.t >= ev[i - 1].t), 'in time order');
  // the merge moves the merged segment's cameras: the frames of one of the two, with its new id
  const m = kinds('merge')[0];
  assert(m.frames.length > 0 && [...m.segments].every(s => s === m.info.a || s === m.info.b), `merge of ${m.info.a}, ${m.info.b}: ${[...new Set(m.segments)]}`);
});

// the retriangulation is long on a long video (minutes): it says how far it is, shows its new points in the theatre,
// and lets the worker post them while it runs
Deno.test('trace: the retriangulation\'s progress and points, the worker free between its parts', async () => {
  const trace = new SfmTrace({ maxPoints: 500 }), seen = [];
  let fired = false;
  const progress = (stage, done, total) => { if (stage === 'retriangulate') seen.push({ done, total, fired }); };
  const { mapper } = await reconstruct(makeSyntheticScene(CUT).input, { ...SEQ, trace: trace.hook, progress });
  assert(seen.length >= 3, `${seen.length} progress reports in the retriangulation`);
  const parts = seen.filter(p => p.total > 0);
  assert(parts.length >= 2 && parts.every((p, i) => p.total === mapper.n && (!i || p.done > parts[i - 1].done)) && parts.at(-1).done === mapper.n,
         parts.map(p => `${p.done}/${p.total}`).join());
  const ev = trace.events, at = ev.findIndex(e => e.kind === 'stage' && e.info.name === 'retriangulate');
  const tracks = ev.slice(at).filter(e => e.kind === 'tracks');
  assert(tracks.length >= 2 && tracks.every(e => e.points?.X.length > 0 && e.info.last >= e.info.first), `${tracks.length} 'tracks' events`);
  assert(eventSays(tracks[0]).includes('new points'), eventSays(tracks[0]));
  // a timer set before a retriangulation runs during it
  setTimeout(() => { fired = true; }, 0);
  seen.length = 0;
  await mapper.extendTracks();
  assert(seen.some(p => p.fired), 'no timer ran during the retriangulation: the worker could not post its progress');
});

// loop closing is long on a long video (minutes: 170 s on autumn_garden_01): it says how far it is through its merges,
// loop pairs, pose graphs and fusions, and lets the worker post its theatre events as they happen
Deno.test('trace: loop closing\'s progress, the worker free during it', async () => {
  const seen = [];
  let fired = null, firedBefore = null;
  const progress = (stage, done, total) => {
    if (stage === 'loops') { fired = false; setTimeout(() => { fired = true; }, 0); }
    if (stage === 'flowJoin') firedBefore = fired;
    if (['loopMerge', 'loopVerify', 'poseGraph', 'loopFuse'].includes(stage)) seen.push({ stage, done, total });
  };
  const { stats } = await reconstruct(makeSyntheticScene(CUT).input, { ...SEQ, trace: new SfmTrace().hook, progress });
  const of = (s) => seen.filter(p => p.stage === s);
  assert(stats.loops.merges >= 1 && of('loopMerge').some(p => p.done >= 1 && p.done <= p.total), JSON.stringify(of('loopMerge')));
  assert(of('loopVerify').length >= 1 && of('poseGraph').length >= 1 && of('loopFuse').length >= 1, seen.map(p => p.stage).join());
  assert(firedBefore === true, 'no timer ran during loop closing: the worker could not post its progress or events');
});

Deno.test('trace: the mapping the same with it as without it', async () => {
  const a = await reconstruct(makeSyntheticScene(CUT).input, {});
  const b = await reconstruct(makeSyntheticScene(CUT).input, { trace: new SfmTrace().hook });
  for (let k = 0; k < a.mapper.n; k++) {
    const P = a.mapper.poses[k], Q = b.mapper.poses[k];
    assert(!P === !Q && (!P || [...P.R, ...P.t].every((x, c) => x === [...Q.R, ...Q.t][c])), `frame ${k}`);
  }
});

// keyframes renumbered (a repair's insert): the event after it holds every pose, and the replay starts there
Deno.test('trace: a reindex resets the replay', () => {
  const I = [1, 0, 0, 0, 1, 0, 0, 0, 1], fake = (n, posed) => ({
    n, poses: Array.from({ length: n }, (_, k) => posed.includes(k) ? { R: I, t: [k, 0, 0] } : null),
    segment: Int32Array.from({ length: n }, (_, k) => posed.includes(k) ? 0 : -1), points: [] });
  const tr = new SfmTrace();
  tr.record(fake(3, [0, 1, 2]), 'register');
  const same = tr.record(fake(3, [0, 1, 2]), 'ba');
  assert(same.frames.length === 0 && !same.reset, 'nothing changed');
  const r = tr.record(fake(4, [0, 2, 3]), 'stage');  // a frame inserted at 1
  assert(r.reset && r.frames.join() === '0,2,3', `${r.reset} ${r.frames}`);
  const { poses, segments } = stateAt(tr.events);
  assert(segments.join() === '0,-1,0,0' && poses[12 * 3 + 9] === -3 && Number.isNaN(poses[12]), `${segments} ${poses[12 * 3 + 9]}`);
});

Deno.test('trace: encoded and decoded, the same events', async () => {
  const trace = new SfmTrace({ maxPoints: 200, pointsEvery: 5 });
  await reconstruct(makeSyntheticScene({ ...ORBIT, numFrames: 24 }).input, { trace: trace.hook });
  const { meta, events: back } = decodeTrace(encodeTrace(trace.events, { model: 'm1' }));
  assert(meta.model === 'm1', JSON.stringify(meta));
  assert(back.length === trace.events.length && back[0].images?.length === 24, `${back.length} of ${trace.events.length}`);
  const same = (a, b) => a.length === b.length && a.every((x, i) => Object.is(x, b[i]));
  trace.events.forEach((e, k) => {
    const d = back[k];
    assert(d.kind === e.kind && d.t === e.t && JSON.stringify(d.info) === JSON.stringify(e.info), `event ${k}: ${d.kind}`);
    assert(same(d.frames, e.frames) && same(d.poses, e.poses) && same(d.segments, e.segments), `event ${k}: arrays`);
    assert(!e.points === !d.points && (!e.points || (same(d.points.X, e.points.X) && same(d.points.segment, e.points.segment) && d.points.live === e.points.live)), `event ${k}: points`);
  });
});

// a fake mapper: frames 0..n-1, those in `posed` at x = k, points by id (null: deleted)
const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const fakeMapper = (names, posed, points = []) => ({
  n: names.length, images: names.map(name => ({ name, time: 0 })),
  poses: names.map((_, k) => posed.includes(k) ? { R: I3, t: [-k, 0, 0] } : null),
  segment: Int32Array.from(names, (_, k) => posed.includes(k) ? 0 : -1),
  points: points.map(x => x === null ? null : { X: [x, 0, 0], obs: new Map([[0, 0]]) }),
});

Deno.test('trace: going on from recorded events, only what changed since', () => {
  const m = fakeMapper(['a', 'b', 'c'], [0, 1]), tr = new SfmTrace();
  tr.record(m, 'register');
  const { events } = decodeTrace(encodeTrace(tr.events));
  const on = SfmTrace.from(events), m2 = fakeMapper(['a', 'b', 'c'], [0, 1]);  // the same keyframes, another mapper (a restore)
  const same = on.record(m2, 'stage', { name: 'repair' });
  assert(same.frames.length === 0 && !same.reset && !same.images, `${same.frames} reset ${same.reset}`);
  m2.poses[2] = { R: I3, t: [-2, 0, 0] }; m2.segment[2] = 0;
  const moved = on.record(m2, 'ba');
  assert(moved.frames.join() === '2' && moved.t >= events.at(-1).t, `${moved.frames}, t ${moved.t}`);
  assert(on.events.length === events.length + 2, `${on.events.length}`);
});

Deno.test('trace: other keyframes, as many: a reset with their names', () => {
  const tr = new SfmTrace();
  tr.record(fakeMapper(['a', 'b', 'c'], [0, 1, 2]), 'register');
  const r = tr.record(fakeMapper(['a', 'x', 'c'], [0, 1, 2]), 'stage');  // b dropped, x inserted
  assert(r.reset && r.images?.map(im => im.name).join() === 'a,x,c' && r.frames.length === 3, `${r.reset} ${r.frames}`);
  const same = tr.record(fakeMapper(['a', 'x', 'c'], [0, 1, 2]), 'ba');  // another array, the same names
  assert(!same.reset && !same.images && same.frames.length === 0, `${same.reset} ${same.frames}`);
});

Deno.test('trace: the points sampled by id, the same ones whatever is deleted', () => {
  const tr = new SfmTrace({ maxPoints: 4 }), pts = Array.from({ length: 16 }, (_, i) => i);
  const ids = (s) => [...s.X].filter((_, k) => k % 3 === 0);
  const a = ids(tr.samplePoints(fakeMapper(['a'], [0], pts)));
  pts[1] = null;  // a point before the sampled ones goes
  const b = ids(tr.samplePoints(fakeMapper(['a'], [0], pts)));
  assert(a.join() === '0,4,8,12' && b.join() === a.join(), `${a} then ${b}`);
  pts.push(...Array.from({ length: 16 }, (_, i) => 16 + i));  // twice as many: every 8th id, half of those before
  const c = ids(tr.samplePoints(fakeMapper(['a'], [0], pts)));
  assert(c.join() === '0,8,16,24', `${c}`);
});

Deno.test('trace: nothing recorded past maxEvents', () => {
  const tr = new SfmTrace({ maxEvents: 2 }), m = fakeMapper(['a'], [0]);
  tr.record(m, 'register'); tr.record(m, 'ba');
  assert(tr.record(m, 'ba') === null && tr.events.length === 2 && tr.full, `${tr.events.length} ${tr.full}`);
});

// numbers not finite come back from JSON as null: the theatre's caption says '?' instead of throwing
Deno.test('trace: the theatre\'s captions with numbers lost to JSON', () => {
  const back = JSON.parse(JSON.stringify({ initialCost: NaN, cost: Infinity, scale: NaN, turnDeg: NaN, angleDeg: NaN }));
  for (const kind of ['init', 'register', 'ba', 'segment', 'loops', 'posegraph', 'merge', 'join', 'unregister', 'undo', 'end']) {
    const s = eventSays({ kind, info: back });
    assert(typeof s === 'string', `${kind}: ${s}`);
  }
  assert(eventSays({ kind: 'posegraph', info: back }).includes('? → ?'), eventSays({ kind: 'posegraph', info: back }));
  assert(stageText('retry')[0] === 'Repair: retry' && stageText('map')[0] === 'Sequential mapping', stageText('retry')[0]);
});
