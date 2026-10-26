// Keyframe quality signals (quality.js, DESIGN-sfm §6.1): green on a clean synthetic map, red where
// a pose is damaged; (slow) on jardin5, few red frames, among them frame 176 whose pose disagrees
// with its own matches and with COLMAP, and a redo there that leaves every COLMAP piece passing.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
import { keyframeQuality } from '../src/sfm/quality.js';
import { SfmSession } from '../src/sfm/session.js';
import { expSO3, mul3 } from '../src/sfm/linalg.js';
import { assert, SLOW, DUMP, HAS_REF, loadInput, assessColmap, ORBIT } from './util.js';
import { tablePairSource } from './session_util.js';

// The synthetic orbit mapped once, each test given it clean (mapper.snapshot / restore)
let base = null;
function cleanMap() {
  if (!base) {
    const { mapper, stats } = reconstruct(makeSyntheticScene(ORBIT).input, {});
    base = { mapper, stats, keep: mapper.snapshot() };
  }
  base.mapper.restore(base.keep);
  base.keep = base.mapper.snapshot();
  return base;
}

Deno.test('quality: a clean map is green; a damaged pose is red, with its reason', () => {
  const { mapper, stats } = cleanMap();
  const q = keyframeQuality(mapper, { loops: stats.loops.loops });
  assert(q.counts.green === 50 && q.loops.length > 0 && !q.loops.some(l => l.suspect), `clean: ${JSON.stringify(q.counts)}`);
  const k = 20, p = mapper.poses[k];
  mapper.poses[k] = { R: mul3(expSO3([0, 0.005, 0]), p.R), t: p.t };  // 0.3 deg
  const hurt = keyframeQuality(mapper, {}).frames[k];
  assert(hurt.score === 'red' && /reprojection/.test(hurt.reason), `damaged: ${hurt.score} ${hurt.reason}`);
  mapper.poses[k] = p;
  for (const kp of mapper.kpPoint[k]) if (kp >= 0 && mapper.points[kp]) mapper.points[kp].obs.delete(k);
  mapper.kpPoint[k].fill(-1);
  const bare = keyframeQuality(mapper, {}).frames[k];
  assert(bare.score === 'red' && /weak/.test(bare.reason), `no points: ${bare.score} ${bare.reason}`);
});

Deno.test('quality: loops checked on the map are not suspect; a loop frame moved off the map is', () => {
  const { mapper, stats } = cleanMap();
  const loops = stats.loops.loops, q = keyframeQuality(mapper, { loops, checkLoops: true });
  const checked = q.loops.filter(l => l.reloc);
  assert(checked.length >= 0.8 * loops.length && !q.loops.some(l => l.suspect),
         `clean: ${checked.length}/${loops.length} relocalized, worst ${Math.max(...checked.map(l => l.reloc.offset)).toFixed(4)} x depth`);
  const { j } = loops[0], p = mapper.poses[j], depth = keyframeQuality(mapper, {}).frames[j].depth;
  mapper.poses[j] = { R: p.R, t: [p.t[0] + 0.3 * depth, p.t[1], p.t[2]] };  // its points stay
  const moved = keyframeQuality(mapper, { loops, checkLoops: true }).loops.filter(l => l.i === j || l.j === j);
  assert(moved.length && moved.every(l => l.suspect), `moved: ${moved.map(l => JSON.stringify(l.reloc)).join(', ')}`);
});

Deno.test({ name: 'quality on jardin5 (SFM_DUMP): few red frames, 176 among them; redo there merges the map and keeps COLMAP pieces passing',
            ignore: !SLOW || !DUMP || !HAS_REF, fn: async () => {
  const input = await loadInput(), names = input.images.map(im => im.name);
  const s = new SfmSession(input, { pairSource: tablePairSource(input) });
  await s.reconstruct();
  const q = s.quality();
  const red = q.frames.filter(f => f.score === 'red' && f.segment === q.main);
  console.log(`  ${JSON.stringify(q.counts)}; red in the main map: ${red.map(f => `${f.index} (${f.reason})`).join(', ')}`);
  assert(red.length <= 5, `${red.length} red frames in the main map`);
  assert(red.some(f => f.index === 176 && /disagrees with its matches/.test(f.reason)), 'frame 176 flagged');
  assert(!q.loops.some(l => l.suspect), 'no suspect loop');
  const r = await s.redo(names.slice(168, 185));
  const { fails } = await assessColmap(s.model(), input);
  const segments = new Set(s.model().images.map(im => im.segment)).size;
  console.log(`  redo 168..184: ${r.maps} map(s), ${r.merges} merges -> ${segments} segment(s), ${s.model().images.length} registered`);
  assert(segments === 1 && s.model().images.length === names.length, 'one segment with every keyframe');
  assert(fails.length === 0, fails.join('; '));
} });
