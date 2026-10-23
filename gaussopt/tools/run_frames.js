// run_frames.js — The in-browser SfM end to end on extracted frames, headless (Deno + WebGPU):
// GPU SIFT -> loop candidates by appearance -> GPU matching (sequential window + loops) ->
// verification -> reconstruction, with geometric loop candidates once the sequential map exists.
//   deno run --allow-read --allow-write tools/run_frames.js PGM_DIR [--ref MODEL] [--features N]
//            [--window N] [--no-loops] [--limit N] [--export OUT_DIR] [--loop-ref DUMP_DIR]
// PGM_DIR: frames converted by tools/gray.py (name order = time order). --ref and --export work as
// in tools/map.js; --loop-ref reports how many of a COLMAP dump's verified loop pairs we found.
"use strict";

import { GpuSift } from '../src/sfm/sift.js';
import { GpuMatcher } from '../src/sfm/match.js';
import { reconstructFromFeatures } from '../src/sfm/pipeline.js';
import { datasetFromMapper } from '../src/sfm/export.js';
import { readPGM } from './pgm.js';
import { cli, assessAgainstColmap, writeDataset, formatTimings, printAssessment } from './common.js';

const c = cli('run_frames.js PGM_DIR [--ref MODEL] [--features N] [--window N] [--no-loops] [--limit N] [--export OUT_DIR] [--loop-ref DUMP_DIR]');
const ref = c.option('--ref'), maxFeatures = Number(c.option('--features', 4096)), window = Number(c.option('--window', 10));
const limit = Number(c.option('--limit', 0)), exportDir = c.option('--export'), loopRef = c.option('--loop-ref');
const noLoops = c.flag('--no-loops');
const [dir] = c.done(1);
const seconds = (t) => `${((performance.now() - t) / 1000).toFixed(1)} s`;

const names = [...Deno.readDirSync(dir)].map(e => e.name).filter(n => n.endsWith('.pgm')).sort().slice(0, limit || undefined);
const t0 = performance.now();
const sift = await GpuSift.create();
const matcher = await GpuMatcher.create(sift.device);
const feats = [];
let size = null;
for (const name of names) {
  const img = await readPGM(`${dir}/${name}`);
  size ??= [img.width, img.height];
  feats.push({ name: name.replace(/\.pgm$/, '.jpg'), ...await sift.extract(img.data, img.width, img.height, { maxFeatures }) });
}
console.log(`features: ${feats.length} frames, ${Math.round(feats.reduce((s, f) => s + f.keypoints.length / 4, 0) / feats.length)} per frame in ${seconds(t0)}`);

const t1 = performance.now();
const { mapper, model, stats, input, matched } = await reconstructFromFeatures(feats, size[0], size[1], matcher,
  { maxFeatures, window, loops: !noLoops, log: (s) => console.log(s) });
console.log(`  (${formatTimings(stats.timings)}, ${seconds(t1)} after the features)`);
if (loopRef) {
  const meta = JSON.parse(await Deno.readTextFile(`${loopRef}/dump.json`));
  const index = new Map(feats.map((f, k) => [f.name, k]));
  const key = (i, j) => `${i},${j}`;
  const theirs = meta.pairs.filter(p => p.type === 'loop' && p.numMatches >= 50)
    .map(p => [index.get(meta.images[p.i].name), index.get(meta.images[p.j].name)])
    .filter(([i, j]) => i !== undefined && j !== undefined && j - i > 2 * window);
  const ours = new Set(matched.pairs.filter(p => p.type === 'loop').map(p => key(p.i, p.j)));
  console.log(`COLMAP's verified loops (>= 50 inliers): ${theirs.length}, of which we verified ${theirs.filter(([i, j]) => ours.has(key(i, j))).length}; ` +
              `we verified ${ours.size} loop pairs`);
}
console.log(`total ${seconds(t0)}`);

const assessment = ref ? await assessAgainstColmap(model, ref, input) : null;
if (assessment) printAssessment(assessment);
if (exportDir) await writeDataset(exportDir, datasetFromMapper(mapper));
if (assessment?.fails.length) Deno.exit(1);
