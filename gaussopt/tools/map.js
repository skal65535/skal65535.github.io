// map.js — Run the SfM pipeline (src/sfm/reconstruct.js: sequential mapping, loop closure and
// segment merging, refinement sweeps, retriangulation) on a COLMAP dump (tools/colmap_dump.py):
//   deno run --allow-read --allow-write tools/map.js DUMP_DIR [--ref MODEL] [--window N] [--no-loops]
//            [--limit N] [--export OUT_DIR [--segment N]] [--json]
// --ref compares with a COLMAP model (e.g. OUT/sparse_sift/0), its own defects excluded
// (src/sfm/reference.js); exit code 1 when that comparison fails.
// --no-loops stops after the sequential mapper (sequential pairs only); --limit maps the first N images.
// --export writes one segment (default: the largest) as a COLMAP model for Gauss-Opt; then
// tools/undistort.py SOURCE_IMAGES OUT_DIR completes the gs/ dataset.
"use strict";

import { loadDump, denoReader } from '../src/sfm/dump.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
import { datasetFromMapper } from '../src/sfm/export.js';
import { cli, assessAgainstColmap, writeDataset, formatTimings, printAssessment } from './common.js';

const c = cli('map.js DUMP_DIR [--ref MODEL] [--window N] [--no-loops] [--limit N] [--export OUT_DIR [--segment N]] [--json]');
const ref = c.option('--ref'), window = c.option('--window'), limit = Number(c.option('--limit', 0));
const exportDir = c.option('--export'), segment = c.option('--segment');
const noLoops = c.flag('--no-loops'), json = c.flag('--json');
const [dumpDir] = c.done(1);

const input = await loadDump(denoReader(dumpDir));
if (limit) {
  input.images = input.images.slice(0, limit);
  input.pairs = input.pairs.filter(p => p.j < limit);
}
const t0 = performance.now();
const { mapper, model, stats } = reconstruct(input, {
  loops: !noLoops, sweeps: noLoops ? 0 : 2, retriangulate: !noLoops,
  ...(window ? { window: Number(window) } : {}),
  log: (s) => { if (!json) console.log(`  ${s}`); },
});
const seconds = (performance.now() - t0) / 1000;

const assessment = ref ? await assessAgainstColmap(model, ref, input) : null;
if (json) {
  const a = assessment && {
    defects: { badFrames: Object.fromEntries(assessment.defects.badFrames), breaks: Object.fromEntries(assessment.defects.breaks) },
    local: { ...assessment.local, pairs: undefined },
    pieces: assessment.pieces.map(({ metrics, ...p }) => metrics ? { ...p, rmsCenterRel: metrics.rmsCenterRel, medianRotDeg: metrics.medianRotDeg } : p),
    fails: assessment.fails,
  };
  console.log(JSON.stringify({ seconds, registered: model.images.length, numImages: input.images.length, focal: model.focal,
                               focalInit: mapper.stats.focalInit, focalSource: mapper.stats.focalSource, numPoints: model.numPoints,
                               timings: stats.timings, numSegments: stats.numSegments, loops: stats.loops, reference: a }, null, 1));
} else {
  console.log(`mapped ${model.images.length}/${input.images.length} images in ${stats.numSegments} segment(s), ${model.numPoints} points, ` +
              `focal ${model.focal.toFixed(1)} px (${mapper.stats.focalSource}) in ${seconds.toFixed(1)} s (${formatTimings(stats.timings)})`);
  if (assessment) printAssessment(assessment, { verbose: true });
}
if (exportDir) await writeDataset(exportDir, datasetFromMapper(mapper, { segment: segment === null ? null : Number(segment) }));
if (assessment?.fails.length) Deno.exit(1);
