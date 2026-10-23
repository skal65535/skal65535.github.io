// compare.js — Compare two COLMAP sparse models by image name (DESIGN-sfm §8.2):
//   deno run --allow-read tools/compare.js EST_MODEL REF_MODEL [--check truth|colmap] [--json]
// A model is a directory with cameras.bin and images.bin (points3D.bin optional), or one with a
// sparse/ subdirectory (a gs/ dataset). Exit code 1 when --check thresholds fail.
"use strict";

import { compareModels, checkThresholds, formatMetrics } from '../src/sfm/compare.js';
import { cli, readColmapModel } from './common.js';

const c = cli('compare.js EST_MODEL REF_MODEL [--check truth|colmap] [--json]');
const json = c.flag('--json'), check = c.option('--check');
const [est, ref] = await Promise.all(c.done(2).map(readColmapModel));
const m = compareModels(est, ref);
const fails = check ? checkThresholds(m, check) : [];
if (json) {
  console.log(JSON.stringify({ ...m, perImage: undefined, fails }, null, 1));
} else {
  console.log(formatMetrics(m));
  const worst = [...m.perImage].sort((a, b) => b.centerErr - a.centerErr).slice(0, 5);
  if (worst.length) console.log(`worst       ${worst.map(p => `${p.name} (${(100 * p.centerErr / m.scaleNorm).toFixed(2)}%, ${p.rotDeg.toFixed(2)} deg)`).join(', ')}`);
  if (check) console.log(fails.length ? `FAIL (${check}):\n  ${fails.join('\n  ')}` : `PASS (${check})`);
}
Deno.exit(fails.length ? 1 : 0);
