// Slow: dumps a real COLMAP database (jardin5 by default, SFM_DATABASE to override) with
// tools/colmap_dump.py and checks the loaded result against the database's own geometry.
import { loadDump, denoReader } from '../src/sfm/dump.js';
import { assert, SLOW, env, exists } from './util.js';

const DATABASE = env('SFM_DATABASE') ?? `${env('HOME')}/gskal/brush/jardin5/out/database_sift.db`;

Deno.test({ name: 'COLMAP dump: layout, indices, and epipolar consistency of the matches', ignore: !SLOW || !exists(DATABASE), fn: async () => {
  const out = await Deno.makeTempDir({ prefix: 'sfm-dump-' });
  try {
    const tool = new URL('../tools/colmap_dump.py', import.meta.url).pathname;
    const run = await new Deno.Command('python3', { args: [tool, DATABASE, out], stdout: 'piped', stderr: 'piped' }).output();
    assert(run.success, new TextDecoder().decode(run.stderr));
    const dump = await loadDump(denoReader(out));
    const n = dump.images.length;
    assert(n > 10, `${n} images`);
    assert(dump.images.every((im, k) => k === 0 || im.name > dump.images[k - 1].name), 'name order');
    for (const { i, j, type, matches } of dump.pairs) {
      assert(i < j && j < n, `pair ${i}-${j}`);
      assert(type === (j - i <= dump.meta.overlap ? 'seq' : 'loop'), 'pair type');
      const ni = dump.images[i].keypoints.length / 2, nj = dump.images[j].keypoints.length / 2;
      for (let m = 0; m < matches.length; m += 2) assert(matches[m] < ni && matches[m + 1] < nj, 'match index in range');
    }
    // x_j^T F x_i ~ 0: Sampson distance of the stored inliers (validates keypoint order and F orientation)
    let checked = 0, within = 0;
    for (const { i, j, F, config, matches } of dump.pairs) {
      if (!F || config !== 'UNCALIBRATED') continue;
      const ki = dump.images[i].keypoints, kj = dump.images[j].keypoints;
      for (let m = 0; m < matches.length; m += 2) {
        const x1 = [ki[2 * matches[m]], ki[2 * matches[m] + 1], 1], x2 = [kj[2 * matches[m + 1]], kj[2 * matches[m + 1] + 1], 1];
        const Fx1 = [0, 1, 2].map(r => F[3 * r] * x1[0] + F[3 * r + 1] * x1[1] + F[3 * r + 2]);
        const Ftx2 = [0, 1, 2].map(c => F[c] * x2[0] + F[3 + c] * x2[1] + F[6 + c]);
        const e = x2[0] * Fx1[0] + x2[1] * Fx1[1] + Fx1[2];
        const sampson = e * e / (Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2);
        checked++;
        within += sampson < 16;  // 4 px, COLMAP's default max_error
      }
    }
    assert(checked > 1000, `${checked} matches checked`);
    assert(within / checked > 0.99, `${(100 * within / checked).toFixed(2)}% of inliers within 4 px of their epipolar line`);
  } finally {
    await Deno.remove(out, { recursive: true });
  }
} });
