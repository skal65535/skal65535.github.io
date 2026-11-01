// stipple.e2e.js — The stipple seeds made off the page's thread (trainer.js _seedJob, stipple_worker.js) are the ones
// the page makes (stipple.js seedsFromView), for two jobs in flight at once, each answered as its own.
//   cd e2e && node stipple.e2e.js   (env: see harness.js)
import { APP, PORT, fail, urlOf, run } from './harness.js';
import { join } from 'node:path';

await run(async (page) => {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'DESIGN.md'))}`);  // same origin, no app (and no dataset load)
  const r = await page.evaluate(async () => {
    const { GaussOptTrainer } = await import('./src/trainer.js');
    const { seedsFromView } = await import('./src/stipple.js');
    const { makeRng } = await import('./src/math.js');
    const W = 160, H = 120;
    // a view's readback: an under-fit blob (gt != hat), pix_state covered on the left half at depth 2
    const args = (seed, budget) => {
      const rng = makeRng(seed), buf = new ArrayBuffer(W * H * 24);
      const gt = new Uint8Array(buf, 0, W * H * 4), hat = new Uint8Array(buf, W * H * 4, W * H * 4), pix = new Float32Array(buf, W * H * 8, W * H * 4);
      for (let i = 0; i < W * H; i++) {
        const x = i % W, y = (i / W) | 0, e = 150 * Math.exp(-((x - 60) ** 2 + (y - 50) ** 2) / 400) + 20 * rng();
        for (let c = 0; c < 3; c++) { gt[4 * i + c] = 60 + c * 40; hat[4 * i + c] = Math.min(255, 60 + c * 40 + e); }
        gt[4 * i + 3] = hat[4 * i + 3] = 255;
        pix.set(x < W / 2 ? [0.1, 2, 0, 0] : [0.9, 0, 0, 0], 4 * i);
      }
      return { gt, hat, pix, W, H, bytesPerRow: W * 4, budget, seed, img: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], center: [0.5, 0, -1] },
               fx: 150, fy: 150, cx: W / 2, cy: H / 2, depthGrid: { grid: new Float32Array(1), gw: 1, gh: 1, cell: W, median: 3 } };
    };
    const want = [seedsFromView(args(1, 300)), seedsFromView(args(2, 200))];
    const tr = Object.create(GaussOptTrainer.prototype);  // _seedJob needs no GPU
    const timeout = new Promise((_, no) => setTimeout(() => no(new Error('no answer from the worker in 10 s')), 10000));
    const got = await Promise.race([Promise.all([tr._seedJob(args(1, 300)), tr._seedJob(args(2, 200))]), timeout]);
    tr.seedWorker.terminate();
    return got.map((g, k) => ({ count: g.count, want: want[k].count, covered: g.covered, wantCovered: want[k].covered,
                                diff: want[k].params.reduce((n, x, i) => n + (x !== g.params[i]), 0) + Math.abs(g.params.length - want[k].params.length) }));
  });
  r.forEach((j, k) => {
    console.log(`job ${k + 1}: ${j.count} seeds (${j.covered} at the rendered depth), the page's ${j.want} (${j.wantCovered}), ${j.diff} values differ`);
    if (!(j.want >= 100 && j.wantCovered > 0 && j.wantCovered < j.want)) fail(`job ${k + 1}: ${j.want} seeds, ${j.wantCovered} covered: not a test`);
    else if (j.count !== j.want || j.diff) fail(`job ${k + 1}: the worker's seeds are not the page's`);
  });
  if (r[0].count === r[1].count) fail('the two jobs made as many seeds: their answers could be swapped unseen');
  return 'the worker\'s stipple seeds are the page\'s, two jobs in flight';
});
