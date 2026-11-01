// tile_overflow.e2e.js — A forward render (trainer._renderViewForward: the compare view, the validation, the held-out
// fit, the mesh's depth, train.mjs's checks) comes out whole when its (tile, splat) entries overflow the tile lists.
// The lists follow the entry count read back after each render (_pollTileEntries): a render needing far more than the
// last ones (a scene of a million splats just loaded: 2.4M entries against the first 1M) dropped the overflow, the tiles
// filled last empty (its lower half black), and only the next render had room. A synthetic scene of 6000 sheets
// across the whole view (192 tiles: 1.15M entries) on a new trainer (lists of 2^20): its first render against a second.
//   cd e2e && node tile_overflow.e2e.js   (env: see harness.js)
import { APP, PORT, fail, urlOf, run } from './harness.js';
import { join } from 'node:path';

await run(async (page) => {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'DESIGN.md'))}`);  // same origin, no app (and no dataset load)
  const r = await page.evaluate(async () => {
    const { GaussOptTrainer } = await import('./src/trainer.js');
    const { LooseOctree } = await import('./src/octree.js');
    const { setPose, SPLAT_FLOATS } = await import('./src/colmap.js');
    const { makeRng } = await import('./src/math.js');
    const W = 256, H = 192, N = 6000, rng = makeRng(11);
    const device = await (await navigator.gpu.requestAdapter()).requestDevice();
    // sheets 10 m wide, 1 mm thick, facing the camera at depths 2..6, faint (opacity 0.02), each its own color
    const params = new Float32Array(N * SPLAT_FLOATS);
    for (let i = 0; i < N; i++) {
      params.set([0.2 * (rng() - 0.5), 0.2 * (rng() - 0.5), 2 + 4 * rng(), Math.log(0.02 / 0.98),
                  Math.log(10), Math.log(10), Math.log(1e-3), 0, 1, 0, 0, 0, 3 * rng() - 1.5, 3 * rng() - 1.5, 3 * rng() - 1.5, 0], i * SPLAT_FLOATS);
    }
    const img = setPose({ name: 'v0', isValidation: false, zMin: 1, zMax: 8,
                          camera: { width: W, height: H, fx: 200, fy: 200, cx: W / 2, cy: H / 2 } }, [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);
    const octree = new LooseOctree({ maxLeafSize: 64 }).build(params, N, [img]);
    const tr = await GaussOptTrainer.create(device, octree, { count: N, params }, [img], { downsample: 1 });
    tr._loadGtTexture = async () => device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const cap0 = tr.tileListCapacity, shots = [];
    for (let k = 0; k < 2; k++) {
      tr._invalidateCaches();
      const out = await tr._renderViewForward(0);
      const [hat] = await tr._readback('overflow', [{ tex: out.hatTex }]);
      let empty = 0;  // rows with no splat drawn (the background is black)
      for (let y = 0; y < H; y++) { let s = 0; for (let x = 0; x < W * 4; x++) if ((x & 3) !== 3) s += hat[y * tr.rowPitch + x]; if (s === 0) empty++; }
      shots.push({ hat: Array.from(hat), empty, cap: tr.tileListCapacity });
    }
    await tr.binStatsPending;
    let diff = 0;
    for (let i = 0; i < shots[0].hat.length; i++) diff = Math.max(diff, Math.abs(shots[0].hat[i] - shots[1].hat[i]));
    device.destroy();
    return { cap0, visible: tr.numVisible, empty: shots.map(s => s.empty), caps: shots.map(s => s.cap), diff, H };
  });
  if (!(r.caps[1] > r.cap0)) fail(`the lists did not grow (${r.cap0} -> ${r.caps.join(', ')}): no overflow, nothing tested`);
  if (r.empty[0] || r.empty[1]) fail(`empty rows: ${r.empty[0]} of ${r.H} in the first render, ${r.empty[1]} in the second`);
  if (r.diff > 0) fail(`the first render differs from the second by up to ${r.diff} levels`);
  return `${r.visible} sheets over the view: the lists ${r.cap0} -> ${r.caps[1]} entries; the first render whole (${r.empty[0]} empty rows) and the same as the second`;
});
