// render_cache.e2e.js — trainer.readView's render is drawn again when what it shows changed: the view's pose, its
// focal (zoom), the background (setEnv), or another camera drawn into hatTex in between; and the same as before when
// they are back. A synthetic scene of 200 sheets on a new trainer, no training step in between (the step count the same).
//   cd e2e && node render_cache.e2e.js   (env: see harness.js)
import { APP, PORT, fail, urlOf, run } from './harness.js';
import { join } from 'node:path';

await run(async (page) => {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'DESIGN.md'))}`);  // same origin, no app (and no dataset load)
  const r = await page.evaluate(async () => {
    const { GaussOptTrainer } = await import('./src/trainer.js');
    const { LooseOctree } = await import('./src/octree.js');
    const { setPose, SPLAT_FLOATS } = await import('./src/colmap.js');
    const { makeRng } = await import('./src/math.js');
    const W = 128, H = 96, N = 200, rng = makeRng(5);
    const device = await (await navigator.gpu.requestAdapter()).requestDevice();
    // small opaque sheets at depths 3..5 over the middle of the view, each its own color; the background around them
    const params = new Float32Array(N * SPLAT_FLOATS);
    for (let i = 0; i < N; i++) {
      params.set([1.2 * (rng() - 0.5), 0.8 * (rng() - 0.5), 3 + 2 * rng(), Math.log(0.9 / 0.1),
                  Math.log(0.05), Math.log(0.05), Math.log(1e-3), 0, 1, 0, 0, 0, 3 * rng() - 1.5, 3 * rng() - 1.5, 3 * rng() - 1.5, 0], i * SPLAT_FLOATS);
    }
    const camera = { width: W, height: H, fx: 100, fy: 100, cx: W / 2, cy: H / 2 };
    const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const img = setPose({ name: 'v0', isValidation: false, zMin: 1, zMax: 8, camera }, I, [0, 0, 0]);
    const octree = new LooseOctree({ maxLeafSize: 64 }).build(params, N, [img]);
    const tr = await GaussOptTrainer.create(device, octree, { count: N, params }, [img], { downsample: 1 });
    tr._loadGtTexture = async () => device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const diff = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i])); return d; };
    const hat = async (opts) => (await tr.readView(0, ['hat'], opts)).hat;
    const a = await hat();
    setPose(img, I, [0.3, 0, 0]);
    const moved = diff(a, await hat());
    setPose(img, I, [0, 0, 0]);
    const back = diff(a, await hat());
    const zoomed = diff(a, await hat({ zoom: 0.5 }));
    const other = setPose({ name: 'other', isValidation: true, zMin: 1, zMax: 8, camera }, I, [0, 0.4, 0]);
    await hat();  // the view's render cached, then another camera's drawn over it
    const drewOther = !!(await tr.readView(other, ['hat']));
    const afterOther = diff(a, await hat());
    const env = await tr.readEnv();
    tr.setEnv(env.map(() => 255));
    const white = diff(a, await hat());
    tr.setEnv(env);
    const envBack = diff(a, await hat());
    const steps = tr.stepCount;
    device.destroy();
    return { moved, back, zoomed, drewOther, afterOther, white, envBack, steps };
  });
  if (r.steps !== 0) fail(`${r.steps} training steps: the step count changed the cache key`);
  if (!(r.moved > 0)) fail('the view moved: the render the same (a stale cache)');
  if (r.back !== 0) fail(`the view back at its pose: the render differs by ${r.back} levels`);
  if (!(r.zoomed > 0)) fail('zoom 0.5: the render the same');
  if (!r.drewOther) fail('another camera: nothing drawn');
  if (r.afterOther !== 0) fail(`after another camera's render: the view's differs by ${r.afterOther} levels (hatTex held the other's)`);
  if (!(r.white > 0)) fail('a white background: the render the same');
  if (r.envBack !== 0) fail(`the background back: the render differs by ${r.envBack} levels`);
  return `drawn again on a pose (${r.moved} levels), a zoom (${r.zoomed}), the background (${r.white}) and after another camera; the same when back`;
});
