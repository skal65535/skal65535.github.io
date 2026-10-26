// posgrad.e2e.js — adam_step's chain from the 2D gradients (grads_2d: dL/du0, dv0 and dL/dconic a, b, c) to
// the splat's world position, scales and rotation, against central differences of project_splats' output: for
// fixed weights w, L = w . (u0, v0, a, b, c) of one splat; its gradient injected into grads_2d, adam_step run with
// zero learning rates; Adam's first moment after one step is 0.1 g (|g| < 1, its clamp). Splats tilted and off
// axis, seen by a camera turned off the world axes (every term of the projection's Jacobian non-zero).
//   cd e2e && node posgrad.e2e.js   (env: see harness.js)
import { APP, PORT, fail, urlOf, run } from './harness.js';
import { join } from 'node:path';

await run(async (page) => {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'DESIGN.md'))}`);  // same origin, no app (and no dataset load)
  const r = await page.evaluate(async () => {
    const { GaussOptTrainer } = await import('./src/trainer.js');
    const { LooseOctree } = await import('./src/octree.js');
    const { setPose, SPLAT_FLOATS } = await import('./src/colmap.js');
    const W = 256, H = 192;
    const [ca, sa, cb, sb] = [Math.cos(0.35), Math.sin(0.35), Math.cos(0.25), Math.sin(0.25)];
    const R = [cb, sa * sb, -ca * sb, 0, ca, sa, sb, -sa * cb, ca * cb];  // world -> camera, orthonormal
    const toWorld = (xc) => [0, 1, 2].map(j => R[j] * xc[0] + R[3 + j] * xc[1] + R[6 + j] * xc[2]);  // t = 0
    const img = setPose({ name: 'v0', isValidation: false, zMin: 1, zMax: 8,
                          camera: { width: W, height: H, fx: 200, fy: 200, cx: W / 2, cy: H / 2 } }, R, [0, 0, 0]);
    // [camera-frame center, log scales, quaternion (w, x, y, z)]: flat, tilted, off axis; one beyond the frustum
    // clamp of x / z (1.3 x the half-width), where J's x column stops depending on x
    const splats = [
      [[0.3, -0.2, 3], [Math.log(0.2), Math.log(0.1), Math.log(0.004)], [0.9, 0.3, -0.2, 0.25]],
      [[-0.8, 0.5, 2.2], [Math.log(0.05), Math.log(0.15), Math.log(0.08)], [0.6, -0.5, 0.4, 0.5]],
      [[0.1, 0.6, 5], [Math.log(0.3), Math.log(0.01), Math.log(0.3)], [0.2, 0.9, 0.3, -0.25]],
      [[1.8, 0.2, 2], [Math.log(0.3), Math.log(0.2), Math.log(0.05)], [0.7, 0.1, 0.7, 0.1]],
    ];
    const weights = [[0.003, -0.002, 0.004, -0.003, 0.002], [-0.002, 0.003, -0.001, 0.004, 0.003],
                     [0.001, 0.002, 0.003, 0.002, -0.004], [0.002, -0.003, 0.002, 0.001, 0.003]];
    const device = await (await navigator.gpu.requestAdapter()).requestDevice();
    const out = [];
    for (let s = 0; s < splats.length; s++) {
      const [xc, ls, q] = splats[s], qn = Math.hypot(...q);
      const base = [...toWorld(xc), 0, ...ls, 0, ...q.map(x => x / qn), 0.2, 0.3, 0.4, 0];
      const params = Float32Array.from(base);
      const octree = new LooseOctree({ maxLeafSize: 64 }).build(params, 1, [img]);
      const tr = await GaussOptTrainer.create(device, octree, { count: 1, params }, [img], { downsample: 1 });
      const k = tr._intrinsics(img);
      tr.updateSortingForView(img.center, img.view, k.fxNorm, k.fyNorm);
      // lrs 0, Adam step 1, limits that never bind, no loss extras, no regularizers
      tr._writeUniforms(img, k, [0, 0, 0, 0, 0, 0, 0, 1, 100, 100, Math.log(1e4), 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      const U = tr.trainUniformBuf;
      // project_splats' (u0, v0, a, b, c) of the splat at `p` (16 floats)
      const project = async (p) => {
        device.queue.writeBuffer(tr.splatsBuf, 0, Float32Array.from(p));
        const enc = device.createCommandEncoder();
        enc.clearBuffer(tr.tileCountsBuf);
        tr._pass(enc, tr.pipeProject, [U, tr.splatsBuf, tr.sortedIdxBuf, tr.projSplatsBuf, tr.projTilesBuf, tr.tileCountsBuf], 1);
        device.queue.submit([enc.finish()]);
        const [pr] = await tr._readback('proj', [{ buf: tr.projSplatsBuf, bytes: 32 }]);
        return [pr[0], pr[1], pr[4], pr[5], pr[6], pr[3]];
      };
      const p0 = await project(base);
      // two weight sets: all five outputs, and the conic only (its path through the covariance and J, which the
      // center's terms would swamp)
      for (const [set, w] of [['all', weights[s]], ['conic', [0, 0, ...weights[s].slice(2).map(x => 10 * x)]]]) {
        const L = (v) => w.reduce((t, wi, i) => t + wi * v[i], 0);
        // analytic: grads_2d slots 4..8 = w (slot 3, dL/dlogit opacity, non-zero so the step runs), adam_step
        const g2 = new Float32Array(12);
        g2[3] = 1e-3; g2[4] = w[0]; g2[5] = w[1]; g2[6] = w[2]; g2[7] = w[3]; g2[8] = w[4]; g2[9] = 1;
        device.queue.writeBuffer(tr.grads2dBuf, 0, g2);
        device.queue.writeBuffer(tr.splatsBuf, 0, Float32Array.from(base));
        device.queue.writeBuffer(tr.adamMBuf, 0, new Float32Array(16));
        device.queue.writeBuffer(tr.adamVBuf, 0, new Float32Array(16));
        const enc = device.createCommandEncoder();
        tr._pass(enc, tr.pipeAdam, [U, tr.splatsBuf, tr.grads2dBuf, tr.adamMBuf, tr.adamVBuf, tr.momentsBuf, tr.growAccBuf], 1);
        device.queue.submit([enc.finish()]);
        const [m] = await tr._readback('m', [{ buf: tr.adamMBuf, bytes: 64 }]);
        const analytic = { pos: [0, 1, 2].map(i => 10 * m[i]), scale: [4, 5, 6].map(i => 10 * m[i]) };
        // central differences: world position (h 1e-3), log scales (h 1e-3)
        const fd = async (i, h) => {
          const a = base.slice(), b = base.slice();
          a[i] += h; b[i] -= h;
          return (L(await project(a)) - L(await project(b))) / (2 * h);
        };
        const numeric = { pos: [], scale: [] };
        for (const i of [0, 1, 2]) numeric.pos.push(await fd(i, 1e-3));
        for (const i of [4, 5, 6]) numeric.scale.push(await fd(i, 1e-3));
        out.push({ s, set, radius: p0[5], uv: [p0[0], p0[1]], analytic, numeric });
      }
      tr.destroy?.();
    }
    device.destroy();
    return out;
  });

  let checked = 0;
  for (const { s, set, radius, uv, analytic, numeric } of r) {
    if (!(radius > 0)) { fail(`splat ${s}: culled (radius ${radius})`); continue; }
    for (const key of ['pos', 'scale']) {
      const a = analytic[key], n = numeric[key], scale = Math.max(...n.map(Math.abs));
      const err = Math.max(...a.map((x, i) => Math.abs(x - n[i]))) / scale;
      console.log(`splat ${s} (${set}) at (${uv.map(x => x.toFixed(1)).join(', ')}) px, dL/d${key}: adam_step [${a.map(x => x.toExponential(3)).join(', ')}], ` +
                  `differences [${n.map(x => x.toExponential(3)).join(', ')}], max error ${(100 * err).toFixed(2)}%`);
      if (a.some(x => Math.abs(x) > 0.99)) fail(`splat ${s} (${set}) ${key}: a component at Adam's clamp (|g| <= 1)`);
      else if (!(scale > 1e-6)) fail(`splat ${s} (${set}) ${key}: the differences are all ~0, no test`);
      else if (!(err < 0.02)) fail(`splat ${s} (${set}) dL/d${key}: off by ${(100 * err).toFixed(2)}% of its largest component`);
      else checked++;
    }
  }
  return `adam_step's gradients of the position and the log scales match central differences of project_splats (${checked} checks)`;
});
