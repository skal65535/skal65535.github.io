// camgrad.e2e.js — adam_step's camera gradient (shaders.wgsl, poses.js; DESIGN.md E19 2, 3), summed by cam_reduce
// into loss_acc, against central differences of project_splats' output: for fixed weights w, L = w . (u0, v0, a, b, c)
// of one splat, its gradient injected into grads_2d (as posgrad.e2e.js), adam_step run with zero learning rates and
// the camera's sums on; numerically, the view through a camera-frame increment (R = Exp(phi) R0, t = Exp(phi) t0 + rho)
// and the intrinsics through log f, cx, cy, k1 (the lens's residual distortion, distort_jac). The 10 values (rho, phi,
// log f, cx, cy, k1) within 1% of the largest, on splats tilted and off axis seen by a camera turned off the world
// axes, at k1 0 and 0.08 (D's terms visible); the weights on the conic alone check the covariance's terms (its turn
// with the camera, J's dependence on the focal, D's on x_c and k1), which the centre's would swamp.
//   cd e2e && PORT=8791 node camgrad.e2e.js   (env: see harness.js)
import { APP, PORT, fail, urlOf, run } from './harness.js';
import { join } from 'node:path';

await run(async (page) => {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'DESIGN.md'))}`);  // same origin, no app (and no dataset load)
  const r = await page.evaluate(async () => {
    const { GaussOptTrainer } = await import('./src/trainer.js');
    const { LooseOctree } = await import('./src/octree.js');
    const { setPose } = await import('./src/colmap.js');
    const { expRot } = await import('./src/poses.js');
    const { ENV_TEXELS } = await import('./src/env.js');
    const CAM_AT = 4 + 4 * ENV_TEXELS + 8;
    const W = 256, H = 192;
    const [ca, sa, cb, sb] = [Math.cos(0.35), Math.sin(0.35), Math.cos(0.25), Math.sin(0.25)];
    const R0 = [cb, sa * sb, -ca * sb, 0, ca, sa, sb, -sa * cb, ca * cb];  // world -> camera, orthonormal
    const t0 = [0.2, -0.1, 0.3];
    const toWorld = (xc) => [0, 1, 2].map(j => R0[j] * (xc[0] - t0[0]) + R0[3 + j] * (xc[1] - t0[1]) + R0[6 + j] * (xc[2] - t0[2]));
    const cam = { width: W, height: H, fx: 200, fy: 200, cx: W / 2 + 7, cy: H / 2 - 5 };
    const img = setPose({ name: 'v0', isValidation: false, zMin: 1, zMax: 8, camera: cam }, R0, t0);
    const splats = [
      [[0.3, -0.2, 3], [Math.log(0.2), Math.log(0.1), Math.log(0.004)], [0.9, 0.3, -0.2, 0.25]],
      [[-0.8, 0.5, 2.2], [Math.log(0.05), Math.log(0.15), Math.log(0.08)], [0.6, -0.5, 0.4, 0.5]],
      [[0.1, 0.6, 5], [Math.log(0.3), Math.log(0.01), Math.log(0.3)], [0.2, 0.9, 0.3, -0.25]],
      [[0.5, 0.3, 2.5], [Math.log(0.12), Math.log(0.3), Math.log(0.02)], [0.7, 0.1, 0.7, 0.1]],
    ];
    const weights = [[0.003, -0.002, 0.004, -0.003, 0.002], [-0.002, 0.003, -0.001, 0.004, 0.003],
                     [0.001, 0.002, 0.003, 0.002, -0.004], [0.002, -0.003, 0.002, 0.001, 0.003]];
    const device = await (await navigator.gpu.requestAdapter()).requestDevice();
    const out = [];
    for (const K1 of [0, 0.08]) for (let s = 0; s < splats.length; s++) {
      const [xc, ls, q] = splats[s], qn = Math.hypot(...q);
      const base = Float32Array.from([...toWorld(xc), 0, ...ls, 0, ...q.map(x => x / qn), 0.2, 0.3, 0.4, 0]);
      const octree = new LooseOctree({ maxLeafSize: 64 }).build(base, 1, [img]);
      const tr = await GaussOptTrainer.create(device, octree, { count: 1, params: base }, [img], { downsample: 1 });
      const k0 = { ...tr.intrinsics(img), k1: K1 };
      tr.updateSortingForView(img.center, img.view, k0.fxNorm, k0.fyNorm);
      const extra = [0, 0, 0, 0, 0, 0, 0, 1, 100, 100, Math.log(1e4), 0, 0, 0, 0, 0, 0, 0, 0, 0];
      const U = tr.trainUniformBuf;
      // project_splats' (u0, v0, a, b, c) through the increment d = [rho, phi, log f, cx, cy, k1]
      const project = async (d) => {
        const E = expRot([d[3], d[4], d[5]]), mul = (A, B) => [0, 1, 2].flatMap(i => [0, 1, 2].map(j => A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]));
        const t = [0, 1, 2].map(i => E[3 * i] * t0[0] + E[3 * i + 1] * t0[1] + E[3 * i + 2] * t0[2] + d[i]);
        const im = setPose({ name: 'v0', camera: cam }, mul(E, R0), t);
        const f = Math.exp(d[6]);
        tr._writeUniforms(im, { fx: k0.fx * f, fy: k0.fy * f, cx: k0.cx + d[7], cy: k0.cy + d[8], k1: K1 + d[9] }, extra);
        const enc = device.createCommandEncoder();
        enc.clearBuffer(tr.tileCountsBuf);
        tr._pass(enc, tr.pipeProject, [U, tr.splatsBuf, tr.sortedIdxBuf, tr.projSplatsBuf, tr.projTilesBuf, tr.tileCountsBuf], 1);
        device.queue.submit([enc.finish()]);
        const [pr] = await tr._readback('proj', [{ buf: tr.projSplatsBuf, bytes: 32 }]);
        return [pr[0], pr[1], pr[4], pr[5], pr[6]];
      };
      for (const [set, w] of [['all', weights[s]], ['conic', [0, 0, ...weights[s].slice(2).map(x => 10 * x)]]]) {
        const L = (v) => w.reduce((a, wi, i) => a + wi * v[i], 0);
        const g2 = new Float32Array(12);
        g2[3] = 1e-3; g2[4] = w[0]; g2[5] = w[1]; g2[6] = w[2]; g2[7] = w[3]; g2[8] = w[4]; g2[9] = 1;
        device.queue.writeBuffer(tr.grads2dBuf, 0, g2);
        tr._writeUniforms(img, k0, extra, false, true);
        const enc = device.createCommandEncoder();
        tr._pass(enc, tr.pipeAdam, [U, tr.splatsBuf, tr.grads2dBuf, tr.adamMBuf, tr.adamVBuf, tr.momentsBuf, tr.sortedIdxBuf, tr.camPartBuf], 1);
        tr._pass(enc, tr.pipeCamReduce, [U, tr.camPartBuf, tr.lossBuf], 1);
        device.queue.submit([enc.finish()]);
        const [loss] = await tr._readback('cam', [{ buf: tr.lossBuf, bytes: tr.lossBuf.size }]);
        const analytic = Array.from(loss.slice(CAM_AT, CAM_AT + 10));
        const h = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.05, 0.05, 1e-3], numeric = [];
        for (let j = 0; j < 10; j++) {
          const a = new Array(10).fill(0), b = new Array(10).fill(0);
          a[j] = h[j]; b[j] = -h[j];
          numeric.push((L(await project(a)) - L(await project(b))) / (2 * h[j]));
        }
        const scale = Math.max(...numeric.map(Math.abs)), err = Math.max(...numeric.map((x, j) => Math.abs(x - analytic[j]))) / scale;
        out.push({ splat: s, k1: K1, set, err, analytic, numeric });
      }
      await tr.destroy();
    }
    return out;
  });
  let worst = 0;
  for (const c of r) {
    worst = Math.max(worst, c.err);
    if (!(c.err < 0.01)) fail(`splat ${c.splat} (${c.set}, k1 ${c.k1}): off by ${(100 * c.err).toFixed(2)}% of the largest\n  analytic ${c.analytic.map(x => x.toExponential(3)).join(' ')}\n  numeric  ${c.numeric.map(x => x.toExponential(3)).join(' ')}`);
  }
  return `the camera's gradient (rho, phi, log f, cx, cy, k1; at k1 0 and 0.08) = central differences of project_splats, ${r.length} checks, within ${(100 * worst).toFixed(3)}% of the largest`;
});
