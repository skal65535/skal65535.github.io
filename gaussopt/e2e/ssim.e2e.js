// ssim.e2e.js — The D-SSIM loss (params.ssim, kernels ssim_coef / ssim_grad, tile_backward):
//   - ssim_grad_tex against central differences of ssimSum (src/metrics.js) at the render's
//     8-bit values, on a textured render of random splats: borders, corners and inner pixels;
//   - tile_backward with weight 1 on one opaque sheet: its dL/drgb (grads_2d slots 0..2) is
//     sum_p -(1/2) G(p) / N * w(p) * SH_C0, G = ssim_grad_tex, w = 1 - T_final.
// and DropGaussian (params.drop_rate, project_splats, tile_forward): at rate 0.5 about half the splats
//   of a render are culled; the opacity factor 1 / (1 - rate) doubles a sheet's alpha (0.3 -> 0.6).
//   cd e2e && node ssim.e2e.js   (env: see harness.js)
import { APP, PORT, fail, urlOf, run } from './harness.js';
import { join } from 'node:path';

await run(async (page) => {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'DESIGN.md'))}`);  // same origin, no app (and no dataset load)
  const r = await page.evaluate(async () => {
    const { GaussOptTrainer } = await import('./src/trainer.js');
    const { LooseOctree } = await import('./src/octree.js');
    const { setPose, SPLAT_FLOATS } = await import('./src/colmap.js');
    const { makeRng } = await import('./src/math.js');
    const W = 64, H = 64;  // the smallest training size; 64 x 16 B: rows of ssim_grad_tex 256-byte aligned
    const target = new Uint8Array(W * H * 4).map((_, i) => i % 4 === 3 ? 255 : 128 + 100 * Math.sin(i * 0.013 + (i % 4)));

    // One training render of `splats` ([camera-frame xyz, logit opacity, log scales xyz, rgb dc], camera at
    // the origin looking down +z) with D-SSIM weight `ssim`, DropGaussian [rate, opacity factor] `drop`,
    // adam_step skipped. Returns grads_2d, pix_state, the render (hat_tex, 8-bit), ssim_grad_tex and the
    // number of splats drawn.
    const render = async (splats, ssim, drop = [0, 0]) => {
      const device = await (await navigator.gpu.requestAdapter()).requestDevice();
      const N = splats.length, params = new Float32Array(N * SPLAT_FLOATS);
      splats.forEach(([xc, op, ls, rgb], i) => params.set([...xc, op, ...ls, 0, 1, 0, 0, 0, ...rgb, 0], i * SPLAT_FLOATS));
      const img = setPose({ name: 'v0', isValidation: false, zMin: 1, zMax: 8,
                            camera: { width: W, height: H, fx: 50, fy: 50, cx: W / 2, cy: H / 2 } }, [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);
      const octree = new LooseOctree({ maxLeafSize: 64 }).build(params, N, [img]);
      const tr = await GaussOptTrainer.create(device, octree, { count: N, params }, [img], { downsample: 1, ssim });
      tr._loadGtTexture = async () => {
        const tex = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        device.queue.writeTexture({ texture: tex }, target, { bytesPerRow: W * 4 }, [W, H]);
        return tex;
      };
      const k = tr._intrinsics(img), gt = await tr.ensureGtTexture(0);
      tr.updateSortingForView(img.center, img.view, k.fxNorm, k.fyNorm);
      tr._writeUniforms(img, k, [0, 0, 0, 0, 0, 0, 0, 1, 0.03, 1, Math.log(10), 0, ssim, drop[0], drop[1], 0]);
      const pass = tr._pass;
      tr._pass = function (enc, pipe, ...a) { if (pipe !== this.pipeAdam) pass.call(this, enc, pipe, ...a); };
      tr._runPasses(gt);
      const [g, pix, hat] = await tr._readback('out', [{ buf: tr.grads2dBuf, bytes: N * 48 }, { buf: tr.pixStateBuf, bytes: W * H * 16 },
                                                       { tex: tr.hatTex }]);
      const staging = device.createBuffer({ size: W * H * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tr.ssimGradTex }, { buffer: staging, bytesPerRow: W * 16 }, [W, H]);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const G = new Float32Array(staging.getMappedRange().slice(0));
      const [proj] = await tr._readback('proj', [{ buf: tr.projSplatsBuf, bytes: tr.numVisible * 32 }]);
      const drawn = Array.from({ length: tr.numVisible }, (_, k) => proj[8 * k + 3] > 0).filter(Boolean).length;
      const out = { g: Array.from(g), pix: Array.from(pix), hat: Array.from(hat), G: Array.from(G), bpr: tr.rowPitch, drawn };
      device.destroy();
      return out;
    };

    const rng = makeRng(11), random = [];
    for (let i = 0; i < 600; i++) {
      const z = 2 + 4 * rng(), s = Math.log(0.02 + 0.08 * rng());
      random.push([[(rng() - 0.5) * 1.4 * z, (rng() - 0.5) * z, z], 3 * rng() - 1, [s, s + rng() - 0.5, s],
                   [2 * rng() - 1, 2 * rng() - 1, 2 * rng() - 1]]);
    }
    const tex = await render(random, 1);
    const sheet = await render([[[0, 0, 3], Math.log(0.95 / 0.05), [Math.log(20), Math.log(20), Math.log(1e-3)], [0.1, -0.2, 0.3]]], 1);
    const dropped = [await render(random, 0), await render(random, 0, [0.5, 2])];
    const faint = (f) => render([[[0, 0, 3], Math.log(0.3 / 0.7), [Math.log(20), Math.log(20), Math.log(1e-3)], [0, 0, 0]]], 0, [0, f]);
    const alpha = [(await faint(0)).pix, (await faint(2)).pix].map(pix => 1 - pix[4 * (32 * W + 32)]);
    return { W, H, target: Array.from(target), tex, sheet, drawn: dropped.map(d => d.drawn), alpha };
  });

  const { ssimSum } = await import('../src/metrics.js');
  const { W, H, target, tex, sheet, drawn, alpha } = r, N = W * H;
  const plane = (src, c, bpr) => Float64Array.from({ length: N }, (_, p) => src[(p / W | 0) * bpr + (p % W) * 4 + c] / 255);

  // G(p) = d(sum_q S(q))/dy(p) per channel, against central differences
  {
    const pixels = [0, W - 1, N - W, N - 1, 3, 5 * W, 7 * W + 9, 20 * W + 31, 30 * W + 50, 47 * W + 30, 24 * W + 63, 12 * W + 4];
    let maxErr = 0, maxG = 0, bad = [];
    for (let c = 0; c < 3; c++) {
      const x = plane(target, c, W * 4), y = plane(tex.hat, c, tex.bpr);
      for (const p of pixels) {
        const eps = 1e-3, y0 = y[p];
        y[p] = y0 + eps; const up = ssimSum(x, y, W, H);
        y[p] = y0 - eps; const dn = ssimSum(x, y, W, H);
        y[p] = y0;
        const want = (up - dn) / (2 * eps), got = tex.G[4 * p + c];
        maxG = Math.max(maxG, Math.abs(want));
        maxErr = Math.max(maxErr, Math.abs(got - want));
        if (!(Math.abs(got - want) < 2e-3 * Math.max(1, Math.abs(want)))) bad.push(`pixel ${p % W},${p / W | 0} channel ${c}: ${got}, expected ${want}`);
      }
    }
    console.log(`ssim_grad: ${pixels.length * 3} samples, |dS/dy| up to ${maxG.toFixed(3)}, max error ${maxErr.toExponential(2)}`);
    if (!(maxG > 0.1)) fail(`ssim_grad: the samples' gradients are all near 0 (max ${maxG})`);
    if (bad.length) fail(`ssim_grad: ${bad.length} samples off, e.g.\n  ${bad.slice(0, 4).join('\n  ')}`);
  }
  // tile_backward, weight 1: dL/dC(p) = -(1/2) G(p) / N, through one splat of weight 1 - T_final
  {
    const SH_C0 = 0.28209479177387814;
    for (let c = 0; c < 3; c++) {
      let want = 0;
      for (let p = 0; p < N; p++) want += -0.5 * sheet.G[4 * p + c] / N * (1 - sheet.pix[4 * p]) * SH_C0;
      const got = sheet.g[c];
      console.log(`sheet dL/drgb[${c}]: ${got.toExponential(4)} (want ${want.toExponential(4)})`);
      if (!(Math.abs(want) > 1e-6)) fail(`sheet dL/drgb[${c}]: expected gradient ${want} is ~0, no test`);
      else if (!(Math.abs(got - want) < 1e-3 * Math.abs(want))) fail(`sheet dL/drgb[${c}]: ${got}, expected ${want} (0.1%)`);
    }
  }
  // DropGaussian
  {
    const kept = drawn[1] / drawn[0];
    console.log(`DropGaussian 0.5: ${drawn[1]} of ${drawn[0]} splats drawn (${(100 * kept).toFixed(1)}%); opacity factor 2: alpha ${alpha[0].toFixed(3)} -> ${alpha[1].toFixed(3)}`);
    if (!(drawn[0] > 300 && Math.abs(kept - 0.5) < 0.06)) fail(`DropGaussian 0.5: ${drawn[1]} of ${drawn[0]} drawn`);
    if (!(Math.abs(alpha[0] - 0.3) < 0.01 && Math.abs(alpha[1] - 0.6) < 0.01)) fail(`DropGaussian factor 2: alpha ${alpha[0]} -> ${alpha[1]}, expected 0.3 -> 0.6`);
  }
  return 'ssim_grad = central differences of ssimSum; tile_backward takes it as the D-SSIM color gradient; ' +
         'DropGaussian culls about half the splats at 0.5 and scales opacities by its factor';
});
