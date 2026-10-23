// tiles.e2e.js — The tile rasterizer kernels on synthetic scenes (Deno's WebGPU has no
// 'subgroups', hence a browser test):
//   - tile_backward with subgroups vs without (the per-thread path, lines tagged @nosg): one
//     render on two devices, same splats and target; grads_2d and the WLBG moments agree slot
//     by slot (only the float summation order differs);
//   - tile_forward's depth spread (pix_state.w) on large flat splats: one near-opaque sheet,
//     two sheets apart, and two sheets close.
//   cd e2e && node tiles.e2e.js   (env: see harness.js)
import { APP, PORT, NAME, fail, urlOf, server, browser } from './harness.js';
import { join } from 'node:path';

const srv = await server();
const { page, close, errors } = await browser();
try {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'DESIGN.md'))}`);  // same origin, no app (and no dataset load)
  const r = await page.evaluate(async () => {
    const { GaussOptTrainer } = await import('./src/trainer.js');
    const { LooseOctree } = await import('./src/octree.js');
    const { setPose, SPLAT_FLOATS } = await import('./src/colmap.js');
    const { makeRng } = await import('./src/math.js');
    const W = 256, H = 192;
    // a camera turned off the world axes (every moment slot non-zero)
    const [ca, sa, cb, sb] = [Math.cos(0.35), Math.sin(0.35), Math.cos(0.25), Math.sin(0.25)];
    const R = [cb, sa * sb, -ca * sb, 0, ca, sa, sb, -sa * cb, ca * cb];  // world -> camera, orthonormal
    const toWorld = (xc) => [0, 1, 2].map(j => R[j] * xc[0] + R[3 + j] * xc[1] + R[6 + j] * xc[2]);  // t = 0
    const target = new Uint8Array(W * H * 4).map((_, i) => i % 4 === 3 ? 255 : 128 + 100 * Math.sin(i * 0.013 + (i % 4)));

    // One render of `splats` ([camera-frame xyz, logit opacity, log scales xyz, rgb dc]) on a
    // new device; backward: also tile_backward (adam_step skipped: it would consume grads_2d)
    const render = async (splats, { features = [], backward = false } = {}) => {
      const device = await (await navigator.gpu.requestAdapter()).requestDevice({ requiredFeatures: features });
      const N = splats.length, params = new Float32Array(N * SPLAT_FLOATS);
      splats.forEach(([xc, op, ls, rgb], i) => params.set([...toWorld(xc), op, ...ls, 0, 1, 0, 0, 0, ...rgb, 0], i * SPLAT_FLOATS));
      const img = setPose({ name: 'v0', isValidation: false, zMin: 1, zMax: 8,
                            camera: { width: W, height: H, fx: 200, fy: 200, cx: W / 2, cy: H / 2 } }, R, [0, 0, 0]);
      const octree = new LooseOctree({ maxLeafSize: 64 }).build(params, N, [img]);
      const tr = await GaussOptTrainer.create(device, octree, { count: N, params }, [img], { downsample: 1 });
      tr._loadGtTexture = async () => {
        const tex = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        device.queue.writeTexture({ texture: tex }, target, { bytesPerRow: W * 4 }, [W, H]);
        return tex;
      };
      const k = tr._intrinsics(img), gt = await tr.ensureGtTexture(0);
      tr.updateSortingForView(img.center, img.view, k.fxNorm, k.fyNorm);
      tr._writeUniforms(img, k, [0, 0, 0, 0, 0, 0, 0, 1, 0.03, 1, Math.log(10), 0]);
      const pass = tr._pass;
      tr._pass = function (enc, pipe, ...a) { if (pipe !== this.pipeAdam) pass.call(this, enc, pipe, ...a); };
      tr._runPasses(backward ? gt : null);
      const [g, m, pix] = await tr._readback('out', [{ buf: tr.grads2dBuf, bytes: N * 48 }, { buf: tr.momentsBuf, bytes: N * 48 },
                                                     { buf: tr.pixStateBuf, bytes: W * H * 16 }]);
      const out = { sg: device.features.has('subgroups'), visible: tr.numVisible, g: Array.from(g), m: Array.from(m), pix };
      device.destroy();
      return out;
    };

    // backward: overlapping splats at depths 2..6, many hits per tile and batch
    const rng = makeRng(7), random = [];
    for (let i = 0; i < 4000; i++) {
      const z = 2 + 4 * rng(), s = Math.log(0.02 + 0.06 * rng());
      random.push([[(rng() - 0.5) * 1.4 * z, (rng() - 0.5) * z, z], 2 * rng() - 1, [s, s + rng() - 0.5, s],
                   [3 * rng() - 1.5, 3 * rng() - 1.5, 3 * rng() - 1.5]]);
    }
    const a = await render(random, { features: ['subgroups'], backward: true });
    const b = await render(random, { backward: true });
    const backward = { sgA: a.sg, sgB: b.sg, visible: a.visible, slots: [] };
    // slot c of every splat: relative L2 difference; grads slot 11 and moments slot 1 (S) stay 0
    for (const [name, x, y, slots] of [['grads_2d', a.g, b.g, 11], ['moments', a.m, b.m, 12]]) {
      for (let c = 0; c < slots; c++) {
        if (name === 'moments' && c === 1) continue;
        let d = 0, n = 0;
        for (let i = c; i < x.length; i += 12) { d += (x[i] - y[i]) ** 2; n += y[i] ** 2; }
        backward.slots.push({ name: `${name}[${c}]`, norm: n, rel: Math.sqrt(d / n) });
      }
    }

    // depth spread: sheets = one splat each, 10 m wide and 1 mm thick (alpha ~ its opacity on
    // every pixel); spread over the pixels where it is measured (T_final < 0.25)
    const sheet = (z, opacity) => [[0, 0, z], Math.log(opacity / (1 - opacity)), [Math.log(10), Math.log(10), Math.log(1e-3)], [0, 0, 0]];
    const spread = {};
    for (const [name, splats] of [['one sheet', [sheet(2, 0.99)]], ['2 and 4', [sheet(2, 0.6), sheet(4, 0.6)]],
                                  ['2 and 2.2', [sheet(2, 0.6), sheet(2.2, 0.6)]], ['T > 0.25', [sheet(2, 0.6)]]]) {
      const { pix } = await render(splats);
      let lo = Infinity, hi = -Infinity, measured = 0;
      for (let p = 0; p < pix.length; p += 4) {
        if (pix[p] >= 0.25) continue;
        measured++; lo = Math.min(lo, pix[p + 3]); hi = Math.max(hi, pix[p + 3]);
      }
      spread[name] = { measured, lo, hi };
    }
    return { backward, spread, pixels: W * H };
  });

  const bw = r.backward;
  if (!bw.sgA || bw.sgB) throw new Error(`devices: subgroups ${bw.sgA} / ${bw.sgB}, expected true / false`);
  console.log(`backward: ${bw.visible} visible splats`);
  for (const s of bw.slots) {
    if (!(s.norm > 0)) fail(`${s.name} is 0 everywhere without subgroups`);
    else if (!(s.rel < 1e-4)) fail(`${s.name}: subgroups differ by ${s.rel.toExponential(2)} (relative L2)`);
  }
  // (z_q3 - z_q1) / z_med: 0 on one sheet, (4 - 2) / 2 and (2.2 - 2) / 2 on two
  for (const [name, want, all] of [['one sheet', 0, true], ['2 and 4', 1, true], ['2 and 2.2', 0.1, true], ['T > 0.25', null, false]]) {
    const s = r.spread[name];
    console.log(`spread ${name}: ${s.measured} / ${r.pixels} pixels measured, in [${s.lo}, ${s.hi}]`);
    if (!all) { if (s.measured) fail(`spread ${name}: ${s.measured} pixels measured, expected none`); continue; }
    if (s.measured !== r.pixels) fail(`spread ${name}: ${s.measured} / ${r.pixels} pixels measured`);
    else if (!(Math.abs(s.lo - want) < 1e-3 && Math.abs(s.hi - want) < 1e-3)) fail(`spread ${name}: [${s.lo}, ${s.hi}], expected ${want}`);
  }
  const errs = await errors();
  if (errs.length) fail(`page errors:\n${errs.join('\n')}`);
  if (!process.exitCode) console.log(`OK (${NAME}): tile_backward with subgroups = without on every slot; depth spread of flat sheets`);
} catch (e) {
  fail(e.message);
} finally {
  await close();
  srv.kill();
}
