// xfer.e2e.js — the trainer's transfers in pieces (trainer.xferBytes: _readback's staging, _write's writes, _clear):
// the same bytes as in one piece. 5000 random splats on a new trainer; their splats, moments and Adam m / v written
// in pieces of 3996 bytes (a row is 64) and read back in one, then in pieces of 4004 and of 70004 with a buffer of
// 12 bytes and the render (hatTex; in the larger pieces in the same staging as the 12 bytes) among them;
// _uploadSplats: the splats as cpuParams, the moments, 2D grads and Adam state 0 on the GPU.
//   cd e2e && node xfer.e2e.js   (env: see harness.js)
import { APP, PORT, fail, urlOf, run } from './harness.js';
import { join } from 'node:path';

await run(async (page) => {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'DESIGN.md'))}`);  // same origin, no app (and no dataset load)
  const r = await page.evaluate(async () => {
    const { GaussOptTrainer } = await import('./src/trainer.js');
    const { LooseOctree, MOMENT_FLOATS: M } = await import('./src/octree.js');
    const { setPose, SPLAT_FLOATS } = await import('./src/colmap.js');
    const { makeRng } = await import('./src/math.js');
    const W = 128, H = 96, N = 5000, rng = makeRng(7);
    const device = await (await navigator.gpu.requestAdapter()).requestDevice();
    const params = new Float32Array(N * SPLAT_FLOATS);
    for (let i = 0; i < N; i++) {
      params.set([2 * (rng() - 0.5), 2 * (rng() - 0.5), 3 + 2 * rng(), Math.log(0.5),
                  Math.log(0.05), Math.log(0.05), Math.log(0.05), 0, 1, 0, 0, 0, rng(), rng(), rng(), 0], i * SPLAT_FLOATS);
    }
    const camera = { width: W, height: H, fx: 100, fy: 100, cx: W / 2, cy: H / 2 };
    const img = setPose({ name: 'v0', isValidation: false, zMin: 1, zMax: 8, camera }, [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);
    const octree = new LooseOctree({ maxLeafSize: 64 }).build(params, N, [img]);
    const tr = await GaussOptTrainer.create(device, octree, { count: N, params }, [img], { downsample: 1 });
    tr._loadGtTexture = async () => device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const rand = (n) => Float32Array.from({ length: n }, () => rng() * 2 - 1);
    const same = (a, b) => { if (a.length !== b.length) return `length ${a.length} vs ${b.length}`;
      const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
      for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return `differs at float ${i} of ${x.length}`; return null; };
    const want = { splats: rand(N * SPLAT_FLOATS), moments: rand(N * M), adamM: rand(N * SPLAT_FLOATS), adamV: rand(N * SPLAT_FLOATS) };
    tr.xferBytes = 3996;
    tr._write(tr.splatsBuf, 0, want.splats);
    tr._write(tr.momentsBuf, 0, want.moments);
    tr._write(tr.adamMBuf, 0, want.adamM);
    tr._write(tr.adamVBuf, 0, want.adamV);
    tr.xferBytes = 1 << 30;
    const one = await tr._readSplatState('one');
    const out = {};
    ['splats', 'moments', 'adamM', 'adamV'].forEach((k, i) => { out[`written in pieces, ${k}`] = same(one[i], want[k]); });
    await tr.readView(0, ['hat']);  // a render in hatTex
    const small = { buf: tr.lossBuf, bytes: 12 };
    const [s1, t1, ...rest1] = await tr._readback('whole', [small, { tex: tr.hatTex }, { buf: tr.splatsBuf, bytes: N * SPLAT_FLOATS * 4 }, { buf: tr.momentsBuf, bytes: N * M * 4 }]);
    tr.xferBytes = 4004;
    const [s2, t2, ...rest2] = await tr._readback('pieces', [small, { tex: tr.hatTex }, { buf: tr.splatsBuf, bytes: N * SPLAT_FLOATS * 4 }, { buf: tr.momentsBuf, bytes: N * M * 4 }]);
    out['read in pieces, the 12 bytes'] = same(s1, s2);
    out['read in pieces, the render'] = t1.every((v, i) => v === t2[i]) ? null : 'differs';
    out['read in pieces, the splats'] = same(rest1[0], rest2[0]);
    out['read in pieces, the moments'] = same(rest1[1], rest2[1]);
    tr.xferBytes = 70004;  // more than the render's 49152 bytes: it follows the 12 in one staging
    const [s3, t3, ...rest3] = await tr._readback('mid', [small, { tex: tr.hatTex }, { buf: tr.splatsBuf, bytes: N * SPLAT_FLOATS * 4 }, { buf: tr.momentsBuf, bytes: N * M * 4 }]);
    out['read in larger pieces, the 12 bytes'] = same(s1, s3);
    out['read in larger pieces, the render'] = t1.every((v, i) => v === t3[i]) ? null : 'differs';
    out['read in larger pieces, the splats'] = same(rest1[0], rest3[0]);
    out['read in larger pieces, the moments'] = same(rest1[1], rest3[1]);
    let lit = 0; for (let i = 3; i < t1.length; i += 4) lit += t1[i] > 0 ? 1 : 0;
    tr._uploadSplats();
    tr.xferBytes = 1 << 30;
    const up = await tr._readSplatState('up');
    const [g2d] = await tr._readback('g2d', [{ buf: tr.grads2dBuf, bytes: N * M * 4 }]);
    out['upload: the splats as cpuParams'] = same(up[0], tr.cpuParams.subarray(0, N * SPLAT_FLOATS));
    const zero = (a) => a.every(v => v === 0) ? null : `${a.filter(v => v !== 0).length} not 0`;
    out['upload: the moments 0'] = zero(up[1]);
    out['upload: Adam m 0'] = zero(up[2]);
    out['upload: Adam v 0'] = zero(up[3]);
    out['upload: 2D grads 0'] = zero(g2d);
    device.destroy();
    return { out, lit };
  });
  if (!(r.lit > 0)) fail('the render has no lit pixel: its readback untested');
  for (const [k, v] of Object.entries(r.out)) if (v) fail(`${k}: ${v}`);
  return `${Object.keys(r.out).length} checks the same in pieces as in one (the render ${r.lit} pixels lit)`;
});
