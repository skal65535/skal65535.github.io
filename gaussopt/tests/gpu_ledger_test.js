// gpu_ledger_test.js — The GPU ledger (src/sfm/gpu_ledger.js): the bytes made, less those destroyed, by the site that
// made them.
import { gpuLedger } from '../src/sfm/gpu_ledger.js';
import { assert, hasGPU } from './util.js';

Deno.test({ name: 'gpu ledger: buffers and textures held, destroyed, by their site', ignore: !hasGPU, fn: async () => {
  const device = await (await navigator.gpu.requestAdapter()).requestDevice();
  const ledger = gpuLedger(() => {}, 0), before = ledger.held();
  function makeThree() {  // the site: its name in the line
    const out = [];
    for (const m of [3, 5, 2]) out.push(device.createBuffer({ size: m * 2 ** 20, usage: GPUBufferUsage.STORAGE }));
    return out;
  }
  const bufs = makeThree();
  const tex = device.createTexture({ size: [1024, 512], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
  // a texel's bytes by its format: r32float 4, rgba16float 8 (the 'a' of 'float' not a channel)
  const r32 = device.createTexture({ size: [256, 256], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING });
  const h16 = device.createTexture({ size: [256, 256], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING });
  assert(ledger.held() - before === 10 * 2 ** 20 + 2 * 2 ** 20 + 65536 * 12, `${ledger.held() - before} bytes held`);
  r32.destroy(); h16.destroy();
  bufs[1].destroy();
  tex.destroy();
  bufs[1].destroy();  // twice: counted once
  assert(ledger.held() - before === 5 * 2 ** 20, `${ledger.held() - before} bytes held after the destroys`);
  assert(/5 MB in 2 at makeThree/.test(ledger.line()), ledger.line());
  for (const b of bufs) b.destroy();
  device.destroy();
}});
