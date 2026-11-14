// write_pieces.e2e.js — Uploads of growing sizes through common/webgpu.js writeInPieces leave Chrome's memory as it
// was: 64 writes of 1 to 64 MB into one buffer (as the GPU bundle adjustment's windows grow), the GPU process's and
// the page's footprint (macOS footprint) at most 400 MB above where they started, the 64 MB buffer included (whole
// writes: 1.1 GB more in each).
//   cd e2e && node write_pieces.e2e.js
import { execFileSync } from 'node:child_process';
import { appUrl, browser, fail, NAME, server } from './harness.js';

const srv = await server();
const { page, close, profile } = await browser();
// the footprint (MB) of this browser's GPU process and page
const footprint = () => {
  const out = { gpu: 0, page: 0 };
  for (const l of execFileSync('ps', ['-Ao', 'pid=,args=']).toString().split('\n')) {
    if (!l.includes(profile) || !/--type=(renderer|gpu-process)/.test(l) || /--top-chrome-webui/.test(l)) continue;
    const pid = l.trim().split(/\s+/)[0], f = execFileSync('footprint', ['-f', 'bytes', pid]).toString();
    const mb = Number(/phys_footprint: (\d+)/.exec(f)?.[1] ?? 0) / 2 ** 20, k = /--type=gpu/.test(l) ? 'gpu' : 'page';
    out[k] = Math.max(out[k], mb);
  }
  return out;
};
try {
  await page.goto(appUrl('?dataset=[synth]'));
  await page.waitForFunction(() => !!window.app, null, { timeout: 120000 });
  const before = footprint();
  await page.evaluate(async () => {
    const { writeInPieces } = await import(new URL('../common/webgpu.js', location.href).href);
    const device = await (await navigator.gpu.requestAdapter()).requestDevice({ requiredLimits: { maxBufferSize: 2 ** 27 } });
    const buf = device.createBuffer({ size: 2 ** 26 + 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    for (let k = 1; k <= 64; k++) {
      writeInPieces(device.queue, buf, 0, new Uint8Array(k * 2 ** 20 + 16 * k));  // a size of its own each time
      await device.queue.onSubmittedWorkDone();
    }
    window.__device = device;  // kept: its memory counted after
  });
  await new Promise(r => setTimeout(r, 3000));
  const after = footprint(), more = { gpu: after.gpu - before.gpu, page: after.page - before.page };
  if (more.gpu > 400 || more.page > 400) fail(`64 writes of 1 to 64 MB: the GPU process ${more.gpu.toFixed(0)} MB more, the page ${more.page.toFixed(0)} MB`);
  else console.log(`PASS in ${NAME}: the GPU process ${more.gpu.toFixed(0)} MB more, the page ${more.page.toFixed(0)} MB`);
} finally {
  await close();
  srv.kill();
}
