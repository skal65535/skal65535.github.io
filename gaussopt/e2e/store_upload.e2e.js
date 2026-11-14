// store_upload.e2e.js — Large files into a serve.py project (src/project/store.js ServerStore.write) under Playwright,
// as train.mjs's runs write their models and features:
//   - 150 MB of an ArrayBuffer (a COLMAP images.bin of 674 keyframes is 114 MB), written whole, the run going on after
//     it (an ArrayBuffer body's bytes went into the network events, a message too long for Playwright's string,
//     which stopped the run);
//   - a container of 1.1 GB (project.js packPieces; the features of 674 keyframes are about that), written whole and
//     read back (one Blob of it could not be sent: Chrome fails a fetch of a Blob of 800 MB).
//   cd e2e && node store_upload.e2e.js
import { mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { appUrl, fail, run } from './harness.js';

const projects = mkdtempSync(join(tmpdir(), 'gaussopt-upload-'));
await run(async (page) => {
  await page.goto(appUrl('?dataset=[synth]'));
  const r = await page.evaluate(async () => {
    const { ServerStore } = await import('./src/project/store.js'), { packPieces, unpackProject } = await import('./src/sfm/project.js');
    const store = new ServerStore('upload'), a = new Uint8Array(150e6);
    for (let i = 0; i < a.length; i += 4096) a[i] = i & 255;
    await store.write('cache/big.bin', a.buffer);
    const parts = {};  // 11 parts of 100 MB, each its index all through
    for (let k = 0; k < 11; k++) parts[`p${k}`] = new Uint8Array(100e6).fill(k);
    await store.write('cache/features.gosfm', packPieces({ parts: 11 }, parts));
    const back = await unpackProject(await store.read('cache/features.gosfm')), last = await back.get('p10');
    return { json: back.json, last: [last.length, last[0], last.at(-1)] };
  });
  const size = (f) => statSync(join(projects, 'upload', 'cache', f)).size;
  if (size('big.bin') !== 150e6) fail(`cache/big.bin: ${size('big.bin')} bytes, 150 MB written`);
  if (r.json.parts !== 11 || r.last.join() !== '100000000,10,10') fail(`features.gosfm read back: ${JSON.stringify(r)}`);
  return `150 MB written; a container of ${(size('features.gosfm') / 1e6).toFixed(0)} MB written and read back`;
}, { serverArgs: ['--projects', projects], cleanup: () => rmSync(projects, { recursive: true, force: true }) });
