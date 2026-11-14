// store.js writes: a container's pieces (project.js packPieces) into serve.py by requests of PIECE bytes at most, in
// order (?append=offset, &last=1 on the last), each its bytes (no Blob: Chrome's blob storage fills), never all in one;
// the same container as packProject's Blob.
import { ServerStore, MemoryStore } from '../src/project/store.js';
import { packProject, packPieces, unpackProject } from '../src/sfm/project.js';
import { assert } from './util.js';

const PIECE = 32 * 2 ** 20;

Deno.test('store: a container of 100 MB into serve.py in pieces, in order, the same bytes', async () => {
  const big = new Uint8Array(70e6).map((_, k) => k % 251), small = Float32Array.from({ length: 1000 }, (_, k) => k);
  const parts = { big, small, jpeg: new Blob([new Uint8Array(30e6).fill(7)], { type: 'image/jpeg' }) };
  const sent = [], real = globalThis.fetch;
  globalThis.fetch = async (url, { method, body }) => { sent.push({ url, method, body }); return new Response('{}'); };
  try {
    await new ServerStore('p').write('cache/features.gosfm', packPieces({ a: 1 }, parts));
  } finally {
    globalThis.fetch = real;
  }
  const whole = new Uint8Array(await packProject({ a: 1 }, parts).arrayBuffer());
  const at = sent.map(s => Number(new URL(s.url, 'http://x').searchParams.get('append')));
  assert(sent.length === Math.ceil(whole.length / PIECE) && sent.every(s => s.body instanceof Uint8Array && s.body.length <= PIECE), sent.map(s => s.body.length).join());
  assert(at.every((a, k) => a === k * PIECE) && sent.map(s => s.url.endsWith('&last=1')).join() === sent.map((_, k) => k === sent.length - 1).join(), sent.map(s => s.url).join(' '));
  const got = new Uint8Array(await new Blob(sent.map(s => s.body)).arrayBuffer());
  assert(got.length === whole.length && got.every((x, k) => x === whole[k]), 'the bytes sent are not the container');
  // small: one request, as before
  sent.length = 0;
  globalThis.fetch = async (url, { body }) => { sent.push({ url, body }); return new Response('{}'); };
  try { await new ServerStore('p').write('a.bin', packPieces({}, { small })); } finally { globalThis.fetch = real; }
  assert(sent.length === 1 && !sent[0].url.includes('append'), sent.map(s => s.url).join());
});

Deno.test('store: pieces into memory, read back as the container', async () => {
  const s = new MemoryStore('m'), small = Float32Array.from({ length: 10 }, (_, k) => k / 2);
  await s.write('x.gosfm', packPieces({ b: 2 }, { small }));
  const back = await unpackProject(await s.read('x.gosfm'));
  assert(back.json.b === 2 && (await back.get('small'))[3] === 1.5, JSON.stringify(back.json));
});
