// Project folders (src/project/): the memory store and the .zip of a project, which
// Python's zipfile must read back.
import { MemoryStore } from '../src/project/store.js';
import { zipBlob, crc32 } from '../src/project/zip.js';
import { assert, env, SLOW } from './util.js';

async function sampleProject() {
  const store = new MemoryStore('p');
  await store.write('project.json', '{"format":"gaussopt-project/1"}');
  await store.write('images/t0000000.jpg', new Uint8Array([0xff, 0xd8, 1, 2, 3, 0xff, 0xd9]));
  await store.write('gs/sparse/cameras.bin', new ArrayBuffer(24));
  assert(JSON.stringify(await store.list()) === JSON.stringify([{ name: 'gs', kind: 'directory' }, { name: 'images', kind: 'directory' },
                                                                 { name: 'project.json', kind: 'file' }]), 'list');
  assert(JSON.stringify((await store.list('gs/sparse')).map(e => e.name)) === '["cameras.bin"]', 'list of a subfolder');
  return store;
}

Deno.test('project: memory store listing, CRC-32', async () => {
  assert(crc32(new TextEncoder().encode('hello')) === 0x3610a686, 'crc32("hello")');
  await sampleProject();
});

// (slow: runs python3)
Deno.test({ name: 'zip: Python zipfile reads a project .zip back', ignore: !SLOW, fn: async () => {
  const store = await sampleProject();
  const zip = new Uint8Array(await (await zipBlob([...store.files])).arrayBuffer());
  const dir = await Deno.makeTempDir();
  try {
    await Deno.writeFile(`${dir}/p.zip`, zip);
    const py = 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None; print(",".join(f"{i.filename}:{i.file_size}" for i in z.infolist()))';
    const out = await new Deno.Command(env('PYTHON') ?? 'python3', { args: ['-I', '-c', py, `${dir}/p.zip`] }).output();
    const listing = new TextDecoder().decode(out.stdout).trim();
    assert(out.success && listing === 'project.json:31,images/t0000000.jpg:7,gs/sparse/cameras.bin:24', `zipfile: ${listing} ${new TextDecoder().decode(out.stderr)}`);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
} });
