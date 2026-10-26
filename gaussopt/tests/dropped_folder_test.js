// loadColmapFromFiles (colmap.js): a dropped folder's files, found by relative path or bare name
import { loadColmapFromFiles } from '../src/colmap.js';
import { assert } from './util.js';

// A File as a dropped folder gives it: name, path under the dropped root, contents
const file = (rel, bytes = new ArrayBuffer(8)) =>
  ({ name: rel.split('/').pop(), webkitRelativePath: `scene/${rel}`, arrayBuffer: async () => bytes });

Deno.test('a dropped folder lists each checkpoint once, by its relative path (meshes are not checkpoints)', async () => {
  const empty = new ArrayBuffer(8);  // no cameras / no images (uint64 count 0)
  const files = [file('sparse/cameras.bin', empty), file('sparse/images.bin', empty),
                 file('splats/a.ply'), file('b.splat'), file('splats/gaussopt_01000.mesh.ply')];
  const { availablePlys } = await loadColmapFromFiles(files);
  const names = availablePlys.map(p => p.name).sort().join();
  assert(names === 'b.splat,splats/a.ply', `checkpoints: ${names}`);
});
