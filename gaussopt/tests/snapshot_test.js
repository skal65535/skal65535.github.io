// snapshot.js and scene_glb.js training: a snapshot is a scene .glb with its training state, read back bit for bit (the
// splats as trained, raw: leaf ids and flags too; the optimizer's moments; the leaf permutation as integers), its counters
// and settings, and the scene's octree and leveling; the same file is still a scene a viewer reads (its splats as
// KHR_gaussian_splatting). A scene saved without it has no snapshot. An older .gosnap converted (gosnapToSceneGlb) gives
// the same snapshot as the .gosnap itself; the kind is told by the first bytes.
import { writeSceneGlb, readSceneGlb, TRAINING_FORMAT } from '../src/scene_glb.js';
import { encodeSnapshot, decodeSnapshot, snapshotKind, snapshotOfScene, gosnapToSceneGlb } from '../src/snapshot.js';
import { parseGlb, readAccessor } from '../src/glb.js';
import { LooseOctree } from '../src/octree.js';
import { SPLAT_FLOATS } from '../src/colmap.js';
import { makeRng, levelTransform, IDENTITY4 } from '../src/math.js';
import { assert } from './util.js';

const rng = makeRng(11), N = 500, MOMENTS = 6;
const floats = (n, f = () => rng() * 2 - 1) => Float32Array.from({ length: n }, f);
function state() {
  const params = floats(N * SPLAT_FLOATS);
  for (let i = 0; i < N; i++) { params[i * SPLAT_FLOATS + 7] = i % 37; params[i * SPLAT_FLOATS + 15] = 3; }  // leaf id, flags
  const octree = new LooseOctree({ maxLeafSize: 32, maxDepth: 5, beta: 0.3 });
  octree.build(params, N);
  const blobs = { params, moments: floats(N * MOMENTS), adam_m: floats(N * SPLAT_FLOATS), adam_v: floats(N * SPLAT_FLOATS, () => rng() * 1e-6),
                  leaf_idx: Uint32Array.from({ length: N }, (_, i) => (i * 7919) % N) };
  const header = { savedAt: '2026-11-06T13:00:00.000Z', dataset_url: '/dataset', numImages: 12, firstImage: 'f000.jpg', splatFloats: SPLAT_FLOATS,
                   params: { lr_pos: 0.00016, max_splats: 300000 }, state: { count: N, stepCount: 6000, adamStepCount: 5990, lastPsnr: 21.5, history: [[5000, 20.9]] } };
  return { params, octree, blobs, header, transform: levelTransform(IDENTITY4, [0, 0, 1], 7) };
}
const same = (a, b) => a.constructor === b.constructor && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

Deno.test('snapshot: a scene .glb with its training state, read back bit for bit; still a scene for a viewer', () => {
  const s = state();
  const bytes = writeSceneGlb({ count: N, params: s.params, octree: s.octree, transform: s.transform, training: { header: s.header, blobs: s.blobs } });
  assert(snapshotKind(bytes.buffer) === 'glb', `kind ${snapshotKind(bytes.buffer)}`);
  const r = readSceneGlb(bytes.buffer), snap = snapshotOfScene(r);
  assert(snap, 'no snapshot in the scene');
  for (const [k, a] of Object.entries(s.blobs)) assert(same(snap.blobs[k], a), `blob ${k}: ${snap.blobs[k]?.constructor.name} of ${snap.blobs[k]?.length}, not the same`);
  assert(JSON.stringify(snap.header.state) === JSON.stringify(s.header.state) && JSON.stringify(snap.header.params) === JSON.stringify(s.header.params) &&
         snap.header.numImages === 12 && snap.header.firstImage === 'f000.jpg' && snap.header.format === TRAINING_FORMAT, `header ${JSON.stringify(snap.header).slice(0, 300)}`);
  assert(JSON.stringify(snap.header.transform) === JSON.stringify(s.transform), `transform ${snap.header.transform}`);
  const oct = s.octree.toJSON();
  assert(snap.header.octree && JSON.stringify(snap.header.octree.nodes) === JSON.stringify(oct.nodes) && snap.header.octree.leafIndices === null, 'octree');
  // a viewer: the splats as KHR_gaussian_splatting, the training state in extras only
  const g = parseGlb(bytes.buffer), prim = g.json.meshes.find(m => m.name === 'splats').primitives[0];
  assert(readAccessor(g, prim.attributes['KHR_gaussian_splatting:OPACITY']).data.length === N, 'the splats for a viewer');
  assert(r.count === N, `scene count ${r.count}`);
});

Deno.test('snapshot: a scene saved without its training state has none', () => {
  const s = state(), r = readSceneGlb(writeSceneGlb({ count: N, params: s.params, octree: s.octree }).buffer);
  assert(r.training === null && snapshotOfScene(r) === null, `training ${r.training}`);
});

Deno.test('snapshot: a .gosnap converted to a scene .glb gives the same snapshot', () => {
  const s = state(), oct = s.octree.toJSON();
  delete oct.leafIndices;
  const gosnap = encodeSnapshot({ ...s.header, octree: oct, transform: [...s.transform] }, s.blobs);
  assert(snapshotKind(gosnap) === 'gosnap', `kind ${snapshotKind(gosnap)}`);
  const old = decodeSnapshot(gosnap), glb = gosnapToSceneGlb(gosnap), snap = snapshotOfScene(readSceneGlb(glb.buffer));
  for (const [k, a] of Object.entries(old.blobs)) assert(same(snap.blobs[k], a), `blob ${k} after the conversion`);
  for (const k of ['state', 'params', 'numImages', 'firstImage', 'dataset_url', 'savedAt', 'splatFloats', 'transform'])
    assert(JSON.stringify(snap.header[k]) === JSON.stringify(old.header[k]), `${k}: ${JSON.stringify(snap.header[k])} vs ${JSON.stringify(old.header[k])}`);
  assert(JSON.stringify(snap.header.octree.nodes) === JSON.stringify(old.header.octree.nodes), 'octree after the conversion');
  assert(snapshotKind(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]).buffer) === null, 'garbage has a kind');
});
