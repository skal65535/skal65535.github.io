// ply_octree_test.js — .ply checkpoint with an embedded Loose Octree: round trip of the tree and
// the splats' leaf ids, and octree-culled vs exhaustive sorts keeping the same splats.
import { LooseOctree, nearCameraCone, splatExtents } from '../src/octree.js';
import { SPLAT_FLOATS, exportPlyBuffer, parsePlyBuffer } from '../src/colmap.js';
import { makeRng, projectPinhole } from '../src/math.js';
import { expSO3, mul3 } from '../src/sfm/linalg.js';
import { assert } from './util.js';

// n splats, 95% in a dense cluster (deep, unbalanced tree), the rest spread out
function clusteredSplats(n, rng) {
  const p = new Float32Array(n * SPLAT_FLOATS);
  for (let i = 0; i < n; i++) {
    const b = i * SPLAT_FLOATS, dense = i < 0.95 * n;
    for (let k = 0; k < 3; k++) p[b + k] = dense ? -1 - 0.5 * rng() : 4 * (rng() - 0.5);
    p[b + 3] = 4 * rng() - 2;
    for (let k = 4; k < 7; k++) p[b + k] = -5 + 2 * rng();
    p[b + 8] = 1;
  }
  return p;
}

Deno.test('.ply octree round trip: leaf ids, max depth, leaf size, looseness', () => {
  const n = 3000, p = clusteredSplats(n, makeRng(1));
  const tree = new LooseOctree({ maxLeafSize: 128, maxDepth: 7, beta: 0.3 });
  tree.build(p, n);
  const parsed = parsePlyBuffer(exportPlyBuffer(p, n, tree));
  const t = LooseOctree.fromSerializedNodes(parsed.octreeData.meta, parsed.octreeData.nodes);
  assert(t.maxDepth === 7 && t.maxLeafSize === 128 && Math.abs(t.beta - 0.3) < 1e-6, `meta ${t.maxDepth} ${t.maxLeafSize} ${t.beta}`);
  assert(t.leaves.length === tree.leaves.length, `${t.leaves.length} leaves, expected ${tree.leaves.length}`);
  let bad = 0;
  for (let i = 0; i < n; i++) {
    const lid = parsed.params[i * SPLAT_FLOATS + 7], leaf = t.leaves[lid];
    if (lid !== p[i * SPLAT_FLOATS + 7] || !(i >= leaf.start && i < leaf.start + leaf.count)) bad++;
  }
  assert(bad === 0, `${bad} splats with a wrong leaf id`);
});

Deno.test('octree cull keeps the splats the exhaustive cull keeps', () => {
  const rng = makeRng(2), n = 20000, p = new Float32Array(n * SPLAT_FLOATS);
  for (let i = 0; i < n; i++) {
    const b = i * SPLAT_FLOATS;
    for (let k = 0; k < 3; k++) p[b + k] = 8 * (rng() - 0.5);
    p[b + 3] = 0;
    for (let k = 4; k < 7; k++) p[b + k] = (i % 64) ? -9 + 2 * rng() : -4 + 3.5 * rng();  // small, a few larger than their leaf
    p[b + 8] = 1;
  }
  const tree = new LooseOctree({ maxLeafSize: 4 });
  tree.build(p, n);
  tree.updateExtents(p);
  const a = new Uint32Array(n), b = new Uint32Array(n);
  for (let c = 0; c < 16; c++) {  // cameras around the origin looking at it, some inside the cloud
    const ang = c * 0.7, r = 1 + (c % 4) * 2.5, C = [r * Math.cos(ang), 0.3 * (c % 3), r * Math.sin(ang)];
    const f = [-Math.cos(ang), 0, -Math.sin(ang)], d = [0, 1, 0], x = [d[1] * f[2] - d[2] * f[1], d[2] * f[0] - d[0] * f[2], d[0] * f[1] - d[1] * f[0]];
    const dot = (u) => u[0] * C[0] + u[1] * C[1] + u[2] * C[2];
    const view = new Float32Array([x[0], d[0], f[0], 0, x[1], d[1], f[1], 0, x[2], d[2], f[2], 0, -dot(x), -dot(d), -dot(f), 1]);
    const fx = 1 + (c % 3), fy = fx * 1.3;
    const nRef = tree.sortGlobalExact(p, n, view, fx, fy, a);
    const nOct = tree.sortOctreeFrontToBack(p, C, view, fx, fy, b).numVisible;
    const ref = new Set(a.subarray(0, nRef)), oct = new Set(b.subarray(0, nOct));
    assert(nRef === nOct && [...ref].every(i => oct.has(i)), `camera ${c}: exhaustive ${nRef} visible, octree ${nOct}`);
  }
});

Deno.test('viewer sort (the gallery: a .ply tree, its index): the splats of the exhaustive cull, in its exact depth order', () => {
  const rng = makeRng(5), n = 20000, p = new Float32Array(n * SPLAT_FLOATS);
  for (let i = 0; i < n; i++) {
    const b = i * SPLAT_FLOATS;
    for (let k = 0; k < 3; k++) p[b + k] = 8 * (rng() - 0.5);
    p[b + 3] = (i % 50) ? 0 : -7;  // a few under the opacity cull
    for (let k = 4; k < 7; k++) p[b + k] = (i % 64) ? -9 + 2 * rng() : -4 + 3.5 * rng();  // small, a few larger than their leaf
    p[b + 8] = 1;
  }
  // as the gallery reads it: the tree from the .ply, the positions, the extents
  const ply = parsePlyBuffer(exportPlyBuffer(p, n, new LooseOctree({ maxLeafSize: 4 }).build(p, n)));
  const tree = LooseOctree.fromSerializedNodes(ply.octreeData.meta, ply.octreeData.nodes);
  const pos = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) pos.set(ply.params.subarray(i * SPLAT_FLOATS, i * SPLAT_FLOATS + 3), 3 * i);
  const extents = splatExtents(ply.params, n);
  tree.setExtents(extents);
  const a = new Uint32Array(n), b = new Uint32Array(n);
  for (let c = 0; c < 16; c++) {  // cameras around the origin looking at it, some inside the cloud
    const ang = c * 0.7, r = 1 + (c % 4) * 2.5, C = [r * Math.cos(ang), 0.3 * (c % 3), r * Math.sin(ang)];
    const f = [-Math.cos(ang), 0, -Math.sin(ang)], d = [0, 1, 0], x = [d[1] * f[2] - d[2] * f[1], d[2] * f[0] - d[0] * f[2], d[0] * f[1] - d[1] * f[0]];
    const dot = (u) => u[0] * C[0] + u[1] * C[1] + u[2] * C[2];
    const view = new Float32Array([x[0], d[0], f[0], 0, x[1], d[1], f[1], 0, x[2], d[2], f[2], 0, -dot(x), -dot(d), -dot(f), 1]);
    const gl = new Float32Array(view);  // the gallery's view (from_opencv): x and z negated, zSign -1
    for (const k of [0, 4, 8, 12, 2, 6, 10, 14]) gl[k] = -gl[k];
    const fx = 1 + (c % 3), fy = fx * 1.3;
    const nRef = tree.sortGlobalExact(ply.params, n, view, fx, fy, a);
    const nView = tree.sortIndexFrontToBack(pos, 3, C, gl, fx, fy, b, -1.0, extents);
    // the same splats; depths (f32, as sorted) never decreasing: ties in leaf order, the reference's in index order
    const ref = new Set(a.subarray(0, nRef)), z = (i) => Math.fround(view[2] * pos[3 * i] + view[6] * pos[3 * i + 1] + view[10] * pos[3 * i + 2] + view[14]);
    let bad = 0;
    for (let k = 1; k < nView; k++) if (z(b[k]) < z(b[k - 1])) bad++;
    assert(nRef === nView && [...b.subarray(0, nView)].every(i => ref.has(i)) && bad === 0, `camera ${c}: exhaustive ${nRef} visible, viewer ${nView}, ${bad} out of order`);
  }
});

Deno.test('near-camera cone: the sphere grid finds exactly the splats a scan of every camera finds', () => {
  const r = makeRng(7), uv = [0, 0];
  const images = Array.from({ length: 60 }, (_, c) => {
    const R = mul3(expSO3([r() - 0.5, 6 * r(), r() - 0.5]), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const C = [8 * r() - 4, r() - 0.5, 8 * r() - 4], t = [0, 1, 2].map(i => -(R[3 * i] * C[0] + R[3 * i + 1] * C[1] + R[3 * i + 2] * C[2]));
    const camera = { fx: 500 + 500 * r(), fy: 500 + 500 * r(), cx: 400 + 200 * r(), cy: 300 + 100 * r(), width: 1000, height: 700 };
    return { R, t, camera, zMin: 0.2 + 2 * r(), isValidation: c % 8 === 4 };
  });
  const near = nearCameraCone(images), train = images.filter(im => !im.isValidation);
  const scan = (x, y, z) => train.some(({ R, t, camera, zMin }) => {
    const zc = projectPinhole(R, t, camera, x, y, z, uv);
    return zc > 0 && zc < 0.65 * zMin && uv[0] >= 0 && uv[0] <= camera.width && uv[1] >= 0 && uv[1] <= camera.height;
  });
  let hits = 0;
  for (let k = 0; k < 200000; k++) {
    const x = 10 * r() - 5, y = 3 * r() - 1.5, z = 10 * r() - 5, s = scan(x, y, z);
    assert(near(x, y, z) === s, `point ${k} (${x}, ${y}, ${z}): grid ${!s}, scan ${s}`);
    hits += s;
  }
  assert(hits > 1000, `only ${hits} points in a cone`);
});

Deno.test('near-camera cone, spare: a splat near an SfM point is in no cone (a scan of every point)', () => {
  const r = makeRng(11), uv = [0, 0];
  const images = Array.from({ length: 40 }, () => {
    const R = mul3(expSO3([r() - 0.5, 6 * r(), r() - 0.5]), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const C = [8 * r() - 4, r() - 0.5, 8 * r() - 4], t = [0, 1, 2].map(i => -(R[3 * i] * C[0] + R[3 * i + 1] * C[1] + R[3 * i + 2] * C[2]));
    return { R, t, camera: { fx: 700, fy: 700, cx: 500, cy: 350, width: 1000, height: 700 }, zMin: 1 + 2 * r() };
  });
  const count = 3000, positions = Float32Array.from({ length: 3 * count }, (_, i) => i % 3 === 1 ? 3 * r() - 1.5 : 10 * r() - 5);
  const radius = 0.3, near = nearCameraCone(images, { spare: { positions, count, radius } });
  const inCone = (x, y, z) => images.some(({ R, t, camera, zMin }) => {
    const zc = projectPinhole(R, t, camera, x, y, z, uv);
    return zc > 0 && zc < 0.65 * zMin && uv[0] >= 0 && uv[0] <= camera.width && uv[1] >= 0 && uv[1] <= camera.height;
  });
  const nearPoint = (x, y, z) => { for (let p = 0; p < count; p++) if ((positions[3 * p] - x) ** 2 + (positions[3 * p + 1] - y) ** 2 + (positions[3 * p + 2] - z) ** 2 <= radius * radius) return true; return false; };
  let spared = 0, carved = 0;
  for (let k = 0; k < 20000; k++) {
    const x = 10 * r() - 5, y = 3 * r() - 1.5, z = 10 * r() - 5, c = inCone(x, y, z), s = c && nearPoint(x, y, z);
    assert(near(x, y, z) === (c && !s), `point ${k}: cone ${c}, near a point ${s}, got ${near(x, y, z)}`);
    spared += s; carved += c && !s;
  }
  assert(spared > 100 && carved > 100, `spared ${spared}, carved ${carved}`);
  assert(nearCameraCone(images, { spare: { positions, count, radius: 0 } })(positions[0], positions[1], positions[2]) === inCone(positions[0], positions[1], positions[2]), 'radius 0: off');
});

Deno.test('front-to-back sort: a leaf far over maxLeafSize (maxDepth reached) sorted in O(k log k), stable', () => {
  const n = 100000, rng = makeRng(3), pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { pos[3 * i] = rng() - 0.5; pos[3 * i + 1] = rng() - 0.5; pos[3 * i + 2] = -1 - Math.floor(10000 * rng()) / 100; }
  const tree = new LooseOctree({ maxLeafSize: 256, maxDepth: 0 }).buildIndexOnly(pos, n);
  assert(tree.leaves.length === 1 && tree.leaves[0].count === n, `${tree.leaves.length} leaves`);
  const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), out = new Uint32Array(n);
  const t0 = performance.now();
  const m = tree.sortIndexFrontToBack(pos, 3, [0, 0, 0], view, 0, 0, out, -1.0);
  const ms = performance.now() - t0;
  // depths in 10000 levels: ties kept in index order (the insertion sort's order)
  const ref = Array.from({ length: n }, (_, i) => i).sort((a, b) => pos[3 * b + 2] - pos[3 * a + 2] || a - b);
  assert(m === n, `${m} sorted of ${n}`);
  let bad = 0;
  for (let k = 0; k < n; k++) if (out[k] !== ref[k]) bad++;
  assert(bad === 0, `${bad} out of order`);
  assert(ms < 300, `${ms.toFixed(0)} ms for one leaf of ${n} (the insertion sort: ~1.5 s)`);
});
