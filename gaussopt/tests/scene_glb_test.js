// scene_glb.js: a scene written as one .glb reads back: the splats (positions and colors exactly; rotations, scales,
// opacities through the linear forms KHR_gaussian_splatting stores, to float precision), their octree and leaf ids,
// the cameras (views, intrinsics, held-out flags), the path, the sparse points, the mesh, the background, the leveling;
// the glTF parts a viewer reads are standard: the splats' attributes as the extension defines them, the root turning
// the scene's up to +Y, the cameras looking where the views look, the path's animation through them. The photos'
// .glb reads back byte for byte.
import { writeSceneGlb, readSceneGlb, writePhotosGlb, readPhotosGlb, SCENE_FORMAT } from '../src/scene_glb.js';
import { parseGlb, readAccessor, readGlb } from '../src/glb.js';
import { LooseOctree } from '../src/octree.js';
import { SPLAT_FLOATS } from '../src/colmap.js';
import { makeRng, levelTransform, upFromTransform, quatToMat3, sigmoid, IDENTITY4, normalize } from '../src/math.js';
import { assert, lookAt } from './util.js';

const rng = makeRng(5), N = 2000;
function splats() {
  const p = new Float32Array(N * SPLAT_FLOATS);
  for (let i = 0; i < N; i++) {
    const b = i * SPLAT_FLOATS, q = [rng() - 0.5, rng() - 0.5, rng() - 0.5, rng() - 0.5];
    for (let k = 0; k < 3; k++) p[b + k] = 6 * rng() - 3;
    p[b + 3] = 12 * rng() - 6;                                      // logit opacity
    for (let k = 4; k < 7; k++) p[b + k] = -9 + 7 * rng();          // log scale
    const n = Math.hypot(...q);
    p.set(q.map(v => v / n), b + 8);                                // w x y z
    for (let k = 12; k < 15; k++) p[b + k] = 4 * rng() - 2;         // SH 0
  }
  return p;
}
// a camera at c looking along f (y down), as the dataset's views: column-major world -> camera
function camera(name, c, f, isValidation = false) {
  return { name, view: lookAt(c, normalize(f)).view,
           camera: { fx: 500, fy: 520, cx: 321.5, cy: 239.25, width: 640, height: 480 }, zMin: 0.5, zMax: 9.25, isValidation };
}

function scene() {
  const params = splats(), octree = new LooseOctree({ maxLeafSize: 64, maxDepth: 6, beta: 0.3 });
  octree.build(params, N);
  const cameras = [0, 1, 2, 3, 4].map(k => camera(`f${k}.jpg`, [Math.cos(k), -0.2 * k, Math.sin(k) - 4], [-Math.cos(k) * 0.2, 0.1, 1], k === 2));
  const transform = levelTransform(IDENTITY4, [0, 0, 1], 12);
  return {
    count: N, params, octree, cameras, path: [4, 3, 0, 1, 2], transform, upEstimate: [0.05, -0.99, 0.1],
    points: { positions: Float32Array.from({ length: 300 }, () => rng() * 4 - 2), colors: Uint8Array.from({ length: 300 }, () => Math.floor(256 * rng())) },
    mesh: { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0.5]), faces: new Uint32Array([0, 1, 2, 1, 3, 2]),
            colors: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9]), voxel: 0.02 },
    env: { n: 4, rgb: Uint8Array.from({ length: 6 * 4 * 4 * 3 }, (_, i) => i % 251) },
    extras: { settings: { lr_pos: 0.0002 }, step: 30000 },
  };
}

const close = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

Deno.test('scene .glb: the splats, octree, cameras, path, points, mesh, background and leveling read back', () => {
  const s = scene(), bytes = writeSceneGlb(s), r = readSceneGlb(bytes.buffer);
  assert(r.count === N, `count ${r.count}`);
  let worst = { pos: 0, dc: 0, q: 0, s: 0, a: 0 };
  for (let i = 0; i < N; i++) {
    const b = i * SPLAT_FLOATS, P = s.params, Q = r.params;
    for (let k = 0; k < 3; k++) worst.pos = Math.max(worst.pos, Math.abs(Q[b + k] - P[b + k]));
    for (let k = 12; k < 15; k++) worst.dc = Math.max(worst.dc, Math.abs(Q[b + k] - P[b + k]));
    for (let k = 8; k < 12; k++) worst.q = Math.max(worst.q, Math.abs(Q[b + k] - P[b + k]));
    for (let k = 4; k < 7; k++) worst.s = Math.max(worst.s, Math.abs(Q[b + k] - P[b + k]));
    worst.a = Math.max(worst.a, Math.abs(sigmoid(Q[b + 3]) - sigmoid(P[b + 3])));
    assert(Q[b + 7] === P[b + 7], `splat ${i}: leaf ${Q[b + 7]}, built ${P[b + 7]}`);
  }
  assert(worst.pos === 0 && worst.dc === 0, `positions / SH 0 not exact: ${worst.pos}, ${worst.dc}`);
  assert(worst.q < 1e-6 && worst.s < 1e-5 && worst.a < 1e-6, `rotation ${worst.q}, log scale ${worst.s}, opacity ${worst.a}`);
  const t = LooseOctree.fromJSON(r.octree);
  assert(t.leaves.length === s.octree.leaves.length && t.maxDepth === 6 && t.beta === 0.3, `octree: ${t.leaves.length} leaves`);
  assert(r.cameras.length === 5, `${r.cameras.length} cameras`);
  for (const [k, c] of r.cameras.entries()) {
    const o = s.cameras[k];
    assert(c.name === o.name && c.isValidation === o.isValidation && c.zMax === o.zMax, `camera ${k}: ${c.name}`);
    assert(Object.entries(o.camera).every(([key, v]) => c.camera[key] === v), `camera ${k}: intrinsics`);
    for (let j = 0; j < 16; j++) assert(close(c.view[j], o.view[j], 1e-6), `camera ${k}: view[${j}] ${c.view[j]} vs ${o.view[j]}`);
  }
  assert(r.path.join() === '4,3,0,1,2', `path ${r.path}`);
  assert(r.points.positions.every((v, i) => v === s.points.positions[i]) && r.points.colors.every((v, i) => v === s.points.colors[i]), 'points');
  assert(r.mesh.faces.join() === '0,1,2,1,3,2' && r.mesh.voxel === 0.02 && r.mesh.colors[0] === 255, 'mesh faces, voxel, colors');
  for (let i = 0; i < 12; i++) assert(Math.abs(r.mesh.positions[i] - s.mesh.positions[i]) < 1e-4, `mesh vertex ${i}`);
  assert(r.env.n === 4 && r.env.rgb.every((v, i) => v === s.env.rgb[i]), 'background');
  assert(r.transform.every((v, i) => v === s.transform[i]) && r.upEstimate.join() === s.upEstimate.join(), 'transform');
  assert(r.info.settings.lr_pos === 0.0002 && r.info.step === 30000 && r.info.format === SCENE_FORMAT, 'extras');
  // a mesh viewer (glb.js readGlb, the app's Open) finds the mesh in it, in the scene's frame
  const m = readGlb(bytes.buffer);
  assert(m.faces.length === 6 && Math.abs(m.positions[3] - 1) < 1e-4, 'readGlb of a scene: its mesh');
});

Deno.test('scene .glb: standard glTF where a viewer reads it (KHR_gaussian_splatting, the root upright, the cameras, the path)', () => {
  const s = scene(), g = parseGlb(writeSceneGlb(s).buffer), { json } = g;
  assert(json.asset.version === '2.0' && json.extensionsUsed.includes('KHR_gaussian_splatting'), 'extension listed');
  const root = json.nodes[json.scenes[0].nodes[0]], M = root.matrix;
  // the root turns the world's up (the Level transform applied to the cameras' up) to +Y
  const up = upFromTransform(s.transform, s.upEstimate), y = [0, 1, 2].map(r => M[r] * up[0] + M[4 + r] * up[1] + M[8 + r] * up[2]);
  assert(Math.abs(y[0]) < 1e-9 && Math.abs(y[1] - 1) < 1e-9 && Math.abs(y[2]) < 1e-9, `up -> ${y}`);
  const plain = parseGlb(writeSceneGlb({ ...s, transform: null, upEstimate: [0, 1, 0] }).buffer).json;  // already +Y up: no matrix (glTF: no default one)
  assert(!plain.nodes[plain.scenes[0].nodes[0]].matrix, 'an identity root matrix written');
  const prim = json.meshes[json.nodes.find(n => n.name === 'splats').mesh].primitives[0], A = prim.attributes;
  assert(prim.mode === 0 && prim.extensions.KHR_gaussian_splatting.kernel === 'ellipse', 'points with the ellipse kernel');
  const acc = (k) => json.accessors[A[k]];
  assert(acc('POSITION').min && acc('POSITION').max && acc('KHR_gaussian_splatting:ROTATION').type === 'VEC4' && acc('KHR_gaussian_splatting:OPACITY').type === 'SCALAR', 'accessor types');
  const q = readAccessor(g, A['KHR_gaussian_splatting:ROTATION']).data, a = readAccessor(g, A['KHR_gaussian_splatting:OPACITY']).data;
  const sc = readAccessor(g, A['KHR_gaussian_splatting:SCALE']).data;
  for (let i = 0; i < N; i++) {
    assert(Math.abs(Math.hypot(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]) - 1) < 1e-6, `rotation ${i} not unit`);
    assert(a[i] > 0 && a[i] < 1 && sc[3 * i] > 0, `opacity ${a[i]} / scale ${sc[3 * i]} of ${i}`);
    // glTF's order x y z w: the same rotation as ours (w x y z)
    const b = i * SPLAT_FLOATS, R0 = quatToMat3([s.params[b + 8], s.params[b + 9], s.params[b + 10], s.params[b + 11]]), R1 = quatToMat3([q[4 * i + 3], q[4 * i], q[4 * i + 1], q[4 * i + 2]]);
    assert(R0.every((v, j) => Math.abs(v - R1[j]) < 1e-6), `rotation ${i} differs`);
  }
  // a camera node looks where its view looks: glTF's -Z is the view's forward, its translation the center
  const cam = json.nodes.find(n => n.name === 'f1.jpg'), v = s.cameras[1].view, C = cam.matrix;
  const fwd = [v[2], v[6], v[10]];
  assert([0, 1, 2].every(k => Math.abs(-C[8 + k] - fwd[k]) < 1e-6), 'camera -Z along the view');
  const yfov = json.cameras[cam.camera].perspective.yfov;
  assert(Math.abs(yfov - 2 * Math.atan(240 / 520)) < 1e-12, `yfov ${yfov}`);
  // the path's animation: through the cameras in path order, its first key at camera 4
  const an = json.animations[0], tIn = readAccessor(g, an.samplers[0].input).data, tOut = readAccessor(g, an.samplers[0].output).data;
  const c4 = json.nodes.find(n => n.name === 'f4.jpg').matrix;
  assert(tIn.length === 5 && tIn[0] === 0 && [0, 1, 2].every(k => Math.abs(tOut[k] - c4[12 + k]) < 1e-6), 'path starts at camera 4');
  assert(an.channels.every(ch => !json.nodes[ch.target.node].matrix), 'the animated node has no matrix (glTF: TRS only)');
});

Deno.test('photos .glb: images by name, read back byte for byte', () => {
  const photos = [0, 1, 2].map(k => ({ name: `f${k}.jpg`, bytes: Uint8Array.from({ length: 1000 + 7 * k }, (_, i) => (i * (k + 3)) & 255), width: 640, height: 480 }));
  const r = readPhotosGlb(writePhotosGlb(photos, { scene: 'room' }).buffer);
  assert(r.info.scene === 'room' && r.photos.length === 3, 'photos and extras');
  for (const [k, p] of r.photos.entries()) {
    assert(p.name === photos[k].name && p.mimeType === 'image/jpeg' && p.width === 640, `photo ${k}`);
    assert(p.bytes.length === photos[k].bytes.length && p.bytes.every((b, i) => b === photos[k].bytes[i]), `photo ${k}: bytes`);
  }
  let threw = false;
  try { readSceneGlb(writePhotosGlb(photos).buffer); } catch { threw = true; }
  assert(threw, 'a photo set read as a scene');
});
