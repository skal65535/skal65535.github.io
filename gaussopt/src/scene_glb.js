// scene_glb.js — A trained scene as one binary glTF 2.0 file (.glb), standard parts only:
//   the splats       a point primitive with KHR_gaussian_splatting (ratified by Khronos): POSITION, ROTATION (unit, x y z w),
//                    SCALE (linear), OPACITY (0-1), SH_DEGREE_0_COEF_0, and COLOR_0 (linear, opacity in alpha) for viewers
//                    without the extension (they draw the points);
//   the mesh         a triangle mesh (glb.js addMesh: 16-bit positions, colors or a texture), when there is one;
//   the cameras      a glTF camera and a node per training / held-out view (perspective; their exact intrinsics in extras);
//   the path         an animation of a camera node through the views in path order (0.5 s a view);
//   the points       the SfM's sparse points, a point primitive (COLOR_0 linear; _SRGB: the colors as they were);
//   the background   env.js's cube map as rgb bytes, a buffer view named in extras (n texels a face: ENV_N learned, more a
//                    dome, src/dome.js; sky: the dome's share of the photos' pixels).
// Everything under one root node whose matrix is the scene's leveling: the world's up (the cameras' up through the
// Level transform, math.js upFromTransform) turned to glTF's +Y, so any viewer shows it upright; the data below it
// stays in the scene's own frame (COLMAP's, as trained), read back as written. The rest is JSON in the root's
// extras.gaussopt: the transform, the up, the octree (octree.js toJSON), the page's settings and anything the caller adds.
// The photos go in a .glb of their own (writePhotosGlb): only the trainer needs them.
"use strict";

import { GlbWriter, addMesh, parseGlb, readAccessor, viewBytes, readMeshNode, rotationToY } from './glb.js';
import { SPLAT_FLOATS, assignLeafIds } from './colmap.js';
import { sigmoid, mat3ToQuat, upFromTransform, parseTransform } from './math.js';

export const SCENE_FORMAT = 'gaussopt-scene/1', PHOTOS_FORMAT = 'gaussopt-photos/1';
const KHR = 'KHR_gaussian_splatting', SH_C0 = 0.28209479177387814, PATH_STEP = 0.5;
const srgbToLinear = (c) => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;

// scene: { count, params (Float32Array, colmap.js SPLAT_FLOATS a splat), octree (LooseOctree) | null,
//   cameras: [{ name, view (16, column-major world -> camera, OpenCV axes), camera: { fx, fy, cx, cy, width, height },
//               zMin, zMax, isValidation }], path: camera indices in path order | null,
//   points: { positions: Float32Array 3N, colors: Uint8Array 3N (sRGB) } | null,
//   transform (16, row-major, the Level transform), upEstimate (the cameras' up),
//   mesh: glb.js's mesh ({ positions, faces, colors?, uvs?, texture?, voxel? }) | null, env: { n, rgb: Uint8Array } | null,
//   extras: more JSON for extras.gaussopt (settings, step, ...) }
export function writeSceneGlb({ count, params, octree = null, cameras = [], path = null, points = null, transform = null,
                                upEstimate = [0, -1, 0], mesh = null, env = null, extras = {} }) {
  const T = parseTransform(transform), up = upFromTransform(T, upEstimate);
  const w = new GlbWriter(), json = {
    asset: { version: '2.0', generator: 'Gauss-Opt scene_glb.js' },
    extensionsUsed: [KHR], scene: 0, scenes: [{ nodes: [0] }],
    nodes: [], meshes: [], materials: [],
  };
  const R = rotationToY(up), root = { name: 'scene', ...(R.some((v, i) => v !== [1, 0, 0, 0, 1, 0, 0, 0, 1][i]) && { matrix: mat4(R, [0, 0, 0]) }), children: [] };
  json.nodes.push(root);
  const child = (node) => { root.children.push(json.nodes.push(node) - 1); return node; };

  // the splats
  const P = new Float32Array(3 * count), Q = new Float32Array(4 * count), S = new Float32Array(3 * count);
  const A = new Float32Array(count), DC = new Float32Array(3 * count), C = new Uint8Array(4 * count);
  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS, q = [params[b + 8], params[b + 9], params[b + 10], params[b + 11]], n = Math.hypot(...q) || 1;
    P.set([params[b], params[b + 1], params[b + 2]], 3 * i);
    Q.set([q[1] / n, q[2] / n, q[3] / n, q[0] / n], 4 * i);
    S.set([Math.exp(params[b + 4]), Math.exp(params[b + 5]), Math.exp(params[b + 6])], 3 * i);
    A[i] = sigmoid(params[b + 3]);
    DC.set([params[b + 12], params[b + 13], params[b + 14]], 3 * i);
    for (let k = 0; k < 3; k++) C[4 * i + k] = Math.round(255 * srgbToLinear(Math.min(1, Math.max(0, 0.5 + SH_C0 * params[b + 12 + k]))));
    C[4 * i + 3] = Math.round(255 * A[i]);
  }
  const attributes = {
    POSITION: w.floats(P, 3, { minMax: true }),
    COLOR_0: w.accessor({ bufferView: w.view(C, { target: 34962 }), componentType: 5121, normalized: true, count, type: 'VEC4' }),
    [`${KHR}:ROTATION`]: w.floats(Q, 4), [`${KHR}:SCALE`]: w.floats(S, 3), [`${KHR}:OPACITY`]: w.floats(A, 1),
    [`${KHR}:SH_DEGREE_0_COEF_0`]: w.floats(DC, 3),
  };
  json.meshes.push({ name: 'splats', primitives: [{ attributes, mode: 0, extensions: { [KHR]: { kernel: 'ellipse', colorSpace: 'srgb_rec709_display' } } }] });
  child({ name: 'splats', mesh: json.meshes.length - 1 });

  // the sparse points
  if (points?.positions?.length) {
    const n = points.positions.length / 3, lin = new Uint8Array(4 * n);
    for (let i = 0; i < n; i++) { for (let k = 0; k < 3; k++) lin[4 * i + k] = Math.round(255 * srgbToLinear(points.colors[3 * i + k] / 255)); lin[4 * i + 3] = 255; }
    const srgb = new Uint8Array(4 * n);
    for (let i = 0; i < n; i++) srgb.set([points.colors[3 * i], points.colors[3 * i + 1], points.colors[3 * i + 2], 255], 4 * i);
    const bytes = (u8) => w.accessor({ bufferView: w.view(u8, { target: 34962 }), componentType: 5121, normalized: true, count: n, type: 'VEC4' });
    json.meshes.push({ name: 'points', primitives: [{ mode: 0, attributes: { POSITION: w.floats(points.positions, 3, { minMax: true }), COLOR_0: bytes(lin), _SRGB: bytes(srgb) } }] });
    child({ name: 'points', mesh: json.meshes.length - 1 });
  }

  // the mesh (in the scene's frame: the root turns it upright)
  if (mesh) {
    addMesh(w, json, mesh, { texture: mesh.texture ?? null });
    const node = json.nodes.pop();
    child({ name: 'mesh', ...node, ...(mesh.voxel && { extras: { gaussopt_voxel: mesh.voxel } }) });
    json.extensionsUsed.push('KHR_mesh_quantization', 'KHR_materials_unlit');
    json.extensionsRequired = ['KHR_mesh_quantization'];
  }
  if (!json.materials.length) delete json.materials;

  // the cameras: glTF's look down -Z with +Y up, OpenCV's down +Z with +Y down
  json.cameras = [];
  const camNodes = cameras.map((im, k) => {
    const { fx, fy, cx, cy, width, height } = im.camera;
    json.cameras.push({ type: 'perspective', perspective: { yfov: 2 * Math.atan(0.5 * height / fy), aspectRatio: (width * fy) / (height * fx), znear: 0.01 } });
    return child({ name: im.name, camera: k, matrix: cameraMatrix(im.view),
                   extras: { gaussopt: { fx, fy, cx, cy, width, height, zMin: im.zMin, zMax: im.zMax, validation: !!im.isValidation } } });
  });
  if (!json.cameras.length) delete json.cameras;

  // the path: a camera flying through the views in path order
  const order = (path ?? cameras.map((_, i) => i)).filter(i => i >= 0 && i < cameras.length);
  if (order.length >= 2) {
    const times = order.map((_, k) => k * PATH_STEP), tr = [], rot = [];
    let prev = null;
    for (const i of order) {
      const M = camNodes[i].matrix;
      tr.push(M[12], M[13], M[14]);
      let q = mat3ToQuat([M[0], M[4], M[8], M[1], M[5], M[9], M[2], M[6], M[10]]);  // w x y z
      if (prev && q[0] * prev[0] + q[1] * prev[1] + q[2] * prev[2] + q[3] * prev[3] < 0) q = q.map(v => -v);  // the short way round
      rot.push(q[1], q[2], q[3], q[0]);
      prev = q;
    }
    const node = json.nodes.push({ name: 'path', camera: order[0], translation: tr.slice(0, 3), rotation: rot.slice(0, 4) }) - 1;
    root.children.push(node);
    const input = w.accessor({ bufferView: w.view(Float32Array.from(times)), componentType: 5126, count: times.length, type: 'SCALAR', min: [0], max: [Math.fround(times.at(-1))] });
    const tAcc = w.accessor({ bufferView: w.view(Float32Array.from(tr)), componentType: 5126, count: order.length, type: 'VEC3' });
    const rAcc = w.accessor({ bufferView: w.view(Float32Array.from(rot)), componentType: 5126, count: order.length, type: 'VEC4' });
    json.animations = [{ name: 'path', samplers: [{ input, output: tAcc, interpolation: 'LINEAR' }, { input, output: rAcc, interpolation: 'LINEAR' }],
                         channels: [{ sampler: 0, target: { node, path: 'translation' } }, { sampler: 1, target: { node, path: 'rotation' } }] }];
  }

  root.extras = { gaussopt: {
    format: SCENE_FORMAT, transform: T, upEstimate: [...upEstimate], up, splats: count, path: path ? [...path] : null,
    ...(octree && { octree: { ...octree.toJSON(), leafIndices: undefined } }),
    ...(env && { env: { n: env.n, bufferView: w.view(env.rgb), ...(env.sky !== undefined && { sky: env.sky }) } }),
    ...extras,
  } };
  return w.finish(json);
}

// A scene .glb back: { count, params (leaf_id from the octree, flags 0), octree: the octree's JSON (octree.js fromJSON) | null,
// cameras (writeSceneGlb's, with center), path, points, transform, upEstimate, up, mesh | null, env | null, info (extras.gaussopt) }
export function readSceneGlb(buffer) {
  const g = parseGlb(buffer), { json } = g, root = sceneRoot(json), info = root?.extras?.gaussopt;
  if (info?.format !== SCENE_FORMAT) throw new Error(`not a Gauss-Opt scene (.glb without extras.gaussopt.format ${SCENE_FORMAT})`);
  const nodes = (root.children ?? []).map(i => json.nodes[i]), named = (name) => nodes.find(n => n.name === name && n.mesh !== undefined);
  // the splats
  const prim = json.meshes[named('splats').mesh].primitives[0], at = (k) => readAccessor(g, prim.attributes[k]).data;
  const P = at('POSITION'), Q = at(`${KHR}:ROTATION`), S = at(`${KHR}:SCALE`), A = at(`${KHR}:OPACITY`), DC = at(`${KHR}:SH_DEGREE_0_COEF_0`);
  const count = A.length, params = new Float32Array(count * SPLAT_FLOATS);
  for (let i = 0; i < count; i++) {
    const p = Math.min(1 - 2 ** -24, Math.max(2 ** -24, A[i]));
    params.set([P[3 * i], P[3 * i + 1], P[3 * i + 2], Math.log(p / (1 - p)),
                Math.log(S[3 * i]), Math.log(S[3 * i + 1]), Math.log(S[3 * i + 2]), 0,
                Q[4 * i + 3], Q[4 * i], Q[4 * i + 1], Q[4 * i + 2],
                DC[3 * i], DC[3 * i + 1], DC[3 * i + 2], 0], i * SPLAT_FLOATS);
  }
  if (info.octree) assignLeafIds(params, count, info.octree.nodes);
  // the points
  let points = null;
  const pn = named('points');
  if (pn) {
    const pp = json.meshes[pn.mesh].primitives[0], pos = readAccessor(g, pp.attributes.POSITION).data, c = readAccessor(g, pp.attributes._SRGB);
    points = { positions: Float32Array.from(pos), colors: Uint8Array.from({ length: pos.length }, (_, j) => Math.round(255 * c.data[c.n * Math.floor(j / 3) + j % 3])) };
  }
  // the mesh
  const mn = named('mesh'), mesh = mn ? readMeshNode(g, mn) : null;
  // the cameras
  const cameras = nodes.filter(n => n.camera !== undefined && n.extras?.gaussopt).map(n => {
    const e = n.extras.gaussopt, M = n.matrix, center = [M[12], M[13], M[14]];
    const R = [M[0], M[1], M[2], -M[4], -M[5], -M[6], -M[8], -M[9], -M[10]];  // rows: right, down, forward
    const t = [0, 1, 2].map(r => -(R[3 * r] * center[0] + R[3 * r + 1] * center[1] + R[3 * r + 2] * center[2]));
    const view = new Float32Array([R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, t[0], t[1], t[2], 1]);
    return { name: n.name, view, center, camera: { fx: e.fx, fy: e.fy, cx: e.cx, cy: e.cy, width: e.width, height: e.height },
             zMin: e.zMin, zMax: e.zMax, isValidation: e.validation };
  });
  const env = info.env ? { n: info.env.n, rgb: viewBytes(g, info.env.bufferView).slice(), ...(info.env.sky !== undefined && { sky: info.env.sky }) } : null;
  return { count, params, octree: info.octree ?? null, cameras, path: info.path ?? null, points, mesh, env,
           transform: parseTransform(info.transform), upEstimate: info.upEstimate, up: info.up, info };
}

// The root node of a .glb's default scene (a scene's: extras.gaussopt holds its JSON) | undefined
export const sceneRoot = (json) => json.nodes?.[json.scenes?.[json.scene ?? 0]?.nodes?.[0]];

// photos: [{ name, bytes (a JPEG or PNG file), mimeType, width?, height? }] as glTF images (no scene: a viewer shows nothing)
export function writePhotosGlb(photos, extras = {}) {
  const w = new GlbWriter();
  const json = { asset: { version: '2.0', generator: 'Gauss-Opt scene_glb.js', extras: { gaussopt: { format: PHOTOS_FORMAT, ...extras } } },
                 images: photos.map(p => ({ name: p.name, mimeType: p.mimeType ?? 'image/jpeg', bufferView: w.view(p.bytes),
                                            ...(p.width && { extras: { width: p.width, height: p.height } }) })) };
  return w.finish(json);
}

// [{ name, bytes, mimeType, width?, height? }] and the extras
export function readPhotosGlb(buffer) {
  const g = parseGlb(buffer), info = g.json.asset?.extras?.gaussopt;
  if (info?.format !== PHOTOS_FORMAT) throw new Error(`not a Gauss-Opt photo set (.glb without asset.extras.gaussopt.format ${PHOTOS_FORMAT})`);
  return { photos: (g.json.images ?? []).map(im => ({ name: im.name, mimeType: im.mimeType, bytes: viewBytes(g, im.bufferView).slice(), ...im.extras })), info };
}

// A photos .glb on a server, read as needed: its header and JSON by a Range request (the JSON chunk comes first in
// a .glb), then each photo by one of its own, kept in the browser's Cache Storage (PHOTO_CACHE: the file's URL, its
// version (ETag, Last-Modified or size) and the photo's name) for the next time. A server that ignores Range sends
// the whole file: the photos are then read from it. { names, info, photo(name) -> Promise<Blob>, ranged }
export const PHOTO_CACHE = 'gaussopt-photos';
const HEAD_BYTES = 65536;
export async function openPhotosGlb(url) {
  const abs = new URL(url, globalThis.location?.href).href;
  const get = async (a, b) => {
    const r = await fetch(abs, { headers: { Range: `bytes=${a}-${b}` } });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r;
  };
  const check = (json) => {
    const info = json.asset?.extras?.gaussopt;
    if (info?.format !== PHOTOS_FORMAT) throw new Error(`${url}: not a Gauss-Opt photo set`);
    return info;
  };
  let r = await get(0, HEAD_BYTES - 1);
  if (r.status !== 206) {  // the whole file
    const g = parseGlb(await r.arrayBuffer()), info = check(g.json), byName = new Map((g.json.images ?? []).map(im => [im.name, im]));
    return { names: [...byName.keys()], info, ranged: false, photo: async (name) => {
      const im = byName.get(name);
      if (!im) throw new Error(`${url}: no photo ${name}`);
      return new Blob([viewBytes(g, im.bufferView)], { type: im.mimeType });
    } };
  }
  let head = new Uint8Array(await r.arrayBuffer());
  const dv = new DataView(head.buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error(`${url}: not a .glb (no glTF magic)`);
  const jsonLen = dv.getUint32(12, true), binStart = 20 + jsonLen + 8;
  if (head.length < 20 + jsonLen) head = new Uint8Array(await (await get(0, 20 + jsonLen - 1)).arrayBuffer());
  const json = JSON.parse(new TextDecoder().decode(head.subarray(20, 20 + jsonLen))), info = check(json);
  const version = r.headers.get('etag') ?? r.headers.get('last-modified') ?? r.headers.get('content-range')?.split('/')[1] ?? '';
  const byName = new Map((json.images ?? []).map(im => [im.name, im]));
  let cache;  // null without Cache Storage (an insecure origin, a browser setting)
  const openCache = async () => cache !== undefined ? cache : (cache = await globalThis.caches?.open(PHOTO_CACHE).catch(() => null) ?? null);
  return { names: [...byName.keys()], info, ranged: true, photo: async (name) => {
    const im = byName.get(name);
    if (!im) throw new Error(`${url}: no photo ${name}`);
    const key = `${abs}?photo=${encodeURIComponent(name)}&v=${encodeURIComponent(version)}`, c = await openCache();
    const hit = await c?.match(key).catch(() => null);
    if (hit) return hit.blob();
    const bv = json.bufferViews[im.bufferView], a = binStart + (bv.byteOffset ?? 0);
    const got = await get(a, a + bv.byteLength - 1);
    if (got.status !== 206) throw new Error(`${url}: photo ${name}: ${got.status}, not the range`);
    const blob = new Blob([await got.arrayBuffer()], { type: im.mimeType });
    if (blob.size !== bv.byteLength) throw new Error(`${url}: photo ${name}: ${blob.size} bytes, ${bv.byteLength} asked`);
    await c?.put(key, new Response(blob)).catch(() => {});
    return blob;
  } };
}

// A camera's node matrix (column-major) from its view: columns right, -down, -forward (glTF's axes), center
function cameraMatrix(v) {
  const R = [v[0], v[4], v[8], v[1], v[5], v[9], v[2], v[6], v[10]], t = [v[12], v[13], v[14]];  // row-major world -> camera
  const c = [0, 1, 2].map(j => -(R[j] * t[0] + R[3 + j] * t[1] + R[6 + j] * t[2]));
  return [R[0], R[1], R[2], 0, -R[3], -R[4], -R[5], 0, -R[6], -R[7], -R[8], 0, c[0], c[1], c[2], 1];
}

// A column-major 4x4 from a row-major rotation and a translation
const mat4 = (R, t) => [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, t[0], t[1], t[2], 1];

