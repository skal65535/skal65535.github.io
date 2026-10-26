// mesh.js: TSDF fusion + surface nets on exact depth maps of a sphere, seen from cameras all around it.
import { depthBounds, Tsdf, snapToSurface, colorVertices, finishColors, meshToPly, meshFromPly } from '../src/mesh.js';
import { assert } from './util.js';

const RADIUS = 1, W = 128, H = 128, F = 120;

// A camera at `c` looking at the origin: R rows = right, down, forward (world -> camera), t = -R c
function lookAtOrigin(c) {
  const f = c.map(x => -x / Math.hypot(...c));
  const up = Math.abs(f[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => a.map(x => x / Math.hypot(...a));
  const r = norm(cross(f, up)), d = cross(f, r);
  const R = [...r, ...d, ...f];
  return { R, t: [0, 1, 2].map(k => -(R[3 * k] * c[0] + R[3 * k + 1] * c[1] + R[3 * k + 2] * c[2])) };
}

// 20 cameras on a Fibonacci sphere (at 64 x 64 pixels the coarse depth folds surface nets into
// a few non-manifold edges; 128 x 128 gives a closed mesh at res 32 to 64) of radius 4; depth = camera z of the ray's first sphere hit (0: none).
// floater: in view 0, a patch of pixels at depth 2.5 (a blob 0.5 in front of the sphere seen by that view only).
// background: the rays that miss the sphere hit a sphere of radius 6 around it (a room); without one,
// they have no depth and cannot carve the floater.
// seeThrough: in views 1..k, a patch of pixels sees past the sphere to the background (the splats of a plain
// wall rendered transparent from a few views).
function views({ floater = false, background = false, seeThrough = 0 } = {}) {
  const out = [];
  for (let n = 0; n < 20; n++) {
    const y = 1 - 2 * (n + 0.5) / 20, r = Math.sqrt(1 - y * y), a = n * Math.PI * (3 - Math.sqrt(5));
    const c = [4 * r * Math.cos(a), 4 * y, 4 * r * Math.sin(a)], { R, t } = lookAtOrigin(c);
    const depth = new Float32Array(W * H), rgb = new Uint8Array(4 * W * H);
    for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
      const dc = [(u + 0.5 - W / 2) / F, (v + 0.5 - H / 2) / F, 1];
      const dw = [0, 1, 2].map(k => R[k] * dc[0] + R[3 + k] * dc[1] + R[6 + k] * dc[2]);  // R^T dc, camera z = 1
      const aa = dw[0] ** 2 + dw[1] ** 2 + dw[2] ** 2, bb = 2 * (c[0] * dw[0] + c[1] * dw[1] + c[2] * dw[2]);
      const disc = bb * bb - 4 * aa * (c[0] ** 2 + c[1] ** 2 + c[2] ** 2 - RADIUS ** 2);
      if (disc > 0) depth[v * W + u] = (-bb - Math.sqrt(disc)) / (2 * aa);
      else if (background) depth[v * W + u] = (-bb + Math.sqrt(bb * bb - 4 * aa * (c[0] ** 2 + c[1] ** 2 + c[2] ** 2 - 36))) / (2 * aa);
      if (floater && n === 0 && Math.abs(u - 64) < 12 && Math.abs(v - 64) < 12) depth[v * W + u] = 2.5;
      if (n >= 1 && n <= seeThrough && Math.abs(u - 64) < 10 && Math.abs(v - 64) < 10)
        depth[v * W + u] = (-bb + Math.sqrt(bb * bb - 4 * aa * (c[0] ** 2 + c[1] ** 2 + c[2] ** 2 - 36))) / (2 * aa);
      rgb.set([200, 100, 50, 255], 4 * (v * W + u));
    }
    out.push({ R, t, fx: F, fy: F, cx: W / 2, cy: H / 2, W, H, depth, rgb, pitch: 4 * W });
  }
  return out;
}

function mesh(vs, res = 48, bounds = depthBounds(vs), freeWeight = 1) {
  const grid = new Tsdf(bounds, { res, trunc: 3, freeWeight });
  for (const v of vs) grid.fuse(v);
  return { grid, mesh: grid.extract() };
}

Deno.test('mesh: a sphere seen from all around is a closed surface on the sphere, faces outwards', () => {
  const { grid, mesh: m } = mesh(views());
  const P = m.positions, F3 = m.faces, nv = P.length / 3, nf = F3.length / 3;
  assert(nv > 500, `${nv} vertices`);
  // in voxels (0.045; a pixel spans 3 / F = 0.025 at the sphere), the depth being one per pixel:
  // measured median 0.044, max 0.365
  const err = Array.from({ length: nv }, (_, i) => Math.abs(Math.hypot(P[3 * i], P[3 * i + 1], P[3 * i + 2]) - RADIUS) / grid.voxel).sort((a, b) => a - b);
  assert(err[nv >> 1] < 0.1 && err[nv - 1] < 0.5, `vertex off the sphere: median ${err[nv >> 1].toFixed(3)}, max ${err[nv - 1].toFixed(3)} voxels`);
  // closed: every edge in exactly two faces, once in each direction; Euler characteristic 2
  const edges = new Map();
  for (let f = 0; f < nf; f++) for (let k = 0; k < 3; k++) {
    const a = F3[3 * f + k], b = F3[3 * f + (k + 1) % 3], key = `${a},${b}`;
    edges.set(key, (edges.get(key) ?? 0) + 1);
  }
  let bad = 0;
  for (const [key, n] of edges) { const [a, b] = key.split(','); if (n !== 1 || edges.get(`${b},${a}`) !== 1) bad++; }
  assert(bad === 0, `${bad} directed edges not matched by exactly one opposite edge`);
  assert(nv - edges.size / 2 + nf === 2, `Euler characteristic ${nv - edges.size / 2 + nf}`);
  // outward: the normal of (almost) every face points away from the center
  let inward = 0;
  for (let f = 0; f < nf; f++) {
    const [a, b, c] = [0, 1, 2].map(k => [0, 1, 2].map(d => P[3 * F3[3 * f + k] + d]));
    const e1 = [0, 1, 2].map(d => b[d] - a[d]), e2 = [0, 1, 2].map(d => c[d] - a[d]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    if (n[0] * a[0] + n[1] * a[1] + n[2] * a[2] < 0) inward++;
  }
  assert(inward === 0, `${inward} of ${nf} faces point inwards`);
});

Deno.test('mesh: a floater seen by one view only is carved away by the views that see through it', () => {
  // fixed bounds: the depth percentiles leave the floater out (and the background in)
  const { grid, mesh: m } = mesh(views({ floater: true, background: true }), 64, { min: [-2, -2, -2], max: [2, 2, 2] });
  const P = m.positions;
  let off = 0;
  for (let i = 0; i < P.length / 3; i++) if (Math.abs(Math.hypot(P[3 * i], P[3 * i + 1], P[3 * i + 2]) - RADIUS) > grid.voxel) off++;
  assert(off === 0, `${off} vertices off the sphere`);
});

// directed edges of a mesh not matched by exactly one opposite edge (0: closed)
function openEdges({ faces: F3 }) {
  const edges = new Map();
  for (let f = 0; f < F3.length / 3; f++) for (let k = 0; k < 3; k++) {
    const key = `${F3[3 * f + k]},${F3[3 * f + (k + 1) % 3]}`;
    edges.set(key, (edges.get(key) ?? 0) + 1);
  }
  let bad = 0;
  for (const [key, n] of edges) { const [a, b] = key.split(','); if (n !== 1 || edges.get(`${b},${a}`) !== 1) bad++; }
  return bad;
}

Deno.test('mesh: a surface a few views see past stays, with their free-space votes at freeWeight (Save mesh: 0.1)', () => {
  // 4 of 20 views see through a patch of the sphere: counted in full, they carve a hole in it
  const vs = views({ background: true, seeThrough: 4 }), box = { min: [-2, -2, -2], max: [2, 2, 2] };
  const full = mesh(vs, 48, box, 1), light = mesh(vs, 48, box, 0.1);
  assert(openEdges(full.mesh) > 0, 'freeWeight 1: no hole');
  assert(openEdges(light.mesh) === 0, `freeWeight 0.1: ${openEdges(light.mesh)} open edges`);
  // the floater (1 view) is still carved at 0.1
  const fl = mesh(views({ floater: true, background: true }), 64, box, 0.1), P = fl.mesh.positions;
  let off = 0;
  for (let i = 0; i < P.length / 3; i++) if (Math.abs(Math.hypot(P[3 * i], P[3 * i + 1], P[3 * i + 2]) - RADIUS) > fl.grid.voxel) off++;
  assert(off === 0, `freeWeight 0.1: ${off} vertices off the sphere (the floater)`);
});

Deno.test('mesh: vertex colors from the views that see them; .ply layout', () => {
  const vs = views(), { grid, mesh: m } = mesh(vs, 24);
  const acc = new Float64Array(4 * m.positions.length / 3);
  for (const v of vs) colorVertices(m, v, grid.mu, acc);
  m.colors = finishColors(acc);
  const nv = m.positions.length / 3, nf = m.faces.length / 3;
  let seen = 0;
  for (let i = 0; i < nv; i++) if (m.colors[3 * i] === 200 && m.colors[3 * i + 1] === 100 && m.colors[3 * i + 2] === 50) seen++;
  assert(seen === nv, `${seen} of ${nv} vertices colored`);
  const ply = meshToPly(m), text = new TextDecoder().decode(ply.subarray(0, 400));
  const head = text.slice(0, text.indexOf('end_header\n') + 11);
  assert(head.includes(`element vertex ${nv}\n`) && head.includes(`element face ${nf}\n`), head);
  assert(ply.length === head.length + 15 * nv + 13 * nf, `${ply.length} bytes`);
  const dv = new DataView(ply.buffer, head.length + 15 * nv);
  assert(dv.getUint8(0) === 3 && dv.getInt32(1, true) === m.faces[0], 'first face');
  const back = meshFromPly(ply.buffer);
  assert(back.positions.every((x, i) => x === m.positions[i]) && back.faces.every((x, i) => x === m.faces[i]) &&
         back.colors.every((x, i) => x === m.colors[i]), 'meshFromPly(meshToPly(m)) differs from m');
});

Deno.test('mesh: snapToSurface moves centers near the fused sphere onto it, leaves far ones alone', () => {
  const { grid } = mesh(views());
  // 200 splat centers (16 floats each) in random directions: 100 at radius 1 +- 1.5 voxels, 100 at 1.5
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const n = 200, P = new Float32Array(16 * n), r0 = [];
  for (let i = 0; i < n; i++) {
    const z = 2 * rnd() - 1, a = 2 * Math.PI * rnd(), s = Math.sqrt(1 - z * z);
    const r = i < 100 ? RADIUS + (rnd() * 3 - 1.5) * grid.voxel : 1.5;
    P.set([r * s * Math.cos(a), r * s * Math.sin(a), r * z], 16 * i);
    r0.push(r);
  }
  const radius = (i) => Math.hypot(P[16 * i], P[16 * i + 1], P[16 * i + 2]);
  const before = Array.from({ length: 100 }, (_, i) => radius(i));
  const st = snapToSurface(P, n, grid, { amount: 1, minViews: 3 });
  // 87 move: some cells inside the sphere have a corner seen by fewer than 3 views
  const moved = before.map((r, i) => i).filter(i => Math.abs(radius(i) - before[i]) > 1e-7);
  assert(st.moved === moved.length && st.moved >= 80, `${st.moved} moved`);
  const mean = (f) => moved.reduce((a, i) => a + f(i), 0) / moved.length;
  const e0 = mean(i => Math.abs(before[i] - RADIUS)), e1 = mean(i => Math.abs(radius(i) - RADIUS));
  // measured: 0.032 -> 0.011 mean, at most 0.56 voxel off (2.8 without the 1-voxel cap)
  const worst = Math.max(...moved.map(i => Math.abs(radius(i) - RADIUS))) / grid.voxel;
  assert(e1 < 0.4 * e0 && worst < 0.7, `mean |r - 1| of the moved ${e0.toFixed(4)} -> ${e1.toFixed(4)}, worst ${worst.toFixed(2)} voxel`);
  for (let i = 100; i < n; i++) assert(Math.abs(radius(i) - r0[i]) < 1e-6, `far splat ${i} moved`);
});
