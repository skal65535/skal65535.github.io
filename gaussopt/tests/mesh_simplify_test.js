// mesh_simplify.js: quadric edge collapse on a flat grid with a hole and on a sphere.
import { simplify } from '../src/mesh_simplify.js';
import { assert } from './util.js';

// An n x n grid of unit squares at z = 0 (2 triangles each, counter-clockwise seen from +z), minus
// the squares in `hole` ([i0, i1) x [j0, j1)); color = (x, y, 0) * 10
function grid(n, hole = null) {
  const positions = [], colors = [], faces = [];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) { positions.push(i, j, 0); colors.push(10 * i, 10 * j, 0); }
  const id = (i, j) => j * (n + 1) + i;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    if (hole && i >= hole[0] && i < hole[1] && j >= hole[0] && j < hole[1]) continue;
    faces.push(id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1));
  }
  return { positions: new Float32Array(positions), colors: new Uint8Array(colors), faces: new Uint32Array(faces) };
}

// directed edges: each must have exactly one opposite (closed) or none (border); returns the border count
function edgeCheck(faces) {
  const e = new Map();
  for (let f = 0; f < faces.length; f += 3) for (let k = 0; k < 3; k++) {
    const key = `${faces[f + k]},${faces[f + (k + 1) % 3]}`;
    e.set(key, (e.get(key) ?? 0) + 1);
  }
  let border = 0;
  for (const [key, n] of e) {
    assert(n === 1, `directed edge ${key} in ${n} faces`);
    const [a, b] = key.split(',');
    if (!e.has(`${b},${a}`)) border++;
  }
  return { border, edges: e.size };
}

const area = (m) => {
  let s = 0;
  const P = m.positions, F = m.faces;
  for (let f = 0; f < F.length; f += 3) {
    const [a, b, c] = [0, 1, 2].map(k => [0, 1, 2].map(d => P[3 * F[f + k] + d]));
    s += ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2;  // signed, z = 0
  }
  return s;
};

Deno.test('simplify: a flat grid with a square hole keeps its plane, outline, hole and orientation in few faces', () => {
  const m = grid(20, [8, 12]);
  const { mesh } = simplify(m, { maxError: 1e-6 });
  const nf = mesh.faces.length / 3;
  assert(nf < 40, `${m.faces.length / 3} -> ${nf} faces`);
  for (let i = 0; i < mesh.positions.length; i += 3) assert(Math.abs(mesh.positions[i + 2]) < 1e-9, 'vertex off the plane');
  // the same area (400 - 16), all faces still counter-clockwise from +z: none flipped
  assert(Math.abs(area(mesh) - 384) < 1e-6, `area ${area(mesh)}`);
  const F = mesh.faces, P = mesh.positions;
  for (let f = 0; f < F.length; f += 3) {
    const [a, b, c] = [0, 1, 2].map(k => [P[3 * F[f + k]], P[3 * F[f + k] + 1]]);
    assert((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0, `face ${f / 3} flipped or degenerate`);
  }
  // the outline's and the hole's 8 corners are still vertices; colors stay with their vertices
  for (const [x, y] of [[0, 0], [20, 0], [20, 20], [0, 20], [8, 8], [12, 8], [12, 12], [8, 12]]) {
    let found = -1;
    for (let i = 0; i < P.length / 3; i++) if (P[3 * i] === x && P[3 * i + 1] === y) found = i;
    assert(found >= 0, `corner (${x}, ${y}) lost`);
    assert(mesh.colors[3 * found] === 10 * x && mesh.colors[3 * found + 1] === 10 * y, `corner (${x}, ${y}) color`);
  }
  edgeCheck(F);
});

Deno.test('simplify: a sphere stays closed and within maxError, with fewer faces', () => {
  // a UV-free sphere: an icosahedron subdivided 4 times (5120 faces), radius 1
  const t = (1 + Math.sqrt(5)) / 2;
  let V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let T = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
           [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const norm = (p) => p.map(x => x / Math.hypot(...p));
  V = V.map(norm);
  for (let s = 0; s < 4; s++) {
    const mid = new Map(), m = (a, b) => {
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      if (!mid.has(k)) { mid.set(k, V.length); V.push(norm(V[a].map((x, i) => (x + V[b][i]) / 2))); }
      return mid.get(k);
    };
    T = T.flatMap(([a, b, c]) => { const ab = m(a, b), bc = m(b, c), ca = m(c, a); return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]; });
  }
  const sphere = { positions: new Float32Array(V.flat()), faces: new Uint32Array(T.flat()) };
  const maxError = 0.02, { mesh } = simplify(sphere, { maxError });
  const nf = mesh.faces.length / 3;
  assert(nf < 0.5 * T.length && nf > 20, `${T.length} -> ${nf} faces`);
  const { border, edges } = edgeCheck(mesh.faces);
  const nv = mesh.positions.length / 3;
  assert(border === 0 && nv - edges / 2 + nf === 2, `border ${border}, Euler ${nv - edges / 2 + nf}`);
  // vertices within maxError of the sphere (they move only to an end or a midpoint: inside, by at most
  // the chord's sagitta)
  for (let i = 0; i < nv; i++) {
    const r = Math.hypot(mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2]);
    assert(Math.abs(r - 1) < maxError, `vertex ${i} at radius ${r}`);
  }
});

Deno.test('simplify: an icosahedron simplified without limit stays a closed surface (no pinched edges)', () => {
  const t = (1 + Math.sqrt(5)) / 2;
  const V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  const T = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
             [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const { mesh } = simplify({ positions: new Float32Array(V.flat()), faces: new Uint32Array(T.flat()) }, {});
  const nf = mesh.faces.length / 3, nv = mesh.positions.length / 3;
  const { border, edges } = edgeCheck(mesh.faces);
  assert(nf >= 4 && nf < 20 && border === 0 && nv - edges / 2 + nf === 2, `${nf} faces, ${nv} vertices, border ${border}, Euler ${nv - edges / 2 + nf}`);
});

Deno.test('simplify: a noisy plane (bumps of a third of a cell) does not fold: every face still faces up', () => {
  let s = 1;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const n = 60, P = [], F = [];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) P.push(i, j, (rnd() - 0.5) * 0.6);
  const id = (i, j) => j * (n + 1) + i;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) F.push(id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1));
  const { mesh } = simplify({ positions: new Float32Array(P), faces: new Uint32Array(F) }, { maxError: 1 });
  const Q = mesh.positions, G = mesh.faces;
  assert(G.length / 3 < 0.2 * F.length / 3, `${F.length / 3} -> ${G.length / 3} faces`);
  let down = 0;
  for (let f = 0; f < G.length; f += 3) {
    const [a, b, c] = [0, 1, 2].map(k => 3 * G[f + k]);
    if ((Q[b] - Q[a]) * (Q[c + 1] - Q[a + 1]) - (Q[b + 1] - Q[a + 1]) * (Q[c] - Q[a]) < 0) down++;
  }
  assert(down === 0, `${down} faces of ${G.length / 3} turned down`);
});
