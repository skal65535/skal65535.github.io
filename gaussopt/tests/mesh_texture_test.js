// mesh_texture.js: the atlas of a sphere (charts that do not overlap, uvs in the image, the space
// between the charts filled), and a plane
// baked from a view of a known image (the texels get the image's colors at their points; a view whose
// depth says something is in front adds nothing; texels no view sees keep their vertex colors; the
// gutters are filled).
import { atlas, texelsOf, bakeView, finishTexture } from '../src/mesh_texture.js';
import { mipLevels } from '../src/mesh_image.js';
import { assert } from './util.js';

function sphere(subdiv) {
  const t = (1 + Math.sqrt(5)) / 2, norm = (p) => p.map(x => x / Math.hypot(...p));
  let V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map(norm);
  let T = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
           [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  for (let s = 0; s < subdiv; s++) {
    const mid = new Map(), m = (a, b) => {
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      if (!mid.has(k)) { mid.set(k, V.length); V.push(norm(V[a].map((x, i) => (x + V[b][i]) / 2))); }
      return mid.get(k);
    };
    T = T.flatMap(([a, b, c]) => { const ab = m(a, b), bc = m(b, c), ca = m(c, a); return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]; });
  }
  return { positions: new Float32Array(V.flat()), faces: new Uint32Array(T.flat()), colors: new Uint8Array(3 * V.length).fill(77) };
}

Deno.test('atlas: a sphere in a few charts, uvs inside the image, no texel covered twice, every face with texels', () => {
  const m = sphere(3), size = 512, a = atlas(m, { size });
  assert(a.info.charts >= 4 && a.info.charts <= 40, `${a.info.charts} charts`);
  assert(a.mesh.faces.length === m.faces.length, 'faces kept');
  for (let i = 0; i < a.mesh.uvs.length; i++) assert(a.mesh.uvs[i] >= 0 && a.mesh.uvs[i] <= 1, `uv ${i}: ${a.mesh.uvs[i]}`);
  // the duplicated vertices are copies of the sphere's
  for (let i = 0; i < a.mesh.positions.length / 3; i++) {
    const r = Math.hypot(a.mesh.positions[3 * i], a.mesh.positions[3 * i + 1], a.mesh.positions[3 * i + 2]);
    assert(Math.abs(r - 1) < 1e-6 && a.mesh.colors[3 * i] === 77, `vertex ${i}`);
  }
  const tx = texelsOf(a.mesh, size);
  assert(tx.overlap === 0, `${tx.overlap} texels covered twice`);
  assert(tx.index.length > 0.3 * size * size, `${tx.index.length} texels used of ${size * size}`);
  // unseen: every texel the vertex color, also those far from any chart (filled, alpha 0)
  const { image } = finishTexture(tx, size, null);
  let far = 0;
  for (let t = 0; t < size * size; t++) {
    assert(image[4 * t] === 77 && image[4 * t + 2] === 77, `texel ${t}: ${image[4 * t]}`);
    if (image[4 * t + 3] === 0) far++;
  }
  assert(far > 0.05 * size * size, `${far} texels away from the charts`);
  // texel points on the sphere, texel normals radial
  for (let i = 0; i < tx.index.length; i += 97) {
    const p = [0, 1, 2].map(k => tx.pos[3 * i + k]), r = Math.hypot(...p), n = [0, 1, 2].map(k => tx.normal[3 * i + k]);
    assert(r > 0.98 && r < 1.0001, `texel ${i} at radius ${r}`);
    assert(Math.abs(n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) / r > 0.97, `texel ${i} normal`);
  }
});

// The unit square at z = 0 (an 8 x 8 grid), seen from (0.5, 0.5, 2) looking down; the image's color at
// a pixel: (255 x, 255 y, 40) of the plane's point there
function plane() {
  const n = 8, P = [], Fc = [];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) P.push(i / n, j / n, 0);
  const id = (i, j) => j * (n + 1) + i;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) Fc.push(id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1));
  return { positions: new Float32Array(P), faces: new Uint32Array(Fc), colors: new Uint8Array(3 * P.length / 3).fill(200) };
}
function view(depthValue, color, scale = 1) {
  const W = 160, H = 160, fx = 120, fy = 120, cx = 80, cy = 80;
  const R = [1, 0, 0, 0, -1, 0, 0, 0, -1], C = [0.5, 0.5, 2], t = [0, 1, 2].map(r => -(R[3 * r] * C[0] + R[3 * r + 1] * C[1] + R[3 * r + 2] * C[2]));
  // the image at `scale` times the view's size (scale > 1: as `levels`, a photo larger than the depth)
  const w = scale * W, h = scale * H, pitch = 4 * w, rgb = new Uint8ClampedArray(pitch * h), depth = new Float32Array(W * H).fill(depthValue);
  for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
    const xc = ((u + 0.5) / scale - cx) / fx * 2, yc = ((v + 0.5) / scale - cy) / fy * 2, zc = 2;  // camera coordinates of the plane point
    const X = [0, 1, 2].map(k => R[k] * (xc - t[0]) + R[3 + k] * (yc - t[1]) + R[6 + k] * (zc - t[2]));
    const c = color(X, u, v);
    rgb.set([c[0], c[1], c[2], 255], v * pitch + 4 * u);
  }
  return { R, t, fx, fy, cx, cy, W, H, depth, ...(scale > 1 ? { levels: mipLevels({ data: rgb, width: w, height: h }) } : { rgb, pitch }) };
}

Deno.test('bakeView + finishTexture: a plane gets the view\'s colors; an occluded view adds nothing; unseen texels keep the vertex colors; gutters filled', () => {
  const size = 128, a = atlas(plane(), { size, pad: 2 }), tx = texelsOf(a.mesh, size);
  assert(a.info.charts === 1, `${a.info.charts} charts`);
  const acc = new Float32Array(4 * tx.index.length);
  bakeView(tx, view(2, (X) => [255 * X[0], 255 * X[1], 40]), 0.05, acc);
  bakeView(tx, view(0.5, () => [0, 0, 255]), 0.05, acc);  // something in front of the plane everywhere: no texel sees this view
  const { image, seen } = finishTexture(tx, size, acc);
  assert(seen === 1, `seen ${seen}`);
  let worst = 0;
  for (let i = 0; i < tx.index.length; i++) {
    const t = tx.index[i], x = tx.pos[3 * i], y = tx.pos[3 * i + 1];
    worst = Math.max(worst, Math.abs(image[4 * t] - 255 * x), Math.abs(image[4 * t + 1] - 255 * y), Math.abs(image[4 * t + 2] - 40));
  }
  assert(worst <= 3, `texel color off by ${worst}`);
  // no view: the vertex colors; texels next to the chart (gutter) filled
  const bare = finishTexture(tx, size, null).image, used = new Uint8Array(size * size);
  for (const t of tx.index) { used[t] = 1; assert(bare[4 * t] === 200 && bare[4 * t + 3] === 255, `unseen texel ${t}`); }
  let gutter = 0;
  for (let y = 1; y < size - 1; y++) for (let x = 1; x < size - 1; x++) {
    const t = y * size + x;
    if (used[t]) continue;
    const near = used[t - 1] || used[t + 1] || used[t - size] || used[t + size];
    if (near) { gutter++; assert(bare[4 * t + 3] === 255 && bare[4 * t] === 200, `gutter texel ${x},${y}`); }
  }
  assert(gutter > 0, 'no gutter texel');
});

Deno.test('bakeView: a photo 8 x the depth\'s size is read at the level of the texels\' size (a one-pixel checker averages out)', () => {
  const size = 128, a = atlas(plane(), { size, pad: 2 }), tx = texelsOf(a.mesh, size);
  // ~3.9 photo pixels per texel: level 1 (2 x 2 means) cancels the checker; level 0 would sample it
  const checker = (X, u, v) => { const d = (u + v) % 2 ? 50 : -50; return [50 + 155 * X[0] + d, 50 + 155 * X[1] + d, 100 + d]; };  // within 0-255
  const acc = new Float32Array(4 * tx.index.length);
  bakeView(tx, view(2, checker, 8), 0.05, acc);
  const { image } = finishTexture(tx, size, acc);
  let worst = 0;
  for (let i = 0; i < tx.index.length; i++) {
    const t = tx.index[i], x = tx.pos[3 * i], y = tx.pos[3 * i + 1];
    worst = Math.max(worst, Math.abs(image[4 * t] - 50 - 155 * x), Math.abs(image[4 * t + 1] - 50 - 155 * y), Math.abs(image[4 * t + 2] - 100));
  }
  assert(worst <= 4, `texel color off by ${worst}`);
});
