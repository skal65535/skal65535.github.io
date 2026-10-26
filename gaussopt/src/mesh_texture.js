// mesh_texture.js — A texture for the meshes of src/mesh.js (DESIGN.md §9 E5), without a UV library:
// the mesh cut into charts (faces grown from a seed while they face within `maxAngle` of it), each
// chart projected flat along its mean normal, turned to its principal axes, and packed by shelves
// into a square atlas at the largest texel density that fits. Each texel of a face then gets the
// weighted mean of the views' pixels that see its point (the photos or the splat renders, as
// colorVertices does for vertices); texels no view sees keep the vertex colors, and the gutters
// around the charts are filled from their neighbours (bilinear lookups at chart borders).
"use strict";

// mesh: { positions: Float32Array 3V, faces: Uint32Array 3F, colors?: Uint8Array 3V }
// size: the atlas side (texels); pad: empty texels around each chart
// Returns { mesh: { positions, faces, colors, uvs: Float32Array 2V' } (vertices on chart borders
// duplicated, one per chart), size, density (texels per world unit), info }.
export function atlas(mesh, { size = 2048, maxAngle = 60, pad = 2, smooth = 2 } = {}) {
  const P = mesh.positions, F = mesh.faces, nf = F.length / 3, nv = P.length / 3;
  const fn = new Float64Array(3 * nf), fa = new Float64Array(nf);
  for (let f = 0; f < nf; f++) {
    const a = 3 * F[3 * f], b = 3 * F[3 * f + 1], c = 3 * F[3 * f + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx, l = Math.hypot(x, y, z);
    fa[f] = l / 2;
    if (l > 0) { fn[3 * f] = x / l; fn[3 * f + 1] = y / l; fn[3 * f + 2] = z / l; }
  }
  // faces around each vertex (charts grow across shared vertices: non-manifold edges and pinches included)
  const start = new Uint32Array(nv + 1);
  for (let i = 0; i < F.length; i++) start[F[i] + 1]++;
  for (let v = 0; v < nv; v++) start[v + 1] += start[v];
  const vf = new Uint32Array(F.length), fill = start.slice(0, nv);
  for (let i = 0; i < F.length; i++) vf[fill[F[i]]++] = (i / 3) | 0;
  // smoothed face normals (face -> vertex -> face, `smooth` times): the TSDF surface is bumpy at the
  // voxel scale, and simplified it folds (6% of the area at full size, 20-40% simplified, corner1)
  let sn = fn;
  for (let it = 0; it < smooth; it++) {
    const vn = new Float64Array(3 * nv), next = new Float64Array(3 * nf);
    for (let f = 0; f < nf; f++) for (let k = 0; k < 3; k++) for (let c = 0; c < 3; c++) vn[3 * F[3 * f + k] + c] += fa[f] * sn[3 * f + c];
    for (let f = 0; f < nf; f++) {
      let x = 0, y = 0, z = 0;
      for (let k = 0; k < 3; k++) { const v = 3 * F[3 * f + k]; x += vn[v]; y += vn[v + 1]; z += vn[v + 2]; }
      const l = Math.hypot(x, y, z);
      if (l > 0) { next[3 * f] = x / l; next[3 * f + 1] = y / l; next[3 * f + 2] = z / l; }
    }
    sn = next;
  }
  // charts, seeded from the largest faces left, grown while the smoothed normals stay within maxAngle
  // of the seed's (the folds inside a chart overlap in it; texelsOf gives their texels to the face
  // that faces the chart's normal most)
  const chart = new Int32Array(nf).fill(-1), cosMax = Math.cos(maxAngle * Math.PI / 180);
  const order = Array.from({ length: nf }, (_, i) => i).sort((i, j) => fa[j] - fa[i]);
  const charts = [], queue = new Int32Array(nf);
  for (const s of order) {
    if (chart[s] >= 0) continue;
    const id = charts.length, sx = sn[3 * s], sy = sn[3 * s + 1], sz = sn[3 * s + 2];
    let head = 0, tail = 0;
    queue[tail++] = s; chart[s] = id;
    while (head < tail) {
      const f = queue[head++];
      for (let k = 0; k < 3; k++) for (let j = start[F[3 * f + k]], je = start[F[3 * f + k] + 1]; j < je; j++) {
        const g = vf[j];
        if (chart[g] >= 0) continue;
        if (fa[g] > 0 && sn[3 * g] * sx + sn[3 * g + 1] * sy + sn[3 * g + 2] * sz < cosMax) continue;
        chart[g] = id; queue[tail++] = g;
      }
    }
    charts.push(queue.slice(0, tail));
  }
  // each chart flat: its vertices (duplicated per chart) in the plane of its mean normal, along its
  // principal axes; xy: 2 per output vertex, in world units from the chart's corner
  const vstamp = new Int32Array(nv).fill(-1), vlocal = new Int32Array(nv);
  const src = [], xy = [], outF = new Uint32Array(3 * nf), rects = [];
  let nOut = 0, o = 0;
  for (let c = 0; c < charts.length; c++) {
    const fs = charts[c];
    let nx = 0, ny = 0, nz = 0;
    for (const f of fs) { nx += fa[f] * fn[3 * f]; ny += fa[f] * fn[3 * f + 1]; nz += fa[f] * fn[3 * f + 2]; }
    let l = Math.hypot(nx, ny, nz);
    if (!(l > 0)) { nx = 0; ny = 0; nz = 1; l = 1; }
    nx /= l; ny /= l; nz /= l;
    // basis u, v perpendicular to n
    const ax = Math.abs(nx) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let ux = ny * ax[2] - nz * ax[1], uy = nz * ax[0] - nx * ax[2], uz = nx * ax[1] - ny * ax[0];
    const ul = Math.hypot(ux, uy, uz); ux /= ul; uy /= ul; uz /= ul;
    const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
    const first = nOut, facing = (f) => fn[3 * f] * nx + fn[3 * f + 1] * ny + fn[3 * f + 2] * nz;
    fs.sort((f, g) => facing(g) - facing(f));
    for (const f of fs) for (let k = 0; k < 3; k++) {
      const v = F[3 * f + k];
      if (vstamp[v] !== c) {
        vstamp[v] = c; vlocal[v] = nOut++; src.push(v);
        const x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2];
        xy.push(x * ux + y * uy + z * uz, x * vx + y * vy + z * vz);
      }
      outF[o++] = vlocal[v];
    }
    // principal axes of the chart's vertices: the long one along x
    let mx = 0, my = 0;
    for (let i = first; i < nOut; i++) { mx += xy[2 * i]; my += xy[2 * i + 1]; }
    mx /= nOut - first; my /= nOut - first;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = first; i < nOut; i++) { const dx = xy[2 * i] - mx, dy = xy[2 * i + 1] - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy), co = Math.cos(th), si = Math.sin(th);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = first; i < nOut; i++) {
      const dx = xy[2 * i] - mx, dy = xy[2 * i + 1] - my, x = co * dx + si * dy, y = -si * dx + co * dy;
      xy[2 * i] = x; xy[2 * i + 1] = y;
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    for (let i = first; i < nOut; i++) { xy[2 * i] -= x0; xy[2 * i + 1] -= y0; }
    rects.push({ c, w: x1 - x0, h: y1 - y0, first, end: nOut });
  }
  // the largest density whose shelf packing fits (binary search on a log scale)
  const pack = (d, place) => {
    const sorted = rects.map((r, i) => i).sort((i, j) => rects[j].h - rects[i].h);
    let x = 0, y = 0, shelf = 0;
    for (const i of sorted) {
      const r = rects[i], w = Math.ceil(r.w * d) + 2 * pad, h = Math.ceil(r.h * d) + 2 * pad;
      if (w > size) return false;
      if (x + w > size) { y += shelf; x = 0; shelf = 0; }
      if (y + h > size) return false;
      if (place) { r.x = x; r.y = y; }
      x += w; shelf = Math.max(shelf, h);
    }
    return true;
  };
  let area = 0;
  for (let f = 0; f < nf; f++) area += fa[f];
  let lo = 0, hi = size / Math.sqrt(Math.max(area, 1e-30));
  for (let it = 0; it < 40 && hi - lo > 1e-4 * hi; it++) { const d = (lo + hi) / 2; if (pack(d, false)) lo = d; else hi = d; }
  const density = lo;
  if (!(density > 0) || !pack(density, true)) throw new Error(`atlas: ${charts.length} charts do not fit ${size} x ${size}`);
  // output vertices: texel coordinates -> uv in [0, 1]
  const positions = new Float32Array(3 * nOut), colors = new Uint8Array(3 * nOut), uvs = new Float32Array(2 * nOut);
  const tx = new Float64Array(2 * nOut);
  for (const r of rects) for (let i = r.first; i < r.end; i++) {
    tx[2 * i] = r.x + pad + xy[2 * i] * density; tx[2 * i + 1] = r.y + pad + xy[2 * i + 1] * density;
    uvs[2 * i] = tx[2 * i] / size; uvs[2 * i + 1] = tx[2 * i + 1] / size;
  }
  for (let i = 0; i < nOut; i++) for (let k = 0; k < 3; k++) {
    positions[3 * i + k] = P[3 * src[i] + k];
    colors[3 * i + k] = mesh.colors ? mesh.colors[3 * src[i] + k] : 128;
  }
  const sizes = charts.map(c => c.length).sort((a, b) => a - b);
  return { mesh: { positions, faces: outF, colors, uvs }, size, density,
           info: { charts: charts.length, single: sizes.filter(n => n === 1).length, medianFaces: sizes[sizes.length >> 1] ?? 0, vertices: nOut,
                   rects: rects.reduce((s, r) => s + (Math.ceil(r.w * density) + 2 * pad) * (Math.ceil(r.h * density) + 2 * pad), 0) / (size * size),
                   largest: Math.max(...rects.map(r => (Math.ceil(r.w * density) + 2 * pad) * (Math.ceil(r.h * density) + 2 * pad))) / (size * size) } };
}

// The texels whose centers the faces of `mesh` (with uvs, in [0, 1]) cover in a size x size image:
// { index (y * size + x), pos (the 3D point), normal (the face's), base (the interpolated vertex
// color) }, the texels covered by more than one face (overlap, kept by the first in face order), and
// texelSize: a texel's side in world units (from the faces' areas in 3D and in the image).
export function texelsOf({ positions, faces, colors, uvs }, size) {
  const nf = faces.length / 3, outF = faces;
  const owner = new Int32Array(size * size).fill(-1);
  const idx = [], pos = [], nrm = [], base = [];
  let overlap = 0, area3 = 0, area2 = 0;
  for (let f = 0; f < nf; f++) {
    const a = outF[3 * f], b = outF[3 * f + 1], c = outF[3 * f + 2];
    const ax = uvs[2 * a] * size, ay = uvs[2 * a + 1] * size, bx = uvs[2 * b] * size, by = uvs[2 * b + 1] * size, cx = uvs[2 * c] * size, cy = uvs[2 * c + 1] * size;
    const pa = 3 * a, pb = 3 * b, pc = 3 * c;
    const ux = positions[pb] - positions[pa], uy = positions[pb + 1] - positions[pa + 1], uz = positions[pb + 2] - positions[pa + 2];
    const vx = positions[pc] - positions[pa], vy = positions[pc + 1] - positions[pa + 1], vz = positions[pc + 2] - positions[pa + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
    area3 += nl / 2; area2 += Math.abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2;
    const den = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (!(Math.abs(den) > 1e-12)) continue;
    const xs = Math.max(0, Math.floor(Math.min(ax, bx, cx))), xe = Math.min(size - 1, Math.ceil(Math.max(ax, bx, cx)));
    const ys = Math.max(0, Math.floor(Math.min(ay, by, cy))), ye = Math.min(size - 1, Math.ceil(Math.max(ay, by, cy)));
    for (let y = ys; y <= ye; y++) for (let x = xs; x <= xe; x++) {
      const px = x + 0.5, py = y + 0.5;
      const wb = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / den, wc = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / den, wa = 1 - wb - wc;
      if (wa < 0 || wb < 0 || wc < 0) continue;
      const t = y * size + x;
      if (owner[t] >= 0) { overlap++; continue; }
      owner[t] = f;
      idx.push(t); nrm.push(nx, ny, nz);
      for (let k = 0; k < 3; k++) {
        pos.push(wa * positions[3 * a + k] + wb * positions[3 * b + k] + wc * positions[3 * c + k]);
        base.push(colors ? Math.round(wa * colors[3 * a + k] + wb * colors[3 * b + k] + wc * colors[3 * c + k]) : 128);
      }
    }
  }
  return { index: Uint32Array.from(idx), pos: Float32Array.from(pos), normal: Float32Array.from(nrm), base: Uint8Array.from(base), overlap,
           texelSize: area2 > 0 ? Math.sqrt(area3 / area2) : 0 };
}

// One view into the texels' running sums (acc: Float64Array 4T, r g b weight), as colorVertices:
// a texel counts when it is within `tol` of the view's depth; weight: the view's pixels per unit of
// surface there, (cos * f / z)^2 (near, head-on views count most), to the power `power` (> 1: the best
// view dominates, fewer ghosts where the views disagree). view: { R, t, fx, fy, cx, cy, W, H,
// depth, rgb, pitch }, rgb sampled bilinearly; or with `levels` (the photo larger than W x H, and its
// 2 x 2 means: mesh_image.js mipLevels) in place of rgb, the level whose pixels are the closest below
// the texel's size there (texels.texelSize at depth z), so a texel larger than the photo's pixels
// gets their mean.
export function bakeView(texels, view, tol, acc, { weight = 'footprint', power = 1 } = {}) {
  const { R, t, fx, fy, cx, cy, W, H, depth, levels } = view, { pos, normal } = texels, n = texels.index.length;
  const lv = levels ?? [{ data: view.rgb, width: W, height: H, pitch: view.pitch }];
  const ts = fx * (lv[0].width / W) * texels.texelSize;  // level-0 pixels per texel at depth 1
  const ox = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), oy = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), oz = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
  const f2 = fx * fy;
  for (let i = 0; i < n; i++) {
    const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    const zc = R[6] * x + R[7] * y + R[8] * z + t[2];
    if (zc <= 1e-6) continue;
    const u = fx * (R[0] * x + R[1] * y + R[2] * z + t[0]) / zc + cx, v = fy * (R[3] * x + R[4] * y + R[5] * z + t[1]) / zc + cy;
    const ui = Math.floor(u), vi = Math.floor(v);
    if (ui < 0 || vi < 0 || ui >= W || vi >= H || !(Math.abs(depth[vi * W + ui] - zc) < tol)) continue;
    const dx = ox - x, dy = oy - y, dz = oz - z;
    const cos = Math.abs(dx * normal[3 * i] + dy * normal[3 * i + 1] + dz * normal[3 * i + 2]) / Math.hypot(dx, dy, dz);
    const fp = cos * cos * f2 / (zc * zc), w = weight === 'uniform' ? 1 : weight === 'cos' ? cos : power === 1 ? fp : fp ** power;
    // bilinear, between pixel centers (clamped at the image border)
    const L = lv.length > 1 && ts > zc ? Math.min(lv.length - 1, Math.floor(Math.log2(ts / zc))) : 0, img = lv[L], rgb = img.data;
    const iw = img.width, ih = img.height, pitch = img.pitch ?? 4 * iw, k = iw / W;
    const sx = Math.min(Math.max(u * k - 0.5, 0), iw - 1), sy = Math.min(Math.max(v * k - 0.5, 0), ih - 1);
    const x0 = Math.min(Math.floor(sx), iw - 2), y0 = Math.min(Math.floor(sy), ih - 2), ax = sx - x0, ay = sy - y0;
    const p00 = y0 * pitch + 4 * x0, p10 = p00 + 4, p01 = p00 + pitch, p11 = p01 + 4;
    for (let c = 0; c < 3; c++) {
      const top = rgb[p00 + c] + ax * (rgb[p10 + c] - rgb[p00 + c]), bot = rgb[p01 + c] + ax * (rgb[p11 + c] - rgb[p01 + c]);
      acc[4 * i + c] += w * (top + ay * (bot - top));
    }
    acc[4 * i + 3] += w;
  }
}

// The atlas image (RGBA bytes, size x size): the texels' weighted means (their vertex colors where no
// view saw them), then `grow` rings of empty texels around them filled with their filled neighbours'
// mean (what bilinear lookups at chart borders read), then (fill) every other texel from a pyramid of
// means of the filled ones (pull-push), so that mip levels and JPEG blocks at chart edges do not mix in
// black. Alpha: 255 on the faces' texels and the rings, 0 on the rest.
export function finishTexture(texels, size, acc, { grow = 4, fill = true } = {}) {
  const img = new Uint8ClampedArray(4 * size * size), filled = new Uint8Array(size * size);
  let seen = 0;
  for (let i = 0, n = texels.index.length; i < n; i++) {
    const t = texels.index[i], w = acc ? acc[4 * i + 3] : 0;
    for (let c = 0; c < 3; c++) img[4 * t + c] = w > 0 ? acc[4 * i + c] / w : texels.base[3 * i + c];
    img[4 * t + 3] = 255; filled[t] = 1;
    if (w > 0) seen++;
  }
  let ring = [];
  for (let g = 0; g < grow; g++) {
    ring = [];
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const t = y * size + x;
      if (filled[t]) continue;
      let r = 0, gr = 0, b = 0, k = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= size || yy >= size || filled[yy * size + xx] !== 1) continue;
        const q = 4 * (yy * size + xx);
        r += img[q]; gr += img[q + 1]; b += img[q + 2]; k++;
      }
      if (k) { img[4 * t] = r / k; img[4 * t + 1] = gr / k; img[4 * t + 2] = b / k; img[4 * t + 3] = 255; ring.push(t); }
    }
    for (const t of ring) filled[t] = 1;
    if (!ring.length) break;
  }
  if (fill) pullPush(img, filled, size);
  return { image: img, seen: texels.index.length ? seen / texels.index.length : 0 };
}

// The texels not `filled` of an RGBA image (size x size) get the mean of the filled ones at the finest
// level of a 2 x 2 mean pyramid that has some (their alpha stays 0)
function pullPush(img, filled, size) {
  const levels = [{ c: Float32Array.from({ length: 3 * size * size }, (_, i) => img[4 * ((i / 3) | 0) + i % 3]), w: Float32Array.from(filled), n: size }];
  while (levels.at(-1).n > 1) {
    const f = levels.at(-1), n = Math.ceil(f.n / 2), c = new Float32Array(3 * n * n), w = new Float32Array(n * n);
    for (let y = 0; y < f.n; y++) for (let x = 0; x < f.n; x++) {
      const p = y * f.n + x, q = (y >> 1) * n + (x >> 1), wp = f.w[p];
      if (!wp) continue;
      c[3 * q] += wp * f.c[3 * p]; c[3 * q + 1] += wp * f.c[3 * p + 1]; c[3 * q + 2] += wp * f.c[3 * p + 2]; w[q] += wp;
    }
    for (let q = 0; q < n * n; q++) if (w[q] > 0) { c[3 * q] /= w[q]; c[3 * q + 1] /= w[q]; c[3 * q + 2] /= w[q]; w[q] = 1; }
    levels.push({ c, w, n });
  }
  for (let l = levels.length - 2; l >= 0; l--) {
    const f = levels[l], g = levels[l + 1];
    for (let y = 0; y < f.n; y++) for (let x = 0; x < f.n; x++) {
      const p = y * f.n + x, q = (y >> 1) * g.n + (x >> 1);
      if (f.w[p] || !g.w[q]) continue;
      f.c[3 * p] = g.c[3 * q]; f.c[3 * p + 1] = g.c[3 * q + 1]; f.c[3 * p + 2] = g.c[3 * q + 2]; f.w[p] = 1;
    }
  }
  const c = levels[0].c;
  for (let p = 0; p < size * size; p++) if (!filled[p]) { img[4 * p] = c[3 * p]; img[4 * p + 1] = c[3 * p + 1]; img[4 * p + 2] = c[3 * p + 2]; }
}
