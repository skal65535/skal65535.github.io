'use strict';

// Phase 2: initial state computation.
// Browser: depends on triangle-core.js + triangle-ans-enc.js loaded first.
// Node.js: auto-requires triangle-core.js for globals (Delaunay, RGBtoYCoCg, ...).
if (typeof module !== 'undefined' && typeof Delaunay === 'undefined') {
  Object.assign(global, require('./triangle-core.js'));
}

// ---------------------------------------------------------------------------
// Downsample imageData to gx×gy grid; returns Float32Array [gx*gy*4] RGBA.
function sampleGrid(imageData, gx, gy) {
  const { data, width: sw, height: sh } = imageData;
  const out = new Float32Array(gx * gy * 4);
  for (let gy_i = 0; gy_i < gy; ++gy_i) {
    for (let gx_i = 0; gx_i < gx; ++gx_i) {
      const x0 = Math.round(gx_i * sw / gx);
      const x1 = Math.max(x0 + 1, Math.round((gx_i + 1) * sw / gx));
      const y0 = Math.round(gy_i * sh / gy);
      const y1 = Math.max(y0 + 1, Math.round((gy_i + 1) * sh / gy));
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let py = y0; py < Math.min(y1, sh); ++py) {
        for (let px = x0; px < Math.min(x1, sw); ++px) {
          const off = (py * sw + px) * 4;
          r += data[off]; g += data[off+1]; b += data[off+2]; a += data[off+3]; ++n;
        }
      }
      const s = 1 / Math.max(n, 1), idx = (gy_i * gx + gx_i) * 4;
      out[idx] = r*s; out[idx+1] = g*s; out[idx+2] = b*s; out[idx+3] = a*s;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Weighted YCoCg perceptual distance.
function ycoCgDist(a, b) {
  const dy = a.y-b.y, dco = a.co-b.co, dcg = a.cg-b.cg;
  return 0.3*dy*dy + 0.1*dco*dco + 0.6*dcg*dcg;
}

// ---------------------------------------------------------------------------
// K-means palette in YCoCg space (max-min init, 20 iterations).
function buildPalette(grid, nb_colors) {
  const n = grid.length >> 2;
  const pts = new Array(n);
  for (let i = 0; i < n; ++i)
    pts[i] = RGBtoYCoCg(grid[i*4], grid[i*4+1], grid[i*4+2], grid[i*4+3]);

  const centers = [Object.assign({}, pts[0])];
  while (centers.length < nb_colors) {
    let best = 0, bestD = -1;
    for (let i = 0; i < n; ++i) {
      let d = Infinity;
      for (const c of centers) { const dd = ycoCgDist(pts[i], c); if (dd < d) d = dd; }
      if (d > bestD) { bestD = d; best = i; }
    }
    centers.push(Object.assign({}, pts[best]));
  }

  const labels = new Int32Array(n);
  for (let iter = 0; iter < 20; ++iter) {
    for (let i = 0; i < n; ++i) {
      let best = 0, bestD = Infinity;
      for (let k = 0; k < nb_colors; ++k) {
        const d = ycoCgDist(pts[i], centers[k]); if (d < bestD) { bestD = d; best = k; }
      }
      labels[i] = best;
    }
    const sy = new Float32Array(nb_colors), sco = new Float32Array(nb_colors);
    const scg = new Float32Array(nb_colors), sa = new Float32Array(nb_colors);
    const cnt = new Int32Array(nb_colors);
    for (let i = 0; i < n; ++i) {
      const k = labels[i], p = pts[i];
      sy[k] += p.y; sco[k] += p.co; scg[k] += p.cg; sa[k] += p.a; cnt[k]++;
    }
    for (let k = 0; k < nb_colors; ++k) {
      if (cnt[k] > 0) centers[k] = {
        y: Math.round(sy[k]/cnt[k]), co: Math.round(sco[k]/cnt[k]),
        cg: Math.round(scg[k]/cnt[k]), a: sa[k]/cnt[k] > 0.5 ? 1 : 0,
      };
    }
  }
  return centers;
}

// ---------------------------------------------------------------------------
// Barycentric coordinates of (px,py) in triangle (x0,y0)-(x1,y1)-(x2,y2).
// Returns [u,v,w] (all >= 0 if inside) or null if degenerate/outside.
function computeBary(px, py, x0, y0, x1, y1, x2, y2) {
  const d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
  if (Math.abs(d) < 1e-8) return null;
  const u = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / d;
  const v = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / d;
  const w = 1 - u - v;
  return (u >= -1e-4 && v >= -1e-4 && w >= -1e-4) ? [u, v, w] : null;
}

// ---------------------------------------------------------------------------
// CPU Gouraud rasterizer onto gx×gy grid.
// vtx: array of Vtx; triangles: from Delaunay.getTriangles(); palRGB: [{r,g,b}].
// Returns Float32Array [gx*gy*4] RGBA.
function rasterizeGrid(gx, gy, vtx, triangles, palRGB) {
  const out = new Float32Array(gx * gy * 4);
  for (let y = 0; y < gy; ++y) {
    for (let x = 0; x < gx; ++x) {
      for (const tri of triangles) {
        const v0 = vtx[tri.vtx[0]], v1 = vtx[tri.vtx[1]], v2 = vtx[tri.vtx[2]];
        const bary = computeBary(x, y, v0.x, v0.y, v1.x, v1.y, v2.x, v2.y);
        if (!bary) continue;
        const c0 = palRGB[v0.idx], c1 = palRGB[v1.idx], c2 = palRGB[v2.idx];
        const i = (y*gx + x)*4;
        out[i]   = bary[0]*c0.r + bary[1]*c1.r + bary[2]*c2.r;
        out[i+1] = bary[0]*c0.g + bary[1]*c1.g + bary[2]*c2.g;
        out[i+2] = bary[0]*c0.b + bary[1]*c1.b + bary[2]*c2.b;
        out[i+3] = bary[0]*c0.a + bary[1]*c1.a + bary[2]*c2.a;
        break;
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Find nearest palette entry (YCoCg distance) for a given YCoCg sample.
function findClosestPalette(ycocg, palette) {
  let best = 0, bestD = Infinity;
  for (let k = 0; k < palette.length; ++k) {
    const d = ycoCgDist(ycocg, palette[k]); if (d < bestD) { bestD = d; best = k; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Greedy vertex placement: 4 corners + nb_border evenly-spaced boundary pts +
// greedy SAD-residual interior pts to fill the remaining budget.
// nb_border defaults to Math.round((gx + gy) / 4).
function placeVertices(grid, gx, gy, palette, nb_pts, nb_border, random_init = false) {
  if (nb_border === undefined) nb_border = Math.round((gx + gy) / 4);
  const palRGB = palette.map(p => YCoCg_to_RGB(p.y, p.co, p.cg, p.a));
  const getYCoCg = (x, y) => RGBtoYCoCg(
    grid[(y*gx+x)*4], grid[(y*gx+x)*4+1], grid[(y*gx+x)*4+2], grid[(y*gx+x)*4+3]);
  const mkVtx   = (x, y) => ({ x, y, idx: findClosestPalette(getYCoCg(x, y), palette) });

  const qpts  = [mkVtx(0, 0), mkVtx(gx-1, 0), mkVtx(0, gy-1), mkVtx(gx-1, gy-1)];
  const vtxSet = new Set(qpts.map(v => v.y*gx + v.x));

  // Phase 1: evenly-spaced boundary vertices (left, right, top, bottom edges).
  const borderPts = [];
  for (let i = 1; i < gy-1; ++i) { borderPts.push({x:0,    y:i}); borderPts.push({x:gx-1, y:i}); }
  for (let i = 1; i < gx-1; ++i) { borderPts.push({x:i,    y:0}); borderPts.push({x:i,    y:gy-1}); }
  const n_border = Math.min(nb_border, nb_pts, borderPts.length);
  for (let k = 0; k < n_border; ++k) {
    const { x, y } = borderPts[Math.round(k * borderPts.length / n_border) % borderPts.length];
    const key = y*gx + x;
    if (!vtxSet.has(key)) { qpts.push(mkVtx(x, y)); vtxSet.add(key); }
  }

  // Phase 2: place remaining points — greedily up to MAX_GREEDY, then randomly.
  // Greedy is O(n²): rebuilds Delaunay+rasterize each step, so cap it to stay responsive.
  const MAX_GREEDY = 128;
  const nRemaining = nb_pts - (qpts.length - 4);
  const n_greedy   = random_init ? 0 : Math.min(nRemaining, MAX_GREEDY);
  const n_random   = nRemaining - n_greedy;

  for (let k = 0; k < n_greedy; ++k) {
    const del      = new Delaunay(gx, gy, qpts);
    const rendered = rasterizeGrid(gx, gy, del.vtx, del.getTriangles(), palRGB);

    let bestKey = -1, bestScore = -1;
    for (let gy_i = 0; gy_i < gy; ++gy_i) {
      for (let gx_i = 0; gx_i < gx; ++gx_i) {
        if ((gx_i === 0 || gx_i === gx-1) && (gy_i === 0 || gy_i === gy-1)) continue;
        const key = gy_i*gx + gx_i;
        if (vtxSet.has(key)) continue;
        const i = key*4;
        const dr = rendered[i]-grid[i], dg = rendered[i+1]-grid[i+1], db = rendered[i+2]-grid[i+2];
        const score = 0.3*dr*dr + 0.6*dg*dg + 0.1*db*db;
        if (score > bestScore) { bestScore = score; bestKey = key; }
      }
    }
    if (bestKey < 0) break;
    const bx = bestKey % gx, by = Math.floor(bestKey / gx);
    qpts.push(mkVtx(bx, by));
    vtxSet.add(bestKey);
  }

  if (n_random > 0) {
    const candidates = [];
    for (let y = 0; y < gy; ++y) for (let x = 0; x < gx; ++x) {
      const key = y*gx + x;
      if (!vtxSet.has(key)) candidates.push({x, y, key});
    }
    for (let i = candidates.length - 1; i > 0; --i) {
      const j = Math.floor(Math.random() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    for (let k = 0; k < Math.min(n_random, candidates.length); ++k) {
      const {x, y, key} = candidates[k];
      qpts.push(mkVtx(x, y));
      vtxSet.add(key);
    }
  }

  return qpts;
}

// ---------------------------------------------------------------------------
// Sort palette by (cg, co, y, a) as required by the encoder.
// Returns { sortedPalette, remap } where remap[oldIdx] = newIdx.
function sortPalette(palette) {
  const order = palette.map((_, i) => i).sort((a, b) => {
    const pa = palette[a], pb = palette[b];
    return pa.cg !== pb.cg ? pa.cg-pb.cg : pa.co !== pb.co ? pa.co-pb.co :
           pa.y  !== pb.y  ? pa.y -pb.y  : pa.a - pb.a;
  });
  const sortedPalette = order.map(i => palette[i]);
  const remap = new Int32Array(palette.length);
  order.forEach((oldIdx, newIdx) => { remap[oldIdx] = newIdx; });
  return { sortedPalette, remap };
}

// ---------------------------------------------------------------------------
// Top-level: build a Preview + color_data from an ImageData-like object.
// options: { grid_x, grid_y, nb_colors, nb_pts, has_alpha, use_noise }
function buildInitialState(imageData, options = {}) {
  const {
    grid_x = 32, grid_y = 32,
    nb_colors = 8, nb_pts = 64,
    has_alpha = null, use_noise = false, random_init = false,
  } = options;

  const grid = sampleGrid(imageData, grid_x, grid_y);
  const actualHasAlpha = has_alpha !== null ? has_alpha :
    grid.some((v, i) => (i & 3) === 3 && v < 255);
  const palette = buildPalette(grid, nb_colors);
  const qpts = placeVertices(grid, grid_x, grid_y, palette, nb_pts, undefined, random_init);

  const { sortedPalette, remap } = sortPalette(palette);

  const corners = qpts.slice(0, 4).map(v => ({...v, idx: remap[v.idx]}));
  const nonCorners = qpts.slice(4)
    .map(v => ({...v, idx: remap[v.idx]}))
    .sort((a, b) => a.y !== b.y ? a.y-b.y : a.x-b.x);

  const preview = {
    grid_x, grid_y, use_noise, nb_colors, has_alpha: actualHasAlpha,
    nb_pts: nonCorners.length,
    qpts: [...corners, ...nonCorners],
  };
  return { preview, color_data: sortedPalette };
}

// ---------------------------------------------------------------------------
// Rebuild palette from imageData and re-assign vertex color indices in preview.
// Keeps vertex positions; only colors change. Returns { preview, color_data }.
function rebuildColormap(preview, imageData, gx, gy, nb_colors) {
  const grid = sampleGrid(imageData, gx, gy);
  const palette = buildPalette(grid, nb_colors);
  const { sortedPalette, remap } = sortPalette(palette);
  const newQpts = preview.qpts.map(v => {
    const x = Math.max(0, Math.min(gx - 1, Math.round(v.x)));
    const y = Math.max(0, Math.min(gy - 1, Math.round(v.y)));
    const ycocg = RGBtoYCoCg(grid[(y*gx+x)*4], grid[(y*gx+x)*4+1],
                              grid[(y*gx+x)*4+2], grid[(y*gx+x)*4+3]);
    return { ...v, idx: remap[findClosestPalette(ycocg, palette)] };
  });
  return { preview: { ...preview, nb_colors, qpts: newQpts }, color_data: sortedPalette };
}

// ---------------------------------------------------------------------------
if (typeof module !== 'undefined') {
  // For Node.js testing: require triangle-core.js first to get Delaunay, RGBtoYCoCg.
  module.exports = { sampleGrid, buildPalette, rasterizeGrid, placeVertices, sortPalette, buildInitialState, rebuildColormap };
}
