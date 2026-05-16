'use strict';

// CPU Gouraud rasterizer — bounding-box scanline, O(triangle_area).

function computeBary(px, py, x0, y0, x1, y1, x2, y2) {
  const d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
  if (Math.abs(d) < 1e-8) return null;
  const u = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / d;
  const v = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / d;
  const w = 1 - u - v;
  return (u >= -1e-4 && v >= -1e-4 && w >= -1e-4) ? [u, v, w] : null;
}

// vtx: array of {x,y,idx} in pixel coords; triangles: from Delaunay.getTriangles().
// palRGB: [{r,g,b,a}]. Returns Float32Array [w*h*4] RGBA.
function rasterizeGrid(w, h, vtx, triangles, palRGB) {
  const out = new Float32Array(w * h * 4);
  for (const tri of triangles) {
    const v0 = vtx[tri.vtx[0]], v1 = vtx[tri.vtx[1]], v2 = vtx[tri.vtx[2]];
    const x0 = v0.x, y0 = v0.y, x1 = v1.x, y1 = v1.y, x2 = v2.x, y2 = v2.y;
    const d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
    if (Math.abs(d) < 1e-8) continue;
    const inv_d = 1 / d;
    const c0 = palRGB[v0.idx], c1 = palRGB[v1.idx], c2 = palRGB[v2.idx];
    const minX = Math.max(0,   Math.floor(Math.min(x0, x1, x2)));
    const maxX = Math.min(w-1, Math.ceil (Math.max(x0, x1, x2)));
    const minY = Math.max(0,   Math.floor(Math.min(y0, y1, y2)));
    const maxY = Math.min(h-1, Math.ceil (Math.max(y0, y1, y2)));
    for (let y = minY; y <= maxY; ++y) {
      for (let x = minX; x <= maxX; ++x) {
        const u = ((y1-y2)*(x-x2) + (x2-x1)*(y-y2)) * inv_d;
        const v = ((y2-y0)*(x-x2) + (x0-x2)*(y-y2)) * inv_d;
        const w_ = 1 - u - v;
        if (u < -1e-4 || v < -1e-4 || w_ < -1e-4) continue;
        const p = (y*w + x)*4;
        out[p]   = u*c0.r + v*c1.r + w_*c2.r;
        out[p+1] = u*c0.g + v*c1.g + w_*c2.g;
        out[p+2] = u*c0.b + v*c1.b + w_*c2.b;
        out[p+3] = u*c0.a + v*c1.a + w_*c2.a;
      }
    }
  }
  return out;
}

// Weighted SAD matching GPU formula; normalized per pixel.
// zoom: render at gx*zoom × gy*zoom against zGrid (Float32Array, same size, RGBA [0,255]).
function computeCPULoss(gx, gy, del, palRGB, zGrid, zoom) {
  zoom = zoom || 1;
  const zx = gx * zoom, zy = gy * zoom;
  // scale vtx grid coords → pixel coords in zoomed space
  const scaledVtx = del.vtx.map(v => ({ x: v.x * zoom, y: v.y * zoom, idx: v.idx }));
  const rendered = rasterizeGrid(zx, zy, scaledVtx, del.getTriangles(), palRGB);
  let sad = 0;
  const inv255 = 1 / 255;
  for (let i = 0, n = zx * zy; i < n; ++i) {
    const b = i * 4;
    const ref_a = zGrid[b+3] * inv255;
    const dr = (rendered[b]   - zGrid[b])   * inv255;
    const dg = (rendered[b+1] - zGrid[b+1]) * inv255;
    const db = (rendered[b+2] - zGrid[b+2]) * inv255;
    const da = (rendered[b+3] - zGrid[b+3]) * inv255;
    sad += ref_a * (0.3 * Math.abs(dr) + 0.6 * Math.abs(dg) + 0.1 * Math.abs(db))
         + Math.abs(da);
  }
  return sad / (zx * zy);
}

if (typeof module !== 'undefined') {
  module.exports = { computeBary, rasterizeGrid, computeCPULoss };
}
