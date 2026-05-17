'use strict';

// CPU Gouraud rasterizer — bounding-box scanline, O(triangle_area).
if (typeof module !== 'undefined' && typeof kBaryEps === 'undefined') {
  Object.assign(global, require('./triangle-core.js'));
}

// vtx: array of {x,y,idx} in pixel coords; triangles: from Delaunay.getTriangles().
// palRGB: [{r,g,b,a}]. Returns Float32Array [w*h*4] RGBA.
function rasterizeGrid(w, h, vtx, triangles, palRGB) {
  const out = new Float32Array(w * h * 4);
  for (const tri of triangles) {
    const v0 = vtx[tri.vtx[0]], v1 = vtx[tri.vtx[1]], v2 = vtx[tri.vtx[2]];
    const x0 = v0.x, y0 = v0.y, x1 = v1.x, y1 = v1.y, x2 = v2.x, y2 = v2.y;
    const d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
    if (Math.abs(d) < kDegenerateEps) continue;
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
        if (u < -kBaryEps || v < -kBaryEps || w_ < -kBaryEps) continue;
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
// maskWeights: optional Float32Array [zx*zy], per-pixel importance weights [0..1].
// roiStrength: multiplier — pixel loss *= (1 + maskWeights[i] * roiStrength).
function computeCPULoss(gx, gy, del, palRGB, zGrid, zoom, maskWeights, roiStrength) {
  zoom = zoom || 1;
  const zx = gx * zoom, zy = gy * zoom, n = zx * zy;
  const scaledVtx = del.vtx.map(v => ({ x: v.x * zoom, y: v.y * zoom, idx: v.idx }));

  let maskScale = 0;
  if (maskWeights) {
    let sum = 0;
    for (let i = 0; i < n; ++i) sum += maskWeights[i];
    maskScale = sum > 0 ? n / sum : 0;
  }

  const inv255 = 1 / 255;
  let sad = 0;

  for (const tri of del.getTriangles()) {
    const v0 = scaledVtx[tri.vtx[0]], v1 = scaledVtx[tri.vtx[1]], v2 = scaledVtx[tri.vtx[2]];
    const x0 = v0.x, y0 = v0.y, x1 = v1.x, y1 = v1.y, x2 = v2.x, y2 = v2.y;
    const d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
    if (Math.abs(d) < kDegenerateEps) continue;
    const inv_d = 1 / d;
    const c0 = palRGB[v0.idx], c1 = palRGB[v1.idx], c2 = palRGB[v2.idx];
    const r0=c0.r, g0=c0.g, b0=c0.b, a0=c0.a;
    const r1=c1.r, g1=c1.g, b1=c1.b, a1=c1.a;
    const r2=c2.r, g2=c2.g, b2=c2.b, a2=c2.a;
    const minX = Math.max(0,    Math.floor(Math.min(x0, x1, x2)));
    const maxX = Math.min(zx-1, Math.ceil (Math.max(x0, x1, x2)));
    const minY = Math.max(0,    Math.floor(Math.min(y0, y1, y2)));
    const maxY = Math.min(zy-1, Math.ceil (Math.max(y0, y1, y2)));
    // incremental barycentric: du/dx = A00, dv/dx = A10
    const A00 = (y1-y2)*inv_d, A01 = (x2-x1)*inv_d;
    const A10 = (y2-y0)*inv_d, A11 = (x0-x2)*inv_d;
    for (let y = minY; y <= maxY; ++y) {
      const py = y - y2;
      let u = A00*(minX-x2) + A01*py;
      let v = A10*(minX-x2) + A11*py;
      let p = (y*zx + minX)*4;
      for (let x = minX; x <= maxX; ++x, u += A00, v += A10, p += 4) {
        const w_ = 1 - u - v;
        if (u < -kBaryEps || v < -kBaryEps || w_ < -kBaryEps) continue;
        const ref_a = zGrid[p+3];
        const dr = u*r0 + v*r1 + w_*r2 - zGrid[p];
        const dg = u*g0 + v*g1 + w_*g2 - zGrid[p+1];
        const db = u*b0 + v*b1 + w_*b2 - zGrid[p+2];
        const da = u*a0 + v*a1 + w_*a2 - zGrid[p+3];
        const pix = ref_a * (kLumaR*Math.abs(dr) + kLumaG*Math.abs(dg) + kLumaB*Math.abs(db)) * inv255 + Math.abs(da);
        sad += maskScale > 0 ? pix * (1 + maskWeights[y*zx+x] * maskScale * roiStrength) : pix;
      }
    }
  }
  return sad * inv255 / n;
}

if (typeof module !== 'undefined') {
  module.exports = { rasterizeGrid, computeCPULoss };
}
