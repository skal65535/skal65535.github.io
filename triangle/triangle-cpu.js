'use strict';

// CPU Gouraud rasterizer — bounding-box scanline, O(triangle_area).
if (typeof module !== 'undefined' && typeof kBaryEps === 'undefined') {
  Object.assign(global, require('./triangle-core.js'));
}

// vtx: array of {x,y,idx} in grid coords; zoom scales vtx coords to pixel coords.
// palRGB: [{r,g,b,a}]. Returns Float32Array [w*h*4] RGBA.
function rasterizeGrid(w, h, vtx, triangles, palRGB, zoom = 1) {
  const out = new Float32Array(w * h * 4);
  for (const tri of triangles) {
    const v0 = vtx[tri.vtx[0]], v1 = vtx[tri.vtx[1]], v2 = vtx[tri.vtx[2]];
    const x0 = v0.x * zoom, y0 = v0.y * zoom, x1 = v1.x * zoom, y1 = v1.y * zoom, x2 = v2.x * zoom, y2 = v2.y * zoom;
    const d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
    if (Math.abs(d) < kDegenerateEps) continue;
    const inv_d = 1 / d;
    const c0 = palRGB[v0.idx], c1 = palRGB[v1.idx], c2 = palRGB[v2.idx];
    const r2=c2.r, g2=c2.g, b2=c2.b, a2=c2.a;
    const r0=c0.r - r2, g0=c0.g - g2, b0=c0.b - b2, a0=c0.a - a2;
    const r1=c1.r - r2, g1=c1.g - g2, b1=c1.b - b2, a1=c1.a - a2;

    const minX = Math.max(0,   Math.floor(Math.min(x0, x1, x2)));
    const maxX = Math.min(w-1, Math.ceil (Math.max(x0, x1, x2)));
    const minY = Math.max(0,   Math.floor(Math.min(y0, y1, y2)));
    const maxY = Math.min(h-1, Math.ceil (Math.max(y0, y1, y2)));
    const A00 = (y1 - y2) * inv_d, A01 = (x2 - x1) * inv_d;
    const A10 = (y2 - y0) * inv_d, A11 = (x0 - x2) * inv_d;
    let ru = A00 * (minX - x2) + A01 * (minY - y2);
    let rv = A10 * (minX - x2) + A11 * (minY - y2);
    for (let y = minY; y <= maxY; ++y, ru += A01, rv += A11) {
      let p = y*w + minX;
      let u = ru, v = rv;
      for (let x = minX; x <= maxX; ++x, u += A00, v += A10, p += 1) {
        const w_ = 1 - u - v;
        if (u < -kBaryEps || v < -kBaryEps || w_ < -kBaryEps) continue;
        out[4*p+0] = u*r0 + v*r1 + r2;
        out[4*p+1] = u*g0 + v*g1 + g2;
        out[4*p+2] = u*b0 + v*b1 + b2;
        out[4*p+3] = u*a0 + v*a1 + a2;
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

  let maskScale = 0;
  if (maskWeights) {
    let sum = 0;
    for (let i = 0; i < n; ++i) sum += maskWeights[i];
    maskScale = sum > 0 ? n / sum : 0;
  }

  const inv255 = 1 / 255;
  let sad = 0;

  for (const tri of del.getTriangles()) {
    const v0 = del.vtx[tri.vtx[0]], v1 = del.vtx[tri.vtx[1]], v2 = del.vtx[tri.vtx[2]];
    const x0 = v0.x * zoom, y0 = v0.y * zoom, x1 = v1.x * zoom, y1 = v1.y * zoom, x2 = v2.x * zoom, y2 = v2.y * zoom;
    const d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
    if (Math.abs(d) < kDegenerateEps) continue;
    const inv_d = 1 / d;
    const c0 = palRGB[v0.idx], c1 = palRGB[v1.idx], c2 = palRGB[v2.idx];
    const r2=c2.r, g2=c2.g, b2=c2.b, a2=c2.a;
    const dr0=c0.r - r2, dg0=c0.g - g2, db0=c0.b - b2, da0=c0.a - a2;
    const dr1=c1.r - r2, dg1=c1.g - g2, db1=c1.b - b2, da1=c1.a - a2;

    const minX = Math.max(0,    Math.floor(Math.min(x0, x1, x2)));
    const maxX = Math.min(zx-1, Math.ceil (Math.max(x0, x1, x2)));
    const minY = Math.max(0,    Math.floor(Math.min(y0, y1, y2)));
    const maxY = Math.min(zy-1, Math.ceil (Math.max(y0, y1, y2)));
    const A00 = (y1-y2)*inv_d, A01 = (x2-x1)*inv_d;
    const A10 = (y2-y0)*inv_d, A11 = (x0-x2)*inv_d;
    let ru = A00*(minX-x2) + A01*(minY-y2);
    let rv = A10*(minX-x2) + A11*(minY-y2);
    for (let y = minY; y <= maxY; ++y, ru += A01, rv += A11) {
      let u = ru, v = rv;
      let p = y*zx + minX;
      for (let x = minX; x <= maxX; ++x, u += A00, v += A10, ++p) {
        const w_ = 1 - u - v;
        if (u < -kBaryEps || v < -kBaryEps || w_ < -kBaryEps) continue;
        const ref_a = zGrid[4*p+3];
        const dr = u*dr0 + v*dr1 + r2 - zGrid[4*p+0];
        const dg = u*dg0 + v*dg1 + g2 - zGrid[4*p+1];
        const db = u*db0 + v*db1 + b2 - zGrid[4*p+2];
        const da = u*da0 + v*da1 + a2 - zGrid[4*p+3];
        const pix = ref_a * (kLumaR*Math.abs(dr) + kLumaG*Math.abs(dg) + kLumaB*Math.abs(db)) * inv255 + Math.abs(da);
        sad += maskScale > 0 ? pix * (1 + maskWeights[p] * maskScale * roiStrength) : pix;
      }
    }
  }
  return sad * inv255 / n;
}

if (typeof module !== 'undefined') {
  module.exports = { rasterizeGrid, computeCPULoss };
}
