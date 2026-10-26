// mesh_image.js — A mesh texture's image (DESIGN.md §9 E5) between RGBA bytes, a JPEG file (the .glb's)
// and its GPU mip levels; shared by the app and the gallery. encodeJpeg / decodeImage need a browser
// (OffscreenCanvas, createImageBitmap).
"use strict";

// A mesh texture (RGBA bytes, size x size) as a JPEG file, and an image file (a Blob, or a .glb's image:
// { bytes, mimeType }) as RGBA ({ data, width, height })
export async function encodeJpeg(rgba, size, quality = 0.9) {
  const c = new OffscreenCanvas(size, size);
  c.getContext('2d').putImageData(new ImageData(rgba, size, size), 0, 0);
  return new Uint8Array(await (await c.convertToBlob({ type: 'image/jpeg', quality })).arrayBuffer());
}
export async function decodeImage(src) {
  const bmp = await createImageBitmap(src instanceof Blob ? src : new Blob([src.bytes], { type: src.mimeType }));
  const c = new OffscreenCanvas(bmp.width, bmp.height), cx = c.getContext('2d');
  cx.drawImage(bmp, 0, 0);
  bmp.close();
  return { data: cx.getImageData(0, 0, c.width, c.height).data, width: c.width, height: c.height };
}

// The mip levels of an RGBA image ({ data, width, height }), each texel the alpha-weighted mean of the 2 x 2
// below it (a mesh texture's texels off its faces, alpha 0, do not blur into its charts' edges far
// away), the plain mean where all four have alpha 0
export function mipLevels(img) {
  const levels = [{ data: img.data instanceof Uint8Array ? img.data : new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength), width: img.width, height: img.height }];
  for (let l = levels[0]; l.width > 1 || l.height > 1; ) {
    const W = Math.max(1, l.width >> 1), H = Math.max(1, l.height >> 1), d = new Uint8Array(4 * W * H), s = l.data;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0, a = 0, amax = 0, r0 = 0, g0 = 0, b0 = 0;
      for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
        const q = 4 * (Math.min(2 * y + j, l.height - 1) * l.width + Math.min(2 * x + i, l.width - 1)), w = s[q + 3];
        r += w * s[q]; g += w * s[q + 1]; b += w * s[q + 2]; a += w; amax = Math.max(amax, w);
        r0 += s[q]; g0 += s[q + 1]; b0 += s[q + 2];
      }
      const o = 4 * (y * W + x);
      if (a > 0) { d[o] = Math.round(r / a); d[o + 1] = Math.round(g / a); d[o + 2] = Math.round(b / a); d[o + 3] = amax; }
      else { d[o] = Math.round(r0 / 4); d[o + 1] = Math.round(g0 / 4); d[o + 2] = Math.round(b0 / 4); }
    }
    levels.push(l = { data: d, width: W, height: H });
  }
  return levels;
}
