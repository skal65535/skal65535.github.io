// exif.js — The few EXIF fields a photo set needs (DESIGN-sfm §13.3 P1, P2, P8), from a JPEG's
// APP1 segment: capture time, focal length (mm and 35 mm equivalent), digital zoom, camera and lens,
// orientation. No dependency; anything missing or malformed is left out of the result.
"use strict";

const TAGS = {
  0x010F: 'make', 0x0110: 'model', 0x0112: 'orientation', 0x8769: 'exifIFD',
  0x9003: 'dateTimeOriginal', 0x9011: 'offsetTimeOriginal', 0x9291: 'subSecTimeOriginal',
  0x920A: 'focalLength', 0xA404: 'digitalZoomRatio', 0xA405: 'focalLength35mm', 0xA434: 'lensModel',
};

// buffer: ArrayBuffer of (at least the start of) a JPEG file. Returns { make, model, orientation,
// dateTimeOriginal, subSecTimeOriginal, offsetTimeOriginal, focalLength, digitalZoomRatio,
// focalLength35mm, lensModel } (each only when present), or {} when there is no EXIF.
export function readExif(buffer) {
  const dv = new DataView(buffer);
  if (dv.byteLength < 4 || dv.getUint16(0) !== 0xFFD8) return {};
  for (let off = 2; off + 4 <= dv.byteLength;) {
    if (dv.getUint8(off) !== 0xFF) return {};
    const marker = dv.getUint8(off + 1), len = dv.getUint16(off + 2);
    if (marker === 0xDA || marker === 0xD9) return {};  // image data: no EXIF before it
    if (marker === 0xE1 && off + 10 <= dv.byteLength && dv.getUint32(off + 4) === 0x45786966) {  // 'Exif'
      try { return readTiff(dv, off + 10, Math.min(dv.byteLength, off + 2 + len)); } catch { return {}; }
    }
    off += 2 + len;
  }
  return {};
}

function readTiff(dv, base, end) {
  const little = dv.getUint16(base) === 0x4949;
  const u16 = (o) => dv.getUint16(base + o, little), u32 = (o) => dv.getUint32(base + o, little);
  if (u16(2) !== 42) return {};
  const out = {};
  const ifd = (at) => {
    if (at <= 0 || base + at + 2 > end) return;
    const n = u16(at);
    for (let k = 0; k < n; k++) {
      const e = at + 2 + 12 * k;
      if (base + e + 12 > end) return;
      const name = TAGS[u16(e)];
      if (!name) continue;
      const v = value(u16(e + 2), u32(e + 4), e + 8);
      if (v !== undefined) out[name] = v;
    }
  };
  const SIZES = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 };
  const value = (type, count, at) => {
    const size = (SIZES[type] ?? 0) * count;
    if (!size) return undefined;
    const o = size > 4 ? u32(at) : at;
    if (base + o + size > end) return undefined;
    if (type === 2) {  // ASCII, NUL-terminated
      let s = '';
      for (let i = 0; i < count; i++) { const c = dv.getUint8(base + o + i); if (!c) break; s += String.fromCharCode(c); }
      return s.trim();
    }
    if (type === 3) return u16(o);
    if (type === 4) return u32(o);
    if (type === 5) { const d = u32(o + 4); return d ? u32(o) / d : 0; }
    if (type === 10) { const d = dv.getInt32(base + o + 4, little); return d ? dv.getInt32(base + o, little) / d : 0; }
    return undefined;
  };
  ifd(u32(4));
  if (out.exifIFD) ifd(out.exifIFD);
  delete out.exifIFD;
  return out;
}

// Capture time in ms since the epoch, from DateTimeOriginal ('YYYY:MM:DD HH:MM:SS'), its sub-seconds
// and offset when present (without an offset, read as UTC: only differences matter), or null
export function captureTime(exif) {
  const m = /^(\d{4}):(\d\d):(\d\d) (\d\d):(\d\d):(\d\d)$/.exec(exif.dateTimeOriginal ?? '');
  if (!m) return null;
  const [, Y, M, D, h, mi, s] = m.map(Number);
  let t = Date.UTC(Y, M - 1, D, h, mi, s);
  const sub = /^\d+$/.test(exif.subSecTimeOriginal ?? '') ? Number(`0.${exif.subSecTimeOriginal}`) : 0;
  t += 1000 * sub;
  const off = /^([+-])(\d\d):(\d\d)$/.exec(exif.offsetTimeOriginal ?? '');
  if (off) t -= (off[1] === '-' ? -1 : 1) * (Number(off[2]) * 60 + Number(off[3])) * 60000;
  return t;
}

// Size of a JPEG as shown: [width, height] from its frame header (SOF0-SOF15 but DHT, JPG, DAC),
// swapped for EXIF orientations 5-8 (a 90° turn), or null when no frame header is found in `buffer`
export function jpegSize(buffer, orientation = 1) {
  const dv = new DataView(buffer);
  if (dv.byteLength < 4 || dv.getUint16(0) !== 0xFFD8) return null;
  for (let off = 2; off + 9 <= dv.byteLength;) {
    if (dv.getUint8(off) !== 0xFF) return null;
    const marker = dv.getUint8(off + 1);
    if (marker >= 0xC0 && marker <= 0xCF && marker !== 0xC4 && marker !== 0xC8 && marker !== 0xCC) {
      const h = dv.getUint16(off + 5), w = dv.getUint16(off + 7);
      return orientation >= 5 && orientation <= 8 ? [h, w] : [w, h];
    }
    if (marker === 0xDA) return null;
    off += 2 + dv.getUint16(off + 2);
  }
  return null;
}
