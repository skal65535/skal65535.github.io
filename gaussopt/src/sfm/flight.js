// flight.js — A drone video's own flight record, for display only (a noisy GPS: never an input to the SfM).
// DJI's djmd track (dvtm_*.proto, e.g. dvtm_dji_neo.proto): a protobuf a video frame, its field numbers read off the
// wire (no .proto published): frame 3 > 4 (the drone) > 4 (position) > 1 (2: latitude, 3: longitude, radians,
// double), 2 (altitude, mm); 3 > 4 > 3 the drone's pitch, roll, yaw and 3 > 5 > 2 the gimbal's (0.1 degree,
// signed varints). alignTrack puts the track into the cameras' world by a similarity on the keyframes' times.
"use strict";

import { dataTrack } from './mp4.js';
import { robustUmeyama } from './align.js';
import { apply } from './sim3.js';

const EARTH = 6378137;  // m, WGS84's equatorial radius (the local tangent plane: a few hundred metres)

// The fields of a protobuf message: Map field number -> first value (varint: a Number or a BigInt beyond 2^53;
// 64-bit: a double; 32-bit: a float; length-delimited: the bytes)
export function pbFields(b) {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength), out = new Map();
  for (let i = 0; i < b.length;) {
    let key = 0, shift = 0, c;
    do { c = b[i++]; key += (c & 0x7f) * 2 ** shift; shift += 7; } while (c & 0x80);
    const fn = Math.floor(key / 8), wt = key % 8;
    let v;
    if (wt === 0) {
      let x = 0n; shift = 0n;
      do { c = b[i++]; x |= BigInt(c & 0x7f) << shift; shift += 7n; } while (c & 0x80);
      v = x <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(x) : x;
    } else if (wt === 1) { v = dv.getFloat64(i, true); i += 8; }
    else if (wt === 5) { v = dv.getFloat32(i, true); i += 4; }
    else if (wt === 2) {
      let n = 0; shift = 0;
      do { c = b[i++]; n += (c & 0x7f) * 2 ** shift; shift += 7; } while (c & 0x80);
      v = b.subarray(i, i + n); i += n;
    } else throw new Error(`protobuf wire type ${wt}`);
    if (i > b.length) throw new Error('protobuf message cut short');
    if (!out.has(fn)) out.set(fn, v);
  }
  return out;
}

// The message at a field path (null where absent or not a message)
const at = (b, ...path) => {
  for (const fn of path) { const v = b && pbFields(b).get(fn); b = v instanceof Uint8Array ? v : null; }
  return b;
};
const sint64 = (v) => v === undefined ? 0 : Number(BigInt.asIntN(64, BigInt(v)));  // a negative int64 is 10 varint bytes
const tenths = (m) => m ? [1, 2, 3].map(k => sint64(pbFields(m).get(k)) / 10) : [0, 0, 0];

// One djmd sample: { lat, lon (degrees), alt (m), attitude, gimbal ([pitch, roll, yaw] degrees) }, or null (no fix)
export function djmdFix(b) {
  const frame = at(b, 3), pos = at(frame, 4, 4), ll = at(pos, 1);
  const f = ll && pbFields(ll), lat = f?.get(2), lon = f?.get(3);
  if (typeof lat !== 'number' || typeof lon !== 'number' || (lat === 0 && lon === 0)) return null;
  return { lat: lat * 180 / Math.PI, lon: lon * 180 / Math.PI, alt: Number(pbFields(pos).get(2) ?? 0) / 1000,
           attitude: tenths(at(frame, 4, 3)), gimbal: tenths(at(frame, 5, 2)) };
}

// The flight record of an MP4 (reader: mp4.js's), or null: { source, t: [s], lat, lon, alt, attitude, gimbal,
// enu: [[east, north, up]] metres from the first fix }
export async function readTelemetry(reader) {
  const track = await dataTrack(reader, 'djmd');
  if (!track?.samples.length) return null;
  const fixes = [], BATCH = 512;
  for (let k = 0; k < track.samples.length; k += BATCH) {  // a few hundred bytes each, interleaved with the video
    const part = track.samples.slice(k, k + BATCH);
    const bytes = await Promise.all(part.map(s => reader.read(s.offset, s.size)));
    part.forEach((s, j) => {
      try { const f = djmdFix(bytes[j]); if (f) fixes.push({ t: s.pts / track.timescale, ...f }); } catch { /* a sample of another layout */ }
    });
  }
  if (!fixes.length) return null;
  fixes.sort((a, b) => a.t - b.t);
  const col = (k) => fixes.map(f => f[k]);
  return { source: 'djmd', t: col('t'), lat: col('lat'), lon: col('lon'), alt: col('alt'), attitude: col('attitude'), gimbal: col('gimbal'), enu: enuOf(fixes, fixes[0]) };
}

// Fixes { lat, lon (degrees), alt } in metres east, north and up from `origin` (one of them)
const enuOf = (fixes, origin) => {
  const lat0 = origin.lat * Math.PI / 180, lon0 = origin.lon * Math.PI / 180;
  return fixes.map(f => [EARTH * Math.cos(lat0) * (f.lon * Math.PI / 180 - lon0), EARTH * (f.lat * Math.PI / 180 - lat0), f.alt - origin.alt]);
};

// Several videos' records as one (parts: [{ tel (readTelemetry's, or null), offset (s) }], worker.js clips): each
// one's times `offset` on, as its keyframes', the positions from the first fix of the first, breaks: the first fix of
// each but the first (nothing between two: positionAt); null when none has any
export function joinTelemetry(parts) {
  const got = parts.filter(p => p.tel);
  if (!got.length) return null;
  const out = { source: got[0].tel.source, t: [], lat: [], lon: [], alt: [], attitude: [], gimbal: [], breaks: [] };
  for (const { tel, offset } of got) {
    if (out.t.length) out.breaks.push(out.t.length);
    out.t.push(...tel.t.map(t => t + offset));
    for (const k of ['lat', 'lon', 'alt', 'attitude', 'gimbal']) out[k].push(...tel[k]);
  }
  const fix = (k) => ({ lat: out.lat[k], lon: out.lon[k], alt: out.alt[k] });
  return { ...out, enu: enuOf(out.t.map((_, k) => fix(k)), fix(0)) };
}

// The track's position at time s (linear between fixes; null outside it)
// (several videos' joined, joinTelemetry: null between two)
export function positionAt(tel, s) {
  const { t, enu } = tel;
  if (!(s >= t[0] && s <= t.at(-1))) return null;
  let lo = 0, hi = t.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] <= s) lo = m; else hi = m; }
  if (tel.breaks?.includes(hi)) return null;
  const w = t[hi] > t[lo] ? (s - t[lo]) / (t[hi] - t[lo]) : 0;
  return enu[lo].map((x, k) => x + w * (enu[hi][k] - x));
}

// A keyframe's video time from its name (worker.js keyframeName: t<ms>.jpg), or null
export const keyframeTime = (name) => { const m = /(?:^|\/)t(\d+)\.jpg$/.exec(name); return m ? Number(m[1]) / 1000 : null; };

// The track in the world of `images` (their centres against the track at their keyframes' times, robustUmeyama):
// { points: [[x, y, z]] world, every `step` s of the track; cameras: the number used; residual: their median distance
// to the track (m); scale: world units a metre }, or null with fewer than minCameras or a track that does not move
export function alignTrack(tel, images, { step = 0.1, minCameras = 5 } = {}) {
  const src = [], dst = [];
  for (const im of images) {
    const s = keyframeTime(im.name), p = s === null ? null : positionAt(tel, s);
    if (p && im.center) { src.push(p); dst.push(im.center); }
  }
  if (src.length < minCameras) return null;
  const spread = Math.max(...[0, 1, 2].map(k => Math.max(...src.map(p => p[k])) - Math.min(...src.map(p => p[k]))));
  if (!(spread > 1)) return null;  // a hover: a metre of GPS noise sets no scale
  const G = robustUmeyama(src, dst);
  const res = src.map((p, k) => Math.hypot(...apply(G, p).map((x, d) => x - dst[k][d])) / G.s).sort((a, b) => a - b);
  const points = [];
  for (let s = tel.t[0]; s <= tel.t.at(-1); s += step) { const p = positionAt(tel, s); if (p) points.push(apply(G, p)); }
  return { points, cameras: src.length, residual: res[res.length >> 1], scale: G.s };
}
