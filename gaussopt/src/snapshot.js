// snapshot.js — Full training-state snapshot (.gosnap) codec.
//
// Layout (little-endian):
//   [0..7]   magic 'GOSNAP01'
//   [8..11]  u32 JSON header byte length (incl. padding to 4 bytes)
//   [12..]   UTF-8 JSON header: { format, version, state: { count, stepCount, ... }, ...,
//            transform (the scene's leveling, 4x4 row-major; math.js levelTransform; absent before it),
//            blobs: { name: [byteOffset, byteLength] } }  (see GaussOptApp._encodeSnapshot)
//   [...]    raw blobs (offsets relative to the start of the blob section, 4-byte aligned):
//              params  : Float32 count * SPLAT_FLOATS   (raw GPU splat layout, incl. leaf_id/flags)
//              moments : Float32 count * MOMENT_FLOATS  (WLBG accumulators since last rebalance)
//              adam_m  : Float32 count * SPLAT_FLOATS
//              adam_v  : Float32 count * SPLAT_FLOATS
//              leaf_idx: Uint32 (optional octree leaf index permutation)
"use strict";

const MAGIC = 'GOSNAP01';
export const SNAPSHOT_FORMAT  = 'gaussopt_snapshot';
export const SNAPSHOT_VERSION = 1;
export const SNAPSHOT_EXT     = '.gosnap';

export function isSnapshotName(name) {
  return typeof name === 'string' && name.toLowerCase().endsWith(SNAPSHOT_EXT);
}

// header: plain JSON-able object; blobs: { name: TypedArray }
export function encodeSnapshot(header, blobs) {
  const table = {};
  let off = 0;
  for (const [name, arr] of Object.entries(blobs)) {
    if (!arr) continue;
    table[name] = [off, arr.byteLength];
    off += (arr.byteLength + 3) & ~3;
  }
  const json = new TextEncoder().encode(JSON.stringify({
    ...header, format: SNAPSHOT_FORMAT, version: SNAPSHOT_VERSION, blobs: table,
  }));
  const jsonLen = (json.byteLength + 3) & ~3;
  const base = 12 + jsonLen;
  const out = new Uint8Array(base + off);
  for (let i = 0; i < 8; i++) out[i] = MAGIC.charCodeAt(i);
  new DataView(out.buffer).setUint32(8, jsonLen, true);
  out.set(json, 12);
  out.fill(0x20, 12 + json.byteLength, base);  // pad JSON with spaces
  for (const [name, [o]] of Object.entries(table)) {
    const arr = blobs[name];
    out.set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength), base + o);
  }
  return out.buffer;
}

export function decodeSnapshot(buf) {
  const u8 = new Uint8Array(buf);
  if (u8.byteLength < 12 || String.fromCharCode(...u8.subarray(0, 8)) !== MAGIC) {
    throw new Error('Not a Gauss-Opt snapshot (bad magic)');
  }
  const jsonLen = new DataView(buf).getUint32(8, true);
  const base = 12 + jsonLen;
  if (base > u8.byteLength) throw new Error('Truncated snapshot header');
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(12, base)));
  if (header.format !== SNAPSHOT_FORMAT || header.version > SNAPSHOT_VERSION) {
    throw new Error(`Unsupported snapshot format ${header.format} v${header.version}`);
  }
  const blobs = {};
  for (const [name, [o, len]] of Object.entries(header.blobs || {})) {
    if (base + o + len > u8.byteLength) throw new Error(`Truncated snapshot blob "${name}"`);
    // copy so blobs are 4-byte aligned regardless of the source buffer
    const bytes = u8.slice(base + o, base + o + len);
    blobs[name] = (name === 'leaf_idx') ? new Uint32Array(bytes.buffer) : new Float32Array(bytes.buffer);
  }
  return { header, blobs };
}
