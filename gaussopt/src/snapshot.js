// snapshot.js — A full training-state snapshot: training goes on from it as it was. Written as a scene .glb with its
// training state (scene_glb.js training: the splats as trained, the optimizer's moments, the counters and settings), the
// scene's other parts with it (cameras, points, background, mesh): one file to view, share and resume. Read also from
// the older .gosnap (below), which gosnapToSceneGlb turns into one.
//
// The .gosnap layout (little-endian):
//   [0..7]   magic 'GOSNAP01'
//   [8..11]  u32 JSON header byte length (incl. padding to 4 bytes)
//   [12..]   UTF-8 JSON header: { format, version, state: { count, stepCount, ... }, ...,
//            transform (the scene's leveling, 4x4 row-major; math.js levelTransform; absent in older files),
//            blobs: { name: [byteOffset, byteLength] } }  (see GaussOptApp._encodeSnapshot)
//   [...]    raw blobs (offsets relative to the start of the blob section, 4-byte aligned):
//              params  : Float32 count * SPLAT_FLOATS   (raw GPU splat layout, incl. leaf_id/flags)
//              moments : Float32 count * MOMENT_FLOATS  (WLBG accumulators since last rebalance)
//              adam_m  : Float32 count * SPLAT_FLOATS
//              adam_v  : Float32 count * SPLAT_FLOATS
//              leaf_idx: Uint32 (optional octree leaf index permutation)
"use strict";

import { writeSceneGlb } from './scene_glb.js';

const MAGIC = 'GOSNAP01';
export const SNAPSHOT_FORMAT  = 'gaussopt_snapshot';
export const SNAPSHOT_VERSION = 1;
export const SNAPSHOT_EXT     = '.scene.glb';  // a snapshot's name: gaussopt_snap_NNNNN.scene.glb
export const GOSNAP_EXT       = '.gosnap';

// An older snapshot by its name (.gosnap); a .glb's kind is in its content (snapshotKind)
export function isSnapshotName(name) {
  return typeof name === 'string' && name.toLowerCase().endsWith(GOSNAP_EXT);
}

// 'gosnap', 'glb' (a scene, with or without its training state) or null, by the file's first bytes
export function snapshotKind(buf) {
  const u8 = new Uint8Array(buf, 0, Math.min(8, buf.byteLength));
  if (u8.length === 8 && String.fromCharCode(...u8) === MAGIC) return 'gosnap';
  if (u8.length >= 4 && new DataView(buf).getUint32(0, true) === 0x46546c67) return 'glb';  // 'glTF'
  return null;
}

// A scene .glb's snapshot (readSceneGlb's result): decodeSnapshot's { header, blobs }, the octree and the leveling
// from the scene; null without a training state (a scene saved to view: its splats load, the optimizer starts anew)
export function snapshotOfScene(r) {
  if (!r.training) return null;
  const { header, blobs } = r.training;
  return { header: { ...header, octree: r.octree ? { ...r.octree, leafIndices: null } : null, transform: r.info.transform ?? null }, blobs };
}

// A .gosnap as a scene .glb with its training state: the same splats, octree, optimizer and counters; scene: the parts
// a .gosnap has not (cameras, path, points, upEstimate, env, mesh, extras), when known (none: a scene of splats alone)
export function gosnapToSceneGlb(buf, scene = {}) {
  const { header, blobs } = decodeSnapshot(buf), { octree, transform, blobs: _, format: __, version: ___, ...rest } = header;
  const count = header.state.count | 0;
  return writeSceneGlb({ ...scene, count, params: blobs.params, octree: octree ? { toJSON: () => octree } : null, transform: transform ?? null,
                         extras: { step: header.state.stepCount | 0, savedAt: header.savedAt ?? null, from: 'gosnap', ...(scene.extras ?? {}) },
                         training: { header: rest, blobs } });
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
