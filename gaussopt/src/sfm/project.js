// project.js — The .gosfm container (DESIGN-sfm §4 persistence): a JSON header and binary parts
// (typed arrays, JPEG blobs), written as one Blob without copying the parts, read back lazily by
// slicing. The worker keeps two of them per video in OPFS (features: analysis, keyframe JPEGs,
// keypoints and descriptors, written once; model: session state, after every change), and
// "Save project" writes both as one file.
//   'GOSFM1\n\0' | u32 header length | header JSON | parts (8-byte aligned)
"use strict";

const MAGIC = 'GOSFM1\n\0';
const TYPES = { Float64Array, Float32Array, Int32Array, Uint32Array, Uint8Array };

// json: any JSON; parts: { key: TypedArray | Blob | { raw: Blob, type, blob } (a part of another
// container, copied as is) }. Returns a Blob.
export function packProject(json, parts) {
  const table = [], chunks = [];
  let offset = 0;
  for (const [key, value] of Object.entries(parts)) {
    const part = value.raw instanceof Blob ? value : null, isBlob = part ? part.blob : value instanceof Blob;
    const length = part ? part.raw.size : isBlob ? value.size : value.byteLength;
    const type = part ? part.type : isBlob ? (value.type || 'blob') : value.constructor.name;
    table.push({ key, offset, length, type, blob: isBlob });
    chunks.push(part ? part.raw : isBlob ? value : new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
    const pad = (8 - (length % 8)) % 8;
    if (pad) chunks.push(new Uint8Array(pad));
    offset += length + pad;
  }
  const header = new TextEncoder().encode(JSON.stringify({ json, parts: table }));
  const pad = (8 - ((MAGIC.length + 4 + header.length) % 8)) % 8;
  const head = new Uint8Array(MAGIC.length + 4 + header.length + pad);
  head.set(new TextEncoder().encode(MAGIC));
  new DataView(head.buffer).setUint32(MAGIC.length, header.length, true);
  head.set(header, MAGIC.length + 4);
  return new Blob([head, ...chunks]);
}

// Blob / File -> { json, keys, get(key) -> Promise<TypedArray | Blob> }
export async function unpackProject(blob) {
  const start = new Uint8Array(await blob.slice(0, MAGIC.length + 4).arrayBuffer());
  if (new TextDecoder().decode(start.subarray(0, MAGIC.length)) !== MAGIC) throw new Error('not a .gosfm project');
  const n = new DataView(start.buffer).getUint32(MAGIC.length, true);
  const { json, parts } = JSON.parse(new TextDecoder().decode(await blob.slice(MAGIC.length + 4, MAGIC.length + 4 + n).arrayBuffer()));
  const base = MAGIC.length + 4 + n + (8 - ((MAGIC.length + 4 + n) % 8)) % 8;
  const byKey = new Map(parts.map(p => [p.key, p]));
  return {
    json,
    keys: parts.map(p => p.key),
    async get(key) {
      const { raw, type, blob: isBlob } = this.raw(key);
      return isBlob ? raw : new TYPES[type](await raw.arrayBuffer());
    },
    raw(key) {  // the part's bytes, unread: { raw: Blob, type, blob }
      const p = byKey.get(key);
      if (!p) throw new Error(`project has no part ${key}`);
      return { raw: blob.slice(base + p.offset, base + p.offset + p.length, p.blob ? p.type : undefined), type: p.type, blob: p.blob };
    },
  };
}

// Several containers as one (Save project): the parts of each under its prefix, not read
export async function mergeProjects(named) {
  const json = {}, parts = {};
  for (const [prefix, blob] of Object.entries(named)) {
    const p = await unpackProject(blob);
    json[prefix] = p.json;
    for (const key of p.keys) parts[`${prefix}/${key}`] = p.raw(key);
  }
  return packProject(json, parts);
}

// The parts of `prefix` in a merged project, as their own container view
export async function projectPart(blob, prefix) {
  const p = await unpackProject(blob);
  if (!(prefix in p.json)) return null;
  return { json: p.json[prefix], keys: p.keys.filter(k => k.startsWith(`${prefix}/`)).map(k => k.slice(prefix.length + 1)),
           get: (key) => p.get(`${prefix}/${key}`), raw: (key) => p.raw(`${prefix}/${key}`) };
}
