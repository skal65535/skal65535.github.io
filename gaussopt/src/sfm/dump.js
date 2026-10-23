// dump.js — Loader for COLMAP database dumps written by tools/colmap_dump.py (DESIGN-sfm §8.2).
// Returns the same `input` shape as synth.js:
//   { camera, images: [{ name, keypoints: Float32Array(2n), descriptors?: Uint8Array(128n) }],
//     pairs: [{ i, j, type: 'seq' | 'loop', config, F, matches: Uint32Array(2m), raw?: Uint32Array }] }
// (raw: matches before geometric verification, when dumped with --raw-matches)
// `read(relPath)` returns a Promise<ArrayBuffer> (denoReader for a local directory).
"use strict";

import { makeCamera } from './camera.js';

export async function loadDump(read, { descriptors = false } = {}) {
  const meta = JSON.parse(new TextDecoder().decode(await read('dump.json')));
  if (meta.version !== 1) throw new Error(`Unsupported dump version ${meta.version}`);
  const kp = new Float32Array(await read('keypoints.f32'));
  const mt = new Uint32Array(await read('matches.u32'));
  const desc = descriptors && meta.descriptors ? new Uint8Array(await read('descriptors.u8')) : null;
  const raw = meta.pairs.some(p => p.rawMatches) ? new Uint32Array(await read('raw.u32')) : null;
  const { model, width, height, params } = meta.camera;
  return {
    meta,
    camera: makeCamera(model, width, height, params),
    images: meta.images.map(({ name, numKeypoints: n, offset }) => ({
      name,
      keypoints: kp.subarray(2 * offset, 2 * (offset + n)),
      ...(desc ? { descriptors: desc.subarray(128 * offset, 128 * (offset + n)) } : {}),
    })),
    pairs: meta.pairs.map(({ i, j, type, config, F, numMatches: n, offset, rawOffset, rawMatches }) =>
      ({ i, j, type, config, F, matches: mt.subarray(2 * offset, 2 * (offset + n)),
         ...(raw && rawMatches ? { raw: raw.subarray(2 * rawOffset, 2 * (rawOffset + rawMatches)) } : {}) })),
  };
}

// read() for a local directory under Deno
export const denoReader = (dir) => async (rel) => {
  const u8 = await Deno.readFile(`${dir}/${rel}`);
  return u8.byteOffset === 0 && u8.byteLength === u8.buffer.byteLength ? u8.buffer : u8.slice().buffer;
};

