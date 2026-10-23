// common.js — Shared pieces of the Deno tools: command-line parsing, COLMAP model loading, gs/
// dataset export, and the report against a COLMAP reference.
"use strict";

import { parseCamerasBin, parseImagesBin, parsePoints3DBin } from '../src/colmap.js';
import { fromColmap } from '../src/sfm/compare.js';
import { writeColmapModel } from '../src/sfm/export.js';
import { assessAgainstReference } from '../src/sfm/reference.js';
import { makeRng } from '../src/math.js';

// Deno.args -> { flag(name), option(name, fallback), done(count) }; done() returns the positional
// arguments, and exits with `usage` unless exactly `count` remain once every flag / option is read.
export function cli(usage) {
  const args = [...Deno.args];
  const fail = () => { console.error(`usage: ${usage}`); Deno.exit(2); };
  return {
    flag: (name) => { const i = args.indexOf(name); if (i < 0) return false; args.splice(i, 1); return true; },
    option: (name, fallback = null) => {
      const i = args.indexOf(name);
      if (i < 0) return fallback;
      if (i + 1 >= args.length) { console.error(`${name} needs a value`); fail(); }
      return args.splice(i, 2)[1];
    },
    done(count) {
      if (args.length !== count || args.some(a => a.startsWith('--'))) fail();
      return args;
    },
  };
}

export const exists = (path) => { try { Deno.statSync(path); return true; } catch { return false; } };

// The directory holding a COLMAP model's .bin files: `dir` itself, or a gs/ dataset's sparse/; null without one
export function colmapModelDir(dir) {
  const base = exists(`${dir}/cameras.bin`) ? dir : `${dir}/sparse`;
  return exists(`${base}/images.bin`) ? base : null;
}

// COLMAP model (a directory with cameras.bin / images.bin / optional points3D.bin, or a gs/
// dataset with them in sparse/) -> compare.js model
export async function readColmapModel(dir) {
  const base = colmapModelDir(dir);
  if (!base) throw new Error(`No COLMAP model in ${dir}`);
  const read = async (f) => (await Deno.readFile(`${base}/${f}`)).slice().buffer;
  const images = parseImagesBin(await read('images.bin'), parseCamerasBin(await read('cameras.bin')));
  const points3D = exists(`${base}/points3D.bin`) ? parsePoints3DBin(await read('points3D.bin')) : null;
  return fromColmap({ images, points3D });
}

// `model` against the COLMAP model in `refDir` restricted to the images of `input` (reference.js,
// the reference's own defects excluded): { ref, defects, local, pieces, fails }
export async function assessAgainstColmap(model, refDir, input) {
  const ref = await readColmapModel(refDir), names = new Set(input.images.map(im => im.name));
  ref.images = ref.images.filter(im => names.has(im.name));
  return { ref, ...assessAgainstReference(model, ref, { input, rng: makeRng(1) }) };
}

// src/sfm/export.js dataset -> DIR/sparse/*.bin and DIR/undistort.json (input of tools/undistort.py)
export async function writeDataset(dir, ds) {
  await Deno.mkdir(`${dir}/sparse`, { recursive: true });
  for (const [name, buf] of Object.entries(writeColmapModel(ds))) await Deno.writeFile(`${dir}/sparse/${name}`, new Uint8Array(buf));
  await Deno.writeTextFile(`${dir}/undistort.json`, JSON.stringify({ distorted: ds.distorted, pinhole: ds.camera, images: ds.images.map(im => im.name) }, null, 1));
  console.log(`exported segment ${ds.segment}: ${ds.images.length} images, ${ds.points.length} points -> ${dir}/sparse ` +
              `(camera ${ds.distorted.model} [${ds.distorted.params.map(p => +p.toPrecision(6)).join(', ')}] -> PINHOLE)`);
}

export const formatTimings = (timings) => Object.entries(timings).map(([k, v]) => `${k} ${(v / 1000).toFixed(1)} s`).join(', ');

// reference.js assessAgainstReference() result, printed
export function printAssessment({ defects, local, pieces, fails }, { verbose = false } = {}) {
  console.log(`vs COLMAP: ${defects.badFrames.size} weak frames and ${defects.breaks.size} breaks of the reference excluded`);
  if (verbose) {
    for (const [n, why] of defects.badFrames) console.log(`  frame ${n}: ${why}`);
    for (const [n, why] of defects.breaks) console.log(`  break before ${n}: ${why}`);
  }
  console.log(`local: ${local.numPairs} consecutive pairs, relative rotation median ${local.medianRelRotDeg.toFixed(3)} / p90 ${local.p90RelRotDeg.toFixed(3)} deg, ` +
              `baseline direction median ${local.medianDirDeg.toFixed(2)} / p90 ${local.p90DirDeg.toFixed(2)} deg`);
  for (const p of pieces) {
    console.log(p.skipped ? `piece ${p.first}..${p.last} (${p.size}, segment ${p.segment}): too short, skipped`
      : `piece ${p.first}..${p.last} (${p.size}, segment ${p.segment}): RMS center ${(100 * p.metrics.rmsCenterRel).toFixed(2)}%, ` +
        `median rotation ${p.metrics.medianRotDeg.toFixed(2)} deg`);
  }
  console.log(fails.length ? `FAIL:\n  ${fails.join('\n  ')}` : 'PASS');
}
