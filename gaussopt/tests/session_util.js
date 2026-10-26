// Helpers of the session tests (session.js, DESIGN-sfm §6.3): synthetic scenes with exact ground
// truth, their verified pairs served by name, damage, accuracy against the truth.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SfmSession, key } from '../src/sfm/session.js';
import { compareModels, checkThresholds } from '../src/sfm/compare.js';
import { expSO3, mul3 } from '../src/sfm/linalg.js';
import { ORBIT } from './util.js';

export { key };

// Verified pairs of a full scene, served by name (what GPU matching does in the worker)
export function tablePairSource(full) {
  const names = full.images.map(im => im.name);
  const table = new Map(full.pairs.map(p => [key(names[p.i], names[p.j]), p]));
  return async (todo) => todo.flatMap(([a, b, type]) => {
    const p = table.get(key(a, b));
    return p ? [{ a: names[p.i], b: names[p.j], type, matches: p.matches }] : [];
  });
}

// The scene restricted to some frames (pairs re-indexed)
function subset(input, keep) {
  const at = new Map(keep.map((k, n) => [k, n]));
  return { ...input, images: keep.map(k => input.images[k]),
           pairs: input.pairs.filter(p => at.has(p.i) && at.has(p.j)).map(p => ({ ...p, i: at.get(p.i), j: at.get(p.j) })) };
}

// Accuracy against the truth, over the session's keyframes
export function accuracy(session, scene) {
  const names = new Set(session.images.map(im => im.name));
  const truth = { ...scene.truth.model, images: scene.truth.model.images.filter(im => names.has(im.name)) };
  const m = compareModels(session.model(), truth);
  return { m, fails: checkThresholds(m, 'truth'), tag: `${m.numEst}/${m.numRef} registered, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg` };
}

// The session tests' orbit: 30 frames (4.9 s to map, 50: 10.7 s); the stretches they damage fit in it
export const SESSION_ORBIT = { ...ORBIT, numFrames: 30 };

export async function session(options = {}, keep = null) {
  const scene = makeSyntheticScene({ ...SESSION_ORBIT, ...options });
  const input = keep ? subset(scene.input, keep) : scene.input;
  const s = new SfmSession(input, { pairSource: tablePairSource(scene.input) });
  await s.reconstruct();
  return { scene, s };
}

// Damage a stretch: rotate its poses by `deg` and move them sideways (the map's points stay)
export function damage(s, names, deg = 2) {
  const ix = new Map(s.images.map((im, k) => [im.name, k]));
  for (const n of names) {
    const k = ix.get(n), p = s.mapper.poses[k];
    s.mapper.poses[k] = { R: mul3(expSO3([0, deg * Math.PI / 180, 0]), p.R), t: [p.t[0] + 0.5, p.t[1], p.t[2]] };
  }
}

export const posesOf = (s) => JSON.stringify(s.mapper.poses);

