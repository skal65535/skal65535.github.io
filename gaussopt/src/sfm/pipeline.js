// pipeline.js — S1..S4 on keyframes (DESIGN-sfm §5.2-5.5), shared by tools/run_frames.js and the
// browser worker: GPU SIFT -> loop candidates by appearance -> GPU matching (sequential window +
// loops) -> verification -> reconstruction, with geometric loop candidates once the sequential
// map exists.
"use strict";

import { matchFeatures, matchPairs } from './frontend.js';
import { buildVocabulary, bagOfWords, loopCandidates, geometricCandidates } from './loopdetect.js';
import { SfmSession, withNames } from './session.js';
import { defaultCamera } from './camera.js';
import { makeRng } from '../math.js';
import { PIPELINE_DEFAULTS } from './defaults.js';

const seconds = (t) => `${((performance.now() - t) / 1000).toFixed(1)} s`;

// features: [{ name, keypoints, descriptors }] in time order, all `width` x `height`.
// mapping: mapper / reconstruct options (focal, hfovDeg, sweeps, ...); log(line), progress(stage, done, total).
// Returns { session (session.js, for repairs), mapper, model, stats, input, matched }.
export async function reconstructFromFeatures(features, width, height, matcher, options = {}) {
  const o = { ...PIPELINE_DEFAULTS, log: () => {}, progress: () => {}, ...options };
  const rng = makeRng(o.seed ?? 1), log = o.log;
  const t1 = performance.now();
  let candidates = [];
  if (o.loops) {
    candidates = loopCandidates(await bagOfWords(features, await buildVocabulary(features, matcher, { rng }), matcher), { minGap: 2 * o.window });
    log(`loop detection: ${candidates.length} candidate pairs in ${seconds(t1)}`);
  }
  const tm = performance.now();
  const matched = await matchFeatures(features, matcher, { rng, window: o.window, loopCandidates: candidates,
                                                          onProgress: (d, n) => o.progress('match', d, n) });
  const ms = matched.stats;
  log(`matching: ${ms.matched} pairs, ${ms.verified} verified (loops ${ms.loopsVerified}/${ms.loopsTried}), ` +
      `${Math.round(ms.inliers / Math.max(1, ms.verified))} inliers per pair in ${seconds(tm)}`);

  const input = { camera: o.camera ?? defaultCamera('OPENCV', width, height), images: matched.images, pairs: matched.pairs };
  const have = new Set(matched.pairs.map(p => `${p.i},${p.j}`));
  const proposeLoops = async (mapper) => {
    const geo = geometricCandidates(mapper, { minGap: 2 * o.window, perFrame: 16, rng }).filter(([i, j]) => !have.has(`${i},${j}`));
    const { pairs, stats } = await matchPairs(features, matcher, geo.map(([i, j]) => [i, j, 'loop']),
                                              { rng, onProgress: (d, n) => o.progress('loopMatch', d, n) });
    log(`  geometric loops: ${geo.length} candidates, ${stats.verified} verified`);
    return pairs;
  };
  const t2 = performance.now();
  const session = new SfmSession(input, { pairSource: featurePairSource(features, matcher), mapping: { window: o.window, ...o.mapping },
                                          loops: o.loops, window: o.window, seed: o.seed ?? 1, log: (s) => log(`  ${s}`), progress: o.progress });
  const { mapper, model, stats } = await session.reconstruct({ proposeLoops });
  log(`reconstruct: ${model.images.length}/${features.length} registered, ${stats.numSegments} segment(s), ` +
      `focal ${model.focal.toFixed(1)} px in ${seconds(t2)}`);
  return { session, mapper, model, stats, input, matched };
}

// session.js pairSource over keyframe features: GPU matching + verification by keyframe name.
// `features` may grow (inserted keyframes): names are looked up at each call. rng: the action's
// (session.js), so a repair does not depend on what was matched before it.
export function featurePairSource(features, matcher) {
  return async (todo, { match = {}, rng = makeRng(1) } = {}) => {
    const index = new Map(features.map((f, k) => [f.name, k]));
    const jobs = todo.map(([a, b, type]) => [index.get(a), index.get(b), type]).filter(([i, j]) => i !== undefined && j !== undefined);
    const { pairs } = await matchPairs(features, matcher, jobs, { rng, match });
    return withNames(pairs, features);
  };
}
