// pipeline.js — S1..S4 on keyframes (DESIGN-sfm §5.2-5.5), shared by tools/run_frames.js and the
// browser worker: GPU SIFT -> loop candidates by appearance -> GPU matching (sequential window +
// loops) -> verification -> reconstruction, with geometric loop candidates once the sequential
// map exists.
"use strict";

import { matchFeatures, matchPairs, clipsOf } from './frontend.js';
import { mergeTrackPairs } from './track.js';
import { buildVocabulary, bagOfWords, loopCandidates, geometricCandidates } from './loopdetect.js';
import { SfmSession, withNames } from './session.js';
import { timingsLine } from './reconstruct.js';
import { defaultCamera } from './camera.js';
import { makeRng } from '../math.js';
import { PIPELINE_DEFAULTS } from './defaults.js';
import { seconds } from './util.js';

// features: [{ name, keypoints, descriptors }] in time order, all `width` x `height`.
// mapping: mapper / reconstruct options (focal, hfovDeg, sweeps, ...); log(line), progress(stage, done, total); trace: the
// mapper's (trace.js), null: none;
// trackPairs: the matches of the keypoints tracked through the video (track.js, the features' indices); seq: the
// sequential pairs already matched (frontend.js adaptFeatures), the loop candidates then matched alone.
// checkpoint(session), checkpointMs: session.js reconstruct's, during the mapping; resume: a session from such a checkpoint
// (SfmSession.fromState): no matching (its pairs), its mapping gone on with.
// Returns { session (session.js, for repairs), mapper, model, stats, input, matched (null on a resume) }.
export async function reconstructFromFeatures(features, width, height, matcher, options = {}) {
  const o = { ...PIPELINE_DEFAULTS, log: () => {}, progress: () => {}, ...options };
  const rng = makeRng(o.seed ?? 1), log = o.log;
  if (o.resume) return reconstructSession(o.resume, o.resume._input(), null, { features, matcher, o, rng, resume: true });
  const t1 = performance.now();
  let candidates = [];
  if (o.loops) {
    o.progress('loopDetect', 0, 0);
    candidates = loopCandidates(await bagOfWords(features, await buildVocabulary(features, matcher, { rng }), matcher),
                                { minGap: 2 * o.window, clips: clipsOf(features) });
    log(`loop detection: ${candidates.length} candidate pairs in ${seconds(t1)}`);
  }
  const tm = performance.now();
  const matched = await matchFeatures(features, matcher, { rng, window: o.window, loopCandidates: candidates, seq: o.seq,
                                                          onProgress: (d, n) => o.progress('match', d, n) });
  const ms = matched.stats;
  log(`matching: ${ms.matched} pairs, ${ms.verified} verified (loops ${ms.loopsVerified}/${ms.loopsTried}), ` +
      `${Math.round(ms.inliers / Math.max(1, ms.verified))} inliers per pair in ${seconds(tm)} (GPU matching ${(ms.gpuMs / 1000).toFixed(1)} s, verification ${(ms.verifyMs / 1000).toFixed(1)} s of CPU ` +
      `${ms.workers ? `on ${ms.workers} workers, done ${(ms.waitMs / 1000).toFixed(1)} s after the GPU` : 'inline'})`);
  const clips = clipsOf(features);
  if (clips) {  // several videos: what ties them
    const across = matched.pairs.filter(p => clips[p.i] !== clips[p.j]);
    log(`matching: ${across.length} verified pairs between two videos (${across.reduce((s, p) => s + p.matches.length / 2, 0)} inliers)`);
  }
  if (o.trackPairs?.length) {
    const t = performance.now(), { pairs, stats } = await mergeTrackPairs(matched.pairs, matched.images, o.trackPairs,
                                                                  { seed: o.seed ?? 1, onProgress: (d, n) => o.progress('trackMerge', d, n) });
    matched.pairs = pairs;
    log(`tracks: ${stats.pairs} pairs with their matches, ${stats.added} matches more than SIFT's, in ${seconds(t)}`);
  }

  // a camera per lens when given (o.cameras, the features' `camera` index: photo sets, DESIGN-sfm §13 P8)
  const input = { camera: o.camera ?? o.cameras?.[0] ?? defaultCamera('OPENCV', width, height), ...(o.cameras ? { cameras: o.cameras } : {}),
                  images: matched.images, pairs: matched.pairs };
  const session = new SfmSession(input, { pairSource: featurePairSource(features, matcher), mapping: { window: o.window, ...o.mapping },
                                          loops: o.loops, window: o.window, seed: o.seed ?? 1, log: (s) => log(`  ${s}`), progress: o.progress, trace: o.trace });
  return reconstructSession(session, input, matched, { features, matcher, o, rng, resume: false });
}

// The session's reconstruction (S2..S4), with the geometric loop candidates matched as it goes
async function reconstructSession(session, input, matched, { features, matcher, o, rng, resume }) {
  const log = o.log;
  const have = new Set(input.pairs.map(p => `${p.i},${p.j}`));
  // before (after the segment merges): each frame's segment before them, the frames then apart looked at (lateLoops:
  // false, none)
  const proposeLoops = async (mapper, before) => {
    if (before && o.lateLoops === false) return [];
    const geo = geometricCandidates(mapper, { minGap: 2 * o.window, perFrame: 16, before, rng }).filter(([i, j]) => !have.has(`${i},${j}`));
    for (const [i, j] of geo) have.add(`${i},${j}`);
    const { pairs, stats } = await matchPairs(features, matcher, geo.map(([i, j]) => [i, j, 'loop']),
                                              { rng, onProgress: (d, n) => o.progress('loopMatch', d, n) });
    log(`  geometric loops${before ? ' after the merges' : ''}: ${geo.length} candidates, ${stats.verified} verified`);
    return pairs;
  };
  const t2 = performance.now();
  const { mapper, model, stats } = await session.reconstruct({ proposeLoops, checkpoint: o.checkpoint ?? null, checkpointMs: o.checkpointMs, resume });
  log(`reconstruct: ${model.images.length}/${features.length} registered, ${stats.numSegments} segment(s), ` +
      `focal ${model.focal.toFixed(1)} px in ${seconds(t2)}`);
  log(`  time: ${timingsLine(stats)}`);
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
