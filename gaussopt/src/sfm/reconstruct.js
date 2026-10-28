// reconstruct.js — The S2..S4 pipeline on matched keyframes (DESIGN-sfm §5): sequential mapping
// (mapper.js), loop closure and segment merging (loop.js), gliding-window refinement and final
// retriangulation. Input: the `input` shape of synth.js / dump.js.
"use strict";

import { makeRng } from '../math.js';
import { SequentialMapper } from './mapper.js';
import { closeLoops } from './loop.js';
import { joinByFlow } from './flowjoin.js';

// options: mapper options, plus { loops: true, sweeps: 2, retriangulate: true, flowJoin: true }
export function reconstruct(input, options = {}) {
  return finish(input, mapSequential(input, options), options);
}

const split = ({ loops = true, sweeps = 2, retriangulate = true, flowJoin = true, ...mapperOptions }) => ({ loops, sweeps, retriangulate, flowJoin, mapperOptions });

export function mapSequential(input, options) {
  const t = performance.now();
  const mapper = new SequentialMapper(input, { ...split(options).mapperOptions, pairTypes: ['seq'] });
  mapper.run();
  return { mapper, timings: { map: performance.now() - t } };
}

// S3 + S4 on a sequential map (session.js replays them after rejecting a loop)
export function finish(input, { mapper, timings }, options) {
  const { loops, sweeps, retriangulate, flowJoin, mapperOptions } = split(options);
  const stage = (name, fn) => { mapperOptions.progress?.(name, 0, 0); const t = performance.now(); const r = fn(); timings[name] = (timings[name] ?? 0) + performance.now() - t; return r; };
  let loopStats = null;
  if (loops) {
    loopStats = stage('loops', () => closeLoops(mapper, input.pairs, { rng: makeRng(mapperOptions.seed ?? 1), log: mapperOptions.log }));
    mapper.addPairs(input.pairs);
  }
  // pieces still apart across a turn in place: joined by the video's optical flow (flowjoin.js; keyframes
  // without flow, e.g. photos, are left as they are)
  const flowStats = flowJoin ? stage('flowJoin', () => joinByFlow(mapper, { log: mapperOptions.log })) : null;
  if (sweeps) stage('refine', () => mapper.refine({ sweeps }));
  if (retriangulate) stage('retriangulate', () => mapper.extendTracks());
  // refinement and retriangulation extend tracks across segment gaps: segments that could not be
  // merged before may be now (jardin5: the stretch after the hover, DESIGN-sfm R5)
  if (loops && sweeps) {
    const late = stage('lateMerge', () => closeLoops(mapper, input.pairs, { rng: makeRng((mapperOptions.seed ?? 1) + 1), log: mapperOptions.log, mergeOnly: true }));
    if (late.merges) {  // tracks across the joint first, then the sweeps (R5)
      loopStats.merges += late.merges;
      stage('retriangulate', () => mapper.extendTracks());
      stage('refine', () => mapper.refine({ sweeps }));
      stage('retriangulate', () => mapper.extendTracks());
    }
  }
  const model = mapper.model();
  return { mapper, model, stats: { ...mapper.stats, loops: loopStats, flowJoin: flowStats, numSegments: new Set(model.images.map(im => im.segment)).size, timings } };
}
