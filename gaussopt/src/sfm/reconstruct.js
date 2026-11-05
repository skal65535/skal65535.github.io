// reconstruct.js — The S2..S4 pipeline on matched keyframes (DESIGN-sfm §5): sequential mapping
// (mapper.js), loop closure and segment merging (loop.js), gliding-window refinement and final
// retriangulation. Input: the `input` shape of synth.js / dump.js.
"use strict";

import { makeRng } from '../math.js';
import { SequentialMapper } from './mapper.js';
import { closeLoops, mergeSegments } from './loop.js';
import { joinByFlow } from './flowjoin.js';

// options: mapper options, plus { loops: true, sweeps: 2, retriangulate: true, flowJoin: true }
export async function reconstruct(input, options = {}) {
  return finish(input, await mapSequential(input, options), options);
}

const split = ({ loops = true, sweeps = 2, retriangulate = true, flowJoin = true, ...mapperOptions }) => ({ loops, sweeps, retriangulate, flowJoin, mapperOptions });

export async function mapSequential(input, options) {
  const t = performance.now();
  const mapper = new SequentialMapper(input, { ...split(options).mapperOptions, pairTypes: ['seq'] });
  await mapper.run();
  return { mapper, timings: { map: performance.now() - t, 'map.ba': mapper.time.ba } };
}

// S3 + S4 on a sequential map (session.js replays them after rejecting a loop)
export async function finish(input, { mapper, timings }, options) {
  const { loops, sweeps, retriangulate, flowJoin, mapperOptions } = split(options);
  if (timings.map === undefined) mapper.resetTime();  // a replay (session.js): its own time, not the map's since it was made
  // timings[name]: a stage's ms, timings[name + '.ba']: of which in window BAs
  const stage = async (name, fn) => {
    mapperOptions.progress?.(name, 0, 0);
    const t = performance.now(), ba = mapper.time.ba, r = await fn();
    timings[name] = (timings[name] ?? 0) + performance.now() - t;
    timings[`${name}.ba`] = (timings[`${name}.ba`] ?? 0) + mapper.time.ba - ba;
    return r;
  };
  let loopStats = null;
  if (loops) {
    loopStats = await stage('loops', () => closeLoops(mapper, input.pairs, { rng: makeRng(mapperOptions.seed ?? 1), log: mapperOptions.log }));
    mapper.addPairs(input.pairs);
  }
  // pieces still apart across a turn in place: joined by the video's optical flow (flowjoin.js; keyframes
  // without flow, e.g. photos, are left as they are)
  const flowStats = flowJoin ? await stage('flowJoin', () => joinByFlow(mapper, { log: mapperOptions.log })) : null;
  if (sweeps) await stage('refine', () => mapper.refine({ sweeps }));
  if (retriangulate) await stage('retriangulate', () => mapper.extendTracks());
  // refinement and retriangulation extend tracks across segment gaps: segments that could not be
  // merged before may be now (DESIGN-sfm R5)
  if (loops && sweeps) {
    const late = await stage('lateMerge', () => mergeSegments(mapper, input.pairs, { rng: makeRng((mapperOptions.seed ?? 1) + 1), log: mapperOptions.log }));
    if (late.merges) {  // tracks across the joint first, then the sweeps (R5)
      loopStats.merges += late.merges;
      await stage('retriangulate', () => mapper.extendTracks());
      await stage('refine', () => mapper.refine({ sweeps }));
      await stage('retriangulate', () => mapper.extendTracks());
    }
  }
  const model = mapper.model();
  return { mapper, model, stats: { ...mapper.stats, loops: loopStats, flowJoin: flowStats, numSegments: new Set(model.images.map(im => im.segment)).size, timings, time: copyTime(mapper.time) } };
}

// A copy of mapper.time the mapper's later window BAs leave as it is
const reasons = (fallback) => Object.entries(Object.groupBy(fallback, x => x)).map(([k, v]) => `${v.length} ${k}`).join(', ');
const copyTime = (t) => ({ ...t, windows: t.windows.slice(), fallback: t.fallback.slice(), route: { ...t.route } });

// The reconstruction's time by stage, in s (stats: finish's): 'map 812.3 s (BA 700.1 s, 433 windows, ...), loops ...'
export function timingsLine({ timings, time }) {
  const s = (ms) => `${(ms / 1000).toFixed(1)} s`;
  const ba = (name) => timings[`${name}.ba`] ? ` (BA ${s(timings[`${name}.ba`])})` : '';
  const parts = [`map ${s(timings.map)} (BA ${s(timings['map.ba'])}; outside it, initial pairs ${s(time.init)}, focal ${s(time.focal)})`];
  for (const name of ['loopMatch', 'loops', 'flowJoin', 'refine', 'retriangulate', 'lateMerge'])
    if (timings[name] !== undefined) parts.push(`${name} ${s(timings[name])}${ba(name)}`);
  return `${parts.join(', ')}; BA in all ${s(time.ba)} over ${time.baCalls} windows, ${s(time.solve)} of it solving, ${time.route.gpu} on the GPU solver, ${time.route.cpu} on the CPU's` + (time.fallback.length ? `, ${time.fallback.length} failed on the GPU, redone by ba.js (${reasons(time.fallback)})` : '') + windowsLine(time.windows);
}

// The window BAs' sizes (median and max) and their solve time by part
function windowsLine(windows) {
  if (!windows?.length) return '';
  const stat = (k) => {
    const v = windows.map(w => w[k]).sort((a, b) => a - b);
    return `${v[v.length >> 1]} (max ${v.at(-1)})`;
  };
  const sum = (k) => windows.reduce((a, w) => a + (w[k] ?? 0), 0);
  // ba.js's parts, and the GPU solver's (ba_gpu.js) when it solved some
  const gpu = ['setup', 'build', 'eval'].filter(k => windows.some(w => w[k] !== undefined));
  const parts = ['linearize', 'normal', 'schur', 'cholesky', 'back', 'cost', ...gpu].map(k => `${k} ${(sum(k) / 1000).toFixed(1)} s`);
  return `; a window: ${stat('obs')} observations, ${stat('freeObs')} of them by free frames, ${stat('points')} points, ` +
         `${stat('free')} free frames; solving: ${parts.join(', ')}, ${sum('solves')} solves, ${sum('evals')} cost evaluations`;
}
