// GPU matcher (match.wgsl) against the CPU reference, and (slow) against COLMAP's own matches.
import { GpuMatcher, matchCPU } from '../src/sfm/match.js';
import { loadDump, denoReader } from '../src/sfm/dump.js';
import { assert, rng, SLOW, env, hasGPU } from './util.js';
import { gaussian } from '../src/sfm/synth.js';

// COLMAP-like descriptors: unit L2 x 512, clamped to 255; pairs share a noisy copy of some rows
function descriptors(r, n, base = null, noise = 0.15) {
  const out = new Uint8Array(128 * n);
  for (let i = 0; i < n; i++) {
    const v = Array.from({ length: 128 }, (_, k) => base && i < base.length / 128 ? base[128 * i + k] / 512 + noise * Math.abs(gaussian(r)) * 0.1 : Math.abs(gaussian(r)));
    const norm = Math.hypot(...v) || 1;
    for (let k = 0; k < 128; k++) out[128 * i + k] = Math.min(255, Math.round(512 * v[k] / norm));
  }
  return out;
}

Deno.test({ name: 'GPU matcher equals the CPU reference', ignore: !hasGPU, fn: async () => {
  const r = rng(60);
  const A = descriptors(r, 700), B = descriptors(r, 900, A);
  const matcher = await GpuMatcher.create();
  for (const opts of [{}, { crossCheck: false }, { maxRatio: 0.6 }]) {
    const gpu = await matcher.match(A, B, opts), cpu = matchCPU(A, B, opts);
    assert(gpu.length === cpu.length && gpu.every((x, k) => x === cpu[k]), `${JSON.stringify(opts)}: ${gpu.length / 2} vs ${cpu.length / 2} matches`);
    assert(gpu.length / 2 > 300, `${gpu.length / 2} matches`);
  }
} });

const DUMP = env('SFM_DESC_DUMP');

// COLMAP's matcher (SiftGPU, OpenGL) and ours implement the same rule: on jardin5 (4096 features,
// under COLMAP's 16384 clamp) the matches are identical (91,900 on 20 pairs).
Deno.test({ name: 'GPU matcher vs COLMAP raw matches (SFM_DESC_DUMP)', ignore: !SLOW || !DUMP || !hasGPU, fn: async () => {
  const dump = await loadDump(denoReader(DUMP), { descriptors: true });
  const matcher = await GpuMatcher.create();
  let ours = 0, theirs = 0, common = 0, ms = 0, pairs = 0;
  for (const p of dump.pairs.filter(p => p.raw && p.j - p.i <= 2).slice(0, 20)) {
    const t = performance.now();
    const m = await matcher.match(dump.images[p.i].descriptors, dump.images[p.j].descriptors, {}, [p.i, p.j]);
    ms += performance.now() - t; pairs++;
    const set = new Set();
    for (let k = 0; k < m.length; k += 2) set.add(m[k] * 65536 + m[k + 1]);
    for (let k = 0; k < p.raw.length; k += 2) common += set.has(p.raw[k] * 65536 + p.raw[k + 1]);
    ours += m.length / 2; theirs += p.raw.length / 2;
  }
  console.log(`  ${pairs} pairs: ours ${ours}, COLMAP ${theirs}, common ${common} (${(100 * common / theirs).toFixed(1)}% of COLMAP's); ${(ms / pairs).toFixed(1)} ms per pair`);
  assert(common === theirs && ours === theirs, 'same matches as COLMAP');
} });
