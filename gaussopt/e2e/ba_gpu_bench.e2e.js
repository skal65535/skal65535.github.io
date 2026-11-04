// ba_gpu_bench.e2e.js — The GPU bundle adjustment (src/sfm/ba_gpu.js: the reduced system built on the GPU, solved on the
// CPU in f64) timed against ba.js in headless Chrome, on synthetic windows shaped like the mapper's (their logs' 'a window:'
// sizes: the photo sets', the pavilion's, gare2's typical and largest; two smaller, as the first windows), each from the
// same start, 10 iterations at most.
// Deno's WebGPU is no measure of it: a round trip there costs about 16 ms.
//   cd e2e && PORT=8792 node ba_gpu_bench.e2e.js   (env: BENCH=name,name to run some only)
import { join } from 'node:path';
import { run, fail, HERE, PORT, urlOf } from './harness.js';

const WINDOWS = [
  { name: 'tiny', nFree: 4, nPts: 300, track: 2.5 },
  { name: 'small', nFree: 8, nPts: 1200, track: 2.7 },
  { name: 'photos', nFree: 12, nPts: 3500, track: 2.7 },
  { name: 'pavilion', nFree: 48, nPts: 2450, track: 4.4 },
  { name: 'gare2', nFree: 10, nPts: 8700, track: 7 },
  { name: 'gare2-calB', nFree: 20, nPts: 15000, track: 10.7, reps: 2 },
  { name: 'gare2-max', nFree: 60, nPts: 94000, track: 6.4, reps: 1 },
].filter(c => !process.env.BENCH || process.env.BENCH.split(',').includes(c.name));

const f = (x) => x.toFixed(0);
await run(async (page) => {
  await page.goto(`http://localhost:${PORT}${urlOf(join(HERE, 'ba_gpu_bench.html'))}`);
  const r = await page.evaluate(async (windows) => (await import('./ba_gpu_bench.js')).run(windows), WINDOWS);
  console.log(`a round trip (64 KB copied and mapped back): ${r.roundTrip.toFixed(2)} ms`);
  for (const w of r.windows) {
    const { cpu, gpu } = w;
    console.log(`${w.name}: ${w.obs} obs, ${w.nPts} points, ${w.nFree} free, ${gpu.terms} system terms in ${gpu.chunks} chunks`);
    console.log(`  ba.js ${f(cpu.ms)} ms (${cpu.iterations} iterations, ${cpu.solves} solves); GPU ${f(gpu.ms)} ms (${gpu.iterations} iterations, ` +
                `${gpu.solves} solves: setup ${f(gpu.setup)}, build ${f(gpu.build)}, cholesky ${f(gpu.cholesky)}, eval ${f(gpu.eval)}): ` +
                `x ${(gpu.ms / cpu.ms).toFixed(2)}; cost ${(gpu.cost / cpu.cost - 1).toExponential(1)} off, poses ${w.rotApartDeg.toExponential(1)} deg apart`);
    if (!(Math.abs(gpu.cost / cpu.cost - 1) < 1e-5)) fail(`${w.name}: the GPU's cost ${gpu.cost} against ba.js's ${cpu.cost}`);
  }
});
