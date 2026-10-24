// FramePacer (pace.js): steps per frame from the GPU time of earlier frames, on simulated
// frame timings.
import { FramePacer } from '../src/pace.js';
import { assert, assertClose } from './util.js';

// Frames back to back on a GPU taking `msPerStep` per step, submitted `lagMs` after the previous
// frame's work was done (or earlier, lagMs < 0: queued behind it); returns the last step count
function simulate(pacer, msPerStep, frames, { lagMs = 0, maxSteps = 20 } = {}) {
  let doneT = Math.max(0, pacer.lastDoneT), n = 0;
  for (let f = 0; f < frames; f++) {
    n = pacer.steps(maxSteps);
    const start = f === 0 ? doneT : doneT + lagMs;  // the first one on an idle GPU
    doneT = Math.max(start, doneT) + n * msPerStep;
    pacer.done(start, doneT, n);
  }
  return n;
}

Deno.test('pace: 2 steps before any frame is measured, then the GPU time fills the target', () => {
  const p = new FramePacer(25);
  assert(p.steps(20) === 2, `first frame: ${p.steps(20)} steps`);
  assert(p.steps(1) === 1, 'bounded by maxSteps');
  assert(simulate(p, 7.5, 60) === 3, `7.5 ms/step, 25 ms: ${p.steps(20)} steps`);
  assertClose(p.msPerStep, 7.5, 1e-9, 'estimate');
});

Deno.test('pace: a frame queued behind the previous one counts from when that one was done', () => {
  // submitted 10 ms before the GPU was free: the queued time is not this frame's work
  const p = new FramePacer(25);
  simulate(p, 2.5, 60, { lagMs: -10 });
  assertClose(p.msPerStep, 2.5, 1e-9, 'estimate with a backlog');
  // submitted 30 ms after the GPU went idle: the idle time is not either
  const q = new FramePacer(25);
  simulate(q, 2.5, 60, { lagMs: 30 });
  assertClose(q.msPerStep, 2.5, 1e-9, 'estimate with an idle GPU');
  assert(q.steps(20) === 10 && q.steps(8) === 8, `2.5 ms/step: ${q.steps(20)} steps, ${q.steps(8)} under maxSteps 8`);
});

Deno.test('pace: one slow frame (a rebalance) moves the estimate by at most 10%', () => {
  const p = new FramePacer(25);
  simulate(p, 7.5, 60);
  p.done(p.lastDoneT, p.lastDoneT + 150, 3);  // 150 ms for 3 steps
  assert(p.msPerStep <= 7.5 * 1.1 + 1e-9, `after the slow frame: ${p.msPerStep} ms/step`);
  assert(p.steps(20) === 3, `${p.steps(20)} steps after the slow frame`);
  // a slower scene: the estimate follows within ~25 frames
  assert(simulate(p, 12.5, 25) === 2, `12.5 ms/step: ${p.steps(20)} steps`);
});
