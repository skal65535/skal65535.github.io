// FramePacer: how many training steps one frame runs, from the GPU time they took in earlier
// frames. A frame's steps (and its draw) should fill about `targetMs` of GPU time: fewer leave the
// GPU idle while the CPU waits for the next display refresh, more delay every displayed frame.

export class FramePacer {
  constructor(targetMs = 25) {
    this.targetMs  = targetMs;
    this.msPerStep = 0;          // running estimate of the GPU time per step (0: none yet)
    this.lastDoneT = -Infinity;  // when the previous frame's work was done on the GPU
  }

  // Steps for the next frame, in [1, maxSteps]; 2 until a frame was measured
  steps(maxSteps) {
    const n = this.msPerStep > 0 ? Math.floor(this.targetMs / this.msPerStep) : 2;
    return Math.max(1, Math.min(maxSteps, n));
  }

  // A frame ran `steps` steps, the first submitted at `startT`, all done on the GPU at `doneT`
  // (ms). Its GPU work started at the earliest when it was submitted and when the previous
  // frame's was done. A sample is capped at twice the estimate: a frame with a rebalance pauses
  // the CPU between its steps, and the GPU idles meanwhile.
  done(startT, doneT, steps) {
    const gpuMs = doneT - Math.max(startT, this.lastDoneT);
    this.lastDoneT = doneT;
    if (!(steps > 0)) return;
    const sample = gpuMs / steps;
    this.msPerStep = this.msPerStep > 0 ? this.msPerStep + 0.1 * (Math.min(sample, 2 * this.msPerStep) - this.msPerStep) : sample;
  }
}
