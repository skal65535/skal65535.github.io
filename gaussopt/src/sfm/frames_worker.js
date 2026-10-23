// frames_worker.js — S0 analysis of a chunk of consecutive frames (frames.js), for the pool in
// worker.js: in { frames: [Uint8Array gray], w, h, options, overlap }, out { sharpness, motion }.
// With overlap, frames[0] is the last frame of the previous chunk: it only provides the motion to
// frames[1] and is not reported.
"use strict";

import { analyzeFrame, motion } from './frames.js';

self.onmessage = ({ data: { frames, w, h, options, overlap } }) => {
  const sharpness = [], steps = [];
  let prev = null;
  frames.forEach((gray, k) => {
    const a = analyzeFrame(gray, w, h, options);
    if (k > 0 || !overlap) {
      sharpness.push(a.sharpness);
      steps.push(prev ? motion(prev, a.flow) : 0);
    }
    prev = a.flow;
  });
  self.postMessage({ sharpness, motion: steps });
};
