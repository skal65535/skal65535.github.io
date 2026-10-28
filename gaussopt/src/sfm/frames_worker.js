// frames_worker.js — S0 analysis of a chunk of consecutive frames (frames.js), for the pool in
// worker.js: in { frames: [Uint8Array gray], w, h, options, overlap }, out { sharpness, motion, zoom, flowH }
// (per step, of frames.js step(): zoom [scale, fit] or null, flowH its homography H or null).
// With overlap, frames[0] is the last frame of the previous chunk: it only provides the motion to
// frames[1] and is not reported.
"use strict";

import { analyzeFrame, step } from './frames.js';

self.onmessage = ({ data: { frames, w, h, options, overlap } }) => {
  const sharpness = [], steps = [], zoom = [], flowH = [];
  let prev = null;
  frames.forEach((gray, k) => {
    const a = analyzeFrame(gray, w, h, options);
    if (k > 0 || !overlap) {
      const s = prev ? step(prev, a.flow) : { motion: 0, scale: null, H: null };
      sharpness.push(a.sharpness);
      steps.push(s.motion);
      zoom.push(s.scale === null ? null : [s.scale, s.fit]);
      flowH.push(s.H);
    }
    prev = a.flow;
  });
  self.postMessage({ sharpness, motion: steps, zoom, flowH });
};
