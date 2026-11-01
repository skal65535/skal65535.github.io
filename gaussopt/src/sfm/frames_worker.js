// frames_worker.js — S0 analysis of a chunk of consecutive frames (frames.js), for the pool in
// worker.js: in { frames: [Uint8Array gray], w, h, options, overlap, mask }, out { sharpness, motion, zoom, flowH }
// (per step, of frames.js step(): zoom [scale, fit] or null, flowH its homography H or null).
// With overlap, frames[0] is the last frame of the previous chunk: it only provides the motion to
// frames[1] and is not reported. mask: an on-screen display's (overlay.js unpackMask): no corner tracked on it
// nor within TRACK_MARGIN of it (its edges would hold the track still).
"use strict";

import { analyzeFrame, step, TRACK_MARGIN } from './frames.js';
import { maskAt, grownMask } from './overlay.js';

self.onmessage = ({ data: { frames, w, h, options, overlap, mask } }) => {
  const sharpness = [], steps = [], zoom = [], flowH = [];
  let grown = null;  // the mask grown by TRACK_MARGIN pixels of the W-wide pyramid base
  const keep = mask ? (x, y, W, H) => {
    if (grown?.W !== W) grown = { W, ov: grownMask(mask, Math.ceil(TRACK_MARGIN * mask.w / W)) };
    return !maskAt(grown.ov, x, y, W, H);
  } : null;
  let prev = null;
  frames.forEach((gray, k) => {
    const a = analyzeFrame(gray, w, h, options);
    if (k > 0 || !overlap) {
      const s = prev ? step(prev, a.flow, keep) : { motion: 0, scale: null, H: null };
      sharpness.push(a.sharpness);
      steps.push(s.motion);
      zoom.push(s.scale === null ? null : [s.scale, s.fit]);
      flowH.push(s.H);
    }
    prev = a.flow;
  });
  self.postMessage({ sharpness, motion: steps, zoom, flowH });
};
