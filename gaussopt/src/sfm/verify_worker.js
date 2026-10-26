// verify_worker.js — Geometric verification of matched pairs for the pool in frontend.js:
// in { id, xy (frontend.js packPair), seed, maxError, minInliers }, out { id, result (verifyPacked), ms }.
"use strict";

import { verifyPacked } from './frontend.js';

self.onmessage = ({ data: { id, xy, seed, maxError, minInliers } }) => {
  const t = performance.now(), result = verifyPacked(xy, seed, { maxError, minInliers });
  self.postMessage({ id, result, ms: performance.now() - t }, result ? [result.inliers.buffer] : []);
};
