// defaults.js — Options of the video -> cameras session (DESIGN-sfm §6): the worker's defaults,
// which the panel also shows and presets from the URL. Light imports only (the page loads it).
"use strict";

import { FRAME_DEFAULTS } from './frames.js';

export const PIPELINE_DEFAULTS = { window: 10, loops: true };

export const SESSION_DEFAULTS = {
  sampleFps: 30, ...FRAME_DEFAULTS, maxDim: 1920, jpegQuality: 0.95,
  maxFeatures: 4096, siftPeak: 0.02 / 3, ...PIPELINE_DEFAULTS, focal: 'auto', hfovDeg: 0, sweeps: 2,  // siftPeak: sift.js SIFT_DEFAULTS.peak
  autoRepair: true,  // session.repair() once the reconstruction is done (undoable)
  overlay: true,     // a video's on-screen display (burnt-in text, boxes, a reticle) masked: the features and the training leave it out (overlay.js)
  trainMinObs: 0,    // the trainer's points: seen by at least this many keyframes (export.js minObs); 0: 3 for a video, 2 for photos (DESIGN-sfm §13.3 P6)
};

// Mapping a photo set (DESIGN-sfm §13.3 P3): every pair matched (the window spans the set), a frame
// placed from one neighbour when PnP has none left, initial pairs by match count, smaller segments
// regrown into the largest (without it a set can stay split), and k2 refined (with f, k1 only, the
// rotations of a wide lens drift from COLMAP's)
export const PHOTO_MAPPING = { relativeRegistration: true, initOrder: 'strongest', regrowSegments: true, refine: ['f', 'k1', 'k2'] };
