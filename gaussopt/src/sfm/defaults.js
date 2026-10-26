// defaults.js — Options of the video -> cameras session (DESIGN-sfm §6): the worker's defaults,
// which the panel also shows and presets from the URL. Light imports only (the page loads it).
"use strict";

import { FRAME_DEFAULTS } from './frames.js';

export const PIPELINE_DEFAULTS = { window: 10, loops: true };

export const SESSION_DEFAULTS = {
  sampleFps: 30, ...FRAME_DEFAULTS, maxDim: 1920, jpegQuality: 0.95,
  maxFeatures: 4096, ...PIPELINE_DEFAULTS, focal: 'auto', hfovDeg: 0, sweeps: 2,
  autoRepair: true,  // session.repair() once the reconstruction is done (undoable)
  trainMinObs: 0,    // the trainer's points: seen by at least this many keyframes (export.js minObs); 0: 3 for a video, 2 for photos (DESIGN-sfm §13.3 P6)
};

// Mapping a photo set (DESIGN-sfm §13.3 P3, measured on cuisine / salon over 2 inputs x 2 feature
// counts, photo_runs/NOTES.md): every pair matched (the window spans the set), a frame placed from
// one neighbour when PnP has none left, initial pairs by match count, smaller segments regrown into
// the largest (cuisine: 7 of COLMAP's 12 photos in one segment on 1 of 4 inputs without it, 12 on
// all with it), and k2 refined (COLMAP fits k1 = 0.09, k2 = -0.14 on cuisine; with f, k1 only,
// rotations differ from COLMAP's by 1.1-1.5 deg)
export const PHOTO_MAPPING = { relativeRegistration: true, initOrder: 'strongest', regrowSegments: true, refine: ['f', 'k1', 'k2'] };
