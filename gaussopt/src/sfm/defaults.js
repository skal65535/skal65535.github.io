// defaults.js — Options of the video -> cameras session (DESIGN-sfm §6): the worker's defaults,
// which the panel also shows and presets from the URL. Light imports only (the page loads it).
"use strict";

import { FRAME_DEFAULTS } from './frames.js';

export const PIPELINE_DEFAULTS = { maxFeatures: 4096, window: 10, loops: true };  // maxFeatures: for the SIFT before

export const SESSION_DEFAULTS = {
  sampleFps: 30, ...FRAME_DEFAULTS, maxDim: 1920, jpegQuality: 0.95,
  ...PIPELINE_DEFAULTS, focal: 'auto', hfovDeg: 0, sweeps: 2,
  autoRepair: true,  // session.repair() once the reconstruction is done (undoable)
};
