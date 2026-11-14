// defaults.js — Options of the video -> cameras session (DESIGN-sfm §6): the worker's defaults,
// which the panel also shows and presets from the URL. Light imports only (the page loads it).
"use strict";

import { FRAME_DEFAULTS } from './frames.js';

// The sequential window, in keyframes: each one matched with the next WINDOW (frontend.js), the window BA's
// free frames (mapper.js), a session's neighbours (session.js)
export const WINDOW = 10;

// lateLoops: geometric loop candidates again after the segment merges, between the frames of segments apart before
// them (loop.js closeLoops afterMerge)
export const PIPELINE_DEFAULTS = { window: WINDOW, loops: true, lateLoops: true };

// The options a stage's result depends on: the keyframes' features (worker.js reuses them while these are the same), all
// the keyframes' (their selection, FRAME_DEFAULTS, too), the cameras' (the mapping); the stepper's "Options changed"
// (ui/phases.js) compares them with the last run's
export const FEATURE_KEYS = ['maxDim', 'maxFeatures', 'siftPeak', 'siftLift', 'rootSift', 'jpegQuality', 'overlay', 'track', 'adaptFeatures', 'denseFeatures', 'densePeak', 'minLink'];
export const KEYFRAME_KEYS = ['sampleFps', ...Object.keys(FRAME_DEFAULTS), ...FEATURE_KEYS];
export const CAMERA_KEYS = [...Object.keys(PIPELINE_DEFAULTS), 'focal', 'hfovDeg', 'sweeps', 'globalBARatio', 'globalRefinements', 'localBA', 'reMinRatio'];

export const SESSION_DEFAULTS = {
  sampleFps: 30, ...FRAME_DEFAULTS, maxDim: 1920, jpegQuality: 0.95,
  maxFeatures: 8192, siftPeak: 0.02 / 3, siftLift: 'none', rootSift: false, ...PIPELINE_DEFAULTS, focal: 'auto', hfovDeg: 0, sweeps: 2,  // siftPeak: sift.js SIFT_DEFAULTS.peak
  globalBARatio: 1.1, globalRefinements: 5, localBA: 'covisible', reMinRatio: 0.05,
  autoRepair: true,  // session.repair() once the reconstruction is done (undoable)
  overlay: true,     // a video's on-screen display (burnt-in text, boxes, a reticle) masked: the features and the training leave it out (overlay.js)
  track: true,       // a video's keyframe keypoints tracked through every sampled frame: longer tracks, the matches of keyframes farther apart (track.js)
  // a video's keyframes extracted with up to denseFeatures at densePeak, kept so where a link between keyframes is weak
  // (the sequential pairs' inliers across it under minLink of their median), trimmed to maxFeatures at siftPeak elsewhere
  // (frontend.js adaptFeatures)
  adaptFeatures: true, denseFeatures: 10000, densePeak: 0.02 / 6, minLink: 0.1,
  trace: true,       // the mapping recorded for the SfM theatre (trace.js, ui/theatre.js): streamed to the page, kept in a project's cache/trace.bin
  trainMinObs: 0,    // the trainer's points: seen by at least this many keyframes (export.js minObs); 0: 3 for a video, 2 for photos (DESIGN-sfm §13.3 P6)
};

// Mapping a photo set (DESIGN-sfm §13.3 P3): every pair matched (the window spans the set), a frame
// placed from one neighbour when PnP has none left, initial pairs by match count, smaller segments
// regrown into the largest (without it a set can stay split), the pieces no 3D points merge then placed with it
// (joinPieces, joinpieces.js), k2 refined (with f, k1 only, the rotations of a wide lens drift from COLMAP's),
// whatever the focal's source, while the first 50 photos register, and
// the points of two-view tracks kept (ignoreTwoView false: a sparse set's photos share few points with three or more;
// cuisine's 18: 12 cameras to the trainer and 1981 points with them, 11 and 651 without)
export const PHOTO_MAPPING = { relativeRegistration: true, initOrder: 'strongest', regrowSegments: true, joinPieces: true, refine: ['f', 'k1', 'k2'],
                               refineUnobserved: ['f', 'k1', 'k2'], refineIntrinsicsUntil: 50, ignoreTwoView: false };
