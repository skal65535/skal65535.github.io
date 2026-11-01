// loss_acc.js — The layout of loss_acc, the training step's sums (tile_backward, env_adam, cam_reduce; trainer.js
// reads it), in 4-byte words: the loss, then the background's gradient, then the camera's (the exposure's, then the
// pose's and the lens's). LOSS_WGSL: the same as WGSL constants, the `_lossacc` include of shaders.wgsl.
"use strict";

import { ENV_TEXELS } from './env.js';
import { EXPO_FLOATS } from './exposure.js';

export const LOSS_ENV_AT  = 4;                            // after the loss (2 words: a 64-bit sum) and 2 pad: a vec4f a texel
export const LOSS_EXPO_AT = LOSS_ENV_AT + 4 * ENV_TEXELS;  // the exposure's EXPO_FLOATS: dL/dlog g (rgb), pad, dL/db (rgb), pad
export const LOSS_CAM_AT  = LOSS_EXPO_AT + EXPO_FLOATS;    // the pose's and the lens's: rho, phi, log f, cx, cy, k1, 2 pad
export const CAMG_FLOATS  = EXPO_FLOATS + 12;              // the camera's words, read back together (from LOSS_EXPO_AT)
export const LOSS_WORDS   = LOSS_EXPO_AT + CAMG_FLOATS;

export const LOSS_WGSL = `
const LOSS_ENV_AT  : u32 = ${LOSS_ENV_AT}u;
const LOSS_EXPO_AT : u32 = ${LOSS_EXPO_AT}u;
const LOSS_CAM_AT  : u32 = ${LOSS_CAM_AT}u;
`;
