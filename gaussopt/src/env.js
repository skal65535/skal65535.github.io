// env.js — The background behind the splats (DESIGN.md §9 E13): a cube map of ENV_N x ENV_N texels a face over the
// world's directions, rgb in [0, 1], drawn where a pixel's splats leave it transparent (C + T_final env(dir)) and
// learned with them (`bg_lr`): what is far away (sky, the far end of a valley) goes there instead of onto a sheet of
// splats at the scene's edge. One copy of the lookup (ENV_WGSL, the `_env` include of shaders.wgsl, shaders.js loadShaders)
// for the trainer, the app's view and the gallery.
"use strict";

export const ENV_N = 32;
export const ENV_TEXELS = 6 * ENV_N * ENV_N;

// WGSL: env_taps(d) -> the 4 bilinear taps (texel indices, weights) of world direction d. Faces 0..5: +x -x +y -y +z -z;
// within a face, bilinear between texel centers, clamped at its edges (no filtering across faces). Texel i of the
// layout is (face i / ENV_N^2, row, column): a buffer of vec4f, or a texture ENV_N wide and 6 ENV_N high.
export const ENV_WGSL = `
const ENV_N : u32 = ${ENV_N}u;
struct EnvTaps { i : vec4u, w : vec4f };
fn env_taps(d : vec3f) -> EnvTaps { return env_taps_n(d, ENV_N); }
// at n texels a face: the learned background's ENV_N, or a dome's (trainer.setDome: its texture's width)
fn env_taps_n(d : vec3f, nf : u32) -> EnvTaps {
  let a = abs(d);
  var face = 0u;
  var uv = vec2f(0.0);
  if (a.x >= a.y && a.x >= a.z) {
    face = select(1u, 0u, d.x > 0.0); uv = vec2f(select(d.z, -d.z, d.x > 0.0), -d.y) / max(a.x, 1e-12);
  } else if (a.y >= a.z) {
    face = select(3u, 2u, d.y > 0.0); uv = vec2f(d.x, select(-d.z, d.z, d.y > 0.0)) / max(a.y, 1e-12);
  } else {
    face = select(5u, 4u, d.z > 0.0); uv = vec2f(select(-d.x, d.x, d.z > 0.0), -d.y) / max(a.z, 1e-12);
  }
  let n = f32(nf);
  let st = clamp((uv * 0.5 + 0.5) * n - 0.5, vec2f(0.0), vec2f(n - 1.0));
  let s0 = min(vec2u(floor(st)), vec2u(nf - 2u));
  let f = st - vec2f(s0);
  let b = face * nf * nf + s0.y * nf + s0.x;
  return EnvTaps(vec4u(b, b + 1u, b + nf, b + nf + 1u),
                 vec4f((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y));
}
// the background at 4 taps, from its texture (n wide, 6 n high: texel i at (i % n, i / n))
fn env_texel(t : texture_2d<f32>, i : u32) -> vec3f { let n = textureDimensions(t).x; return textureLoad(t, vec2u(i % n, i / n), 0).rgb; }
// world direction d's taps in texture t, at its own size
fn env_taps_tex(t : texture_2d<f32>, d : vec3f) -> EnvTaps { return env_taps_n(d, textureDimensions(t).x); }
fn env_bilinear(t : texture_2d<f32>, tap : EnvTaps) -> vec3f {
  return env_texel(t, tap.i.x) * tap.w.x + env_texel(t, tap.i.y) * tap.w.y + env_texel(t, tap.i.z) * tap.w.z + env_texel(t, tap.i.w) * tap.w.w;
}
`;

// The cube map as bytes (rgb, ENV_TEXELS x 3) from its GPU layout (a vec4f a texel), and back
export function envToBytes(f32) {
  const out = new Uint8Array(3 * ENV_TEXELS);
  for (let t = 0; t < ENV_TEXELS; t++) for (let c = 0; c < 3; c++) out[3 * t + c] = Math.round(255 * Math.min(1, Math.max(0, f32[4 * t + c])));
  return out;
}
export function envFromBytes(u8) {
  const out = new Float32Array(4 * ENV_TEXELS);
  for (let t = 0; t < ENV_TEXELS; t++) for (let c = 0; c < 3; c++) out[4 * t + c] = u8[3 * t + c] / 255;
  return out;
}
