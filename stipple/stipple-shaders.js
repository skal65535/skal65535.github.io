// stipple-shaders.js — load and cache the WGSL kernels.
//
// The .wgsl file contains multiple kernels separated by `// === <name> ===`
// markers; this loader splits the source and compiles each as an independent
// GPUShaderModule. Modules are cached per device.
"use strict";

let _cache = null;

function splitSource(src) {
  const out = {};
  const re = /\/\/\s*===\s*([a-zA-Z0-9_]+)\s*===\s*\n/g;
  const parts = src.split(re);
  for (let i = 1; i < parts.length; i += 2) out[parts[i]] = parts[i + 1];
  return out;
}

export async function loadShaders(device) {
  if (_cache && _cache.device === device) return _cache.mods;
  const url = new URL('./stipple-shaders.wgsl', import.meta.url);
  const src = await (await fetch(url)).text();
  const map = splitSource(src);
  const mods = {};
  for (const [name, code] of Object.entries(map)) {
    mods[name] = device.createShaderModule({ code, label: name });
  }
  _cache = { device, mods };
  return mods;
}
