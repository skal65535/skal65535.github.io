// shaders.js — shaders.wgsl's sections as shader modules (the trainer's and the renderer's), with the shared code
// inserted (env.js's _env, loss_acc.js's _lossacc and the file's own `_*` sections).
"use strict";

import { ENV_WGSL } from './env.js';
import { LOSS_WGSL } from './loss_acc.js';

let _shaderCache = null;

// `// === <name> ===` sections -> { name: code }, the `_*` ones (shared code, and env.js's and loss_acc.js's) inserted
// at each `// #include _<name>` line and left out
function splitWgslSource(src) {
  const sections = { _env: ENV_WGSL, _lossacc: LOSS_WGSL }, out = {};
  const re = /\/\/\s*===\s*([a-zA-Z0-9_]+)\s*===\s*\n/g;
  const parts = src.split(re);
  for (let i = 1; i < parts.length; i += 2) sections[parts[i]] = parts[i + 1];
  for (const [name, code] of Object.entries(sections)) {
    if (name.startsWith('_')) continue;
    out[name] = code.replace(/^\/\/ #include (_\w+)$/gm, (_, inc) => {
      if (!(inc in sections)) throw new Error(`shaders.wgsl ${name}: no section ${inc}`);
      return sections[inc];
    });
  }
  return out;
}

// Lines ending in `// @sg` are kept only when the device has 'subgroups' (and the module
// then gets `enable subgroups;`), lines ending in `// @nosg` only when it hasn't.
function selectSubgroupLines(code, device) {
  if (!/\/\/ @(no)?sg\s*$/m.test(code)) return code;
  const sg = device.features.has('subgroups');
  const lines = code.split('\n').filter(l => !(sg ? /\/\/ @nosg\s*$/ : /\/\/ @sg\s*$/).test(l));
  return (sg ? 'enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n' : '') + lines.join('\n');
}

export async function loadShaders(device) {
  if (_shaderCache && _shaderCache.device === device) return _shaderCache.mods;
  const url = new URL('./shaders.wgsl', import.meta.url);
  const src = await (await fetch(url)).text();
  const map = splitWgslSource(src);
  const mods = {};
  for (const [name, code] of Object.entries(map)) {
    mods[name] = device.createShaderModule({ code: selectSubgroupLines(code, device), label: name });
  }
  _shaderCache = { device, mods };
  return mods;
}
