// gpu.js — WebGPU compute helpers shared by the trainer, the renderer and the SfM kernels.
"use strict";

// Bind group `group` of `pipeline`: resources[binding] (an array, or an object for sparse
// bindings), GPUBuffers wrapped as { buffer }
export function bindGroup(device, pipeline, resources, group = 0) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(group),
    entries: Object.entries(resources).map(([binding, r]) =>
      ({ binding: Number(binding), resource: (typeof r.mapAsync === 'function') ? { buffer: r } : r })),
  });
}

// One compute pass on `enc`: `pipeline`, bind group 0 = `resources` (see bindGroup), `groups`
// workgroups
export function dispatch(device, enc, pipeline, resources, ...groups) {
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup(device, pipeline, resources));
  pass.dispatchWorkgroups(...groups);
  pass.end();
}

// { entry: compute pipeline } from the WGSL file at `url`, its code passed through patch() first
export async function computePipelines(device, url, entries = ['main'], patch = code => code) {
  const module = device.createShaderModule({ code: patch(await (await fetch(url)).text()) });
  return Object.fromEntries(entries.map(e => [e, device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: e } })]));
}
