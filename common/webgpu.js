// common/webgpu.js — shared WebGPU plumbing.
//
// ES module. Exports the small set of helpers each demo currently rewrites
// (device init + canvas configure, buffer / texture / bind-group creation).

export async function initGPU(canvas, { alphaMode = 'premultiplied' } = {}) {
  if (!navigator.gpu) throw new Error('WebGPU not supported.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("Couldn't request WebGPU adapter.");
  const device = await adapter.requestDevice();
  if (!device)  throw new Error("Couldn't request WebGPU logical device.");
  device.addEventListener('uncapturederror',
    e => console.error('WebGPU uncaptured error:', e.error.message));
  const format = navigator.gpu.getPreferredCanvasFormat();
  if (!canvas) return { device, adapter, format, context: null };
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Could not get WebGPU canvas context.');
  context.configure({ device, format, alphaMode });
  return { device, adapter, context, format };
}

export function createGPUBuffer(device, src, extraUsage = 0) {
  const buf = device.createBuffer({
    size:  (src.byteLength + 3) & ~3,
    usage: extraUsage | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new src.constructor(buf.getMappedRange()).set(src);
  buf.unmap();
  return buf;
}

export function createUniformBuffer(device, size) {
  return device.createBuffer({
    size:  (size + 3) & ~3,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function createStorageBuffer(device, size, extraUsage = 0) {
  return device.createBuffer({
    size:  (size + 3) & ~3,
    usage: extraUsage | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

export function createTexture(device, w, h, format, usage, sampleCount = 1, label) {
  return device.createTexture({ label, size: [w, h], format, usage, sampleCount });
}

export function createBindGroup(device, pipeline, resources, layout = 0) {
  return device.createBindGroup({
    layout:  pipeline.getBindGroupLayout(layout),
    entries: resources.map((r, i) => ({ binding: i, resource: r })),
  });
}
