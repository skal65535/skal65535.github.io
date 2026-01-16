//
// common WebGPU functions
//
////////////////////////////////////////////////////////////////////////////////

function Create_GPU_Buffer(device, src_buf, buf_usage = 0) {
  const desc = {
    size: (src_buf.byteLength + 3) & ~3,  // needs to be a multiple of 4
    usage: buf_usage | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  };
  const gpu_buf = device.createBuffer(desc);
  new src_buf.constructor(gpu_buf.getMappedRange()).set(src_buf);
  gpu_buf.unmap();
  return gpu_buf;
}

function Create_Uniform_Buffer(device, size) {
  return device.createBuffer({
    size: (size + 3) & ~3,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function Create_Storage_Buffer(device, size, usage = 0) {
  return device.createBuffer({
    size: (size + 3) & ~3,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | usage,
  });
}

function Create_Texture(device, width, height, format, usage, sampleCount = 1, label = undefined) {
  return device.createTexture({
    label: label,
    size: [width, height],
    format: format,
    usage: usage,
    sampleCount: sampleCount,
  });
}

function Create_Bind_Group(device, pipeline, buffers, layout = 0) {
  let entries = [];
  for (let i = 0; i < buffers.length; ++i) {
    entries.push({ binding: i, resource: buffers[i],});
  }
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(layout),
    entries: entries,
  });
}

async function Init_WebGPU(onError) {
  if (!navigator.gpu) {
    const msg = "WebGPU not supported.";
    if (onError) onError(msg);
    throw new Error(msg);
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    const msg = "Couldn’t request WebGPU adapter.";
    if (onError) onError(msg);
    throw new Error(msg);
  }
  const device = await adapter.requestDevice();
  if (!device) {
    const msg = "Couldn’t request WebGPU logical device.";
    if (onError) onError(msg);
    throw new Error(msg);
  }

  device.addEventListener('uncapturederror', (event) => {
    console.error("WebGPU uncaptured error:", event.error.message);
  });

  const format = navigator.gpu.getPreferredCanvasFormat();
  return { device, adapter, format };
}

////////////////////////////////////////////////////////////////////////////////

function RandomRange(a, b) { return Math.random() * (b - a) + a; }
