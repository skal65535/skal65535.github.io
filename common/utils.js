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

function RandomRange(a, b) { return Math.random() * (b - a) + a; }
