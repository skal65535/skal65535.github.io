function Create_GPU_Buffer(device, src_buf, buf_usage = 0, as_u32 = false) {
  const gpu_buf = device.createBuffer({
    size: src_buf.length * 4,
    usage: buf_usage | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  if (as_u32) {
    new Uint32Array(gpu_buf.getMappedRange()).set(src_buf);
  } else {
    new Float32Array(gpu_buf.getMappedRange()).set(src_buf);
  }
  gpu_buf.unmap();
  return gpu_buf;
}

function RandomRange(a, b) { return Math.random() * (b - a) + a; }
