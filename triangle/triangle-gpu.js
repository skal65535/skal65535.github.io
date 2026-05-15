'use strict';

// Phase 3: WebGPU rasterizer + SAD reduction.
// Renders triangles at grid resolution (gx×gy), computes weighted SAD vs reference.

const RENDER_WGSL = /* wgsl */`
struct Uniforms {
  inv_gx : f32,
  inv_gy : f32,
};
@group(0) @binding(0) var<uniform> uni : Uniforms;
@group(0) @binding(1) var<storage, read> positions : array<f32>;   // [nv*2] x,y in grid coords
@group(0) @binding(2) var<storage, read> colorIdx  : array<u32>;   // [nv]
@group(0) @binding(3) var<storage, read> palette   : array<f32>;   // [nc*3] r,g,b in [0,255]

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       col : vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  let gx = positions[vi * 2u];
  let gy = positions[vi * 2u + 1u];
  let ndcx =  gx * uni.inv_gx * 2.0 - 1.0;
  let ndcy = -(gy * uni.inv_gy * 2.0 - 1.0);
  let ci = colorIdx[vi] * 3u;
  var o : VSOut;
  o.pos = vec4<f32>(ndcx, ndcy, 0.0, 1.0);
  o.col = vec3<f32>(palette[ci], palette[ci+1u], palette[ci+2u]);
  return o;
}

@fragment
fn fs_main(@location(0) col : vec3<f32>) -> @location(0) vec4<f32> {
  return vec4<f32>(col / 255.0, 1.0);
}
`;

const SAD_WGSL = /* wgsl */`
@group(0) @binding(0) var      rendered  : texture_2d<f32>;
@group(0) @binding(1) var<storage, read>       reference : array<f32>;  // [gx*gy*4] RGBA [0,255]
@group(0) @binding(2) var<storage, read_write> accum     : atomic<u32>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let sz = textureDimensions(rendered);
  if (gid.x >= sz.x || gid.y >= sz.y) { return; }
  let pixel = textureLoad(rendered, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
  let base  = (gid.y * sz.x + gid.x) * 4u;
  let dr = pixel.r - reference[base]     / 255.0;
  let dg = pixel.g - reference[base + 1u] / 255.0;
  let db = pixel.b - reference[base + 2u] / 255.0;
  let sad = 0.3 * abs(dr) + 0.6 * abs(dg) + 0.1 * abs(db);
  let fixed = u32(sad * 65536.0);
  atomicAdd(&accum, fixed);
}
`;

// ---------------------------------------------------------------------------

class TriangleGPU {
  constructor(device, gx, gy) {
    this.device = device;
    this.gx = gx;
    this.gy = gy;
    this._ref = null;   // GPUBuffer for reference grid
    this._buildRenderPipeline();
    this._buildSADPipeline();
    this._buildAccumBuffer();
  }

  _buildRenderPipeline() {
    const { device, gx, gy } = this;
    const mod = device.createShaderModule({ code: RENDER_WGSL });

    this._renderTarget = device.createTexture({
      size: [gx, gy],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    const bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ]});

    this._renderPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex:   { module: mod, entryPoint: 'vs_main' },
      fragment: { module: mod, entryPoint: 'fs_main',
                  targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });

    this._renderBGL = bgl;

    this._uniBuffer = device.createBuffer({
      size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this._uniBuffer, 0,
      new Float32Array([1 / (gx - 1 || 1), 1 / (gy - 1 || 1)]));
  }

  _buildSADPipeline() {
    const { device, gx, gy } = this;
    const mod = device.createShaderModule({ code: SAD_WGSL });

    const bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'float', viewDimension: '2d' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]});

    this._sadPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: mod, entryPoint: 'cs_main' },
    });

    this._sadBGL = bgl;

    // reference storage buffer (written by setReference)
    this._refBuffer = device.createBuffer({
      size: gx * gy * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  _buildAccumBuffer() {
    this._accumBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this._readBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  // grid: Float32Array RGBA [0,255], length gx*gy*4
  setReference(grid) {
    this.device.queue.writeBuffer(this._refBuffer, 0, grid);
  }

  // preview: { grid_x, grid_y, nb_colors, qpts, ... }
  // color_data: [{ r, g, b }]  (already converted to RGB)
  async computeLoss(del, palRGB) {
    const { device, gx, gy } = this;
    const { positions, indices, colorIdx } = del.getFlatBuffers();

    const nv = positions.length / 2;
    const nc = palRGB.length;

    // --- upload vertex data ---
    const posBuf = device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(posBuf, 0, positions);

    const idxBuf = device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(idxBuf, 0, indices);

    const cidxBuf = device.createBuffer({
      size: colorIdx.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(cidxBuf, 0, colorIdx);

    const palArr = new Float32Array(nc * 3);
    for (let i = 0; i < nc; ++i) {
      palArr[i*3] = palRGB[i].r; palArr[i*3+1] = palRGB[i].g; palArr[i*3+2] = palRGB[i].b;
    }
    const palBuf = device.createBuffer({
      size: palArr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(palBuf, 0, palArr);

    // --- reset accum ---
    device.queue.writeBuffer(this._accumBuffer, 0, new Uint32Array([0]));

    const renderBG = device.createBindGroup({
      layout: this._renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this._uniBuffer } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: cidxBuf } },
        { binding: 3, resource: { buffer: palBuf } },
      ],
    });

    const sadBG = device.createBindGroup({
      layout: this._sadBGL,
      entries: [
        { binding: 0, resource: this._renderTarget.createView() },
        { binding: 1, resource: { buffer: this._refBuffer } },
        { binding: 2, resource: { buffer: this._accumBuffer } },
      ],
    });

    const enc = device.createCommandEncoder();

    // render pass
    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: this._renderTarget.createView(),
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    rp.setPipeline(this._renderPipeline);
    rp.setBindGroup(0, renderBG);
    rp.setIndexBuffer(idxBuf, 'uint32');
    rp.drawIndexed(indices.length);
    rp.end();

    // compute pass
    const cp = enc.beginComputePass();
    cp.setPipeline(this._sadPipeline);
    cp.setBindGroup(0, sadBG);
    cp.dispatchWorkgroups(Math.ceil(gx / 8), Math.ceil(gy / 8));
    cp.end();

    enc.copyBufferToBuffer(this._accumBuffer, 0, this._readBuffer, 0, 4);
    device.queue.submit([enc.finish()]);

    // cleanup temp buffers after submit
    device.queue.onSubmittedWorkDone().then(() => {
      posBuf.destroy(); idxBuf.destroy(); cidxBuf.destroy(); palBuf.destroy();
    });

    await this._readBuffer.mapAsync(GPUMapMode.READ);
    const raw = new Uint32Array(this._readBuffer.getMappedRange())[0];
    this._readBuffer.unmap();
    return raw / 65536 / (gx * gy);   // normalized [0,1]
  }

  destroy() {
    this._renderTarget.destroy();
    this._uniBuffer.destroy();
    this._refBuffer.destroy();
    this._accumBuffer.destroy();
    this._readBuffer.destroy();
  }
}

// ---------------------------------------------------------------------------

async function createWebGPUDevice() {
  if (!navigator.gpu) throw new Error('WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  return adapter.requestDevice();
}

// ---------------------------------------------------------------------------

if (typeof module !== 'undefined') {
  module.exports = { TriangleGPU, createWebGPUDevice };
}
