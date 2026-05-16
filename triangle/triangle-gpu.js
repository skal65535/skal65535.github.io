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
@group(0) @binding(3) var<storage, read> palette   : array<f32>;   // [nc*4] r,g,b,a in [0,255]

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       col : vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  let vx = positions[vi * 2u];
  let vy = positions[vi * 2u + 1u];
  let ndcx =  vx * uni.inv_gx * 2.0 - 1.0;
  let ndcy = -(vy * uni.inv_gy * 2.0 - 1.0);
  let ci = colorIdx[vi] * 4u;
  var o : VSOut;
  o.pos = vec4<f32>(ndcx, ndcy, 0.0, 1.0);
  o.col = vec4<f32>(palette[ci], palette[ci+1u], palette[ci+2u], palette[ci+3u]);
  return o;
}

@fragment
fn fs_main(@location(0) col : vec4<f32>) -> @location(0) vec4<f32> {
  return col / 255.0;
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
  let ref_a = reference[base + 3u] / 255.0;
  let dr = pixel.r - reference[base]      / 255.0;
  let dg = pixel.g - reference[base + 1u] / 255.0;
  let db = pixel.b - reference[base + 2u] / 255.0;
  let da = pixel.a - ref_a;
  let sad = ref_a * (0.3 * abs(dr) + 0.6 * abs(dg) + 0.1 * abs(db))
          + abs(da);
  let fixed = u32(sad * 65536.0);
  atomicAdd(&accum, fixed);
}
`;

// ---------------------------------------------------------------------------

class TriangleGPU {
  constructor(device, gx, gy, zoom = 1) {
    this.device = device;
    this.gx = gx;
    this.gy = gy;
    this.zoom = zoom;
    this._ref = null;   // GPUBuffer for reference grid
    this._buildRenderPipeline();
    this._buildSADPipeline();
    this._buildAccumBuffer();
  }

  _buildRenderPipeline() {
    const { device, gx, gy } = this;
    const mod = device.createShaderModule({ code: RENDER_WGSL });

    this._renderTarget = device.createTexture({
      size: [gx * this.zoom, gy * this.zoom],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
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

    // reference storage buffer (written by setReference), sized for zoomed resolution
    this._refBuffer = device.createBuffer({
      size: gx * this.zoom * gy * this.zoom * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  _buildAccumBuffer() {
    this._accumBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
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

    const palArr = new Float32Array(nc * 4);
    for (let i = 0; i < nc; ++i) {
      palArr[i*4]   = palRGB[i].r;
      palArr[i*4+1] = palRGB[i].g;
      palArr[i*4+2] = palRGB[i].b;
      palArr[i*4+3] = palRGB[i].a;
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

    try {
      await this._readBuffer.mapAsync(GPUMapMode.READ);
    } catch (e) {
      if (this._destroyed) return 0;
      throw e;
    }
    const raw = new Uint32Array(this._readBuffer.getMappedRange())[0];
    this._readBuffer.unmap();
    return raw / 65536 / (gx * gy);  // normalized per grid-point (zoom provides accuracy, not scale)
  }

  destroy() {
    this._destroyed = true;
    this._renderTarget.destroy();
    this._uniBuffer.destroy();
    this._refBuffer.destroy();
    this._accumBuffer.destroy();
    this._readBuffer.destroy();
  }
}

// ---------------------------------------------------------------------------
// Phase 5: full-GPU hill-climbing optimizer.
// Single workgroup of WG_SIZE threads; runs N_ITERS sequential greedy steps
// with parallel SAD reduction per step.  Only handles mutations that leave the
// Delaunay topology unchanged (vertex-move, color-index-move, color-move,
// flip-alpha).  Topology mutations (add/remove vertex/color) are still done
// on the CPU between GPU batches.

const OPT_WG_SIZE = 256;

const OPT_WGSL = /* wgsl */`
const WG_SIZE : u32 = ${OPT_WG_SIZE}u;

struct OptUni {
  gx : u32, gy : u32, nv : u32, nt : u32,
  nc : u32, n_iters : u32, iter_offset : u32, max_iters : u32,
  p_vm : u32, p_ci : u32, p_cm : u32, p_fa : u32,
  tol_fp : u32,            // score_tolerance * 65536
  has_alpha : u32,
  vm_amp : u32,            // vertex move amplitude (max delta per axis)
  vm_border_escape : u32,  // probability (0-100) of leaving the boundary
  zoom : u32,              // reference/render zoom (1,2,4,8): gx*zoom x gy*zoom pixels
}

@group(0) @binding(0) var<uniform>             uni  : OptUni;
@group(0) @binding(1) var<storage,read_write>  pos  : array<i32>;  // [nv*2] int coords
@group(0) @binding(2) var<storage,read_write>  cidx : array<u32>;  // [nv]
@group(0) @binding(3) var<storage,read_write>  pal  : array<u32>;  // [nc*4] y,co,cg,a 0..63
@group(0) @binding(4) var<storage,read>        tri  : array<u32>;  // [nt*3]
@group(0) @binding(5) var<storage,read>        ref_ : array<f32>;  // [gx*gy*4] RGBA 0..255
@group(0) @binding(6) var<storage,read_write>  score_buf : array<f32>; // [1]
@group(0) @binding(7) var<storage,read_write>  rng_buf   : array<u32>; // [1]

var<workgroup> wg_partial  : array<u32, WG_SIZE>;
var<workgroup> wg_sad_cur  : u32;
var<workgroup> wg_rng      : u32;
var<workgroup> wg_mut_type : u32;  // 0=vm 1=ci 2=cm 3=fa 0xFFFF=noop
var<workgroup> wg_mut_idx  : u32;  // index into pos/cidx/pal
var<workgroup> wg_mut_old  : i32;  // saved value for revert

fn xor32(s: u32) -> u32 {
  var x = s;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  return x;
}

fn ycog_to_rgb(y: u32, co: u32, cg: u32) -> vec3f {
  let yf  = f32(y)  * (255.0 / 63.0);
  let cgf = f32(cg) * (255.0 / 63.0) - 128.0;
  let cof = f32(co) * (255.0 / 63.0) - 128.0;
  let d   = yf - cgf;
  return clamp(vec3f(d + cof, yf + cgf, d - cof), vec3f(0.0), vec3f(255.0));
}

fn pixel_sad(px: u32, py: u32) -> u32 {
  // px,py are in zoomed pixel space; convert to grid-space for triangle coverage test
  let pfx  = (f32(px) + 0.5) / f32(uni.zoom);
  let pfy  = (f32(py) + 0.5) / f32(uni.zoom);
  let base = (py * uni.gx * uni.zoom + px) * 4u;
  var col  = vec4f(0.0);
  for (var t = 0u; t < uni.nt; t += 1u) {
    let i0 = tri[t*3u]; let i1 = tri[t*3u+1u]; let i2 = tri[t*3u+2u];
    let ax = f32(pos[i0*2u]); let ay = f32(pos[i0*2u+1u]);
    let bx = f32(pos[i1*2u]); let by = f32(pos[i1*2u+1u]);
    let cx = f32(pos[i2*2u]); let cy = f32(pos[i2*2u+1u]);
    let denom = (by-cy)*(ax-cx) + (cx-bx)*(ay-cy);
    if (abs(denom) < 0.001) { continue; }
    let inv = 1.0 / denom;
    let w0 = ((by-cy)*(pfx-cx) + (cx-bx)*(pfy-cy)) * inv;
    let w1 = ((cy-ay)*(pfx-cx) + (ax-cx)*(pfy-cy)) * inv;
    let w2 = 1.0 - w0 - w1;
    if (w0 < -0.001 || w1 < -0.001 || w2 < -0.001) { continue; }
    let c0 = cidx[i0]*4u; let c1 = cidx[i1]*4u; let c2 = cidx[i2]*4u;
    let r0 = ycog_to_rgb(pal[c0],pal[c0+1u],pal[c0+2u]);
    let r1 = ycog_to_rgb(pal[c1],pal[c1+1u],pal[c1+2u]);
    let r2 = ycog_to_rgb(pal[c2],pal[c2+1u],pal[c2+2u]);
    let a  = w0*f32(pal[c0+3u]) + w1*f32(pal[c1+3u]) + w2*f32(pal[c2+3u]);
    col = vec4f(w0*r0 + w1*r1 + w2*r2, a * 255.0);
    break;
  }
  let ra = ref_[base+3u] / 255.0;
  let dr = (col.r - ref_[base])     / 255.0;
  let dg = (col.g - ref_[base+1u])  / 255.0;
  let db = (col.b - ref_[base+2u])  / 255.0;
  let da = (col.a - ref_[base+3u])  / 255.0;
  let s  = ra * (0.3*abs(dr) + 0.6*abs(dg) + 0.1*abs(db)) + abs(da);
  return u32(s * 65536.0);
}

fn reduce_sad(tid: u32) -> u32 {
  let zx   = uni.gx * uni.zoom;
  let npix = zx * uni.gy * uni.zoom;
  let ppt  = (npix + WG_SIZE - 1u) / WG_SIZE;
  let st   = tid * ppt;
  let en   = min(st + ppt, npix);
  var acc  = 0u;
  for (var p = st; p < en; p += 1u) {
    acc += pixel_sad(p % zx, p / zx);
  }
  wg_partial[tid] = acc;
  workgroupBarrier();
  for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
    if (tid < s) { wg_partial[tid] += wg_partial[tid + s]; }
    workgroupBarrier();
  }
  return wg_partial[0];
}

fn propose() {
  let r = wg_rng % 100u;
  wg_rng = xor32(wg_rng);
  var acc = 0u;

  // vertex move
  acc += uni.p_vm;
  if (r < acc && uni.nv > 4u) {
    let vi  = (wg_rng % (uni.nv - 4u)) + 4u;           wg_rng = xor32(wg_rng);
    let mag = i32(wg_rng % uni.vm_amp) + 1;             wg_rng = xor32(wg_rng);
    let d   = select(-mag, mag, (wg_rng & 1u) == 1u);   wg_rng = xor32(wg_rng);
    let px  = pos[vi * 2u];
    let py  = pos[vi * 2u + 1u];
    let GX  = i32(uni.gx); let GY = i32(uni.gy);
    let onL = px == 0; let onR = px == GX-1;
    let onT = py == 0; let onB = py == GY-1;
    let onBorder = onL || onR || onT || onB;
    var axis : u32; var lo : i32; var hi : i32;
    let esc = wg_rng % 100u; wg_rng = xor32(wg_rng);
    if (onBorder && esc >= uni.vm_border_escape) {
      // Slide along the edge (avoid corners: clamp inner coord to [1, dim-2]).
      axis = select(0u, 1u, onL || onR);
      lo = 1; hi = select(GX-2, GY-2, axis == 1u);
    } else {
      // Free move; allow reaching boundary.
      axis = wg_rng & 1u; wg_rng = xor32(wg_rng);
      lo = 0; hi = select(GX-1, GY-1, axis == 1u);
    }
    let idx    = vi * 2u + axis;
    let old    = pos[idx];
    let newval = clamp(old + d, lo, hi);
    // Reject landing on a corner (already occupied by a fixed vertex).
    let new_px = select(px, newval, axis == 0u);
    let new_py = select(py, newval, axis == 1u);
    let isCorner = (new_px == 0 || new_px == GX-1) && (new_py == 0 || new_py == GY-1);
    if (!isCorner) {
      wg_mut_type = 0u; wg_mut_idx = idx; wg_mut_old = old;
      pos[idx] = newval;
    } else {
      wg_mut_type = 0xFFFFu;
    }
    return;
  }

  // color index move
  acc += uni.p_ci;
  if (r < acc) {
    let vi  = wg_rng % uni.nv; wg_rng = xor32(wg_rng);
    let ni  = wg_rng % uni.nc; wg_rng = xor32(wg_rng);
    let old = cidx[vi];
    wg_mut_type = 1u; wg_mut_idx = vi; wg_mut_old = i32(old);
    cidx[vi] = ni;
    return;
  }

  // color move (y/co/cg channel)
  acc += uni.p_cm;
  if (r < acc) {
    let ci = wg_rng % uni.nc;  wg_rng = xor32(wg_rng);
    let ch = wg_rng % 3u;      wg_rng = xor32(wg_rng);
    let d  = select(-1, 1, (wg_rng & 1u) == 1u); wg_rng = xor32(wg_rng);
    let bi = ci * 4u + ch;
    let old = i32(pal[bi]);
    wg_mut_type = 2u; wg_mut_idx = bi; wg_mut_old = old;
    pal[bi] = u32(clamp(old + d, 0, 63));
    return;
  }

  // flip alpha
  if (uni.has_alpha != 0u) {
    let ci  = wg_rng % uni.nc; wg_rng = xor32(wg_rng);
    let bi  = ci * 4u + 3u;
    let old = i32(pal[bi]);
    wg_mut_type = 3u; wg_mut_idx = bi; wg_mut_old = old;
    pal[bi] = 1u - pal[bi];
    return;
  }

  wg_mut_type = 0xFFFFu;  // noop
}

fn revert() {
  if      (wg_mut_type == 0u) { pos[wg_mut_idx]  = wg_mut_old; }
  else if (wg_mut_type == 1u) { cidx[wg_mut_idx] = u32(wg_mut_old); }
  else if (wg_mut_type <= 3u) { pal[wg_mut_idx]  = u32(wg_mut_old); }
}

@compute @workgroup_size(WG_SIZE)
fn opt_main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let tid = lid.x;
  if (tid == 0u) { wg_rng = rng_buf[0]; }
  workgroupBarrier();

  let init_sad = reduce_sad(tid);
  if (tid == 0u) { wg_sad_cur = init_sad; }
  workgroupBarrier();

  for (var iter = 0u; iter < uni.n_iters; iter += 1u) {
    if (tid == 0u) { propose(); }
    storageBarrier();
    workgroupBarrier();

    let new_sad = reduce_sad(tid);

    if (tid == 0u) {
      let g_iter = iter + uni.iter_offset;
      let rem    = select(0u, uni.max_iters - g_iter, g_iter < uni.max_iters);
      let tol    = u32(f32(uni.tol_fp) * f32(rem) / f32(uni.max_iters)
                       * f32(uni.gx * uni.gy));
      if (new_sad <= wg_sad_cur + tol) {
        wg_sad_cur = new_sad;
      } else {
        revert();
      }
    }
    storageBarrier();
    workgroupBarrier();
  }

  if (tid == 0u) {
    score_buf[0] = f32(wg_sad_cur) / 65536.0 / f32(uni.gx * uni.gy);
    rng_buf[0]   = wg_rng;
  }
}
`;

// ---------------------------------------------------------------------------

class TriangleOptGPU {
  constructor(device, gx, gy, zoom = 1) {
    this.device = device;
    this.gx = gx;
    this.gy = gy;
    this.zoom = zoom;
    this._nv = 0;
    this._nt = 0;
    this._nc = 0;
    this._destroyed = false;
    this._buildPipeline();
    this._buildFixedBuffers();
  }

  _buildPipeline() {
    const { device } = this;
    const mod = device.createShaderModule({ code: OPT_WGSL });

    const bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]});

    this._pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: mod, entryPoint: 'opt_main' },
    });
    this._bgl = bgl;

    this._uniBuffer = device.createBuffer({
      size: 80,  // 20 u32s: 16 original + zoom + 3 padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._scoreBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this._rngBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._scoreReadBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  _buildFixedBuffers() {
    // Placeholder — real buffers allocated in uploadState
    this._posBuf = null; this._cidxBuf = null; this._palBuf = null; this._triBuf = null;
    this._refBuf = null;
    this._posReadBuf = null; this._cidxReadBuf = null; this._palReadBuf = null;
  }

  _allocBuf(size, usage) {
    return this.device.createBuffer({ size: Math.max(size, 4), usage });
  }

  // posI32: Int32Array [nv*2], cidxU32: Uint32Array [nv], palU32: Uint32Array [nc*4],
  // triU32: Uint32Array [nt*3], refF32: Float32Array [gx*gy*4]
  uploadState(posI32, cidxU32, palU32, triU32, refF32) {
    const { device } = this;
    const nv = posI32.length / 2;
    const nc = palU32.length / 4;
    const nt = triU32.length / 3;

    // Reallocate if sizes changed
    if (nv !== this._nv || nc !== this._nc || nt !== this._nt) {
      [this._posBuf, this._cidxBuf, this._palBuf, this._triBuf,
       this._posReadBuf, this._cidxReadBuf, this._palReadBuf,
       this._refBuf].forEach(b => b && b.destroy());

      const STO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
      const RO  = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const RB  = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;

      this._posBuf  = this._allocBuf(nv * 2 * 4, STO);
      this._cidxBuf = this._allocBuf(nv * 4,     STO);
      this._palBuf  = this._allocBuf(nc * 4 * 4, STO);
      this._triBuf  = this._allocBuf(nt * 3 * 4, RO);
      this._refBuf  = this._allocBuf(this.gx * this.zoom * this.gy * this.zoom * 4 * 4, RO);
      this._posReadBuf  = this._allocBuf(nv * 2 * 4, RB);
      this._cidxReadBuf = this._allocBuf(nv * 4,     RB);
      this._palReadBuf  = this._allocBuf(nc * 4 * 4, RB);
      this._nv = nv; this._nc = nc; this._nt = nt;
    }

    device.queue.writeBuffer(this._posBuf,  0, posI32);
    device.queue.writeBuffer(this._cidxBuf, 0, cidxU32);
    device.queue.writeBuffer(this._palBuf,  0, palU32);
    device.queue.writeBuffer(this._triBuf,  0, triU32);
    device.queue.writeBuffer(this._refBuf,  0, refF32);
  }

  // Upload everything except the reference grid (ref unchanged since last uploadState).
  uploadStateNoRef(posI32, cidxU32, palU32, triU32) {
    const { device } = this;
    const nv = posI32.length / 2;
    const nc = palU32.length / 4;
    const nt = triU32.length / 3;
    if (nv !== this._nv || nc !== this._nc) {
      throw new Error('nv/nc changed — use uploadState with ref');
    }
    // nt can change after vertex moves (re-triangulation); reallocate tri buffer if needed.
    if (nt !== this._nt) {
      this._triBuf && this._triBuf.destroy();
      const RO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      this._triBuf = this._allocBuf(nt * 3 * 4, RO);
      this._nt = nt;
    }
    device.queue.writeBuffer(this._posBuf,  0, posI32);
    device.queue.writeBuffer(this._cidxBuf, 0, cidxU32);
    device.queue.writeBuffer(this._palBuf,  0, palU32);
    device.queue.writeBuffer(this._triBuf,  0, triU32);
  }

  // Run nIters GPU hill-climbing steps.  Returns distortion score [0,1].
  async runBatch(nIters, iterOffset, maxIters, opts = {}) {
    const { device, gx, gy, _nv: nv, _nt: nt, _nc: nc } = this;
    const {
      p_vm = 50, p_ci = 25, p_cm = 20, p_fa = 25,
      score_tolerance = 0.0002,
      has_alpha = 0,
      vm_amp = 1,
      vm_border_escape = 5,
      rng_seed,
    } = opts;

    // Scale down GPU iterations by zoom² to keep wall-clock time per dispatch constant.
    const gpuIters = Math.max(1, Math.round(nIters / (this.zoom * this.zoom)));

    // Write uniforms
    const uni = new Uint32Array(20);
    uni[0] = gx; uni[1] = gy; uni[2] = nv; uni[3] = nt;
    uni[4] = nc; uni[5] = gpuIters; uni[6] = iterOffset; uni[7] = maxIters;
    uni[8] = p_vm; uni[9] = p_ci; uni[10] = p_cm; uni[11] = p_fa;
    uni[12] = Math.round(score_tolerance * 65536);
    uni[13] = has_alpha ? 1 : 0;
    uni[14] = Math.max(1, vm_amp | 0);
    uni[15] = Math.max(0, Math.min(100, vm_border_escape | 0));
    uni[16] = Math.max(1, this.zoom | 0);
    device.queue.writeBuffer(this._uniBuffer, 0, uni);

    // Seed RNG once per batch (only on first call or when seed changes)
    if (rng_seed !== undefined) {
      device.queue.writeBuffer(this._rngBuffer, 0, new Uint32Array([rng_seed >>> 0]));
    }

    const bg = device.createBindGroup({
      layout: this._bgl,
      entries: [
        { binding: 0, resource: { buffer: this._uniBuffer } },
        { binding: 1, resource: { buffer: this._posBuf } },
        { binding: 2, resource: { buffer: this._cidxBuf } },
        { binding: 3, resource: { buffer: this._palBuf } },
        { binding: 4, resource: { buffer: this._triBuf } },
        { binding: 5, resource: { buffer: this._refBuf } },
        { binding: 6, resource: { buffer: this._scoreBuffer } },
        { binding: 7, resource: { buffer: this._rngBuffer } },
      ],
    });

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this._pipeline);
    cp.setBindGroup(0, bg);
    cp.dispatchWorkgroups(1);   // single workgroup, all iterations are sequential inside
    cp.end();
    enc.copyBufferToBuffer(this._scoreBuffer, 0, this._scoreReadBuffer, 0, 4);
    device.queue.submit([enc.finish()]);

    try {
      await this._scoreReadBuffer.mapAsync(GPUMapMode.READ);
    } catch (e) {
      if (this._destroyed) return 0;
      throw e;
    }
    const score = new Float32Array(this._scoreReadBuffer.getMappedRange())[0];
    this._scoreReadBuffer.unmap();
    return score;
  }

  // Read back pos/cidx/pal after a batch.
  async readState() {
    const { device, _nv: nv, _nc: nc } = this;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this._posBuf,  0, this._posReadBuf,  0, nv * 2 * 4);
    enc.copyBufferToBuffer(this._cidxBuf, 0, this._cidxReadBuf, 0, nv * 4);
    enc.copyBufferToBuffer(this._palBuf,  0, this._palReadBuf,  0, nc * 4 * 4);
    device.queue.submit([enc.finish()]);

    try {
      await Promise.all([
        this._posReadBuf.mapAsync(GPUMapMode.READ),
        this._cidxReadBuf.mapAsync(GPUMapMode.READ),
        this._palReadBuf.mapAsync(GPUMapMode.READ),
      ]);
    } catch (e) {
      if (this._destroyed) return null;
      throw e;
    }
    const posI32  = new Int32Array(this._posReadBuf.getMappedRange().slice(0));
    const cidxU32 = new Uint32Array(this._cidxReadBuf.getMappedRange().slice(0));
    const palU32  = new Uint32Array(this._palReadBuf.getMappedRange().slice(0));
    this._posReadBuf.unmap();
    this._cidxReadBuf.unmap();
    this._palReadBuf.unmap();
    return { posI32, cidxU32, palU32 };
  }

  destroy() {
    this._destroyed = true;
    [this._uniBuffer, this._scoreBuffer, this._rngBuffer, this._scoreReadBuffer,
     this._posBuf, this._cidxBuf, this._palBuf, this._triBuf, this._refBuf,
     this._posReadBuf, this._cidxReadBuf, this._palReadBuf]
      .forEach(b => b && b.destroy());
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
  module.exports = { TriangleGPU, TriangleOptGPU, createWebGPUDevice };
}
