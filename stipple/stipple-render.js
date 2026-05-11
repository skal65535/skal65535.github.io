// stipple-render.js — pure draw layer. Takes already-prepared state.
//
// GPURenderer holds the points / target / voronoi render pipelines and a
// reusable site_color buffer. All render* methods are pure draws — they read
// from GPU resources passed in by the caller and never trigger compute.
"use strict";

import { loadShaders } from './stipple-shaders.js';

export class GPURenderer {
  static async create(device, context, format) {
    const r = new GPURenderer();
    await r._init(device, context, format);
    return r;
  }

  async _init(device, context, format) {
    this._device  = device;
    this._context = context;
    this._format  = format;
    const mods = await loadShaders(device);

    this._pipe_points = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.points_render, entryPoint: 'vs' },
      fragment: { module: mods.points_render, entryPoint: 'fs',
                  targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
    });
    this._pipe_target = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.target_render, entryPoint: 'vs' },
      fragment: { module: mods.target_render, entryPoint: 'fs',
                  targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
    });
    this._pipe_voro = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.voronoi_render, entryPoint: 'vs' },
      fragment: { module: mods.voronoi_render, entryPoint: 'fs',
                  targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
    });

    this._pp_buf = device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'point_params',
    });
    this._sampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });
    this._vp_buf = device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'voro_params',
    });
    this._color_buf      = null;
    this._color_capacity = 0;
  }

  _bgColor(invert) { return invert ? { r:0,g:0,b:0,a:1 } : { r:1,g:1,b:1,a:1 }; }

  renderClear({ invert }) {
    const enc  = this._device.createCommandEncoder({ label: 'render_clear' });
    const view = this._context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: this._bgColor(invert),
      }],
    });
    pass.end();
    this._device.queue.submit([enc.finish()]);
  }

  renderPoints({ sitesBuf, N, canvas, radius, invert }) {
    if (!sitesBuf || N <= 0) { this.renderClear({ invert }); return; }
    const device = this._device;
    const pp = new ArrayBuffer(32);
    const f  = new Float32Array(pp);
    f[0] = radius;
    f[1] = canvas.width;
    f[2] = canvas.height;
    f[3] = invert ? 1 : 0;
    device.queue.writeBuffer(this._pp_buf, 0, pp);

    const bg = device.createBindGroup({
      layout: this._pipe_points.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._pp_buf } },
        { binding: 1, resource: { buffer: sitesBuf } },
      ],
    });
    const enc  = device.createCommandEncoder({ label: 'render_points' });
    const view = this._context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: this._bgColor(invert),
      }],
    });
    pass.setPipeline(this._pipe_points);
    pass.setBindGroup(0, bg);
    pass.draw(4, N);  // 4 verts per point, N instances
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  renderTarget({ grayTex }) {
    const device = this._device;
    const bg = device.createBindGroup({
      layout: this._pipe_target.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: grayTex.createView() },
        { binding: 1, resource: this._sampler },
      ],
    });
    const enc  = device.createCommandEncoder({ label: 'render_target' });
    const view = this._context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: { r:0, g:0, b:0, a:1 },
      }],
    });
    pass.setPipeline(this._pipe_target);
    pass.setBindGroup(0, bg);
    pass.draw(4);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  renderVoronoi({ siteIdsTex, points, scale, W, H }) {
    const device = this._device;
    if (this._color_capacity < points.length) {
      this._color_buf?.destroy?.();
      this._color_buf = device.createBuffer({
        size: Math.max(1024, points.length * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'site_color',
      });
      this._color_capacity = points.length;
    }
    const colors = new Float32Array(points.length);
    for (let i = 0; i < points.length; i++) colors[i] = (points[i].c ?? 0) / 255;
    device.queue.writeBuffer(this._color_buf, 0, colors.buffer, 0, colors.byteLength);

    const vp = new ArrayBuffer(16);
    const u  = new Uint32Array(vp); const f = new Float32Array(vp);
    u[0] = W; u[1] = H; f[2] = scale;
    device.queue.writeBuffer(this._vp_buf, 0, vp);

    const bg = device.createBindGroup({
      layout: this._pipe_voro.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: siteIdsTex.createView() },
        { binding: 1, resource: { buffer: this._color_buf } },
        { binding: 2, resource: { buffer: this._vp_buf } },
      ],
    });
    const enc  = device.createCommandEncoder({ label: 'render_voronoi' });
    const view = this._context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view, loadOp: 'clear', storeOp: 'store',
        clearValue: { r:0, g:0, b:0, a:1 },
      }],
    });
    pass.setPipeline(this._pipe_voro);
    pass.setBindGroup(0, bg);
    pass.draw(4);
    pass.end();
    device.queue.submit([enc.finish()]);
  }
}
