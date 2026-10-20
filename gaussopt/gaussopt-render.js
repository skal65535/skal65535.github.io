// gaussopt-render.js — Pure draw layer for Gauss-Opt:
//   - Free-view & COLMAP-pose splat renderer (shapes + diagnostic heatmaps)
//   - 3D wireframe overlays (Loose Octree boxes, COLMAP camera frustums, WLBG split vectors)
//   - Split-screen Target / Render / Residual comparison
//   - Side-canvas rolling PSNR & Octree/WLBG/MC/SA telemetry plot
"use strict";

let _shaderCache = null;

function splitWgslSource(src) {
  const out = {};
  const re = /\/\/\s*===\s*([a-zA-Z0-9_]+)\s*===\s*\n/g;
  const parts = src.split(re);
  for (let i = 1; i < parts.length; i += 2) out[parts[i]] = parts[i + 1];
  return out;
}

export async function loadShaders(device) {
  if (_shaderCache && _shaderCache.device === device) return _shaderCache.mods;
  const url = new URL('./gaussopt-shaders.wgsl', import.meta.url);
  const src = await (await fetch(url)).text();
  const map = splitWgslSource(src);
  const mods = {};
  for (const [name, code] of Object.entries(map)) {
    mods[name] = device.createShaderModule({ code, label: name });
  }
  _shaderCache = { device, mods };
  return mods;
}

// Build OpenCV (+Z forward, +Y down, +X right) view matrix for the Free-View Orbit Camera
export function buildOrbitViewMatrix(target, radius, thetaDeg, phiDeg, up = [0, 1, 0]) {
  const theta = thetaDeg * Math.PI / 180;
  const phi   = phiDeg   * Math.PI / 180;
  // Camera position in world space
  const cx = target[0] + radius * Math.cos(theta) * Math.cos(phi);
  const cy = target[1] - radius * Math.sin(phi);
  const cz = target[2] + radius * Math.sin(theta) * Math.cos(phi);

  // Forward (+Z in camera space) points from camera to target
  let fx = target[0] - cx, fy = target[1] - cy, fz = target[2] - cz;
  const fl = Math.hypot(fx, fy, fz) || 1;
  fx /= fl; fy /= fl; fz /= fl;

  // Right (+X in camera space) = cross(forward, world_up)
  let rx = fy * up[2] - fz * up[1];
  let ry = fz * up[0] - fx * up[2];
  let rz = fx * up[1] - fy * up[0];
  const rl = Math.hypot(rx, ry, rz) || 1;
  rx /= rl; ry /= rl; rz /= rl;

  // Down (+Y in OpenCV camera space) = cross(forward, right)
  const dx = fy * rz - fz * ry;
  const dy = fz * rx - fx * rz;
  const dz = fx * ry - fy * rx;

  const tx = -(rx * cx + ry * cy + rz * cz);
  const ty = -(dx * cx + dy * cy + dz * cz);
  const tz = -(fx * cx + fy * cy + fz * cz);

  const view = new Float32Array([
    rx, dx, fx, 0,
    ry, dy, fy, 0,
    rz, dz, fz, 0,
    tx, ty, tz, 1,
  ]);
  return { view, center: [cx, cy, cz] };
}

export class GaussOptRenderer {
  static async create(device, context, format) {
    const r = new GaussOptRenderer();
    await r._init(device, context, format);
    return r;
  }

  async _init(device, context, format) {
    this.device  = device;
    this.context = context;
    this.format  = format;
    const mods   = await loadShaders(device);

    const blendPremul = {
      color: { srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add' },
    };

    this.pipeSplats = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.splats_render, entryPoint: 'vs' },
      fragment: { module: mods.splats_render, entryPoint: 'fs', targets: [{ format, blend: blendPremul }] },
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
    });

    this.pipeWire = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.wireframe_render, entryPoint: 'vs' },
      fragment: { module: mods.wireframe_render, entryPoint: 'fs', targets: [{ format, blend: blendPremul }] },
      primitive: { topology: 'line-list' },
    });

    this.pipeBlit = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module: mods.compare_blit, entryPoint: 'vs' },
      fragment: { module: mods.compare_blit, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-strip' },
    });

    this.renderUniformBuf = device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'render_uniforms',
    });
    this.wireUniformBuf = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'wire_uniforms',
    });
    this.blitUniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'blit_uniforms',
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    this.wireBuf      = null;
    this.wireCapacity = 0;
    this.wireCount    = 0;
  }

  // Rebuild 3D wireframe geometry buffer for Octree leaves + COLMAP camera frustums + WLBG split lines
  updateWireframes({ octree, images, activeCamIdx = -1, showOctree = true, showLoose = false, showCameras = true, splitVectors = null }) {
    const chunks = [];
    let totalVerts = 0;

    if (showOctree && octree && octree.leaves.length > 0) {
      const { buffer, vertexCount } = octree.buildWireframeVertices({ showLoose });
      chunks.push(buffer);
      totalVerts += vertexCount;
    }

    if (showCameras && images && images.length > 0) {
      const scale = (octree ? octree.sceneRadius : 2.0) * 0.06;
      const camVerts = new Float32Array(images.length * 16 * 4);
      const camU32   = new Uint32Array(camVerts.buffer);
      let ptr = 0;

      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const { R, center, camera } = img;
        const w = scale * (camera.width / camera.fx) * 0.5;
        const h = scale * (camera.height / camera.fy) * 0.5;
        const z = scale;

        const camToWorld = (x, y, zc) => [
          center[0] + R[0] * x + R[3] * y + R[6] * zc,
          center[1] + R[1] * x + R[4] * y + R[7] * zc,
          center[2] + R[2] * x + R[5] * y + R[8] * zc,
        ];
        const c0 = center;
        const p1 = camToWorld(-w, -h, z);
        const p2 = camToWorld( w, -h, z);
        const p3 = camToWorld( w,  h, z);
        const p4 = camToWorld(-w,  h, z);

        let col = img.isValidation ? 0xffffff00 : 0xff22ccff; // ABGR: cyan for val, yellow for train
        if (i === activeCamIdx) col = 0xff22ff22;             // bright green for active camera

        const lines = [
          [c0, p1], [c0, p2], [c0, p3], [c0, p4],
          [p1, p2], [p2, p3], [p3, p4], [p4, p1],
        ];
        for (const [a, b] of lines) {
          camVerts[ptr + 0] = a[0]; camVerts[ptr + 1] = a[1]; camVerts[ptr + 2] = a[2]; camU32[ptr + 3] = col;
          camVerts[ptr + 4] = b[0]; camVerts[ptr + 5] = b[1]; camVerts[ptr + 6] = b[2]; camU32[ptr + 7] = col;
          ptr += 8;
        }
      }
      chunks.push(camVerts);
      totalVerts += images.length * 16;
    }

    if (splitVectors && splitVectors.length >= 6) {
      const nLines = Math.floor(splitVectors.length / 6);
      const sv = new Float32Array(nLines * 2 * 4);
      const su = new Uint32Array(sv.buffer);
      let ptr = 0;
      for (let i = 0; i < nLines; i++) {
        const o = i * 6;
        sv[ptr + 0] = splitVectors[o + 0]; sv[ptr + 1] = splitVectors[o + 1]; sv[ptr + 2] = splitVectors[o + 2]; su[ptr + 3] = 0xff66ff66;
        sv[ptr + 4] = splitVectors[o + 3]; sv[ptr + 5] = splitVectors[o + 4]; sv[ptr + 6] = splitVectors[o + 5]; su[ptr + 7] = 0xff66ff66;
        ptr += 8;
      }
      chunks.push(sv);
      totalVerts += nLines * 2;
    }

    this.wireCount = totalVerts;
    if (totalVerts === 0) return;

    const totalFloats = totalVerts * 4;
    if (!this.wireBuf || this.wireCapacity < totalFloats) {
      this.wireBuf?.destroy?.();
      this.wireCapacity = Math.max(1024, totalFloats * 2);
      this.wireBuf = this.device.createBuffer({
        size: this.wireCapacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'wireframes',
      });
    }

    const merged = new Float32Array(totalFloats);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }
    this.device.queue.writeBuffer(this.wireBuf, 0, merged.buffer, 0, merged.byteLength);
  }

  renderScene({
    splatsBuf, momentsBuf, sortedIdxBuf, numVisible,
    view, proj, camCenter, sceneRadius = 2.0,
    fx, fy, cx, cy, W, H,
    visuMode = 0, colorMode = 0, haze = 1.0, avgR = 0.01,
    drawWireframes = true,
  }) {
    const uBuf = new ArrayBuffer(256);
    const f32  = new Float32Array(uBuf);
    const u32  = new Uint32Array(uBuf);
    f32.set(view, 0);
    if (proj) f32.set(proj, 16);
    f32[32] = camCenter[0]; f32[33] = camCenter[1]; f32[34] = camCenter[2]; f32[35] = sceneRadius;
    f32[36] = fx; f32[37] = fy; f32[38] = cx; f32[39] = cy;
    f32[40] = W;  f32[41] = H;  f32[42] = numVisible; f32[43] = haze;
    u32[44] = visuMode; u32[45] = colorMode; u32[46] = 1; u32[47] = 0;
    f32[48] = Math.max(1e-5, avgR); f32[49] = 1.0; f32[50] = 0.05; f32[51] = 200.0;
    this.device.queue.writeBuffer(this.renderUniformBuf, 0, uBuf);

    const enc = this.device.createCommandEncoder({ label: 'render_scene' });
    const canvasView = this.context.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: canvasView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1.0 },
      }],
    });

    if (numVisible > 0 && splatsBuf && sortedIdxBuf) {
      const bg = this.device.createBindGroup({
        layout: this.pipeSplats.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.renderUniformBuf } },
          { binding: 1, resource: { buffer: splatsBuf } },
          { binding: 2, resource: { buffer: momentsBuf } },
          { binding: 3, resource: { buffer: sortedIdxBuf } },
        ],
      });
      pass.setPipeline(this.pipeSplats);
      pass.setBindGroup(0, bg);
      pass.draw(4, numVisible);
    }

    if (drawWireframes && this.wireCount > 0 && this.wireBuf) {
      const wBuf = new ArrayBuffer(96);
      const wf   = new Float32Array(wBuf);
      wf.set(view, 0);
      wf[16] = fx; wf[17] = fy; wf[18] = cx; wf[19] = cy;
      wf[20] = W;  wf[21] = H;  wf[22] = 0.05; wf[23] = 200.0;
      this.device.queue.writeBuffer(this.wireUniformBuf, 0, wBuf);

      const wbg = this.device.createBindGroup({
        layout: this.pipeWire.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.wireUniformBuf } },
          { binding: 1, resource: { buffer: this.wireBuf } },
        ],
      });
      pass.setPipeline(this.pipeWire);
      pass.setBindGroup(0, wbg);
      pass.draw(this.wireCount);
    }

    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  renderCompare({ gtTex, hatTex, mode = 0, splitX = 0.5, amp = 3.0 }) {
    if (!gtTex || !hatTex) return;
    const b = new ArrayBuffer(16);
    new Uint32Array(b)[0] = mode;
    const f = new Float32Array(b);
    f[1] = splitX;
    f[2] = amp;
    this.device.queue.writeBuffer(this.blitUniformBuf, 0, b);

    const bg = this.device.createBindGroup({
      layout: this.pipeBlit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.blitUniformBuf } },
        { binding: 1, resource: gtTex.createView() },
        { binding: 2, resource: hatTex.createView() },
        { binding: 3, resource: this.sampler },
      ],
    });
    const enc = this.device.createCommandEncoder({ label: 'render_compare' });
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipeBlit);
    pass.setBindGroup(0, bg);
    pass.draw(4);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}

// Render live telemetry & WLBG/Octree/MC/SA statistics on `#side-canvas`
export function drawTelemetryPanel(canvas, history, octree, statsText = '') {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = '#121318';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = '#2a2d3a';
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  ctx.font = '12px monospace';
  ctx.fillStyle = '#e0e6f0';
  const leaves = octree ? octree.leaves.length : 0;
  const s = octree ? octree.lastStats : { split: 0, cloned: 0, pruned: 0, floaters: 0, merged: 0, teleported: 0 };
  const teleStr = s.teleported ? `, ↻${s.teleported} mcmc` : '';
  ctx.fillText(
    `Octree Leaves: ${leaves} | Rebalance: +${s.split} split, +${s.cloned} clone, -${s.pruned} prune, -${s.floaters} floaters, -${s.merged} merge${teleStr}`,
    12, 18
  );
  if (statsText) {
    ctx.fillStyle = '#8ef';
    ctx.fillText(statsText, 12, 36);
  }

  if (!history || history.length < 2) return;

  const padL = 44, padR = 16, padT = 46, padB = 20;
  const pw = W - padL - padR;
  const ph = H - padT - padB;

  let minPsnr = 10, maxPsnr = 35;
  for (const h of history) {
    if (h.psnr < minPsnr) minPsnr = Math.floor(h.psnr - 1);
    if (h.psnr > maxPsnr) maxPsnr = Math.ceil(h.psnr + 1);
  }

  ctx.strokeStyle = '#252834';
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padT + (ph * i) / 4;
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    const val = (maxPsnr - ((maxPsnr - minPsnr) * i) / 4).toFixed(1);
    ctx.fillStyle = '#889';
    ctx.fillText(`${val}dB`, 4, y + 4);
  }
  ctx.stroke();

  // Plot Train PSNR curve
  ctx.strokeStyle = '#4af';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = padL + (i / (history.length - 1)) * pw;
    const t = (history[i].psnr - minPsnr) / Math.max(1, maxPsnr - minPsnr);
    const y = padT + ph * (1 - Math.min(1, Math.max(0, t)));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
