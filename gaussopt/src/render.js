// render.js — Pure draw layer for Gauss-Opt:
//   - Free-view & COLMAP-pose splat renderer (shapes + diagnostic heatmaps)
//   - 3D wireframe overlays (Loose Octree boxes, COLMAP camera frustums, WLBG split vectors)
//   - Split-screen Target / Render / Residual comparison
//   - Side-canvas rolling PSNR & Octree/WLBG/MC/SA telemetry plot
"use strict";

import { clamp, sub, madd, normalize, viewMatToQuat, quatSlerp, viewFromQuat, viewTransform, fitIntrinsics } from './math.js';
import { bindGroup } from './gpu.js';

let _shaderCache = null;

// `// === <name> ===` sections -> { name: code }, the `_*` ones (shared code) inserted at each
// `// #include _<name>` line and left out
function splitWgslSource(src) {
  const sections = {}, out = {};
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

// Fly-to animation: cubic Bezier for the camera center (bowed outward from `sceneCenter`
// and slightly towards -Y, so it arcs instead of cutting through the scene), slerp for the
// orientation, linear blend of the intrinsics.
export function createCameraSplineAnimation({
  fromView, fromCenter, fromFx, fromFy, fromCx, fromCy,
  toImg, W, H, sceneCenter = [0, 0, 0], durationMs = 680,
}) {
  const p0 = [...fromCenter], p3 = [...toImg.center];
  const d = sub(p3, p0);
  const chord = Math.hypot(...d);
  const out = normalize(sub(madd(p0, d, 0.5), sceneCenter));
  const ctrl = (a) => {
    const p = madd(madd(p0, d, a), out, 0.18 * chord);
    p[1] -= 0.04 * chord;
    return p;
  };
  const { fx: toFx, fy: toFy, cx: toCx, cy: toCy } = fitIntrinsics(toImg.camera, W, H);
  return {
    startTime: performance.now(), durationMs,
    p0, p1: ctrl(0.33), p2: ctrl(0.67), p3,
    q0: viewMatToQuat(fromView), q1: viewMatToQuat(toImg.view),
    fromFx, fromFy, fromCx, fromCy, toFx, toFy, toCx, toCy,
  };
}

export function evaluateCameraSpline(anim, nowMs = performance.now()) {
  const t = clamp((nowMs - anim.startTime) / anim.durationMs, 0, 1);
  const s = t * t * t * (t * (t * 6 - 15) + 10);  // quintic smoothstep (C2 easing)
  const u = 1 - s;
  const b = [u * u * u, 3 * u * u * s, 3 * u * s * s, s * s * s];
  const camCenter = [0, 1, 2].map(k => b[0] * anim.p0[k] + b[1] * anim.p1[k] + b[2] * anim.p2[k] + b[3] * anim.p3[k]);
  const lerp = (a, z) => a + (z - a) * s;
  return {
    done: t >= 1,
    view: viewFromQuat(quatSlerp(anim.q0, anim.q1, s), camCenter),
    camCenter,
    fx: lerp(anim.fromFx, anim.toFx), fy: lerp(anim.fromFy, anim.toFy),
    cx: lerp(anim.fromCx, anim.toCx), cy: lerp(anim.fromCy, anim.toCy),
  };
}

// ── GPU renderer ─────────────────────────────────────────────────────────────

const ZNEAR = 0.05, ZFAR = 200.0;  // display clip range
// Octree boxes: full alpha up to 0.25 scene radii of depth, fading linearly to 2 radii, never below 0.2
const OCTREE_FADE = [0.25, 2.0, 0.2];
const blend = (srcFactor, dstFactor) => {
  const c = { srcFactor, dstFactor, operation: 'add' };
  return { color: c, alpha: c };
};
const BLEND_UNDER = blend('one-minus-dst-alpha', 'one');  // front-to-back splats
const BLEND_OVER  = blend('one', 'one-minus-src-alpha');  // premultiplied overlays

// Camera frustum drawn as a pyramid: points [apex, p1..p4] (image-plane corners, clockwise)
const FRUSTUM_EDGES = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]];
const FRUSTUM_TRIS  = [[1, 2, 3], [1, 3, 4], [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]];
const COL_TRAIN = 0xff22ccff, COL_VAL = 0xffffff00, COL_ACTIVE = 0xff22ff22, COL_SPLIT = 0xff66ff66;  // ABGR
const COL_PATH = 0xffff55ff;  // camera path: magenta

function pointInTri(u, v, a, b, c) {
  const d1 = (u - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (v - b[1]);
  const d2 = (u - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (v - c[1]);
  const d3 = (u - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (v - a[1]);
  return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
}

function distToSeg(u, v, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy;
  const t = (l2 < 1e-4) ? 0 : clamp(((u - a[0]) * dx + (v - a[1]) * dy) / l2, 0, 1);
  return Math.hypot(u - (a[0] + t * dx), v - (a[1] + t * dy));
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
    const mods   = await loadShaders(device);
    const pipe = (module, topology, blend) => device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [blend ? { format, blend } : { format }] },
      primitive: { topology, cullMode: 'none' },
    });
    this.pipeSplats     = pipe(mods.splats_render,    'triangle-strip', BLEND_UNDER);
    this.pipeWire       = pipe(mods.wireframe_render, 'line-list',      BLEND_OVER);
    this.pipeFrustumImg = pipe(mods.frustum_image,    'triangle-strip', BLEND_OVER);
    this.pipeBlit       = pipe(mods.compare_blit,     'triangle-strip', null);

    const ubuf = (size, label) => device.createBuffer({ size, label, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.renderUniformBuf = ubuf(128, 'render_uniforms');
    this.wireUniformBuf   = ubuf(112, 'wire_uniforms');
    this.blitUniformBuf   = ubuf(16,  'blit_uniforms');
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    this.wireBuf           = null;  // line vertices: xyz + bitcast ABGR color
    this.wireCount         = 0;
    this.octreeVertCount   = 0;     // the first wireBuf vertices are octree boxes (faded with depth)
    this.frustumQuadBuf    = null;  // per camera: 4 strip corners (xyz + alpha)
    this.frustumGeoms      = [];    // per camera: { camIdx, pts: [apex, p1..p4] } for picking
    this.frustumBindGroups = new WeakMap();  // texture -> bind group (reset when frustumQuadBuf moves)
  }

  // Bind group 0 of `pipeline`, binding i = resources[i] (GPUBuffers wrapped as { buffer })
  _bindGroup(pipeline, ...resources) { return bindGroup(this.device, pipeline, resources); }

  // Grow-only storage buffer this[key] of at least nFloats floats; true if (re)allocated
  _ensureStorage(key, nFloats, minFloats, label) {
    if (this[key] && this[key].size >= nFloats * 4) return false;
    this[key]?.destroy?.();
    this[key] = this.device.createBuffer({
      size: Math.max(minFloats, nFloats * 2) * 4, label,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    return true;
  }

  // Rebuild the line buffer (octree leaves + camera frustums + camera path + WLBG split vectors)
  // and frustum quads. splitVectors: flat [x0, y0, z0, x1, y1, z1, ...] segments.
  // The camera path joins the camera centers in image-name order (= capture order for video frames).
  updateWireframes({
    octree, images, activeCamIdx = -1,
    showOctree = true, showLoose = false, showCameras = true, showPath = false,
    frustumScale = 1.0, splitVectors = null,
  }) {
    const octVerts = (showOctree && octree && octree.leaves.length > 0) ? octree.buildWireframeVertices({ showLoose }) : null;
    const numCams  = (showCameras && images) ? images.length : 0;
    const numSplit = splitVectors ? Math.floor(splitVectors.length / 6) : 0;
    const path = (showPath && images?.length > 1) ? images.map(im => im.center) : [];  // images are name-sorted
    const numPath = Math.max(0, path.length - 1);
    const totalVerts = (octVerts ? octVerts.vertexCount : 0) + numCams * 16 + numSplit * 2 + numPath * 2;

    const verts = new Float32Array(totalVerts * 4);
    const vu32  = new Uint32Array(verts.buffer);
    let ptr = 0;
    const put = (p, col, o = 0) => { verts[ptr] = p[o]; verts[ptr + 1] = p[o + 1]; verts[ptr + 2] = p[o + 2]; vu32[ptr + 3] = col; ptr += 4; };
    if (octVerts) { verts.set(octVerts.buffer); ptr = octVerts.vertexCount * 4; }
    this.octreeVertCount = octVerts ? octVerts.vertexCount : 0;

    this.frustumGeoms = [];
    if (numCams > 0) {
      const scale = (octree ? octree.sceneRadius : 2.0) * 0.06 * frustumScale;
      const quads = new Float32Array(numCams * 16);
      images.forEach((img, i) => {
        const { R, center, camera } = img;
        const w = scale * (camera.width / camera.fx) * 0.5;
        const h = scale * (camera.height / camera.fy) * 0.5;
        const toWorld = (x, y) => [0, 1, 2].map(k => center[k] + R[k] * x + R[3 + k] * y + R[6 + k] * scale);
        const pts = [center, toWorld(-w, -h), toWorld(w, -h), toWorld(w, h), toWorld(-w, h)];
        this.frustumGeoms.push({ camIdx: i, pts });
        const active = (i === activeCamIdx);
        [1, 2, 4, 3].forEach((k, j) => quads.set([...pts[k], active ? 0.94 : 0.80], i * 16 + j * 4));
        const col = active ? COL_ACTIVE : img.wireColor ?? (img.isValidation ? COL_VAL : COL_TRAIN);  // wireColor: SfM quality
        for (const [a, b] of FRUSTUM_EDGES) { put(pts[a], col); put(pts[b], col); }
      });
      if (this._ensureStorage('frustumQuadBuf', quads.length, 64, 'frustum_quads')) this.frustumBindGroups = new WeakMap();
      this.device.queue.writeBuffer(this.frustumQuadBuf, 0, quads);
    }

    for (let i = 0; i < 2 * numSplit; i++) put(splitVectors, COL_SPLIT, 3 * i);
    for (let i = 0; i < numPath; i++) { put(path[i], COL_PATH); put(path[i + 1], COL_PATH); }

    this.wireCount = totalVerts;
    if (totalVerts === 0) return;
    this._ensureStorage('wireBuf', verts.length, 1024, 'wireframes');
    this.device.queue.writeBuffer(this.wireBuf, 0, verts);
  }

  // Camera frustum under pixel (px, py) for the given view / intrinsics, or -1:
  // nearest frustum whose projected pyramid contains the point, else closest edge within 14 px.
  pickFrustum(px, py, view, fx, fy, cx, cy) {
    let bestIdx = -1, bestZ = Infinity, bestDist = 14.0;
    for (const { camIdx, pts } of this.frustumGeoms) {
      const s = pts.map(p => {
        const [x, y, z] = viewTransform(view, p);
        return (z > 0.08) ? [fx * (x / z) + cx, fy * (y / z) + cy, z] : null;
      });
      if (s.includes(null)) continue;
      const avgZ = (s[0][2] + s[1][2] + s[2][2] + s[3][2] + s[4][2]) * 0.2;
      if (FRUSTUM_TRIS.some(([a, b, c]) => pointInTri(px, py, s[a], s[b], s[c]))) {
        if (bestDist > 0 || avgZ < bestZ) { bestDist = 0; bestZ = avgZ; bestIdx = camIdx; }
      } else if (bestDist > 0) {
        const d = Math.min(...FRUSTUM_EDGES.map(([a, b]) => distToSeg(px, py, s[a], s[b])));
        if (d < bestDist) { bestDist = d; bestZ = avgZ; bestIdx = camIdx; }
      }
    }
    return bestIdx;
  }

  // Splats (sorted, front-to-back) + frustum image quads + wireframes onto the canvas.
  // frustumTextures: iterable of [camIdx, GPUTexture].
  renderScene({
    splatsBuf, momentsBuf, sortedIdxBuf, numVisible,
    view, sceneRadius = 2.0,
    fx, fy, cx, cy, W, H,
    visuMode = 0, colorMode = 0, haze = 1.0, avgR = 0.01,
    frustumTextures = null,
  }) {
    const dev = this.device;
    // RenderUniforms: view, intrinsics, screen, modes, thresholds
    const uBuf = new ArrayBuffer(128);
    const f32  = new Float32Array(uBuf);
    const u32  = new Uint32Array(uBuf);
    f32.set(view, 0);
    f32.set([fx, fy, cx, cy, W, H, haze, sceneRadius], 16);
    u32.set([visuMode, colorMode], 24);
    f32.set([Math.max(1e-5, avgR), ZNEAR, ZFAR], 28);
    dev.queue.writeBuffer(this.renderUniformBuf, 0, uBuf);

    const enc = dev.createCommandEncoder({ label: 'render_scene' });
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });

    if (numVisible > 0 && splatsBuf && sortedIdxBuf) {
      pass.setPipeline(this.pipeSplats);
      pass.setBindGroup(0, this._bindGroup(this.pipeSplats, this.renderUniformBuf, splatsBuf, momentsBuf, sortedIdxBuf));
      pass.draw(4, numVisible);
    }

    const drawWires  = this.wireCount > 0 && this.wireBuf;
    const drawImages = frustumTextures && frustumTextures.length > 0 && this.frustumQuadBuf;
    if (drawWires || drawImages) {  // WireUniforms: view, intrinsics, screen (W, H, znear, zfar), fade
      const wf = new Float32Array(28);
      wf.set(view, 0);
      wf.set([fx, fy, cx, cy, W, H, ZNEAR, ZFAR], 16);
      wf.set([this.octreeVertCount, OCTREE_FADE[0] * sceneRadius, OCTREE_FADE[1] * sceneRadius, OCTREE_FADE[2]], 24);
      dev.queue.writeBuffer(this.wireUniformBuf, 0, wf);
    }
    if (drawImages) {
      pass.setPipeline(this.pipeFrustumImg);
      for (const [camIdx, tex] of frustumTextures) {
        if (!tex) continue;
        let bg = this.frustumBindGroups.get(tex);
        if (!bg) {
          bg = this._bindGroup(this.pipeFrustumImg, this.wireUniformBuf, this.frustumQuadBuf, tex.createView(), this.sampler);
          this.frustumBindGroups.set(tex, bg);
        }
        pass.setBindGroup(0, bg);
        pass.draw(4, 1, 0, camIdx);  // first_instance selects the camera's quad
      }
    }
    if (drawWires) {
      pass.setPipeline(this.pipeWire);
      pass.setBindGroup(0, this._bindGroup(this.pipeWire, this.wireUniformBuf, this.wireBuf));
      pass.draw(this.wireCount);
    }
    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  // Fullscreen target / render / residual / depth-spread comparison (mode: see compare_blit
  // BlitParams); pixState: the trainer's pix_state, of the render in hatTex
  renderCompare({ gtTex, hatTex, pixState, mode = 0, splitX = 0.5, amp = mode === 3 ? 5.0 : 3.0 }) {
    if (!gtTex || !hatTex || !pixState) return;
    const b = new ArrayBuffer(16);
    new Uint32Array(b)[0] = mode;
    new Float32Array(b).set([splitX, amp], 1);
    this.device.queue.writeBuffer(this.blitUniformBuf, 0, b);

    const enc = this.device.createCommandEncoder({ label: 'render_compare' });
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipeBlit);
    pass.setBindGroup(0, this._bindGroup(this.pipeBlit, this.blitUniformBuf, gtTex.createView(), hatTex.createView(), this.sampler, pixState));
    pass.draw(4);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}

// ── Side-canvas telemetry ────────────────────────────────────────────────────

// Octree / rebalance stats, `statsText`, and the train PSNR curve (blue, by sample index)
// with held-out validation PSNR (orange points, by step).
export function drawTelemetryPanel(canvas, history, octree, statsText = '', valHistory = []) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.lineWidth = 1;
  ctx.fillStyle = '#121318';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#2a2d3a';
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  ctx.font = '12px monospace';
  ctx.fillStyle = '#e0e6f0';
  const s = octree ? octree.lastStats : { split: 0, cloned: 0, pruned: 0, floaters: 0, merged: 0, teleported: 0 };
  ctx.fillText(`Octree Leaves: ${octree ? octree.leaves.length : 0} | Rebalance: +${s.split} split, +${s.cloned} clone, ` +
               `-${s.pruned} prune, -${s.floaters} floaters, -${s.merged} merge${s.teleported ? `, ↻${s.teleported} mcmc` : ''}`, 12, 18);
  if (statsText) {
    ctx.fillStyle = '#8ef';
    ctx.fillText(statsText, 12, 36);
  }
  if (!history || history.length < 2) return;

  const padL = 44, padR = 16, padT = 46, padB = 20;
  const pw = W - padL - padR, ph = H - padT - padB;
  const s0 = history[0].step, s1 = history[history.length - 1].step;
  const vals = (valHistory || []).filter(v => v.step >= s0 && v.step <= s1);
  let minPsnr = 10, maxPsnr = 35;
  for (const h of [...history, ...vals]) {
    if (h.psnr < minPsnr) minPsnr = Math.floor(h.psnr - 1);
    if (h.psnr > maxPsnr) maxPsnr = Math.ceil(h.psnr + 1);
  }
  const yOf = (psnr) => padT + ph * (1 - clamp((psnr - minPsnr) / Math.max(1, maxPsnr - minPsnr), 0, 1));

  ctx.strokeStyle = '#252834';
  ctx.fillStyle = '#889';
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padT + (ph * i) / 4;
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.fillText(`${(maxPsnr - ((maxPsnr - minPsnr) * i) / 4).toFixed(1)}dB`, 4, y + 4);
  }
  ctx.stroke();

  ctx.strokeStyle = '#4af';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  history.forEach((h, i) => ctx.lineTo(padL + (i / (history.length - 1)) * pw, yOf(h.psnr)));
  ctx.stroke();

  if (vals.length > 0 && s1 > s0) {
    ctx.strokeStyle = ctx.fillStyle = '#fa4';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (const v of vals) {
      const x = padL + ((v.step - s0) / (s1 - s0)) * pw, y = yOf(v.psnr);
      ctx.lineTo(x, y);
      ctx.fillRect(x - 2, y - 2, 4, 4);
    }
    ctx.stroke();
    ctx.fillText(`val ${vals[vals.length - 1].psnr.toFixed(2)} dB`, W - padR - 90, padT - 4);
  }
}
