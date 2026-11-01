// render.js — Pure draw layer for Gauss-Opt:
//   - Free-view & COLMAP-pose splat renderer (shapes + diagnostic heatmaps)
//   - 3D wireframe overlays (Loose Octree boxes, COLMAP camera frustums, camera path, WLBG split vectors)
//   - Split-screen Target / Render / Residual comparison
//   - Side-canvas rolling PSNR & Octree/WLBG/MC/SA telemetry plot
"use strict";

import { clamp, sub, madd, normalize, viewMatToQuat, quatSlerp, viewFromQuat, viewTransform, fitIntrinsics, fitRect } from './math.js';
import { bindGroup } from './gpu.js';
import { mipLevels } from './mesh_image.js';
import { meshVoxel } from './glb.js';
import { ENV_WGSL } from './env.js';

let _shaderCache = null;

// `// === <name> ===` sections -> { name: code }, the `_*` ones (shared code) inserted at each
// `// #include _<name>` line and left out; `shared`: more `_*` sections (env.js's _env)
export function splitWgslSource(src, shared = { _env: ENV_WGSL }) {
  const sections = { ...shared }, out = {};
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
// orientation, linear blend of the intrinsics, over FLY_MS.
const FLY_MS = 680;
export function createCameraSplineAnimation({
  fromView, fromCenter, fromFx, fromFy, fromCx, fromCy,
  toImg, W, H, sceneCenter = [0, 0, 0],
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
    startTime: performance.now(),
    p0, p1: ctrl(0.33), p2: ctrl(0.67), p3,
    q0: viewMatToQuat(fromView), q1: viewMatToQuat(toImg.view),
    fromFx, fromFy, fromCx, fromCy, toFx, toFy, toCx, toCy,
  };
}

export function evaluateCameraSpline(anim) {
  const t = clamp((performance.now() - anim.startTime) / FLY_MS, 0, 1);
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
// The free view's depth buffer (linear view depth, see mesh_render): only the mesh writes it
const DEPTH_FORMAT = 'depth32float';
const depthState = (depthCompare, depthWriteEnabled = false) => ({ format: DEPTH_FORMAT, depthCompare, depthWriteEnabled });
// mesh display modes (params.js MESH_VIEWS)
export const MESH_OFF = 0, MESH_OVER = 1, MESH_ONLY = 2, MESH_WIRE = 3;

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
    // depth: the free view's pass has a depth attachment (renderScene); the compare blit's has none
    const pipe = (module, topology, blend, { depth = depthState('always'), fs = 'fs', buffers = undefined } = {}) => device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs', buffers },
      fragment: { module, entryPoint: fs, targets: [blend === undefined ? { format, writeMask: 0 } : blend ? { format, blend } : { format }] },
      primitive: { topology, cullMode: 'none' },
      ...(depth && { depthStencil: depth }),
    });
    this.pipeSplats     = pipe(mods.splats_render,    'triangle-strip', BLEND_UNDER);
    this.pipeSplatsHid  = pipe(mods.splats_render,    'triangle-strip', BLEND_UNDER, { depth: depthState('less') });  // behind the mesh: hidden
    this.pipeSplatsDepth = pipe(mods.splats_render,   'triangle-strip', undefined, { depth: depthState('less', true), fs: 'fs_depth' });  // depth only, for the wires
    this.pipeWire       = pipe(mods.wireframe_render, 'line-list',      BLEND_OVER, { depth: depthState('less') });
    this.pipeWireHid    = pipe(mods.wireframe_render, 'line-list',      BLEND_OVER, { depth: depthState('greater-equal') });
    this.pipeEnv        = pipe(mods.env_render,       'triangle-list',  BLEND_UNDER);  // the background, under the splats
    this.envBuf = null;  // setEnv(): the background's buffer (env.js), the trainer's own while it trains
    this.pipeFrustumImg = pipe(mods.frustum_image,    'triangle-strip', BLEND_OVER);
    this.pipeBlit       = pipe(mods.compare_blit,     'triangle-strip', null, { depth: null });
    const meshBuffers = [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
                         { arrayStride: 4,  attributes: [{ shaderLocation: 1, offset: 0, format: 'unorm8x4' }] },
                         { arrayStride: 8,  attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x2' }] }];
    const mesh = (topology, blend, fs, depth) => pipe(mods.mesh_render, topology, blend, { fs, depth, buffers: meshBuffers });
    this.pipeMeshDepth  = mesh('triangle-list', undefined,   'fs_depth_back', depthState('less', true));  // no color written; meshMargin behind
    this.pipeMeshFront  = mesh('triangle-list', undefined,   'fs_depth',  depthState('less', true));  // its own depth, after the splats
    this.pipeMeshColor  = mesh('triangle-list', BLEND_UNDER, 'fs_color',  depthState('less-equal'));
    this.pipeMeshTex    = mesh('triangle-list', BLEND_UNDER, 'fs_tex',    depthState('less-equal'));
    this.pipeMeshShaded = mesh('triangle-list', BLEND_UNDER, 'fs_shaded', depthState('less-equal'));
    this.pipeMeshLine   = mesh('line-list',     BLEND_OVER,  'fs_line',   depthState('less-equal'));
    this.depthTex = null;
    this.mesh = null;  // setMesh(): { pos, col, uv, tex, idx, idxCount, edges, edgeCount, voxel, bg: per pipeline }
    this.meshSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });

    const ubuf = (size, label) => device.createBuffer({ size, label, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.renderUniformBuf = ubuf(128, 'render_uniforms');
    this.wireUniformBuf   = ubuf(128, 'wire_uniforms');
    this.blitUniformBuf   = ubuf(16,  'blit_uniforms');
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    this.wireBuf           = null;  // line vertices: xyz + bitcast ABGR color
    this.wireCount         = 0;
    this.octreeVertCount   = 0;     // the first wireBuf vertices are octree boxes (faded with depth)
    this.frustumQuadBuf    = null;  // per camera: 4 strip corners (xyz + alpha)
    this.frustumGeoms      = [];    // per camera: { camIdx, pts: [apex, p1..p4] } for picking
    this.frustumBindGroups = new WeakMap();  // texture -> bind group (reset when frustumQuadBuf moves)
  }

  // The mesh to show (DESIGN.md §9 E5): { positions: Float32Array 3V, faces: Uint32Array 3F, colors?:
  // Uint8Array 3V, uvs?: Float32Array 2V, image?: its texture, { data: RGBA bytes, width, height } }
  // or null; uploaded once, with its unique edges for the wireframe and the texture's mip levels
  setMesh(m) {
    for (const k of ['pos', 'col', 'uv', 'tex', 'idx', 'edges']) this.mesh?.[k]?.destroy();
    this.mesh = null;
    if (!m || !m.faces.length) return;
    const dev = this.device, nv = m.positions.length / 3;
    const buf = (data, usage, label) => {  // padded to 4 bytes, as writeBuffer wants
      const bytes = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
      bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      const b = dev.createBuffer({ size: bytes.length, usage: usage | GPUBufferUsage.COPY_DST, label });
      dev.queue.writeBuffer(b, 0, bytes);
      return b;
    };
    const col = new Uint8Array(4 * nv).fill(255);
    for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) col[4 * i + k] = m.colors ? m.colors[3 * i + k] : 180;
    const seen = new Set(), edges = [];
    for (let f = 0; f < m.faces.length; f += 3) for (let k = 0; k < 3; k++) {
      const a = m.faces[f + k], b = m.faces[f + (k + 1) % 3], key = a < b ? a * nv + b : b * nv + a;
      if (!seen.has(key)) { seen.add(key); edges.push(a, b); }
    }
    let tex = null;
    if (m.uvs && m.image) {
      const levels = mipLevels(m.image);
      tex = dev.createTexture({ size: [m.image.width, m.image.height], format: 'rgba8unorm', mipLevelCount: levels.length,
                                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, label: 'mesh_tex' });
      levels.forEach((l, i) => dev.queue.writeTexture({ texture: tex, mipLevel: i }, l.data, { bytesPerRow: 4 * l.width }, [l.width, l.height]));
    }
    this.mesh = {
      pos: buf(m.positions, GPUBufferUsage.VERTEX, 'mesh_pos'), col: buf(col, GPUBufferUsage.VERTEX, 'mesh_col'),
      uv: buf(tex ? m.uvs : new Float32Array(2 * nv), GPUBufferUsage.VERTEX, 'mesh_uv'), tex,
      idx: buf(m.faces, GPUBufferUsage.INDEX, 'mesh_idx'), idxCount: m.faces.length,
      edges: buf(Uint32Array.from(edges), GPUBufferUsage.INDEX, 'mesh_edges'), edgeCount: edges.length, voxel: meshVoxel(m),
      bg: new Map(),
    };
  }

  _drawMesh(pass, pipeline, lines = false) {
    const m = this.mesh;
    let bg = m.bg.get(pipeline);
    if (!bg) m.bg.set(pipeline, bg = pipeline === this.pipeMeshTex ? this._bindGroup(pipeline, this.wireUniformBuf, m.tex.createView(), this.meshSampler)
                                                                   : this._bindGroup(pipeline, this.wireUniformBuf));
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.setVertexBuffer(0, m.pos);
    pass.setVertexBuffer(1, m.col);
    pass.setVertexBuffer(2, m.uv);
    pass.setIndexBuffer(lines ? m.edges : m.idx, 'uint32');
    pass.drawIndexed(lines ? m.edgeCount : m.idxCount);
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
  // The camera path joins the camera centers in pathOrder (indices into images; null: image-name
  // order, = capture order for video frames and photo sets; DESIGN-sfm §13.3 P7).
  updateWireframes({
    octree, images, activeCamIdx = -1,
    showOctree = true, showLoose = false, showCameras = true, showPath = false,
    frustumScale = 1.0, splitVectors = null, pathOrder = null,
  }) {
    const octVerts = (showOctree && octree && octree.leaves.length > 0) ? octree.buildWireframeVertices({ showLoose }) : null;
    const numCams  = (showCameras && images) ? images.length : 0;
    const numSplit = splitVectors ? Math.floor(splitVectors.length / 6) : 0;
    const path = (showPath && images?.length > 1) ? (pathOrder ?? images.map((_, i) => i)).map(i => images[i].center) : [];
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
  // meshMode (with setMesh's mesh): MESH_OVER: the mesh's depth first, the splats behind it hidden, then
  // the mesh blended under the rest; MESH_ONLY: the mesh alone; MESH_WIRE: its depth only (the splats
  // behind it hidden), its visible edges over the rest. meshShaded: gray lit by the camera instead of the vertex colors.
  // meshMargin: the splats hidden only when this many of the mesh's voxels (glb.js meshVoxel) behind it.
  renderScene({
    splatsBuf, momentsBuf, sortedIdxBuf, numVisible,
    view, sceneRadius = 2.0,
    fx, fy, cx, cy, W, H,
    visuMode = 0, colorMode = 0, haze = 1.0, avgR = 0.01,
    frustumTextures = null, meshMode = MESH_OFF, meshShaded = false, meshMargin = 0, octreeHidden = 1,
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
    const target = this.context.getCurrentTexture();
    if (!this.depthTex || this.depthTex.width !== target.width || this.depthTex.height !== target.height) {
      this.depthTex?.destroy();
      this.depthTex = dev.createTexture({ size: [target.width, target.height], format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT, label: 'view_depth' });
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: target.createView(),
        loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });

    const mesh = this.mesh && meshMode !== MESH_OFF;
    const drawWires  = this.wireCount > 0 && this.wireBuf;
    const drawImages = frustumTextures && frustumTextures.length > 0 && this.frustumQuadBuf;
    if (drawWires || drawImages || mesh) {  // WireUniforms: view, intrinsics, screen (W, H, znear, zfar), fade, mesh
      const wf = new Float32Array(32);
      wf.set(view, 0);
      wf.set([fx, fy, cx, cy, W, H, ZNEAR, ZFAR], 16);
      wf.set([this.octreeVertCount, OCTREE_FADE[0] * sceneRadius, OCTREE_FADE[1] * sceneRadius, OCTREE_FADE[2]], 24);
      wf[28] = mesh ? meshMargin * this.mesh.voxel : 0;
      wf[29] = octreeHidden;
      dev.queue.writeBuffer(this.wireUniformBuf, 0, wf);
    }
    if (mesh) this._drawMesh(pass, this.pipeMeshDepth);
    if (numVisible > 0 && splatsBuf && sortedIdxBuf && !(mesh && meshMode === MESH_ONLY)) {
      const p = mesh ? this.pipeSplatsHid : this.pipeSplats;
      pass.setPipeline(p);
      pass.setBindGroup(0, this._bindGroup(p, this.renderUniformBuf, splatsBuf, momentsBuf, sortedIdxBuf));
      pass.draw(4, numVisible);
    }
    if (mesh && meshMargin > 0) this._drawMesh(pass, this.pipeMeshFront);  // its faces within the margin of its front ones stay hidden
    if (mesh && meshMode !== MESH_WIRE) this._drawMesh(pass, meshShaded ? this.pipeMeshShaded : this.mesh.tex ? this.pipeMeshTex : this.pipeMeshColor);
    if (this.envBuf && visuMode === 0 && colorMode === 0 && !(mesh && meshMode === MESH_ONLY)) {  // the background, in the splats' own colors only
      pass.setPipeline(this.pipeEnv);
      pass.setBindGroup(0, this._bindGroup(this.pipeEnv, this.renderUniformBuf, this.envBuf));
      pass.draw(3);
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
      const splatDepth = this.octreeVertCount > 0 && octreeHidden < 1 && numVisible > 0 && splatsBuf && sortedIdxBuf && !(mesh && meshMode === MESH_ONLY);
      if (splatDepth) {  // the splats' depth (with the mesh's): the boxes behind them faint
        pass.setPipeline(this.pipeSplatsDepth);
        pass.setBindGroup(0, this._bindGroup(this.pipeSplatsDepth, this.renderUniformBuf, splatsBuf, momentsBuf, sortedIdxBuf));
        pass.draw(4, numVisible);
      }
      for (const [p, hidden] of [[this.pipeWire, 0], [this.pipeWireHid, 1]]) {
        pass.setPipeline(p);
        pass.setBindGroup(0, this._bindGroup(p, this.wireUniformBuf, this.wireBuf));
        pass.draw(this.wireCount, 1, 0, hidden);
      }
    }
    if (mesh && meshMode === MESH_WIRE) this._drawMesh(pass, this.pipeMeshLine, true);
    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  // Fullscreen target / render / residual / depth-spread comparison (mode: see compare_blit
  // BlitParams); pixState: the trainer's pix_state, of the render in hatTex
  renderCompare({ gtTex, hatTex, pixState, mode = 0, splitX = 0.5 }) {
    if (!gtTex || !hatTex || !pixState) return;
    const amp = mode === 3 ? 5.0 : 3.0;
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
    // the photo keeps its aspect: in the largest box of it in the canvas, black around
    const target = this.context.canvas, box = fitRect(gtTex.width, gtTex.height, target.width, target.height);
    pass.setViewport(box.x, box.y, box.w, box.h, 0, 1);
    pass.setPipeline(this.pipeBlit);
    pass.setBindGroup(0, this._bindGroup(this.pipeBlit, this.blitUniformBuf, gtTex.createView(), hatTex.createView(), this.sampler, pixState));
    pass.draw(4);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}

// ── Side-canvas telemetry ────────────────────────────────────────────────────

// Each sample's mean PSNR over the samples of the last `passSteps` steps (history: { step, psnr }, steps rising)
export function passMean(history, passSteps) {
  let lo = 0, sum = 0;
  return history.map((h, i) => {
    sum += h.psnr;
    while (history[lo].step <= h.step - passSteps) sum -= history[lo++].psnr;
    return sum / (i - lo + 1);
  });
}

// Octree / rebalance stats, `statsText`, and the train PSNR curve (by sample index) with held-out validation PSNR
// (orange points, by step). Each sample is one training view's PSNR, the views in a fixed round robin, so the raw
// curve (faint) repeats how hard each view is, once a pass; the bold line is the mean over the last `passSteps` steps
// (one pass over the training views; 0: the raw curve alone).
export function drawTelemetryPanel(canvas, history, octree, statsText = '', valHistory = [], passSteps = 0) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  // its size on screen (CSS pixels; the canvas's own when not laid out), its pixels at the device's density
  const W = canvas.clientWidth || canvas.width, H = canvas.clientHeight || canvas.height, dpr = canvas.clientWidth ? devicePixelRatio || 1 : 1;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.lineWidth = 1;
  ctx.fillStyle = '#121318';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#2a2d3a';
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  ctx.font = W < 500 ? '9px monospace' : '12px monospace';  // a phone's pane: the header lines fit more
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

  const xOf = (i) => padL + (i / (history.length - 1)) * pw;
  ctx.strokeStyle = passSteps > 1 ? '#4af5' : '#4af';
  ctx.lineWidth = passSteps > 1 ? 1 : 1.8;
  ctx.beginPath();
  history.forEach((h, i) => ctx.lineTo(xOf(i), yOf(h.psnr)));
  ctx.stroke();
  if (passSteps > 1) {
    ctx.strokeStyle = '#4af';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    passMean(history, passSteps).forEach((m, i) => ctx.lineTo(xOf(i), yOf(m)));
    ctx.stroke();
  }

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
