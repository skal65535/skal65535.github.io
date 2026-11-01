// mesh_gpu.js — A mesh's GPU buffers for drawing (render.js, the gallery): positions, colors (gray without), uvs and its
// texture's mip levels, faces and the unique edges (the wireframe), and its voxel (glb.js meshVoxel).
"use strict";

import { mipLevels } from './mesh_image.js';
import { meshVoxel } from './glb.js';

export function meshBuffers(dev, m) {
  const nv = m.positions.length / 3;
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
  return {
    pos: buf(m.positions, GPUBufferUsage.VERTEX, 'mesh_pos'), col: buf(col, GPUBufferUsage.VERTEX, 'mesh_col'),
    uv: buf(tex ? m.uvs : new Float32Array(2 * nv), GPUBufferUsage.VERTEX, 'mesh_uv'), tex,
    idx: buf(m.faces, GPUBufferUsage.INDEX, 'mesh_idx'), idxCount: m.faces.length,
    edges: buf(Uint32Array.from(edges), GPUBufferUsage.INDEX, 'mesh_edges'), edgeCount: edges.length, voxel: meshVoxel(m),
  };
}
