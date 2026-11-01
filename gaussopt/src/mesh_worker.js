// mesh_worker.js — Simplify a mesh and write it as .glb off the page's thread (DESIGN.md §9 E5): a large
// mesh takes seconds to simplify; or first grow it behind the splats (mesh_grow.js, up to 28 bytes a grid
// cell). In: { mesh, voxel } or { grow: { splats, cameras, bounds, opts } }, simplify (voxels), up, glb;
// out: { glb (when asked), mesh, info: { voxel, grow (growBacking's info), ... } } or { error }.
import { simplify } from './mesh_simplify.js';
import { writeGlb } from './glb.js';
import { growBacking } from './mesh_grow.js';

self.onmessage = ({ data: { mesh, voxel, grow, simplify: maxVoxels, up, glb: wantGlb = true } }) => {
  try {
    const t0 = performance.now();
    let grown = null;
    if (grow) {
      grown = growBacking(grow.splats, grow.cameras, grow.bounds, grow.opts);
      ({ mesh } = grown);
      voxel = grown.info.voxel;
    }
    const s = maxVoxels > 0 ? simplify(mesh, { maxError: maxVoxels * voxel }) : { mesh, info: { passes: 0, error: 0 } };
    const glb = wantGlb ? writeGlb(s.mesh, { up, voxel }) : null;
    const m = s.mesh;
    self.postMessage({ glb, mesh: m, info: { ...s.info, voxel, grow: grown?.info, ms: performance.now() - t0 } },
                     [...(glb ? [glb.buffer] : []), m.positions.buffer, m.faces.buffer, ...(m.colors ? [m.colors.buffer] : [])]);
  } catch (e) {
    self.postMessage({ error: e.message });
  }
};
