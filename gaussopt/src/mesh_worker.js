// mesh_worker.js — Simplify a mesh and write it as .glb off the page's thread (DESIGN.md §9 E5): a 1.3M
// face mesh takes 10-15 s to simplify. In: { mesh, maxError, up, glb }; out: { glb (when asked), mesh, info } or { error }.
import { simplify } from './mesh_simplify.js';
import { writeGlb } from './glb.js';

self.onmessage = ({ data: { mesh, maxError, up, glb: wantGlb = true } }) => {
  try {
    const t0 = performance.now();
    const s = maxError > 0 ? simplify(mesh, { maxError }) : { mesh, info: { passes: 0, error: 0 } };
    const glb = wantGlb ? writeGlb(s.mesh, { up }) : null;
    const m = s.mesh;
    self.postMessage({ glb, mesh: m, info: { ...s.info, ms: performance.now() - t0 } },
                     [...(glb ? [glb.buffer] : []), m.positions.buffer, m.faces.buffer, ...(m.colors ? [m.colors.buffer] : [])]);
  } catch (e) {
    self.postMessage({ error: e.message });
  }
};
