// mesh_grow.js: the mesh behind the splats of a flat wall seen by a row of cameras in front of it.
import { growBacking } from '../src/mesh_grow.js';
import { SPLAT_FLOATS, rgbToSh0 } from '../src/colmap.js';
import { makeViewMatrix } from '../src/math.js';
import { assert } from './util.js';

const WALL = 5, BOUNDS = { min: [-3, -2, 3], max: [3, 2, 8] }, RES = 64;  // voxel 6 / 63 = 0.095

const splat = (x, y, z, sx, sy, sz) => [x, y, z, Math.log(0.9 / 0.1), Math.log(sx), Math.log(sy), Math.log(sz), 0, 1, 0, 0, 0,
                                        rgbToSh0(0.9), rgbToSh0(0.5), rgbToSh0(0.1), 0];

// Flat splats on the plane z = WALL over |x| < 2, |y| < 1.5 every 0.05, orange; none where hole(x, y); more: other splats
function wall(hole = () => false, more = []) {
  const out = [...more];
  for (let x = -2; x <= 2; x += 0.05) for (let y = -1.5; y <= 1.5; y += 0.05) {
    if (hole(x, y)) continue;
    out.push(splat(x, y, WALL, 0.04, 0.04, 0.005));
  }
  return pack(out);
}
function pack(out) {
  const params = new Float32Array(out.length * SPLAT_FLOATS);
  out.forEach((p, i) => params.set(p, i * SPLAT_FLOATS));
  return { params, count: out.length };
}

// 5 cameras at z = 0, x in [-1, 1], looking down +z (160 x 90, f 100: the view spans |x| < 4 at the wall)
const cameras = [-1, -0.5, 0, 0.5, 1].map(x => {
  const R = [1, 0, 0, 0, 1, 0, 0, 0, 1], t = [-x, 0, 0];
  return { center: [x, 0, 0], view: makeViewMatrix(R, t), fx: 100, fy: 100, cx: 80, cy: 45, width: 160, height: 90 };
});

const opts = { res: RES, close: 2, inset: 4, depth: 12, rayW: 96 };

Deno.test('mesh_grow: a sheet behind the wall, inset from its front, open behind, colored as the splats', () => {
  const { mesh, info } = growBacking(wall(), cameras, BOUNDS, opts);
  const P = mesh.positions, v = info.voxel, used = new Set(mesh.faces);
  assert(mesh.faces.length / 3 > 500, `${mesh.faces.length / 3} faces`);
  let zMin = Infinity, zMax = -Infinity, xMin = Infinity, xMax = -Infinity;
  for (const i of used) {
    xMin = Math.min(xMin, P[3 * i]); xMax = Math.max(xMax, P[3 * i]);
    if (Math.abs(P[3 * i]) > 1.5 || Math.abs(P[3 * i + 1]) > 1) continue;  // the sheet wraps back around the wall's edges
    zMin = Math.min(zMin, P[3 * i + 2]); zMax = Math.max(zMax, P[3 * i + 2]);
  }
  // behind the splats' plane, and not at the solid's back (WALL + depth voxels): the faces there are cut
  assert(zMin > WALL, `in front of the wall: z ${zMin.toFixed(3)}`);
  assert(zMax < WALL + (opts.close + opts.inset) * v, `behind the inset: z ${zMax.toFixed(3)} (voxel ${v.toFixed(3)})`);
  assert(info.cut > 0, `no faces cut`);
  // not past the wall's sides (faces with no splats within the color blur are dropped)
  const blur = (opts.close + opts.inset + 2) * v;
  assert(xMin > -2 - blur && xMax < 2 + blur, `x ${xMin.toFixed(2)} .. ${xMax.toFixed(2)}`);
  const c = [0, 0, 0];
  for (const i of used) for (let k = 0; k < 3; k++) c[k] += mesh.colors[3 * i + k] / used.size;
  assert(Math.abs(c[0] - 230) < 12 && Math.abs(c[1] - 128) < 12 && Math.abs(c[2] - 26) < 12, `mean color ${c.map(Math.round)}`);
});

Deno.test('mesh_grow: a hole in the wall narrower than twice the closing is backed, a wider one is not', () => {
  const v = (BOUNDS.max[0] - BOUNDS.min[0]) / (RES - 1);
  const covered = (halfWidth) => {
    const { mesh } = growBacking(wall((x, y) => Math.abs(x) < halfWidth && Math.abs(y) < 0.5), cameras, BOUNDS, opts);
    const P = mesh.positions;
    let n = 0;
    for (const i of new Set(mesh.faces)) if (Math.abs(P[3 * i]) < 0.5 * halfWidth && Math.abs(P[3 * i + 1]) < 0.25) n++;
    return n;
  };
  const narrow = covered(1.8 * v), wide = covered(8 * v);  // 3.6 voxels: open in the occupancy, under twice the closing
  assert(narrow > 0, 'the narrow hole is not backed');
  assert(wide === 0, `the wide hole is backed: ${wide} vertices`);
});

Deno.test('mesh_grow: a slot under twice the closing, seen lengthwise, stays open (no lid over it)', () => {
  // the wall with a slit |x| < g / 2 (g 3.6 voxels: open in the splats, closed by the closing), behind it a corridor
  // (walls x = +-g / 2 from WALL to WALL + 2.8, 29 voxels: over the default passage) to a back wall, the cameras looking down it
  const v = (BOUNDS.max[0] - BOUNDS.min[0]) / (RES - 1), g = 3.6 * v, more = [];
  for (let z = WALL; z <= WALL + 2.8; z += 0.05) for (let y = -1.5; y <= 1.5; y += 0.05)
    for (const x of [-g / 2, g / 2]) more.push(splat(x, y, z, 0.005, 0.04, 0.04));
  for (let x = -0.5; x <= 0.5; x += 0.05) for (let y = -1.5; y <= 1.5; y += 0.05) more.push(splat(x, y, WALL + 2.8, 0.04, 0.04, 0.005));
  const splats = wall((x) => Math.abs(x) < g / 2, more);
  const inSlot = (passage) => {  // vertices in the slot's middle, in front of the back wall
    const { mesh, info } = growBacking(splats, cameras, BOUNDS, { ...opts, passage });
    const P = mesh.positions;
    let n = 0;
    for (const i of new Set(mesh.faces)) if (Math.abs(P[3 * i]) < g / 4 && Math.abs(P[3 * i + 1]) < 1 && P[3 * i + 2] < WALL + 2.3) n++;
    return [n, info.passages];
  };
  const [open, passages] = inSlot(undefined), [lid] = inSlot(Infinity);
  assert(lid > 0, 'no lid over the slot even without passages: the test does not test');
  assert(open === 0 && passages > 0, `a lid over the slot: ${open} vertices (${passages} rays through it)`);
});
