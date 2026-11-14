// edit.js: the splat under a pixel (pickSplat), the cluster grown from a splat (growCluster, through SplatGrid's
// touching splats) and the floaters apart from the scene (findClusters).
import { SPLAT_FLOATS } from '../src/colmap.js';
import { SplatGrid, pickSplat, growCluster, findClusters, components, splatRadius } from '../src/edit.js';
import { assert, rng } from './util.js';

const logit = (a) => Math.log(a / (1 - a));
// splats [x, y, z, alpha, r] or [x, y, z, alpha, [rx, ry, rz], [qw, qx, qy, qz]]
function splats(rows) {
  const p = new Float32Array(rows.length * SPLAT_FLOATS);
  rows.forEach(([x, y, z, a, r, q = [1, 0, 0, 0]], i) => {
    const s = Array.isArray(r) ? r : [r, r, r];
    p.set([x, y, z, logit(a), ...s.map(Math.log), 0, ...q, 0, 0, 0, 0], i * SPLAT_FLOATS);
  });
  return { params: p, count: rows.length };
}
// a camera at the origin looking down +z (world-to-camera: the identity), 200 x 200, f 200
const CAM = { view: Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), fx: 200, fy: 200, cx: 100, cy: 100, W: 200, H: 200 };

Deno.test('pickSplat: the splat adding the most to the pixel, front to back; its ellipse turned with it', () => {
  const two = (aFront) => splats([[0, 0, 4, 0.9, 0.2], [0, 0, 2, aFront, 0.2]]);
  let s = two(0.9);
  let hit = pickSplat(s.params, s.count, CAM, 100, 100);
  assert(hit?.index === 1 && Math.abs(hit.depth - 2) < 1e-6, `opaque in front: ${JSON.stringify(hit)}`);
  s = two(0.05);
  hit = pickSplat(s.params, s.count, CAM, 100, 100);
  assert(hit?.index === 0, `faint in front: the one behind, ${JSON.stringify(hit)}`);
  assert(pickSplat(s.params, s.count, CAM, 190, 190) === null, 'a pixel off both: none');
  // a splat long along x (1 by 0.02) at depth 2: at 40 px right of its centre (0.4 m), covered; 40 px down, not;
  // turned 90 deg about z, the other way round; turned 45 deg, along the diagonal down right (x right, y down), not up
  const along = splats([[0, 0, 2, 0.9, [1, 0.02, 0.02]]]);
  const turned = splats([[0, 0, 2, 0.9, [1, 0.02, 0.02], [Math.SQRT1_2, 0, 0, Math.SQRT1_2]]]);
  assert(pickSplat(along.params, 1, CAM, 140, 100) && !pickSplat(along.params, 1, CAM, 100, 140), 'long along x: right yes, down no');
  assert(!pickSplat(turned.params, 1, CAM, 140, 100) && pickSplat(turned.params, 1, CAM, 100, 140), 'turned: right no, down yes');
  const c = Math.cos(Math.PI / 8), d = splats([[0, 0, 2, 0.9, [1, 0.02, 0.02], [c, 0, 0, Math.sin(Math.PI / 8)]]]);
  assert(pickSplat(d.params, 1, CAM, 128, 128) && !pickSplat(d.params, 1, CAM, 128, 72), 'turned 45 deg: down right yes, up right no');
});

// a floor: 40 x 40 splats 0.05 apart, r 0.03 (1.67 sigmas: they touch at reach 2); above it a blob of 27 splats 0.008
// apart, r 0.01 (0.8 sigmas)
function scene({ blobY = 0.5, blobN = 3, extra = [] } = {}) {
  const rows = [];
  for (let i = 0; i < 40; i++) for (let k = 0; k < 40; k++) rows.push([i * 0.05, 0, k * 0.05, 0.8, 0.03]);
  const floor = rows.length;
  for (let a = 0; a < blobN; a++) for (let b = 0; b < blobN; b++) for (let c = 0; c < blobN; c++) {
    rows.push([1 + a * 0.008, blobY + b * 0.008, 1 + c * 0.008, 0.8, 0.01]);
  }
  rows.push(...extra);
  return { ...splats(rows), floor, blob: rows.length - floor - extra.length };
}

Deno.test('growCluster: a floating blob exactly, the floor attached; the reach cuts a blob off the floor', () => {
  const s = scene();
  const grid = new SplatGrid(s.params, s.count, { reach: 2 });
  const blob = growCluster(grid, s.floor, 200);
  assert(!blob.attached && blob.indices.length === 27 && blob.indices[0] === s.floor && blob.indices.at(-1) === s.floor + 26,
         `blob: ${blob.indices.length}, attached ${blob.attached}`);
  const floor = growCluster(grid, 0, 200);
  assert(floor.attached && floor.indices.length === 201, `floor: ${floor.indices.length}, attached ${floor.attached}`);
  // the blob 0.05 above the floor: 1.67 of the floor's sigmas, touching it at reach 2, not at reach 1
  const near = scene({ blobY: 0.05 });
  const at2 = growCluster(new SplatGrid(near.params, near.count, { reach: 2 }), near.floor, 200);
  const at1 = growCluster(new SplatGrid(near.params, near.count, { reach: 1 }), near.floor, 200);
  assert(at2.attached, `reach 2: the blob 0.07 above the floor touches it (${at2.indices.length})`);
  assert(!at1.attached && at1.indices.length === 27, `reach 1: cut off, ${at1.indices.length}`);
});

Deno.test('SplatGrid: a large splat and a small one touching found from both, at any level', () => {
  // r 1 and r 0.001, 1.9 apart: within 2 of the large one's sigmas, not 1; 200 small ones apart, so the large one is
  // on a level of its own, in another cell than the small one
  const rows = [[-0.4, 1, 1, 0.8, 1], [1.5, 1, 1, 0.8, 0.001]];
  for (let i = 0; i < 200; i++) rows.push([10 + i * 0.1, 5, 5, 0.8, 0.001]);
  const s = splats(rows);
  const near = (reach, i) => { const g = new SplatGrid(s.params, s.count, { reach }), out = []; g.neighbours(i, j => out.push(j)); return out; };
  assert(near(2, 0).join() === '1' && near(2, 1).join() === '0', `reach 2: ${near(2, 0)} / ${near(2, 1)}`);
  assert(near(1, 0).length === 0 && near(1, 1).length === 0, `reach 1: ${near(1, 0)} / ${near(1, 1)}`);
  assert(Math.abs(splatRadius(s.params, 0) - 1) < 1e-6, 'r: the largest axis');
  // a flat splat (0.5, 0.5, 0.01) turned 90 deg about x (its thin axis along y): a small one 0.8 off along its plane
  // (z) touches it, 0.8 across it (y) does not
  const t = Math.SQRT1_2, flat = splats([[0, 0, 0, 0.8, [0.5, 0.5, 0.01], [t, t, 0, 0]], [0, 0, 0.8, 0.8, 0.001], [0, 0.8, 0, 0.8, 0.001]]);
  const g = new SplatGrid(flat.params, 3, { reach: 2 }), out = [];
  g.neighbours(0, j => out.push(j));
  assert(out.join() === '1', `flat: along its plane yes, across no: ${out}`);
});

Deno.test('findClusters: the floaters apart from the floor, the most visible first; faint splats left out', () => {
  // the 27-splat blob, a 2-splat pair of r 0.05 (more mass) far off, a lone splat, a faint lone one (alpha 0.01)
  const s = scene({ extra: [[3, 1, 3, 0.8, 0.05], [3.05, 1, 3, 0.8, 0.05], [-2, 1, -2, 0.8, 0.005], [5, 5, 5, 0.01, 0.05]] });
  const res = findClusters(new SplatGrid(s.params, s.count, { reach: 2 }), 200);
  const sizes = res.clusters.map(c => c.indices.length).join();
  assert(res.largest === s.floor && res.total === 3 && sizes === '2,27,1', `floor ${res.largest}, ${res.total} floaters of ${sizes}`);
  assert(res.clusters[0].indices.join() === `${s.count - 4},${s.count - 3}`, `the pair first: ${res.clusters[0].indices}`);
  const c = res.clusters[1].center;
  assert(Math.abs(c[0] - 1.008) < 1e-5 && Math.abs(c[1] - 0.508) < 1e-5, `the blob's centre ${c}`);
  // the largest cluster never a floater, whatever maxSize
  const big = findClusters(new SplatGrid(s.params, s.count, { reach: 2 }), 5000);
  assert(big.clusters.map(c => c.indices.length).join() === '2,27,1', `maxSize 5000: ${big.clusters.map(c => c.indices.length)}`);
  // at least 2 splats: the lone one left out
  const two = findClusters(new SplatGrid(s.params, s.count, { reach: 2 }), 200, { minSplats: 2 });
  assert(two.clusters.map(c => c.indices.length).join() === '2,27', `minSplats 2: ${two.clusters.map(c => c.indices.length)}`);
  // no floater over maxSize: the blob's 27 at 20 is the scene's
  const small = findClusters(new SplatGrid(s.params, s.count, { reach: 2 }), 20);
  assert(small.clusters.map(c => c.indices.length).join() === '2,1', `maxSize 20: ${small.clusters.map(c => c.indices.length)}`);
});

Deno.test('findClusters: the isolation keeps the floaters still apart at a larger reach; the cameras rank by their view', () => {
  // the blob 0.1 above the floor: 3.3 of the floor's sigmas, apart at reach 2, touching at reach 4; a pair of r 0.05
  // far off (more mass) and a lone splat of r 0.02 near a camera at (-2, 1.2, -2)
  const s = scene({ blobY: 0.1, extra: [[3, 1, 3, 0.8, 0.05], [3.05, 1, 3, 0.8, 0.05], [-2, 1, -2, 0.8, 0.02]] });
  const sizes = (r) => r.clusters.map(c => c.indices.length).join();
  const at2 = new SplatGrid(s.params, s.count, { reach: 2 }), at4 = new SplatGrid(s.params, s.count, { reach: 4 });
  const plain = findClusters(at2, 200), iso = findClusters(at2, 200, { isolation: at4 });
  assert(sizes(plain) === '2,27,1' && plain.joined === 0, `no isolation: ${sizes(plain)}`);
  assert(sizes(iso) === '2,1' && iso.joined === 1, `isolation at reach 4: ${sizes(iso)}, ${iso.joined} joined`);
  const cam = findClusters(at2, 200, { isolation: at4, cameras: [[-2, 1.2, -2]] });
  assert(sizes(cam) === '1,2' && Math.abs(cam.clusters[0].near - 0.2) < 1e-6, `by the camera's view: ${sizes(cam)}, near ${cam.clusters[0].near}`);
});

Deno.test('components: each pair from its smaller splat, the same clusters as every splat searched fully', () => {
  // 600 splats of r 0.002 to 0.5 (log-uniform), flat or round, turned at random, in a 3-unit box
  // (472, 129 and 22 clusters at reach 1, 2, 3)
  const r = rng(11), rows = [];
  for (let i = 0; i < 600; i++) {
    const s = 0.002 * 250 ** r(), flat = r() < 0.5 ? 0.05 : 1, q = [r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5];
    rows.push([3 * r(), 3 * r(), 3 * r(), 0.8, [s, s * (0.3 + 0.7 * r()), s * flat], q]);
  }
  const s = splats(rows);
  for (const reach of [1, 2, 3]) {
    const g = new SplatGrid(s.params, s.count, { reach }), { root } = components(g);
    // the reference: every splat's full search, a flood fill
    const label = new Int32Array(s.count).fill(-1);
    let pairs = 0;
    for (let i = 0; i < s.count; i++) {
      if (label[i] >= 0) continue;
      label[i] = i;
      for (const st = [i]; st.length;) g.neighbours(st.pop(), (j) => { pairs++; if (label[j] < 0) { label[j] = i; st.push(j); } });
    }
    // the same partition: each root one label and each label one root
    const a = new Map(), b = new Map();
    let same = true;
    for (let i = 0; i < s.count; i++) {
      if ((a.get(root[i]) ?? label[i]) !== label[i] || (b.get(label[i]) ?? root[i]) !== root[i]) same = false;
      a.set(root[i], label[i]); b.set(label[i], root[i]);
    }
    assert(same && a.size > 10 && a.size < s.count - 50 && pairs > 100, `reach ${reach}: the same clusters ${same}, ${a.size} of them (${pairs} touching pairs seen)`);
    assert(new Set(g.level).size >= 5, `levels: ${new Set(g.level).size}`);
  }
});

Deno.test('findClusters small: the isolated floaters, the splats of the groups under minSplats still apart at a larger reach', () => {
  // a lone splat far off, one 0.1 above the floor (apart at reach 2, joining it at 4), the pair and the blob
  const s = scene({ extra: [[3, 1, 3, 0.8, 0.05], [3.05, 1, 3, 0.8, 0.05], [-2, 1, -2, 0.8, 0.02], [0.5, 0.1, 0.5, 0.8, 0.01]] });
  const at2 = new SplatGrid(s.params, s.count, { reach: 2 }), at4 = new SplatGrid(s.params, s.count, { reach: 4 });
  const far = s.count - 2, low = s.count - 1;
  const res = findClusters(at2, 200, { isolation: at4, minSplats: 3, small: true });
  // at 3: the pair and the far one (2 groups), not the one near the floor
  assert(res.small.join() === `${far - 2},${far - 1},${far}` && res.smallGroups === 2, `isolated at minSplats 3: ${res.small} (${res.smallGroups} groups)`);
  assert(res.clusters.map(c => c.indices.length).join() === '27', `the clusters: ${res.clusters.map(c => c.indices.length)}`);
  const two = findClusters(at2, 200, { isolation: at4, minSplats: 2, small: true });
  assert(two.small.join() === `${far}` && two.clusters.map(c => c.indices.length).join() === '2,27', `minSplats 2: ${two.small}, ${two.clusters.map(c => c.indices.length)}`);
  const none = findClusters(at2, 200, { isolation: at4, minSplats: 3 });
  assert(none.small === undefined, 'without small: none');
  const noIso = findClusters(at2, 200, { minSplats: 3, small: true });
  assert(noIso.small.join() === `${s.floor + 27},${s.floor + 28},${far},${low}`, `no isolation: ${noIso.small}`);
});
