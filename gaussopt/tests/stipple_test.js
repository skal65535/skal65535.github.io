// stipple.js (E4 seeding): the Voronoi of the Lloyd steps only assigns the pixels with density, the others
// weigh nothing; the sites and cells are those of the full assignment.
import { residualDensity, stippleSites } from '../src/stipple.js';
import { computeVoronoi } from '../../stipple/stipple.js';
import { makeRng } from '../src/math.js';
import { assert } from './util.js';

// A residual like a training view's: an under-fit blob, a thin edge and noise, about a third of it above the mean
function density(W, H) {
  const rng = makeRng(3), gt = new Uint8Array(W * H * 4), hat = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4, blob = Math.exp(-((x - 0.3 * W) ** 2 + (y - 0.6 * H) ** 2) / (0.02 * W * W));
    const edge = Math.abs(x - 0.7 * W - 0.1 * y) < 2 ? 1 : 0, e = 120 * blob + 90 * edge + 12 * rng();
    for (let c = 0; c < 3; c++) { gt[o + c] = 100; hat[o + c] = Math.min(255, 100 + e); }
  }
  return residualDensity(gt, hat, W, H);
}

// stippleSites with every pixel assigned (the reference)
function sitesFull(dens, W, H, N, lloydIters = 4, seed = 1234) {
  const cdf = new Float64Array(W * H);
  let acc = 0;
  for (let i = 0; i < W * H; i++) { acc += dens[i]; cdf[i] = acc; }
  const rand = makeRng(seed), pts = [];
  for (let n = 0; n < N; n++) {
    const u = rand() * acc;
    let lo = 0, hi = W * H - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    pts.push({ x: ((lo % W) + rand()) / W, y: (((lo / W) | 0) + rand()) / H });
  }
  let area = null;
  for (let it = 0; it <= lloydIters; it++) {
    const owner = computeVoronoi(W, H, pts);
    const m = new Float64Array(N), mx = new Float64Array(N), my = new Float64Array(N);
    area = new Float32Array(N);
    for (let y = 0, i = 0; y < H; y++) for (let x = 0; x < W; x++, i++) {
      const c = owner[i], r = dens[i];
      if (r > 0) area[c]++;
      const w = r * r;
      m[c] += w; mx[c] += w * (x + 0.5); my[c] += w * (y + 0.5);
    }
    if (it === lloydIters) break;
    for (let n = 0; n < N; n++) if (m[n] > 0) pts[n] = { x: mx[n] / m[n] / W, y: my[n] / m[n] / H };
  }
  return { xs: Float32Array.from(pts, p => p.x), ys: Float32Array.from(pts, p => p.y), area };
}

Deno.test('stipple: the masked Voronoi assigns the pixels with density as the full one, and the sites are the same', () => {
  const W = 240, H = 180, d = density(W, H);
  let zero = 0;
  for (const r of d) if (!(r > 0)) zero++;
  assert(zero > 0.3 * W * H && zero < 0.9 * W * H, `${zero} / ${W * H} pixels without density: not a test of the mask`);

  const rng = makeRng(5), pts = Array.from({ length: 300 }, () => ({ x: rng(), y: rng() }));
  const full = computeVoronoi(W, H, pts), masked = computeVoronoi(W, H, pts, d);
  let diff = 0;
  for (let i = 0; i < W * H; i++) if (d[i] > 0 ? masked[i] !== full[i] : masked[i] !== 0) diff++;
  assert(diff === 0, `${diff} pixels assigned differently by the masked Voronoi`);

  const a = stippleSites(d, W, H, 500), b = sitesFull(d, W, H, 500);
  for (const k of ['xs', 'ys', 'area']) {
    let n = 0;
    for (let i = 0; i < a[k].length; i++) if (a[k][i] !== b[k][i]) n++;
    assert(a[k].length === 500 && n === 0, `stippleSites ${k}: ${n} of ${a[k].length} differ from the full assignment`);
  }
});
