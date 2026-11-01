// The sky dome from the photos alone (src/dome.js): synthetic views along a path, a sky that depends on the
// direction only (far), ground and walls that depend on where the camera is (near), each view under its own
// exposure gain.
import { buildDome, cubeDir, fillDome } from '../src/dome.js';
import { assert } from './util.js';

const UP = [0, -1, 0];  // the cameras' world: +y down
const sky = (d) => {  // a direction's sky: bluer and darker up high
  const e = Math.asin(-d[1] / Math.hypot(...d));
  return [0.55 - 0.3 * e / (Math.PI / 2), 0.7 - 0.2 * e / (Math.PI / 2), 0.95];
};
const checker = (a, b) => ((Math.floor(a * 2) + Math.floor(b * 2)) & 1 ? [0.2, 0.5, 0.15] : [0.6, 0.4, 0.2]);

// k cameras along x, turning, pitched up by pitchDeg; rays above 3 deg of elevation see the sky when `withSky`
function views({ K = 24, w = 64, h = 48, f = 40, pitchDeg = 10, withSky = true, flatGround = false, gains = (k) => 1 + 0.15 * Math.sin(k) }) {
  const out = [];
  for (let k = 0; k < K; k++) {
    const yaw = k * 0.45, ph = pitchDeg * Math.PI / 180, p = [0.4 * k, 0, 0];
    const fw = [Math.sin(yaw) * Math.cos(ph), -Math.sin(ph), Math.cos(yaw) * Math.cos(ph)];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let x = cross(fw, UP); const lx = Math.hypot(...x); x = x.map(v => v / lx);
    const y = cross(fw, x), R = [...x, ...y, ...fw];
    const rgba = new Uint8Array(4 * w * h), g = gains(k);
    for (let py = 0; py < h; py++) for (let px = 0; px < w; px++) {
      const rx = (px + 0.5 - w / 2) / f, ry = (py + 0.5 - h / 2) / f;
      const d = [0, 1, 2].map(i => R[i] * rx + R[3 + i] * ry + R[6 + i]);
      const e = Math.asin(-d[1] / Math.hypot(...d)) * 180 / Math.PI;
      let c;
      if (withSky && e > 3) c = sky(d);
      else if (flatGround && e < 0) c = [0.1, 0.12, 0.08];  // a shadow: the same from everywhere
      else if (d[1] > 0.05) { const t = 1 / d[1]; c = checker(p[0] + t * d[0], p[2] + t * d[2]); }  // the ground, 1 below
      else { const t = 3 / Math.hypot(d[0], d[2]); c = checker(p[0] + t * d[0] + 7 * d[1], p[2] + t * d[2]); }  // a wall around, 3 away
      for (let ch = 0; ch < 3; ch++) rgba[4 * (py * w + px) + ch] = Math.round(255 * Math.min(1, c[ch] * g));
      rgba[4 * (py * w + px) + 3] = 255;
    }
    out.push({ R, fx: f, fy: f, cx: w / 2, cy: h / 2, w, h, rgba });
  }
  return out;
}
const texel = (dome, d) => {  // the cube texel nearest direction d: the face of its largest axis, then u, v
  const n = dome.n;
  let best = -1, bi = 0;
  for (let fc = 0; fc < 6; fc++) for (let r = 0; r < n; r += 4) for (let c = 0; c < n; c += 4) {
    const t = cubeDir(fc, r, c, n), l = Math.hypot(...t), dd = Math.hypot(...d);
    const cos = (t[0] * d[0] + t[1] * d[1] + t[2] * d[2]) / l / dd;
    if (cos > best) { best = cos; bi = (fc * n + r) * n + c; }
  }
  return [0, 1, 2].map(ch => dome.cube[3 * bi + ch] / 255);
};
const opts = { n: 32, width: 256, minPatchDeg2: 20 };

Deno.test('dome: the sky found, its exposure gains, the zenith filled; the walls and ground left out', async () => {
  const d = await buildDome(views({}), UP, opts);
  assert(d.worth && d.farShare > 0.1 && d.farShare < 0.6, `far share ${d.farShare}`);
  // each view's gain (1 + 0.15 sin k), over their median as the estimates are
  const g = Array.from({ length: 24 }, (_, k) => 1 + 0.15 * Math.sin(k)), med = [...g].sort((a, b) => a - b)[12];
  const worst = Math.max(...g.map((x, k) => Math.abs(d.viewGains[k] / (x / med) - 1)));
  assert(worst < 0.02, `the views' gains off by up to ${(100 * worst).toFixed(1)}%`);
  // the sky at 30 deg up, every direction, filled everywhere (fill 'all'); the zenith (no camera sees above 41 deg):
  // the sky up there, roughly, with either fill
  const all = await buildDome(views({}), UP, { ...opts, fill: 'all' });
  for (const yaw of [0, 1.5, 3, 4.5]) {
    const dir = [Math.sin(yaw) * Math.cos(Math.PI / 6), -Math.sin(Math.PI / 6), Math.cos(yaw) * Math.cos(Math.PI / 6)];
    const got = texel(all, dir), want = sky(dir);
    assert(got.every((v, i) => Math.abs(v - want[i]) < 0.05), `30 deg up, yaw ${yaw}: ${got} for ${want}`);
  }
  const z = texel(d, [0, -1, 0]), want = sky([0, -Math.sin(0.7), Math.cos(0.7)]);  // never seen: filled
  assert(z.every((v, i) => Math.abs(v - want[i]) < 0.1), `zenith ${z} against ${want}`);
  // 9 deg below the horizon (the cameras see down to -21 deg): seen all around, never far (magenta in the observed map)
  const { W, H, observed } = d.latlong, row = Math.floor(H * 0.55);
  for (let x = 0; x < W; x++) {
    const s = row * W + x;
    assert(observed[3 * s] === 204 && observed[3 * s + 1] === 0, `the ground at column ${x}: ${observed.slice(3 * s, 3 * s + 3)}`);
  }
  // seen, not far: black in the dome (the splats' own), unless fill 'all'
  const gp = Math.floor(H * 0.55) * W + 10, black = d.latlong.filled.slice(3 * gp, 3 * gp + 3).every(v => v === 0);
  assert(black && all.latlong.filled.slice(3 * gp, 3 * gp + 3).some(v => v > 50), `the ground in the dome: ${d.latlong.filled.slice(3 * gp, 3 * gp + 3)}, filled ${all.latlong.filled.slice(3 * gp, 3 * gp + 3)}`);
});

Deno.test('dome: filled again from its maps; the panorama fill keeps every seen direction its color', async () => {
  const d = await buildDome(views({}), UP, opts), all = await buildDome(views({}), UP, { ...opts, fill: 'all' });
  // the same dome as built with that fill, without the photos
  const again = fillDome(d.maps, 'all', d.n);
  assert(again.cube.every((v, i) => v === all.cube[i]), 'fillDome(all) is not buildDome with fill all');
  // 'seen': the ground (seen, not far) its own mean color, not black nor the sky's; the sky as found; the zenith filled
  const seen = fillDome(d.maps, 'seen', d.n), { W, H } = d.latlong, gp = Math.floor(H * 0.55) * W + 10;
  const own = [0, 1, 2].map(c => Math.round(255 * Math.min(1, d.maps.mean[3 * gp + c])));
  const got = Array.from(seen.filled.slice(3 * gp, 3 * gp + 3));
  assert(got.every((v, c) => v === own[c]) && got.some(v => v > 50), `the ground in the panorama: ${got}, its mean ${own}`);
  const dir = [0, -Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)], sk = texel({ n: d.n, cube: seen.cube }, dir), want = sky(dir);
  assert(sk.every((v, i) => Math.abs(v - want[i]) < 0.05), `the sky 30 deg up in the panorama: ${sk} for ${want}`);
  const z = texel({ n: d.n, cube: seen.cube }, [0, -1, 0]);
  assert(z.every(v => v > 0.05), `the zenith in the panorama: ${z}`);
  let threw = false;
  try { fillDome(d.maps, 'nope', d.n); } catch { threw = true; }
  assert(threw, 'an unknown fill did not throw');
});

Deno.test('dome: not worth it in a closed room', async () => {
  const d = await buildDome(views({ withSky: false }), UP, opts);
  assert(!d.worth && d.farShare < 0.01, `far share ${d.farShare}`);
});

Deno.test('dome: a uniform dark ground agrees across the views but is below the horizon: not far', async () => {
  const d = await buildDome(views({ flatGround: true }), UP, opts);
  const { W, H, observed } = d.latlong, row = Math.floor(H * 0.55);
  let far = 0;
  for (let x = 0; x < W; x++) if (observed[3 * (row * W + x) + 1] !== 0) far++;
  assert(d.worth && far === 0, `${far} of ${W} texels 9 deg below the horizon far`);
});
