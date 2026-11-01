// Each view's exposure (exposure.js, DESIGN.md E19): Adam on tile_backward's sums recovers known gains and offsets
// (about mid-grey: c' = g (c - 1/2) + 1/2 + o), the gauge keeps the mean log-gain and offset at 0, a snapshot's JSON
// gives the same uniforms back.
import { Exposure, EXPO_FLOATS } from '../src/exposure.js';
import { assert, assertClose } from './util.js';

// tile_backward's sums for one view: the render c (the splats, exact), the photo G c + O, the loss 0.8 smooth-L1 + 0.4 L2
// (dL/dc' per pixel and channel, over the pixel count), dL/dlog g = dL/dc' (c' - b), dL/db = dL/dc'
function sums(e, name, colors, G, O) {
  const [g0, g1, g2, , b0, b1, b2] = e.uniforms(name), g = [g0, g1, g2], b = [b0, b1, b2];
  const out = new Float32Array(EXPO_FLOATS), N = colors.length / 3;
  for (let p = 0; p < N; p++) for (let c = 0; c < 3; c++) {
    const hat = g[c] * colors[3 * p + c] + b[c], diff = hat - (G[c] * colors[3 * p + c] + O[c]);
    const d = (0.8 * Math.max(-1, Math.min(1, 25 * diff)) + 0.4 * diff) / N;
    out[c] += d * (hat - b[c]);
    out[4 + c] += d;
  }
  return out;
}

Deno.test('exposure: known gains and offsets recovered within 2%, the mean log-gain and offset at 0', () => {
  let s = 7;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const colors = Float64Array.from({ length: 3 * 2000 }, () => 0.1 + 0.6 * rnd());
  const V = 12, lg = Array.from({ length: V }, () => [0, 1, 2].map(() => 0.2 * (rnd() - 0.5)));
  const off = Array.from({ length: V }, () => [0, 1, 2].map(() => 0.06 * (rnd() - 0.5)));
  for (const t of [lg, off]) for (let c = 0; c < 3; c++) {  // the truth in the gauge (o's mean 0)
    const m = t.reduce((a, r) => a + r[c], 0) / V;
    for (const r of t) r[c] -= m;
  }
  const b = (v) => off[v].map((o, c) => o + 0.5 * (1 - Math.exp(lg[v][c])));  // the photo's g c + b
  const e = new Exposure();
  for (let round = 0; round < 300; round++) {
    for (let v = 0; v < V; v++) e.update(`v${v}`, sums(e, `v${v}`, colors, lg[v].map(Math.exp), b(v)), 0.01);
    e.center();
  }
  let worst = 0, worstB = 0;
  for (let v = 0; v < V; v++) {
    const u = e.uniforms(`v${v}`);
    for (let c = 0; c < 3; c++) {
      worst = Math.max(worst, Math.abs(u[c] / Math.exp(lg[v][c]) - 1));
      worstB = Math.max(worstB, Math.abs(u[4 + c] - b(v)[c]));
    }
  }
  assert(worst < 0.02, `gains off by up to ${(100 * worst).toFixed(2)}%`);
  assert(worstB < 0.01, `offsets off by up to ${worstB.toFixed(4)}`);
  for (let j = 0; j < 6; j++) {
    const m = [...e.views.values()].reduce((a, st) => a + st.p[j], 0) / V;
    assertClose(m, 0, 1e-12, `the gauge, parameter ${j}`);
  }
  const back = Exposure.fromJSON(JSON.parse(JSON.stringify(e.toJSON())));
  for (let v = 0; v < V; v++) e.uniforms(`v${v}`).forEach((x, j) => assertClose(back.uniforms(`v${v}`)[j], x, 1e-9, `v${v} [${j}] from JSON`));
  assert(new Exposure().uniforms('held out').join() === '1,1,1,1,0,0,0,0', 'no view with an exposure: identity');
});

Deno.test('exposure: about mid-grey, a low-contrast view converges as fast (gain and offset not trading)', () => {
  let s = 11;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const colors = Float64Array.from({ length: 3 * 2000 }, () => 0.4 + 0.2 * rnd());
  const V = 12, lg = Array.from({ length: V }, () => [0, 1, 2].map(() => 0.2 * (rnd() - 0.5)));
  for (let c = 0; c < 3; c++) { const m = lg.reduce((a, r) => a + r[c], 0) / V; for (const r of lg) r[c] -= m; }
  const e = new Exposure();
  // 100 visits a view: 0.05% off (about 0, the shader's c' = g c + b as the parameters: 1.0%)
  for (let round = 0; round < 100; round++) {
    for (let v = 0; v < V; v++) e.update(`v${v}`, sums(e, `v${v}`, colors, lg[v].map(Math.exp), lg[v].map(x => 0.5 * (1 - Math.exp(x)))), 0.01);
    e.center();
  }
  let worst = 0;
  for (let v = 0; v < V; v++) for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(e.uniforms(`v${v}`)[c] / Math.exp(lg[v][c]) - 1));
  assert(worst < 0.003, `gains off by up to ${(100 * worst).toFixed(2)}%`);
});

Deno.test('exposure: a view without one takes its neighbours\' in name order, by frame number', () => {
  const e = Exposure.fromJSON({ 't0010.jpg': [1, 1, 1, 0, 0, 0], 't0020.jpg': [1.2, 1.1, 0.9, 0.1, 0, -0.1], 't0100.jpg': [0.8, 1, 1, 0, 0, 0] });
  const p = (n) => e.uniforms(n);
  const lerp = (t) => [0, 1, 2].map(c => Math.exp((1 - t) * Math.log(p('t0010.jpg')[c]) + t * Math.log(p('t0020.jpg')[c])));
  lerp(0.3).forEach((g, c) => assertClose(p('t0013.jpg')[c], g, 1e-12, `t0013's gain ${c}: 0.3 of the way (log)`));
  p('t0001.jpg').forEach((x, j) => assertClose(x, p('t0010.jpg')[j], 1e-12, `before the first: the first's [${j}]`));
  p('t0200.jpg').forEach((x, j) => assertClose(x, p('t0100.jpg')[j], 1e-12, `after the last: the last's [${j}]`));
  // 3 of 8 of the way from t0020 (1.2) to t0100 (0.8) in log
  assertClose(p('t0050.jpg')[0], Math.exp(0.625 * Math.log(1.2) + 0.375 * Math.log(0.8)), 1e-12, 't0050 by frame number');
  assertClose(Exposure.fromJSON({ a: [1.2, 1, 1, 0, 0, 0], c: [0.8, 1, 1, 0, 0, 0] }).uniforms('b')[0], Math.sqrt(1.2 * 0.8), 1e-12, 'no frame numbers: halfway');
});
