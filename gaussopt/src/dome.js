// dome.js — The sky dome (the Dome phase, before Train): a background at infinity made from the photos alone,
// before any splat, then fixed while the splats train (trainer.setDome: env.js's cube map at n texels a face).
// Every pixel of the training views goes along its ray onto a lat-long map around the cameras' up. Per view a gain
// (auto exposure), per texel a trimmed mean over the views. A texel is far when enough views see it and they agree
// (no parallax over the camera path), above the horizon (below it, dark ground agrees too), in a patch of some
// size (a dark trunk or roof edge agreeing across views is not sky). The dome is worth it when the far texels take
// a share of the photos' pixels; the directions no camera saw (the zenith) are filled from the far texels by a rough
// diffusion (push-pull), the ones seen but not far are black (fill 'unseen': the splats' own, as without a dome; 'all'
// filled too lost 0.33 dB held out on canyon3, 0.08 so); fill 'seen': every seen direction its own color (what the
// views saw there, far or not: the near content as billboards at infinity), only the unseen ones diffused from them.
// fillDome fills a built dome again (another fill) without the photos. photo_runs/tools/dome.py is the same offline.
// DOM-free (tests/dome_test.js).
"use strict";

export const DOME_DEFAULTS = {
  n: 256,           // the cube map's texels a face
  width: 1024,      // the lat-long map's (height: half); its texel 360 / width degrees
  tau: 0.03,        // the views' spread (rgb in [0, 1], their channels' mean) under which a texel is far
  worth: 0.01,      // the share of the photos' pixels on far texels for a dome
  minViews: 5,      // views a far texel needs
  inlierFrac: 0.75, // of its samples near its mean
  minElevDeg: 0,    // far texels above this elevation only
  minPatchDeg2: 6,  // far patches under this many square degrees dropped
  iterations: 4,    // gain / mean rounds
  fill: 'unseen',   // DOME_FILLS: 'unseen' (the far texels' colors spread where no camera looked; the seen ones black) | 'all' | 'seen'
};
// fill -> what it does (the Dome step's choice, params.js dome_fill)
export const DOME_FILLS = {
  unseen: 'sky only: the sky found, spread where no camera looked; what the cameras saw that is not sky left to the splats',
  seen: 'panorama: every direction a camera saw, its color (the far sky and the near things as billboards at infinity, behind the splats); the rest spread from it',
  all: 'sky everywhere: the sky found, spread over every other direction (seen or not)',
};
const LUM = [0.299, 0.587, 0.114];

// up (unit, world) -> { x, y, z }: the lat-long frame, y up
export function domeFrame(up) {
  const l = Math.hypot(...up), y = up.map(v => v / l);
  const a = Math.abs(y[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
  const d = a[0] * y[0] + a[1] * y[1] + a[2] * y[2];
  let x = a.map((v, i) => v - d * y[i]); const lx = Math.hypot(...x); x = x.map(v => v / lx);
  const z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  return { x, y, z };
}

// world direction (not unit) -> lat-long texel (W x H), and the (fractional) map coordinates
function latlong(dx, dy, dz, F, W, H) {
  const l = Math.hypot(dx, dy, dz);
  const a = (dx * F.x[0] + dy * F.x[1] + dz * F.x[2]) / l, b = (dx * F.y[0] + dy * F.y[1] + dz * F.y[2]) / l,
        c = (dx * F.z[0] + dy * F.z[1] + dz * F.z[2]) / l;
  const lon = Math.atan2(a, -c), lat = Math.asin(Math.max(-1, Math.min(1, b)));
  return [(lon / (2 * Math.PI) + 0.5) * W, (0.5 - lat / Math.PI) * H];
}

// The far texels' patches of at least nMin texels (4-connected, longitude wrapping), in place
function keepBigPatches(far, W, H, nMin) {
  const seen = new Uint8Array(W * H), stack = [], patch = [];
  for (let s = 0; s < W * H; s++) {
    if (!far[s] || seen[s]) continue;
    seen[s] = 1; stack.push(s); patch.length = 0;
    while (stack.length) {
      const t = stack.pop(), y = Math.floor(t / W), x = t - y * W;
      patch.push(t);
      for (const u of [y > 0 ? t - W : -1, y < H - 1 ? t + W : -1, y * W + (x + 1) % W, y * W + (x + W - 1) % W]) {
        if (u >= 0 && far[u] && !seen[u]) { seen[u] = 1; stack.push(u); }
      }
    }
    if (patch.length < nMin) for (const t of patch) far[t] = 0;
  }
}

// Fill where w = 0 from where w > 0 (v: W x H x 3 float, w: W x H in [0, 1]): a pyramid of weighted means, each
// level's holes from the coarser one upsampled and blurred (3 x 3, twice; longitude wraps)
export function pushPull(v, w, W, H) {
  let total = 0;
  for (let i = 0; i < W * H; i++) total += w[i];
  if (W <= 2 || H <= 2 || total === 0) {
    const m = [0, 0, 0];
    for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) m[c] += v[3 * i + c] * w[i];
    const out = new Float32Array(3 * W * H);
    for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = total > 0 ? m[c] / total : 0;
    return out;
  }
  const w2 = Math.ceil(W / 2), h2 = Math.ceil(H / 2), vs = new Float32Array(3 * w2 * h2), ws = new Float32Array(w2 * h2);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, j = (y >> 1) * w2 + (x >> 1);
    ws[j] += w[i];
    for (let c = 0; c < 3; c++) vs[3 * j + c] += v[3 * i + c] * w[i];
  }
  for (let j = 0; j < w2 * h2; j++) {
    if (ws[j] > 0) for (let c = 0; c < 3; c++) vs[3 * j + c] /= ws[j];
    ws[j] = Math.min(1, ws[j]);
  }
  const coarse = pushPull(vs, ws, w2, h2);
  let up = new Float32Array(3 * W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const j = (y >> 1) * w2 + (x >> 1);
    for (let c = 0; c < 3; c++) up[3 * (y * W + x) + c] = coarse[3 * j + c];
  }
  for (let pass = 0; pass < 2; pass++) {
    const b = new Float32Array(3 * W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      for (let dy = -1; dy <= 1; dy++) {
        const yy = Math.max(0, Math.min(H - 1, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const s = 3 * (yy * W + (x + dx + W) % W);
          for (let c = 0; c < 3; c++) b[3 * (y * W + x) + c] += up[s + c] / 9;
        }
      }
    }
    up = b;
  }
  for (let i = 0; i < W * H; i++) {
    const a = Math.min(1, Math.max(0, w[i]));
    for (let c = 0; c < 3; c++) up[3 * i + c] = a * v[3 * i + c] + (1 - a) * up[3 * i + c];
  }
  return up;
}

// The cube map texel (face, row, col)'s direction (env.js env_taps inverted; faces +x -x +y -y +z -z)
export function cubeDir(f, row, col, n) {
  const v = (row + 0.5) / n * 2 - 1, u = (col + 0.5) / n * 2 - 1;
  return [[1, -v, -u], [-1, -v, u], [u, 1, v], [u, -1, -v], [u, -v, 1], [-u, -v, -1]][f];
}

// views: [{ R (row-major world to camera, +z ahead, +y down), fx, fy, cx, cy (at the pixels' size), w, h,
//           rgba: Uint8Array w x h x 4 (alpha 0: not a photo pixel, an on-screen display) }]
// up: the world's up. Returns { worth, farShare, farTexels, seen, gains: [p5, p95], viewGains (each view's, median 1), fill, n, cube: Uint8Array(18 n^2) rgb,
// latlong: { W, H, observed, filled: Uint8Array rgb (observed: the far texels, magenta where seen and not far) }, maps (fillDome's) }.
// onProgress(fraction) is awaited now and then (the page stays responsive).
export async function buildDome(views, up, opts = {}, onProgress = null) {
  const o = { ...DOME_DEFAULTS, ...opts }, W = o.width, H = W / 2, F = domeFrame(up), K = views.length;
  // each view's pixels' texels, once
  const tex = views.map((v) => {
    const t = new Int32Array(v.w * v.h), R = v.R;
    for (let y = 0; y < v.h; y++) for (let x = 0; x < v.w; x++) {
      const i = y * v.w + x;
      if (v.rgba[4 * i + 3] < 128) { t[i] = -1; continue; }
      const rx = (x + 0.5 - v.cx) / v.fx, ry = (y + 0.5 - v.cy) / v.fy;  // R^T (rx, ry, 1)
      const [mx, my] = latlong(R[0] * rx + R[3] * ry + R[6], R[1] * rx + R[4] * ry + R[7], R[2] * rx + R[5] * ry + R[8], F, W, H);
      t[i] = Math.min(H - 1, Math.max(0, Math.floor(my))) * W + Math.min(W - 1, Math.max(0, Math.floor(mx)));
    }
    return t;
  });
  const nViews = new Uint16Array(W * H), stamp = new Int32Array(W * H).fill(-1);
  tex.forEach((t, k) => { for (const s of t) if (s >= 0 && stamp[s] !== k) { stamp[s] = k; nViews[s]++; } });
  const above = (s) => (0.5 - (Math.floor(s / W) + 0.5) / H) * 180 >= o.minElevDeg;
  const gain = new Float32Array(K).fill(1);
  const cnt = new Float32Array(W * H), sum = new Float32Array(3 * W * H), sq = new Float32Array(3 * W * H);
  const mean = new Float32Array(3 * W * H), std = new Float32Array(W * H), ci = new Float32Array(W * H);
  const far = new Uint8Array(W * H), plain = new Float32Array(3 * W * H);
  const each = (fn) => views.forEach((v, k) => { const t = tex[k], g = gain[k]; for (let i = 0; i < t.length; i++) if (t[i] >= 0) fn(t[i], v.rgba, i, g, k); });
  const stats = (useInliers) => {
    cnt.fill(0); sum.fill(0); sq.fill(0);
    each((s, px, i, g) => {
      const r = px[4 * i] / 255 / g, gr = px[4 * i + 1] / 255 / g, b = px[4 * i + 2] / 255 / g;
      if (useInliers && Math.max(Math.abs(r - mean[3 * s]), Math.abs(gr - mean[3 * s + 1]), Math.abs(b - mean[3 * s + 2])) >= Math.max(2 * std[s], 0.04)) return;
      cnt[s]++; sum[3 * s] += r; sum[3 * s + 1] += gr; sum[3 * s + 2] += b; sq[3 * s] += r * r; sq[3 * s + 1] += gr * gr; sq[3 * s + 2] += b * b;
    });
    for (let s = 0; s < W * H; s++) {
      const n = Math.max(cnt[s], 1);
      let sd = 0;
      for (let c = 0; c < 3; c++) { const m = sum[3 * s + c] / n; mean[3 * s + c] = m; sd += Math.sqrt(Math.max(0, sq[3 * s + c] / n - m * m)); }
      std[s] = sd / 3;
    }
  };
  for (let it = 0; it < o.iterations; it++) {
    stats(false);
    const all = Float32Array.from(cnt);
    plain.set(mean);
    stats(true);
    ci.set(cnt);
    for (let s = 0; s < W * H; s++) far[s] = above(s) && nViews[s] >= o.minViews && ci[s] >= o.inlierFrac * Math.max(all[s], 1) && std[s] < o.tau ? 1 : 0;
    // each view's gain: its median ratio to the texels' means on the far texels
    const ratios = Array.from({ length: K }, () => []);
    views.forEach((v, k) => {
      const t = tex[k];
      for (let i = 0; i < t.length; i++) {
        const s = t[i];
        if (s < 0 || !far[s]) continue;
        const lm = LUM[0] * mean[3 * s] + LUM[1] * mean[3 * s + 1] + LUM[2] * mean[3 * s + 2];
        const lc = (LUM[0] * v.rgba[4 * i] + LUM[1] * v.rgba[4 * i + 1] + LUM[2] * v.rgba[4 * i + 2]) / 255;
        ratios[k].push(lc / Math.max(lm, 1e-3));
      }
    });
    ratios.forEach((r, k) => { if (r.length >= 50) { r.sort((a, b) => a - b); gain[k] = r[r.length >> 1]; } });
    const med = Float32Array.from(gain).sort()[K >> 1] || 1;
    for (let k = 0; k < K; k++) gain[k] /= med;
    await onProgress?.((it + 1) / (o.iterations + 1));
  }
  // off by one texel at the silhouettes, then the small patches dropped
  const eroded = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const s = y * W + x;
    eroded[s] = far[s] && y > 0 && y < H - 1 && far[s - W] && far[s + W] && far[y * W + (x + 1) % W] && far[y * W + (x + W - 1) % W] ? 1 : 0;
  }
  keepBigPatches(eroded, W, H, Math.ceil(o.minPatchDeg2 / (360 / W) ** 2));
  far.set(eroded);
  let onFar = 0, total = 0, farTexels = 0, seen = 0;
  for (const t of tex) for (const s of t) if (s >= 0) { total++; onFar += far[s]; }
  for (let s = 0; s < W * H; s++) { farTexels += far[s]; seen += nViews[s] > 0 ? 1 : 0; }
  // a texel whose views all disagree (near content, parallax) keeps no sample: its plain mean (the panorama's color there)
  for (let s = 0; s < W * H; s++) if (ci[s] === 0) for (let c = 0; c < 3; c++) mean[3 * s + c] = plain[3 * s + c];
  const observed = new Uint8Array(3 * W * H);
  for (let s = 0; s < W * H; s++) for (let c = 0; c < 3; c++)
    observed[3 * s + c] = far[s] ? Math.round(255 * Math.min(1, mean[3 * s + c])) : nViews[s] > 0 ? [204, 0, 204][c] : 0;
  const maps = { W, H, F, mean, far, nViews };
  const { cube, filled } = fillDome(maps, o.fill, o.n);
  await onProgress?.(1);
  const sortedGain = Float32Array.from(gain).sort();
  return {
    worth: total > 0 && onFar / total >= o.worth, farShare: total > 0 ? onFar / total : 0, farTexels: farTexels / Math.max(seen, 1),
    seen: seen / (W * H), gains: [sortedGain[Math.floor(0.05 * (K - 1))], sortedGain[Math.ceil(0.95 * (K - 1))]], viewGains: gain,
    fill: o.fill, n: o.n, cube, latlong: { W, H, observed, filled }, maps,
  };
}

// A built dome's maps (buildDome's maps: the lat-long frame F, the texels' means, far, views) filled (DOME_FILLS) into
// an n x n a face cube map: { cube: Uint8Array(18 n^2) rgb, filled: Uint8Array rgb, the lat-long map }
export function fillDome({ W, H, F, mean, far, nViews }, fill, n) {
  if (!(fill in DOME_FILLS)) throw new Error(`dome fill '${fill}': not one of ${Object.keys(DOME_FILLS).join(', ')}`);
  const w = new Float32Array(W * H);  // the texels kept as they are, the rest diffused from them
  for (let s = 0; s < W * H; s++) w[s] = fill === 'seen' ? (nViews[s] > 0 ? 1 : 0) : far[s];
  const filled = pushPull(mean, w, W, H);
  if (fill === 'unseen') for (let s = 0; s < W * H; s++) if (nViews[s] > 0 && !far[s]) filled[3 * s] = filled[3 * s + 1] = filled[3 * s + 2] = 0;
  const filled8 = new Uint8Array(3 * W * H);
  for (let i = 0; i < 3 * W * H; i++) filled8[i] = Math.round(255 * Math.min(1, Math.max(0, filled[i])));
  // the cube: each texel's direction back to the filled map, bilinear (longitude wrapping)
  const cube = new Uint8Array(18 * n * n);
  for (let f = 0; f < 6; f++) for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
    const d = cubeDir(f, row, col, n), [mx, my] = latlong(d[0], d[1], d[2], F, W, H);
    const x = mx - 0.5, y = Math.min(H - 1.001, Math.max(0, my - 0.5)), x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const at = (yy, xx, c) => filled[3 * (yy * W + ((xx % W) + W) % W) + c];
    for (let c = 0; c < 3; c++) {
      const val = at(y0, x0, c) * (1 - fx) * (1 - fy) + at(y0, x0 + 1, c) * fx * (1 - fy) + at(y0 + 1, x0, c) * (1 - fx) * fy + at(y0 + 1, x0 + 1, c) * fx * fy;
      cube[3 * ((f * n + row) * n + col) + c] = Math.round(255 * Math.min(1, Math.max(0, val)));
    }
  }
  return { cube, filled: filled8 };
}
