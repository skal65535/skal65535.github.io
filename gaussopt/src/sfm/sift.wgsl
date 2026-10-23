// sift.wgsl — SIFT on the GPU (DESIGN-sfm §5.2, S1), following VLFeat as configured by COLMAP:
// octaves of S + 3 Gaussian levels (sigma_k = sigma0 2^(k/S)), S + 2 difference-of-Gaussian
// levels, extrema over 26 neighbours with sub-pixel refinement, contrast and edge tests, 36-bin
// orientation histograms (up to 2 orientations), 4x4x8 descriptors normalized like COLMAP
// (L2, clamp 0.2, L2, then L1-root, x 512 -> u8). Each octave's Gaussian and DoG levels live in
// one stacked buffer (level k at offset k w h). Entry points share @group(0) bindings by number;
// each pipeline only uses its own.

struct Params {
  w: u32, h: u32, radius: u32, src_off: u32,
  dst_off: u32, level: u32, octave: u32, levels: u32,   // levels = S
  sigma0: f32, peak: f32, edge: f32, scale: f32,         // scale = 2^octave (image px per octave px)
  max_kp: u32, _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> P: Params;

// ── to_float: u8 gray (packed 4 per u32) -> f32 in [0, 1] ───────────────────
@group(0) @binding(1) var<storage, read> gray_u8: array<u32>;
@group(0) @binding(2) var<storage, read_write> img_out: array<f32>;

@compute @workgroup_size(256)
fn to_float(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= P.w * P.h) { return; }
  let b = (gray_u8[i / 4u] >> (8u * (i % 4u))) & 0xffu;
  img_out[P.dst_off + i] = f32(b) / 255.0;
}

// ── separable Gaussian blur, clamped borders: stack[dst] = blur(stack[src]) ──
@group(0) @binding(3) var<storage, read_write> stack: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> tmp: array<f32>;

@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.w || g.y >= P.h) { return; }
  let r = i32(P.radius);
  var s = 0.0;
  for (var k = -r; k <= r; k++) {
    let x = clamp(i32(g.x) + k, 0, i32(P.w) - 1);
    s += weights[u32(k + r)] * stack[P.src_off + g.y * P.w + u32(x)];
  }
  tmp[g.y * P.w + g.x] = s;
}

@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.w || g.y >= P.h) { return; }
  let r = i32(P.radius);
  var s = 0.0;
  for (var k = -r; k <= r; k++) {
    let y = clamp(i32(g.y) + k, 0, i32(P.h) - 1);
    s += weights[u32(k + r)] * tmp[u32(y) * P.w + g.x];
  }
  stack[P.dst_off + g.y * P.w + g.x] = s;
}

// ── downsample by 2 (every other pixel): next octave base from this octave's level S ──
@group(0) @binding(6) var<storage, read> src_stack: array<f32>;

@compute @workgroup_size(16, 16)
fn downsample(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.w || g.y >= P.h) { return; }  // P.w, P.h: destination size
  let sw = P.radius;                          // source width (radius slot reused)
  stack[P.dst_off + g.y * P.w + g.x] = src_stack[P.src_off + (2u * g.y) * sw + 2u * g.x];
}

// ── DoG: dog[k] = gauss[k + 1] - gauss[k], one level (P.level) per dispatch ──
@group(0) @binding(7) var<storage, read_write> dog: array<f32>;

@compute @workgroup_size(256)
fn diff(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = P.w * P.h;
  if (g.x >= n) { return; }
  let k = P.level;
  dog[k * n + g.x] = src_stack[(k + 1u) * n + g.x] - src_stack[k * n + g.x];
}

// ── extrema + refinement: one thread per pixel of DoG level P.level (1 .. S) ──
// keypoint record (8 f32): x, y (octave px), s (level, refined), octave, sigma (image px), response, angle, 0
@group(0) @binding(8) var<storage, read_write> kp: array<f32>;
// counters: [0] candidates, [1] oriented, [2 + 2 o] / [3 + 2 o] both counts before octave o
@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(1)
fn snapshot() {
  atomicStore(&counters[2u + 2u * P.octave], atomicLoad(&counters[0]));
  atomicStore(&counters[3u + 2u * P.octave], atomicLoad(&counters[1]));
}

fn D(x: i32, y: i32, s: i32) -> f32 {
  let n = P.w * P.h;
  return src_stack[u32(s) * n + u32(y) * P.w + u32(x)];
}

@compute @workgroup_size(16, 16)
fn extrema(@builtin(global_invocation_id) g: vec3<u32>) {
  let b = 5;  // border
  var x = i32(g.x);
  var y = i32(g.y);
  let s = i32(P.level);
  if (x < b || y < b || x >= i32(P.w) - b || y >= i32(P.h) - b) { return; }
  let v = D(x, y, s);
  if (abs(v) < 0.8 * P.peak) { return; }
  var is_max = true;
  var is_min = true;
  for (var ds = -1; ds <= 1; ds++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0 && ds == 0) { continue; }
        let u = D(x + dx, y + dy, s + ds);
        is_max = is_max && v > u;
        is_min = is_min && v < u;
      }
    }
  }
  if (!is_max && !is_min) { return; }
  // Newton refinement in (x, y, s); moves to the neighbouring pixel while the offset exceeds 0.6
  var off = vec3<f32>(0.0);
  var grad = vec3<f32>(0.0);
  var Dxx = 0.0; var Dyy = 0.0; var Dxy = 0.0;
  var ok = false;
  for (var it = 0; it < 5; it++) {
    let c = D(x, y, s);
    grad = 0.5 * vec3<f32>(D(x + 1, y, s) - D(x - 1, y, s), D(x, y + 1, s) - D(x, y - 1, s), D(x, y, s + 1) - D(x, y, s - 1));
    Dxx = D(x + 1, y, s) + D(x - 1, y, s) - 2.0 * c;
    Dyy = D(x, y + 1, s) + D(x, y - 1, s) - 2.0 * c;
    let Dss = D(x, y, s + 1) + D(x, y, s - 1) - 2.0 * c;
    Dxy = 0.25 * (D(x + 1, y + 1, s) - D(x - 1, y + 1, s) - D(x + 1, y - 1, s) + D(x - 1, y - 1, s));
    let Dxs = 0.25 * (D(x + 1, y, s + 1) - D(x - 1, y, s + 1) - D(x + 1, y, s - 1) + D(x - 1, y, s - 1));
    let Dys = 0.25 * (D(x, y + 1, s + 1) - D(x, y - 1, s + 1) - D(x, y + 1, s - 1) + D(x, y - 1, s - 1));
    let H = mat3x3<f32>(vec3<f32>(Dxx, Dxy, Dxs), vec3<f32>(Dxy, Dyy, Dys), vec3<f32>(Dxs, Dys, Dss));
    let det = determinant(H);
    if (abs(det) < 1e-12) { break; }
    off = -(inverse3(H, det) * grad);
    let mx = select(0, select(-1, 1, off.x > 0.0), abs(off.x) > 0.6);
    let my = select(0, select(-1, 1, off.y > 0.0), abs(off.y) > 0.6);
    if (mx == 0 && my == 0) { ok = true; break; }
    x += mx; y += my;
    if (x < b || y < b || x >= i32(P.w) - b || y >= i32(P.h) - b) { return; }
  }
  if (!ok || abs(off.x) > 1.5 || abs(off.y) > 1.5 || abs(off.z) > 1.5) { return; }
  let value = D(x, y, s) + 0.5 * dot(grad, off);
  if (abs(value) < P.peak) { return; }
  let tr = Dxx + Dyy;
  let det2 = Dxx * Dyy - Dxy * Dxy;
  if (det2 <= 0.0 || tr * tr * P.edge >= (P.edge + 1.0) * (P.edge + 1.0) * det2) { return; }
  let i = atomicAdd(&counters[0], 1u);
  if (i >= P.max_kp) { return; }
  let sl = f32(s) + off.z;
  kp[8u * i + 0u] = f32(x) + off.x;
  kp[8u * i + 1u] = f32(y) + off.y;
  kp[8u * i + 2u] = sl;
  kp[8u * i + 3u] = f32(P.octave);
  kp[8u * i + 4u] = P.sigma0 * exp2(sl / f32(P.levels)) * P.scale;
  kp[8u * i + 5u] = abs(value);
  kp[8u * i + 6u] = 0.0;
  kp[8u * i + 7u] = 0.0;
}

fn inverse3(m: mat3x3<f32>, det: f32) -> mat3x3<f32> {
  let a = m[0]; let b = m[1]; let c = m[2];
  let r0 = cross(b, c); let r1 = cross(c, a); let r2 = cross(a, b);
  return transpose(mat3x3<f32>(r0, r1, r2)) * (1.0 / det);
}

// Gaussian level nearest to a keypoint's scale, and its gradient (central differences)
fn G(level: u32, x: i32, y: i32) -> f32 {
  let n = P.w * P.h;
  let xc = clamp(x, 0, i32(P.w) - 1);
  let yc = clamp(y, 0, i32(P.h) - 1);
  return src_stack[level * n + u32(yc) * P.w + u32(xc)];
}

fn gradient(level: u32, x: i32, y: i32) -> vec2<f32> {
  return 0.5 * vec2<f32>(G(level, x + 1, y) - G(level, x - 1, y), G(level, x, y + 1) - G(level, x, y - 1));
}

const PI2: f32 = 6.283185307179586;

// ── orientation: one thread per candidate of this octave ──
@group(0) @binding(10) var<storage, read_write> okp: array<f32>;  // oriented keypoints (same record)

@compute @workgroup_size(64)
fn orient(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = atomicLoad(&counters[2u + 2u * P.octave]) + g.x;
  if (i >= min(atomicLoad(&counters[0]), P.max_kp)) { return; }
  let x = kp[8u * i]; let y = kp[8u * i + 1u]; let sl = kp[8u * i + 2u];
  let level = u32(clamp(round(sl), 0.0, f32(P.levels + 2u)));
  let sigma = P.sigma0 * exp2(sl / f32(P.levels));
  let sw = 1.5 * sigma;
  let r = i32(ceil(3.0 * sw));
  var hist: array<f32, 36>;
  for (var k = 0; k < 36; k++) { hist[k] = 0.0; }
  let xi = i32(round(x)); let yi = i32(round(y));
  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let px = xi + dx; let py = yi + dy;
      if (px <= 0 || py <= 0 || px >= i32(P.w) - 1 || py >= i32(P.h) - 1) { continue; }
      let fx = f32(px) - x; let fy = f32(py) - y;
      let r2 = fx * fx + fy * fy;
      if (r2 > f32(r * r) + 0.5) { continue; }
      let gr = gradient(level, px, py);
      let mag = length(gr);
      var ang = atan2(gr.y, gr.x);
      if (ang < 0.0) { ang += PI2; }
      let fbin = 36.0 * ang / PI2 - 0.5;
      let b0 = i32(floor(fbin));
      let t = fbin - f32(b0);
      let w = mag * exp(-r2 / (2.0 * sw * sw));
      hist[(b0 + 36) % 36] += (1.0 - t) * w;
      hist[(b0 + 37) % 36] += t * w;
    }
  }
  for (var it = 0; it < 6; it++) {  // circular [1 1 1] / 3 smoothing
    var prev = hist[35];
    let first = hist[0];
    for (var k = 0; k < 36; k++) {
      let cur = hist[k];
      let next = select(hist[min(k + 1, 35)], first, k == 35);
      hist[k] = (prev + cur + next) / 3.0;
      prev = cur;
    }
  }
  var hmax = 0.0;
  for (var k = 0; k < 36; k++) { hmax = max(hmax, hist[k]); }
  // up to 2 orientations: peaks >= 0.8 max, strongest first, parabolic interpolation
  var a1 = -1.0; var v1 = 0.0; var a2 = -1.0; var v2 = 0.0;
  for (var k = 0; k < 36; k++) {
    let h0 = hist[(k + 35) % 36]; let h1 = hist[k]; let h2 = hist[(k + 1) % 36];
    if (h1 > h0 && h1 > h2 && h1 >= 0.8 * hmax) {
      let di = 0.5 * (h0 - h2) / (h0 - 2.0 * h1 + h2);
      let a = PI2 * (f32(k) + di + 0.5) / 36.0;
      if (h1 > v1) { a2 = a1; v2 = v1; a1 = a; v1 = h1; } else if (h1 > v2) { a2 = a; v2 = h1; }
    }
  }
  for (var o = 0; o < 2; o++) {
    let a = select(a2, a1, o == 0);
    if (a < 0.0) { continue; }
    let j = atomicAdd(&counters[1], 1u);
    if (j >= P.max_kp) { return; }
    for (var k = 0u; k < 6u; k++) { okp[8u * j + k] = kp[8u * i + k]; }
    okp[8u * j + 6u] = a;
    okp[8u * j + 7u] = 0.0;
  }
}

// ── descriptor: one thread per oriented keypoint of this octave ──
@group(0) @binding(11) var<storage, read_write> desc: array<u32>;  // 32 u32 (128 u8) per keypoint

@compute @workgroup_size(64)
fn describe(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = atomicLoad(&counters[3u + 2u * P.octave]) + g.x;
  if (i >= min(atomicLoad(&counters[1]), P.max_kp)) { return; }
  let x = okp[8u * i]; let y = okp[8u * i + 1u]; let sl = okp[8u * i + 2u]; let angle = okp[8u * i + 6u];
  let level = u32(clamp(round(sl), 0.0, f32(P.levels + 2u)));
  let sigma = P.sigma0 * exp2(sl / f32(P.levels));
  let sbp = 3.0 * sigma;                          // spatial bin size (magnif 3)
  let W = i32(floor(sqrt(2.0) * sbp * 5.0 / 2.0 + 0.5));
  let c = cos(angle); let s = sin(angle);
  var h: array<f32, 128>;
  for (var k = 0; k < 128; k++) { h[k] = 0.0; }
  let xi = i32(round(x)); let yi = i32(round(y));
  for (var dy = -W; dy <= W; dy++) {
    for (var dx = -W; dx <= W; dx++) {
      let px = xi + dx; let py = yi + dy;
      if (px <= 0 || py <= 0 || px >= i32(P.w) - 1 || py >= i32(P.h) - 1) { continue; }
      let fx = f32(px) - x; let fy = f32(py) - y;
      // bin coordinates in the rotated frame, centred: 0 .. 4 over the window
      let nx = (c * fx + s * fy) / sbp;
      let ny = (-s * fx + c * fy) / sbp;
      let gr = gradient(level, px, py);
      let mag = length(gr);
      var t = atan2(gr.y, gr.x) - angle;
      t = t - PI2 * floor(t / PI2);
      let nt = 8.0 * t / PI2;
      let wgt = mag * exp(-(nx * nx + ny * ny) / 8.0);   // sigma = NBP / 2 = 2 bins
      let bx = nx + 1.5; let by = ny + 1.5;             // bin centres at -1.5 .. 1.5
      let x0 = i32(floor(bx)); let y0 = i32(floor(by)); let t0 = i32(floor(nt));
      let rx = bx - f32(x0); let ry = by - f32(y0); let rt = nt - f32(t0);
      for (var oy = 0; oy < 2; oy++) {
        let yb = y0 + oy;
        if (yb < 0 || yb > 3) { continue; }
        let wy = select(1.0 - ry, ry, oy == 1);
        for (var ox = 0; ox < 2; ox++) {
          let xb = x0 + ox;
          if (xb < 0 || xb > 3) { continue; }
          let wx = select(1.0 - rx, rx, ox == 1);
          for (var ot = 0; ot < 2; ot++) {
            let tb = (t0 + ot) % 8;
            let wt = select(1.0 - rt, rt, ot == 1);
            h[(yb * 4 + xb) * 8 + tb] += wgt * wx * wy * wt;
          }
        }
      }
    }
  }
  // L2, clamp 0.2, L2, then COLMAP's L1-root, x 512 -> u8
  var n2 = 0.0;
  for (var k = 0; k < 128; k++) { n2 += h[k] * h[k]; }
  let inv = 1.0 / max(sqrt(n2), 1e-12);
  n2 = 0.0;
  for (var k = 0; k < 128; k++) { h[k] = min(h[k] * inv, 0.2); n2 += h[k] * h[k]; }
  let inv2 = 1.0 / max(sqrt(n2), 1e-12);
  var l1 = 0.0;
  for (var k = 0; k < 128; k++) { h[k] *= inv2; l1 += h[k]; }
  let inv1 = 1.0 / max(l1, 1e-12);
  for (var k = 0u; k < 32u; k++) {
    var word = 0u;
    for (var b = 0u; b < 4u; b++) {
      let v = u32(clamp(round(512.0 * sqrt(h[4u * k + b] * inv1)), 0.0, 255.0));
      word |= v << (8u * b);
    }
    desc[32u * i + k] = word;
  }
}
