// stipple-shaders.wgsl — WGSL kernels for the GPU stippling pipeline.
//
// Kernels are separated by "// === <name> ===" lines; the loader splits on
// those markers and compiles each as an independent module. Bindings are
// per-module (each module starts with its own @group(0) @binding(...)).
//
// Phase 1 ships only the JFA kernels and a no-op lloyd kernel so the
// end-to-end pipeline can run. Accumulate + gather land in Phase 3.

// === clear_jfa ===
struct Params {
  W            : u32,
  H            : u32,
  full_W       : u32,
  full_H       : u32,
  N            : u32,
  N_max        : u32,
  jfa_step     : u32,
  jfa_min_step : u32,
  pyramid_lvl  : u32,
  full_moments : u32,
  _pad0        : u32,
  _pad1        : u32,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var jfa : texture_storage_2d<r32uint, write>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let W = params.W;
  let H = params.H;
  let idx = gid.x;
  if (idx >= W * H) { return; }
  textureStore(jfa, vec2<i32>(i32(idx % W), i32(idx / W)),
               vec4<u32>(0xffffffffu, 0u, 0u, 0u));
}

// === seed_jfa ===
struct Params {
  W            : u32, H            : u32, full_W       : u32, full_H       : u32,
  N            : u32, N_max        : u32, jfa_step     : u32, jfa_min_step : u32,
  pyramid_lvl  : u32, full_moments : u32, _pad0        : u32, _pad1        : u32,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read>  sites : array<vec2<f32>>;
@group(0) @binding(2) var jfa : texture_storage_2d<r32uint, write>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) { return; }
  let s = sites[i];
  let fx = clamp(s.x, 0.0, 0.99999);
  let fy = clamp(s.y, 0.0, 0.99999);
  let px = i32(fx * f32(params.W));
  let py = i32(fy * f32(params.H));
  textureStore(jfa, vec2<i32>(px, py), vec4<u32>(i, 0u, 0u, 0u));
}

// === jfa_step ===
struct Params {
  W            : u32, H            : u32, full_W       : u32, full_H       : u32,
  N            : u32, N_max        : u32, jfa_step     : u32, jfa_min_step : u32,
  pyramid_lvl  : u32, full_moments : u32, _pad0        : u32, _pad1        : u32,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read>  sites : array<vec2<f32>>;
@group(0) @binding(2) var jfa_in  : texture_2d<u32>;
@group(0) @binding(3) var jfa_out : texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let W = params.W;
  let H = params.H;
  if (gid.x >= W || gid.y >= H) { return; }
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  let step = i32(params.jfa_step);

  let X = (f32(gid.x) + 0.5) / f32(W);
  let Y = (f32(gid.y) + 0.5) / f32(H);

  var best_id : u32 = textureLoad(jfa_in, p, 0).x;
  var best_d2 : f32 = 1.0e30;
  if (best_id != 0xffffffffu) {
    let s = sites[best_id];
    let ddx = X - s.x;
    let ddy = Y - s.y;
    best_d2 = ddx * ddx + ddy * ddy;
  }

  for (var dy : i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx : i32 = -1; dx <= 1; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      let q = p + vec2<i32>(dx * step, dy * step);
      if (q.x < 0 || q.y < 0 || q.x >= i32(W) || q.y >= i32(H)) { continue; }
      let v = textureLoad(jfa_in, q, 0).x;
      if (v == 0xffffffffu) { continue; }
      let s = sites[v];
      let ddx = X - s.x;
      let ddy = Y - s.y;
      let d2 = ddx * ddx + ddy * ddy;
      if (d2 < best_d2) {
        best_d2 = d2;
        best_id = v;
      }
    }
  }
  textureStore(jfa_out, p, vec4<u32>(best_id, 0u, 0u, 0u));
}

// === edge_detect ===
// Sample source RGBA texture, compute luma + center-vs-neighbor edge detect,
// store grayscale into the rgba8unorm output. Also reduces the grayscale sum
// (in u8 units, 0-255 per pixel) into a single u32 atomic for image_average.
// Reduction is workgroup-local (64 threads contend on shared atomic) then one
// global atomicAdd per workgroup to bound contention.
struct EdgeParams {
  W      : u32,
  H      : u32,
  alpha  : f32,
  amp    : f32,
  strength : f32,
  _pad0  : f32,
  _pad1  : f32,
  _pad2  : f32,
};
@group(0) @binding(0) var<uniform> ep      : EdgeParams;
@group(0) @binding(1) var          src     : texture_2d<f32>;
@group(0) @binding(2) var          srcSamp : sampler;
@group(0) @binding(3) var          dst     : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> image_sum : atomic<u32>;

var<workgroup> wg_sum : atomic<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32) {
  if (lid == 0u) { atomicStore(&wg_sum, 0u); }
  workgroupBarrier();

  if (gid.x < ep.W && gid.y < ep.H) {
    let inv = vec2<f32>(1.0 / f32(ep.W), 1.0 / f32(ep.H));
    let uv  = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) * inv;

    var col = textureSampleLevel(src, srcSamp, uv, 0.0).rgb;
    let amp = 0.25 * (1.0 - ep.alpha);
    col = col - amp * textureSampleLevel(src, srcSamp, uv + vec2<f32>( inv.x, 0.0), 0.0).rgb;
    col = col - amp * textureSampleLevel(src, srcSamp, uv + vec2<f32>(-inv.x, 0.0), 0.0).rgb;
    col = col - amp * textureSampleLevel(src, srcSamp, uv + vec2<f32>(0.0,  inv.y), 0.0).rgb;
    col = col - amp * textureSampleLevel(src, srcSamp, uv + vec2<f32>(0.0, -inv.y), 0.0).rgb;

    let luma = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    let v    = min(ep.strength * abs(luma), 1.0);
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(v, v, v, 1.0));
    atomicAdd(&wg_sum, u32(v * 255.0 + 0.5));
  }

  workgroupBarrier();
  if (lid == 0u) { atomicAdd(&image_sum, atomicLoad(&wg_sum)); }
}

// === luma ===
// Plain grayscale (used when edge detection is off). Same image_sum reduction
// as edge_detect.
struct EdgeParams {
  W : u32, H : u32, alpha : f32, amp : f32,
  strength : f32, _pad0 : f32, _pad1 : f32, _pad2 : f32,
};
@group(0) @binding(0) var<uniform> ep      : EdgeParams;
@group(0) @binding(1) var          src     : texture_2d<f32>;
@group(0) @binding(2) var          srcSamp : sampler;
@group(0) @binding(3) var          dst     : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> image_sum : atomic<u32>;

var<workgroup> wg_sum : atomic<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32) {
  if (lid == 0u) { atomicStore(&wg_sum, 0u); }
  workgroupBarrier();

  if (gid.x < ep.W && gid.y < ep.H) {
    let inv = vec2<f32>(1.0 / f32(ep.W), 1.0 / f32(ep.H));
    let uv  = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) * inv;
    let col = textureSampleLevel(src, srcSamp, uv, 0.0).rgb;
    let luma = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(luma, luma, luma, 1.0));
    atomicAdd(&wg_sum, u32(luma * 255.0 + 0.5));
  }

  workgroupBarrier();
  if (lid == 0u) { atomicAdd(&image_sum, atomicLoad(&wg_sum)); }
}

// === points_render ===
// Vertex+fragment for drawing N point sprites. Each point is a quad of 4
// vertices (TriangleStrip in instanced form). instance_index = point id.
struct PointParams {
  radius_px : f32,
  canvas_w  : f32,
  canvas_h  : f32,
  invert    : f32,   // 0 = black points on white, 1 = white on black
  _pad      : vec4<f32>,
};
@group(0) @binding(0) var<uniform> pp : PointParams;
@group(0) @binding(1) var<storage, read> sites : array<vec2<f32>>;

struct VOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       quad_uv : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index)   vid : u32,
      @builtin(instance_index) iid : u32) -> VOut {
  let corners = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let c = corners[vid];
  let s = sites[iid];
  // Site position in NDC: x in [-1,1], y flipped (image space).
  let center_ndc = vec2<f32>(s.x * 2.0 - 1.0, 1.0 - s.y * 2.0);
  let radius_ndc = vec2<f32>(pp.radius_px * 2.0 / pp.canvas_w,
                             pp.radius_px * 2.0 / pp.canvas_h);
  var o : VOut;
  o.pos     = vec4<f32>(center_ndc + c * radius_ndc, 0.0, 1.0);
  o.quad_uv = c;
  return o;
}

@fragment
fn fs(in : VOut) -> @location(0) vec4<f32> {
  if (dot(in.quad_uv, in.quad_uv) > 1.0) { discard; }
  let c = pp.invert;   // 1 = white point, 0 = black
  return vec4<f32>(c, c, c, 1.0);
}

// === target_render ===
// Fullscreen blit of an r8unorm grayscale texture into the swapchain.
@group(0) @binding(0) var t  : texture_2d<f32>;
@group(0) @binding(1) var ts : sampler;

struct TVOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid : u32) -> TVOut {
  let corners = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let uvs = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
  );
  var o : TVOut;
  o.pos = vec4<f32>(corners[vid], 0.0, 1.0);
  o.uv  = uvs[vid];
  return o;
}

@fragment
fn fs(in : TVOut) -> @location(0) vec4<f32> {
  let g = textureSampleLevel(t, ts, in.uv, 0.0).r;
  return vec4<f32>(g, g, g, 1.0);
}

// === voronoi_render ===
// Like target_render but reads the r32uint site_ids texture, fetches each
// site's intensity from a sites_color buffer, and renders.
@group(0) @binding(0) var site_ids   : texture_2d<u32>;
@group(0) @binding(1) var<storage, read> site_color : array<f32>;
struct VParams {
  W : u32, H : u32, scale : f32, _pad : f32,
};
@group(0) @binding(2) var<uniform> vp : VParams;

struct VVOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid : u32) -> VVOut {
  let corners = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
  );
  let uvs = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
  );
  var o : VVOut;
  o.pos = vec4<f32>(corners[vid], 0.0, 1.0);
  o.uv  = uvs[vid];
  return o;
}

@fragment
fn fs(in : VVOut) -> @location(0) vec4<f32> {
  let px = vec2<i32>(i32(in.uv.x * f32(vp.W)),
                     i32(in.uv.y * f32(vp.H)));
  let id = textureLoad(site_ids, px, 0).x;
  if (id == 0xffffffffu) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
  let c = clamp(site_color[id] * vp.scale, 0.0, 1.0);
  return vec4<f32>(c, c, c, 1.0);
}

// === clear_cells ===
// Zero the per-cell f32-via-u32 atomic accumulators. 7 slots per cell:
//   [acc, r, rx, ry, rxx, rxy, ryy]   stored as bitcast<u32>(f32).
//   0u == bitcast<u32>(0.0f), so zeroing is direct.
struct Params {
  W : u32, H : u32, full_W : u32, full_H : u32,
  N : u32, N_max : u32, jfa_step : u32, jfa_min_step : u32,
  pyramid_lvl : u32, full_moments : u32, _pad0 : u32, _pad1 : u32,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read_write> cells : array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.N * 7u) { return; }
  atomicStore(&cells[i], 0u);
}

// === accumulate ===
// Per pixel: f32-atomicAdd into cells[sid * 9 + slot] via CAS-loop on u32.
// WebGPU has no native f32 atomics; we bitcast f32↔u32 and use
// atomicCompareExchangeWeak to perform the add. Contention is low (~13
// pixels per cell at 1024² + N=80K) so retries are minimal.
struct Params {
  W : u32, H : u32, full_W : u32, full_H : u32,
  N : u32, N_max : u32, jfa_step : u32, jfa_min_step : u32,
  pyramid_lvl : u32, full_moments : u32, _pad0 : u32, _pad1 : u32,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var pixels   : texture_2d<f32>;
@group(0) @binding(2) var site_ids : texture_2d<u32>;
@group(0) @binding(3) var<storage, read_write> cells : array<atomic<u32>>;

fn add_f32(idx : u32, value : f32) {
  var old : u32 = atomicLoad(&cells[idx]);
  loop {
    let new_bits = bitcast<u32>(bitcast<f32>(old) + value);
    let r = atomicCompareExchangeWeak(&cells[idx], old, new_bits);
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.W || gid.y >= params.H) { return; }
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  let sid = textureLoad(site_ids, p, 0).x;
  if (sid >= params.N) { return; }

  let r = textureLoad(pixels, p, 0).r * 255.0;
  let X = (f32(gid.x) + 0.5) / f32(params.W);    // [0, 1]
  let Y = (f32(gid.y) + 0.5) / f32(params.H);

  let base = sid * 7u;
  add_f32(base + 0u, 1.0);                          // acc
  add_f32(base + 1u, r);                            // r_acc
  add_f32(base + 2u, r * X);                        // rx_acc
  add_f32(base + 3u, r * Y);                        // ry_acc

  if (params.full_moments != 0u) {
    add_f32(base + 4u, r * X * X);                  // rxx_acc
    add_f32(base + 5u, r * X * Y);                  // rxy_acc
    add_f32(base + 6u, r * Y * Y);                  // ryy_acc
  }
}

// === lloyd_update ===
// Reads cells as u32 (bitcast f32). Centroid = rx / r, ry / r directly.
struct Params {
  W : u32, H : u32, full_W : u32, full_H : u32,
  N : u32, N_max : u32, jfa_step : u32, jfa_min_step : u32,
  pyramid_lvl : u32, full_moments : u32, _pad0 : u32, _pad1 : u32,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read>       cells : array<u32>;
@group(0) @binding(2) var<storage, read_write> sites : array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) { return; }
  let base = i * 7u;
  let r = bitcast<f32>(cells[base + 1u]);
  if (r > 0.0) {
    let rx = bitcast<f32>(cells[base + 2u]);
    let ry = bitcast<f32>(cells[base + 3u]);
    sites[i] = vec2<f32>(rx / r, ry / r);
  }
}
