// shaders.wgsl — WGSL kernels for Gauss-Opt.
//
// Each `// === <name> ===` section is compiled as an independent module by
// loadShaders() (src/render.js), so shared structs/helpers are duplicated per
// module and must stay byte-identical. Layouts must match src/render.js and
// src/trainer.js.
//
// Splat layout (4 vec4f): [x, y, z, logit_op], [log_sx, log_sy, log_sz, leaf_id],
//                         [qw, qx, qy, qz], [f_dc_r, f_dc_g, f_dc_b, flags]
// Moments (12 f32): W, S, R, R1.xyz, R2.(xx, yy, zz, xy, xz, yz)
// Grads   (12 f32): dL/drgb.xyz, dL/dlogit_op, dL/du0, dL/dv0,
//                   dL/dconic.(a, b, c), w_view, w_err_view, unused

// === splats_render ===
// Instanced quad renderer for the interactive view: 2.5-sigma screen-space
// quads per sorted splat, shape modes (vs/fs) and diagnostic color modes.
// Bindings: 0 RenderUniforms, 1 splats, 2 moments, 3 sorted_idx.

struct RenderUniforms {
  view       : mat4x4f,
  proj       : mat4x4f, // unused
  cam_pos    : vec4f,   // xyz = camera center (unused), w = scene_radius
  intrinsics : vec4f,   // fx, fy, cx, cy
  screen     : vec4f,   // W, H, num_visible (unused), haze (scale multiplier)
  modes      : vec4u,   // x = visu_mode, y = color_mode, zw unused
  thresholds : vec4f,   // x = avg_R, y = opacity_scale, z = znear, w = zfar
};

@group(0) @binding(0) var<uniform> u : RenderUniforms;
@group(0) @binding(1) var<storage, read> splats     : array<vec4f>; // 4 per splat
@group(0) @binding(2) var<storage, read> moments    : array<vec4f>; // 3 per splat
@group(0) @binding(3) var<storage, read> sorted_idx : array<u32>;

struct VOut {
  @builtin(position) pos   : vec4f,
  @location(0)       uv    : vec2f,
  @location(1) @interpolate(flat) color : vec4f,
};

const kQuad = array<vec2f, 4>(
  vec2f(-2.5, -2.5),
  vec2f( 2.5, -2.5),
  vec2f(-2.5,  2.5),
  vec2f( 2.5,  2.5),
);

const SH_C0 : f32 = 0.28209479177387814;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -15.0, 15.0)));
}

// q = (w, x, y, z), normalized here. Returns column-major R.
fn quat_to_mat3(q: vec4f) -> mat3x3f {
  let n = max(1e-8, length(q));
  let w = q.x / n; let x = q.y / n; let y = q.z / n; let z = q.w / n;
  return mat3x3f(
    vec3f(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y)),
    vec3f(2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
    vec3f(2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y))
  );
}

// EWA: T = J * W (pixel Jacobian at view-space point vpos times view rotation).
// Screen covariance is T * Sigma * T^T. As in 3DGS, J is evaluated with x/z, y/z clamped
// to `lim` (1.3x the frustum, see frustum_lim): off-axis splats near the camera would
// otherwise project to huge ellipses.
fn ewa_jacobian(view: mat4x4f, vpos: vec3f, fx: f32, fy: f32, lim: vec2f) -> mat3x2f {
  let inv_z  = 1.0 / vpos.z;
  let t = clamp(vpos.xy * inv_z, -lim, lim);  // clamped x/z, y/z
  let J = mat3x2f(
    vec2f(fx * inv_z, 0.0),
    vec2f(0.0, fy * inv_z),
    vec2f(-fx * t.x * inv_z, -fy * t.y * inv_z)
  );
  return J * mat3x3f(view[0].xyz, view[1].xyz, view[2].xyz);
}

// |x/z|, |y/z| bounds of 1.3x the view frustum (W x H, principal point cx, cy).
fn frustum_lim(W: f32, H: f32, fx: f32, fy: f32, cx: f32, cy: f32) -> vec2f {
  return 1.3 * vec2f(max(cx, W - cx) / fx, max(cy, H - cy) / fy);
}

fn palette_leaf(leaf_id: u32) -> vec3f {
  let h = (leaf_id * 2654435761u) ^ (leaf_id >> 13u);
  let r = f32((h >>  0u) & 255u) / 255.0;
  let g = f32((h >>  8u) & 255u) / 255.0;
  let b = f32((h >> 16u) & 255u) / 255.0;
  return 0.25 + 0.75 * vec3f(r, g, b);
}

fn heatmap_coolwarm(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  return clamp(vec3f(1.5*x, 1.2*sin(x * 3.14159), 1.5*(1.0 - x)), vec3f(0.0), vec3f(1.0));
}

@vertex
fn vs(@builtin(vertex_index) vid: u32,
      @builtin(instance_index) iid: u32) -> VOut {
  var out : VOut;
  let idx = sorted_idx[iid];
  let b = idx * 4u;
  let p0 = splats[b + 0u];
  let p1 = splats[b + 1u];
  let p2 = splats[b + 2u];
  let p3 = splats[b + 3u];

  let vpos = u.view * vec4f(p0.xyz, 1.0);
  if (vpos.z <= u.thresholds.z) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0); // clipped
    return out;
  }

  let fx = u.intrinsics.x;
  let fy = u.intrinsics.y;
  let inv_z = 1.0 / vpos.z;
  let u0 = fx * vpos.x * inv_z + u.intrinsics.z;
  let v0 = fy * vpos.y * inv_z + u.intrinsics.w;

  let R = quat_to_mat3(p2);
  let s = exp(clamp(p1.xyz, vec3f(-12.0), vec3f(4.0))) * u.screen.w;
  let M = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let T = ewa_jacobian(u.view, vpos.xyz, fx, fy,
                       frustum_lim(u.screen.x, u.screen.y, fx, fy, u.intrinsics.z, u.intrinsics.w));
  let cov2d = T * (M * transpose(M)) * transpose(T);

  // Eigen-decomposition of the dilated 2x2 covariance -> quad axes in pixels.
  let sxx = cov2d[0][0] + 0.3;
  let syy = cov2d[1][1] + 0.3;
  let sxy = cov2d[0][1];
  let mid  = 0.5 * (sxx + syy);
  let dist = 0.5 * (sxx - syy);
  let det  = length(vec2f(dist, sxy));
  let lambda1 = clamp(mid + det, 0.1, 16384.0);
  let lambda2 = clamp(mid - det, 0.1, 16384.0);
  var diag = vec2f(1.0, 0.0);
  if (abs(sxy) > 1e-6 || abs(det - dist) > 1e-6) {
    diag = normalize(vec2f(sxy, det - dist));
  }
  let major_px = sqrt(lambda1) * diag;
  let minor_px = sqrt(lambda2) * vec2f(diag.y, -diag.x);

  let quad = kQuad[vid];
  let px = vec2f(u0, v0) + quad.x * major_px + quad.y * minor_px;
  let ndc_x = (px.x / u.screen.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (px.y / u.screen.y) * 2.0;
  let clip_z = clamp((vpos.z - u.thresholds.z) / (u.thresholds.w - u.thresholds.z), 0.0, 0.9999);
  out.pos = vec4f(ndc_x, ndc_y, clip_z, 1.0);
  out.uv  = quad;

  let alpha = min(0.99, sigmoid(p0.w) * u.thresholds.y);
  var rgb = clamp(vec3f(0.5) + SH_C0 * p3.xyz, vec3f(0.0), vec3f(1.0));
  let cmode = u.modes.y;
  if (cmode == 1u) {        // octree leaf id
    rgb = palette_leaf(u32(max(0.0, p1.w)));
  } else if (cmode == 2u) { // WLBG error mass R_i, relative to 2 * avg_R
    let m0 = moments[idx * 3u];
    rgb = heatmap_coolwarm(m0.z / max(1e-5, u.thresholds.x * 2.0));
  } else if (cmode == 3u) { // N_eff = W^2 / S: magenta = single-view, green = multi-view
    let m0 = moments[idx * 3u];
    let n_eff = select(1.0, (m0.x * m0.x) / max(1e-6, m0.y), m0.y > 1e-6);
    rgb = mix(vec3f(1.0, 0.1, 0.8), vec3f(0.1, 0.95, 0.4), clamp((n_eff - 1.0) / 4.0, 0.0, 1.0));
  } else if (cmode == 4u) { // depth
    rgb = heatmap_coolwarm(1.0 - clamp(vpos.z / max(1.0, u.cam_pos.w * 2.5), 0.0, 1.0));
  }
  out.color = vec4f(rgb, alpha);
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let v  = in.uv; // in units of sigma
  let r2 = dot(v, v);
  let r  = sqrt(r2);
  let vmode = u.modes.x;
  var weight : f32 = 0.0;
  if (vmode == 0u) {        // gaussian
    if (r2 > 6.25) { discard; }
    weight = exp(-0.5 * r2);
  } else if (vmode == 1u) { // cone
    if (r > 2.0) { discard; }
    weight = 1.0 - 0.5 * r;
  } else if (vmode == 2u) { // ellipsis ring
    if (r > 2.0) { discard; }
    weight = smoothstep(1.6, 1.85, r) * (1.0 - step(2.0, r)) + 0.15 * exp(-0.5 * r2);
  } else if (vmode == 3u) { // small dot
    if (r > 0.35) { discard; }
    weight = 1.0;
  } else {                  // principal axes cross
    if (r > 2.0) { discard; }
    weight = 1.0 - smoothstep(0.0, 0.12, min(abs(v.x), abs(v.y)));
  }
  let a = in.color.a * weight;
  if (a < 1.0 / 255.0) { discard; }
  return vec4f(in.color.rgb * a, a);
}

// === wireframe_render ===
// 3D line segments (octree boxes, camera frustums, WLBG split axes).
// Bindings: 0 WireUniforms, 1 verts (xyz, bitcast<f32>(abgr u32)).

struct WireUniforms {
  view       : mat4x4f,
  intrinsics : vec4f, // fx, fy, cx, cy
  screen     : vec4f, // W, H, znear, zfar
};

@group(0) @binding(0) var<uniform> wu : WireUniforms;
@group(0) @binding(1) var<storage, read> verts : array<vec4f>;

// World point -> (ndc_x, ndc_y, clip_z, view z). clip_z = 2 (clipped) if z <= 0.05.
fn wire_project(p: vec3f) -> vec4f {
  let vpos = wu.view * vec4f(p, 1.0);
  let z = max(0.05, vpos.z);
  let px = wu.intrinsics.xy * (vpos.xy / z) + wu.intrinsics.zw;
  let ndc_x = (px.x / wu.screen.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (px.y / wu.screen.y) * 2.0;
  let clip_z = select(2.0, clamp((z - wu.screen.z) / (wu.screen.w - wu.screen.z), 0.0, 0.999), vpos.z > 0.05);
  return vec4f(ndc_x, ndc_y, clip_z, vpos.z);
}

struct WireOut {
  @builtin(position) pos   : vec4f,
  @location(0)       color : vec4f,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> WireOut {
  var out : WireOut;
  let v = verts[vid];
  out.pos   = vec4f(wire_project(v.xyz).xyz, 1.0);
  out.color = unpack4x8unorm(bitcast<u32>(v.w));
  return out;
}

@fragment
fn fs(in: WireOut) -> @location(0) vec4f {
  return vec4f(in.color.rgb * in.color.a, in.color.a);
}

// === frustum_image ===
// Textured image planes inside the COLMAP camera frustums (one instance per camera).
// Bindings: 0 WireUniforms, 1 quad_verts (4 corners per camera, .w = alpha),
//           2 cam_tex, 3 cam_samp.

struct WireUniforms {
  view       : mat4x4f,
  intrinsics : vec4f, // fx, fy, cx, cy
  screen     : vec4f, // W, H, znear, zfar
};

@group(0) @binding(0) var<uniform> wu : WireUniforms;
@group(0) @binding(1) var<storage, read> quad_verts : array<vec4f>;
@group(0) @binding(2) var cam_tex  : texture_2d<f32>;
@group(0) @binding(3) var cam_samp : sampler;

// World point -> (ndc_x, ndc_y, clip_z, view z). clip_z = 2 (clipped) if z <= 0.05.
fn wire_project(p: vec3f) -> vec4f {
  let vpos = wu.view * vec4f(p, 1.0);
  let z = max(0.05, vpos.z);
  let px = wu.intrinsics.xy * (vpos.xy / z) + wu.intrinsics.zw;
  let ndc_x = (px.x / wu.screen.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (px.y / wu.screen.y) * 2.0;
  let clip_z = select(2.0, clamp((z - wu.screen.z) / (wu.screen.w - wu.screen.z), 0.0, 0.999), vpos.z > 0.05);
  return vec4f(ndc_x, ndc_y, clip_z, vpos.z);
}

struct FrustumImgOut {
  @builtin(position) pos   : vec4f,
  @location(0)       uv    : vec2f,
  @location(1)       alpha : f32,
  @location(2)       z_cam : f32,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> FrustumImgOut {
  var out : FrustumImgOut;
  let v = quad_verts[iid * 4u + vid];
  let c = wire_project(v.xyz);
  let z = max(0.05, c.w);
  out.pos   = vec4f(c.xyz * z, z); // w = z: perspective-correct uv
  out.uv    = array<vec2f, 4>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0))[vid];
  out.alpha = v.w;
  out.z_cam = c.w;
  return out;
}

@fragment
fn fs(in: FrustumImgOut) -> @location(0) vec4f {
  if (in.z_cam < 0.05 || in.alpha <= 0.0) { discard; }
  let rgb = textureSampleLevel(cam_tex, cam_samp, in.uv, 0.0).rgb;
  return vec4f(rgb * in.alpha, in.alpha);
}

// === compare_blit ===
// Fullscreen ground-truth / render / residual comparison.
// Bindings: 0 BlitParams, 1 gt_tex, 2 hat_tex, 3 samp.

struct BlitParams {
  mode    : u32, // 0 = target, 1 = split (target left), 2 = residual |gt - hat| * amp
  split_x : f32, // in [0, 1]
  amp     : f32,
  _pad    : f32,
};

@group(0) @binding(0) var<uniform> bp      : BlitParams;
@group(0) @binding(1) var          gt_tex  : texture_2d<f32>;
@group(0) @binding(2) var          hat_tex : texture_2d<f32>;
@group(0) @binding(3) var          samp    : sampler;

struct BlitOut {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> BlitOut {
  let c = array<vec2f, 4>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(1,1))[vid];
  var o : BlitOut;
  o.pos = vec4f(c, 0.0, 1.0);
  o.uv  = vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
  return o;
}

@fragment
fn fs(in: BlitOut) -> @location(0) vec4f {
  let gt  = textureSampleLevel(gt_tex, samp, in.uv, 0.0).rgb;
  let hat = textureSampleLevel(hat_tex, samp, in.uv, 0.0).rgb;
  if (bp.mode == 0u) { return vec4f(gt, 1.0); }
  if (bp.mode == 2u) { return vec4f(abs(gt - hat) * bp.amp, 1.0); }
  if (abs(in.uv.x - bp.split_x) < 0.002) { return vec4f(1.0, 0.9, 0.2, 1.0); } // wipe line
  return vec4f(select(hat, gt, in.uv.x < bp.split_x), 1.0);
}

// === sort_refine ===
// Incremental GPU depth sort of the display list (cf. splats/): each thread insertion-sorts
// a CHUNK of indices in place by camera-space z (front-to-back). Alternating the chunk offset
// (0 / CHUNK/2) between passes lets entries migrate across chunk borders, so the exact CPU
// order of a recent pose keeps tracking the moving camera.
// Bindings: 0 RefineParams, 1 splats, 2 idx_list.
struct RefineParams {
  zrow   : vec4f, // row 2 of the view matrix: z_cam = dot(zrow.xyz, p) + zrow.w
  n      : u32,
  offset : u32,
  _pad0  : u32,
  _pad1  : u32,
};

const CHUNK : u32 = 32u;

@group(0) @binding(0) var<uniform> rp : RefineParams;
@group(0) @binding(1) var<storage, read> splats : array<vec4f>; // 4 vec4f per splat, [0].xyz = position
@group(0) @binding(2) var<storage, read_write> idx_list : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x * CHUNK + rp.offset;
  if (s >= rp.n) { return; }
  let m = min(CHUNK, rp.n - s);
  var ids : array<u32, CHUNK>;
  var zs  : array<f32, CHUNK>;
  for (var i = 0u; i < m; i++) {
    let id = idx_list[s + i];
    ids[i] = id;
    zs[i]  = dot(rp.zrow.xyz, splats[id * 4u].xyz) + rp.zrow.w;
  }
  var changed = false;
  for (var i = 1u; i < m; i++) {
    let a  = ids[i];
    let za = zs[i];
    var j = i;
    while (j > 0u && zs[j - 1u] > za) {
      ids[j] = ids[j - 1u];
      zs[j]  = zs[j - 1u];
      j--;
    }
    if (j != i) {
      ids[j] = a;
      zs[j]  = za;
      changed = true;
    }
  }
  if (changed) {
    for (var i = 0u; i < m; i++) { idx_list[s + i] = ids[i]; }
  }
}

// === project_splats ===
// Projects the num_visible depth-sorted splats to screen space, one thread each.
// proj_splats (2 vec4f per sorted slot): [u0, v0, z_cam, radius_px],
// [conic_a, conic_b, conic_c, bitcast<f32>(orig_idx)]; radius_px = 0 = culled.
// proj_tiles (per slot): inclusive 16x16-tile rect, packed [x0 | y0 << 16, x1 | y1 << 16]
// (x0 > x1 = none), of the tiles holding a pixel center inside the 3-sigma ellipse
// (power <= 4.5, the rasterizers' cut): its bounding box, half-widths 3 sqrt(cov_xx|yy).
// Each covered tile's tile_counts entry is incremented (tile_scan / fill / sort build the lists).
// Bindings: 0 TrainUniforms, 1 splats, 2 sorted_idx, 3 proj_splats, 4 proj_tiles, 5 tile_counts.

struct TrainUniforms {
  view       : mat4x4f,
  cam_pos    : vec4f, // xyz = camera center, w = Adam step (1-based)
  intrinsics : vec4f, // fx, fy, cx, cy
  dims       : vec4u, // W, H, num_visible, total_splats
  lrs        : vec4f, // lr_pos, lr_scale, lr_quat, lr_color
  opt_params : vec4f, // x = opt_method (0 Adam, 1 MC, 2 SA), y = mc_noise | sa_temp, z = rng seed, w = scene_radius
  limits     : vec4f, // x = max splat angle (scale / depth), y = max world scale, z = log(max anisotropy), w = lr_opacity
};

@group(0) @binding(0) var<uniform> tu : TrainUniforms;
@group(0) @binding(1) var<storage, read>       splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       sorted_idx  : array<u32>;
@group(0) @binding(3) var<storage, read_write> proj_splats : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> proj_tiles  : array<vec2u>;
@group(0) @binding(5) var<storage, read_write> tile_counts : array<atomic<u32>>;

// q = (w, x, y, z), normalized here. Returns column-major R.
fn quat_to_mat3(q: vec4f) -> mat3x3f {
  let n = max(1e-8, length(q));
  let w = q.x / n; let x = q.y / n; let y = q.z / n; let z = q.w / n;
  return mat3x3f(
    vec3f(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y)),
    vec3f(2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
    vec3f(2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y))
  );
}

// EWA: T = J * W (pixel Jacobian at view-space point vpos times view rotation).
// Screen covariance is T * Sigma * T^T. As in 3DGS, J is evaluated with x/z, y/z clamped
// to `lim` (1.3x the frustum, see frustum_lim): off-axis splats near the camera would
// otherwise project to huge ellipses.
fn ewa_jacobian(view: mat4x4f, vpos: vec3f, fx: f32, fy: f32, lim: vec2f) -> mat3x2f {
  let inv_z  = 1.0 / vpos.z;
  let t = clamp(vpos.xy * inv_z, -lim, lim);  // clamped x/z, y/z
  let J = mat3x2f(
    vec2f(fx * inv_z, 0.0),
    vec2f(0.0, fy * inv_z),
    vec2f(-fx * t.x * inv_z, -fy * t.y * inv_z)
  );
  return J * mat3x3f(view[0].xyz, view[1].xyz, view[2].xyz);
}

// |x/z|, |y/z| bounds of 1.3x the view frustum (W x H, principal point cx, cy).
fn frustum_lim(W: f32, H: f32, fx: f32, fy: f32, cx: f32, cy: f32) -> vec2f {
  return 1.3 * vec2f(max(cx, W - cx) / fx, max(cy, H - cy) / fy);
}

fn write_culled(k: u32, idx: u32) {
  proj_splats[k * 2u + 0u] = vec4f(-1e5, -1e5, -1.0, 0.0);
  proj_splats[k * 2u + 1u] = vec4f(0.0, 0.0, 0.0, bitcast<f32>(idx));
  proj_tiles[k] = vec2u(1u, 0u);  // empty rect
}

// Tiles t whose pixel centers [16 t + 0.5, 16 t + 15.5] meet [c - e, c + e], padded for rounding.
fn tile_span(c: f32, e: f32, n: u32) -> vec2u {
  let lo = i32(floor((c - e - 15.6) / 16.0));
  let hi = i32(floor((c + e - 0.4) / 16.0));
  return vec2u(u32(clamp(lo, 0, i32(n) - 1)), u32(clamp(hi, 0, i32(n) - 1)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= tu.dims.z) { return; }

  let idx = sorted_idx[k];
  let p0  = splats[idx * 4u + 0u];
  let p1  = splats[idx * 4u + 1u];
  let p2  = splats[idx * 4u + 2u];

  let vpos = tu.view * vec4f(p0.xyz, 1.0);
  if (vpos.z <= 0.1) { write_culled(k, idx); return; }

  let fx = tu.intrinsics.x;
  let fy = tu.intrinsics.y;
  let inv_z = 1.0 / vpos.z;
  let u0 = fx * vpos.x * inv_z + tu.intrinsics.z;
  let v0 = fy * vpos.y * inv_z + tu.intrinsics.w;

  let R = quat_to_mat3(p2);
  let s = exp(clamp(p1.xyz, vec3f(-10.0), vec3f(3.0)));
  let M = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let lim = frustum_lim(f32(tu.dims.x), f32(tu.dims.y), fx, fy, tu.intrinsics.z, tu.intrinsics.w);
  let T = ewa_jacobian(tu.view, vpos.xyz, fx, fy, lim);
  let cov2d = T * (M * transpose(M)) * transpose(T);

  // 0.3 px^2 low-pass dilation (must match adam_step), conic = inverse covariance.
  let sxx = cov2d[0][0] + 0.3;
  let syy = cov2d[1][1] + 0.3;
  let sxy = cov2d[0][1];
  let det = max(1e-6, sxx * syy - sxy * sxy);
  let conic = vec3f(syy, -sxy, sxx) * (1.0 / det);
  let mid = 0.5 * (sxx + syy);
  let radius_px = ceil(3.0 * sqrt(mid + sqrt(max(0.1, mid * mid - det))));
  if (u0 + radius_px < 0.0 || u0 - radius_px > f32(tu.dims.x) ||
      v0 + radius_px < 0.0 || v0 - radius_px > f32(tu.dims.y)) {
    write_culled(k, idx);
    return;
  }
  proj_splats[k * 2u + 0u] = vec4f(u0, v0, vpos.z, radius_px);
  proj_splats[k * 2u + 1u] = vec4f(conic, bitcast<f32>(idx));
  let tx = (tu.dims.x + 15u) / 16u;
  let sx = tile_span(u0, 3.0 * sqrt(sxx), tx);
  let sy = tile_span(v0, 3.0 * sqrt(syy), (tu.dims.y + 15u) / 16u);
  proj_tiles[k] = vec2u(sx.x | (sy.x << 16u), sx.y | (sy.y << 16u));
  for (var y = sy.x; y <= sy.y; y++) {
    for (var x = sx.x; x <= sx.y; x++) { atomicAdd(&tile_counts[y * tx + x], 1u); }
  }
}

// === tile_scan ===
// Per-tile list ranges from tile_counts (one workgroup): exclusive prefix sum ->
// tile_ranges[t] = [start, count] (count clipped to the list capacity), tile_cursor[t] =
// start (for tile_fill), bin_stats[0] = total entries (unclipped, for resizing).
// Bindings: 0 BinUniforms, 1 tile_counts, 2 tile_ranges, 3 tile_cursor, 4 bin_stats.

struct BinUniforms {
  n_vis    : u32,  // sorted slots
  tiles_x  : u32,
  n_tiles  : u32,
  capacity : u32,  // tile_list length
};

@group(0) @binding(0) var<uniform> bu : BinUniforms;
@group(0) @binding(1) var<storage, read>       tile_counts : array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_ranges : array<vec2u>;
@group(0) @binding(3) var<storage, read_write> tile_cursor : array<u32>;
@group(0) @binding(4) var<storage, read_write> bin_stats   : array<u32>;

var<workgroup> wg_sum : array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  // Each thread owns a contiguous chunk of tiles; scan the 256 chunk sums, then the chunks.
  let chunk = (bu.n_tiles + 255u) / 256u;
  let t0 = lid * chunk;
  let t1 = min(t0 + chunk, bu.n_tiles);
  var s = 0u;
  for (var t = t0; t < t1; t++) { s += tile_counts[t]; }
  wg_sum[lid] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off *= 2u) {  // inclusive Hillis-Steele scan
    let v = select(0u, wg_sum[lid - off], lid >= off);
    workgroupBarrier();
    wg_sum[lid] += v;
    workgroupBarrier();
  }
  var start = wg_sum[lid] - s;
  for (var t = t0; t < t1; t++) {
    let c = tile_counts[t];
    let clipped = min(c, select(0u, bu.capacity - start, start < bu.capacity));
    tile_ranges[t] = vec2u(start, clipped);
    tile_cursor[t] = start;
    start += c;
  }
  if (lid == 255u) { bin_stats[0] = wg_sum[255]; }
}

// === tile_fill ===
// Appends every sorted slot to the list of each tile of its proj_tiles rect (unordered;
// tile_sort restores the depth order). Entries past the capacity are dropped.
// Bindings: 0 BinUniforms, 1 proj_tiles, 2 tile_cursor, 3 tile_list.

struct BinUniforms {
  n_vis    : u32,  // sorted slots
  tiles_x  : u32,
  n_tiles  : u32,
  capacity : u32,  // tile_list length
};

@group(0) @binding(0) var<uniform> bu : BinUniforms;
@group(0) @binding(1) var<storage, read>       proj_tiles  : array<vec2u>;
@group(0) @binding(2) var<storage, read_write> tile_cursor : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tile_list   : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= bu.n_vis) { return; }
  let r = proj_tiles[k];
  let x0 = r.x & 0xffffu;
  let y0 = r.x >> 16u;
  let x1 = r.y & 0xffffu;
  let y1 = r.y >> 16u;
  if (x0 > x1 || y0 > y1) { return; }
  for (var y = y0; y <= y1; y++) {
    for (var x = x0; x <= x1; x++) {
      let pos = atomicAdd(&tile_cursor[y * bu.tiles_x + x], 1u);
      if (pos < bu.capacity) { tile_list[pos] = k; }
    }
  }
}

// === tile_sort ===
// One workgroup per tile sorts its list ascending (= front-to-back slot order): bottom-up
// merge sort ping-ponging between tile_list and tile_scratch, each element placed at its
// rank (own index + binary-searched position in the sibling run). Keys are unique.
// Bindings: 0 BinUniforms, 1 tile_ranges, 2 tile_list, 3 tile_scratch.

struct BinUniforms {
  n_vis    : u32,  // sorted slots
  tiles_x  : u32,
  n_tiles  : u32,
  capacity : u32,  // tile_list length
};

@group(0) @binding(0) var<uniform> bu : BinUniforms;
@group(0) @binding(1) var<storage, read>       tile_ranges  : array<vec2u>;
@group(0) @binding(2) var<storage, read_write> tile_list    : array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_scratch : array<u32>;

var<workgroup> wg_range : vec2u;

fn load(from_list: bool, i: u32) -> u32 {
  if (from_list) { return tile_list[i]; }
  return tile_scratch[i];
}

// Number of elements < v (strict) or <= v in the sorted run [lo, hi).
fn rank_in(from_list: bool, lo: u32, hi: u32, v: u32, strict: bool) -> u32 {
  var a = lo;
  var b = hi;
  while (a < b) {
    let m = (a + b) / 2u;
    let x = load(from_list, m);
    if (x < v || (!strict && x == v)) { a = m + 1u; } else { b = m; }
  }
  return a - lo;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) { wg_range = tile_ranges[wgid.y * bu.tiles_x + wgid.x]; }
  let range = workgroupUniformLoad(&wg_range);
  let start = range.x;
  let n = range.y;
  var in_list = true;  // where the current runs live
  for (var w = 1u; w < n; w *= 2u) {
    for (var j = lid; j < n; j += 256u) {
      let v = load(in_list, start + j);
      let pair = (j / (2u * w)) * 2u * w;   // start of this pair of runs
      let mid  = min(pair + w, n);
      let end  = min(pair + 2u * w, n);
      var dst : u32;
      if (j < mid) { dst = pair + (j - pair) + rank_in(in_list, start + mid, start + end, v, true); }
      else         { dst = pair + (j - mid) + rank_in(in_list, start + pair, start + mid, v, false); }
      if (in_list) { tile_scratch[start + dst] = v; } else { tile_list[start + dst] = v; }
    }
    storageBarrier();
    workgroupBarrier();
    in_list = !in_list;
  }
  if (!in_list) {
    for (var j = lid; j < n; j += 256u) { tile_list[start + j] = tile_scratch[start + j]; }
  }
}

// === tile_forward ===
// 16x16-pixel tile rasterizer: one workgroup per tile composites its list of sorted
// slots (tile_list[start .. start + count), front-to-back) in shared-memory batches;
// stops at T < 1e-4.
// Writes color to out_tex and pix_state = [T_final, z_med, bitcast<f32>(last_k), 0],
// z_med = depth where T crosses 0.5, last_k = 1 + last contributing sorted slot.
// Bindings: 0 TrainUniforms, 1 splats, 2 proj_splats, 3 out_tex, 4 pix_state,
//           5 tile_ranges ([start, count] per tile, row-major), 6 tile_list.

struct TrainUniforms {
  view       : mat4x4f,
  cam_pos    : vec4f, // xyz = camera center, w = Adam step (1-based)
  intrinsics : vec4f, // fx, fy, cx, cy
  dims       : vec4u, // W, H, num_visible, total_splats
  lrs        : vec4f, // lr_pos, lr_scale, lr_quat, lr_color
  opt_params : vec4f, // x = opt_method (0 Adam, 1 MC, 2 SA), y = mc_noise | sa_temp, z = rng seed, w = scene_radius
  limits     : vec4f, // x = max splat angle (scale / depth), y = max world scale, z = log(max anisotropy), w = lr_opacity
};

@group(0) @binding(0) var<uniform> tu : TrainUniforms;
@group(0) @binding(1) var<storage, read>       splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       proj_splats : array<vec4f>;
@group(0) @binding(3) var                      out_tex     : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> pix_state   : array<vec4f>;
@group(0) @binding(5) var<storage, read>       tile_ranges : array<vec2u>;
@group(0) @binding(6) var<storage, read>       tile_list   : array<u32>;

const BATCH : u32 = 64u;
const kNoSplat = vec4f(-1e5, -1e5, -1.0, 0.0); // radius 0 = skip

var<workgroup> wg_u0v0zr : array<vec4f, BATCH>;
var<workgroup> wg_conic  : array<vec4f, BATCH>;
var<workgroup> wg_rgba   : array<vec4f, BATCH>;
var<workgroup> wg_k      : array<u32, BATCH>;
var<workgroup> wg_hits   : atomic<u32>;

const SH_C0 : f32 = 0.28209479177387814;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -15.0, 15.0)));
}

// Conservative tile test: the splat's 3-sigma box overlaps the 16x16 tile.
fn tile_hit(pr0: vec4f, tile_c: vec2f) -> bool {
  return pr0.w > 0.0 && all(abs(pr0.xy - tile_c) <= vec2f(pr0.w + 8.5));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(workgroup_id) wgid: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let W = tu.dims.x;
  let H = tu.dims.y;
  let num_vis = tu.dims.z;
  let in_bounds = gid.x < W && gid.y < H;
  let px = vec2f(gid.xy) + 0.5;
  let tile_c = vec2f(wgid.xy * 16u) + 8.0;

  var C = vec3f(0.0);
  var T = 1.0;
  var z_med = 0.0;
  var last_k = 0u;
  var done = !in_bounds;

  let range = tile_ranges[wgid.y * ((W + 15u) / 16u) + wgid.x];
  let num_batches = (range.y + BATCH - 1u) / BATCH;
  for (var batch = 0u; batch < num_batches; batch++) {
    if (lid == 0u) { atomicStore(&wg_hits, 0u); }
    workgroupBarrier();

    // Load: one thread per batch entry; entries missing this tile get kNoSplat.
    if (lid < BATCH) {
      let j = batch * BATCH + lid;
      var e = kNoSplat;
      if (j < range.y) {
        let k = tile_list[range.x + j];
        wg_k[lid] = k;
        let pr0 = proj_splats[k * 2u + 0u];
        if (tile_hit(pr0, tile_c)) {
          let pr1 = proj_splats[k * 2u + 1u];
          let idx = bitcast<u32>(pr1.w);
          let p0  = splats[idx * 4u + 0u];
          let p3  = splats[idx * 4u + 3u];
          wg_conic[lid] = pr1;
          wg_rgba[lid]  = vec4f(clamp(vec3f(0.5) + SH_C0 * p3.xyz, vec3f(0.0), vec3f(1.0)), sigmoid(p0.w));
          atomicAdd(&wg_hits, 1u);
          e = pr0;
        }
      }
      wg_u0v0zr[lid] = e;
    }
    workgroupBarrier();

    if (!done && atomicLoad(&wg_hits) > 0u) {
      let batch_end = min(BATCH, range.y - batch * BATCH);
      for (var i = 0u; i < batch_end; i++) {
        let u0v0zr = wg_u0v0zr[i];
        let r = u0v0zr.w;
        if (r <= 0.0) { continue; }
        let dx = px.x - u0v0zr.x;
        let dy = px.y - u0v0zr.y;
        if (abs(dx) > r || abs(dy) > r) { continue; }
        let conic = wg_conic[i].xyz;
        let power = 0.5 * (conic.x * dx * dx + 2.0 * conic.y * dx * dy + conic.z * dy * dy);
        if (power > 4.5) { continue; }
        let rgba  = wg_rgba[i];
        let alpha = min(0.99, rgba.w * exp(-power));
        if (alpha < 1.0 / 255.0) { continue; }

        let next_T = T * (1.0 - alpha);
        if (T >= 0.5 && next_T < 0.5) { z_med = u0v0zr.z; }
        C += (T * alpha) * rgba.rgb;
        T = next_T;
        last_k = wg_k[i] + 1u;
        if (T < 1e-4) { done = true; break; }
      }
    }
    workgroupBarrier();
  }

  if (in_bounds) {
    textureStore(out_tex, vec2i(gid.xy), vec4f(C, 1.0));
    pix_state[gid.y * W + gid.x] = vec4f(T, z_med, bitcast<f32>(last_k), 0.0);
  }
}

// === tile_backward ===
// Backward pass of tile_forward over the same tile lists, batches walked back-to-front
// up to each pixel's last_k. T_i is recovered as T_{i+1} / (1 - alpha_i).
// Per splat, accumulates the 2D grads (see header) into grads_2d and the WLBG
// error moments into moments (S is done in adam_step). Contributions are summed
// over the subgroup first (when the device has 'subgroups'; lines tagged @sg / @nosg
// are selected by loadShaders()), then in workgroup memory: one global CAS per slot
// per batch per tile.
// Loss: 0.8 * smooth-L1 + 0.2 * L2 per channel, normalized by pixel count;
// loss_acc[0..1] = 64-bit sum (lo, hi) of the per-pixel MSE in 16.16 fixed point.
// Bindings: 0 TrainUniforms, 1 splats, 2 proj_splats, 3 gt_tex, 4 hat_tex,
//           5 pix_state, 6 grads_2d, 7 moments, 8 loss_acc, 9 tile_ranges, 10 tile_list.

struct TrainUniforms {
  view       : mat4x4f,
  cam_pos    : vec4f, // xyz = camera center, w = Adam step (1-based)
  intrinsics : vec4f, // fx, fy, cx, cy
  dims       : vec4u, // W, H, num_visible, total_splats
  lrs        : vec4f, // lr_pos, lr_scale, lr_quat, lr_color
  opt_params : vec4f, // x = opt_method (0 Adam, 1 MC, 2 SA), y = mc_noise | sa_temp, z = rng seed, w = scene_radius
  limits     : vec4f, // x = max splat angle (scale / depth), y = max world scale, z = log(max anisotropy), w = lr_opacity
};

@group(0) @binding(0) var<uniform> tu : TrainUniforms;
@group(0) @binding(1) var<storage, read>       splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       proj_splats : array<vec4f>;
@group(0) @binding(3) var                      gt_tex      : texture_2d<f32>;
@group(0) @binding(4) var                      hat_tex     : texture_2d<f32>;
@group(0) @binding(5) var<storage, read>       pix_state   : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> grads_2d    : array<atomic<u32>>; // 12 f32 per splat
@group(0) @binding(7) var<storage, read_write> moments     : array<atomic<u32>>; // 12 f32 per splat
@group(0) @binding(8) var<storage, read_write> loss_acc    : array<atomic<u32>, 4>;
@group(0) @binding(9) var<storage, read>       tile_ranges : array<vec2u>;
@group(0) @binding(10) var<storage, read>      tile_list   : array<u32>;

const BATCH : u32 = 32u;
const SLOTS : u32 = BATCH * 12u;
const kNoSplat = vec4f(-1e5, -1e5, -1.0, 0.0); // radius 0 = skip

var<workgroup> wg_u0v0zr     : array<vec4f, BATCH>;
var<workgroup> wg_conic      : array<vec4f, BATCH>;
var<workgroup> wg_rgba       : array<vec4f, BATCH>;
var<workgroup> wg_cmask      : array<vec4f, BATCH>; // 1 where rgb is not clamped
var<workgroup> wg_k          : array<u32, BATCH>;   // sorted slot of each batch entry
var<workgroup> wg_grads      : array<atomic<u32>, SLOTS>;
var<workgroup> wg_moms       : array<atomic<u32>, SLOTS>;
var<workgroup> wg_mse        : atomic<u32>;
var<workgroup> wg_hits       : atomic<u32>;
var<workgroup> wg_max_last_k : atomic<u32>;

const SH_C0 : f32 = 0.28209479177387814;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -15.0, 15.0)));
}

// Conservative tile test: the splat's 3-sigma box overlaps the 16x16 tile.
fn tile_hit(pr0: vec4f, tile_c: vec2f) -> bool {
  return pr0.w > 0.0 && all(abs(pr0.xy - tile_c) <= vec2f(pr0.w + 8.5));
}

// f32 atomic adds via CAS (WGSL has no pointer params to workgroup/storage
// atomics, hence one copy per array). Tiny values are skipped.
fn add_wg_grad(i: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&wg_grads[i]);
  loop {
    let r = atomicCompareExchangeWeak(&wg_grads[i], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn add_wg_mom(i: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&wg_moms[i]);
  loop {
    let r = atomicCompareExchangeWeak(&wg_moms[i], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn add_global_grad(i: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&grads_2d[i]);
  loop {
    let r = atomicCompareExchangeWeak(&grads_2d[i], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn add_global_mom(i: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&moments[i]);
  loop {
    let r = atomicCompareExchangeWeak(&moments[i], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(workgroup_id) wgid: vec3u,
        @builtin(subgroup_invocation_id) sg_lane: u32,  // @sg
        @builtin(local_invocation_index) lid: u32) {
  let W = tu.dims.x;
  let H = tu.dims.y;
  let num_vis = tu.dims.z;
  let in_bounds = gid.x < W && gid.y < H;
  let px = vec2f(gid.xy) + 0.5;
  let tile_c = vec2f(wgid.xy * 16u) + 8.0;

  if (lid == 0u) {
    atomicStore(&wg_mse, 0u);
    atomicStore(&wg_hits, 0u);
    atomicStore(&wg_max_last_k, 0u);
  }
  for (var s = lid; s < SLOTS; s += 256u) {
    atomicStore(&wg_grads[s], 0u);
    atomicStore(&wg_moms[s], 0u);
  }
  workgroupBarrier();

  var dL_dC = vec3f(0.0);
  var err_l1 : f32 = 0.0;
  var T : f32 = 1.0;
  var last_k = 0u;
  var accum_rec = vec3f(0.0); // color behind the current splat, unattenuated by T

  if (in_bounds) {
    let gt   = textureLoad(gt_tex,  vec2i(gid.xy), 0).rgb;
    let hat  = textureLoad(hat_tex, vec2i(gid.xy), 0).rgb;
    let diff = hat - gt;
    atomicAdd(&wg_mse, u32(dot(diff, diff) / 3.0 * 65536.0 + 0.5));
    let sign_d = clamp(diff * 25.0, vec3f(-1.0), vec3f(1.0));
    dL_dC  = (0.8 * sign_d + 0.4 * diff) * (1.0 / f32(W * H));
    err_l1 = (abs(diff.x) + abs(diff.y) + abs(diff.z)) / 3.0;

    let st = pix_state[gid.y * W + gid.x];
    T = st.x;
    last_k = bitcast<u32>(st.z);
    atomicMax(&wg_max_last_k, last_k);
  }
  workgroupBarrier();
  if (lid == 0u) {  // 64-bit add: carry into loss_acc[1] when the low word wraps
    let v = atomicLoad(&wg_mse);
    let old = atomicAdd(&loss_acc[0], v);
    if (old + v < old) { atomicAdd(&loss_acc[1], 1u); }
  }

  let tile_max_k = atomicLoad(&wg_max_last_k);
  let fx = tu.intrinsics.x;
  let fy = tu.intrinsics.y;
  let cam_to_world = transpose(mat3x3f(tu.view[0].xyz, tu.view[1].xyz, tu.view[2].xyz));

  let range = tile_ranges[wgid.y * ((W + 15u) / 16u) + wgid.x];
  let num_batches = (range.y + BATCH - 1u) / BATCH;
  for (var b_rev = 0u; b_rev < num_batches; b_rev++) {
    let batch = num_batches - 1u - b_rev;
    let base = batch * BATCH;

    // Load (entries past every pixel's last_k are skipped).
    if (lid < BATCH) {
      let j = base + lid;
      var e = kNoSplat;
      var k = 0xffffffffu;
      if (j < range.y) { k = tile_list[range.x + j]; }
      wg_k[lid] = k;
      if (k < tile_max_k) {
        let pr0 = proj_splats[k * 2u + 0u];
        if (tile_hit(pr0, tile_c)) {
          let pr1 = proj_splats[k * 2u + 1u];
          let idx = bitcast<u32>(pr1.w);
          let p0  = splats[idx * 4u + 0u];
          let p3  = splats[idx * 4u + 3u];
          let rgb_raw = vec3f(0.5) + SH_C0 * p3.xyz;
          let rgb = clamp(rgb_raw, vec3f(0.0), vec3f(1.0));
          wg_conic[lid] = pr1;
          wg_rgba[lid]  = vec4f(rgb, sigmoid(p0.w));
          wg_cmask[lid] = vec4f(select(vec3f(0.0), vec3f(1.0), rgb == rgb_raw), 0.0);
          atomicAdd(&wg_hits, 1u);
          e = pr0;
        }
      }
      wg_u0v0zr[lid] = e;
    }
    workgroupBarrier();

    // Every thread walks the batch (uniformly, for the subgroup sums); `hit` = this
    // pixel gets a contribution from entry i.
    let has_hits = atomicLoad(&wg_hits) > 0u;
    if (has_hits) {
      let batch_end = min(BATCH, range.y - base);
      for (var ir = 0u; ir < batch_end; ir++) {
        let i = batch_end - 1u - ir;
        let u0v0zr = wg_u0v0zr[i];
        let r = u0v0zr.w;
        let dx = px.x - u0v0zr.x;
        let dy = px.y - u0v0zr.y;
        let conic = wg_conic[i].xyz;
        let power = 0.5 * (conic.x * dx * dx + 2.0 * conic.y * dx * dy + conic.z * dy * dy);
        let rgba  = wg_rgba[i];
        let alpha = min(0.99, rgba.w * exp(-power));
        let hit = in_bounds && wg_k[i] < last_k && r > 0.0 && abs(dx) <= r && abs(dy) <= r &&
                     power <= 4.5 && alpha >= 1.0 / 255.0;
        if (!hit) { continue; }                                      // @nosg

        var g = array<f32, 11>();  // grads_2d slots 0..10
        var m = array<f32, 12>();  // moments slots 0..11 (1 = S, done in adam_step)
        if (hit) {
          T = T / max(1e-4, 1.0 - alpha); // T_i
          let weight = T * alpha;
          let w_err  = weight * err_l1;

          let dL_drgb   = dL_dC * weight * SH_C0 * wg_cmask[i].xyz; // w.r.t. f_dc, zero where clamped
          let dL_dalpha = T * dot(dL_dC, rgba.rgb - accum_rec);
          accum_rec = alpha * rgba.rgb + (1.0 - alpha) * accum_rec;

          let dL_dpower = -dL_dalpha * alpha;
          g[0] = dL_drgb.x;
          g[1] = dL_drgb.y;
          g[2] = dL_drgb.z;
          g[3] = dL_dalpha * alpha * (1.0 - rgba.w);                // dL/dlogit_opacity
          g[4] = -dL_dpower * (conic.x * dx + conic.y * dy);        // dL/du0
          g[5] = -dL_dpower * (conic.y * dx + conic.z * dy);        // dL/dv0
          g[6] = 0.5 * dL_dpower * dx * dx;                         // dL/dconic.a
          g[7] =       dL_dpower * dx * dy;                         // dL/dconic.b (full, b appears twice)
          g[8] = 0.5 * dL_dpower * dy * dy;                         // dL/dconic.c
          g[9] = weight;                                            // w_view
          g[10] = w_err;                                            // w_err_view (SA energy)

          // WLBG moments; delta = world-space pixel offset from the center at its depth.
          let d = cam_to_world * vec3f(dx * u0v0zr.z / fx, dy * u0v0zr.z / fy, 0.0);
          m[0] = weight;
          m[2] = w_err;
          m[3] = w_err * d.x;
          m[4] = w_err * d.y;
          m[5] = w_err * d.z;
          m[6] = w_err * d.x * d.x;
          m[7] = w_err * d.y * d.y;
          m[8] = w_err * d.z * d.z;
          m[9] = w_err * d.x * d.y;
          m[10] = w_err * d.x * d.z;
          m[11] = w_err * d.y * d.z;
        }
        let gbase = i * 12u;
        if (subgroupAny(hit)) {                                   // @sg
          for (var c = 0u; c < 11u; c++) { g[c] = subgroupAdd(g[c]); }  // @sg
          for (var c = 0u; c < 12u; c++) { m[c] = subgroupAdd(m[c]); }  // @sg
          if (sg_lane == 0u) {                                       // @sg
            for (var c = 0u; c < 11u; c++) { add_wg_grad(gbase + c, g[c]); }  // @sg
            for (var c = 0u; c < 12u; c++) { add_wg_mom(gbase + c, m[c]); }   // @sg
          }                                                          // @sg
        }                                                            // @sg
        if (hit) {                                                // @nosg
          for (var c = 0u; c < 11u; c++) { add_wg_grad(gbase + c, g[c]); }  // @nosg
          for (var c = 0u; c < 12u; c++) { add_wg_mom(gbase + c, m[c]); }   // @nosg
        }                                                            // @nosg
      }
    }
    workgroupBarrier();

    // Flush workgroup sums to global memory and zero them for the next batch.
    if (has_hits) {
      for (var s = lid; s < SLOTS; s += 256u) {
        let g = bitcast<f32>(atomicExchange(&wg_grads[s], 0u));
        let m = bitcast<f32>(atomicExchange(&wg_moms[s], 0u));
        let j = s / 12u;
        if (wg_u0v0zr[j].w > 0.0) {
          let gi = bitcast<u32>(wg_conic[j].w) * 12u + s % 12u;
          add_global_grad(gi, g);
          add_global_mom(gi, m);
        }
      }
      if (lid == 0u) { atomicStore(&wg_hits, 0u); }
    }
    workgroupBarrier();
  }
}

// === adam_step ===
// Per-splat update, one thread per splat:
//   1. reads and clears grads_2d, adds w_view^2 to moments.S;
//   2. chains the 2D grads to 3D (position incl. through the EWA Jacobian,
//      log-scale, quaternion), Adam step (sparse: splats with no grad are
//      skipped), clamps and shape limits;
//   3. opt_method 1 (MC/SGLD): covariance-shaped position noise ~ (1 - alpha)^2;
//      opt_method 2 (SA): Metropolis accept/reject of position jumps against
//      the per-view mean error. SA state lives in the spare Adam .w slots:
//      m[1].w, m[3].w, v[3].w = last jump xyz, v[1].w = energy baseline.
// Bindings: 0 TrainUniforms, 1 splats, 2 grads_2d, 3 adam_m, 4 adam_v, 5 moments.

struct TrainUniforms {
  view       : mat4x4f,
  cam_pos    : vec4f, // xyz = camera center, w = Adam step (1-based)
  intrinsics : vec4f, // fx, fy, cx, cy
  dims       : vec4u, // W, H, num_visible, total_splats
  lrs        : vec4f, // lr_pos, lr_scale, lr_quat, lr_color
  opt_params : vec4f, // x = opt_method (0 Adam, 1 MC, 2 SA), y = mc_noise | sa_temp, z = rng seed, w = scene_radius
  limits     : vec4f, // x = max splat angle (scale / depth), y = max world scale, z = log(max anisotropy), w = lr_opacity
};

@group(0) @binding(0) var<uniform> tu : TrainUniforms;
@group(0) @binding(1) var<storage, read_write> splats   : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> grads_2d : array<u32>;
@group(0) @binding(3) var<storage, read_write> adam_m   : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> adam_v   : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> moments  : array<u32>;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -15.0, 15.0)));
}

// q = (w, x, y, z), normalized here. Returns column-major R.
fn quat_to_mat3(q: vec4f) -> mat3x3f {
  let n = max(1e-8, length(q));
  let w = q.x / n; let x = q.y / n; let y = q.z / n; let z = q.w / n;
  return mat3x3f(
    vec3f(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y)),
    vec3f(2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
    vec3f(2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y))
  );
}

// EWA: T = J * W (pixel Jacobian at view-space point vpos times view rotation).
// Screen covariance is T * Sigma * T^T. As in 3DGS, J is evaluated with x/z, y/z clamped
// to `lim` (1.3x the frustum, see frustum_lim): off-axis splats near the camera would
// otherwise project to huge ellipses.
fn ewa_jacobian(view: mat4x4f, vpos: vec3f, fx: f32, fy: f32, lim: vec2f) -> mat3x2f {
  let inv_z  = 1.0 / vpos.z;
  let t = clamp(vpos.xy * inv_z, -lim, lim);  // clamped x/z, y/z
  let J = mat3x2f(
    vec2f(fx * inv_z, 0.0),
    vec2f(0.0, fy * inv_z),
    vec2f(-fx * t.x * inv_z, -fy * t.y * inv_z)
  );
  return J * mat3x3f(view[0].xyz, view[1].xyz, view[2].xyz);
}

// |x/z|, |y/z| bounds of 1.3x the view frustum (W x H, principal point cx, cy).
fn frustum_lim(W: f32, H: f32, fx: f32, fy: f32, cx: f32, cy: f32) -> vec2f {
  return 1.3 * vec2f(max(cx, W - cx) / fx, max(cy, H - cy) / fy);
}

fn pcg_hash(v_in: u32) -> u32 {
  let state = v_in * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return clamp(f32(pcg_hash(seed) & 0x00ffffffu) / 16777216.0, 1e-6, 0.999999);
}

// 3 standard normal samples (Box-Muller).
fn rand_normal3(idx: u32, step_u: u32) -> vec3f {
  let s0 = idx * 1973u + step_u * 9277u + u32(tu.opt_params.z * 65536.0);
  let u1 = rand01(s0 + 1u);
  let u2 = rand01(s0 + 2u);
  let u3 = rand01(s0 + 3u);
  let u4 = rand01(s0 + 4u);
  let r1 = sqrt(-2.0 * log(u1));
  let r2 = sqrt(-2.0 * log(u3));
  return vec3f(r1 * cos(6.2831853 * u2), r1 * sin(6.2831853 * u2), r2 * cos(6.2831853 * u4));
}

// Adam (b1 = 0.9, b2 = 0.999, grads clipped to [-1, 1]); bc = bias corrections.
fn adam4(p: vec4f, grad: vec4f, m: ptr<function, vec4f>, v: ptr<function, vec4f>, lr: vec4f, bc: vec2f) -> vec4f {
  let g = clamp(grad, vec4f(-1.0), vec4f(1.0));
  *m = 0.9 * (*m) + 0.1 * g;
  *v = 0.999 * (*v) + 0.001 * g * g;
  return p - lr * ((*m) / bc.x) / (sqrt((*v) / bc.y) + 1e-15);
}

// Same on xyz only; m.w / v.w are left untouched.
fn adam3(p: vec3f, grad: vec3f, m: ptr<function, vec4f>, v: ptr<function, vec4f>, lr: f32, bc: vec2f) -> vec3f {
  let g = clamp(grad, vec3f(-1.0), vec3f(1.0));
  let mn = 0.9 * (*m).xyz + 0.1 * g;
  let vn = 0.999 * (*v).xyz + 0.001 * g * g;
  *m = vec4f(mn, (*m).w);
  *v = vec4f(vn, (*v).w);
  return p - lr * (mn / bc.x) / (sqrt(vn / bc.y) + 1e-15);
}

fn load_grad(i: u32) -> f32 {
  return bitcast<f32>(grads_2d[i]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= tu.dims.w) { return; }

  let gbase = idx * 12u;
  let dL_drgb = vec3f(load_grad(gbase + 0u), load_grad(gbase + 1u), load_grad(gbase + 2u));
  let dL_dop  = load_grad(gbase + 3u);
  let dL_du0  = load_grad(gbase + 4u);
  let dL_dv0  = load_grad(gbase + 5u);
  let dL_da   = load_grad(gbase + 6u);
  let dL_db   = load_grad(gbase + 7u);
  let dL_dc   = load_grad(gbase + 8u);
  let w_view  = load_grad(gbase + 9u);
  let w_err_v = load_grad(gbase + 10u);
  for (var c = 0u; c < 12u; c++) { grads_2d[gbase + c] = 0u; }

  // S_i += w_{i,c}^2 (view concentration, N_eff = W^2 / S).
  if (w_view > 1e-6) {
    moments[gbase + 1u] = bitcast<u32>(bitcast<f32>(moments[gbase + 1u]) + w_view * w_view);
  }

  let grad_norm = abs(dL_drgb.x) + abs(dL_drgb.y) + abs(dL_drgb.z) + abs(dL_dop) + abs(dL_du0) + abs(dL_dv0);
  if (grad_norm < 1e-10) { return; }

  let b  = idx * 4u;
  var p0 = splats[b + 0u];
  var p1 = splats[b + 1u];
  var p2 = splats[b + 2u];
  var p3 = splats[b + 3u];

  let vpos = tu.view * vec4f(p0.xyz, 1.0);
  if (vpos.z <= 0.1) { return; }

  let fx = tu.intrinsics.x;
  let fy = tu.intrinsics.y;
  let inv_z  = 1.0 / vpos.z;
  let inv_z2 = inv_z * inv_z;
  let inv_z3 = inv_z2 * inv_z;
  let W3 = mat3x3f(tu.view[0].xyz, tu.view[1].xyz, tu.view[2].xyz);

  // (u0, v0) -> view-space position.
  let dL_dvpos = vec3f(
    dL_du0 * fx * inv_z,
    dL_dv0 * fy * inv_z,
    -(dL_du0 * fx * vpos.x + dL_dv0 * fy * vpos.y) * inv_z2
  );

  // Recompute the forward covariance (must match project_splats).
  let qn = max(1e-8, length(p2));
  let qw = p2.x / qn; let qx = p2.y / qn; let qy = p2.z / qn; let qz = p2.w / qn;
  let R = quat_to_mat3(p2);
  let s = exp(clamp(p1.xyz, vec3f(-10.0), vec3f(3.0)));
  let M = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let Sigma = M * transpose(M);
  let lim = frustum_lim(f32(tu.dims.x), f32(tu.dims.y), fx, fy, tu.intrinsics.z, tu.intrinsics.w);
  let T = ewa_jacobian(tu.view, vpos.xyz, fx, fy, lim);
  let cov2d = T * Sigma * transpose(T);
  let sxx = cov2d[0][0] + 0.3;
  let syy = cov2d[1][1] + 0.3;
  let sxy = cov2d[0][1];
  let det = max(1e-6, sxx * syy - sxy * sxy);
  let inv_det = 1.0 / det;
  let ca = syy * inv_det;
  let cb = -sxy * inv_det;
  let cc = sxx * inv_det;

  // dL/dcov2d = -conic * dL/dconic * conic, with dL/dconic = [[da, db/2], [db/2, dc]].
  let dL_dsxx = -(ca * ca * dL_da + ca * cb * dL_db + cb * cb * dL_dc);
  let dL_dsyy = -(cb * cb * dL_da + cb * cc * dL_db + cc * cc * dL_dc);
  let dL_dsxy = -(ca * cb * dL_da + 0.5 * (ca * cc + cb * cb) * dL_db + cb * cc * dL_dc);
  let dL_dcov2d = mat2x2f(vec2f(dL_dsxx, dL_dsxy), vec2f(dL_dsxy, dL_dsyy));
  let dL_dSigma = transpose(T) * dL_dcov2d * T;
  let dL_dM = 2.0 * dL_dSigma * M;

  // cov2d also depends on the view-space position through J.
  let dL_dJ = 2.0 * dL_dcov2d * T * Sigma * transpose(W3);
  // A clamped x/z (y/z) makes J's x (y) column independent of x (y), and halves its z term.
  let tq = vpos.xy * inv_z;
  let free = vec2f(select(0.0, 1.0, abs(tq.x) <= lim.x), select(0.0, 1.0, abs(tq.y) <= lim.y));
  let tc = clamp(tq, -lim, lim) * vpos.z;  // clamped x, y
  let dL_dvpos_cov = vec3f(
    -fx * inv_z2 * dL_dJ[2][0] * free.x,
    -fy * inv_z2 * dL_dJ[2][1] * free.y,
    -fx * inv_z2 * dL_dJ[0][0] - fy * inv_z2 * dL_dJ[1][1] +
      inv_z3 * ((1.0 + free.x) * fx * tc.x * dL_dJ[2][0] + (1.0 + free.y) * fy * tc.y * dL_dJ[2][1])
  );
  let dL_dpos = transpose(W3) * (dL_dvpos + dL_dvpos_cov);

  let dL_dlog_s = vec3f(dot(R[0], dL_dM[0]) * s.x, dot(R[1], dL_dM[1]) * s.y, dot(R[2], dL_dM[2]) * s.z);

  // dL/dq from dL/dR = [G0, G1, G2] (columns, column-major R).
  let G0 = dL_dM[0] * s.x;
  let G1 = dL_dM[1] * s.y;
  let G2 = dL_dM[2] * s.z;
  let dL_dq = 2.0 * vec4f(
    qx * (G1.z - G2.y) + qy * (G2.x - G0.z) + qz * (G0.y - G1.x),
    qy * (G0.y + G1.x) + qz * (G0.z + G2.x) + qw * (G1.z - G2.y) - 2.0 * qx * (G1.y + G2.z),
    qx * (G0.y + G1.x) + qw * (G2.x - G0.z) + qz * (G1.z + G2.y) - 2.0 * qy * (G0.x + G2.z),
    qw * (G0.y - G1.x) + qx * (G0.z + G2.x) + qy * (G1.z + G2.y) - 2.0 * qz * (G0.x + G1.y)
  );

  let step = max(1.0, tu.cam_pos.w);
  let step_u = u32(step);
  let bc = vec2f(1.0 - pow(0.9, step), 1.0 - pow(0.999, step));

  var m0 = adam_m[b + 0u]; var v0 = adam_v[b + 0u];
  var m1 = adam_m[b + 1u]; var v1 = adam_v[b + 1u];
  var m2 = adam_m[b + 2u]; var v2 = adam_v[b + 2u];
  var m3 = adam_m[b + 3u]; var v3 = adam_v[b + 3u];

  // Position + logit opacity.
  p0 = adam4(p0, vec4f(dL_dpos, dL_dop), &m0, &v0, vec4f(vec3f(tu.lrs.x), tu.limits.w), bc);
  p0.w = clamp(p0.w, -8.0, 8.0);

  // Log-scale, then shape limits: cap each axis by the angular size in this
  // view and by an absolute world size, and the largest axis by the median
  // one times max anisotropy (flat discs stay legal, needles don't).
  p1 = vec4f(clamp(adam3(p1.xyz, dL_dlog_s, &m1, &v1, tu.lrs.y, bc), vec3f(-9.5), vec3f(2.0)), p1.w);
  let log_cap = log(max(1e-8, min(tu.limits.x * vpos.z, tu.limits.y)));
  let ls_med = max(min(p1.x, p1.y), min(max(p1.x, p1.y), p1.z));
  let ls_max = min(log_cap, ls_med + tu.limits.z);
  p1 = vec4f(min(p1.xyz, vec3f(ls_max)), p1.w);

  p2 = normalize(adam4(p2, dL_dq, &m2, &v2, vec4f(tu.lrs.z), bc));
  p3 = vec4f(adam3(p3.xyz, dL_drgb, &m3, &v3, tu.lrs.w, bc), p3.w);

  // Stochastic extensions. m1/v1/m3/v3 .w still hold the SA state read above.
  let opt_mode = u32(tu.opt_params.x + 0.5);
  var energy = v1.w;
  var jump = vec3f(m1.w, m3.w, v3.w);
  if (opt_mode == 1u) {
    let noise_amp = tu.opt_params.y * (1.0 - sigmoid(p0.w)) * (1.0 - sigmoid(p0.w));
    p0 = vec4f(p0.xyz + noise_amp * (M * rand_normal3(idx, step_u)), p0.w);
  } else if (opt_mode == 2u && w_view > 1e-4) {
    let e_cur = w_err_v / w_view;
    let temp  = max(1e-5, tu.opt_params.y);
    let dE    = e_cur - energy;
    let u_acc = rand01(idx * 3571u + step_u * 6151u);
    if (energy > 0.0 && dE > 0.0 && u_acc > exp(-max(0.0, dE) / temp)) {
      // Reject: undo the last jump, try a fresh one.
      p0 = vec4f(p0.xyz - jump, p0.w);
      jump = (temp * 0.25) * (M * rand_normal3(idx + 17u, step_u));
    } else {
      // Accept: track the energy baseline and keep exploring.
      energy = select(e_cur, 0.7 * energy + 0.3 * e_cur, energy > 0.0);
      jump = (temp * 0.15 * max(0.05, e_cur)) * (M * rand_normal3(idx, step_u));
    }
    p0 = vec4f(p0.xyz + jump, p0.w);
  }

  adam_m[b + 0u] = m0;                        adam_v[b + 0u] = v0;
  adam_m[b + 1u] = vec4f(m1.xyz, jump.x);     adam_v[b + 1u] = vec4f(v1.xyz, energy);
  adam_m[b + 2u] = m2;                        adam_v[b + 2u] = v2;
  adam_m[b + 3u] = vec4f(m3.xyz, jump.y);     adam_v[b + 3u] = vec4f(v3.xyz, jump.z);

  splats[b + 0u] = p0;
  splats[b + 1u] = p1;
  splats[b + 2u] = p2;
  splats[b + 3u] = p3;
}
