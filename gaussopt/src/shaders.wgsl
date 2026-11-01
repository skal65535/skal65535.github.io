// shaders.wgsl — WGSL kernels for Gauss-Opt.
//
// Each `// === <name> ===` section is compiled as an independent module by
// loadShaders() (src/render.js), except the `_*` ones: those hold the shared
// structs/helpers, inserted where a module has `// #include _<name>`. Layouts must
// match src/render.js and src/trainer.js.
//
// Splat layout (4 vec4f): [x, y, z, logit_op], [log_sx, log_sy, log_sz, leaf_id],
//                         [qw, qx, qy, qz], [f_dc_r, f_dc_g, f_dc_b, flags]
// Moments (12 f32): W, S, R, R1.xyz, R2.(xx, yy, zz, xy, xz, yz)
// Grads   (12 f32): dL/drgb.xyz, dL/dlogit_op, dL/du0, dL/dv0,
//                   dL/dconic.(a, b, c), w_view, w_err_view, (unused)

// === _math ===
const SH_C0 : f32 = 0.28209479177387814;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -15.0, 15.0)));
}

fn pcg_hash(v_in: u32) -> u32 {
  let state = v_in * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
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

// The camera-to-world rotation of world-to-camera `view`, and pixel p's ray in the camera's frame (z = 1) at
// intrinsics k = (fx, fy, cx, cy)
fn view_to_world(view: mat4x4f) -> mat3x3f { return transpose(mat3x3f(view[0].xyz, view[1].xyz, view[2].xyz)); }
fn pixel_ray(p: vec2f, k: vec4f) -> vec3f { return vec3f((p - k.zw) / k.xy, 1.0); }

// |x/z|, |y/z| bounds of 1.3x the view frustum (W x H, principal point cx, cy).
fn frustum_lim(W: f32, H: f32, fx: f32, fy: f32, cx: f32, cy: f32) -> vec2f {
  return 1.3 * vec2f(max(cx, W - cx) / fx, max(cy, H - cy) / fy);
}

// A refined lens's residual radial distortion (poses.js, DESIGN.md E19 3; TrainUniforms cam.z), what the SfM's
// undistortion of the photos left: u = cx + fx x (1 + k1 r^2), v = cy + fy y (1 + k1 r^2), with x, y = x_c / z_c,
// y_c / z_c and r^2 = x^2 + y^2. D: its pixel Jacobian d(u, v) / d(u, v of the pinhole) at (x, y), by columns; the
// splat's screen covariance is (D T) Sigma (D T)^T.
fn distort_jac(t: vec2f, k1: f32, fx: f32, fy: f32) -> mat2x2f {
  let c = 2.0 * k1 * t.x * t.y;
  return mat2x2f(vec2f(1.0 + k1 * (3.0 * t.x * t.x + t.y * t.y), c * fy / fx),
                 vec2f(c * fx / fy, 1.0 + k1 * (t.x * t.x + 3.0 * t.y * t.y)));
}

// dL/dx, dL/dy, dL/dk1 through D's entries, given G = dL/dD
fn distort_jac_grad(t: vec2f, k1: f32, fx: f32, fy: f32, G: mat2x2f) -> vec3f {
  let a = fy / fx * G[0][1] + fx / fy * G[1][0];
  return vec3f(k1 * (6.0 * t.x * G[0][0] + 2.0 * t.y * a + 2.0 * t.x * G[1][1]),
               k1 * (2.0 * t.y * G[0][0] + 2.0 * t.x * a + 6.0 * t.y * G[1][1]),
               (3.0 * t.x * t.x + t.y * t.y) * G[0][0] + 2.0 * t.x * t.y * a + (t.x * t.x + 3.0 * t.y * t.y) * G[1][1]);
}

// === _train ===
struct TrainUniforms {
  view       : mat4x4f,
  intrinsics : vec4f, // fx, fy, cx, cy
  dims       : vec4u, // W, H, num_visible, total_splats
  lrs        : vec4f, // lr_pos, lr_scale, lr_quat, lr_color
  opt_params : vec4f, // x = opt_method (0 Adam, 1 MC), y = mc_noise, z = rng seed, w = Adam step (1-based)
  limits     : vec4f, // x = max splat angle (scale / depth), y = max world scale, z = log(max anisotropy), w = lr_opacity
  loss       : vec4f, // x = D-SSIM weight (ssim_*, tile_backward), y = DropGaussian rate (project_splats),
                      // z = its opacity factor 1 / (1 - y) (tile_forward / backward; 0 = 1), w unused
  reg        : vec4f, // monte_carlo's L1 terms, already over the splat count: x = on opacity, y = on scale (per axis); zw unused
  expo_g     : vec4f, // the view's exposure (exposure.js): the render as expo_g.rgb c + expo_b.rgb; w = 1 while it learns
  expo_b     : vec4f, //   (tile_backward sums its gradient)
  cam        : vec4f, // x = 1: adam_step sums the view's pose and lens gradient (cam_reduce; poses.js); y = 1: only that,
                      //   the splats, their Adam state and moments left as they are (a held-out camera fitted);
                      //   z: the lens's residual k1 (distort_jac; 0: the pinhole)
};

@group(0) @binding(0) var<uniform> tu : TrainUniforms;

// The opacity a splat renders with in training renders: DropGaussian's compensation of the dropped
fn train_opacity_factor() -> f32 { return select(1.0, tu.loss.z, tu.loss.z > 0.0); }

// === _proj ===
// A splat's screen covariance in the training view (tu): EWA at view-space vpos through the lens's residual distortion
// (distort_jac), the 0.3 px^2 low-pass dilation added. project_splats draws with it, adam_step chains its gradient.
struct SplatCov {
  R     : mat3x3f,  // the rotation (quat_to_mat3)
  scale : vec3f,    // the scales, exp of the log-scales clamped to [-10, 3]
  M     : mat3x3f,  // R x scales
  Sigma : mat3x3f,  // M M^T
  Tp    : mat3x2f,  // the pinhole's EWA Jacobian (ewa_jacobian)
  D     : mat2x2f,  // the distortion's (distort_jac, at tcn)
  T     : mat3x2f,  // D Tp
  tcn   : vec2f,    // x/z, y/z clamped to lim
  lim   : vec2f,    // frustum_lim
  cov   : vec3f,    // sxx, syy, sxy: the screen covariance, low-passed
}

fn splat_cov(vpos: vec3f, p1: vec4f, p2: vec4f) -> SplatCov {
  let fx = tu.intrinsics.x;
  let fy = tu.intrinsics.y;
  let R = quat_to_mat3(p2);
  let s = exp(clamp(p1.xyz, vec3f(-10.0), vec3f(3.0)));
  let M = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let Sigma = M * transpose(M);
  let lim = frustum_lim(f32(tu.dims.x), f32(tu.dims.y), fx, fy, tu.intrinsics.z, tu.intrinsics.w);
  let tcn = clamp(vpos.xy * (1.0 / vpos.z), -lim, lim);
  let D = distort_jac(tcn, tu.cam.z, fx, fy);
  let Tp = ewa_jacobian(tu.view, vpos, fx, fy, lim);
  let T = D * Tp;
  let cov2d = T * Sigma * transpose(T);
  return SplatCov(R, s, M, Sigma, Tp, D, T, tcn, lim, vec3f(cov2d[0][0] + 0.3, cov2d[1][1] + 0.3, cov2d[0][1]));
}

// === _ssim ===
// SSIM as rgbSsim (src/metrics.js): 11x11 Gaussian window (sigma 1.5, these weights per axis),
// truncated at the borders, C1 = 0.01^2, C2 = 0.03^2
const SSIM_R  : i32 = 5;
const SSIM_C1 : f32 = 0.0001;
const SSIM_C2 : f32 = 0.0009;
const SSIM_G = array<f32, 11>(0.00102838008, 0.00759875814, 0.0360007721, 0.109360690, 0.213005538, 0.266011725,
                              0.213005538, 0.109360690, 0.0360007721, 0.00759875814, 0.00102838008);

// The window moments around pixel q of gt_tex (x) and hat_tex (y), the includer's bindings
struct SsimMoments { mx: vec3f, my: vec3f, exx: vec3f, eyy: vec3f, exy: vec3f }
fn ssim_moments(q: vec2i, W: i32, H: i32) -> SsimMoments {
  var G = SSIM_G;
  var m = SsimMoments(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
  for (var dy = -SSIM_R; dy <= SSIM_R; dy++) {
    let py = q.y + dy;
    if (py < 0 || py >= H) { continue; }
    for (var dx = -SSIM_R; dx <= SSIM_R; dx++) {
      let px = q.x + dx;
      if (px < 0 || px >= W) { continue; }
      let g = G[dy + SSIM_R] * G[dx + SSIM_R];
      let x = textureLoad(gt_tex,  vec2i(px, py), 0).rgb;
      let y = textureLoad(hat_tex, vec2i(px, py), 0).rgb;
      m.mx += g * x; m.my += g * y;
      m.exx += g * x * x; m.eyy += g * y * y; m.exy += g * x * y;
    }
  }
  return m;
}

// === _raster ===
const kNoSplat = vec4f(-1e5, -1e5, -1.0, 0.0); // radius 0 = skip

// Conservative tile test: the splat's 3-sigma box overlaps the 16x16 tile.
fn tile_hit(pr0: vec4f, tile_c: vec2f) -> bool {
  return pr0.w > 0.0 && all(abs(pr0.xy - tile_c) <= vec2f(pr0.w + 8.5));
}

// === _bin ===
struct BinUniforms {
  n_vis    : u32,  // sorted slots
  tiles_x  : u32,
  n_tiles  : u32,
  capacity : u32,  // tile_list length
};

@group(0) @binding(0) var<uniform> bu : BinUniforms;

// === _wire ===
struct WireUniforms {
  view       : mat4x4f,
  intrinsics : vec4f, // fx, fy, cx, cy
  screen     : vec4f, // W, H, znear, zfar
  fade       : vec4f, // octree vertex count, full-alpha depth, faded depth, min alpha
  mesh       : vec4f, // x: mesh_render's margin (view z) behind its surface before a splat is hidden;
                      // y: the octree boxes' alpha behind the splats (wireframe_render)
};

@group(0) @binding(0) var<uniform> wu : WireUniforms;

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

// === splats_render ===
// Instanced quad renderer for the interactive view: 2.5-sigma screen-space
// quads per sorted splat, shape modes (vs/fs) and diagnostic color modes.
// Bindings: 0 RenderUniforms, 1 splats, 2 moments, 3 sorted_idx.

struct RenderUniforms {
  view       : mat4x4f,
  intrinsics : vec4f,   // fx, fy, cx, cy
  screen     : vec4f,   // W, H, haze (scale multiplier), scene_radius
  modes      : vec4u,   // x = visu_mode, y = color_mode, zw unused
  thresholds : vec4f,   // x = avg_R, y = znear, z = zfar, w unused
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

// #include _math

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
  if (vpos.z <= u.thresholds.y) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0); // clipped
    return out;
  }

  let fx = u.intrinsics.x;
  let fy = u.intrinsics.y;
  let inv_z = 1.0 / vpos.z;
  let u0 = fx * vpos.x * inv_z + u.intrinsics.z;
  let v0 = fy * vpos.y * inv_z + u.intrinsics.w;

  let R = quat_to_mat3(p2);
  let s = exp(clamp(p1.xyz, vec3f(-12.0), vec3f(4.0))) * u.screen.z;
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
  let clip_z = clamp((vpos.z - u.thresholds.y) / (u.thresholds.z - u.thresholds.y), 0.0, 0.9999);
  out.pos = vec4f(ndc_x, ndc_y, clip_z, 1.0);
  out.uv  = quad;

  let alpha = min(0.99, sigmoid(p0.w));
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
    rgb = heatmap_coolwarm(1.0 - clamp(vpos.z / max(1.0, u.screen.w * 2.5), 0.0, 1.0));
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

// The splats' depth for the overlays drawn after them (no color): the fragments of an opacity of 0.1 or more, each at
// its splat centre's depth (the wireframes behind them are drawn faint, wireframe_render). A surface is often many
// splats of 0.2-0.5, hence the low threshold
@fragment
fn fs_depth(in: VOut) {
  if (in.color.a * exp(-0.5 * dot(in.uv, in.uv)) < 0.1) { discard; }
}

// === env_render ===
// The background (env.js, DESIGN.md E13) in the free view: a full-screen triangle drawn under the splats and the mesh
// (front-to-back blending: it fills what they leave), each pixel's world direction from the view's intrinsics, as
// tile_forward's, at its size (n texels a face: ENV_N learned, more a dome). Bindings: 0 RenderUniforms (view, intrinsics,
// screen: its first 24 floats), 1 env.

// #include _env
// #include _math
struct EnvView { view : mat4x4f, intrinsics : vec4f, screen : vec4f };
@group(0) @binding(0) var<uniform>       ev  : EnvView;
@group(0) @binding(1) var<storage, read> env : array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  return vec4f(f32((i << 1u) & 2u) * 2.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);  // (-1, -1), (3, -1), (-1, 3)
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let n = u32(round(sqrt(f32(arrayLength(&env)) / 6.0)));
  let tap = env_taps_n(view_to_world(ev.view) * pixel_ray(pos.xy, ev.intrinsics), n);
  return vec4f(env[tap.i.x].rgb * tap.w.x + env[tap.i.y].rgb * tap.w.y + env[tap.i.z].rgb * tap.w.z + env[tap.i.w].rgb * tap.w.w, 1.0);
}

// === wireframe_render ===
// 3D line segments (octree boxes, camera frustums, WLBG split axes), drawn twice: in front of the depth
// (instance 0), behind it (instance 1: the octree boxes faint).
// Bindings: 0 WireUniforms, 1 verts (xyz, bitcast<f32>(abgr u32)).

// #include _wire
@group(0) @binding(1) var<storage, read> verts : array<vec4f>;

struct WireOut {
  @builtin(position) pos   : vec4f,
  @location(0)       color : vec4f,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) hidden: u32) -> WireOut {
  var out : WireOut;
  let v = verts[vid];
  let p = wire_project(v.xyz);
  out.pos   = vec4f(p.xyz, 1.0);
  out.color = unpack4x8unorm(bitcast<u32>(v.w));
  // octree boxes (the first vertices) fade with depth, so the far ones don't clog the view, and are faint where
  // behind the splats (instance 1: the draw whose depth test keeps the hidden fragments)
  if (f32(vid) < wu.fade.x) {
    let f = clamp((wu.fade.z - p.w) / (wu.fade.z - wu.fade.y), 0.0, 1.0);
    out.color.a = max(min(out.color.a, wu.fade.w), out.color.a * f) * select(1.0, wu.mesh.y, hidden == 1u);
  }
  return out;
}

@fragment
fn fs(in: WireOut) -> @location(0) vec4f {
  return vec4f(in.color.rgb * in.color.a, in.color.a);
}

// === mesh_render ===
// The mesh of DESIGN.md §9 E5 (src/mesh.js, .glb) in the free view: indexed triangles or their edges
// (vertex buffers: 0 position, 1 color, 2 uv). Its depth is linear in view z, written per fragment exactly as
// the splats write theirs (splats_render), so the depth test between the two is fair even on large
// simplified faces. fs_depth_back: depth only, wu.mesh.x behind the surface (the pre-pass that hides the
// splats further behind); fs_depth: its own depth only (laid after the splats, so only its front faces draw);
// fs_color: vertex colors; fs_tex: its texture; fs_shaded: gray, lit by the camera (shows the shape);
// fs_line: wireframe, pulled slightly towards the camera so it wins against its own faces.
// Bindings: 0 WireUniforms; fs_tex: 1 mesh_tex, 2 mesh_samp.

// #include _wire

struct MeshOut {
  @builtin(position) pos   : vec4f,
  @location(0)       vpos  : vec3f,
  @location(1)       color : vec4f,
  @location(2)       uv    : vec2f,
};

@group(0) @binding(1) var mesh_tex  : texture_2d<f32>;
@group(0) @binding(2) var mesh_samp : sampler;

struct MeshFrag {
  @location(0)          color : vec4f,
  @builtin(frag_depth)  depth : f32,
};

@vertex
fn vs(@location(0) p: vec3f, @location(1) c: vec4f, @location(2) uv: vec2f) -> MeshOut {
  let v = (wu.view * vec4f(p, 1.0)).xyz;
  // clip coordinates of the pinhole projection with w = view z: faces crossing the near plane are
  // clipped there (the depth written is fs's, not this z)
  let W = wu.screen.x; let H = wu.screen.y; let n = wu.screen.z; let f = wu.screen.w;
  let x = 2.0 * (wu.intrinsics.x * v.x + wu.intrinsics.z * v.z) / W - v.z;
  let y = v.z - 2.0 * (wu.intrinsics.y * v.y + wu.intrinsics.w * v.z) / H;
  var out : MeshOut;
  out.pos   = vec4f(x, y, f * (v.z - n) / (f - n), v.z);
  out.vpos  = v;
  out.color = c;
  out.uv    = uv;
  return out;
}

fn mesh_depth(z: f32) -> f32 { return clamp((z - wu.screen.z) / (wu.screen.w - wu.screen.z), 0.0, 0.9999); }

@fragment
fn fs_depth_back(in: MeshOut) -> @builtin(frag_depth) f32 { return mesh_depth(in.vpos.z + wu.mesh.x); }

@fragment
fn fs_depth(in: MeshOut) -> @builtin(frag_depth) f32 { return mesh_depth(in.vpos.z); }

@fragment
fn fs_color(in: MeshOut) -> MeshFrag { return MeshFrag(vec4f(in.color.rgb, 1.0), mesh_depth(in.vpos.z)); }

@fragment
fn fs_tex(in: MeshOut) -> MeshFrag { return MeshFrag(vec4f(textureSample(mesh_tex, mesh_samp, in.uv).rgb, 1.0), mesh_depth(in.vpos.z)); }

@fragment
fn fs_shaded(in: MeshOut) -> MeshFrag {
  let nrm = normalize(cross(dpdx(in.vpos), dpdy(in.vpos)));
  let g = 0.15 + 0.8 * abs(dot(nrm, normalize(in.vpos)));
  return MeshFrag(vec4f(g, g, g, 1.0), mesh_depth(in.vpos.z));
}

@fragment
fn fs_line(in: MeshOut) -> MeshFrag { return MeshFrag(vec4f(0.04, 0.35, 0.4, 0.4), mesh_depth(in.vpos.z) * 0.998); }

// === frustum_image ===
// Textured image planes inside the COLMAP camera frustums (one instance per camera).
// Bindings: 0 WireUniforms, 1 quad_verts (4 corners per camera, .w = alpha),
//           2 cam_tex, 3 cam_samp.

// #include _wire
@group(0) @binding(1) var<storage, read> quad_verts : array<vec4f>;
@group(0) @binding(2) var cam_tex  : texture_2d<f32>;
@group(0) @binding(3) var cam_samp : sampler;

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
// Fullscreen ground-truth / render / residual / depth-spread comparison.
// Bindings: 0 BlitParams, 1 gt_tex, 2 hat_tex, 3 samp, 4 pix_state (tile_forward's, of hat_tex).

struct BlitParams {
  mode    : u32, // 0 = target, 1 = split (target left), 2 = residual |gt - hat| * amp,
                 // 3 = depth spread (pix_state.w) as heat over the gray render, full at 1 / amp;
                 //     blue where T_final > 0.25 (no spread measured)
  split_x : f32, // in [0, 1]
  amp     : f32,
  _pad    : f32,
};

@group(0) @binding(0) var<uniform> bp      : BlitParams;
@group(0) @binding(1) var          gt_tex  : texture_2d<f32>;
@group(0) @binding(2) var          hat_tex : texture_2d<f32>;
@group(0) @binding(3) var          samp    : sampler;
@group(0) @binding(4) var<storage, read> pix_state : array<vec4f>;

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
  if (bp.mode == 3u) {
    let dim = textureDimensions(hat_tex);
    let p = min(vec2u(in.uv * vec2f(dim)), dim - 1u);
    let st = pix_state[p.y * dim.x + p.x];
    let gray = vec3f(0.35 * dot(hat, vec3f(0.299, 0.587, 0.114)));
    let s = clamp(st.w * bp.amp, 0.0, 1.0);
    let heat = clamp(vec3f(3.0 * s, 3.0 * s - 1.0, 3.0 * s - 2.0), vec3f(0.0), vec3f(1.0));  // black-red-yellow-white
    return vec4f(select(gray + heat, vec3f(0.0, 0.0, 0.25), st.x > 0.25), 1.0);
  }
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

// #include _train
@group(0) @binding(1) var<storage, read>       splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       sorted_idx  : array<u32>;
@group(0) @binding(3) var<storage, read_write> proj_splats : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> proj_tiles  : array<vec2u>;
@group(0) @binding(5) var<storage, read_write> tile_counts : array<atomic<u32>>;

// #include _math
// #include _proj

fn write_culled(k: u32, idx: u32) {
  proj_splats[k * 2u + 0u] = vec4f(-1e5, -1e5, -1.0, 0.0);
  proj_splats[k * 2u + 1u] = vec4f(0.0, 0.0, 0.0, bitcast<f32>(idx));
  proj_tiles[k] = vec2u(1u, 0u);  // empty rect
}

// Tiles t whose pixel centers [16 t + 0.5, 16 t + 15.5] meet [c - e, c + e], padded by 0.1 px
// for rounding: the first has 16 t + 15.5 >= c - e, the last 16 t + 0.5 <= c + e.
fn tile_span(c: f32, e: f32, n: u32) -> vec2u {
  let lo = i32(ceil((c - e - 15.6) / 16.0));
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

  // DropGaussian: a training render skips each splat with probability tu.loss.y
  if (tu.loss.y > 0.0) {
    let h = pcg_hash(idx ^ pcg_hash(u32(tu.opt_params.z * 16777216.0)));
    if (f32(h & 0x00ffffffu) / 16777216.0 < tu.loss.y) { write_culled(k, idx); return; }
  }

  let vpos = tu.view * vec4f(p0.xyz, 1.0);
  if (vpos.z <= 0.1) { write_culled(k, idx); return; }

  let fx = tu.intrinsics.x;
  let fy = tu.intrinsics.y;
  let inv_z = 1.0 / vpos.z;
  let tn = vpos.xy * inv_z;
  let k1 = tu.cam.z;
  let sd = 1.0 + k1 * dot(tn, tn);  // the lens's residual distortion (distort_jac)
  let u0 = fx * tn.x * sd + tu.intrinsics.z;
  let v0 = fy * tn.y * sd + tu.intrinsics.w;

  let c = splat_cov(vpos.xyz, p1, p2);
  let sxx = c.cov.x;
  let syy = c.cov.y;
  let sxy = c.cov.z;
  // conic = inverse covariance
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

// #include _bin
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

// #include _bin
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

// #include _bin
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
// Writes color to out_tex and pix_state = [T_final, z_med, bitcast<f32>(last_k), spread],
// z_med = depth where T crosses 0.5: where the pixel's ray crosses the plane of that splat when it is flat
// (its thinnest scale at most half the middle one; the thinnest axis as the normal, within 3 of its largest
// scales of its center), else its center depth, as 2DGS / RaDe-GS read depth: a large tilted splat's center depth is off by up to its size
// along the ray (DESIGN.md §9 E5), last_k = 1 + last contributing sorted slot,
// spread = (z_q3 - z_q1) / z_med, z_q1 and z_q3 = depths (the same way) where T crosses 0.75 and 0.25: how
// far apart along the ray the middle half of the blending weight is (a facade made of two
// sheets = high); 0 when T stays above 0.25.
// The color written is C + T_final env(the pixel's world direction): the background (env.js; black until learned),
// through the view's exposure (tu.expo_g, expo_b; exposure.js).
// Bindings: 0 TrainUniforms, 1 splats, 2 proj_splats, 3 out_tex, 4 pix_state,
//           5 tile_ranges ([start, count] per tile, row-major), 6 tile_list, 7 env_tex (env.js, the trainer's copy).

// #include _train
// #include _env
@group(0) @binding(1) var<storage, read>       splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       proj_splats : array<vec4f>;
@group(0) @binding(3) var                      out_tex     : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> pix_state   : array<vec4f>;
@group(0) @binding(5) var<storage, read>       tile_ranges : array<vec2u>;
@group(0) @binding(6) var<storage, read>       tile_list   : array<u32>;
@group(0) @binding(7) var                      env_tex     : texture_2d<f32>;

const BATCH : u32 = 64u;

var<workgroup> wg_u0v0zr : array<vec4f, BATCH>;
var<workgroup> wg_conic  : array<vec4f, BATCH>;
var<workgroup> wg_rgba   : array<vec4f, BATCH>;
var<workgroup> wg_k      : array<u32, BATCH>;
var<workgroup> wg_plane  : array<vec4f, BATCH>;  // the splat's plane, view space: normal, normal . center
var<workgroup> wg_reach  : array<f32, BATCH>;    // 3 x its largest scale
var<workgroup> wg_hits   : atomic<u32>;

// #include _math
// #include _raster

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(workgroup_id) wgid: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let W = tu.dims.x;
  let H = tu.dims.y;
  let in_bounds = gid.x < W && gid.y < H;
  let px = vec2f(gid.xy) + 0.5;
  let tile_c = vec2f(wgid.xy * 16u) + 8.0;
  let ray = pixel_ray(px, tu.intrinsics);  // view space, z = 1

  var C = vec3f(0.0);
  var T = 1.0;
  var z_med = 0.0;
  var last_k = 0u;
  var done = !in_bounds;
  var z_q1 = 0.0;
  var z_q3 = 0.0;

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
          let p1  = splats[idx * 4u + 1u];
          let p3  = splats[idx * 4u + 3u];
          let ax  = quat_to_mat3(splats[idx * 4u + 2u]);
          let thin = select(select(2u, 1u, p1.y <= p1.z), 0u, p1.x <= p1.y && p1.x <= p1.z);
          let n = normalize((tu.view * vec4f(ax[thin], 0.0)).xyz);
          wg_plane[lid] = vec4f(n, dot(n, (tu.view * vec4f(p0.xyz, 1.0)).xyz));
          // flat (the thinnest scale at most half the middle one): a plane; else none (reach 0: the center depth)
          let mid = p1.x + p1.y + p1.z - min(p1.x, min(p1.y, p1.z)) - max(p1.x, max(p1.y, p1.z));
          wg_reach[lid] = select(0.0, 3.0 * exp(max(p1.x, max(p1.y, p1.z))), p1[thin] + 0.693 <= mid);
          wg_conic[lid] = pr1;
          wg_rgba[lid]  = vec4f(clamp(vec3f(0.5) + SH_C0 * p3.xyz, vec3f(0.0), vec3f(1.0)),
                                min(1.0, sigmoid(p0.w) * train_opacity_factor()));
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
        if (T >= 0.25 && next_T < 0.75) {  // a quartile's depth: on the splat's plane (flat ones), else its center
          let pl = wg_plane[i];
          let nd = dot(pl.xyz, ray);
          let zp = select(u0v0zr.z, pl.w / nd, abs(nd) > 1e-3);
          let z = select(u0v0zr.z, zp, abs(zp - u0v0zr.z) < wg_reach[i]);
          if (T >= 0.75 && next_T < 0.75) { z_q1 = z; }
          if (T >= 0.5 && next_T < 0.5) { z_med = z; }
          if (next_T < 0.25) { z_q3 = z; }
        }
        C += (T * alpha) * rgba.rgb;
        T = next_T;
        last_k = wg_k[i] + 1u;
        if (T < 1e-4) { done = true; break; }
      }
    }
    workgroupBarrier();
  }

  if (in_bounds) {
    let tap = env_taps_tex(env_tex, view_to_world(tu.view) * ray);
    let bg = env_bilinear(env_tex, tap);
    textureStore(out_tex, vec2i(gid.xy), vec4f(tu.expo_g.rgb * (C + T * bg) + tu.expo_b.rgb, 1.0));
    let spread = select(0.0, max(0.0, z_q3 - z_q1) / max(1e-6, z_med), T < 0.25);
    pix_state[gid.y * W + gid.x] = vec4f(T, z_med, bitcast<f32>(last_k), spread);
  }
}

// === ssim_coef ===
// D-SSIM loss (tu.loss.x > 0), first half. Per pixel q and channel, with x = gt, y = hat and
// the window moments mu_x, mu_y, E[xx], E[yy], E[xy] (see _ssim):
//   S = A1 A2 / (B1 B2),  A1 = 2 mu_x mu_y + C1,  A2 = 2 cov_xy + C2,
//                         B1 = mu_x^2 + mu_y^2 + C1,  B2 = var_x + var_y + C2;
// writes the partials of S(q) w.r.t. mu_y, E[yy], E[xy] (the variances taken from them):
//   c = 2 A1 / (B1 B2),  b = -S / B2,  a = 2 mu_x A2 / (B1 B2) - 2 mu_y S / B1 - mu_x c - 2 mu_y b;
// ssim_grad sums them back: dS(q)/dy(p) = g(p - q) (a(q) + 2 y(p) b(q) + x(p) c(q)).
// Bindings: 0 TrainUniforms, 1 gt_tex, 2 hat_tex, 3 coef (3 vec4f per pixel: a, b, c).

// #include _train
// #include _ssim
@group(0) @binding(1) var                      gt_tex  : texture_2d<f32>;
@group(0) @binding(2) var                      hat_tex : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> coef    : array<vec4f>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let W = i32(tu.dims.x);
  let H = i32(tu.dims.y);
  let q = vec2i(gid.xy);
  if (q.x >= W || q.y >= H) { return; }
  let M = ssim_moments(q, W, H);
  let mx = M.mx; let my = M.my; let exx = M.exx; let eyy = M.eyy; let exy = M.exy;
  let A1 = 2.0 * mx * my + SSIM_C1;
  let A2 = 2.0 * (exy - mx * my) + SSIM_C2;
  let B1 = mx * mx + my * my + SSIM_C1;
  let B2 = (exx - mx * mx) + (eyy - my * my) + SSIM_C2;
  let D = 1.0 / (B1 * B2);
  let S = A1 * A2 * D;
  let c = 2.0 * A1 * D;
  let b = -S / B2;
  let a = 2.0 * mx * A2 * D - 2.0 * my * S / B1 - mx * c - 2.0 * my * b;
  let o = (u32(q.y) * u32(W) + u32(q.x)) * 3u;
  coef[o] = vec4f(a, 0.0);
  coef[o + 1u] = vec4f(b, 0.0);
  coef[o + 2u] = vec4f(c, 0.0);
}

// === ssim_map ===
// The held-out SSIM (evaluateValidation; rgbSsim on the GPU): per pixel q, S (see ssim_coef) summed over
// the RGB channels, x = gt, y = hat; summed again per 16 x 16 workgroup, one f32 per workgroup (row major).
// Bindings: 0 TrainUniforms, 1 gt_tex, 2 hat_tex, 3 ssim_sums.

// #include _train
// #include _ssim
@group(0) @binding(1) var                      gt_tex    : texture_2d<f32>;
@group(0) @binding(2) var                      hat_tex   : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> ssim_sums : array<f32>;

var<workgroup> part : array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let W = i32(tu.dims.x);
  let H = i32(tu.dims.y);
  let q = vec2i(gid.xy);
  var v = 0.0;
  if (q.x < W && q.y < H) {
    let M = ssim_moments(q, W, H);
    let S = ((2.0 * M.mx * M.my + SSIM_C1) * (2.0 * (M.exy - M.mx * M.my) + SSIM_C2)) /
            ((M.mx * M.mx + M.my * M.my + SSIM_C1) * ((M.exx - M.mx * M.mx) + (M.eyy - M.my * M.my) + SSIM_C2));
    v = S.r + S.g + S.b;
  }
  part[li] = v;
  workgroupBarrier();
  for (var n = 128u; n > 0u; n >>= 1u) {
    if (li < n) { part[li] += part[li + n]; }
    workgroupBarrier();
  }
  if (li == 0u) { ssim_sums[wid.y * nwg.x + wid.x] = part[0]; }
}

// === ssim_grad ===
// D-SSIM loss, second half: per pixel p and channel, sum_q dS(q)/dy(p) over the window around p
// (see ssim_coef), into ssim_grad_tex for tile_backward.
// Bindings: 0 TrainUniforms, 1 gt_tex, 2 hat_tex, 3 coef, 4 ssim_grad_tex.

// #include _train
// #include _ssim
@group(0) @binding(1) var                      gt_tex        : texture_2d<f32>;
@group(0) @binding(2) var                      hat_tex       : texture_2d<f32>;
@group(0) @binding(3) var<storage, read>       coef          : array<vec4f>;
@group(0) @binding(4) var                      ssim_grad_tex : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let W = i32(tu.dims.x);
  let H = i32(tu.dims.y);
  let p = vec2i(gid.xy);
  if (p.x >= W || p.y >= H) { return; }
  var G = SSIM_G;
  let x = textureLoad(gt_tex,  p, 0).rgb;
  let y = textureLoad(hat_tex, p, 0).rgb;
  var sum = vec3f(0.0);
  for (var dy = -SSIM_R; dy <= SSIM_R; dy++) {
    let qy = p.y + dy;
    if (qy < 0 || qy >= H) { continue; }
    for (var dx = -SSIM_R; dx <= SSIM_R; dx++) {
      let qx = p.x + dx;
      if (qx < 0 || qx >= W) { continue; }
      let o = (u32(qy) * u32(W) + u32(qx)) * 3u;
      sum += G[dy + SSIM_R] * G[dx + SSIM_R] * (coef[o].rgb + 2.0 * y * coef[o + 1u].rgb + x * coef[o + 2u].rgb);
    }
  }
  textureStore(ssim_grad_tex, p, vec4f(sum, 0.0));
}

// === tile_backward ===
// Backward pass of tile_forward over the same tile lists, batches walked back-to-front
// up to each pixel's last_k. T_i is recovered as T_{i+1} / (1 - alpha_i).
// Per splat, accumulates the 2D grads (see header) into grads_2d and the WLBG
// error moments into moments (S is done in adam_step): nv values per hit, 10 grads (slot 9,
// the weight, is also moments' W), R, then the 9 directional moments (moments 3..11) with DIR_MOMS:
// only split_mode 'error' reads them (densify.js splitSplat), the trainer's other pipeline leaves
// them out (their code alone, even skipped, slows a step noticeably). Contributions are summed over the subgroup first (when the device
// has 'subgroups'; lines tagged @sg / @nosg are selected by loadShaders()), then in
// workgroup memory: one global CAS per slot per batch per tile.
// Loss: 0.8 * smooth-L1 + 0.2 * L2 per channel, normalized by pixel count; with a D-SSIM weight
// l = tu.loss.x > 0, (1 - l) * smooth-L1 + l * (1 - SSIM) / 2 instead (as 3DGS), the SSIM
// gradient sum_q dS(q)/dy(p) read from ssim_grad_tex (ssim_coef, ssim_grad);
// loss_acc[0..1] = 64-bit sum (lo, hi) of the per-pixel MSE in 16.16 fixed point.
// The background (env_tex, tile_forward): the color behind the last splat is env(dir), not black, and the background's
// gradient dL/dC T_final is summed over the tile and added to the 4 texels at the tile's center (16 pixels span at
// most ~1.5 texels at 960 x 540), with the sum of T_final (w: how much the pixels show of it), into loss_acc after its first 4 words (a vec4f of f32 bits a texel, read by
// env_adam: a binding of its own would be the 9th storage buffer, over WebGPU's default of 8 a stage).
// The view's exposure (exposure.js): hat_tex is g c + b; dL/dc' is taken back to the render c (times g) for the splats
// and the background, and while it learns (tu.expo_g.w = 1) its gradient, dL/dlog g = dL/dc' (c' - b) and dL/db = dL/dc'
// summed over the pixels, goes after the background's in loss_acc (LOSS_EXPO_AT, 8 words: log g rgb, pad, b rgb, pad).
// Bindings: 0 TrainUniforms, 1 splats, 2 proj_splats, 3 gt_tex, 4 hat_tex,
//           5 pix_state, 6 grads_2d, 7 moments, 8 loss_acc (+ the background's gradient), 9 tile_ranges, 10 tile_list,
//           11 ssim_grad_tex, 12 env_tex.

// #include _train
// #include _env
// #include _lossacc
@group(0) @binding(1) var<storage, read>       splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       proj_splats : array<vec4f>;
@group(0) @binding(3) var                      gt_tex      : texture_2d<f32>;
@group(0) @binding(4) var                      hat_tex     : texture_2d<f32>;
@group(0) @binding(5) var<storage, read>       pix_state   : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> grads_2d    : array<atomic<u32>>; // 12 f32 per splat
@group(0) @binding(7) var<storage, read_write> moments     : array<atomic<u32>>; // 12 f32 per splat
@group(0) @binding(8) var<storage, read_write> loss_acc    : array<atomic<u32>>;
@group(0) @binding(9) var<storage, read>       tile_ranges : array<vec2u>;
@group(0) @binding(10) var<storage, read>      tile_list   : array<u32>;
@group(0) @binding(11) var                     ssim_grad_tex : texture_2d<f32>;
@group(0) @binding(12) var                     env_tex     : texture_2d<f32>;

const BATCH : u32 = 32u;
override DIR_MOMS : bool = false;  // sum the directional moments (pipeline constant, see the header)
const NV    : u32 = 20u;  // values per hit: 10 grads, R, 9 directional moments
const SLOTS : u32 = BATCH * NV;

var<workgroup> wg_u0v0zr     : array<vec4f, BATCH>;
var<workgroup> wg_conic      : array<vec4f, BATCH>;
var<workgroup> wg_rgba       : array<vec4f, BATCH>;
var<workgroup> wg_cmask      : array<vec4f, BATCH>; // 1 where rgb is not clamped
var<workgroup> wg_k          : array<u32, BATCH>;   // sorted slot of each batch entry
var<workgroup> wg_acc        : array<atomic<u32>, SLOTS>;
var<workgroup> wg_mse        : atomic<u32>;
var<workgroup> wg_hits       : atomic<u32>;
var<workgroup> wg_max_last_k : atomic<u32>;
var<workgroup> wg_env        : array<atomic<u32>, 4>;  // the tile's background gradient (rgb) and its sum of T_final
var<workgroup> wg_expo       : array<atomic<u32>, 6>;  // the tile's exposure gradient: log g rgb, b rgb
// Only pixels with T_final over this teach the background: below it, it learned the splats' residual (a pixel the splats
// render too red pushed it to green), colors no photo has where the views barely saw it
const ENV_MIN_T : f32 = 0.5;

// #include _math
// #include _raster

// f32 atomic adds via CAS (WGSL has no pointer params to workgroup/storage
// atomics, hence one copy per array). Tiny values are skipped.
fn add_wg(i: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&wg_acc[i]);
  loop {
    let r = atomicCompareExchangeWeak(&wg_acc[i], old, bitcast<u32>(bitcast<f32>(old) + val));
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

fn add_wg_env(i: u32, val: f32) {
  if (abs(val) < 1e-12) { return; }
  var old = atomicLoad(&wg_env[i]);
  loop {
    let r = atomicCompareExchangeWeak(&wg_env[i], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn add_wg_expo(i: u32, val: f32) {
  if (abs(val) < 1e-12) { return; }
  var old = atomicLoad(&wg_expo[i]);
  loop {
    let r = atomicCompareExchangeWeak(&wg_expo[i], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn add_loss_acc(i: u32, val: f32) {
  if (abs(val) < 1e-12) { return; }
  var old = atomicLoad(&loss_acc[i]);
  loop {
    let r = atomicCompareExchangeWeak(&loss_acc[i], old, bitcast<u32>(bitcast<f32>(old) + val));
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
        @builtin(subgroup_size) sg_size: u32,           // @sg
        @builtin(local_invocation_index) lid: u32) {
  let W = tu.dims.x;
  let H = tu.dims.y;
  let in_bounds = gid.x < W && gid.y < H;
  let px = vec2f(gid.xy) + 0.5;
  let tile_c = vec2f(wgid.xy * 16u) + 8.0;
  let learned_env = textureDimensions(env_tex).x == ENV_N;  // a dome (trainer.setDome) is fixed: no gradient

  if (lid == 0u) {
    atomicStore(&wg_mse, 0u);
    atomicStore(&wg_hits, 0u);
    atomicStore(&wg_max_last_k, 0u);
    for (var c = 0u; c < 4u; c++) { atomicStore(&wg_env[c], 0u); }
    for (var c = 0u; c < 6u; c++) { atomicStore(&wg_expo[c], 0u); }
  }
  for (var s = lid; s < SLOTS; s += 256u) { atomicStore(&wg_acc[s], 0u); }
  workgroupBarrier();

  var dL_dC = vec3f(0.0);
  var err_l1 : f32 = 0.0;
  var T : f32 = 1.0;
  var last_k = 0u;
  var accum_rec = vec3f(0.0); // color behind the current splat, unattenuated by T
  var expo = array<f32, 6>();  // this pixel's exposure gradient: log g rgb, b rgb

  if (in_bounds) {
    let gt4  = textureLoad(gt_tex,  vec2i(gid.xy), 0);
    let gt   = gt4.rgb;
    let hat  = textureLoad(hat_tex, vec2i(gid.xy), 0).rgb;
    let diff = (hat - gt) * gt4.a;  // alpha 0: a pixel left out of the loss (a video's on-screen display, overlay.js)
    atomicAdd(&wg_mse, u32(dot(diff, diff) / 3.0 * 65536.0 + 0.5));
    let sign_d = clamp(diff * 25.0, vec3f(-1.0), vec3f(1.0));
    let lam_ssim = tu.loss.x;
    if (lam_ssim > 0.0) {
      let gs = textureLoad(ssim_grad_tex, vec2i(gid.xy), 0).rgb;
      dL_dC = ((1.0 - lam_ssim) * sign_d - 0.5 * lam_ssim * gs * gt4.a) * (1.0 / f32(W * H));
    } else {
      dL_dC = (0.8 * sign_d + 0.4 * diff) * (1.0 / f32(W * H));
    }
    err_l1 = (abs(diff.x) + abs(diff.y) + abs(diff.z)) / 3.0;
    let dg = dL_dC * (hat - tu.expo_b.rgb);
    expo = array<f32, 6>(dg.x, dg.y, dg.z, dL_dC.x, dL_dC.y, dL_dC.z);
    dL_dC *= tu.expo_g.rgb;  // dL/dc, the render before the exposure

    let st = pix_state[gid.y * W + gid.x];
    T = st.x;
    last_k = bitcast<u32>(st.z);
    atomicMax(&wg_max_last_k, last_k);
    let tap = env_taps_tex(env_tex, view_to_world(tu.view) * pixel_ray(px, tu.intrinsics));
    accum_rec = env_bilinear(env_tex, tap);
    if (T > ENV_MIN_T && learned_env) {  // the pixels it dominates teach it, not the splats' residual where they nearly cover
      let g = dL_dC * T;
      for (var c = 0u; c < 3u; c++) { add_wg_env(c, g[c]); }
      add_wg_env(3u, T);  // how much of the background this pixel shows
    }
  }
  if (tu.expo_g.w > 0.0) {  // uniform: the subgroup sums first
    for (var c = 0u; c < 6u; c++) { expo[c] = subgroupAdd(expo[c]); }          // @sg
    for (var c = sg_lane; c < 6u; c += sg_size) { add_wg_expo(c, expo[c]); }   // @sg
    for (var c = 0u; c < 6u; c++) { add_wg_expo(c, expo[c]); }                 // @nosg
  }
  workgroupBarrier();
  if (lid == 0u && learned_env) {  // the tile's background gradient onto the texels at its center
    let tap = env_taps(view_to_world(tu.view) * pixel_ray(tile_c, tu.intrinsics));
    for (var c = 0u; c < 4u; c++) {
      let g = bitcast<f32>(atomicLoad(&wg_env[c]));
      add_loss_acc(LOSS_ENV_AT + tap.i.x * 4u + c, g * tap.w.x);
      add_loss_acc(LOSS_ENV_AT + tap.i.y * 4u + c, g * tap.w.y);
      add_loss_acc(LOSS_ENV_AT + tap.i.z * 4u + c, g * tap.w.z);
      add_loss_acc(LOSS_ENV_AT + tap.i.w * 4u + c, g * tap.w.w);
    }
  }
  if (lid == 0u && tu.expo_g.w > 0.0) {
    for (var c = 0u; c < 6u; c++) { add_loss_acc(LOSS_EXPO_AT + c + c / 3u, bitcast<f32>(atomicLoad(&wg_expo[c]))); }
  }
  if (lid == 0u) {  // 64-bit add: carry into loss_acc[1] when the low word wraps
    let v = atomicLoad(&wg_mse);
    let old = atomicAdd(&loss_acc[0], v);
    if (old + v < old) { atomicAdd(&loss_acc[1], 1u); }
  }

  let tile_max_k = atomicLoad(&wg_max_last_k);
  let dir_moms = DIR_MOMS;
  let nv = select(11u, NV, dir_moms);
  let fx = tu.intrinsics.x;
  let fy = tu.intrinsics.y;
  let cam_to_world = view_to_world(tu.view);

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
          wg_rgba[lid]  = vec4f(rgb, min(1.0, sigmoid(p0.w) * train_opacity_factor()));
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

        var v = array<f32, 11>();  // grads_2d 0..9, then moments 2 (R)
        var dm = array<f32, 9>();  // moments 3..11 (dir_moms)
        if (hit) {
          T = T / max(1e-4, 1.0 - alpha); // T_i
          let weight = T * alpha;
          let w_err  = weight * err_l1;

          let dL_drgb   = dL_dC * weight * SH_C0 * wg_cmask[i].xyz; // w.r.t. f_dc, zero where clamped
          let dL_dalpha = T * dot(dL_dC, rgba.rgb - accum_rec);
          accum_rec = alpha * rgba.rgb + (1.0 - alpha) * accum_rec;

          let dL_dpower = -dL_dalpha * alpha;
          v[0] = dL_drgb.x;
          v[1] = dL_drgb.y;
          v[2] = dL_drgb.z;
          v[3] = dL_dalpha * alpha * (1.0 - rgba.w / train_opacity_factor());  // dL/dlogit_opacity
          v[4] = -dL_dpower * (conic.x * dx + conic.y * dy);        // dL/du0
          v[5] = -dL_dpower * (conic.y * dx + conic.z * dy);        // dL/dv0
          v[6] = 0.5 * dL_dpower * dx * dx;                         // dL/dconic.a
          v[7] =       dL_dpower * dx * dy;                         // dL/dconic.b (full, b appears twice)
          v[8] = 0.5 * dL_dpower * dy * dy;                         // dL/dconic.c
          v[9] = weight;                                            // w_view (also moments' W)
          v[10] = w_err;                                            // moments' R

          // WLBG moments; delta = world-space pixel offset from the center at its depth.
          if (dir_moms) {
            let d = cam_to_world * vec3f(dx * u0v0zr.z / fx, dy * u0v0zr.z / fy, 0.0);
            dm[0] = w_err * d.x;
            dm[1] = w_err * d.y;
            dm[2] = w_err * d.z;
            dm[3] = w_err * d.x * d.x;
            dm[4] = w_err * d.y * d.y;
            dm[5] = w_err * d.z * d.z;
            dm[6] = w_err * d.x * d.y;
            dm[7] = w_err * d.x * d.z;
            dm[8] = w_err * d.y * d.z;
          }
        }
        let gbase = i * nv;
        // constant loop bounds (a bound of nv makes the pass much slower)
        if (subgroupAny(hit)) {                                   // @sg
          for (var c = 0u; c < 11u; c++) { v[c] = subgroupAdd(v[c]); }  // @sg
          // one lane per slot, not a CAS loop per slot on lane 0      // @sg
          for (var c = sg_lane; c < 11u; c += sg_size) { add_wg(gbase + c, v[c]); }  // @sg
          if (dir_moms) {                                           // @sg
            for (var c = 0u; c < 9u; c++) { dm[c] = subgroupAdd(dm[c]); }  // @sg
            for (var c = sg_lane; c < 9u; c += sg_size) { add_wg(gbase + 11u + c, dm[c]); }  // @sg
          }                                                          // @sg
        }                                                            // @sg
        if (hit) {                                                // @nosg
          for (var c = 0u; c < 11u; c++) { add_wg(gbase + c, v[c]); }  // @nosg
          if (dir_moms) { for (var c = 0u; c < 9u; c++) { add_wg(gbase + 11u + c, dm[c]); } }  // @nosg
        }                                                            // @nosg
      }
    }
    workgroupBarrier();

    // Flush workgroup sums to global memory and zero them for the next batch.
    if (has_hits) {
      for (var s = lid; s < BATCH * nv; s += 256u) {  // slot c of entry j at j * nv + c
        let c = s % nv;
        let j = s / nv;
        let val = bitcast<f32>(atomicExchange(&wg_acc[s], 0u));
        if (wg_u0v0zr[j].w > 0.0) {
          let gi = bitcast<u32>(wg_conic[j].w) * 12u;
          if (c < 10u) { add_global_grad(gi + c, val); }
          if (c == 9u) { add_global_mom(gi, val); }                  // W
          else if (c == 10u) { add_global_mom(gi + 2u, val); }       // R
          else if (c > 10u) { add_global_mom(gi + c - 8u, val); }
        }
      }
      if (lid == 0u) { atomicStore(&wg_hits, 0u); }
    }
    workgroupBarrier();
  }
}

// === env_adam ===
// The background's update (env.js, tile_backward's gradient in loss_acc after its first vec4): one thread a texel, Adam
// on rgb (b1 0.9, b2 0.999, the grads clipped to [-1, 1] as adam_step's) at lr tu.reg.z x wbar / (wbar + ENV_W0), kept in
// [0, 1]; wbar (env_m.w): the running mean of the texel's T_final sum a step. Adam's step does not shrink with the gradient,
// so a texel seen only through nearly opaque splats would walk to saturated colors no training view showed; scaled by
// how much it is seen, it stays black. The gradient is read
// and cleared; the trainer then copies env to the texture the tile passes read.
// Bindings: 0 TrainUniforms, 1 env, 2 loss_acc, 3 env_m, 4 env_v.

// #include _train
// #include _lossacc
@group(0) @binding(1) var<storage, read_write> env      : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> loss_acc : array<vec4u>;
@group(0) @binding(3) var<storage, read_write> env_m    : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> env_v    : array<vec4f>;
const ENV_W0 : f32 = 16.0;  // a texel's T_final sum a step at which it learns at half the rate

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= arrayLength(&env)) { return; }
  let raw = bitcast<vec4f>(loss_acc[LOSS_ENV_AT / 4u + t]);
  loss_acc[LOSS_ENV_AT / 4u + t] = vec4u(0u);
  let g = clamp(raw.rgb, vec3f(-1.0), vec3f(1.0));
  let step = max(1.0, tu.opt_params.w);
  let m = 0.9 * env_m[t].rgb + 0.1 * g;
  let v = 0.999 * env_v[t].rgb + 0.001 * g * g;
  let wbar = 0.99 * env_m[t].w + 0.01 * raw.w;
  env_m[t] = vec4f(m, wbar);
  env_v[t] = vec4f(v, 0.0);
  let upd = (m / (1.0 - pow(0.9, step))) / (sqrt(v / (1.0 - pow(0.999, step))) + 1e-15);
  let lr = tu.reg.z * wbar / (wbar + ENV_W0);
  env[t] = vec4f(clamp(env[t].rgb - lr * upd, vec3f(0.0), vec3f(1.0)), 0.0);
}

// === cam_reduce ===
// The view's pose and lens gradient (adam_step's cam_part, 12 floats a workgroup of 64 visible splats) summed into
// loss_acc after the exposure's (LOSS_CAM_AT, 10 words: rho, phi, log f, cx, cy, k1). One workgroup.
// Bindings: 0 TrainUniforms, 1 cam_part, 2 loss_acc.

// #include _train
// #include _lossacc
@group(0) @binding(1) var<storage, read>       cam_part : array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_acc : array<f32>;
var<workgroup> wg_sum : array<array<f32, 10>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  var acc = array<f32, 10>();
  for (var w = lid; w < (tu.dims.z + 63u) / 64u; w += 256u) {
    for (var c = 0u; c < 10u; c++) { acc[c] += cam_part[w * 12u + c]; }
  }
  wg_sum[lid] = acc;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (lid < h) { for (var c = 0u; c < 10u; c++) { wg_sum[lid][c] += wg_sum[lid + h][c]; } }
    workgroupBarrier();
  }
  if (lid == 0u) { for (var c = 0u; c < 10u; c++) { loss_acc[LOSS_CAM_AT + c] = wg_sum[0][c]; } }
}

// === adam_step ===
// Per-splat update, one thread per visible splat (sorted_idx, the training sort: only those have grads):
//   1. reads and clears grads_2d, adds w_view^2 to moments.S;
//   2. chains the 2D grads to 3D (position incl. through the EWA Jacobian,
//      log-scale, quaternion), Adam step (sparse: splats with no grad are
//      skipped), clamps and shape limits;
//   3. opt_method 1 (MC/SGLD, 3DGS-MCMC): L1 terms on opacity and scale, noise on the centers of
//      nearly transparent splats.
// While the camera learns (tu.cam.x = 1): each splat's share of the view's pose and lens gradient (poses.js), from the
// same chain: a camera-frame increment x_c -> x_c + rho + phi x x_c turns the splat's covariance with it too, so
// dL/drho = dL/dx_c (through u0, v0 and J), dL/dphi = x_c x dL/dx_c + the covariance's term (with A = dL/dW W^T =
// 2 W dL/dSigma Sigma W^T: A_21 - A_12, A_02 - A_20, A_10 - A_01), dL/dlog f = dL/du0 (u0 - cx) + dL/dv0 (v0 - cy)
// + sum dL/dJ * J, dL/dcx = dL/du0, dL/dcy = dL/dv0 (the frustum clamp's dependence on them left out), dL/dk1 = (dL/du0
// (u0 - cx) + dL/dv0 (v0 - cy)) r^2 / (1 + k1 r^2) + sum dL/dD * dD/dk1 (distort_jac: the centre's gradient goes back
// to the pinhole's through D^T, the covariance's through D and D's own dependence on x_c); summed over the
// workgroup into cam_part (12 floats a workgroup), then over the workgroups by cam_reduce.
// Bindings: 0 TrainUniforms, 1 splats, 2 grads_2d, 3 adam_m, 4 adam_v, 5 moments, 6 sorted_idx, 7 cam_part.

// #include _train
@group(0) @binding(7) var<storage, read_write> cam_part : array<f32>;
var<private> cam_g : array<f32, 10>;          // this thread's splat's share: rho, phi, log f, cx, cy, k1
var<workgroup> wg_cam : array<array<f32, 10>, 64>;
@group(0) @binding(1) var<storage, read_write> splats   : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> grads_2d : array<u32>;
@group(0) @binding(3) var<storage, read_write> adam_m   : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> adam_v   : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> moments  : array<u32>;
@group(0) @binding(6) var<storage, read>       sorted_idx : array<u32>;

// #include _math
// #include _proj

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
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3u) {
  step_splat(gid.x);
  if (tu.cam.x > 0.0) {  // uniform: the workgroup's sum
    wg_cam[lid] = cam_g;
    workgroupBarrier();
    for (var h = 32u; h > 0u; h >>= 1u) {
      if (lid < h) { for (var c = 0u; c < 10u; c++) { wg_cam[lid][c] += wg_cam[lid + h][c]; } }
      workgroupBarrier();
    }
    if (lid == 0u) { for (var c = 0u; c < 10u; c++) { cam_part[wgid.x * 12u + c] = wg_cam[0][c]; } }
  }
}

fn step_splat(k: u32) {
  if (k >= tu.dims.z) { return; }
  let idx = sorted_idx[k];

  let gbase = idx * 12u;
  let dL_drgb = vec3f(load_grad(gbase + 0u), load_grad(gbase + 1u), load_grad(gbase + 2u));
  let dL_dop  = load_grad(gbase + 3u);
  let dL_du0  = load_grad(gbase + 4u);
  let dL_dv0  = load_grad(gbase + 5u);
  let dL_da   = load_grad(gbase + 6u);
  let dL_db   = load_grad(gbase + 7u);
  let dL_dc   = load_grad(gbase + 8u);
  let w_view  = load_grad(gbase + 9u);
  for (var c = 0u; c < 12u; c++) { grads_2d[gbase + c] = 0u; }

  // S_i += w_{i,c}^2 (view concentration, N_eff = W^2 / S).
  if (w_view > 1e-6 && tu.cam.y == 0.0) {
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

  // (u0, v0) -> the pinhole's (through the lens's residual distortion, distort_jac) -> view-space position.
  let tn = vpos.xy * inv_z;
  let k1 = tu.cam.z;
  let r2 = dot(tn, tn);
  let dL_dpin = transpose(distort_jac(tn, k1, fx, fy)) * vec2f(dL_du0, dL_dv0);
  let dL_dvpos = vec3f(
    dL_dpin.x * fx * inv_z,
    dL_dpin.y * fy * inv_z,
    -(dL_dpin.x * fx * vpos.x + dL_dpin.y * fy * vpos.y) * inv_z2
  );

  // the forward covariance, project_splats'
  let qn = max(1e-8, length(p2));
  let qw = p2.x / qn; let qx = p2.y / qn; let qy = p2.z / qn; let qz = p2.w / qn;
  let c = splat_cov(vpos.xyz, p1, p2);
  let R = c.R; let s = c.scale; let M = c.M; let Sigma = c.Sigma; let Tp = c.Tp; let D = c.D; let T = c.T; let tcn = c.tcn; let lim = c.lim;
  let sxx = c.cov.x;
  let syy = c.cov.y;
  let sxy = c.cov.z;
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

  // cov2d also depends on the view-space position through J, and through D (distort_jac).
  let dL_dT = 2.0 * dL_dcov2d * T * Sigma;
  let dL_dJ = transpose(D) * dL_dT * transpose(W3);  // the pinhole's J
  let dL_dD = distort_jac_grad(tcn, k1, fx, fy, dL_dT * transpose(Tp));  // dL/dx, dL/dy (clamped), dL/dk1
  // A clamped x/z (y/z) makes J's x (y) column independent of x (y), and halves its z term.
  let tq = tn;
  let free = vec2f(select(0.0, 1.0, abs(tq.x) <= lim.x), select(0.0, 1.0, abs(tq.y) <= lim.y));
  let tc = tcn * vpos.z;  // clamped x, y
  let dDf = dL_dD.xy * free;
  let dL_dvpos_cov = vec3f(
    -fx * inv_z2 * dL_dJ[2][0] * free.x + dDf.x * inv_z,
    -fy * inv_z2 * dL_dJ[2][1] * free.y + dDf.y * inv_z,
    -fx * inv_z2 * dL_dJ[0][0] - fy * inv_z2 * dL_dJ[1][1] +
      inv_z3 * ((1.0 + free.x) * fx * tc.x * dL_dJ[2][0] + (1.0 + free.y) * fy * tc.y * dL_dJ[2][1]) -
      (dDf.x * tq.x + dDf.y * tq.y) * inv_z
  );
  let dL_dpos = transpose(W3) * (dL_dvpos + dL_dvpos_cov);
  if (tu.cam.x > 0.0) {  // the camera's share (the header)
    let g = dL_dvpos + dL_dvpos_cov;
    let A = 2.0 * W3 * dL_dSigma * Sigma * transpose(W3);
    let rot = cross(vpos.xyz, g) + vec3f(A[1][2] - A[2][1], A[2][0] - A[0][2], A[0][1] - A[1][0]);
    let J = Tp * transpose(W3);
    let dc = dL_du0 * fx * tn.x + dL_dv0 * fy * tn.y;  // dL/du0 (u0 - cx) + dL/dv0 (v0 - cy), over 1 + k1 r^2
    let dlogf = dc * (1.0 + k1 * r2) + dot(dL_dJ[0], J[0]) + dot(dL_dJ[1], J[1]) + dot(dL_dJ[2], J[2]);
    cam_g = array<f32, 10>(g.x, g.y, g.z, rot.x, rot.y, rot.z, dlogf, dL_du0, dL_dv0, dc * r2 + dL_dD.z);
  }

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

  if (tu.cam.y > 0.0) { return; }  // the camera only: its share summed above
  let step = max(1.0, tu.opt_params.w);
  let step_u = u32(step);
  let bc = vec2f(1.0 - pow(0.9, step), 1.0 - pow(0.999, step));

  var m0 = adam_m[b + 0u]; var v0 = adam_v[b + 0u];
  var m1 = adam_m[b + 1u]; var v1 = adam_v[b + 1u];
  var m2 = adam_m[b + 2u]; var v2 = adam_v[b + 2u];
  var m3 = adam_m[b + 3u]; var v3 = adam_v[b + 3u];

  // 3DGS-MCMC's lambda_o mean|o| + lambda_s mean|s| (tu.reg: 0 but with monte_carlo)
  let alpha0 = sigmoid(p0.w);
  let dL_dop_r = dL_dop + tu.reg.x * alpha0 * (1.0 - alpha0);
  let dL_dlog_s_r = dL_dlog_s + tu.reg.y * s;

  // Position + logit opacity.
  p0 = adam4(p0, vec4f(dL_dpos, dL_dop_r), &m0, &v0, vec4f(vec3f(tu.lrs.x), tu.limits.w), bc);
  p0.w = clamp(p0.w, -8.0, 8.0);

  // Log-scale, then shape limits: cap each axis by the angular size in this
  // view and by an absolute world size, and the largest axis by the median
  // one times max anisotropy (flat discs stay legal, needles don't).
  p1 = vec4f(clamp(adam3(p1.xyz, dL_dlog_s_r, &m1, &v1, tu.lrs.y, bc), vec3f(-9.5), vec3f(2.0)), p1.w);
  let log_cap = log(max(1e-8, min(tu.limits.x * vpos.z, tu.limits.y)));
  let ls_med = max(min(p1.x, p1.y), min(max(p1.x, p1.y), p1.z));
  let ls_max = min(log_cap, ls_med + tu.limits.z);
  p1 = vec4f(min(p1.xyz, vec3f(ls_max)), p1.w);

  p2 = normalize(adam4(p2, dL_dq, &m2, &v2, vec4f(tu.lrs.z), bc));
  p3 = vec4f(adam3(p3.xyz, dL_drgb, &m3, &v3, tu.lrs.w, bc), p3.w);

  // Monte-Carlo
  if (u32(tu.opt_params.x + 0.5) == 1u) {
    // SGLD (3DGS-MCMC eq. 9): lambda_noise lr_pos sigmoid(-k (alpha - t)) Sigma xi, k = 100, t = 0.005:
    // the nearly transparent splats wander, the others are left to the gradient
    let gate = 1.0 / (1.0 + exp(100.0 * (sigmoid(p0.w) - 0.005)));
    p0 = vec4f(p0.xyz + (tu.opt_params.y * tu.lrs.x * gate) * (Sigma * rand_normal3(idx, step_u)), p0.w);
  }

  adam_m[b + 0u] = m0; adam_v[b + 0u] = v0;
  adam_m[b + 1u] = m1; adam_v[b + 1u] = v1;
  adam_m[b + 2u] = m2; adam_v[b + 2u] = v2;
  adam_m[b + 3u] = m3; adam_v[b + 3u] = v3;

  splats[b + 0u] = p0;
  splats[b + 1u] = p1;
  splats[b + 2u] = p2;
  splats[b + 3u] = p3;
}
