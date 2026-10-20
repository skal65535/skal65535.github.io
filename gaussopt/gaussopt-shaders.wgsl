// gaussopt-shaders.wgsl — WGSL kernels for Gauss-Opt (WebGPU 3DGS Optimizer).
//
// Kernels are separated by `// === <name> ===` markers and compiled into
// independent GPUShaderModules by loadShaders().

// === splats_render ===
// Hardware-instanced quad renderer for interactive Free-View and COLMAP camera
// inspection, supporting all shape modes (kVisus) and diagnostic color heatmaps.

struct RenderUniforms {
  view       : mat4x4f,
  proj       : mat4x4f,
  cam_pos    : vec4f,    // xyz = camera world center, w = scene_radius
  intrinsics : vec4f,    // fx, fy, cx, cy
  screen     : vec4f,    // W, H, num_visible, haze
  modes      : vec4u,    // x = visu_mode, y = color_mode, z = show_sh, w = pad
  thresholds : vec4f,    // x = avg_R (for wlbg_error heatmap), y = opacity_scale, z = znear, w = zfar
};

@group(0) @binding(0) var<uniform> u : RenderUniforms;
@group(0) @binding(1) var<storage, read> splats  : array<vec4f>; // 4 vec4f per splat (16 floats)
@group(0) @binding(2) var<storage, read> moments : array<vec4f>; // 3 vec4f per splat (12 floats)
@group(0) @binding(3) var<storage, read> sorted_idx : array<u32>;

struct VOut {
  @builtin(position) pos   : vec4f,
  @location(0)       uv    : vec2f,
  @location(1) @interpolate(flat) color : vec4f,
};

const SH_C0 : f32 = 0.28209479177387814;
const kQuad = array<vec2f, 4>(
  vec2f(-2.5, -2.5),
  vec2f( 2.5, -2.5),
  vec2f(-2.5,  2.5),
  vec2f( 2.5,  2.5),
);

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(x, -15.0, 15.0)));
}

fn quat_to_mat3(q: vec4f) -> mat3x3f {
  let n = max(1e-8, length(q));
  let w = q.x / n; let x = q.y / n; let y = q.z / n; let z = q.w / n;
  // Column-major mat3x3f
  return mat3x3f(
    vec3f(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y)),
    vec3f(2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
    vec3f(2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y))
  );
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
  let p0 = splats[b + 0u]; // x, y, z, logit_op
  let p1 = splats[b + 1u]; // log_sx, log_sy, log_sz, leaf_id
  let p2 = splats[b + 2u]; // qw, qx, qy, qz
  let p3 = splats[b + 3u]; // f_dc_0, f_dc_1, f_dc_2, flags

  let world_pos = p0.xyz;
  let vpos = u.view * vec4f(world_pos, 1.0);
  if (vpos.z <= u.thresholds.z) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }

  let fx = u.intrinsics.x;
  let fy = u.intrinsics.y;
  let cx = u.intrinsics.z;
  let cy = u.intrinsics.w;
  let W  = u.screen.x;
  let H  = u.screen.y;

  let inv_z  = 1.0 / vpos.z;
  let inv_z2 = inv_z * inv_z;
  let u0 = fx * vpos.x * inv_z + cx;
  let v0 = fy * vpos.y * inv_z + cy;

  let R = quat_to_mat3(p2);
  let s = exp(clamp(p1.xyz, vec3f(-12.0), vec3f(4.0))) * u.screen.w; // haze multiplier
  let M = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let Sigma = M * transpose(M);

  let W3 = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
  let J = mat3x2f(
    vec2f(fx * inv_z, 0.0),
    vec2f(0.0, fy * inv_z),
    vec2f(-fx * vpos.x * inv_z2, -fy * vpos.y * inv_z2)
  );
  let T = J * W3;
  let cov2d = T * Sigma * transpose(T);

  let sxx = cov2d[0][0] + 0.3;
  let syy = cov2d[1][1] + 0.3;
  let sxy = cov2d[0][1];

  let mid = 0.5 * (sxx + syy);
  let dist = 0.5 * (sxx - syy);
  let det = length(vec2f(dist, sxy));
  let lambda1 = clamp(mid + det, 0.1, 16384.0);
  let lambda2 = clamp(mid - det, 0.1, 16384.0);

  var diag = vec2f(1.0, 0.0);
  if (abs(sxy) > 1e-6 || abs(det - dist) > 1e-6) {
    diag = normalize(vec2f(sxy, det - dist));
  }
  let major_px = sqrt(lambda1) * diag;
  let minor_px = sqrt(lambda2) * vec2f(diag.y, -diag.x);

  let quad = kQuad[vid];
  let px_pos = vec2f(u0, v0) + quad.x * major_px + quad.y * minor_px;

  // Convert pixel coordinate (u, v) to WebGPU NDC [-1, 1]
  let ndc_x = (px_pos.x / W) * 2.0 - 1.0;
  let ndc_y = 1.0 - (px_pos.y / H) * 2.0;
  let clip_z = clamp((vpos.z - u.thresholds.z) / (u.thresholds.w - u.thresholds.z), 0.0, 0.9999);

  out.pos = vec4f(ndc_x, ndc_y, clip_z, 1.0);
  out.uv  = quad;

  let alpha = min(0.99, sigmoid(p0.w) * u.thresholds.y);
  var rgb = clamp(vec3f(0.5) + SH_C0 * p3.xyz, vec3f(0.0), vec3f(1.0));

  let cmode = u.modes.y;
  if (cmode == 1u) {
    // Octree leaf ID color
    rgb = palette_leaf(u32(max(0.0, p1.w)));
  } else if (cmode == 2u) {
    // WLBG error mass R_i heatmap
    let m0 = moments[idx * 3u];
    let norm_r = m0.z / max(1e-5, u.thresholds.x * 2.0);
    rgb = heatmap_coolwarm(norm_r);
  } else if (cmode == 3u) {
    // Multi-view diversity N_eff = W^2 / S (magenta = floater < 1.8, green = multi-view)
    let m0 = moments[idx * 3u];
    let n_eff = select(1.0, (m0.x * m0.x) / max(1e-6, m0.y), m0.y > 1e-6);
    let t = clamp((n_eff - 1.0) / 4.0, 0.0, 1.0);
    rgb = mix(vec3f(1.0, 0.1, 0.8), vec3f(0.1, 0.95, 0.4), t);
  } else if (cmode == 4u) {
    // Depth heatmap
    let zn = clamp(vpos.z / max(1.0, u.cam_pos.w * 2.5), 0.0, 1.0);
    rgb = heatmap_coolwarm(1.0 - zn);
  }

  out.color = vec4f(rgb, alpha);
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let v = in.uv;
  let r2 = dot(v, v);
  let r  = sqrt(r2);
  let vmode = u.modes.x;

  var weight : f32 = 0.0;
  if (vmode == 0u) {
    // Standard 2D Gaussian exp(-0.5 * r^2)
    if (r2 > 6.25) { discard; }
    weight = exp(-0.5 * r2);
  } else if (vmode == 1u) {
    // Cone
    if (r > 2.0) { discard; }
    weight = 1.0 - 0.5 * r;
  } else if (vmode == 2u) {
    // Ellipsis ring
    if (r > 2.0) { discard; }
    weight = smoothstep(1.6, 1.85, r) * (1.0 - step(2.0, r)) + 0.15 * exp(-0.5 * r2);
  } else if (vmode == 3u) {
    // Small center dot
    if (r > 0.35) { discard; }
    weight = 1.0;
  } else {
    // Principal axes cross
    if (r > 2.0) { discard; }
    weight = 1.0 - smoothstep(0.0, 0.12, min(abs(v.x), abs(v.y)));
  }

  let a = in.color.a * weight;
  if (a < 1.0 / 255.0) { discard; }
  return vec4f(in.color.rgb * a, a);
}

// === wireframe_render ===
// Renders 3D line segments (octree boxes, camera frustums, WLBG split vectors).
struct WireUniforms {
  view       : mat4x4f,
  intrinsics : vec4f, // fx, fy, cx, cy
  screen     : vec4f, // W, H, znear, zfar
};

@group(0) @binding(0) var<uniform> wu : WireUniforms;
@group(0) @binding(1) var<storage, read> verts : array<vec4f>; // xyz, bitcast<f32>(abgr_u32)

struct WireOut {
  @builtin(position) pos   : vec4f,
  @location(0)       color : vec4f,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> WireOut {
  var out : WireOut;
  let v = verts[vid];
  let vpos = wu.view * vec4f(v.xyz, 1.0);
  let z = max(0.05, vpos.z);
  let u_px = wu.intrinsics.x * (vpos.x / z) + wu.intrinsics.z;
  let v_px = wu.intrinsics.y * (vpos.y / z) + wu.intrinsics.w;
  let ndc_x = (u_px / wu.screen.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (v_px / wu.screen.y) * 2.0;
  let clip_z = select(2.0, clamp((z - wu.screen.z) / (wu.screen.w - wu.screen.z), 0.0, 0.999), vpos.z > 0.05);
  out.pos = vec4f(ndc_x, ndc_y, clip_z, 1.0);
  out.color = unpack4x8unorm(bitcast<u32>(v.w));
  return out;
}

@fragment
fn fs(in: WireOut) -> @location(0) vec4f {
  return vec4f(in.color.rgb * in.color.a, in.color.a);
}

// === compare_blit ===
// Fullscreen overlay for Ground-Truth / Rendered / Residual Split-Screen comparison.
struct BlitParams {
  mode    : u32, // 0=target_only, 1=split_target_left, 2=residual_heatmap
  split_x : f32, // in [0, 1]
  amp     : f32, // residual amplification
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
  if (bp.mode == 0u) {
    return vec4f(gt, 1.0);
  }
  if (bp.mode == 2u) {
    let err = abs(gt - hat) * bp.amp;
    return vec4f(err, 1.0);
  }
  // Split screen wipe
  if (abs(in.uv.x - bp.split_x) < 0.002) {
    return vec4f(1.0, 0.9, 0.2, 1.0);
  }
  let col = select(hat, gt, in.uv.x < bp.split_x);
  return vec4f(col, 1.0);
}

// === project_splats ===
// Projects the `num_visible` front-to-back ordered splats into 2D screen space
// (`proj_splats`: 2 vec4f per visible splat = [u0, v0, z_cam, radius_px], [conic_a, conic_b, conic_c, orig_idx]).

struct TrainUniforms {
  view       : mat4x4f,
  cam_pos    : vec4f, // xyz = camera world center, w = step_num
  intrinsics : vec4f, // fx, fy, cx, cy
  dims       : vec4u, // W, H, num_visible, total_splats
  lrs        : vec4f, // lr_pos, lr_scale, lr_quat, lr_color_op
  opt_params : vec4f, // x = opt_method (0=Adam, 1=MC, 2=SA), y = mc_noise/sa_temp, z = seed, w = scene_radius
};

@group(0) @binding(0) var<uniform> tu : TrainUniforms;
@group(0) @binding(1) var<storage, read>       splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read>       sorted_idx  : array<u32>;
@group(0) @binding(3) var<storage, read_write> proj_splats : array<vec4f>; // 2 vec4f per visible splat

fn quat_to_mat3_p(q: vec4f) -> mat3x3f {
  let n = max(1e-8, length(q));
  let w = q.x / n; let x = q.y / n; let y = q.z / n; let z = q.w / n;
  return mat3x3f(
    vec3f(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y)),
    vec3f(2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
    vec3f(2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y))
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= tu.dims.z) { return; }

  let idx = sorted_idx[k];
  let b   = idx * 4u;
  let p0  = splats[b + 0u];
  let p1  = splats[b + 1u];
  let p2  = splats[b + 2u];

  let vpos = tu.view * vec4f(p0.xyz, 1.0);
  if (vpos.z <= 0.1) {
    proj_splats[k * 2u + 0u] = vec4f(-1e5, -1e5, -1.0, 0.0);
    proj_splats[k * 2u + 1u] = vec4f(0.0, 0.0, 0.0, bitcast<f32>(idx));
    return;
  }

  let fx = tu.intrinsics.x;
  let fy = tu.intrinsics.y;
  let cx = tu.intrinsics.z;
  let cy = tu.intrinsics.w;
  let inv_z  = 1.0 / vpos.z;
  let inv_z2 = inv_z * inv_z;
  let u0 = fx * vpos.x * inv_z + cx;
  let v0 = fy * vpos.y * inv_z + cy;

  let R = quat_to_mat3_p(p2);
  let s = exp(clamp(p1.xyz, vec3f(-10.0), vec3f(3.0)));
  let M = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let Sigma = M * transpose(M);

  let W3 = mat3x3f(tu.view[0].xyz, tu.view[1].xyz, tu.view[2].xyz);
  let J = mat3x2f(
    vec2f(fx * inv_z, 0.0),
    vec2f(0.0, fy * inv_z),
    vec2f(-fx * vpos.x * inv_z2, -fy * vpos.y * inv_z2)
  );
  let T = J * W3;
  let cov2d = T * Sigma * transpose(T);

  let sxx = cov2d[0][0] + 0.3;
  let syy = cov2d[1][1] + 0.3;
  let sxy = cov2d[0][1];
  let det = max(1e-6, sxx * syy - sxy * sxy);
  let inv_det = 1.0 / det;

  let conic = vec3f(syy * inv_det, -sxy * inv_det, sxx * inv_det);
  let mid   = 0.5 * (sxx + syy);
  let lambda_max = mid + sqrt(max(0.1, mid * mid - det));
  let radius_px  = ceil(3.0 * sqrt(lambda_max));

  proj_splats[k * 2u + 0u] = vec4f(u0, v0, vpos.z, radius_px);
  proj_splats[k * 2u + 1u] = vec4f(conic, bitcast<f32>(idx));
}

// === tile_forward ===
// 16x16 tile compute rasterizer: integrates front-to-back in batches of 64 splats
// loaded into workgroup shared memory, writing rendered color + final T + last_k.

struct TrainUniforms {
  view       : mat4x4f,
  cam_pos    : vec4f,
  intrinsics : vec4f,
  dims       : vec4u, // W, H, num_visible, total_splats
  lrs        : vec4f,
  opt_params : vec4f,
};

@group(0) @binding(0) var<uniform> tu          : TrainUniforms;
@group(0) @binding(1) var<storage, read> splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read> proj_splats : array<vec4f>;
@group(0) @binding(3) var out_tex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> pix_state : array<vec4f>; // [T_final, z_med, bitcast<f32>(last_k), 0]

const SH_C0_F : f32 = 0.28209479177387814;
var<workgroup> wg_u0v0zr : array<vec4f, 64>;
var<workgroup> wg_conic  : array<vec4f, 64>;
var<workgroup> wg_rgba   : array<vec4f, 64>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let W = tu.dims.x;
  let H = tu.dims.y;
  let num_vis = tu.dims.z;
  let in_bounds = (gid.x < W && gid.y < H);
  let px = vec2f(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  var C = vec3f(0.0);
  var T = 1.0;
  var z_med = 0.0;
  var last_k : u32 = 0u;
  var done = !in_bounds;

  let num_batches = (num_vis + 63u) / 64u;
  for (var batch = 0u; batch < num_batches; batch = batch + 1u) {
    let k = batch * 64u + lid;
    if (lid < 64u) {
      if (k < num_vis) {
        let pr0 = proj_splats[k * 2u + 0u];
        let pr1 = proj_splats[k * 2u + 1u];
        let idx = bitcast<u32>(pr1.w);
        let p0  = splats[idx * 4u + 0u];
        let p3  = splats[idx * 4u + 3u];
        let rgb = clamp(vec3f(0.5) + SH_C0_F * p3.xyz, vec3f(0.0), vec3f(1.0));
        let op  = 1.0 / (1.0 + exp(-clamp(p0.w, -15.0, 15.0)));
        wg_u0v0zr[lid] = pr0;
        wg_conic[lid]  = pr1;
        wg_rgba[lid]   = vec4f(rgb, op);
      } else {
        wg_u0v0zr[lid] = vec4f(-1e5, -1e5, -1.0, 0.0);
      }
    }
    workgroupBarrier();

    if (!done) {
      let batch_end = min(64u, num_vis - batch * 64u);
      for (var i = 0u; i < batch_end; i = i + 1u) {
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
        if (T >= 0.5 && next_T < 0.5) {
          z_med = u0v0zr.z;
        }
        C = C + (T * alpha) * rgba.rgb;
        T = next_T;
        last_k = batch * 64u + i + 1u;
        if (T < 1e-4) {
          done = true;
          break;
        }
      }
    }
    workgroupBarrier();
  }

  if (in_bounds) {
    textureStore(out_tex, vec2i(i32(gid.x), i32(gid.y)), vec4f(C, 1.0));
    let pid = gid.y * W + gid.x;
    pix_state[pid] = vec4f(T, z_med, bitcast<f32>(last_k), 0.0);
  }
}

// === tile_backward ===
// 16x16 tile backward pass + 3D WLBG error-moment accumulation.
// Traverses batches back-to-front, recovers T_i, and accumulates:
//   - 2D parameter gradients in `grads_2d` (11 floats per splat: dL/drgb(3), dL/dop(1), dL/duv(2), dL/dconic(3), w_view(1), w_err_view(1))
//   - 3D WLBG moments in `moments` (12 floats per splat: W_i, S_i, R_i, R1(3), R2(6))
// Uses workgroup reduction so at most 1 global atomic CAS per splat per tile is executed!

struct TrainUniforms {
  view       : mat4x4f,
  cam_pos    : vec4f,
  intrinsics : vec4f,
  dims       : vec4u,
  lrs        : vec4f,
  opt_params : vec4f,
};

@group(0) @binding(0) var<uniform> tu          : TrainUniforms;
@group(0) @binding(1) var<storage, read> splats      : array<vec4f>;
@group(0) @binding(2) var<storage, read> proj_splats : array<vec4f>;
@group(0) @binding(3) var gt_tex  : texture_2d<f32>;
@group(0) @binding(4) var hat_tex : texture_2d<f32>;
@group(0) @binding(5) var<storage, read> pix_state   : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> grads_2d : array<atomic<u32>>; // 12 u32 per splat
@group(0) @binding(7) var<storage, read_write> moments  : array<atomic<u32>>; // 12 u32 per splat
@group(0) @binding(8) var<storage, read_write> loss_acc : atomic<u32>;        // fixed-point MSE sum

const SH_C0_B : f32 = 0.28209479177387814;

var<workgroup> b_u0v0zr : array<vec4f, 32>;
var<workgroup> b_conic  : array<vec4f, 32>;
var<workgroup> b_rgba   : array<vec4f, 32>;
var<workgroup> b_wpos   : array<vec4f, 32>;
var<workgroup> wg_grads : array<atomic<u32>, 384>; // 32 splats * 12 slots
var<workgroup> wg_moms  : array<atomic<u32>, 384>; // 32 splats * 12 slots
var<workgroup> wg_mse   : atomic<u32>;

fn add_wg_grad(slot: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&wg_grads[slot]);
  loop {
    let r = atomicCompareExchangeWeak(&wg_grads[slot], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn add_wg_mom(slot: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&wg_moms[slot]);
  loop {
    let r = atomicCompareExchangeWeak(&wg_moms[slot], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn add_global_grad(idx: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&grads_2d[idx]);
  loop {
    let r = atomicCompareExchangeWeak(&grads_2d[idx], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

fn add_global_mom(idx: u32, val: f32) {
  if (abs(val) < 1e-10) { return; }
  var old = atomicLoad(&moments[idx]);
  loop {
    let r = atomicCompareExchangeWeak(&moments[idx], old, bitcast<u32>(bitcast<f32>(old) + val));
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_index) lid: u32) {
  let W = tu.dims.x;
  let H = tu.dims.y;
  let num_vis = tu.dims.z;
  let in_bounds = (gid.x < W && gid.y < H);
  let px = vec2f(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  if (lid == 0u) { atomicStore(&wg_mse, 0u); }
  workgroupBarrier();

  var dL_dC = vec3f(0.0);
  var err_l1 : f32 = 0.0;
  var T : f32 = 1.0;
  var last_k : u32 = 0u;
  var accum_rec = vec3f(0.0);

  if (in_bounds) {
    let p_coord = vec2i(i32(gid.x), i32(gid.y));
    let gt  = textureLoad(gt_tex, p_coord, 0).rgb;
    let hat = textureLoad(hat_tex, p_coord, 0).rgb;
    let diff = hat - gt;
    let mse = dot(diff, diff) / 3.0;
    atomicAdd(&wg_mse, u32(mse * 65536.0 + 0.5));

    // Combined 0.8 * L1 + 0.2 * L2 pixel derivative, normalized by image area
    let inv_npix = 1.0 / f32(W * H);
    let sign_d = clamp(diff * 25.0, vec3f(-1.0), vec3f(1.0));
    dL_dC = (0.8 * sign_d + 0.4 * diff) * inv_npix;
    err_l1 = (abs(diff.x) + abs(diff.y) + abs(diff.z)) / 3.0;

    let st = pix_state[gid.y * W + gid.x];
    T = st.x;
    last_k = bitcast<u32>(st.z);
  }
  workgroupBarrier();
  if (lid == 0u) { atomicAdd(&loss_acc, atomicLoad(&wg_mse)); }

  let fx = tu.intrinsics.x;
  let fy = tu.intrinsics.y;
  let W3_t = transpose(mat3x3f(tu.view[0].xyz, tu.view[1].xyz, tu.view[2].xyz));

  let num_batches = (num_vis + 31u) / 32u;
  for (var b_rev = 0u; b_rev < num_batches; b_rev = b_rev + 1u) {
    let batch = num_batches - 1u - b_rev;
    let k = batch * 32u + lid;

    // Zero workgroup accumulators (384 u32 / 256 threads => up to 2 writes per thread)
    if (lid < 256u) {
      atomicStore(&wg_grads[lid], 0u);
      atomicStore(&wg_moms[lid], 0u);
    }
    if (lid + 256u < 384u) {
      atomicStore(&wg_grads[lid + 256u], 0u);
      atomicStore(&wg_moms[lid + 256u], 0u);
    }

    if (lid < 32u) {
      if (k < num_vis) {
        let pr0 = proj_splats[k * 2u + 0u];
        let pr1 = proj_splats[k * 2u + 1u];
        let idx = bitcast<u32>(pr1.w);
        let p0  = splats[idx * 4u + 0u];
        let p3  = splats[idx * 4u + 3u];
        let rgb = clamp(vec3f(0.5) + SH_C0_B * p3.xyz, vec3f(0.0), vec3f(1.0));
        let op  = 1.0 / (1.0 + exp(-clamp(p0.w, -15.0, 15.0)));
        b_u0v0zr[lid] = pr0;
        b_conic[lid]  = pr1;
        b_rgba[lid]   = vec4f(rgb, op);
        b_wpos[lid]   = p0;
      } else {
        b_u0v0zr[lid] = vec4f(-1e5, -1e5, -1.0, 0.0);
      }
    }
    workgroupBarrier();

    if (in_bounds && last_k > batch * 32u) {
      let batch_end = min(32u, num_vis - batch * 32u);
      for (var ir = 0u; ir < batch_end; ir = ir + 1u) {
        let i = batch_end - 1u - ir;
        let global_k = batch * 32u + i;
        if (global_k >= last_k) { continue; }

        let u0v0zr = b_u0v0zr[i];
        let r = u0v0zr.w;
        if (r <= 0.0) { continue; }
        let dx = px.x - u0v0zr.x;
        let dy = px.y - u0v0zr.y;
        if (abs(dx) > r || abs(dy) > r) { continue; }

        let conic = b_conic[i].xyz;
        let power = 0.5 * (conic.x * dx * dx + 2.0 * conic.y * dx * dy + conic.z * dy * dy);
        if (power > 4.5) { continue; }

        let rgba = b_rgba[i];
        let G = exp(-power);
        let alpha = min(0.99, rgba.w * G);
        if (alpha < 1.0 / 255.0) { continue; }

        // Recover transmittance T_i before splat i
        T = T / max(1e-4, 1.0 - alpha);
        let weight = T * alpha;
        let w_err  = weight * err_l1;

        let dL_dc = dL_dC * weight * SH_C0_B;
        let dL_dalpha = T * dot(dL_dC, rgba.rgb - accum_rec);
        accum_rec = alpha * rgba.rgb + (1.0 - alpha) * accum_rec;

        let dL_dlogit_op = dL_dalpha * alpha * (1.0 - rgba.w);
        let dL_dP = -dL_dalpha * alpha;
        let dL_du0 = -dL_dP * (conic.x * dx + conic.y * dy);
        let dL_dv0 = -dL_dP * (conic.y * dx + conic.z * dy);
        let dL_da  =  0.5 * dL_dP * dx * dx;
        let dL_db  =        dL_dP * dx * dy;
        let dL_dc_conic = 0.5 * dL_dP * dy * dy;

        let gbase = i * 12u;
        add_wg_grad(gbase + 0u, dL_dc.x);
        add_wg_grad(gbase + 1u, dL_dc.y);
        add_wg_grad(gbase + 2u, dL_dc.z);
        add_wg_grad(gbase + 3u, dL_dlogit_op);
        add_wg_grad(gbase + 4u, dL_du0);
        add_wg_grad(gbase + 5u, dL_dv0);
        add_wg_grad(gbase + 6u, dL_da);
        add_wg_grad(gbase + 7u, dL_db);
        add_wg_grad(gbase + 8u, dL_dc_conic);
        add_wg_grad(gbase + 9u, weight); // per-view visibility weight w_{i,c}
        add_wg_grad(gbase + 10u, w_err); // per-view error weight r_{i,c} (for Simulated Annealing energy)

        // 3D WLBG error moments (Section 5.1)
        let delta_cam = vec3f(dx * u0v0zr.z / fx, dy * u0v0zr.z / fy, 0.0);
        let delta_w   = W3_t * delta_cam;
        add_wg_mom(gbase + 0u, weight);
        add_wg_mom(gbase + 2u, w_err);
        add_wg_mom(gbase + 3u, w_err * delta_w.x);
        add_wg_mom(gbase + 4u, w_err * delta_w.y);
        add_wg_mom(gbase + 5u, w_err * delta_w.z);
        add_wg_mom(gbase + 6u, w_err * delta_w.x * delta_w.x);
        add_wg_mom(gbase + 7u, w_err * delta_w.y * delta_w.y);
        add_wg_mom(gbase + 8u, w_err * delta_w.z * delta_w.z);
        add_wg_mom(gbase + 9u, w_err * delta_w.x * delta_w.y);
        add_wg_mom(gbase + 10u, w_err * delta_w.x * delta_w.z);
        add_wg_mom(gbase + 11u, w_err * delta_w.y * delta_w.z);
      }
    }
    workgroupBarrier();

    // Flush workgroup-reduced gradients & moments to global memory (1 thread per slot)
    let slot_a = lid;
    if (slot_a < 384u) {
      let s_in_batch = slot_a / 12u;
      let c_in_splat = slot_a % 12u;
      let gk = batch * 32u + s_in_batch;
      if (gk < num_vis) {
        let orig_idx = bitcast<u32>(b_conic[s_in_batch].w);
        let gval = bitcast<f32>(atomicLoad(&wg_grads[slot_a]));
        let mval = bitcast<f32>(atomicLoad(&wg_moms[slot_a]));
        add_global_grad(orig_idx * 12u + c_in_splat, gval);
        add_global_mom(orig_idx * 12u + c_in_splat, mval);
      }
    }
    let slot_b = lid + 256u;
    if (slot_b < 384u) {
      let s_in_batch = slot_b / 12u;
      let c_in_splat = slot_b % 12u;
      let gk = batch * 32u + s_in_batch;
      if (gk < num_vis) {
        let orig_idx = bitcast<u32>(b_conic[s_in_batch].w);
        let gval = bitcast<f32>(atomicLoad(&wg_grads[slot_b]));
        let mval = bitcast<f32>(atomicLoad(&wg_moms[slot_b]));
        add_global_grad(orig_idx * 12u + c_in_splat, gval);
        add_global_mom(orig_idx * 12u + c_in_splat, mval);
      }
    }
    workgroupBarrier();
  }
}

// === adam_step ===
// Per-splat compute pass supporting three selectable optimizer methods:
//   - opt_method = 0: Deterministic Adam
//   - opt_method = 1: Monte-Carlo / SGLD (3DGS-MCMC covariance-scaled Langevin noise)
//   - opt_method = 2: Simulated Annealing (Metropolis-Hastings per-splat energy acceptance)

struct TrainUniforms {
  view       : mat4x4f,
  cam_pos    : vec4f, // w = step_num (1-indexed)
  intrinsics : vec4f,
  dims       : vec4u, // W, H, num_visible, total_splats
  lrs        : vec4f, // lr_pos, lr_scale, lr_quat, lr_color_op
  opt_params : vec4f, // x = opt_method (0=Adam, 1=MC, 2=SA), y = mc_noise/sa_temp, z = rng_seed, w = scene_radius
};

@group(0) @binding(0) var<uniform> tu       : TrainUniforms;
@group(0) @binding(1) var<storage, read_write> splats   : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> grads_2d : array<u32>;
@group(0) @binding(3) var<storage, read_write> adam_m   : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> adam_v   : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> moments  : array<u32>;

fn pcg_hash(v_in: u32) -> u32 {
  let state = v_in * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return clamp(f32(pcg_hash(seed) & 0x00ffffffu) / 16777216.0, 1e-6, 0.999999);
}

fn rand_normal3(idx: u32, step_u: u32) -> vec3f {
  let s0 = idx * 1973u + step_u * 9277u + u32(tu.opt_params.z * 65536.0);
  let u1 = rand01(s0 + 1u);
  let u2 = rand01(s0 + 2u);
  let u3 = rand01(s0 + 3u);
  let u4 = rand01(s0 + 4u);
  let r1 = sqrt(-2.0 * log(u1));
  let r2 = sqrt(-2.0 * log(u3));
  return vec3f(
    r1 * cos(6.2831853 * u2),
    r1 * sin(6.2831853 * u2),
    r2 * cos(6.2831853 * u4)
  );
}

fn quat_to_mat3_a(q: vec4f) -> mat3x3f {
  let n = max(1e-8, length(q));
  let w = q.x / n; let x = q.y / n; let y = q.z / n; let z = q.w / n;
  return mat3x3f(
    vec3f(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y)),
    vec3f(2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x)),
    vec3f(2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y))
  );
}

fn adam_update(param: f32, grad: f32, m_in: ptr<function, f32>, v_in: ptr<function, f32>, lr: f32, b1_corr: f32, b2_corr: f32) -> f32 {
  let g = clamp(grad, -1.0, 1.0);
  let m = 0.9 * (*m_in) + 0.1 * g;
  let v = 0.999 * (*v_in) + 0.001 * g * g;
  *m_in = m;
  *v_in = v;
  let m_hat = m / b1_corr;
  let v_hat = v / b2_corr;
  return param - lr * m_hat / (sqrt(v_hat) + 1e-8);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= tu.dims.w) { return; }

  let gbase = idx * 12u;
  let dL_dr     = bitcast<f32>(grads_2d[gbase + 0u]);
  let dL_dg     = bitcast<f32>(grads_2d[gbase + 1u]);
  let dL_db_c   = bitcast<f32>(grads_2d[gbase + 2u]);
  let dL_dop    = bitcast<f32>(grads_2d[gbase + 3u]);
  let dL_du0    = bitcast<f32>(grads_2d[gbase + 4u]);
  let dL_dv0    = bitcast<f32>(grads_2d[gbase + 5u]);
  let dL_da     = bitcast<f32>(grads_2d[gbase + 6u]);
  let dL_db     = bitcast<f32>(grads_2d[gbase + 7u]);
  let dL_dc     = bitcast<f32>(grads_2d[gbase + 8u]);
  let w_view    = bitcast<f32>(grads_2d[gbase + 9u]);
  let w_err_v   = bitcast<f32>(grads_2d[gbase + 10u]);

  // Clear 2D grads for next step
  for (var c = 0u; c < 12u; c = c + 1u) {
    grads_2d[gbase + c] = 0u;
  }

  // Update exact view-concentration S_i = sum_c (w_{i,c}^2) for N_eff = W_i^2 / S_i
  if (w_view > 1e-6) {
    let S_old = bitcast<f32>(moments[gbase + 1u]);
    moments[gbase + 1u] = bitcast<u32>(S_old + w_view * w_view);
  }

  let grad_norm = abs(dL_dr) + abs(dL_dg) + abs(dL_db_c) + abs(dL_dop) + abs(dL_du0) + abs(dL_dv0);
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

  // Chain rule (dL/du0, dL/dv0) -> dL/dvpos -> dL/dpos_world
  let dL_dvpos = vec3f(
    dL_du0 * fx * inv_z,
    dL_dv0 * fy * inv_z,
    -(dL_du0 * fx * vpos.x + dL_dv0 * fy * vpos.y) * inv_z2
  );
  let W3 = mat3x3f(tu.view[0].xyz, tu.view[1].xyz, tu.view[2].xyz);
  let dL_dpos = transpose(W3) * dL_dvpos;

  // Chain rule (dL/da, dL/db, dL/dc) -> dL/dSigma2D -> dL/dSigma3D -> dL/dscale & dL/dquat
  let qn = max(1e-8, length(p2));
  let qw = p2.x / qn; let qx = p2.y / qn; let qy = p2.z / qn; let qz = p2.w / qn;
  let R = quat_to_mat3_a(p2);
  let s = exp(clamp(p1.xyz, vec3f(-10.0), vec3f(3.0)));
  let M = mat3x3f(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let Sigma = M * transpose(M);

  let J = mat3x2f(
    vec2f(fx * inv_z, 0.0),
    vec2f(0.0, fy * inv_z),
    vec2f(-fx * vpos.x * inv_z2, -fy * vpos.y * inv_z2)
  );
  let T = J * W3;
  let cov2d = T * Sigma * transpose(T);
  let sxx = cov2d[0][0] + 0.3;
  let syy = cov2d[1][1] + 0.3;
  let sxy = cov2d[0][1];
  let det = max(1e-6, sxx * syy - sxy * sxy);
  let inv_det = 1.0 / det;
  let ca = syy * inv_det;
  let cb = -sxy * inv_det;
  let cc = sxx * inv_det;

  // dL/d(cov2d) = -conic * (dL/dconic) * conic
  let dL_dsxx = -(ca * ca * dL_da + 2.0 * ca * cb * dL_db + cb * cb * dL_dc);
  let dL_dsyy = -(cb * cb * dL_da + 2.0 * cb * cc * dL_db + cc * cc * dL_dc);
  let dL_dsxy = -(ca * cb * dL_da + (ca * cc + cb * cb) * dL_db + cb * cc * dL_dc);

  let dL_dcov2d = mat2x2f(vec2f(dL_dsxx, dL_dsxy), vec2f(dL_dsxy, dL_dsyy));
  let dL_dSigma = transpose(T) * dL_dcov2d * T;
  let dL_dM = 2.0 * dL_dSigma * M;

  let dL_dlog_s = vec3f(
    dot(R[0], dL_dM[0]) * s.x,
    dot(R[1], dL_dM[1]) * s.y,
    dot(R[2], dL_dM[2]) * s.z
  );

  // Quaternion gradient dL/dq from dL/dR = [dL_dM[0]*s.x, dL_dM[1]*s.y, dL_dM[2]*s.z]
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
  let b1_corr = 1.0 - pow(0.9, step);
  let b2_corr = 1.0 - pow(0.999, step);

  var m0 = adam_m[b + 0u]; var v0 = adam_v[b + 0u];
  var m1 = adam_m[b + 1u]; var v1 = adam_v[b + 1u];
  var m2 = adam_m[b + 2u]; var v2 = adam_v[b + 2u];
  var m3 = adam_m[b + 3u]; var v3 = adam_v[b + 3u];

  var fm0x = m0.x; var fv0x = v0.x;
  p0.x = adam_update(p0.x, dL_dpos.x, &fm0x, &fv0x, tu.lrs.x, b1_corr, b2_corr);
  var fm0y = m0.y; var fv0y = v0.y;
  p0.y = adam_update(p0.y, dL_dpos.y, &fm0y, &fv0y, tu.lrs.x, b1_corr, b2_corr);
  var fm0z = m0.z; var fv0z = v0.z;
  p0.z = adam_update(p0.z, dL_dpos.z, &fm0z, &fv0z, tu.lrs.x, b1_corr, b2_corr);
  var fm0w = m0.w; var fv0w = v0.w;
  p0.w = clamp(adam_update(p0.w, dL_dop, &fm0w, &fv0w, tu.lrs.w * 2.0, b1_corr, b2_corr), -8.0, 8.0);
  adam_m[b + 0u] = vec4f(fm0x, fm0y, fm0z, fm0w);
  adam_v[b + 0u] = vec4f(fv0x, fv0y, fv0z, fv0w);

  var fm1x = m1.x; var fv1x = v1.x;
  p1.x = clamp(adam_update(p1.x, dL_dlog_s.x, &fm1x, &fv1x, tu.lrs.y, b1_corr, b2_corr), -9.5, 2.0);
  var fm1y = m1.y; var fv1y = v1.y;
  p1.y = clamp(adam_update(p1.y, dL_dlog_s.y, &fm1y, &fv1y, tu.lrs.y, b1_corr, b2_corr), -9.5, 2.0);
  var fm1z = m1.z; var fv1z = v1.z;
  p1.z = clamp(adam_update(p1.z, dL_dlog_s.z, &fm1z, &fv1z, tu.lrs.y, b1_corr, b2_corr), -9.5, 2.0);

  var fm2x = m2.x; var fv2x = v2.x;
  p2.x = adam_update(p2.x, dL_dq.x, &fm2x, &fv2x, tu.lrs.z, b1_corr, b2_corr);
  var fm2y = m2.y; var fv2y = v2.y;
  p2.y = adam_update(p2.y, dL_dq.y, &fm2y, &fv2y, tu.lrs.z, b1_corr, b2_corr);
  var fm2z = m2.z; var fv2z = v2.z;
  p2.z = adam_update(p2.z, dL_dq.z, &fm2z, &fv2z, tu.lrs.z, b1_corr, b2_corr);
  var fm2w = m2.w; var fv2w = v2.w;
  p2.w = adam_update(p2.w, dL_dq.w, &fm2w, &fv2w, tu.lrs.z, b1_corr, b2_corr);
  p2 = normalize(p2);
  adam_m[b + 2u] = vec4f(fm2x, fm2y, fm2z, fm2w);
  adam_v[b + 2u] = vec4f(fv2x, fv2y, fv2z, fv2w);

  var fm3x = m3.x; var fv3x = v3.x;
  p3.x = adam_update(p3.x, dL_dr, &fm3x, &fv3x, tu.lrs.w, b1_corr, b2_corr);
  var fm3y = m3.y; var fv3y = v3.y;
  p3.y = adam_update(p3.y, dL_dg, &fm3y, &fv3y, tu.lrs.w, b1_corr, b2_corr);
  var fm3z = m3.z; var fv3z = v3.z;
  p3.z = adam_update(p3.z, dL_db_c, &fm3z, &fv3z, tu.lrs.w, b1_corr, b2_corr);

  // ── Stochastic Optimizer Extensions (Section 5.4) ──
  let opt_mode = u32(tu.opt_params.x + 0.5);
  let alpha_cur = 1.0 / (1.0 + exp(-clamp(p0.w, -15.0, 15.0)));
  var prev_energy = v1.w;
  var prev_jump_x = m1.w;
  var prev_jump_y = m3.w;
  var prev_jump_z = v3.w;

  if (opt_mode == 1u) {
    // Mode 1: Monte-Carlo / SGLD (3DGS-MCMC covariance-scaled Langevin diffusion)
    // Low-opacity splats diffuse along their 3D covariance axes M * xi.
    let noise_amp = tu.opt_params.y * (1.0 - alpha_cur) * (1.0 - alpha_cur);
    let xi = rand_normal3(idx, step_u);
    let cov_noise = M * xi;
    p0 = vec4f(p0.xyz + noise_amp * cov_noise, p0.w);
  } else if (opt_mode == 2u && w_view > 1e-4) {
    // Mode 2: Simulated Annealing with Metropolis-Hastings Energy Acceptance
    let e_cur = w_err_v / w_view;
    let temp  = max(1e-5, tu.opt_params.y);
    let dE    = e_cur - prev_energy;
    let u_acc = rand01(idx * 3571u + step_u * 6151u);
    let p_acc = exp(-max(0.0, dE) / temp);

    if (prev_energy > 0.0 && dE > 0.0 && u_acc > p_acc) {
      // Reject uphill move: undo previous stochastic jump and sample a fresh direction
      p0 = vec4f(p0.xyz - vec3f(prev_jump_x, prev_jump_y, prev_jump_z), p0.w);
      let xi = rand_normal3(idx + 17u, step_u);
      let jump = (temp * 0.25) * (M * xi);
      p0 = vec4f(p0.xyz + jump, p0.w);
      prev_jump_x = jump.x;
      prev_jump_y = jump.y;
      prev_jump_z = jump.z;
    } else {
      // Accept move: update baseline energy and apply temperature-scaled perturbation
      prev_energy = select(e_cur, 0.7 * prev_energy + 0.3 * e_cur, prev_energy > 0.0);
      let xi = rand_normal3(idx, step_u);
      let jump = (temp * 0.15 * max(0.05, e_cur)) * (M * xi);
      p0 = vec4f(p0.xyz + jump, p0.w);
      prev_jump_x = jump.x;
      prev_jump_y = jump.y;
      prev_jump_z = jump.z;
    }
  }

  adam_m[b + 1u] = vec4f(fm1x, fm1y, fm1z, prev_jump_x);
  adam_v[b + 1u] = vec4f(fv1x, fv1y, fv1z, prev_energy);
  adam_m[b + 3u] = vec4f(fm3x, fm3y, fm3z, prev_jump_y);
  adam_v[b + 3u] = vec4f(fv3x, fv3y, fv3z, prev_jump_z);

  splats[b + 0u] = p0;
  splats[b + 1u] = p1;
  splats[b + 2u] = p2;
  splats[b + 3u] = p3;
}
