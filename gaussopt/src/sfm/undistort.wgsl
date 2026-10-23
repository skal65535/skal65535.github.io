// undistort.wgsl — Remap an RGBA image from the SfM camera (OPENCV distortion) to a pinhole
// camera of the same size, as cv2.undistort: bilinear, black outside the source (DESIGN-sfm §4).
// Pixels are packed RGBA8 in u32 (little endian: R in the low byte); pixel centers at +0.5.

struct Params {
  w: u32, h: u32, _a: u32, _b: u32,
  pinhole: vec4f,     // fx, fy, cx, cy of the output
  camera: vec4f,      // fx, fy, cx, cy of the source
  dist: vec4f,        // k1, k2, p1, p2
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn texel(x: i32, y: i32) -> vec4f {
  if (x < 0 || y < 0 || x >= i32(P.w) || y >= i32(P.h)) { return vec4f(0.0); }
  return unpack4x8unorm(src[u32(y) * P.w + u32(x)]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.w || id.y >= P.h) { return; }
  let x = (f32(id.x) + 0.5 - P.pinhole.z) / P.pinhole.x;
  let y = (f32(id.y) + 0.5 - P.pinhole.w) / P.pinhole.y;
  let r2 = x * x + y * y;
  let radial = P.dist.x * r2 + P.dist.y * r2 * r2;
  let xd = x + x * radial + 2.0 * P.dist.z * x * y + P.dist.w * (r2 + 2.0 * x * x);
  let yd = y + y * radial + 2.0 * P.dist.w * x * y + P.dist.z * (r2 + 2.0 * y * y);
  // source pixel coordinates, in texel indices
  let u = P.camera.x * xd + P.camera.z - 0.5;
  let v = P.camera.y * yd + P.camera.w - 0.5;
  let x0 = i32(floor(u));
  let y0 = i32(floor(v));
  let fx = u - f32(x0);
  let fy = v - f32(y0);
  let c = mix(mix(texel(x0, y0), texel(x0 + 1, y0), fx), mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), fx), fy);
  dst[id.y * P.w + id.x] = pack4x8unorm(c);
}
