// match.wgsl — Brute-force descriptor matching (DESIGN-sfm §5.3): for each query descriptor, the
// best and second-best dot product over all reference descriptors and the best index. Descriptors
// are 128 x u8 packed in 32 u32; COLMAP's normalization (unit L2 x 512) makes the dot product of
// two descriptors 512^2 cos(angle).
//
// Tiled like a matrix product: a workgroup of 16 x 16 threads owns 64 queries; reference tiles of
// 64 stream through workgroup memory, stored transposed (word k of reference r at k * 64 + r) so
// the 16 threads of a row read consecutive words. Each thread accumulates a 4 x 4 block of dot
// products (queries ty * 4 + i, references tx * 4 + j) and keeps the top two per query; the 16
// threads sharing a query merge their top two at the end. Ties resolve to the smallest reference
// index, like the sequential CPU reference (match.js). dot4U8Packed needs
// 'packed_4x8_integer_dot_product' (match.js swaps in an unpack4xU8 fallback otherwise).

struct Params { num_query: u32, num_ref: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<u32>;
@group(0) @binding(2) var<storage, read> refs: array<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<vec4<u32>>;  // best idx, best dot, 2nd dot, 0

const NONE: u32 = 0xffffffffu;
var<workgroup> qs: array<u32, 2048>;  // 64 queries x 32 words, row-major
var<workgroup> rs: array<u32, 2048>;  // 64 references x 32 words, transposed (k * 64 + r)

fn dot4(a: u32, b: u32) -> u32 { return dot4U8Packed(a, b); }

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tx = lid.x;
  let ty = lid.y;
  let t = ty * 16u + tx;
  let q0 = wg.x * 64u;
  for (var k = 0u; k < 8u; k++) {  // 2048 words, 8 per thread
    let w = t * 8u + k;
    let q = q0 + w / 32u;
    qs[w] = select(0u, query[min(q, params.num_query - 1u) * 32u + w % 32u], q < params.num_query);
  }
  var best: array<u32, 4>;
  var second: array<u32, 4>;
  var idx: array<u32, 4>;
  for (var i = 0u; i < 4u; i++) { best[i] = 0u; second[i] = 0u; idx[i] = NONE; }

  for (var base = 0u; base < params.num_ref; base += 64u) {
    workgroupBarrier();
    for (var k = 0u; k < 8u; k++) {
      let w = t * 8u + k;              // word w % 32 of reference w / 32
      let r = base + w / 32u;
      rs[(w % 32u) * 64u + w / 32u] = select(0u, refs[min(r, params.num_ref - 1u) * 32u + w % 32u], r < params.num_ref);
    }
    workgroupBarrier();
    var acc: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0u; }
    for (var k = 0u; k < 32u; k++) {
      let a0 = qs[(ty * 4u + 0u) * 32u + k]; let a1 = qs[(ty * 4u + 1u) * 32u + k];
      let a2 = qs[(ty * 4u + 2u) * 32u + k]; let a3 = qs[(ty * 4u + 3u) * 32u + k];
      let b0 = rs[k * 64u + tx * 4u + 0u]; let b1 = rs[k * 64u + tx * 4u + 1u];
      let b2 = rs[k * 64u + tx * 4u + 2u]; let b3 = rs[k * 64u + tx * 4u + 3u];
      acc[0] += dot4(a0, b0); acc[1] += dot4(a0, b1); acc[2] += dot4(a0, b2); acc[3] += dot4(a0, b3);
      acc[4] += dot4(a1, b0); acc[5] += dot4(a1, b1); acc[6] += dot4(a1, b2); acc[7] += dot4(a1, b3);
      acc[8] += dot4(a2, b0); acc[9] += dot4(a2, b1); acc[10] += dot4(a2, b2); acc[11] += dot4(a2, b3);
      acc[12] += dot4(a3, b0); acc[13] += dot4(a3, b1); acc[14] += dot4(a3, b2); acc[15] += dot4(a3, b3);
    }
    for (var i = 0u; i < 4u; i++) {
      for (var j = 0u; j < 4u; j++) {
        let r = base + tx * 4u + j;
        if (r >= params.num_ref) { continue; }
        let s = acc[i * 4u + j];
        if (s > best[i]) { second[i] = best[i]; best[i] = s; idx[i] = r; }
        else if (s > second[i]) { second[i] = s; }
      }
    }
  }

  // merge the 16 threads of each query row: qs / rs reused as (best, second, idx) per thread
  workgroupBarrier();
  for (var i = 0u; i < 4u; i++) {
    let slot = ((ty * 4u + i) * 16u + tx);  // 64 queries x 16 threads = 1024 slots
    qs[slot] = best[i];
    qs[1024u + slot] = second[i];
    rs[slot] = idx[i];
  }
  workgroupBarrier();
  if (tx == 0u) {
    for (var i = 0u; i < 4u; i++) {
      let ql = ty * 4u + i;
      let q = q0 + ql;
      if (q >= params.num_query) { continue; }
      var bt = 0u; var bb = 0u; var bi = NONE;
      for (var u = 0u; u < 16u; u++) {  // best: largest value, then smallest index
        let v = qs[ql * 16u + u]; let id = rs[ql * 16u + u];
        if (id != NONE && (v > bb || (v == bb && id < bi))) { bb = v; bi = id; bt = u; }
      }
      var sec = 0u;
      for (var u = 0u; u < 16u; u++) {
        let v = select(qs[ql * 16u + u], qs[1024u + ql * 16u + u], u == bt);
        if (rs[ql * 16u + u] != NONE || u == bt) { sec = max(sec, v); }
      }
      result[q] = vec4<u32>(bi, bb, sec, 0u);
    }
  }
}
