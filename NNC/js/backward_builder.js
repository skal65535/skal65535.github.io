// backward_builder.js
// Generates WGSL shaders for GPU backward pass + Adam update.
// Fixed-point accumulation: gradients stored as atomic<i32> scaled by FP_SCALE.

export const FP_SCALE = 1 << 14; // 16384 — safe for all expected gradient magnitudes

// Common uniform block shared by all backward shaders (group 0, binding 0)
const BWD_UNIFORMS = `
struct BwdUniforms {
    width:         u32,
    height:        u32,
    stride:        u32,
    sampled_count: u32,
}
@group(0) @binding(0) var<uniform> uni: BwdUniforms;
`;

export function buildBackwardShaders(config) {
    const { gridSize, embeddingChannels: embCh, mlpWidth, smoothInterpolation } = config;
    return {
        gradOutput:      gradOutputShader(),
        gradL3:          gradL3Shader(mlpWidth),
        gradL2:          gradL2Shader(mlpWidth),
        gradL1:          gradL1Shader(gridSize, embCh, mlpWidth, smoothInterpolation),
        adamStep:        adamStepShader(),
        packEmbeddings:  packEmbeddingsShader(embCh),
    };
}

// ---------------------------------------------------------------------------
// Shader 1: compute grad_final from (out_final - target) per pixel
// ---------------------------------------------------------------------------
function gradOutputShader() {
    return `${BWD_UNIFORMS}
@group(0) @binding(1) var<storage, read>       out_final:  array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:        array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_final: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= uni.width * uni.height) { return; }
    if (uni.stride > 1u && p % uni.stride != 0u) {
        grad_final[p*4u]   = 0.0; grad_final[p*4u+1u] = 0.0;
        grad_final[p*4u+2u] = 0.0; grad_final[p*4u+3u] = 0.0;
        return;
    }
    grad_final[p*4u]   = 2.0 * (out_final[p*4u]   - tgt[p*4u]);
    grad_final[p*4u+1u] = 2.0 * (out_final[p*4u+1u] - tgt[p*4u+1u]);
    grad_final[p*4u+2u] = 2.0 * (out_final[p*4u+2u] - tgt[p*4u+2u]);
    grad_final[p*4u+3u] = 2.0 * (out_final[p*4u+3u] - tgt[p*4u+3u]);
}
`;
}

// ---------------------------------------------------------------------------
// Shader 2: backprop through Layer 3 (linear: inter2_activated -> output)
// Reads pre-activation inter_layer2; input to L3 is sin(inter_layer2).
// ---------------------------------------------------------------------------
function gradL3Shader(mlpWidth) {
    return `${BWD_UNIFORMS}
@group(0) @binding(1) var<storage, read>       grad_final:         array<f32>;
@group(0) @binding(2) var<storage, read>       inter_layer2:       array<f32>;
@group(0) @binding(3) var<storage, read>       layer3_weights:     array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_inter2_preact: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_l3_weights:    array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> grad_l3_biases:     array<atomic<i32>>;

const MW: u32 = ${mlpWidth}u;
const FPS: f32 = ${FP_SCALE}.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= uni.width * uni.height || (uni.stride > 1u && p % uni.stride != 0u)) { return; }

    let gf = vec4<f32>(grad_final[p*4u], grad_final[p*4u+1u], grad_final[p*4u+2u], grad_final[p*4u+3u]);
    for (var i = 0u; i < 4u; i++) { atomicAdd(&grad_l3_biases[i], i32(gf[i] * FPS)); }

    // g2[j] = W3^T * gf  via dot over the 4 output channels
    for (var j = 0u; j < MW; j++) {
        let col = vec4<f32>(layer3_weights[0u*MW+j], layer3_weights[1u*MW+j], layer3_weights[2u*MW+j], layer3_weights[3u*MW+j]);
        grad_inter2_preact[p*MW+j] = dot(gf, col);
    }
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < MW; j++) {
            atomicAdd(&grad_l3_weights[i*MW+j], i32(gf[i] * sin(inter_layer2[p*MW+j]) * FPS));
        }
    }
}
`;
}

// ---------------------------------------------------------------------------
// Shader 3: backprop through Layer 2's sin activation + linear layer
// ---------------------------------------------------------------------------
function gradL2Shader(mlpWidth) {
    return `${BWD_UNIFORMS}
@group(0) @binding(1) var<storage, read>       grad_inter2_preact: array<f32>;
@group(0) @binding(2) var<storage, read>       inter_layer2:       array<f32>;
@group(0) @binding(3) var<storage, read>       inter_layer1:       array<f32>;
@group(0) @binding(4) var<storage, read>       layer2_weights:     array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_inter1_preact: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_l2_weights:    array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> grad_l2_biases:     array<atomic<i32>>;

const MW: u32 = ${mlpWidth}u;
const FPS: f32 = ${FP_SCALE}.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= uni.width * uni.height || (uni.stride > 1u && p % uni.stride != 0u)) { return; }

    var g2: array<f32, ${mlpWidth}>;
    for (var i = 0u; i < MW; i++) {
        g2[i] = grad_inter2_preact[p*MW+i] * cos(inter_layer2[p*MW+i]);
        atomicAdd(&grad_l2_biases[i], i32(g2[i] * FPS));
    }

    // Pack g2 into vec4 chunks for dot() below
    var gv: array<vec4<f32>, ${mlpWidth >> 2}>;
    for (var k = 0u; k < ${mlpWidth >> 2}u; k++) {
        let b = k * 4u;
        gv[k] = vec4<f32>(g2[b], g2[b+1u], g2[b+2u], g2[b+3u]);
    }

    // g1[j] = W2^T * g2  via dot over vec4 chunks of g2
    for (var j = 0u; j < MW; j++) {
        var acc = 0.0;
        for (var k = 0u; k < ${mlpWidth >> 2}u; k++) {
            let b = k * 4u;
            let col = vec4<f32>(layer2_weights[b*MW+j], layer2_weights[(b+1u)*MW+j], layer2_weights[(b+2u)*MW+j], layer2_weights[(b+3u)*MW+j]);
            acc += dot(gv[k], col);
        }
        grad_inter1_preact[p*MW+j] = acc;
    }
    for (var i = 0u; i < MW; i++) {
        for (var j = 0u; j < MW; j++) {
            atomicAdd(&grad_l2_weights[i*MW+j], i32(g2[i] * sin(inter_layer1[p*MW+j]) * FPS));
        }
    }
}
`;
}

// ---------------------------------------------------------------------------
// Shader 4: backprop through Layer 1's sin activation + linear layer
//           + bilinear/smoothstep scatter to embedding gradients
// Phase 2: 8×8 spatial workgroup so adjacent pixels map to nearby grid cells,
//           reducing atomic contention on grad_embeddings.
// ---------------------------------------------------------------------------
function gradL1Shader(gridSize, embCh, mlpWidth, smoothInterp) {
    const smoothCode = smoothInterp ? `
    tx = tx*tx*(3.0 - 2.0*tx);
    ty = ty*ty*(3.0 - 2.0*ty);` : '';

    return `${BWD_UNIFORMS}
@group(0) @binding(1) var<storage, read>       grad_inter1_preact: array<f32>;
@group(0) @binding(2) var<storage, read>       inter_layer1:       array<f32>;
@group(0) @binding(3) var<storage, read>       layer1_weights:     array<f32>;
@group(0) @binding(4) var<storage, read>       embeddings:         array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_l1_weights:    array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> grad_l1_biases:     array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> grad_embeddings:    array<atomic<i32>>;

const MW:  u32 = ${mlpWidth}u;
const EC:  u32 = ${embCh}u;
const GS:  u32 = ${gridSize}u;
const FPS: f32 = ${FP_SCALE}.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px_x = gid.x;
    let px_y = gid.y;
    if (px_x >= uni.width || px_y >= uni.height) { return; }
    let p = px_y * uni.width + px_x;
    if (uni.stride > 1u && p % uni.stride != 0u) { return; }

    let sx = f32(px_x) / f32(uni.width  - 1u) * f32(GS - 1u);
    let sy = f32(px_y) / f32(uni.height - 1u) * f32(GS - 1u);
    let x0 = u32(sx); let y0 = u32(sy);
    let x1 = min(x0 + 1u, GS - 1u);
    let y1 = min(y0 + 1u, GS - 1u);
    var tx = sx - f32(x0);
    var ty = sy - f32(y0);
    ${smoothCode}
    let w00 = (1.0-tx)*(1.0-ty);
    let w10 = tx*(1.0-ty);
    let w01 = (1.0-tx)*ty;
    let w11 = tx*ty;
    let idx00 = (y0*GS+x0)*EC;
    let idx10 = (y0*GS+x1)*EC;
    let idx01 = (y1*GS+x0)*EC;
    let idx11 = (y1*GS+x1)*EC;

    let bilinear_w = vec4<f32>(w00, w10, w01, w11);
    var interp: array<f32, ${embCh}>;
    for (var j = 0u; j < EC; j++) {
        interp[j] = dot(bilinear_w, vec4<f32>(embeddings[idx00+j], embeddings[idx10+j], embeddings[idx01+j], embeddings[idx11+j]));
    }

    var g: array<f32, ${mlpWidth}>;
    for (var i = 0u; i < MW; i++) {
        g[i] = grad_inter1_preact[p*MW+i] * cos(inter_layer1[p*MW+i]);
        atomicAdd(&grad_l1_biases[i], i32(g[i] * FPS));
        for (var j = 0u; j < EC; j++) {
            atomicAdd(&grad_l1_weights[i*EC+j], i32(g[i] * interp[j] * FPS));
        }
    }

    // ge[j] = W1^T * g  via dot over vec4 chunks; then scatter to embedding corners
    var gv: array<vec4<f32>, ${mlpWidth >> 2}>;
    for (var k = 0u; k < ${mlpWidth >> 2}u; k++) {
        let b = k * 4u;
        gv[k] = vec4<f32>(g[b], g[b+1u], g[b+2u], g[b+3u]);
    }
    for (var j = 0u; j < EC; j++) {
        var ge = 0.0;
        for (var k = 0u; k < ${mlpWidth >> 2}u; k++) {
            let b = k * 4u;
            let col = vec4<f32>(layer1_weights[b*EC+j], layer1_weights[(b+1u)*EC+j], layer1_weights[(b+2u)*EC+j], layer1_weights[(b+3u)*EC+j]);
            ge += dot(gv[k], col);
        }
        atomicAdd(&grad_embeddings[idx00+j], i32(w00 * ge * FPS));
        atomicAdd(&grad_embeddings[idx10+j], i32(w10 * ge * FPS));
        atomicAdd(&grad_embeddings[idx01+j], i32(w01 * ge * FPS));
        atomicAdd(&grad_embeddings[idx11+j], i32(w11 * ge * FPS));
    }
}
`;
}

// ---------------------------------------------------------------------------
// Shader 5: Adam update — one dispatch per parameter tensor.
// Reads fixed-point grad from atomic<i32> buffer, updates param in-place.
// l2_lambda > 0 adds L2 regularisation: g += l2_lambda * param[k]
// do_clamp == 1 clamps output to [-1, 1] (for embeddings).
// ---------------------------------------------------------------------------
function adamStepShader() {
    return `
struct AdamUniforms {
    lr:        f32,
    beta1:     f32,
    beta2:     f32,
    epsilon:   f32,
    t:         u32,
    size:      u32,
    fp_scale:  f32,
    l2_lambda: f32,
    do_clamp:      u32,
    sampled_count: u32,
}
@group(0) @binding(0) var<uniform>             adam:  AdamUniforms;
@group(0) @binding(1) var<storage, read_write> grad:  array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> param: array<f32>;
@group(0) @binding(3) var<storage, read_write> m:     array<f32>;
@group(0) @binding(4) var<storage, read_write> v:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= adam.size) { return; }

    var g = f32(atomicLoad(&grad[k])) / (adam.fp_scale * f32(adam.sampled_count));
    g += adam.l2_lambda * param[k];

    let bc1 = 1.0 - pow(adam.beta1, f32(adam.t));
    let bc2 = 1.0 - pow(adam.beta2, f32(adam.t));
    m[k] = adam.beta1 * m[k] + (1.0 - adam.beta1) * g;
    v[k] = adam.beta2 * v[k] + (1.0 - adam.beta2) * g * g;
    var w = param[k] - adam.lr * (m[k] / bc1) / (sqrt(v[k] / bc2) + adam.epsilon);
    if (adam.do_clamp == 1u) { w = clamp(w, -1.0, 1.0); }
    param[k] = w;
}
`;
}

// ---------------------------------------------------------------------------
// Shader 6: pack f32 embeddings → u32 (4 channels per u32 via pack4x8snorm).
// Applies per-channel range scaling so the full int8 range maps to [mn, mx].
// Run after each Adam step for embeddings; forward shader reads packed buffer.
// ---------------------------------------------------------------------------
function packEmbeddingsShader(embCh) {
    const numPlanes = embCh / 4;
    return `
@group(0) @binding(0) var<storage, read>       src:       array<f32>;
@group(0) @binding(1) var<storage, read_write>  dst:       array<u32>;
@group(0) @binding(2) var<storage, read>        emb_range: array<f32>;

const NP: u32 = ${numPlanes}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= arrayLength(&dst)) { return; }
    let i    = k * 4u;
    let base = (k % NP) * 4u;
    let mn   = vec4<f32>(emb_range[(base)*2u],   emb_range[(base+1u)*2u],   emb_range[(base+2u)*2u],   emb_range[(base+3u)*2u]);
    let mx   = vec4<f32>(emb_range[(base)*2u+1u], emb_range[(base+1u)*2u+1u], emb_range[(base+2u)*2u+1u], emb_range[(base+3u)*2u+1u]);
    let span = mx - mn;
    let safe = select(span, vec4<f32>(2.0), span < vec4<f32>(1e-9));
    let v    = vec4<f32>(src[i], src[i+1u], src[i+2u], src[i+3u]);
    let s    = clamp(2.0 * (v - mn) / safe - vec4<f32>(1.0), vec4<f32>(-1.0), vec4<f32>(1.0));
    dst[k]   = pack4x8snorm(s);
}
`;
}
