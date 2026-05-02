// backward_builder.js
// Generates WGSL shaders for GPU backward pass + Adam update.
// Fixed-point accumulation: gradients stored as atomic<i32> scaled by FP_SCALE.

export const FP_SCALE = 1 << 10; // 1024 — prevents i32 overflow at 512×512 (262144*2*1024 ≈ 537M << INT32_MAX)

function wgslActivFns(activation) {
    switch (activation) {
        case 'tanh':
            return `fn activ(x: f32) -> f32 { return tanh(x); }
fn activ_prime(x: f32) -> f32 { let t = tanh(x); return 1.0 - t * t; }`;
        case 'softsign':
            return `fn activ(x: f32) -> f32 { return x / (1.0 + abs(x)); }
fn activ_prime(x: f32) -> f32 { let d = 1.0 + abs(x); return 1.0 / (d * d); }`;
        case 'none':
            return `fn activ(x: f32) -> f32 { return x; }
fn activ_prime(x: f32) -> f32 { return 1.0; }`;
        default:
            return `fn activ(x: f32) -> f32 { return sin(x); }
fn activ_prime(x: f32) -> f32 { return cos(x); }`;
    }
}

// Common uniform block shared by all backward shaders (group 0, binding 0)
const BWD_UNIFORMS = `
struct BwdUniforms {
    width:         u32,
    height:        u32,
    stride:        u32,
    sampled_count: u32,
    roi_strength:  f32,
}
@group(0) @binding(0) var<uniform> uni: BwdUniforms;
`;

export function buildBackwardShaders(config) {
    const { gridSize, embeddingChannels: embCh, mlpWidth1, mlpWidth2, smoothInterpolation, activation = 'sin' } = config;
    const embBits = config.embBits || 8;
    const outCh = config.hasAlpha ? 4 : 3;
    return {
        gradOutput:      gradOutputShader(outCh),
        gradL3:          gradL3Shader(mlpWidth2, activation, outCh),
        gradL2:          gradL2Shader(mlpWidth1, mlpWidth2, activation),
        gradL1:          gradL1Shader(gridSize, embCh, mlpWidth1, smoothInterpolation, embBits, activation),
        adamStep:        adamStepShader(),
        packEmbeddings:  packEmbeddingsShader(embCh, embBits),
    };
}

// ---------------------------------------------------------------------------
// Shader 1: compute grad_final from (out_final - target) per pixel
// ---------------------------------------------------------------------------
function gradOutputShader(outCh) {
    return `${BWD_UNIFORMS}
@group(0) @binding(1) var<storage, read>       out_final:  array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:        array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_final: array<f32>;
@group(0) @binding(4) var<storage, read>       roi_mask:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= uni.width * uni.height) { return; }
    if (uni.stride > 1u && p % uni.stride != 0u) {
        grad_final[p*4u]    = 0.0; grad_final[p*4u+1u] = 0.0;
        grad_final[p*4u+2u] = 0.0; grad_final[p*4u+3u] = 0.0;
        return;
    }
    let wt = 1.0 + roi_mask[p] * uni.roi_strength;
    grad_final[p*4u]    = wt * 2.0 * 0.299 * (out_final[p*4u]    - tgt[p*4u]);
    grad_final[p*4u+1u] = wt * 2.0 * 0.587 * (out_final[p*4u+1u] - tgt[p*4u+1u]);
    grad_final[p*4u+2u] = wt * 2.0 * 0.114 * (out_final[p*4u+2u] - tgt[p*4u+2u]);
    grad_final[p*4u+3u] = ${outCh === 4 ? 'wt * 2.0 * (out_final[p*4u+3u] - tgt[p*4u+3u])' : '0.0'};
}
`;
}

// ---------------------------------------------------------------------------
// Shader 2: backprop through Layer 3 (linear: inter2_activated -> output)
// Reads pre-activation inter_layer2; input to L3 is activ(inter_layer2).
// ---------------------------------------------------------------------------
function gradL3Shader(mlpWidth, activation, outCh) {
    const OC = outCh;
    const vecT = `vec${OC}<f32>`;
    const gfInit = Array.from({length: OC}, (_, k) => `grad_final[p*4u+${k}u]`).join(', ');
    const colInit = Array.from({length: OC}, (_, k) => `layer3_weights[${k}u*MW+j]`).join(', ');
    return `${BWD_UNIFORMS}
${wgslActivFns(activation)}
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

    let gf = ${vecT}(${gfInit});
    for (var i = 0u; i < ${OC}u; i++) { atomicAdd(&grad_l3_biases[i], i32(gf[i] * FPS)); }

    // g2[j] = W3^T * gf  via dot over the ${OC} output channels
    for (var j = 0u; j < MW; j++) {
        let col = ${vecT}(${colInit});
        grad_inter2_preact[p*MW+j] = dot(gf, col);
    }
    for (var i = 0u; i < ${OC}u; i++) {
        for (var j = 0u; j < MW; j++) {
            atomicAdd(&grad_l3_weights[i*MW+j], i32(gf[i] * activ(inter_layer2[p*MW+j]) * FPS));
        }
    }
}
`;
}

// ---------------------------------------------------------------------------
// Shader 3: backprop through Layer 2's activation + linear layer
// mlpWidth1 = W1 (inter1 width = L2 input), mlpWidth2 = W2 (inter2 width = L2 output)
// ---------------------------------------------------------------------------
function gradL2Shader(mlpWidth1, mlpWidth2, activation) {
    return `${BWD_UNIFORMS}
${wgslActivFns(activation)}
@group(0) @binding(1) var<storage, read>       grad_inter2_preact: array<f32>;
@group(0) @binding(2) var<storage, read>       inter_layer2:       array<f32>;
@group(0) @binding(3) var<storage, read>       inter_layer1:       array<f32>;
@group(0) @binding(4) var<storage, read>       layer2_weights:     array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_inter1_preact: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_l2_weights:    array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> grad_l2_biases:     array<atomic<i32>>;

const MW1: u32 = ${mlpWidth1}u; // inter1 / L2 input width
const MW2: u32 = ${mlpWidth2}u; // inter2 / L2 output width
const FPS: f32 = ${FP_SCALE}.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= uni.width * uni.height || (uni.stride > 1u && p % uni.stride != 0u)) { return; }

    var g2: array<f32, ${mlpWidth2}>;
    for (var i = 0u; i < MW2; i++) {
        g2[i] = grad_inter2_preact[p*MW2+i] * activ_prime(inter_layer2[p*MW2+i]);
        atomicAdd(&grad_l2_biases[i], i32(g2[i] * FPS));
    }

    // Pack g2 into vec4 chunks for dot() below
    var gv: array<vec4<f32>, ${mlpWidth2 >> 2}>;
    for (var k = 0u; k < ${mlpWidth2 >> 2}u; k++) {
        let b = k * 4u;
        gv[k] = vec4<f32>(g2[b], g2[b+1u], g2[b+2u], g2[b+3u]);
    }

    // g1[j] = W2^T * g2  via dot over vec4 chunks of g2
    // layer2_weights layout: row i (0..MW2), col j (0..MW1) → weights[i*MW1+j]
    for (var j = 0u; j < MW1; j++) {
        var acc = 0.0;
        for (var k = 0u; k < ${mlpWidth2 >> 2}u; k++) {
            let b = k * 4u;
            let col = vec4<f32>(layer2_weights[b*MW1+j], layer2_weights[(b+1u)*MW1+j], layer2_weights[(b+2u)*MW1+j], layer2_weights[(b+3u)*MW1+j]);
            acc += dot(gv[k], col);
        }
        grad_inter1_preact[p*MW1+j] = acc;
    }
    for (var i = 0u; i < MW2; i++) {
        for (var j = 0u; j < MW1; j++) {
            atomicAdd(&grad_l2_weights[i*MW1+j], i32(g2[i] * activ(inter_layer1[p*MW1+j]) * FPS));
        }
    }
}
`;
}

// ---------------------------------------------------------------------------
// Shader 4: backprop through Layer 1's activation + linear layer
//           + bilinear/smoothstep scatter to embedding gradients
// Phase 2: 8×8 spatial workgroup so adjacent pixels map to nearby grid cells,
//           reducing atomic contention on grad_embeddings.
// ---------------------------------------------------------------------------
function gradL1Shader(gridSize, embCh, mlpWidth1, smoothInterp, embBits, activation) {
    const channelsPerU32 = 32 / (embBits || 8);
    const numU32 = embCh / channelsPerU32;
    const smoothCode = smoothInterp ? `
    tx = tx*tx*(3.0 - 2.0*tx);
    ty = ty*ty*(3.0 - 2.0*ty);` : '';

    return `${BWD_UNIFORMS}
${wgslActivFns(activation)}
@group(0) @binding(1) var<storage, read>       grad_inter1_preact: array<f32>;
@group(0) @binding(2) var<storage, read>       inter_layer1:       array<f32>;
@group(0) @binding(3) var<storage, read>       layer1_weights:     array<f32>;
@group(0) @binding(4) var<storage, read>       embeddings:         array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_l1_weights:    array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> grad_l1_biases:     array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> grad_embeddings:    array<atomic<i32>>;
@group(0) @binding(8) var<storage, read>       emb_offsets:         array<f32>;

const MW:  u32 = ${mlpWidth1}u;
const EC:  u32 = ${embCh}u;
const GS:  u32 = ${gridSize}u;
const NU:  u32 = ${numU32}u;
const CPG: u32 = ${channelsPerU32}u;
const FPS: f32 = ${FP_SCALE}.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px_x = gid.x;
    let px_y = gid.y;
    if (px_x >= uni.width || px_y >= uni.height) { return; }
    let p = px_y * uni.width + px_x;
    if (uni.stride > 1u && p % uni.stride != 0u) { return; }

    // Compute g[] from upstream gradient — independent of embedding layout
    var g: array<f32, ${mlpWidth1}>;
    for (var i = 0u; i < MW; i++) {
        g[i] = grad_inter1_preact[p*MW+i] * activ_prime(inter_layer1[p*MW+i]);
        atomicAdd(&grad_l1_biases[i], i32(g[i] * FPS));
    }
    var gv: array<vec4<f32>, ${mlpWidth1 >> 2}>;
    for (var k = 0u; k < ${mlpWidth1 >> 2}u; k++) {
        let bk = k * 4u;
        gv[k] = vec4<f32>(g[bk], g[bk+1u], g[bk+2u], g[bk+3u]);
    }

    let uvx = f32(px_x) / f32(uni.width  - 1u);
    let uvy = f32(px_y) / f32(uni.height - 1u);

    // Per-u32 group: each has its own UV offset → distinct bilinear footprint
    for (var grp = 0u; grp < NU; grp++) {
        let ox = emb_offsets[grp * 2u];
        let oy = emb_offsets[grp * 2u + 1u];
        let sx = clamp(uvx + ox, 0.0, 1.0) * f32(GS - 1u);
        let sy = clamp(uvy + oy, 0.0, 1.0) * f32(GS - 1u);
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

        for (var b = 0u; b < CPG; b++) {
            let j = grp * CPG + b;
            let interp_j = w00*embeddings[idx00+j] + w10*embeddings[idx10+j]
                         + w01*embeddings[idx01+j] + w11*embeddings[idx11+j];
            for (var i = 0u; i < MW; i++) {
                atomicAdd(&grad_l1_weights[i*EC+j], i32(g[i] * interp_j * FPS));
            }
            var ge = 0.0;
            for (var k = 0u; k < ${mlpWidth1 >> 2}u; k++) {
                let bk = k * 4u;
                let col = vec4<f32>(layer1_weights[bk*EC+j], layer1_weights[(bk+1u)*EC+j],
                                    layer1_weights[(bk+2u)*EC+j], layer1_weights[(bk+3u)*EC+j]);
                ge += dot(gv[k], col);
            }
            atomicAdd(&grad_embeddings[idx00+j], i32(w00 * ge * FPS));
            atomicAdd(&grad_embeddings[idx10+j], i32(w10 * ge * FPS));
            atomicAdd(&grad_embeddings[idx01+j], i32(w01 * ge * FPS));
            atomicAdd(&grad_embeddings[idx11+j], i32(w11 * ge * FPS));
        }
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
const PACK_BINDINGS = `
@group(0) @binding(0) var<storage, read>       src:       array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:       array<u32>;
@group(0) @binding(2) var<storage, read>       emb_range: array<f32>;
`;

function packEmbeddingsShader(embCh, embBits) {
    const channelsPerU32 = 32 / (embBits || 8);
    const numU32 = embCh / channelsPerU32;
    if ((embBits || 8) === 4) {
        return `${PACK_BINDINGS}
const EC: u32 = ${embCh}u;
const NG: u32 = ${numU32}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= arrayLength(&dst)) { return; }
    let grid_i = k / NG;
    let g      = k % NG;
    var packed: u32 = 0u;
    for (var b = 0u; b < 8u; b++) {
        let c    = g * 8u + b;
        let v    = src[grid_i * EC + c];
        let mn   = emb_range[c * 2u];
        let mx   = emb_range[c * 2u + 1u];
        let span = mx - mn;
        let s    = select(0.0, clamp(2.0 * (v - mn) / span - 1.0, -1.0, 1.0), span >= 1e-9);
        let nib  = u32(i32(clamp(round(s * 7.0), -7.0, 7.0)) & 0xF);
        packed  |= nib << (b * 4u);
    }
    dst[k] = packed;
}
`;
    }
    // 8-bit path (default)
    return `${PACK_BINDINGS}
const NP: u32 = ${numU32}u;

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
