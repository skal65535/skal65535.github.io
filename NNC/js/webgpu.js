// webgpu.js
// Centralized WebGPU management and shader generation.

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
        default: // 'sin' / SIREN
            return `fn activ(x: f32) -> f32 { return sin(x); }
fn activ_prime(x: f32) -> f32 { return cos(x); }`;
    }
}

// --- WebGPU Manager ---

export async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
                 ?? await navigator.gpu.requestAdapter({ powerPreference: 'low-power' })
                 ?? await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice({ requiredFeatures: [] });
    if (!device) {
        throw new Error("No appropriate GPUDevice found.");
    }

    return {
        device,
        createBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label = '') {
            const buffer = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true, label });
            new Float32Array(buffer.getMappedRange()).set(data);
            buffer.unmap();
            return buffer;
        },
        zeroBuffer(count, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label = '') {
            return device.createBuffer({ size: count * 4, usage, label });
        },
        uniformBuffer(size, label = '') {
            return device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
        },
        storageBuffer(size, label = '') {
            return device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, label });
        },
        outputBuffer(size, label = '') {
            return device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label });
        },
        readbackBuffer(size, label = '') {
            return device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label });
        },
        async readBackBuffers(bufMap) {
            const rbBufs = {};
            const ce = device.createCommandEncoder({ label: 'readBackBuffers' });
            for (const [k, { buf, size }] of Object.entries(bufMap)) {
                rbBufs[k] = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                ce.copyBufferToBuffer(buf, 0, rbBufs[k], 0, size * 4);
            }
            device.queue.submit([ce.finish()]);
            await Promise.all(Object.values(rbBufs).map(b => b.mapAsync(GPUMapMode.READ)));
            const result = {};
            for (const [k, b] of Object.entries(rbBufs)) {
                result[k] = new Float32Array(b.getMappedRange()).slice();
                b.unmap();
                b.destroy();
            }
            return result;
        },
        writeBuffer(buf, data) {
            device.queue.writeBuffer(buf, 0, data);
        },
        writeBufferAt(buf, byteOffset, data) {
            device.queue.writeBuffer(buf, byteOffset, data);
        },
        clearBuffer(buf) {
            const ce = device.createCommandEncoder();
            ce.clearBuffer(buf, 0, buf.size);
            device.queue.submit([ce.finish()]);
        },
        uploadModelWeights(model, tensors) {
            device.queue.writeBuffer(model.embeddings,     0, tensors.embeddings);
            device.queue.writeBuffer(model.layer1.weights, 0, tensors.layer1_weights);
            device.queue.writeBuffer(model.layer1.biases,  0, tensors.layer1_biases);
            device.queue.writeBuffer(model.layer2.weights, 0, tensors.layer2_weights);
            device.queue.writeBuffer(model.layer2.biases,  0, tensors.layer2_biases);
            device.queue.writeBuffer(model.layer3.weights, 0, tensors.layer3_weights);
            device.queue.writeBuffer(model.layer3.biases,  0, tensors.layer3_biases);
            if (model.mlp_weights) {
                const ly = model.mlpLayout;
                for (const [data, off] of [
                    [tensors.layer1_weights, ly.l1w], [tensors.layer1_biases,  ly.l1b],
                    [tensors.layer2_weights, ly.l2w], [tensors.layer2_biases,  ly.l2b],
                    [tensors.layer3_weights, ly.l3w], [tensors.layer3_biases,  ly.l3b],
                ]) device.queue.writeBuffer(model.mlp_weights, off * 4, data);
            }
        },
    };
}

// --- Forward Shader Builder ---

import { mlpWeightsLayout } from './model.js';

// ---------------------------------------------------------------------------
// --- Backward Shader Builders ---
// ---------------------------------------------------------------------------

export const FP_SCALE = 1 << 10;
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

const MW1: u32 = ${mlpWidth1}u;
const MW2: u32 = ${mlpWidth2}u;
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
    var gv: array<vec4<f32>, ${mlpWidth2 >> 2}>;
    for (var k = 0u; k < ${mlpWidth2 >> 2}u; k++) {
        let b = k * 4u;
        gv[k] = vec4<f32>(g2[b], g2[b+1u], g2[b+2u], g2[b+3u]);
    }
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
    let px_x = gid.x; let px_y = gid.y;
    if (px_x >= uni.width || px_y >= uni.height) { return; }
    let p = px_y * uni.width + px_x;
    if (uni.stride > 1u && p % uni.stride != 0u) { return; }
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
        let w00 = (1.0-tx)*(1.0-ty); let w10 = tx*(1.0-ty);
        let w01 = (1.0-tx)*ty; let w11 = tx*ty;
        let idx00 = (y0*GS+x0)*EC; let idx10 = (y0*GS+x1)*EC;
        let idx01 = (y1*GS+x0)*EC; let idx11 = (y1*GS+x1)*EC;
        for (var b = 0u; b < CPG; b++) {
            let j = grp * CPG + b;
            let interp_j = w00*embeddings[idx00+j] + w10*embeddings[idx10+j] + w01*embeddings[idx01+j] + w11*embeddings[idx11+j];
            for (var i = 0u; i < MW; i++) { atomicAdd(&grad_l1_weights[i*EC+j], i32(g[i] * interp_j * FPS)); }
            var ge = 0.0;
            for (var k = 0u; k < ${mlpWidth1 >> 2}u; k++) {
                let bk = k * 4u;
                let col = vec4<f32>(layer1_weights[bk*EC+j], layer1_weights[(bk+1u)*EC+j], layer1_weights[(bk+2u)*EC+j], layer1_weights[(bk+3u)*EC+j]);
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
    let grid_i = k / NG; let g = k % NG;
    var packed: u32 = 0u;
    for (var b = 0u; b < 8u; b++) {
        let c = g * 8u + b; let v = src[grid_i * EC + c];
        let mn = emb_range[c * 2u]; let mx = emb_range[c * 2u + 1u];
        let span = mx - mn;
        let s = select(0.0, clamp(2.0 * (v - mn) / span - 1.0, -1.0, 1.0), span >= 1e-9);
        let nib = u32(i32(clamp(round(s * 7.0), -7.0, 7.0)) & 0xF);
        packed |= nib << (b * 4u);
    }
    dst[k] = packed;
}
`;
    }
    return `${PACK_BINDINGS}
const NP: u32 = ${numU32}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= arrayLength(&dst)) { return; }
    let i = k * 4u; let base = (k % NP) * 4u;
    let mn = vec4<f32>(emb_range[(base)*2u], emb_range[(base+1u)*2u], emb_range[(base+2u)*2u], emb_range[(base+3u)*2u]);
    let mx = vec4<f32>(emb_range[(base)*2u+1u], emb_range[(base+1u)*2u+1u], emb_range[(base+2u)*2u+1u], emb_range[(base+3u)*2u+1u]);
    let span = mx - mn;
    let safe = select(span, vec4<f32>(2.0), span < vec4<f32>(1e-9));
    let v = vec4<f32>(src[i], src[i+1u], src[i+2u], src[i+3u]);
    let s = clamp(2.0 * (v - mn) / safe - vec4<f32>(1.0), vec4<f32>(-1.0), vec4<f32>(1.0));
    dst[k] = pack4x8snorm(s);
}
`;
}

// Re-implementing buildForwardShader to be exactly like the original but using wgslActivFns
export function buildShader(config) {
    const { gridSize, embeddingChannels, mlpWidth1, mlpWidth2, quantization, smoothInterpolation, activation = 'sin' } = config;
    const outCh = config.hasAlpha ? 4 : 3;
    const embBits = config.embBits || 8;
    const channelsPerU32 = 32 / embBits;
    const numU32 = embeddingChannels / channelsPerU32;
    const { l1w: L1W_OFF, l1b: L1B_OFF, l2w: L2W_OFF, l2b: L2B_OFF, l3w: L3W_OFF, l3b: L3B_OFF } = mlpWeightsLayout(config);

    const smoothCode = smoothInterpolation ? 'tx = tx*tx*(3.0-2.0*tx); ty = ty*ty*(3.0-2.0*ty);' : '';

    const interpolationFunctions = embBits === 8 ? `
fn sample_embedding_plane(uv: vec2<f32>, plane: u32) -> vec4<f32> {
    let off = uniforms.emb_offsets[plane >> 1u];
    let ox = select(off.x, off.z, (plane & 1u) == 1u);
    let oy = select(off.y, off.w, (plane & 1u) == 1u);
    let uvo = clamp(uv + vec2<f32>(ox, oy), vec2<f32>(0.0), vec2<f32>(1.0));
    let scaled = uvo * vec2<f32>(f32(uniforms.gridSize - 1u));
    let c      = floor(scaled);
    let x0 = u32(c.x); let y0 = u32(c.y);
    let x1 = min(x0 + 1u, uniforms.gridSize - 1u);
    let y1 = min(y0 + 1u, uniforms.gridSize - 1u);
    var tx = scaled.x - c.x;
    var ty = scaled.y - c.y;
    ${smoothCode}
    let gs = uniforms.gridSize;
    let c00 = unpack4x8snorm(embeddings_q[(y0*gs+x0)*${numU32}u+plane]);
    let c10 = unpack4x8snorm(embeddings_q[(y0*gs+x1)*${numU32}u+plane]);
    let c01 = unpack4x8snorm(embeddings_q[(y1*gs+x0)*${numU32}u+plane]);
    let c11 = unpack4x8snorm(embeddings_q[(y1*gs+x1)*${numU32}u+plane]);
    let interp = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
    let mn = uniforms.emb_range[plane * 2u];
    let mx = uniforms.emb_range[plane * 2u + 1u];
    return (interp + vec4<f32>(1.0)) * 0.5 * (mx - mn) + mn;
}
` : `
fn dequant4(n: u32) -> f32 { return f32(select(i32(n), i32(n) - 16, n >= 8u)) / 7.0; }
`;

    const makeDotLoop = inSize => quantization === 'qat8'
        ? `        for (var j: u32 = 0u; j < ${inSize}u; j = j + 4u) {
            let base = mat_base + i * ${inSize}u + j;
            let w = vec4<f32>(quantize_dequantize_8bit(mlp_weights[base]), quantize_dequantize_8bit(mlp_weights[base+1u]), quantize_dequantize_8bit(mlp_weights[base+2u]), quantize_dequantize_8bit(mlp_weights[base+3u]));
            let v = vec4<f32>(vec_in[j], vec_in[j+1u], vec_in[j+2u], vec_in[j+3u]);
            sum += dot(w, v);
        }`
        : `        for (var j: u32 = 0u; j < ${inSize}u; j = j + 4u) {
            let base = mat_base + i * ${inSize}u + j;
            let w = vec4<f32>(mlp_weights[base], mlp_weights[base+1u], mlp_weights[base+2u], mlp_weights[base+3u]);
            let v = vec4<f32>(vec_in[j], vec_in[j+1u], vec_in[j+2u], vec_in[j+3u]);
            sum += dot(w, v);
        }`;

    const matVecMulFn = (name, inSize, outSize) => `
fn ${name}(mat_base: u32, vec_in: array<f32, ${inSize}>) -> array<f32, ${outSize}> {
    var result: array<f32, ${outSize}>;
    for (var i: u32 = 0u; i < ${outSize}u; i = i + 1u) {
        var sum = 0.0;
${makeDotLoop(inSize)}
        result[i] = sum;
    }
    return result;
}`;

    return `
fn quantize_dequantize_8bit(val: f32) -> f32 {
    return round(clamp(val, -1.0, 1.0) * 127.0) / 127.0;
}
${interpolationFunctions}
${matVecMulFn('mat_vec_mul_l1', embeddingChannels, mlpWidth1)}
${matVecMulFn('mat_vec_mul_l2', mlpWidth1,         mlpWidth2)}
${matVecMulFn('mat_vec_mul_l3', mlpWidth2,         outCh)}
${wgslActivFns(activation)}

struct Uniforms {
    gridSize:          u32,
    embeddingChannels: u32,
    mlpWidth1:         u32,
    canvasWidth:       u32,
    canvasHeight:      u32,
    channelMask: u32, mlpWidth2: u32, _p2: u32,
    emb_range:   array<vec4<f32>, 8>,
    emb_offsets: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform>      uniforms:     Uniforms;
@group(0) @binding(1) var<storage, read> embeddings_q: array<u32>;
@group(0) @binding(2) var<storage, read> mlp_weights:  array<f32>;
@group(0) @binding(3) var<storage, read_write> out_inter_layer1: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_inter_layer2: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_final: array<f32>;

const L1W_OFF: u32 = ${L1W_OFF}u;
const L1B_OFF: u32 = ${L1B_OFF}u;
const L2W_OFF: u32 = ${L2W_OFF}u;
const L2B_OFF: u32 = ${L2B_OFF}u;
const L3W_OFF: u32 = ${L3W_OFF}u;
const L3B_OFF: u32 = ${L3B_OFF}u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x; let y = global_id.y;
    if (x >= uniforms.canvasWidth || y >= uniforms.canvasHeight) { return; }
    let uv = vec2<f32>(f32(x) / f32(uniforms.canvasWidth - 1u), f32(y) / f32(uniforms.canvasHeight - 1u));
    var embedding_vector: array<f32, ${embeddingChannels}>;
    ${embBits === 8 ? `
    for (var plane = 0u; plane < ${numU32}u; plane++) {
        let v = sample_embedding_plane(uv, plane);
        embedding_vector[plane*4u+0u] = v.x;
        embedding_vector[plane*4u+1u] = v.y;
        embedding_vector[plane*4u+2u] = v.z;
        embedding_vector[plane*4u+3u] = v.w;
    }` : `
    for (var grp = 0u; grp < ${numU32}u; grp++) {
        let off = uniforms.emb_offsets[grp >> 1u];
        let ox = select(off.x, off.z, (grp & 1u) == 1u);
        let oy = select(off.y, off.w, (grp & 1u) == 1u);
        let uvo = clamp(uv + vec2<f32>(ox, oy), vec2<f32>(0.0), vec2<f32>(1.0));
        let scaled = uvo * vec2<f32>(f32(uniforms.gridSize - 1u));
        let c = floor(scaled);
        let x0 = u32(c.x); let y0 = u32(c.y);
        let x1 = min(x0+1u, uniforms.gridSize-1u);
        let y1 = min(y0+1u, uniforms.gridSize-1u);
        var tx = scaled.x - c.x; var ty = scaled.y - c.y;
        ${smoothCode}
        let gs = uniforms.gridSize;
        let q00 = embeddings_q[(y0*gs+x0)*${numU32}u+grp];
        let q10 = embeddings_q[(y0*gs+x1)*${numU32}u+grp];
        let q01 = embeddings_q[(y1*gs+x0)*${numU32}u+grp];
        let q11 = embeddings_q[(y1*gs+x1)*${numU32}u+grp];
        for (var b = 0u; b < 8u; b++) {
            let ch = grp * 8u + b; let plane = ch / 4u; let comp = ch % 4u;
            let mn = uniforms.emb_range[plane * 2u][comp];
            let mx = uniforms.emb_range[plane * 2u + 1u][comp];
            let s = dequant4((q00 >> (b*4u)) & 0xFu)*(1.0-tx)*(1.0-ty)
                  + dequant4((q10 >> (b*4u)) & 0xFu)*tx*(1.0-ty)
                  + dequant4((q01 >> (b*4u)) & 0xFu)*(1.0-tx)*ty
                  + dequant4((q11 >> (b*4u)) & 0xFu)*tx*ty;
            embedding_vector[ch] = (s + 1.0) * 0.5 * (mx - mn) + mn;
        }
    }`}
    for (var ch = 0u; ch < ${embeddingChannels}u; ch++) {
        if ((uniforms.channelMask & (1u << ch)) == 0u) { embedding_vector[ch] = 0.0; }
    }
    let pixel_index = y * uniforms.canvasWidth + x;
    var layer1_out = mat_vec_mul_l1(L1W_OFF, embedding_vector);
    for (var i: u32 = 0u; i < ${mlpWidth1}u; i = i + 1u) {
        layer1_out[i] = layer1_out[i] + mlp_weights[L1B_OFF + i];
        out_inter_layer1[pixel_index * ${mlpWidth1}u + i] = layer1_out[i];
        layer1_out[i] = activ(layer1_out[i]);
    }
    var layer2_out = mat_vec_mul_l2(L2W_OFF, layer1_out);
    for (var i: u32 = 0u; i < ${mlpWidth2}u; i = i + 1u) {
        layer2_out[i] = layer2_out[i] + mlp_weights[L2B_OFF + i];
        out_inter_layer2[pixel_index * ${mlpWidth2}u + i] = layer2_out[i];
        layer2_out[i] = activ(layer2_out[i]);
    }
    var layer3_out = mat_vec_mul_l3(L3W_OFF, layer2_out);
    for (var i: u32 = 0u; i < ${outCh}u; i = i + 1u) { layer3_out[i] = layer3_out[i] + mlp_weights[L3B_OFF + i]; }
    out_final[pixel_index * 4 + 0] = clamp(layer3_out[0], 0.0, 1.0);
    out_final[pixel_index * 4 + 1] = clamp(layer3_out[1], 0.0, 1.0);
    out_final[pixel_index * 4 + 2] = clamp(layer3_out[2], 0.0, 1.0);
    out_final[pixel_index * 4 + 3] = ${outCh === 4 ? 'clamp(layer3_out[3], 0.0, 1.0)' : '1.0'};
}
`;
}
