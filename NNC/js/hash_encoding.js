// hash_encoding.js
// Instant-NGP multi-scale hash encoding for 2D image UV coordinates.
import { FP_SCALE } from './webgpu.js';

// Uniform struct layout (160 bytes):
//   [0..7]   8 scalars: width, height, L, F_, LF, minRes(f32), maxRes(f32), _pad
//   [8..23]  levelT[16]      packed as array<vec4<u32>, 4>  (offset 32)
//   [24..39] levelOffset[16] packed as array<vec4<u32>, 4>  (offset 96)
// WGSL requires array element stride ≥16 in uniform → use vec4<u32> groups.

const HASH_STRUCT = `
struct HashUniforms {
    width: u32, height: u32,
    L: u32, F_: u32, LF: u32,
    minRes: f32, maxRes: f32,
    _pad: u32,
    levelT:      array<vec4<u32>, 4>,
    levelOffset: array<vec4<u32>, 4>,
}
fn level_T(l: u32)      -> u32 { return uni.levelT[l >> 2u][l & 3u]; }
fn level_off(l: u32)    -> u32 { return uni.levelOffset[l >> 2u][l & 3u]; }
fn hash2d(x: u32, y: u32, T: u32) -> u32 { return (x ^ (y * 2654435761u)) % T; }
`;

const HASH_FORWARD_SHADER = `
${HASH_STRUCT}
@group(0) @binding(0) var<uniform>             uni:          HashUniforms;
@group(0) @binding(1) var<storage, read>       hash_tables:  array<f32>;
@group(0) @binding(2) var<storage, read_write> features_out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= uni.width || py >= uni.height) { return; }
    let p   = py * uni.width + px;
    let uvx = f32(px) / f32(uni.width  - 1u);
    let uvy = f32(py) / f32(uni.height - 1u);
    let Lm1 = max(uni.L - 1u, 1u);
    for (var l = 0u; l < uni.L; l++) {
        let res_f = uni.minRes * pow(uni.maxRes / uni.minRes, f32(l) / f32(Lm1));
        let r  = max(u32(res_f), 2u);
        let sx = uvx * f32(r - 1u);
        let sy = uvy * f32(r - 1u);
        let x0 = u32(sx); let y0 = u32(sy);
        let x1 = min(x0 + 1u, r - 1u);
        let y1 = min(y0 + 1u, r - 1u);
        let tx = sx - f32(x0); let ty = sy - f32(y0);
        let w00 = (1.0-tx)*(1.0-ty); let w10 = tx*(1.0-ty);
        let w01 = (1.0-tx)*ty;       let w11 = tx*ty;
        let tl   = level_T(l);
        let h00  = hash2d(x0, y0, tl); let h10 = hash2d(x1, y0, tl);
        let h01  = hash2d(x0, y1, tl); let h11 = hash2d(x1, y1, tl);
        let base = level_off(l) * uni.F_;
        for (var f = 0u; f < uni.F_; f++) {
            let v = w00 * hash_tables[base + h00*uni.F_+f]
                  + w10 * hash_tables[base + h10*uni.F_+f]
                  + w01 * hash_tables[base + h01*uni.F_+f]
                  + w11 * hash_tables[base + h11*uni.F_+f];
            features_out[p * uni.LF + l * uni.F_ + f] = v;
        }
    }
}`;

const HASH_BACKWARD_SHADER = (fpScale) => `
const FPS: f32 = ${fpScale}.0;
${HASH_STRUCT}
@group(0) @binding(0) var<uniform>             uni:          HashUniforms;
@group(0) @binding(1) var<storage, read>       grad_features: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_hash:     array<atomic<i32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= uni.width || py >= uni.height) { return; }
    let p   = py * uni.width + px;
    let uvx = f32(px) / f32(uni.width  - 1u);
    let uvy = f32(py) / f32(uni.height - 1u);
    let Lm1 = max(uni.L - 1u, 1u);
    for (var l = 0u; l < uni.L; l++) {
        let res_f = uni.minRes * pow(uni.maxRes / uni.minRes, f32(l) / f32(Lm1));
        let r  = max(u32(res_f), 2u);
        let sx = uvx * f32(r - 1u);
        let sy = uvy * f32(r - 1u);
        let x0 = u32(sx); let y0 = u32(sy);
        let x1 = min(x0 + 1u, r - 1u);
        let y1 = min(y0 + 1u, r - 1u);
        let tx = sx - f32(x0); let ty = sy - f32(y0);
        let w00 = (1.0-tx)*(1.0-ty); let w10 = tx*(1.0-ty);
        let w01 = (1.0-tx)*ty;       let w11 = tx*ty;
        let tl   = level_T(l);
        let h00  = hash2d(x0, y0, tl); let h10 = hash2d(x1, y0, tl);
        let h01  = hash2d(x0, y1, tl); let h11 = hash2d(x1, y1, tl);
        let base = level_off(l) * uni.F_;
        for (var f = 0u; f < uni.F_; f++) {
            let g = grad_features[p * uni.LF + l * uni.F_ + f];
            atomicAdd(&grad_hash[base + h00*uni.F_+f], i32(g * w00 * FPS));
            atomicAdd(&grad_hash[base + h10*uni.F_+f], i32(g * w10 * FPS));
            atomicAdd(&grad_hash[base + h01*uni.F_+f], i32(g * w01 * FPS));
            atomicAdd(&grad_hash[base + h11*uni.F_+f], i32(g * w11 * FPS));
        }
    }
}`;

const HASH_ADAM_SHADER = `
struct AdamUniforms {
    lr: f32, beta1: f32, beta2: f32, epsilon: f32,
    t: u32, size: u32, fp_scale: f32, l2_lambda: f32,
    do_clamp: u32, sampled_count: u32,
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
    let bc1 = 1.0 - pow(adam.beta1, f32(adam.t));
    let bc2 = 1.0 - pow(adam.beta2, f32(adam.t));
    m[k] = adam.beta1 * m[k] + (1.0 - adam.beta1) * g;
    v[k] = adam.beta2 * v[k] + (1.0 - adam.beta2) * g * g;
    param[k] -= adam.lr * (m[k] / bc1) / (sqrt(v[k] / bc2) + adam.epsilon);
}`;

// Compute per-level T and offsets: T_l = min(T_max, res_l^2).
// Returns { levelT, levelOffset, totalEntries }.
export function computeHashLevels(L, T, minRes, maxRes) {
    const Lm1 = Math.max(L - 1, 1);
    const levelT = [], levelOffset = [];
    let totalEntries = 0;
    for (let l = 0; l < L; l++) {
        const res = Math.max(Math.round(minRes * Math.pow(maxRes / minRes, l / Lm1)), 2);
        const tl  = Math.min(T, res * res);
        levelOffset.push(totalEntries);
        levelT.push(tl);
        totalEntries += tl;
    }
    return { levelT, levelOffset, totalEntries };
}

export function hashLevelsFromConfig(config) {
    const { ngpNumLevels: L, ngpHashTableSize: T, gW, gH } = config;
    return computeHashLevels(L, T, 4, Math.max(gW || 16, gH || 16) * 2);
}

export function createHashEncoding(ctx, W, H, config) {
    const L      = config.ngpNumLevels       || 8;
    const T      = config.ngpHashTableSize   || 512;
    const F      = config.ngpFeaturesPerLevel || 2;
    const LF     = L * F;
    const { device } = ctx;
    const minRes = 4, maxRes = Math.max(config.gW || 16, config.gH || 16) * 2;

    const { levelT, levelOffset, totalEntries } = hashLevelsFromConfig(config);
    const totalParams = totalEntries * F;

    const initData   = new Float32Array(totalParams).map(() => Math.random() * 2 - 1);
    const hashTables = ctx.createBuffer(initData, undefined, 'ngp/tables');
    const hashM      = ctx.zeroBuffer(totalParams, undefined, 'ngp/m');
    const hashV      = ctx.zeroBuffer(totalParams, undefined, 'ngp/v');
    const gradHash   = ctx.zeroBuffer(totalParams, undefined, 'ngp/grad');

    const featuresOut     = device.createBuffer({ size: W * H * LF * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: 'ngp/features_out' });
    const gradFeaturesOut = device.createBuffer({ size: W * H * LF * 4, usage: GPUBufferUsage.STORAGE, label: 'ngp/grad_features' });

    // Uniform buffer: 160 bytes (see layout comment at top of file)
    const hashUniforms = ctx.uniformBuffer(160, 'ngp/hash_uni');
    const huAB = new ArrayBuffer(160);
    const huU  = new Uint32Array(huAB);
    const huF  = new Float32Array(huAB);
    huU[0] = W; huU[1] = H; huU[2] = L; huU[3] = F; huU[4] = LF;
    huF[5] = minRes; huF[6] = maxRes; // huU[7] = _pad (zero)
    for (let l = 0; l < 16; l++) {
        huU[8  + l] = l < L ? levelT[l]      : 0;
        huU[24 + l] = l < L ? levelOffset[l] : 0;
    }
    ctx.writeBuffer(hashUniforms, huAB);

    // Adam uniforms
    const adamUniforms = ctx.uniformBuffer(10 * 4, 'ngp/adam_uni');
    const adamAB = new ArrayBuffer(10 * 4);
    const adamF  = new Float32Array(adamAB);
    const adamU  = new Uint32Array(adamAB);

    const makePL = code => device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    const makeBG = (pl, bufs) => device.createBindGroup({
        layout: pl.getBindGroupLayout(0),
        entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    const fwdPipeline  = makePL(HASH_FORWARD_SHADER);
    const bwdPipeline  = makePL(HASH_BACKWARD_SHADER(FP_SCALE));
    const adamPipeline = makePL(HASH_ADAM_SHADER);

    const fwdBG  = makeBG(fwdPipeline,  [hashUniforms, hashTables, featuresOut]);
    const bwdBG  = makeBG(bwdPipeline,  [hashUniforms, gradFeaturesOut, gradHash]);
    const adamBG = makeBG(adamPipeline, [adamUniforms, gradHash, hashTables, hashM, hashV]);

    return {
        featuresOut,
        gradFeaturesOut,
        gradHash,
        hashTables,
        LF,
        totalParams,

        prepareAdamStep(lr, t, sampledCount) {
            adamF[0] = lr; adamF[1] = 0.9; adamF[2] = 0.999; adamF[3] = 1e-8;
            adamU[4] = t; adamU[5] = totalParams; adamF[6] = FP_SCALE;
            adamF[7] = 0; adamU[8] = 0; adamU[9] = sampledCount;
            ctx.writeBuffer(adamUniforms, adamAB);
        },

        dispatchForward(pass) {
            pass.setPipeline(fwdPipeline);
            pass.setBindGroup(0, fwdBG);
            pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
        },

        dispatchBackward(pass) {
            pass.setPipeline(bwdPipeline);
            pass.setBindGroup(0, bwdBG);
            pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
        },

        dispatchAdam(pass) {
            pass.setPipeline(adamPipeline);
            pass.setBindGroup(0, adamBG);
            pass.dispatchWorkgroups(Math.ceil(totalParams / 64));
        },

        destroy() {
            hashTables.destroy(); hashM.destroy(); hashV.destroy(); gradHash.destroy();
            featuresOut.destroy(); gradFeaturesOut.destroy();
            hashUniforms.destroy(); adamUniforms.destroy();
        },
    };
}
