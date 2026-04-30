// shader_builder.js
// Generates the WGSL forward-pass compute shader for the embedding + MLP model.
function wgslActivFns(activation) {
    switch (activation) {
        case 'tanh':
            return `fn activ(x: f32) -> f32 { return tanh(x); }
fn activ_prime(x: f32) -> f32 { let t = tanh(x); return 1.0 - t * t; }`;
        case 'softsign':
            return `fn activ(x: f32) -> f32 { return x / (1.0 + abs(x)); }
fn activ_prime(x: f32) -> f32 { let d = 1.0 + abs(x); return 1.0 / (d * d); }`;
        case 'hardtanh':
            return `fn activ(x: f32) -> f32 { return clamp(x, -1.0, 1.0); }
fn activ_prime(x: f32) -> f32 { return f32(abs(x) < 1.0); }`;
        default:
            return `fn activ(x: f32) -> f32 { return sin(x); }
fn activ_prime(x: f32) -> f32 { return cos(x); }`;
    }
}

export function buildShader(config) {
    const { gridSize, embeddingChannels, mlpWidth, quantization, smoothInterpolation, activation = 'sin' } = config;
    const embBits = config.embBits || 8;
    const channelsPerU32 = 32 / embBits;  // 4 for 8-bit, 8 for 4-bit
    const numU32 = embeddingChannels / channelsPerU32;

    const offsets = config.embOffsets || new Float32Array(numU32 * 2);
    const offsetVals = Array.from(offsets).map(v => v.toFixed(6) + 'f').join(', ');
    const embOffsetConst = `const EMB_OFFSETS: array<f32, ${numU32 * 2}> = array<f32, ${numU32 * 2}>(${offsetVals});`;

    // keep numPlanes for 8-bit MLP/range indexing (always embCh/4)
    const numPlanes = embeddingChannels / 4;

    const quantizationFunctions = `
fn quantize_dequantize_8bit(val: f32) -> f32 {
    return round(clamp(val, -1.0, 1.0) * 127.0) / 127.0;
}
`;

    const smoothCode = smoothInterpolation
        ? 'tx = tx*tx*(3.0-2.0*tx); ty = ty*ty*(3.0-2.0*ty);'
        : '';

    const interpolationFunctions = embBits === 8 ? `
${embOffsetConst}
fn sample_embedding_plane(uv: vec2<f32>, plane: u32) -> vec4<f32> {
    let ox = EMB_OFFSETS[plane * 2u];
    let oy = EMB_OFFSETS[plane * 2u + 1u];
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
${embOffsetConst}
fn dequant4(n: u32) -> f32 { return f32(select(i32(n), i32(n) - 16, n >= 8u)) / 7.0; }
`;

    // Generate dot-product inner loop (width must be a multiple of 4)
    const dotLoop = quantization === 'qat8'
        ? `        for (var j: u32 = 0u; j < width; j = j + 4u) {
            let base = i * width + j;
            let w = vec4<f32>(quantize_dequantize_8bit((*mat)[base]), quantize_dequantize_8bit((*mat)[base+1u]), quantize_dequantize_8bit((*mat)[base+2u]), quantize_dequantize_8bit((*mat)[base+3u]));
            let v = vec4<f32>(vec_in[j], vec_in[j+1u], vec_in[j+2u], vec_in[j+3u]);
            sum += dot(w, v);
        }`
        : `        for (var j: u32 = 0u; j < width; j = j + 4u) {
            let base = i * width + j;
            let w = vec4<f32>((*mat)[base], (*mat)[base+1u], (*mat)[base+2u], (*mat)[base+3u]);
            let v = vec4<f32>(vec_in[j], vec_in[j+1u], vec_in[j+2u], vec_in[j+3u]);
            sum += dot(w, v);
        }`;

    const matVecMulFn = (name, inSize, outSize) => `
fn ${name}(mat: ptr<storage, array<f32>, read>, vec_in: array<f32, ${inSize}>, width: u32, height: u32) -> array<f32, ${outSize}> {
    var result: array<f32, ${outSize}>;
    for (var i: u32 = 0u; i < height; i = i + 1u) {
        var sum = 0.0;
${dotLoop}
        result[i] = sum;
    }
    return result;
}`;

    const mlpFunctions = `
${matVecMulFn('mat_vec_mul',        embeddingChannels, mlpWidth)}
${matVecMulFn('mat_vec_mul_hidden', mlpWidth,          mlpWidth)}
${matVecMulFn('mat_vec_mul_output', mlpWidth,          4)}
`;

    const shaderCode = `
${quantizationFunctions}
${interpolationFunctions}
${mlpFunctions}
${wgslActivFns(activation)}

struct Uniforms {
    gridSize:          u32,
    embeddingChannels: u32,
    mlpWidth:          u32,
    canvasWidth:       u32,
    canvasHeight:      u32,
    channelMask: u32, _p1: u32, _p2: u32,  // channelMask: bit i=0 → zero out emb channel i
    emb_range: array<vec4<f32>, 8>,  // [mn_plane0, mx_plane0, mn_plane1, mx_plane1, ...]
};

@group(0) @binding(0) var<uniform>      uniforms:     Uniforms;
@group(0) @binding(1) var<storage, read> embeddings_q: array<u32>;
@group(0) @binding(2) var<storage, read> layer1_weights: array<f32>;
@group(0) @binding(3) var<storage, read> layer1_biases: array<f32>;
@group(0) @binding(4) var<storage, read> layer2_weights: array<f32>;
@group(0) @binding(5) var<storage, read> layer2_biases: array<f32>;
@group(0) @binding(6) var<storage, read> layer3_weights: array<f32>;
@group(0) @binding(7) var<storage, read> layer3_biases: array<f32>;

// Output buffers for backpropagation
@group(0) @binding(8)  var<storage, read_write> out_inter_layer1: array<f32>;
@group(0) @binding(9)  var<storage, read_write> out_inter_layer2: array<f32>;
@group(0) @binding(10) var<storage, read_write> out_final: array<f32>;


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.canvasWidth || y >= uniforms.canvasHeight) {
        return;
    }

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
        let ox = EMB_OFFSETS[grp * 2u];
        let oy = EMB_OFFSETS[grp * 2u + 1u];
        let uvo = clamp(uv + vec2<f32>(ox, oy), vec2<f32>(0.0), vec2<f32>(1.0));
        let scaled = uvo * vec2<f32>(f32(uniforms.gridSize - 1u));
        let c = floor(scaled);
        let x0 = u32(c.x); let y0 = u32(c.y);
        let x1 = min(x0+1u, uniforms.gridSize-1u);
        let y1 = min(y0+1u, uniforms.gridSize-1u);
        var tx = scaled.x - c.x;
        var ty = scaled.y - c.y;
        ${smoothCode}
        let gs = uniforms.gridSize;
        let q00 = embeddings_q[(y0*gs+x0)*${numU32}u+grp];
        let q10 = embeddings_q[(y0*gs+x1)*${numU32}u+grp];
        let q01 = embeddings_q[(y1*gs+x0)*${numU32}u+grp];
        let q11 = embeddings_q[(y1*gs+x1)*${numU32}u+grp];
        for (var b = 0u; b < 8u; b++) {
            let ch = grp * 8u + b;
            let plane = ch / 4u;
            let comp  = ch % 4u;
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

    // --- MLP FORWARD PASS ---

    let pixel_index = y * uniforms.canvasWidth + x;

    // Layer 1: embeddingChannels -> mlpWidth
    var layer1_out = mat_vec_mul(&layer1_weights, embedding_vector, uniforms.embeddingChannels, uniforms.mlpWidth);
    for (var i: u32 = 0u; i < uniforms.mlpWidth; i = i + 1u) {
        layer1_out[i] = layer1_out[i] + layer1_biases[i];
        out_inter_layer1[pixel_index * uniforms.mlpWidth + i] = layer1_out[i]; // pre-activation
        layer1_out[i] = activ(layer1_out[i]);
    }

    // Layer 2: mlpWidth -> mlpWidth
    var layer2_out = mat_vec_mul_hidden(&layer2_weights, layer1_out, uniforms.mlpWidth, uniforms.mlpWidth);
    for (var i: u32 = 0u; i < uniforms.mlpWidth; i = i + 1u) {
        layer2_out[i] = layer2_out[i] + layer2_biases[i];
        out_inter_layer2[pixel_index * uniforms.mlpWidth + i] = layer2_out[i]; // pre-activation
        layer2_out[i] = activ(layer2_out[i]);
    }

    // Layer 3: mlpWidth -> 4 (RGBA)
    var layer3_out = mat_vec_mul_output(&layer3_weights, layer2_out, uniforms.mlpWidth, 4u);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        layer3_out[i] = layer3_out[i] + layer3_biases[i];
    }

    // Final color
    out_final[pixel_index * 4 + 0] = clamp(layer3_out[0], 0.0, 1.0); // R
    out_final[pixel_index * 4 + 1] = clamp(layer3_out[1], 0.0, 1.0); // G
    out_final[pixel_index * 4 + 2] = clamp(layer3_out[2], 0.0, 1.0); // B
    out_final[pixel_index * 4 + 3] = 1.0;                             // A
}
`;
    return shaderCode;
}
