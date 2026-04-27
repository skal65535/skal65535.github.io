// shader_builder.js

export function buildShader(config) {
    const { gridSize, embeddingChannels, mlpWidth, quantization, smoothInterpolation } = config;
    const numPlanes = embeddingChannels / 4;

    const quantizationFunctions = `
fn quantize_dequantize_8bit(val: f32) -> f32 {
    return round(clamp(val, -1.0, 1.0) * 127.0) / 127.0;
}
`;

    const smoothCode = smoothInterpolation
        ? 'tx = tx*tx*(3.0-2.0*tx); ty = ty*ty*(3.0-2.0*ty);'
        : '';

    // Samples one vec4 plane from the packed u32 embedding buffer with bilinear interpolation.
    const interpolationFunctions = `
fn sample_embedding_plane(uv: vec2<f32>, plane: u32) -> vec4<f32> {
    let scaled = uv * vec2<f32>(f32(uniforms.gridSize - 1u));
    let c      = floor(scaled);
    let x0 = u32(c.x); let y0 = u32(c.y);
    let x1 = min(x0 + 1u, uniforms.gridSize - 1u);
    let y1 = min(y0 + 1u, uniforms.gridSize - 1u);
    var tx = scaled.x - c.x;
    var ty = scaled.y - c.y;
    ${smoothCode}
    let gs = uniforms.gridSize;
    let c00 = unpack4x8snorm(embeddings_q[(y0*gs+x0)*${numPlanes}u+plane]);
    let c10 = unpack4x8snorm(embeddings_q[(y0*gs+x1)*${numPlanes}u+plane]);
    let c01 = unpack4x8snorm(embeddings_q[(y1*gs+x0)*${numPlanes}u+plane]);
    let c11 = unpack4x8snorm(embeddings_q[(y1*gs+x1)*${numPlanes}u+plane]);
    let interp = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
    // unscale from packed [-1,1] using per-plane range stored in uniforms
    let mn = uniforms.emb_range[plane * 2u];
    let mx = uniforms.emb_range[plane * 2u + 1u];
    return (interp + vec4<f32>(1.0)) * 0.5 * (mx - mn) + mn;
}
`;

    // Generate dot-product inner loop (width must be a multiple of 4)
    const dotLoop = (vecType) => quantization === 'qat8'
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

    const mlpFunctions = `
fn mat_vec_mul(mat: ptr<storage, array<f32>, read>, vec_in: array<f32, ${embeddingChannels}>, width: u32, height: u32) -> array<f32, ${mlpWidth}> {
    var result: array<f32, ${mlpWidth}>;
    for (var i: u32 = 0u; i < height; i = i + 1u) {
        var sum = 0.0;
${dotLoop('embeddingChannels')}
        result[i] = sum;
    }
    return result;
}

fn mat_vec_mul_hidden(mat: ptr<storage, array<f32>, read>, vec_in: array<f32, ${mlpWidth}>, width: u32, height: u32) -> array<f32, ${mlpWidth}> {
    var result: array<f32, ${mlpWidth}>;
    for (var i: u32 = 0u; i < height; i = i + 1u) {
        var sum = 0.0;
${dotLoop('mlpWidth')}
        result[i] = sum;
    }
    return result;
}

fn mat_vec_mul_output(mat: ptr<storage, array<f32>, read>, vec_in: array<f32, ${mlpWidth}>, width: u32, height: u32) -> array<f32, 4> {
    var result: array<f32, 4>;
    for (var i: u32 = 0u; i < height; i = i + 1u) {
        var sum = 0.0;
${dotLoop('mlpWidth')}
        result[i] = sum;
    }
    return result;
}
`;

    const shaderCode = `
${quantizationFunctions}
${interpolationFunctions}
${mlpFunctions}

struct Uniforms {
    gridSize:          u32,
    embeddingChannels: u32,
    mlpWidth:          u32,
    canvasWidth:       u32,
    canvasHeight:      u32,
    _p0: u32, _p1: u32, _p2: u32,   // pad to align vec4 at offset 32
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
    for (var plane = 0u; plane < ${numPlanes}u; plane++) {
        let v = sample_embedding_plane(uv, plane);
        embedding_vector[plane*4u+0u] = v.x;
        embedding_vector[plane*4u+1u] = v.y;
        embedding_vector[plane*4u+2u] = v.z;
        embedding_vector[plane*4u+3u] = v.w;
    }

    // --- MLP FORWARD PASS ---

    let pixel_index = y * uniforms.canvasWidth + x;

    // Layer 1: embeddingChannels -> mlpWidth
    var layer1_out = mat_vec_mul(&layer1_weights, embedding_vector, uniforms.embeddingChannels, uniforms.mlpWidth);
    for (var i: u32 = 0u; i < uniforms.mlpWidth; i = i + 1u) {
        layer1_out[i] = layer1_out[i] + layer1_biases[i];
        out_inter_layer1[pixel_index * uniforms.mlpWidth + i] = layer1_out[i]; // pre-activation
        layer1_out[i] = sin(layer1_out[i]);
    }

    // Layer 2: mlpWidth -> mlpWidth
    var layer2_out = mat_vec_mul_hidden(&layer2_weights, layer1_out, uniforms.mlpWidth, uniforms.mlpWidth);
    for (var i: u32 = 0u; i < uniforms.mlpWidth; i = i + 1u) {
        layer2_out[i] = layer2_out[i] + layer2_biases[i];
        out_inter_layer2[pixel_index * uniforms.mlpWidth + i] = layer2_out[i]; // pre-activation
        layer2_out[i] = sin(layer2_out[i]);
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
