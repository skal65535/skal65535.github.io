// shader_exporter.js

function matMul(rows, cols, inVec, outVec, weights, biases, activation) {
    let s = '';
    for (let i = 0; i < rows; i++) {
        s += `    ${outVec}[${i}] = ${biases}[${i}]`;
        for (let j = 0; j < cols; j++) {
            s += ` + ${weights}[${i*cols + j}] * ${inVec}[${j}]`;
        }
        s += ';\n';
    }
    if (activation) {
        s += `    for (int _i = 0; _i < ${rows}; _i++) ${outVec}[_i] = ${activation}(${outVec}[_i]);\n`;
    }
    return s;
}

export function export_to_glsl(config, weights) {
    const { gridSize, embeddingChannels, mlpWidth, smoothInterpolation, quantization } = config;
    const numPlanes = embeddingChannels / 4;
    const comps = ['r', 'g', 'b', 'a'];

    // One texture per embedding plane (up to 4 = iChannel0..3)
    const textureDecls = Array.from({length: numPlanes}, (_, p) =>
        `uniform sampler2D iChannel${p}; // embedding plane ${p}: channels ${p*4}–${p*4+3}`
    ).join('\n');

    const sampleFn = smoothInterpolation ? `
vec4 sample_plane(sampler2D smp, vec2 uv) {
    vec2 px = uv * vec2(${gridSize}.0);
    vec2 f  = fract(px);
    vec2 iu = (floor(px) + 0.5) / vec2(${gridSize}.0);
    vec2 fu = (floor(px) + 1.5) / vec2(${gridSize}.0);
    vec4 v00 = texture(smp, iu);
    vec4 v10 = texture(smp, vec2(fu.x, iu.y));
    vec4 v01 = texture(smp, vec2(iu.x, fu.y));
    vec4 v11 = texture(smp, fu);
    vec2 t = smoothstep(0.0, 1.0, f);
    return mix(mix(v00, v10, t.x), mix(v01, v11, t.x), t.y);
}` : `
vec4 sample_plane(sampler2D smp, vec2 uv) { return texture(smp, uv); }`;

    const quantizeFn = quantization === 'qat8' ? `
vec3 quantize(vec3 v) { return floor(v * 255.0 + 0.5) / 255.0; }` : '';

    // Embedding init: sample all planes into l0_out[]
    let embInit = `    float l0_out[${embeddingChannels}];\n`;
    for (let p = 0; p < numPlanes; p++) {
        embInit += `    { vec4 e = sample_plane(iChannel${p}, uv); `;
        for (let b = 0; b < 4; b++) embInit += `l0_out[${p*4+b}] = e.${comps[b]}; `;
        embInit += '}\n';
    }

    const shader = `precision highp float;
uniform vec2 iResolution;
${textureDecls}
${sampleFn}
${quantizeFn}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xy;

    // Layer 0: embeddings (${embeddingChannels} channels from ${numPlanes} texture(s))
${embInit}
    // Layer 1 (${embeddingChannels} → ${mlpWidth})
    const float l1_w[${mlpWidth * embeddingChannels}] = float[](${weights.layer1_weights.join(', ')});
    const float l1_b[${mlpWidth}] = float[](${weights.layer1_biases.join(', ')});
    float l1_out[${mlpWidth}];
${matMul(mlpWidth, embeddingChannels, 'l0_out', 'l1_out', 'l1_w', 'l1_b', 'sin')}
    // Layer 2 (${mlpWidth} → ${mlpWidth})
    const float l2_w[${mlpWidth * mlpWidth}] = float[](${weights.layer2_weights.join(', ')});
    const float l2_b[${mlpWidth}] = float[](${weights.layer2_biases.join(', ')});
    float l2_out[${mlpWidth}];
${matMul(mlpWidth, mlpWidth, 'l1_out', 'l2_out', 'l2_w', 'l2_b', 'sin')}
    // Layer 3 (${mlpWidth} → 4)
    const float l3_w[${4 * mlpWidth}] = float[](${weights.layer3_weights.join(', ')});
    const float l3_b[4] = float[](${weights.layer3_biases.join(', ')});
    vec4 l3_out;
${matMul(4, mlpWidth, 'l2_out', 'l3_out', 'l3_w', 'l3_b', '')}
    fragColor = clamp(vec4(l3_out.rgb, 1.0), 0.0, 1.0);
    ${quantization === 'qat8' ? 'fragColor.rgb = quantize(fragColor.rgb);' : ''}
}
`;
    return shader;
}
