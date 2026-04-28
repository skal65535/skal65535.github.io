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
    // GLSL export always uses fp32 textures (one texture per 4-channel plane), regardless of embBits
    const offsets = config.embOffsets
        ? (() => {
            // embOffsets may be sized for 4-bit (numGroups) or 8-bit (numPlanes); expand to numPlanes
            const embBits = config.embBits || 8;
            const numU32 = embeddingChannels / (32 / embBits);
            const out = new Float32Array(numPlanes * 2);
            // each u32 group covers channelsPerU32 channels; map groups → planes
            for (let p = 0; p < numPlanes; p++) {
                const grp = embBits === 4 ? Math.floor(p / 2) : p;
                out[p * 2]     = config.embOffsets[grp * 2]     || 0;
                out[p * 2 + 1] = config.embOffsets[grp * 2 + 1] || 0;
            }
            return out;
          })()
        : new Float32Array(numPlanes * 2);

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

    // For QAT8: snap weights to the same int8 grid used during training forward pass.
    // Biases and embeddings are not quantized (training doesn't quantize them).
    const qw = quantization === 'qat8'
        ? w => Math.round(Math.max(-1, Math.min(1, w)) * 127) / 127
        : w => w;

    // Embedding init: sample all planes into l0_out[], each with its UV offset
    let embInit = `    float l0_out[${embeddingChannels}];\n`;
    for (let p = 0; p < numPlanes; p++) {
        const ox = offsets[p * 2].toFixed(6), oy = offsets[p * 2 + 1].toFixed(6);
        const uvExpr = (ox === '0.000000' && oy === '0.000000')
            ? 'uv'
            : `clamp(uv + vec2(${ox}, ${oy}), 0.0, 1.0)`;
        embInit += `    { vec4 e = sample_plane(iChannel${p}, ${uvExpr}); `;
        for (let b = 0; b < 4; b++) embInit += `l0_out[${p*4+b}] = e.${comps[b]}; `;
        embInit += '}\n';
    }

    const shader = `precision highp float;
uniform vec2 iResolution;
${textureDecls}
${sampleFn}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xy;

    // Layer 0: embeddings (${embeddingChannels} channels from ${numPlanes} texture(s))
${embInit}
    // Layer 1 (${embeddingChannels} → ${mlpWidth})
    const float l1_w[${mlpWidth * embeddingChannels}] = float[](${Array.from(weights.layer1_weights).map(qw).join(', ')});
    const float l1_b[${mlpWidth}] = float[](${weights.layer1_biases.join(', ')});
    float l1_out[${mlpWidth}];
${matMul(mlpWidth, embeddingChannels, 'l0_out', 'l1_out', 'l1_w', 'l1_b', 'sin')}
    // Layer 2 (${mlpWidth} → ${mlpWidth})
    const float l2_w[${mlpWidth * mlpWidth}] = float[](${Array.from(weights.layer2_weights).map(qw).join(', ')});
    const float l2_b[${mlpWidth}] = float[](${weights.layer2_biases.join(', ')});
    float l2_out[${mlpWidth}];
${matMul(mlpWidth, mlpWidth, 'l1_out', 'l2_out', 'l2_w', 'l2_b', 'sin')}
    // Layer 3 (${mlpWidth} → 4)
    const float l3_w[${4 * mlpWidth}] = float[](${Array.from(weights.layer3_weights).map(qw).join(', ')});
    const float l3_b[4] = float[](${weights.layer3_biases.join(', ')});
    vec4 l3_out;
${matMul(4, mlpWidth, 'l2_out', 'l3_out', 'l3_w', 'l3_b', '')}
    fragColor = clamp(vec4(l3_out.rgb, 1.0), 0.0, 1.0);
}
`;
    return shader;
}
