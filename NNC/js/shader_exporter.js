// shader_exporter.js

function fmtFloatArray(arr, fmt, valsPerLine = 4) {
    const vals = Array.from(arr).map(fmt);
    const lines = [];
    for (let i = 0; i < vals.length; i += valsPerLine)
        lines.push('    ' + vals.slice(i, i + valsPerLine).join(', '));
    return lines.join(',\n');
}

// Format float array as vec4[](…) body — groups of 4 floats per vec4, one vec4 per line.
function fmtVec4Array(arr, fmt) {
    const lines = [];
    for (let i = 0; i < arr.length; i += 4)
        lines.push(`    vec4(${fmt(arr[i])}, ${fmt(arr[i+1])}, ${fmt(arr[i+2])}, ${fmt(arr[i+3])})`);
    return lines.join(',\n');
}

// All arrays (weights, biases, inputs, outputs) are vec4[].
// rows and cols must be multiples of 4.
// Computes rows/4 output vec4s, each from 4 consecutive rows of the matrix.
function matMul(rows, cols, inVec, outVec, weights, biases, activation) {
    console.assert(rows % 4 === 0 && cols % 4 === 0,
        `matMul: rows=${rows} and cols=${cols} must be multiples of 4`);
    const nc = cols / 4;
    let s = '';
    for (let g = 0; g < rows / 4; g++) {
        s += `    ${outVec}[${g}] = ${activation ? activation + '(' : ''}${biases}[${g}] + vec4(\n`;
        for (let r = 0; r < 4; r++) {
            const base = (g * 4 + r) * nc;
            // two dots per line, continuation lines indented
            const dots = Array.from({length: nc}, (_, p) =>
                `dot(${weights}[${base + p}], ${inVec}[${p}])`);
            const lines = [];
            for (let p = 0; p < dots.length; p += 2)
                lines.push((p === 0 ? '        ' : '            + ') +
                    dots.slice(p, p + 2).join(' + '));
            s += lines.join('\n') + (r < 3 ? ',\n' : '\n');
        }
        s += activation ? '    ));\n' : '    );\n';
    }
    return s;
}

// Pack one 4-channel embedding plane as uint32 array (signed int8 per channel).
// embeddings: Float32Array [gridCount, embCh], normalized to [-1,1].
function packEmbPlane(embeddings, planeIdx, gridCount, embCh) {
    const u32 = new Uint32Array(gridCount);
    for (let i = 0; i < gridCount; i++) {
        let p = 0;
        for (let b = 0; b < 4; b++) {
            const v = embeddings[i * embCh + planeIdx * 4 + b];
            p |= (Math.max(-127, Math.min(127, Math.round(v * 127))) & 0xFF) << (b * 8);
        }
        u32[i] = p >>> 0;
    }
    return u32;
}

// Format uint32 array as uvec4[](…) body (groups of 4 uint32 → one uvec4).
function fmtUvec4Array(u32Array, perLine = 2) {
    const vecs = [];
    for (let i = 0; i < u32Array.length; i += 4) {
        const a = u32Array[i], b = u32Array[i+1] || 0;
        const c = u32Array[i+2] || 0, d = u32Array[i+3] || 0;
        vecs.push(`uvec4(${a}u,${b}u,${c}u,${d}u)`);
    }
    const lines = [];
    for (let i = 0; i < vecs.length; i += perLine)
        lines.push('    ' + vecs.slice(i, i + perLine).join(', '));
    return lines.join(',\n');
}

// Common tab GLSL — shared constant and macro for all embedding buffers.
function genCommonGLSL(gridSize) {
    return `\
// === Common tab ===
// Paste into the Common tab in Shadertoy (shared by all Buffer tabs).

const int embed_texture = ${gridSize};

#define EMB_BUFFER_MAINIMAGE(quant_arr)                                                 \\
void mainImage(out vec4 fragColor, in vec2 fragCoord) {                                 \\
    ivec2 gxy = ivec2(fragCoord);                                                       \\
    if (gxy.x >= embed_texture || gxy.y >= embed_texture) { fragColor = vec4(0.0); return; } \\
    int idx = gxy.x + gxy.y * embed_texture;                                           \\
    uint val = quant_arr[idx >> 2][idx & 3];                                            \\
    ivec4 s = ivec4((uvec4(val) >> uvec4(0u, 8u, 16u, 24u)) & uvec4(255u));            \\
    s -= 256 * ivec4(greaterThan(s, ivec4(127)));                                       \\
    fragColor = clamp(vec4(s) / 127.0, -1.0, 1.0);                                     \\
}`;
}

// Generate Buffer GLSL for one embedding plane.
// Buffer is gridSize×gridSize pixels — each pixel stores one raw grid value.
// GPU sampler bilinear handles interpolation; main shader rescales UV.
function genBufferGLSL(p, gridSize, u32Data) {
    const label = String.fromCharCode(65 + p); // 'A', 'B', ...
    const N = u32Data.length / 4;              // number of uvec4s
    return `\
// === Buffer ${label} — embedding plane ${p} (channels ${p*4}–${p*4+3}) ===
// Paste into Buffer ${label} tab. Set iChannel${p} = Buffer ${label} in Image tab.

const uvec4 embed${p}_quant[${N}] = uvec4[${N}](
${fmtUvec4Array(u32Data)}
);
EMB_BUFFER_MAINIMAGE(embed${p}_quant)`;
}

export function export_to_glsl(config, weights) {
    const { gridSize, embeddingChannels, mlpWidth, smoothInterpolation, quantization, width, height } = config;
    const numPlanes = embeddingChannels / 4;

    const offsets = config.embOffsets
        ? (() => {
            const embBits = config.embBits || 8;
            const out = new Float32Array(numPlanes * 2);
            for (let p = 0; p < numPlanes; p++) {
                const grp = embBits === 4 ? Math.floor(p / 2) : p;
                out[p * 2]     = config.embOffsets[grp * 2]     || 0;
                out[p * 2 + 1] = config.embOffsets[grp * 2 + 1] || 0;
            }
            return out;
          })()
        : new Float32Array(numPlanes * 2);

    const textureDecls = Array.from({length: numPlanes}, (_, p) =>
        `uniform sampler2D iChannel${p}; // embedding plane ${p}: channels ${p*4}–${p*4+3}`
    ).join('\n');

    // Rescale uv [0,1] to the gridSize×gridSize region with align-corners=true,
    // so the GPU sampler bilinear matches the training interpolation.
    const sampleFn = `\nvec4 sample_plane(sampler2D smp, vec2 uv) {
    vec2 bufUV = (uv * float(embed_texture - 1) + 0.5) / iResolution.xy;
    return texture(smp, bufUV);
}`;

    // Fixed-width float: 8 decimal places, always with sign padding for alignment.
    const fmtF = v => {
        const s = v.toFixed(8);
        return v >= 0 ? ' ' + s : s;
    };
    const qw = quantization === 'qat8'
        ? w => fmtF(Math.round(Math.max(-1, Math.min(1, w)) * 127) / 127)
        : w => fmtF(w);

    let embInit = `    vec4 l0_out[${numPlanes}];\n`;
    for (let p = 0; p < numPlanes; p++) {
        const ox = offsets[p * 2].toFixed(6), oy = offsets[p * 2 + 1].toFixed(6);
        const uvExpr = (ox === '0.000000' && oy === '0.000000')
            ? 'uv'
            : `clamp(uv + vec2(${ox}, ${oy}), 0.0, 1.0)`;
        embInit += `    l0_out[${p}] = sample_plane(iChannel${p}, ${uvExpr});\n`;
    }

    const N1 = mlpWidth * embeddingChannels / 4, N2 = mlpWidth * mlpWidth / 4;
    const NW = mlpWidth / 4;
    const weightDecls = `
// Layer 1 weights/biases  (${embeddingChannels} → ${mlpWidth})
const vec4 l1_w[${N1}] = vec4[${N1}](\n${fmtVec4Array(weights.layer1_weights, qw)}\n);
const vec4 l1_b[${NW}] = vec4[${NW}](\n${fmtVec4Array(weights.layer1_biases, qw)}\n);

// Layer 2 weights/biases  (${mlpWidth} → ${mlpWidth})
const vec4 l2_w[${N2}] = vec4[${N2}](\n${fmtVec4Array(weights.layer2_weights, qw)}\n);
const vec4 l2_b[${NW}] = vec4[${NW}](\n${fmtVec4Array(weights.layer2_biases, qw)}\n);

// Layer 3 weights/biases  (${mlpWidth} → 4)
const vec4 l3_w[${mlpWidth}] = vec4[${mlpWidth}](\n${fmtVec4Array(weights.layer3_weights, qw)}\n);
const vec4 l3_b[1] = vec4[1](\n${fmtVec4Array(weights.layer3_biases, qw)}\n);`;

    const mainShader = `precision highp float;
uniform vec2 iResolution;
${textureDecls}
${sampleFn}
${weightDecls}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv_raw = vec2(fragCoord.x, iResolution.y - fragCoord.y) / iResolution.xy;
    // Fit image (${width}×${height}) centered in canvas, preserving aspect ratio
    float ca = iResolution.x / iResolution.y;
    float ia = float(${width}) / float(${height});
    vec2 uv = ca > ia
        ? vec2((uv_raw.x - 0.5) * ca / ia + 0.5, uv_raw.y)
        : vec2(uv_raw.x, (uv_raw.y - 0.5) * ia / ca + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        { fragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }

    // Layer 0: embeddings (${embeddingChannels} channels from ${numPlanes} buffer(s))
${embInit}
    // Layer 1
    vec4 l1_out[${NW}];
${matMul(mlpWidth, embeddingChannels, 'l0_out', 'l1_out', 'l1_w', 'l1_b', 'sin')}
    // Layer 2
    vec4 l2_out[${NW}];
${matMul(mlpWidth, mlpWidth, 'l1_out', 'l2_out', 'l2_w', 'l2_b', 'sin')}
    // Layer 3
    vec4 l3_out[1];
${matMul(4, mlpWidth, 'l2_out', 'l3_out', 'l3_w', 'l3_b', '')}
    fragColor = clamp(vec4(l3_out[0].rgb, 1.0), 0.0, 1.0);
}
`;

    // Generate commented-out buffer code for each embedding plane.
    const gridCount = gridSize * gridSize;
    const buffers = Array.from({length: numPlanes}, (_, p) => {
        const u32 = packEmbPlane(weights.embeddings, p, gridCount, embeddingChannels);
        return genBufferGLSL(p, gridSize, u32);
    });

    const commonSection = `/*\n${genCommonGLSL(gridSize)}\n*/`;
    const bufferSection = buffers.map(code => `/*\n${code}\n*/`).join('\n\n');

    return `${mainShader}\n// ============================================================\n// Shadertoy buffer code — paste each block into its tab\n// ============================================================\n\n${commonSection}\n\n${bufferSection}\n`;
}
