// shader_exporter.js
// Exports the trained model weights as a self-contained GLSL shader for ShaderToy.
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

// Pack float array as uint array (int8×4 per uint, same layout as embedding buffers).
function packMlpWeights(floatArr) {
    const n = Math.ceil(floatArr.length / 4);
    const u32 = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        let p = 0;
        for (let b = 0; b < 4; b++) {
            const v = floatArr[i * 4 + b] || 0;
            const q = Math.max(-127, Math.min(127, Math.round(v * 127)));
            p |= (q & 0xFF) << (b * 8);
        }
        u32[i] = p >>> 0;
    }
    return u32;
}

function fmtUintPackedArray(u32Array, perLine = 6) {
    const vals = Array.from(u32Array).map(v =>
        `0x${(v >>> 0).toString(16).padStart(8, '0').toUpperCase()}u`);
    const lines = [];
    for (let i = 0; i < vals.length; i += perLine)
        lines.push('    ' + vals.slice(i, i + perLine).join(', '));
    return lines.join(',\n');
}

function glslActivDecl(activation) {
    switch (activation) {
        case 'softsign': return `vec4 softsign(vec4 x) { return x / (1.0 + abs(x)); }`;
        case 'none': return '';
        default: return '';
    }
}
function glslActivName(activation) {
    return { sin: 'sin', tanh: 'tanh', softsign: 'softsign', none: '' }[activation] ?? 'sin';
}

// All arrays (weights, biases, inputs, outputs) are vec4[].
// rows and cols must be multiples of 4.
// Computes rows/4 output vec4s, each from 4 consecutive rows of the matrix.
// packedWeights=true: weights array is uint[] (int8×4), unpacked via unpack8().
function matMul(rows, cols, inVec, outVec, weights, biases, activation, packedWeights = false) {
    if (rows % 4 !== 0 || cols % 4 !== 0)
        throw new Error(`matMul: rows=${rows} and cols=${cols} must be multiples of 4`);
    const nc = cols / 4;
    let s = '';
    for (let g = 0; g < rows / 4; g++) {
        s += `    ${outVec}[${g}] = ${activation ? activation + '(' : ''}${biases}[${g}] + vec4(\n`;
        for (let r = 0; r < 4; r++) {
            const base = (g * 4 + r) * nc;
            const dots = Array.from({length: nc}, (_, p) => {
                const wv = packedWeights
                    ? `unpack8(${weights}[${base + p}])`
                    : `${weights}[${base + p}]`;
                return `dot(${wv}, ${inVec}[${p}])`;
            });
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

// Format uint32 array as plain uint[](…) body.
function fmtUintArray(u32Array, perLine = 8) {
    const lines = [];
    for (let i = 0; i < u32Array.length; i += perLine)
        lines.push('    ' + Array.from(u32Array.slice(i, i + perLine)).map(v => `${v}u`).join(', '));
    return lines.join(',\n');
}

// Common tab GLSL — shared constant and macro for all embedding buffers.
function genCommonGLSL(gW, gH) {
    return `\
// === Common tab ===
// Paste into the Common tab in Shadertoy (shared by Image + all Buffer tabs).

const int embed_w = ${gW};
const int embed_h = ${gH};

vec4 unpack8(uint u) {   // no unpackSnorm4x8 !!
    ivec4 s = ivec4((uvec4(u) >> uvec4(0u,8u,16u,24u)) & uvec4(255u));
    s -= 256 * ivec4(greaterThan(s, ivec4(127)));
    return vec4(s) / 127.0;
}

#define EMB_BUFFER_MAINIMAGE(quant_arr)                                \\
void mainImage(out vec4 fragColor, in vec2 fragCoord) {                \\
    ivec2 gxy = ivec2(fragCoord);                                      \\
    if (gxy.x >= embed_w || gxy.y >= embed_h) { discard; }            \\
    int idx = gxy.x + gxy.y * embed_w;                                \\
    uint val = quant_arr[idx];                                         \\
    fragColor = unpack8(val);                                          \\
}`;
}

// Generate Buffer GLSL for one embedding plane.
// Buffer is gW×gH pixels — each pixel stores one raw grid value.
// GPU sampler bilinear handles interpolation; main shader rescales UV.
function genBufferGLSL(p, gW, gH, u32Data) {
    const label = String.fromCharCode(65 + p); // 'A', 'B', ...
    const N = u32Data.length;
    return `\
// === Buffer ${label} — embedding plane ${p} (channels ${p*4}–${p*4+3}) ===
// Paste into Buffer ${label} tab. Set iChannel${p} = Buffer ${label} in Image tab.
precision highp float;
precision highp int;

const uint embed${p}[${N}] = uint[${N}](
${fmtUintArray(u32Data)}
);
EMB_BUFFER_MAINIMAGE(embed${p})`;
}

export function export_to_glsl(config, weights) {
    const { gW, gH, embeddingChannels, mlpWidth1, mlpWidth2, smoothInterpolation, quantization, width, height, activation = 'sin' } = config;
    const outCh = config.hasAlpha ? 4 : 3;
    const numPlanes = embeddingChannels / 4;

    if (numPlanes > 4)
        throw new Error(`Cannot export to GLSL: ${numPlanes} embedding planes require iChannel0–iChannel${numPlanes-1}, but ShaderToy supports at most 4 iChannels. Reduce embeddingChannels to ≤16.`);

    const embMax = weights.embeddings.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    if (embMax > 1.01)
        console.warn(`export_to_glsl: embeddings range ±${embMax.toFixed(2)} (expected ≤1). CPU-trained models are not normalized — exported GLSL will have saturated channels. Re-train on GPU for best results.`);

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

    // Rescale uv [0,1] to the gW×gH region with align-corners=true.
    // When smoothInterpolation is on, use manual 4-tap texelFetch + smoothstep to
    // match training; otherwise let the GPU sampler handle plain bilinear.
    const sampleFn = smoothInterpolation
        ? `\nvec4 sample_plane(sampler2D smp, vec2 uv) {
    vec2 scaled = uv * vec2(float(embed_w - 1), float(embed_h - 1));
    ivec2 c0 = ivec2(floor(scaled));
    ivec2 c1 = min(c0 + ivec2(1), ivec2(embed_w - 1, embed_h - 1));
    vec2 t = scaled - vec2(c0);
    t = t * t * (3.0 - 2.0 * t);
    vec4 v00 = texelFetch(smp, c0,                0);
    vec4 v10 = texelFetch(smp, ivec2(c1.x, c0.y), 0);
    vec4 v01 = texelFetch(smp, ivec2(c0.x, c1.y), 0);
    vec4 v11 = texelFetch(smp, c1,                0);
    return mix(mix(v00, v10, t.x), mix(v01, v11, t.x), t.y);
}`
        : `\nvec4 sample_plane(sampler2D smp, vec2 uv) {
    vec2 bufUV = (uv * vec2(float(embed_w - 1), float(embed_h - 1)) + 0.5) / vec2(float(embed_w), float(embed_h));
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

    const N1 = mlpWidth1 * embeddingChannels / 4, N2 = mlpWidth2 * mlpWidth1 / 4;
    const NW1 = mlpWidth1 / 4, NW2 = mlpWidth2 / 4;
    const packWeights = quantization === 'qat8';

    const wType = packWeights ? 'uint' : 'vec4';
    const fmtW = (arr, n) => packWeights
        ? `uint[${n}](\n${fmtUintPackedArray(packMlpWeights(arr))}\n)`
        : `vec4[${n}](\n${fmtVec4Array(arr, qw)}\n)`;

    const weightDecls = `
// Layer 1 weights/biases  (${embeddingChannels} → ${mlpWidth1})
const ${wType} l1_w[${N1}] = ${fmtW(weights.layer1_weights, N1)};
const vec4 l1_b[${NW1}] = vec4[${NW1}](
${fmtVec4Array(weights.layer1_biases, qw)}
);

// Layer 2 weights/biases  (${mlpWidth1} → ${mlpWidth2})
const ${wType} l2_w[${N2}] = ${fmtW(weights.layer2_weights, N2)};
const vec4 l2_b[${NW2}] = vec4[${NW2}](
${fmtVec4Array(weights.layer2_biases, qw)}
);

// Layer 3 weights/biases  (${mlpWidth2} → ${outCh})
${(() => {
    // pad to 4 rows if outCh===3 so matMul(4,...) works; extra row is zero
    const w4 = outCh === 4 ? weights.layer3_weights
        : (() => { const p = new Float32Array(mlpWidth2 * 4); p.set(weights.layer3_weights); return p; })();
    const b4 = outCh === 4 ? weights.layer3_biases
        : (() => { const p = new Float32Array(4); p.set(weights.layer3_biases); return p; })();
    return `const ${wType} l3_w[${mlpWidth2}] = ${fmtW(w4, mlpWidth2)};
const vec4 l3_b[1] = vec4[1](
${fmtVec4Array(b4, qw)}
);`;
})()}`;

    const activDecl = glslActivDecl(activation);
    const activName = glslActivName(activation);

    const embBitsNote = (config.embBits || 8) === 4
        ? `// Note: model was trained with 4-bit embeddings; buffers re-encoded as 8-bit (int8×4) for ShaderToy.\n` : '';

    const mainShader = `// === Image tab === (requires Common tab for embed_texture${packWeights ? ', unpack8' : ''})
${embBitsNote}
precision highp float;
uniform vec2 iResolution;
${textureDecls}
${sampleFn}
${activDecl ? activDecl + '\n' : ''}${weightDecls}

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
    vec4 l1_out[${NW1}];
${matMul(mlpWidth1, embeddingChannels, 'l0_out', 'l1_out', 'l1_w', 'l1_b', activName, packWeights)}
    // Layer 2
    vec4 l2_out[${NW2}];
${matMul(mlpWidth2, mlpWidth1, 'l1_out', 'l2_out', 'l2_w', 'l2_b', activName, packWeights)}
    // Layer 3
    vec4 l3_out[1];
${matMul(4, mlpWidth2, 'l2_out', 'l3_out', 'l3_w', 'l3_b', '', packWeights)}
    fragColor = clamp(${outCh === 4 ? 'l3_out[0]' : 'vec4(l3_out[0].rgb, 1.0)'}, 0.0, 1.0);
}
`;

    // Generate commented-out buffer code for each embedding plane.
    const gridCount = gW * gH;
    const buffers = Array.from({length: numPlanes}, (_, p) => {
        const u32 = packEmbPlane(weights.embeddings, p, gridCount, embeddingChannels);
        return genBufferGLSL(p, gW, gH, u32);
    });

    const commonSection = `/*\n${genCommonGLSL(gW, gH)}\n*/`;
    const bufferSection = buffers.map(code => `/*\n${code}\n*/`).join('\n\n');

    return `${mainShader}\n// ============================================================\n// Shadertoy buffer code — paste each block into its tab\n// ============================================================\n\n${commonSection}\n\n${bufferSection}\n`;
}
