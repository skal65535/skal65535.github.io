// model.js
// Creates and manages GPU weight, Adam moment, and grad accumulation buffers.

export const UNIFORM_OFFSET_RANGE = 32;
export const UNIFORM_OFFSET_OFFSETS = 160;
export const UNIFORM_OFFSET_MASK = 20;

// f32 element offsets for each sub-tensor in the flat mlp_weights buffer.
// Order: [L1W | L1B | L2W | L2B | L3W | L3B]
export function mlpWeightsLayout(config) {
    const { embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = config;
    const outCh = config.hasAlpha ? 4 : 3;
    const l1w = 0, l1b = l1w + embCh * mlpWidth1,
          l2w = l1b + mlpWidth1, l2b = l2w + mlpWidth1 * mlpWidth2,
          l3w = l2b + mlpWidth2, l3b = l3w + mlpWidth2 * outCh;
    return { l1w, l1b, l2w, l2b, l3w, l3b };
}

export function computeTensorSizes(config) {
    const { gridSize, embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = config;
    const outCh = config.hasAlpha ? 4 : 3;
    return {
        embeddings:     gridSize * gridSize * embCh,
        layer1_weights: embCh * mlpWidth1,
        layer1_biases:  mlpWidth1,
        layer2_weights: mlpWidth1 * mlpWidth2,
        layer2_biases:  mlpWidth2,
        layer3_weights: mlpWidth2 * outCh,
        layer3_biases:  outCh,
    };
}

export class ModelTensors {
    static KEYS = ['embeddings', 'layer1_weights', 'layer1_biases', 'layer2_weights', 'layer2_biases', 'layer3_weights', 'layer3_biases'];

    constructor(init) { Object.assign(this, init); }

    static create(fn) {
        const obj = {};
        for (const k of ModelTensors.KEYS) obj[k] = fn(k);
        return new ModelTensors(obj);
    }
}

export function createModel(ctx, config) {
    const sizes = computeTensorSizes(config);
    const { gridSize, embeddingChannels, mlpWidth1, mlpWidth2 } = config;
    const embBits = config.embBits || 8;
    const channelsPerU32 = 32 / embBits;

    const embeddingData = new Float32Array(sizes.embeddings).map(() => Math.random() * 2 - 1);

    // SIREN init: first layer 1/fan_in, hidden sqrt(6/fan_in)
    const layer1Weights = new Float32Array(sizes.layer1_weights).map(() => (Math.random() * 2 - 1) / embeddingChannels);
    const layer1Biases = new Float32Array(sizes.layer1_biases).fill(0);

    const s2 = Math.sqrt(6 / mlpWidth1);
    const layer2Weights = new Float32Array(sizes.layer2_weights).map(() => (Math.random() * 2 - 1) * s2);
    const layer2Biases = new Float32Array(sizes.layer2_biases).fill(0);

    const outCh = config.hasAlpha ? 4 : 3;
    const s3 = Math.sqrt(6 / mlpWidth2);
    const layer3Weights = new Float32Array(sizes.layer3_weights).map(() => (Math.random() * 2 - 1) * s3);
    const layer3Biases = new Float32Array(sizes.layer3_biases).fill(0);

    const zeroTensors = (prefix) => ModelTensors.create(k => ctx.zeroBuffer(sizes[k], undefined, `${prefix}/${k}`));

    const numU32 = embeddingChannels / channelsPerU32;
    const embOffsets = config.embOffsets || new Float32Array(numU32 * 2);

    const layout  = mlpWeightsLayout(config);
    const mlpData = new Float32Array(layout.l3b + outCh);
    mlpData.set(layer1Weights, layout.l1w); mlpData.set(layer1Biases, layout.l1b);
    mlpData.set(layer2Weights, layout.l2w); mlpData.set(layer2Biases, layout.l2b);
    mlpData.set(layer3Weights, layout.l3w); mlpData.set(layer3Biases, layout.l3b);

    const modelBuffers = {
        embeddings:       ctx.createBuffer(embeddingData, undefined, 'emb'),
        embeddings_q:     ctx.zeroBuffer(sizes.embeddings / channelsPerU32, undefined, 'emb_q'),
        embeddings_range: ctx.zeroBuffer(embeddingChannels * 2, undefined, 'emb_range'),
        emb_offsets:      ctx.createBuffer(embOffsets, undefined, 'emb_offsets'),
        mlp_weights:      ctx.createBuffer(mlpData, undefined, 'mlp_weights'),
        mlpLayout:        layout,
        layer1: { weights: ctx.createBuffer(layer1Weights, undefined, 'L1/weights'), biases: ctx.createBuffer(layer1Biases, undefined, 'L1/biases') },
        layer2: { weights: ctx.createBuffer(layer2Weights, undefined, 'L2/weights'), biases: ctx.createBuffer(layer2Biases, undefined, 'L2/biases') },
        layer3: { weights: ctx.createBuffer(layer3Weights, undefined, 'L3/weights'), biases: ctx.createBuffer(layer3Biases, undefined, 'L3/biases') },
        adamM:      zeroTensors('adamM'),
        adamV:      zeroTensors('adamV'),
        gradAtomic: zeroTensors('grad'),
    };

    return {
        buffers: modelBuffers,
        weights: new ModelTensors({
            embeddings:     embeddingData,
            layer1_weights: layer1Weights, layer1_biases: layer1Biases,
            layer2_weights: layer2Weights, layer2_biases: layer2Biases,
            layer3_weights: layer3Weights, layer3_biases: layer3Biases,
        }),
    };
}

export function initCpuWeights(config) {
    const sizes = computeTensorSizes(config);
    const { embeddingChannels, mlpWidth1, mlpWidth2 } = config;
    const s2 = Math.sqrt(6 / mlpWidth1), s3 = Math.sqrt(6 / mlpWidth2);
    return {
        embeddings:     new Float32Array(sizes.embeddings).map(() => Math.random() * 2 - 1),
        layer1_weights: new Float32Array(sizes.layer1_weights).map(() => (Math.random() * 2 - 1) / embeddingChannels),
        layer1_biases:  new Float32Array(sizes.layer1_biases),
        layer2_weights: new Float32Array(sizes.layer2_weights).map(() => (Math.random() * 2 - 1) * s2),
        layer2_biases:  new Float32Array(sizes.layer2_biases),
        layer3_weights: new Float32Array(sizes.layer3_weights).map(() => (Math.random() * 2 - 1) * s3),
        layer3_biases:  new Float32Array(sizes.layer3_biases),
    };
}

export function destroyModel(m) {
    if (!m) return;
    m.embeddings?.destroy();
    m.embeddings_q?.destroy();
    m.embeddings_range?.destroy();
    m.emb_offsets?.destroy();
    m.mlp_weights?.destroy();
    m.layer1?.weights?.destroy();  m.layer1?.biases?.destroy();
    m.layer2?.weights?.destroy();  m.layer2?.biases?.destroy();
    m.layer3?.weights?.destroy();  m.layer3?.biases?.destroy();
    for (const mt of [m.adamM, m.adamV, m.gradAtomic]) {
        if (!mt) continue;
        for (const k of ModelTensors.KEYS) mt[k]?.destroy();
    }
}

const SHAKE_EMB_AMPLITUDE = 0.40;
const SHAKE_MLP_AMPLITUDE = 0.01;

export function shakeEmbeddings(ctx, model, embData) {
    for (let i = 0; i < embData.length; i++) embData[i] += (Math.random() * 2 - 1) * SHAKE_EMB_AMPLITUDE;
    ctx.writeBuffer(model.embeddings, embData);
    ctx.clearBuffer(model.adamM.embeddings);
    ctx.clearBuffer(model.adamV.embeddings);
}

export function shakeMlp(ctx, model, rb) {
    const ly = model.mlpLayout;
    const entries = [
        { k: 'layer1_weights', buf: model.layer1.weights, off: ly?.l1w },
        { k: 'layer1_biases',  buf: model.layer1.biases,  off: ly?.l1b },
        { k: 'layer2_weights', buf: model.layer2.weights, off: ly?.l2w },
        { k: 'layer2_biases',  buf: model.layer2.biases,  off: ly?.l2b },
        { k: 'layer3_weights', buf: model.layer3.weights, off: ly?.l3w },
        { k: 'layer3_biases',  buf: model.layer3.biases,  off: ly?.l3b },
    ];
    for (const { k, buf, off } of entries) {
        const w = rb[k];
        for (let i = 0; i < w.length; i++) w[i] += (Math.random() * 2 - 1) * SHAKE_MLP_AMPLITUDE;
        ctx.writeBuffer(buf, w);
        ctx.clearBuffer(model.adamM[k]);
        ctx.clearBuffer(model.adamV[k]);
        if (model.mlp_weights) ctx.writeBufferAt(model.mlp_weights, off * 4, w);
    }
}

// --- Merged from emb_utils.js ---

export function generateEmbOffsets(embCh, embBits, gridSize, noOffset) {
    const numU32 = embCh / (32 / embBits);
    if (noOffset) return new Float32Array(numU32 * 2);
    const scale = 1.0 / gridSize;
    return new Float32Array(numU32 * 2).map(() => (Math.random() - 0.5) * scale);
}

export function computeEmbRange(embData, embCh, gridCount) {
    const range = new Float32Array(embCh * 2);
    for (let c = 0; c < embCh; c++) {
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < gridCount; i++) {
            const v = embData[i * embCh + c];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        range[c * 2]     = mn === Infinity  ? -1 : mn;
        range[c * 2 + 1] = mx === -Infinity ?  1 : mx;
    }
    return range;
}

export function normalizeEmbAndAdjustL1(embData, l1Weights, l1Biases, embCh, mlpWidth1) {
    const gridCount = embData.length / embCh;
    for (let c = 0; c < embCh; c++) {
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < gridCount; i++) {
            const v = embData[i * embCh + c];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        const span = mx - mn;
        if (span < 1e-9) continue;
        const scale  = span * 0.5;
        const center = (mx + mn) * 0.5;
        for (let i = 0; i < gridCount; i++) {
            embData[i * embCh + c] = (embData[i * embCh + c] - center) / scale;
        }
        for (let j = 0; j < mlpWidth1; j++) {
            const w = l1Weights[j * embCh + c];
            l1Weights[j * embCh + c] = w * scale;
            l1Biases[j] += w * center;
        }
    }
}

// 4 planes × 2 (min/max) × 4 channels = 32; matches shader struct `emb_range: array<vec4<f32>, 8>`
const EMB_RANGE_F32_SIZE = 32;

export function buildEmbRangeF32(range, embCh) {
    const f32 = new Float32Array(EMB_RANGE_F32_SIZE);
    for (let p = 0; p < 4; p++) {
        f32[p*8]=f32[p*8+1]=f32[p*8+2]=f32[p*8+3]=-1;
        f32[p*8+4]=f32[p*8+5]=f32[p*8+6]=f32[p*8+7]=1;
    }
    if (range) {
        const numPlanes = embCh / 4;
        for (let p = 0; p < numPlanes; p++) {
            for (let b = 0; b < 4; b++) {
                const c = p*4+b;
                f32[p*8+b]   = range[c*2];
                f32[p*8+4+b] = range[c*2+1];
            }
        }
    }
    return f32;
}

export function buildFwdUniforms(gSize, embCh, mlpW1, mlpW2, w, h, range, offsets) {
    const ab  = new ArrayBuffer(224);
    const u32 = new Uint32Array(ab);
    u32[0] = gSize; u32[1] = embCh; u32[2] = mlpW1; u32[3] = w; u32[4] = h; u32[6] = mlpW2;
    new Float32Array(ab, UNIFORM_OFFSET_RANGE).set(buildEmbRangeF32(range, embCh));
    if (offsets) new Float32Array(ab, UNIFORM_OFFSET_OFFSETS).set(offsets.slice(0, 16));
    return ab;
}

export function uploadEmbRange(range, embCh, buf, device) {
    device.queue.writeBuffer(buf, UNIFORM_OFFSET_RANGE, buildEmbRangeF32(range, embCh));
}

export function uploadEmbOffsets(offsets, buf, device) {
    if (!offsets) return;
    const data = new Float32Array(16);
    data.set(offsets.slice(0, 16));
    device.queue.writeBuffer(buf, UNIFORM_OFFSET_OFFSETS, data);
}

export function uploadChannelMask(mask, buf, device) {
    device.queue.writeBuffer(buf, UNIFORM_OFFSET_MASK, new Uint32Array([mask]));
}

export function cpuPackEmbeddings(embF32, embCh, range, embBits = 8) {
    const chPerGroup = 32 / embBits;
    const maxVal     = (1 << (embBits - 1)) - 1;
    const mask       = (1 << embBits) - 1;
    const numGroups  = embCh / chPerGroup;
    const gridCount  = embF32.length / embCh;
    const packed     = new Uint32Array(gridCount * numGroups);
    for (let i = 0; i < gridCount; i++) {
        for (let g = 0; g < numGroups; g++) {
            let p = 0;
            for (let b = 0; b < chPerGroup; b++) {
                const c = g * chPerGroup + b;
                const v = embF32[i * embCh + c];
                const mn = range[c * 2], mx = range[c * 2 + 1];
                const span = mx - mn;
                const s = span < 1e-9 ? 0 : Math.max(-1, Math.min(1, 2 * (v - mn) / span - 1));
                p |= (Math.max(-maxVal, Math.min(maxVal, Math.round(s * maxVal))) & mask) << (b * embBits);
            }
            packed[i * numGroups + g] = p >>> 0;
        }
    }
    return packed;
}

