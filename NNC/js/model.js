// model.js
// Creates and manages GPU weight, Adam moment, and grad accumulation buffers.
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
    const { gridSize, embeddingChannels, mlpWidth1, mlpWidth2 } = config;
    const embBits = config.embBits || 8;
    const channelsPerU32 = 32 / embBits;

    const embeddingSize = gridSize * gridSize * embeddingChannels;
    const embeddingData = new Float32Array(embeddingSize).map(() => Math.random() * 2 - 1);

    // SIREN init: first layer 1/fan_in, hidden sqrt(6/fan_in)
    const layer1Size = embeddingChannels * mlpWidth1;
    const layer1Weights = new Float32Array(layer1Size).map(() => (Math.random() * 2 - 1) / embeddingChannels);
    const layer1Biases = new Float32Array(mlpWidth1).fill(0);

    const layer2Size = mlpWidth1 * mlpWidth2;
    const s2 = Math.sqrt(6 / mlpWidth1);
    const layer2Weights = new Float32Array(layer2Size).map(() => (Math.random() * 2 - 1) * s2);
    const layer2Biases = new Float32Array(mlpWidth2).fill(0);

    const outCh = config.hasAlpha ? 4 : 3;
    const layer3Size = mlpWidth2 * outCh;
    const s3 = Math.sqrt(6 / mlpWidth2);
    const layer3Weights = new Float32Array(layer3Size).map(() => (Math.random() * 2 - 1) * s3);
    const layer3Biases = new Float32Array(outCh).fill(0);

    const sizes = {
        embeddings:     embeddingSize,
        layer1_weights: layer1Size,  layer1_biases: mlpWidth1,
        layer2_weights: layer2Size,  layer2_biases: mlpWidth2,
        layer3_weights: layer3Size,  layer3_biases: outCh,
    };
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
        embeddings_q:     ctx.zeroBuffer(embeddingSize / channelsPerU32, undefined, 'emb_q'),
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
    const { gridSize, embeddingChannels, mlpWidth1, mlpWidth2 } = config;
    const outCh = config.hasAlpha ? 4 : 3;
    const s2 = Math.sqrt(6 / mlpWidth1), s3 = Math.sqrt(6 / mlpWidth2);
    return {
        embeddings:     new Float32Array(gridSize * gridSize * embeddingChannels).map(() => Math.random() * 2 - 1),
        layer1_weights: new Float32Array(embeddingChannels * mlpWidth1).map(() => (Math.random() * 2 - 1) / embeddingChannels),
        layer1_biases:  new Float32Array(mlpWidth1),
        layer2_weights: new Float32Array(mlpWidth1 * mlpWidth2).map(() => (Math.random() * 2 - 1) * s2),
        layer2_biases:  new Float32Array(mlpWidth2),
        layer3_weights: new Float32Array(mlpWidth2 * outCh).map(() => (Math.random() * 2 - 1) * s3),
        layer3_biases:  new Float32Array(outCh),
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
