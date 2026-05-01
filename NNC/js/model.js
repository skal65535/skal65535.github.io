// model.js
// Creates and manages GPU weight, Adam moment, and grad accumulation buffers.
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
    const zeroTensors = () => ModelTensors.create(k => ctx.zeroBuffer(sizes[k]));

    const numU32 = embeddingChannels / channelsPerU32;
    const embOffsets = config.embOffsets || new Float32Array(numU32 * 2);

    const modelBuffers = {
        embeddings:       ctx.createBuffer(embeddingData),
        embeddings_q:     ctx.zeroBuffer(embeddingSize / channelsPerU32),  // packed u32: channelsPerU32 per u32
        embeddings_range: ctx.zeroBuffer(embeddingChannels * 2), // [min, max] per channel (f32)
        emb_offsets:      ctx.createBuffer(embOffsets),
        layer1: { weights: ctx.createBuffer(layer1Weights), biases: ctx.createBuffer(layer1Biases) },
        layer2: { weights: ctx.createBuffer(layer2Weights), biases: ctx.createBuffer(layer2Biases) },
        layer3: { weights: ctx.createBuffer(layer3Weights), biases: ctx.createBuffer(layer3Biases) },
        adamM:      zeroTensors(),
        adamV:      zeroTensors(),
        gradAtomic: zeroTensors(),
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

export function destroyModel(m) {
    if (!m) return;
    m.embeddings?.destroy();
    m.embeddings_q?.destroy();
    m.embeddings_range?.destroy();
    m.emb_offsets?.destroy();
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
    const layerMap = {
        layer1_weights: model.layer1.weights, layer1_biases: model.layer1.biases,
        layer2_weights: model.layer2.weights, layer2_biases: model.layer2.biases,
        layer3_weights: model.layer3.weights, layer3_biases: model.layer3.biases,
    };
    for (const [k, buf] of Object.entries(layerMap)) {
        const w = rb[k];
        for (let i = 0; i < w.length; i++) w[i] += (Math.random() * 2 - 1) * SHAKE_MLP_AMPLITUDE;
        ctx.writeBuffer(buf, w);
        ctx.clearBuffer(model.adamM[k]);
        ctx.clearBuffer(model.adamV[k]);
    }
}
