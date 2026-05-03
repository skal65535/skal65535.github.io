// model_io.js — safetensors save/load for nn_compress model

const TENSOR_ORDER = [
    'embeddings', 'layer1_weights', 'layer1_biases',
    'layer2_weights', 'layer2_biases', 'layer3_weights', 'layer3_biases',
];

function tensorShapes(config) {
    const { gridSize, embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = config;
    const outCh = config.hasAlpha ? 4 : 3;
    return {
        embeddings:     [gridSize * gridSize, embCh],
        layer1_weights: [mlpWidth1, embCh],
        layer1_biases:  [mlpWidth1],
        layer2_weights: [mlpWidth2, mlpWidth1],
        layer2_biases:  [mlpWidth2],
        layer3_weights: [outCh, mlpWidth2],
        layer3_biases:  [outCh],
    };
}

export function saveModelSafetensors(config, tensors) {
    const { gridSize, embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = config;
    const shapes = tensorShapes(config);

    const meta = { gridSize: String(gridSize), embeddingChannels: String(embCh), mlpWidth1: String(mlpWidth1), mlpWidth2: String(mlpWidth2) };
    for (const [k, v] of Object.entries(config)) {
        if (!['gridSize', 'embeddingChannels', 'mlpWidth1', 'mlpWidth2', 'width', 'height', 'embOffsets'].includes(k))
            meta[k] = String(v);
    }
    if (config.embOffsets) meta.emb_offsets = JSON.stringify(Array.from(config.embOffsets));

    let offset = 0;
    const headerObj = { __metadata__: meta };
    for (const k of TENSOR_ORDER) {
        const byteLen = tensors[k].byteLength;
        headerObj[k] = { dtype: 'F32', shape: shapes[k], data_offsets: [offset, offset + byteLen] };
        offset += byteLen;
    }

    // Pad header to multiple of 8 for alignment
    let headerStr = JSON.stringify(headerObj);
    headerStr += ' '.repeat((8 - (headerStr.length % 8)) % 8);
    const headerBytes = new TextEncoder().encode(headerStr);

    const buf = new ArrayBuffer(8 + headerBytes.length + offset);
    new DataView(buf).setBigUint64(0, BigInt(headerBytes.length), true);
    new Uint8Array(buf, 8, headerBytes.length).set(headerBytes);

    let dataOff = 8 + headerBytes.length;
    for (const k of TENSOR_ORDER) {
        const src = tensors[k];
        new Uint8Array(buf, dataOff, src.byteLength).set(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
        dataOff += src.byteLength;
    }
    return buf;
}

export async function loadModelSafetensors(file) {
    const ab = await file.arrayBuffer();
    const headerLen = Number(new DataView(ab).getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 8, headerLen)));

    const meta = header.__metadata__ || {};
    // Backward compat: old files have mlpWidth; new files have mlpWidth1 + mlpWidth2
    if (meta.mlpWidth && !meta.mlpWidth1)
        console.warn(`Loading legacy single-width model: mlpWidth1=mlpWidth2=${meta.mlpWidth}`);
    const mlpW1 = parseInt(meta.mlpWidth1 ?? meta.mlpWidth);
    const mlpW2 = parseInt(meta.mlpWidth2 ?? meta.mlpWidth);
    const config = {
        gridSize:          parseInt(meta.gridSize),
        embeddingChannels: parseInt(meta.embeddingChannels),
        mlpWidth1:         mlpW1,
        mlpWidth2:         mlpW2,
        embBits:           meta.embBits ? parseInt(meta.embBits) : 8,
        embOffsets:        meta.emb_offsets ? new Float32Array(JSON.parse(meta.emb_offsets)) : null,
    };
    if ([config.gridSize, config.embeddingChannels, config.mlpWidth1, config.mlpWidth2].some(isNaN))
        throw new Error('Missing or invalid config metadata');

    const dataStart = 8 + headerLen;
    const tensors = {};
    for (const k of TENSOR_ORDER) {
        if (!header[k]) throw new Error(`Missing tensor: ${k}`);
        const [start, end] = header[k].data_offsets;
        tensors[k] = new Float32Array(ab.slice(dataStart + start, dataStart + end));
    }

    const MODEL_KEYS = new Set(['gridSize', 'embeddingChannels', 'mlpWidth', 'mlpWidth1', 'mlpWidth2', 'embBits', 'emb_offsets']);
    const uiSettings = Object.fromEntries(Object.entries(meta).filter(([k]) => !MODEL_KEYS.has(k)));

    return { config, tensors, uiSettings };
}
