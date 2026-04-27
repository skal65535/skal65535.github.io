// model_io.js — safetensors save/load for nn_compress model

const TENSOR_ORDER = [
    'embeddings', 'layer1_weights', 'layer1_biases',
    'layer2_weights', 'layer2_biases', 'layer3_weights', 'layer3_biases',
];

function tensorShapes(config) {
    const { gridSize, embeddingChannels: embCh, mlpWidth } = config;
    return {
        embeddings:     [gridSize * gridSize, embCh],
        layer1_weights: [mlpWidth, embCh],
        layer1_biases:  [mlpWidth],
        layer2_weights: [mlpWidth, mlpWidth],
        layer2_biases:  [mlpWidth],
        layer3_weights: [4, mlpWidth],
        layer3_biases:  [4],
    };
}

export function saveModelSafetensors(config, tensors) {
    const { gridSize, embeddingChannels: embCh, mlpWidth } = config;
    const shapes = tensorShapes(config);

    const meta = { gridSize: String(gridSize), embeddingChannels: String(embCh), mlpWidth: String(mlpWidth) };
    for (const [k, v] of Object.entries(config)) {
        if (!['gridSize', 'embeddingChannels', 'mlpWidth', 'width', 'height'].includes(k))
            meta[k] = String(v);
    }

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
    const config = {
        gridSize:          parseInt(meta.gridSize),
        embeddingChannels: parseInt(meta.embeddingChannels),
        mlpWidth:          parseInt(meta.mlpWidth),
    };
    if ([config.gridSize, config.embeddingChannels, config.mlpWidth].some(isNaN))
        throw new Error('Missing or invalid config metadata');

    const dataStart = 8 + headerLen;
    const tensors = {};
    for (const k of TENSOR_ORDER) {
        if (!header[k]) throw new Error(`Missing tensor: ${k}`);
        const [start, end] = header[k].data_offsets;
        tensors[k] = new Float32Array(ab.slice(dataStart + start, dataStart + end));
    }

    const MODEL_KEYS = new Set(['gridSize', 'embeddingChannels', 'mlpWidth']);
    const uiSettings = Object.fromEntries(Object.entries(meta).filter(([k]) => !MODEL_KEYS.has(k)));

    return { config, tensors, uiSettings };
}
