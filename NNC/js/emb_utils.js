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

// Normalize embeddings to [-1,1] per channel; absorb scale+center into L1 weights/biases.
// Math: e_norm = (e - center) / scale → W1_new[:,c] = W1[:,c]*scale, b1_new += W1[:,c]*center
export function normalizeEmbAndAdjustL1(embData, l1Weights, l1Biases, embCh, mlpWidth) {
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
        for (let j = 0; j < mlpWidth; j++) {
            const w = l1Weights[j * embCh + c];
            l1Weights[j * embCh + c] = w * scale;
            l1Biases[j] += w * center;
        }
    }
}

// Builds 32-float emb_range block: identity [-1,1] defaults, overwritten with actual range.
export function buildEmbRangeF32(range, embCh) {
    const f32 = new Float32Array(32);
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

// Build 160-byte ArrayBuffer for the forward Uniforms struct.
// Layout: 5 u32 + 3 u32 padding + 8×vec4<f32> emb_range (offset 32).
export function buildFwdUniforms(gSize, embCh, mlpW, w, h, range) {
    const ab  = new ArrayBuffer(160);
    const u32 = new Uint32Array(ab);
    u32[0] = gSize; u32[1] = embCh; u32[2] = mlpW; u32[3] = w; u32[4] = h;
    new Float32Array(ab, 32).set(buildEmbRangeF32(range, embCh));
    return ab;
}

// Update only the emb_range portion of fwdUniformsBuf (byte offset 32).
export function uploadEmbRange(range, embCh, buf, device) {
    device.queue.writeBuffer(buf, 32, buildEmbRangeF32(range, embCh));
}

// Update only the channelMask field of fwdUniformsBuf (byte offset 20, u32[5]).
export function uploadChannelMask(mask, buf, device) {
    device.queue.writeBuffer(buf, 20, new Uint32Array([mask]));
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
