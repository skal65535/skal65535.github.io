export const MAX_LOSS_HISTORY = 400;

export function drawOutputCanvas(canvas, outputData) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const px = imageData.data;
    for (let i = 0; i < outputData.length; i += 4) {
        px[i]   = outputData[i]   * 255 | 0;
        px[i+1] = outputData[i+1] * 255 | 0;
        px[i+2] = outputData[i+2] * 255 | 0;
        px[i+3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

// All embedding channels as grayscale thumbnails; uses same bilinear sampling as forward shader.
export function drawEmbeddings(canvas, lastWeights, config) {
    if (!lastWeights?.embeddings || !config.gridSize) return;
    const { gridSize, embeddingChannels } = config;
    if (!Number.isInteger(embeddingChannels) || embeddingChannels < 1) return;
    const emb = lastWeights.embeddings;
    const w = canvas.width, h = canvas.height;
    const ctx = canvas.getContext('2d');
    const cols = Math.max(1, Math.ceil(Math.sqrt(embeddingChannels)));
    const rows = Math.max(1, Math.ceil(embeddingChannels / cols));
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    for (let screenY = 0; screenY < h; screenY++) {
        for (let screenX = 0; screenX < w; screenX++) {
            const col = Math.floor(screenX * cols / w);
            const row = Math.floor(screenY * rows / h);
            const ch  = row * cols + col;
            let v = 0;
            if (ch < embeddingChannels) {
                const localX = screenX - Math.floor(col * w / cols);
                const localY = screenY - Math.floor(row * h / rows);
                const localW = Math.floor(w / cols);
                const localH = Math.floor(h / rows);
                if (localX === 0 || localY === 0 || localX === localW - 1 || localY === localH - 1) {
                    v = 20;
                } else {
                    const gx = Math.min((localX - 1) * gridSize / Math.max(1, localW - 2) | 0, gridSize - 1);
                    const gy = Math.min((localY - 1) * gridSize / Math.max(1, localH - 2) | 0, gridSize - 1);
                    const val = emb[(gy * gridSize + gx) * embeddingChannels + ch];
                    v = Math.max(0, Math.min(255, (val * 0.5 + 0.5) * 255 | 0));
                }
            }
            const idx = (screenY * w + screenX) * 4;
            data[idx] = v; data[idx+1] = v; data[idx+2] = v; data[idx+3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

// L1/L2/L3 weight matrices as stacked grayscale bands with per-layer EMA range.
// Returns updated layerRangeEma (pass null to reset).
export function drawLayers(canvas, lastWeights, config, layerRangeEma) {
    const w = lastWeights;
    if (!w?.layer1Weights || !config.mlpWidth) return layerRangeEma;
    const { mlpWidth, embeddingChannels } = config;
    const ctx = canvas.getContext('2d');
    const cw = canvas.width, ch = canvas.height;

    const layers = [
        { data: w.layer1Weights, rows: mlpWidth, cols: embeddingChannels, label: 'L1' },
        { data: w.layer2Weights, rows: mlpWidth, cols: mlpWidth,          label: 'L2' },
        { data: w.layer3Weights, rows: 4,        cols: mlpWidth,          label: 'L3' },
    ];
    const totalRows = layers.reduce((s, l) => s + l.rows, 0);
    const bandHs    = layers.map(l => Math.round(l.rows / totalRows * ch));
    const imgData   = ctx.createImageData(cw, ch);
    const data      = imgData.data;

    const EMA_ALPHA = 0.98; // ~50-step lag
    if (!layerRangeEma) layerRangeEma = layers.map(() => ({ mn: null, mx: null }));

    let yOffset = 0;
    for (let li = 0; li < layers.length; li++) {
        const layer = layers[li], bandH = bandHs[li];
        let cmn = Infinity, cmx = -Infinity;
        for (let i = 0; i < layer.data.length; i++) {
            if (layer.data[i] < cmn) cmn = layer.data[i];
            if (layer.data[i] > cmx) cmx = layer.data[i];
        }
        const ema = layerRangeEma[li];
        if (ema.mn === null) { ema.mn = cmn; ema.mx = cmx; }
        else {
            ema.mn = EMA_ALPHA * ema.mn + (1 - EMA_ALPHA) * cmn;
            ema.mx = EMA_ALPHA * ema.mx + (1 - EMA_ALPHA) * cmx;
        }
        const mn = ema.mn, mx = ema.mx;
        const scale = mx > mn ? 255 / (mx - mn) : 1;
        for (let py = yOffset; py < yOffset + bandH; py++) {
            const row = Math.min(Math.floor((py - yOffset) / bandH * layer.rows), layer.rows - 1);
            for (let px = 0; px < cw; px++) {
                const col = Math.min(Math.floor(px / cw * layer.cols), layer.cols - 1);
                const v   = (layer.data[row * layer.cols + col] - mn) * scale | 0;
                const idx = (py * cw + px) * 4;
                data[idx] = v; data[idx+1] = v; data[idx+2] = v; data[idx+3] = 255;
            }
        }
        yOffset += bandH;
    }
    ctx.putImageData(imgData, 0, 0);

    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.font = '11px monospace';
    yOffset = 0;
    for (let li = 0; li < layers.length; li++) {
        const layer = layers[li], bandH = bandHs[li];
        ctx.fillText(layer.label, 4, yOffset + 12);
        if (yOffset > 0) ctx.fillRect(0, yOffset, cw, 1);
        yOffset += bandH;
    }

    return layerRangeEma;
}

export function drawLossCurve(canvas, lossHistory) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const hist = lossHistory.slice(-MAX_LOSS_HISTORY);
    if (hist.length < 2) return;

    const maxL  = Math.max(...hist);
    const minL  = Math.min(...hist);
    const range = maxL - minL || maxL || 1;
    const pad   = 5, n = hist.length;

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 3; i++) {
        const y = pad + (i / 3) * (h - pad * 2);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    const toX = i => (i / (n - 1)) * w;
    const toY = v => h - pad - ((v - minL) / range) * (h - pad * 2);

    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(245,166,36,0.14)');
    grad.addColorStop(1, 'rgba(245,166,36,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(toX(0), h);
    for (let i = 0; i < n; i++) ctx.lineTo(toX(i), toY(hist[i]));
    ctx.lineTo(toX(n - 1), h);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = '#f5a624';
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = 'rgba(245,166,36,0.55)';
    ctx.shadowBlur  = 5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        i === 0 ? ctx.moveTo(toX(i), toY(hist[i])) : ctx.lineTo(toX(i), toY(hist[i]));
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
}
