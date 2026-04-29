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

// Horizontal flow diagram: Emb → L1 → Act1 → L2 → Act2 → L3 → RGBA
// interLayer1/2: Float32Array [imgW*imgH*mlpWidth] or null (shown as placeholder)
// finalOutput: Float32Array [imgW*imgH*4] or null
// Returns updated ema (pass null to reset).
export function drawFlowDiagram(canvas, weights, interLayer1, interLayer2, finalOutput, imgW, imgH, config, channelMask, ema) {
    const w = weights;
    if (!w?.layer1Weights || !config.mlpWidth) return ema;
    const { mlpWidth, embeddingChannels: embCh, gridSize } = config;
    const ctx = canvas.getContext('2d');
    const cw = canvas.width, ch = canvas.height;
    const EMA_ALPHA = 0.98;

    if (!ema) ema = [{mn:null,mx:null}, {mn:null,mx:null}, {mn:null,mx:null}];

    const matDefs = [
        { data: w.layer1Weights, rows: mlpWidth, cols: embCh },
        { data: w.layer2Weights, rows: mlpWidth, cols: mlpWidth },
        { data: w.layer3Weights, rows: 4,        cols: mlpWidth },
    ];
    for (let i = 0; i < 3; i++) {
        let mn = Infinity, mx = -Infinity;
        for (const v of matDefs[i].data) { if (v < mn) mn = v; if (v > mx) mx = v; }
        const e = ema[i];
        if (e.mn === null) { e.mn = mn; e.mx = mx; }
        else { e.mn = EMA_ALPHA*e.mn + (1-EMA_ALPHA)*mn; e.mx = EMA_ALPHA*e.mx + (1-EMA_ALPHA)*mx; }
    }

    const PAD = 4, THUMB_W = 48;
    const BODY_H = ch - PAD * 2;

    // Matrices use 72% of non-thumb space; remainder goes to the 6 gaps → wider fan-line areas
    const spaceForMatsAndGaps = cw - 2*PAD - 4*THUMB_W;
    const totalMatCols = embCh + 2 * mlpWidth;
    const cell = Math.max(1, Math.min(
        Math.floor(spaceForMatsAndGaps * 0.72 / totalMatCols),
        Math.floor(BODY_H / mlpWidth)
    ));
    const totalMatWidth = totalMatCols * cell;
    const GAP = Math.max(6, Math.floor((spaceForMatsAndGaps - totalMatWidth) / 6));

    const matW_L1 = embCh * cell, matW_L2 = mlpWidth * cell, matW_L3 = mlpWidth * cell;

    const xEmb  = PAD;
    const xL1   = xEmb  + THUMB_W + GAP;
    const xAct1 = xL1   + matW_L1 + GAP;
    const xL2   = xAct1 + THUMB_W + GAP;
    const xAct2 = xL2   + matW_L2 + GAP;
    const xL3   = xAct2 + THUMB_W + GAP;
    const xRGBA = xL3   + matW_L3 + GAP;

    const imgData = ctx.createImageData(cw, ch);
    const px = imgData.data;
    for (let i = 0; i < px.length; i += 4) { px[i]=px[i+1]=px[i+2]=22; px[i+3]=255; }

    // Square-pixel weight matrix, vertically centered; returns {y0, h} for fan lines
    function drawMatrix(matData, rows, cols, x0, matW, e) {
        const matH = rows * cell;
        const oy = PAD + ((BODY_H - matH) >> 1);
        const mn = e.mn, scale = e.mx > e.mn ? 255 / (e.mx - e.mn) : 1;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const v = Math.max(0, Math.min(255, (matData[r * cols + c] - mn) * scale | 0));
                for (let dy = 0; dy < cell; dy++)
                    for (let dx = 0; dx < cell; dx++) {
                        const idx = ((oy + r*cell + dy) * cw + x0 + c*cell + dx) * 4;
                        px[idx]=v*3/4|0; px[idx+1]=v*7/8|0; px[idx+2]=v; px[idx+3]=255;
                    }
            }
        }
        return { y0: oy, h: matH };
    }

    // N channels stacked vertically; getValue(c, lx, ly, lw, lh)→float; tints[c]=[r,g,b] optional
    // Returns {y0, h} for fan lines
    function drawChannelStack(nCh, x0, w0, getValue, tints) {
        for (let c = 0; c < nCh; c++) {
            const cy0 = PAD + (c * BODY_H / nCh | 0);
            const cy1 = PAD + ((c+1) * BODY_H / nCh | 0);
            const lh = cy1 - cy0, tint = tints?.[c];
            let mn = Infinity, mx = -Infinity;
            for (let py = cy0; py < cy1; py++)
                for (let qx = x0; qx < x0+w0; qx++) {
                    const v = getValue(c, qx-x0, py-cy0, w0, lh);
                    if (v < mn) mn = v; if (v > mx) mx = v;
                }
            const sc = mx > mn ? 1 / (mx - mn) : 1;
            for (let py = cy0; py < cy1; py++)
                for (let qx = x0; qx < x0+w0; qx++) {
                    const vf  = (getValue(c, qx-x0, py-cy0, w0, lh) - mn) * sc;
                    const idx = (py * cw + qx) * 4;
                    if (tint) {
                        px[idx]=(tint[0]*vf)|0; px[idx+1]=(tint[1]*vf)|0; px[idx+2]=(tint[2]*vf)|0;
                    } else { const v=vf*255|0; px[idx]=px[idx+1]=px[idx+2]=v; }
                    px[idx+3]=255;
                }
            if (c > 0)
                for (let qx = x0; qx < x0+w0; qx++) {
                    const idx=(cy0*cw+qx)*4; px[idx]=px[idx+1]=px[idx+2]=40; px[idx+3]=255;
                }
        }
        return { y0: PAD, h: BODY_H };
    }

    function drawPlaceholder(x0, w0) {
        for (let py = PAD; py < PAD+BODY_H; py++)
            for (let qx = x0; qx < x0+w0; qx++) {
                const idx=(py*cw+qx)*4; px[idx]=px[idx+1]=px[idx+2]=30; px[idx+3]=255;
            }
        return { y0: PAD, h: BODY_H };
    }

    const embTints = Array.from({length: embCh}, (_, c) =>
        ((channelMask >>> c) & 1) ? null : [200, 40, 40]);

    const bbEmb = w.embeddings
        ? drawChannelStack(embCh, xEmb, THUMB_W, (c, lx, ly, lw, lh) => {
            const gx = Math.min(lx*gridSize/lw|0, gridSize-1);
            const gy = Math.min(ly*gridSize/lh|0, gridSize-1);
            return w.embeddings[(gy*gridSize+gx)*embCh+c];
          }, embTints)
        : drawPlaceholder(xEmb, THUMB_W);

    const bbL1 = drawMatrix(w.layer1Weights, mlpWidth, embCh,   xL1, matW_L1, ema[0]);
    const bbL2 = drawMatrix(w.layer2Weights, mlpWidth, mlpWidth, xL2, matW_L2, ema[1]);
    const bbL3 = drawMatrix(w.layer3Weights, 4,        mlpWidth, xL3, matW_L3, ema[2]);

    const bbAct1 = interLayer1
        ? drawChannelStack(mlpWidth, xAct1, THUMB_W, (c, lx, ly, lw, lh) => {
            const sx = Math.min(lx*imgW/lw|0, imgW-1), sy = Math.min(ly*imgH/lh|0, imgH-1);
            return interLayer1[(sy*imgW+sx)*mlpWidth+c];
          }, null)
        : drawPlaceholder(xAct1, THUMB_W);

    const bbAct2 = interLayer2
        ? drawChannelStack(mlpWidth, xAct2, THUMB_W, (c, lx, ly, lw, lh) => {
            const sx = Math.min(lx*imgW/lw|0, imgW-1), sy = Math.min(ly*imgH/lh|0, imgH-1);
            return interLayer2[(sy*imgW+sx)*mlpWidth+c];
          }, null)
        : drawPlaceholder(xAct2, THUMB_W);

    const bbRGBA = finalOutput
        ? drawChannelStack(4, xRGBA, THUMB_W, (c, lx, ly, lw, lh) => {
            const sx = Math.min(lx*imgW/lw|0, imgW-1), sy = Math.min(ly*imgH/lh|0, imgH-1);
            return finalOutput[(sy*imgW+sx)*4+c];
          }, [[255,80,80],[80,220,80],[80,120,255],[180,180,180]])
        : drawPlaceholder(xRGBA, THUMB_W);

    ctx.putImageData(imgData, 0, 0);

    // Fan-out lines between adjacent blocks
    function fanLines(x1, src, x2, dst, n) {
        ctx.beginPath();
        const nLines = Math.min(n, 24);
        for (let i = 0; i < nLines; i++) {
            const t = (i + 0.5) / nLines;
            ctx.moveTo(x1, src.y0 + t * src.h);
            ctx.lineTo(x2, dst.y0 + t * dst.h);
        }
        ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(255,210,0,0.75)';
    ctx.lineWidth = 0.8;
    fanLines(xEmb + THUMB_W,      bbEmb,  xL1,            bbL1,  embCh);
    fanLines(xL1  + matW_L1,      bbL1,   xAct1,          bbAct1, mlpWidth);
    fanLines(xAct1 + THUMB_W,     bbAct1, xL2,            bbL2,  mlpWidth);
    fanLines(xL2  + matW_L2,      bbL2,   xAct2,          bbAct2, mlpWidth);
    fanLines(xAct2 + THUMB_W,     bbAct2, xL3,            bbL3,  mlpWidth);
    fanLines(xL3  + matW_L3,      bbL3,   xRGBA,          bbRGBA, 4);

    // Labels
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(200,200,200,0.8)';
    const labels = [
        ['Emb',     xEmb,  THUMB_W], ['Layer 1', xL1,  matW_L1], ['Act1', xAct1, THUMB_W],
        ['Layer 2', xL2,   matW_L2], ['Act2',   xAct2, THUMB_W], ['Layer 3', xL3, matW_L3],
        ['RGBA',xRGBA, THUMB_W],
    ];
    for (const [lbl, lx, lw] of labels) ctx.fillText(lbl, lx + lw/2, PAD + 10);

    return ema;
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
