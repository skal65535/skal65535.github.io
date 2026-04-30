// viz.js
// Canvas rendering helpers for output image, embedding thumbnails, and loss curve.
export const MAX_LOSS_HISTORY = 400;
export const FLOW_PAD = 4, FLOW_EMB_W = 48;

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

// Horizontal flow diagram: Emb → L1 → Act1 → L2 → Act2 → L3 → RGBA
// interLayer1/2: Float32Array [imgW*imgH*mlpWidth] or null (shown as placeholder)
// finalOutput: Float32Array [imgW*imgH*4] or null
// Returns updated ema (pass null to reset).
export function drawFlowDiagram(canvas, weights, interLayer1, interLayer2, finalOutput, imgW, imgH, config, channelMask, ema, hoverState = null) {
    const w = weights;
    if (!w || !config.mlpWidth) return ema;
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

    const PAD = FLOW_PAD, THUMB_W = FLOW_EMB_W, ZOOM_SIZE = 140;
    const BODY_H = ch - PAD * 2;

    // Matrices use 72% of non-thumb space; zoom inset overlays the empty lower-right corner
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
    const xZoom = cw - PAD - ZOOM_SIZE;

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
    // cellH: fixed per-channel height (aspect-ratio mode); null → divide BODY_H equally
    // Returns {y0, h} for fan lines
    function drawChannelStack(nCh, x0, w0, getValue, tints, cellH) {
        const slotH = cellH != null ? cellH : (BODY_H / nCh | 0);
        const totalH = nCh * slotH;
        const baseY  = PAD + ((BODY_H - totalH) >> 1);
        for (let c = 0; c < nCh; c++) {
            const cy0 = baseY + c * slotH;
            const cy1 = cy0 + slotH;
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
        return { y0: baseY, h: totalH };
    }

    function drawPlaceholder(x0, w0) {
        for (let py = PAD; py < PAD+BODY_H; py++)
            for (let qx = x0; qx < x0+w0; qx++) {
                const idx=(py*cw+qx)*4; px[idx]=px[idx+1]=px[idx+2]=30; px[idx+3]=255;
            }
        return { y0: PAD, h: BODY_H };
    }

    const embTints = Array.from({length: embCh}, (_, c) =>
        ((channelMask >>> c) & 1) ? [191, 223, 255] : [200, 40, 40]);

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

    const rgbaCellH = imgW > 0 ? Math.min(Math.round(THUMB_W * imgH / imgW), BODY_H >> 2) : null;
    const bbRGBA = finalOutput
        ? drawChannelStack(4, xRGBA, THUMB_W, (c, lx, ly, lw, lh) => {
            const sx = Math.min(lx*imgW/lw|0, imgW-1), sy = Math.min(ly*imgH/lh|0, imgH-1);
            return finalOutput[(sy*imgW+sx)*4+c];
          }, [[255,80,80],[80,220,80],[80,120,255],[180,180,180]], rgbaCellH)
        : drawPlaceholder(xRGBA, THUMB_W);

    let zoomBounds = null;
    if (hoverState) {
        const hCol = hoverState.col, hCh = hoverState.ch;
        const xZ = xZoom, yZ = ch - PAD - ZOOM_SIZE;
        const fillZoom = (srcW, srcH, getVal, tint) => {
            const scale = Math.min(ZOOM_SIZE / srcW, ZOOM_SIZE / srcH);
            const rW = Math.round(srcW * scale), rH = Math.round(srcH * scale);
            const rox = (ZOOM_SIZE - rW) >> 1, roy = (ZOOM_SIZE - rH) >> 1;
            let mn = Infinity, mx = -Infinity;
            for (let py = 0; py < rH; py++)
                for (let qx = 0; qx < rW; qx++) {
                    const v = getVal(qx * srcW / rW | 0, py * srcH / rH | 0);
                    if (v < mn) mn = v; if (v > mx) mx = v;
                }
            const zsc = mx > mn ? 1 / (mx - mn) : 1;
            for (let py = 0; py < rH; py++)
                for (let qx = 0; qx < rW; qx++) {
                    const vf = (getVal(qx * srcW / rW | 0, py * srcH / rH | 0) - mn) * zsc;
                    const idx = ((yZ + roy + py) * cw + xZ + rox + qx) * 4;
                    if (tint) {
                        px[idx]=(tint[0]*vf)|0; px[idx+1]=(tint[1]*vf)|0; px[idx+2]=(tint[2]*vf)|0;
                    } else { const v=vf*255|0; px[idx]=px[idx+1]=px[idx+2]=v; }
                    px[idx+3] = 255;
                }
            zoomBounds = { xZ, yZ, rox, roy, rW, rH };
        };
        if      (hCol==='emb'  && w.embeddings)  fillZoom(gridSize, gridSize,
            (sx,sy) => w.embeddings[(sy*gridSize+sx)*embCh+hCh], [191,223,255]);
        else if (hCol==='l1')                     fillZoom(embCh, mlpWidth,
            (sx,sy) => w.layer1Weights[sy*embCh+sx], [191,223,255]);
        else if (hCol==='act1' && interLayer1)    fillZoom(imgW, imgH,
            (sx,sy) => interLayer1[(sy*imgW+sx)*mlpWidth+hCh], null);
        else if (hCol==='l2')                     fillZoom(mlpWidth, mlpWidth,
            (sx,sy) => w.layer2Weights[sy*mlpWidth+sx], [191,223,255]);
        else if (hCol==='act2' && interLayer2)    fillZoom(imgW, imgH,
            (sx,sy) => interLayer2[(sy*imgW+sx)*mlpWidth+hCh], null);
        else if (hCol==='l3')                     fillZoom(mlpWidth, 4,
            (sx,sy) => w.layer3Weights[sy*mlpWidth+sx], [191,223,255]);
        else if (hCol==='rgba' && finalOutput)    fillZoom(imgW, imgH,
            (sx,sy) => finalOutput[(sy*imgW+sx)*4+hCh],
            [[255,80,80],[80,220,80],[80,120,255],[180,180,180]][hCh]);
    }

    ctx.putImageData(imgData, 0, 0);

    if (hoverState) {
        const hCol = hoverState.col, hCh = hoverState.ch;
        ctx.strokeStyle = 'rgba(191,223,255,0.85)';
        ctx.lineWidth = 1.5;
        const hilightStack = (bb, nCh, x0, w0) => {
            const slotH = bb.h / nCh;
            ctx.strokeRect(x0 + 0.5, bb.y0 + hCh * slotH + 0.5, w0 - 1, slotH - 1);
        };
        const hilightMat = (bb, x0, w0) => ctx.strokeRect(x0 + 0.5, bb.y0 + 0.5, w0 - 1, bb.h - 1);
        switch (hCol) {
            case 'emb':  hilightStack(bbEmb,  embCh,    xEmb,  THUMB_W); break;
            case 'l1':   hilightMat  (bbL1,             xL1,   matW_L1); break;
            case 'act1': hilightStack(bbAct1, mlpWidth, xAct1, THUMB_W); break;
            case 'l2':   hilightMat  (bbL2,             xL2,   matW_L2); break;
            case 'act2': hilightStack(bbAct2, mlpWidth, xAct2, THUMB_W); break;
            case 'l3':   hilightMat  (bbL3,             xL3,   matW_L3); break;
            case 'rgba': hilightStack(bbRGBA, 4,        xRGBA, THUMB_W); break;
        }
        if (zoomBounds) {
            const { xZ, yZ, rox, roy, rW, rH } = zoomBounds;
            const zoomLabel = ({ emb:`emb ch ${hCh}`, l1:'Layer 1', act1:`act1 ch ${hCh}`,
                l2:'Layer 2', act2:`act2 ch ${hCh}`, l3:'Layer 3',
                rgba:['R','G','B','α'][hCh] })[hCol] ?? '';
            ctx.strokeStyle = 'rgba(191,223,255,0.7)';
            ctx.lineWidth = 1;
            ctx.strokeRect(xZ + rox - 0.5, yZ + roy - 0.5, rW + 1, rH + 1);
            ctx.font = '9px monospace';
            ctx.textAlign = 'left';
            const tw = ctx.measureText(zoomLabel).width;
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(xZ + rox + 2, yZ + roy + 1, tw + 4, 12);
            ctx.fillStyle = 'rgba(191,223,255,0.95)';
            ctx.fillText(zoomLabel, xZ + rox + 4, yZ + roy + 10);
        }
    }

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

    return {
        ema,
        cols: [
            { name:'emb',  x:xEmb,  w:THUMB_W, y0:bbEmb.y0,  slotH:bbEmb.h/embCh,    nCh:embCh    },
            { name:'l1',   x:xL1,   w:matW_L1, y0:bbL1.y0,   slotH:null,              nCh:mlpWidth },
            { name:'act1', x:xAct1, w:THUMB_W, y0:bbAct1.y0, slotH:bbAct1.h/mlpWidth, nCh:mlpWidth },
            { name:'l2',   x:xL2,   w:matW_L2, y0:bbL2.y0,   slotH:null,              nCh:mlpWidth },
            { name:'act2', x:xAct2, w:THUMB_W, y0:bbAct2.y0, slotH:bbAct2.h/mlpWidth, nCh:mlpWidth },
            { name:'l3',   x:xL3,   w:matW_L3, y0:bbL3.y0,   slotH:null,              nCh:4        },
            { name:'rgba', x:xRGBA, w:THUMB_W, y0:bbRGBA.y0, slotH:bbRGBA.h/4,        nCh:4        },
        ],
    };
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
