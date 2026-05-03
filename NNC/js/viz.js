// viz.js
// Canvas rendering helpers for output image, embedding thumbnails, and loss curve.
export const MAX_LOSS_HISTORY = 400;
export const FLOW_PAD = 4, FLOW_EMB_W = 48;


export function drawOutputCanvas(canvas, outputData, hasAlpha = true) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const px = imageData.data;
    for (let i = 0; i < outputData.length; i += 4) {
        px[i]   = outputData[i]   * 255 | 0;
        px[i+1] = outputData[i+1] * 255 | 0;
        px[i+2] = outputData[i+2] * 255 | 0;
        px[i+3] = hasAlpha ? (outputData[i+3] * 255 | 255) : 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

// Horizontal flow diagram: Emb → L1 → Act1 → L2 → Act2 → L3 → RGBA
// interLayer1/2: Float32Array [imgW*imgH*mlpWidth] or null (shown as placeholder)
// finalOutput: Float32Array [imgW*imgH*4] or null
// Returns updated ema (pass null to reset).
export function drawFlowDiagram(canvas, { weights, inter1: interLayer1, inter2: interLayer2, finalOutput, imgW, imgH, config, channelMask, ema, hoverState = null }) {
    const w = weights;
    if (!w || !config.mlpWidth1) return ema;
    const { mlpWidth1, mlpWidth2, embeddingChannels: embCh, gW, gH } = config;
    const outCh = config.hasAlpha ? 4 : 3;
    const ctx = canvas.getContext('2d');
    const cw = canvas.width, ch = canvas.height;
    const EMA_ALPHA = 0.98;

    if (!ema) ema = [{mn:null,mx:null}, {mn:null,mx:null}, {mn:null,mx:null}];

    const matDefs = [
        { data: w.layer1_weights, rows: mlpWidth1, cols: embCh },
        { data: w.layer2_weights, rows: mlpWidth2, cols: mlpWidth1 },
        { data: w.layer3_weights, rows: outCh,      cols: mlpWidth2 },
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
    const totalMatCols = embCh + mlpWidth1 + mlpWidth2;
    const cell = Math.max(1, Math.min(
        Math.floor(spaceForMatsAndGaps * 0.72 / totalMatCols),
        Math.floor(BODY_H / Math.max(mlpWidth1, mlpWidth2))
    ));
    const totalMatWidth = totalMatCols * cell;
    const GAP = Math.max(6, Math.floor((spaceForMatsAndGaps - totalMatWidth) / 6));

    const matW_L1 = embCh * cell, matW_L2 = mlpWidth1 * cell, matW_L3 = mlpWidth2 * cell;

    const xEmb  = PAD;
    const xL1   = xEmb  + THUMB_W + GAP;
    const xAct1 = xL1   + matW_L1 + GAP;
    const xL2   = xAct1 + THUMB_W + GAP;
    const xAct2 = xL2   + matW_L2 + GAP;
    const xL3   = xAct2 + THUMB_W + GAP;
    const xRGBA = xL3   + matW_L3 + GAP;
    const xZoom = cw - PAD - ZOOM_SIZE;

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const imgData = ctx.createImageData(cw, ch);
    const px = imgData.data;
    const bgV = isLight ? 220 : 0;
    for (let i = 0; i < px.length; i += 4) { px[i]=bgV; px[i+1]=bgV; px[i+2]=bgV; px[i+3]=255; }

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
        const ph = (isLight ? bgV - 20 : bgV + 30) | 0;
        for (let py = PAD; py < PAD+BODY_H; py++)
            for (let qx = x0; qx < x0+w0; qx++) {
                const idx=(py*cw+qx)*4; px[idx]=px[idx+1]=px[idx+2]=ph; px[idx+3]=255;
            }
        return { y0: PAD, h: BODY_H };
    }

    const embTints = Array.from({length: embCh}, (_, c) =>
        ((channelMask >>> c) & 1) ? [191, 223, 255] : [200, 40, 40]);

    const embCellH = gW > 0 ? Math.round(THUMB_W * gH / gW) : (BODY_H / embCh | 0);
    const bbEmb = w.embeddings
        ? drawChannelStack(embCh, xEmb, THUMB_W, (c, lx, ly, lw, lh) => {
            const gx = Math.min(lx*gW/lw|0, gW-1);
            const gy = Math.min(ly*gH/lh|0, gH-1);
            return w.embeddings[(gy*gW+gx)*embCh+c];
          }, embTints, embCellH)
        : drawPlaceholder(xEmb, THUMB_W);

    const aspectCellH = imgW > 0 ? Math.round(THUMB_W * imgH / imgW) : BODY_H;
    const maxActCh = Math.max(mlpWidth1, mlpWidth2, outCh);
    const actCellH = Math.min(aspectCellH, BODY_H / maxActCh | 0);

    const bbL1 = drawMatrix(w.layer1_weights, mlpWidth1, embCh,    xL1, matW_L1, ema[0]);
    const bbL2 = drawMatrix(w.layer2_weights, mlpWidth2, mlpWidth1, xL2, matW_L2, ema[1]);
    const bbL3 = drawMatrix(w.layer3_weights, outCh,      mlpWidth2, xL3, matW_L3, ema[2]);

    const bbAct1 = interLayer1
        ? drawChannelStack(mlpWidth1, xAct1, THUMB_W, (c, lx, ly, lw, lh) => {
            const sx = Math.min(lx*imgW/lw|0, imgW-1), sy = Math.min(ly*imgH/lh|0, imgH-1);
            return interLayer1[(sy*imgW+sx)*mlpWidth1+c];
          }, null, actCellH)
        : drawPlaceholder(xAct1, THUMB_W);

    const bbAct2 = interLayer2
        ? drawChannelStack(mlpWidth2, xAct2, THUMB_W, (c, lx, ly, lw, lh) => {
            const sx = Math.min(lx*imgW/lw|0, imgW-1), sy = Math.min(ly*imgH/lh|0, imgH-1);
            return interLayer2[(sy*imgW+sx)*mlpWidth2+c];
          }, null, actCellH)
        : drawPlaceholder(xAct2, THUMB_W);

    const rgbaTints = [[255,80,80],[80,220,80],[80,120,255],[180,180,180]].slice(0, outCh);
    const bbRGBA = finalOutput
        ? drawChannelStack(outCh, xRGBA, THUMB_W, (c, lx, ly, lw, lh) => {
            const sx = Math.min(lx*imgW/lw|0, imgW-1), sy = Math.min(ly*imgH/lh|0, imgH-1);
            return finalOutput[(sy*imgW+sx)*4+c];
          }, rgbaTints, Math.min(aspectCellH, BODY_H / outCh | 0))
        : drawPlaceholder(xRGBA, THUMB_W);

    let zoomBounds = null;
    if (hoverState) {
        const hCol = hoverState.col, hCh = hoverState.ch;
        const xZ = xZoom, yZ = ch - PAD - ZOOM_SIZE;
        const fillZoom = (srcW, srcH, getVal, tint) => {
            const scale = Math.min(ZOOM_SIZE / srcW, ZOOM_SIZE / srcH);
            const rW = Math.round(srcW * scale), rH = Math.round(srcH * scale);
            const rox = (ZOOM_SIZE - rW) >> 1, roy = (ZOOM_SIZE - rH) >> 1;
            const inv_rW = srcW / rW, inv_rH = srcH / rH;
            let mn = Infinity, mx = -Infinity;
            for (let py = 0; py < rH; py++)
                for (let qx = 0; qx < rW; qx++) {
                    const v = getVal(qx * inv_rW | 0, py * inv_rH | 0);
                    if (v < mn) mn = v; if (v > mx) mx = v;
                }
            const zsc = mx > mn ? 1 / (mx - mn) : 1;
            for (let py = 0; py < rH; py++)
                for (let qx = 0; qx < rW; qx++) {
                    const vf = (getVal(qx * inv_rW | 0, py * inv_rH | 0) - mn) * zsc;
                    const idx = ((yZ + roy + py) * cw + xZ + rox + qx) * 4;
                    if (tint) {
                        px[idx]=(tint[0]*vf)|0; px[idx+1]=(tint[1]*vf)|0; px[idx+2]=(tint[2]*vf)|0;
                    } else { const v=vf*255|0; px[idx]=px[idx+1]=px[idx+2]=v; }
                    px[idx+3] = 255;
                }
            zoomBounds = { xZ, yZ, rox, roy, rW, rH };
        };
        if      (hCol==='emb'  && w.embeddings)  fillZoom(gW, gH,
            (sx,sy) => w.embeddings[(sy*gW+sx)*embCh+hCh], [191,223,255]);
        else if (hCol==='l1')                     fillZoom(embCh, mlpWidth1,
            (sx,sy) => w.layer1_weights[sy*embCh+sx], [191,223,255]);
        else if (hCol==='act1' && interLayer1)    fillZoom(imgW, imgH,
            (sx,sy) => interLayer1[(sy*imgW+sx)*mlpWidth1+hCh], null);
        else if (hCol==='l2')                     fillZoom(mlpWidth1, mlpWidth2,
            (sx,sy) => w.layer2_weights[sy*mlpWidth1+sx], [191,223,255]);
        else if (hCol==='act2' && interLayer2)    fillZoom(imgW, imgH,
            (sx,sy) => interLayer2[(sy*imgW+sx)*mlpWidth2+hCh], null);
        else if (hCol==='l3')                     fillZoom(mlpWidth2, outCh,
            (sx,sy) => w.layer3_weights[sy*mlpWidth2+sx], [191,223,255]);
        else if (hCol==='rgba' && finalOutput)    fillZoom(imgW, imgH,
            (sx,sy) => finalOutput[(sy*imgW+sx)*4+hCh],
            [[255,80,80],[80,220,80],[80,120,255],[180,180,180]][hCh]);
    }

    ctx.putImageData(imgData, 0, 0);

    // Theme-aware colors for labels and highlights
    const labelDefault = isLight ? 'rgba(30,50,80,0.95)' : 'rgba(191,223,255,0.95)';
    const labelBg      = isLight ? 'rgba(255,255,255,0.75)' : 'rgba(0,0,0,0.6)';
    const labelBlue    = isLight ? 'rgba(42,112,192,0.95)' : 'rgba(100,160,255,0.9)';
    const highlightBox = isLight ? 'rgba(42,112,192,0.85)' : 'rgba(191,223,255,0.85)';

    // Draws text with a background pill; cx/cy is the text anchor point.
    function drawLabel(text, cx, cy, font = '10px monospace', align = 'center', color = labelDefault) {
        ctx.font = font;
        ctx.textAlign = align;
        const tw = ctx.measureText(text).width;
        const ox = align === 'center' ? tw / 2 : align === 'right' ? tw : 0;
        ctx.fillStyle = labelBg;
        ctx.fillRect(cx - ox - 2, cy - 9, tw + 4, 12);
        ctx.fillStyle = color;
        ctx.fillText(text, cx, cy);
    }

    if (hoverState) {
        const hCol = hoverState.col, hCh = hoverState.ch;
        ctx.strokeStyle = highlightBox;
        ctx.lineWidth = 1.5;
        const hilightStack = (bb, nCh, x0, w0) => {
            const slotH = bb.h / nCh;
            ctx.strokeRect(x0 + 0.5, bb.y0 + hCh * slotH + 0.5, w0 - 1, slotH - 1);
        };
        const hilightMat = (bb, x0, w0) => ctx.strokeRect(x0 + 0.5, bb.y0 + 0.5, w0 - 1, bb.h - 1);
        switch (hCol) {
            case 'emb':  hilightStack(bbEmb,  embCh,    xEmb,  THUMB_W); break;
            case 'l1':   hilightMat  (bbL1,             xL1,   matW_L1); break;
            case 'act1': hilightStack(bbAct1, mlpWidth1, xAct1, THUMB_W); break;
            case 'l2':   hilightMat  (bbL2,              xL2,   matW_L2); break;
            case 'act2': hilightStack(bbAct2, mlpWidth2, xAct2, THUMB_W); break;
            case 'l3':   hilightMat  (bbL3,             xL3,   matW_L3); break;
            case 'rgba': hilightStack(bbRGBA, outCh,    xRGBA, THUMB_W); break;
        }
        if (zoomBounds) {
            const { xZ, yZ, rox, roy, rW, rH } = zoomBounds;
            const zoomLabel = ({ emb:`emb ch ${hCh}`, l1:'Layer 1', act1:`act1 ch ${hCh}`,
                l2:'Layer 2', act2:`act2 ch ${hCh}`, l3:'Layer 3',
                rgba:['R','G','B','α'].slice(0,outCh)[hCh] })[hCol] ?? '';
            ctx.strokeStyle = isLight ? 'rgba(42,112,192,0.7)' : 'rgba(191,223,255,0.7)';
            ctx.lineWidth = 1;
            ctx.strokeRect(xZ + rox - 0.5, yZ + roy - 0.5, rW + 1, rH + 1);
            drawLabel(zoomLabel, xZ + rox + 4, yZ + roy + 10, '9px monospace', 'left');
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
    ctx.strokeStyle = isLight ? 'rgba(40,60,100,0.65)' : 'rgba(255,210,0,0.75)';
    ctx.lineWidth = 0.8;
    fanLines(xEmb + THUMB_W,      bbEmb,  xL1,            bbL1,   embCh);
    fanLines(xL1  + matW_L1,      bbL1,   xAct1,          bbAct1, mlpWidth1);
    fanLines(xAct1 + THUMB_W,     bbAct1, xL2,            bbL2,   mlpWidth1);
    fanLines(xL2  + matW_L2,      bbL2,   xAct2,          bbAct2, mlpWidth2);
    fanLines(xAct2 + THUMB_W,     bbAct2, xL3,            bbL3,   mlpWidth2);
    fanLines(xL3  + matW_L3,      bbL3,   xRGBA,          bbRGBA, outCh);

    // Labels at top for non-matrix columns
    for (const [lbl, lx, lw] of [
        ['Emb', xEmb, THUMB_W], ['Act1', xAct1, THUMB_W], ['Act2', xAct2, THUMB_W],
    ]) drawLabel(lbl, lx + lw/2, PAD + 10);
    drawLabel(config.hasAlpha ? 'RGBA' : 'RGB', xRGBA + THUMB_W/2, bbRGBA.y0 - 16);

    // Learnt / inferred legend — stacked above RGBA column
    drawLabel('■ learnt',   xRGBA + THUMB_W/2, PAD + 10, '9px monospace', 'center', labelBlue);
    drawLabel('■ inferred', xRGBA + THUMB_W/2, PAD + 22, '9px monospace', 'center', labelDefault);

    // Labels just above each weight matrix: name + channel count, in blue
    for (const [name, ch, lx, lw, my0] of [
        ['Layer 1', `${embCh}→${mlpWidth1}`,     xL1, matW_L1, bbL1.y0],
        ['Layer 2', `${mlpWidth1}→${mlpWidth2}`,  xL2, matW_L2, bbL2.y0],
        ['Layer 3', `${mlpWidth2}→${outCh}`,      xL3, matW_L3, bbL3.y0],
    ]) {
        drawLabel(name, lx + lw/2, my0 - 28, '10px monospace', 'center', labelBlue);
        drawLabel(ch,   lx + lw/2, my0 - 16, '13px monospace', 'center', labelBlue);
    }

    return {
        ema,
        cols: [
            { name:'emb',  x:xEmb,  w:THUMB_W, y0:bbEmb.y0,  h:bbEmb.h,  slotH:bbEmb.h/embCh,     nCh:embCh    },
            { name:'l1',   x:xL1,   w:matW_L1, y0:bbL1.y0,   h:bbL1.h,   slotH:null,               nCh:mlpWidth1 },
            { name:'act1', x:xAct1, w:THUMB_W, y0:bbAct1.y0, h:bbAct1.h, slotH:bbAct1.h/mlpWidth1, nCh:mlpWidth1 },
            { name:'l2',   x:xL2,   w:matW_L2, y0:bbL2.y0,   h:bbL2.h,   slotH:null,               nCh:mlpWidth2 },
            { name:'act2', x:xAct2, w:THUMB_W, y0:bbAct2.y0, h:bbAct2.h, slotH:bbAct2.h/mlpWidth2, nCh:mlpWidth2 },
            { name:'l3',   x:xL3,   w:matW_L3, y0:bbL3.y0,   h:bbL3.h,   slotH:null,               nCh:outCh    },
            { name:'rgba', x:xRGBA, w:THUMB_W, y0:bbRGBA.y0, h:bbRGBA.h, slotH:bbRGBA.h/outCh,     nCh:outCh    },
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

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const accentColor = isLight ? '#c07800' : '#f5a624';
    const accentRgb   = isLight ? '192,120,0' : '245,166,36';

    ctx.strokeStyle = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.04)';
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 3; i++) {
        const y = pad + (i / 3) * (h - pad * 2);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    const toX = i => (i / (n - 1)) * w;
    const toY = v => h - pad - ((v - minL) / range) * (h - pad * 2);

    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, `rgba(${accentRgb},0.15)`);
    grad.addColorStop(1, `rgba(${accentRgb},0)`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(toX(0), h);
    for (let i = 0; i < n; i++) ctx.lineTo(toX(i), toY(hist[i]));
    ctx.lineTo(toX(n - 1), h);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = accentColor;
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = `rgba(${accentRgb},${isLight ? 0.35 : 0.55})`;
    ctx.shadowBlur  = 5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const X = toX(i), Y = toY(hist[i]);
        i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
}
