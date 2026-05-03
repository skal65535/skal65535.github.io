import { GpuSession, CpuSession } from "./session.js";
import { initWebGPU, buildShader, buildBackwardShaders } from './webgpu.js';
import { ModelTensors, createModel, destroyModel, initCpuWeights, shakeEmbeddings, shakeMlp, computeEmbRange, buildFwdUniforms, uploadEmbRange, uploadEmbOffsets, uploadChannelMask, cpuPackEmbeddings, generateEmbOffsets, computeTensorSizes, computeGridDims } from './model.js';
import { get_target_pixels } from './loss.js';
import { export_to_glsl } from './shader_exporter.js';
import { saveModelSafetensors, loadModelSafetensors } from './model_io.js';
import { ROIMask } from './roi_mask.js';
import { drawOutputCanvas, drawFlowDiagram, drawLossCurve, FLOW_PAD, FLOW_EMB_W } from './viz.js';
import { ui, GRID_SIZES, init as initUI, drawPlaceholder, updateSizeDisplay, updateDirtyIndicators, syncButtonStates, updateStartLabel, setStatus, restoreUISettings, syncSliderDisplay, sliderToLR, updateEmbBitsOptions, fitSidePanels, applyUrlParams } from './ui.js?v=2';
import { init as initROI, startDecayLoop, stopDecayLoop, hasPainted } from './roi_controls.js';
import { init as initFileHandler, getUrlExample } from './file_handler.js';
import { SweepOverlay } from './sweep.js';
import { Trainer, runInferencePass, runZoomInferencePass } from './trainer.js';
import { CpuTrainer, forward } from './optimizer.js';

const gpuAvailable = !!navigator.gpu;

function disableGpu() {
    const opt = ui.engineSelect.querySelector('option[value="gpu"]');
    if (opt) opt.disabled = true;
    if (ui.engineSelect.value === 'gpu') ui.engineSelect.value = 'cpu';
}

// --- State ---
let BASE_CANVAS_W = ui.canvas.width;
let BASE_CANVAS_H = ui.canvas.height;
let gridW = 64, gridH = 64;  // computed once from numPts + image dims; updated on image load / dropdown change
let config = {};
let imageHasAlpha = false;  // set from image pixels in loadImageOntoCanvas
let loadedImage   = null;
let webGpuContext = null;
let activeSession = null;
let trainer       = null;
let lastWeights    = null;
let snapshotWeights = null;
let layerRangeEma  = null;
let lastInterData  = { inter1: null, inter2: null };
let lastConfig     = null;
let channelMask    = 0xFFFFFFFF;
let hoverState     = null;
let layerCols      = [];
let activeInferencePromise = null;
let inferPending           = false;
const sweep        = new SweepOverlay(ui.sweepCanvas, ui.sourceSweepCanvas);

const isTraining   = () => trainer?.active ?? false;
const configReady  = () => !!config.mlpWidth1;
const canInfer     = () => !!lastWeights && !isTraining();
const hasCpuWeights = () => !!lastWeights?.layer1_biases;  // full weight tensors loaded (not partial GPU readback)
const doInference  = () => activeSession ? runInference() : (hasCpuWeights() ? runInferenceCpu() : null);

function stopTrainer() {
    trainer?.stop();
}

async function teardownTrainer() {
    if (!trainer) return;
    trainer.stop();
    await trainer.waitForIdle();
    trainer.destroy();
    trainer = null;
}

// ROI mask state
const roiMask = new ROIMask(1, 1);

// --- Source canvas ---
function drawSourceImage() {
    if (!loadedImage) return;
    const ctx = ui.sourceCanvas.getContext('2d', { willReadFrequently: true });
    ctx.clearRect(0, 0, ui.sourceCanvas.width, ui.sourceCanvas.height);
    ctx.drawImage(loadedImage, 0, 0, ui.sourceCanvas.width, ui.sourceCanvas.height);
    roiMask.drawOverlay(ctx);
    ui.dropOverlay.classList.add('hidden');
}

ui.outputZoomInput.addEventListener('input', () => {
    const scale = parseFloat(ui.outputZoomInput.value);
    ui.outputZoomVal.textContent = scale + '×';
    const wrap = ui.canvas.closest('.canvas-wrap');
    const zoomed = scale > 1;
    wrap.classList.toggle('zoomed', zoomed);
    if (zoomed) {
        const aspect = BASE_CANVAS_W / BASE_CANVAS_H;
        const ww = wrap.clientWidth, wh = wrap.clientHeight;
        const fitW = Math.min(ww, wh * aspect);
        ui.canvas.style.width  = Math.round(fitW * scale) + 'px';
        ui.canvas.style.height = Math.round(fitW * scale / aspect) + 'px';
    } else {
        ui.canvas.style.width = '';
        ui.canvas.style.height = '';
    }
    if (canInfer()) {
        const W = Math.round(BASE_CANVAS_W * scale);
        const H = Math.round(BASE_CANVAS_H * scale);
        ui.outputResEl.textContent = `${W}×${H}`;
        runZoomInference(W, H);
    }
});

const VALID_EMB_CHANNELS = [4, 8, 16];
const VALID_MLP_WIDTHS1  = [4, 8, 16, 32, 64];
const VALID_MLP_WIDTHS2  = [8, 16, 32, 64];
const VALID_EMB_BITS     = [4, 8];

function buildAlphaCellMask(targetPixels, W, H, gW, gH) {
    const mask = new Float32Array(gW * gH).fill(1);
    const stepX = (W - 1) / (gW - 1);
    const stepY = (H - 1) / (gH - 1);
    for (let gy = 0; gy < gH; gy++) {
        for (let gx = 0; gx < gW; gx++) {
            const cx  = gx * stepX;
            const cy  = gy * stepY;
            const px0 = Math.max(0, Math.floor(cx - stepX));
            const py0 = Math.max(0, Math.floor(cy - stepY));
            const px1 = Math.min(W - 1, Math.ceil(cx + stepX));
            const py1 = Math.min(H - 1, Math.ceil(cy + stepY));
            let opaque = false;
            outer: for (let py = py0; py <= py1; py++) {
                for (let px = px0; px <= px1; px++) {
                    if (targetPixels[(py * W + px) * 4 + 3] > 0) { opaque = true; break outer; }
                }
            }
            if (!opaque) mask[gy * gW + gx] = 0;
        }
    }
    return mask;
}

function isValidModelConfig({ gW, gH, embeddingChannels, mlpWidth1, mlpWidth2, embBits = 8 }) {
    return Number.isInteger(gW) && gW >= 2 && gW <= 512 &&
           Number.isInteger(gH) && gH >= 2 && gH <= 512 &&
           VALID_EMB_CHANNELS.includes(embeddingChannels) &&
           VALID_MLP_WIDTHS1.includes(mlpWidth1)     &&
           VALID_MLP_WIDTHS2.includes(mlpWidth2)     &&
           VALID_EMB_BITS.includes(embBits);
}

function configCompatible(a, b) {
    return a && b &&
        a.gW === b.gW &&
        a.gH === b.gH &&
        a.embeddingChannels === b.embeddingChannels &&
        a.mlpWidth1 === b.mlpWidth1 &&
        a.mlpWidth2 === b.mlpWidth2;
}

function updateGridDims() {
    const numPts = GRID_SIZES[parseInt(ui.gridSizeSelect.value)];
    ({ gW: gridW, gH: gridH } = computeGridDims(numPts, BASE_CANVAS_W, BASE_CANVAS_H));
    sweep.setGrid(gridW, gridH);
    ui.gridSizeLabelEl.textContent = `${gridW * gridH} pts (${gridW} × ${gridH})`;
}

function currentModelConfig() {
    const mlpWidth1 = parseInt(ui.mlpWidth1Select.value);
    const mlpWidth2 = parseInt(ui.mlpWidth2Select.value);
    const embeddingChannels = parseInt(ui.embeddingChannelsSelect.value);
    return {
        gW: gridW, gH: gridH,
        embeddingChannels: Math.ceil(embeddingChannels / 4) * 4,
        mlpWidth1:         Math.ceil(mlpWidth1 / 4) * 4,
        mlpWidth2:         Math.ceil(mlpWidth2 / 4) * 4,
        embBits:           parseInt(ui.embBitsSelect.value),
    };
}

window.addEventListener('resize', () => { if (loadedImage) fitSidePanels(BASE_CANVAS_W, BASE_CANVAS_H); });

window.addEventListener('theme-changed', () => {
    if (lastWeights) {
        ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(ui.layersCanvas,
            { weights: lastWeights, inter1: lastInterData.inter1, inter2: lastInterData.inter2,
              finalOutput: lastWeights.finalOutput || null, imgW: ui.canvas.width, imgH: ui.canvas.height,
              config, channelMask, ema: layerRangeEma, hoverState }));
        sweep.setCols(layerCols);
        sweep.triggerFwd(); // Force a redraw of the sweep to apply the new theme
    }
    if (trainer && trainer._lossHistory) {
        drawLossCurve(ui.lossCanvas, trainer._lossHistory);
    } else if (model && trainer === null) {
         // During inference-only, loss might be drawn elsewhere or is empty.
         // Assuming if there's no trainer, we don't have an active loss curve to redraw.
    }
});

function loadImageOntoCanvas(img) {
    loadedImage = img;
    const MAX = 512;
    const aspect = img.naturalWidth / img.naturalHeight;
    BASE_CANVAS_W = aspect >= 1 ? MAX : Math.round(MAX * aspect);
    BASE_CANVAS_H = aspect >= 1 ? Math.round(MAX / aspect) : MAX;
    ui.sourceCanvas.width  = BASE_CANVAS_W;
    ui.sourceCanvas.height = BASE_CANVAS_H;
    ui.sourceSweepCanvas.width  = BASE_CANVAS_W;
    ui.sourceSweepCanvas.height = BASE_CANVAS_H;
    ui.canvas.width  = BASE_CANVAS_W;
    ui.canvas.height = BASE_CANVAS_H;
    ui.canvas.style.width = '';
    ui.canvas.style.height = '';
    ui.canvas.style.maxWidth = '';
    ui.canvas.style.maxHeight = '';
    roiMask.resize(BASE_CANVAS_W, BASE_CANVAS_H);
    updateGridDims();
    ui.sourcePanel.classList.add('has-image');
    ui.sourceResEl.textContent = `${BASE_CANVAS_W}×${BASE_CANVAS_H}`;
    ui.outputResEl.textContent = `${BASE_CANVAS_W}×${BASE_CANVAS_H}`;
    drawSourceImage();
    {
        const ctx = ui.sourceCanvas.getContext('2d');
        const px = ctx.getImageData(0, 0, BASE_CANVAS_W, BASE_CANVAS_H).data;
        let hasAlpha = false;
        for (let i = 3; i < px.length; i += 4) { if (px[i] < 255) { hasAlpha = true; break; } }
        imageHasAlpha = hasAlpha;
        config.hasAlpha = hasAlpha;
    }
    const outCh = config.hasAlpha ? 4 : 3;
    const bytes = BASE_CANVAS_W * BASE_CANVAS_H * outCh;
    ui.inputSizeEl.textContent = bytes >= 1024 ? (bytes / 1024).toFixed(1) + ' KB' : bytes + ' B';
    requestAnimationFrame(() => fitSidePanels(BASE_CANVAS_W, BASE_CANVAS_H));
}

async function resetToRandomModel() {
    config = buildConfigFromUI();
    config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gW, config.gH, ui.noOffsetCheckbox.checked);
    if (activeSession) activeSession.destroy();
    activeSession = new GpuSession(webGpuContext, config, ui.canvas.width, ui.canvas.height);
    lastConfig = currentModelConfig();
    updateDirtyIndicators(lastConfig, currentModelConfig());
    lastWeights = weightsViewFrom(activeSession.initialWeights);
    await runInference();
}

async function previewGeometry() {
    if (!webGpuContext || !loadedImage) return;
    await teardownTrainer();
    clearTrainingUI();
    layerRangeEma = null;
    await resetToRandomModel();
    syncButtonStates(isTraining(), !!activeSession, !!snapshotWeights);
}

async function startTraining(fullReset) {
    if (isTraining()) return;

    if (ui.engineSelect.value === 'gpu' && !gpuAvailable) {
        alert("WebGPU is not supported in this browser. Falling back to CPU training (slow).");
        ui.engineSelect.value = 'cpu';
    }

    const useCpu = ui.engineSelect.value === 'cpu';
    const curConfig = currentModelConfig();
    const configChanged = !configCompatible(lastConfig, curConfig);
    const needsNewBuffers = fullReset || configChanged || !activeSession;

    // We can transfer weights from CPU trainer if we are continuing without a config change
    const previousEngine = trainer?.type;
    const canTransferCpu = previousEngine === 'cpu' && !fullReset && !configChanged;
    let prevCpuWeights = (!useCpu && canTransferCpu) ? trainer.getWeights() : null;

    await teardownTrainer();
    resetCanvasToBase();

    const prevOffsets = config?.embOffsets;
    const prevEmbBits = config?.embBits;
    config = buildConfigFromUI();

    let startWeights = null;

    if (needsNewBuffers) {
        config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gW, config.gH, ui.noOffsetCheckbox.checked);

        if (activeSession) activeSession.destroy();

        if (useCpu) {
            activeSession = new CpuSession(config);
            startWeights = activeSession.initialWeights;
        } else {
            try {
                if (!webGpuContext) webGpuContext = await initWebGPU();
                activeSession = new GpuSession(webGpuContext, config, ui.canvas.width, ui.canvas.height);
                startWeights = activeSession.initialWeights;
            } catch (err) {
                console.error("GPU device init or session creation failed:", err);
                disableGpu();
                ui.engineSelect.value = 'cpu';
                return startTraining(fullReset); // retry recursively as CPU
            }
        }

        lastConfig = curConfig;
        updateDirtyIndicators(lastConfig, curConfig);
        updateSizeDisplay(ui.canvas.width, ui.canvas.height);
        snapshotWeights = null;
    } else {
        // Continue branch
        config.embOffsets = (config.embBits === prevEmbBits) ? prevOffsets
            : generateEmbOffsets(config.embeddingChannels, config.embBits, config.gW, config.gH, ui.noOffsetCheckbox.checked);

        const currentEngine  = useCpu ? 'cpu' : 'gpu';

        if (currentEngine === 'gpu') {
            const m = activeSession.model;
            webGpuContext.writeBuffer(m.emb_offsets, config.embOffsets);
            if (previousEngine === 'cpu' && prevCpuWeights) {
                // Switching CPU -> GPU
                webGpuContext.uploadModelWeights(m, prevCpuWeights);
                for (const k of ModelTensors.KEYS) {
                    webGpuContext.clearBuffer(m.adamM[k]);
                    webGpuContext.clearBuffer(m.adamV[k]);
                }
                lastWeights = weightsViewFrom(prevCpuWeights);
            }
            if (activeSession.shaderChanged(config)) {
                activeSession.rebuildPipeline({
                    smoothInterpolation: config.smoothInterpolation,
                    activation:          config.activation,
                    quantization:        config.quantization,
                    embOffsets:          config.embOffsets,
                });
            } else {
                activeSession.rebuildBindGroup();
            }
            // If GPU -> GPU, we just reuse the existing buffers, startWeights remains null
        } else if (currentEngine === 'cpu') {
            if (previousEngine === 'gpu' && activeSession.model && webGpuContext) {
                // Switching GPU -> CPU: Must read back buffers to JS arrays
                startWeights = await readBackAllWeights();
            } else {
                // Switching CPU -> CPU: Inject the JS arrays into the new CpuTrainer
                startWeights = prevCpuWeights || lastWeights;
            }
        }
    }

    clearTrainingUI();
    setStatus('training', true, !!activeSession, true, !!snapshotWeights);
    sweep.resetDecay();

    trainer = useCpu ? makeCpuTrainer() : makeTrainer();
    if (!useCpu && !startWeights) trainer.lastWeights = lastWeights;

    // trainer.start takes freshWeights to initialize optimizer buffers.
    // If continuing from same engine, we pass null.
    // If starting fresh or crossing engine boundaries, we pass weights.
    trainer.start(startWeights);
}

async function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = async () => {
            loadImageOntoCanvas(img);
            await startTraining(true);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function resetCanvasToBase() {
    ui.canvas.width  = BASE_CANVAS_W;
    ui.canvas.height = BASE_CANVAS_H;
    ui.canvas.style.width = '';
    ui.canvas.style.height = '';
    ui.canvas.parentElement.style.overflow = 'hidden';
}

function buildConfigFromUI() {
    return {
        gW: gridW, gH: gridH,
        embeddingChannels:   parseInt(ui.embeddingChannelsSelect.value),
        mlpWidth1:           parseInt(ui.mlpWidth1Select.value),
        mlpWidth2:           parseInt(ui.mlpWidth2Select.value),
        quantization:        ui.quantizationSelect.value,
        embBits:             parseInt(ui.embBitsSelect.value),
        activation:          ui.activationSelect.value,
        smoothInterpolation: ui.smoothInterpolationCheckbox.checked,
        hasAlpha:            imageHasAlpha,
        width:  BASE_CANVAS_W,
        height: BASE_CANVAS_H,
    };
}



function clearTrainingUI() {
    stopDecayLoop();
    ui.lossCanvas.getContext('2d').clearRect(0, 0, ui.lossCanvas.width, ui.lossCanvas.height);
    ui.canvas.getContext('2d').clearRect(0, 0, ui.canvas.width, ui.canvas.height);
    ui.lossValueEl.textContent   = '—';
    ui.stepCounterEl.textContent = '0';
    ui.rateDisplayEl.textContent = '—';
    layerRangeEma = null;
    channelMask   = 0xFFFFFFFF;
    layerCols     = [];
    hoverState    = null;
    lastInterData = { inter1: null, inter2: null };
}

function setStoppedStatus() {
    setStatus('stopped', configCompatible(lastConfig, currentModelConfig()), !!activeSession, false, !!snapshotWeights);
}

function trainerCallbacks() {
    return {
        onStep({ loss, step, rate, lastWeights: w, inter1, inter2, lossHistory }) {
            lastWeights = w;
            if (inter1 !== null) lastInterData = { inter1, inter2 };
            ui.stepCounterEl.textContent = step.toLocaleString();
            ui.rateDisplayEl.textContent = rate === '—' ? rate : rate + ' it/s';
            if (w.finalOutput !== null) {
                ui.lossValueEl.textContent = loss < 1e-4 ? loss.toExponential(3) : loss.toFixed(6);
                drawOutputCanvas(ui.canvas, w.finalOutput, config.hasAlpha);
                ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(ui.layersCanvas,
                    { weights: w, inter1: lastInterData.inter1, inter2: lastInterData.inter2,
                      finalOutput: w.finalOutput, imgW: ui.canvas.width, imgH: ui.canvas.height,
                      config, channelMask, ema: layerRangeEma, hoverState }));
                sweep.setCols(layerCols);
                drawLossCurve(ui.lossCanvas, lossHistory);
                drawSourceImage();
            }
            if (inter1 !== null) sweep.triggerStep();
        },
        onStop() {
            setStoppedStatus();
            if (roiMask.isActive()) startDecayLoop();
        },
    };
}

function makeCpuTrainer() {
    return new CpuTrainer({
        config,
        targetPixels: get_target_pixels(loadedImage, ui.canvas),
        getHyperparams: () => ({
            stride:      parseInt(ui.bwdStrideInput.value) || 1,
            embedLr:     sliderToLR(parseInt(ui.embedLrInput.value)),
            mlpLr:       sliderToLR(parseInt(ui.mlpLrInput.value)),
            maxIter:     parseInt(ui.maxIterInput.value),
            vizInterval: parseInt(ui.vizIntervalSelect.value),
        }),
        ...trainerCallbacks(),
    });
}

function makeTrainer() {
    const targetPixels = get_target_pixels(loadedImage, ui.canvas);
    const alphaCellMask = config.hasAlpha
        ? buildAlphaCellMask(targetPixels, ui.canvas.width, ui.canvas.height, config.gW, config.gH)
        : null;
    return new Trainer({
        webGpuContext, canvas: ui.canvas, config,
        model: activeSession.model, pipeline: activeSession.pipeline, bindGroup: activeSession.bindGroup,
        fwdUniformsBuf: activeSession.fwdUniformsBuf, outputBuffers: activeSession.outputBuffers,
        readbackBuffers: activeSession.readbackBuffers,
        targetPixels,
        alphaCellMask,
        roiMask,
        getHyperparams: () => ({
            stride:      parseInt(ui.bwdStrideInput.value) || 1,
            mlpRatio:    parseInt(ui.mlpRatioInput.value)  || 1,
            numLoops:      parseInt(ui.numLoopsInput.value)      || 1,
            embedLr:     sliderToLR(parseInt(ui.embedLrInput.value)),
            mlpLr:       sliderToLR(parseInt(ui.mlpLrInput.value)),
            roiStrength: parseFloat(ui.roiStrengthInput.value),
            roiFreeze:   ui.roiFreezeChk.checked,
            maxIter:     parseInt(ui.maxIterInput.value),
            vizInterval:          parseInt(ui.vizIntervalSelect.value),
            offsetSampleInterval: parseInt(ui.offsetSampleIntervalSelect.value),
        }),
        ...trainerCallbacks(),
    });
}

// --- Start / Stop / Reset buttons ---
ui.startBtn.addEventListener('click', async () => {
    if (isTraining()) {
        stopTrainer();
        setStoppedStatus();
        if (roiMask.isActive()) startDecayLoop();
        return;
    }
    if (!loadedImage) {
        alert("Please load an image first.");
        return;
    }
    await startTraining(false);
});

ui.resetBtn.addEventListener('click', async () => {
    if (!loadedImage) return;
    await startTraining(true);
});

// --- Shake: add small noise to escape local minima ---
ui.shakeEmbBtn.addEventListener('click', () => {
    if (!lastWeights) return;
    shakeEmbeddings(webGpuContext, activeSession.model, lastWeights.embeddings);
});

ui.shakeMlpBtn.addEventListener('click', async () => {
    if (!activeSession || activeSession.type === 'cpu') return;
    shakeMlp(webGpuContext, activeSession.model, await readBackAllWeights());
});

// --- Weight readback (for export) ---
async function readBackAllWeights() {
    const sizes = computeTensorSizes(config);
    const m = activeSession.model;
    const tensors = {
        embeddings:     { buf: m.embeddings,     size: sizes.embeddings },
        layer1_weights: { buf: m.layer1.weights, size: sizes.layer1_weights },
        layer1_biases:  { buf: m.layer1.biases,  size: sizes.layer1_biases },
        layer2_weights: { buf: m.layer2.weights, size: sizes.layer2_weights },
        layer2_biases:  { buf: m.layer2.biases,  size: sizes.layer2_biases },
        layer3_weights: { buf: m.layer3.weights, size: sizes.layer3_weights },
        layer3_biases:  { buf: m.layer3.biases,  size: sizes.layer3_biases },
    };
    return webGpuContext.readBackBuffers(tensors);
}

function weightsViewFrom(rb) {
    return {
        embeddings:     rb.embeddings,
        layer1_weights: rb.layer1_weights,
        layer1_biases:  rb.layer1_biases,
        layer2_weights: rb.layer2_weights,
        layer2_biases:  rb.layer2_biases,
        layer3_weights: rb.layer3_weights,
        layer3_biases:  rb.layer3_biases,
    };
}

// --- Export button ---
document.getElementById('export-btn').addEventListener('click', async () => {
    if (!activeSession) {
        alert("Train the model first.");
        return;
    }
    const weights = await readBackAllWeights();
    const glsl = export_to_glsl(config, weights);
    const url  = URL.createObjectURL(new Blob([glsl], { type: 'text/plain' }));
    const a    = Object.assign(document.createElement('a'), { href: url, download: 'shader.glsl' });
    a.click();
    URL.revokeObjectURL(url);
});

// --- Save model ---
async function serializeModel() {
    const weights = await readBackAllWeights();
    const saveConfig = {
        ...config,
        maxIter:    ui.maxIterInput.value,
        embedLr:    ui.embedLrInput.value,
        mlpLr:      ui.mlpLrInput.value,
        mlpRatio:   ui.mlpRatioInput.value,
        numLoops:      ui.numLoopsInput.value,
        bwdStride:  ui.bwdStrideInput.value,
        outputZoom: ui.outputZoomInput.value,
        noOffset:   ui.noOffsetCheckbox.checked,
    };
    return saveModelSafetensors(saveConfig, weights);
}

document.getElementById('save-btn').addEventListener('click', async () => {
    if (!activeSession) { alert("Train or load a model first."); return; }
    const buf = await serializeModel();
    const url = URL.createObjectURL(new Blob([buf], { type: 'application/octet-stream' }));
    const a   = Object.assign(document.createElement('a'), { href: url, download: 'model.safetensors' });
    a.click();
    URL.revokeObjectURL(url);
});

// --- Snapshot / Recall ---
ui.snapshotBtn.addEventListener('click', async () => {
    if (!activeSession) { alert("Train or load a model first."); return; }
    snapshotWeights = await serializeModel();
    syncButtonStates(isTraining(), !!activeSession, !!snapshotWeights);
});

ui.recallBtn.addEventListener('click', async () => {
    if (!snapshotWeights) return;
    const saved = snapshotWeights;
    await loadAndResetModelFile(new Blob([saved], { type: 'application/octet-stream' }));
    snapshotWeights = saved;
    syncButtonStates(false, !!activeSession, !!snapshotWeights);
});

// --- Zoom inference: forward pass at arbitrary resolution using temp buffers ---
async function runZoomInference(W, H) {
    const { final } = await runZoomInferencePass({
        webGpuContext, config, model: activeSession.model, pipeline: activeSession.pipeline, lastWeights, channelMask, W, H
    });
    ui.canvas.width  = W;
    ui.canvas.height = H;
    drawOutputCanvas(ui.canvas, final, config.hasAlpha);
}

// --- Channel mask: click / shift+click on the Emb column in the Layers canvas ---
// Map a mouse event to canvas pixel coords, accounting for object-fit:contain letterboxing.
function layersCanvasCoords(e) {
    const rect = ui.layersCanvas.getBoundingClientRect();
    const cw = ui.layersCanvas.width, ch = ui.layersCanvas.height;
    const canvasAspect = cw / ch;
    const rectAspect = rect.width / rect.height;
    let contentW, contentH, ox, oy;
    if (canvasAspect > rectAspect) {
        contentW = rect.width; contentH = rect.width / canvasAspect;
        ox = 0; oy = (rect.height - contentH) / 2;
    } else {
        contentH = rect.height; contentW = rect.height * canvasAspect;
        ox = (rect.width - contentW) / 2; oy = 0;
    }
    return {
        cx: (e.clientX - rect.left - ox) * cw / contentW,
        cy: (e.clientY - rect.top  - oy) * ch / contentH,
    };
}

function redrawLayers() {
    if (!lastWeights || !configReady()) return;
    ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(ui.layersCanvas,
        { weights: lastWeights, inter1: lastInterData.inter1, inter2: lastInterData.inter2,
          finalOutput: lastWeights.finalOutput, imgW: ui.canvas.width, imgH: ui.canvas.height,
          config, channelMask, ema: layerRangeEma, hoverState }));
}

ui.layersCanvas.addEventListener('mousemove', (e) => {
    const { cx, cy } = layersCanvasCoords(e);
    let newState = null;
    for (const col of layerCols) {
        if (cx >= col.x && cx < col.x + col.w) {
            if (col.slotH != null) {
                const ch = Math.min(Math.max(0, ((cy - col.y0) / col.slotH) | 0), col.nCh - 1);
                newState = { col: col.name, ch };
            } else {
                newState = { col: col.name, ch: -1 };
            }
            break;
        }
    }
    ui.layersCanvas.style.cursor = newState?.col === 'emb' && !isTraining() ? 'pointer' : 'default';
    if (newState?.col === hoverState?.col && newState?.ch === hoverState?.ch) return;
    hoverState = newState;
    redrawLayers();
});

ui.layersCanvas.addEventListener('mouseleave', () => {
    ui.layersCanvas.style.cursor = 'default';
    if (hoverState === null) return;
    hoverState = null;
    redrawLayers();
});

ui.layersCanvas.addEventListener('click', (e) => {
    if (isTraining()) return;
    const embCol = layerCols.find(c => c.name === 'emb');
    if (!embCol?.slotH) return;
    const { cx, cy } = layersCanvasCoords(e);
    if (cx < embCol.x || cx >= embCol.x + embCol.w) return;
    const embCh = config?.embeddingChannels ?? 0;
    if (!embCh) return;
    const ch = Math.min(Math.max(0, ((cy - embCol.y0) / embCol.slotH) | 0), embCh - 1);
    const allBits = embCh < 32 ? (1 << embCh) - 1 : 0xFFFFFFFF;
    if (!(channelMask & (1 << ch))) channelMask = 0xFFFFFFFF;
    else channelMask ^= (allBits ^ (1 << ch));
    if (canInfer()) doInference();
    else redrawLayers();
});

// --- Inference: pack embeddings (CPU) then run one forward pass ---
async function runInference() {
    if (activeInferencePromise) { inferPending = true; return activeInferencePromise; }

    activeInferencePromise = (async () => {
        try {
            const { final: inferFinal, inter1: inferInter1, inter2: inferInter2 } = await runInferencePass({
                webGpuContext, config,
                model: activeSession.model, pipeline: activeSession.pipeline, bindGroup: activeSession.bindGroup,
                fwdUniformsBuf: activeSession.fwdUniformsBuf, outputBuffers: activeSession.outputBuffers,
                readbackBuffers: activeSession.readbackBuffers, channelMask,
                canvasWidth: ui.canvas.width, canvasHeight: ui.canvas.height, lastWeights
            });

            sweep.setCols(layerCols); sweep.triggerFwd();

            drawOutputCanvas(ui.canvas, inferFinal, config.hasAlpha);

            lastInterData = { inter1: inferInter1, inter2: inferInter2 };
            lastWeights.finalOutput = inferFinal;
            ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(ui.layersCanvas,
                { weights: lastWeights, inter1: inferInter1, inter2: inferInter2,
                  finalOutput: inferFinal, imgW: ui.canvas.width, imgH: ui.canvas.height,
                  config, channelMask, ema: layerRangeEma, hoverState }));
            sweep.setCols(layerCols);
        } finally {
            activeInferencePromise = null;
            if (inferPending) { inferPending = false; runInference(); }
        }
    })();
    return activeInferencePromise;
}

async function runInferenceCpu() {
    const { gW, gH, embeddingChannels: embCh } = config;
    const range  = computeEmbRange(lastWeights.embeddings, embCh, gW * gH);
    const packed = cpuPackEmbeddings(lastWeights.embeddings, embCh, range, config.embBits);
    const { final, interLayer1, interLayer2 } = forward(config, { ...lastWeights, embeddings_q: packed }, range);
    drawOutputCanvas(ui.canvas, final, config.hasAlpha);
    lastInterData = { inter1: interLayer1, inter2: interLayer2 };
    lastWeights.finalOutput = final;
    ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(ui.layersCanvas,
        { weights: lastWeights, inter1: interLayer1, inter2: interLayer2,
          finalOutput: final, imgW: ui.canvas.width, imgH: ui.canvas.height,
          config, channelMask, ema: layerRangeEma, hoverState }));
    sweep.setCols(layerCols);
}

async function loadAndResetModelFile(file) {
    // Prevent pending inference from retriggering, then wait for any active inference.
    inferPending = false;
    if (activeInferencePromise) await activeInferencePromise;
    await teardownTrainer();
    clearTrainingUI();
    snapshotWeights = null;
    roiMask.clear();
    await loadModelFile(file);
    syncButtonStates(false, !!activeSession, !!snapshotWeights);
}

async function loadModelFile(file) {
    let parsed;
    try {
        parsed = await loadModelSafetensors(file);
    } catch (err) {
        alert('Failed to load model: ' + err.message);
        return;
    }

    const { config: savedConfig, tensors, uiSettings = {} } = parsed;
    if (!isValidModelConfig(savedConfig)) {
        alert('Unsupported model config in file.');
        return;
    }

    const numPtsApprox = savedConfig.gW * savedConfig.gH;
    const nearestIdx = GRID_SIZES.reduce((best, v, i) =>
        Math.abs(v - numPtsApprox) < Math.abs(GRID_SIZES[best] - numPtsApprox) ? i : best, 0);
    ui.gridSizeSelect.value          = String(nearestIdx);
    ui.embeddingChannelsSelect.value = String(savedConfig.embeddingChannels);
    ui.mlpWidth1Select.value         = String(savedConfig.mlpWidth1);
    ui.mlpWidth2Select.value         = String(savedConfig.mlpWidth2);

    restoreUISettings({ savedConfig, uiSettings });

    const loadedEmbBits = savedConfig.embBits || 8;
    ui.embBitsSelect.value = String(loadedEmbBits);

    const outCh = tensors.layer3_biases.length;  // 3 or 4
    imageHasAlpha = outCh === 4;
    const { gW: savedGW, gH: savedGH, embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = savedConfig;
    const expectedSizes = {
        embeddings:     savedGW * savedGH * embCh,
        layer1_weights: embCh * mlpWidth1,
        layer1_biases:  mlpWidth1,
        layer2_weights: mlpWidth1 * mlpWidth2,
        layer2_biases:  mlpWidth2,
        layer3_weights: outCh * mlpWidth2,
        layer3_biases:  outCh,
    };
    for (const [k, expected] of Object.entries(expectedSizes)) {
        if (tensors[k].length !== expected) {
            alert(`Incompatible model: '${k}' has ${tensors[k].length} values, expected ${expected}`);
            return;
        }
    }
    for (const [k, arr] of Object.entries(tensors)) {
        if (arr.some(v => !isFinite(v))) {
            alert(`Corrupt model: '${k}' contains NaN or Inf values.`);
            return;
        }
    }

    config = {
        gW:                  savedGW,
        gH:                  savedGH,
        embeddingChannels:   embCh,
        mlpWidth1:           mlpWidth1,
        mlpWidth2:           mlpWidth2,
        quantization:        ui.quantizationSelect.value,
        embBits:             loadedEmbBits,
        activation:          ui.activationSelect.value,
        smoothInterpolation: ui.smoothInterpolationCheckbox.checked,
        hasAlpha:            outCh === 4,
        width:  ui.canvas.width,
        height: ui.canvas.height,
        embOffsets:          savedConfig.embOffsets ?? new Float32Array(embCh / (32 / loadedEmbBits) * 2),
    };

    gridW = savedGW; gridH = savedGH;
    sweep.setGrid(gridW, gridH);
    lastWeights = weightsViewFrom(tensors);
    lastConfig  = { gW: savedGW, gH: savedGH, embeddingChannels: embCh, mlpWidth1, mlpWidth2, embBits: loadedEmbBits };
    updateDirtyIndicators(lastConfig, currentModelConfig());

    if (gpuAvailable) {
        try {
            if (!webGpuContext) webGpuContext = await initWebGPU();
        } catch (err) {
            console.error('WebGPU device init failed:', err);
            disableGpu();
            await runInferenceCpu();
            return;
        }
        try {
            if (activeSession) activeSession.destroy();
            activeSession = new GpuSession(webGpuContext, config, ui.canvas.width, ui.canvas.height);
            webGpuContext.uploadModelWeights(activeSession.model, tensors);
            await runInference();
        } catch (err) {
            console.error('Load model (GPU) failed:', err);
            await runInferenceCpu();
        }
    } else {
        await runInferenceCpu();
    }
    setStoppedStatus();
    updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
    updateStartLabel(configCompatible(lastConfig, currentModelConfig()), !!activeSession, false, !!snapshotWeights);
}


function refreshStartLabel() {
    updateStartLabel(configCompatible(lastConfig, currentModelConfig()), !!activeSession, isTraining(), !!snapshotWeights);
}

initUI({
    onConfigChange: () => {
        refreshStartLabel();
        updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
    },
    onShaderChange: () => {
        if (isTraining() || !activeSession || activeSession.type !== 'gpu') return;
        activeSession.rebuildPipeline({ smoothInterpolation: ui.smoothInterpolationCheckbox.checked });
        config = { ...config, smoothInterpolation: ui.smoothInterpolationCheckbox.checked };
        if (canInfer()) doInference();
    },
    onSelectChange: () => {
        updateGridDims();
        refreshStartLabel();
        updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
        if (!isTraining()) {
            updateDirtyIndicators(lastConfig, currentModelConfig());
            const cur = currentModelConfig();
            if (lastWeights && configCompatible(lastConfig, cur)) {
                if (canInfer()) doInference(); else redrawLayers();
            } else if (webGpuContext && loadedImage) {
                previewGeometry();
            } else {
                drawPlaceholder(ui.layersCanvas);
                layerRangeEma = null;
                lastInterData = { inter1: null, inter2: null };
            }
        }
    },
});
initROI({ roiMask, isTraining, drawSourceImage });
initFileHandler({ onImageFile: handleFile, onModelFile: loadAndResetModelFile, onExampleSelect: loadExample, hasPainted });

async function loadExample({ image, model: modelUrl }) {
    await new Promise((resolve) => {
        const img = new Image();
        img.onload = () => { loadImageOntoCanvas(img); resolve(); };
        img.src = image;
    });
    try {
        let modelLoaded = false;
        try {
            const resp = await fetch(modelUrl);
            if (resp.ok) { await loadAndResetModelFile(await resp.blob()); modelLoaded = true; }
        } catch (err) { console.warn('Example model fetch failed:', err); }
        if (!modelLoaded && gpuAvailable) {
            try {
                if (!webGpuContext) webGpuContext = await initWebGPU();
            } catch (err) { console.error('WebGPU device init failed:', err); disableGpu(); return; }
            try {
                await teardownTrainer(); clearTrainingUI(); snapshotWeights = null;
                roiMask.clear();
                await resetToRandomModel();
                setStoppedStatus();
                updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
            } catch (err) { console.warn('Example GPU init failed:', err); }
        }
    } catch (err) { console.warn('Example load failed:', err); }
}

if (!gpuAvailable) disableGpu();

// --- Start tooltip: show once startup is complete AND splash is dismissed ---
(function() {
    const tip = document.getElementById('start-tooltip');
    const dismiss = () => tip.remove();
    let startupDone = false;
    let splashGone  = false;
    function maybeShow() {
        if (!startupDone || !splashGone) return;
        const r = ui.startBtn.getBoundingClientRect();
        tip.style.left = (r.left + r.width / 2) + 'px';
        tip.style.top  = (r.bottom + 10) + 'px';
        tip.classList.add('visible');
        tip.addEventListener('animationend', dismiss, { once: true });
    }
    document.addEventListener('startup-complete',  () => { startupDone = true; maybeShow(); }, { once: true });
    document.addEventListener('splash-dismissed',  () => { splashGone  = true; maybeShow(); }, { once: true });
    ui.startBtn.addEventListener('click', dismiss, { once: true });
})();

// --- Startup: load default image then default model (or auto-start from URL params) ---
const urlHasParams = applyUrlParams();
if (urlHasParams) {
    updateGridDims();
    refreshStartLabel();
    updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
}
const urlExample = getUrlExample();
if (urlExample) {
    (async () => {
        await loadExample(urlExample);
        document.dispatchEvent(new Event('startup-complete'));
    })();
} else {
    const startupImg = new Image();
    startupImg.onload = async () => {
        loadImageOntoCanvas(startupImg);
        if (urlHasParams) {
            ui.startBtn.click();
        } else if (!gpuAvailable) {
            try {
                const resp = await fetch('imgs/model.safetensors');
                if (resp.ok) await loadAndResetModelFile(await resp.blob());
            } catch (err) { console.warn('CPU startup load failed:', err); }
            document.dispatchEvent(new Event('startup-complete'));
        } else {
            try { if (!webGpuContext) webGpuContext = await initWebGPU(); }
            catch (err) {
                console.error('WebGPU init failed:', err);
                disableGpu();
                document.dispatchEvent(new Event('startup-complete'));
                return;
            }
            // Try loading the saved model first; fall back to random-weights inference if absent.
            let modelLoaded = false;
            try {
                const resp = await fetch('imgs/model.safetensors');
                if (resp.ok) { await loadAndResetModelFile(await resp.blob()); modelLoaded = true; }
            } catch (err) { console.warn('Auto-load failed:', err); }
            if (!modelLoaded) {
                try { await resetToRandomModel(); }
                catch (err) { console.error('Initial inference failed:', err); }
            }
            document.dispatchEvent(new Event('startup-complete'));
        }
    };
    startupImg.src = 'imgs/model.webp';
}
