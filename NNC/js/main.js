import { initWebGPU } from './webgpu_manager.js';
import { ModelTensors, createModel, destroyModel, initCpuWeights, shakeEmbeddings, shakeMlp } from './model.js';
import { buildShader } from './shader_builder.js';
import { get_target_pixels } from './loss.js';
import { export_to_glsl } from './shader_exporter.js';
import { saveModelSafetensors, loadModelSafetensors } from './model_io.js';
import { ROIMask } from './roi_mask.js';
import { drawOutputCanvas, drawFlowDiagram, drawLossCurve, FLOW_PAD, FLOW_EMB_W } from './viz.js';
import { computeEmbRange, buildFwdUniforms, uploadEmbRange, uploadEmbOffsets, uploadChannelMask, cpuPackEmbeddings, generateEmbOffsets } from './emb_utils.js';
import { initTooltips } from './tooltips.js';
import { Trainer } from './trainer.js';
import { CpuTrainer } from './optimizer.js';
import { DOM, init as initUI, drawPlaceholder, updateSizeDisplay, updateDirtyIndicators, syncButtonStates, updateStartLabel, setStatus, restoreUISettings, syncSliderDisplay, sliderToLR, updateEmbBitsOptions, fitSidePanels, applyUrlParams } from './ui_manager.js';
import { init as initROI, startDecayLoop, stopDecayLoop, hasPainted } from './roi_controls.js';
import { init as initFileHandler, getUrlExample } from './file_handler.js';
import { SweepOverlay } from './sweep.js';

// --- State ---
let BASE_CANVAS_W = DOM.canvas.width;
let BASE_CANVAS_H = DOM.canvas.height;
let config = {};
let imageHasAlpha = false;  // set from image pixels in loadImageOntoCanvas
let loadedImage   = null;
let webGpuContext = null;
let model         = null;
let pipeline      = null;
let bindGroup     = null;
let fwdUniformsBuf = null; // 160-byte forward uniform buffer (includes emb_range)
let outputBuffers  = {};
let readbackBuffers = {};
let trainer        = null;
let lastWeights    = null;
let snapshotWeights = null;
let layerRangeEma  = null;
let lastInterData  = { inter1: null, inter2: null };
let lastConfig     = null;
let channelMask    = 0xFFFFFFFF;
let hoverState     = null;
let layerCols      = [];
let inferRunning   = false;
let inferPending   = false;
const sweep        = new SweepOverlay(DOM.sweepCanvas, DOM.sourceSweepCanvas);

const isTraining = () => trainer?.active ?? false;
const canInfer   = () => !!model && !!lastWeights && !isTraining();

// ROI mask state
const roiMask = new ROIMask(1, 1);

// --- Source canvas ---
function drawSourceImage() {
    if (!loadedImage) return;
    const ctx = DOM.sourceCanvas.getContext('2d');
    ctx.clearRect(0, 0, DOM.sourceCanvas.width, DOM.sourceCanvas.height);
    ctx.drawImage(loadedImage, 0, 0, DOM.sourceCanvas.width, DOM.sourceCanvas.height);
    roiMask.drawOverlay(ctx);
    DOM.dropOverlay.classList.add('hidden');
}

DOM.outputZoomInput.addEventListener('input', () => {
    const scale = parseFloat(DOM.outputZoomInput.value);
    DOM.outputZoomVal.textContent = scale + '×';
    const wrap = DOM.canvas.closest('.canvas-wrap');
    const zoomed = scale > 1;
    wrap.classList.toggle('zoomed', zoomed);
    if (zoomed) {
        const aspect = BASE_CANVAS_W / BASE_CANVAS_H;
        const ww = wrap.clientWidth, wh = wrap.clientHeight;
        const fitW = Math.min(ww, wh * aspect);
        DOM.canvas.style.width  = Math.round(fitW * scale) + 'px';
        DOM.canvas.style.height = Math.round(fitW * scale / aspect) + 'px';
    } else {
        DOM.canvas.style.width = '';
        DOM.canvas.style.height = '';
    }
    if (canInfer()) {
        const W = Math.round(BASE_CANVAS_W * scale);
        const H = Math.round(BASE_CANVAS_H * scale);
        DOM.outputResEl.textContent = `${W}×${H}`;
        runZoomInference(W, H);
    }
});

const VALID_GRID_SIZES   = [16, 32, 64];
const VALID_EMB_CHANNELS = [4, 8, 16];
const VALID_MLP_WIDTHS1  = [4, 8, 16, 32, 64];
const VALID_MLP_WIDTHS2  = [8, 16, 32, 64];
const VALID_EMB_BITS     = [4, 8];

function buildAlphaCellMask(targetPixels, W, H, gridSize) {
    const gs   = gridSize;
    const mask = new Float32Array(gs * gs).fill(1);
    const stepX = (W - 1) / (gs - 1);
    const stepY = (H - 1) / (gs - 1);
    for (let gy = 0; gy < gs; gy++) {
        for (let gx = 0; gx < gs; gx++) {
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
            if (!opaque) mask[gy * gs + gx] = 0;
        }
    }
    return mask;
}

function isValidModelConfig({ gridSize, embeddingChannels, mlpWidth1, mlpWidth2, embBits = 8 }) {
    return VALID_GRID_SIZES.includes(gridSize)       &&
           VALID_EMB_CHANNELS.includes(embeddingChannels) &&
           VALID_MLP_WIDTHS1.includes(mlpWidth1)     &&
           VALID_MLP_WIDTHS2.includes(mlpWidth2)     &&
           VALID_EMB_BITS.includes(embBits);
}

function configCompatible(a, b) {
    return a && b &&
        a.gridSize === b.gridSize &&
        a.embeddingChannels === b.embeddingChannels &&
        a.mlpWidth1 === b.mlpWidth1 &&
        a.mlpWidth2 === b.mlpWidth2;
}

function currentModelConfig() {
    const mlpWidth1 = parseInt(DOM.mlpWidth1Select.value);
    const mlpWidth2 = parseInt(DOM.mlpWidth2Select.value);
    const embeddingChannels = parseInt(DOM.embeddingChannelsSelect.value);
    return {
        gridSize:          parseInt(DOM.gridSizeSelect.value),
        embeddingChannels: Math.ceil(embeddingChannels / 4) * 4,
        mlpWidth1:         Math.ceil(mlpWidth1 / 4) * 4,
        mlpWidth2:         Math.ceil(mlpWidth2 / 4) * 4,
        embBits:           parseInt(DOM.embBitsSelect.value),
    };
}

window.addEventListener('resize', () => { if (loadedImage) fitSidePanels(BASE_CANVAS_W, BASE_CANVAS_H); });

function loadImageOntoCanvas(img) {
    loadedImage = img;
    const MAX = 512;
    const aspect = img.naturalWidth / img.naturalHeight;
    BASE_CANVAS_W = aspect >= 1 ? MAX : Math.round(MAX * aspect);
    BASE_CANVAS_H = aspect >= 1 ? Math.round(MAX / aspect) : MAX;
    DOM.sourceCanvas.width  = BASE_CANVAS_W;
    DOM.sourceCanvas.height = BASE_CANVAS_H;
    DOM.sourceSweepCanvas.width  = BASE_CANVAS_W;
    DOM.sourceSweepCanvas.height = BASE_CANVAS_H;
    DOM.canvas.width  = BASE_CANVAS_W;
    DOM.canvas.height = BASE_CANVAS_H;
    DOM.canvas.style.width = '';
    DOM.canvas.style.height = '';
    DOM.canvas.style.maxWidth = '';
    DOM.canvas.style.maxHeight = '';
    roiMask.resize(BASE_CANVAS_W, BASE_CANVAS_H);
    DOM.sourcePanel.classList.add('has-image');
    DOM.sourceResEl.textContent = `${BASE_CANVAS_W}×${BASE_CANVAS_H}`;
    DOM.outputResEl.textContent = `${BASE_CANVAS_W}×${BASE_CANVAS_H}`;
    drawSourceImage();
    {
        const ctx = DOM.sourceCanvas.getContext('2d');
        const px = ctx.getImageData(0, 0, BASE_CANVAS_W, BASE_CANVAS_H).data;
        let hasAlpha = false;
        for (let i = 3; i < px.length; i += 4) { if (px[i] < 255) { hasAlpha = true; break; } }
        imageHasAlpha = hasAlpha;
        config.hasAlpha = hasAlpha;
    }
    const outCh = config.hasAlpha ? 4 : 3;
    const bytes = BASE_CANVAS_W * BASE_CANVAS_H * outCh;
    DOM.inputSizeEl.textContent = bytes >= 1024 ? (bytes / 1024).toFixed(1) + ' KB' : bytes + ' B';
    requestAnimationFrame(() => fitSidePanels(BASE_CANVAS_W, BASE_CANVAS_H));
}

async function resetToRandomModel() {
    destroyModel(model);
    config = buildConfigFromUI();
    config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, DOM.noOffsetCheckbox.checked);
    const { buffers, weights } = createModel(webGpuContext, config);
    model = buffers;
    await createPipeline();
    createBindGroup();
    channelMask = 0xFFFFFFFF;
    lastConfig = currentModelConfig();
    updateDirtyIndicators(lastConfig, currentModelConfig());
    lastWeights = weightsViewFrom(weights);
    await runInference();
}

async function previewGeometry() {
    if (!webGpuContext || !loadedImage) return;
    trainer?.destroy(); trainer = null;
    clearTrainingUI();
    layerRangeEma = null;
    await resetToRandomModel();
    syncButtonStates(isTraining(), !!model, !!snapshotWeights);
}

async function startTraining(fullReset) {
    const useCpu = DOM.engineSelect.value === 'cpu';
    const configChanged = !configCompatible(lastConfig, currentModelConfig());

    const canTransferCpu = trainer?.type === 'cpu' && !fullReset && !configChanged;
    const prevCpuWeights = (!useCpu && canTransferCpu) ? trainer.getWeights() : null;
    const keepCpu        = ( useCpu && canTransferCpu);
    if (!keepCpu) { trainer?.destroy(); trainer = null; }

    resetCanvasToBase();
    const prevOffsets = config.embOffsets;
    const prevEmbBits = config.embBits;
    config = buildConfigFromUI();

    if (useCpu) {
        if (!trainer) {
            config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, DOM.noOffsetCheckbox.checked);
            lastConfig = currentModelConfig();
            updateDirtyIndicators(lastConfig, currentModelConfig());
            clearTrainingUI();
            snapshotWeights = null;
            trainer = makeCpuTrainer();
            setStatus('training', configCompatible(lastConfig, currentModelConfig()), !!model, isTraining(), !!snapshotWeights);
            const cpuStartWeights = (!fullReset && !configChanged && model && webGpuContext)
                ? await readBackAllWeights()
                : initCpuWeights(config);
            sweep.resetDecay(); trainer.start(cpuStartWeights);
        } else {
            config.embOffsets = (config.embBits === prevEmbBits) ? prevOffsets
                : generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, DOM.noOffsetCheckbox.checked);
            clearTrainingUI();
            setStatus('training', configCompatible(lastConfig, currentModelConfig()), !!model, isTraining(), !!snapshotWeights);
            sweep.resetDecay(); trainer.start(null);
        }
        return;
    }

    try {
        if (!webGpuContext) webGpuContext = await initWebGPU();
        if (fullReset || configChanged || !model) {
            config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, DOM.noOffsetCheckbox.checked);
            const { buffers, weights: freshWeights } = createModel(webGpuContext, config);
            destroyModel(model);
            model = buffers;
            await createPipeline();
            createBindGroup();
            channelMask = 0xFFFFFFFF;
            lastConfig = currentModelConfig();
            updateDirtyIndicators(lastConfig, currentModelConfig());
            clearTrainingUI();
            snapshotWeights = null;
            trainer = makeTrainer();
            setStatus('training', configCompatible(lastConfig, currentModelConfig()), !!model, isTraining(), !!snapshotWeights);
            sweep.resetDecay(); trainer.start(freshWeights);
        } else {
            config.embOffsets = (config.embBits === prevEmbBits) ? prevOffsets
                : generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, DOM.noOffsetCheckbox.checked);
            webGpuContext.writeBuffer(model.emb_offsets, config.embOffsets);
            if (prevCpuWeights) {
                webGpuContext.uploadModelWeights(model, prevCpuWeights);
                for (const k of ModelTensors.KEYS) {
                    webGpuContext.clearBuffer(model.adamM[k]);
                    webGpuContext.clearBuffer(model.adamV[k]);
                }
                lastWeights = weightsViewFrom(prevCpuWeights);
            }
            await createPipeline();
            createBindGroup();
            channelMask = 0xFFFFFFFF;
            clearTrainingUI();
            trainer = makeTrainer();
            trainer.lastWeights = lastWeights;
            setStatus('training', configCompatible(lastConfig, currentModelConfig()), !!model, isTraining(), !!snapshotWeights);
            sweep.resetDecay(); trainer.start(null);
        }
    } catch (err) {
        console.error("Training start failed:", err);
        alert("Training start failed. Check the console for errors.");
    }
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
    DOM.canvas.width  = BASE_CANVAS_W;
    DOM.canvas.height = BASE_CANVAS_H;
    DOM.canvas.style.width = '';
    DOM.canvas.style.height = '';
    DOM.canvas.parentElement.style.overflow = 'hidden';
}

function buildConfigFromUI() {
    return {
        gridSize:            parseInt(DOM.gridSizeSelect.value),
        embeddingChannels:   parseInt(DOM.embeddingChannelsSelect.value),
        mlpWidth1:           parseInt(DOM.mlpWidth1Select.value),
        mlpWidth2:           parseInt(DOM.mlpWidth2Select.value),
        quantization:        DOM.quantizationSelect.value,
        embBits:             parseInt(DOM.embBitsSelect.value),
        activation:          DOM.activationSelect.value,
        smoothInterpolation: DOM.smoothInterpolationCheckbox.checked,
        hasAlpha:            imageHasAlpha,
        width:  BASE_CANVAS_W,
        height: BASE_CANVAS_H,
    };
}

// --- WebGPU setup ---
async function createPipeline() {
    const shaderModule = webGpuContext.device.createShaderModule({ code: buildShader(config) });
    pipeline = webGpuContext.device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
    });
}

function createBindGroup() {
    fwdUniformsBuf?.destroy();
    for (const b of Object.values(outputBuffers)) b?.destroy();
    for (const b of Object.values(readbackBuffers)) b?.destroy();

    const { mlpWidth1, mlpWidth2, gridSize, embeddingChannels } = config;
    fwdUniformsBuf = webGpuContext.uniformBuffer(224, 'fwdUniforms');
    webGpuContext.writeBuffer(fwdUniformsBuf,
        buildFwdUniforms(gridSize, embeddingChannels, mlpWidth1, mlpWidth2, DOM.canvas.width, DOM.canvas.height, null, config.embOffsets));

    const pixelCount = DOM.canvas.width * DOM.canvas.height;
    const embSize    = gridSize * gridSize * embeddingChannels;
    const stride1    = mlpWidth1 * 4; // bytes per pixel in interLayer1
    const stride2    = mlpWidth2 * 4; // bytes per pixel in interLayer2

    outputBuffers.interLayer1 = webGpuContext.outputBuffer(pixelCount * stride1, 'out/inter1');
    outputBuffers.interLayer2 = webGpuContext.outputBuffer(pixelCount * stride2, 'out/inter2');
    outputBuffers.final       = webGpuContext.outputBuffer(pixelCount * 4 * 4,   'out/final');

    readbackBuffers.final         = webGpuContext.readbackBuffer(pixelCount * 4 * 4,                  'rb/final');
    readbackBuffers.embeddings    = webGpuContext.readbackBuffer(embSize * 4,                          'rb/emb');
    readbackBuffers.layer1Weights = webGpuContext.readbackBuffer(mlpWidth1 * embeddingChannels * 4,    'rb/L1w');
    readbackBuffers.layer1Biases  = webGpuContext.readbackBuffer(stride1,                              'rb/L1b');
    readbackBuffers.layer2Weights = webGpuContext.readbackBuffer(mlpWidth2 * stride1,                  'rb/L2w');
    readbackBuffers.layer3Weights = webGpuContext.readbackBuffer((config.hasAlpha ? 4 : 3) * stride2,  'rb/L3w');
    readbackBuffers.interLayer1   = webGpuContext.readbackBuffer(pixelCount * stride1,                 'rb/inter1');
    readbackBuffers.interLayer2   = webGpuContext.readbackBuffer(pixelCount * stride2,                 'rb/inter2');

    bindGroup = webGpuContext.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0,  resource: { buffer: fwdUniformsBuf } },
            { binding: 1,  resource: { buffer: model.embeddings_q } },
            { binding: 2,  resource: { buffer: model.layer1.weights } },
            { binding: 3,  resource: { buffer: model.layer1.biases  } },
            { binding: 4,  resource: { buffer: model.layer2.weights } },
            { binding: 5,  resource: { buffer: model.layer2.biases  } },
            { binding: 6,  resource: { buffer: model.layer3.weights } },
            { binding: 7,  resource: { buffer: model.layer3.biases  } },
            { binding: 8,  resource: { buffer: outputBuffers.interLayer1 } },
            { binding: 9,  resource: { buffer: outputBuffers.interLayer2 } },
            { binding: 10, resource: { buffer: outputBuffers.final       } },
        ],
    });
}

function clearTrainingUI() {
    stopDecayLoop();
    DOM.lossCanvas.getContext('2d').clearRect(0, 0, DOM.lossCanvas.width, DOM.lossCanvas.height);
    DOM.canvas.getContext('2d').clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);
    DOM.lossValueEl.textContent   = '—';
    DOM.stepCounterEl.textContent = '0';
    DOM.rateDisplayEl.textContent = '—';
    layerRangeEma = null;
    layerCols     = [];
    hoverState    = null;
    lastInterData = { inter1: null, inter2: null };
}

function setStoppedStatus() {
    setStatus('stopped', configCompatible(lastConfig, currentModelConfig()), !!model, false, !!snapshotWeights);
}

function trainerCallbacks() {
    return {
        onStep({ loss, step, rate, lastWeights: w, inter1, inter2, lossHistory }) {
            lastWeights = w;
            if (inter1 !== null) lastInterData = { inter1, inter2 };
            DOM.lossValueEl.textContent   = loss < 1e-4 ? loss.toExponential(3) : loss.toFixed(6);
            DOM.stepCounterEl.textContent = step.toLocaleString();
            DOM.rateDisplayEl.textContent = rate === '—' ? rate : rate + ' it/s';
            drawOutputCanvas(DOM.canvas, w.finalOutput, config.hasAlpha);
            ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(DOM.layersCanvas,
                { weights: w, inter1: lastInterData.inter1, inter2: lastInterData.inter2,
                  finalOutput: w.finalOutput, imgW: DOM.canvas.width, imgH: DOM.canvas.height,
                  config, channelMask, ema: layerRangeEma, hoverState }));
            sweep.setCols(layerCols); if (inter1 !== null) sweep.triggerStep();
            drawLossCurve(DOM.lossCanvas, lossHistory);
            drawSourceImage();
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
        targetPixels: get_target_pixels(loadedImage, DOM.canvas),
        getHyperparams: () => ({
            stride:      parseInt(DOM.bwdStrideInput.value) || 1,
            embedLr:     sliderToLR(parseInt(DOM.embedLrInput.value)),
            mlpLr:       sliderToLR(parseInt(DOM.mlpLrInput.value)),
            maxIter:     parseInt(DOM.maxIterInput.value),
            vizInterval: parseInt(DOM.vizIntervalSelect.value),
        }),
        ...trainerCallbacks(),
    });
}

function makeTrainer() {
    const targetPixels = get_target_pixels(loadedImage, DOM.canvas);
    const alphaCellMask = config.hasAlpha
        ? buildAlphaCellMask(targetPixels, DOM.canvas.width, DOM.canvas.height, config.gridSize)
        : null;
    return new Trainer({
        webGpuContext, canvas: DOM.canvas, config, model, pipeline, bindGroup, fwdUniformsBuf,
        outputBuffers, readbackBuffers,
        targetPixels,
        alphaCellMask,
        roiMask,
        getHyperparams: () => ({
            stride:      parseInt(DOM.bwdStrideInput.value) || 1,
            mlpRatio:    parseInt(DOM.mlpRatioInput.value)  || 1,
            numLoops:    parseInt(DOM.numLoopsInput.value)  || 1,
            embedLr:     sliderToLR(parseInt(DOM.embedLrInput.value)),
            mlpLr:       sliderToLR(parseInt(DOM.mlpLrInput.value)),
            roiStrength: parseFloat(DOM.roiStrengthInput.value),
            roiFreeze:   DOM.roiFreezeChk.checked,
            maxIter:     parseInt(DOM.maxIterInput.value),
            vizInterval:          parseInt(DOM.vizIntervalSelect.value),
            offsetSampleInterval: parseInt(DOM.offsetSampleIntervalSelect.value),
        }),
        ...trainerCallbacks(),
    });
}

// --- Start / Stop / Reset buttons ---
DOM.startBtn.addEventListener('click', async () => {
    if (isTraining()) {
        trainer.stop();
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

DOM.resetBtn.addEventListener('click', async () => {
    if (!loadedImage) return;
    await startTraining(true);
});

// --- Shake: add small noise to escape local minima ---
DOM.shakeEmbBtn.addEventListener('click', () => {
    if (!lastWeights) return;
    shakeEmbeddings(webGpuContext, model, lastWeights.embeddings);
});

DOM.shakeMlpBtn.addEventListener('click', async () => {
    if (!model) return;
    shakeMlp(webGpuContext, model, await readBackAllWeights());
});

// --- Weight readback (for export) ---
async function readBackAllWeights() {
    const { gridSize, embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = config;
    const outCh = config.hasAlpha ? 4 : 3;
    const tensors = {
        embeddings:     { buf: model.embeddings,     size: gridSize * gridSize * embCh },
        layer1_weights: { buf: model.layer1.weights, size: mlpWidth1 * embCh },
        layer1_biases:  { buf: model.layer1.biases,  size: mlpWidth1 },
        layer2_weights: { buf: model.layer2.weights, size: mlpWidth2 * mlpWidth1 },
        layer2_biases:  { buf: model.layer2.biases,  size: mlpWidth2 },
        layer3_weights: { buf: model.layer3.weights, size: outCh * mlpWidth2 },
        layer3_biases:  { buf: model.layer3.biases,  size: outCh },
    };
    return webGpuContext.readBackBuffers(tensors);
}

function weightsViewFrom(rb) {
    return {
        embeddings:    rb.embeddings,
        layer1Weights: rb.layer1_weights,
        layer2Weights: rb.layer2_weights,
        layer3Weights: rb.layer3_weights,
    };
}

// --- Export button ---
document.getElementById('export-btn').addEventListener('click', async () => {
    if (!model) {
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
        maxIter:    DOM.maxIterInput.value,
        embedLr:    DOM.embedLrInput.value,
        mlpLr:      DOM.mlpLrInput.value,
        mlpRatio:   DOM.mlpRatioInput.value,
        numLoops:   DOM.numLoopsInput.value,
        bwdStride:  DOM.bwdStrideInput.value,
        outputZoom: DOM.outputZoomInput.value,
        noOffset:   DOM.noOffsetCheckbox.checked,
    };
    return saveModelSafetensors(saveConfig, weights);
}

document.getElementById('save-btn').addEventListener('click', async () => {
    if (!model) { alert("Train or load a model first."); return; }
    const buf = await serializeModel();
    const url = URL.createObjectURL(new Blob([buf], { type: 'application/octet-stream' }));
    const a   = Object.assign(document.createElement('a'), { href: url, download: 'model.safetensors' });
    a.click();
    URL.revokeObjectURL(url);
});

// --- Snapshot / Recall ---
DOM.snapshotBtn.addEventListener('click', async () => {
    if (!model) { alert("Train or load a model first."); return; }
    snapshotWeights = await serializeModel();
    syncButtonStates(isTraining(), !!model, !!snapshotWeights);
});

DOM.recallBtn.addEventListener('click', async () => {
    if (!snapshotWeights) return;
    await loadModelFile(new Blob([snapshotWeights], { type: 'application/octet-stream' }));
});

// --- Zoom inference: forward pass at arbitrary resolution using temp buffers ---
async function runZoomInference(W, H) {
    const { device } = webGpuContext;
    const { gridSize, embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = config;
    const embSize    = gridSize * gridSize * embCh;
    const pixelCount = W * H;

    const embF32 = lastWeights.embeddings;
    const range  = computeEmbRange(embF32, embCh, gridSize * gridSize);
    webGpuContext.writeBuffer(model.embeddings_range, range);
    const packed = cpuPackEmbeddings(embF32, embCh, range, config.embBits);
    webGpuContext.writeBuffer(model.embeddings_q, packed);

    const unifBuf = webGpuContext.uniformBuffer(224);
    webGpuContext.writeBuffer(unifBuf, buildFwdUniforms(gridSize, embCh, mlpWidth1, mlpWidth2, W, H, range, config.embOffsets));
    uploadChannelMask(channelMask, unifBuf, device);
    const outBuf  = webGpuContext.outputBuffer(pixelCount * 4 * 4);
    const interL1 = webGpuContext.storageBuffer(pixelCount * mlpWidth1 * 4);
    const interL2 = webGpuContext.storageBuffer(pixelCount * mlpWidth2 * 4);
    const rbBuf   = webGpuContext.readbackBuffer(pixelCount * 4 * 4);

    const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0,  resource: { buffer: unifBuf } },
            { binding: 1,  resource: { buffer: model.embeddings_q } },
            { binding: 2,  resource: { buffer: model.layer1.weights } },
            { binding: 3,  resource: { buffer: model.layer1.biases  } },
            { binding: 4,  resource: { buffer: model.layer2.weights } },
            { binding: 5,  resource: { buffer: model.layer2.biases  } },
            { binding: 6,  resource: { buffer: model.layer3.weights } },
            { binding: 7,  resource: { buffer: model.layer3.biases  } },
            { binding: 8,  resource: { buffer: interL1 } },
            { binding: 9,  resource: { buffer: interL2 } },
            { binding: 10, resource: { buffer: outBuf  } },
        ],
    });

    const ce = device.createCommandEncoder({ label: 'zoom-inference' });
    const fwdPass = ce.beginComputePass({ label: 'fwd' });
    fwdPass.setPipeline(pipeline);
    fwdPass.setBindGroup(0, bg);
    fwdPass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    fwdPass.end();
    ce.copyBufferToBuffer(outBuf, 0, rbBuf, 0, outBuf.size);
    device.queue.submit([ce.finish()]);

    await rbBuf.mapAsync(GPUMapMode.READ);
    DOM.canvas.width  = W;
    DOM.canvas.height = H;
    drawOutputCanvas(DOM.canvas, new Float32Array(rbBuf.getMappedRange()), config.hasAlpha);
    rbBuf.unmap();

    unifBuf.destroy(); outBuf.destroy(); interL1.destroy(); interL2.destroy(); rbBuf.destroy();
}

// --- Channel mask: click / shift+click on the Emb column in the Layers canvas ---
// Map a mouse event to canvas pixel coords, accounting for object-fit:contain letterboxing.
function layersCanvasCoords(e) {
    const rect = DOM.layersCanvas.getBoundingClientRect();
    const cw = DOM.layersCanvas.width, ch = DOM.layersCanvas.height;
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
    if (!lastWeights || !config?.mlpWidth1) return;
    ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(DOM.layersCanvas,
        { weights: lastWeights, inter1: lastInterData.inter1, inter2: lastInterData.inter2,
          finalOutput: lastWeights.finalOutput, imgW: DOM.canvas.width, imgH: DOM.canvas.height,
          config, channelMask, ema: layerRangeEma, hoverState }));
}

DOM.layersCanvas.addEventListener('mousemove', (e) => {
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
    DOM.layersCanvas.style.cursor = newState?.col === 'emb' && !isTraining() ? 'pointer' : 'default';
    if (newState?.col === hoverState?.col && newState?.ch === hoverState?.ch) return;
    hoverState = newState;
    redrawLayers();
});

DOM.layersCanvas.addEventListener('mouseleave', () => {
    DOM.layersCanvas.style.cursor = 'default';
    if (hoverState === null) return;
    hoverState = null;
    redrawLayers();
});

DOM.layersCanvas.addEventListener('click', (e) => {
    if (isTraining()) return;
    const embCol = layerCols.find(c => c.name === 'emb');
    if (!embCol?.slotH) return;
    const { cx, cy } = layersCanvasCoords(e);
    if (cx < embCol.x || cx >= embCol.x + embCol.w) return;
    const embCh = config?.embeddingChannels ?? 0;
    if (!embCh) return;
    const ch = Math.min(Math.max(0, ((cy - embCol.y0) / embCol.slotH) | 0), embCh - 1);
    const allBits = embCh < 32 ? (1 << embCh) - 1 : 0xFFFFFFFF;
    channelMask ^= (allBits ^ (1 << ch));
    if (canInfer()) runInference();
    else redrawLayers();
});

// --- Inference: pack embeddings (CPU) then run one forward pass ---
async function runInference() {
    if (inferRunning) { inferPending = true; return; }
    inferRunning = true;
    try {
        const { device } = webGpuContext;
        const { gridSize, embeddingChannels: embCh } = config;

        const embF32 = lastWeights.embeddings;
        const range  = computeEmbRange(embF32, embCh, gridSize * gridSize);
        webGpuContext.writeBuffer(model.embeddings_range, range);
        uploadEmbRange(range, embCh, fwdUniformsBuf, device);
        const packed = cpuPackEmbeddings(embF32, embCh, range, config.embBits);
        webGpuContext.writeBuffer(model.embeddings_q, packed);

        uploadChannelMask(channelMask, fwdUniformsBuf, device);
        const ce = device.createCommandEncoder({ label: 'inference' });
        const fwdPass = ce.beginComputePass({ label: 'fwd' });
        fwdPass.setPipeline(pipeline);
        fwdPass.setBindGroup(0, bindGroup);
        fwdPass.dispatchWorkgroups(Math.ceil(DOM.canvas.width / 8), Math.ceil(DOM.canvas.height / 8));
        fwdPass.end();
        ce.copyBufferToBuffer(outputBuffers.final,       0, readbackBuffers.final,       0, outputBuffers.final.size);
        ce.copyBufferToBuffer(outputBuffers.interLayer1, 0, readbackBuffers.interLayer1, 0, readbackBuffers.interLayer1.size);
        ce.copyBufferToBuffer(outputBuffers.interLayer2, 0, readbackBuffers.interLayer2, 0, readbackBuffers.interLayer2.size);
        device.queue.submit([ce.finish()]);
        sweep.setCols(layerCols); sweep.triggerFwd();

        await Promise.all([
            readbackBuffers.final.mapAsync(GPUMapMode.READ),
            readbackBuffers.interLayer1.mapAsync(GPUMapMode.READ),
            readbackBuffers.interLayer2.mapAsync(GPUMapMode.READ),
        ]);
        const inferFinal  = new Float32Array(readbackBuffers.final.getMappedRange()).slice();
        const inferInter1 = new Float32Array(readbackBuffers.interLayer1.getMappedRange()).slice();
        const inferInter2 = new Float32Array(readbackBuffers.interLayer2.getMappedRange()).slice();
        drawOutputCanvas(DOM.canvas, inferFinal, config.hasAlpha);
        readbackBuffers.final.unmap();
        readbackBuffers.interLayer1.unmap();
        readbackBuffers.interLayer2.unmap();

        lastInterData = { inter1: inferInter1, inter2: inferInter2 };
        lastWeights.finalOutput = inferFinal;
        ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(DOM.layersCanvas,
            { weights: lastWeights, inter1: inferInter1, inter2: inferInter2,
              finalOutput: inferFinal, imgW: DOM.canvas.width, imgH: DOM.canvas.height,
              config, channelMask, ema: layerRangeEma, hoverState }));
        sweep.setCols(layerCols);
    } finally {
        inferRunning = false;
        if (inferPending) { inferPending = false; runInference(); }
    }
}

async function loadAndResetModelFile(file) {
    trainer?.destroy();
    trainer = null;
    clearTrainingUI();
    snapshotWeights = null;
    roiMask.clear();
    await loadModelFile(file);
    syncButtonStates(false, !!model, !!snapshotWeights);
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

    DOM.gridSizeSelect.value          = String(savedConfig.gridSize);
    DOM.embeddingChannelsSelect.value = String(savedConfig.embeddingChannels);
    DOM.mlpWidth1Select.value         = String(savedConfig.mlpWidth1);
    DOM.mlpWidth2Select.value         = String(savedConfig.mlpWidth2);

    restoreUISettings({ savedConfig, uiSettings });

    const loadedEmbBits = savedConfig.embBits || 8;
    DOM.embBitsSelect.value = String(loadedEmbBits);

    const outCh = tensors.layer3_biases.length;  // 3 or 4
    imageHasAlpha = outCh === 4;
    const { gridSize, embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = savedConfig;
    const expectedSizes = {
        embeddings:     gridSize * gridSize * embCh,
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
        gridSize:            gridSize,
        embeddingChannels:   embCh,
        mlpWidth1:           mlpWidth1,
        mlpWidth2:           mlpWidth2,
        quantization:        DOM.quantizationSelect.value,
        embBits:             loadedEmbBits,
        activation:          DOM.activationSelect.value,
        smoothInterpolation: DOM.smoothInterpolationCheckbox.checked,
        hasAlpha:            outCh === 4,
        width:  DOM.canvas.width,
        height: DOM.canvas.height,
        embOffsets:          savedConfig.embOffsets ?? new Float32Array(embCh / (32 / loadedEmbBits) * 2),
    };

    try {
        if (!webGpuContext) webGpuContext = await initWebGPU();
        const { buffers } = createModel(webGpuContext, config);
        destroyModel(model);
        model = buffers;

        webGpuContext.uploadModelWeights(model, tensors);

        await createPipeline();
        createBindGroup();
        channelMask = 0xFFFFFFFF;
        lastConfig  = currentModelConfig();
        updateDirtyIndicators(lastConfig, currentModelConfig());

        lastWeights = weightsViewFrom(tensors);

        await runInference();
        setStoppedStatus();
        updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
        updateStartLabel(configCompatible(lastConfig, currentModelConfig()), !!model, false, !!snapshotWeights);
    } catch (err) {
        console.error('Load model failed:', err);
        alert('Load model failed. Check console for errors.');
    }
}


function refreshStartLabel() {
    updateStartLabel(configCompatible(lastConfig, currentModelConfig()), !!model, isTraining(), !!snapshotWeights);
}

initTooltips();
initUI({
    onConfigChange: () => {
        refreshStartLabel();
        updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
    },
    onSelectChange: () => {
        refreshStartLabel();
        updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
        if (!isTraining()) {
            updateDirtyIndicators(lastConfig, currentModelConfig());
            const cur = currentModelConfig();
            if (lastWeights && configCompatible(lastConfig, cur)) {
                if (canInfer()) runInference(); else redrawLayers();
            } else if (webGpuContext && loadedImage) {
                previewGeometry();
            } else {
                drawPlaceholder(DOM.layersCanvas);
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
        if (!webGpuContext) webGpuContext = await initWebGPU();
        let modelLoaded = false;
        try {
            const resp = await fetch(modelUrl);
            if (resp.ok) { await loadAndResetModelFile(await resp.blob()); modelLoaded = true; }
        } catch (err) { console.warn('Example model fetch failed:', err); }
        if (!modelLoaded) {
            trainer?.destroy(); trainer = null; clearTrainingUI(); snapshotWeights = null;
            roiMask.clear();
            await resetToRandomModel();
            setStoppedStatus();
            updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
        }
    } catch (err) { console.warn('Example load failed:', err); }
}

// Default engine: CPU on mobile / no WebGPU, GPU otherwise
if (!navigator.gpu) DOM.engineSelect.value = 'cpu';

// --- Start tooltip: show once startup is complete ---
(function() {
    const tip = document.getElementById('start-tooltip');
    const dismiss = () => tip.remove();
    document.addEventListener('startup-complete', () => {
        const r = DOM.startBtn.getBoundingClientRect();
        tip.style.left = (r.left + r.width / 2) + 'px';
        tip.style.top = (r.bottom + 10) + 'px';
        tip.classList.add('visible');
        tip.addEventListener('animationend', dismiss, { once: true });
    }, { once: true });
    DOM.startBtn.addEventListener('click', dismiss, { once: true });
})();

// --- Startup: load default image then default model (or auto-start from URL params) ---
const urlHasParams = applyUrlParams();
if (urlHasParams) {
    refreshStartLabel();
    updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
}
const urlExample = getUrlExample();
if (urlExample) {
    (async () => {
        try { if (!webGpuContext) webGpuContext = await initWebGPU(); }
        catch (err) { console.error('WebGPU init failed:', err); return; }
        await loadExample(urlExample);
        document.dispatchEvent(new Event('startup-complete'));
    })();
} else {
    const startupImg = new Image();
    startupImg.onload = async () => {
        loadImageOntoCanvas(startupImg);
        if (urlHasParams) {
            DOM.startBtn.click();
        } else {
            try { if (!webGpuContext) webGpuContext = await initWebGPU(); }
            catch (err) { console.error('WebGPU init failed:', err); return; }
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
