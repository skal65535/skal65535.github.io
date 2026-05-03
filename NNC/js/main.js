import { initWebGPU, buildShader, buildBackwardShaders } from './webgpu.js';
import { ModelTensors, createModel, destroyModel, initCpuWeights, shakeEmbeddings, shakeMlp, computeEmbRange, buildFwdUniforms, uploadEmbRange, uploadEmbOffsets, uploadChannelMask, cpuPackEmbeddings, generateEmbOffsets, computeTensorSizes } from './model.js';
import { get_target_pixels } from './loss.js';
import { export_to_glsl } from './shader_exporter.js';
import { saveModelSafetensors, loadModelSafetensors } from './model_io.js';
import { ROIMask } from './roi_mask.js';
import { drawOutputCanvas, drawFlowDiagram, drawLossCurve, FLOW_PAD, FLOW_EMB_W } from './viz.js';
import { ui, init as initUI, drawPlaceholder, updateSizeDisplay, updateDirtyIndicators, syncButtonStates, updateStartLabel, setStatus, restoreUISettings, syncSliderDisplay, sliderToLR, updateEmbBitsOptions, fitSidePanels, applyUrlParams } from './ui.js?v=2';
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
const sweep        = new SweepOverlay(ui.sweepCanvas, ui.sourceSweepCanvas);

const isTraining   = () => trainer?.active ?? false;
const configReady  = () => !!config.mlpWidth1;
const canInfer     = () => !!lastWeights && !isTraining();
const hasCpuWeights = () => !!lastWeights?.layer1_biases;  // full weight tensors loaded (not partial GPU readback)
const doInference  = () => model ? runInference() : (hasCpuWeights() ? runInferenceCpu() : null);

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
    const mlpWidth1 = parseInt(ui.mlpWidth1Select.value);
    const mlpWidth2 = parseInt(ui.mlpWidth2Select.value);
    const embeddingChannels = parseInt(ui.embeddingChannelsSelect.value);
    return {
        gridSize:          parseInt(ui.gridSizeSelect.value),
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
    destroyModel(model);
    config = buildConfigFromUI();
    config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, ui.noOffsetCheckbox.checked);
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
    const useCpu = ui.engineSelect.value === 'cpu';
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
            config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, ui.noOffsetCheckbox.checked);
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
                : generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, ui.noOffsetCheckbox.checked);
            clearTrainingUI();
            setStatus('training', configCompatible(lastConfig, currentModelConfig()), !!model, isTraining(), !!snapshotWeights);
            sweep.resetDecay(); trainer.start(null);
        }
        return;
    }

    try {
        if (!webGpuContext) webGpuContext = await initWebGPU();
        if (fullReset || configChanged || !model) {
            config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, ui.noOffsetCheckbox.checked);
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
                : generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, ui.noOffsetCheckbox.checked);
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
    ui.canvas.width  = BASE_CANVAS_W;
    ui.canvas.height = BASE_CANVAS_H;
    ui.canvas.style.width = '';
    ui.canvas.style.height = '';
    ui.canvas.parentElement.style.overflow = 'hidden';
}

function buildConfigFromUI() {
    return {
        gridSize:            parseInt(ui.gridSizeSelect.value),
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
        buildFwdUniforms(gridSize, embeddingChannels, mlpWidth1, mlpWidth2, ui.canvas.width, ui.canvas.height, null, config.embOffsets));

    const pixelCount = ui.canvas.width * ui.canvas.height;
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
            { binding: 0, resource: { buffer: fwdUniformsBuf } },
            { binding: 1, resource: { buffer: model.embeddings_q } },
            { binding: 2, resource: { buffer: model.mlp_weights } },
            { binding: 3, resource: { buffer: outputBuffers.interLayer1 } },
            { binding: 4, resource: { buffer: outputBuffers.interLayer2 } },
            { binding: 5, resource: { buffer: outputBuffers.final       } },
        ],
    });
}

function clearTrainingUI() {
    stopDecayLoop();
    ui.lossCanvas.getContext('2d').clearRect(0, 0, ui.lossCanvas.width, ui.lossCanvas.height);
    ui.canvas.getContext('2d').clearRect(0, 0, ui.canvas.width, ui.canvas.height);
    ui.lossValueEl.textContent   = '—';
    ui.stepCounterEl.textContent = '0';
    ui.rateDisplayEl.textContent = '—';
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
        ? buildAlphaCellMask(targetPixels, ui.canvas.width, ui.canvas.height, config.gridSize)
        : null;
    return new Trainer({
        webGpuContext, canvas: ui.canvas, config, model, pipeline, bindGroup, fwdUniformsBuf,
        outputBuffers, readbackBuffers,
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

ui.resetBtn.addEventListener('click', async () => {
    if (!loadedImage) return;
    await startTraining(true);
});

// --- Shake: add small noise to escape local minima ---
ui.shakeEmbBtn.addEventListener('click', () => {
    if (!lastWeights) return;
    shakeEmbeddings(webGpuContext, model, lastWeights.embeddings);
});

ui.shakeMlpBtn.addEventListener('click', async () => {
    if (!model) return;
    shakeMlp(webGpuContext, model, await readBackAllWeights());
});

// --- Weight readback (for export) ---
async function readBackAllWeights() {
    const sizes = computeTensorSizes(config);
    const tensors = {
        embeddings:     { buf: model.embeddings,     size: sizes.embeddings },
        layer1_weights: { buf: model.layer1.weights, size: sizes.layer1_weights },
        layer1_biases:  { buf: model.layer1.biases,  size: sizes.layer1_biases },
        layer2_weights: { buf: model.layer2.weights, size: sizes.layer2_weights },
        layer2_biases:  { buf: model.layer2.biases,  size: sizes.layer2_biases },
        layer3_weights: { buf: model.layer3.weights, size: sizes.layer3_weights },
        layer3_biases:  { buf: model.layer3.biases,  size: sizes.layer3_biases },
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
        layer1Weights:  rb.layer1_weights,
        layer2Weights:  rb.layer2_weights,
        layer3Weights:  rb.layer3_weights,
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
    if (!model) { alert("Train or load a model first."); return; }
    const buf = await serializeModel();
    const url = URL.createObjectURL(new Blob([buf], { type: 'application/octet-stream' }));
    const a   = Object.assign(document.createElement('a'), { href: url, download: 'model.safetensors' });
    a.click();
    URL.revokeObjectURL(url);
});

// --- Snapshot / Recall ---
ui.snapshotBtn.addEventListener('click', async () => {
    if (!model) { alert("Train or load a model first."); return; }
    snapshotWeights = await serializeModel();
    syncButtonStates(isTraining(), !!model, !!snapshotWeights);
});

ui.recallBtn.addEventListener('click', async () => {
    if (!snapshotWeights) return;
    await loadModelFile(new Blob([snapshotWeights], { type: 'application/octet-stream' }));
});

// --- Zoom inference: forward pass at arbitrary resolution using temp buffers ---
async function runZoomInference(W, H) {
    const { final } = await runZoomInferencePass({
        webGpuContext, config, model, pipeline, lastWeights, channelMask, W, H
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
    if (inferRunning) { inferPending = true; return; }
    inferRunning = true;
    try {
        const { final: inferFinal, inter1: inferInter1, inter2: inferInter2 } = await runInferencePass({
            webGpuContext, config, model, pipeline, bindGroup, fwdUniformsBuf, outputBuffers, readbackBuffers, channelMask,
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
        inferRunning = false;
        if (inferPending) { inferPending = false; runInference(); }
    }
}

async function runInferenceCpu() {
    const { gridSize, embeddingChannels: embCh } = config;
    const range  = computeEmbRange(lastWeights.embeddings, embCh, gridSize * gridSize);
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

    ui.gridSizeSelect.value          = String(savedConfig.gridSize);
    ui.embeddingChannelsSelect.value = String(savedConfig.embeddingChannels);
    ui.mlpWidth1Select.value         = String(savedConfig.mlpWidth1);
    ui.mlpWidth2Select.value         = String(savedConfig.mlpWidth2);

    restoreUISettings({ savedConfig, uiSettings });

    const loadedEmbBits = savedConfig.embBits || 8;
    ui.embBitsSelect.value = String(loadedEmbBits);

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
        quantization:        ui.quantizationSelect.value,
        embBits:             loadedEmbBits,
        activation:          ui.activationSelect.value,
        smoothInterpolation: ui.smoothInterpolationCheckbox.checked,
        hasAlpha:            outCh === 4,
        width:  ui.canvas.width,
        height: ui.canvas.height,
        embOffsets:          savedConfig.embOffsets ?? new Float32Array(embCh / (32 / loadedEmbBits) * 2),
    };

    lastWeights = weightsViewFrom(tensors);
    lastConfig  = currentModelConfig();
    updateDirtyIndicators(lastConfig, currentModelConfig());
    channelMask = 0xFFFFFFFF;

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
            const { buffers } = createModel(webGpuContext, config);
            destroyModel(model);
            model = buffers;
            webGpuContext.uploadModelWeights(model, tensors);
            await createPipeline();
            createBindGroup();
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
    updateStartLabel(configCompatible(lastConfig, currentModelConfig()), !!model, false, !!snapshotWeights);
}


function refreshStartLabel() {
    updateStartLabel(configCompatible(lastConfig, currentModelConfig()), !!model, isTraining(), !!snapshotWeights);
}

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
                trainer?.destroy(); trainer = null; clearTrainingUI(); snapshotWeights = null;
                roiMask.clear();
                await resetToRandomModel();
                setStoppedStatus();
                updateSizeDisplay(BASE_CANVAS_W, BASE_CANVAS_H);
            } catch (err) { console.warn('Example GPU init failed:', err); }
        }
    } catch (err) { console.warn('Example load failed:', err); }
}

if (!gpuAvailable) disableGpu();

// --- Start tooltip: show once startup is complete ---
(function() {
    const tip = document.getElementById('start-tooltip');
    const dismiss = () => tip.remove();
    document.addEventListener('startup-complete', () => {
        const r = ui.startBtn.getBoundingClientRect();
        tip.style.left = (r.left + r.width / 2) + 'px';
        tip.style.top = (r.bottom + 10) + 'px';
        tip.classList.add('visible');
        tip.addEventListener('animationend', dismiss, { once: true });
    }, { once: true });
    ui.startBtn.addEventListener('click', dismiss, { once: true });
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
