// main.js
// Orchestrates UI, WebGPU setup, and GPU training loop (forward + backward + Adam).
import { initWebGPU } from './webgpu_manager.js';
import { createModel } from './model.js';
import { buildShader } from './shader_builder.js';
import { get_target_pixels } from './loss.js';
import { export_to_glsl } from './shader_exporter.js';
import { saveModelSafetensors, loadModelSafetensors } from './model_io.js';
import { ROIMask } from './roi_mask.js';
import { drawOutputCanvas, drawFlowDiagram, drawLossCurve, FLOW_PAD, FLOW_EMB_W } from './viz.js';
import { computeEmbRange, buildFwdUniforms, uploadEmbRange, uploadChannelMask, cpuPackEmbeddings, generateEmbOffsets } from './emb_utils.js';
import { initTooltips } from './tooltips.js';
import { Trainer, setVizInterval } from './trainer.js';

// --- DOM references ---
const sourcePanel  = document.getElementById('source-panel');
const outputPanel  = document.getElementById('output-panel');
const canvasRow    = document.querySelector('.canvas-row');
const sourceCanvas = document.getElementById('source-canvas');
const layersCanvas = document.getElementById('layers-canvas');
const canvas       = document.getElementById('canvas');
const lossCanvas   = document.getElementById('loss-canvas');
const fileInput    = document.getElementById('file-input');
const dropOverlay  = document.getElementById('drop-overlay');

const gridSizeSelect            = document.getElementById('grid-size');
const embeddingChannelsSelect   = document.getElementById('embedding-channels');
const mlpWidthSelect            = document.getElementById('mlp-width');
const quantizationSelect        = document.getElementById('quantization');
const embBitsSelect             = document.getElementById('emb-bits');
const activationSelect          = document.getElementById('activation');
function syncActivationButtons() {
    document.querySelectorAll('.activ-btn').forEach(btn => {
        btn.classList.toggle('selected', btn.dataset.val === activationSelect.value);
    });
}
document.querySelectorAll('.activ-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        activationSelect.value = btn.dataset.val;
        syncActivationButtons();
        activationSelect.dispatchEvent(new Event('change'));
    });
});
syncActivationButtons();
const smoothInterpolationCheckbox = document.getElementById('smooth-interpolation');
const noOffsetCheckbox            = document.getElementById('no-offset');
const maxIterInput  = document.getElementById('max-iter');
const embedLrInput  = document.getElementById('embed-lr');
const mlpLrInput    = document.getElementById('mlp-lr');
const maxIterVal    = document.getElementById('max-iter-val');
const embedLrVal    = document.getElementById('embed-lr-val');
const mlpLrVal      = document.getElementById('mlp-lr-val');
const mlpRatioInput  = document.getElementById('mlp-ratio');
const mlpRatioVal    = document.getElementById('mlp-ratio-val');
const numLoopsInput  = document.getElementById('num-loops');
const numLoopsVal    = document.getElementById('num-loops-val');
const bwdStrideInput = document.getElementById('bwd-stride');
const bwdStrideVal   = document.getElementById('bwd-stride-val');
const outputZoomInput = document.getElementById('output-zoom');
const outputZoomVal   = document.getElementById('output-zoom-val');
const startBtn      = document.getElementById('start-btn');
const resetBtn      = document.getElementById('reset-btn');
const shakeBtn      = document.getElementById('shake-btn');
const saveBtn       = document.getElementById('save-btn');
const loadBtn       = document.getElementById('load-btn');
const snapshotBtn   = document.getElementById('snapshot-btn');
const modelFileInput = document.getElementById('model-file-input');
const lossValueEl   = document.getElementById('loss-value');
const stepCounterEl = document.getElementById('step-counter');
const rateDisplayEl = document.getElementById('rate-display');
const modelSizeEl   = document.getElementById('model-size');
const inputSizeEl   = document.getElementById('input-size');
const statusDotEl   = document.getElementById('status-dot');
const statusTextEl  = document.getElementById('status-text');
const sourceResEl   = document.getElementById('source-res');
const outputResEl   = document.getElementById('output-res');
const roiBrushInput    = document.getElementById('roi-brush');
const roiBrushVal      = document.getElementById('roi-brush-val');
const roiStrengthInput = document.getElementById('roi-strength');
const roiStrengthVal   = document.getElementById('roi-strength-val');
const roiFreezeChk     = document.getElementById('roi-freeze');
const roiClearBtn      = document.getElementById('roi-clear-btn');
const roiAutoBtn       = document.getElementById('roi-auto-btn');

// --- State ---
let BASE_CANVAS_W = canvas.width;
let BASE_CANVAS_H = canvas.height;
let config = {};
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

const isTraining = () => trainer?.active ?? false;
const canInfer   = () => !!model && !!lastWeights && !isTraining();

// ROI mask state
const roiMask  = new ROIMask(1, 1);
let decayRafId = null;

function drawPlaceholder(ctx_canvas) {
    const ctx = ctx_canvas.getContext('2d');
    ctx.clearRect(0, 0, ctx_canvas.width, ctx_canvas.height);
    ctx.fillStyle = '#1a1a2a';
    ctx.fillRect(0, 0, ctx_canvas.width, ctx_canvas.height);
    ctx.fillStyle = '#555';
    ctx.font = `${Math.max(12, Math.min(18, ctx_canvas.width / 14))}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('train or load a model', ctx_canvas.width / 2, ctx_canvas.height / 2);
}
const vizIntervalSelect = document.getElementById('viz-interval');
vizIntervalSelect.addEventListener('change', () => setVizInterval(parseInt(vizIntervalSelect.value)));

// --- Compressed size estimate ---
function computeModelSize() {
    const gs   = parseInt(gridSizeSelect.value);
    const embCh = parseInt(embeddingChannelsSelect.value);
    const mlpW  = parseInt(mlpWidthSelect.value);
    const emb   = gs * gs * embCh;
    const mlp   = embCh*mlpW + mlpW + mlpW*mlpW + mlpW + mlpW*4 + 4;
    const mlpBpp = quantizationSelect.value === 'none' ? 4 : 1;
    const embBpp = parseInt(embBitsSelect.value) / 8;
    return emb * embBpp + mlp * mlpBpp;
}

function updateSizeDisplay() {
    const b = computeModelSize();
    const pixels = BASE_CANVAS_W * BASE_CANVAS_H;
    const bpp = pixels > 0 ? (b * 8 / pixels).toFixed(2) : '—';
    const sizeStr = b >= 1024 ? (b / 1024).toFixed(1) + ' KB' : b + ' B';
    modelSizeEl.textContent = `${sizeStr} (${bpp} bpp)`;
}

// --- LR slider helpers (log scale: slider 0–100 → 1e-4 … ~1.2e-1; v=80 → 3e-2) ---
const sliderToLR = v => 1e-4 * Math.pow(300, v / 80);
const formatLR   = lr => lr.toExponential(1);

function syncSliderDisplay(input, display) {
    display.textContent = formatLR(sliderToLR(parseInt(input.value)));
}

syncSliderDisplay(embedLrInput, embedLrVal);
syncSliderDisplay(mlpLrInput,   mlpLrVal);
updateSizeDisplay();

embedLrInput.addEventListener('input', () => syncSliderDisplay(embedLrInput, embedLrVal));
mlpLrInput.addEventListener('input',   () => syncSliderDisplay(mlpLrInput, mlpLrVal));
maxIterInput.addEventListener('input', () => {
    const v = parseInt(maxIterInput.value);
    maxIterVal.textContent = v === 0 ? '∞' : v.toLocaleString();
});
mlpRatioInput.addEventListener('input', () => { mlpRatioVal.textContent = mlpRatioInput.value; });
numLoopsInput.addEventListener('input', () => { numLoopsVal.textContent = numLoopsInput.value; });
bwdStrideInput.addEventListener('input', () => { bwdStrideVal.textContent = bwdStrideInput.value; });
outputZoomInput.addEventListener('input', () => {
    const scale = parseFloat(outputZoomInput.value);
    outputZoomVal.textContent = scale + '×';
    if (canInfer()) {
        const W = Math.round(BASE_CANVAS_W * scale);
        const H = Math.round(BASE_CANVAS_H * scale);
        outputResEl.textContent = `${W}×${H}`;
        runZoomInference(W, H);
    }
});

function isValidModelConfig({ gridSize, embeddingChannels, mlpWidth }) {
    return [16, 32, 64].includes(gridSize) &&
           [4, 8, 16].includes(embeddingChannels) &&
           [4, 8, 16, 32, 64].includes(mlpWidth);
}

function configCompatible(a, b) {
    return a && b &&
        a.gridSize === b.gridSize &&
        a.embeddingChannels === b.embeddingChannels &&
        a.mlpWidth === b.mlpWidth;
}

function currentModelConfig() {
    const mlpWidth = parseInt(mlpWidthSelect.value);
    const embeddingChannels = parseInt(embeddingChannelsSelect.value);
    return {
        gridSize:          parseInt(gridSizeSelect.value),
        embeddingChannels: Math.ceil(embeddingChannels / 4) * 4,
        mlpWidth:          Math.ceil(mlpWidth / 4) * 4,
        embBits:           parseInt(embBitsSelect.value),
    };
}

const STRUCTURAL_CONTROLS = [gridSizeSelect, embeddingChannelsSelect, mlpWidthSelect];
function updateDirtyIndicators() {
    if (!lastConfig) { STRUCTURAL_CONTROLS.forEach(el => el.removeAttribute('data-dirty')); return; }
    const cur = currentModelConfig();
    gridSizeSelect.toggleAttribute('data-dirty',          lastConfig.gridSize            !== cur.gridSize);
    embeddingChannelsSelect.toggleAttribute('data-dirty', lastConfig.embeddingChannels   !== cur.embeddingChannels);
    mlpWidthSelect.toggleAttribute('data-dirty',          lastConfig.mlpWidth            !== cur.mlpWidth);
}

function syncButtonStates() {
    const training      = isTraining();
    const hasModel      = !!model;
    const shouldDisable = !hasModel || training;
    resetBtn.disabled        = shouldDisable;
    shakeBtn.disabled        = shouldDisable;
    outputZoomInput.disabled = shouldDisable;
    snapshotBtn.disabled     = !hasModel || (training && snapshotWeights !== null);
}

function updateStartLabel() {
    if (configCompatible(lastConfig, currentModelConfig()) && model) {
        startBtn.textContent = '▶ Continue';
    } else {
        startBtn.textContent = '▶ Start';
    }
    syncButtonStates();
}

// --- Status indicator ---
function setStatus(s) {
    const labels = { idle: 'IDLE', training: 'TRAINING', stopped: 'STOPPED' };
    statusTextEl.textContent = labels[s] || s;
    statusDotEl.className    = 'status-dot ' + s;
    if (s === 'training') {
        startBtn.textContent = '■ Stop';
        startBtn.classList.add('stopping');
    } else {
        startBtn.classList.remove('stopping');
        updateStartLabel();
        return;
    }
    syncButtonStates();
}

// --- Source canvas ---
function drawSourceImage() {
    if (!loadedImage) return;
    const ctx = sourceCanvas.getContext('2d');
    ctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
    ctx.drawImage(loadedImage, 0, 0, sourceCanvas.width, sourceCanvas.height);
    roiMask.drawOverlay(ctx);
    dropOverlay.classList.add('hidden');
}

function startDecayLoop() {
    if (decayRafId !== null || !roiMask.isActive()) return;
    function tick(now) {
        if (!roiFreezeChk.checked) roiMask.decay(now);
        drawSourceImage();
        decayRafId = (roiFreezeChk.checked || roiMask.isActive()) ? requestAnimationFrame(tick) : null;
    }
    decayRafId = requestAnimationFrame(tick);
}

function stopDecayLoop() {
    if (decayRafId !== null) { cancelAnimationFrame(decayRafId); decayRafId = null; }
}

// --- Collapsible sidebar sections ---
document.querySelectorAll('#sidebar .section-label').forEach(label => {
    label.addEventListener('click', () => label.closest('.ctrl-section').classList.toggle('collapsed'));
});

// --- ROI mask controls ---
roiBrushInput.addEventListener('input', () => { roiBrushVal.textContent = roiBrushInput.value; });
roiStrengthInput.addEventListener('input', () => { roiStrengthVal.textContent = roiStrengthInput.value; });
roiFreezeChk.addEventListener('change', () => {
    if (!roiFreezeChk.checked && roiMask.isActive() && !isTraining()) startDecayLoop();
});
roiClearBtn.addEventListener('click', () => {
    roiMask.clear();
    stopDecayLoop();
    drawSourceImage();
});
roiAutoBtn.addEventListener('click', () => {
    if (!loadedImage) return;
    const ctx = sourceCanvas.getContext('2d');
    const id  = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    roiMask.autoMask(id.data);
    roiFreezeChk.checked = true;
    stopDecayLoop();
    drawSourceImage();
});

{
    let isPainting = false, didPaint = false;
    function sourceCanvasCoords(e) {
        const r = sourceCanvas.getBoundingClientRect();
        return [
            (e.clientX - r.left) * sourceCanvas.width  / r.width,
            (e.clientY - r.top)  * sourceCanvas.height / r.height,
        ];
    }
    sourceCanvas.addEventListener('mousedown', (e) => {
        if (!loadedImage) return;
        isPainting = true;
        didPaint = false;
        const [x, y] = sourceCanvasCoords(e);
        roiMask.paint(x, y, parseInt(roiBrushInput.value));
        drawSourceImage();
        if (!isTraining()) startDecayLoop();
    });
    sourceCanvas.addEventListener('mousemove', (e) => {
        if (!isPainting) return;
        didPaint = true;
        const [x, y] = sourceCanvasCoords(e);
        roiMask.paint(x, y, parseInt(roiBrushInput.value));
        drawSourceImage();
    });
    window.addEventListener('mouseup',    () => { isPainting = false; });
    sourceCanvas.addEventListener('mouseleave', () => { isPainting = false; });

    // --- File / drop zone ---
    dropOverlay.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
    sourcePanel.addEventListener('click', (e) => {
        if (e.target !== fileInput && !(e.target === sourceCanvas && didPaint)) fileInput.click();
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) { handleFile(e.target.files[0]); fileInput.value = ''; }
    });
}
sourcePanel.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('hidden');
    dropOverlay.classList.add('dragover');
});
sourcePanel.addEventListener('dragleave', () => {
    dropOverlay.classList.remove('dragover');
    if (loadedImage) dropOverlay.classList.add('hidden');
});
sourcePanel.addEventListener('drop', (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function fitSidePanels() {
    if (!loadedImage) return;
    const rowH = canvasRow.clientHeight;
    [sourcePanel, outputPanel].forEach(panel => {
        const hdrH = panel.querySelector('.panel-header').offsetHeight;
        const w = Math.round((rowH - hdrH) * BASE_CANVAS_W / BASE_CANVAS_H);
        panel.style.width = w + 'px';
    });
}
window.addEventListener('resize', fitSidePanels);

function loadImageOntoCanvas(img) {
    loadedImage = img;
    const MAX = 512;
    const aspect = img.naturalWidth / img.naturalHeight;
    BASE_CANVAS_W = aspect >= 1 ? MAX : Math.round(MAX * aspect);
    BASE_CANVAS_H = aspect >= 1 ? Math.round(MAX / aspect) : MAX;
    sourceCanvas.width  = BASE_CANVAS_W;
    sourceCanvas.height = BASE_CANVAS_H;
    canvas.width  = BASE_CANVAS_W;
    canvas.height = BASE_CANVAS_H;
    canvas.style.width = '';
    canvas.style.height = '';
    canvas.style.maxWidth = '';
    canvas.style.maxHeight = '';
    roiMask.resize(BASE_CANVAS_W, BASE_CANVAS_H);
    sourcePanel.classList.add('has-image');
    sourceResEl.textContent = `${BASE_CANVAS_W}×${BASE_CANVAS_H}`;
    outputResEl.textContent = `${BASE_CANVAS_W}×${BASE_CANVAS_H}`;
    drawSourceImage();
    {
        const ctx = sourceCanvas.getContext('2d');
        const px = ctx.getImageData(0, 0, BASE_CANVAS_W, BASE_CANVAS_H).data;
        let hasAlpha = false;
        for (let i = 3; i < px.length; i += 4) { if (px[i] < 255) { hasAlpha = true; break; } }
        config.hasAlpha = hasAlpha;
    }
    const outCh = config.hasAlpha ? 4 : 3;
    const bytes = BASE_CANVAS_W * BASE_CANVAS_H * outCh;
    inputSizeEl.textContent = bytes >= 1024 ? (bytes / 1024).toFixed(1) + ' KB' : bytes + ' B';
    requestAnimationFrame(fitSidePanels);
}

async function previewGeometry() {
    if (!webGpuContext || !loadedImage) return;
    trainer?.destroy();
    trainer = null;
    config = buildConfigFromUI();
    config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, noOffsetCheckbox.checked);
    const { buffers, weights: freshWeights } = createModel(webGpuContext, config);
    model = buffers;
    await createPipeline();
    createBindGroup();
    channelMask = 0xFFFFFFFF;
    lastConfig = currentModelConfig();
    updateDirtyIndicators();
    clearTrainingUI();
    trainer = new Trainer({
        webGpuContext, canvas, config, model, pipeline, bindGroup, fwdUniformsBuf,
        outputBuffers, readbackBuffers,
        targetPixels: get_target_pixels(loadedImage, canvas),
        roiMask,
        getHyperparams: () => ({
            stride: 1, mlpRatio: 1, numLoops: 1,
            embedLr: sliderToLR(parseInt(embedLrInput.value)),
            mlpLr:   sliderToLR(parseInt(mlpLrInput.value)),
            roiStrength: 0, roiFreeze: true, maxIter: 1,
        }),
        onStep({ lastWeights: w, inter1, inter2 }) {
            lastWeights = w;
            if (inter1 !== null) lastInterData = { inter1, inter2 };
            ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(
                layersCanvas, w, lastInterData.inter1, lastInterData.inter2,
                w.finalOutput, canvas.width, canvas.height, config, channelMask, null, hoverState));
        },
        onStop() { trainer?.destroy(); trainer = null; syncButtonStates(); },
    });
    trainer.start(freshWeights);
}

async function startTraining(fullReset) {
    trainer?.destroy();
    trainer = null;
    resetCanvasToBase();
    const prevOffsets = config.embOffsets;
    const prevEmbBits = config.embBits;
    config = buildConfigFromUI();
    try {
        if (!webGpuContext) webGpuContext = await initWebGPU();
        if (fullReset || !configCompatible(lastConfig, currentModelConfig()) || !model) {
            config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, noOffsetCheckbox.checked);
            const { buffers, weights: freshWeights } = createModel(webGpuContext, config);
            model = buffers;
            await createPipeline();
            createBindGroup();
            channelMask = 0xFFFFFFFF;
            lastConfig = currentModelConfig();
            updateDirtyIndicators();
            clearTrainingUI();
            trainer = makeTrainer();
            setStatus('training');
            trainer.start(freshWeights);
        } else {
            config.embOffsets = (config.embBits === prevEmbBits) ? prevOffsets
                : generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, noOffsetCheckbox.checked);
            await createPipeline();
            createBindGroup();
            channelMask = 0xFFFFFFFF;
            clearTrainingUI();
            trainer = makeTrainer();
            trainer.lastWeights = lastWeights;
            setStatus('training');
            trainer.start(null);
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
    canvas.width  = BASE_CANVAS_W;
    canvas.height = BASE_CANVAS_H;
    canvas.style.width = '';
    canvas.style.height = '';
    canvas.parentElement.style.overflow = 'hidden';
}

function buildConfigFromUI() {
    return {
        gridSize:            parseInt(gridSizeSelect.value),
        embeddingChannels:   parseInt(embeddingChannelsSelect.value),
        mlpWidth:            parseInt(mlpWidthSelect.value),
        quantization:        quantizationSelect.value,
        embBits:             parseInt(embBitsSelect.value),
        activation:          activationSelect.value,
        smoothInterpolation: smoothInterpolationCheckbox.checked,
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
    const { mlpWidth, gridSize, embeddingChannels } = config;
    fwdUniformsBuf = webGpuContext.uniformBuffer(160);
    webGpuContext.device.queue.writeBuffer(fwdUniformsBuf, 0,
        buildFwdUniforms(gridSize, embeddingChannels, mlpWidth, canvas.width, canvas.height, null));

    const pixelCount = canvas.width * canvas.height;
    const embSize    = gridSize * gridSize * embeddingChannels;
    const stride     = mlpWidth * 4; // bytes per pixel in inter-layer buffers (mlpWidth f32)

    outputBuffers.interLayer1 = webGpuContext.outputBuffer(pixelCount * stride);
    outputBuffers.interLayer2 = webGpuContext.outputBuffer(pixelCount * stride);
    outputBuffers.final       = webGpuContext.outputBuffer(pixelCount * 4 * 4);

    readbackBuffers.final         = webGpuContext.readbackBuffer(pixelCount * 4 * 4);
    readbackBuffers.embeddings    = webGpuContext.readbackBuffer(embSize * 4);
    readbackBuffers.layer1Weights = webGpuContext.readbackBuffer(mlpWidth * embeddingChannels * 4);
    readbackBuffers.layer1Biases  = webGpuContext.readbackBuffer(stride);
    readbackBuffers.layer2Weights = webGpuContext.readbackBuffer(mlpWidth * stride);
    readbackBuffers.layer3Weights = webGpuContext.readbackBuffer(4 * stride);
    readbackBuffers.interLayer1   = webGpuContext.readbackBuffer(pixelCount * stride);
    readbackBuffers.interLayer2   = webGpuContext.readbackBuffer(pixelCount * stride);

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
    lossCanvas.getContext('2d').clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    lossValueEl.textContent   = '—';
    stepCounterEl.textContent = '0';
    rateDisplayEl.textContent = '—';
    layerRangeEma = null;
    layerCols     = [];
    hoverState    = null;
    lastInterData = { inter1: null, inter2: null };
}

function makeTrainer() {
    return new Trainer({
        webGpuContext, canvas, config, model, pipeline, bindGroup, fwdUniformsBuf,
        outputBuffers, readbackBuffers,
        targetPixels: get_target_pixels(loadedImage, canvas),
        roiMask,
        getHyperparams: () => ({
            stride:      parseInt(bwdStrideInput.value) || 1,
            mlpRatio:    parseInt(mlpRatioInput.value)  || 1,
            numLoops:    parseInt(numLoopsInput.value)  || 1,
            embedLr:     sliderToLR(parseInt(embedLrInput.value)),
            mlpLr:       sliderToLR(parseInt(mlpLrInput.value)),
            roiStrength: parseFloat(roiStrengthInput.value),
            roiFreeze:   roiFreezeChk.checked,
            maxIter:     parseInt(maxIterInput.value),
        }),
        onStep({ loss, step, rate, lastWeights: w, inter1, inter2, lossHistory }) {
            lastWeights = w;
            if (inter1 !== null) lastInterData = { inter1, inter2 };
            lossValueEl.textContent   = loss < 1e-4 ? loss.toExponential(3) : loss.toFixed(6);
            stepCounterEl.textContent = step.toLocaleString();
            rateDisplayEl.textContent = rate === '—' ? rate : rate + ' it/s';
            drawOutputCanvas(canvas, w.finalOutput, config.hasAlpha);
            ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(layersCanvas, w, lastInterData.inter1, lastInterData.inter2,
                w.finalOutput, canvas.width, canvas.height, config, channelMask, layerRangeEma, hoverState));
            drawLossCurve(lossCanvas, lossHistory);
            drawSourceImage();
        },
        onStop() {
            setStatus('stopped');
            if (roiMask.isActive()) startDecayLoop();
        },
    });
}

// --- Start / Stop / Reset buttons ---
startBtn.addEventListener('click', async () => {
    if (isTraining()) {
        trainer.stop();
        setStatus('stopped');
        if (roiMask.isActive()) startDecayLoop();
        return;
    }
    if (!loadedImage) {
        alert("Please load an image first.");
        return;
    }
    await startTraining(false);
});

resetBtn.addEventListener('click', async () => {
    if (!loadedImage) return;
    await startTraining(true);
});

// --- Shake: add small noise to embeddings to escape local minima ---
const SHAKE_AMPLITUDE = 0.02;
shakeBtn.addEventListener('click', async () => {
    if (!lastWeights) return;
    const emb = lastWeights.embeddings;
    for (let i = 0; i < emb.length; i++) emb[i] += (Math.random() * 2 - 1) * SHAKE_AMPLITUDE;
    const { device } = webGpuContext;
    device.queue.writeBuffer(model.embeddings, 0, emb);
    // Reset Adam moments so accumulated momentum doesn't immediately undo the perturbation
    const zeros = new Float32Array(emb.length);
    device.queue.writeBuffer(model.adamM.embeddings, 0, zeros);
    device.queue.writeBuffer(model.adamV.embeddings, 0, zeros);
});

// --- Weight readback (for export) ---
async function readBackAllWeights() {
    const { device } = webGpuContext;
    const { gridSize, embeddingChannels: embCh, mlpWidth } = config;
    const tensors = {
        embeddings:     { buf: model.embeddings,     size: gridSize * gridSize * embCh },
        layer1_weights: { buf: model.layer1.weights, size: mlpWidth * embCh },
        layer1_biases:  { buf: model.layer1.biases,  size: mlpWidth },
        layer2_weights: { buf: model.layer2.weights, size: mlpWidth * mlpWidth },
        layer2_biases:  { buf: model.layer2.biases,  size: mlpWidth },
        layer3_weights: { buf: model.layer3.weights, size: 4 * mlpWidth },
        layer3_biases:  { buf: model.layer3.biases,  size: 4 },
    };
    const rbBufs = {};
    const ce = device.createCommandEncoder();
    for (const [k, { buf, size }] of Object.entries(tensors)) {
        rbBufs[k] = webGpuContext.readbackBuffer(size * 4);
        ce.copyBufferToBuffer(buf, 0, rbBufs[k], 0, size * 4);
    }
    device.queue.submit([ce.finish()]);
    await Promise.all(Object.values(rbBufs).map(b => b.mapAsync(GPUMapMode.READ)));
    const result = {};
    for (const [k, b] of Object.entries(rbBufs)) {
        result[k] = new Float32Array(b.getMappedRange()).slice();
        b.unmap();
        b.destroy();
    }
    return result;
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
document.getElementById('save-btn').addEventListener('click', async () => {
    if (!model) {
        alert("Train or load a model first.");
        return;
    }
    const weights = await readBackAllWeights();
    const saveConfig = {
        ...config,
        maxIter:    maxIterInput.value,
        embedLr:    embedLrInput.value,
        mlpLr:      mlpLrInput.value,
        mlpRatio:   mlpRatioInput.value,
        numLoops:   numLoopsInput.value,
        bwdStride:  bwdStrideInput.value,
        outputZoom: outputZoomInput.value,
        noOffset:   noOffsetCheckbox.checked,
    };
    const buf = saveModelSafetensors(saveConfig, weights);
    const url = URL.createObjectURL(new Blob([buf], { type: 'application/octet-stream' }));
    const a   = Object.assign(document.createElement('a'), { href: url, download: 'model.safetensors' });
    a.click();
    URL.revokeObjectURL(url);
});

// --- Snapshot / Recall ---
snapshotBtn.addEventListener('click', async () => {
    if (!model) { alert("Train or load a model first."); return; }
    if (!snapshotWeights) {
        snapshotWeights = await readBackAllWeights();
        snapshotBtn.textContent = '↺ Recall';
    } else {
        const { device } = webGpuContext;
        device.queue.writeBuffer(model.embeddings,     0, snapshotWeights.embeddings);
        device.queue.writeBuffer(model.layer1.weights, 0, snapshotWeights.layer1_weights);
        device.queue.writeBuffer(model.layer1.biases,  0, snapshotWeights.layer1_biases);
        device.queue.writeBuffer(model.layer2.weights, 0, snapshotWeights.layer2_weights);
        device.queue.writeBuffer(model.layer2.biases,  0, snapshotWeights.layer2_biases);
        device.queue.writeBuffer(model.layer3.weights, 0, snapshotWeights.layer3_weights);
        device.queue.writeBuffer(model.layer3.biases,  0, snapshotWeights.layer3_biases);
        snapshotWeights = null;
        snapshotBtn.textContent = '● Snapshot';
    }
});

// --- Zoom inference: forward pass at arbitrary resolution using temp buffers ---
async function runZoomInference(W, H) {
    const { device } = webGpuContext;
    const { gridSize, embeddingChannels: embCh, mlpWidth } = config;
    const embSize    = gridSize * gridSize * embCh;
    const pixelCount = W * H;

    const embF32 = lastWeights.embeddings;
    const range  = computeEmbRange(embF32, embCh, gridSize * gridSize);
    device.queue.writeBuffer(model.embeddings_range, 0, range);
    const packed = cpuPackEmbeddings(embF32, embCh, range, config.embBits);
    device.queue.writeBuffer(model.embeddings_q, 0, packed);

    const unifBuf = webGpuContext.uniformBuffer(160);
    device.queue.writeBuffer(unifBuf, 0, buildFwdUniforms(gridSize, embCh, mlpWidth, W, H, range));
    uploadChannelMask(channelMask, unifBuf, device);
    const outBuf  = webGpuContext.outputBuffer(pixelCount * 4 * 4);
    const stride  = mlpWidth * 4;
    const interL1 = webGpuContext.storageBuffer(pixelCount * stride);
    const interL2 = webGpuContext.storageBuffer(pixelCount * stride);
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

    const ce = device.createCommandEncoder();
    const fwdPass = ce.beginComputePass();
    fwdPass.setPipeline(pipeline);
    fwdPass.setBindGroup(0, bg);
    fwdPass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    fwdPass.end();
    ce.copyBufferToBuffer(outBuf, 0, rbBuf, 0, outBuf.size);
    device.queue.submit([ce.finish()]);

    await rbBuf.mapAsync(GPUMapMode.READ);
    canvas.width  = W;
    canvas.height = H;
    drawOutputCanvas(canvas, new Float32Array(rbBuf.getMappedRange()), config.hasAlpha);
    rbBuf.unmap();

    unifBuf.destroy(); outBuf.destroy(); interL1.destroy(); interL2.destroy(); rbBuf.destroy();
}

// --- Channel mask: click / shift+click on the Emb column in the Layers canvas ---
// Map a mouse event to canvas pixel coords, accounting for object-fit:contain letterboxing.
function layersCanvasCoords(e) {
    const rect = layersCanvas.getBoundingClientRect();
    const cw = layersCanvas.width, ch = layersCanvas.height;
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
    if (!lastWeights || !config?.mlpWidth) return;
    ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(layersCanvas, lastWeights,
        lastInterData.inter1, lastInterData.inter2,
        lastWeights.finalOutput, canvas.width, canvas.height,
        config, channelMask, layerRangeEma, hoverState));
}

layersCanvas.addEventListener('mousemove', (e) => {
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
    layersCanvas.style.cursor = (newState?.col === 'emb' && !isTraining()) ? 'pointer' : 'default';
    if (newState?.col === hoverState?.col && newState?.ch === hoverState?.ch) return;
    hoverState = newState;
    redrawLayers();
});

layersCanvas.addEventListener('mouseleave', () => {
    layersCanvas.style.cursor = 'default';
    if (hoverState === null) return;
    hoverState = null;
    redrawLayers();
});

layersCanvas.addEventListener('click', (e) => {
    if (isTraining()) return;
    const embCol = layerCols.find(c => c.name === 'emb');
    if (!embCol?.slotH) return;
    const { cx, cy } = layersCanvasCoords(e);
    if (cx < embCol.x || cx >= embCol.x + embCol.w) return;
    const embCh = config?.embeddingChannels ?? 0;
    if (!embCh) return;
    const ch = Math.min(Math.max(0, ((cy - embCol.y0) / embCol.slotH) | 0), embCh - 1);
    const allBits = embCh < 32 ? (1 << embCh) - 1 : 0xFFFFFFFF;
    if (e.shiftKey) {
        channelMask = (channelMask ^ allBits) | (1 << ch);
    } else {
        channelMask ^= (1 << ch);
    }
    if (canInfer()) runInference();
    else redrawLayers();
});

// --- Inference: pack embeddings (CPU) then run one forward pass ---
async function runInference() {
    const { device } = webGpuContext;
    const { gridSize, embeddingChannels: embCh } = config;

    const embF32 = lastWeights.embeddings;
    const range  = computeEmbRange(embF32, embCh, gridSize * gridSize);
    device.queue.writeBuffer(model.embeddings_range, 0, range);
    uploadEmbRange(range, embCh, fwdUniformsBuf, device);
    const packed = cpuPackEmbeddings(embF32, embCh, range, config.embBits);
    device.queue.writeBuffer(model.embeddings_q, 0, packed);

    uploadChannelMask(channelMask, fwdUniformsBuf, device);
    const ce = device.createCommandEncoder();
    const fwdPass = ce.beginComputePass();
    fwdPass.setPipeline(pipeline);
    fwdPass.setBindGroup(0, bindGroup);
    fwdPass.dispatchWorkgroups(Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));
    fwdPass.end();
    ce.copyBufferToBuffer(outputBuffers.final,      0, readbackBuffers.final,      0, outputBuffers.final.size);
    ce.copyBufferToBuffer(outputBuffers.interLayer1, 0, readbackBuffers.interLayer1, 0, readbackBuffers.interLayer1.size);
    ce.copyBufferToBuffer(outputBuffers.interLayer2, 0, readbackBuffers.interLayer2, 0, readbackBuffers.interLayer2.size);
    device.queue.submit([ce.finish()]);

    await Promise.all([
        readbackBuffers.final.mapAsync(GPUMapMode.READ),
        readbackBuffers.interLayer1.mapAsync(GPUMapMode.READ),
        readbackBuffers.interLayer2.mapAsync(GPUMapMode.READ),
    ]);
    const inferFinal  = new Float32Array(readbackBuffers.final.getMappedRange()).slice();
    const inferInter1 = new Float32Array(readbackBuffers.interLayer1.getMappedRange()).slice();
    const inferInter2 = new Float32Array(readbackBuffers.interLayer2.getMappedRange()).slice();
    drawOutputCanvas(canvas, inferFinal, config.hasAlpha);
    readbackBuffers.final.unmap();
    readbackBuffers.interLayer1.unmap();
    readbackBuffers.interLayer2.unmap();

    lastInterData = { inter1: inferInter1, inter2: inferInter2 };
    lastWeights.finalOutput = inferFinal;
    ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(layersCanvas, lastWeights, inferInter1, inferInter2,
        inferFinal, canvas.width, canvas.height, config, channelMask, layerRangeEma, hoverState));
}

async function loadModelFile(file) {
    trainer?.destroy();
    trainer = null;
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

    gridSizeSelect.value          = String(savedConfig.gridSize);
    embeddingChannelsSelect.value = String(savedConfig.embeddingChannels);
    mlpWidthSelect.value          = String(savedConfig.mlpWidth);

    // Restore saved UI settings
    if (uiSettings.quantization) quantizationSelect.value = uiSettings.quantization;
    if (savedConfig.embBits) embBitsSelect.value = String(savedConfig.embBits);
    if (savedConfig.activation) { activationSelect.value = savedConfig.activation; syncActivationButtons(); }
    if (uiSettings.smoothInterpolation !== undefined)
        smoothInterpolationCheckbox.checked = uiSettings.smoothInterpolation === 'true';
    if (uiSettings.noOffset !== undefined)
        noOffsetCheckbox.checked = uiSettings.noOffset === 'true';
    if (uiSettings.maxIter !== undefined) {
        maxIterInput.value = uiSettings.maxIter;
        maxIterInput.dispatchEvent(new Event('input'));
    }
    if (uiSettings.embedLr !== undefined) syncSliderDisplay(Object.assign(embedLrInput, { value: uiSettings.embedLr }), embedLrVal);
    if (uiSettings.mlpLr   !== undefined) syncSliderDisplay(Object.assign(mlpLrInput,   { value: uiSettings.mlpLr   }), mlpLrVal);
    if (uiSettings.mlpRatio  !== undefined) { mlpRatioInput.value  = uiSettings.mlpRatio;  mlpRatioVal.textContent  = uiSettings.mlpRatio; }
    if (uiSettings.numLoops  !== undefined) { numLoopsInput.value  = uiSettings.numLoops;  numLoopsVal.textContent  = uiSettings.numLoops; }
    if (uiSettings.bwdStride !== undefined) { bwdStrideInput.value = uiSettings.bwdStride; bwdStrideVal.textContent = uiSettings.bwdStride; }
    if (uiSettings.outputZoom !== undefined) { outputZoomInput.value = uiSettings.outputZoom; outputZoomVal.textContent = uiSettings.outputZoom + '×'; }

    const loadedEmbBits = savedConfig.embBits || 8;
    config = {
        gridSize:            savedConfig.gridSize,
        embeddingChannels:   savedConfig.embeddingChannels,
        mlpWidth:            savedConfig.mlpWidth,
        quantization:        quantizationSelect.value,
        embBits:             loadedEmbBits,
        activation:          savedConfig.activation || 'sin',
        smoothInterpolation: smoothInterpolationCheckbox.checked,
        width:  canvas.width,
        height: canvas.height,
        embOffsets:          savedConfig.embOffsets || generateEmbOffsets(savedConfig.embeddingChannels, loadedEmbBits, savedConfig.gridSize, noOffsetCheckbox.checked),
    };

    try {
        if (!webGpuContext) webGpuContext = await initWebGPU();
        const { buffers } = createModel(webGpuContext, config);
        model = buffers;

        const { device } = webGpuContext;
        device.queue.writeBuffer(model.embeddings,     0, tensors.embeddings);
        device.queue.writeBuffer(model.layer1.weights, 0, tensors.layer1_weights);
        device.queue.writeBuffer(model.layer1.biases,  0, tensors.layer1_biases);
        device.queue.writeBuffer(model.layer2.weights, 0, tensors.layer2_weights);
        device.queue.writeBuffer(model.layer2.biases,  0, tensors.layer2_biases);
        device.queue.writeBuffer(model.layer3.weights, 0, tensors.layer3_weights);
        device.queue.writeBuffer(model.layer3.biases,  0, tensors.layer3_biases);

        await createPipeline();
        createBindGroup();
        channelMask = 0xFFFFFFFF;
        lastConfig  = currentModelConfig();
        updateDirtyIndicators();

        lastWeights = {
            embeddings:    tensors.embeddings,
            layer1Weights: tensors.layer1_weights,
            layer2Weights: tensors.layer2_weights,
            layer3Weights: tensors.layer3_weights,
        };

        await runInference();
        setStatus('stopped');
        updateSizeDisplay();
        updateStartLabel();
    } catch (err) {
        console.error('Load model failed:', err);
        alert('Load model failed. Check console for errors.');
    }
}

loadBtn.addEventListener('click', () => modelFileInput.click());
modelFileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    modelFileInput.value = '';
    await loadModelFile(file);
});

// Page-level drop for .safetensors (image drops still handled by source panel)
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', async (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith('.safetensors')) await loadModelFile(file);
});

function updateEmbBitsOptions() {
    const embCh = parseInt(embeddingChannelsSelect.value);
    const opt4 = embBitsSelect.querySelector('option[value="4"]');
    opt4.disabled = embCh < 8;
    if (opt4.disabled && embBitsSelect.value === '4') embBitsSelect.value = '8';
}

// Update start button label and size when model config dropdowns change
[gridSizeSelect, embeddingChannelsSelect, mlpWidthSelect, quantizationSelect, embBitsSelect, activationSelect].forEach(sel => {
    sel.addEventListener('change', () => {
        updateEmbBitsOptions(); updateStartLabel(); updateSizeDisplay();
        if (!isTraining()) {
            updateDirtyIndicators();
            const cur = currentModelConfig();
            if (lastWeights && configCompatible(lastConfig, cur)) {
                ({ ema: layerRangeEma, cols: layerCols } = drawFlowDiagram(layersCanvas, lastWeights, lastInterData.inter1, lastInterData.inter2,
                    lastWeights.finalOutput, canvas.width, canvas.height, lastConfig, channelMask, layerRangeEma, hoverState));
            } else if (webGpuContext && loadedImage) {
                previewGeometry();
            } else {
                drawPlaceholder(layersCanvas);
                layerRangeEma = null;
                lastInterData = { inter1: null, inter2: null };
            }
        }
    });
});
updateEmbBitsOptions();

// --- URL parameter parsing ---
// Supported: ?grid=64&EMB=16&MLP=8&iters=4000&8qat&4b&numLoops=3
// Flags: 8qat (MLP 8-bit QAT), 4b (4-bit embeddings), 8b (8-bit embeddings)
function applyUrlParams() {
    const params = new URLSearchParams(window.location.search);
    let hasParams = false;

    const setSelect = (el, val, allowed) => {
        const n = parseInt(val);
        if (val !== null && allowed.includes(n)) { el.value = String(n); hasParams = true; }
    };

    setSelect(gridSizeSelect,          params.get('grid'), [16, 32, 64]);
    setSelect(embeddingChannelsSelect, params.get('EMB'),  [4, 8, 16]);
    setSelect(mlpWidthSelect,          params.get('MLP'),  [4, 8, 16, 32, 64]);

    if (params.has('8qat')) { quantizationSelect.value = 'qat8'; hasParams = true; }
    if (params.has('none')) { quantizationSelect.value = 'none'; hasParams = true; }
    if (params.has('4b'))   { embBitsSelect.value = '4'; hasParams = true; }
    if (params.has('8b'))   { embBitsSelect.value = '8'; hasParams = true; }

    const iters = params.get('iters');
    if (iters !== null) {
        const v = parseInt(iters);
        if (!isNaN(v) && v >= 0) {
            maxIterInput.value = v;
            maxIterInput.dispatchEvent(new Event('input'));
            hasParams = true;
        }
    }

    const setLR = (el, valEl, param) => {
        const v = parseInt(params.get(param));
        if (!isNaN(v) && v >= 0 && v <= 100) {
            syncSliderDisplay(Object.assign(el, { value: v }), valEl);
            hasParams = true;
        }
    };
    setLR(embedLrInput, embedLrVal, 'embedLr');
    setLR(mlpLrInput,   mlpLrVal,   'mlpLr');

    const numLoopsParam = parseInt(params.get('numLoops'));
    if (!isNaN(numLoopsParam) && numLoopsParam >= 1 && numLoopsParam <= 8) {
        numLoopsInput.value = numLoopsParam;
        numLoopsVal.textContent = numLoopsParam;
        hasParams = true;
    }

    if (params.has('smooth')) { smoothInterpolationCheckbox.checked = true;  hasParams = true; }
    if (params.has('nosmooth')) { smoothInterpolationCheckbox.checked = false; hasParams = true; }

    if (hasParams) { updateEmbBitsOptions(); updateStartLabel(); updateSizeDisplay(); }
    return hasParams;
}

initTooltips();

// --- Startup: load default image then default model (or auto-start from URL params) ---
const urlHasParams = applyUrlParams();
const startupImg = new Image();
startupImg.onload = async () => {
    loadImageOntoCanvas(startupImg);
    if (urlHasParams) {
        startBtn.click();
    } else {
        // Always init model + run one inference (random weights) for immediate visual feedback.
        // Then overwrite with saved weights if available.
        try {
            if (!webGpuContext) webGpuContext = await initWebGPU();
            config = buildConfigFromUI();
            config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, noOffsetCheckbox.checked);
            const { buffers, weights: initialWeights } = createModel(webGpuContext, config);
            model = buffers;
            await createPipeline();
            createBindGroup();
            channelMask = 0xFFFFFFFF;
            lastConfig = currentModelConfig();
            updateDirtyIndicators();
            lastWeights = {
                embeddings:    new Float32Array(initialWeights.embeddings),
                layer1Weights: new Float32Array(initialWeights.layer1_weights),
                layer2Weights: new Float32Array(initialWeights.layer2_weights),
                layer3Weights: new Float32Array(initialWeights.layer3_weights),
            };
            await runInference();
        } catch (err) { console.error('Initial inference failed:', err); }
        try {
            const resp = await fetch('mona_lisa.safetensors');
            if (resp.ok) await loadModelFile(await resp.blob());
        } catch (_) {}
    }
};
startupImg.src = 'Mona_Lisa.webp';
