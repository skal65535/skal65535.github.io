import { initWebGPU } from './webgpu_manager.js';
import { createModel, ModelTensors } from './model.js';
import { buildShader } from './shader_builder.js';
import { buildBackwardShaders, FP_SCALE } from './backward_builder.js';
import { get_target_pixels, calculate_loss } from './loss.js';
import { export_to_glsl } from './shader_exporter.js';
import { saveModelSafetensors, loadModelSafetensors } from './model_io.js';
import { ROIMask } from './roi_mask.js';
import { drawOutputCanvas, drawEmbeddings, drawLayers, drawLossCurve } from './viz.js';
import { computeEmbRange, normalizeEmbAndAdjustL1, buildFwdUniforms, uploadEmbRange, cpuPackEmbeddings, generateEmbOffsets } from './emb_utils.js';
import { initTooltips } from './tooltips.js';

// --- DOM references ---
const sourcePanel  = document.getElementById('source-panel');
const sourceCanvas = document.getElementById('source-canvas');
const embedCanvas  = document.getElementById('embed-canvas');
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
const smoothInterpolationCheckbox = document.getElementById('smooth-interpolation');
const noOffsetCheckbox            = document.getElementById('no-offset');
const maxIterInput  = document.getElementById('max-iter');
const embedLrInput  = document.getElementById('embed-lr');
const mlpLrInput    = document.getElementById('mlp-lr');
const maxIterVal    = document.getElementById('max-iter-val');
const embedLrVal    = document.getElementById('embed-lr-val');
const mlpLrVal      = document.getElementById('mlp-lr-val');
const mlpRatioInput = document.getElementById('mlp-ratio');
const mlpRatioVal   = document.getElementById('mlp-ratio-val');
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
const iterLabelEl   = document.getElementById('iter-label');
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
let outputBuffers  = {};
let readbackBuffers = {};
let targetPixels  = null;
let trainingActive  = false;
let animationFrameId = null;
let lastWeights     = null;
let snapshotWeights = null;
let layerRangeEma = null; // EMA of [min, max] per layer for visualization
let stepCount     = 0;
let lossHistory   = [];
let lastStepTime  = 0;
let lastConfig    = null; // { gridSize, embeddingChannels, mlpWidth } from last full init
let adamT         = 0;   // GPU Adam timestep (incremented each step)

// ROI mask state
const roiMask  = new ROIMask(1, 1);
let decayRafId = null;

// Backward pipelines + bind groups (rebuilt on model change)
let fwdUniformsBuf   = null; // 160-byte forward uniform buffer (includes emb_range)
let bwdPipelines     = null;
let bwdBindGroups    = null;
let bwdBuffers       = {}; // intermediate per-pixel grad buffers (GPU only)
let targetGpuBuf     = null; // target image uploaded once per training run
let bwdUniformsBuf   = null; // BwdUniforms (width, height, stride, sampled_count)
let adamUniformsBufs = {}; // AdamUniforms uniform buffer per parameter tensor

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

// --- LR slider helpers (log scale: slider 0–100 → 1e-4 … 1e-1) ---
const sliderToLR = v => Math.pow(10, -4 + (v / 100) * 3);
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
bwdStrideInput.addEventListener('input', () => { bwdStrideVal.textContent = bwdStrideInput.value; });
outputZoomInput.addEventListener('input', () => {
    const scale = parseFloat(outputZoomInput.value);
    outputZoomVal.textContent = scale + '×';
    if (!trainingActive && model && config.mlpWidth && lastWeights?.embeddings) {
        const W = Math.round(BASE_CANVAS_W * scale);
        const H = Math.round(BASE_CANVAS_H * scale);
        outputResEl.textContent = `${W}×${H}`;
        if (scale <= 1.0) {
            canvas.style.width = '';
            canvas.style.height = '';
            canvas.style.maxWidth = '';
            canvas.style.maxHeight = '';
            canvas.parentElement.style.overflow = 'hidden';
        } else {
            canvas.style.maxWidth  = 'none';
            canvas.style.maxHeight = 'none';
            canvas.style.width  = W + 'px';
            canvas.style.height = H + 'px';
            canvas.parentElement.style.overflow = 'auto';
        }
        runZoomInference(W, H);
    }
});

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

function updateStartLabel() {
    if (configCompatible(lastConfig, currentModelConfig()) && model) {
        startBtn.textContent = '▶ Continue';
    } else {
        startBtn.textContent = '▶ Start';
    }
    resetBtn.disabled = !model || trainingActive;
    shakeBtn.disabled = !model;
}

// --- Status indicator ---
function setStatus(s) {
    const labels = { idle: 'IDLE', training: 'TRAINING', stopped: 'STOPPED' };
    statusTextEl.textContent = labels[s] || s;
    statusDotEl.className    = 'status-dot ' + s;
    if (s === 'training') {
        startBtn.textContent = '■ Stop';
        startBtn.classList.add('stopping');
        resetBtn.disabled = true;
        outputZoomInput.disabled = true;
    } else {
        startBtn.classList.remove('stopping');
        updateStartLabel();
        outputZoomInput.disabled = false;
    }
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
    if (!roiFreezeChk.checked && roiMask.isActive() && !trainingActive) startDecayLoop();
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
    let isPainting = false;
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
        const [x, y] = sourceCanvasCoords(e);
        roiMask.paint(x, y, parseInt(roiBrushInput.value));
        drawSourceImage();
        if (!trainingActive) startDecayLoop();
    });
    sourceCanvas.addEventListener('mousemove', (e) => {
        if (!isPainting) return;
        const [x, y] = sourceCanvasCoords(e);
        roiMask.paint(x, y, parseInt(roiBrushInput.value));
        drawSourceImage();
    });
    window.addEventListener('mouseup',    () => { isPainting = false; });
    sourceCanvas.addEventListener('mouseleave', () => { isPainting = false; });
}

// --- File / drop zone ---
sourcePanel.addEventListener('click', (e) => {
    if (e.target !== fileInput && !loadedImage) fileInput.click();
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});
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
    const bytes = BASE_CANVAS_W * BASE_CANVAS_H * 3;
    inputSizeEl.textContent = bytes >= 1024 ? (bytes / 1024).toFixed(1) + ' KB' : bytes + ' B';
}

async function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => loadImageOntoCanvas(img);
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

    outputBuffers.interLayer1 = webGpuContext.storageBuffer(pixelCount * mlpWidth * 4);
    outputBuffers.interLayer2 = webGpuContext.storageBuffer(pixelCount * mlpWidth * 4);
    outputBuffers.final       = webGpuContext.outputBuffer(pixelCount * 4 * 4);

    readbackBuffers.final         = webGpuContext.readbackBuffer(pixelCount * 4 * 4);
    readbackBuffers.embeddings    = webGpuContext.readbackBuffer(embSize * 4);
    readbackBuffers.layer1Weights = webGpuContext.readbackBuffer(mlpWidth * embeddingChannels * 4);
    readbackBuffers.layer1Biases  = webGpuContext.readbackBuffer(mlpWidth * 4);
    readbackBuffers.layer2Weights = webGpuContext.readbackBuffer(mlpWidth * mlpWidth * 4);
    readbackBuffers.layer3Weights = webGpuContext.readbackBuffer(4 * mlpWidth * 4);

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

function createBackwardPipelines() {
    const { device } = webGpuContext;
    const { gridSize, embeddingChannels: embCh, mlpWidth } = config;
    const pixelCount = canvas.width * canvas.height;
    const shaders = buildBackwardShaders(config);

    const makePipeline = code => device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });

    bwdPipelines = {
        gradOutput:     makePipeline(shaders.gradOutput),
        gradL3:         makePipeline(shaders.gradL3),
        gradL2:         makePipeline(shaders.gradL2),
        gradL1:         makePipeline(shaders.gradL1),
        adamStep:       makePipeline(shaders.adamStep),
        packEmbeddings: makePipeline(shaders.packEmbeddings),
    };

    // Upload target image once
    if (targetGpuBuf) targetGpuBuf.destroy();
    targetGpuBuf = webGpuContext.zeroBuffer(targetPixels.length);
    device.queue.writeBuffer(targetGpuBuf, 0, targetPixels);

    // BwdUniforms (width, height, stride, sampled_count, roi_strength)
    bwdUniformsBuf = webGpuContext.uniformBuffer(5 * 4);

    // ROI mask (one f32 per pixel, uploaded from CPU each step)
    bwdBuffers.roiMask = webGpuContext.zeroBuffer(pixelCount);

    // Per-pixel intermediate gradient buffers (no races, plain f32, GPU-only)
    bwdBuffers.gradFinal        = webGpuContext.storageBuffer(pixelCount * 4 * 4);
    bwdBuffers.gradInter2Preact = webGpuContext.storageBuffer(pixelCount * mlpWidth * 4);
    bwdBuffers.gradInter1Preact = webGpuContext.storageBuffer(pixelCount * mlpWidth * 4);

    // Adam uniform buffers (one per parameter tensor, updated each step)
    const makeAdamBuf = () => webGpuContext.uniformBuffer(10 * 4);
    adamUniformsBufs = ModelTensors.create(() => makeAdamBuf());

    // Pre-allocate scratch buffers reused every step (avoids per-frame heap allocation)
    bwdBuffers.adamAB = new ArrayBuffer(10 * 4);
    bwdBuffers.adamF  = new Float32Array(bwdBuffers.adamAB);
    bwdBuffers.adamU  = new Uint32Array(bwdBuffers.adamAB);
    bwdBuffers.uniAB  = new ArrayBuffer(5 * 4);
    bwdBuffers.uniU32 = new Uint32Array(bwdBuffers.uniAB);
    bwdBuffers.uniF32 = new Float32Array(bwdBuffers.uniAB);
    bwdBuffers.gradZero = {};
    for (const [k, buf] of Object.entries(model.gradAtomic)) {
        bwdBuffers.gradZero[k] = new Int32Array(buf.size / 4); // zero-filled, reused
    }

    // Helper: bind group from flat buffer list
    const makeBG = (pipeline, bufs) => device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    const m = model;
    bwdBindGroups = {
        gradOutput: makeBG(bwdPipelines.gradOutput, [
            bwdUniformsBuf, outputBuffers.final, targetGpuBuf, bwdBuffers.gradFinal, bwdBuffers.roiMask,
        ]),
        gradL3: makeBG(bwdPipelines.gradL3, [
            bwdUniformsBuf, bwdBuffers.gradFinal, outputBuffers.interLayer2, m.layer3.weights,
            bwdBuffers.gradInter2Preact, m.gradAtomic.layer3_weights, m.gradAtomic.layer3_biases,
        ]),
        gradL2: makeBG(bwdPipelines.gradL2, [
            bwdUniformsBuf, bwdBuffers.gradInter2Preact, outputBuffers.interLayer2,
            outputBuffers.interLayer1, m.layer2.weights,
            bwdBuffers.gradInter1Preact, m.gradAtomic.layer2_weights, m.gradAtomic.layer2_biases,
        ]),
        gradL1: makeBG(bwdPipelines.gradL1, [
            bwdUniformsBuf, bwdBuffers.gradInter1Preact, outputBuffers.interLayer1,
            m.layer1.weights, m.embeddings,
            m.gradAtomic.layer1_weights, m.gradAtomic.layer1_biases, m.gradAtomic.embeddings,
        ]),
        adam: {
            embeddings:     makeBG(bwdPipelines.adamStep, [adamUniformsBufs.embeddings,     m.gradAtomic.embeddings,     m.embeddings,     m.adamM.embeddings,     m.adamV.embeddings]),
            layer1_weights: makeBG(bwdPipelines.adamStep, [adamUniformsBufs.layer1_weights,  m.gradAtomic.layer1_weights, m.layer1.weights, m.adamM.layer1_weights, m.adamV.layer1_weights]),
            layer1_biases:  makeBG(bwdPipelines.adamStep, [adamUniformsBufs.layer1_biases,   m.gradAtomic.layer1_biases,  m.layer1.biases,  m.adamM.layer1_biases,  m.adamV.layer1_biases]),
            layer2_weights: makeBG(bwdPipelines.adamStep, [adamUniformsBufs.layer2_weights,  m.gradAtomic.layer2_weights, m.layer2.weights, m.adamM.layer2_weights, m.adamV.layer2_weights]),
            layer2_biases:  makeBG(bwdPipelines.adamStep, [adamUniformsBufs.layer2_biases,   m.gradAtomic.layer2_biases,  m.layer2.biases,  m.adamM.layer2_biases,  m.adamV.layer2_biases]),
            layer3_weights: makeBG(bwdPipelines.adamStep, [adamUniformsBufs.layer3_weights,  m.gradAtomic.layer3_weights, m.layer3.weights, m.adamM.layer3_weights, m.adamV.layer3_weights]),
            layer3_biases:  makeBG(bwdPipelines.adamStep, [adamUniformsBufs.layer3_biases,   m.gradAtomic.layer3_biases,  m.layer3.biases,  m.adamM.layer3_biases,  m.adamV.layer3_biases]),
        },
        packEmbeddings: makeBG(bwdPipelines.packEmbeddings, [m.embeddings, m.embeddings_q, m.embeddings_range]),
    };

    // Pack initial f32 embeddings → u32 before first forward pass
    {
        const { gridSize, embeddingChannels: embCh } = config;
        const ce = device.createCommandEncoder();
        const p  = ce.beginComputePass();
        p.setPipeline(bwdPipelines.packEmbeddings);
        p.setBindGroup(0, bwdBindGroups.packEmbeddings);
        p.dispatchWorkgroups(Math.ceil(gridSize * gridSize * embCh / 4 / 64));
        p.end();
        device.queue.submit([ce.finish()]);
    }
}

// --- Training loop ---
async function train() {
    const { device } = webGpuContext;
    const { gridSize, embeddingChannels: embCh, mlpWidth, embBits } = config;
    const pixelCount = canvas.width * canvas.height;
    const embSize    = gridSize * gridSize * embCh;

    const stride       = parseInt(bwdStrideInput.value) || 1;
    const sampledCount = Math.ceil(pixelCount / stride);
    adamT++;

    const embedLR   = sliderToLR(parseInt(embedLrInput.value));
    const mlpRatio  = parseInt(mlpRatioInput.value) || 1;
    const activeMLP = (adamT % mlpRatio === 0);
    const mlpLR     = activeMLP ? sliderToLR(parseInt(mlpLrInput.value)) : 0;
    const tensorCfg = {
        embeddings:     { lr: embedLR, size: embSize,             l2: 0.002 / embSize, clamp: 1 },
        layer1_weights: { lr: mlpLR,   size: mlpWidth * embCh,    l2: 0, clamp: 0 },
        layer1_biases:  { lr: mlpLR,   size: mlpWidth,             l2: 0, clamp: 0 },
        layer2_weights: { lr: mlpLR,   size: mlpWidth * mlpWidth,  l2: 0, clamp: 0 },
        layer2_biases:  { lr: mlpLR,   size: mlpWidth,             l2: 0, clamp: 0 },
        layer3_weights: { lr: mlpLR,   size: 4 * mlpWidth,         l2: 0, clamp: 0 },
        layer3_biases:  { lr: mlpLR,   size: 4,                    l2: 0, clamp: 0 },
    };

    if (!roiFreezeChk.checked) roiMask.decay(performance.now());
    device.queue.writeBuffer(bwdBuffers.roiMask, 0, roiMask.weights);

    for (const [k, buf] of Object.entries(model.gradAtomic)) {
        device.queue.writeBuffer(buf, 0, bwdBuffers.gradZero[k]);
    }
    const { adamAB, adamF, adamU, uniAB, uniU32, uniF32 } = bwdBuffers;
    uniU32[0] = canvas.width; uniU32[1] = canvas.height;
    uniU32[2] = stride;       uniU32[3] = sampledCount;
    uniF32[4] = parseFloat(roiStrengthInput.value);
    device.queue.writeBuffer(bwdUniformsBuf, 0, uniAB);
    adamF[1] = 0.9; adamF[2] = 0.999; adamF[3] = 1e-8; adamF[6] = FP_SCALE;
    adamU[4] = adamT; adamU[9] = sampledCount;
    for (const [k, tc] of Object.entries(tensorCfg)) {
        adamF[0] = tc.lr; adamU[5] = tc.size; adamF[7] = tc.l2; adamU[8] = tc.clamp;
        device.queue.writeBuffer(adamUniformsBufs[k], 0, adamAB);
    }

    const ce = device.createCommandEncoder();

    const fwdPass = ce.beginComputePass();
    fwdPass.setPipeline(pipeline);
    fwdPass.setBindGroup(0, bindGroup);
    fwdPass.dispatchWorkgroups(Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));
    fwdPass.end();

    // separate compute pass per shader = implicit memory barrier between them
    const bwdDispatch = Math.ceil(pixelCount / 64);
    const runPass = (pl, bg, dx, dy = 1) => {
        const p = ce.beginComputePass();
        p.setPipeline(pl); p.setBindGroup(0, bg);
        p.dispatchWorkgroups(dx, dy); p.end();
    };
    runPass(bwdPipelines.gradOutput, bwdBindGroups.gradOutput, bwdDispatch);
    runPass(bwdPipelines.gradL3,     bwdBindGroups.gradL3,     bwdDispatch);
    runPass(bwdPipelines.gradL2,     bwdBindGroups.gradL2,     bwdDispatch);
    runPass(bwdPipelines.gradL1,     bwdBindGroups.gradL1,     Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));

    const adamPass = ce.beginComputePass();
    for (const k of ModelTensors.KEYS) {
        adamPass.setPipeline(bwdPipelines.adamStep);
        adamPass.setBindGroup(0, bwdBindGroups.adam[k]);
        adamPass.dispatchWorkgroups(Math.ceil(tensorCfg[k].size / 64));
    }
    adamPass.end();

    ce.copyBufferToBuffer(outputBuffers.final,    0, readbackBuffers.final,         0, outputBuffers.final.size);
    ce.copyBufferToBuffer(model.embeddings,       0, readbackBuffers.embeddings,    0, model.embeddings.size);
    ce.copyBufferToBuffer(model.layer1.weights,   0, readbackBuffers.layer1Weights, 0, model.layer1.weights.size);
    ce.copyBufferToBuffer(model.layer1.biases,    0, readbackBuffers.layer1Biases,  0, model.layer1.biases.size);
    ce.copyBufferToBuffer(model.layer2.weights,   0, readbackBuffers.layer2Weights, 0, model.layer2.weights.size);
    ce.copyBufferToBuffer(model.layer3.weights,   0, readbackBuffers.layer3Weights, 0, model.layer3.weights.size);

    device.queue.submit([ce.finish()]);

    await Promise.all([
        readbackBuffers.final.mapAsync(GPUMapMode.READ),
        readbackBuffers.embeddings.mapAsync(GPUMapMode.READ),
        readbackBuffers.layer1Weights.mapAsync(GPUMapMode.READ),
        readbackBuffers.layer1Biases.mapAsync(GPUMapMode.READ),
        readbackBuffers.layer2Weights.mapAsync(GPUMapMode.READ),
        readbackBuffers.layer3Weights.mapAsync(GPUMapMode.READ),
    ]);

    const finalData = new Float32Array(readbackBuffers.final.getMappedRange());
    const embData   = new Float32Array(readbackBuffers.embeddings.getMappedRange()).slice();
    const l1wData   = new Float32Array(readbackBuffers.layer1Weights.getMappedRange()).slice();
    const l1bData   = new Float32Array(readbackBuffers.layer1Biases.getMappedRange()).slice();
    const l2wData   = new Float32Array(readbackBuffers.layer2Weights.getMappedRange()).slice();
    const l3wData   = new Float32Array(readbackBuffers.layer3Weights.getMappedRange()).slice();

    drawOutputCanvas(canvas, finalData);
    const loss = calculate_loss(finalData, targetPixels);

    readbackBuffers.final.unmap();
    readbackBuffers.embeddings.unmap();
    readbackBuffers.layer1Weights.unmap();
    readbackBuffers.layer1Biases.unmap();
    readbackBuffers.layer2Weights.unmap();
    readbackBuffers.layer3Weights.unmap();

    // Normalize embeddings to [-1,1] per channel; absorb scale+center into L1
    normalizeEmbAndAdjustL1(embData, l1wData, l1bData, embCh, mlpWidth);
    device.queue.writeBuffer(model.embeddings,    0, embData);
    device.queue.writeBuffer(model.layer1.weights, 0, l1wData);
    device.queue.writeBuffer(model.layer1.biases,  0, l1bData);

    // CPU pack with fixed [-1,1] range → upload to embeddings_q for next forward pass
    const identityRange = new Float32Array(embCh * 2);
    for (let c = 0; c < embCh; c++) { identityRange[c * 2] = -1; identityRange[c * 2 + 1] = 1; }
    device.queue.writeBuffer(model.embeddings_q, 0, cpuPackEmbeddings(embData, embCh, identityRange, embBits));
    // Ensure fwdUniformsBuf emb_range matches identity packing (guards against stale range from restart)
    uploadEmbRange(null, embCh, fwdUniformsBuf, device);

    lastWeights = {
        embeddings:    embData,
        layer1Weights: l1wData,
        layer2Weights: l2wData,
        layer3Weights: l3wData,
    };

    stepCount++;
    lossHistory.push(loss);

    const now = performance.now();
    const rate = lastStepTime > 0 ? (1000 / (now - lastStepTime)).toFixed(1) : '—';
    lastStepTime = now;

    lossValueEl.textContent   = loss < 1e-4 ? loss.toExponential(3) : loss.toFixed(6);
    stepCounterEl.textContent = stepCount.toLocaleString();
    rateDisplayEl.textContent = rate === '—' ? rate : rate + ' it/s';
    iterLabelEl.textContent   = `step ${stepCount}`;

    drawEmbeddings(embedCanvas, lastWeights, config);
    layerRangeEma = drawLayers(layersCanvas, lastWeights, config, layerRangeEma);
    drawLossCurve(lossCanvas, lossHistory);
    drawSourceImage();

    const maxIter = parseInt(maxIterInput.value);
    if (maxIter > 0 && stepCount >= maxIter) {
        trainingActive = false;
        setStatus('stopped');
        if (roiMask.isActive()) startDecayLoop();
        return;
    }

    if (trainingActive) {
        animationFrameId = requestAnimationFrame(train);
    }
}

function run(initialWeights) {
    stopDecayLoop();
    targetPixels  = get_target_pixels(loadedImage, canvas);
    stepCount     = 0;
    lossHistory   = [];
    lastStepTime  = 0;
    layerRangeEma = null;

    // Clear visualizations
    embedCanvas.getContext('2d').clearRect(0, 0, embedCanvas.width, embedCanvas.height);
    lossCanvas.getContext('2d').clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    lossValueEl.textContent   = '—';
    stepCounterEl.textContent = '0';
    rateDisplayEl.textContent = '—';
    iterLabelEl.textContent   = '—';

    {
        const { gridSize, embeddingChannels: embCh } = config;
        const src = initialWeights !== null ? initialWeights.embeddings : lastWeights?.embeddings;
        if (initialWeights !== null) {
            adamT = 0;
            lastWeights = { embeddings: new Float32Array(initialWeights.embeddings) };
        }
        if (src) {
            const range = computeEmbRange(src, embCh, gridSize * gridSize);
            webGpuContext.device.queue.writeBuffer(model.embeddings_range, 0, range);
            uploadEmbRange(range, embCh, fwdUniformsBuf, webGpuContext.device);
        }
    }

    createBackwardPipelines();

    trainingActive   = true;
    animationFrameId = requestAnimationFrame(train);
}

// --- Start / Stop / Reset buttons ---
startBtn.addEventListener('click', async () => {
    if (trainingActive) {
        trainingActive = false;
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
        setStatus('stopped');
        if (roiMask.isActive()) startDecayLoop();
        return;
    }
    if (!loadedImage) {
        alert("Please load an image first.");
        return;
    }
    resetCanvasToBase();
    const prevOffsets = config.embOffsets;
    const prevEmbBits = config.embBits;
    config = buildConfigFromUI();
    try {
        if (!webGpuContext) webGpuContext = await initWebGPU();
        const isRestart = configCompatible(lastConfig, currentModelConfig()) && model;
        if (isRestart) {
            // Preserve offsets only when embBits unchanged (same buffer layout)
            config.embOffsets = (config.embBits === prevEmbBits) ? prevOffsets
                : generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, noOffsetCheckbox.checked);
            // Rebuild pipeline (quantization/smooth may have changed), reuse model buffers + Adam state
            await createPipeline();
            createBindGroup();
            setStatus('training');
            run(null);
        } else {
            config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, noOffsetCheckbox.checked);
            const { buffers, weights: initialWeights } = createModel(webGpuContext, config);
            model = buffers;
            await createPipeline();
            createBindGroup();
            lastConfig = currentModelConfig();
            setStatus('training');
            run(initialWeights);
        }
    } catch (err) {
        console.error("Training setup failed:", err);
        alert("Training setup failed. Check the console for errors.");
    }
});

resetBtn.addEventListener('click', async () => {
    if (trainingActive || !loadedImage) return;
    resetCanvasToBase();
    config = buildConfigFromUI();
    config.embOffsets = generateEmbOffsets(config.embeddingChannels, config.embBits, config.gridSize, noOffsetCheckbox.checked);
    try {
        if (!webGpuContext) webGpuContext = await initWebGPU();
        const { buffers, weights: initialWeights } = createModel(webGpuContext, config);
        model = buffers;
        await createPipeline();
        createBindGroup();
        lastConfig = currentModelConfig();
        setStatus('training');
        run(initialWeights);
    } catch (err) {
        console.error("Reset failed:", err);
        alert("Reset failed. Check the console for errors.");
    }
});

// --- Shake: add small noise to embeddings to escape local minima ---
const SHAKE_AMPLITUDE = 0.02;
shakeBtn.addEventListener('click', async () => {
    if (!model || !lastWeights?.embeddings) return;
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
    if (!model || !config.mlpWidth) {
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
    if (!model || !config.mlpWidth) {
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
    if (!model || !config.mlpWidth) { alert("Train or load a model first."); return; }
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
    const outBuf  = webGpuContext.outputBuffer(pixelCount * 4 * 4);
    const interL1 = webGpuContext.storageBuffer(pixelCount * mlpWidth * 4);
    const interL2 = webGpuContext.storageBuffer(pixelCount * mlpWidth * 4);
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
    drawOutputCanvas(canvas, new Float32Array(rbBuf.getMappedRange()));
    rbBuf.unmap();

    unifBuf.destroy(); outBuf.destroy(); interL1.destroy(); interL2.destroy(); rbBuf.destroy();
}

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

    const ce = device.createCommandEncoder();
    const fwdPass = ce.beginComputePass();
    fwdPass.setPipeline(pipeline);
    fwdPass.setBindGroup(0, bindGroup);
    fwdPass.dispatchWorkgroups(Math.ceil(canvas.width / 8), Math.ceil(canvas.height / 8));
    fwdPass.end();
    ce.copyBufferToBuffer(outputBuffers.final, 0, readbackBuffers.final, 0, outputBuffers.final.size);
    device.queue.submit([ce.finish()]);

    await readbackBuffers.final.mapAsync(GPUMapMode.READ);
    drawOutputCanvas(canvas, new Float32Array(readbackBuffers.final.getMappedRange()));
    readbackBuffers.final.unmap();

    drawEmbeddings(embedCanvas, lastWeights, config);
    layerRangeEma = drawLayers(layersCanvas, lastWeights, config, layerRangeEma);
    iterLabelEl.textContent = 'loaded';
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
    if (![16, 32, 64].includes(savedConfig.gridSize) ||
        ![4, 8, 16].includes(savedConfig.embeddingChannels) ||
        ![4, 8, 16, 32, 64].includes(savedConfig.mlpWidth)) {
        alert('Unsupported model config in file.');
        return;
    }

    gridSizeSelect.value          = String(savedConfig.gridSize);
    embeddingChannelsSelect.value = String(savedConfig.embeddingChannels);
    mlpWidthSelect.value          = String(savedConfig.mlpWidth);

    // Restore saved UI settings
    if (uiSettings.quantization) quantizationSelect.value = uiSettings.quantization;
    if (savedConfig.embBits) embBitsSelect.value = String(savedConfig.embBits);
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
    if (uiSettings.bwdStride !== undefined) { bwdStrideInput.value = uiSettings.bwdStride; bwdStrideVal.textContent = uiSettings.bwdStride; }
    if (uiSettings.outputZoom !== undefined) { outputZoomInput.value = uiSettings.outputZoom; outputZoomVal.textContent = uiSettings.outputZoom + '×'; }

    const loadedEmbBits = savedConfig.embBits || 8;
    config = {
        gridSize:            savedConfig.gridSize,
        embeddingChannels:   savedConfig.embeddingChannels,
        mlpWidth:            savedConfig.mlpWidth,
        quantization:        quantizationSelect.value,
        embBits:             loadedEmbBits,
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
        lastConfig  = currentModelConfig();
        adamT       = 0;
        stepCount   = 0;
        lossHistory = [];

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
[gridSizeSelect, embeddingChannelsSelect, mlpWidthSelect, quantizationSelect, embBitsSelect].forEach(sel => {
    sel.addEventListener('change', () => { updateEmbBitsOptions(); updateStartLabel(); updateSizeDisplay(); });
});
updateEmbBitsOptions();

// --- URL parameter parsing ---
// Supported: ?grid=64&EMB=16&MLP=8&iters=4000&8qat&4b
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
        try {
            const resp = await fetch('mona_lisa.safetensors');
            if (resp.ok) await loadModelFile(await resp.blob());
        } catch (_) {}
    }
};
startupImg.src = 'Mona_Lisa.jpg';
