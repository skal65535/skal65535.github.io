// ui.js
// Centralized UI management, DOM access, and tooltips.

export const GRID_SIZES = [256, 400, 625, 1024, 1600, 2500, 4000, 6400, 8100, 10000];

const elements = {
    sourcePanel:  document.getElementById('source-panel'),
    outputPanel:  document.getElementById('output-panel'),
    canvasRow:    document.querySelector('.canvas-row'),
    sourceCanvas: document.getElementById('source-canvas'),
    layersCanvas: document.getElementById('layers-canvas'),
    sweepCanvas:       document.getElementById('sweep-canvas'),
    sourceSweepCanvas: document.getElementById('source-sweep-canvas'),
    canvas:       document.getElementById('canvas'),
    lossCanvas:   document.getElementById('loss-canvas'),
    fileInput:    document.getElementById('file-input'),
    dropOverlay:  document.getElementById('drop-overlay'),

    gridSizeSelect:            document.getElementById('grid-size'),
    gridSizeLabelEl:           document.getElementById('grid-size-label'),
    embeddingChannelsSelect:   document.getElementById('embedding-channels'),
    mlpWidth1Select:           document.getElementById('mlp-width1'),
    mlpWidth2Select:           document.getElementById('mlp-width2'),
    quantizationSelect:        document.getElementById('quantization'),
    embBitsSelect:             document.getElementById('emb-bits'),
    activationSelect:          document.getElementById('activation'),
    smoothInterpolationCheckbox: document.getElementById('smooth-interpolation'),
    noOffsetCheckbox:            document.getElementById('no-offset'),
    maxIterInput:  document.getElementById('max-iter'),
    embedLrInput:  document.getElementById('embed-lr'),
    mlpLrInput:    document.getElementById('mlp-lr'),
    maxIterVal:    document.getElementById('max-iter-val'),
    embedLrVal:    document.getElementById('embed-lr-val'),
    mlpLrVal:      document.getElementById('mlp-lr-val'),
    mlpRatioInput:  document.getElementById('mlp-ratio'),
    mlpRatioVal:    document.getElementById('mlp-ratio-val'),
    numLoopsInput:       document.getElementById('num-loops'),
    numLoopsVal:         document.getElementById('num-loops-val'),
    engineSelect:   document.getElementById('engine'),
    bwdStrideInput: document.getElementById('bwd-stride'),
    bwdStrideVal:   document.getElementById('bwd-stride-val'),
    outputZoomInput: document.getElementById('output-zoom'),
    outputZoomVal:   document.getElementById('output-zoom-val'),
    startBtn:      document.getElementById('start-btn'),
    resetBtn:      document.getElementById('reset-btn'),
    shakeEmbBtn:   document.getElementById('shake-emb-btn'),
    shakeMlpBtn:   document.getElementById('shake-mlp-btn'),
    saveBtn:       document.getElementById('save-btn'),
    loadBtn:       document.getElementById('load-btn'),
    exampleSelect: document.getElementById('example-select'),
    snapshotBtn:   document.getElementById('snapshot-btn'),
    recallBtn:     document.getElementById('recall-btn'),
    modelFileInput: document.getElementById('model-file-input'),
    lossValueEl:   document.getElementById('loss-value'),
    stepCounterEl: document.getElementById('step-counter'),
    rateDisplayEl: document.getElementById('rate-display'),
    modelSizeEl:   document.getElementById('model-size'),
    inputSizeEl:   document.getElementById('input-size'),
    statusDotEl:   document.getElementById('status-dot'),
    sourceResEl:   document.getElementById('source-res'),
    outputResEl:   document.getElementById('output-res'),
    roiBrushInput:    document.getElementById('roi-brush'),
    roiBrushVal:      document.getElementById('roi-brush-val'),
    roiStrengthInput: document.getElementById('roi-strength'),
    roiStrengthVal:   document.getElementById('roi-strength-val'),
    roiFreezeChk:     document.getElementById('roi-freeze'),
    roiClearBtn:      document.getElementById('roi-clear-btn'),
    roiAutoBtn:       document.getElementById('roi-auto-btn'),

    vizIntervalSelect:          document.getElementById('viz-interval'),
    offsetSampleIntervalSelect: document.getElementById('offset-sample-interval'),
};

// --- Public Exports ---

export const ui = elements;

export function drawPlaceholder(ctx_canvas) {
    const ctx = ctx_canvas.getContext('2d');
    const style = getComputedStyle(document.documentElement);
    const bg  = style.getPropertyValue('--canvas-bg').trim() || '#1a1a2a';
    const fg  = style.getPropertyValue('--text-dim').trim()  || '#555';
    ctx.clearRect(0, 0, ctx_canvas.width, ctx_canvas.height);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, ctx_canvas.width, ctx_canvas.height);
    ctx.fillStyle = fg;
    ctx.font = `${Math.max(12, Math.min(18, ctx_canvas.width / 14))}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('train or load a model', ctx_canvas.width / 2, ctx_canvas.height / 2);
}

function computeModelSize() {
    const numPts = GRID_SIZES[parseInt(elements.gridSizeSelect.value)];
    const embCh = parseInt(elements.embeddingChannelsSelect.value);
    const mlpW1 = parseInt(elements.mlpWidth1Select.value);
    const mlpW2 = parseInt(elements.mlpWidth2Select.value);
    const emb   = numPts * embCh;
    const outCh = 3;
    const mlp   = embCh*mlpW1 + mlpW1 + mlpW1*mlpW2 + mlpW2 + mlpW2*outCh + outCh;
    const mlpBpp = elements.quantizationSelect.value === 'none' ? 4 : 1;
    const embBpp = parseInt(elements.embBitsSelect.value) / 8;
    return emb * embBpp + mlp * mlpBpp;
}

export function updateSizeDisplay(baseCanvasW, baseCanvasH) {
    const b = computeModelSize();
    const pixels = baseCanvasW * baseCanvasH;
    const bpp = pixels > 0 ? (b * 8 / pixels).toFixed(2) : '—';
    const sizeStr = b >= 1024 ? (b / 1024).toFixed(1) + ' KB' : b + ' B';
    elements.modelSizeEl.textContent = `${sizeStr} (${bpp} bpp)`;
}

export const sliderToLR = v => 1e-4 * Math.pow(300, v / 80);
const formatLR   = lr => lr.toExponential(1);

export function syncSliderDisplay(input, display) {
    display.textContent = formatLR(sliderToLR(parseInt(input.value)));
}

function syncActivationButtons() {
    document.querySelectorAll('.activ-btn').forEach(btn => {
        btn.classList.toggle('selected', btn.dataset.val === elements.activationSelect.value);
    });
}

const STRUCTURAL_CONTROLS = [elements.gridSizeSelect, elements.embeddingChannelsSelect, elements.mlpWidth1Select, elements.mlpWidth2Select];
export function updateDirtyIndicators(lastConfig, currentModelConfig) {
    if (!lastConfig) { STRUCTURAL_CONTROLS.forEach(el => el.removeAttribute('data-dirty')); return; }
    const cur = currentModelConfig;
    elements.gridSizeSelect.toggleAttribute('data-dirty',          lastConfig.gW !== cur.gW || lastConfig.gH !== cur.gH);
    elements.embeddingChannelsSelect.toggleAttribute('data-dirty', lastConfig.embeddingChannels   !== cur.embeddingChannels);
    elements.mlpWidth1Select.toggleAttribute('data-dirty',         lastConfig.mlpWidth1           !== cur.mlpWidth1);
    elements.mlpWidth2Select.toggleAttribute('data-dirty',         lastConfig.mlpWidth2           !== cur.mlpWidth2);
}

export function syncButtonStates(isTraining, hasModel, snapshotWeights) {
    const training      = isTraining;
    const shouldDisable = !hasModel || training;
    elements.resetBtn.disabled        = shouldDisable;
    elements.shakeEmbBtn.disabled     = shouldDisable;
    elements.shakeMlpBtn.disabled     = shouldDisable;
    elements.outputZoomInput.disabled = shouldDisable;
    elements.snapshotBtn.disabled = !hasModel || training;
    elements.recallBtn.disabled   = !snapshotWeights;
    elements.loadBtn.disabled      = training;
    elements.exampleSelect.disabled = training;
    elements.engineSelect.disabled = training;
}

export function updateStartLabel(isCompatible, hasModel, isTraining, snapshotWeights) {
    if (isCompatible && hasModel) {
        elements.startBtn.textContent = '▶ Train';
    } else {
        elements.startBtn.textContent = '▶ Start';
    }
    syncButtonStates(isTraining, hasModel, snapshotWeights);
}

export function setStatus(s, isCompatible, hasModel, isTraining, snapshotWeights) {
    elements.statusDotEl.className = 'status-dot ' + s;
    const modelControls = document.getElementById('model-controls');
    if (s === 'training') {
        elements.startBtn.textContent = '■ Stop';
        elements.startBtn.classList.add('stopping');
        modelControls.inert = true;
        elements.loadBtn.disabled      = true;
        elements.exampleSelect.disabled = true;
    } else {
        elements.startBtn.classList.remove('stopping');
        modelControls.inert = false;
        updateStartLabel(isCompatible, hasModel, isTraining, snapshotWeights);
        return;
    }
    syncButtonStates(isTraining, hasModel, snapshotWeights);
}

export function restoreUISettings({ uiSettings }) {
    if (uiSettings.maxIter !== undefined) {
        elements.maxIterInput.value = uiSettings.maxIter;
        elements.maxIterInput.dispatchEvent(new Event('input'));
    }
    if (uiSettings.embedLr !== undefined) {
        elements.embedLrInput.value = uiSettings.embedLr;
        syncSliderDisplay(elements.embedLrInput, elements.embedLrVal);
    }
    if (uiSettings.mlpLr !== undefined) {
        elements.mlpLrInput.value = uiSettings.mlpLr;
        syncSliderDisplay(elements.mlpLrInput, elements.mlpLrVal);
    }
    if (uiSettings.mlpRatio !== undefined) {
        elements.mlpRatioInput.value = uiSettings.mlpRatio;
        elements.mlpRatioVal.textContent = uiSettings.mlpRatio;
    }
    if (uiSettings.numLoops !== undefined) {
        elements.numLoopsInput.value = uiSettings.numLoops;
        elements.numLoopsVal.textContent = uiSettings.numLoops;
    }
    if (uiSettings.bwdStride !== undefined) {
        elements.bwdStrideInput.value = uiSettings.bwdStride;
        elements.bwdStrideVal.textContent = uiSettings.bwdStride;
    }
    if (uiSettings.outputZoom !== undefined) {
        elements.outputZoomInput.value = uiSettings.outputZoom;
        elements.outputZoomVal.textContent = uiSettings.outputZoom + '×';
    }
    if (uiSettings.noOffset !== undefined) {
        elements.noOffsetCheckbox.checked = uiSettings.noOffset === 'true';
    }
    if (uiSettings.activation !== undefined) {
        elements.activationSelect.value = uiSettings.activation;
        syncActivationButtons();
    }
    if (uiSettings.smoothInterpolation !== undefined) {
        elements.smoothInterpolationCheckbox.checked = uiSettings.smoothInterpolation === 'true';
    }
    if (uiSettings.quantization !== undefined) {
        elements.quantizationSelect.value = uiSettings.quantization;
    }
}

export function updateEmbBitsOptions() {
    const embCh = parseInt(elements.embeddingChannelsSelect.value);
    const opt4 = elements.embBitsSelect.querySelector('option[value="4"]');
    opt4.disabled = embCh < 8;
    if (opt4.disabled && elements.embBitsSelect.value === '4') elements.embBitsSelect.value = '8';
}

export function fitSidePanels(baseW, baseH) {
    const isMobile = window.innerWidth <= 768;
    const rowH = elements.canvasRow.clientHeight;
    const rowW = elements.canvasRow.clientWidth;
    [elements.sourcePanel, elements.outputPanel].forEach(panel => {
        const hdrH = panel.querySelector('.panel-header').offsetHeight;
        const ftrH = panel.querySelector('.layers-legend').offsetHeight;
        let w;
        if (isMobile) {
            // On mobile, max height is 80vw or 450px, width can span the row
            const maxWrapH = Math.min(window.innerWidth * 0.8, 450);
            const wFromH = Math.round(maxWrapH * baseW / baseH);
            const wFromW = rowW;
            w = Math.min(wFromH, wFromW);
        } else {
            // On desktop, height is bound by the row, width shares space with center panel
            const wFromH = Math.round((rowH - hdrH - ftrH) * baseW / baseH);
            const wFromW = Math.floor((rowW - 200 - 12) / 2); // keep ≥200px for center
            w = Math.max(0, Math.min(wFromH, wFromW, 512));
        }
        panel.style.width = w + 'px';
        panel.style.height = '';
        panel.style.maxWidth = '';
        panel.style.maxHeight = '';
        panel.querySelector('.canvas-wrap').style.aspectRatio = `${baseW} / ${baseH}`;
    });
}

export function applyUrlParams() {
    const params = new URLSearchParams(window.location.search);
    let hasParams = false;

    const setSelect = (el, val, allowed) => {
        const n = parseInt(val);
        if (val !== null && allowed.includes(n)) { el.value = String(n); hasParams = true; }
    };

    const gridParam = parseInt(params.get('grid'));
    if (!isNaN(gridParam)) {
        const idx = GRID_SIZES.reduce((best, v, i) =>
            Math.abs(v - gridParam) < Math.abs(GRID_SIZES[best] - gridParam) ? i : best, 0);
        elements.gridSizeSelect.value = String(idx);
        hasParams = true;
    }
    setSelect(elements.embeddingChannelsSelect, params.get('EMB'),  [4, 8, 16]);
    setSelect(elements.mlpWidth1Select,         params.get('MLP1'), [4, 8, 16, 32, 64]);
    setSelect(elements.mlpWidth2Select,         params.get('MLP2'), [4, 8, 16, 32, 64]);

    if (params.has('8qat')) { elements.quantizationSelect.value = 'qat8'; hasParams = true; }
    if (params.has('none')) { elements.quantizationSelect.value = 'none'; hasParams = true; }
    if (params.has('4b'))   { elements.embBitsSelect.value = '4'; hasParams = true; }
    if (params.has('8b'))   { elements.embBitsSelect.value = '8'; hasParams = true; }

    const iters = params.get('iters');
    if (iters !== null) {
        const v = parseInt(iters);
        if (!isNaN(v) && v >= 0) {
            elements.maxIterInput.value = v;
            elements.maxIterInput.dispatchEvent(new Event('input'));
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
    setLR(elements.embedLrInput, elements.embedLrVal, 'embedLr');
    setLR(elements.mlpLrInput,   elements.mlpLrVal,   'mlpLr');

    const numLoopsParam = parseInt(params.get('numLoops'));
    if (!isNaN(numLoopsParam) && numLoopsParam >= 1 && numLoopsParam <= 8) {
        elements.numLoopsInput.value = numLoopsParam;
        elements.numLoopsVal.textContent = numLoopsParam;
        hasParams = true;
    }

    if (params.has('smooth'))   { elements.smoothInterpolationCheckbox.checked = true;  hasParams = true; }
    if (params.has('nosmooth')) { elements.smoothInterpolationCheckbox.checked = false; hasParams = true; }

    if (hasParams) updateEmbBitsOptions();
    return hasParams;
}

// --- Tooltips ---

const TOOLTIPS = {
    'theme-btn':            'Toggle light/dark mode.',
    'model-label':          'Model configuration: grid size, layers, and quantization.',
    'training-label':       'Training parameters: engine, learning rates, and iterations.',
    'roi-label':            'Region of Interest: paint areas to prioritize during training.',
    'help-btn':             'Show help — project overview, URL params, UI guide.',
    'start-btn':            'Start or stop training.',
    'reset-btn':            'Reset model weights and optimizer state.',
    'load-btn':             'Load a .safetensors model file.',
    'save-btn':             'Download model weights as .safetensors.',
    'export-btn':           'Export trained model as a GLSL fragment shader.',
    'grid-size':            'Approximate number of embedding grid points.\nGrid is sized to match image aspect ratio (gW×gH ≈ N pts).\nLarger = finer detail, bigger model.',
    'embedding-channels':   'Feature channels per grid cell.\nMore = richer representation.',
    'mlp-width1':           'Output width of Layer 1 (embCh → W1).\nLarger = more capacity, slower training.',
    'mlp-width2':           'Output width of Layer 2 (W1 → W2).\nLarger = more capacity, slower training.',
    'quantization':         'MLP weight precision.\nQAT trains with simulated 8-bit quantization for compact GLSL export.',
    'activation':           'Hidden-layer activation function.\nSIREN (sin) works well for natural images.',
    'emb-bits':             'Bits per embedding channel.\n4-bit = 2× smaller; requires EMB ≥ 8.',
    'smooth-interpolation': 'Bicubic-like smooth grid sampling (IQ\'s technique).\nImproves reconstruction quality.',
    'no-offset':            'Disable per-plane UV jitter.\nEach plane normally samples at a random offset to increase effective resolution.',
    'max-iter':             'Stop training after N steps. 0 = run forever.',
    'embed-lr':             'Adam learning rate for the embedding grid (log scale, 1e-4 … 1e-1).',
    'mlp-lr':               'Adam learning rate for MLP weights (log scale, 1e-4 … 1e-1).',
    'mlp-ratio':            'Embedding-only Adam steps per MLP step.\nHigher = focus more on the grid.',
    'num-loops':            'Repeat each phase N times per super-cycle.\nratio=5, loops=3 → 15 EMB steps then 3 MLP steps.',
    'bwd-stride':           'Subsample every Nth pixel during backprop.\nHigher = faster steps, noisier gradients.',
    'offset-sample-interval': 'Resample per-plane UV offsets every N steps.\n0 = fixed offsets. Lower = more jitter diversity, slightly noisier.',
    'shake-emb-btn':        'Add small noise to embeddings to escape local minima. Resets embedding Adam moments.',
    'shake-mlp-btn':        'Add small noise to MLP weights and biases to escape local minima. Resets MLP Adam moments.',
    'roi-brush':            'Brush radius for painting the ROI mask (pixels).',
    'roi-strength':         'Loss multiplier inside painted regions.\nHigher = network focuses more on those areas.',
    'roi-freeze':           'Freeze the mask — prevent it from decaying over time.',
    'roi-auto-btn':         'Auto-detect high-variance (high-detail) regions as ROI.',
    'roi-clear-btn':        'Clear the entire ROI mask.',
    'snapshot-btn':         'Save current weights as an in-memory checkpoint.',
    'recall-btn':           'Restore weights from the last in-memory snapshot.',
    'example-select':       'Load a preset image + model.',
    'engine':               'Training backend.\nGPU runs the full backward pass in WGSL; CPU is slower but works without WebGPU.',
    'output-zoom':          'Upscale the output canvas for a closer look.\nRuns a full forward pass at the target resolution.',
    'viz-interval':         'How often the Layers diagram refreshes during training (every N steps).',
    'source-header':        'Click or drop an image to load it.\nDrag on the image to paint a training mask (ROI).',
    'layers-header':        'Flow diagram of the network.\nClick an embedding channel to isolate it; click again to restore all.',
};

function initTooltips() {
    const box = document.createElement('div');
    box.className = 'tooltip-box';
    document.body.appendChild(box);

    let hideTimer = null;
    let showTimer = null;

    function show(text, el) {
        clearTimeout(hideTimer);
        clearTimeout(showTimer);
        box.style.display = 'none';
        showTimer = setTimeout(() => {
            box.textContent = text;
            box.style.visibility = 'hidden';
            box.style.display    = 'block';
            position(el);
            box.style.visibility = 'visible';
        }, 1000);
    }

    function position(el) {
        const r  = el.getBoundingClientRect();
        const bw = box.offsetWidth, bh = box.offsetHeight;
        let left = r.left + r.width / 2 - bw / 2 + 10;
        let top  = r.top + window.scrollY - bh - 14;
        left = Math.max(6, Math.min(left, window.innerWidth - bw - 6));
        if (top < window.scrollY + 6) top = r.bottom + window.scrollY + 8;
        box.style.left = left + 'px';
        box.style.top  = top  + 'px';
    }

    function hide() {
        clearTimeout(showTimer);
        hideTimer = setTimeout(() => { box.style.display = 'none'; }, 60);
    }

    function anchor(el) {
        if (el.tagName === 'BUTTON') return el;
        const row = el.closest('.ctrl-row, .slider-row');
        if (row) return row.querySelector('.ctrl-label, .slider-header') ?? el;
        return el.closest('label') ?? el;
    }

    for (const [id, text] of Object.entries(TOOLTIPS)) {
        const el = document.getElementById(id);
        if (!el) continue;
        const a = anchor(el);
        a.addEventListener('mouseenter', () => show(text, a));
        a.addEventListener('mousemove',  () => position(a));
        a.addEventListener('mouseleave', hide);
    }
}

// --- Init ---

export function init(callbacks) {
    initTooltips();
    syncSliderDisplay(elements.embedLrInput, elements.embedLrVal);
    syncSliderDisplay(elements.mlpLrInput,   elements.mlpLrVal);
    updateSizeDisplay(elements.canvas.width, elements.canvas.height);

    elements.embedLrInput.addEventListener('input', () => syncSliderDisplay(elements.embedLrInput, elements.embedLrVal));
    elements.mlpLrInput.addEventListener('input',   () => syncSliderDisplay(elements.mlpLrInput, elements.mlpLrVal));
    elements.mlpRatioInput.addEventListener('input',  () => { elements.mlpRatioVal.textContent  = elements.mlpRatioInput.value; });
    elements.numLoopsInput.addEventListener('input',       () => { elements.numLoopsVal.textContent       = elements.numLoopsInput.value; });
    elements.bwdStrideInput.addEventListener('input', () => { elements.bwdStrideVal.textContent = elements.bwdStrideInput.value; });
    elements.maxIterInput.addEventListener('input', () => {
        const v = parseInt(elements.maxIterInput.value);
        elements.maxIterVal.textContent = v === 0 ? '∞' : v.toLocaleString();
    });

    document.querySelectorAll('.activ-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.activationSelect.value = btn.dataset.val;
            syncActivationButtons();
            callbacks.onConfigChange();
        });
    });
    syncActivationButtons();

    elements.gridSizeSelect.addEventListener('input', () => {
        updateEmbBitsOptions();
        callbacks.onSelectChange();
    });
    [elements.embeddingChannelsSelect, elements.mlpWidth1Select, elements.mlpWidth2Select,
     elements.quantizationSelect, elements.embBitsSelect, elements.activationSelect].forEach(sel => {
        sel.addEventListener('change', () => {
            updateEmbBitsOptions();
            callbacks.onSelectChange();
        });
    });
    updateEmbBitsOptions();

    document.querySelectorAll('#sidebar .section-label').forEach(label => {
        label.addEventListener('click', () => {
            const section = label.closest('.ctrl-section');
            const wasCollapsed = section.classList.contains('collapsed');
            document.querySelectorAll('.ctrl-section').forEach(s => s.classList.add('collapsed'));
            if (wasCollapsed) section.classList.remove('collapsed');
        });
    });
    document.addEventListener('click', e => {
        if (!e.target.closest('.ctrl-section'))
            document.querySelectorAll('.ctrl-section:not(.collapsed)').forEach(s => s.classList.add('collapsed'));
    });

    const helpOverlay = document.getElementById('help-overlay');
    const openHelp = () => helpOverlay.classList.remove('hidden');
    document.getElementById('help-btn').addEventListener('click', openHelp);
    document.getElementById('app-title').addEventListener('click', openHelp);
    document.getElementById('help-close').addEventListener('click', () => helpOverlay.classList.add('hidden'));
    helpOverlay.addEventListener('click', e => { if (e.target === helpOverlay) helpOverlay.classList.add('hidden'); });
    document.querySelectorAll('input, select, button:not(#start-btn)').forEach(el => {
        el.addEventListener('keydown', e => {
            e.stopPropagation();
            if (e.key === ' ' || e.key === 'Enter') e.preventDefault();
        });
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') { helpOverlay.classList.add('hidden'); return; }
    }, true);
    document.addEventListener('keydown', e => {
        if (e.key === ' ' && e.target === document.body) {
            e.preventDefault();
            elements.startBtn.click();
        }
    });
}
