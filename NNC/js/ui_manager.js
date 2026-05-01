// ui_manager.js

// This module handles all direct DOM interactions and UI updates.
// It is initialized by main.js and communicates back through callbacks.

export const DOM = {
    sourcePanel:  document.getElementById('source-panel'),
    outputPanel:  document.getElementById('output-panel'),
    canvasRow:    document.querySelector('.canvas-row'),
    sourceCanvas: document.getElementById('source-canvas'),
    layersCanvas: document.getElementById('layers-canvas'),
    sweepCanvas:  document.getElementById('sweep-canvas'),
    canvas:       document.getElementById('canvas'),
    lossCanvas:   document.getElementById('loss-canvas'),
    fileInput:    document.getElementById('file-input'),
    dropOverlay:  document.getElementById('drop-overlay'),

    gridSizeSelect:            document.getElementById('grid-size'),
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
    numLoopsInput:  document.getElementById('num-loops'),
    numLoopsVal:    document.getElementById('num-loops-val'),
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
    statusTextEl:  document.getElementById('status-text'),
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

export function drawPlaceholder(ctx_canvas) {
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

function computeModelSize() {
    const gs   = parseInt(DOM.gridSizeSelect.value);
    const embCh = parseInt(DOM.embeddingChannelsSelect.value);
    const mlpW1 = parseInt(DOM.mlpWidth1Select.value);
    const mlpW2 = parseInt(DOM.mlpWidth2Select.value);
    const emb   = gs * gs * embCh;
    const outCh = 3; // conservative estimate (no alpha)
    const mlp   = embCh*mlpW1 + mlpW1 + mlpW1*mlpW2 + mlpW2 + mlpW2*outCh + outCh;
    const mlpBpp = DOM.quantizationSelect.value === 'none' ? 4 : 1;
    const embBpp = parseInt(DOM.embBitsSelect.value) / 8;
    return emb * embBpp + mlp * mlpBpp;
}

export function updateSizeDisplay(baseCanvasW, baseCanvasH) {
    const b = computeModelSize();
    const pixels = baseCanvasW * baseCanvasH;
    const bpp = pixels > 0 ? (b * 8 / pixels).toFixed(2) : '—';
    const sizeStr = b >= 1024 ? (b / 1024).toFixed(1) + ' KB' : b + ' B';
    DOM.modelSizeEl.textContent = `${sizeStr} (${bpp} bpp)`;
}

export const sliderToLR = v => 1e-4 * Math.pow(300, v / 80);
const formatLR   = lr => lr.toExponential(1);

export function syncSliderDisplay(input, display) {
    display.textContent = formatLR(sliderToLR(parseInt(input.value)));
}

function syncActivationButtons() {
    document.querySelectorAll('.activ-btn').forEach(btn => {
        btn.classList.toggle('selected', btn.dataset.val === DOM.activationSelect.value);
    });
}

const STRUCTURAL_CONTROLS = [DOM.gridSizeSelect, DOM.embeddingChannelsSelect, DOM.mlpWidth1Select, DOM.mlpWidth2Select];
export function updateDirtyIndicators(lastConfig, currentModelConfig) {
    if (!lastConfig) { STRUCTURAL_CONTROLS.forEach(el => el.removeAttribute('data-dirty')); return; }
    const cur = currentModelConfig;
    DOM.gridSizeSelect.toggleAttribute('data-dirty',          lastConfig.gridSize            !== cur.gridSize);
    DOM.embeddingChannelsSelect.toggleAttribute('data-dirty', lastConfig.embeddingChannels   !== cur.embeddingChannels);
    DOM.mlpWidth1Select.toggleAttribute('data-dirty',         lastConfig.mlpWidth1           !== cur.mlpWidth1);
    DOM.mlpWidth2Select.toggleAttribute('data-dirty',         lastConfig.mlpWidth2           !== cur.mlpWidth2);
}

export function syncButtonStates(isTraining, hasModel, snapshotWeights) {
    const training      = isTraining;
    const shouldDisable = !hasModel || training;
    DOM.resetBtn.disabled        = shouldDisable;
    DOM.shakeEmbBtn.disabled     = shouldDisable;
    DOM.shakeMlpBtn.disabled     = shouldDisable;
    DOM.outputZoomInput.disabled = shouldDisable;
    DOM.snapshotBtn.disabled = !hasModel || training;
    DOM.recallBtn.disabled   = !snapshotWeights;
    DOM.loadBtn.disabled      = training;
    DOM.exampleSelect.disabled = training;
    DOM.engineSelect.disabled = training;
}

export function updateStartLabel(isCompatible, hasModel, isTraining, snapshotWeights) {
    if (isCompatible && hasModel) {
        DOM.startBtn.textContent = '▶ Resume';
    } else {
        DOM.startBtn.textContent = '▶ Start';
    }
    syncButtonStates(isTraining, hasModel, snapshotWeights);
}

export function setStatus(s, isCompatible, hasModel, isTraining, snapshotWeights) {
    const labels = { idle: 'IDLE', training: 'TRAINING', stopped: 'STOPPED' };
    DOM.statusTextEl.textContent = labels[s] || s;
    DOM.statusDotEl.className    = 'status-dot ' + s;
    const modelControls = document.getElementById('model-controls');
    if (s === 'training') {
        DOM.startBtn.textContent = '■ Stop';
        DOM.startBtn.classList.add('stopping');
        modelControls.inert = true;
        DOM.loadBtn.disabled      = true;
        DOM.exampleSelect.disabled = true;
    } else {
        DOM.startBtn.classList.remove('stopping');
        modelControls.inert = false;
        updateStartLabel(isCompatible, hasModel, isTraining, snapshotWeights);
        return;
    }
    syncButtonStates(isTraining, hasModel, snapshotWeights);
}

export function restoreUISettings({ uiSettings }) {
    if (uiSettings.maxIter !== undefined) {
        DOM.maxIterInput.value = uiSettings.maxIter;
        DOM.maxIterInput.dispatchEvent(new Event('input'));
    }
    if (uiSettings.embedLr !== undefined) {
        DOM.embedLrInput.value = uiSettings.embedLr;
        syncSliderDisplay(DOM.embedLrInput, DOM.embedLrVal);
    }
    if (uiSettings.mlpLr !== undefined) {
        DOM.mlpLrInput.value = uiSettings.mlpLr;
        syncSliderDisplay(DOM.mlpLrInput, DOM.mlpLrVal);
    }
    if (uiSettings.mlpRatio !== undefined) {
        DOM.mlpRatioInput.value = uiSettings.mlpRatio;
        DOM.mlpRatioVal.textContent = uiSettings.mlpRatio;
    }
    if (uiSettings.numLoops !== undefined) {
        DOM.numLoopsInput.value = uiSettings.numLoops;
        DOM.numLoopsVal.textContent = uiSettings.numLoops;
    }
    if (uiSettings.bwdStride !== undefined) {
        DOM.bwdStrideInput.value = uiSettings.bwdStride;
        DOM.bwdStrideVal.textContent = uiSettings.bwdStride;
    }
    if (uiSettings.outputZoom !== undefined) {
        DOM.outputZoomInput.value = uiSettings.outputZoom;
        DOM.outputZoomVal.textContent = uiSettings.outputZoom + '×';
    }
    if (uiSettings.noOffset !== undefined) {
        DOM.noOffsetCheckbox.checked = uiSettings.noOffset === 'true';
    }
    if (uiSettings.activation !== undefined) {
        DOM.activationSelect.value = uiSettings.activation;
        syncActivationButtons();
    }
    if (uiSettings.smoothInterpolation !== undefined) {
        DOM.smoothInterpolationCheckbox.checked = uiSettings.smoothInterpolation === 'true';
    }
    if (uiSettings.quantization !== undefined) {
        DOM.quantizationSelect.value = uiSettings.quantization;
    }
}

export function updateEmbBitsOptions() {
    const embCh = parseInt(DOM.embeddingChannelsSelect.value);
    const opt4 = DOM.embBitsSelect.querySelector('option[value="4"]');
    opt4.disabled = embCh < 8;
    if (opt4.disabled && DOM.embBitsSelect.value === '4') DOM.embBitsSelect.value = '8';
}

export function fitSidePanels(baseW, baseH) {
    const rowH = DOM.canvasRow.clientHeight;
    [DOM.sourcePanel, DOM.outputPanel].forEach(panel => {
        const hdrH = panel.querySelector('.panel-header').offsetHeight;
        const w = Math.round((rowH - hdrH) * baseW / baseH);
        panel.style.width = '';
        panel.style.maxWidth = w + 'px';
    });
}

export function applyUrlParams() {
    const params = new URLSearchParams(window.location.search);
    let hasParams = false;

    const setSelect = (el, val, allowed) => {
        const n = parseInt(val);
        if (val !== null && allowed.includes(n)) { el.value = String(n); hasParams = true; }
    };

    setSelect(DOM.gridSizeSelect,          params.get('grid'), [16, 32, 64]);
    setSelect(DOM.embeddingChannelsSelect, params.get('EMB'),  [4, 8, 16]);
    setSelect(DOM.mlpWidth1Select,         params.get('MLP1'), [4, 8, 16, 32, 64]);
    setSelect(DOM.mlpWidth2Select,         params.get('MLP2'), [4, 8, 16, 32, 64]);

    if (params.has('8qat')) { DOM.quantizationSelect.value = 'qat8'; hasParams = true; }
    if (params.has('none')) { DOM.quantizationSelect.value = 'none'; hasParams = true; }
    if (params.has('4b'))   { DOM.embBitsSelect.value = '4'; hasParams = true; }
    if (params.has('8b'))   { DOM.embBitsSelect.value = '8'; hasParams = true; }

    const iters = params.get('iters');
    if (iters !== null) {
        const v = parseInt(iters);
        if (!isNaN(v) && v >= 0) {
            DOM.maxIterInput.value = v;
            DOM.maxIterInput.dispatchEvent(new Event('input'));
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
    setLR(DOM.embedLrInput, DOM.embedLrVal, 'embedLr');
    setLR(DOM.mlpLrInput,   DOM.mlpLrVal,   'mlpLr');

    const numLoopsParam = parseInt(params.get('numLoops'));
    if (!isNaN(numLoopsParam) && numLoopsParam >= 1 && numLoopsParam <= 8) {
        DOM.numLoopsInput.value = numLoopsParam;
        DOM.numLoopsVal.textContent = numLoopsParam;
        hasParams = true;
    }

    if (params.has('smooth'))   { DOM.smoothInterpolationCheckbox.checked = true;  hasParams = true; }
    if (params.has('nosmooth')) { DOM.smoothInterpolationCheckbox.checked = false; hasParams = true; }

    if (hasParams) updateEmbBitsOptions();
    return hasParams;
}

export function init(callbacks) {
    syncSliderDisplay(DOM.embedLrInput, DOM.embedLrVal);
    syncSliderDisplay(DOM.mlpLrInput,   DOM.mlpLrVal);
    updateSizeDisplay(DOM.canvas.width, DOM.canvas.height);

    DOM.embedLrInput.addEventListener('input', () => syncSliderDisplay(DOM.embedLrInput, DOM.embedLrVal));
    DOM.mlpLrInput.addEventListener('input',   () => syncSliderDisplay(DOM.mlpLrInput, DOM.mlpLrVal));
    DOM.maxIterInput.addEventListener('input', () => {
        const v = parseInt(DOM.maxIterInput.value);
        DOM.maxIterVal.textContent = v === 0 ? '∞' : v.toLocaleString();
    });

    document.querySelectorAll('.activ-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            DOM.activationSelect.value = btn.dataset.val;
            syncActivationButtons();
            callbacks.onConfigChange();
        });
    });
    syncActivationButtons();

    [DOM.gridSizeSelect, DOM.embeddingChannelsSelect, DOM.mlpWidth1Select, DOM.mlpWidth2Select,
     DOM.quantizationSelect, DOM.embBitsSelect, DOM.activationSelect].forEach(sel => {
        sel.addEventListener('change', () => {
            updateEmbBitsOptions();
            callbacks.onSelectChange();
        });
    });
    updateEmbBitsOptions();

    // Collapsible topbar sections
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

    // Help modal
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
        if (e.key === ' ' && e.target === document.body) {
            e.preventDefault();
            DOM.startBtn.click();
        }
    });
}
