// tooltips.js
// All tooltip text is defined here and applied programmatically.
const TOOLTIPS = {
    'help-btn':             'Show help — project overview, URL params, UI guide.',
    'start-btn':            'Start or stop training.',
    'reset-btn':            'Reset model weights and optimizer state.',
    'load-btn':             'Load a .safetensors model file.',
    'save-btn':             'Download model weights as .safetensors.',
    'export-btn':           'Export trained model as a GLSL fragment shader.',
    'grid-size':            'Spatial resolution of the feature grid (N×N).\nLarger = finer detail, bigger model.',
    'embedding-channels':   'Feature channels per grid cell.\nMore = richer representation.',
    'mlp-width1':           'Output width of Layer 1 (embCh → W1).\nLarger = more capacity, slower training.',
    'mlp-width2':           'Output width of Layer 2 (W1 → W2).\nLarger = more capacity, slower training.',
    'quantization':         'MLP weight precision.\nQAT trains with simulated 8-bit quantization for compact GLSL export.',
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
};

export function initTooltips() {
    const box = document.createElement('div');
    box.className = 'tooltip-box';
    document.body.appendChild(box);

    let hideTimer = null;
    let showTimer = null;

    function show(text, el) {
        clearTimeout(hideTimer);
        clearTimeout(showTimer);
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
