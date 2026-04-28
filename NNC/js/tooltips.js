const TIPS = {
    'grid-size':            'Spatial resolution of the feature grid (N×N).\nLarger = finer detail, bigger model.',
    'embedding-channels':   'Feature channels per grid cell.\nMore = richer representation.',
    'mlp-width':            'Hidden layer size of the MLP decoder.\nLarger = more capacity, slower training.',
    'quantization':         'MLP weight precision.\nQAT trains with simulated 8-bit quantization for compact GLSL export.',
    'emb-bits':             'Bits per embedding channel.\n4-bit = 2× smaller; requires EMB ≥ 8.',
    'smooth-interpolation': "Bicubic-like smooth grid sampling (IQ's technique).\nImproves reconstruction quality.",
    'no-offset':            'Disable per-plane UV jitter.\nEach plane normally samples at a random offset to increase effective resolution.',
    'max-iter':             'Stop training after N steps. 0 = run forever.',
    'embed-lr':             'Adam learning rate for the embedding grid (log scale, 1e-4 … 1e-1).',
    'mlp-lr':               'Adam learning rate for MLP weights (log scale, 1e-4 … 1e-1).',
    'mlp-ratio':            'Embedding-only Adam steps per MLP step.\nHigher = focus more on the grid.',
    'bwd-stride':           'Subsample every Nth pixel during backprop.\nHigher = faster steps, noisier gradients.',
    'roi-brush':            'Brush radius for painting the ROI mask (pixels).',
    'roi-strength':         'Loss multiplier inside painted regions.\nHigher = network focuses more on those areas.',
    'roi-freeze':           'Freeze the mask — prevent it from decaying over time.',
    'roi-auto-btn':         'Auto-detect high-variance (high-detail) regions as ROI.',
    'roi-clear-btn':        'Clear the entire ROI mask.',
    'start-btn':            'Start or stop training.',
    'shake-btn':            'Add small noise to embeddings to escape local minima.',
    'reset-btn':            'Reset model weights and optimizer state.',
    'snapshot-btn':         'Save a weight checkpoint in memory, or restore it.',
    'save-btn':             'Download model weights as .safetensors.',
    'load-btn':             'Load a .safetensors model file.',
    'export-btn':           'Export trained model as a GLSL fragment shader.',
    'help-btn':             'Show help — project overview, URL params, UI guide.',
};

export function initTooltips() {
    const box = document.createElement('div');
    box.className = 'tooltip-box';
    document.body.appendChild(box);

    let hideTimer = null;

    function show(text, el) {
        clearTimeout(hideTimer);
        box.textContent = text;
        box.style.visibility = 'hidden';
        box.style.display    = 'block';
        position(el);
        box.style.visibility = 'visible';
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
        hideTimer = setTimeout(() => { box.style.display = 'none'; }, 60);
    }

    function anchor(el) {
        if (el.tagName === 'BUTTON') return el;
        const row = el.closest('.ctrl-row, .slider-row');
        if (row) return row.querySelector('.ctrl-label, .slider-header') ?? el;
        return el.closest('label') ?? el;
    }

    for (const [id, text] of Object.entries(TIPS)) {
        const el = document.getElementById(id);
        if (!el) continue;
        const a = anchor(el);
        a.addEventListener('mouseenter', () => show(text, a));
        a.addEventListener('mousemove',  () => position(a));
        a.addEventListener('mouseleave', hide);
    }
}
