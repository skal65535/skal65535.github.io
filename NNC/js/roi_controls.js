import { DOM } from './ui_manager.js';

let _roiMask, _isTraining, _drawSourceImage;
let decayRafId = null;
let isPainting  = false;
let didPaint    = false;

export const hasPainted = () => didPaint;

export function startDecayLoop() {
    if (decayRafId !== null || !_roiMask.isActive()) return;
    function tick(now) {
        if (!DOM.roiFreezeChk.checked) _roiMask.decay(now);
        _drawSourceImage();
        decayRafId = (DOM.roiFreezeChk.checked || _roiMask.isActive()) ? requestAnimationFrame(tick) : null;
    }
    decayRafId = requestAnimationFrame(tick);
}

export function stopDecayLoop() {
    if (decayRafId !== null) { cancelAnimationFrame(decayRafId); decayRafId = null; }
}

function sourceCanvasCoords(e) {
    const r = DOM.sourceCanvas.getBoundingClientRect();
    return [
        (e.clientX - r.left) * DOM.sourceCanvas.width  / r.width,
        (e.clientY - r.top)  * DOM.sourceCanvas.height / r.height,
    ];
}

export function init({ roiMask, isTraining, drawSourceImage }) {
    _roiMask         = roiMask;
    _isTraining      = isTraining;
    _drawSourceImage = drawSourceImage;

    DOM.roiBrushInput.addEventListener('input', () => { DOM.roiBrushVal.textContent = DOM.roiBrushInput.value; });
    DOM.roiStrengthInput.addEventListener('input', () => { DOM.roiStrengthVal.textContent = DOM.roiStrengthInput.value; });
    DOM.roiFreezeChk.addEventListener('change', () => {
        if (!DOM.roiFreezeChk.checked && _roiMask.isActive() && !_isTraining()) startDecayLoop();
    });
    DOM.roiClearBtn.addEventListener('click', () => {
        _roiMask.clear();
        stopDecayLoop();
        _drawSourceImage();
    });
    DOM.roiAutoBtn.addEventListener('click', () => {
        if (!DOM.sourcePanel.classList.contains('has-image')) return;
        const ctx = DOM.sourceCanvas.getContext('2d');
        const id  = ctx.getImageData(0, 0, DOM.sourceCanvas.width, DOM.sourceCanvas.height);
        _roiMask.autoMask(id.data);
        DOM.roiFreezeChk.checked = true;
        stopDecayLoop();
        _drawSourceImage();
    });

    DOM.sourceCanvas.addEventListener('mousedown', (e) => {
        if (!DOM.sourcePanel.classList.contains('has-image')) return;
        isPainting = true;
        didPaint   = false;
        const [x, y] = sourceCanvasCoords(e);
        _roiMask.paint(x, y, parseInt(DOM.roiBrushInput.value));
        _drawSourceImage();
        if (!_isTraining()) startDecayLoop();
    });
    DOM.sourceCanvas.addEventListener('mousemove', (e) => {
        if (!isPainting) return;
        didPaint = true;
        const [x, y] = sourceCanvasCoords(e);
        _roiMask.paint(x, y, parseInt(DOM.roiBrushInput.value));
        _drawSourceImage();
    });
    window.addEventListener('mouseup', () => { isPainting = false; });
    DOM.sourceCanvas.addEventListener('mouseleave', () => { isPainting = false; });
}
