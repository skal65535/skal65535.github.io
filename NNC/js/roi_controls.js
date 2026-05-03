import { ui } from './ui.js?v=2';

let _roiMask, _isTraining, _drawSourceImage;
let decayRafId = null;
let isPainting  = false;
let didPaint    = false;

export const hasPainted = () => didPaint;

export function startDecayLoop() {
    if (decayRafId !== null || !_roiMask.isActive()) return;
    function tick(now) {
        if (!ui.roiFreezeChk.checked) _roiMask.decay(now);
        _drawSourceImage();
        decayRafId = (ui.roiFreezeChk.checked || _roiMask.isActive()) ? requestAnimationFrame(tick) : null;
    }
    decayRafId = requestAnimationFrame(tick);
}

export function stopDecayLoop() {
    if (decayRafId !== null) { cancelAnimationFrame(decayRafId); decayRafId = null; }
}

function sourceCanvasCoords(e) {
    const r = ui.sourceCanvas.getBoundingClientRect();
    return [
        (e.clientX - r.left) * ui.sourceCanvas.width  / r.width,
        (e.clientY - r.top)  * ui.sourceCanvas.height / r.height,
    ];
}

export function init({ roiMask, isTraining, drawSourceImage }) {
    _roiMask         = roiMask;
    _isTraining      = isTraining;
    _drawSourceImage = drawSourceImage;

    ui.roiBrushInput.addEventListener('input', () => { ui.roiBrushVal.textContent = ui.roiBrushInput.value; });
    ui.roiStrengthInput.addEventListener('input', () => { ui.roiStrengthVal.textContent = ui.roiStrengthInput.value; });
    ui.roiFreezeChk.addEventListener('change', () => {
        if (!ui.roiFreezeChk.checked && _roiMask.isActive() && !_isTraining()) startDecayLoop();
    });
    ui.roiClearBtn.addEventListener('click', () => {
        _roiMask.clear();
        stopDecayLoop();
        _drawSourceImage();
    });
    ui.roiAutoBtn.addEventListener('click', () => {
        if (!ui.sourcePanel.classList.contains('has-image')) return;
        const ctx = ui.sourceCanvas.getContext('2d');
        const id  = ctx.getImageData(0, 0, ui.sourceCanvas.width, ui.sourceCanvas.height);
        _roiMask.autoMask(id.data);
        ui.roiFreezeChk.checked = true;
        stopDecayLoop();
        _drawSourceImage();
    });

    ui.sourceCanvas.addEventListener('mousedown', (e) => {
        if (!ui.sourcePanel.classList.contains('has-image')) return;
        isPainting = true;
        didPaint   = false;
        const [x, y] = sourceCanvasCoords(e);
        _roiMask.paint(x, y, parseInt(ui.roiBrushInput.value));
        _drawSourceImage();
        if (!_isTraining()) startDecayLoop();
    });
    ui.sourceCanvas.addEventListener('mousemove', (e) => {
        if (!isPainting) return;
        didPaint = true;
        const [x, y] = sourceCanvasCoords(e);
        _roiMask.paint(x, y, parseInt(ui.roiBrushInput.value));
        _drawSourceImage();
    });
    window.addEventListener('mouseup', () => { isPainting = false; });
    ui.sourceCanvas.addEventListener('mouseleave', () => { isPainting = false; });
}
