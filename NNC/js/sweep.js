// sweep.js
// Animates column bounding boxes in the layers panel: each box lights up and fades
// in sequence, left→right (fwd, cyan) then right→left (bwd, orange).

const SWEEP_MS  = 320;  // time for wave to travel across all columns
const FADE_MS   = 380;  // how long each box glows after activation
const TOTAL_MS  = SWEEP_MS + FADE_MS;
const DECAY_MS  = 10000; // training sweep fades to zero over this duration

export class SweepOverlay {
    constructor(canvas, srcCanvas = null) {
        this._canvas = canvas;
        this._ctx    = canvas.getContext('2d');
        this._cols   = [];
        this._phase  = 'idle'; // 'fwd' | 'bwd' | 'idle'
        this._t0     = 0;
        this._bwdPending = false;
        this._rafId  = null;
        this._lastTrigger = 0;
        this._decayStart  = 0; // 0 = not yet started
        this._srcCanvas = srcCanvas;
        this._srcCtx    = srcCanvas ? srcCanvas.getContext('2d') : null;
        this._gW = 0;
        this._gH = 0;
    }

    setCols(cols) { this._cols = cols; }
    setGrid(gW, gH) { this._gW = gW; this._gH = gH; }

    // Call when a new training session starts to reset the decay timer.
    resetDecay() { this._decayStart = 0; this._lastTrigger = 0; }

    triggerStep() {
        const now = performance.now();
        if (this._decayStart === 0) this._decayStart = now;
        if (now - this._decayStart >= DECAY_MS) return; // fully faded — stay silent
        if (now - this._lastTrigger < 1000) return;
        this._lastTrigger = now;
        this._phase = 'fwd';
        this._t0 = now;
        this._bwdPending = true;
        this._schedule();
    }

    triggerFwd() {
        this._phase = 'fwd';
        this._t0 = performance.now();
        this._bwdPending = false;
        this._schedule();
    }

    _schedule() {
        if (!this._rafId) this._rafId = requestAnimationFrame(() => this._tick());
    }

    _tick() {
        this._rafId = null;
        const t = performance.now() - this._t0;
        this._draw(Math.min(t, TOTAL_MS));

        if (t < TOTAL_MS) {
            this._rafId = requestAnimationFrame(() => this._tick());
        } else {
            this._clear();
            if (this._bwdPending) {
                this._bwdPending = false;
                this._phase = 'bwd';
                this._t0 = performance.now();
                this._rafId = requestAnimationFrame(() => this._tick());
            } else {
                this._phase = 'idle';
            }
        }
    }

    _draw(t) {
        const cols = this._cols;
        if (!cols.length) return;
        const ctx = this._ctx;
        const cv  = this._canvas;
        ctx.clearRect(0, 0, cv.width, cv.height);

        const n   = cols.length;
        const isFwd = this._phase === 'fwd';
        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        let r, g, b;
        if (isFwd) {
            [r, g, b] = isLight ? [42, 112, 192] : [0, 210, 255];
        } else {
            [r, g, b] = isLight ? [192, 120, 0] : [255, 150, 0];
        }
        const decay = this._decayStart > 0
            ? Math.max(0, 1 - (performance.now() - this._decayStart) / DECAY_MS)
            : 1;

        ctx.lineWidth = 2;
        ctx.shadowBlur = (isLight ? 4 : 8) * decay;
        ctx.shadowColor = `rgb(${r},${g},${b})`;

        for (let i = 0; i < n; i++) {
            const col = cols[i];
            const seqIdx = isFwd ? i : (n - 1 - i);
            const tActivate = (seqIdx / (n - 1)) * SWEEP_MS;
            const elapsed = t - tActivate;
            if (elapsed <= 0) continue;
            const alpha = Math.max(0, 1 - elapsed / FADE_MS) * decay;
            ctx.strokeStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
            ctx.strokeRect(col.x + 0.5, col.y0 + 0.5, col.w - 1, col.h - 1);
        }
        ctx.shadowBlur = 0;

        // Source canvas: activates first in fwd, last in bwd.
        if (this._srcCtx) {
            const sc = this._srcCanvas;
            this._srcCtx.clearRect(0, 0, sc.width, sc.height);
            const tActivate = isFwd ? 0 : SWEEP_MS;
            const elapsed = t - tActivate;
            if (elapsed > 0) {
                const alpha = Math.max(0, 1 - elapsed / FADE_MS) * decay;
                this._srcCtx.lineWidth = 3;
                this._srcCtx.shadowBlur = (isLight ? 5 : 10) * decay;
                this._srcCtx.shadowColor = `rgb(${r},${g},${b})`;
                this._srcCtx.strokeStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
                this._srcCtx.strokeRect(2, 2, sc.width - 4, sc.height - 4);
                this._srcCtx.shadowBlur = 0;

                if (this._gW > 1 && this._gH > 1) {
                    const gW = this._gW, gH = this._gH;
                    const cellW = sc.width  / (gW - 1);
                    const cellH = sc.height / (gH - 1);
                    const cr = Math.max(1.5, Math.min(4, Math.min(cellW, cellH) * 0.22));
                    this._srcCtx.fillStyle = `rgba(${r},${g},${b},${(alpha * 0.35).toFixed(3)})`;
                    this._srcCtx.beginPath();
                    for (let iy = 0; iy < gH; iy++) {
                        for (let ix = 0; ix < gW; ix++) {
                            const cx = ix * cellW;
                            const cy = iy * cellH;
                            this._srcCtx.moveTo(cx + cr, cy);
                            this._srcCtx.arc(cx, cy, cr, 0, Math.PI * 2);
                        }
                    }
                    this._srcCtx.fill();
                }
            }
        }
    }

    _clear() {
        this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
        if (this._srcCtx) this._srcCtx.clearRect(0, 0, this._srcCanvas.width, this._srcCanvas.height);
    }
}
