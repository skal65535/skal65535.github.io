// sweep.js
// Animates column bounding boxes in the layers panel: each box lights up and fades
// in sequence, left→right (fwd, cyan) then right→left (bwd, orange).

const SWEEP_MS  = 320;  // time for wave to travel across all columns
const FADE_MS   = 380;  // how long each box glows after activation
const TOTAL_MS  = SWEEP_MS + FADE_MS;
const DECAY_MS  = 10000; // training sweep fades to zero over this duration

export class SweepOverlay {
    constructor(canvas) {
        this._canvas = canvas;
        this._ctx    = canvas.getContext('2d');
        this._cols   = [];
        this._phase  = 'idle'; // 'fwd' | 'bwd' | 'idle'
        this._t0     = 0;
        this._bwdPending = false;
        this._rafId  = null;
        this._lastTrigger = 0;
        this._decayStart  = 0; // 0 = not yet started
    }

    setCols(cols) { this._cols = cols; }

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
        const [r, g, b] = isFwd ? [0, 210, 255] : [255, 150, 0];
        const decay = this._decayStart > 0
            ? Math.max(0, 1 - (performance.now() - this._decayStart) / DECAY_MS)
            : 1;

        ctx.lineWidth = 2;
        ctx.shadowBlur = 8 * decay;
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
    }

    _clear() {
        this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    }
}
