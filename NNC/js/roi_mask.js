// roi_mask.js — Per-pixel ROI loss-weight mask.
// Paint with mouse; weights decay exponentially toward zero.

const HALF_LIFE_MS = 3000;
const DECAY_RATE   = Math.LN2 / HALF_LIFE_MS; // weight halves every 3 s

export class ROIMask {
    constructor(w, h) {
        this.w = w;
        this.h = h;
        this.weights = new Float32Array(w * h);
        this._lastDecayAt = 0;
        this._offscreen   = null;
    }

    resize(w, h) {
        if (this.w === w && this.h === h) return;
        this.w = w;
        this.h = h;
        this.weights      = new Float32Array(w * h);
        this._lastDecayAt = 0;
        this._offscreen   = null;
    }

    isActive() {
        for (let i = 0; i < this.weights.length; i++) {
            if (this.weights[i] > 0.005) return true;
        }
        return false;
    }

    // dt-based exponential decay; call with performance.now()
    decay(now) {
        if (this._lastDecayAt === 0) { this._lastDecayAt = now; return; }
        const dt = now - this._lastDecayAt;
        this._lastDecayAt = now;
        if (dt <= 0) return;
        const factor = Math.exp(-DECAY_RATE * dt);
        for (let i = 0; i < this.weights.length; i++) this.weights[i] *= factor;
    }

    // Paint a soft radial brush at canvas coordinates (cx, cy).
    paint(cx, cy, radius) {
        const r  = Math.ceil(radius);
        const r2 = radius * radius;
        const x0 = Math.max(0,          (cx - r) | 0);
        const x1 = Math.min(this.w - 1, (cx + r) | 0);
        const y0 = Math.max(0,          (cy - r) | 0);
        const y1 = Math.min(this.h - 1, (cy + r) | 0);
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                const d2 = (x - cx) ** 2 + (y - cy) ** 2;
                if (d2 <= r2) {
                    const i = y * this.w + x;
                    this.weights[i] = Math.min(1, this.weights[i] + (1 - d2 / r2) * 0.08);
                }
            }
        }
        if (this._lastDecayAt === 0) this._lastDecayAt = performance.now();
    }

    clear() {
        this.weights.fill(0);
        this._lastDecayAt = 0;
    }

    // Composite an orange tint over whatever is already drawn on ctx.
    drawOverlay(ctx) {
        const { w, h } = this;
        if (!this._offscreen || this._offscreen.width !== w || this._offscreen.height !== h) {
            this._offscreen = new OffscreenCanvas(w, h);
        }
        const octx = this._offscreen.getContext('2d');
        const id   = octx.createImageData(w, h);
        const d    = id.data;
        for (let i = 0; i < this.weights.length; i++) {
            const a = this.weights[i];
            if (a < 0.005) continue;
            const b  = i * 4;
            d[b]     = 255;
            d[b + 1] = 100;
            d[b + 2] = 0;
            d[b + 3] = (a * 180) | 0;
        }
        octx.putImageData(id, 0, 0);
        ctx.drawImage(this._offscreen, 0, 0);
    }
}
