// optimizer.js
// CPU-side Adam optimizer + forward/backward passes (mobile/fallback path).
import { cpuPackEmbeddings, computeEmbRange } from './model.js';

// --- Activation helpers ---
function makeActiv(name) {
    switch (name) {
        case 'tanh':
            return { f: Math.tanh, df: x => { const t = Math.tanh(x); return 1 - t * t; } };
        case 'softsign':
            return { f: x => x / (1 + Math.abs(x)), df: x => { const d = 1 + Math.abs(x); return 1 / (d * d); } };
        case 'none':
            return { f: x => x, df: _ => 1 };
        default: // 'sin' / SIREN
            return { f: Math.sin, df: Math.cos };
    }
}

function qat8(w) { return Math.round(Math.max(-1, Math.min(1, w)) * 127) / 127; }

function unpack4x8snorm(u) {
    const out = new Float32Array(4);
    for (let i = 0; i < 4; i++) {
        let b = (u >>> (i * 8)) & 0xFF;
        if (b > 127) b -= 256;
        out[i] = Math.max(b / 127.0, -1.0);
    }
    return out;
}

function dequant4(n) { return (n >= 8 ? n - 16 : n) / 7.0; }

// --- Forward pass ---
// range: Float32Array from computeEmbRange — [min_ch0, max_ch0, min_ch1, max_ch1, ...]
export function forward(config, weights, range) {
    const { gW, gH, embeddingChannels, mlpWidth1, mlpWidth2,
            quantization, smoothInterpolation, activation = 'sin',
            embBits = 8, hasAlpha = false, embOffsets, width, height } = config;
    const outCh    = hasAlpha ? 4 : 3;
    const numU32   = embeddingChannels / (32 / embBits);
    const pixelCount = width * height;
    const { f: activ } = makeActiv(activation);
    const useQat   = quantization === 'qat8';

    const final       = new Float32Array(pixelCount * 4);
    const interLayer1 = new Float32Array(pixelCount * mlpWidth1);
    const interLayer2 = new Float32Array(pixelCount * mlpWidth2);
    const emb_q       = weights.embeddings_q;  // Uint32Array
    const invW = 1 / (width  - 1);
    const invH = 1 / (height - 1);

    for (let py = 0; py < height; py++) {
        for (let px = 0; px < width; px++) {
            const pidx = py * width + px;
            const uvx  = px * invW;
            const uvy  = py * invH;

            // --- Sample & dequantize embedding ---
            const emb = new Float32Array(embeddingChannels);

            if (embBits === 8) {
                for (let grp = 0; grp < numU32; grp++) {
                    const ox = embOffsets ? embOffsets[grp * 2]     : 0;
                    const oy = embOffsets ? embOffsets[grp * 2 + 1] : 0;
                    const u  = Math.max(0, Math.min(1, uvx + ox));
                    const v  = Math.max(0, Math.min(1, uvy + oy));
                    const sx = u * (gW - 1), sy = v * (gH - 1);
                    const x0 = sx | 0, y0 = sy | 0;
                    const x1 = Math.min(x0 + 1, gW - 1), y1 = Math.min(y0 + 1, gH - 1);
                    let tx = sx - x0, ty = sy - y0;
                    if (smoothInterpolation) { tx = tx*tx*(3-2*tx); ty = ty*ty*(3-2*ty); }

                    const c00 = unpack4x8snorm(emb_q[(y0*gW+x0)*numU32+grp]);
                    const c10 = unpack4x8snorm(emb_q[(y0*gW+x1)*numU32+grp]);
                    const c01 = unpack4x8snorm(emb_q[(y1*gW+x0)*numU32+grp]);
                    const c11 = unpack4x8snorm(emb_q[(y1*gW+x1)*numU32+grp]);

                    for (let b = 0; b < 4; b++) {
                        const ch     = grp * 4 + b;
                        const interp = c00[b]*(1-tx)*(1-ty) + c10[b]*tx*(1-ty)
                                     + c01[b]*(1-tx)*ty     + c11[b]*tx*ty;
                        const mn = range ? range[ch * 2]     : -1;
                        const mx = range ? range[ch * 2 + 1] :  1;
                        emb[ch] = (interp + 1) * 0.5 * (mx - mn) + mn;
                    }
                }
            } else { // 4-bit
                for (let grp = 0; grp < numU32; grp++) {
                    const ox = embOffsets ? embOffsets[grp * 2]     : 0;
                    const oy = embOffsets ? embOffsets[grp * 2 + 1] : 0;
                    const u  = Math.max(0, Math.min(1, uvx + ox));
                    const v  = Math.max(0, Math.min(1, uvy + oy));
                    const sx = u * (gW - 1), sy = v * (gH - 1);
                    const x0 = sx | 0, y0 = sy | 0;
                    const x1 = Math.min(x0 + 1, gW - 1), y1 = Math.min(y0 + 1, gH - 1);
                    let tx = sx - x0, ty = sy - y0;
                    if (smoothInterpolation) { tx = tx*tx*(3-2*tx); ty = ty*ty*(3-2*ty); }

                    const q00 = emb_q[(y0*gW+x0)*numU32+grp];
                    const q10 = emb_q[(y0*gW+x1)*numU32+grp];
                    const q01 = emb_q[(y1*gW+x0)*numU32+grp];
                    const q11 = emb_q[(y1*gW+x1)*numU32+grp];

                    for (let b = 0; b < 8; b++) {
                        const ch    = grp * 8 + b;
                        const shift = b * 4;
                        const s = dequant4((q00 >>> shift) & 0xF)*(1-tx)*(1-ty)
                                + dequant4((q10 >>> shift) & 0xF)*tx*(1-ty)
                                + dequant4((q01 >>> shift) & 0xF)*(1-tx)*ty
                                + dequant4((q11 >>> shift) & 0xF)*tx*ty;
                        const mn = range ? range[ch * 2]     : -1;
                        const mx = range ? range[ch * 2 + 1] :  1;
                        emb[ch] = (s + 1) * 0.5 * (mx - mn) + mn;
                    }
                }
            }

            // --- Layer 1: embeddingChannels → mlpWidth1 ---
            const l1 = new Float32Array(mlpWidth1);
            for (let i = 0; i < mlpWidth1; i++) {
                let sum = weights.layer1_biases[i];
                const row = i * embeddingChannels;
                for (let j = 0; j < embeddingChannels; j++) {
                    const w = useQat ? qat8(weights.layer1_weights[row + j]) : weights.layer1_weights[row + j];
                    sum += w * emb[j];
                }
                interLayer1[pidx * mlpWidth1 + i] = sum;
                l1[i] = activ(sum);
            }

            // --- Layer 2: mlpWidth1 → mlpWidth2 ---
            const l2 = new Float32Array(mlpWidth2);
            for (let i = 0; i < mlpWidth2; i++) {
                let sum = weights.layer2_biases[i];
                const row = i * mlpWidth1;
                for (let j = 0; j < mlpWidth1; j++) {
                    const w = useQat ? qat8(weights.layer2_weights[row + j]) : weights.layer2_weights[row + j];
                    sum += w * l1[j];
                }
                interLayer2[pidx * mlpWidth2 + i] = sum;
                l2[i] = activ(sum);
            }

            // --- Layer 3: mlpWidth2 → outCh ---
            for (let i = 0; i < outCh; i++) {
                let sum = weights.layer3_biases[i];
                const row = i * mlpWidth2;
                for (let j = 0; j < mlpWidth2; j++) {
                    const w = useQat ? qat8(weights.layer3_weights[row + j]) : weights.layer3_weights[row + j];
                    sum += w * l2[j];
                }
                final[pidx * 4 + i] = Math.max(0, Math.min(1, sum));
            }
            if (!hasAlpha) final[pidx * 4 + 3] = 1.0;
        }
    }
    return { final, interLayer1, interLayer2 };
}

// --- Backward pass ---
export function backward(config, model, outputs, targetImage, weights, stride = 1) {
    const { gW, gH, embeddingChannels, mlpWidth1, mlpWidth2,
            activation = 'sin' } = config;
    const { f: activ_f, df: activ_prime } = makeActiv(activation);
    const pixelCount   = outputs.final.length / 4;
    const sampledCount = Math.ceil(pixelCount / stride);

    const hasAlpha = config.hasAlpha || false;
    const outCh    = hasAlpha ? 4 : 3;
    const invSampled = 1 / sampledCount;
    const lumaW = [0.299 * 2 * invSampled, 0.587 * 2 * invSampled,
                   0.114 * 2 * invSampled, (hasAlpha ? 1.0 : 0.0) * 2 * invSampled];

    // 1. Output gradient (luminance-weighted, alpha zeroed when !hasAlpha)
    const grad_final = new Float32Array(outputs.final.length);
    for (let p = 0; p < pixelCount; p += stride) {
        for (let c = 0; c < 4; c++) {
            grad_final[p * 4 + c] = lumaW[c] * (outputs.final[p * 4 + c] - targetImage[p * 4 + c]);
        }
    }

    // 2. Layer 3 backward
    const grad_l3_weights    = new Float32Array(weights.layer3_weights.length);
    const grad_l3_biases     = new Float32Array(weights.layer3_biases.length);
    const grad_inter2_output = new Float32Array(outputs.interLayer2.length);

    const l2_activated = new Float32Array(outputs.interLayer2.length);
    for (let i = 0; i < l2_activated.length; i++) {
        l2_activated[i] = activ_f(outputs.interLayer2[i]);
    }

    for (let p = 0; p < pixelCount; p += stride) {
        for (let i = 0; i < outCh; i++) {
            const g = grad_final[p * 4 + i];
            for (let j = 0; j < mlpWidth2; j++) {
                grad_l3_weights[i * mlpWidth2 + j] += g * l2_activated[p * mlpWidth2 + j];
                grad_inter2_output[p * mlpWidth2 + j] += g * weights.layer3_weights[i * mlpWidth2 + j];
            }
            grad_l3_biases[i] += g;
        }
    }

    // 3. Through L2 activation
    const grad_inter2_input = new Float32Array(outputs.interLayer2.length);
    for (let i = 0; i < grad_inter2_input.length; i++) {
        grad_inter2_input[i] = grad_inter2_output[i] * activ_prime(outputs.interLayer2[i]);
    }

    // 4. Layer 2 backward
    const grad_l2_weights    = new Float32Array(weights.layer2_weights.length);
    const grad_l2_biases     = new Float32Array(weights.layer2_biases.length);
    const grad_inter1_output = new Float32Array(outputs.interLayer1.length);

    const l1_activated = new Float32Array(outputs.interLayer1.length);
    for (let i = 0; i < l1_activated.length; i++) {
        l1_activated[i] = activ_f(outputs.interLayer1[i]);
    }

    for (let p = 0; p < pixelCount; p += stride) {
        for (let i = 0; i < mlpWidth2; i++) {
            const g = grad_inter2_input[p * mlpWidth2 + i];
            for (let j = 0; j < mlpWidth1; j++) {
                grad_l2_weights[i * mlpWidth1 + j] += g * l1_activated[p * mlpWidth1 + j];
                grad_inter1_output[p * mlpWidth1 + j] += g * weights.layer2_weights[i * mlpWidth1 + j];
            }
            grad_l2_biases[i] += g;
        }
    }

    // 5. Through L1 activation
    const grad_inter1_input = new Float32Array(outputs.interLayer1.length);
    for (let i = 0; i < grad_inter1_input.length; i++) {
        grad_inter1_input[i] = grad_inter1_output[i] * activ_prime(outputs.interLayer1[i]);
    }

    // 6. Layer 1 backward + embedding gradient
    const grad_l1_weights = new Float32Array(weights.layer1_weights.length);
    const grad_l1_biases  = new Float32Array(weights.layer1_biases.length);
    const grad_embeddings = new Float32Array(weights.embeddings.length);

    const embBits  = config.embBits || 8;
    const chPerGrp = 32 / embBits;
    const numU32   = embeddingChannels / chPerGrp;
    const embOffsets = config.embOffsets;
    const invBW = 1 / (config.width  - 1);
    const invBH = 1 / (config.height - 1);
    for (let p = 0; p < pixelCount; p += stride) {
        const puvx = (p % config.width) * invBW;
        const puvy = (p / config.width | 0) * invBH;

        for (let i = 0; i < mlpWidth1; i++) {
            grad_l1_biases[i] += grad_inter1_input[p * mlpWidth1 + i];
        }

        for (let grp = 0; grp < numU32; grp++) {
            const ox = embOffsets ? embOffsets[grp * 2]     : 0;
            const oy = embOffsets ? embOffsets[grp * 2 + 1] : 0;
            const u  = Math.max(0, Math.min(1, puvx + ox));
            const v  = Math.max(0, Math.min(1, puvy + oy));
            const sx = u * (gW - 1), sy = v * (gH - 1);
            const x0 = sx | 0, y0 = sy | 0;
            const x1 = Math.min(x0 + 1, gW - 1), y1 = Math.min(y0 + 1, gH - 1);
            let tx = sx - x0, ty = sy - y0;
            if (config.smoothInterpolation) { tx = tx*tx*(3-2*tx); ty = ty*ty*(3-2*ty); }
            const w00 = (1-tx)*(1-ty), w10 = tx*(1-ty), w01 = (1-tx)*ty, w11 = tx*ty;
            const idx00 = (y0*gW+x0)*embeddingChannels;
            const idx10 = (y0*gW+x1)*embeddingChannels;
            const idx01 = (y1*gW+x0)*embeddingChannels;
            const idx11 = (y1*gW+x1)*embeddingChannels;

            for (let b = 0; b < chPerGrp; b++) {
                const j = grp * chPerGrp + b;
                const emb_j = w00*weights.embeddings[idx00+j] + w10*weights.embeddings[idx10+j]
                            + w01*weights.embeddings[idx01+j] + w11*weights.embeddings[idx11+j];
                let ge = 0;
                for (let i = 0; i < mlpWidth1; i++) {
                    const g = grad_inter1_input[p * mlpWidth1 + i];
                    grad_l1_weights[i * embeddingChannels + j] += g * emb_j;
                    ge += g * weights.layer1_weights[i * embeddingChannels + j];
                }
                grad_embeddings[idx00+j] += w00 * ge;
                grad_embeddings[idx10+j] += w10 * ge;
                grad_embeddings[idx01+j] += w01 * ge;
                grad_embeddings[idx11+j] += w11 * ge;
            }
        }
    }

    // L2 regularisation on embeddings
    const embN = weights.embeddings.length;
    const l2Scale = 0.002 / embN;
    for (let i = 0; i < embN; i++) {
        grad_embeddings[i] += l2Scale * weights.embeddings[i];
    }

    return {
        layer1_weights: grad_l1_weights, layer1_biases: grad_l1_biases,
        layer2_weights: grad_l2_weights, layer2_biases: grad_l2_biases,
        layer3_weights: grad_l3_weights, layer3_biases: grad_l3_biases,
        embeddings: grad_embeddings,
    };
}

export class AdamOptimizer {
    constructor(parameters, learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, learningRateMap = null) {
        this.parameters   = parameters;
        this.learningRate = learningRate;
        this.beta1        = beta1;
        this.beta2        = beta2;
        this.epsilon      = epsilon;
        this.learningRateMap = learningRateMap;
        this.m = {};
        this.v = {};
        this.t = 0;
        for (const key in parameters) {
            this.m[key] = new Float32Array(parameters[key].length);
            this.v[key] = new Float32Array(parameters[key].length);
        }
    }

    _getLR(key) {
        if (this.learningRateMap) {
            if (this.learningRateMap[key]     !== undefined) return this.learningRateMap[key];
            if (this.learningRateMap.default  !== undefined) return this.learningRateMap.default;
        }
        return this.learningRate;
    }

    step(gradients) {
        this.t++;
        const invBc1 = 1 / (1 - Math.pow(this.beta1, this.t));
        const invBc2 = 1 / (1 - Math.pow(this.beta2, this.t));
        for (const key in this.parameters) {
            const param = this.parameters[key];
            const grad  = gradients[key];
            if (!grad) continue;
            const lr = this._getLR(key);
            for (let i = 0; i < param.length; i++) {
                this.m[key][i] = this.beta1 * this.m[key][i] + (1 - this.beta1) * grad[i];
                this.v[key][i] = this.beta2 * this.v[key][i] + (1 - this.beta2) * grad[i] * grad[i];
                param[i] -= lr * (this.m[key][i] * invBc1) / (Math.sqrt(this.v[key][i] * invBc2) + this.epsilon);
            }
        }
    }
}

export class CpuTrainer {
    constructor({ config, targetPixels, getHyperparams, onStep, onStop }) {
        this._config         = config;
        this._target         = targetPixels;
        this._getHyperparams = getHyperparams;
        this._onStep         = onStep;
        this._onStop         = onStop;
        this.type            = 'cpu';
        this.active          = false;
        this.lastWeights     = null;
        this._w              = null;
        this._adam           = null;
        this._stepCount      = 0;
        this._lossHistory    = [];
        this._lastTime       = 0;
        this._rafId          = null;
    }

    // freshWeights non-null → full reset; null → resume from current state
    start(freshWeights) {
        if (freshWeights) {
            this._w = {};
            for (const k of Object.keys(freshWeights))
                this._w[k] = new Float32Array(freshWeights[k]);
            this._stepCount   = 0;
            this._lossHistory = [];
            this._adam        = null;
        }
        if (!this._adam) this._adam = new AdamOptimizer(this._w, 0.001);
        this.active    = true;
        this._lastTime = 0;
        this._rafId    = requestAnimationFrame(() => this._loop());
    }

    stop()       { this.active = false; cancelAnimationFrame(this._rafId); this._rafId = null; this._onStop(); }
    async waitForIdle() {}
    destroy()    { this.active = false; cancelAnimationFrame(this._rafId); this._rafId = null; }
    getWeights() { return this._w; }

    _loop() {
        if (!this.active) return;
        const hp  = this._getHyperparams();
        const cfg = this._config;
        const embCh = cfg.embeddingChannels, embBits = cfg.embBits || 8;

        this._adam.learningRateMap = { embeddings: hp.embedLr, default: hp.mlpLr };

        const range        = computeEmbRange(this._w.embeddings, embCh, cfg.gW * cfg.gH);
        const embeddings_q = cpuPackEmbeddings(this._w.embeddings, embCh, range, embBits);
        const outputs      = forward(cfg, { ...this._w, embeddings_q }, range);

        // Luminance-weighted MSE loss
        const outCh = cfg.hasAlpha ? 4 : 3;
        const luma  = [0.299, 0.587, 0.114, 1.0];
        const n     = cfg.width * cfg.height;
        let loss = 0;
        for (let p = 0; p < n; p++)
            for (let c = 0; c < outCh; c++) {
                const d = outputs.final[p*4+c] - this._target[p*4+c];
                loss += luma[c] * d * d;
            }
        loss /= n;

        const grads = backward(cfg, null, outputs, this._target, this._w, hp.stride || 1);
        this._adam.step(grads);

        this._stepCount++;
        this._lossHistory.push(loss);
        const now  = performance.now();
        const rate = this._lastTime > 0 ? (1000 / (now - this._lastTime)).toFixed(1) : '—';
        this._lastTime = now;

        this.lastWeights = {
            embeddings:     this._w.embeddings,
            layer1_weights: this._w.layer1_weights,
            layer2_weights: this._w.layer2_weights,
            layer3_weights: this._w.layer3_weights,
            finalOutput:    outputs.final,
        };

        const doViz = (this._stepCount % (hp.vizInterval || 10) === 0);
        this._onStep({
            loss, step: this._stepCount, rate,
            lastWeights: this.lastWeights,
            inter1: doViz ? outputs.interLayer1 : null,
            inter2: doViz ? outputs.interLayer2 : null,
            lossHistory: this._lossHistory,
        });

        if (hp.maxIter > 0 && this._stepCount >= hp.maxIter) {
            this.active = false;
            this._onStop();
            return;
        }
        this._rafId = requestAnimationFrame(() => this._loop());
    }
}
