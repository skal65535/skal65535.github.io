// optimizer.js
// CPU-side Adam optimizer (reference/fallback; hot path runs on GPU via backward_builder.js).
export class AdamOptimizer {
    constructor(parameters, learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, learningRateMap = null) {
        this.parameters = parameters;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.learningRateMap = learningRateMap; // e.g. { embeddings: 0.01, default: 0.001 }

        this.m = {}; // First moment vector
        this.v = {}; // Second moment vector
        this.t = 0;   // Timestep

        // Initialize moment vectors
        for (const key in parameters) {
            this.m[key] = new Float32Array(parameters[key].length).fill(0);
            this.v[key] = new Float32Array(parameters[key].length).fill(0);
        }
    }

    _getLR(key) {
        if (this.learningRateMap) {
            if (this.learningRateMap[key] !== undefined) return this.learningRateMap[key];
            if (this.learningRateMap.default !== undefined) return this.learningRateMap.default;
        }
        return this.learningRate;
    }

    step(gradients) {
        this.t++;
        const bc1 = 1 - Math.pow(this.beta1, this.t);
        const bc2 = 1 - Math.pow(this.beta2, this.t);
        for (const key in this.parameters) {
            const param = this.parameters[key];
            const grad  = gradients[key];
            if (!grad) continue;
            const lr = this._getLR(key);
            for (let i = 0; i < param.length; i++) {
                this.m[key][i] = this.beta1 * this.m[key][i] + (1 - this.beta1) * grad[i];
                this.v[key][i] = this.beta2 * this.v[key][i] + (1 - this.beta2) * (grad[i] * grad[i]);
                const m_hat = this.m[key][i] / bc1;
                const v_hat = this.v[key][i] / bc2;
                param[i] -= lr * m_hat / (Math.sqrt(v_hat) + this.epsilon);
            }
        }
    }
}


/**
 * Computes the gradients of the loss with respect to the model parameters.
 * @param {object} config - The network configuration.
 * @param {object} model - The model parameters (GPU buffers).
 * @param {object} outputs - The intermediate outputs from the forward pass.
 * @param {Float32Array} targetImage - The target pixel data.
 * @returns {object} - An object containing the gradients for each parameter.
 */
export function backward(config, model, outputs, targetImage, weights, stride = 1) {
    const { gridSize, embeddingChannels, mlpWidth1, mlpWidth2 } = config;
    const pixelCount = outputs.final.length / 4;
    const sampledCount = Math.ceil(pixelCount / stride);

    // --- 1. Gradient of the loss with respect to the final output ---
    const grad_final = new Float32Array(outputs.final.length);
    for (let p = 0; p < pixelCount; p += stride) {
        for (let c = 0; c < 4; c++) {
            grad_final[p * 4 + c] = 2.0 * (outputs.final[p * 4 + c] - targetImage[p * 4 + c]) / sampledCount;
        }
    }

    // --- 2. Backprop through Layer 3 ---
    const grad_l3_weights = new Float32Array(weights.layer3_weights.length).fill(0);
    const grad_l3_biases = new Float32Array(weights.layer3_biases.length).fill(0);
    const grad_inter2_output = new Float32Array(outputs.interLayer2.length).fill(0);

    // Input to L3 is activ(interLayer2); interLayer2 stride is mlpWidth2
    const l2_output_activated = new Float32Array(outputs.interLayer2.length);
    for (let i = 0; i < l2_output_activated.length; i++) {
        l2_output_activated[i] = Math.sin(outputs.interLayer2[i]);
    }

    for (let p = 0; p < pixelCount; p += stride) {
        for (let i = 0; i < 4; i++) { // Output channels (RGBA)
            const grad_out_p = grad_final[p * 4 + i];
            for (let j = 0; j < mlpWidth2; j++) {
                grad_l3_weights[i * mlpWidth2 + j] += grad_out_p * l2_output_activated[p * mlpWidth2 + j];
                grad_inter2_output[p * mlpWidth2 + j] += grad_out_p * weights.layer3_weights[i * mlpWidth2 + j];
            }
            grad_l3_biases[i] += grad_out_p;
        }
    }

    // --- 3. Backprop through Layer 2's sine activation ---
    const grad_inter2_input = new Float32Array(outputs.interLayer2.length);
    for (let i = 0; i < grad_inter2_input.length; i++) {
        grad_inter2_input[i] = grad_inter2_output[i] * Math.cos(outputs.interLayer2[i]);
    }

    // --- 4. Backprop through Layer 2 ---
    // L2 weights: [mlpWidth2 × mlpWidth1], interLayer1 stride = mlpWidth1
    const grad_l2_weights = new Float32Array(weights.layer2_weights.length).fill(0);
    const grad_l2_biases = new Float32Array(weights.layer2_biases.length).fill(0);
    const grad_inter1_output = new Float32Array(outputs.interLayer1.length).fill(0);

    const l1_output_activated = new Float32Array(outputs.interLayer1.length);
    for (let i = 0; i < l1_output_activated.length; i++) {
        l1_output_activated[i] = Math.sin(outputs.interLayer1[i]);
    }

    for (let p = 0; p < pixelCount; p += stride) {
        for (let i = 0; i < mlpWidth2; i++) {
            const grad_l2_in_p = grad_inter2_input[p * mlpWidth2 + i];
            for (let j = 0; j < mlpWidth1; j++) {
                grad_l2_weights[i * mlpWidth1 + j] += grad_l2_in_p * l1_output_activated[p * mlpWidth1 + j];
                grad_inter1_output[p * mlpWidth1 + j] += grad_l2_in_p * weights.layer2_weights[i * mlpWidth1 + j];
            }
            grad_l2_biases[i] += grad_l2_in_p;
        }
    }

    // --- 5. Backprop through Layer 1's sine activation ---
    const grad_inter1_input = new Float32Array(outputs.interLayer1.length);
    for (let i = 0; i < grad_inter1_input.length; i++) {
        grad_inter1_input[i] = grad_inter1_output[i] * Math.cos(outputs.interLayer1[i]);
    }

    // --- 6. Backprop through Layer 1 ---
    // L1 weights: [mlpWidth1 × embeddingChannels], interLayer1 stride = mlpWidth1
    const grad_l1_weights = new Float32Array(weights.layer1_weights.length).fill(0);
    const grad_l1_biases = new Float32Array(weights.layer1_biases.length).fill(0);
    const grad_embeddings = new Float32Array(weights.embeddings.length).fill(0);

    for (let p = 0; p < pixelCount; p += stride) {
        const px = (p % config.width) / (config.width - 1);
        const py = Math.floor(p / config.width) / (config.height - 1);

        const scaledX = px * (gridSize - 1);
        const scaledY = py * (gridSize - 1);
        const x0 = Math.floor(scaledX);
        const y0 = Math.floor(scaledY);
        const x1 = Math.min(x0 + 1, gridSize - 1);
        const y1 = Math.min(y0 + 1, gridSize - 1);
        let tx = scaledX - x0;
        let ty = scaledY - y0;
        if (config.smoothInterpolation) {
            tx = tx * tx * (3 - 2 * tx);
            ty = ty * ty * (3 - 2 * ty);
        }

        const w00 = (1 - tx) * (1 - ty);
        const w10 = tx * (1 - ty);
        const w01 = (1 - tx) * ty;
        const w11 = tx * ty;

        const idx00 = (y0 * gridSize + x0) * embeddingChannels;
        const idx10 = (y0 * gridSize + x1) * embeddingChannels;
        const idx01 = (y1 * gridSize + x0) * embeddingChannels;
        const idx11 = (y1 * gridSize + x1) * embeddingChannels;

        const interp = new Float32Array(embeddingChannels);
        for (let j = 0; j < embeddingChannels; j++) {
            interp[j] = w00 * weights.embeddings[idx00 + j]
                      + w10 * weights.embeddings[idx10 + j]
                      + w01 * weights.embeddings[idx01 + j]
                      + w11 * weights.embeddings[idx11 + j];
        }

        for (let i = 0; i < mlpWidth1; i++) {
            const grad_l1_in_p = grad_inter1_input[p * mlpWidth1 + i];
            for (let j = 0; j < embeddingChannels; j++) {
                grad_l1_weights[i * embeddingChannels + j] += grad_l1_in_p * interp[j];
                const g = grad_l1_in_p * weights.layer1_weights[i * embeddingChannels + j];
                grad_embeddings[idx00 + j] += w00 * g;
                grad_embeddings[idx10 + j] += w10 * g;
                grad_embeddings[idx01 + j] += w01 * g;
                grad_embeddings[idx11 + j] += w11 * g;
            }
            grad_l1_biases[i] += grad_l1_in_p;
        }
    }


    // L2 regularisation on embeddings: grad of 0.001 * mean(emb^2) = 0.002 * emb / n
    const emb = weights.embeddings;
    const embN = emb.length;
    for (let i = 0; i < embN; i++) {
        grad_embeddings[i] += 0.002 * emb[i] / embN;
    }

    return {
        layer1_weights: grad_l1_weights,
        layer1_biases: grad_l1_biases,
        layer2_weights: grad_l2_weights,
        layer2_biases: grad_l2_biases,
        layer3_weights: grad_l3_weights,
        layer3_biases: grad_l3_biases,
        embeddings: grad_embeddings,
    };
}
