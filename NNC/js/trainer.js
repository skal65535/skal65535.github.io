// trainer.js
// GPU Trainer class: runs forward pass, backward pass, and Adam update each step.
import { buildBackwardShaders, FP_SCALE } from './backward_builder.js';
import { calculate_loss } from './loss.js';
import { ModelTensors } from './model.js';
import { computeEmbRange, normalizeEmbAndAdjustL1, uploadEmbRange, uploadEmbOffsets, uploadChannelMask, cpuPackEmbeddings, generateEmbOffsets } from './emb_utils.js';


const OFFSET_SAMPLE_INTERVAL   = 100;
const OFFSET_SAMPLE_CANDIDATES = 8;

export class Trainer {
    // webGpuContext, canvas, config, model, pipeline, bindGroup, fwdUniformsBuf,
    // outputBuffers, readbackBuffers, targetPixels, roiMask,
    // getHyperparams: () => { stride, mlpRatio, numLoops, embedLr, mlpLr, roiStrength, roiFreeze, maxIter, vizInterval }
    // onStep: ({ loss, step, rate, lastWeights, inter1, inter2, lossHistory }) => void
    // onStop: () => void
    constructor({ webGpuContext, canvas, config, model, pipeline, bindGroup, fwdUniformsBuf,
                  outputBuffers, readbackBuffers, targetPixels, roiMask,
                  getHyperparams, onStep, onStop }) {
        this._ctx          = webGpuContext;
        this._canvas       = canvas;
        this._config       = config;
        this._model        = model;
        this._pipeline     = pipeline;
        this._bindGroup    = bindGroup;
        this._fwdUniforms  = fwdUniformsBuf;
        this._outBufs      = outputBuffers;
        this._rbBufs       = readbackBuffers;
        this._target       = targetPixels;
        this._roiMask      = roiMask;
        this._getHyperparams = getHyperparams;
        this._onStep       = onStep;
        this._onStop       = onStop;

        this.active        = false;
        this.lastWeights   = null;

        this._rafId        = null;
        this._adamT        = 0;
        this._stepCount    = 0;
        this._lossHistory  = [];
        this._lastStepTime = 0;
        this._lastInter    = { inter1: null, inter2: null };

        this._bwdPipelines     = null;
        this._bwdBindGroups    = null;
        this._bwdBufs          = {};
        this._targetGpuBuf     = null;
        this._bwdUniformsBuf   = null;
        this._adamUniformsBufs = null;
    }

    // freshWeights: non-null = full reset (adamT=0); null = continue from lastWeights
    start(freshWeights) {
        const { device } = this._ctx;
        const { gridSize, embeddingChannels: embCh } = this._config;

        this._stepCount    = 0;
        this._lossHistory  = [];
        this._lastStepTime = 0;
        this._lastInter    = { inter1: null, inter2: null };

        const src = freshWeights !== null ? freshWeights.embeddings : this.lastWeights?.embeddings;
        if (freshWeights !== null) {
            this._adamT = 0;
            this.lastWeights = { embeddings: new Float32Array(freshWeights.embeddings) };
        }
        if (src) {
            const range = computeEmbRange(src, embCh, gridSize * gridSize);
            device.queue.writeBuffer(this._model.embeddings_range, 0, range);
            uploadEmbRange(range, embCh, this._fwdUniforms, device);
        }

        this._initBackward();
        this.active = true;
        this._rafId = requestAnimationFrame(() => this._train());
    }

    stop() {
        this.active = false;
        if (this._rafId !== null) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    }

    destroy() {
        this.stop();
        this._targetGpuBuf?.destroy();
        this._bwdUniformsBuf?.destroy();
        for (const v of Object.values(this._bwdBufs)) { if (v?.destroy) v.destroy(); }
        if (this._adamUniformsBufs) {
            for (const k of ModelTensors.KEYS) this._adamUniformsBufs[k]?.destroy();
        }
    }

    _initBackward() {
        const { device } = this._ctx;
        const { gridSize, embeddingChannels: embCh, mlpWidth } = this._config;
        const cv  = this._canvas;
        const m   = this._model;
        const ob  = this._outBufs;
        const ctx = this._ctx;
        const pixelCount = cv.width * cv.height;
        const stride     = mlpWidth * 4;
        const shaders    = buildBackwardShaders(this._config);

        const makePL = code => device.createComputePipeline({
            layout: 'auto',
            compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
        });
        const makeBG = (pl, bufs) => device.createBindGroup({
            layout: pl.getBindGroupLayout(0),
            entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });

        this._bwdPipelines = {
            gradOutput:     makePL(shaders.gradOutput),
            gradL3:         makePL(shaders.gradL3),
            gradL2:         makePL(shaders.gradL2),
            gradL1:         makePL(shaders.gradL1),
            adamStep:       makePL(shaders.adamStep),
            packEmbeddings: makePL(shaders.packEmbeddings),
        };
        const pl = this._bwdPipelines;
        const bb = this._bwdBufs;

        if (this._targetGpuBuf) this._targetGpuBuf.destroy();
        this._targetGpuBuf = ctx.zeroBuffer(this._target.length);
        device.queue.writeBuffer(this._targetGpuBuf, 0, this._target);

        if (this._bwdUniformsBuf) this._bwdUniformsBuf.destroy();
        this._bwdUniformsBuf = ctx.uniformBuffer(5 * 4);

        if (bb.roiMask?.destroy) bb.roiMask.destroy();
        bb.roiMask = ctx.zeroBuffer(pixelCount);

        if (bb.gradFinal?.destroy)        bb.gradFinal.destroy();
        if (bb.gradInter2Preact?.destroy)  bb.gradInter2Preact.destroy();
        if (bb.gradInter1Preact?.destroy)  bb.gradInter1Preact.destroy();
        bb.gradFinal        = ctx.storageBuffer(pixelCount * 4 * 4);
        bb.gradInter2Preact = ctx.storageBuffer(pixelCount * stride);
        bb.gradInter1Preact = ctx.storageBuffer(pixelCount * stride);

        if (this._adamUniformsBufs) {
            for (const k of ModelTensors.KEYS) this._adamUniformsBufs[k]?.destroy();
        }
        this._adamUniformsBufs = ModelTensors.create(() => ctx.uniformBuffer(10 * 4));
        const au = this._adamUniformsBufs;

        bb.adamAB = new ArrayBuffer(10 * 4);
        bb.adamF  = new Float32Array(bb.adamAB);
        bb.adamU  = new Uint32Array(bb.adamAB);
        bb.uniAB  = new ArrayBuffer(5 * 4);
        bb.uniU32 = new Uint32Array(bb.uniAB);
        bb.uniF32 = new Float32Array(bb.uniAB);
        bb.gradZero = {};
        for (const [k, buf] of Object.entries(m.gradAtomic)) {
            bb.gradZero[k] = new Int32Array(buf.size / 4);
        }

        const bu = this._bwdUniformsBuf;
        const tgt = this._targetGpuBuf;
        this._bwdBindGroups = {
            gradOutput: makeBG(pl.gradOutput, [bu, ob.final, tgt, bb.gradFinal, bb.roiMask]),
            gradL3:     makeBG(pl.gradL3, [
                bu, bb.gradFinal, ob.interLayer2, m.layer3.weights,
                bb.gradInter2Preact, m.gradAtomic.layer3_weights, m.gradAtomic.layer3_biases,
            ]),
            gradL2:     makeBG(pl.gradL2, [
                bu, bb.gradInter2Preact, ob.interLayer2, ob.interLayer1, m.layer2.weights,
                bb.gradInter1Preact, m.gradAtomic.layer2_weights, m.gradAtomic.layer2_biases,
            ]),
            gradL1:     makeBG(pl.gradL1, [
                bu, bb.gradInter1Preact, ob.interLayer1,
                m.layer1.weights, m.embeddings,
                m.gradAtomic.layer1_weights, m.gradAtomic.layer1_biases, m.gradAtomic.embeddings,
                m.emb_offsets,
            ]),
            adam: {
                embeddings:     makeBG(pl.adamStep, [au.embeddings,     m.gradAtomic.embeddings,     m.embeddings,     m.adamM.embeddings,     m.adamV.embeddings]),
                layer1_weights: makeBG(pl.adamStep, [au.layer1_weights, m.gradAtomic.layer1_weights, m.layer1.weights, m.adamM.layer1_weights, m.adamV.layer1_weights]),
                layer1_biases:  makeBG(pl.adamStep, [au.layer1_biases,  m.gradAtomic.layer1_biases,  m.layer1.biases,  m.adamM.layer1_biases,  m.adamV.layer1_biases]),
                layer2_weights: makeBG(pl.adamStep, [au.layer2_weights, m.gradAtomic.layer2_weights, m.layer2.weights, m.adamM.layer2_weights, m.adamV.layer2_weights]),
                layer2_biases:  makeBG(pl.adamStep, [au.layer2_biases,  m.gradAtomic.layer2_biases,  m.layer2.biases,  m.adamM.layer2_biases,  m.adamV.layer2_biases]),
                layer3_weights: makeBG(pl.adamStep, [au.layer3_weights, m.gradAtomic.layer3_weights, m.layer3.weights, m.adamM.layer3_weights, m.adamV.layer3_weights]),
                layer3_biases:  makeBG(pl.adamStep, [au.layer3_biases,  m.gradAtomic.layer3_biases,  m.layer3.biases,  m.adamM.layer3_biases,  m.adamV.layer3_biases]),
            },
            packEmbeddings: makeBG(pl.packEmbeddings, [m.embeddings, m.embeddings_q, m.embeddings_range]),
        };

        // Pack initial f32 embeddings → u32 before first forward pass
        const ce = device.createCommandEncoder();
        const p  = ce.beginComputePass();
        p.setPipeline(pl.packEmbeddings);
        p.setBindGroup(0, this._bwdBindGroups.packEmbeddings);
        p.dispatchWorkgroups(Math.ceil(gridSize * gridSize * embCh / 4 / 64));
        p.end();
        device.queue.submit([ce.finish()]);
    }

    async _train() {
        if (!this.active) return;

        const { device } = this._ctx;
        const { gridSize, embeddingChannels: embCh, mlpWidth, embBits } = this._config;
        const outCh = this._config.hasAlpha ? 4 : 3;
        const cv  = this._canvas;
        const m   = this._model;
        const ob  = this._outBufs;
        const rb  = this._rbBufs;
        const bb  = this._bwdBufs;
        const pl  = this._bwdPipelines;
        const bg  = this._bwdBindGroups;
        const au  = this._adamUniformsBufs;
        const pixelCount = cv.width * cv.height;
        const embSize    = gridSize * gridSize * embCh;

        const hp           = this._getHyperparams();
        const stride       = hp.stride;
        const sampledCount = Math.ceil(pixelCount / stride);
        this._adamT++;

        const period    = (hp.mlpRatio + 1) * hp.numLoops;
        const phasePos  = this._adamT % period;
        const activeMLP = phasePos >= hp.mlpRatio * hp.numLoops;
        const embedLR   = activeMLP ? 0 : hp.embedLr;
        const mlpLR     = activeMLP ? hp.mlpLr : 0;
        const tensorCfg = {
            embeddings:     { lr: embedLR, size: embSize,            l2: 0.002 / embSize, clamp: 1 },
            layer1_weights: { lr: mlpLR,   size: mlpWidth * embCh,   l2: 0, clamp: 0 },
            layer1_biases:  { lr: mlpLR,   size: mlpWidth,            l2: 0, clamp: 0 },
            layer2_weights: { lr: mlpLR,   size: mlpWidth * mlpWidth, l2: 0, clamp: 0 },
            layer2_biases:  { lr: mlpLR,   size: mlpWidth,            l2: 0, clamp: 0 },
            layer3_weights: { lr: mlpLR,   size: outCh * mlpWidth,    l2: 0, clamp: 0 },
            layer3_biases:  { lr: mlpLR,   size: outCh,               l2: 0, clamp: 0 },
        };

        if (!hp.roiFreeze) this._roiMask.decay(performance.now());
        if (this._roiMask.dirty) {
            device.queue.writeBuffer(bb.roiMask, 0, this._roiMask.weights);
            this._roiMask.dirty = false;
        }

        for (const [k, buf] of Object.entries(m.gradAtomic)) {
            device.queue.writeBuffer(buf, 0, bb.gradZero[k]);
        }
        const { adamAB, adamF, adamU, uniAB, uniU32, uniF32 } = bb;
        uniU32[0] = cv.width; uniU32[1] = cv.height;
        uniU32[2] = stride;   uniU32[3] = sampledCount;
        uniF32[4] = hp.roiStrength;
        device.queue.writeBuffer(this._bwdUniformsBuf, 0, uniAB);
        adamF[1] = 0.9; adamF[2] = 0.999; adamF[3] = 1e-8; adamF[6] = FP_SCALE;
        adamU[4] = this._adamT; adamU[9] = sampledCount;
        for (const [k, tc] of Object.entries(tensorCfg)) {
            adamF[0] = tc.lr; adamU[5] = tc.size; adamF[7] = tc.l2; adamU[8] = tc.clamp;
            device.queue.writeBuffer(au[k], 0, adamAB);
        }

        uploadChannelMask(0xFFFFFFFF, this._fwdUniforms, device);
        const ce = device.createCommandEncoder();

        const fwdPass = ce.beginComputePass();
        fwdPass.setPipeline(this._pipeline);
        fwdPass.setBindGroup(0, this._bindGroup);
        fwdPass.dispatchWorkgroups(Math.ceil(cv.width / 8), Math.ceil(cv.height / 8));
        fwdPass.end();

        // Separate compute pass per shader = implicit memory barrier between them
        const bwdDispatch = Math.ceil(pixelCount / 64);
        const runPass = (pipeline, bindGroup, dx, dy = 1) => {
            const p = ce.beginComputePass();
            p.setPipeline(pipeline); p.setBindGroup(0, bindGroup);
            p.dispatchWorkgroups(dx, dy); p.end();
        };
        runPass(pl.gradOutput, bg.gradOutput, bwdDispatch);
        runPass(pl.gradL3,     bg.gradL3,     bwdDispatch);
        runPass(pl.gradL2,     bg.gradL2,     bwdDispatch);
        runPass(pl.gradL1,     bg.gradL1,     Math.ceil(cv.width / 8), Math.ceil(cv.height / 8));

        const adamPass = ce.beginComputePass();
        for (const k of ModelTensors.KEYS) {
            adamPass.setPipeline(pl.adamStep);
            adamPass.setBindGroup(0, bg.adam[k]);
            adamPass.dispatchWorkgroups(Math.ceil(tensorCfg[k].size / 64));
        }
        adamPass.end();

        const doViz = this._stepCount % hp.vizInterval === 0;

        ce.copyBufferToBuffer(ob.final,         0, rb.final,         0, ob.final.size);
        ce.copyBufferToBuffer(m.embeddings,     0, rb.embeddings,    0, m.embeddings.size);
        ce.copyBufferToBuffer(m.layer1.weights, 0, rb.layer1Weights, 0, m.layer1.weights.size);
        ce.copyBufferToBuffer(m.layer1.biases,  0, rb.layer1Biases,  0, m.layer1.biases.size);
        ce.copyBufferToBuffer(m.layer2.weights, 0, rb.layer2Weights, 0, m.layer2.weights.size);
        ce.copyBufferToBuffer(m.layer3.weights, 0, rb.layer3Weights, 0, m.layer3.weights.size);
        if (doViz) {
            ce.copyBufferToBuffer(ob.interLayer1, 0, rb.interLayer1, 0, rb.interLayer1.size);
            ce.copyBufferToBuffer(ob.interLayer2, 0, rb.interLayer2, 0, rb.interLayer2.size);
        }
        device.queue.submit([ce.finish()]);

        const maps = [
            rb.final.mapAsync(GPUMapMode.READ),
            rb.embeddings.mapAsync(GPUMapMode.READ),
            rb.layer1Weights.mapAsync(GPUMapMode.READ),
            rb.layer1Biases.mapAsync(GPUMapMode.READ),
            rb.layer2Weights.mapAsync(GPUMapMode.READ),
            rb.layer3Weights.mapAsync(GPUMapMode.READ),
        ];
        if (doViz) {
            maps.push(rb.interLayer1.mapAsync(GPUMapMode.READ));
            maps.push(rb.interLayer2.mapAsync(GPUMapMode.READ));
        }
        await Promise.all(maps);

        const finalData = new Float32Array(rb.final.getMappedRange());
        const embData   = new Float32Array(rb.embeddings.getMappedRange()).slice();
        const l1wData   = new Float32Array(rb.layer1Weights.getMappedRange()).slice();
        const l1bData   = new Float32Array(rb.layer1Biases.getMappedRange()).slice();
        const l2wData   = new Float32Array(rb.layer2Weights.getMappedRange()).slice();
        const l3wData   = new Float32Array(rb.layer3Weights.getMappedRange()).slice();
        const loss      = calculate_loss(finalData, this._target);
        const finalSlice = finalData.slice();

        rb.final.unmap(); rb.embeddings.unmap();
        rb.layer1Weights.unmap(); rb.layer1Biases.unmap();
        rb.layer2Weights.unmap(); rb.layer3Weights.unmap();

        let inter1 = null, inter2 = null;
        if (doViz) {
            inter1 = new Float32Array(rb.interLayer1.getMappedRange()).slice();
            inter2 = new Float32Array(rb.interLayer2.getMappedRange()).slice();
            rb.interLayer1.unmap(); rb.interLayer2.unmap();
            this._lastInter = { inter1, inter2 };
        }

        // Normalize embeddings to [-1,1] per channel; absorb scale+center into L1
        normalizeEmbAndAdjustL1(embData, l1wData, l1bData, embCh, mlpWidth);
        device.queue.writeBuffer(m.embeddings,     0, embData);
        device.queue.writeBuffer(m.layer1.weights, 0, l1wData);
        device.queue.writeBuffer(m.layer1.biases,  0, l1bData);

        // CPU pack with fixed [-1,1] range → upload to embeddings_q for next forward pass
        const identityRange = new Float32Array(embCh * 2);
        for (let c = 0; c < embCh; c++) { identityRange[c*2] = -1; identityRange[c*2+1] = 1; }
        device.queue.writeBuffer(m.embeddings_q, 0, cpuPackEmbeddings(embData, embCh, identityRange, embBits));
        // Guard against stale range from restart
        uploadEmbRange(null, embCh, this._fwdUniforms, device);

        this.lastWeights = {
            embeddings:    embData,
            layer1Weights: l1wData,
            layer2Weights: l2wData,
            layer3Weights: l3wData,
            finalOutput:   finalSlice,
        };
        this._stepCount++;
        this._lossHistory.push(loss);

        const now  = performance.now();
        const rate = this._lastStepTime > 0 ? (1000 / (now - this._lastStepTime)).toFixed(1) : '—';
        this._lastStepTime = now;

        this._onStep({
            loss, step: this._stepCount, rate,
            lastWeights: this.lastWeights,
            inter1: doViz ? inter1 : null,
            inter2: doViz ? inter2 : null,
            lossHistory: this._lossHistory,
        });

        if (hp.maxIter > 0 && this._stepCount >= hp.maxIter) {
            this.active = false;
            this._onStop();
            return;
        }
        if (this.active && this._config.embOffsets?.some(v => v !== 0) && this._stepCount % OFFSET_SAMPLE_INTERVAL === 0) {
            await this._sampleOffsets();
        }
        if (this.active) this._rafId = requestAnimationFrame(() => this._train());
    }

    async _sampleOffsets() {
        const { device } = this._ctx;
        const { gridSize, embeddingChannels: embCh, embBits } = this._config;
        const cv = this._canvas;
        const pixelCount = cv.width * cv.height;

        const evalOffsets = async (offsets) => {
            uploadEmbOffsets(offsets, this._fwdUniforms, device);
            const ce = device.createCommandEncoder();
            const p  = ce.beginComputePass();
            p.setPipeline(this._pipeline);
            p.setBindGroup(0, this._bindGroup);
            p.dispatchWorkgroups(Math.ceil(cv.width / 8), Math.ceil(cv.height / 8));
            p.end();
            const rb = device.createBuffer({ size: pixelCount * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            ce.copyBufferToBuffer(this._outBufs.final, 0, rb, 0, pixelCount * 16);
            device.queue.submit([ce.finish()]);
            await rb.mapAsync(GPUMapMode.READ);
            const loss = calculate_loss(new Float32Array(rb.getMappedRange()), this._target);
            rb.unmap(); rb.destroy();
            return loss;
        };

        const current  = this._config.embOffsets;
        let bestOffsets = current;
        let bestLoss    = await evalOffsets(current);

        for (let i = 0; i < OFFSET_SAMPLE_CANDIDATES; i++) {
            const candidate = generateEmbOffsets(embCh, embBits, gridSize, false);
            const loss = await evalOffsets(candidate);
            if (loss < bestLoss) { bestLoss = loss; bestOffsets = candidate; }
        }

        // Upload winner to both uniform (fwd) and storage (bwd gradL1)
        if (bestOffsets !== current) {
            this._config.embOffsets = bestOffsets;
        }
        uploadEmbOffsets(bestOffsets, this._fwdUniforms, device);
        device.queue.writeBuffer(this._model.emb_offsets, 0, bestOffsets);
    }
}
