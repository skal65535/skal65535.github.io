// trainer.js
// GPU Trainer class: runs forward pass, backward pass, and Adam update each step.
import { buildBackwardShaders, FP_SCALE } from './webgpu.js';
import { calculate_loss } from './loss.js';
import { computeEmbRange, normalizeEmbAndAdjustL1, uploadEmbRange, uploadEmbOffsets, uploadChannelMask, cpuPackEmbeddings, generateEmbOffsets, ModelTensors, buildFwdUniforms } from './model.js';



export class Trainer {
    // webGpuContext, canvas, config, model, pipeline, bindGroup, fwdUniformsBuf,
    // outputBuffers, readbackBuffers, targetPixels, roiMask,
    // getHyperparams: () => { stride, mlpRatio, numLoops, embedLr, mlpLr, roiStrength, roiFreeze, maxIter, vizInterval }
    // onStep: ({ loss, step, rate, lastWeights, inter1, inter2, lossHistory }) => void
    // onStop: () => void
    constructor({ webGpuContext, canvas, config, model, pipeline, bindGroup, fwdUniformsBuf,
                  outputBuffers, readbackBuffers, targetPixels, alphaCellMask, roiMask,
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
        this._alphaCellMask = alphaCellMask;
        this._roiMask      = roiMask;
        this._getHyperparams = getHyperparams;
        this._onStep       = onStep;
        this._onStop       = onStop;

        this.active        = false;
        this.type          = 'gpu';
        this.lastWeights   = null;

        this._rafId      = null;
        this._stepRunning = false;
        this._adamT  = 0;
        this._resetStats();

        this._bwdPipelines     = null;
        this._bwdBindGroups    = null;
        this._bwdBufs          = {};
        this._targetGpuBuf     = null;
        this._bwdUniformsBuf   = null;
        this._adamUniformsBufs = null;
        this._offsetRbBuf      = null;
    }

    // freshWeights: non-null = full reset (adamT=0); null = continue from lastWeights
    start(freshWeights) {
        const { device } = this._ctx;
        const { gW, gH, embeddingChannels: embCh } = this._config;

        this._resetStats();

        const src = freshWeights !== null ? freshWeights.embeddings : this.lastWeights?.embeddings;
        if (freshWeights !== null) {
            this._adamT = 0;
            this.lastWeights = { embeddings: new Float32Array(freshWeights.embeddings) };
        }
        if (src) {
            const range = computeEmbRange(src, embCh, gW * gH);
            this._ctx.writeBuffer(this._model.embeddings_range, range);
            uploadEmbRange(range, embCh, this._fwdUniforms, device);
        }

        this._initBackward();
        this.active = true;
        this._rafId = requestAnimationFrame(t => this._train(t));
    }

    stop() {
        this.active = false;
        if (this._rafId !== null) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    }

    async waitForIdle() {
        while (this._stepRunning) await new Promise(r => setTimeout(r, 4));
    }

    destroy() {
        this.stop();
        this._targetGpuBuf?.destroy();
        this._bwdUniformsBuf?.destroy();
        this._offsetRbBuf?.destroy();
        for (const v of Object.values(this._bwdBufs)) { if (v?.destroy) v.destroy(); }
        if (this._adamUniformsBufs) {
            for (const k of ModelTensors.KEYS) this._adamUniformsBufs[k]?.destroy();
        }
    }

    _resetStats() {
        this._stepCount     = 0;
        this._lossHistory   = [];
        this._lastStepTime  = 0;
        this._lastRafT      = 0;
        this._frameInterval = 16.67;  // ms; updated from rAF timestamps
        this._stepsPerFrame = 4;
        this._lastLoss       = 0;
        this._lastFinalSlice = null;
    }

    _initBackward() {
        const { device } = this._ctx;
        const { gW, gH, embeddingChannels: embCh, mlpWidth1, mlpWidth2 } = this._config;
        const cv  = this._canvas;
        const m   = this._model;
        const ob  = this._outBufs;
        const ctx = this._ctx;
        const pixelCount = cv.width * cv.height;
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
        ctx.writeBuffer(this._targetGpuBuf, this._target);

        if (this._bwdUniformsBuf) this._bwdUniformsBuf.destroy();
        this._bwdUniformsBuf = ctx.uniformBuffer(5 * 4);

        if (bb.roiMask?.destroy) bb.roiMask.destroy();
        bb.roiMask = ctx.zeroBuffer(pixelCount);

        if (bb.gradFinal?.destroy)        bb.gradFinal.destroy();
        if (bb.gradInter2Preact?.destroy)  bb.gradInter2Preact.destroy();
        if (bb.gradInter1Preact?.destroy)  bb.gradInter1Preact.destroy();
        bb.gradFinal        = ctx.storageBuffer(pixelCount * 4 * 4);
        bb.gradInter2Preact = ctx.storageBuffer(pixelCount * mlpWidth2 * 4);
        bb.gradInter1Preact = ctx.storageBuffer(pixelCount * mlpWidth1 * 4);

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

        this._offsetRbBuf?.destroy();
        this._offsetRbBuf = ctx.readbackBuffer(cv.width * cv.height * 4 * 4, 'rb/offsets');

        // Pack initial f32 embeddings → u32 before first forward pass
        const ce = device.createCommandEncoder();
        const p  = ce.beginComputePass();
        p.setPipeline(pl.packEmbeddings);
        p.setBindGroup(0, this._bwdBindGroups.packEmbeddings);
        p.dispatchWorkgroups(Math.ceil(gW * gH * embCh / 4 / 64));
        p.end();
        device.queue.submit([ce.finish()]);

        const embSize = gW * gH * embCh;
        const outCh = this._config.hasAlpha ? 4 : 3;
        this._tensorCfgs = {
            embeddings:     { lr: 0, size: embSize,               l2: 0.002 / embSize, clamp: 1 },
            layer1_weights: { lr: 0, size: mlpWidth1 * embCh,     l2: 0, clamp: 0 },
            layer1_biases:  { lr: 0, size: mlpWidth1,             l2: 0, clamp: 0 },
            layer2_weights: { lr: 0, size: mlpWidth2 * mlpWidth1, l2: 0, clamp: 0 },
            layer2_biases:  { lr: 0, size: mlpWidth2,             l2: 0, clamp: 0 },
            layer3_weights: { lr: 0, size: outCh * mlpWidth2,     l2: 0, clamp: 0 },
            layer3_biases:  { lr: 0, size: outCh,                 l2: 0, clamp: 0 },
        };
    }

    _syncMlpWeights(ce, m) {
        const { l1w, l1b, l2w, l2b, l3w, l3b } = m.mlpLayout;
        ce.copyBufferToBuffer(m.layer1.weights, 0, m.mlp_weights, l1w * 4, m.layer1.weights.size);
        ce.copyBufferToBuffer(m.layer1.biases,  0, m.mlp_weights, l1b * 4, m.layer1.biases.size);
        ce.copyBufferToBuffer(m.layer2.weights, 0, m.mlp_weights, l2w * 4, m.layer2.weights.size);
        ce.copyBufferToBuffer(m.layer2.biases,  0, m.mlp_weights, l2b * 4, m.layer2.biases.size);
        ce.copyBufferToBuffer(m.layer3.weights, 0, m.mlp_weights, l3w * 4, m.layer3.weights.size);
        ce.copyBufferToBuffer(m.layer3.biases,  0, m.mlp_weights, l3b * 4, m.layer3.biases.size);
    }


    _writeUniforms(hp, sampledCount) {
        const period    = (hp.mlpRatio + 1) * hp.numLoops;
        const activeMLP = (this._adamT % period) >= hp.mlpRatio * hp.numLoops;
        const mlpLR     = activeMLP ? hp.mlpLr : 0;

        const tCfg = this._tensorCfgs;
        tCfg.embeddings.lr     = activeMLP ? 0 : hp.embedLr;
        tCfg.layer1_weights.lr = mlpLR;
        tCfg.layer1_biases.lr  = mlpLR;
        tCfg.layer2_weights.lr = mlpLR;
        tCfg.layer2_biases.lr  = mlpLR;
        tCfg.layer3_weights.lr = mlpLR;
        tCfg.layer3_biases.lr  = mlpLR;

        const { adamAB: aAB, adamF: aF, adamU: aU, uniAB: uAB, uniU32: uU32, uniF32: uF32 } = this._bwdBufs;
        uU32[0] = this._canvas.width; uU32[1] = this._canvas.height;
        uU32[2] = hp.stride;   uU32[3] = sampledCount;
        uF32[4] = hp.roiStrength;
        this._ctx.writeBuffer(this._bwdUniformsBuf, uAB);

        aF[1] = 0.9; aF[2] = 0.999; aF[3] = 1e-8; aF[6] = FP_SCALE;
        aU[4] = this._adamT; aU[9] = sampledCount;
        for (const [k, tc] of Object.entries(tCfg)) {
            aF[0] = tc.lr; aU[5] = tc.size; aF[7] = tc.l2; aU[8] = tc.clamp;
            this._ctx.writeBuffer(this._adamUniformsBufs[k], aAB);
        }
    }

    async _train(rafT = 0) {
        if (!this.active) return;
        this._stepRunning = true;
        try {
            await this._doTrain(rafT);
        } finally {
            this._stepRunning = false;
        }
    }

    async _doTrain(rafT = 0) {
        if (!this.active) return;

        const { device } = this._ctx;
        const { gW, gH, embeddingChannels: embCh, mlpWidth1, mlpWidth2, embBits } = this._config;
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
        const embSize    = gW * gH * embCh;

        const hp           = this._getHyperparams();
        const stride       = hp.stride;
        const sampledCount = Math.ceil(pixelCount / stride);
        const stepsPerFrame = this._stepsPerFrame;

        // Fast inner steps: GPU-only, no readback, GPU pack instead of CPU normalize+pack
        for (let s = 0; s < stepsPerFrame - 1 && this.active; s++) {
            this._adamT++;
            if (!hp.roiFreeze) this._roiMask.decay(performance.now());
            if (this._roiMask.dirty) {
                this._ctx.writeBuffer(bb.roiMask, this._roiMask.weights);
                this._roiMask.dirty = false;
            }
            this._writeUniforms(hp, sampledCount);

            uploadChannelMask(0xFFFFFFFF, this._fwdUniforms, device);

            const ce2 = device.createCommandEncoder();
            for (const buf of Object.values(m.gradAtomic)) ce2.clearBuffer(buf, 0, buf.size);
            const fp2 = ce2.beginComputePass();
            fp2.setPipeline(this._pipeline); fp2.setBindGroup(0, this._bindGroup);
            fp2.dispatchWorkgroups(Math.ceil(cv.width / 8), Math.ceil(cv.height / 8));
            fp2.end();

            const bwdD2 = Math.ceil(pixelCount / 64);
            const runPass2 = (pipeline, bindGroup, dx, dy = 1) => {
                const p = ce2.beginComputePass();
                p.setPipeline(pipeline); p.setBindGroup(0, bindGroup);
                p.dispatchWorkgroups(dx, dy); p.end();
            };
            runPass2(pl.gradOutput, bg.gradOutput, bwdD2);
            runPass2(pl.gradL3,     bg.gradL3,     bwdD2);
            runPass2(pl.gradL2,     bg.gradL2,     bwdD2);
            runPass2(pl.gradL1,     bg.gradL1,     Math.ceil(cv.width / 8), Math.ceil(cv.height / 8));

            const ap2 = ce2.beginComputePass();
            for (const k of ModelTensors.KEYS) {
                ap2.setPipeline(pl.adamStep);
                ap2.setBindGroup(0, bg.adam[k]);
                ap2.dispatchWorkgroups(Math.ceil(this._tensorCfgs[k].size / 64));
            }
            ap2.end();
            this._syncMlpWeights(ce2, m);

            const pp2 = ce2.beginComputePass();
            pp2.setPipeline(pl.packEmbeddings);
            pp2.setBindGroup(0, bg.packEmbeddings);
            pp2.dispatchWorkgroups(Math.ceil(gW * gH * embCh / 4 / 64));
            pp2.end();

            device.queue.submit([ce2.finish()]);

            this._stepCount++;
            if (hp.maxIter > 0 && this._stepCount >= hp.maxIter) {
                this.active = false;
                this._onStop();
                return;
            }
        }

        this._adamT++;

        if (!hp.roiFreeze) this._roiMask.decay(performance.now());
        if (this._roiMask.dirty) {
            this._ctx.writeBuffer(bb.roiMask, this._roiMask.weights);
            this._roiMask.dirty = false;
        }

        this._writeUniforms(hp, sampledCount);

        uploadChannelMask(0xFFFFFFFF, this._fwdUniforms, device);
        const ce = device.createCommandEncoder();
        for (const buf of Object.values(m.gradAtomic)) ce.clearBuffer(buf, 0, buf.size);

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
            adamPass.dispatchWorkgroups(Math.ceil(this._tensorCfgs[k].size / 64));
        }
        adamPass.end();
        this._syncMlpWeights(ce, m);

        const doViz = this._stepCount % hp.vizInterval === 0;

        ce.copyBufferToBuffer(m.embeddings,     0, rb.embeddings,    0, m.embeddings.size);
        ce.copyBufferToBuffer(m.layer1.weights, 0, rb.layer1Weights, 0, m.layer1.weights.size);
        ce.copyBufferToBuffer(m.layer1.biases,  0, rb.layer1Biases,  0, m.layer1.biases.size);
        if (doViz) {
            ce.copyBufferToBuffer(ob.final,         0, rb.final,         0, ob.final.size);
            ce.copyBufferToBuffer(m.layer2.weights, 0, rb.layer2Weights, 0, m.layer2.weights.size);
            ce.copyBufferToBuffer(m.layer3.weights, 0, rb.layer3Weights, 0, m.layer3.weights.size);
            ce.copyBufferToBuffer(ob.interLayer1, 0, rb.interLayer1, 0, rb.interLayer1.size);
            ce.copyBufferToBuffer(ob.interLayer2, 0, rb.interLayer2, 0, rb.interLayer2.size);
        }
        device.queue.submit([ce.finish()]);

        const maps = [
            rb.embeddings.mapAsync(GPUMapMode.READ),
            rb.layer1Weights.mapAsync(GPUMapMode.READ),
            rb.layer1Biases.mapAsync(GPUMapMode.READ),
        ];
        if (doViz) {
            maps.push(rb.final.mapAsync(GPUMapMode.READ));
            maps.push(rb.layer2Weights.mapAsync(GPUMapMode.READ));
            maps.push(rb.layer3Weights.mapAsync(GPUMapMode.READ));
            maps.push(rb.interLayer1.mapAsync(GPUMapMode.READ));
            maps.push(rb.interLayer2.mapAsync(GPUMapMode.READ));
        }
        await Promise.all(maps);

        const embData = new Float32Array(rb.embeddings.getMappedRange()).slice();
        const l1wData = new Float32Array(rb.layer1Weights.getMappedRange()).slice();
        const l1bData = new Float32Array(rb.layer1Biases.getMappedRange()).slice();
        rb.embeddings.unmap(); rb.layer1Weights.unmap(); rb.layer1Biases.unmap();

        let loss = this._lastLoss, finalSlice = null;
        let l2wData = null, l3wData = null;
        let inter1 = null, inter2 = null;
        if (doViz) {
            const finalData = new Float32Array(rb.final.getMappedRange());
            loss = calculate_loss(finalData, this._target, outCh);
            finalSlice = finalData.slice();
            l2wData = new Float32Array(rb.layer2Weights.getMappedRange()).slice();
            l3wData = new Float32Array(rb.layer3Weights.getMappedRange()).slice();
            inter1  = new Float32Array(rb.interLayer1.getMappedRange()).slice();
            inter2  = new Float32Array(rb.interLayer2.getMappedRange()).slice();
            rb.final.unmap(); rb.layer2Weights.unmap(); rb.layer3Weights.unmap();
            rb.interLayer1.unmap(); rb.interLayer2.unmap();
            this._lastLoss       = loss;
            this._lastFinalSlice = finalSlice;
        }

        // Normalize embeddings to [-1,1] per channel; absorb scale+center into L1
        normalizeEmbAndAdjustL1(embData, l1wData, l1bData, embCh, mlpWidth1);
        if (this._alphaCellMask) {
            for (let cell = 0; cell < this._alphaCellMask.length; cell++) {
                if (this._alphaCellMask[cell] === 0) embData.fill(0, cell * embCh, (cell + 1) * embCh);
            }
        }
        this._ctx.writeBuffer(m.embeddings,     embData);
        this._ctx.writeBuffer(m.layer1.weights, l1wData);
        this._ctx.writeBuffer(m.layer1.biases,  l1bData);
        this._ctx.writeBufferAt(m.mlp_weights, m.mlpLayout.l1w * 4, l1wData);
        this._ctx.writeBufferAt(m.mlp_weights, m.mlpLayout.l1b * 4, l1bData);

        // CPU pack with fixed [-1,1] range → upload to embeddings_q for next forward pass
        const identityRange = new Float32Array(embCh * 2);
        for (let c = 0; c < embCh; c++) { identityRange[c*2] = -1; identityRange[c*2+1] = 1; }
        this._ctx.writeBuffer(m.embeddings_range, identityRange);
        this._ctx.writeBuffer(m.embeddings_q, cpuPackEmbeddings(embData, embCh, identityRange, embBits));
        // Guard against stale range from restart
        uploadEmbRange(null, embCh, this._fwdUniforms, device);

        this.lastWeights = {
            embeddings:     embData,
            layer1_weights: l1wData,
            layer2_weights: doViz ? l2wData : this.lastWeights?.layer2_weights ?? null,
            layer3_weights: doViz ? l3wData : this.lastWeights?.layer3_weights ?? null,
            finalOutput:    doViz ? finalSlice : this._lastFinalSlice,
        };
        this._stepCount++;
        this._lossHistory.push(loss);

        const now = performance.now();
        const rate = this._lastStepTime > 0 ? (stepsPerFrame * 1000 / (now - this._lastStepTime)).toFixed(1) : '—';
        if (this._lastStepTime > 0) {
            // track actual display refresh interval from rAF timestamps
            if (this._lastRafT > 0) {
                const dt = rafT - this._lastRafT;
                if (dt > 1) this._frameInterval = 0.9 * this._frameInterval + 0.1 * dt;
            }
            const elapsed   = now - this._lastStepTime;
            const msPerStep = elapsed / stepsPerFrame;
            const next = Math.max(1, Math.round(this._frameInterval / msPerStep));
            this._stepsPerFrame = next;
        }
        this._lastRafT     = rafT;
        this._lastStepTime = now;

        this._onStep({
            loss, step: this._stepCount, rate,
            lastWeights: this.lastWeights,
            inter1, inter2,
            lossHistory: this._lossHistory,
        });

        if (hp.maxIter > 0 && this._stepCount >= hp.maxIter) {
            this.active = false;
            this._onStop();
            return;
        }
        if (this.active && this._config.embOffsets?.some(v => v !== 0) && hp.offsetSampleInterval > 0 && this._stepCount % hp.offsetSampleInterval === 0) {
            await this._sampleOffsets();
        }
        if (this.active) this._rafId = requestAnimationFrame(t => this._train(t));
    }

    async _sampleOffsets() {
        const { device } = this._ctx;
        const { gW, gH, embeddingChannels: embCh, embBits } = this._config;
        const outCh = this._config.hasAlpha ? 4 : 3;
        const cv = this._canvas;
        const pixelCount = cv.width * cv.height;
        const rb = this._offsetRbBuf;

        const evalOffsets = async (offsets) => {
            uploadEmbOffsets(offsets, this._fwdUniforms, device);
            const ce = device.createCommandEncoder();
            const p  = ce.beginComputePass();
            p.setPipeline(this._pipeline);
            p.setBindGroup(0, this._bindGroup);
            p.dispatchWorkgroups(Math.ceil(cv.width / 8), Math.ceil(cv.height / 8));
            p.end();
            ce.copyBufferToBuffer(this._outBufs.final, 0, rb, 0, pixelCount * 4 * 4);
            device.queue.submit([ce.finish()]);
            await rb.mapAsync(GPUMapMode.READ);
            const loss = calculate_loss(new Float32Array(rb.getMappedRange()), this._target, outCh);
            rb.unmap();
            return loss;
        };

        const numCandidates = 8;
        const current  = this._config.embOffsets;
        let bestOffsets = current;
        let bestLoss    = await evalOffsets(current);

        for (let i = 0; i < numCandidates; i++) {
            const candidate = generateEmbOffsets(embCh, embBits, gW, gH, false);
            const loss = await evalOffsets(candidate);
            if (loss < bestLoss) { bestLoss = loss; bestOffsets = candidate; }
        }

        // Upload winner to both uniform (fwd) and storage (bwd gradL1)
        if (bestOffsets !== current) {
            this._config.embOffsets = bestOffsets;
        }
        uploadEmbOffsets(bestOffsets, this._fwdUniforms, device);
        this._ctx.writeBuffer(this._model.emb_offsets, bestOffsets);
    }
}

export async function runInferencePass({ webGpuContext, config, model, pipeline, bindGroup, fwdUniformsBuf, outputBuffers, readbackBuffers, channelMask, canvasWidth, canvasHeight, lastWeights }) {
    const { device } = webGpuContext;
    const { gW, gH, embeddingChannels: embCh, embBits, embOffsets } = config;

    const range = computeEmbRange(lastWeights.embeddings, embCh, gW * gH);
    webGpuContext.writeBuffer(model.embeddings_range, range);
    uploadEmbRange(range, embCh, fwdUniformsBuf, device);
    const packed = cpuPackEmbeddings(lastWeights.embeddings, embCh, range, embBits);
    webGpuContext.writeBuffer(model.embeddings_q, packed);

    uploadChannelMask(channelMask, fwdUniformsBuf, device);
    const ce = device.createCommandEncoder({ label: 'inference' });
    const fwdPass = ce.beginComputePass({ label: 'fwd' });
    fwdPass.setPipeline(pipeline);
    fwdPass.setBindGroup(0, bindGroup);
    fwdPass.dispatchWorkgroups(Math.ceil(canvasWidth / 8), Math.ceil(canvasHeight / 8));
    fwdPass.end();
    ce.copyBufferToBuffer(outputBuffers.final,       0, readbackBuffers.final,       0, outputBuffers.final.size);
    ce.copyBufferToBuffer(outputBuffers.interLayer1, 0, readbackBuffers.interLayer1, 0, readbackBuffers.interLayer1.size);
    ce.copyBufferToBuffer(outputBuffers.interLayer2, 0, readbackBuffers.interLayer2, 0, readbackBuffers.interLayer2.size);
    device.queue.submit([ce.finish()]);

    await Promise.all([
        readbackBuffers.final.mapAsync(GPUMapMode.READ),
        readbackBuffers.interLayer1.mapAsync(GPUMapMode.READ),
        readbackBuffers.interLayer2.mapAsync(GPUMapMode.READ),
    ]);
    const final  = new Float32Array(readbackBuffers.final.getMappedRange()).slice();
    const inter1 = new Float32Array(readbackBuffers.interLayer1.getMappedRange()).slice();
    const inter2 = new Float32Array(readbackBuffers.interLayer2.getMappedRange()).slice();
    readbackBuffers.final.unmap();
    readbackBuffers.interLayer1.unmap();
    readbackBuffers.interLayer2.unmap();

    return { final, inter1, inter2 };
}

export async function runZoomInferencePass({ webGpuContext, config, model, pipeline, lastWeights, channelMask, W, H }) {
    const { device } = webGpuContext;
    const { gW, gH, embeddingChannels: embCh, mlpWidth1, mlpWidth2, embBits, embOffsets } = config;
    const pixelCount = W * H;

    const embF32 = lastWeights.embeddings;
    const range  = computeEmbRange(embF32, embCh, gW * gH);
    webGpuContext.writeBuffer(model.embeddings_range, range);
    const packed = cpuPackEmbeddings(embF32, embCh, range, embBits);
    webGpuContext.writeBuffer(model.embeddings_q, packed);

    const unifBuf = webGpuContext.uniformBuffer(224);
    webGpuContext.writeBuffer(unifBuf, buildFwdUniforms(gW, gH, embCh, mlpWidth1, mlpWidth2, W, H, range, embOffsets));
    uploadChannelMask(channelMask, unifBuf, device);
    const outBuf  = webGpuContext.outputBuffer(pixelCount * 4 * 4);
    const interL1 = webGpuContext.storageBuffer(pixelCount * mlpWidth1 * 4);
    const interL2 = webGpuContext.storageBuffer(pixelCount * mlpWidth2 * 4);
    const rbBuf   = webGpuContext.readbackBuffer(pixelCount * 4 * 4);

    const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: unifBuf } },
            { binding: 1, resource: { buffer: model.embeddings_q } },
            { binding: 2, resource: { buffer: model.mlp_weights } },
            { binding: 3, resource: { buffer: interL1 } },
            { binding: 4, resource: { buffer: interL2 } },
            { binding: 5, resource: { buffer: outBuf  } },
        ],
    });

    const ce = device.createCommandEncoder({ label: 'zoom-inference' });
    const fwdPass = ce.beginComputePass({ label: 'fwd' });
    fwdPass.setPipeline(pipeline);
    fwdPass.setBindGroup(0, bg);
    fwdPass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    fwdPass.end();
    ce.copyBufferToBuffer(outBuf, 0, rbBuf, 0, outBuf.size);
    device.queue.submit([ce.finish()]);

    await rbBuf.mapAsync(GPUMapMode.READ);
    const final = new Float32Array(rbBuf.getMappedRange()).slice();
    rbBuf.unmap();

    unifBuf.destroy(); outBuf.destroy(); interL1.destroy(); interL2.destroy(); rbBuf.destroy();
    return { final };
}
