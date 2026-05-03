import { createModel, destroyModel, initCpuWeights, buildFwdUniforms } from './model.js';
import { buildShader } from './webgpu.js';

export class GpuSession {
    constructor(webGpuContext, config, canvasWidth, canvasHeight) {
        this.type = 'gpu';
        this.ctx = webGpuContext;
        this.config = config;
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;

        this.model = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.fwdUniformsBuf = null;
        this.outputBuffers = {};
        this.readbackBuffers = {};

        const { buffers, weights } = createModel(webGpuContext, config);
        this.model = buffers;
        this.initialWeights = weights;

        this._createPipeline();
        this.rebuildBindGroup();
    }

    _createPipeline() {
        const shaderModule = this.ctx.device.createShaderModule({ code: buildShader(this.config) });
        this.pipeline = this.ctx.device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
    }

    rebuildBindGroup() {
        this.fwdUniformsBuf?.destroy();
        for (const b of Object.values(this.outputBuffers)) b?.destroy();
        for (const b of Object.values(this.readbackBuffers)) b?.destroy();

        const { mlpWidth1, mlpWidth2, gW, gH, embeddingChannels, embOffsets } = this.config;
        this.fwdUniformsBuf = this.ctx.uniformBuffer(224, 'fwdUniforms');
        this.ctx.writeBuffer(this.fwdUniformsBuf,
            buildFwdUniforms(gW, gH, embeddingChannels, mlpWidth1, mlpWidth2, this.canvasWidth, this.canvasHeight, null, embOffsets));

        const pixelCount = this.canvasWidth * this.canvasHeight;
        const embSize    = gW * gH * embeddingChannels;
        const stride1    = mlpWidth1 * 4;
        const stride2    = mlpWidth2 * 4;

        this.outputBuffers.interLayer1 = this.ctx.outputBuffer(pixelCount * stride1, 'out/inter1');
        this.outputBuffers.interLayer2 = this.ctx.outputBuffer(pixelCount * stride2, 'out/inter2');
        this.outputBuffers.final       = this.ctx.outputBuffer(pixelCount * 4 * 4,   'out/final');

        this.readbackBuffers.final         = this.ctx.readbackBuffer(pixelCount * 4 * 4,                  'rb/final');
        this.readbackBuffers.embeddings    = this.ctx.readbackBuffer(embSize * 4,                          'rb/emb');
        this.readbackBuffers.layer1Weights = this.ctx.readbackBuffer(mlpWidth1 * embeddingChannels * 4,    'rb/L1w');
        this.readbackBuffers.layer1Biases  = this.ctx.readbackBuffer(stride1,                              'rb/L1b');
        this.readbackBuffers.layer2Weights = this.ctx.readbackBuffer(mlpWidth2 * stride1,                  'rb/L2w');
        this.readbackBuffers.layer3Weights = this.ctx.readbackBuffer((this.config.hasAlpha ? 4 : 3) * stride2,  'rb/L3w');
        this.readbackBuffers.interLayer1   = this.ctx.readbackBuffer(pixelCount * stride1,                 'rb/inter1');
        this.readbackBuffers.interLayer2   = this.ctx.readbackBuffer(pixelCount * stride2,                 'rb/inter2');

        this.bindGroup = this.ctx.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.fwdUniformsBuf } },
                { binding: 1, resource: { buffer: this.model.embeddings_q } },
                { binding: 2, resource: { buffer: this.model.mlp_weights } },
                { binding: 3, resource: { buffer: this.outputBuffers.interLayer1 } },
                { binding: 4, resource: { buffer: this.outputBuffers.interLayer2 } },
                { binding: 5, resource: { buffer: this.outputBuffers.final       } },
            ],
        });
    }

    shaderChanged(newConfig) {
        const c = this.config;
        return c.smoothInterpolation !== newConfig.smoothInterpolation
            || c.activation          !== newConfig.activation
            || c.quantization        !== newConfig.quantization;
    }

    rebuildPipeline(updates) {
        Object.assign(this.config, updates);
        this._createPipeline();
        this.rebuildBindGroup();
    }

    destroy() {
        if (this.model) destroyModel(this.model);
        this.fwdUniformsBuf?.destroy();
        for (const b of Object.values(this.outputBuffers)) b?.destroy();
        for (const b of Object.values(this.readbackBuffers)) b?.destroy();

        this.model = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.fwdUniformsBuf = null;
    }
}

export class CpuSession {
    constructor(config) {
        this.type = 'cpu';
        this.config = config;
        this.initialWeights = initCpuWeights(config);
    }

    // CPU session doesn't need to rebuild bind groups, but we provide the interface
    rebuildBindGroup() {}

    // CPU arrays are garbage collected
    destroy() {
        this.initialWeights = null;
    }
}
