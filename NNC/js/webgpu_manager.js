// webgpu_manager.js
// Initialises the WebGPU adapter and device; returns the webGpuContext object.
export async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
                 ?? await navigator.gpu.requestAdapter({ powerPreference: 'low-power' })
                 ?? await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice({ requiredFeatures: [] });
    if (!device) {
        throw new Error("No appropriate GPUDevice found.");
    }

    return {
        device,
        createBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label = '') {
            const buffer = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true, label });
            new Float32Array(buffer.getMappedRange()).set(data);
            buffer.unmap();
            return buffer;
        },
        zeroBuffer(count, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label = '') {
            return device.createBuffer({ size: count * 4, usage, label });
        },
        uniformBuffer(size, label = '') {
            return device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
        },
        storageBuffer(size, label = '') {
            return device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, label });
        },
        outputBuffer(size, label = '') {
            return device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label });
        },
        readbackBuffer(size, label = '') {
            return device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label });
        },
        // Read multiple GPU buffers in one submit. bufMap: {key: {buf, size}} → {key: Float32Array}
        async readBackBuffers(bufMap) {
            const rbBufs = {};
            const ce = device.createCommandEncoder({ label: 'readBackBuffers' });
            for (const [k, { buf, size }] of Object.entries(bufMap)) {
                rbBufs[k] = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                ce.copyBufferToBuffer(buf, 0, rbBufs[k], 0, size * 4);
            }
            device.queue.submit([ce.finish()]);
            await Promise.all(Object.values(rbBufs).map(b => b.mapAsync(GPUMapMode.READ)));
            const result = {};
            for (const [k, b] of Object.entries(rbBufs)) {
                result[k] = new Float32Array(b.getMappedRange()).slice();
                b.unmap();
                b.destroy();
            }
            return result;
        },
        writeBuffer(buf, data) {
            device.queue.writeBuffer(buf, 0, data);
        },
        writeBufferAt(buf, byteOffset, data) {
            device.queue.writeBuffer(buf, byteOffset, data);
        },
        clearBuffer(buf) {
            device.queue.writeBuffer(buf, 0, new Uint8Array(buf.size));
        },
        // Upload all MLP weight tensors to their GPU buffers.
        uploadModelWeights(model, tensors) {
            device.queue.writeBuffer(model.embeddings,     0, tensors.embeddings);
            device.queue.writeBuffer(model.layer1.weights, 0, tensors.layer1_weights);
            device.queue.writeBuffer(model.layer1.biases,  0, tensors.layer1_biases);
            device.queue.writeBuffer(model.layer2.weights, 0, tensors.layer2_weights);
            device.queue.writeBuffer(model.layer2.biases,  0, tensors.layer2_biases);
            device.queue.writeBuffer(model.layer3.weights, 0, tensors.layer3_weights);
            device.queue.writeBuffer(model.layer3.biases,  0, tensors.layer3_biases);
            if (model.mlp_weights) {
                const order = ['layer1_weights', 'layer1_biases', 'layer2_weights', 'layer2_biases', 'layer3_weights', 'layer3_biases'];
                let off = 0;
                for (const k of order) {
                    device.queue.writeBuffer(model.mlp_weights, off * 4, tensors[k]);
                    off += tensors[k].length;
                }
            }
        },
    };
}
