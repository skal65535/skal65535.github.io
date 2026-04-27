// webgpu_manager.js

export async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBuffersPerShaderStage: 10
        }
    });
    if (!device) {
        throw new Error("No appropriate GPUDevice found.");
    }

    return {
        device,
        createBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
            const buffer = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
            new Float32Array(buffer.getMappedRange()).set(data);
            buffer.unmap();
            return buffer;
        },
        zeroBuffer(count, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
            return device.createBuffer({ size: count * 4, usage });
        },
        uniformBuffer(size) {
            return device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        },
        storageBuffer(size) {
            return device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
        },
        outputBuffer(size) {
            return device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        },
        readbackBuffer(size) {
            return device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        },
    };
}
