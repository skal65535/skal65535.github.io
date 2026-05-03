// loss.js
// MSE loss computation between network output and target pixel values.
export function calculate_loss(output, target, outCh = 4) {
    let loss = 0;
    const pixelCount = output.length / 4;
    for (let p = 0; p < pixelCount; p++) {
        for (let c = 0; c < outCh; c++) {
            const diff = output[p * 4 + c] - target[p * 4 + c];
            loss += diff * diff;
        }
    }
    return loss / (pixelCount * outCh);
}

export function get_target_pixels(image, canvas) {
    const ctx = document.createElement('canvas').getContext('2d');
    ctx.canvas.width = canvas.width;
    ctx.canvas.height = canvas.height;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const float32Array = new Float32Array(imageData.data.length);
    for (let i = 0; i < imageData.data.length; i++) {
        float32Array[i] = imageData.data[i] / 255.0;
    }
    return float32Array;
}
