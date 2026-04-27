// loss.js

export function calculate_loss(output, target) {
    let loss = 0;
    for (let i = 0; i < output.length; i++) {
        const diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / output.length;
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
