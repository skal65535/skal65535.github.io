//
// Inspired from https://kazuhikoarase.github.io/qrcode-generator/js/demo/
//
// Licensed under the MIT license:
//   http://www.opensource.org/licenses/mit-license.php
//
// The word 'QR Code' is registered trademark of
// DENSO WAVE INCORPORATED
//   http://www.denso-wave.com/qrcode/faqpatent-e.html
//

////////////////////////////////////////////////////////////////////////////////
// Drag'n'Drop

function PreventDefaults (e) {
  e.preventDefault();
  e.stopPropagation();
}
function HandleDrop(e) {
  HandleFile(e.dataTransfer.files[0]);
}
function HandleFile(file) {
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = function() { SetImage(reader.result); }
}
function SetupDragAndDrop() {
  const dropArea = document.getElementById('drop-area');
  function highlight(e) { dropArea.classList.add('highlight'); }
  function unhighlight(e) { dropArea.classList.remove('highlight'); }

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(
    name => { dropArea.addEventListener(name, PreventDefaults, false); }
  );
  ['dragenter', 'dragover'].forEach(
    name => { dropArea.addEventListener(name, highlight, false); }
  );
  ['dragleave', 'drop'].forEach(
    name => { dropArea.addEventListener(name, unhighlight, false); }
  );
  dropArea.addEventListener('drop', HandleDrop, false)
}

////////////////////////////////////////////////////////////////////////////////
// Params

var params = {  // Global parameters
  pixelSize: 2,
  dithering: 'diffusion',
  num_levels: 2,
  luma_adjust: 128,
  alpha_limit: 128,
  image: null,
  text: "https://skal65535.github.io/QR",
  QRsize: 6,   // in [1, 10], 0=Auto
  error_level: 'H',
  background: 'image',
  print() {
    console.log("text = [" + this.text + "]\n" +
                "lvl  = [" + this.errorLevel + "]\n" +
                "QRsize = [" + this.QRsize + "]\n" +
                "Background = [" + this.background + "]\n" +
                "dithering = [" + this.dithering + "]\n" +
                "num_levels = [" + this.num_levels + "]\n" +
                "luma_adjust = [" + this.luma_adjust + "]\n" +
                "pixelSize = [" + this.pixelSize + "]\n");
  },
};


//---------------------------------------------------------------------

function HalftoneQR(QRBytes, controlBytes, image) {
  const pixelSize = params.pixelSize;
  const blockSize = 3 * pixelSize;
  const dim = QRBytes.length * blockSize;
  const dim_out = dim + 2 * blockSize;   // including the border

  ['#imageInput', '#imageDithered', '#imagePixel'].forEach(name => {
    const obj = document.querySelector(name);
    obj.width = dim;
    obj.height = dim;
  });

  const canvas = document.querySelector('#output');
  canvas.width = dim_out;
  canvas.height = dim_out;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.rect(0, 0, dim_out, dim_out);
  ctx.fill();
  let transp = null;
  let stride = 0;
  if (params.image != null) {
    Render();
    if (params.background === 'image') {
      const canvasDithered = document.querySelector('#imageDithered');
      const ctxDithered = canvasDithered.getContext('2d');
      ctx.drawImage(canvasDithered, blockSize, blockSize, dim, dim);
      if (params.alpha_limit > 0) {
        const canvasPixel = document.querySelector('#imagePixel');
        const ctxPixel = canvasPixel.getContext('2d');
        const pixels = ctxPixel.getImageData(0, 0, canvasPixel.width, canvasPixel.height);
        transp = pixels.data;
        stride = canvasPixel.width;
      }
    }
  }

  for (let Y = 0; Y < QRBytes.length; ++Y) {
    for (let X = 0; X < QRBytes[Y].length; ++X) {
      const off_x = (X + 1) * blockSize;
      const off_y = (Y + 1) * blockSize;
      if (params.image == null || params.background === 'noise') {
        // Draw random bytes
        for (let y = 0; y < 3; ++y) {
          for (let x = 0; x < 3; ++x) {
            ctx.fillStyle = (Math.random() < 0.5) ? 'white' : 'black';
            ctx.fillRect(off_y + y * pixelSize, off_x + x * pixelSize,
                         pixelSize, pixelSize);
          }
        }
      }
      // Re-draw control bytes
      if (controlBytes[Y][X] !== null) {
        ctx.fillStyle = (controlBytes[Y][X] === true) ? 'black' : 'white';
        ctx.fillRect(off_y, off_x, blockSize, blockSize);
      } else {   // Middle Cell
        ctx.fillStyle = QRBytes[Y][X] ? 'black' : 'white';
        ctx.fillRect(off_y + pixelSize, off_x + pixelSize, pixelSize, pixelSize);
        if (transp != null) {
          // TODO(skal): average alpha over a 3x3 block??
          const alpha = transp[(Y + X * stride) * 4 * blockSize + 3];
          if (alpha < params.alpha_limit) {
            ctx.fillRect(off_y, off_x, blockSize, blockSize);
          }
        }
      }
    }
  }
  const result = document.querySelector('#output').toDataURL('image/webp', 1);
  document.querySelector('#download').href = result;
}

//---------------------------------------------------------------------

function DisableSmoothing(ctx) {
  ctx.imageSmoothingEnabled = false;
  ctx.mozImageSmoothingEnabled = false;
  ctx.msImageSmoothingEnabled = false;
  ctx.webkitImageSmoothingEnabled =  false;
}

function Render() {
  const canvasColour = document.querySelector('#imageInput');
  const ctxColour = canvasColour.getContext('2d');
  ctxColour.clearRect(0, 0, canvasColour.width, canvasColour.height);
  ctxColour.drawImage(params.image, 0, 0, canvasColour.width, canvasColour.height);

  const canvasPixel = document.querySelector('#imagePixel');
  const ctxPixel = canvasPixel.getContext('2d');

  const canvasTmp = document.createElement('canvas');
  // reduced resolution:
  canvasTmp.width = canvasTmp.height = canvasPixel.width / params.pixelSize;
  const ctxTmp = canvasTmp.getContext('2d');
  DisableSmoothing(ctxTmp);
  ctxTmp.drawImage(canvasColour, 0, 0, canvasTmp.width, canvasTmp.height);
  ctxPixel.drawImage(canvasTmp, 0, 0, canvasPixel.width, canvasPixel.height);

  // DitherImage
  const canvasDithered = document.querySelector('#imageDithered');
  const ctxDithered = canvasDithered.getContext('2d');

  const pixels = ctxPixel.getImageData(0, 0, canvasPixel.width, canvasPixel.height);
  const d = pixels.data;
  const num_levels = params.num_levels;
  const adjust = params.luma_adjust - 128;

  for (let i = 0; i < d.length; i += 4) {
    const r = d[i + 0], g = d[i + 1], b = d[i + 2];
    d[i + 0] = Math.floor(r * 0.2126 + g * 0.7152 + b * 0.0722);
    d[i + 1] = 0;   // reset error
  }
  const stride = 4 * canvasPixel.width;
  for (let y = 0; y < canvasPixel.height; ++y) {
    for (let x = 0; x < canvasPixel.width; ++x) {
      const off0 = y * stride + x * 4;
      let correction = 0;
      if (params.dithering == 'diffusion') {
        const off1 = off0 - stride;
        correction = 0;
        correction += 7 * d[off0 - 1 * 4 + 1];
        correction += 1 * d[off1 - 1 * 4 + 1];
        correction += 5 * d[off1 + 0 * 4 + 1];
        correction += 3 * d[off1 + 1 * 4 + 1];
        correction /= 16;
        correction *= num_levels;
      } else if (params.dithering == 'random') {
        correction = Math.floor(Math.random() * 255) - 128;
      } else if (params.dithering == 'ordered') {
        const dithering_4x4 = [  0,  8,  2, 10,
                                12,  4, 14,  6,
                                 3, 11,  1,  9,
                                15,  7, 13,  5 ];
        const x4 = x % 4, y4 = y % 4;
        correction = dithering_4x4[x4 + 4 * y4] * 16 - 112;
      } else {
        correction = 64;
      }
      const gray = d[off0 + 0];
      const q_gray = Math.floor((gray * num_levels + correction + adjust) / 256);
      const d_gray =
        Math.max(0, Math.min(Math.floor(q_gray * 255 / (num_levels - 1)), 255));
      const error = gray - d_gray;
      d[off0 + 0] = d_gray;
      d[off0 + 1] = error;
    }
  }
  for (let i = 0; i < d.length; i += 4) {
    d[i + 1] = d[i + 2] = d[i + 0];
  }
  ctxDithered.putImageData(pixels, 0, 0);
}

function ParseParams(p) {
  p.text = document.querySelector('#text').value;
  p.errorLevel = document.querySelector('#error_level_selector').value;
  p.pixelSize = parseInt(document.querySelector('#pixel_size_selector').value);
  p.dithering = document.querySelector('#dithering_selector').value;
  p.num_levels = document.querySelector('#gray_selector').value;
  p.luma_adjust = parseInt(document.querySelector('#luma_selector').value);
  p.alpha_limit = parseInt(document.querySelector('#alpha_selector').value);
  p.background = document.querySelector('#background').value;

  let dim_text = document.querySelector('#resolution');
  dim_text.innerHTML = "";
  const sizes = {
    // 10 levels for each quality
    L: [152, 272, 440, 640, 864, 1088, 1248, 1552, 1856, 1240],
    M: [128, 224, 352, 512, 688, 864,  992,  700,  700,  524],
    Q: [104, 176, 272, 384, 286, 608,  508,  376,  608,  434],
    H: [72,  128, 208, 288, 214, 480,  164,  296,  464,  346]
  };
  const userSize = parseInt(document.querySelector('#size_selector').value);

  p.QRsize = 0;
  const len = 8 * p.text.length;  // in bits
  if (userSize === 0) {    // 'Auto'
    for (let i = 0; i < sizes[p.errorLevel].length; i++) {
      if (len < sizes[p.errorLevel][i]) {
        p.QRsize = i + 1;
        break;
      }
    }
    if (p.QRsize === 0) {
      if (p.errorLevel === 'H') {
        alert('Too much text.');
      } else {
        alert('Too much text. Try decreasing the error level.');
      }
    }
  } else {
    if (len < sizes[p.errorLevel][userSize - 1]) {
      p.QRsize = userSize;
    } else {
      alert('Text is too long (length: ' + len +
            ' bits). Try decreasing the error level (' + p.errorLevel +
            ') or increasing the size (' + userSize + ').');
    }
  }
  if (p.QRsize == 0) return false;
  const dim = (p.QRsize * 4 + 17 + 2) * p.pixelSize * 3;
  dim_text.innerHTML = ' (' + dim + ' x ' + dim + ')';
  return true;
}

function GenerateQRCode() {
  let new_params = params;
  if (!ParseParams(new_params)) return;
  params = new_params;
  params.print();

  const payload = CreateData(params.QRsize, params.errorLevel, params.text);

  const qr = new qrcode(params.QRsize, params.errorLevel);
  const best_pattern_id = qr.GetBestPattern(payload);
  const bytes = qr.MakeData(payload, false, best_pattern_id, false);
  const control_bytes = qr.MakeData(payload, false, best_pattern_id, true);

  HalftoneQR(bytes, control_bytes);
}

function SetImage(url) {
  const new_image = new Image();
  new_image.onload = function() {
    params.image = this;
    GenerateQRCode();
  }
  new_image.src = url;
}
