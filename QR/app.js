// Halftone QR-code generator.
//   - pixelate the source image at a reduced resolution
//   - dither it (grayscale or per-channel color) at `num_levels` per channel
//   - composite QR control + data cells on top, preserving the dithered
//     background everywhere except the 1-pixel centers (carrying the QR bits)
//     and the immovable control bytes.

const $ = sel => document.querySelector(sel);

const SIZES = {  // text-length limits (bits) per QR version 1..10
  L: [152, 272, 440, 640, 864, 1088, 1248, 1552, 1856, 1240],
  M: [128, 224, 352, 512, 688,  864,  992,  700,  700,  524],
  Q: [104, 176, 272, 384, 286,  608,  508,  376,  608,  434],
  H: [ 72, 128, 208, 288, 214,  480,  164,  296,  464,  346],
};

const BAYER4 = [ 0,  8,  2, 10,
                12,  4, 14,  6,
                 3, 11,  1,  9,
                15,  7, 13,  5];

let params = {
  pixelSize:   2,
  dithering:   'diffusion',
  num_levels:  2,
  luma_adjust: 128,
  alpha_limit: 128,
  image:       null,
  text:        'https://skal65535.github.io/QR',
  QRsize:      6,           // [1..10], or 0 = auto
  errorLevel:  'H',
  background:  'image',     // 'image' | 'color' | 'noise' | 'trans'
};

////////////////////////////////////////////////////////////////////////////////
// Drag & drop

function setupDragAndDrop() {
  const area = document.getElementById('main-area');
  const stop = e => { e.preventDefault(); e.stopPropagation(); };
  ['dragenter','dragover','dragleave','drop'].forEach(n => area.addEventListener(n, stop));
  ['dragenter','dragover'].forEach(n => area.addEventListener(n, () => area.classList.add('highlight')));
  ['dragleave','drop'].forEach(n => area.addEventListener(n, () => area.classList.remove('highlight')));
  area.addEventListener('drop', e => loadFile(e.dataTransfer.files[0]));
  $('#file_dialog').addEventListener('change', e => loadFile(e.target.files[0]));
}

function loadFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onloadend = () => setImage(reader.result);
  reader.readAsDataURL(file);
}

function setImage(url) {
  const img = new Image();
  img.onload = () => { params.image = img; GenerateQRCode(); };
  img.src = url;
}

////////////////////////////////////////////////////////////////////////////////
// UI ↔ params

function readUI() {
  params.text        = $('#text').value;
  params.errorLevel  = $('#error_level_selector').value;
  params.pixelSize   = parseInt($('#pixel_size_selector').value);
  params.dithering   = $('#dithering_selector').value;
  params.num_levels  = parseInt($('#gray_selector').value);
  params.luma_adjust = parseInt($('#luma_selector').value);
  params.alpha_limit = parseInt($('#alpha_selector').value);
  params.background  = $('#background').value;

  const resolution = $('#resolution');
  resolution.textContent = '';
  const want = parseInt($('#size_selector').value);   // 0 = Auto
  const bits = 8 * params.text.length;
  const caps = SIZES[params.errorLevel];

  let size = 0;
  if (want === 0) {
    const i = caps.findIndex(c => bits < c);
    if (i >= 0) size = i + 1;
  } else if (bits < caps[want - 1]) {
    size = want;
  }
  if (size === 0) {
    alert(params.errorLevel === 'H'
      ? 'Too much text.'
      : 'Too much text. Try decreasing the error level.');
    return false;
  }
  params.QRsize = size;
  const dim = (size * 4 + 17 + 2) * params.pixelSize * 3;
  resolution.textContent = ` (${dim} × ${dim})`;
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Dithering

// Generic dither on a channel-interleaved Float32 buffer (length w*h*c).
// Quantizes to `levels` steps per channel, writing back into `src`.
//
// Errors are stored in a Uint8ClampedArray — that's not for memory savings,
// it's behavior: the assignment rounds to int and clamps negatives to 0,
// which is the asymmetric error-diffusion the original relies on for its
// halftone look. Neighbour reads are intentionally not bounds-checked so the
// boundary semantics (OOB → 0; edge wraparound to the prior row's tail) match
// the original Uint8ClampedArray-on-imageData implementation.
function dither(src, w, h, c, mode, levels, adjust) {
  const err = new Uint8ClampedArray(src.length);
  const stride = c * w;
  for (let y = 0; y < h; ++y) {
    for (let x = 0; x < w; ++x) {
      const off0 = y * stride + x * c;
      const off1 = off0 - stride;
      for (let k = 0; k < c; ++k) {
        let correction = 0;
        if (mode === 'diffusion') {
          correction = 7 * err[off0 - c + k]
                     + 1 * err[off1 - c + k]
                     + 5 * err[off1     + k]
                     + 3 * err[off1 + c + k];
          correction = (correction / 16) * levels;
        } else if (mode === 'random') {
          correction = Math.floor(Math.random() * 255) - 128;
        } else if (mode === 'ordered') {
          correction = BAYER4[(x & 3) + 4 * (y & 3)] * 16 - 112;
        } else {
          correction = 64;
        }
        const v = src[off0 + k];
        const q = Math.floor((v * levels + correction + adjust) / 256);
        const d = Math.max(0, Math.min(Math.floor(q * 255 / (levels - 1)), 255));
        err[off0 + k] = v - d;     // Uint8ClampedArray: rounds + clamps to [0,255]
        src[off0 + k] = d;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Pipeline

function disableSmoothing(ctx) {
  ctx.imageSmoothingEnabled = false;
  ctx.mozImageSmoothingEnabled = false;
  ctx.msImageSmoothingEnabled = false;
  ctx.webkitImageSmoothingEnabled = false;
}

// Down-sample the source into a tiny canvas (1/pixelSize), then blow it back
// up — that's the "pixelated" preview. Returns the ImageData of #imagePixel.
function pixelate() {
  const input = $('#imageInput');
  const ictx = input.getContext('2d');
  ictx.clearRect(0, 0, input.width, input.height);
  ictx.drawImage(params.image, 0, 0, input.width, input.height);

  const pixel = $('#imagePixel');
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = pixel.width / params.pixelSize;
  const tctx = tmp.getContext('2d');
  disableSmoothing(tctx);
  tctx.drawImage(input, 0, 0, tmp.width, tmp.height);

  // Crucially, the upscale step leaves smoothing ENABLED — bilinear filtering
  // gives a small gradient inside each "pixel block", which FS dither needs
  // to diffuse smoothly. With nearest-neighbor here, flat blocks force the
  // raster-order dither into row-aligned patterns → horizontal banding.
  const pctx = pixel.getContext('2d');
  pctx.drawImage(tmp, 0, 0, pixel.width, pixel.height);
  return pctx.getImageData(0, 0, pixel.width, pixel.height);
}

// Render the pixelated + dithered preview into #imageDithered.
function renderPreview() {
  const pixels = pixelate();
  const d = pixels.data;
  const w = pixels.width, h = pixels.height;
  const adjust = params.luma_adjust - 128;
  const levels = params.num_levels;
  const color  = (params.background === 'color');

  if (color) {
    // Dither R, G, B independently.
    const src = new Float32Array(w * h * 3);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
      src[j + 0] = d[i + 0];
      src[j + 1] = d[i + 1];
      src[j + 2] = d[i + 2];
    }
    dither(src, w, h, 3, params.dithering, levels, adjust);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
      d[i + 0] = src[j + 0];
      d[i + 1] = src[j + 1];
      d[i + 2] = src[j + 2];
    }
  } else {
    // Luma + grayscale. Floor to match the original integer pipeline so the
    // Uint8ClampedArray-stored errors don't accrete rounding drift.
    const src = new Float32Array(w * h);
    for (let i = 0, j = 0; i < d.length; i += 4, ++j) {
      src[j] = Math.floor(d[i + 0] * 0.2126 + d[i + 1] * 0.7152 + d[i + 2] * 0.0722);
    }
    dither(src, w, h, 1, params.dithering, levels, adjust);
    for (let i = 0, j = 0; i < d.length; i += 4, ++j) {
      d[i + 0] = d[i + 1] = d[i + 2] = src[j];
    }
  }
  $('#imageDithered').getContext('2d').putImageData(pixels, 0, 0);
}

// Build the final QR canvas: dithered background, then control/data cells.
function renderQR(qrBits, ctrlBits) {
  const pixelSize = params.pixelSize;
  const blockSize = 3 * pixelSize;
  const dim    = qrBits.length * blockSize;
  const dimOut = dim + 2 * blockSize;             // including white quiet zone

  for (const id of ['#imageInput', '#imageDithered', '#imagePixel']) {
    const c = $(id); c.width = dim; c.height = dim;
    c.style.width = c.style.height = dim + 'px';   // pin CSS = bitmap, no resample
  }
  const out = $('#output');
  out.width = out.height = dimOut;
  out.style.width = out.style.height = dimOut + 'px';
  const ctx = out.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, dimOut, dimOut);

  // Optional background + alpha map (for cutting out transparent regions).
  let transp = null, stride = 0;
  const hasImageBg = params.image && (params.background === 'image' || params.background === 'color');
  if (hasImageBg) {
    renderPreview();
    ctx.drawImage($('#imageDithered'), blockSize, blockSize, dim, dim);
    if (params.alpha_limit > 0) {
      const pixel = $('#imagePixel');
      stride = pixel.width;
      transp = pixel.getContext('2d').getImageData(0, 0, stride, pixel.height).data;
    }
  }

  // Draw cells. Note: in the original, Y is row, X is column, but fillRect's
  // x argument is `off_y` (= (Y+1)*blockSize); kept verbatim for compat.
  for (let Y = 0; Y < qrBits.length; ++Y) {
    for (let X = 0; X < qrBits[Y].length; ++X) {
      const off_x = (X + 1) * blockSize;
      const off_y = (Y + 1) * blockSize;

      if (!hasImageBg) {   // either no image, or 'noise' / 'trans'
        if (params.background === 'noise' || !params.image) {
          for (let y = 0; y < 3; ++y) {
            for (let x = 0; x < 3; ++x) {
              ctx.fillStyle = (Math.random() < 0.5) ? 'white' : 'black';
              ctx.fillRect(off_y + y * pixelSize, off_x + x * pixelSize, pixelSize, pixelSize);
            }
          }
        }
      }

      if (ctrlBits[Y][X] !== null) {
        ctx.fillStyle = ctrlBits[Y][X] ? 'black' : 'white';
        ctx.fillRect(off_y, off_x, blockSize, blockSize);
      } else {
        // Center pixel carries the data bit.
        ctx.fillStyle = qrBits[Y][X] ? 'black' : 'white';
        ctx.fillRect(off_y + pixelSize, off_x + pixelSize, pixelSize, pixelSize);
        // Transparent regions of the source are forced to white so the QR
        // doesn't pick up garbage from the background.
        if (transp) {
          const alpha = transp[(Y * blockSize + X * blockSize * stride) * 4 + 3];
          if (alpha < params.alpha_limit) {
            ctx.fillRect(off_y, off_x, blockSize, blockSize);
          }
        }
      }
    }
  }
  $('#download').href = out.toDataURL('image/webp', 1);
}

////////////////////////////////////////////////////////////////////////////////

function GenerateQRCode() {
  if (!readUI()) return;
  const payload = MakePayload(params.QRsize, params.errorLevel, params.text);
  const qr      = new qrcode(params.QRsize, params.errorLevel);
  const pat     = qr.GetBestPattern(payload);
  const bytes   = qr.MakeData(payload, false, pat, false);
  const ctrl    = qr.MakeData(payload, false, pat, true);
  renderQR(bytes, ctrl);
}

window.onload = () => {
  setupDragAndDrop();
  setImage('./QR.webp');
};
