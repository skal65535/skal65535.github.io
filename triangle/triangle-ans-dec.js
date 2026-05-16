'use strict';

function decodePreview(b64) {
  const data = atob(b64);
  const dec = new ANSDec(data);

  const preview = {};
  preview.grid_x = dec.ReadRange(kPreviewMinGridSize, kPreviewMaxGridSize);
  preview.grid_y = dec.ReadRange(kPreviewMinGridSize, kPreviewMaxGridSize);
  preview.use_noise = dec.NextBit(kPreviewNoiseProba) ? true : false;
  preview.nb_colors = dec.ReadRange(kPreviewMinNumColors, kPreviewMaxNumColors);
  preview.has_alpha = dec.NextBit(kPreviewOpaqueProba) ? true : false;

  const color_data = new Array(preview.nb_colors);
  const alpha_stats = new ANSBinSymbol(2, 2);
  const syco = new ValueStats(), scg = new ValueStats();
  let pa = 1, py = kYCoCgMax >> 1, pco = kYCoCgMax >> 1, pcg = 0;

  for (let i = 0; i < preview.nb_colors; i++) {
    if (preview.has_alpha) {
      if (dec.NextAdaptiveBit(alpha_stats)) pa = 1 - pa;
    }
    const y  = py  = dec.ReadAValue(syco, false, py);
    const co = pco = dec.ReadAValue(syco, false, pco);
    const cg = pcg = dec.ReadAValue(scg,  true,  pcg);
    color_data[i] = { y, co, cg, a: pa };
  }

  const nb_nc = preview.grid_x * preview.grid_y - 4;
  const min_pts = Math.max(preview.nb_colors - 4, kPreviewMinNumVertices);
  const max_pts = Math.min(nb_nc, kPreviewMaxNumVertices);
  preview.nb_pts = dec.ReadRange(min_pts, max_pts);

  preview.qpts = new Array(preview.nb_pts + 4);
  const istats = new ANSBinSymbol(2, 2);
  const decIdx = (pred) => {
    if (dec.NextAdaptiveBit(istats)) {
      const v = dec.ReadRange(0, preview.nb_colors - 2);
      return v < pred ? v : v + 1;
    }
    return pred;
  };

  let pidx = 0;
  pidx = decIdx(pidx); preview.qpts[0] = { x: 0, y: 0, idx: pidx };
  pidx = decIdx(pidx); preview.qpts[1] = { x: preview.grid_x - 1, y: 0, idx: pidx };
  pidx = decIdx(pidx); preview.qpts[2] = { x: 0, y: preview.grid_y - 1, idx: pidx };
  pidx = decIdx(pidx); preview.qpts[3] = { x: preview.grid_x - 1, y: preview.grid_y - 1, idx: pidx };

  let cells = nb_nc, pts = preview.nb_pts;
  for (let y = 0, k = 4; y < preview.grid_y; ++y) {
    for (let x = 0; x < preview.grid_x; ++x) {
      if ((x === 0 || x === preview.grid_x - 1) && (y === 0 || y === preview.grid_y - 1)) continue;
      if (pts > 0) {
        const proba = kProbaMax - Math.floor((pts << 16) / cells);
        if (dec.NextBit(proba)) {
          pidx = decIdx(pidx);
          preview.qpts[k++] = { x, y, idx: pidx };
          --pts;
        }
      }
      --cells;
    }
  }

  return { preview, color_data };
}

if (typeof module !== 'undefined') {
  module.exports = { decodePreview };
}
