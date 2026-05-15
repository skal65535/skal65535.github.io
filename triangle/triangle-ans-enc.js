'use strict';

// ---------------------------------------------------------------------------

class ANSEnc {
  constructor() { this._tok = []; }

  PutBit(bit, p0) {
    bit = bit ? 1 : 0;
    if (p0 > 0 && p0 < kProbaMax) this._tok.push({bit, p0});
    return bit;
  }

  PutABit(bit, stats) {
    const p0 = stats.Proba();
    stats.Update(bit);
    return this.PutBit(bit, p0);
  }

  PutRange(value, min, max) {
    this._tok.push({v: value - min, R: max - min + 1});
    return value;
  }

  PutAValue(stats, is_positive, pred, value) {
    const delta = value - pred;
    if (delta === 0) { this.PutABit(1, stats.zero); return value; }
    this.PutABit(0, stats.zero);
    const mag = Math.abs(delta) - 1;
    for (let k = 0; k < kYCoCgBitDepth; ++k) this.PutABit((mag >> k) & 1, stats.bits[k]);
    if (!is_positive) this.PutABit(delta < 0 ? 1 : 0, stats.sign);
    return value;
  }

  Assemble() {
    let e = 0xf3 * kProbaMax;
    const words = [];
    const emit = () => { words.push(e % kProbaMax); e = Math.floor(e / kProbaMax) >>> 0; };
    const kLim = kProbaMax * kProbaMax;

    for (let i = this._tok.length - 1; i >= 0; --i) {
      const tok = this._tok[i];
      if (tok.p0 !== undefined) {
        const proba = tok.bit ? kProbaMax - tok.p0 : tok.p0;
        if (e >= proba * kProbaMax) emit();
        const q = Math.floor(e / proba), r = e - q * proba;
        e = (q * kProbaMax + r + (tok.bit ? tok.p0 : 0)) >>> 0;
      } else {
        const en = e * tok.R + tok.v;
        if (en >= kLim) { words.push(en % kProbaMax); e = Math.floor(en / kProbaMax) >>> 0; }
        else e = en >>> 0;
      }
    }
    emit(); emit();

    const n = words.length;
    const out = new Uint8Array(n * 2);
    for (let i = 0; i < n; ++i) {
      const w = words[n - 1 - i];
      out[2*i] = (w >> 8) & 0xff; out[2*i+1] = w & 0xff;
    }
    return out;
  }

  ToBase64() {
    const b = this.Assemble();
    let s = '';
    for (let i = 0; i < b.length; ++i) s += String.fromCharCode(b[i]);
    return btoa(s);
  }

  get numTokens() { return this._tok.length; }
}

// ---------------------------------------------------------------------------

function encodePreview(preview, color_data) {
  const enc = new ANSEnc();
  const {grid_x: gx, grid_y: gy, nb_colors: nc, nb_pts} = preview;

  enc.PutRange(gx, kPreviewMinGridSize, kPreviewMaxGridSize);
  enc.PutRange(gy, kPreviewMinGridSize, kPreviewMaxGridSize);
  enc.PutBit(preview.use_noise ? 1 : 0, kPreviewNoiseProba);
  enc.PutRange(nc, kPreviewMinNumColors, kPreviewMaxNumColors);
  enc.PutBit(preview.has_alpha ? 1 : 0, kPreviewOpaqueProba);

  const alpha_stats = new ANSBinSymbol(2, 2);
  const syco = new ValueStats(), scg = new ValueStats();
  let pa = 1, py = kYCoCgMax >> 1, pco = kYCoCgMax >> 1, pcg = 0;

  for (const c of color_data) {
    if (preview.has_alpha) {
      if (enc.PutABit(c.a !== pa ? 1 : 0, alpha_stats)) pa = 1 - pa;
    }
    py  = enc.PutAValue(syco, false, py,  c.y);
    pco = enc.PutAValue(syco, false, pco, c.co);
    pcg = enc.PutAValue(scg,  true,  pcg, c.cg);
  }

  const nb_nc = gx * gy - 4;
  const min_pts = Math.max(nc - 4, kPreviewMinNumVertices);
  const max_pts = Math.min(nb_nc, kPreviewMaxNumVertices);
  enc.PutRange(nb_pts, min_pts, max_pts);

  const vtx_map = new Map();
  for (let k = 4; k < nb_pts + 4; ++k) {
    const v = preview.qpts[k];
    vtx_map.set(v.y * gx + v.x, v.idx);
  }

  const istats = new ANSBinSymbol(2, 2);
  const encIdx = (idx, pred) => {
    if (enc.PutABit(idx !== pred ? 1 : 0, istats)) {
      enc.PutRange(idx < pred ? idx : idx - 1, 0, nc - 2);
    }
    return idx;
  };

  let pidx = 0;
  for (let k = 0; k < 4; ++k) pidx = encIdx(preview.qpts[k].idx, pidx);

  let cells = nb_nc, pts = nb_pts;
  for (let y = 0; y < gy; ++y) {
    for (let x = 0; x < gx; ++x) {
      if ((x === 0 || x === gx-1) && (y === 0 || y === gy-1)) continue;
      if (pts > 0) {
        const key = y * gx + x, has = vtx_map.has(key) ? 1 : 0;
        enc.PutBit(has, kProbaMax - Math.floor((pts << 16) / cells));
        if (has) { pidx = encIdx(vtx_map.get(key), pidx); --pts; }
      }
      --cells;
    }
  }

  return enc.ToBase64();
}

// ---------------------------------------------------------------------------

if (typeof module !== 'undefined') {
  module.exports = { ANSEnc, encodePreview };
}
