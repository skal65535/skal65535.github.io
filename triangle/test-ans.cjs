'use strict';
const fs = require('fs');
const path = require('path');

// ---- Constants ----
const kProbaMax              = 1 << 16;
const kYCoCgBitDepth         = 6;
const kYCoCgMax              = (1 << kYCoCgBitDepth) - 1;
const kPreviewMinNumVertices = 0;
const kPreviewMaxNumVertices = 1024;
const kPreviewMinNumColors   = 2;
const kPreviewMaxNumColors   = 32;
const kPreviewMinGridSize    = 2;
const kPreviewMaxGridSize    = 256;
const kPreviewOpaqueProba    = 3 * kProbaMax / 4;
const kPreviewNoiseProba     = kProbaMax / 2;
const kMaxSum                = 256;

// ---- ANSBinSymbol / ValueStats ----
function ANSBinSymbol(p0, p1) { this.p0_ = p0; this.sum_ = p0 + p1; }
ANSBinSymbol.prototype.Update = function(bit) {
  if (this.sum_ < kMaxSum) { if (!bit) ++this.p0_; ++this.sum_; }
  return bit;
};
ANSBinSymbol.prototype.Proba = function() {
  return Math.floor((this.p0_ << 16) / this.sum_);
};
function ValueStats() {
  this.zero = new ANSBinSymbol(1, 1);
  this.sign = new ANSBinSymbol(1, 1);
  this.bits = [];
  for (let i = 0; i < kYCoCgBitDepth; ++i) this.bits.push(new ANSBinSymbol(1, 1));
}

// ---- ANSDec ----
function ANSDec(data) {
  this.data_ = data; this.pos_ = 0; this.max_pos_ = data.length;
  this.state_ = new Uint32Array(1);
  this.NextWord(); this.NextWord();
}
ANSDec.prototype.NextWord = function() {
  const pos = this.pos_; this.pos_ = pos + 2;
  const hi = (pos     < this.max_pos_) ? this.data_.charCodeAt(pos)     : 0;
  const lo = (pos + 1 < this.max_pos_) ? this.data_.charCodeAt(pos + 1) : 0;
  this.state_[0] = (this.state_[0] << 16) | ((lo + 256 * hi) || 0);
};
ANSDec.prototype.NextBit = function(p0) {
  const q0 = kProbaMax - p0;
  if (p0 === 0) return 1;
  if (q0 === 0) return 0;
  const xfrac = this.state_[0] % kProbaMax;
  const bit = (xfrac >= p0) ? 1 : 0;
  if (!bit) this.state_[0] = p0 * (this.state_[0] >>> 16) + xfrac;
  else      this.state_[0] = q0 * (this.state_[0] >>> 16) + xfrac - p0;
  if (this.state_[0] < kProbaMax) this.NextWord();
  return bit;
};
ANSDec.prototype.NextAdaptiveBit = function(s) {
  return s.Update(this.NextBit(s.Proba()));
};
ANSDec.prototype.ReadRange = function(min_range, max_range) {
  max_range = max_range + 1 - min_range;
  let s = this.state_[0];
  let value = s % max_range;
  s = Math.floor(s / max_range);
  if (s < kProbaMax) {
    this.NextWord();
    const s_lo = this.state_[0] & 0xffff;
    value = (value << 16) + s_lo;
    s = (s << 16) + Math.floor(value / max_range);
    value = value % max_range;
  }
  this.state_[0] = s;
  return value + min_range;
};
ANSDec.prototype.ReadAValue = function(stats, is_positive, pred) {
  if (this.NextAdaptiveBit(stats.zero)) return pred;
  let v = 1;
  for (let i = 0, j = 1; j <= kYCoCgMax; j <<= 1)
    if (this.NextAdaptiveBit(stats.bits[i++])) v += j;
  if (!is_positive && this.NextAdaptiveBit(stats.sign)) v = -v;
  return pred + v;
};

// ---- Load ANSEnc, stripping the leading const block ----
{
  const src = fs.readFileSync(path.join(__dirname, 'triangle-ans-enc.js'), 'utf8');
  // Find first function/class definition (after the constants block).
  const idx = src.search(/^(function|class) /m);
  if (idx < 0) throw new Error('no function/class found');
  const trimmed = src.slice(idx).replace("if (typeof module !== 'undefined')", 'if (true)');
  const fn = new Function(
    'require', 'module', 'exports', '__filename', '__dirname',
    'kProbaMax', 'kYCoCgBitDepth', 'kYCoCgMax',
    'kPreviewMinNumVertices', 'kPreviewMaxNumVertices',
    'kPreviewMinNumColors', 'kPreviewMaxNumColors',
    'kPreviewMinGridSize', 'kPreviewMaxGridSize',
    'kPreviewOpaqueProba', 'kPreviewNoiseProba',
    trimmed
  );
  const fakeModule = { exports: {} };
  fn(
    require, fakeModule, fakeModule.exports, __filename, __dirname,
    kProbaMax, kYCoCgBitDepth, kYCoCgMax,
    kPreviewMinNumVertices, kPreviewMaxNumVertices,
    kPreviewMinNumColors, kPreviewMaxNumColors,
    kPreviewMinGridSize, kPreviewMaxGridSize,
    kPreviewOpaqueProba, kPreviewNoiseProba
  );
  // Inject exported symbols into global scope for test access
  Object.assign(global, fakeModule.exports);
}

// ---- Test harness ----
let pass = 0, fail = 0;
function check(label, got, exp) {
  if (got === exp) { console.log('OK  ' + label); ++pass; }
  else { console.log('FAIL ' + label + '  got=' + got + ' exp=' + exp); ++fail; }
}
function makeReader(fn) {
  const enc = new ANSEnc();  // eslint-disable-line no-undef
  fn(enc);
  const bytes = enc.Assemble();
  let s = '';
  for (let i = 0; i < bytes.length; ++i) s += String.fromCharCode(bytes[i]);
  return new ANSDec(s);
}

// PutRange / ReadRange
for (const [v, lo, hi] of [
  [0, 0, 0], [0, 0, 1], [1, 0, 1],
  [0, 0, 255], [127, 0, 255], [255, 0, 255],
  [3, 2, 256], [2, 2, 256], [256, 2, 256], [5, 5, 5],
]) {
  check(`PutRange(${v},${lo},${hi})`, makeReader(e => e.PutRange(v, lo, hi)).ReadRange(lo, hi), v);
}

// PutBit / NextBit
for (const [b, p0] of [
  [0, kProbaMax >> 1], [1, kProbaMax >> 1],
  [0, 1], [1, 1],
  [0, kProbaMax - 1], [1, kProbaMax - 1],
]) {
  check(`PutBit(${b},${p0})`, makeReader(e => e.PutBit(b, p0)).NextBit(p0), b);
}

// Mixed sequence
{
  const dec = makeReader(enc => {
    enc.PutRange(42, 0, 255);
    enc.PutBit(1, kProbaMax >> 1);
    enc.PutRange(7, 2, 256);
    enc.PutBit(0, kProbaMax >> 2);
    enc.PutRange(200, 0, 255);
  });
  check('seq range1', dec.ReadRange(0, 255), 42);
  check('seq bit1',   dec.NextBit(kProbaMax >> 1), 1);
  check('seq range2', dec.ReadRange(2, 256), 7);
  check('seq bit2',   dec.NextBit(kProbaMax >> 2), 0);
  check('seq range3', dec.ReadRange(0, 255), 200);
}

// PutAValue / ReadAValue
for (const [value, pred, pos] of [
  [10, 0, false], [10, 10, false], [12, 10, false], [8, 10, false],
  [5, 0, true], [63, 20, true], [63, 63, false], [0, 0, false],
  [31, 31, false], [1, 0, true], [63, 0, true],
]) {
  const dec = makeReader(enc => enc.PutAValue(new ValueStats(), pos, pred, value));
  check(`PutAValue(${value},pred=${pred},pos=${pos})`,
        dec.ReadAValue(new ValueStats(), pos, pred), value);
}

// Adaptive ValueStats sequence
{
  const vals = [5, 5, 10, 5, 20, 5, 0, 63, 30];
  const enc = new ANSEnc();  // eslint-disable-line no-undef
  const st = new ValueStats();
  let pred = 0;
  for (const v of vals) { enc.PutAValue(st, false, pred, v); pred = v; }
  const bytes = enc.Assemble();
  let s = '';
  for (let i = 0; i < bytes.length; ++i) s += String.fromCharCode(bytes[i]);
  const dec = new ANSDec(s);
  const st2 = new ValueStats();
  pred = 0;
  let ok = true;
  for (const v of vals) {
    const g = dec.ReadAValue(st2, false, pred);
    if (g !== v) { ok = false; console.log(`FAIL adaptive: exp ${v} got ${g}`); }
    pred = g;
  }
  if (ok) { console.log('OK  adaptive sequence'); ++pass; } else ++fail;
}

// encodePreview round-trip: encode a hand-crafted preview, decode and verify fields.
{
  // Minimal: 4x4 grid, 2 colors, no alpha, 1 non-corner vertex
  const grid_x = 4, grid_y = 4;
  const nb_colors = 2;
  // YCoCg values, sorted non-decreasingly by cg
  const color_data = [
    { y: 20, co: 30, cg: 10, a: 1 },
    { y: 40, co: 35, cg: 25, a: 1 },
  ];
  // qpts[0..3] = corners TL,TR,BL,BR; qpts[4] = one non-corner vertex
  const preview = {
    grid_x, grid_y,
    use_noise: false,
    nb_colors,
    has_alpha: false,
    nb_pts: 1,   // one non-corner vertex
    qpts: [
      { x: 0,         y: 0,         idx: 0 },  // TL
      { x: grid_x-1,  y: 0,         idx: 1 },  // TR
      { x: 0,         y: grid_y-1,  idx: 0 },  // BL
      { x: grid_x-1,  y: grid_y-1,  idx: 1 },  // BR
      { x: 1,         y: 1,         idx: 0 },  // non-corner
    ],
  };

  // Encode
  const b64 = encodePreview(preview, color_data);
  const raw = Buffer.from(b64, 'base64');
  let s = '';
  for (let i = 0; i < raw.length; ++i) s += String.fromCharCode(raw[i]);
  const dec = new ANSDec(s);

  // Decode header
  const kPreviewMinGridSize2 = 2, kPreviewMaxGridSize2 = 256;
  const kPreviewMinNumColors2 = 2, kPreviewMaxNumColors2 = 32;
  const r_gx = dec.ReadRange(kPreviewMinGridSize2, kPreviewMaxGridSize2);
  const r_gy = dec.ReadRange(kPreviewMinGridSize2, kPreviewMaxGridSize2);
  const r_noise = dec.NextBit(kPreviewNoiseProba);
  check('preview grid_x', r_gx, grid_x);
  check('preview grid_y', r_gy, grid_y);
  check('preview use_noise', r_noise, 0);

  // Decode palette
  const r_nb = dec.ReadRange(kPreviewMinNumColors2, kPreviewMaxNumColors2);
  const r_alpha = dec.NextBit(kPreviewOpaqueProba);
  check('preview nb_colors', r_nb, nb_colors);
  check('preview has_alpha', r_alpha, 0);

  const stats_yco = new ValueStats(), stats_cg = new ValueStats();
  let py = kYCoCgMax >> 1, pco = kYCoCgMax >> 1, pcg = 0;
  for (let i = 0; i < nb_colors; ++i) {
    py  = dec.ReadAValue(stats_yco, false, py);
    pco = dec.ReadAValue(stats_yco, false, pco);
    pcg = dec.ReadAValue(stats_cg,  true,  pcg);
    check(`color[${i}].y`,  py,  color_data[i].y);
    check(`color[${i}].co`, pco, color_data[i].co);
    check(`color[${i}].cg`, pcg, color_data[i].cg);
  }

  // Decode vertex count
  const nb_non_corner = grid_x * grid_y - 4;
  const min_pts = Math.max(nb_colors - 4, 0);
  const max_pts = Math.min(nb_non_corner, kPreviewMaxNumVertices);
  const r_nb_pts = dec.ReadRange(min_pts, max_pts);
  check('preview nb_pts', r_nb_pts, 1);

  // Decode 4 corner indices
  const idx_stats = new ANSBinSymbol(2, 2);
  const decodeColorIdx = (pred) => {
    if (dec.NextAdaptiveBit(idx_stats)) {
      const tmp = dec.ReadRange(0, nb_colors - 2);
      return tmp >= pred ? tmp + 1 : tmp;
    }
    return pred;
  };
  let ci = 0;
  ci = decodeColorIdx(ci); check('corner TL', ci, preview.qpts[0].idx);
  ci = decodeColorIdx(ci); check('corner TR', ci, preview.qpts[1].idx);
  ci = decodeColorIdx(ci); check('corner BL', ci, preview.qpts[2].idx);
  ci = decodeColorIdx(ci); check('corner BR', ci, preview.qpts[3].idx);

  // Decode non-corner grid scan
  let pts_left = r_nb_pts, cells_left = nb_non_corner;
  let found_vtx = false;
  for (let y = 0; y < grid_y && pts_left > 0; ++y) {
    for (let x = 0; x < grid_x && pts_left > 0; ++x) {
      if ((x === 0 || x === grid_x-1) && (y === 0 || y === grid_y-1)) continue;
      const proba = kProbaMax - Math.floor((pts_left << 16) / cells_left);
      const bit = dec.NextBit(proba);
      if (bit) {
        ci = decodeColorIdx(ci);
        if (x === 1 && y === 1) { check('vtx(1,1).idx', ci, preview.qpts[4].idx); found_vtx = true; }
        --pts_left;
      }
      --cells_left;
    }
  }
  if (!found_vtx) { console.log('FAIL vtx(1,1) not found in scan'); ++fail; }
}

console.log(`\n=== ${fail === 0 ? 'ALL PASS' : fail + ' FAILURES'} (${pass} passed) ===`);
process.exit(fail > 0 ? 1 : 0);
