// Node.js round-trip test for triangle-ans-enc.js
// Run: node test-ans.mjs
'use strict';

import { readFileSync } from 'fs';
import { createRequire } from 'module';

// ---- Shared constants (defined once) ----
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
  const hi = (pos     < this.max_pos_) ? this.data_.charCodeAt(pos)   : 0;
  const lo = (pos + 1 < this.max_pos_) ? this.data_.charCodeAt(pos+1) : 0;
  this.state_[0] = (this.state_[0] << 16) | ((lo + 256 * hi) || 0);
};
ANSDec.prototype.NextBit = function(p0) {
  const q0 = kProbaMax - p0;
  if (p0 == 0) return 1;
  if (q0 == 0) return 0;
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

// ---- Load ANSEnc (strip const block to avoid redeclaration) ----
const src = readFileSync(new URL('./triangle-ans-enc.js', import.meta.url), 'utf8');
const marker = '// Token types';
const idx = src.indexOf(marker);
if (idx < 0) { console.error('marker not found'); process.exit(1); }
// eval in current scope so ANSEnc, encodePreview etc. become visible below
const encSrc = src.slice(idx).replace(/^if \(typeof module/, '// $&');
eval(encSrc);  // defines T_BIT, T_RANGE, ANSEnc, encodePreview, etc.

// ---- Test harness ----
let pass = 0, fail = 0;

function check(label, got, expected) {
  if (got === expected) {
    console.log(`OK  ${label}`);
    ++pass;
  } else {
    console.log(`FAIL ${label}  got=${got} expected=${expected}`);
    ++fail;
  }
}

function makeReader(fn) {
  const enc = new ANSEnc();
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
  [3, 2, 256], [2, 2, 256], [256, 2, 256],
  [5, 5, 5],
]) {
  const dec = makeReader(enc => enc.PutRange(v, lo, hi));
  check(`PutRange(${v},${lo},${hi})`, dec.ReadRange(lo, hi), v);
}

// PutBit / NextBit
for (const [b, p0] of [
  [0, kProbaMax >> 1], [1, kProbaMax >> 1],
  [0, 1], [1, 1],
  [0, kProbaMax - 1], [1, kProbaMax - 1],
]) {
  const dec = makeReader(enc => enc.PutBit(b, p0));
  check(`PutBit(${b},${p0})`, dec.NextBit(p0), b);
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
for (const [value, pred, is_positive] of [
  [10, 0, false], [10, 10, false], [12, 10, false], [8, 10, false],
  [0, 5, true], [63, 20, true], [63, 63, false], [0, 0, false],
  [31, 31, false], [1, 0, true], [63, 0, true],
]) {
  const dec = makeReader(enc => enc.PutAValue(new ValueStats(), is_positive, pred, value));
  check(`PutAValue(${value},pred=${pred},pos=${is_positive})`,
        dec.ReadAValue(new ValueStats(), is_positive, pred), value);
}

// Adaptive ValueStats sequence
{
  const vals = [5, 5, 10, 5, 20, 5, 0, 63, 30];
  const enc = new ANSEnc();
  const enc_st = new ValueStats();
  let pred = 0;
  for (const v of vals) { enc.PutAValue(enc_st, false, pred, v); pred = v; }
  const bytes = enc.Assemble();
  let s = ''; for (let i = 0; i < bytes.length; ++i) s += String.fromCharCode(bytes[i]);
  const dec = new ANSDec(s);
  const dec_st = new ValueStats();
  pred = 0;
  let ok = true;
  for (const v of vals) {
    const g = dec.ReadAValue(dec_st, false, pred);
    if (g !== v) { ok = false; console.log(`FAIL AValue adaptive: expected ${v} got ${g}`); }
    pred = g;
  }
  if (ok) { console.log('OK  AValue adaptive sequence'); ++pass; }
  else ++fail;
}

console.log(`\n=== ${fail === 0 ? 'ALL PASS' : fail + ' FAILURES'} (${pass} passed) ===`);
process.exit(fail > 0 ? 1 : 0);
