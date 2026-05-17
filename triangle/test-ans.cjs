'use strict';
const fs   = require('fs');
const path = require('path');

// Pull all shared symbols from triangle-core.js (single source of truth).
const {
  ANSBinSymbol, ValueStats, ANSDec,
  kProbaMax, kYCoCgBitDepth, kYCoCgMax,
  kPreviewMinNumVertices, kPreviewMaxNumVertices,
  kPreviewMinNumColors, kPreviewMaxNumColors,
  kPreviewMinGridSize, kPreviewMaxGridSize,
  kPreviewOpaqueProba, kPreviewNoiseProba,
} = require('./triangle-core.js');

// ---- Load ANSEnc from triangle-ans-enc.js ----
{
  const src = fs.readFileSync(path.join(__dirname, 'triangle-ans-enc.js'), 'utf8');
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
    'ANSBinSymbol', 'ValueStats',
    trimmed
  );
  const fakeModule = { exports: {} };
  fn(
    require, fakeModule, fakeModule.exports, __filename, __dirname,
    kProbaMax, kYCoCgBitDepth, kYCoCgMax,
    kPreviewMinNumVertices, kPreviewMaxNumVertices,
    kPreviewMinNumColors, kPreviewMaxNumColors,
    kPreviewMinGridSize, kPreviewMaxGridSize,
    kPreviewOpaqueProba, kPreviewNoiseProba,
    ANSBinSymbol, ValueStats
  );
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
  const grid_x = 4, grid_y = 4, nb_colors = 2;
  const color_data = [
    { y: 20, co: 30, cg: 10, a: 1 },
    { y: 40, co: 35, cg: 25, a: 1 },
  ];
  const preview = {
    grid_x, grid_y, use_noise: false, nb_colors, has_alpha: false, nb_pts: 1,
    qpts: new Int16Array([0,0,0,  grid_x-1,0,1,  0,grid_y-1,0,  grid_x-1,grid_y-1,1,  1,1,0]),
  };

  const b64 = encodePreview(preview, color_data);  // eslint-disable-line no-undef
  const raw = Buffer.from(b64, 'base64');
  let s = '';
  for (let i = 0; i < raw.length; ++i) s += String.fromCharCode(raw[i]);
  const dec = new ANSDec(s);

  const r_gx = dec.ReadRange(kPreviewMinGridSize, kPreviewMaxGridSize);
  const r_gy = dec.ReadRange(kPreviewMinGridSize, kPreviewMaxGridSize);
  check('preview grid_x',    r_gx, grid_x);
  check('preview grid_y',    r_gy, grid_y);
  check('preview use_noise', dec.NextBit(kPreviewNoiseProba), 0);
  check('preview nb_colors', dec.ReadRange(kPreviewMinNumColors, kPreviewMaxNumColors), nb_colors);
  check('preview has_alpha', dec.NextBit(kPreviewOpaqueProba), 0);

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

  const nb_non_corner = grid_x * grid_y - 4;
  const min_pts = Math.max(nb_colors - 4, 0);
  const max_pts = Math.min(nb_non_corner, kPreviewMaxNumVertices);
  check('preview nb_pts', dec.ReadRange(min_pts, max_pts), 1);

  const idx_stats = new ANSBinSymbol(2, 2);
  const decodeColorIdx = (pred) => {
    if (dec.NextAdaptiveBit(idx_stats)) {
      const tmp = dec.ReadRange(0, nb_colors - 2);
      return tmp >= pred ? tmp + 1 : tmp;
    }
    return pred;
  };
  let ci = 0;
  ci = decodeColorIdx(ci); check('corner TL', ci, preview.qpts[0*3+2]);
  ci = decodeColorIdx(ci); check('corner TR', ci, preview.qpts[1*3+2]);
  ci = decodeColorIdx(ci); check('corner BL', ci, preview.qpts[2*3+2]);
  ci = decodeColorIdx(ci); check('corner BR', ci, preview.qpts[3*3+2]);

  let pts_left = 1, cells_left = nb_non_corner, found_vtx = false;
  for (let y = 0; y < grid_y && pts_left > 0; ++y) {
    for (let x = 0; x < grid_x && pts_left > 0; ++x) {
      if ((x === 0 || x === grid_x-1) && (y === 0 || y === grid_y-1)) continue;
      const bit = dec.NextBit(kProbaMax - Math.floor((pts_left << 16) / cells_left));
      if (bit) {
        ci = decodeColorIdx(ci);
        if (x === 1 && y === 1) { check('vtx(1,1).idx', ci, preview.qpts[4*3+2]); found_vtx = true; }
        --pts_left;
      }
      --cells_left;
    }
  }
  if (!found_vtx) { console.log('FAIL vtx(1,1) not found in scan'); ++fail; }
}

console.log(`\n=== ${fail === 0 ? 'ALL PASS' : fail + ' FAILURES'} (${pass} passed) ===`);
process.exit(fail > 0 ? 1 : 0);
