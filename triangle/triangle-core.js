'use strict';

// Format constants
const kYCoCgBitDepth = 6;
const kYCoCgMax = (1 << kYCoCgBitDepth) - 1;
const kPreviewMinNumVertices = 0;
const kPreviewMaxNumVertices = 1024;
const kPreviewMinNumColors = 2;
const kPreviewMaxNumColors = 32;
const kPreviewMinGridSize = 2;
const kPreviewMaxGridSize = 256;
const kProbaMax = 1 << 16;
const kPreviewOpaqueProba = 3 * kProbaMax / 4;
const kPreviewNoiseProba = kProbaMax / 2;
const kMaxSum = 256;
const kBaryEps = 1e-4;       // inside-triangle tolerance
const kDegenerateEps = 1e-8; // degenerate-triangle threshold
const kLumaR = 0.3, kLumaG = 0.6, kLumaB = 0.1;  // perceptual RGB weights

////////////////////////////////////////////////////////////////////////////////
// ANS
function ANSBinSymbol(p0, p1) {
  this.p0_ = p0;
  this.sum_ = p0 + p1;
}
ANSBinSymbol.prototype.Update = function(bit) {
  if (this.sum_ < kMaxSum) {
    if (!bit) ++this.p0_;
    ++this.sum_;
  }
  return bit;
}
ANSBinSymbol.prototype.Proba = function() {
  return Math.floor((this.p0_ << 16) / this.sum_);
}

function ValueStats() {
  this.zero = new ANSBinSymbol(1, 1);
  this.sign = new ANSBinSymbol(1, 1);
  this.bits = new Array(kYCoCgBitDepth);
  for (let i = 0; i < kYCoCgBitDepth; ++i) {
    this.bits[i] = new ANSBinSymbol(1, 1);
  }
  return this;
}

function ANSDec(data) {
  this.data_ = data;
  this.pos_ = 0;
  this.max_pos_ = data.length;
  this.state_ = new Uint32Array(1);
  this.state_[0] = 0;
  this.NextWord();
  this.NextWord();
}
ANSDec.prototype.NextWord = function() {
  const pos = this.pos_;
  this.pos_ = pos + 2;
  const hi = this.data_.charCodeAt(pos + 0);
  const lo = this.data_.charCodeAt(pos + 1);
  const val = (lo + 256 * hi) || 0;
  this.state_[0] = (this.state_[0] << 16) | val;
}
ANSDec.prototype.NextBit = function(p0) {
  const q0 = kProbaMax - p0;
  if (p0 == 0) return 1;
  if (q0 == 0) return 0;
  const xfrac = this.state_[0] % kProbaMax;
  const bit = (xfrac >= p0) ? 1 : 0;
  if (!bit) {
    this.state_[0] = p0 * (this.state_[0] >>> 16) + xfrac;
  } else {
    this.state_[0] = q0 * (this.state_[0] >>> 16) + xfrac - p0;
  }
  if (this.state_[0] < kProbaMax) this.NextWord();
  return bit;
}
ANSDec.prototype.NextAdaptiveBit = function(s) {
  const bit = this.NextBit(s.Proba());
  return s.Update(bit);
}
ANSDec.prototype.ReadRange = function(min_range, max_range) {
  max_range = max_range + 1 - min_range;
  let s = this.state_[0];
  let value = s % max_range;
  s /= max_range;
  if (s < kProbaMax) {
    this.NextWord();
    const s_lo = this.state_[0] & 0xffff;
    value = (value << 16) + s_lo;
    s = (s << 16) + Math.floor(value / max_range);
    value = value % max_range;
  }
  this.state_[0] = s;
  return value + min_range;
}
ANSDec.prototype.ReadAValue = function(stats, is_positive, pred) {
  if (this.NextAdaptiveBit(stats.zero)) return pred;
  let v = 1;
  for (let i = 0, j = 1; j <= kYCoCgMax; j <<= 1) {
    if (this.NextAdaptiveBit(stats.bits[i++])) v += j;
  }
  if (!is_positive && this.NextAdaptiveBit(stats.sign)) v = -v;
  return pred + v;
}

////////////////////////////////////////////////////////////////////////////////
// Preview data model
class Vtx {
  constructor(x, y, idx) { this.x = x; this.y = y; this.idx = idx; }
  eq(p) {
    const eps = 0.0001;
    return Math.abs(this.x - p.x) < eps && Math.abs(this.y - p.y) < eps;
  }
}

function Color(r, g, b, a) {
  this.r = r;
  this.g = g;
  this.b = b;
  this.a = (a > 0) ? 255 : 0;
  return this;
}

function clip8b(v) {
  return (v < 0) ? 0 : (v > 255) ? 255 : v;
}

function YCoCg_to_RGB(y, co, cg, a) {
  y  = Math.floor((y  * 255) / (64 - 1));
  cg = Math.floor((cg * 255) / (64 - 1)) - 128;
  co = Math.floor((co * 255) / (64 - 1)) - 128;
  const diff = y - cg;
  const r = clip8b(diff + co);
  const g = clip8b(   y + cg);
  const b = clip8b(diff - co);
  return new Color(r, g, b, a);
}

class Preview {
  constructor() {
    this.grid_x = 0;
    this.grid_y = 0;
    this.nb_colors = 0;
    this.nb_pts = 0;
    this.has_alpha = false;
    this.use_noise = false;
    this.txt = "";
    this.qpts = null;
    this.counts = null;
  }
  GetBinText() { return this.txt; }
  GetInfoText() {
    let info = "[";
    info += " grid = " + this.grid_x + " x " + this.grid_y;
    info += ", nb_colors = " + this.nb_colors;
    info += " nb_pts = " + this.nb_pts;
    if (this.has_alpha) info += " (with alpha)";
    info += "]";
    return info;
  }
  GetCMapText() {
    let cmap_txt = "<h5>Colormap (w/ use counts):<br/>";
    for (let i = 0; i < this.nb_colors; ++i) {
      const c = this.cmap[i];
      const tip = `Y=${c.y} Co=${c.co} Cg=${c.cg} a=${c.a > 0 ? 1 : 0} | R=${c.r} G=${c.g} B=${c.b} A=${c.a}`;
      cmap_txt += `<div title="${tip}" style='width:30px; height:20px; `;
      cmap_txt += `background-color: rgba(${c.r},${c.g},${c.b},${c.a > 0 ? 1 : 0});`;
      cmap_txt += " display: inline-block;'>";
      cmap_txt += "</div>(" + this.counts[i] + ")&nbsp;";
    }
    cmap_txt += "</h5>";
    return cmap_txt;
  }
  ReadPalette(reader) {
    this.nb_colors = reader.ReadRange(kPreviewMinNumColors, kPreviewMaxNumColors);
    this.has_alpha = reader.NextBit(kPreviewOpaqueProba);
    this.txt += this.grid_x + " " + this.grid_y + " ";
    this.txt += this.nb_colors + " " + (this.has_alpha ? 1 : 0) + "\n";
    this.cmap = new Array(this.nb_colors);
    const alpha = new ANSBinSymbol(2, 2);
    const stats_yco = new ValueStats;
    const stats_cg = new ValueStats;
    const pred = new Color(kYCoCgMax >> 1, kYCoCgMax >> 1, 0, 255);
    for (let i = 0; i < this.nb_colors; ++i) {
      if (this.has_alpha && reader.NextAdaptiveBit(alpha)) pred.a = (pred.a > 0) ? 0 : 255;
      const y = pred.r = reader.ReadAValue(stats_yco, false, pred.r);
      const co = pred.g = reader.ReadAValue(stats_yco, false, pred.g);
      const cg = pred.b = reader.ReadAValue(stats_cg, true, pred.b);
      const c = YCoCg_to_RGB(y, co, cg, pred.a);
      c.y = y; c.co = co; c.cg = cg;
      this.cmap[i] = c;
      this.txt += y + " " + co + " " + cg + " " + (pred.a > 0 ? 1 : 0) + "\n";
    }
  }
  IsCorner(x, y) {
    return ((x == 0 || x == this.grid_x - 1) && (y == 0 || y == this.grid_y - 1));
  }
  DecodeColorIdx(reader, stats, idx) {
    if (reader.NextAdaptiveBit(stats)) {
      const tmp = reader.ReadRange(0, this.nb_colors - 2);
      idx = (tmp >= idx ? tmp + 1 : tmp);
    }
    ++this.counts[idx];
    return idx;
  }
  ReadVertices(reader) {
    let pts_left = this.grid_x * this.grid_y - 4;
    const min_num_pts = Math.max(Math.max(this.nb_colors - 4, 0),
      kPreviewMinNumVertices);
    const max_num_pts = Math.min(pts_left, kPreviewMaxNumVertices);
    this.nb_pts = reader.ReadRange(min_num_pts, max_num_pts);
    const q = this.qpts = new Int16Array((this.nb_pts + 4) * 3);
    this.counts = new Array(this.nb_colors).fill(0);
    const stats_idx = new ANSBinSymbol(2, 2);
    let idx = 0;
    idx = this.DecodeColorIdx(reader, stats_idx, idx);
    q[0]=0; q[1]=0; q[2]=idx;
    idx = this.DecodeColorIdx(reader, stats_idx, idx);
    q[3]=this.grid_x - 1; q[4]=0; q[5]=idx;
    idx = this.DecodeColorIdx(reader, stats_idx, idx);
    q[6]=0; q[7]=this.grid_y - 1; q[8]=idx;
    idx = this.DecodeColorIdx(reader, stats_idx, idx);
    q[9]=this.grid_x - 1; q[10]=this.grid_y - 1; q[11]=idx;
    let k = 0;
    let den = pts_left;
    let num = this.nb_pts;
    for (let y = 0; y < this.grid_y; ++y) {
      for (let x = 0; x < this.grid_x; ++x) {
        let letter = '.';
        if (this.IsCorner(x, y)) {
          const c = q[((x != 0 ? 1 : 0) + (y != 0 ? 2 : 0)) * 3 + 2];
          letter = String.fromCharCode(c + 97);
        } else if (k < this.nb_pts) {
          const proba = kProbaMax - Math.floor((num << 16) / den);
          const bit = reader.NextBit(proba);
          if (bit) {
            idx = this.DecodeColorIdx(reader, stats_idx, idx);
            q[(k + 4)*3]=x; q[(k + 4)*3+1]=y; q[(k + 4)*3+2]=idx;
            letter = String.fromCharCode(idx + 97);
            --num;
            ++k;
          }
          --den;
        }
        this.txt += letter;
      }
      this.txt += "\n";
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Delaunay triangulation
const Delaunay = (function() {
  function Edge(p0, p1) { this.pts = [p0, p1]; }
  Edge.prototype = {
    eq: function(edge, vtx) {
      const a = this.pts, b = edge.pts;
      const a0 = vtx[a[0]], a1 = vtx[a[1]];
      const b0 = vtx[b[0]], b1 = vtx[b[1]];
      return a0.eq(b1) && a1.eq(b0);
    }
  }

  function Triangle(P0, P1, P2, vtx) {
    this.vtx = [P0, P1, P2];
    this.edges = [new Edge(P0, P1), new Edge(P1, P2), new Edge(P2, P0)];
    const circle = this.circle = {};
    const p0 = vtx[P0], p1 = vtx[P1], p2 = vtx[P2];
    const ax = p1.x - p0.x, ay = p1.y - p0.y;
    const bx = p2.x - p0.x, by = p2.y - p0.y;
    const t = (p1.x * p1.x - p0.x * p0.x + p1.y * p1.y - p0.y * p0.y);
    const u = (p2.x * p2.x - p0.x * p0.x + p2.y * p2.y - p0.y * p0.y);
    const norm = 0.5 / (ax * by - ay * bx);
    const x = circle.x = ((p2.y - p0.y) * t + (p0.y - p1.y) * u) * norm;
    const y = circle.y = ((p0.x - p2.x) * t + (p1.x - p0.x) * u) * norm;
    const dx = p0.x - x;
    const dy = p0.y - y;
    circle.r2 = dx * dx + dy * dy;
  }
  Triangle.prototype = {
    InCircle: function(x, y) {
      const dx = x - this.circle.x;
      const dy = y - this.circle.y;
      const r2 = dx * dx + dy * dy;
      // '<=' matters: enables replacing a vertex.
      return (r2 <= this.circle.r2 + 0.001);
    }
  }

  function Delaunay(width, height, pts) {
    this.width = width;
    this.height = height;
    this.triangles = null;
    this.vtx = new Array();
    this.Init(width, height);
    if (pts != null) {
      if (pts instanceof Int16Array) {
        for (let i = 0; i < pts.length; i += 3) this.Insert({x: pts[i], y: pts[i+1], idx: pts[i+2]});
      } else {
        for (const pt of pts) this.Insert(pt);
      }
    }
  }
  Delaunay.prototype = {
    NewVtx: function(x, y, idx) {
      return this.vtx.push(new Vtx(x, y, idx)) - 1;
    },
    Init: function(w, h) {
      const p0 = this.NewVtx(    0,     0, 0);
      const p1 = this.NewVtx(w - 1,     0, 0);
      const p2 = this.NewVtx(w - 1, h - 1, 0);
      const p3 = this.NewVtx(    0, h - 1, 0);
      this.triangles = [
        new Triangle(p0, p1, p2, this.vtx),
        new Triangle(p0, p2, p3, this.vtx)
      ];
    },
    Insert: function(new_pt) {
      const triangles = this.triangles;
      let tri = [];
      let edges = [];
      for (const v of this.vtx) {
        if (v.x == new_pt.x && v.y == new_pt.y) {
          v.idx = new_pt.idx;
          return;
        }
      }
      for (const t of triangles) {
        if (t.InCircle(new_pt.x, new_pt.y)) {
          edges.push(t.edges[0], t.edges[1], t.edges[2]);
        } else {
          tri.push(t);
        }
      }
      let polygons = [];
      mainLoop:
        for (const edge of edges) {
          for (const polygon of polygons) {
            if (edge.eq(polygon, this.vtx)) {
              polygons.splice(polygons.indexOf(polygon), 1);
              continue mainLoop;
            }
          }
          polygons.push(edge);
        }
      for (const edge of polygons) {
        const new_vtx = this.NewVtx(new_pt.x, new_pt.y, new_pt.idx);
        tri.push(new Triangle(edge.pts[0], edge.pts[1], new_vtx, this.vtx));
      }
      this.triangles = tri;
    },
    getTriangles: function() {
      return this.triangles.slice();
    },
  }
  return Delaunay;
})();

////////////////////////////////////////////////////////////////////////////////
// RGB -> YCoCg (encoder direction)
function RGBtoYCoCg(r, g, b, a) {
  const clamp63 = v => Math.max(0, Math.min(kYCoCgMax, v));
  const q = v => clamp63(Math.round((v + 128) * kYCoCgMax / 255));
  return {
    y:  clamp63(Math.round((2*g + r + b) / 4 * kYCoCgMax / 255)),
    co: q((r - b) / 2),
    cg: q((2*g - r - b) / 4),
    a:  a > 0 ? 1 : 0,
  };
}

// Canonical palette sort order: (cg, co, y, a).
function comparePaletteEntries(pa, pb) {
  return pa.cg !== pb.cg ? pa.cg - pb.cg :
         pa.co !== pb.co ? pa.co - pb.co :
         pa.y  !== pb.y  ? pa.y  - pb.y  : pa.a - pb.a;
}

// Barycentric coordinates of (px,py) in triangle (x0,y0)-(x1,y1)-(x2,y2).
// Returns [u,v,w] (all >= -kBaryEps if inside) or null if degenerate/outside.
function computeBary(px, py, x0, y0, x1, y1, x2, y2) {
  const d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2);
  if (Math.abs(d) < kDegenerateEps) return null;
  const u = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / d;
  const v = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / d;
  const w = 1 - u - v;
  return (u >= -kBaryEps && v >= -kBaryEps && w >= -kBaryEps) ? [u, v, w] : null;
}

if (typeof module !== 'undefined') {
  module.exports = {
    kProbaMax, kYCoCgBitDepth, kYCoCgMax, kMaxSum,
    kPreviewMinNumVertices, kPreviewMaxNumVertices,
    kPreviewMinNumColors, kPreviewMaxNumColors,
    kPreviewMinGridSize, kPreviewMaxGridSize,
    kPreviewOpaqueProba, kPreviewNoiseProba,
    kBaryEps, kDegenerateEps, kLumaR, kLumaG, kLumaB,
    ANSBinSymbol, ValueStats, ANSDec,
    Vtx, Color, clip8b, YCoCg_to_RGB, RGBtoYCoCg,
    Preview, Delaunay, comparePaletteEntries, computeBary,
  };
}
