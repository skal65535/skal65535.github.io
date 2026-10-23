// mp4.js — ISO BMFF (.mp4 / .mov) demuxer for the first video track (DESIGN-sfm §5.1, §5.7):
// the sample table (offset, size, decode and presentation time, key flag) and a WebCodecs
// VideoDecoder configuration (codec string, avcC / hvcC description, coded size, rotation) for
// H.264, HEVC and VP9.
// Reads through `read(offset, length) -> Promise<Uint8Array>`, so a browser File and a Deno file
// work alike, and only the boxes are read, not the media data.
"use strict";

const CONTAINERS = new Set(['moov', 'trak', 'mdia', 'minf', 'stbl', 'edts', 'dinf']);
const fourcc = (b, o) => String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);

// Browser File / Blob reader
export const blobReader = (blob) => ({
  size: blob.size,
  read: async (offset, length) => new Uint8Array(await blob.slice(offset, offset + length).arrayBuffer()),
});

// Child boxes of a box payload: [{ type, start (payload offset in b), end }]
function boxes(b, start = 0, end = b.length) {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength), out = [];
  for (let o = start; o + 8 <= end;) {
    let size = dv.getUint32(o), header = 8;
    if (size === 1) { size = Number(dv.getBigUint64(o + 8)); header = 16; }
    else if (size === 0) size = end - o;
    if (size < header || o + size > end) break;
    out.push({ type: fourcc(b, o + 4), start: o + header, end: o + size });
    o += size;
  }
  return out;
}

// Locate and read the top-level moov box (it can sit after the media data)
async function readMoov({ size, read }) {
  for (let o = 0; o + 8 <= size;) {
    const h = await read(o, 16), dv = new DataView(h.buffer, h.byteOffset, h.byteLength);
    let len = dv.getUint32(0), header = 8;
    if (len === 1) { len = Number(dv.getBigUint64(8)); header = 16; }
    else if (len === 0) len = size - o;
    if (len < header) break;
    if (fourcc(h, 4) === 'moov') return read(o + header, len - header);
    o += len;
  }
  throw new Error('not an MP4 / MOV file (no moov box)');
}

// Map of box type -> [payload], recursively through the container boxes of a trak
function index(b, start, end, out = new Map()) {
  for (const box of boxes(b, start, end)) {
    if (!out.has(box.type)) out.set(box.type, []);
    out.get(box.type).push(b.subarray(box.start, box.end));
    if (CONTAINERS.has(box.type)) index(b, box.start, box.end, out);
  }
  return out;
}

const view = (p) => new DataView(p.buffer, p.byteOffset, p.byteLength);
const hex2 = (v) => v.toString(16).padStart(2, '0');

// RFC 6381 codec string of an hvcC record
function hevcCodec(type, c) {
  const space = ['', 'A', 'B', 'C'][c[1] >> 6], tier = c[1] & 0x20 ? 'H' : 'L', profile = c[1] & 0x1f;
  let compat = view(c).getUint32(2), rev = 0;
  for (let k = 0; k < 32; k++, compat >>>= 1) rev = (rev << 1) | (compat & 1);
  const cons = [...c.subarray(6, 12)];
  while (cons.length && cons.at(-1) === 0) cons.pop();
  return [`${type}.${space}${profile}`, (rev >>> 0).toString(16), `${tier}${c[12]}`, ...cons.map(v => v.toString(16))].join('.');
}

// VP9 codec string (vp09.PP.LL.DD): level and bit depth from the vpcC record, the profile from the
// first frame's header, which the decoder follows (some muxers write a wrong one: jardin10.mp4's
// vpcC says profile 1, 4:4:4 for a profile 0, 4:2:0 stream)
function vp9Codec(c, frame) {
  const profile = ((frame >> 5) & 1) | (((frame >> 4) & 1) << 1), depth = profile < 2 ? 8 : c[6] >> 4;
  return ['vp09', profile, c[5], depth].map((v, k) => k ? String(v).padStart(2, '0') : v).join('.');
}

// Sample description: codec string, coded size and decoder description
function sampleEntry(stsd) {
  const entry = boxes(stsd, 8)[0];
  if (!entry) throw new Error('empty sample description');
  const dv = view(stsd), type = fourcc(stsd, entry.start - 4);
  const width = dv.getUint16(entry.start + 24), height = dv.getUint16(entry.start + 26);
  const children = boxes(stsd, entry.start + 78, entry.end);
  const config = (t) => children.find(c => c.type === t);
  let codec = type, description = null;
  if (type === 'avc1' || type === 'avc3') {
    const c = config('avcC');
    if (c) { description = stsd.slice(c.start, c.end); codec = `${type}.${hex2(description[1])}${hex2(description[2])}${hex2(description[3])}`; }
  } else if (type === 'hvc1' || type === 'hev1') {
    const c = config('hvcC');
    if (c) { description = stsd.slice(c.start, c.end); codec = hevcCodec(type, description); }
  } else if (type === 'vp09') {
    const c = config('vpcC');
    if (!c) throw new Error('vp09 without a vpcC box');
    description = stsd.slice(c.start, c.end);  // for vp9Codec in demux(); VP9 decoders take none
  } else if (type === 'av01') {
    throw new Error('av01 in MP4 is not supported yet; transcode to H.264');
  }
  return { type, codec, codedWidth: width, codedHeight: height, description };
}

// Rotation in degrees (0, 90, 180, 270) from the tkhd display matrix
function rotation(tkhd) {
  const dv = view(tkhd), m = tkhd[0] === 1 ? 52 : 40;
  const a = dv.getInt32(m) / 65536, b = dv.getInt32(m + 4) / 65536;
  return (Math.round(Math.atan2(b, a) * 180 / Math.PI / 90) * 90 + 360) % 360;
}

// Sample table: [{ offset, size, dts, pts, key }] in decode order, times in timescale units
function sampleTable(t, mediaTime) {
  const one = (k) => t.get(k)?.[0];
  if (!one('stsz')) throw new Error(one('stz2') ? 'compact sample sizes (stz2) are not supported; remux, e.g. ffmpeg -i in.mp4 -c copy out.mp4' : 'no sample sizes (stsz)');
  for (const k of ['stsc', 'stts']) if (!one(k)) throw new Error(`no sample table box ${k}`);
  if (!one('stco') && !one('co64')) throw new Error('no chunk offsets (stco / co64)');
  const stsz = view(one('stsz')), fixed = stsz.getUint32(4), n = stsz.getUint32(8);
  const sizes = Array.from({ length: n }, (_, k) => fixed || stsz.getUint32(12 + 4 * k));
  const chunks = [];
  if (one('stco')) { const d = view(one('stco')); for (let k = 0; k < d.getUint32(4); k++) chunks.push(d.getUint32(8 + 4 * k)); }
  else { const d = view(one('co64')); for (let k = 0; k < d.getUint32(4); k++) chunks.push(Number(d.getBigUint64(8 + 8 * k))); }
  const samples = new Array(n);
  // chunk layout (stsc): runs of chunks with the same number of samples
  const stsc = view(one('stsc')), runs = stsc.getUint32(4);
  let s = 0;
  for (let r = 0; r < runs; r++) {
    const first = stsc.getUint32(8 + 12 * r) - 1, per = stsc.getUint32(12 + 12 * r);
    const last = r + 1 < runs ? stsc.getUint32(20 + 12 * r) - 1 : chunks.length;
    for (let c = first; c < last && s < n; c++) {
      let off = chunks[c];
      for (let k = 0; k < per && s < n; k++, s++) { samples[s] = { offset: off, size: sizes[s] }; off += sizes[s]; }
    }
  }
  // decode times (stts), composition offsets (ctts, signed in version 1 and in practice)
  const stts = view(one('stts'));
  let dts = 0;
  s = 0;
  for (let e = 0; e < stts.getUint32(4); e++) {
    const count = stts.getUint32(8 + 8 * e), delta = stts.getUint32(12 + 8 * e);
    for (let k = 0; k < count && s < n; k++, s++) { samples[s].dts = dts; dts += delta; }
  }
  for (const x of samples) x.pts = x.dts - mediaTime;
  if (one('ctts')) {
    const ctts = view(one('ctts'));
    s = 0;
    for (let e = 0; e < ctts.getUint32(4); e++) {
      const count = ctts.getUint32(8 + 8 * e), offset = ctts.getInt32(12 + 8 * e);
      for (let k = 0; k < count && s < n; k++, s++) samples[s].pts += offset;
    }
  }
  if (one('stss')) {
    const stss = view(one('stss'));
    for (const x of samples) x.key = false;
    for (let k = 0; k < stss.getUint32(4); k++) samples[stss.getUint32(8 + 4 * k) - 1].key = true;
  } else {
    for (const x of samples) x.key = true;
  }
  return samples;
}

// Returns { codec, codedWidth, codedHeight, description, rotation, timescale, duration (s),
//           samples: [{ offset, size, dts, pts, key }] (decode order; pts in timescale units) }
export async function demux(reader) {
  const moov = await readMoov(reader);
  const remux = 'remux it, e.g. ffmpeg -i in.mp4 -c copy out.mp4';
  if (boxes(moov).some(b => b.type === 'mvex')) throw new Error(`fragmented MP4 is not supported: ${remux}`);
  for (const trak of boxes(moov).filter(b => b.type === 'trak')) {
    const t = index(moov, trak.start, trak.end);
    const hdlr = t.get('hdlr')?.[0];
    if (!hdlr || fourcc(hdlr, 8) !== 'vide') continue;
    const mdhd = t.get('mdhd')[0], md = view(mdhd);
    const timescale = mdhd[0] === 1 ? md.getUint32(20) : md.getUint32(12);
    // first edit that maps media time: presentation starts there (B-frame delay)
    let mediaTime = 0;
    const elst = t.get('elst')?.[0];
    if (elst) {
      const e = view(elst), v1 = elst[0] === 1;
      for (let k = 0, o = 8; k < e.getUint32(4); k++, o += v1 ? 20 : 12) {
        const mt = v1 ? Number(e.getBigInt64(o + 8)) : e.getInt32(o + 4);
        if (mt >= 0) { mediaTime = mt; break; }
      }
    }
    const samples = sampleTable(t, mediaTime);
    if (!samples.length) throw new Error(`no samples in the video track (fragmented MP4?): ${remux}`);
    const last = samples.reduce((m, x) => Math.max(m, x.pts), 0);
    const frame = samples.length > 1 ? (samples.at(-1).dts - samples[0].dts) / (samples.length - 1) : 0;
    const entry = sampleEntry(t.get('stsd')[0]);
    if (entry.type === 'vp09') {
      entry.codec = vp9Codec(entry.description, (await reader.read(samples[0].offset, 1))[0]);
      entry.description = null;
    }
    return { ...entry, rotation: rotation(t.get('tkhd')[0]), timescale, duration: (last + frame) / timescale, samples };
  }
  throw new Error('no video track');
}
