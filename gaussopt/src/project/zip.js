// zip.js — A .zip of files, stored without compression (the content is JPEGs and binary floats):
// the download of a project when there is no folder to write it into. The parts are Blobs, so the
// archive is assembled without copying them; each is read once for its CRC-32. No ZIP64: up to
// 65535 files and 4 GB.
"use strict";

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes, crc = 0) {
  let c = ~crc >>> 0;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}

// MS-DOS time and date of a Date
function dosTime(d) {
  return [(d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
          ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate()];
}

// files: [[path, Blob]] -> Blob (application/zip)
export async function zipBlob(files, date = new Date()) {
  if (files.length > 0xffff) throw new Error(`zip: ${files.length} files (at most 65535 without ZIP64)`);
  const enc = new TextEncoder(), [time, day] = dosTime(date);
  const parts = [], central = [];
  let offset = 0;
  for (const [path, blob] of files) {
    const name = enc.encode(path), size = blob.size;
    const crc = crc32(new Uint8Array(await blob.arrayBuffer()));
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true);  // UTF-8 names
    local.setUint16(8, 0, true);       // stored
    local.setUint16(10, time, true);
    local.setUint16(12, day, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, size, true);
    local.setUint32(22, size, true);
    local.setUint16(26, name.length, true);
    parts.push(local.buffer, name, blob);
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true);
    cd.setUint16(6, 20, true);
    cd.setUint16(8, 0x0800, true);
    cd.setUint16(12, time, true);
    cd.setUint16(14, day, true);
    cd.setUint32(16, crc, true);
    cd.setUint32(20, size, true);
    cd.setUint32(24, size, true);
    cd.setUint16(28, name.length, true);
    cd.setUint32(42, offset, true);
    central.push(cd.buffer, name);
    offset += 30 + name.length + size;
    if (offset > 0xffffffff) throw new Error('zip: over 4 GB (no ZIP64)');
  }
  const cdSize = central.reduce((s, p) => s + p.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, cdSize, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
}
