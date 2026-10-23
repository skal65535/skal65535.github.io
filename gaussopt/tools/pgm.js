// pgm.js — Minimal binary PGM (P5, 8-bit) reader for the Deno tools and tests;
// tools/gray.py converts frames to PGM (Deno has no JPEG decoder).
"use strict";

export function parsePGM(bytes) {
  let off = 0;
  const token = () => {
    for (;;) {
      while (/\s/.test(String.fromCharCode(bytes[off]))) off++;
      if (bytes[off] === 0x23) { while (bytes[off] !== 0x0a) off++; continue; }  // comment
      break;
    }
    let s = '';
    while (!/\s/.test(String.fromCharCode(bytes[off]))) s += String.fromCharCode(bytes[off++]);
    return s;
  };
  if (token() !== 'P5') throw new Error('Not a binary PGM (P5)');
  const width = Number(token()), height = Number(token()), max = Number(token());
  if (max > 255) throw new Error('16-bit PGM not supported');
  off++;  // single whitespace after maxval
  return { width, height, data: bytes.slice(off, off + width * height) };
}

export const readPGM = async (path) => parsePGM(await Deno.readFile(path));
