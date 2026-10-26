// download.js — Hands a Blob to the browser as a file download.
"use strict";

// The URL is revoked after 60 s, once the browser has surely started reading it
export function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
}
