// files.js — What the page tells apart in picked or dropped files (app.js, sfm_panel.js), and a file chooser.
"use strict";

export const isPhoto = (f) => /^image\/jpeg$/.test(f.type ?? '') || /\.jpe?g$/i.test(f.name ?? '');
export const isVideo = (f) => (f.type ?? '').startsWith('video/') || /\.(mp4|mov|m4v|gosfm)$/i.test(f.name ?? '');

// A file chooser for `accept` (several files with `multiple`); onFiles([File]) once some are chosen
export function chooseFiles(accept, onFiles, multiple = false) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = accept;
  input.multiple = multiple;
  input.onchange = () => { if (input.files.length) onFiles([...input.files]); };
  input.click();
}
