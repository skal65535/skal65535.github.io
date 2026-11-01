// per_view.js — What the per-view models share (exposure.js, poses.js): a small Adam state a view (or a lens), and a
// view's neighbours in name order (a video's frames: in time) for the views without their own.
"use strict";

const B1 = 0.9, B2 = 0.999;

// An Adam state of n zeros, or starting at p (its moments zero)
export const adamState = (n) => ({ p: new Float64Array(n), m: new Float64Array(n), v: new Float64Array(n), t: 0 });
export const adamFrom = (p) => ({ ...adamState(p.length), p: Float64Array.from(p) });

// Adam on coordinate j of state s (its step count s.t already incremented) at rate lr, gradient g
export function adamStep(s, j, g, lr, eps) {
  s.m[j] = B1 * s.m[j] + (1 - B1) * g;
  s.v[j] = B2 * s.v[j] + (1 - B2) * g * g;
  s.p[j] -= lr * (s.m[j] / (1 - B1 ** s.t)) / (Math.sqrt(s.v[j] / (1 - B2 ** s.t)) + eps);
}

export const byName = (a, b) => a.localeCompare(b, undefined, { numeric: true });
const frameNumber = (name) => { const m = name.match(/(\d+)\D*$/); return m ? Number(m[1]) : null; };

// Image `name` between two of `names` (sorted byName): { a, b, t }, interpolated by the frame number in the names (else
// halfway); before the first or past the last, that one (a = b, t = 0)
export function neighbours(names, name) {
  let lo = 0, hi = names.length;  // the first name after `name`
  while (lo < hi) { const mid = (lo + hi) >> 1; if (byName(names[mid], name) < 0) lo = mid + 1; else hi = mid; }
  const a = names[Math.max(0, lo - 1)], b = names[Math.min(names.length - 1, lo)];
  const [fa, fb, f] = [a, b, name].map(frameNumber);
  const t = a === b ? 0 : fa !== null && fb !== null && f !== null && fb > fa ? Math.min(1, Math.max(0, (f - fa) / (fb - fa))) : 0.5;
  return { a, b, t };
}
