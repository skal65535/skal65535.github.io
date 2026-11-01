// url.js — The page's URL following its controls (app.js, ../gallery): DOM-free, no imports.
"use strict";

// The URL query (location.search) once a UI change set a param to `value`, or null when it is not in
// the URL: only a param given in the URL (by the first of `names` it has) is rewritten, the others
// stay as typed, so the URL keeps reloading the page as shown. A boolean is written 1 / 0, or is there
// or not when its name is one of `flags` (read by presence: args.js parse_arg_bool); an array
// comma-separated.
export function urlAfterChange(search, names, value, flags = []) {
  const parts = search.replace(/^\?/, '').split('&').filter(Boolean);
  const nameOf = (kv) => decodeURIComponent(kv.split('=')[0]);
  const name = names.find(n => parts.some(kv => nameOf(kv) === n));
  if (!name) return null;
  const i = parts.findIndex(kv => nameOf(kv) === name);
  if (typeof value === 'boolean' && flags.includes(name)) {
    if (value) return null;
    parts.splice(i, 1);
  } else {
    const v = typeof value === 'boolean' ? (value ? '1' : '0') : Array.isArray(value) ? value.join(',') : String(value);
    parts[i] = `${name}=${encodeURIComponent(v).replace(/%2F/g, '/').replace(/%2C/g, ',')}`;
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

// A boolean URL value: off for 0, false, off and no (any case), else on
export const urlFlagOn = (v) => !['0', 'false', 'off', 'no'].includes(v.toLowerCase());

// The options of `defaults` given in `search` as <prefix><name>, of the defaults' types (the SfM panel's sfm_<name>)
export function urlOptions(defaults, search, prefix) {
  const args = new URLSearchParams(search), out = {};
  for (const [k, v] of Object.entries(defaults)) {
    const a = args.get(prefix + k);
    if (a === null) continue;
    out[k] = typeof v === 'number' ? Number(a) : typeof v === 'boolean' ? urlFlagOn(a) : a;
  }
  return out;
}

// `search` with `first` ([name, value] pairs, a null value left out) in front, any of their names it had taken out;
// values encoded as urlAfterChange's (app.js _syncUrl: dataset, ply)
export function urlWithFirst(search, first) {
  const names = new Set(first.map(([n]) => n));
  const rest = search.replace(/^\?/, '').split('&').filter(kv => kv && !names.has(decodeURIComponent(kv.split('=')[0])));
  const q = [...first.filter(([, v]) => v !== null).map(([n, v]) => `${n}=${encodeURIComponent(v).replace(/%2F/g, '/')}`), ...rest].join('&');
  return q ? `?${q}` : '';
}

// The page's URL query set to `q` (history.replaceState), unless null or the same
export function replaceSearch(q) {
  if (q !== null && q !== location.search) history.replaceState(null, '', `${location.pathname}${q}${location.hash}`);
}
