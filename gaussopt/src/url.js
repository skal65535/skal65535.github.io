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
