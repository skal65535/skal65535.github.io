// pace.js — A long synchronous stage in the worker gives way to the event loop every `ms` of work, so its timers
// (the theatre's trace posts, trace.js through worker.js) run while it goes on. ms 0: at every call.
"use strict";

export function pacer(ms = 100) {
  let last = performance.now();
  return async () => {
    if (performance.now() - last < ms) return;
    await new Promise(r => setTimeout(r, 0));
    last = performance.now();
  };
}
