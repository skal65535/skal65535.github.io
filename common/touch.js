// common/touch.js — pointer / touch normalization helpers.
//
// Uses Pointer Events so the same callback handles mouse, finger and pen.
// Demos should also set `touch-action: none` on the canvas in CSS (already
// done for #main-canvas in common/main.css) so the browser does not steal
// pan / pinch gestures.

// Map an event to canvas-local pixel coords (accounts for CSS scaling and DPR).
export function canvasXY(canvas, ev) {
  const r = canvas.getBoundingClientRect();
  const x = (ev.clientX - r.left) * (canvas.width  / r.width);
  const y = (ev.clientY - r.top)  * (canvas.height / r.height);
  return [x, y];
}

// Same, but normalized to [0,1].
export function canvasUV(canvas, ev) {
  const r = canvas.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width,
          (ev.clientY - r.top)  / r.height];
}

// Register pointer-down/move/up handlers. Returns a disposer.
// `handlers` is { down?, move?, up?, cancel? }. Each receives the raw
// PointerEvent (use canvasXY/canvasUV if you want positions).
export function onDrag(target, handlers) {
  let active = null;
  const onDown = ev => {
    if (active !== null) return;
    active = ev.pointerId;
    target.setPointerCapture?.(ev.pointerId);
    handlers.down?.(ev);
  };
  const onMove = ev => { if (ev.pointerId === active) handlers.move?.(ev); };
  const onEnd  = (which) => ev => {
    if (ev.pointerId !== active) return;
    active = null;
    target.releasePointerCapture?.(ev.pointerId);
    handlers[which]?.(ev);
  };
  const onUp     = onEnd('up');
  const onCancel = onEnd('cancel');
  target.addEventListener('pointerdown',  onDown);
  target.addEventListener('pointermove',  onMove);
  target.addEventListener('pointerup',    onUp);
  target.addEventListener('pointercancel', onCancel);
  return () => {
    target.removeEventListener('pointerdown',  onDown);
    target.removeEventListener('pointermove',  onMove);
    target.removeEventListener('pointerup',    onUp);
    target.removeEventListener('pointercancel', onCancel);
  };
}

// Fire `cb(ev)` on a fast second tap/click within `windowMs` and `slopPx`.
// Works for mouse-double-click AND touch-double-tap (which most mobile
// browsers do NOT synthesize dblclick for on a <canvas>).
export function onDoubleTap(target, cb, { windowMs = 320, slopPx = 30 } = {}) {
  let last = { t: 0, x: 0, y: 0 };
  const handler = ev => {
    const now = performance.now();
    const dt  = now - last.t;
    const dx  = ev.clientX - last.x, dy = ev.clientY - last.y;
    if (dt < windowMs && (dx * dx + dy * dy) < slopPx * slopPx) {
      cb(ev);
      last = { t: 0, x: 0, y: 0 };
    } else {
      last = { t: now, x: ev.clientX, y: ev.clientY };
    }
  };
  target.addEventListener('pointerdown', handler);
  return () => target.removeEventListener('pointerdown', handler);
}

// Two-finger pinch. `cb({ scale, cx, cy })` is called on each move, where
// `scale` is current-distance / start-distance and (cx, cy) is the midpoint
// in canvas-local pixel coords.
export function onPinch(canvas, cb) {
  const ptrs = new Map();
  let startDist = 0;
  const dist = () => {
    const [a, b] = [...ptrs.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  const mid  = () => {
    const [a, b] = [...ptrs.values()];
    return [(a.x + b.x) / 2, (a.y + b.y) / 2];
  };
  const onDown = ev => {
    ptrs.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    if (ptrs.size === 2) startDist = dist();
  };
  const onMove = ev => {
    if (!ptrs.has(ev.pointerId)) return;
    ptrs.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    if (ptrs.size === 2 && startDist > 0) {
      const [mx, my] = mid();
      const r = canvas.getBoundingClientRect();
      cb({
        scale: dist() / startDist,
        cx: (mx - r.left) * (canvas.width  / r.width),
        cy: (my - r.top)  * (canvas.height / r.height),
      });
    }
  };
  const onUp = ev => {
    ptrs.delete(ev.pointerId);
    if (ptrs.size < 2) startDist = 0;
  };
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup',   onUp);
  canvas.addEventListener('pointercancel', onUp);
  return () => {
    canvas.removeEventListener('pointerdown', onDown);
    canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerup',   onUp);
    canvas.removeEventListener('pointercancel', onUp);
  };
}
