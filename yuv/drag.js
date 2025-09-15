//
// make an html element draggable (with re-drawing callback)
//
////////////////////////////////////////////////////////////////////////////////

function make_draggable(id, callback = {}) {
  const elmt = document.getElementById(id);
  if (!elmt) return;

  elmt.is_dragging = false;
  elmt.drag_x = 0;
  elmt.drag_y = 0;
  elmt.draw_cb = callback;

  const mouse_down = (e, pos) => {
    const elmt = e.target;
    elmt.is_dragging = true;
    const rect = elmt.getBoundingClientRect();
    elmt.drag_x = pos.clientX - rect.left;
    elmt.drag_y = pos.clientY - rect.top;
    elmt.style.cursor = 'grabbing';
    e.preventDefault();
  };
  const mouse_move = (e, pos) => {
    const elmt = e.target;
    if (!elmt.is_dragging) return;
    const x = pos.clientX - elmt.drag_x + window.scrollX;
    const y = pos.clientY - elmt.drag_y + window.scrollY;
    elmt.style.left = `${x}px`;
    elmt.style.top = `${y}px`;
    elmt.draw_cb();
  };
  const mouse_up = (e) => {
    const elmt = e.target;
    elmt.is_dragging = false;
    elmt.style.cursor = 'grab';
  }

  elmt.addEventListener('mousedown', (e) => mouse_down(e, e));
  elmt.addEventListener('mousemove', (e) => mouse_move(e, e));
  elmt.addEventListener('mouseup', mouse_up);
  elmt.addEventListener('touchstart', (e) => mouse_down(e, e.touches[0]));
  elmt.addEventListener('touchmove', (e) => mouse_move(e, e.touches[0]));
  document.addEventListener('touchend', mouse_up);
}
