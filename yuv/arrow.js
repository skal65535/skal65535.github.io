// Some SVG utility functions

// Adjust the SVG element's dimensions and viewbox to match scrollable area.
function get_adjusted_svg_container(name) {
  const svg = document.getElementById(name);
  const W = document.documentElement.scrollWidth;
  const H = document.documentElement.scrollHeight;
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  return svg;
}

// Create an SVG element with the correct namespace.
function create_svg_element(tag, attributes = {}) {
  const elmt = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const key in attributes) elmt.setAttribute(key, attributes[key]);
  return elmt;
}

// Get the absolute position (including scroll) and dimensions of an HTML element.
function get_element_rectangle(name) {
  const element = document.getElementById(name);
  const rect = element.getBoundingClientRect();
  return {
      x: rect.left + window.scrollX,
      y: rect.top + window.scrollY,
      width: rect.width,
      height: rect.height
  };
}

// Draws an arrow (or a line) from (x0, y0) to (x1, y1)
function svg_draw_arrow(svgContainer, arrowId, x0, y0, x1, y1, options = {}) {
  const {
    color = 'white',
    stroke = 'black',
    line_width = 3,
    head_length = 10,
    head_width = 10
  } = options;

  const old_path = svgContainer.querySelector(`#${arrowId}`);
  if (old_path) svgContainer.removeChild(old_path);

  const angle = Math.atan2(y1 - y0, x1 - x0);
  const xb = x1 - head_length * Math.cos(angle);
  const yb = y1 - head_length * Math.sin(angle);

  const d_angle = Math.PI / 2.;
  const wing1_x = xb + (head_width / 2) * Math.cos(angle + d_angle);
  const wing1_y = yb + (head_width / 2) * Math.sin(angle + d_angle);

  const wing2_x = xb + (head_width / 2) * Math.cos(angle - d_angle);
  const wing2_y = yb + (head_width / 2) * Math.sin(angle - d_angle);

  const path_data = `M ${x0},${y0} ` +
                    `L ${xb},${yb} ` +
                    `L ${wing1_x},${wing1_y} ` +
                    `L ${x1},${y1} ` +
                    `L ${wing2_x},${wing2_y} ` +
                    `L ${xb},${yb} ` +
                    `Z`;

  const new_path = create_svg_element('path', {
      id: arrowId,
      d: path_data,
                               stroke: stroke,
      'stroke-width': line_width,
      fill: color
  });
  svgContainer.appendChild(new_path);
}
