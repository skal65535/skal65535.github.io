// Some SVG utility functions for drawing arrows

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

function is_inside_rectangle(x, y, rect) {
  return (x >= rect.x && x < rect.x + rect.width) &&
         (y >= rect.y && y < rect.y + rect.height);
}

// helper function useful for compute line / box intersection
// for a line starting at (x0,y0) and targetting point (x1, y1),
// this find a nice intersection point with box 'rect'.
// 'penetration' ratio (>= 1.) controls how deep the point
// will be inside the rectangle.
// return the updated [x1, y1] point.
function get_arrow_endpoint(x0, y0, x1, y1, rect, penetration) {
  x1 = x1 || (rect.x + rect.width / 2);
  y1 = y1 || (rect.y + rect.height / 2);

  // intersect with box
  function intersect(v0, v1, v) {
    return clamp((v - v0) / (v1 - v0), 0., 1.);
  }

  function interpolate(a, b, x) { return a * (1. - x) + b * x; }

  function mix([v0, w0], [v1, w1], [wm, wM], a) {
    const w = interpolate(w0, w1, a);
    if (w < wm || w > wM) return [v1, w1];
    return [interpolate(v0, v1, a), w];
  }

  const rx0 = rect.x, rx1 = rect.x + rect.width;
  const ry0 = rect.y, ry1 = rect.y + rect.height;
  let a;

  a = intersect(x0, x1, rx0);
  [x1, y1] = mix([x0, y0], [x1, y1], [ry0, ry1], a);
  a = intersect(x0, x1, rx1);
  [x1, y1] = mix([x0, y0], [x1, y1], [ry0, ry1], a);

  a = intersect(y0, y1, ry0);
  [y1, x1] = mix([y0, x0], [y1, x1], [rx0, rx1], a);
  a = intersect(y0, y1, ry1);
  [y1, x1] = mix([y0, x0], [y1, x1], [rx0, rx1], a);

  x1 = interpolate(x0, x1, penetration);
  y1 = interpolate(y0, y1, penetration);
  return [x1, y1];
}

// Draws an arrow (or a line) from (x0, y0) to (x1, y1)
function svg_draw_arrow(svgContainer, arrowId, x0, y0, x1, y1, options = {}) {
  const {
    color = 'white',
    stroke = 'black',
    line_width = 3,
    head_length = 10,
    head_width = 10,
    thickness = 3,
    spike = 1.,
    drop_limit = 1e6,
  } = options;

  //          .P2
  //        P1|\
  //  P0.-----. \
  //    |        .P3
  //  P6.-----. /
  //        P5|/
  //          .P4

  const old_path = svgContainer.querySelector(`#${arrowId}`);
  if (old_path) svgContainer.removeChild(old_path);

  const angle = Math.atan2(y1 - y0, x1 - x0);
  const length = Math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
  if (length < drop_limit) return;  // arrow is too small, don't display it.

  const d_angle = Math.PI / 2.;
  const [lx, ly] = [(length - head_length) * Math.cos(angle), (length - head_length) * Math.sin(angle)];
  const [dx, dy] = [Math.cos(angle + d_angle), Math.sin(angle + d_angle)];
  const [dx2, dy2] = [Math.cos(angle + spike * d_angle), Math.sin(angle + spike * d_angle)];
  const [dx4, dy4] = [Math.cos(angle - spike * d_angle), Math.sin(angle - spike * d_angle)];
  const [P0x, P0y] = [x0 + thickness * dx, y0 + thickness * dy];
  const [P1x, P1y] = [P0x + lx, P0y + ly];
  const [P2x, P2y] = [x0 + lx + head_width * dx2, y0 + ly + head_width * dy2];
  const [P3x, P3y] = [x1, y1];
  const [P4x, P4y] = [x0 + lx + head_width * dx4, y0 + ly + head_width * dy4];
  const [P6x, P6y] = [x0 - thickness * dx, y0 - thickness * dy];
  const [P5x, P5y] = [P6x + lx, P6y + ly];

  const path_data = `M ${P0x},${P0y} ` +
                    `L ${P1x},${P1y} ` +
                    `L ${P2x},${P2y} ` +
                    `L ${P3x},${P3y} ` +
                    `L ${P4x},${P4y} ` +
                    `L ${P5x},${P5y} ` +
                    `L ${P6x},${P6y} ` +
                    `Z`;

  const path = create_svg_element('path', {
      id: arrowId,
      d: path_data,
      stroke: stroke,
      fill: color,
      'stroke-width': line_width,
      'stroke-linejoin': "round",
  });
  svgContainer.appendChild(path);
}
