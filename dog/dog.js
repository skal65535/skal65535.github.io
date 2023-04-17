"use strict";

////////////////////////////////////////////////////////////////////////////////
// DoG: The interesting stuff

const settings = {  // Global parameters
  Sigma: 0.5,
  K: 1.6,
  p: 30.0,
  Epsilon: 0.45,
  Phi: 3.2,
  GrayScale: false,
  original: false,
  image: null,
  gui: null,
}

function ComputeWeights() {
  let w = new Float32Array(9);
  for (let i = 0; i <= 8; ++i) {
    const x = 1. * i / settings.Sigma;
    const iK = 1. / settings.K;
    const amp = 1. / (Math.sqrt(Math.PI) * settings.Sigma);
    const alpha = amp * (1. + settings.p);
    const beta = amp * settings.p * Math.sqrt(iK);
    w[i] = alpha * Math.exp(-x * x) - beta * Math.exp(-x * x * iK);
  }
  return w;
}

////////////////////////////////////////////////////////////////////////////////
// Drag'n'Drop

function PreventDefaults (e) {
  e.preventDefault();
  e.stopPropagation();
}
function HandleDrop(e) {
  HandleFile(e.dataTransfer.files[0]);
}
function HandleFile(file) {
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = function() {
    settings.image = new Image();
    settings.image.src = reader.result;
    settings.image.onload = function() { Render(); };
  }
}
function SetupDragAndDrop() {
  const dropArea = document.getElementById('drop-area');
  function highlight(e) { dropArea.classList.add('highlight'); }
  function unhighlight(e) { dropArea.classList.remove('highlight'); }

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(
    name => { dropArea.addEventListener(name, PreventDefaults, false); }
  );
  ['dragenter', 'dragover'].forEach(
    name => { dropArea.addEventListener(name, highlight, false); }
  );
  ['dragleave', 'drop'].forEach(
    name => { dropArea.addEventListener(name, unhighlight, false); }
  );
  dropArea.addEventListener('drop', HandleDrop, false)
}

////////////////////////////////////////////////////////////////////////////////

function SetupUI() {
  settings.gui = new dat.GUI();
  settings.gui.domElement.id = 'gui';
  settings.gui.add(settings, 'Sigma', 0.01, 5., .01).name('Sigma').listen().onChange(Render);
  settings.gui.add(settings, 'K', 0.5, 4., .05).name('K').listen().onChange(Render);
  settings.gui.add(settings, 'p', 10., 200., .1).name('p').listen().onChange(Render);
  settings.gui.add(settings, 'Epsilon', 0., 1., .01).name('Epsilon').listen().onChange(Render);
  settings.gui.add(settings, 'Phi', 0.01, 10., .1).name('Phi').listen().onChange(Render);
  settings.gui.add(settings, 'GrayScale').name('grayscale').listen().onChange(Render);
  const canvas = document.getElementById('gui-container');
  canvas.appendChild(settings.gui.domElement);
}

////////////////////////////////////////////////////////////////////////////////
// Main WebGL calls

function main() {
  settings.image = new Image();
  settings.image.src = "./SF.webp";
  settings.image.onload = function() { Render(); };

  SetupDragAndDrop();
  SetupUI();
}

function CreateShader(gl, type, src_id) {
  const shader = gl.createShader(type);
  const src = document.getElementById(src_id).text;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  const compiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (!compiled) {
    const error = gl.getShaderInfoLog(shader);
    console.log('*** Shader Compile Error \'' + src_id + '\':' + error + '\n');
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function CreateShaders(gl, v_shader_id, f_shader_id) {
  const f_shader = CreateShader(gl, gl.FRAGMENT_SHADER, f_shader_id);
  const v_shader = CreateShader(gl, gl.VERTEX_SHADER, v_shader_id);
  const shader_program = gl.createProgram();
  gl.attachShader(shader_program, f_shader);
  gl.attachShader(shader_program, v_shader);
  gl.linkProgram(shader_program);
  gl.useProgram(shader_program);
  return shader_program;
}

function Render() {
  const image = settings.image;
  const canvas = document.querySelector("#main-canvas");
  let Wo = canvas.width, Ho = canvas.height;
  let W = image.width, H = image.height;
  if (W > Wo || H > Ho) {   // needs resizing?
    if (W * Ho > Wo > H) {
      H = Math.floor(H * Wo / W);
      W = Wo;
    } else {
      W = Math.floor(W * Ho / H);
      H = Ho;
    }
  }
  image.width = W;
  image.height = H;
  const gl = canvas.getContext("webgl");
  if (!gl) return;

  gl.viewport(0, 0, Wo, Ho);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  const program = CreateShaders(gl,
    "vertex-shader-2d",
    settings.original ? "fragment-shader-basic" : "fragment-shader-2d");

  // rectangle the same size as the image.
  const vtxBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vtxBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([  // x,y, u,v
     0, 0, 0., 0.,
     W, 0, 1., 0.,
     0, H, 0., 1.,
     0, H, 0., 1.,
     W, 0, 1., 0.,
     W, H, 1., 1.
  ]), gl.STATIC_DRAW);
  const vtxLocation = gl.getAttribLocation(program, "vtx");
  gl.enableVertexAttribArray(vtxLocation);
  gl.bindBuffer(gl.ARRAY_BUFFER, vtxBuffer);
  gl.vertexAttribPointer(vtxLocation, 4, gl.FLOAT,
                         /*normalize=*/false, /*stride=*/0, /*offset=*/0);

  // upload image to texture
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

  // uniforms setup
  const kView = gl.getUniformLocation(program, "u_view");
  const iSize = gl.getUniformLocation(program, "iSize");
  const kWeights = gl.getUniformLocation(program, "kWeights");
  const kThreshold = gl.getUniformLocation(program, "Threshold");
  const kGrayScale = gl.getUniformLocation(program, "GrayScale");
  gl.uniform4f(kView, 2. / Wo, -2. / Ho, W / Wo, -H / Ho);
  gl.uniform2f(iSize, 1. / W, 1. / H);
  gl.uniform1fv(kWeights, ComputeWeights());
  gl.uniform2f(kThreshold, settings.Epsilon, settings.Phi);
  gl.uniform1i(kGrayScale, settings.original ? -1 : settings.GrayScale ? 1 : 0);

  // go!
  gl.drawArrays(gl.TRIANGLES, /*offset=*/0, /*count=*/6);
}
