// Product: view paper plane

"use strict;"

function Oops(e) {
  console.log(e);
  throw Error(e);
}

const show_triangles = true;  // TODO: later

function CreateShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  const compiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (!compiled) {
    console.log('Shader compile error: ' + gl.getShaderInfoLog(shader));
  }
  return shader;
}

function initGL(ctx) {
  const v_shader_src = "                                         \
    #version 300 es\n                                            \
    precision mediump float;\n                                   \
                                                                 \
    uniform mat4 proj_mtx;                                       \
    uniform mat4 view_mtx;                                       \
    uniform mat4 transf_mtx;                                     \
    uniform vec3 light_dir;                                      \
                                                                 \
    in vec3 a_position;                                          \
    out vec4 color;                                              \
    const vec3 kVtx[6] = vec3[](                                 \
      vec3( 1.00,  0.0,  0.), vec3(-1.05, -1.1,  .1),            \
      vec3(-1.00, -0.2,  0.), vec3(-1.00,    0,  .5),            \
      vec3(-1.00,  0.2,  0.), vec3(-1.05,  1.1,  .1));           \
    const ivec3 kFaces[4] =                                      \
      ivec3[4](ivec3(0, 1, 2), ivec3(0, 2, 3),                   \
               ivec3(0, 3, 4), ivec3(0, 4, 5));                  \
    const vec4 kColors[3] =                                      \
      vec4[](vec4(1., .0, 0., 1.), vec4(0., .2, .9, 1.),         \
             vec4(1., 1., .9, 1.));                              \
                                                                 \
    vec3 to_vec3(vec4 v) { return vec3(v.x, v.y, v.z); }         \
    void main(void) {                                            \
      ivec3 f = kFaces[gl_VertexID / 3];                         \
      vec3 v0 = kVtx[f.x], v1 = kVtx[f.y], v2 = kVtx[f.z];       \
      vec4 u = transf_mtx * vec4(v1 - v0, 0.);                   \
      vec4 v = transf_mtx * vec4(v2 - v0, 0.);                   \
      vec3 n = cross(to_vec3(u), to_vec3(v));                    \
      float light = max(.1, dot(n, light_dir));                  \
      vec4 diffuse = vec4(.5, .3, .5, 1.);                       \
      vec4 albedo = .3 * kColors[gl_VertexID % 3];               \
      color = (1. - light) * albedo + light * diffuse;           \
                                                                 \
      vec4 vtx = vec4(kVtx[f[gl_VertexID % 3]], 1.);             \
      gl_Position = proj_mtx * view_mtx * transf_mtx * vtx;      \
    }                                                            \
  ";
  const f_shader_src = "                           \
    #version 300 es\n                              \
    precision mediump float;                       \
                                                   \
    in vec4 color;                                 \
    out vec4 o_color;                              \
    void main(void) {                              \
      o_color = color;                             \
    }                                              \
  ";
  // TODO(skal): trap errors
  const f_shader = CreateShader(ctx, ctx.FRAGMENT_SHADER, f_shader_src);
  const v_shader = CreateShader(ctx, ctx.VERTEX_SHADER, v_shader_src);
  const shader_program = ctx.createProgram();
  ctx.attachShader(shader_program, f_shader);
  ctx.attachShader(shader_program, v_shader);
  ctx.linkProgram(shader_program);
  if (!ctx.getProgramParameter(shader_program, ctx.LINK_STATUS)) {
    const info = ctx.getProgramInfoLog(shader_program);
    throw new Error(`Could not compile WebGL program. \n\n${info}`);
  }
  ctx.useProgram(shader_program);
  return shader_program;
}

class GyroView {
  constructor(canvas) {
    this.W = canvas.width;
    this.H = canvas.height;
    this.gl = canvas.getContext("webgl2");
    if (!this.gl) {
      alert("Could not initialise WebGL2.");
      return null;
    }
    this.shader_program = initGL(this.gl);
  }

  set_uniform_matrix(name, transpose, value) {
    const loc = this.gl.getUniformLocation(this.shader_program, name);
    this.gl.uniformMatrix4fv(loc, transpose, value);
  }

  setup_uniforms(gl, filter) {  // uniforms matrices
    // light-dir
    const light_loc = gl.getUniformLocation(this.shader_program, "light_dir");
    gl.uniform3f(light_loc, 1., 1., -1.);

    const fov = 90.;
    const fx = 1. / Math.tan(fov * Math.PI / 360.);
    const fy = this.W / this.H * fx;
    const proj_matrix = perspective(fx, fy, .1, 100.);  // TODO
    const view_matrix = look_at([-3, 0, -1], [0., 0., 0.], [0., 0., -1.]);
    const transf_matrix = new Float32Array(filter.to_dcm().flat());
    this.set_uniform_matrix("proj_mtx", true, proj_matrix);
    this.set_uniform_matrix("view_mtx", true, view_matrix);
    this.set_uniform_matrix("transf_mtx", false, transf_matrix);
  }

  async render(imu) {
    const gl = this.gl;
    if (gl == null) return;

    gl.useProgram(this.shader_program);
    gl.viewport(0, 0, this.W, this.H);
    this.setup_uniforms(gl, imu.filter);

    gl.clearColor(0., 0., 0.2, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 3 * 4);
/*
    if (show_triangles) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ZERO, gl.ZERO);
      gl.bindVertexArray(this.vtx_buffer);
      for (let i = 0; i < kPlaneIdx.length; i += 3) {
        gl.drawElements(gl.LINE_LOOP, 3, gl.UNSIGNED_SHORT, i * 2);
      }
    }
*/
  }
}
