////////////////////////////////////////////////////////////////////////////////
// Regular Polyhedrons
//
//  defines kPolys[] with .vtx[]:vec4f as vertices and .faces[]: vec3i as faces

const icoX = 0.525731112119133606;
const icoZ = 0.850650808352039932;
const kIcoVertex = new Float32Array([  // 3 x 12 floats
  -icoX, 0., icoZ, 1.,  icoX, 0., icoZ, 1.,   -icoX, 0.,-icoZ, 1.,
   icoX, 0.,-icoZ, 1.,  0., icoZ, icoX, 1.,     0.,icoZ,-icoX, 1.,
    0.,-icoZ,icoX, 1.,  0.,-icoZ,-icoX, 1.,   icoZ,icoX,   0., 1.,
    -icoZ,icoX,0., 1.,  icoZ,-icoX, 0., 1.,   -icoZ,-icoX, 0., 1.,]);
const kIcoFaces = new Uint32Array(
[ 0, 4,1,   0,9, 4,   9, 5,4,    4,5,8,   4,8, 1,
  8,10,1,   8,3,10,   5, 3,8,    5,2,3,   2,7, 3,
  7,10,3,   7,6,10,   7,11,6,   11,0,6,   0,1, 6,
  6,1,10,   9,0,11,   9,11,2,    9,2,5,   7,2,11, ]);

const kOctaVertex = new Float32Array([
  0., -1., 0., 1.,
  1., 0., 0., 1.,   0., 0., 1., 1.,   -1., 0., 0., 1.,   0., 0., -1., 1.,
  0., 1., 0., 1. ]);
const kOctaFaces = new Uint32Array(
[ 0,2,1,  0,3,2,  0,4,3,  0,1,4,
  5,1,2,  5,2,3,  5,3,4,  5,4,1,]);

const T0 = 1. / 3.;
const T1 = Math.sqrt(8.) / 3.;
const T2 = Math.sqrt(2.) / 3.;
const T3 = Math.sqrt(6.) / 3.;
const kTetraVertex = new Float32Array([
  0., 1., 0., 1,   -T2, -T0, T3, 1,  -T2, -T0, -T3, 1,   T1, -T0, 0., 1, ]);
const kTetraFaces = new Uint32Array([ 0,1,2,   0,2,3,   0,3,1,   3,2,1,]);

const kPlaneVertex = new Float32Array([ 1., -0.1,  1., 1.,   1., -0.1, -1., 1.,
                                       -1., -0.1, -1., 1.,  -1., -0.1,  1., 1. ]);
const kPlaneFaces = new Uint32Array([ 0,2,1,  0,3,2 ]);

const kCubeVertex = new Float32Array([
    .5, .5,  .5, .5,
   -.5, .5,  .5, .5,
    .5, .5, -.5, .5,
   -.5, .5, -.5, .5,
    .5,-.5,  .5, .5,
   -.5,-.5,  .5, .5,
    .5,-.5, -.5, .5,
   -.5,-.5, -.5, .5,]);

function MakeQuad(a, b, c, d) { return [ a, b, c, c, b, d ]; }
const kCubeFaces = new Uint32Array([
  MakeQuad(0, 1, 2, 3), MakeQuad(4, 6, 5, 7),
  MakeQuad(1, 0, 5, 4), MakeQuad(2, 3, 6, 7),
  MakeQuad(0, 2, 4, 6), MakeQuad(3, 1, 7, 5),
].flat());

function MakeIdx(i, N, j, M) { return (i % N) + N * (j % M); }
function MakeTorus(R, r, N = 10, M = 10) {
  const Vtx = [];
  for (let m = 0; m < M; ++m) {
    const phi = 2. * Math.PI * m / M;
    const RR = R + r * Math.cos(phi);
    for (let n = 0; n < N; ++n) {
      const theta = 2. * Math.PI * n / N;
      const x = RR * Math.cos(theta);
      const y = r * Math.sin(phi);
      const z = RR * Math.sin(theta);
      Vtx.push([x, y, z, 1.]);
    }
  }
  const Faces = [];
  for (let m = 0; m < M; ++m) {
    for (let n = 0; n < N; ++n) {
      Faces.push(
        MakeQuad(MakeIdx(n, N, m,     M), MakeIdx(n + 1, N, m,     M),
                 MakeIdx(n, N, m + 1, M), MakeIdx(n + 1, N, m + 1, M)));
    }
  }
  return { vtx: new Float32Array(Vtx.flat()),
           faces: new Uint32Array(Faces.flat()), };
}

const kPolys = {
  'icosahedron': { vtx: kIcoVertex,   faces: kIcoFaces, },
  'tetrahedron': { vtx: kTetraVertex, faces: kTetraFaces, },
  'octahedron':  { vtx: kOctaVertex,  faces: kOctaFaces, },
  'plane':       { vtx: kPlaneVertex, faces: kPlaneFaces,  },
  'cube':        { vtx: kCubeVertex,  faces: kCubeFaces,  },
  'torus': MakeTorus(0.7, 0.2, 48, 16),
};

