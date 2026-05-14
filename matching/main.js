import { mkGui }                          from '../common/controls.js';
import { recoverPose, triangulate2View } from './geometry.js';

const App = {
  cv: null,
  pendingRun: false,
  _vizAlive: null,

  state: {
    images:          [],
    imageSrcs:       [],
    imageScales:     [],
    features:        [],
    allMatches:      [],
    allCameraPoses: [],
    method:          'orb',
    orbFeatures:     500,
    loweRatio:       0.75,
    ransacThresh:    1.5,
    frustumScale:    0.3,
  },

  init() { this.ui.init(); },

  // opencv.js is loaded async — if the user dropped files before it was ready,
  // pendingRun fires the pipeline now.
  onOpenCvReady() {
    this.cv = window.cv;
    if (this.pendingRun) { this.pendingRun = false; this.run(); }
  },

  run() {
    if (!this.cv) { this.pendingRun = true; return; }
    if (this.state.images.length < 2) return;

    this.ui.showProgress(true);
    this.detectFeatures()
      .then(() => { this.ui.drawFeatures(); })
      .then(() => this.compute())
      .then(() => { this.ui.drawFeatures(); })   // redraw with matched pts in red
      .catch(err => console.error('pipeline error:', err))
      .finally(() => this.ui.showProgress(false));
  },

  // --- UI ---
  ui: {
    init() {
      const fileInput = document.getElementById('fileInput');
      fileInput.addEventListener('change', App.handleImageUpload.bind(App));

      const gui   = mkGui(document.getElementById('gui-container'), { title: 'Parameters' });
      const rerun = () => App.run();
      const reviz = () => App.visualize3D();
      gui.add(App.state, 'method', { 'ORB': 'orb', 'Harris': 'harris' })
         .name('Feature Type').onChange(rerun);
      gui.add(App.state, 'orbFeatures',  200, 3000, 50 ).name('ORB max features')   .onChange(rerun);
      gui.add(App.state, 'loweRatio',    0.6, 0.95, 0.01).name('Lowe ratio')         .onChange(rerun);
      gui.add(App.state, 'ransacThresh', 0.5, 5.0,  0.1).name('RANSAC px threshold').onChange(rerun);
      gui.add(App.state, 'frustumScale', 0.05, 1.0, 0.05).name('Frustum scale')      .onChange(reviz);
    },
    showProgress(v) {
      document.getElementById('progress-container').style.display = v ? 'block' : 'none';
    },
    updateProgress(label, pct) {
      document.getElementById('progress-label').textContent = label;
      document.getElementById('progress-bar').style.width = pct + '%';
    },
    displayImages() {
      const grid = document.getElementById('imagesGrid');
      grid.innerHTML = '';
      App.state.thumbCanvases = [];
      for (let i = 0; i < App.state.images.length; i++) {
        const srcCanvas = App.state.images[i];
        const div = document.createElement('div');
        div.className = 'image-container';
        const c = document.createElement('canvas');
        c.width  = srcCanvas.width;
        c.height = srcCanvas.height;
        c.getContext('2d').drawImage(srcCanvas, 0, 0);
        div.appendChild(c);
        grid.appendChild(div);
        App.state.thumbCanvases.push(c);
        const idx = i;
        div.addEventListener('mouseenter', () => App.ui.showPreview(idx));
      }
      if (App.state.thumbCanvases.length) App.ui.showPreview(0);
    },

    showPreview(i) {
      const thumb = App.state.thumbCanvases?.[i];
      const preview = document.getElementById('previewCanvas');
      if (!thumb || !preview) return;
      // Match the source thumbnail's size; CSS handles the contain-fit.
      preview.width  = thumb.width;
      preview.height = thumb.height;
      preview.getContext('2d').drawImage(thumb, 0, 0);
      preview.setAttribute('data-loaded', '');
      App.state.previewIdx = i;
    },

    drawFeatures() {
      const thumbs  = App.state.thumbCanvases || [];
      const matches = App.state.allMatches    || [];
      // Per-image set of feature indices that found a partner in either
      // neighbouring pair.
      const matched = thumbs.map(() => new Set());
      for (let i = 0; i < matches.length; i++) {
        for (const k of matches[i].idx1) matched[i]    ?.add(k);
        for (const k of matches[i].idx2) matched[i + 1]?.add(k);
      }
      const STYLES = {
        unmatched: { stroke: 'rgba(0, 60, 0, 0.85)',  fill: 'rgba(40, 230, 60, 0.9)'  },
        matched:   { stroke: 'rgba(80, 0, 0, 0.9)',   fill: 'rgba(230, 40, 40, 0.95)' },
      };
      for (let i = 0; i < thumbs.length; i++) {
        const c   = thumbs[i];
        const src = App.state.images[i];
        const pts = App.state.features[i]?.points;
        if (!c || !src) continue;
        const ctx = c.getContext('2d');
        ctx.drawImage(src, 0, 0);          // wipe previous overlay
        if (!pts || !pts.length) continue;
        // Stroke + fill so dots stay visible on both light and dark patches.
        const r = Math.max(2, Math.min(c.width, c.height) / 120);
        ctx.lineWidth = 1;
        for (let k = 0; k < pts.length; k++) {
          const p = pts[k];
          const s = matched[i].has(k) ? STYLES.matched : STYLES.unmatched;
          ctx.strokeStyle = s.stroke;
          ctx.fillStyle   = s.fill;
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
      // Refresh the preview pane in case the currently-shown thumb just had
      // feature dots painted on it.
      if (App.state.previewIdx != null) App.ui.showPreview(App.state.previewIdx);
    },
  },

  // --- Image handling ---
  handleImageUpload(e) {
    const files = Array.from(e.target.files);
    this.ui.updateProgress('Loading images...', 0);
    const promises = files.map(f => new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = ev => {
        const img = new Image();
        img.onload = () => {
          const r = this.resizeImage(img);
          resolve({ src: ev.target.result, canvas: r.canvas, scale: r.scale });
        };
        img.src = ev.target.result;
      };
      reader.readAsDataURL(f);
    }));

    Promise.all(promises).then(loaded => {
      this._clearFeatures?.();
      this.state.images       = loaded.map(l => l.canvas);
      this.state.imageSrcs    = loaded.map(l => l.src);
      this.state.imageScales  = loaded.map(l => l.scale);
      this.state.allMatches   = [];
      this.state.allCameraPoses = [];
      this.ui.displayImages();
      this.run();
    });
  },

  resizeImage(img, maxSize = 400) {
    const ratio = Math.min(1, maxSize / Math.max(img.width, img.height));
    const canvas = document.createElement('canvas');
    canvas.width  = Math.round(img.width  * ratio);
    canvas.height = Math.round(img.height * ratio);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return { canvas, scale: ratio };
  },

  // --- Feature detection ---
  // Each feature record is { points: [{x,y}], desc: cv.Mat, norm: int }.
  // The descriptor stays a cv.Mat so it can be fed straight to BFMatcher
  // (round-tripping through JS arrays was mangling rows for some Mats).
  _clearFeatures() {
    for (const f of this.state.features) f?.desc?.delete?.();
    this.state.features = [];
  },

  detectFeatures() {
    return new Promise((resolve, reject) => {
      if (this.state.images.length === 0) { resolve(); return; }
      this.ui.updateProgress('Detecting features...', 25);
      setTimeout(() => {
        try {
          this._clearFeatures();
          const fn = this.state.method === 'harris'
            ? this.detectHarrisCorners.bind(this)
            : this.detectORB.bind(this);
          this.state.features = this.state.images.map(c => fn(c));
          resolve();
        } catch (err) { reject(err); }
      }, 50);
    });
  },

  detectHarrisCorners(canvas) {
    const cv = this.cv;
    const NORM_L2 = cv.NORM_L2 ?? 4;
    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    const corners = new cv.Mat();
    cv.goodFeaturesToTrack(gray, corners, 500, 0.01, 10);

    const points = [];
    const flat = new Uint8Array(corners.rows * 900);
    for (let i = 0; i < corners.rows; i++) {
      const x = corners.data32F[i * 2];
      const y = corners.data32F[i * 2 + 1];
      points.push({ x, y });
      const patch = this.getDescriptor(canvas, x | 0, y | 0);
      flat.set(patch, i * 900);
    }
    // matFromArray is the safe way to populate a Mat — handles row stride
    // internally so we don't have to guess at padding.
    const desc = points.length
      ? cv.matFromArray(points.length, 900, cv.CV_8U, flat)
      : new cv.Mat();
    src.delete(); gray.delete(); corners.delete();
    return { points, desc, norm: NORM_L2 };
  },

  detectORB(canvas) {
    const cv = this.cv;
    const NORM_HAMMING = cv.NORM_HAMMING ?? 6;
    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    // Constructor takes nfeatures first. Wrap in try/catch in case the build
    // exposes only the no-arg overload.
    let orb;
    try { orb = new cv.ORB(this.state.orbFeatures | 0); }
    catch { orb = new cv.ORB(); }
    const mask = new cv.Mat();
    const kps  = new cv.KeyPointVector();
    const desc = new cv.Mat();
    orb.detectAndCompute(gray, mask, kps, desc);

    const points = [];
    for (let i = 0; i < kps.size(); i++) {
      const kp = kps.get(i);
      points.push({ x: kp.pt.x, y: kp.pt.y });
    }
    src.delete(); gray.delete(); mask.delete();
    orb.delete(); kps.delete();
    return { points, desc, norm: NORM_HAMMING };
  },

  getDescriptor(canvas, x, y) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const sx = Math.max(0, Math.min(x - 15, canvas.width  - 30));
    const sy = Math.max(0, Math.min(y - 15, canvas.height - 30));
    const id = ctx.getImageData(sx, sy, 30, 30);
    const desc = new Uint8Array(900);
    for (let i = 0, j = 0; i < id.data.length; i += 4, j++) {
      desc[j] = (id.data[i] + id.data[i + 1] + id.data[i + 2]) / 3;
    }
    return desc;
  },

  // --- Matching + RANSAC ---
  compute() {
    return new Promise((resolve, reject) => {
      if (this.state.features.length < 2) { resolve(); return; }
      this.ui.updateProgress('Matching features...', 50);
      setTimeout(() => {
        try {
          this.state.allMatches = [];
          for (let i = 0; i < this.state.features.length - 1; i++) {
            this.state.allMatches.push(
              this.matchFeatures(this.state.features[i], this.state.features[i + 1])
            );
          }
          this.computeAndVisualize();
          resolve();
        } catch (err) { reject(err); }
      }, 50);
    });
  },

  matchFeatures(f1, f2) {
    const cv = this.cv;
    const empty = { pts1: [], pts2: [], idx1: [], idx2: [] };
    if (!f1.points.length || !f2.points.length || !f1.desc || !f2.desc) return empty;
    if (f1.desc.empty() || f2.desc.empty()) return empty;

    // Pass the cv.Mat descriptors straight through. Norm is recorded on the
    // feature record at detection time (Hamming for ORB, L2 for patches).
    // Indices are tracked so cross-pair scale chaining can identify which
    // image-i feature was seen in both pair i-1 and pair i.
    const bf  = new cv.BFMatcher(f1.norm, false);
    const knn = new cv.DMatchVectorVector();
    bf.knnMatch(f1.desc, f2.desc, knn, 2);
    const pts1 = [], pts2 = [], idx1 = [], idx2 = [];
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i);
      if (pair.size() < 2) continue;
      const a = pair.get(0), b = pair.get(1);
      if (a.distance < this.state.loweRatio * b.distance) {
        pts1.push(f1.points[a.queryIdx]); idx1.push(a.queryIdx);
        pts2.push(f2.points[a.trainIdx]); idx2.push(a.trainIdx);
      }
    }
    bf.delete(); knn.delete();
    return { pts1, pts2, idx1, idx2 };
  },

  computeAndVisualize() {
    if (this.state.allMatches.length === 0) return;
    this.ui.updateProgress('Running RANSAC...', 75);

    // poses[i] = camera-to-world 4x4. recoverPose's t has ||t|| = 1 per pair,
    // so we have to rescale every new pair to the running world scale.
    //
    // Scale recovery: a feature in image i that was an inlier in BOTH pair
    // i-1 (image i is its right view) AND pair i (image i is its left view)
    // is the same physical 3D point. Pair i-1 already placed it in world
    // coords (X_known). Pair i, triangulated with unit-scale t, places it at
    // C_i + s · dir where dir = poses[i].rotation · Y_unit. The scale s that
    // makes the two agree is (X_known - C_i) · dir / |dir|². We take the
    // median over all common points — robust to a few bad correspondences.
    const poses       = [ new THREE.Matrix4() ];
    const worldPoints = [];
    let prevMap = null;                     // Map<image-i feature idx, Vector3>

    for (let i = 0; i < this.state.allMatches.length; i++) {
      const m   = this.state.allMatches[i];
      const img = this.state.images[i];
      console.log(`pair ${i}: ${m.pts1.length} good matches`);

      const pose = m.pts1.length >= 8
        ? recoverPose(m.pts1, m.pts2, img.width, img.height,
                      { ransacThresh: this.state.ransacThresh })
        : null;

      if (!pose) {
        console.warn(`pose recovery failed for pair ${i}`);
        poses.push(poses[i].clone());
        prevMap = null;                     // chain broken — no scale anchor
        continue;
      }

      const { R, t, mask, K } = pose;

      // Pull out inliers along with their feature indices.
      const inP1 = [], inP2 = [], inI1 = [], inI2 = [];
      for (let k = 0; k < m.pts1.length; k++) {
        if (!mask[k]) continue;
        inP1.push(m.pts1[k]); inI1.push(m.idx1[k]);
        inP2.push(m.pts2[k]); inI2.push(m.idx2[k]);
      }

      // Unit-scale triangulation in cam_i's frame.
      const Y_unit = triangulate2View(K, R, t, inP1, inP2);

      // Recover scale from common points with the previous pair.
      let s = 1;
      let nCommon = 0;
      if (prevMap) {
        const C_i = new THREE.Vector3().setFromMatrixPosition(poses[i]);
        const ratios = [];
        for (let k = 0; k < inP1.length; k++) {
          const Y = Y_unit[k];
          if (!Y) continue;
          const Xk = prevMap.get(inI1[k]);
          if (!Xk) continue;
          nCommon++;
          // dir = (poses[i] · Y_unit) - C_i  (world-frame direction at s=1)
          const dir = new THREE.Vector3(Y[0], Y[1], Y[2])
            .applyMatrix4(poses[i]).sub(C_i);
          const len2 = dir.lengthSq();
          if (len2 < 1e-9) continue;
          const delta = Xk.clone().sub(C_i);
          ratios.push(delta.dot(dir) / len2);
        }
        if (ratios.length) {
          ratios.sort((a, b) => a - b);
          const mid = ratios[ratios.length >> 1];
          if (isFinite(mid) && mid > 0) s = mid;
        }
        console.log(`pair ${i}: scale s=${s.toFixed(3)} (${nCommon} common pts)`);
      }

      // Build the scaled relative pose and chain it.
      const Mrel = new THREE.Matrix4().set(
        R[0], R[1], R[2], s * t[0],
        R[3], R[4], R[5], s * t[1],
        R[6], R[7], R[8], s * t[2],
        0,    0,    0,    1
      );
      poses.push(poses[i].clone().multiply(Mrel.clone().invert()));

      // Triangulated points × s, transformed to world. Build the map keyed by
      // image-{i+1} feature index so the next pair can do scale recovery.
      const nextMap = new Map();
      for (let k = 0; k < inP1.length; k++) {
        const Y = Y_unit[k];
        if (!Y) continue;
        const Xw = new THREE.Vector3(s * Y[0], s * Y[1], s * Y[2])
          .applyMatrix4(poses[i]);
        if (!isFinite(Xw.x) || Xw.length() > 1e4) continue;
        worldPoints.push(Xw.clone());
        nextMap.set(inI2[k], Xw);
      }
      prevMap = nextMap;
    }

    this.state.allCameraPoses = poses;
    this.state.worldPoints    = worldPoints;
    console.log(`triangulated ${worldPoints.length} 3D points`);
    this.visualize3D();
  },

  // --- 3D viz ---
  visualize3D() {
    const vizContainer = document.getElementById('vizContainer');
    const poses = this.state.allCameraPoses;
    if (!poses || poses.length === 0) return;
    this.ui.updateProgress('Visualizing 3D scene...', 100);

    // Kill any previous animation loop so we don't accumulate them on re-run.
    if (this._vizAlive) this._vizAlive.dead = true;
    const alive = this._vizAlive = { dead: false };

    vizContainer.innerHTML = '';
    const width  = vizContainer.clientWidth || 800;
    const height = 500;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.05, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    vizContainer.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    scene.add(new THREE.DirectionalLight(0xffffff, 0.5));
    scene.add(new THREE.AxesHelper(0.5));
    const grid = new THREE.GridHelper(4, 8, 0x888888, 0xcccccc);
    grid.rotation.x = Math.PI / 2;        // OpenCV: Z is forward, lay grid on XY
    scene.add(grid);

    const palette = [0x2a6df4, 0xf48a2a, 0x2ac06f, 0xd02af4, 0x7a2af4, 0xf42a6f];
    const FRUSTUM_SCALE = this.state.frustumScale;

    // Bounding sphere of the *cameras* (the point cloud is ignored for
    // framing — a single noisy outlier point used to drag the orbit center
    // arbitrarily far away).
    const camBox = new THREE.Box3();

    for (let i = 0; i < poses.length; i++) {
      const color = palette[i % palette.length];
      const img   = this.state.images[i];
      const frustum = makeFrustumWithImage(FRUSTUM_SCALE, img, color);
      poses[i].decompose(frustum.position, frustum.quaternion, frustum.scale);
      scene.add(frustum);

      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.03, 12, 12),
        new THREE.MeshBasicMaterial({ color })
      );
      dot.position.setFromMatrixPosition(poses[i]);
      scene.add(dot);
      camBox.expandByPoint(dot.position);
    }

    // Triangulated 3D point cloud (rendered, but not used for framing).
    const pts = this.state.worldPoints || [];
    if (pts.length) {
      const positions = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++) {
        positions[i*3]   = pts[i].x;
        positions[i*3+1] = pts[i].y;
        positions[i*3+2] = pts[i].z;
      }
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const mat = new THREE.PointsMaterial({
        color: 0x1a2240,
        size: Math.max(0.015, FRUSTUM_SCALE * 0.08),
        sizeAttenuation: true,
      });
      scene.add(new THREE.Points(geom, mat));
    }

    // Orbit centered on the camera centroid, distance sized to the camera
    // spread (plus a frustum's worth so frustums stay in view).
    const camSphere = camBox.getBoundingSphere(new THREE.Sphere());
    const centroid  = camSphere.center.clone();
    if (!isFinite(centroid.x)) centroid.set(0, 0, 0);
    const camRadius   = Math.max(camSphere.radius, FRUSTUM_SCALE);
    const initialDist = Math.max(1.5, (camRadius + FRUSTUM_SCALE) * 3);

    // orbit controls
    let dragging = false, prevX = 0, prevY = 0;
    let yaw = 0.6, pitch = 0.35, dist = initialDist;
    const updateCam = () => {
      camera.position.set(
        centroid.x + dist * Math.cos(pitch) * Math.sin(yaw),
        centroid.y + dist * Math.sin(pitch),
        centroid.z + dist * Math.cos(pitch) * Math.cos(yaw)
      );
      camera.lookAt(centroid);
    };
    const el = renderer.domElement;
    el.onmousedown   = e => { dragging = true; prevX = e.clientX; prevY = e.clientY; };
    el.onmouseup     = () => dragging = false;
    el.onmouseleave  = () => dragging = false;
    el.onmousemove   = e => {
      if (!dragging) return;
      yaw   += (e.clientX - prevX) * 0.005;
      pitch += (e.clientY - prevY) * 0.005;
      pitch = Math.max(-Math.PI/2 + 0.01, Math.min(Math.PI/2 - 0.01, pitch));
      prevX = e.clientX; prevY = e.clientY;
    };
    el.onwheel = e => {
      e.preventDefault();
      dist = Math.max(0.3, Math.min(200, dist * (e.deltaY > 0 ? 1.1 : 0.9)));
    };

    const animate = () => {
      if (alive.dead) { renderer.dispose(); return; }
      requestAnimationFrame(animate);
      updateCam();
      renderer.render(scene, camera);
    };
    animate();
  },
};

// Wireframe pyramid pointing along +Z (OpenCV convention), with the source
// photo textured onto the far plane so the image plane is visible from the
// orbital viewer. Half-extents match the implicit K (f = max(W,H), principal
// point centered), so the wireframe and the image plane line up exactly.
function makeFrustumWithImage(scale, image, color) {
  const W = image.width, H = image.height;
  const m = Math.max(W, H);
  const d = scale;
  const fw = (W / (2 * m)) * d;
  const fh = (H / (2 * m)) * d;
  const v = new Float32Array([
    0,0,0,   fw, fh, d,
    0,0,0,  -fw, fh, d,
    0,0,0,   fw,-fh, d,
    0,0,0,  -fw,-fh, d,
     fw, fh, d,  -fw, fh, d,
    -fw, fh, d,  -fw,-fh, d,
    -fw,-fh, d,   fw,-fh, d,
     fw,-fh, d,   fw, fh, d,
  ]);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(v, 3));
  const frustum = new THREE.LineSegments(
    geom, new THREE.LineBasicMaterial({ color, linewidth: 2 })
  );

  // Textured image plane on the far plane (in local frustum coords). flipY
  // is false so canvas pixel (0,0) maps to plane UV (0,0); combined with the
  // OpenCV Y-down camera frame, the image then reads upright when seen from
  // behind its own camera.
  const tex = new THREE.CanvasTexture(image);
  tex.flipY = false;
  if ('sRGBEncoding' in THREE) tex.encoding = THREE.sRGBEncoding;
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(2 * fw, 2 * fh),
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })
  );
  plane.position.set(0, 0, d);
  frustum.add(plane);

  return frustum;
}

document.addEventListener('DOMContentLoaded', () => App.init());
window.onOpenCvReady = () => App.onOpenCvReady();
