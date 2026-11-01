# Gauss-Opt gallery

A decode-only page for trained scenes: the splats, the photo cameras, the camera path and the sparse
points, a fly-by along the path (`../gaussopt/src/demo.js` PathFlight: through each camera's center at its
orientation and field of view, `fly-by smoothing` (`smooth=` in the URL, 0 to 1, default 0.5) averaging them over up to 2 s of flight, 0: exactly through them; `fly-by on the path` off, or `fly-by-loose` in the URL: the app's demo flight, drifting around it). A fork of the
[splats](https://skal65535.github.io/splats/) renderer (sorted quads; the Loose Octree culls, the splats in
the frame are radix-sorted by depth as in training, `../gaussopt/DESIGN.md` §3.5 and E1). A `.ply` dropped on the page replaces the splats shown, a scene's `.glb` (the app's Save scene) the scene, another `.glb` (the app's Save mesh) the mesh.
No video is shipped, and photos only for a scene with `photos.glb` (below): only what a scene's files below hold.

* `index.html?scene=<id>&mode=fly-by|orbit|photo camera|drone` (also `visu=` a splat shape, e.g. `hard_cone`, `mesh=` one of
  the mesh menu's choices, e.g. `mesh%20only`,
  `no-cull`, `no-gui`, `canvas-only`, `w=`, `h=`; `h` lists them): a param given in the URL follows the UI
  (the scene menu, a drag into orbit), so the URL reloads the view shown.
* Drone (`../gaussopt/src/drone.js`, as in the app): `f` toggles drone / orbit, a flight key takes over from any
  view; drag = stick, arrows forward / back and yaw, `e` / `q` climb / descend, space re-centers, wheel tilts.
* World up: the scene's (`scene.glb`): the cameras' up through the scene's leveling (the app's Level).
* Cameras and path: drawn over everything, or with `cameras, path hidden` (`cameras-hidden` in the URL) hidden
  behind the mesh and the splats (where a splat is at least half opaque).
* Mesh (scenes with one; `../gaussopt/DESIGN.md` §9 E5): `splats only` (`off` in older URLs), `mesh + splats` hides the splats behind its surface (those in
  front, floaters, stay), `mesh only`, `wireframe + splats` the splats behind it hidden too, its visible edges over the rest, unfilled; `mesh shaded`: gray,
  lit from the camera, to see its shape.
* Photos (scenes with `photos.glb`): `photos in frustums` (a toggle, off by default; `photos=N` in the URL, 1 to 32: on, N photos) draws the
  photos of the `photos: nearest` cameras (8 when turned on in the menu) nearest the view in their frustums' rectangles, loaded as needed (each by its byte range in `photos.glb`, kept in the browser's Cache Storage; the whole file from a server that ignores Range, as `python3 -m http.server`); not a camera's the view is inside of or about to pass through,
  nor one seen from beyond its photo (mirrored).
* `scenes/index.json`: `[{ "id", "title", "about", "photos" }]`, in menu order; the menu shows `[id]: title`;
  `photos: true`: a `photos.glb`.
* `scenes/<id>/scene.glb`: the scene as one glTF 2.0 file (`../gaussopt/src/scene_glb.js`; any glTF viewer opens it): the splats
  (`KHR_gaussian_splatting`, SH0; a point cloud in viewers without it), the octree, the cameras (glTF cameras; their
  intrinsics, depth range and names in `extras`), the capture path (an animation through them; the page follows the
  shortest path through the cameras instead, `../gaussopt/src/path.js`, up to 100 cameras), the SfM points, the mesh
  when there is one, the background when one was learned, all under a root node turning the scene's up to glTF's +Y;
  the leveling, the up and the octree as JSON in the root's `extras.gaussopt`.
* `scenes/<id>/photos.glb` (optional): the cameras' photos as glTF images (`../gaussopt/src/scene_glb.js` writePhotosGlb), image
  `k` camera k's photo as the trainer saw it (turned as its camera: `width` x `height`), a JPEG 512 pixels on its long side. Made by `scene_photos.py SCENE_DIR` (`~/gskal/brush/photo_runs/tools`) from the run's photos or video.

Made by `train.mjs` with `SCENE=<dir>` (`~/gskal/brush/photo_runs/tools`, outside the repository): a photo set
mapped and trained in the app, then exported (with `MESH=`, the mesh in it). `publish.sh SCENES_DIR DEST` stages the page as one folder (imports rewritten to `./common`,
`./gaussopt/src`, the modules it needs) with the scenes and rsyncs it to DEST.

Locally, `scenes` may be a link to the scenes' folder (`.gitignore`): serve `gaussopt/web` with a plain
static server (`python3 -m http.server`; serve.py does not follow the link) and open `/gallery/`.
