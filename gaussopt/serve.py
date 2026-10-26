#!/usr/bin/env python3
"""Local HTTP bridge for Gauss-Opt (WebGPU 3DGS Optimizer).

Serves `skal65535.github.io/` on http://localhost:<port>/ and mounts:
  - `/dataset/` -> the user-specified COLMAP `gs/` directory
  - `/datasets/jardin1_gs/`, `/datasets/jardin3_gs/`, `/datasets/jardin3-2_gs/` -> built-in test datasets
    (those present on this machine)
  - `GET /dataset/info.json` -> lists available .ply/.splat checkpoints for resuming
  - `POST /save?name=splat_05000.ply&dataset=/datasets/jardin1_gs` -> writes exported .ply/.splat/logs to `<dataset>/splats_gaussopt/`
  - project folders under `--projects DIR` (src/project/project.js for the layout):
    `GET /projects.json` lists them; `GET /project/<name>/<path>` reads a file (`?optional`: 204
    instead of 404 when it is missing, `<dir>/?list`: its entries); `PUT` writes one (directories created, written to a temporary file then renamed);
    `DELETE` removes one.
Writes (POST, PUT, DELETE) are refused unless the request comes from this server's own pages
(Origin) and names it as localhost (Host): another site open in the browser cannot write here.

Usage:
  ./serve.py [PATH_TO_GS_DIR] [--port 8000] [--ply splats/jardin1_05000.ply] [--projects DIR] [--open] [--debug]
"""

import argparse
import errno
import http.server
import json
import os
import re
import shutil
import socket
import tempfile
import sys
import traceback
import urllib.parse
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
DEFAULT_DATASETS = {
    "jardin1_gs": Path("/Users/skal/gskal/skal65535/Hierarchical-Localization/datasets/jardin1_gs"),
    "jardin3_gs": Path("/Users/skal/gskal/skal65535/Hierarchical-Localization/datasets/jardin3/gs"),
    "jardin3-2_gs": Path("/Users/skal/gskal/skal65535/Hierarchical-Localization/datasets/jardin3-2/gs"),
}


DEBUG = False
PROJECT_NAME = re.compile(r"^[\w][\w.-]*$")
CHUNK = 1 << 20


def fail(msg):
    print(f"serve.py: error: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def warn(msg):
    print(f"serve.py: warning: {msg}", file=sys.stderr, flush=True)


def resolve_gs_dir(p: Path) -> Path:
    p = p.resolve()
    if (p / "sparse" / "cameras.bin").exists():
        return p
    if (p / "gs" / "sparse" / "cameras.bin").exists():
        return p / "gs"
    return p


def check_dataset(arg: str) -> Path:
    """The gs/ directory for a command-line path, or exit with what is missing and what to do."""
    p = Path(arg).expanduser()
    if not p.exists():
        fail(f"dataset not found: {p}")
    if not p.is_dir():
        fail(f"dataset must be a directory (a COLMAP gs/ output), got a file: {p}")
    gs = resolve_gs_dir(p)
    sparse = gs / "sparse"
    if not (sparse / "cameras.bin").exists():
        hint = ""
        if (p / "sparse" / "0" / "cameras.bin").exists() or (p / "0" / "cameras.bin").exists():
            hint = (" This looks like a raw COLMAP model (sparse/0/): undistort it first, e.g. "
                    "colmap image_undistorter --image_path IMAGES --input_path MODEL --output_path OUT/gs, "
                    "or video2splat.py OUT --from undistort.")
        elif (p / "out" / "gs").is_dir():
            hint = f" Did you mean {p / 'out' / 'gs'}?"
        fail(f"no COLMAP model in {p}: expected {p}/sparse/cameras.bin or {p}/gs/sparse/cameras.bin.{hint}")
    missing = [f for f in ("images.bin", "points3D.bin") if not (sparse / f).exists()]
    if "images.bin" in missing:
        fail(f"{sparse} has cameras.bin but no images.bin: the model is incomplete")
    if missing:
        warn(f"{sparse} has no points3D.bin: training starts from a stand-in point cloud")
    if not (gs / "images").is_dir():
        warn(f"no images/ next to {sparse}: the page can show the cameras but cannot train")
    return gs


def port_number(text):
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}")
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be between 1 and 65535, got {port}")
    return port


def free_port(start):
    """First port from `start` that can be bound on 127.0.0.1, or None."""
    for port in range(start, min(start + 50, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


class Server(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        """One line per failed request instead of socketserver's traceback; the browser
        cancelling a download (closed connection) is normal and stays silent."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        print(f"[serve.py] error while serving {client_address[0]}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if DEBUG:
            traceback.print_exc()


def scan_plys(gs_dir: Path):
    found = []
    roots = [gs_dir] + ([gs_dir.parent] if gs_dir.name == "gs" else [])
    for r in roots:
        if not r.exists():
            continue
        for pat in ["*.ply", "splats/*.ply", "splats_*/*.ply", "*.splat", "splats/*.splat",
                    "*.gosnap", "splats_*/*.gosnap"]:
            for f in sorted(r.glob(pat)):
                if f.name.endswith(".mesh.ply"):  # src/mesh.js's meshes, not splats
                    continue
                try:
                    rel = f.relative_to(r)
                    found.append(str(rel))
                except ValueError:
                    pass
    return sorted(set(found))


def copy_body(rfile, out, length):
    """`length` bytes of a request body into the open file `out`; returns how many came."""
    done = 0
    while done < length:
        chunk = rfile.read(min(CHUNK, length - done))
        if not chunk:
            break
        out.write(chunk)
        done += len(chunk)
    return done


class GaussOptHandler(http.server.SimpleHTTPRequestHandler):
    gs_map = {}
    default_ds = None     # mount path, set by main()
    projects_root = None  # --projects directory, set by main()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _match_dataset(self, path_str):
        parsed = urllib.parse.urlparse(path_str)
        clean = urllib.parse.unquote(parsed.path)
        if clean == "/datasets.json":
            return "LIST", None, None
        for prefix, root in self.gs_map.items():
            if clean == prefix or clean.startswith(prefix + "/"):
                sub = clean[len(prefix):].lstrip("/")
                return "DS", root, sub
        return None, None, clean

    # ── writes: this server's own pages only ──────────────────────────────────────

    def _local_request(self):
        """The Host names this server as localhost and the Origin, when the browser sends one, is
        this server: a page from another site (or a DNS-rebound name) cannot write."""
        port = self.server.server_address[1]
        own = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in own:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in own}

    def _refuse_foreign(self):
        if self._local_request():
            return False
        self._json(403, {"ok": False, "error": "writes are only accepted from this server's own pages"})
        return True

    def _content_length(self):
        try:
            return int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(411, {"ok": False, "error": f"{self.command} needs a Content-Length header"})
            return None

    # ── project folders ────────────────────────────────────────────────────────────

    def _project_path(self, path_str):
        """(project dir, target path, rest) for /project/<name>/<rest>, or None outside of it (or
        for a name or path that leaves the project)."""
        clean = urllib.parse.unquote(urllib.parse.urlparse(path_str).path)
        if not clean.startswith("/project/") or self.projects_root is None:
            return None
        name, _, rest = clean[len("/project/"):].partition("/")
        if not PROJECT_NAME.match(name):
            return None
        pdir = (self.projects_root / name).resolve()
        target = (pdir / rest).resolve()
        if not pdir.is_relative_to(self.projects_root) or not target.is_relative_to(pdir):
            return None
        return pdir, target, rest

    def _projects_list(self):
        root = self.projects_root
        names = sorted(d.name for d in root.iterdir() if d.is_dir() and PROJECT_NAME.match(d.name)) if root and root.is_dir() else []
        self._json(200, {"root": str(root), "projects": names})

    def _project_get(self, pdir, target):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query, keep_blank_values=True)
        if "list" in query:
            if not target.is_dir():
                self._json(404, {"ok": False, "error": "no such directory"})
                return
            entries = [{"name": e.name, "kind": "directory" if e.is_dir() else "file"}
                       for e in sorted(target.iterdir()) if not e.name.startswith(".")]
            self._json(200, {"entries": entries})
            return
        if not target.is_file():
            if "optional" in query:  # a probe, not an error (the browser logs every 404)
                self.send_response(204)
                self.end_headers()
                return
            self.send_error(404, "Project file not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(target)))
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        with open(target, "rb") as fp:
            while chunk := fp.read(CHUNK):
                self.wfile.write(chunk)

    def do_PUT(self):
        p = self._project_path(self.path)
        if p is None:
            self._json(404, {"ok": False, "error": "PUT is only for /project/<name>/<path>"})
            return
        if self._refuse_foreign():
            return
        pdir, target, rest = p
        if not rest or rest.endswith("/"):
            self._json(400, {"ok": False, "error": "PUT needs a file path"})
            return
        length = self._content_length()
        if length is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".upload-")
            try:
                with os.fdopen(fd, "wb") as out:
                    got = copy_body(self.rfile, out, length)
                if got != length:
                    raise ValueError(f"upload cut short: {got} of {length} bytes")
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except ValueError as e:
            self._json(400, {"ok": False, "error": str(e)})
            return
        except OSError as e:
            self._json(500, {"ok": False, "error": f"cannot write {target}: {e.strerror or e}"})
            return
        if DEBUG:
            print(f"[serve.py] {pdir.name}/{rest} ({length} bytes)", flush=True)
        self._json(200, {"ok": True, "path": str(target), "bytes": length})

    def do_DELETE(self):
        p = self._project_path(self.path)
        if p is None:
            self._json(404, {"ok": False, "error": "DELETE is only for /project/<name>/<path>"})
            return
        if self._refuse_foreign():
            return
        pdir, target, rest = p
        if target == pdir:
            self._json(400, {"ok": False, "error": "a project is not deleted from the page"})
            return
        if not target.exists():
            self._json(404, {"ok": False, "error": "no such file"})
            return
        try:
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        except OSError as e:
            self._json(500, {"ok": False, "error": f"cannot remove {target}: {e.strerror or e}"})
            return
        self._json(200, {"ok": True})

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path == "/projects.json":
            self._projects_list()
            return
        p = self._project_path(self.path)
        if p is not None:
            self._project_get(p[0], p[1])
            return
        if urllib.parse.unquote(urllib.parse.urlparse(self.path).path).startswith("/project/"):
            self.send_error(404, "Not a project path")
            return
        kind, root, sub = self._match_dataset(self.path)
        if kind == "LIST":
            payload = {
                "default": self.default_ds,
                "datasets": {
                    k: {"mount": f"/datasets/{k}", "path": str(v), "plys": scan_plys(v)}
                    for k, v in self.gs_map.items() if k.startswith("/datasets/")
                },
            }
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if kind == "DS":
            if sub == "info.json":
                info = {
                    "path": str(root),
                    "has_sparse": (root / "sparse" / "cameras.bin").exists(),
                    "plys": scan_plys(root),
                }
                data = json.dumps(info, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            target = (root / sub).resolve()
            allowed_base = root.parent.resolve() if root.name == "gs" else root.resolve()
            if not target.is_file() and root.name == "gs":
                target = (root.parent / sub).resolve()
            if not target.is_relative_to(allowed_base) or not target.is_file():
                self.send_error(404, f"Dataset file not found: {sub}")
                return
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(str(target)))
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
            with open(target, "rb") as fp:
                while chunk := fp.read(65536):
                    self.wfile.write(chunk)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/save":
            self.send_error(404, "Only POST /save is supported")
            return
        if self._refuse_foreign():
            return
        qs = urllib.parse.parse_qs(parsed.query)
        name = Path(qs.get("name", ["splat_out.ply"])[0]).name
        ds_key = qs.get("dataset", ["/dataset"])[0].rstrip("/")
        root = self.gs_map.get(ds_key)
        if root is None:
            self._json(404, {"ok": False, "error": f"no dataset mounted at {ds_key}" if self.gs_map
                             else "no dataset mounted: start serve.py with a gs/ directory"})
            return
        out_dir = root / "splats_gaussopt"
        out_file = out_dir / name

        length = self._content_length()
        if length is None:
            return
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as out:
                got = copy_body(self.rfile, out, length)
        except OSError as e:
            print(f"[serve.py] cannot save {out_file}: {e.strerror or e}", file=sys.stderr, flush=True)
            self._json(500, {"ok": False, "error": f"cannot write {out_file}: {e.strerror or e}"})
            return
        if got != length:
            self._json(400, {"ok": False, "error": f"upload cut short: {got} of {length} bytes"})
            return
        print(f"[serve.py] Saved {out_file} ({length} bytes)", flush=True)
        self._json(200, {"ok": True, "path": str(out_file), "bytes": length})

    def _json(self, code, payload):
        resp = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def translate_path(self, path):
        parsed = urllib.parse.urlparse(path)
        rel = urllib.parse.unquote(parsed.path).lstrip("/")
        return str((WEB_ROOT / rel).resolve())

    def send_head(self):
        if not Path(self.translate_path(self.path)).is_relative_to(WEB_ROOT):  # e.g. ..%2f
            self.send_error(404, "File not found")
            return None
        return super().send_head()


def main():
    global DEBUG
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", help="Path to COLMAP gs/ directory (default: jardin1_gs)")
    p.add_argument("--port", type=port_number, default=8000, help="HTTP port (default: 8000)")
    p.add_argument("--ply", help="Optional .ply file (relative to dataset) to resume from")
    p.add_argument("--projects", default="~/gaussopt-projects",
                   help="Directory of the project folders the page can write (default: ~/gaussopt-projects)")
    p.add_argument("--open", action="store_true", help="Open browser automatically")
    p.add_argument("--debug", action="store_true", help="Print tracebacks for request errors")
    args = p.parse_args()
    DEBUG = args.debug

    gs_map = {}
    for k, v in DEFAULT_DATASETS.items():
        if v.exists():
            gs_map[f"/datasets/{k}"] = resolve_gs_dir(v)

    if args.dataset:
        user_gs = check_dataset(args.dataset)
        gs_map["/dataset"] = user_gs
        default_mount = "/dataset"
    else:
        default_mount = "/datasets/jardin1_gs"
        if default_mount in gs_map:
            gs_map["/dataset"] = gs_map[default_mount]
        else:
            default_mount = None
            warn("no dataset given and the built-in test datasets are not on this machine: "
                 "pass a gs/ directory, e.g. ./serve.py OUT/gs")

    if args.ply and "/dataset" in gs_map:
        root = gs_map["/dataset"]
        roots = [root] + ([root.parent] if root.name == "gs" else [])
        if not any((r / args.ply).is_file() for r in roots):
            known = scan_plys(root)
            warn(f"--ply {args.ply} not found under {root}" + (f"; available: {', '.join(known[:8])}" if known else ""))

    GaussOptHandler.gs_map = gs_map
    GaussOptHandler.default_ds = default_mount
    GaussOptHandler.projects_root = Path(args.projects).expanduser().resolve()  # created by the first write

    os.chdir(WEB_ROOT)
    try:
        server = Server(("127.0.0.1", args.port), GaussOptHandler)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            other = free_port(args.port + 1)
            fail(f"port {args.port} is already in use (another serve.py?)."
                 + (f" Use --port {other}, which is free." if other else " Pick another one with --port."))
        if e.errno == errno.EACCES:
            fail(f"no permission to listen on port {args.port}: use a port >= 1024, e.g. --port 8000")
        fail(f"cannot listen on 127.0.0.1:{args.port}: {e.strerror or e}")
    query = ([f"dataset={default_mount}"] if default_mount else []) + ([f"ply={urllib.parse.quote(args.ply)}"] if args.ply else [])
    url = f"http://127.0.0.1:{args.port}/gaussopt/" + ("?" + "&".join(query) if query else "")
    print(f"Gauss-Opt server listening on {url}", flush=True)
    for k, v in gs_map.items():
        print(f"  mounted {k} -> {v}", flush=True)
    if GaussOptHandler.projects_root:
        print(f"  project folders in {GaussOptHandler.projects_root}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
