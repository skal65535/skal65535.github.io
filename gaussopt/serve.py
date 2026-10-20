#!/usr/bin/env python3
"""Local HTTP bridge for Gauss-Opt (WebGPU 3DGS Optimizer).

Serves `skal65535.github.io/` on http://localhost:<port>/ and mounts:
  - `/dataset/` -> the user-specified COLMAP `gs/` directory
  - `/datasets/jardin1_gs/` and `/datasets/jardin3_gs/` -> built-in test datasets
  - `GET /dataset/info.json` -> lists available .ply/.splat checkpoints for resuming
  - `POST /save?name=splat_05000.ply` -> writes exported .ply/.splat/logs to `<dataset>/splats_gaussopt/`

Usage:
  ./serve.py [PATH_TO_GS_DIR] [--port 8000] [--ply splats/jardin1_05000.ply] [--open]
"""

import argparse
import http.server
import json
import os
import urllib.parse
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
DEFAULT_DATASETS = {
    "jardin1_gs": Path("/Users/skal/gskal/skal65535/Hierarchical-Localization/datasets/jardin1_gs"),
    "jardin3_gs": Path("/Users/skal/gskal/skal65535/Hierarchical-Localization/datasets/jardin3/gs"),
}


def resolve_gs_dir(p: Path) -> Path:
    p = p.resolve()
    if (p / "sparse" / "cameras.bin").exists():
        return p
    if (p / "gs" / "sparse" / "cameras.bin").exists():
        return p / "gs"
    return p


def scan_plys(gs_dir: Path):
    found = []
    for root in [gs_dir, gs_dir.parent]:
        if not root.exists():
            continue
        for pat in ["*.ply", "splats/*.ply", "splats_*/*.ply", "*.splat", "splats/*.splat"]:
            for f in sorted(root.glob(pat)):
                try:
                    rel = f.relative_to(gs_dir)
                    found.append(str(rel))
                except ValueError:
                    pass
    return sorted(set(found))


class GaussOptHandler(http.server.SimpleHTTPRequestHandler):
    gs_map = {}
    default_ds = "jardin1_gs"

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
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

    def do_GET(self):
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
            if not str(target).startswith(str(root.resolve())) or not target.is_file():
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
        qs = urllib.parse.parse_qs(parsed.query)
        name = Path(qs.get("name", ["splat_out.ply"])[0]).name
        ds_key = qs.get("dataset", ["/dataset"])[0].rstrip("/")
        root = self.gs_map.get(ds_key) or next(iter(self.gs_map.values()))
        out_dir = root / "splats_gaussopt"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / name

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        out_file.write_bytes(body)
        print(f"[serve.py] Saved {out_file} ({len(body)} bytes)", flush=True)

        resp = json.dumps({"ok": True, "path": str(out_file), "bytes": len(body)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def translate_path(self, path):
        parsed = urllib.parse.urlparse(path)
        rel = urllib.parse.unquote(parsed.path).lstrip("/")
        return str((WEB_ROOT / rel).resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", help="Path to COLMAP gs/ directory (default: jardin1_gs)")
    p.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    p.add_argument("--ply", help="Optional .ply file (relative to dataset) to resume from")
    p.add_argument("--open", action="store_true", help="Open browser automatically")
    args = p.parse_args()

    gs_map = {}
    for k, v in DEFAULT_DATASETS.items():
        if v.exists():
            gs_map[f"/datasets/{k}"] = resolve_gs_dir(v)

    if args.dataset:
        user_gs = resolve_gs_dir(Path(args.dataset))
        gs_map["/dataset"] = user_gs
        default_mount = "/dataset"
    else:
        default_mount = "/datasets/jardin1_gs"
        if default_mount in gs_map:
            gs_map["/dataset"] = gs_map[default_mount]

    GaussOptHandler.gs_map = gs_map
    GaussOptHandler.default_ds = default_mount

    os.chdir(WEB_ROOT)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), GaussOptHandler)
    url = f"http://127.0.0.1:{args.port}/gaussopt/?dataset={default_mount}"
    if args.ply:
        url += f"&ply={urllib.parse.quote(args.ply)}"
    print(f"Gauss-Opt server listening on {url}", flush=True)
    for k, v in gs_map.items():
        print(f"  mounted {k} -> {v}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
