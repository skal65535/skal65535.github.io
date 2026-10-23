#!/usr/bin/env python3
"""serve.py's project folders: reads, writes, listing, and the writes it must refuse (another
site's page, another Host, paths that leave the project).

    python3 tests/serve_test.py
"""

import http.client
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SERVE = Path(__file__).resolve().parent.parent / "serve.py"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServeProjects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "projects"
        cls.port = free_port()
        cls.proc = subprocess.Popen([sys.executable, str(SERVE), "--port", str(cls.port), "--projects", str(cls.root)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", cls.port), timeout=0.1).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("serve.py did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait()
        cls.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Host": f"127.0.0.1:{self.port}", "Origin": f"http://127.0.0.1:{self.port}", **(headers or {})}
        c.putrequest(method, path, skip_host=True)
        for k, v in h.items():
            if v is not None:
                c.putheader(k, v)
        if body is not None:
            c.putheader("Content-Length", str(len(body)))
        c.endheaders()
        if body is not None:
            c.send(body)
        r = c.getresponse()
        return r.status, r.read()

    def test_write_read_list_delete(self):
        self.assertEqual(self.request("PUT", "/project/jardin/images/t0000000.jpg", b"jpeg")[0], 200)
        self.assertEqual((self.root / "jardin" / "images" / "t0000000.jpg").read_bytes(), b"jpeg")
        self.assertEqual(self.request("GET", "/project/jardin/images/t0000000.jpg"), (200, b"jpeg"))
        status, body = self.request("GET", "/project/jardin/?list")
        self.assertEqual((status, body), (200, b'{"entries": [{"name": "images", "kind": "directory"}]}'))
        self.assertIn(b'"jardin"', self.request("GET", "/projects.json")[1])
        self.assertEqual(self.request("DELETE", "/project/jardin/images/t0000000.jpg")[0], 200)
        self.assertEqual(self.request("GET", "/project/jardin/images/t0000000.jpg")[0], 404)
        self.assertEqual(self.request("GET", "/project/jardin/images/t0000000.jpg?optional"), (204, b""))
        self.assertEqual(self.request("PUT", "/project/jardin/a.txt", b"x", {"Origin": None})[0], 200)  # no Origin: not a browser page

    def test_foreign_writes_refused(self):
        self.assertEqual(self.request("PUT", "/project/p/x.txt", b"x", {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.request("PUT", "/project/p/x.txt", b"x", {"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("POST", "/save?name=a.ply", b"x", {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.request("DELETE", "/project/jardin", None, {"Origin": "http://evil.example"})[0], 403)
        self.assertFalse((self.root / "p").exists())

    def test_paths_stay_in_the_project(self):
        for path in ["/project/../x.txt", "/project/p/../../x.txt", "/project/p/%2e%2e/%2e%2e/x.txt", "/project/.hidden/x.txt"]:
            self.assertEqual(self.request("PUT", path, b"x")[0], 404, path)
        self.assertFalse((Path(self.tmp.name) / "x.txt").exists())
        self.assertEqual(self.request("PUT", "/project/p/", b"x")[0], 400)


if __name__ == "__main__":
    unittest.main()
