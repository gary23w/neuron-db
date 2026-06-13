"""One-endpoint HTTP server over a NeuronDB. Standard library only.

    POST /v1/{neuron}        {"message": "..."}   -> {"reply","kind","facts"}
    GET  /v1/{neuron}                              -> stats
    POST /v1/{neuron}/forget {"match": "..."}      -> prune

Optional auth: set NEURON_DB_KEY and send  Authorization: Bearer <key>.
Run:  python -m neuron_db serve --db memory.db --port 8088
"""
from __future__ import annotations
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote
from .db import NeuronDB


def make_handler(db: NeuronDB, api_key: str | None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass  # quiet

        def _send(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("access-control-allow-origin", "*")
            self.send_header("access-control-allow-headers", "authorization, content-type")
            self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _auth(self) -> bool:
            if not api_key: return True
            got = (self.headers.get("authorization") or "").replace("Bearer ", "").strip()
            return got == api_key

        def _parts(self):
            return [p for p in self.path.split("?")[0].split("/") if p]

        def do_OPTIONS(self): self._send({}, 204)

        def do_GET(self):
            p = self._parts()
            if not p: return self._send({"service": "neuron-db", "endpoint": "POST /v1/{neuron}"})
            if not self._auth(): return self._send({"error": "unauthorized"}, 401)
            if len(p) >= 2 and p[0] == "v1":
                return self._send(db.stats(unquote(p[1])[:128]))
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            p = self._parts()
            if not self._auth(): return self._send({"error": "unauthorized"}, 401)
            if len(p) < 2 or p[0] != "v1": return self._send({"error": "POST /v1/{neuron}"}, 404)
            nid = unquote(p[1])[:128]
            n = int(self.headers.get("content-length", 0) or 0)
            try: body = json.loads(self.rfile.read(n) or b"{}")
            except Exception: body = {}
            if len(p) >= 3 and p[2] == "forget":
                return self._send(db.forget(nid, body.get("match")))
            msg = (body.get("message") or "")[:4000]
            if not msg: return self._send({"error": "empty message"}, 400)
            self._send(db.turn(nid, msg))
    return H


def serve(db_path: str = "neurons.db", host: str = "127.0.0.1", port: int = 8088, max_facts: int = 500):
    db = NeuronDB(db_path, max_facts)
    key = os.environ.get("NEURON_DB_KEY")
    httpd = ThreadingHTTPServer((host, port), make_handler(db, key))
    auth = "on (NEURON_DB_KEY set)" if key else "off"
    print(f"neuron-db serving {db_path} at http://{host}:{port}  (auth {auth})")
    print(f"  POST http://{host}:{port}/v1/demo  {{'message':'my name is Gary'}}")
    try: httpd.serve_forever()
    except KeyboardInterrupt: print("\nbye."); httpd.shutdown(); db.close()
