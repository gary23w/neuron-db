from __future__ import annotations
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote
from .db import NeuronDB
def make_handler(db, api_key):
    class H(BaseHTTPRequestHandler):
        def log_message(self,*a): pass
        def _send(self,obj,status=200):
            body=json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("content-type","application/json")
            self.send_header("access-control-allow-origin","*")
            self.send_header("access-control-allow-headers","authorization, content-type")
            self.end_headers(); self.wfile.write(body)
        def _auth(self):
            if not api_key: return True
            return (self.headers.get("authorization") or "").replace("Bearer ","").strip()==api_key
        def _parts(self): return [p for p in self.path.split("?")[0].split("/") if p]
        def do_OPTIONS(self): self._send({},204)
        def do_GET(self):
            p=self._parts()
            if not p: return self._send({"service":"neuron-db","endpoint":"POST /v1/{neuron}"})
            if not self._auth(): return self._send({"error":"unauthorized"},401)
            if len(p)>=2 and p[0]=="v1": return self._send(db.stats(unquote(p[1])[:128]))
            self._send({"error":"not found"},404)
        def do_POST(self):
            p=self._parts()
            if not self._auth(): return self._send({"error":"unauthorized"},401)
            if len(p)<2 or p[0]!="v1": return self._send({"error":"POST /v1/{neuron}"},404)
            nid=unquote(p[1])[:128]
            n=int(self.headers.get("content-length",0) or 0)
            try: body=json.loads(self.rfile.read(n) or b"{}")
            except Exception: body={}
            if len(p)>=3 and p[2]=="forget": return self._send(db.forget(nid,body.get("match")))
            if len(p)>=3 and p[2]=="get":
                q=(body.get("query") or body.get("message") or "")[:4000]
                if not q: return self._send({"error":"empty query"},400)
                return self._send({"value":db.get(nid,q)})
            msg=(body.get("message") or "")[:4000]
            if not msg: return self._send({"error":"empty message"},400)
            self._send(db.turn(nid,msg))
    return H
def serve(db_path="neurons.db", host="127.0.0.1", port=8088, max_facts=500):
    db=NeuronDB(db_path,max_facts); key=os.environ.get("NEURON_DB_KEY")
    httpd=ThreadingHTTPServer((host,port),make_handler(db,key))
    print(f"neuron-db serving {db_path} at http://{host}:{port}  (auth {'on' if key else 'off'})")
    try: httpd.serve_forever()
    except KeyboardInterrupt: print("\nbye."); httpd.shutdown(); db.close()
