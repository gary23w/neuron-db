from __future__ import annotations
import os, hmac, hashlib, base64, json, sqlite3, threading, time
from typing import Optional
from .neuron import _content, _stem
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_AESGCM = True
except Exception:
    _HAS_AESGCM = False
def _hkdf(key, salt, info, n=32):
    prk=hmac.new(salt,key,hashlib.sha256).digest(); out=b""; t=b""; i=1
    while len(out)<n:
        t=hmac.new(prk,t+info+bytes([i]),hashlib.sha256).digest(); out+=t; i+=1
    return out[:n]
def aead_encrypt(key, plaintext, aad=b""):
    nonce=os.urandom(12)
    if _HAS_AESGCM:
        return b"\x01"+nonce+AESGCM(_hkdf(key,b"neuron-aesgcm",b"v1",32)).encrypt(nonce,plaintext,aad)
    ek=_hkdf(key,nonce,b"enc"); mk=_hkdf(key,nonce,b"mac"); ks=b""; c=0
    while len(ks)<len(plaintext): ks+=hmac.new(ek,nonce+c.to_bytes(8,"big"),hashlib.sha256).digest(); c+=1
    ct=bytes(a^b for a,b in zip(plaintext,ks))
    tag=hmac.new(mk,b"\x00"+nonce+aad+ct,hashlib.sha256).digest()[:16]
    return b"\x00"+nonce+tag+ct
def aead_decrypt(key, blob, aad=b""):
    ver,nonce=blob[:1],blob[1:13]
    if ver==b"\x01":
        return AESGCM(_hkdf(key,b"neuron-aesgcm",b"v1",32)).decrypt(nonce,blob[13:],aad)
    tag,ct=blob[13:29],blob[29:]; mk=_hkdf(key,nonce,b"mac")
    if not hmac.compare_digest(tag,hmac.new(mk,b"\x00"+nonce+aad+ct,hashlib.sha256).digest()[:16]):
        raise ValueError("authentication failed: wrong key or tampered data")
    ek=_hkdf(key,nonce,b"enc"); ks=b""; c=0
    while len(ks)<len(ct): ks+=hmac.new(ek,nonce+c.to_bytes(8,"big"),hashlib.sha256).digest(); c+=1
    return bytes(a^b for a,b in zip(ct,ks))
def derive_key(secret, neuron_id):
    return _hkdf(secret.encode(),b"neuron-db/"+neuron_id.encode(),b"key",32)
class SecureNeuron:
    def __init__(self, key, entries=None):
        self._key=key; self._idx_key=_hkdf(key,b"neuron-index",b"v1",32); self.entries=entries or []
    def _keyed(self, stem):
        return base64.b64encode(hmac.new(self._idx_key,stem.encode(),hashlib.sha256).digest()[:8]).decode()
    def _stems(self, phrase): return {self._keyed(s) for s in _stem(_content(phrase))}
    def put(self, key_phrase, value):
        idx=sorted(self._stems(key_phrase))
        if not idx: raise ValueError("key phrase has no indexable content")
        ct=base64.b64encode(aead_encrypt(self._key,value.encode())).decode()
        self.entries.append({"x":idx,"c":ct})
    def get(self, query, min_cover=0.5):
        q=self._stems(query)
        if not q: return None
        best,bk=None,(-1.0,-1)
        for i,e in enumerate(self.entries):
            xs=set(e["x"]); ov=len(q&xs)
            if not ov: continue
            cover=ov/len(xs)
            if cover<min_cover: continue
            sc=(cover,i)
            if sc>bk: bk,best=sc,e
        if best is None: return None
        try: return aead_decrypt(self._key,base64.b64decode(best["c"])).decode()
        except Exception: return None
    def dump(self): return json.dumps(self.entries,separators=(",",":"))
    @classmethod
    def load(cls, key, blob): return cls(key,json.loads(blob or "[]"))
    @property
    def count(self): return len(self.entries)
class SecureNeuronDB:
    SCHEMA="CREATE TABLE IF NOT EXISTS secure (id TEXT PRIMARY KEY, blob TEXT NOT NULL, updated INTEGER NOT NULL);"
    def __init__(self, path="secure.db"):
        self.conn=sqlite3.connect(path,check_same_thread=False); self._lock=threading.Lock()
        self.conn.execute(self.SCHEMA); self.conn.commit()
    def _blob(self, nid):
        row=self.conn.execute("SELECT blob FROM secure WHERE id=?",(nid,)).fetchone(); return row[0] if row else "[]"
    def _write(self, nid, blob):
        self.conn.execute("INSERT INTO secure(id,blob,updated) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET blob=excluded.blob,updated=excluded.updated",(nid,blob,int(time.time()*1000)))
        self.conn.commit()
    def put(self, nid, secret, key_phrase, value):
        with self._lock:
            k=derive_key(secret,nid); n=SecureNeuron.load(k,self._blob(nid)); n.put(key_phrase,value); self._write(nid,n.dump())
    def get(self, nid, secret, query):
        with self._lock:
            k=derive_key(secret,nid); n=SecureNeuron.load(k,self._blob(nid)); return n.get(query)
    def close(self): self.conn.close()
