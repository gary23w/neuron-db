from __future__ import annotations
import sqlite3, time, threading
from collections import OrderedDict
from typing import Optional
from .neuron import Neuron
from .turn import turn as _turn
SCHEMA = "CREATE TABLE IF NOT EXISTS neurons (id TEXT PRIMARY KEY, facts TEXT NOT NULL DEFAULT '[]', created INTEGER NOT NULL, updated INTEGER NOT NULL, turns INTEGER NOT NULL DEFAULT 0);"
class NeuronDB:
    def __init__(self, path="neurons.db", max_facts=500, cache_size=256, model=True, model_dir=None):
        self.path=path; self.max_facts=max_facts
        self._model=model; self._model_dir=model_dir; self._bridge=None; self._bridge_tried=False
        self.conn=sqlite3.connect(path, check_same_thread=False)
        try: self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError: pass
        self._lock=threading.Lock()
        self.conn.execute(SCHEMA); self.conn.commit()
        self._cache=OrderedDict(); self._cache_size=cache_size
    def _load(self, nid):
        hit=self._cache.get(nid)
        if hit is not None:
            self._cache.move_to_end(nid); return hit
        row=self.conn.execute("SELECT facts,created,updated,turns FROM neurons WHERE id=?",(nid,)).fetchone()
        if row: entry=(Neuron.load(row[0],self.max_facts),{"created":row[1],"updated":row[2],"turns":row[3]})
        else:
            now=int(time.time()*1000); entry=(Neuron(self.max_facts),{"created":now,"updated":now,"turns":0})
        self._cache[nid]=entry; self._cache.move_to_end(nid)
        if len(self._cache)>self._cache_size: self._cache.popitem(last=False)
        return entry
    def _save(self, nid, n, meta):
        meta["updated"]=int(time.time()*1000)
        self.conn.execute("INSERT INTO neurons(id,facts,created,updated,turns) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET facts=excluded.facts,updated=excluded.updated,turns=excluded.turns",(nid,n.dump(),meta["created"],meta["updated"],meta["turns"]))
        self.conn.commit()
    def turn(self, nid, message):
        with self._lock:
            n,meta=self._load(nid); at_cap=n.fact_count>=self.max_facts
            r=_turn(n,message)
            if at_cap and r["wrote"]: n.episodes=n.episodes[:self.max_facts]
            meta["turns"]+=1; self._save(nid,n,meta)
            return {**r,"facts":n.fact_count,"capacity_reached":at_cap and bool(r["wrote"])}
    def observe(self, nid, text):
        with self._lock:
            n,meta=self._load(nid); w=n.observe(text); self._save(nid,n,meta); return len(w)
    def recall(self, nid, query):
        with self._lock:
            n,_=self._load(nid); return n.recall(query)
    def get(self, nid, query):
        with self._lock:
            n,_=self._load(nid); hit=n.recall(query); return hit["value"] if hit else None
    def forget(self, nid, match=None):
        with self._lock:
            n,meta=self._load(nid); before=n.fact_count
            n.episodes=[e for e in n.episodes if match.lower() not in e["t"].lower()] if match else []
            self._save(nid,n,meta); return {"forgot":before-n.fact_count,"remaining":n.fact_count}
    def _ensure_bridge(self):
        """Lazily load the gary-neuron cortex the first time think() is called. Cortex is
        ON by default; if the model or its deps are unavailable we fall back to the store
        and never try again (so think() degrades gracefully instead of erroring)."""
        if self._bridge is not None or not self._model or self._bridge_tried:
            return self._bridge
        self._bridge_tried=True
        try:
            from .bridge import GaryNeuronBridge
            self._bridge=GaryNeuronBridge(model_dir=self._model_dir)
        except Exception:
            self._bridge=None; self._model=False
        return self._bridge
    @property
    def model_enabled(self):
        return self._ensure_bridge() is not None
    def think(self, nid, query, k=3):
        """Answer by composing language with the cortex over the bounded working set the
        store selects. Cortex is enabled by default; with no model present this returns the
        store's recalled value instead. get()/recall() stay pure store (microseconds)."""
        with self._lock:
            n,_=self._load(nid)
        b=self._ensure_bridge()
        if b is None:
            hit=n.recall(query)
            return {"answer": hit["value"] if hit else None, "source": "store", "model": False}
        return {"answer": b.think(n, query, k=k), "source": "cortex", "model": True}
    def stats(self, nid):
        with self._lock:
            n,meta=self._load(nid); return {"facts":n.fact_count,"max_facts":self.max_facts,**meta}
    def neurons(self):
        with self._lock:
            return [r[0] for r in self.conn.execute("SELECT id FROM neurons ORDER BY updated DESC")]
    def close(self): self.conn.close()
