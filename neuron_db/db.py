"""NeuronDB -- a database of neurons in a single SQLite file.

Each row is one neuron: an id and its facts (stored minimally as text+flag, the
index rebuilt on load). You write to a neuron and query it by name. There is no
operation that dumps every value of a neuron -- facts come out only via recall.

    db = NeuronDB("memory.db")
    db.turn("alice", "my wifi password is hunter2")   # -> {'reply': 'got it -- hunter2.', ...}
    db.turn("alice", "what is my wifi password?")      # -> {'reply': 'hunter2.', ...}
"""
from __future__ import annotations
import sqlite3, time, threading
from typing import Optional
from .neuron import Neuron
from .turn import turn as _turn

SCHEMA = """
CREATE TABLE IF NOT EXISTS neurons (
  id      TEXT PRIMARY KEY,
  facts   TEXT NOT NULL DEFAULT '[]',
  created INTEGER NOT NULL,
  updated INTEGER NOT NULL,
  turns   INTEGER NOT NULL DEFAULT 0
);
"""


class NeuronDB:
    def __init__(self, path: str = "neurons.db", max_facts: int = 500):
        self.path = path
        self.max_facts = max_facts
        # check_same_thread=False + a lock: safe to share across the server's threads
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def _load(self, nid: str):
        row = self.conn.execute("SELECT facts, created, updated, turns FROM neurons WHERE id=?", (nid,)).fetchone()
        if row:
            return Neuron.load(row[0], self.max_facts), {"created": row[1], "updated": row[2], "turns": row[3]}
        now = int(time.time() * 1000)
        return Neuron(self.max_facts), {"created": now, "updated": now, "turns": 0}

    def _save(self, nid: str, n: Neuron, meta: dict):
        meta["updated"] = int(time.time() * 1000)
        self.conn.execute(
            "INSERT INTO neurons(id,facts,created,updated,turns) VALUES(?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET facts=excluded.facts, updated=excluded.updated, turns=excluded.turns",
            (nid, n.dump(), meta["created"], meta["updated"], meta["turns"]))
        self.conn.commit()

    # --- primary API (each call serialized; SQLite shared across server threads) ---
    def turn(self, nid: str, message: str) -> dict:
        with self._lock:
            n, meta = self._load(nid)
            at_cap = n.fact_count >= self.max_facts
            r = _turn(n, message)
            if at_cap and r["wrote"]:
                n.episodes = n.episodes[:self.max_facts]
            meta["turns"] += 1
            self._save(nid, n, meta)
            return {**r, "facts": n.fact_count, "capacity_reached": at_cap and bool(r["wrote"])}

    def observe(self, nid: str, text: str) -> int:
        with self._lock:
            n, meta = self._load(nid); wrote = n.observe(text); self._save(nid, n, meta); return len(wrote)

    def recall(self, nid: str, query: str) -> Optional[dict]:
        with self._lock:
            n, _ = self._load(nid); return n.recall(query)

    def forget(self, nid: str, match: Optional[str] = None) -> dict:
        with self._lock:
            n, meta = self._load(nid); before = n.fact_count
            if match: n.episodes = [e for e in n.episodes if match.lower() not in e["t"].lower()]
            else: n.episodes = []
            self._save(nid, n, meta)
            return {"forgot": before - n.fact_count, "remaining": n.fact_count}

    def stats(self, nid: str) -> dict:
        with self._lock:
            n, meta = self._load(nid)
            return {"facts": n.fact_count, "max_facts": self.max_facts, **meta}

    def neurons(self) -> list:
        with self._lock:
            return [r[0] for r in self.conn.execute("SELECT id FROM neurons ORDER BY updated DESC")]

    def close(self): self.conn.close()
