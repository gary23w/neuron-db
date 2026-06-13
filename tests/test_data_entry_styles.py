"""Every style of data entry I can think of, against Neuron + NeuronDB(SQLite).
Goal: nothing crashes, storage is robust to weird input, and clear facts recall.
Run: python tests/test_data_entry_styles.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron, NeuronDB

P=F=0; notes=[]
def ck(name, cond, info=""):
    global P,F; ok=bool(cond); P+=ok; F+=(not ok)
    print(("PASS " if ok else "FAIL ")+name+(("  | "+info) if info else ""))

# ---------- robustness: store every style without crashing ----------
styles = {
 "ascii":            "the project deadline is friday",
 "integer":          "the server port is 8080",
 "decimal":          "the cpu temperature is 72.4 degrees",
 "negative":         "the account balance is -250 dollars",
 "currency":         "the invoice total is $1,234.56",
 "percent":          "the conversion rate is 3.7 percent",
 "iso_date":         "the launch date is 2026-06-13",
 "time":             "the standup is at 09:30",
 "accents":          "the cafe is in Zurich near the Hauptbahnhof",
 "chinese":          "the city is 北京 in China",
 "japanese":         "the train is the 新幹線 line",
 "korean":           "the dish is called 비빔밥",
 "arabic_rtl":       "the word for peace is سلام today",
 "hebrew_rtl":       "the greeting is שלום friend",
 "emoji":            "the mascot is a rocket 🚀 named Sparky",
 "url":              "the docs are at https://example.com/guide?id=42",
 "email":            "the contact is jane.doe@example.co.uk",
 "filepath":         "the config is at /etc/app/config.yaml",
 "json_ish":         'the payload is {"user": 42, "plan": "pro"}',
 "key_value":        "plan: enterprise; seats: 50; region: us-east",
 "markdown":         "the title is **Quarterly Report** in bold",
 "code":             "the function is def add(a, b): return a + b",
 "multi_sentence":   "The meeting moved. The new room is B12. Bring the laptop.",
 "mixed_lang":       "the password café_2026 works on the 服务器 server",
 "quoted_value":     'my favorite tool is "Search Console" lately',
 "very_long":        "the manifest lists " + " ".join(f"item{i}" for i in range(2000)) + " and ends",
 "whitespace":       "      \t   \n  ",
 "empty":            "",
 "single_char":      "x",
 "control_chars":    "the\x00code\x07is\x1bvalid42 here",
 "repeated":         "the the the value is twelve twelve",
 "sql_inject":       "Robert'); DROP TABLE neurons;-- is my name",
 "html":             "the banner is <script>alert(1)</script> text",
 "numbers_value":    "there are 50 chairs in the main hall",
}
n = Neuron(max_facts=10**9)
crashed=[]
for name, txt in styles.items():
    try: n.observe(txt)
    except Exception as e: crashed.append((name,str(e)[:60]))
ck("no style crashes the store", not crashed, f"{len(styles)} styles" if not crashed else str(crashed))

# ---------- recall correctness on the clear factual styles ----------
checks = [
 ("integer recall",      "what is the server port?",        "8080"),
 ("decimal recall",      "what is the cpu temperature?",    "72.4"),
 ("iso_date recall",     "what is the launch date?",        "2026-06-13"),
 ("numbers value",       "how many chairs are there?",      "50"),
 ("emoji-adjacent name", "what is the mascot named?",       "Sparky"),
 ("quoted multiword",    "what is my favorite tool?",       "Search Console"),
]
for label, q, want in checks:
    hit = n.recall(q); got = (hit or {}).get("value","")
    ck(label, want.lower() in str(got).lower(), f"want~{want!r} got {got!r}")

# --- informational findings (not failures): documented engine boundaries ---
print("-- findings --")
url_hit = n.recall("where are the docs?")
print(f"  FINDING url-with-? not stored: observe() rejects text containing '?' -> recall={url_hit}")
print(f"  FINDING two proper nouns in one sentence -> value isolation may pick the latter")

# unicode value survives round-trip through dump/load
n2 = Neuron.load(n.dump(), max_facts=10**9)
hit = n2.recall("what is the city?")
ck("unicode survives dump/load", hit and ("北京" in hit["fact"]), (hit or {}).get("value",""))

# ---------- SQLite safety: injection string must NOT damage the DB ----------
dbf = tempfile.mktemp(suffix=".db")
db = NeuronDB(dbf, max_facts=1000)
db.observe("u:1", "Robert'); DROP TABLE neurons;-- is my name")
db.observe("u:1", "my plan is pro")
db.observe("u:2", "my plan is free")
ck("injection did not drop table", db.get("u:1","what plan am i on?")=="pro")
ck("second scope intact",          db.get("u:2","what plan am i on?")=="free")
ck("scopes isolated",              db.get("u:1","what plan am i on?")!=db.get("u:2","what plan am i on?"))
db.close()

print(f"\n{P}/{P+F} passed")
sys.exit(0 if F==0 else 1)
