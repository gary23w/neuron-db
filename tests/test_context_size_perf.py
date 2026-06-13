"""Context retention, scaling, and performance for Neuron + NeuronDB(SQLite).
Single-neuron recall is O(candidates), so latency grows with size for broad cues -- the
documented boundary that motivates per-scope sharding (one neuron per user/session).
Run: python tests/test_context_size_perf.py"""
import os, sys, time, tempfile, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron, NeuronDB

P=F=0
def ck(name, cond, info=""):
    global P,F; ok=bool(cond); P+=ok; F+=(not ok); print(("PASS " if ok else "FAIL ")+name+(("  | "+info) if info else ""))

# CONTEXT: distinct facts coexist in one neuron and each recalls
n=Neuron(max_facts=10**9)
ctx={"deploy region":"the deploy region is us-west-2","db engine":"the database engine is postgres 16",
 "cache ttl":"the cache ttl is 300 seconds","auth method":"the auth method is oauth2",
 "rate limit":"the rate limit is 1000 requests","cdn provider":"the cdn provider is cloudflare",
 "queue system":"the queue system is rabbitmq","log level":"the log level is debug"}
for v in ctx.values(): n.observe(v)
hits=sum(1 for k in ctx if n.recall(f"what is the {k}?"))
ck("context: 8 distinct facts coexist + recall", hits>=7, f"{hits}/{len(ctx)}")

# SIZE: distinct 4-component keys, accuracy + latency
ADJ="north south east west gold iron jade onyx ruby teal".split()
NOUN="server router gateway sensor vault locker node region tenant policy".split()
VERB="holds maps owns logs caches tracks binds stores".split()
def build(N):
    s=Neuron(max_facts=10**9); keys=[]
    for i in range(N):
        key=f"{ADJ[i%10]} {NOUN[(i//10)%10]} {VERB[(i//100)%8]} {i}"; keys.append((key,f"val{i}"))
        s.observe(f"the {key} reads val{i}")
    return s,keys
random.seed(0)
print("  size      build      recall       accuracy")
acc10=0
for N in (1000,10000,50000):
    t=time.perf_counter(); s,keys=build(N); bt=time.perf_counter()-t
    probe=random.sample(keys,100)
    t=time.perf_counter(); correct=sum(1 for k,v in probe if (lambda h: h and h['value']==v)(s.recall(f"what does the {k} read?")))
    rt=(time.perf_counter()-t)/len(probe)*1e6
    print(f"  {N:>6}   {bt:>6.2f}s   {rt:>8.0f} us    {correct}/{len(probe)}")
    if N==10000: acc10=correct
ck("10k distinct-key accuracy = 100%", acc10==100, f"{acc10}/100")

# PERFORMANCE: write throughput + dump size
t=time.perf_counter(); s,_=build(20000); wt=time.perf_counter()-t
ck("in-mem write > 25k facts/s", 20000/wt>25000, f"{20000/wt:.0f}/s")
print(f"  serialized dump: {len(s.dump().encode())/20000:.1f} bytes/fact")

# NeuronDB SQLite: per-user (sharded) ingest + storage + recall
dbf=tempfile.mktemp(suffix=".db"); db=NeuronDB(dbf, max_facts=10**9); M=8000
t=time.perf_counter()
for i in range(M): db.observe(f"user:{i%1000}", f"page p{i%50} loaded in {50+i%400} ms")
it=time.perf_counter()-t; db.close(); size=os.path.getsize(dbf)
ck("SQLite ingest > 800 facts/s", M/it>800, f"{M/it:.0f}/s (commit+blob-rewrite per write)")
print(f"  SQLite file: {size/1024:.0f} KiB ({size/M:.1f} B/fact)")
db=NeuronDB(dbf, max_facts=10**9)
t=time.perf_counter()
for i in range(1000): db.get(f"user:{i}", "what page?")
gt=(time.perf_counter()-t)/1000*1e6; db.close()
ck("sharded per-user recall < 1ms", gt<1000, f"{gt:.0f} us/op (small per-user neuron stays fast)")

print(f"\n{P}/{P+F} passed")
sys.exit(0 if F==0 else 1)
