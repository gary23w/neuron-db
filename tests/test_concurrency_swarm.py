"""A swarm of users submitting measurements at the same time. NeuronDB uses one SQLite
connection (check_same_thread=False) guarded by a threading.Lock, so concurrent access must
be safe: no exceptions, no lost writes, no cross-scope corruption. Run:
python tests/test_concurrency_swarm.py"""
import os, sys, time, tempfile, threading, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import NeuronDB

P=F=0
def ck(name, cond, info=""):
    global P,F; ok=bool(cond); P+=ok; F+=(not ok); print(("PASS " if ok else "FAIL ")+name+(("  | "+info) if info else ""))

METRICS=["page load","click count","scroll depth","time on page","memory used","fps","ttfb","cls"]

# ---------- 1) SWARM to DISTINCT scopes (each user writes their own metrics) ----------
dbf=tempfile.mktemp(suffix=".db"); db=NeuronDB(dbf, max_facts=10**9)
THREADS=100; PER=40
errors=[]; done=threading.Barrier(THREADS+1)
def user_writer(tid):
    try:
        for k in range(PER):
            m=METRICS[k%len(METRICS)]
            db.observe(f"user:{tid}", f"the {m.replace(' ','_')} sample {k} is {100+(tid*7+k)%900}")
    except Exception as e: errors.append(f"w{tid}:{e}")
    finally: done.wait()
t=time.perf_counter()
ths=[threading.Thread(target=user_writer,args=(i,)) for i in range(THREADS)]
for th in ths: th.start()
done.wait()
for th in ths: th.join()
dur=time.perf_counter()-t
ck("no exceptions under 100-thread swarm", not errors, str(errors[:3]))
# every user's facts all landed
total=sum(db.stats(f"user:{i}")["facts"] for i in range(THREADS))
ck("no lost writes across distinct scopes", total==THREADS*PER, f"{total}/{THREADS*PER}")
# spot-check recall correctness per user
okr=sum(1 for i in range(THREADS) if db.recall(f"user:{i}","what is the page_load sample 0?"))
ck("per-user recall intact after swarm", okr==THREADS, f"{okr}/{THREADS}")
print(f"  throughput: {THREADS*PER/dur:.0f} writes/s across {THREADS} concurrent users")
db.close()

# ---------- 2) CONTENTION: many threads hammer the SAME scope ----------
dbf2=tempfile.mktemp(suffix=".db"); db=NeuronDB(dbf2, max_facts=10**9)
N=2000; CT=50; counter=[0]; clock=threading.Lock()
def shared_writer():
    while True:
        with clock:
            i=counter[0]
            if i>=N: return
            counter[0]+=1
        db.observe("shared:metrics", f"measurement {i} reads value {i}")
t=time.perf_counter()
ths=[threading.Thread(target=shared_writer) for _ in range(CT)]
for th in ths: th.start()
for th in ths: th.join()
dur2=time.perf_counter()-t
got=db.stats("shared:metrics")["facts"]
ck("no lost updates on a single hot scope", got==N, f"{got}/{N} facts from {CT} threads")
print(f"  contended throughput: {N/dur2:.0f} writes/s into one scope ({CT} threads)")
db.close()

# ---------- 3) CONCURRENT READ + WRITE (readers must never crash or read torn data) ----------
dbf3=tempfile.mktemp(suffix=".db"); db=NeuronDB(dbf3, max_facts=10**9)
for i in range(200): db.observe(f"user:{i}", f"the plan is {'pro' if i%2 else 'free'}")
stop=threading.Event(); rerr=[]; reads=[0]
def reader():
    while not stop.is_set():
        try:
            u=random.randint(0,199); v=db.get(f"user:{u}","what plan?")
            if v not in ("pro","free",None): rerr.append(f"torn:{v}")
            reads[0]+=1
        except Exception as e: rerr.append(str(e))
def writer():
    for i in range(1000):
        db.observe(f"user:{random.randint(0,199)}", f"event {i} took {i%500} ms")
readers=[threading.Thread(target=reader) for _ in range(20)]
for r in readers: r.start()
w=threading.Thread(target=writer); w.start(); w.join()
stop.set()
for r in readers: r.join()
ck("concurrent readers never crash or read torn data", not rerr, f"{reads[0]} reads, errs={rerr[:2]}")
db.close()

print(f"\n{P}/{P+F} passed")
sys.exit(0 if F==0 else 1)
