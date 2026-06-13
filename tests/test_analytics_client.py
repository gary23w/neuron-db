"""Client-side analytics on neuron-db (SQLite per-user model). Simulates the metrics a JS
analytics SDK collects, computes the standard KPIs, and measures neuron-db ingest, storage,
and per-user recall against ground truth. Run: python tests/test_analytics_client.py"""
import os, sys, time, tempfile, random, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import NeuronDB

random.seed(7)
DEVICES=["mobile","mobile","mobile","desktop","desktop","tablet"]
BROWSERS=["Chrome","Chrome","Safari","Safari","Firefox","Edge"]
OSES=["iOS","Android","Windows","macOS","Linux"]
COUNTRIES=["States","States","States","Canada","Britain","Germany","India","Brazil","Australia","Japan"]
REFERRERS=["google","google","direct","direct","twitter","newsletter","reddit","bing"]
PAGES=["/","/pricing","/product","/blog","/docs","/cart","/checkout","/about"]
FUNNEL=["/","/product","/cart","/checkout"]

U=3000
users=[]; events=[]   # ground-truth event log
day0=1_700_000_000
for uid in range(U):
    dev=random.choice(DEVICES); br=random.choice(BROWSERS); osn=random.choice(OSES); cc=random.choice(COUNTRIES)
    ref=random.choice(REFERRERS); plan=random.choice(["free","free","free","pro","enterprise"])
    first_day=random.randint(0,6)
    n_sessions=random.choices([1,2,3,5,8],[40,25,18,12,5])[0]
    seen_days=set()
    for sname in range(n_sessions):
        d=first_day+random.randint(0,7); seen_days.add(d)
        ts=day0+d*86400+random.randint(0,80000)
        # funnel progression with drop-off
        depth=random.choices([1,2,3,4],[45,28,17,10])[0]
        pages=FUNNEL[:depth]+random.sample(PAGES, random.randint(0,2))
        dur=0
        for pi,pg in enumerate(pages):
            load=max(80, int(random.gauss(900,500)))         # page load ms (web vital)
            ttfb=max(20, int(load*random.uniform(0.2,0.5)))
            dur+=random.randint(3,90)
            err=random.random()<0.02                          # 2% error rate
            events.append(dict(uid=uid,day=d,ts=ts,page=pg,load=load,ttfb=ttfb,err=err,dev=dev,cc=cc))
        users.append(dict(uid=uid,dev=dev,br=br,osn=osn,cc=cc,ref=ref,plan=plan,
                          sess=n_sessions,days=seen_days,first_day=first_day))
        break  # one representative profile row per user (sessions tracked in events)
    # add the rest of the sessions' events
    for s2 in range(1,n_sessions):
        d=first_day+random.randint(0,7); seen_days.add(d); ts=day0+d*86400
        depth=random.choices([1,2,3,4],[45,28,17,10])[0]
        for pg in FUNNEL[:depth]:
            events.append(dict(uid=uid,day=d,ts=ts,page=pg,load=max(80,int(random.gauss(900,500))),
                               ttfb=200,err=random.random()<0.02,dev=dev,cc=cc))

# ---------------- STANDARD CLIENT-SIDE ANALYTICS KPIs (ground truth) ----------------
import collections
sessions=collections.defaultdict(list)
for e in events: sessions[(e["uid"],e["ts"])].append(e)
n_sessions=len(sessions); n_events=len(events); n_users=U
pv=[e for e in events if True]
pages_per_session=[len(v) for v in sessions.values()]
bounce=sum(1 for v in sessions.values() if len(v)==1)/n_sessions
loads=sorted(e["load"] for e in events)
def pct(a,p): return a[min(len(a)-1,int(len(a)*p))]
err_rate=sum(1 for e in events if e["err"])/n_events
dev_mix=collections.Counter(u["dev"] for u in users)
cc_mix=collections.Counter(e["cc"] for e in events)
top_pages=collections.Counter(e["page"] for e in events).most_common(4)
ref_mix=collections.Counter(u["ref"] for u in users)
# funnel conversion
reached=collections.defaultdict(set)
for e in events:
    if e["page"] in FUNNEL: reached[e["page"]].add(e["uid"])
funnel=[(p,len(reached[p])) for p in FUNNEL]
# retention: of users first seen on a day, how many returned on a later day
ret_d1=sum(1 for u in users if any(d>u["first_day"] for d in u["days"]))/n_users
ret_d7=sum(1 for u in users if any(d>=u["first_day"]+7 for d in u["days"]))/n_users

print("==== CLIENT-SIDE ANALYTICS KPIs (simulated SDK collection) ====")
print(f"  users={n_users}  sessions={n_sessions}  events/pageviews={n_events}")
print(f"  pages/session: mean {st.mean(pages_per_session):.2f}  p50 {pct(sorted(pages_per_session),.5)}  p95 {pct(sorted(pages_per_session),.95)}")
print(f"  bounce rate: {bounce*100:.1f}%")
print(f"  page load ms: p50 {pct(loads,.5)}  p75 {pct(loads,.75)}  p95 {pct(loads,.95)}")
print(f"  JS error rate: {err_rate*100:.2f}%")
print(f"  device mix: {dict(dev_mix.most_common())}")
print(f"  top countries: {dict(cc_mix.most_common(5))}")
print(f"  top pages: {top_pages}")
print(f"  referrers: {dict(ref_mix.most_common())}")
print(f"  funnel {'->'.join(FUNNEL)}: {[c for _,c in funnel]}  (conv {funnel[-1][1]/funnel[0][1]*100:.1f}%)")
print(f"  retention: returned-after-day1 {ret_d1*100:.1f}%  day7 {ret_d7*100:.1f}%")

# ---------------- neuron-db AS THE CLIENT-SIDE STORE ----------------
print("\n==== neuron-db storing per-user profiles (SQLite) ====")
dbf=tempfile.mktemp(suffix=".db"); db=NeuronDB(dbf, max_facts=200)
t=time.perf_counter()
last_page={}
for u in users:
    uid=u["uid"]; sc=f"user:{uid}"
    db.observe(sc, f"the plan is {u['plan']}")
    db.observe(sc, f"the device is {u['dev']}")
    db.observe(sc, f"the country is {u['cc']}")
    db.observe(sc, f"the referrer is {u['ref']}")
    # last page seen for this user
    lp=[e["page"] for e in events if e["uid"]==uid][-1] if any(e["uid"]==uid for e in events) else "/"
    last_page[uid]=lp
    db.observe(sc, f"the last page is {lp.replace('/','slash ') or 'home'}")
ingest=time.perf_counter()-t
db.close(); size=os.path.getsize(dbf)
print(f"  ingested {U} user profiles (5 facts each = {U*5}) in {ingest:.2f}s = {U*5/ingest:.0f} facts/s")
print(f"  SQLite file: {size/1024:.0f} KiB ({size/(U*5):.1f} B/fact)")

# recall correctness vs ground truth + latency
db=NeuronDB(dbf, max_facts=200)
sample=random.sample(users, 300)
okp=okd=okc=0
t=time.perf_counter()
for u in sample:
    sc=f"user:{u['uid']}"
    okp += (db.get(sc,"what plan?")==u["plan"])
    okd += (db.get(sc,"what device?")==u["dev"])
    okc += (db.get(sc,"what country?")==u["cc"])
lat=(time.perf_counter()-t)/(len(sample)*3)*1e6
db.close()
print(f"  per-user recall accuracy: plan {okp}/{len(sample)}  device {okd}/{len(sample)}  country {okc}/{len(sample)}")
print(f"  per-user recall latency: {lat:.0f} us/query (from SQLite)")
ok = okp>=295 and okd>=295 and okc>=295
print("\nRESULT:", "PASS per-user store/recall is accurate" if ok else "FAIL")
sys.exit(0 if ok else 1)
