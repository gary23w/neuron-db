#!/usr/bin/env python3
"""Generate the FROZEN head-to-head dataset, fed identically to both engines.
Writes facts.tsv (id<TAB>fact), queries.tsv (query<TAB>gold_id<TAB>gold_value<TAB>class),
corpus.txt (generic synonym-grounding text for the std-only semantic space — the disclosed,
tiny analog of a dense embedder's internet-scale pretraining; it does NOT contain the facts or
queries). Classes: exact-id, exact-lex, paraphrase (disjoint vocab, vectors' turf), distractor
(near-duplicates, precision), none (no-answer, symmetric abstention test)."""
import sys

facts, queries = [], []
def add_fact(text):
    fid = len(facts); facts.append(text); return fid
def add_q(q, fid, val, klass): queries.append((q, str(fid) if fid is not None else "NONE", val, klass))

# ---- A: exact-identifier (opaque tokens; some near-duplicate to stress vector blur) ----
api = [("stripe","sk_9f3a2b71"),("twilio","sk_9f3b2a17"),("sendgrid","SG_4d8e10"),
       ("github","ghp_22aa90"),("aws","AKIA7Z1Q4"),("openai","sk_proj_5b2c"),
       ("datadog","dd_7e44c1"),("slack","xoxb_3391"),("notion","ntn_8842"),("figma","figd_1057")]
for s,k in api:
    f=add_fact(f"the {s} api key is {k}"); add_q(f"what is the {s} api key?", f, k, "exact-id")
wh = [("stripe","whsec_7c1d"),("twilio","whsec_7c1e"),("github","whsec_44b2"),
      ("slack","whsec_44c2"),("datadog","whsec_9001")]
for s,k in wh:
    f=add_fact(f"the {s} webhook secret is {k}"); add_q(f"what is the {s} webhook secret?", f, k, "exact-id")
rel = [("kafka","v4.12.7"),("redis","v7.2.1"),("envoy","v1.29.3"),("vault","v1.15.0"),("nginx","v1.25.4")]
for p,v in rel:
    f=add_fact(f"the {p} release is version {v}"); add_q(f"what version is the {p} release?", f, v, "exact-id")

# ---- B: exact-lexical (answer in the fact's own words) ----
proj = [("atlas","march"),("orion","august"),("vega","november"),("lyra","june"),("draco","april")]
for p,m in proj:
    f=add_fact(f"the {p} deadline is {m}"); add_q(f"when is the {p} deadline?", f, m, "exact-lex")
team = [("backend","0930"),("frontend","1015"),("platform","1100"),("data","1400")]
for t,h in team:
    f=add_fact(f"the {t} standup is at {h}"); add_q(f"what time is the {t} standup?", f, h, "exact-lex")
room = [("aspen","18"),("cedar","12"),("birch","40"),("maple","8"),("willow","24")]
for r,n in room:
    f=add_fact(f"the {r} room capacity is {n} seats"); add_q(f"what is the {r} room capacity?", f, n, "exact-lex")
misc = [("the quarterly report owner is priya","who owns the quarterly report?","priya"),
        ("the staging cluster region is frankfurt","what region is the staging cluster in?","frankfurt"),
        ("the billing service language is rust","what language is the billing service?","rust"),
        ("the mobile app store rating is 4.6","what is the mobile app store rating?","4.6")]
for fact_t,q,v in misc:
    f=add_fact(fact_t); add_q(q, f, v, "exact-lex")

# ---- D: paraphrase-semantic (query vocab disjoint from fact vocab; vectors' turf) ----
para = [
 ("the rooftop garden is watered at dawn","when do they irrigate the elevated terrace plants?","dawn"),
 ("the founder mentors three apprentices on weekends","how many trainees does the company creator guide on days off?","three"),
 ("the vaccine remains potent for nine months","how long does the immunization stay effective?","nine"),
 ("the harbor freezes during midwinter","what season does the port turn to ice?","midwinter"),
 ("the novelist drafts chapters before sunrise","when does the author write sections of the book?","sunrise"),
 ("the alloy withstands extreme heat","what does the metal blend tolerate?","heat"),
 ("the ferry departs every quarter hour","how frequently does the boat leave the dock?","quarter"),
 ("the surgeon operates twice weekly","how often does the doctor perform procedures?","twice"),
 ("the reservoir supplies the entire valley","what region does the water source serve?","valley"),
 ("the comet returns every seventy years","how regularly does the icy body come back?","seventy"),
 ("the bakery discards unsold loaves nightly","what does the bread shop throw away each evening?","loaves"),
 ("the glacier retreats a meter annually","how much does the ice sheet shrink yearly?","meter"),
 ("the orchestra rehearses in the basement","where does the symphony group practice?","basement"),
 ("the startup burned its funding within months","how quickly did the new venture exhaust its capital?","months"),
 ("the antidote neutralizes the venom instantly","what does the cure do to the poison immediately?","instantly"),
 ("the lighthouse guides ships through fog","what does the beacon help vessels navigate?","fog"),
 ("the apprentice forged the blade overnight","when did the trainee craft the sword?","overnight"),
 ("the monastery brews ale for pilgrims","who does the abbey make beer for?","pilgrims"),
]
for fact_t,q,v in para:
    f=add_fact(fact_t); add_q(q, f, v, "paraphrase")

# ---- DD: adversarial near-duplicate distractors (differ only in entity; precision) ----
cities = ["denver","austin","boston","seattle","miami","portland","dallas","atlanta",
          "phoenix","newark","tucson","fresno"]
open_t = ["0700","0730","0800","0830","0900","0930","1000","1030","0715","0745","0815","0845"]
for c,h in zip(cities,open_t):
    f=add_fact(f"the {c} office opens at {h}")
    add_q(f"when does the {c} office open?", f, h, "distractor")
ship = ["lumber","textiles","ceramics","glassware","produce","hardware","apparel","furniture"]
for c,p in zip(cities,ship):
    f=add_fact(f"the {c} warehouse ships {p}")  # near-duplicate set, mostly unqueried
for i,(c,p) in enumerate(zip(cities,ship)):
    if i % 3 == 0: add_q(f"what does the {c} warehouse ship?", facts.index(f"the {c} warehouse ships {p}"), p, "distractor")

# ---- Filler: diverse, unqueried facts so retrieval has real competition ----
subj = ["river","mountain","desert","forest","canyon","prairie","tundra","lagoon","fjord","mesa",
        "library","museum","stadium","airport","harbor","market","garden","temple","castle","bridge"]
pred = ["was mapped in 1962","hosts an annual festival","is protected by statute","draws many visitors",
        "was rebuilt after the flood","appears on the regional seal","is closed on mondays","spans two districts",
        "is named after a poet","was funded by donations"]
for i,s in enumerate(subj):
    for j in range(8):
        add_fact(f"the north {s} {pred[(i+j)%len(pred)]}")

# ---- NONE: no-answer queries (nothing in the store answers them) ----
none_qs = ["what is the heroku api key?","when is the perseus deadline?","when does the london office open?",
           "what is the quantum room capacity?","what version is the mariadb release?","who owns the annual budget?",
           "what time is the design standup?","what does the chicago warehouse ship?","what is the paypal webhook secret?",
           "what region is the production cluster in?","when is the titan deadline?","what is the discord api key?"]
for q in none_qs: add_q(q, None, "", "none")

# ---- corpus.txt: generic synonym co-occurrence (NO facts/queries inside) ----
corpus = """
Sailors trust a beacon. Every lighthouse along the coast guides passing ships and vessels safely
through fog and storm, helping them navigate the dangerous narrows at night.
Gardeners irrigate the soil at dawn. A rooftop garden on an elevated terrace needs watering before
the heat of noon so the plants stay green through the long summer.
A founder often mentors a young trainee. The company creator guides each apprentice patiently,
and on weekends and days off the mentor reviews their progress.
An immunization protects the body. A good vaccine stays potent and remains effective for many
months, so the shot keeps people safe through the season.
The harbor and the port share a shoreline. In midwinter the cold water can freeze, and a sheet of
ice covers the docks where the ferry and the boat depart.
An author is a novelist. The writer drafts chapters and writes long sections of the book before
sunrise, alone at the desk while the town still sleeps.
A metal blend is an alloy. The material withstands extreme heat and can tolerate fire far better
than ordinary iron, so engineers prize it.
A surgeon is a doctor. The physician operates twice a week and performs delicate procedures in the
operating theatre, often for many hours.
A reservoir is a vast water source. It supplies the entire valley and serves the whole region,
feeding the river that waters the farms downstream.
A comet is an icy body. It returns on a long orbit and comes back to our skies only once every
several decades, a rare and dazzling visitor.
A bakery is a bread shop. Each evening it discards the unsold loaves, throwing the day's leftover
bread away so the morning batch is always fresh.
A glacier is a sheet of ice. It retreats and shrinks year by year, losing a meter of length
annually as the climate slowly warms.
An orchestra is a symphony group. The musicians rehearse and practice in the basement hall beneath
the theatre before every concert.
A startup is a new venture. It burned through its funding fast and exhausted its capital within a
few short months, then closed its doors.
An antidote is a cure. It neutralizes the venom and counters the poison almost immediately, saving
the patient within minutes of the bite.
A blade is a sword. The apprentice and the smith forge and craft the steel overnight, hammering the
hot metal until the edge is keen.
A monastery is an abbey. The monks brew ale and make beer for the pilgrims and travellers who stop
to rest on the long road.
Offices open early; a warehouse ships goods like lumber, textiles, and produce to many cities.
A standup meeting starts the workday; a deadline marks its end; a room has a seating capacity.
"""

def w(p, lines):
    with open(p,"w",encoding="utf-8") as f: f.write("\n".join(lines)+"\n")
w("facts.tsv", [f"{i}\t{t}" for i,t in enumerate(facts)])
w("queries.tsv", [f"{q}\t{gid}\t{v}\t{c}" for q,gid,v,c in queries])
open("corpus.txt","w",encoding="utf-8").write(corpus)
from collections import Counter
cc = Counter(c for *_,c in queries)
print(f"facts={len(facts)} queries={len(queries)} byclass={dict(cc)}")
