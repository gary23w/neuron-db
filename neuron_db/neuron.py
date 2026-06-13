from __future__ import annotations
import re, json
from typing import Optional
STOP = set("what is my the a an you your i me do did does how was were it to of and or in on at that this s u g who whats tell about im ive id ill am are be will wont cant dont yes no really have has had its these those there here remember recall know knew think guess again still any some mine get got getting go going want wanted would could should can please us we they them he she him her his hers their our ours".split())
STOPVAL = set("had has have having like likes liked want wants wanted went going goes got get gets day days week thing things something anything nothing everything lot bit time times really very name named names favorite favourite color colour food dog cat hello hi hey thanks thank okay yes lol bye good great nice long suppose supposed gonna wanna kinda sorta maybe probably definitely oh ya yeah yep nah hmm right sure fine cool wow oops used use using still now for with from into onto over under after before out off down up around through there here then than while because but not no so just too also even back well if as by be been being actually never always sometimes today tomorrow yesterday tonight sister brother mom dad mother father wife husband son daughter grandma grandpa aunt uncle cousin friend boss live lives lived drive drives new anyway hows heres theres lets gotta mostly honestly basically literally".split())
IRR = {"drank":"drink","ate":"eat","went":"go","goes":"go","saw":"see","met":"meet","took":"take","bought":"buy","made":"make","ran":"run","drove":"drive","wrote":"write","slept":"sleep","gave":"give","told":"tell","said":"say","felt":"feel","day":"today","days":"today","old":"age","aged":"age","work":"job","working":"job","employed":"job","occupation":"job","profession":"job","career":"job","kids":"kid","children":"kid","hometown":"city","town":"city","favourite":"favorite","colour":"color"}
REL = set("dog cat pet bird fish sister brother mom dad mother father wife husband son daughter grandma grandpa aunt uncle cousin car truck bike cats dogs puppy kitten puppies kittens hamster".split())
PETSRC = ["dog","cat","pet","bird","fish","puppy","kitten","hamster"]
NUMWORDS = set("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty hundred thousand million dozen".split())
ADV = set("actually anyway honestly basically literally oh ok okay well yeah yep nah hmm wow oops so but and also still just then wait sorry hey um uh no yes listen look".split())
ENT = re.compile(r"\b(dog|cat|puppy|kitten|pet|bird|fish|son|daughter|sister|brother|wife|husband|friend|car|truck|bike|boat|house)\b", re.I)
CORE = re.compile(r"^\s*(her|his|its|their|the)\s+name\s+is\s+", re.I)
INTRO = re.compile(r"^\s*(i'?m|i am|this is|call me|my name'?s?|the name is|name is)\s+([^\W\d]\S{1,24}(?:\s+[A-Z]\S{1,24})?)\s*$", re.I | re.U)
AGEIN = re.compile(r"^\s*(i'?m|i am)\s+(\d{1,3})(\s*(years?\s*old|yo))?\s*$", re.I)
SENT = re.compile(r"(?<=[.!?;])\s+|\n+")
LEADIN = re.compile(r"^\s*(remember( th(is|at))?|note|fyi|btw)\s*[:,\-]+\s*", re.I)
COMMAND = re.compile(r"^(count|explain|list|show|give|make|stop|repeat|calculate|compute|draw|sing|translate|define|describe|summarize|continue)\b", re.I)
def _w1(w):
    w=w.lower().strip("?.!,;:'\"’><)([]}{")
    if w.endswith("'s") or w.endswith("’s"): w=w[:-2]
    return w
def _words(s): return {_w1(w) for w in s.split()} - {""}
def _content(s): return _words(s) - STOP
def _stem1(w):
    w=IRR.get(w,w)
    if len(w)>=5 and w.endswith("ies"): w=w[:-3]+"y"
    elif len(w)>=4 and w.endswith("s") and not w.endswith("ss"): w=w[:-1]
    return w[:6] if len(w)>=8 else (w[:4] if len(w)>=4 else w)
def _stem(ws): return {_stem1(w) for w in ws}
REL_S=_stem(REL); PETS=_stem(PETSRC); STOPVAL_S=_stem(STOPVAL)
def _isnum(w): return any(c.isdigit() for c in w) or w.strip("?.!,'\"()").lower() in NUMWORDS
def _clip(s): return s.strip("?.!,;:'\"()[]{}")
def _surprise(w,i):
    s=0.0; core=w.lower()
    if any(c.isdigit() for c in core): s+=3.0
    elif w[:1].isupper() and i>0: s+=2.0
    if len(core)>=7: s+=0.6
    return s
def _sents(u,cap=24):
    parts=[p.strip() for p in SENT.split(u.strip()) if p and p.strip()]
    return (parts or [u.strip()])[:cap]
def _encode(text, entity):
    u=text.strip()
    if not u: return None
    ma=AGEIN.match(u)
    if ma: return {"t":text,"v":ma.group(2),"c":[ma.group(2)],"s":sorted(_stem({"age"})|{ma.group(2)}),"h":"age","self":True}
    mi=INTRO.match(u)
    if mi:
        nm=mi.group(2).strip("?.!,'"); f=nm.split()[0].lower()
        if f not in STOP and f not in ADV:
            return {"t":text,"v":nm,"c":[nm],"s":sorted(_stem({"name"})|{w.lower() for w in nm.split()}),"h":"name","self":True}
    cont=_content(u)
    if len(cont)<2 and not any(w.isdigit() for w in cont): return None
    if len(u.split())<3 and not any(w.isdigit() for w in cont): return None
    uw=_words(u)
    selfish=bool({"my","i","im","mine"}&uw) and not bool({"her","his","its","their","your"}&uw)
    inject=set()
    if entity and CORE.match(u): inject={entity.lower()}
    cands=[]
    for i,raw in enumerate(u.split()):
        w=_clip(raw); wl=w.lower()
        if not wl or wl in STOP or wl in STOPVAL or not any(c.isalnum() for c in wl) or (len(wl)<3 and not wl.isdigit()): continue
        cands.append((w,_surprise(w,i)+0.15*i))
    if not cands: return None
    cands.sort(key=lambda x:-x[1])
    keep=[w for w,_ in cands[:5]]
    for w,_ in cands[5:]:
        if len(keep)>=10: break
        if _isnum(w) or (w[:1].isupper()): keep.append(w)
    self_name=selfish and ("name" in _stem(cont))
    head=""
    for w in u.split():
        x=_w1(w)
        if x and x not in STOP and x not in ADV: head=next(iter(_stem({x}))); break
    return {"t":text,"v":keep[0],"c":keep,"s":sorted(_stem(cont)|_stem(inject)),"h":(next(iter(_stem(inject))) if inject else head),"self":self_name and not inject}
def _pick_value(ep, cue, want_num):
    words=ep["t"].split(); cand=ep.get("c",[ep["v"]])
    cue_pos=[i for i,w in enumerate(words) if _stem({_w1(w)})&cue]
    def pos_of(c):
        cl=_clip(c).lower()
        for i,w in enumerate(words):
            if _clip(w).lower()==cl or _w1(w)==_w1(c): return i
        return 10**6
    pool=[c for c in cand if not (_stem({c.lower()})&cue)]
    if want_num:
        nums=[c for c in pool if _isnum(c)]
        if nums: pool=nums
    if not pool: return ep["v"],True
    if want_num and cue_pos and len(pool)>1:
        pool=sorted(pool,key=lambda c: min((abs(pos_of(c)-p), 0 if pos_of(c)<=p else 1) for p in cue_pos))
    return pool[0],False
class Neuron:
    def __init__(self, max_facts=500):
        self.episodes=[]; self.last_entity=None; self.max_facts=max_facts
        self._index=None; self._index_len=-1
    def _build_index(self):
        idx={}
        for i,e in enumerate(self.episodes):
            for s in e["s"]: idx.setdefault(s,[]).append(i)
        self._index=idx; self._index_len=len(self.episodes)
    def observe(self, text, entity=None):
        if not text.strip() or "?" in text: return []
        out=[]; ent=entity if entity is not None else self.last_entity
        for s in _sents(text):
            s=LEADIN.sub("",s); sw=_words(s)
            if COMMAND.match(s) and not ({"my","i","im","mine"}&sw): continue
            if bool({"your","you","yours"}&sw) and not ({"my","i","im","mine"}&sw) and len(s.split())<8: continue
            e=_encode(s,ent)
            if e: self.episodes.append(e); out.append(e)
            m=ENT.search(s)
            if m: ent=m.group(1).lower()
        me=ENT.search(text)
        if me: self.last_entity=me.group(1).lower()
        if len(self.episodes)>self.max_facts: self.episodes=self.episodes[-self.max_facts:]
        return out
    def recall(self, query):
        cue=_stem(_content(query)) if isinstance(query,str) else _stem(set(query))
        if not cue: return None
        pet_query=bool(cue&_stem({"pet","animal"}))
        name_query=("name" in cue) and not (cue&REL_S)
        if self._index is None or self._index_len!=len(self.episodes): self._build_index()
        cand=set()
        for s in cue: cand.update(self._index.get(s,()))
        if pet_query:
            for s in PETS: cand.update(self._index.get(s,()))
        best=None; bk=(-1,-1,-1,0,-1)
        for i in sorted(cand):
            e=self.episodes[i]
            es=set(e["s"]); ov=len(cue&es); es_pet=bool(es&PETS)
            if ov<1 and pet_query and es_pet: ov=1
            if ov<1: continue
            if (es&REL_S)-cue and not (pet_query and es_pet): continue
            if (cue&REL_S)-es and not (pet_query and es_pet): continue
            selfp=1 if (name_query and e.get("self")) else 0
            spec=-len(es-cue-STOPVAL_S)
            sc=(ov,selfp,1 if e.get("h","") in cue else 0,spec,i)
            if sc>bk: bk=sc; best=e
        if best is None: return None
        bes=set(best["s"]); cov=len(cue&bes)/max(1,len(cue))
        if pet_query and (bes&PETS): cov=1.0
        val,echo=_pick_value(best,cue,bool(cue&_stem({"many","much","number"})))
        return {"fact":best["t"],"value":val,"coverage":cov,"overlap":bk[0],"echo":echo}
    def dump(self): return json.dumps([[e["t"],1 if e.get("self") else 0] for e in self.episodes], ensure_ascii=False)
    @classmethod
    def load(cls, blob, max_facts=500):
        n=cls(max_facts)
        for t,_flag in json.loads(blob or "[]"):
            e=_encode(LEADIN.sub("",str(t)),n.last_entity)
            if e: n.episodes.append(e)
            m=ENT.search(str(t))
            if m: n.last_entity=m.group(1).lower()
        return n
    @property
    def fact_count(self): return len(self.episodes)
