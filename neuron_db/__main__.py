"""CLI:  python -m neuron_db serve   |   python -m neuron_db chat   |   python -m neuron_db demo"""
import sys, argparse
from .db import NeuronDB
from .server import serve


def chat(db_path: str, nid: str):
    db = NeuronDB(db_path)
    print(f"neuron-db chat -- neuron '{nid}' in {db_path}. tell me things, ask them back. ctrl-c to leave.")
    try:
        while True:
            u = input("you: ").strip()
            if not u: continue
            if u == "/forget": print("neuron:", db.forget(nid)); continue
            print("neuron:", db.turn(nid, u)["reply"])
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")


def demo(db_path: str = ":memory:"):
    db = NeuronDB(db_path)
    script = ["hello", "my name is Gary", "the first 1,000 users get 150,000 coins",
              "how many coins?", "how many users?", "the wifi password is hunter2",
              "what is the wifi password?", "what is my name?", "what is my blood type?", "17 + 25"]
    for u in script:
        print(f"you   : {u}\nneuron: {db.turn('demo', u)['reply']}")


def main():
    ap = argparse.ArgumentParser(prog="neuron_db")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve"); s.add_argument("--db", default="neurons.db"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8088); s.add_argument("--max-facts", type=int, default=500)
    c = sub.add_parser("chat"); c.add_argument("--db", default="neurons.db"); c.add_argument("--neuron", default="me")
    sub.add_parser("demo")
    a = ap.parse_args()
    if a.cmd == "serve": serve(a.db, a.host, a.port, a.max_facts)
    elif a.cmd == "chat": chat(a.db, a.neuron)
    elif a.cmd == "demo": demo()
    else: ap.print_help()


if __name__ == "__main__":
    main()
