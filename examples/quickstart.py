"""neuron-db in 20 lines. Run: python examples/quickstart.py"""
from neuron_db import NeuronDB

db = NeuronDB("quickstart.db")          # a database of neurons, one file

# write facts in plain language
for fact in [
    "my name is Marisol",
    "the wifi password is hunter2",
    "only the first 1,000 users score 150,000 coins",
    "i adopted a puppy. her name is Mochi",
]:
    print("you   :", fact)
    print("neuron:", db.turn("alice", fact)["reply"])

# ask them back — even after a restart, they're in quickstart.db
for q in ["what is my name?", "what is the wifi password?",
          "how many coins?", "what is my puppy's name?",
          "what is my favorite color?"]:   # never told -> abstains
    print("you   :", q)
    print("neuron:", db.turn("alice", q)["reply"])

print("\nstats:", db.stats("alice"))
db.close()
