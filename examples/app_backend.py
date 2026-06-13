"""Using neuron-db as a plain app/website database -- NO LLM, NO model, NO neocortex.

Your app is the neocortex: it decides what to store and what to ask. neuron-db is just the
database. This is the whole dependency list: the Python standard library.

    python examples/app_backend.py
"""
from neuron_db import NeuronDB

db = NeuronDB("app.db")          # one SQLite file; durable, never decays

# a web request handler might do:  (neuron id = the user/session/scope)
def on_signup(user_id, name, plan, city):
    db.turn(user_id, f"my name is {name}")
    db.turn(user_id, f"my plan is {plan}")
    db.turn(user_id, f"my city is {city}")

def profile_field(user_id, question):
    return db.get(user_id, question)          # exact value or None

on_signup("user:42", "Marisol", "pro", "Halifax")
on_signup("user:99", "Viktor", "free", "Oslo")

print("user 42 name:", profile_field("user:42", "what is my name?"))   # Marisol
print("user 42 plan:", profile_field("user:42", "what is my plan?"))   # pro
print("user 99 city:", profile_field("user:99", "what is my city?"))   # Oslo
print("unknown field:", profile_field("user:42", "what is my shoe size?"))  # None
print("users in db:", db.neurons())

# durability: reopen the file, data is still there. no model ever loaded.
db.close()
print("name after restart:", NeuronDB("app.db").get("user:42", "what is my name?"))
