"""neuron-db HTTP client (works from any language; this one uses Python requests).
The database is Rust/SQLite under the server — your app just speaks HTTP.
Run: pip install requests && python python_client.py"""
import os, requests
BASE = os.environ.get("NEURON_BASE", "http://localhost:8088")
KEY  = os.environ.get("NEURON_DB_KEY")
H = {"content-type": "application/json", **({"authorization": f"Bearer {KEY}"} if KEY else {})}

def remember(scope, message): return requests.post(f"{BASE}/v1/{scope}", json={"message": message}, headers=H).json()
def ask(scope, query):        return requests.post(f"{BASE}/v1/{scope}/get", json={"query": query}, headers=H).json().get("value")
def forget(scope, match):     return requests.post(f"{BASE}/v1/{scope}/forget", json={"match": match}, headers=H).json()

if __name__ == "__main__":
    remember("user:42", "the plan is pro")
    remember("user:42", "the region is us-west-2")
    print("plan   =", ask("user:42", "what plan?"))
    print("region =", ask("user:42", "where is the region?"))
    print("forget =", forget("user:42", "region"))
