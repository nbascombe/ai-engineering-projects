import sys
import time
import requests

BASE = "http://localhost:8000"
DEFAULTS = [
    "How many players are on court in doubles?",
    "How long can a tennis racket be?",
    "What is the maximum height of the net at the posts?",
]
questions = sys.argv[1:] or DEFAULTS

t0 = time.perf_counter()
jobs = {}
for q in questions:
    r = requests.post(f"{BASE}/jobs", json={"content": q})
    print(f"{time.perf_counter() - t0:5.1f}s  POST {r.status_code}  {q}")
    if r.status_code == 202:
        jobs[r.json()["job_id"]] = q

last = {}
while jobs and any(last.get(j) not in ("finished", "failed") for j in jobs):
    for job_id, q in jobs.items():
        if last.get(job_id) in ("finished", "failed"):
            continue
        s = requests.get(f"{BASE}/jobs/{job_id}").json()["status"]
        if s != last.get(job_id):
            print(f"{time.perf_counter() - t0:5.1f}s  {s:<9} {q}")
            last[job_id] = s
    time.sleep(1)