import sys
import time
import requests

BASE = "http://localhost:8000"
question = " ".join(sys.argv[1:]) or "What is a tiebreak?"

t0 = time.perf_counter()
r = requests.post(f"{BASE}/jobs", json={"content": question})
print(f"POST {r.status_code} in {time.perf_counter() - t0:.3f}s -> {r.json()}")

if r.status_code == 202:
    job_id = r.json()["job_id"]
    while True:
        s = requests.get(f"{BASE}/jobs/{job_id}").json()
        print(f"  {time.perf_counter() - t0:6.1f}s  {s['status']}  {s.get('position', '')}")
        if s["status"] not in ("queued", "started", "scheduled", "deferred"):
            print(s)
            break
        time.sleep(1)