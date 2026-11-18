# RQ learning notes

The experiments behind the `/jobs` design (README section 10). They were run in a scratch project, which is git-ignored, so the findings are recorded here.

| Topic | Finding | Where it shows up in the repo |
|---|---|---|
| Basic flow | Producer enqueues, a worker drains the queue, the client polls by job ID | `POST /jobs`, `GET /jobs/{id}` |
| Failure isolation | A failing job ends `failed` without stopping the worker or other jobs | `failure_ttl=86400` |
| Result retention | Successful results are kept for only 500s by default, so a polling client could find them gone | `result_ttl=3600` |
| Retries | RQ retries only if configured, and does not re-run a job whose worker was killed | `Retry(max=2, interval=[15, 45])`, needs `--with-scheduler` |
| Scaling workers | More workers cap how many calls run at once, not how often they start | shared Redis limiter in `wait_for_slot` |
| Shutdown | A warm shutdown lets the current job finish; a hard kill leaves the job in limbo until RQ's cleanup marks it `failed` | README section 10, limitations |
| macOS | The default forking worker crashed on every job; `rq.SimpleWorker` fixed it | worker command in the README |

Not yet measured: behaviour with several workers under real load (see "What I'd improve" in README section 10).
