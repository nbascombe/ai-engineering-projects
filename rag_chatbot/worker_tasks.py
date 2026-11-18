"""
RQ job: answer one question off the request thread.

Run the worker from the repo root:
    rq worker llm --with-scheduler
(--with-scheduler is needed for Retry(interval=...) to work; check the RQ docs for your installed version.)
"""
import os
import time

import chromadb
from redis import Redis

from rag_chatbot.rag_api import (
    BASE_DIR, FALLBACK_NOTICE, STALE_NOTICE, build_prompt, client,
    find_relevant_chunks, make_cache_key, make_stale_key,
)
from rag_chatbot.rag_common import (
    FALLBACK_CHAIN, K, STALE_TTL_S, SYSTEM_INSTRUCTION, log_llm_call, validate_output,
)
from rag_chatbot.fallback import _forced_fault, is_fallback_worthy

# Text connection for the app's cache and the rate limiter. This is NOT the RQ
# connection (RQ's must not use decode_responses=True).
cache = Redis(decode_responses=True)

GEMINI_RPM = int(os.getenv("GEMINI_RPM", "5"))
GUARDRAIL_MESSAGE = "Sorry, I can't help with that request."

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        chroma = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
        _collection = chroma.get_collection("tennis_rules")
    return _collection


class AllTiersFailed(Exception):
    pass


def wait_for_slot(model, limit=GEMINI_RPM, window=60, max_wait=90):
    """Block until this model has a free request slot in the current window.

    Shared across all workers via Redis. max_wait is longer than one window so a
    job waits for the next window instead of degrading to the backup model at
    once. Raises TimeoutError (fallback-worthy) if no slot appears in time.
    """
    deadline = time.monotonic() + max_wait
    while True:
        key = f"ratelimit:{model}:{int(time.time() // window)}"
        count = cache.incr(key)
        if count == 1:
            cache.expire(key, window * 2)
        if count <= limit:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no {model} rate-limit slot within {max_wait}s")
        time.sleep(1)


def generate_with_fallback(prompt):
    """Sync version of the /output chain: try each model in order."""
    last_exc = None
    for tier, model in enumerate(FALLBACK_CHAIN):
        try:
            fault = _forced_fault(tier)      # FORCE_FAIL works in the worker too
            if fault:
                raise fault
            wait_for_slot(model)
            start = time.perf_counter()
            response = client.models.generate_content(
                model=model,
                config={"system_instruction": SYSTEM_INSTRUCTION},
                contents=prompt,
            )
            text = (response.text or "").strip()
            if not text:
                raise TimeoutError("empty response")   # treated like a failed tier
            return tier, model, text, response.usage_metadata, time.perf_counter() - start
        except Exception as e:
            if not is_fallback_worthy(e):
                raise                        # a bug or bad key: don't hide it
            print(f"[tier {tier} {model} failed: {type(e).__name__}: {e}]")
            last_exc = e
    raise AllTiersFailed("every model tier failed") from last_exc


def answer_question(question):
    """The job. Returns a dict that RQ stores as the job result."""
    # Second cache check: an identical job queued earlier may have finished since.
    cache_key = make_cache_key(question)
    cached = cache.get(cache_key)
    if cached:
        return {"answer": cached, "tier": 0, "model": None, "cached": True}

    try:
        context = find_relevant_chunks(question, get_collection(), K)
        prompt = build_prompt(context, question)
        tier, model, text, usage, latency = generate_with_fallback(prompt)
    except Exception as e:
        if not (isinstance(e, AllTiersFailed) or is_fallback_worthy(e)):
            raise                            # bug: RQ marks the job failed
        stale = cache.get(make_stale_key(question))
        if stale:
            log_llm_call(source="worker", model=None, question=question, prompt=None,
                         response=stale, latency_s=None, tier="stale")
            return {"answer": STALE_NOTICE + stale, "tier": "stale", "model": None, "cached": False}
        raise                                # no stale copy: fail so Retry can try again

    ok, reason = validate_output(text)
    log_llm_call(source="worker", model=model, question=question, prompt=prompt,
                 response=text, latency_s=latency, usage=usage, tier=tier,
                 guardrail=None if ok else reason)
    if not ok:
        # Not streamed, so a bad answer is withheld before the client ever sees it.
        # The reason stays in the log (guardrail=reason above), not in the response.
        return {"answer": GUARDRAIL_MESSAGE, "tier": tier, "model": model,
                "cached": False}
    if tier == 0:                            # only primary answers are cached
        cache.set(cache_key, text, ex=6000)
        cache.set(make_stale_key(question), text, ex=STALE_TTL_S)
    answer = text if tier == 0 else FALLBACK_NOTICE + text
    return {"answer": answer, "tier": tier, "model": model, "cached": False}