"""
Tiered generation with fallback, for /output.

stream_with_fallback() tries each model in FALLBACK_CHAIN in order. It can only
fall back BEFORE the first text chunk arrives: once a token has reached the
client we are committed to that tier, and a later failure propagates to the
caller (which ends the stream and logs it).

Fault injection (for testing the chain, read at call time so no restart is
needed if you change it in the same process, but uvicorn --reload restarts anyway):
    FORCE_FAIL=tier0            tier 0 raises a fake 503 -> falls back
    FORCE_FAIL=tier0,tier1      both fail -> stale cache / degraded message
    FORCE_FAIL=tier0:400        tier 0 raises a fake 400 -> must NOT fall back
"""
import os
import asyncio
from google.genai import errors
from rag_chatbot.rag_common import FALLBACK_CHAIN, TTFT_TIMEOUT_S, SYSTEM_INSTRUCTION
import time

# Worth trying another model for: rate limit and server-side errors.
# Not worth it: 400 (our bug), 401/403 (same key fails on every tier).
FALLBACK_CODES = {429, 500, 502, 503, 504}


class InjectedFault(Exception):
    """Fake API error used to force a tier to fail."""
    def __init__(self, code=503):
        super().__init__(f"injected fault (code {code})")
        self.code = code


def is_fallback_worthy(exc):
    """True if falling back to the next tier could help."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, StopAsyncIteration)):
        return True  # hung call, or an empty stream
    if isinstance(exc, (InjectedFault, errors.ClientError, errors.ServerError)):
        return getattr(exc, "code", None) in FALLBACK_CODES
    return False


def _forced_fault(tier):
    """Return an InjectedFault if FORCE_FAIL targets this tier, else None."""
    for item in os.getenv("FORCE_FAIL", "").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, code = item.partition(":")
        if name == f"tier{tier}":
            return InjectedFault(int(code) if code else 503)
    return None


async def _first_text_chunk(it):
    """Advance the stream to its first chunk that has text (skips metadata-only chunks)."""
    async for chunk in it:
        if chunk.text:
            return chunk
    raise StopAsyncIteration  # stream ended with no text


async def stream_with_fallback(client, prompt):
    """Yield (tier, model, chunk). Yields nothing if every tier failed."""
    for tier, model in enumerate(FALLBACK_CHAIN):
        try:
            fault = _forced_fault(tier)
            if fault:
                raise fault
            t0 = time.perf_counter()
            stream = await client.aio.models.generate_content_stream(
                model=model,
                config={"system_instruction": SYSTEM_INSTRUCTION},
                contents=prompt,
            )
            it = stream.__aiter__()
            first = await asyncio.wait_for(_first_text_chunk(it), TTFT_TIMEOUT_S)
            print(f"[tier {tier} {model}: first text chunk after {time.perf_counter() - t0:.2f}s]")
        except Exception as e:
            if not is_fallback_worthy(e):
                raise  # a bug or auth problem: falling back would hide it
            print(f"[tier {tier} {model} failed: {type(e).__name__}: {e}]")
            continue

        # Committed: the first token exists, so there is no falling back from here.
        yield tier, model, first
        async for chunk in it:
            yield tier, model, chunk
        return
