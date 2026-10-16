"""
Common code for rag_api.py and eval_rag.py.

- MODEL / SYSTEM_INSTRUCTION: one definition. The eval's answer-cache key includes
  both, so a change here invalidates stale cached answers automatically.
- squash(): the single phrase-matching normaliser (letters and digits only), so
  the eval, --ranks and any other script agree on what "the chunk contains this
  phrase" means.
- log_llm_call(): one JSON line per LLM call (prompt, response, latency, tokens)
  appended to rag_chatbot/logs/llm_calls.jsonl.
"""
import os
import re
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "logs", "llm_calls.jsonl")

MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You are TennisRulesBot, a helpful tennis rules assistant. Answer questions "
    "using only the context provided. If the answer is not in the context, say so."
)


def squash(s):
    """Lowercase and keep only letters and digits.

    Survives PDF extraction quirks such as 'tie -break', stray brackets and
    line-break hyphens, which whitespace-only stripping does not.
    """
    return re.sub(r"[^a-z0-9]", "", s.lower())


def usage_dict(usage):
    """Normalise a Gemini usage_metadata object (or None) into plain ints."""
    def get(name):
        return getattr(usage, name, None) if usage is not None else None
    return {
        "prompt_tokens": get("prompt_token_count"),
        "output_tokens": get("candidates_token_count"),
        "total_tokens": get("total_token_count"),
        "thinking_tokens": get("thoughts_token_count")
    }


def log_llm_call(source, model, question, prompt, response, latency_s,
                 usage=None, ttft_s=None):
    """Append one structured record per LLM call. Never raises: logging must not break a request."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "model": model,
            "question": question,
            "prompt": prompt,
            "response": response,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "ttft_s": round(ttft_s, 3) if ttft_s is not None else None,
            **usage_dict(usage),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[log_llm_call failed: {e}]")
