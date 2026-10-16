"""
Eval harness for TennisRulesBot.

Scores retrieval and generation separately against eval_set.json, running each
question several times because identical inputs can give different answers.
Bypasses the Redis cache, so cached API answers can't hide regressions.

API errors (e.g. 429 quota, empty/blocked responses) are recorded as errors, NOT
as failures, are never cached, and are excluded from accuracy.

eval_set.json fields (per question):
  must_contain      list of groups; every group needs at least one matching term
  must_not_contain  terms that make the answer fail if present (wrong-rule answers,
                    or a model answering from its own knowledge on a refusal item)
  expected_phrase   a string, or a list of strings, identifying the answer chunk(s)
  phrase_match      "all" (default): every phrase must be retrieved (multi-fact answers)
                    "any": any one phrase is enough (the PDF repeats the fact)
  should_refuse     true for out-of-scope questions

Quota-friendly features:
  --retrieval-only   score retrieval only; makes no generation calls
  --ranks            rank of every answer chunk across the whole collection, plus the
                     smallest k that would hit each question (replaces diagnose_phrases.py
                     and check_retrieval.py)
  fail-fast          stops calling the API after repeated errors
  answer cache       successful answers are saved to eval_results/answer_cache.json,
                     so a rerun only spends quota on what is missing
  --max-calls        cap on NEW generation calls per run (free tier: 20/day)

LangSmith tracing is switched off for eval runs (they'd burn the free-tier trace
cap). Pass --trace to leave it on.

Run from the repo root:
    python -m rag_chatbot.eval_rag --ranks
    python -m rag_chatbot.eval_rag --retrieval-only --k 8
    python -m rag_chatbot.eval_rag --runs 1 --label baseline_1run
    python -m rag_chatbot.eval_rag --runs 3 --label baseline
    python -m rag_chatbot.eval_rag --compare rag_chatbot/eval_results/a.json rag_chatbot/eval_results/b.json
"""
import os
import sys

# Must happen before rag_api (and therefore langsmith) is imported.
if "--trace" not in sys.argv:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

import re
import json
import time
import hashlib
import argparse
import statistics
from datetime import datetime

import chromadb

# Reuse the real pipeline pieces so the eval tests production logic, not a copy.
# Importing rag_api does not need Redis to be running (the connection is lazy).
from rag_chatbot.rag_api import client, find_relevant_chunks, build_prompt, embed_text
from rag_chatbot.rag_common import MODEL, SYSTEM_INSTRUCTION, squash, usage_dict, log_llm_call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SET_PATH = os.path.join(BASE_DIR, "eval_set.json")
RESULTS_DIR = os.path.join(BASE_DIR, "eval_results")
CACHE_PATH = os.path.join(RESULTS_DIR, "answer_cache.json")

# Phrases that indicate the bot declined to answer. Shared by all should_refuse items.
REFUSAL_MARKERS = [
    "does not contain", "doesn't contain", "not contain", "not provide",
    "no information", "not in the context", "cannot answer", "can't answer",
    "not mentioned", "sorry",
]

# A real answer or refusal never quotes the system prompt. If it does, the model
# echoed its instructions (seen on Q8 at k=3), which must not count as a refusal.
DEGENERATE_MARKERS = [
    "you are tennisrulesbot",
    "answer questions using only the context provided",
]

MAX_CONSECUTIVE_ERRORS = 3   # stop calling the API after this many errors in a row
DEFAULT_MAX_CALLS = 18       # free tier allows 20/day; leave headroom


# ---------- helpers ----------

def contains_term(text, term):
    """Case-insensitive whole-word match.

    Pure-digit terms get stricter rules: the digit must not touch another digit or
    sit in a score like '6-6', so '6' matches 'every 6 points' but not 'at 6-6',
    and '7' doesn't match '17'.
    """
    t = term.lower()
    if t.isdigit():
        pattern = r"(?<!\d)(?<!\d[-\u2013])" + re.escape(t) + r"(?!\d)(?![-\u2013]\d)"
    else:
        pattern = r"(?<!\w)" + re.escape(t) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def load_collection():
    chroma = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
    return chroma.get_collection("tennis_rules")


def load_eval_set():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def is_error(answer):
    return answer.startswith("[ERROR")


def phrases_of(item):
    """expected_phrase as a list (it may be absent, a string, or a list)."""
    p = item.get("expected_phrase")
    if not p:
        return []
    return [p] if isinstance(p, str) else list(p)


# ---------- answer cache ----------

def load_cache():
    """Entries are dicts {answer, latency_s, prompt_tokens, output_tokens}.

    Older caches stored bare strings; those are upgraded on load. Empty answers
    (which an earlier version could cache) are dropped so they get retried.
    """
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    cleaned = {}
    for key, value in raw.items():
        entry = value if isinstance(value, dict) else {"answer": value}
        answer = entry.get("answer")
        if isinstance(answer, str) and answer.strip() and not is_error(answer):
            cleaned[key] = entry
    return cleaned


def save_cache(cache):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def cache_key(prompt, run_idx):
    """Model, system instruction, full prompt and run index. Changing any of them
    invalidates the entry, so a stale answer can't be reused after the bot changes."""
    raw = f"{MODEL}|{SYSTEM_INSTRUCTION}|{prompt}|{run_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------- generation ----------

def generate_answer(question, context, retries=2):
    """One uncached generation. Returns a dict with the answer plus latency and token counts.

    Errors and empty responses come back as '[ERROR ...]' answers, so they are
    never scored as failures and never cached. Gives up immediately on a daily quota.
    """
    prompt = build_prompt(context, question)
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            response = client.models.generate_content(
                model=MODEL,
                config={"system_instruction": SYSTEM_INSTRUCTION},
                contents=prompt,
            )
            latency = time.perf_counter() - start
            text = (response.text or "").strip()
            usage = getattr(response, "usage_metadata", None)

            if not text:
                # Safety block or empty candidate. Not a model failure, and not cacheable.
                return {"answer": "[ERROR: empty response (possibly blocked)]"}

            log_llm_call("eval", MODEL, question, prompt, text, latency, usage)
            counts = usage_dict(usage)
            return {
                "answer": text,
                "latency_s": round(latency, 3),
                "prompt_tokens": counts["prompt_tokens"],
                "output_tokens": counts["output_tokens"],
            }
        except Exception as e:
            message = str(e)
            rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
            # A daily quota won't recover within a run, so don't keep hammering it.
            if rate_limited and ("PerDay" in message or "per day" in message.lower()):
                return {"answer": f"[ERROR: daily quota exhausted. {message[:150]}]"}
            if attempt == retries - 1:
                return {"answer": f"[ERROR: {message[:200]}]"}
            if rate_limited:
                match = re.search(r"retry in ([\d.]+)s", message) or re.search(r"'retryDelay': '(\d+)s'", message)
                wait = float(match.group(1)) + 2 if match else 30 * (attempt + 1)
                print(f"   rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{retries})")
            else:
                wait = 2 ** (attempt + 1)
            time.sleep(wait)


# ---------- scoring ----------

def score_retrieval(item, context):
    """Return (hit, rank). rank is 1-based among the retrieved chunks, None if missed.

    With phrase_match "all" (default) every expected phrase must appear in some
    retrieved chunk and rank is the deepest one needed. With "any", the best
    (shallowest) hit counts.
    """
    phrases = phrases_of(item)
    if not phrases:
        return None, None
    mode = item.get("phrase_match", "all")
    squashed = [squash(c) for c in context]
    ranks = []
    for phrase in phrases:
        target = squash(phrase)
        ranks.append(next((i for i, c in enumerate(squashed, 1) if target in c), None))
    found = [r for r in ranks if r]
    if mode == "any":
        return (True, min(found)) if found else (False, None)
    return (True, max(ranks)) if all(ranks) else (False, None)


def score_answer(item, answer):
    """Return (passed, reason). Call only on non-error answers."""
    lowered = answer.lower()

    if any(marker in lowered for marker in DEGENERATE_MARKERS):
        return False, "degenerate output (echoed the system instruction)"

    banned = [t for t in item.get("must_not_contain", []) if contains_term(answer, t)]
    if banned:
        return False, f"contains forbidden term(s) {banned}"

    if item["should_refuse"]:
        if any(marker in lowered for marker in REFUSAL_MARKERS):
            return True, ""
        return False, "no refusal marker found"

    # Every group must be satisfied by at least one of its terms.
    missing = [g for g in item["must_contain"] if not any(contains_term(answer, t) for t in g)]
    if missing:
        return False, f"missing one of {missing}"
    return True, ""


# ---------- retrieval-only and ranks modes ----------

def run_retrieval_only(k):
    """Embedding calls only. No generation quota used."""
    eval_set = load_eval_set()
    collection = load_collection()
    hits = 0
    scorable = 0
    print(f"\nRetrieval-only eval, k={k}\n")
    for item in eval_set:
        context = find_relevant_chunks(item["question"], collection, k)
        hit, rank = score_retrieval(item, context)
        if hit is not None:
            scorable += 1
            hits += hit
        label = "n/a" if hit is None else (f"hit #{rank}" if hit else "MISS")
        print(f"Q{item['id']:<3} {item['type']:<17} {label:<8} {item['question'][:50]}")
    print(f"\nRetrieval hit rate at k={k}: {hits}/{scorable}")


def run_ranks():
    """Rank of each answer chunk across the WHOLE collection, and the smallest k per question.

    Also validates the evaluator: a phrase found in no chunk is reported so the
    expected_phrase can be fixed before any retrieval number is trusted.
    """
    eval_set = load_eval_set()
    collection = load_collection()
    data = collection.get(include=["documents"])
    ids, squashed_docs = data["ids"], [squash(d) for d in data["documents"]]
    total = collection.count()

    print(f"\nChunk ranks across all {total} chunks (1 = best)\n")
    min_ks = []
    problems = []
    for item in eval_set:
        phrases = phrases_of(item)
        if not phrases:
            continue
        ranking = collection.query(query_embeddings=[embed_text(item["question"])], n_results=total)["ids"][0]
        per_phrase_best = []
        print(f"Q{item['id']:<3} {item['type']:<17} {item['question'][:55]}")
        for phrase in phrases:
            holders = [cid for cid, doc in zip(ids, squashed_docs) if squash(phrase) in doc]
            if not holders:
                print(f"      PHRASE NOT FOUND in any chunk: {phrase!r}  -> fix expected_phrase")
                problems.append(item["id"])
                per_phrase_best.append(None)
                continue
            ranks = {cid: ranking.index(cid) + 1 for cid in holders}
            per_phrase_best.append(min(ranks.values()))
            print(f"      {phrase[:45]!r}: {ranks}")
        if None in per_phrase_best:
            continue
        mode = item.get("phrase_match", "all")
        min_ks.append((item["id"], max(per_phrase_best) if mode == "all" else min(per_phrase_best)))

    if min_ks:
        print("\nSmallest k that retrieves each question's answer:")
        print("  " + ", ".join(f"Q{qid}:{k}" for qid, k in min_ks))
        for k in (3, 5, 8, 10):
            hit = sum(1 for _, mk in min_ks if mk <= k)
            print(f"  k={k:<2} -> {hit}/{len(min_ks)}")
        print(f"  k needed for all: {max(mk for _, mk in min_ks)}  "
              "(a high number means retrieval is weak; raising k treats the symptom)")
    if problems:
        print(f"\nQuestions with unmatched phrases: {sorted(set(problems))}")


# ---------- eval loop ----------

def evaluate_question(item, collection, k, runs, delay, cache, state):
    start = time.perf_counter()
    context = find_relevant_chunks(item["question"], collection, k)
    retrieval_s = time.perf_counter() - start
    retrieval_hit, retrieval_rank = score_retrieval(item, context)
    prompt = build_prompt(context, item["question"])

    run_results = []
    for run_idx in range(runs):
        key = cache_key(prompt, run_idx)
        made_call = False

        if key in cache:
            entry = cache[key]
            state["cached"] += 1
        elif state["stop"]:
            entry = {"answer": "[ERROR: skipped, API failing repeatedly]"}
        elif state["new_calls"] >= state["max_calls"]:
            entry = {"answer": "[ERROR: skipped, call budget reached]"}
        else:
            entry = generate_answer(item["question"], context)
            state["new_calls"] += 1
            made_call = True

        answer = entry["answer"]
        errored = is_error(answer)

        if made_call:
            if errored:
                state["consecutive_errors"] += 1
                if state["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
                    state["stop"] = True
                    print(f"   {MAX_CONSECUTIVE_ERRORS} consecutive API errors, no more generation calls this run.")
            else:
                state["consecutive_errors"] = 0
                cache[key] = entry
                save_cache(cache)

        passed, reason = (None, "") if errored else score_answer(item, answer)
        run_results.append({
            "answer": answer,
            "errored": errored,
            "passed": passed,
            "fail_reason": reason,
            "latency_s": entry.get("latency_s"),
            "prompt_tokens": entry.get("prompt_tokens"),
            "output_tokens": entry.get("output_tokens"),
        })

        # Pace only real, successful calls. Cache hits and errors need no delay.
        if made_call and not errored:
            time.sleep(delay)

    scored = [r for r in run_results if not r["errored"]]
    passed_count = sum(r["passed"] for r in scored)
    errored_count = runs - len(scored)
    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "retrieval_hit": retrieval_hit,
        "retrieval_rank": retrieval_rank,
        "retrieval_s": round(retrieval_s, 3),
        "context_chars": sum(len(c) for c in context),
        "runs": runs,
        "runs_scored": len(scored),
        "runs_errored": errored_count,
        "runs_passed": passed_count,
        "complete": errored_count == 0,
        "all_runs_passed": errored_count == 0 and passed_count == runs,
        "run_results": run_results,
    }


def run_eval(k, runs, label, delay, max_calls, use_cache):
    eval_set = load_eval_set()
    collection = load_collection()
    cache = load_cache() if use_cache else {}
    state = {
        "new_calls": 0, "cached": 0, "consecutive_errors": 0,
        "stop": False, "max_calls": max_calls,
    }

    results = []
    for item in eval_set:
        print(f"Evaluating Q{item['id']}: {item['question']}")
        results.append(evaluate_question(item, collection, k, runs, delay, cache, state))

    retrieval_items = [r for r in results if r["retrieval_hit"] is not None]
    retrieval_hits = sum(r["retrieval_hit"] for r in retrieval_items)
    complete = [r for r in results if r["complete"]]
    strict_pass = sum(r["all_runs_passed"] for r in complete)
    scored_runs = sum(r["runs_scored"] for r in results)
    passed_runs = sum(r["runs_passed"] for r in results)
    errored_runs = sum(r["runs_errored"] for r in results)

    good_runs = [rr for r in results for rr in r["run_results"] if not rr["errored"]]
    latencies = [rr["latency_s"] for rr in good_runs if rr["latency_s"] is not None]
    prompt_tokens = [rr["prompt_tokens"] for rr in good_runs if rr["prompt_tokens"] is not None]
    output_tokens = [rr["output_tokens"] for rr in good_runs if rr["output_tokens"] is not None]

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "config": {"model": MODEL, "k": k, "runs_per_question": runs, "delay_s": delay,
                   "max_calls": max_calls, "cache_used": use_cache},
        "summary": {
            "retrieval_hit_rate": f"{retrieval_hits}/{len(retrieval_items)}",
            "answer_accuracy_strict": f"{strict_pass}/{len(complete)} complete questions",
            "answer_pass_rate_per_run": f"{passed_runs}/{scored_runs} scored runs",
            "errored_runs": errored_runs,
            "incomplete_questions": [r["id"] for r in results if not r["complete"]],
            "new_api_calls": state["new_calls"],
            "answers_from_cache": state["cached"],
            "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
            "mean_prompt_tokens": round(statistics.mean(prompt_tokens)) if prompt_tokens else None,
            "mean_output_tokens": round(statistics.mean(output_tokens)) if output_tokens else None,
            "mean_context_chars": round(statistics.mean(r["context_chars"] for r in results)),
        },
        "results": results,
    }


# ---------- output ----------

def fmt(v, suffix=""):
    return "n/a (answers came from the old cache)" if v is None else f"{v}{suffix}"

def print_report(report):
    print("\n" + "=" * 78)
    print(f"Eval: {report['label']}   k={report['config']['k']}   "
          f"runs={report['config']['runs_per_question']}   {report['timestamp']}")
    print("=" * 78)
    print(f"{'Q':<3} {'type':<17} {'retrieval':<10} {'answer':<10} question")
    for r in report["results"]:
        if r["retrieval_hit"] is None:
            retrieval = "n/a"
        elif r["retrieval_hit"]:
            retrieval = f"hit (#{r['retrieval_rank']})"
        else:
            retrieval = "MISS"
        answer = f"{r['runs_passed']}/{r['runs_scored']}"
        if r["runs_errored"]:
            answer += f" +{r['runs_errored']}err"
        print(f"{r['id']:<3} {r['type']:<17} {retrieval:<10} {answer:<10} {r['question'][:38]}")

    s = report["summary"]
    print("-" * 78)
    print(f"Retrieval hit rate:         {s['retrieval_hit_rate']}")
    print(f"Answer accuracy (strict):   {s['answer_accuracy_strict']}   (passed on every run)")
    print(f"Answer pass rate (per run): {s['answer_pass_rate_per_run']}")
    print(f"API calls this run:         {s['new_api_calls']} new, {s['answers_from_cache']} answers from cache")
    print(f"Cost at k={report['config']['k']}:".ljust(28) +
          f"median latency {fmt(s['median_latency_s'], 's')}, "
          f"mean prompt {fmt(s['mean_prompt_tokens'])} tokens, "
          f"mean output {fmt(s['mean_output_tokens'])} tokens, "
          f"mean context {s['mean_context_chars']} chars")
    if s["errored_runs"]:
        print(f"\n  WARNING: {s['errored_runs']} runs hit API errors or were skipped, and are excluded from accuracy.")
        print(f"  Incomplete questions: {s['incomplete_questions']}. Don't quote these numbers until a run has 0 errors.")
        print("  Rerun later: answers already collected are cached, so only the gaps cost quota.")
    print()

    # Why each failed run failed.
    for r in report["results"]:
        for i, rr in enumerate(r["run_results"], 1):
            if rr["passed"] is False:
                print(f"  Q{r['id']} run {i} FAILED: {rr['fail_reason']}")

    # The interesting cases: retrieval and generation failing independently.
    for r in report["results"]:
        if r["runs_scored"] == 0:
            continue
        if r["retrieval_hit"] is False and r["runs_passed"] > 0:
            print(f"  Q{r['id']}: retrieval missed but the answer still passed sometimes (lucky, model knowledge, or a loose check).")
        if r["retrieval_hit"] is True and r["runs_passed"] < r["runs_scored"]:
            print(f"  Q{r['id']}: right chunk retrieved but answer failed {r['runs_scored'] - r['runs_passed']}x (generation or scoring problem).")


def save_report(report):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"eval_{stamp}_{report['label']}.json" if report["label"] else f"eval_{stamp}.json"
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {path}")


def compare_reports(path_a, path_b):
    """Print a before/after view of two saved runs."""
    with open(path_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        b = json.load(f)

    print(f"\nBEFORE: {a['label'] or path_a}  k={a['config']['k']}  {a['summary']}")
    print(f"AFTER:  {b['label'] or path_b}  k={b['config']['k']}  {b['summary']}\n")
    by_id_a = {r["id"]: r for r in a["results"]}
    for rb in b["results"]:
        ra = by_id_a.get(rb["id"])
        if not ra:
            continue
        before = (ra["retrieval_hit"], ra["runs_passed"], ra["runs_scored"])
        after = (rb["retrieval_hit"], rb["runs_passed"], rb["runs_scored"])
        if before != after:
            print(f"Q{rb['id']}: retrieval {ra['retrieval_hit']} -> {rb['retrieval_hit']}, "
                  f"answer {ra['runs_passed']}/{ra['runs_scored']} -> {rb['runs_passed']}/{rb['runs_scored']}  | {rb['question'][:45]}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TennisRulesBot")
    parser.add_argument("--k", type=int, default=3, help="chunks to retrieve (default 3, matches the API)")
    parser.add_argument("--runs", type=int, default=3, help="generations per question")
    parser.add_argument("--label", default="baseline", help="short name for this run, used in the filename")
    parser.add_argument("--delay", type=float, default=7.0,
                        help="seconds between generation calls (default 7, ~8 requests/min for the free tier)")
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                        help=f"max NEW generation calls this run (default {DEFAULT_MAX_CALLS})")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and don't write the answer cache (forces fresh generations)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="score retrieval only; makes no generation calls")
    parser.add_argument("--ranks", action="store_true",
                        help="show answer-chunk ranks across all chunks and the smallest k per question")
    parser.add_argument("--trace", action="store_true",
                        help="leave LangSmith tracing on (off by default for evals)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="compare two saved result files")
    args = parser.parse_args()

    if args.compare:
        compare_reports(*args.compare)
        return

    if args.ranks:
        run_ranks()
        return

    if args.retrieval_only:
        run_retrieval_only(args.k)
        return

    report = run_eval(args.k, args.runs, args.label, args.delay, args.max_calls, not args.no_cache)
    print_report(report)
    save_report(report)


if __name__ == "__main__":
    main()
