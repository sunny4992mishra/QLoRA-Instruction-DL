"""LLM judge: scores each answer 1-5 for correctness, helpfulness and Hindi fluency (issue #9).

    export GEMINI_API_KEY=...            # never commit it (or OPENROUTER_API_KEY with --provider openrouter)
    python src/judge.py Qwen2.5-1.5B-Instruct_zeroshot
    python src/judge.py --all            # every results/*.jsonl that has no complete judge file

Input:   results/<run>.jsonl            (id, instruction, reference, output)
Output:  results/judge/<run>.jsonl      one line per answer, the format src/analyze.py reads:
    {"id": ..., "correctness": 4, "helpfulness": 3, "fluency": 5, "judge_model": ..., "judged_at": ...}
A score is null if the judge failed on that answer.

The judge prompt (PROMPT below) is fixed and the same for every run. The judge sees the
question, the reference answer and the answer, and replies with JSON only. Decoding is at
temperature 0. Answers are sent BATCH at a time to save requests (the free tier allows
50 requests a day); every answer is scored independently inside the request.

Resumes like generate.py: answers that already have three scores are skipped, and answers
that failed (null) are tried again on the next run. The file is rewritten in test-set order.
If the daily request limit is hit, the script stops cleanly; run it again later.
Uses only the standard library.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

# Repo root, so the script finds results/ wherever it is run from.
ROOT = Path(__file__).resolve().parent.parent
# Two ways to reach a judge model. Gemini is the default and the one used for every score in the
# report. The OpenRouter Gemma model was tried first and dropped: it was throttled and too lenient
# (5/5/5 for a thin one-line answer).
PROVIDERS = {  # name -> (endpoint, environment variable with the key, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
               "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY",
                   "google/gemma-4-31b-it:free"),
}
CRITERIA = ["correctness", "helpfulness", "fluency"]
# Why clip: a very long reference would make one request huge. 3,000 characters is far above any
# answer (at most 512 tokens); it cut the reference of only 3 prompts.
MAX_CHARS = 3000  # per field; answers are at most 512 tokens, a few references are longer

# Why one fixed prompt: scores stay comparable across all nine systems. It counts looping under
# fluency, unlike the human sheet, and looping answers hold most of the 0.35 fluency gap.
# The fixed judge prompt. 'Strict, impartial', 'score the ANSWER not the reference', a 1/3/5 anchor
# for each criterion, 'use the whole range', 'do not reward length' and 'judge each item on its
# own' are there to reduce known LLM-judge biases (leniency, preferring long answers, copying the
# reference's score). It asks for JSON only so the reply can be parsed by a program. The double
# braces {{ }} are literal braces in str.format.
PROMPT = """You are a strict, impartial evaluator of answers written in Hindi.

For each item below you get a QUESTION, a REFERENCE answer written by a human, and the ANSWER to evaluate. Score the ANSWER (not the reference) from 1 to 5 on three criteria:

correctness: Are the facts and the content right? Compare with the REFERENCE and with what you know. 1 = wrong or invented, 3 = partly right, 5 = fully right. An answer that repeats itself or never finishes can still be correct in its first part.
helpfulness: Does it do what the QUESTION asks, in the requested form, and is it useful and complete? 1 = ignores the question, 3 = partly answers, 5 = fully answers.
fluency: Is it natural, grammatical Hindi in Devanagari? 1 = broken, mostly not Hindi or garbled, 3 = understandable but awkward or with many errors, 5 = natural. Repeated phrases, looping text and unwanted switches to another language lower the score.

Use the whole range. Do not reward length. Judge each item on its own.

Reply with JSON only, no other text: a list with one object per item, in this form:
[{{"n": 1, "correctness": 3, "helpfulness": 4, "fluency": 5}}, ...]

{items}"""

# One item inside the prompt. Every item has a number n, and the judge must repeat it in its reply,
# so each score can be matched to the right answer even if the reply is reordered.
ITEM = """### Item {n}
QUESTION:
{question}

REFERENCE:
{reference}

ANSWER:
{answer}
"""


def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Cut a field to MAX_CHARS and mark the cut with [...], so the judge knows text is missing.
def clip(text):
    text = text.strip()
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " [...]"


# One request = the fixed prompt plus up to 10 numbered items (question, reference, answer). An
# empty answer is shown as '(empty)', so the judge does not skip it.
def build_prompt(rows):
    items = "\n".join(ITEM.format(n=i + 1, question=clip(r["instruction"]), reference=clip(r["reference"]),
                                  answer=clip(r["output"]) or "(empty)") for i, r in enumerate(rows))
    return PROMPT.format(items=items)


# Send one request and return the judge's reply text, or None after 6 failed attempts. Uses only
# urllib from the standard library, so it needs no extra packages.
def call(provider, model, prompt, key, retries=6):
    url, _, _ = PROVIDERS[provider]
    if provider == "gemini":
        url = url.format(model=model)
        # The API key goes in a request header and comes from an environment variable
        # (GEMINI_API_KEY), never from the code, so it is never committed to git.
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        # Why temperature 0: to reduce randomness. The judge is still noisy: 45% of 258 identical-
        # answer pairs got different scores.
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                # responseMimeType application/json asks Gemini to answer with valid JSON.
                # maxOutputTokens 8000 leaves room for 10 short score objects.
                "generationConfig": {"temperature": 0, "maxOutputTokens": 8000,
                                     "responseMimeType": "application/json"}}
    else:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 4000}
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            if provider == "gemini":
                return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            return data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            # HTTP 429 with a per-day message = the free daily quota is used up. Stop cleanly
            # (scores so far are already saved) instead of retrying for hours.
            if e.code == 429 and ("per-day" in msg or "PerDay" in msg):
                raise SystemExit("Daily free-request limit reached. Run the same command again later.")
            # These errors (bad request, wrong key, no credit, forbidden, wrong model name) will
            # not fix themselves, so retrying is pointless: stop with the message.
            if e.code in (400, 401, 402, 403, 404):
                raise SystemExit(f"HTTP {e.code}: {msg[:300]}")
            # Why exponential backoff: it gives a rate-limited free API time to recover.
            wait = 10 * 2 ** attempt
        # Network problems, timeouts and malformed replies are temporary: wait and retry.
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, IndexError) as e:
            msg, wait = str(e), 10 * 2 ** attempt
        print(f"  request failed ({msg[:120]}), retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    return None


# Turn the judge's reply into k score dicts, one per item in the request. Anything missing or
# invalid stays None, so that answer is retried on the next run.
def parse(text, k):
    """Return k dicts of scores (None where missing or invalid) from the judge's reply."""
    # Start with all scores None.
    scores = [dict.fromkeys(CRITERIA) for _ in range(k)]
    # Why search for the list: a model can wrap valid JSON in extra text.
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return scores
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return scores
    # Use the item number n from the reply to place each score, and ignore numbers outside 1..k.
    for obj in arr if isinstance(arr, list) else []:
        n = obj.get("n") if isinstance(obj, dict) else None
        if isinstance(n, int) and 1 <= n <= k:
            vals = {c: obj.get(c) for c in CRITERIA}
            # Why reject invalid scores: they stay null and are retried instead of skewing the
            # means.
            # Accept only whole numbers 1 to 5. bool is excluded because True is an int in Python
            # and would pass as a 1.
            if all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5 for v in vals.values()):
                scores[n - 1] = vals
    return scores


# An answer is done when all three criteria have a score.
def complete(row):
    return row is not None and all(row.get(c) is not None for c in CRITERIA)


# Rewrite the whole judge file in test-set order: one line per answer, scored or not (missing
# scores are null).
def write(path, rows, done):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(done.get(r["id"]) or {"id": r["id"], **dict.fromkeys(CRITERIA)}, ensure_ascii=False) + "\n")
    # Why a temp file: a crash during writing can't leave a half-written score file.
    tmp.replace(path)


# Judge one system's 150 answers.
def judge_run(name, args, key):
    rows = load_jsonl(ROOT / "results" / f"{name}.jsonl")
    if not rows:
        print(f"{name}: no results file, skipped")
        return
    out = ROOT / "results" / "judge" / f"{name}.jsonl"
    # Resume: answers that already have three scores are kept and not sent again. So rerunning
    # after a quota stop or crash only does the remaining work, and rerunning on a finished file
    # does nothing.
    done = {r["id"]: r for r in load_jsonl(out) if complete(r)}
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{name}: {len(done)} already scored, {len(todo)} to do", flush=True)
    # Why batches of 10: 1,350 answers fit in 135 free-tier requests. Trade-off: neighbouring
    # answers may affect a score.
    # Send the answers in batches of --batch (10). Answers are in test-set order, so every run's
    # batch k holds the same 10 prompts.
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        text = call(args.provider, args.model, build_prompt(chunk), key)
        for r, s in zip(chunk, parse(text, len(chunk))):
            # Each line records the judge model and the date, because a hosted model can change
            # over time; the scores in the report are from gemini-3.5-flash-lite on 2026-11-21.
            done[r["id"]] = {"id": r["id"], **s, "judge_model": args.model, "judged_at": date.today().isoformat()}
            if not complete(done[r["id"]]):
                # Keep the first 500 characters of a failed reply, to see why parsing failed.
                done[r["id"]]["judge_raw"] = (text or "")[:500]
        # Save after every batch, so a stop loses at most one batch.
        write(out, rows, done)
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
        # Pause between requests (3 s) to stay under the free tier's per-minute limit.
        time.sleep(args.pause)
    failed = sum(not complete(done.get(r["id"])) for r in rows)
    print(f"{name}: {len(rows) - failed} scored, {failed} missing -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="run names (results/<run>.jsonl)")
    ap.add_argument("--all", action="store_true", help="every file in results/")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    ap.add_argument("--model", help="default depends on the provider")
    ap.add_argument("--batch", type=int, default=10, help="answers per request")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N unscored answers per run (testing)")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between requests")
    args = ap.parse_args()
    # Pick the key's environment variable and the default model for the chosen provider.
    _, env_var, default_model = PROVIDERS[args.provider]
    args.model = args.model or default_model
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(f"Set {env_var} in the environment.")
    # With --all: every results/<run>.jsonl, in name order. In the project this judged all 9
    # systems, 1,350 answers in about 135 requests.
    runs = args.runs or ([p.stem for p in sorted((ROOT / "results").glob("*.jsonl"))] if args.all else [])
    if not runs:
        raise SystemExit("Give run names or --all.")
    for name in runs:
        judge_run(name, args, key)


if __name__ == "__main__":
    main()
