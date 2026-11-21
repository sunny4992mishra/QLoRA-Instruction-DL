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

ROOT = Path(__file__).resolve().parent.parent
PROVIDERS = {  # name -> (endpoint, environment variable with the key, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
               "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY",
                   "google/gemma-4-31b-it:free"),
}
CRITERIA = ["correctness", "helpfulness", "fluency"]
MAX_CHARS = 3000  # per field; answers are at most 512 tokens, a few references are longer

PROMPT = """You are a strict, impartial evaluator of answers written in Hindi.

For each item below you get a QUESTION, a REFERENCE answer written by a human, and the ANSWER to evaluate. Score the ANSWER (not the reference) from 1 to 5 on three criteria:

correctness: Are the facts and the content right? Compare with the REFERENCE and with what you know. 1 = wrong or invented, 3 = partly right, 5 = fully right. An answer that repeats itself or never finishes can still be correct in its first part.
helpfulness: Does it do what the QUESTION asks, in the requested form, and is it useful and complete? 1 = ignores the question, 3 = partly answers, 5 = fully answers.
fluency: Is it natural, grammatical Hindi in Devanagari? 1 = broken, mostly not Hindi or garbled, 3 = understandable but awkward or with many errors, 5 = natural. Repeated phrases, looping text and unwanted switches to another language lower the score.

Use the whole range. Do not reward length. Judge each item on its own.

Reply with JSON only, no other text: a list with one object per item, in this form:
[{{"n": 1, "correctness": 3, "helpfulness": 4, "fluency": 5}}, ...]

{items}"""

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


def clip(text):
    text = text.strip()
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " [...]"


def build_prompt(rows):
    items = "\n".join(ITEM.format(n=i + 1, question=clip(r["instruction"]), reference=clip(r["reference"]),
                                  answer=clip(r["output"]) or "(empty)") for i, r in enumerate(rows))
    return PROMPT.format(items=items)


def call(provider, model, prompt, key, retries=6):
    url, _, _ = PROVIDERS[provider]
    if provider == "gemini":
        url = url.format(model=model)
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
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
            if e.code == 429 and ("per-day" in msg or "PerDay" in msg):
                raise SystemExit("Daily free-request limit reached. Run the same command again later.")
            if e.code in (400, 401, 402, 403, 404):
                raise SystemExit(f"HTTP {e.code}: {msg[:300]}")
            wait = 10 * 2 ** attempt
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, IndexError) as e:
            msg, wait = str(e), 10 * 2 ** attempt
        print(f"  request failed ({msg[:120]}), retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    return None


def parse(text, k):
    """Return k dicts of scores (None where missing or invalid) from the judge's reply."""
    scores = [dict.fromkeys(CRITERIA) for _ in range(k)]
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return scores
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return scores
    for obj in arr if isinstance(arr, list) else []:
        n = obj.get("n") if isinstance(obj, dict) else None
        if isinstance(n, int) and 1 <= n <= k:
            vals = {c: obj.get(c) for c in CRITERIA}
            if all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5 for v in vals.values()):
                scores[n - 1] = vals
    return scores


def complete(row):
    return row is not None and all(row.get(c) is not None for c in CRITERIA)


def write(path, rows, done):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(done.get(r["id"]) or {"id": r["id"], **dict.fromkeys(CRITERIA)}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def judge_run(name, args, key):
    rows = load_jsonl(ROOT / "results" / f"{name}.jsonl")
    if not rows:
        print(f"{name}: no results file, skipped")
        return
    out = ROOT / "results" / "judge" / f"{name}.jsonl"
    done = {r["id"]: r for r in load_jsonl(out) if complete(r)}
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{name}: {len(done)} already scored, {len(todo)} to do", flush=True)
    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        text = call(args.provider, args.model, build_prompt(chunk), key)
        for r, s in zip(chunk, parse(text, len(chunk))):
            done[r["id"]] = {"id": r["id"], **s, "judge_model": args.model, "judged_at": date.today().isoformat()}
            if not complete(done[r["id"]]):
                done[r["id"]]["judge_raw"] = (text or "")[:500]
        write(out, rows, done)
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
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
    _, env_var, default_model = PROVIDERS[args.provider]
    args.model = args.model or default_model
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(f"Set {env_var} in the environment.")
    runs = args.runs or ([p.stem for p in sorted((ROOT / "results").glob("*.jsonl"))] if args.all else [])
    if not runs:
        raise SystemExit("Give run names or --all.")
    for name in runs:
        judge_run(name, args, key)


if __name__ == "__main__":
    main()
