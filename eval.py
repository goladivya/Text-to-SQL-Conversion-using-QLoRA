
"""Execution-accuracy evaluation on Spider dev, with an optional error-feedback retry.

Usage:
  python eval.py /path/to/spider baseline                       # base model, no adapter
  python eval.py /path/to/spider finetuned outputs/adapter      # with a LoRA adapter

Env vars:
  N_EVAL=200   number of dev questions (0 = all 1034)
  SHUFFLE=0    1 = shuffle dev with a fixed seed (42) before taking N_EVAL.
               Use the SAME setting for baseline and fine-tuned runs, or the comparison is unfair.
  RETRY=1      1 = if the first SQL fails to execute, send the error back to the model once
  MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct

Metric: a prediction is correct if it returns the same rows as the gold SQL on the real
SQLite database (row order ignored). This is a simplified version of Spider's official
test-suite evaluation, so call it "execution accuracy (simplified)".

Reports two numbers: first-attempt accuracy, and accuracy after the retry.
"""
import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SPIDER = Path(sys.argv[1])
TAG = sys.argv[2]
ADAPTER = sys.argv[3] if len(sys.argv) > 3 else None
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
N_EVAL = int(os.environ.get("N_EVAL", 200))
SHUFFLE = int(os.environ.get("SHUFFLE", 0))
RETRY = int(os.environ.get("RETRY", 1))
BATCH = 8

rows = [json.loads(line) for line in open("data/dev.jsonl")]
if SHUFFLE:
    random.Random(42).shuffle(rows)
if N_EVAL > 0:
    rows = rows[:N_EVAL]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0})
if ADAPTER:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def clean(text):
    text = text.split("###")[0].strip()
    text = text.replace("```sql", "").replace("```", "").strip()
    return " ".join(text.split())


def generate(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [clean(t) for t in texts]


def retry_prompt(prompt, err):
    """Same prompt, with the failed attempt's error inserted before the final '### SQL:' marker."""
    base = prompt[: -len("### SQL:\n")]
    return base + f"### Previous attempt failed with: {err}\n\n### SQL:\n"


def execute(db_id, sql, timeout=10):
    path = SPIDER / "database" / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    deadline = time.time() + timeout
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        return Counter(map(repr, conn.execute(sql).fetchall())), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    finally:
        conn.close()


def is_correct(item):
    return item["gold"] is not None and item["got"] is not None and item["gold"] == item["got"]


first_ok = final_ok = n_retried = n_fixed = 0
results = []
for i in range(0, len(rows), BATCH):
    batch = rows[i : i + BATCH]
    preds = generate([r["prompt"] for r in batch])

    items = []
    for r, pred in zip(batch, preds):
        gold, _ = execute(r["db_id"], r["query"])
        got, err = execute(r["db_id"], pred)
        item = {"r": r, "gold": gold, "pred": pred, "got": got, "err": err,
                "first_pred": pred, "first_err": err, "retried": False}
        item["first_correct"] = is_correct(item)
        items.append(item)

    if RETRY:
        bad = [it for it in items if it["err"]]
        if bad:
            new_preds = generate([retry_prompt(it["r"]["prompt"], it["err"]) for it in bad])
            for it, pred in zip(bad, new_preds):
                got, err = execute(it["r"]["db_id"], pred)
                it.update(pred=pred, got=got, err=err, retried=True)

    for it in items:
        correct = is_correct(it)
        first_ok += it["first_correct"]
        final_ok += correct
        n_retried += it["retried"]
        n_fixed += it["retried"] and correct
        results.append(
            {"db_id": it["r"]["db_id"], "question": it["r"]["question"], "gold": it["r"]["query"],
             "first_pred": it["first_pred"], "first_error": it["first_err"],
             "first_correct": it["first_correct"],
             "pred": it["pred"], "error": it["err"], "correct": correct, "retried": it["retried"]}
        )
    done = len(results)
    print(f"{done}/{len(rows)}  first-attempt acc: {first_ok / done:.3f}  with retry: {final_ok / done:.3f}")

n = len(results)
print(f"\n[{TAG}] n={n} shuffle={SHUFFLE}")
print(f"  first-attempt execution accuracy: {first_ok / n:.3f} ({first_ok}/{n})")
if RETRY:
    print(f"  with error-feedback retry:        {final_ok / n:.3f} ({final_ok}/{n})  "
          f"[retried {n_retried}, fixed {n_fixed}]")
Path("outputs").mkdir(exist_ok=True)
json.dump(
    {"tag": TAG, "n": n, "shuffle": SHUFFLE, "retry": RETRY,
     "accuracy_first_attempt": first_ok / n, "accuracy_with_retry": final_ok / n,
     "retried": n_retried, "fixed_by_retry": n_fixed, "results": results},
    open(f"outputs/results_{TAG}.json", "w"), indent=2)
