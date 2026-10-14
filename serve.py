"""FastAPI service: natural-language question -> validated, read-only SQL result.

Run: DB_PATH=my.sqlite ADAPTER=outputs/adapter uvicorn serve:app --port 8000
"""
import os
import sqlite3

import sqlglot
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlglot import exp
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "outputs/adapter")
DB_PATH = os.environ["DB_PATH"]

tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
from peft import PeftModel  # noqa: E402

model = PeftModel.from_pretrained(model, ADAPTER).eval()
app = FastAPI(title="Text-to-SQL")


def get_schema():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    conn.close()
    return "\n".join(r[0] + ";" for r in rows)


def validate(sql):
    """Allow a single SELECT statement only."""
    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"unparseable SQL: {e}")
    if len(parsed) != 1 or not isinstance(parsed[0], (exp.Select, exp.Union)):
        raise ValueError("only a single SELECT statement is allowed")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Command)
    if any(parsed[0].find(t) for t in forbidden):
        raise ValueError("write operations are not allowed")


def generate_sql(question, error_hint=None):
    prompt = f"### Schema:\n{get_schema()}\n\n### Question:\n{question}\n\n"
    if error_hint:
        prompt += f"### Previous attempt failed with: {error_hint}\n\n"
    prompt += "### SQL:\n"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=160, do_sample=False)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return " ".join(text.split("###")[0].replace("```sql", "").replace("```", "").split())


def run_readonly(sql):
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    try:
        cur = conn.execute(sql)
        return [d[0] for d in cur.description], cur.fetchmany(100)
    finally:
        conn.close()


class Ask(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Ask):
    sql, error = generate_sql(body.question), None
    for attempt in range(2):  # one retry, feeding the error back to the model
        try:
            validate(sql)
            cols, rows = run_readonly(sql)
            return {"sql": sql, "columns": cols, "rows": rows, "retried": attempt == 1}
        except Exception as e:  # noqa: BLE001
            error = str(e)
            if attempt == 0:
                sql = generate_sql(body.question, error_hint=error)
    raise HTTPException(status_code=422, detail={"sql": sql, "error": error})
