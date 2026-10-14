"""Build prompt files from the Spider dataset.

Usage: python prepare_data.py /path/to/spider
Expects: train_spider.json, dev.json, tables.json, database/ inside that folder.
Writes: data/train.jsonl, data/dev.jsonl
"""
import json
import sys
from pathlib import Path

SPIDER = Path(sys.argv[1] if len(sys.argv) > 1 else "spider")
OUT = Path("data")
OUT.mkdir(exist_ok=True)


def load_schemas():
    schemas = {}
    for db in json.load(open(SPIDER / "tables.json")):
        cols = db["column_names_original"]
        types = db["column_types"]
        tables = db["table_names_original"]
        per_table = {i: [] for i in range(len(tables))}
        for idx, (t, c) in enumerate(cols):
            if t >= 0:
                per_table[t].append(f"{c} {types[idx]}")
        lines = [f"CREATE TABLE {tables[i]} ({', '.join(per_table[i])});" for i in per_table]
        for a, b in db["foreign_keys"]:
            lines.append(
                f"-- FK: {tables[cols[a][0]]}.{cols[a][1]} -> {tables[cols[b][0]]}.{cols[b][1]}"
            )
        schemas[db["db_id"]] = "\n".join(lines)
    return schemas


def build_prompt(schema, question):
    return f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n"


def convert(src, dst, schemas):
    data = json.load(open(SPIDER / src))
    with open(OUT / dst, "w") as f:
        for ex in data:
            row = {
                "db_id": ex["db_id"],
                "question": ex["question"],
                "query": ex["query"],
                "prompt": build_prompt(schemas[ex["db_id"]], ex["question"]),
            }
            f.write(json.dumps(row) + "\n")
    print(f"{dst}: {len(data)} examples")


if __name__ == "__main__":
    schemas = load_schemas()
    convert("train_spider.json", "train.jsonl", schemas)
    convert("dev.json", "dev.jsonl", schemas)
