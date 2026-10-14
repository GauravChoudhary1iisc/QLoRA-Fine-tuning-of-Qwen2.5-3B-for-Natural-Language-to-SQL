"""
Week 1, Step 3: Format Spider examples into prompt-completion pairs.

Converts each raw Spider example into:
  prompt    = instruction + schema + question
  completion = SQL query

Then filters (length, duplicates) and splits into train/val/test.

Saves:
  data/processed/train_formatted.json
  data/processed/val_formatted.json
  data/processed/test_formatted.json
  data/processed/stats.json

Run: python src/03_format_prompts.py
"""

import json
import random
from pathlib import Path
from collections import Counter

# ---------- CONFIG ----------
DATA_RAW        = Path("data/raw")
DATA_PROCESSED  = Path("data/processed")
SCHEMA_LOOKUP   = DATA_RAW / "schema_lookup.json"

MAX_PROMPT_CHARS   = 2048   # skip examples with very long prompts
MIN_SQL_CHARS      = 5      # skip trivially short SQL
VAL_SPLIT          = 0.1    # 10% of train -> validation
TEST_SPLIT         = 0.1    # 10% of train -> test
SEED               = 42
# ----------------------------

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
random.seed(SEED)

# ── Prompt template ───────────────────────────────────────────────────────────
# This EXACT format must be used at inference time too.
# The model learns the pattern — any deviation degrades output.

PROMPT_TEMPLATE = """### Task
Convert the following question into a SQL query using the provided database schema.

### Schema
{schema}

### Question
{question}

### SQL
"""


def format_example(question: str, schema: str, sql: str) -> dict:
    """Build one prompt-completion record."""
    prompt = PROMPT_TEMPLATE.format(
        schema=schema,
        question=question,
    )
    return {
        "prompt":     prompt,
        "completion": sql.strip(),
        "full_text":  prompt + sql.strip(),   # for tokenisation length check
    }


def load_and_format(
    data_path: Path,
    schema_lookup: dict,
    split_name: str,
) -> list[dict]:
    """Load a Spider JSON file and format all examples."""
    with open(data_path) as f:
        raw = json.load(f)

    formatted   = []
    skipped_schema  = 0
    skipped_length  = 0
    skipped_sql     = 0

    for ex in raw:
        db_id    = ex["db_id"]
        question = ex["question"].strip()
        sql      = ex["query"].strip()

        # skip if schema missing
        if db_id not in schema_lookup:
            skipped_schema += 1
            continue

        # skip trivially short SQL
        if len(sql) < MIN_SQL_CHARS:
            skipped_sql += 1
            continue

        schema = schema_lookup[db_id]
        record = format_example(question, schema, sql)
        record["db_id"]    = db_id
        record["question"] = question
        record["sql"]      = sql

        # skip if prompt too long
        if len(record["full_text"]) > MAX_PROMPT_CHARS:
            skipped_length += 1
            continue

        formatted.append(record)

    print(f"  {split_name}: {len(raw)} raw → {len(formatted)} kept "
          f"(skipped: {skipped_schema} schema, "
          f"{skipped_length} length, {skipped_sql} short SQL)")
    return formatted


def dedup(records: list[dict]) -> list[dict]:
    """Remove exact duplicate (question, sql) pairs."""
    seen = set()
    deduped = []
    for r in records:
        key = (r["question"].lower(), r["sql"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def split_train_val_test(
    records: list[dict],
    val_frac: float,
    test_frac: float,
) -> tuple[list, list, list]:
    """Shuffle and split into train / val / test."""
    random.shuffle(records)
    n       = len(records)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    test    = records[:n_test]
    val     = records[n_test:n_test + n_val]
    train   = records[n_test + n_val:]
    return train, val, test


def compute_stats(train, val, test, all_records) -> dict:
    """Compute dataset statistics for the stats.json summary."""
    def avg_len(records, key):
        if not records:
            return 0
        return int(sum(len(r[key]) for r in records) / len(records))

    # SQL keyword distribution
    keywords = ["JOIN", "GROUP BY", "ORDER BY", "HAVING",
                "INTERSECT", "UNION", "EXCEPT", "WHERE"]
    kw_counts = Counter()
    for r in all_records:
        sql_upper = r["sql"].upper()
        for kw in keywords:
            if kw in sql_upper:
                kw_counts[kw] += 1

    return {
        "total":             len(all_records),
        "train":             len(train),
        "val":               len(val),
        "test":              len(test),
        "avg_prompt_chars":  avg_len(all_records, "prompt"),
        "avg_sql_chars":     avg_len(all_records, "sql"),
        "sql_keywords":      dict(kw_counts),
        "unique_dbs":        len(set(r["db_id"] for r in all_records)),
    }


def save(records: list[dict], path: Path):
    """Save records to JSON, stripping internal fields not needed for training."""
    to_save = [{"prompt": r["prompt"], "completion": r["completion"]} for r in records]
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"  Saved {len(to_save)} examples → {path}  "
          f"({path.stat().st_size/1024:.0f} KB)")


def main():
    print("Loading schema lookup...")
    with open(SCHEMA_LOOKUP) as f:
        schema_lookup = json.load(f)
    print(f"  {len(schema_lookup)} database schemas loaded\n")

    # ── Format both splits ────────────────────────────────────────────────────
    print("Formatting examples...")
    train_raw = load_and_format(DATA_RAW / "train.json", schema_lookup, "train")
    dev_raw   = load_and_format(DATA_RAW / "dev.json",   schema_lookup, "dev")

    # ── Deduplicate train ─────────────────────────────────────────────────────
    before = len(train_raw)
    train_raw = dedup(train_raw)
    print(f"\n  Dedup: {before} → {len(train_raw)} train examples "
          f"({before - len(train_raw)} duplicates removed)")

    # ── Split train into train / val / test ───────────────────────────────────
    # Dev set from Spider is used as our final held-out test set.
    # We carve val + test from the training data.
    print("\nSplitting train into train/val/test...")
    train, val, test = split_train_val_test(train_raw, VAL_SPLIT, TEST_SPLIT)
    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")
    print(f"  Test  : {len(test)}")
    print(f"  Spider dev (held out for final eval): {len(dev_raw)}")

    # ── Save splits ───────────────────────────────────────────────────────────
    print("\nSaving processed files...")
    save(train, DATA_PROCESSED / "train_formatted.json")
    save(val,   DATA_PROCESSED / "val_formatted.json")
    save(test,  DATA_PROCESSED / "test_formatted.json")

    # also save dev set (Spider's original dev, used for final eval in Week 4)
    save(dev_raw, DATA_PROCESSED / "spider_dev_formatted.json")

    # ── Stats ─────────────────────────────────────────────────────────────────
    all_records = train + val + test
    stats = compute_stats(train, val, test, all_records)
    stats_path = DATA_PROCESSED / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Print a sample prompt ─────────────────────────────────────────────────
    print("\n── Sample prompt ────────────────────────────────────────")
    sample = train[0]
    print(sample["prompt"])
    print(f"### COMPLETION: {sample['completion']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  Train examples     : {stats['train']}")
    print(f"  Val examples       : {stats['val']}")
    print(f"  Test examples      : {stats['test']}")
    print(f"  Avg prompt chars   : {stats['avg_prompt_chars']}")
    print(f"  Avg SQL chars      : {stats['avg_sql_chars']}")
    print(f"  Unique databases   : {stats['unique_dbs']}")
    print(f"\n  SQL complexity:")
    for kw, cnt in sorted(stats['sql_keywords'].items(),
                          key=lambda x: -x[1]):
        pct = cnt / stats['total'] * 100
        print(f"    {kw:<15}: {cnt:>4}  ({pct:.1f}%)")
    print(f"\nStep 3 complete. Files saved to {DATA_PROCESSED}/")
    print("Run next: python src/04_verify_data.py")


if __name__ == "__main__":
    main()