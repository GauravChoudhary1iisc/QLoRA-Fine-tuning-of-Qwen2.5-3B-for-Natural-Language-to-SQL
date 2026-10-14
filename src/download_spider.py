"""
Week 1, Step 1: Download the Spider NL-to-SQL dataset.

Uses HuggingFace datasets library to download xlangai/spider directly.
Saves train.json, dev.json, and tables.json into data/raw/.

Run: python src/01_download_spider.py
"""

import json
from pathlib import Path

# ---------- CONFIG ----------
DATA_RAW = Path("data/raw")
# ----------------------------

DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_spider():
    from datasets import load_dataset

    print("Downloading Spider dataset from HuggingFace (xlangai/spider)...")
    print("This may take a minute on first run.\n")

    # load train and validation splits
    dataset = load_dataset("xlangai/spider", trust_remote_code=True)

    print(f"Splits available: {list(dataset.keys())}")
    print(f"  Train examples : {len(dataset['train'])}")
    print(f"  Dev examples   : {len(dataset['validation'])}")
    print(f"  Columns        : {dataset['train'].column_names}\n")

    # save train split
    train_path = DATA_RAW / "train.json"
    train_data = [dict(row) for row in dataset["train"]]
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved train.json  ({len(train_data)} examples, {train_path.stat().st_size/1024:.0f} KB)")

    # save dev split
    dev_path = DATA_RAW / "dev.json"
    dev_data = [dict(row) for row in dataset["validation"]]
    with open(dev_path, "w") as f:
        json.dump(dev_data, f, indent=2)
    print(f"Saved dev.json    ({len(dev_data)} examples, {dev_path.stat().st_size/1024:.0f} KB)")

    return train_data, dev_data


def inspect_sample(train_data):
    """Print a few samples so we can see the data structure."""
    print("\n── Sample entry ─────────────────────────────────────────")
    sample = train_data[0]
    for k, v in sample.items():
        val = str(v)[:120] + "..." if len(str(v)) > 120 else v
        print(f"  {k:<20}: {val}")

    print("\n── SQL complexity distribution ──────────────────────────")
    from collections import Counter
    keywords = ["JOIN", "GROUP BY", "ORDER BY", "HAVING", "INTERSECT", "UNION", "EXCEPT", "NESTED"]
    counts = Counter()
    for ex in train_data:
        sql = ex.get("query", "").upper()
        for kw in keywords:
            if kw in sql:
                counts[kw] += 1
    for kw, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(train_data) * 100
        print(f"  {kw:<15}: {cnt:>4} ({pct:.1f}%)")

    print("\n── Database IDs (first 10) ──────────────────────────────")
    from collections import Counter
    db_counts = Counter(ex.get("db_id") for ex in train_data)
    for db, cnt in db_counts.most_common(10):
        print(f"  {db:<30}: {cnt} examples")


def main():
    print("Spider NL-to-SQL Dataset Downloader")
    print("=" * 45)

    # check if already downloaded
    if (DATA_RAW / "train.json").exists() and (DATA_RAW / "dev.json").exists():
        print("Files already exist. Loading from disk...")
        with open(DATA_RAW / "train.json") as f:
            train_data = json.load(f)
        with open(DATA_RAW / "dev.json") as f:
            dev_data = json.load(f)
        print(f"  train.json: {len(train_data)} examples")
        print(f"  dev.json  : {len(dev_data)} examples")
    else:
        train_data, dev_data = download_spider()

    inspect_sample(train_data)

    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  Train examples : {len(train_data)}")
    print(f"  Dev examples   : {len(dev_data)}")
    print(f"  Files saved to : {DATA_RAW.resolve()}")
    print("\nStep 1 complete. Run Step 2 next: python src/02_build_schema.py")


if __name__ == "__main__":
    main()