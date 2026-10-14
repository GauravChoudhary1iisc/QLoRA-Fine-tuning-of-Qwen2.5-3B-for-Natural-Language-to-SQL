"""
Week 1, Step 4: Verify the processed dataset before training.

Checks:
  1. All splits load correctly and have expected counts
  2. Every record has prompt + completion fields
  3. Token length distribution (using character proxy)
  4. SQL complexity distribution
  5. Prints 3 random samples for manual inspection

Run: python src/04_verify_data.py
"""

import json
import random
from pathlib import Path
from collections import Counter

# ---------- CONFIG ----------
DATA_PROCESSED  = Path("data/processed")
SEED            = 42

# Rough token estimate: 1 token ≈ 4 chars for English text
CHARS_PER_TOKEN = 4
MAX_TOKENS      = 512   # our training max_seq_length
# ----------------------------

random.seed(SEED)


def load_split(name: str) -> list[dict]:
    path = DATA_PROCESSED / f"{name}_formatted.json"
    with open(path) as f:
        data = json.load(f)
    return data


def check_fields(records: list[dict], split_name: str) -> bool:
    missing = [i for i, r in enumerate(records)
               if "prompt" not in r or "completion" not in r]
    if missing:
        print(f"  ERROR: {len(missing)} records missing fields in {split_name}")
        return False
    print(f"  {split_name:<12}: {len(records):>5} records — all fields present")
    return True


def token_length_stats(records: list[dict]) -> dict:
    """Estimate token lengths using char/4 approximation."""
    lengths = [(len(r["prompt"]) + len(r["completion"])) // CHARS_PER_TOKEN
               for r in records]
    lengths.sort()
    n = len(lengths)
    return {
        "min":    lengths[0],
        "p25":    lengths[n//4],
        "median": lengths[n//2],
        "p75":    lengths[3*n//4],
        "p90":    lengths[int(n*0.9)],
        "p95":    lengths[int(n*0.95)],
        "max":    lengths[-1],
        "over_512": sum(1 for l in lengths if l > MAX_TOKENS),
    }


def sql_complexity(records: list[dict]) -> Counter:
    keywords = ["JOIN", "GROUP BY", "ORDER BY", "HAVING",
                "WHERE", "INTERSECT", "UNION", "EXCEPT"]
    counts = Counter()
    for r in records:
        sql = r["completion"].upper()
        for kw in keywords:
            if kw in sql:
                counts[kw] += 1
    return counts


def main():
    print("Week 1, Step 4 — Data Verification")
    print("=" * 45)

    # ── 1. Load and field-check all splits ───────────────────────────────────
    print("\n── Field checks ─────────────────────────────────────────")
    splits = {}
    all_ok = True
    for name in ["train", "val", "test", "spider_dev"]:
        try:
            data = load_split(name)
            splits[name] = data
            ok = check_fields(data, name)
            all_ok = all_ok and ok
        except FileNotFoundError:
            print(f"  {name:<12}: FILE NOT FOUND")
            all_ok = False

    if not all_ok:
        print("\nSome checks failed. Fix issues before proceeding.")
        return

    # ── 2. Token length distribution ──────────────────────────────────────────
    print("\n── Token length distribution (train) ────────────────────")
    stats = token_length_stats(splits["train"])
    print(f"  Min      : {stats['min']}")
    print(f"  P25      : {stats['p25']}")
    print(f"  Median   : {stats['median']}")
    print(f"  P75      : {stats['p75']}")
    print(f"  P90      : {stats['p90']}")
    print(f"  P95      : {stats['p95']}")
    print(f"  Max      : {stats['max']}")
    print(f"  Over 512 : {stats['over_512']} ({stats['over_512']/len(splits['train'])*100:.1f}%)")

    if stats["p90"] > MAX_TOKENS:
        print(f"\n  WARNING: P90 exceeds {MAX_TOKENS} tokens — consider "
              f"increasing MAX_TOKENS in training config or filtering more aggressively.")
    else:
        print(f"\n  OK: P90 fits within {MAX_TOKENS} token limit.")

    # ── 3. SQL complexity ─────────────────────────────────────────────────────
    print("\n── SQL complexity (train) ───────────────────────────────")
    counts = sql_complexity(splits["train"])
    total  = len(splits["train"])
    for kw, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt * 20 // total)
        print(f"  {kw:<15}: {cnt:>4}  ({cnt/total*100:4.1f}%)  {bar}")

    # ── 4. Prompt structure check ─────────────────────────────────────────────
    print("\n── Prompt structure check ───────────────────────────────")
    required_markers = ["### Task", "### Schema", "### Question", "### SQL"]
    bad = []
    for r in splits["train"]:
        for marker in required_markers:
            if marker not in r["prompt"]:
                bad.append(marker)
                break
    if bad:
        print(f"  WARNING: {len(bad)} records missing prompt markers")
    else:
        print(f"  All {len(splits['train'])} train prompts have correct structure")

    # ── 5. Random sample inspection ───────────────────────────────────────────
    print("\n── Random samples (manual inspection) ───────────────────")
    samples = random.sample(splits["train"], 3)
    for i, s in enumerate(samples):
        print(f"\n  Sample {i+1}:")
        print(f"  Prompt length : {len(s['prompt'])} chars")
        print(f"  SQL           : {s['completion']}")
        # show last 3 lines of prompt (question line)
        prompt_lines = s["prompt"].strip().split("\n")
        print(f"  Question      : {prompt_lines[-3] if len(prompt_lines) >= 3 else prompt_lines[-1]}")

    # ── 6. Final summary ──────────────────────────────────────────────────────
    print("\n── Final summary ────────────────────────────────────────")
    total_examples = sum(len(v) for v in splits.values())
    print(f"  Train          : {len(splits['train'])}")
    print(f"  Val            : {len(splits['val'])}")
    print(f"  Test           : {len(splits['test'])}")
    print(f"  Spider dev     : {len(splits['spider_dev'])}")
    print(f"  Total          : {total_examples}")
    print(f"\n  Median tokens  : {stats['median']}")
    print(f"  Max tokens     : {stats['max']}")

    print("\nWeek 1 complete! Data is ready for training.")
    print("Next: Week 2 — Baseline evaluation (python src/05_baseline.py)")


if __name__ == "__main__":
    main()