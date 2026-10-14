"""
Week 2: Baseline evaluation of Qwen2.5-3B (no fine-tuning).

Measures:
  - Zero-shot execution accuracy (no examples in prompt)
  - Few-shot execution accuracy (3 examples in prompt)

Tests on 100 examples from the spider dev set.
Saves results to eval/baseline_results.json.

Run: python src/05_baseline.py
"""

import json
import time
import sys
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_sql, execution_accuracy, compute_metrics, load_schema_lookup

# ---------- CONFIG ----------
DATA_PROCESSED  = Path("data/processed")
DATA_RAW        = Path("data/raw")
EVAL_DIR        = Path("eval")
OUTPUT_PATH     = EVAL_DIR / "baseline_results.json"

MODEL_NAME      = "Qwen/Qwen2.5-3B-Instruct"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS  = 150
N_EVAL          = 100
N_FEW_SHOT      = 3
SEED            = 42
# ----------------------------

EVAL_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)

PROMPT_TEMPLATE = """### Task
Convert the following question into a SQL query using the provided database schema.

### Schema
{schema}

### Question
{question}

### SQL
"""

FEW_SHOT_EXAMPLE = """### Task
Convert the following question into a SQL query using the provided database schema.

### Schema
{schema}

### Question
{question}

### SQL
{sql}
"""


def load_model(model_name: str):
    print(f"Loading {model_name} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded. {n_params:.1f}B parameters.")
    return model, tokenizer


def generate(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)
    # stop tokens: halt generation at these strings
    stop_strings = ["###", "Human:", "```\n\n", "\n\n\n"]
    stop_ids = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.append(ids[0])

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id] + stop_ids,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_eval(examples, schema_lookup, model, tokenizer, mode, few_shot_exs=None):
    results = []
    for i, ex in enumerate(examples):
        question = ex["question"]
        gold_sql = ex["query"]
        db_id    = ex["db_id"]
        schema   = schema_lookup.get(db_id, "")

        if mode == "zero_shot":
            prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
        else:
            # build few-shot block
            few_block = ""
            for fs in few_shot_exs:
                fs_schema = schema_lookup.get(fs["db_id"], "")
                few_block += FEW_SHOT_EXAMPLE.format(
                    schema=fs_schema,
                    question=fs["question"],
                    sql=fs["query"],
                ) + "\n"
            prompt = few_block + PROMPT_TEMPLATE.format(
                schema=schema, question=question
            )

        t0 = time.time()
        raw = generate(prompt, model, tokenizer)
        elapsed = time.time() - t0

        pred_sql = extract_sql(raw)
        correct  = execution_accuracy(pred_sql, gold_sql, schema)

        results.append({
            "id":            i,
            "question":      question,
            "db_id":         db_id,
            "gold_sql":      gold_sql,
            "predicted_sql": pred_sql,
            "correct":       correct,
            "latency_s":     round(elapsed, 2),
        })

        status = "✓" if correct else "✗"
        print(f"  [{i+1:>3}/{len(examples)}] {status}  "
              f"{question[:55]:<55}  ({elapsed:.1f}s)")

    return results


def main():
    print("Week 2 — Baseline Evaluation")
    print("=" * 45)

    # load data
    print("\nLoading data...")
    schema_lookup = load_schema_lookup(DATA_RAW / "schema_lookup.json")

    with open(DATA_RAW / "dev.json") as f:
        dev_data = json.load(f)
    with open(DATA_RAW / "train.json") as f:
        train_data = json.load(f)

    # filter to examples that have schemas
    eligible = [ex for ex in dev_data if ex["db_id"] in schema_lookup]
    test_examples = random.sample(eligible, min(N_EVAL, len(eligible)))
    few_shot_exs  = random.sample(
        [ex for ex in train_data if ex["db_id"] in schema_lookup],
        N_FEW_SHOT
    )
    print(f"  Test set  : {len(test_examples)} examples")
    print(f"  Few-shot  : {N_FEW_SHOT} examples")

    # load model
    model, tokenizer = load_model(MODEL_NAME)

    # zero-shot
    print(f"\n{'='*45}")
    print(f"ZERO-SHOT ({len(test_examples)} examples)")
    print(f"{'='*45}")
    zero_results = run_eval(
        test_examples, schema_lookup, model, tokenizer, "zero_shot"
    )
    zero_metrics = compute_metrics(zero_results)
    print(f"\n  Zero-shot accuracy: {zero_metrics['execution_accuracy']:.4f} "
          f"({zero_metrics['correct']}/{zero_metrics['total']})")

    # few-shot
    print(f"\n{'='*45}")
    print(f"FEW-SHOT ({N_FEW_SHOT} examples in prompt)")
    print(f"{'='*45}")
    few_results = run_eval(
        test_examples, schema_lookup, model, tokenizer, "few_shot",
        few_shot_exs=few_shot_exs
    )
    few_metrics = compute_metrics(few_results)
    print(f"\n  Few-shot accuracy : {few_metrics['execution_accuracy']:.4f} "
          f"({few_metrics['correct']}/{few_metrics['total']})")

    # save
    output = {
        "model":      MODEL_NAME,
        "n_eval":     N_EVAL,
        "n_few_shot": N_FEW_SHOT,
        "zero_shot":  {"metrics": zero_metrics,  "results": zero_results},
        "few_shot":   {"metrics": few_metrics,   "results": few_results},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # final table
    print(f"\n{'='*45}")
    print("BASELINE RESULTS TABLE")
    print(f"{'='*45}")
    print(f"  {'Strategy':<20} {'Exec Acc':>10} {'Correct':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Zero-shot':<20} "
          f"{zero_metrics['execution_accuracy']:>10.4f} "
          f"{zero_metrics['correct']:>5}/{zero_metrics['total']}")
    print(f"  {'Few-shot (3-ex)':<20} "
          f"{few_metrics['execution_accuracy']:>10.4f} "
          f"{few_metrics['correct']:>5}/{few_metrics['total']}")
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print("\nWeek 2 complete — these are your 'before' numbers.")
    print("Week 3: QLoRA fine-tuning → should significantly improve these.")


if __name__ == "__main__":
    main()