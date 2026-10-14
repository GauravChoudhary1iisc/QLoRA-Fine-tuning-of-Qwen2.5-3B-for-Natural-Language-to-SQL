"""
Week 4: Evaluate the QLoRA fine-tuned model on the Spider test set.

Loads the base model + LoRA adapter from checkpoints/final/,
runs the same 100 test questions used in baseline,
and prints the final comparison table.

Run: python src/07_evaluate.py
"""
import importlib
import sys

# force reload utils to pick up latest version
if 'utils' in sys.modules:
    del sys.modules['utils']

import json
import time
import sys
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_sql, execution_accuracy_v2 as execution_accuracy, compute_metrics, load_schema_lookup

# ---------- CONFIG ----------
DATA_RAW        = Path("data/raw")
EVAL_DIR        = Path("eval")
ADAPTER_PATH    = Path("checkpoints/final")
BASELINE_PATH   = EVAL_DIR / "baseline_results.json"
OUTPUT_PATH     = EVAL_DIR / "finetuned_results.json"
TABLE_PATH      = EVAL_DIR / "comparison_table.csv"

BASE_MODEL = "Qwen/Qwen2.5-3B"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS  = 150
N_EVAL          = 100
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


def load_finetuned_model(base_model: str, adapter_path: Path):
    """Load base model in 4-bit + merge LoRA adapter."""
    print(f"Loading base model: {base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    print("Fine-tuned model ready.")
    return model, tokenizer


def generate(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_eval(examples, schema_lookup, model, tokenizer) -> list[dict]:
    results = []
    for i, ex in enumerate(examples):
        question = ex["question"]
        gold_sql = ex["query"]
        db_id    = ex["db_id"]
        schema   = schema_lookup.get(db_id, "")

        prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)

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


def print_comparison(baseline_metrics: dict, ft_metrics: dict):
    bz = baseline_metrics["execution_accuracy"]
    ft = ft_metrics["execution_accuracy"]
    improvement = (ft - bz) / bz * 100 if bz > 0 else 0

    print(f"\n{'='*55}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*55}")
    print(f"  {'Strategy':<25} {'Exec Acc':>10} {'Correct':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Zero-shot (base)':<25} "
          f"{bz:>10.4f} "
          f"{baseline_metrics['correct']:>5}/{baseline_metrics['total']}")
    print(f"  {'QLoRA fine-tuned':<25} "
          f"{ft:>10.4f} "
          f"{ft_metrics['correct']:>5}/{ft_metrics['total']}")
    print(f"  {'-'*47}")
    print(f"  Improvement: +{ft-bz:.4f} ({improvement:+.1f}%)")
    print(f"{'='*55}")


def save_csv(baseline_metrics: dict, ft_metrics: dict):
    import csv
    rows = [
        {"strategy": "zero_shot_base",
         "execution_accuracy": baseline_metrics["execution_accuracy"],
         "correct": baseline_metrics["correct"],
         "total": baseline_metrics["total"]},
        {"strategy": "qlora_finetuned",
         "execution_accuracy": ft_metrics["execution_accuracy"],
         "correct": ft_metrics["correct"],
         "total": ft_metrics["total"]},
    ]
    with open(TABLE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison table saved to: {TABLE_PATH}")


def main():
    print("Week 4 — Fine-tuned Model Evaluation")
    print("=" * 45)

    # load data
    print("\nLoading data...")
    schema_lookup = load_schema_lookup(DATA_RAW / "schema_lookup.json")
    with open(DATA_RAW / "dev.json") as f:
        dev_data = json.load(f)

    eligible = [ex for ex in dev_data if ex["db_id"] in schema_lookup]
    test_examples = random.sample(eligible, min(N_EVAL, len(eligible)))
    print(f"  Test examples : {len(test_examples)}")

    # load fine-tuned model
    model, tokenizer = load_finetuned_model(BASE_MODEL, ADAPTER_PATH)

    # run evaluation
    print(f"\n{'='*45}")
    print(f"Fine-tuned model evaluation ({len(test_examples)} examples)")
    print(f"{'='*45}")
    results = run_eval(test_examples, schema_lookup, model, tokenizer)
    ft_metrics = compute_metrics(results)

    print(f"\n  Fine-tuned accuracy: {ft_metrics['execution_accuracy']:.4f} "
          f"({ft_metrics['correct']}/{ft_metrics['total']})")

    # save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "model":   f"{BASE_MODEL} + LoRA ({ADAPTER_PATH})",
            "n_eval":  N_EVAL,
            "metrics": ft_metrics,
            "results": results,
        }, f, indent=2)

    # load baseline and compare
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        baseline_metrics = baseline["zero_shot"]["metrics"]
        print_comparison(baseline_metrics, ft_metrics)
        save_csv(baseline_metrics, ft_metrics)
    else:
        print("\nBaseline file not found — run 05_baseline.py first.")
        print(f"Fine-tuned accuracy: {ft_metrics['execution_accuracy']:.4f}")

    print("\nWeek 4 complete. Project done!")


if __name__ == "__main__":
    main()