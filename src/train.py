"""
Week 3: QLoRA fine-tuning of Qwen2.5-3B on Spider NL-to-SQL.

Uses:
  - bitsandbytes for 4-bit quantisation (QLoRA)
  - peft for LoRA adapter injection
  - trl SFTTrainer + SFTConfig for supervised fine-tuning

Run: python src/06_train.py
"""

import json
import yaml
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---------- load config ----------
CONFIG_PATH = Path("config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


def load_data(path: str) -> Dataset:
    with open(path) as f:
        records = json.load(f)
    # combine prompt + completion into single 'text' field
    for r in records:
        r["text"] = r["prompt"] + r["completion"]
    return Dataset.from_list(records)


def main():
    print("Week 3 — QLoRA Fine-tuning")
    print("=" * 45)
    print(f"Model     : {cfg['model_name']}")
    print(f"LoRA rank : {cfg['lora_rank']}")
    print(f"Epochs    : {cfg['num_epochs']}")
    print(f"Device    : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset = load_data(cfg["train_file"])
    val_dataset   = load_data(cfg["val_file"])
    print(f"  Train : {len(train_dataset)} examples")
    print(f"  Val   : {len(val_dataset)} examples")

    # ── Quantisation config (QLoRA) ───────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=True,
    )

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer from {cfg['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model in 4-bit ───────────────────────────────────────────────────
    print("Loading model in 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["gradient_checkpointing"],
    )

    # ── Inject LoRA adapters ──────────────────────────────────────────────────
    print(f"Injecting LoRA adapters (rank={cfg['lora_rank']})...")
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── SFTConfig (replaces TrainingArguments for trl >= 0.16) ───────────────
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=50,
        weight_decay=cfg["weight_decay"],
        bf16=True,
        fp16=False,
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg["seed"],
        report_to="none",
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=cfg["max_seq_length"],   # renamed from max_seq_length
        packing=False,
        dataset_text_field="text",          # column name in our dataset
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    print("\nInitialising SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    final_path = Path(cfg["output_dir"]) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    size_mb = sum(
        p.stat().st_size for p in final_path.rglob("*") if p.is_file()
    ) / 1024**2

    print(f"\nTraining complete!")
    print(f"Final adapter saved to : {final_path}")
    print(f"Adapter size           : {size_mb:.1f} MB")
    print("\nNext: python src/07_evaluate.py")


if __name__ == "__main__":
    main()