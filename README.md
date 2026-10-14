# QLoRA Fine-tuning of Qwen2.5-3B for Natural Language to SQL

Fine-tunes **Qwen2.5-3B** to translate natural-language questions into SQL queries, using **QLoRA** on the **Spider** benchmark. The model trains on a single consumer GPU (NVIDIA RTX 3050).

## Results

Evaluated on the same 100 held-out Spider questions:

| Model | Accuracy | Avg latency |
|---|---|---|
| Qwen2.5-3B-Instruct (zero-shot) | 26% | ~14.5 s |
| **Qwen2.5-3B + QLoRA (ours)** | **56%** | **~1.6 s** |

Fine-tuning more than doubles accuracy. It also makes inference about 9× faster, because the model emits clean SQL instead of long markdown-wrapped answers.

## Approach

- **Data:** Spider, with 6,494 examples across 134 databases after deduplication. Split 5,196 train / 649 validation / 649 test.
- **Prompt:** each database schema is rendered as `CREATE TABLE` statements and placed in a fixed template:
  ```
  ### Task
  Convert the following question into a SQL query using the provided database schema.
  ### Schema
  {schema}
  ### Question
  {question}
  ### SQL
  ```
- **Training:** 4-bit NF4 quantization with double quantization, and LoRA adapters (r=32, α=64, dropout 0.05) on all attention and MLP projections. Runs 5 epochs at learning rate 2e-4 with a cosine schedule, an effective batch size of 16, bf16, and gradient checkpointing. All settings are in [config.yaml](config.yaml).
- **Evaluation:** the predicted and gold SQL both run on an in-memory SQLite database built from the schema, and their results are compared. If execution can't decide a case, the comparison falls back to normalized SQL and keyword matching.

## Project structure

```
src/
  download_spider.py   # download the Spider dataset
  build_schema.py      # tables.json -> CREATE TABLE schema strings
  format_prompt.py     # build prompts, dedup, train/val/test split, stats
  verify_data.py       # sanity-check the processed data
  baseline.py          # zero-/few-shot baseline evaluation
  train.py             # QLoRA supervised fine-tuning (TRL SFTTrainer)
  evaluate.py          # evaluate the fine-tuned adapter, write comparison table
  utils.py             # SQL execution, extraction, metrics
config.yaml            # training hyperparameters
eval/                  # baseline + fine-tuned results, comparison_table.csv
```

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirement.txt

python src/download_spider.py
python src/build_schema.py
python src/format_prompt.py
python src/verify_data.py
python src/baseline.py
python src/train.py        # saves the LoRA adapter to checkpoints/final
python src/evaluate.py
```

**Stack:** PyTorch, Hugging Face Transformers, PEFT, TRL, bitsandbytes, Datasets.

## Limitations and next steps

- The schema-only SQLite databases contain no rows, so the accuracy above approximates true execution accuracy. Next step: evaluate on Spider's official databases with its evaluation script.
- The baseline used the Instruct model. Evaluate the base model and fix the few-shot baseline to get a like-for-like comparison.
- Evaluate on the full 649-example test set.
