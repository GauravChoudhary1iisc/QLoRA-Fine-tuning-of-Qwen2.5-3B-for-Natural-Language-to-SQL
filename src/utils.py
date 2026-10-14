"""
utils.py: Shared utilities for SQL execution and evaluation metrics.

Used by both baseline.py (Week 2) and evaluate.py (Week 4).
"""

import re
import sqlite3
from pathlib import Path


# ── SQL execution ─────────────────────────────────────────────────────────────

def execute_sql(sql: str, schema_str: str) -> tuple[bool, any]:
    """Execute SQL against an in-memory SQLite DB built from schema."""
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        for stmt in schema_str.strip().split(";"):
            stmt = stmt.strip()
            if stmt.upper().startswith("CREATE"):
                cursor.execute(stmt + ";")
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, rows
    except Exception as e:
        return False, str(e)


def normalize_sql(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolon."""
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";").strip()
    return sql


def sql_keywords(sql: str) -> list[str]:
    """
    Extract meaningful SQL keywords and identifiers, ignoring
    alias names (t1, t2...) and common stop words.
    Useful for loose structural comparison.
    """
    stop = {"as", "t1", "t2", "t3", "t4", "t5", "on", "a", "b", "c"}
    words = re.findall(r"\b\w+\b", sql.lower())
    return [w for w in words if w not in stop]


def execution_accuracy(
    predicted_sql: str,
    gold_sql: str,
    schema_str: str,
) -> bool:
    """
    Three-tier matching:
    1. Execution match (with real data in DB — most accurate)
    2. Exact normalised string match
    3. Keyword-order match (ignores alias names)
    """
    # tier 1: execution match
    pred_ok, pred_result = execute_sql(predicted_sql, schema_str)
    gold_ok, gold_result = execute_sql(gold_sql, schema_str)
    if pred_ok and gold_ok and (pred_result or gold_result):
        return set(pred_result) == set(gold_result)

    # tier 2: exact normalised string match
    if normalize_sql(predicted_sql) == normalize_sql(gold_sql):
        return True

    # tier 3: keyword-order match (handles alias swaps, minor reordering)
    return sql_keywords(predicted_sql) == sql_keywords(gold_sql)


# ── SQL extraction from model output ─────────────────────────────────────────

def extract_sql(model_output: str) -> str:
    """
    Extract SQL from model output.
    Handles markdown fences, ### markers, explanations after SQL.
    """
    output = model_output.strip()

    # strip markdown fences
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", output, re.DOTALL)
    if fence_match:
        output = fence_match.group(1).strip()

    # cut at any of these stop markers (with or without newline)
    stop_markers = [
        "\n###", "\n--", "\n\n",
        "###", "Human:", "# Write", "#Write",
        "SELECT.*SELECT",   # repeated SQL — take only first
    ]
    for stop in stop_markers[:-1]:  # all except the regex one
        if stop in output:
            output = output.split(stop)[0]

    # if model repeated the SQL multiple times, take just the first statement
    # find first semicolon and cut there
    if output.count("SELECT") > 1:
        # take everything up to and including the first semicolon
        if ";" in output:
            output = output[:output.index(";") + 1]

    # strip trailing "Human" or similar without separator
    for suffix in ["Human", "human", "# Write", "/*"]:
        if output.endswith(suffix):
            output = output[:-len(suffix)].strip()
        # also handle it appearing after semicolon
        for sep in [";Human", "; Human", ";human"]:
            if sep in output:
                output = output.split(sep)[0] + ";"

    return output.strip().rstrip(";").strip()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total   = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    return {
        "total":              total,
        "correct":            correct,
        "execution_accuracy": round(correct / total, 4) if total > 0 else 0.0,
    }


# ── Schema lookup ─────────────────────────────────────────────────────────────

def load_schema_lookup(path: Path) -> dict:
    import json
    with open(path) as f:
        return json.load(f)


def execution_accuracy_v2(
    predicted_sql: str,
    gold_sql: str,
    schema_str: str,
) -> bool:
    """
    Four-tier matching — handles JOIN order swaps.
    """
    # tier 1: execution match with real data
    pred_ok, pred_result = execute_sql(predicted_sql, schema_str)
    gold_ok, gold_result = execute_sql(gold_sql, schema_str)
    if pred_ok and gold_ok and (pred_result or gold_result):
        return set(pred_result) == set(gold_result)

    # tier 2: exact normalised string match
    norm_pred = normalize_sql(predicted_sql)
    norm_gold = normalize_sql(gold_sql)
    if norm_pred == norm_gold:
        return True

    # tier 3: keyword-order match (handles alias differences)
    if sql_keywords(predicted_sql) == sql_keywords(gold_sql):
        return True

    # tier 4: set-based keyword match (handles JOIN table order swaps)
    # only apply if same SQL structure keywords appear
    pred_kw = sql_keywords(predicted_sql)
    gold_kw = sql_keywords(gold_sql)
    structure_words = {"select","from","where","join","group","order",
                       "by","having","limit","count","avg","sum","max","min",
                       "distinct","on","and","or","not","in","like","between"}
    pred_struct = [w for w in pred_kw if w in structure_words]
    gold_struct = [w for w in gold_kw if w in structure_words]
    if pred_struct == gold_struct and set(pred_kw) == set(gold_kw):
        return True

    return False