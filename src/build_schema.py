"""
Week 1, Step 2: Download tables.json and build a schema lookup.

tables.json contains all 200 Spider database schemas (table names,
column names, column types, foreign keys). This script:
  1. Downloads tables.json from GitHub
  2. Builds a lookup: db_id -> CREATE TABLE string
  3. Saves the lookup to data/raw/schema_lookup.json

The schema string becomes part of every training prompt:
  ### Schema:
  CREATE TABLE singer (Singer_ID INT, Name TEXT, ...);
  CREATE TABLE concert (concert_ID INT, ...);

Run: python src/02_build_schema.py
"""

import json
import requests
from pathlib import Path

# ---------- CONFIG ----------
DATA_RAW        = Path("data/raw")
TABLES_URL = "https://github.com/ygan/Spider-DK/raw/refs/heads/main/tables.json"
TABLES_PATH     = DATA_RAW / "tables.json"
SCHEMA_OUT      = DATA_RAW / "schema_lookup.json"
# ----------------------------

# Spider column type mapping → clean SQL types
TYPE_MAP = {
    "text":    "TEXT",
    "number":  "REAL",
    "time":    "TEXT",
    "boolean": "INTEGER",
    "others":  "TEXT",
}


def download_tables(url: str, dest: Path):
    """Download tables.json from the Spider GitHub repo."""
    if dest.exists():
        print(f"tables.json already exists at {dest}, skipping download.")
        return
    print(f"Downloading tables.json from GitHub...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest, "w") as f:
        f.write(resp.text)
    print(f"  Saved to {dest}  ({dest.stat().st_size/1024:.0f} KB)")


def build_create_table(db: dict) -> str:
    """
    Convert a Spider database entry into a CREATE TABLE string.

    Spider tables.json structure per database:
      db_id          : str
      table_names    : list of table name strings (lowercase)
      column_names   : list of [table_idx, col_name] pairs
                       (table_idx = -1 means it's a wildcard column *)
      column_types   : list of type strings parallel to column_names
      primary_keys   : list of column indices
      foreign_keys   : list of [col_idx, ref_col_idx] pairs
    """
    table_names     = db["table_names_original"]   # preserve original casing
    column_names    = db["column_names_original"]  # [[table_idx, col_name], ...]
    column_types    = db["column_types"]
    primary_keys    = set(db.get("primary_keys", []))

    # group columns by table
    tables: dict[int, list[tuple]] = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(column_names):
        if tbl_idx == -1:
            continue  # skip wildcard *
        sql_type = TYPE_MAP.get(column_types[col_idx], "TEXT")
        is_pk    = col_idx in primary_keys
        tables[tbl_idx].append((col_name, sql_type, is_pk))

    # build CREATE TABLE statements
    statements = []
    for tbl_idx, tbl_name in enumerate(table_names):
        cols = tables.get(tbl_idx, [])
        if not cols:
            continue
        col_defs = []
        for col_name, sql_type, is_pk in cols:
            pk_suffix = " PRIMARY KEY" if is_pk else ""
            col_defs.append(f"{col_name} {sql_type}{pk_suffix}")
        stmt = f"CREATE TABLE {tbl_name} ({', '.join(col_defs)});"
        statements.append(stmt)

    return "\n".join(statements)


def build_schema_lookup(tables_path: Path) -> dict[str, str]:
    """Build db_id -> schema_string lookup for all 200 Spider databases."""
    with open(tables_path) as f:
        tables_data = json.load(f)

    lookup = {}
    for db in tables_data:
        db_id  = db["db_id"]
        schema = build_create_table(db)
        lookup[db_id] = schema

    return lookup


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Step 1: download tables.json
    download_tables(TABLES_URL, TABLES_PATH)

    # Step 2: build schema lookup
    print("\nBuilding schema lookup...")
    lookup = build_schema_lookup(TABLES_PATH)
    print(f"  Built schemas for {len(lookup)} databases")

    # Step 3: save lookup
    with open(SCHEMA_OUT, "w") as f:
        json.dump(lookup, f, indent=2)
    print(f"  Saved to {SCHEMA_OUT}  ({SCHEMA_OUT.stat().st_size/1024:.0f} KB)")

    # Step 4: sanity check — print one example
    print("\n── Sample schema (concert_singer) ───────────────────────")
    if "concert_singer" in lookup:
        print(lookup["concert_singer"])
    else:
        # print first available
        first_key = next(iter(lookup))
        print(f"(concert_singer not found, showing '{first_key}')")
        print(lookup[first_key])

    # Step 5: verify all train/dev db_ids have schemas
    print("\n── Coverage check ───────────────────────────────────────")
    for split_file in ["train.json", "dev.json"]:
        path = DATA_RAW / split_file
        if not path.exists():
            print(f"  {split_file}: not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        missing = [ex["db_id"] for ex in data if ex["db_id"] not in lookup]
        print(f"  {split_file}: {len(data)} examples, "
              f"{len(set(ex['db_id'] for ex in data))} unique dbs, "
              f"{len(missing)} missing schemas")
        if missing:
            print(f"    Missing: {set(missing)}")

    print(f"\nStep 2 complete. Schema lookup saved to {SCHEMA_OUT}")
    print("Run next: python src/03_format_prompts.py")


if __name__ == "__main__":
    main()