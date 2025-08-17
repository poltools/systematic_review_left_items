# modules/embeddings.py
#!/usr/bin/env python3
import argparse
import os
import sys
import time
from typing import List, Optional

import pandas as pd
import requests

DEFAULT_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"

def get_embedding(prompt: str,
                  model: str = DEFAULT_MODEL,
                  url: str = DEFAULT_URL,
                  timeout: int = 120) -> List[float]:
    """Return embedding vector for prompt using the local embeddings API."""
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"model": model, "prompt": prompt},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("embedding", [])
    except requests.exceptions.RequestException as e:
        print(f"[embed] request error: {e}", file=sys.stderr)
    except ValueError as e:
        print(f"[embed] invalid JSON response: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[embed] unexpected error: {e}", file=sys.stderr)
    return []

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    elif ext in {".csv"}:
        return pd.read_csv(path)
    elif ext in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported input format: {ext}")

def main():
    p = argparse.ArgumentParser(
        description="Compute and store text embeddings into a dataframe."
    )
    p.add_argument("--input", required=True, help="Path to input table (xlsx/csv/pkl)")
    p.add_argument("--output", required=True, help="Path to output pickle (.pkl)")
    p.add_argument("--url", default=DEFAULT_URL, help=f"Embeddings HTTP endpoint (default: {DEFAULT_URL})")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    p.add_argument("--raw-col", default="Item content",
                   help='Column with raw text (default: "Item content")')
    p.add_argument("--tokens-col", default="tokens_nouns__unique",
                   help='Column with tokenized/unique text (default: "tokens_nouns__unique")')
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Sleep seconds between requests (throttle)")
    p.add_argument("--no-raw", action="store_true", help="Skip raw text embeddings")
    p.add_argument("--no-tokens", action="store_true", help="Skip token text embeddings")
    args = p.parse_args()

    df = read_table(args.input)
    print(f">>> Loaded {len(df):,} rows from {args.input}")

    # Decide what to compute
    do_tokens = not args.no_tokens and (args.tokens_col in df.columns)
    do_raw = not args.no_raw and (args.raw_col in df.columns)

    if not do_tokens and not do_raw:
        cols = ", ".join(df.columns[:10])
        raise SystemExit(
            f"Nothing to embed. Checked for '{args.tokens_col}' and '{args.raw_col}' "
            f"but neither column exists. Available columns (first 10): {cols}"
        )

    # Helper with minimal progress printing (no extra deps)
    def iter_rows(series: pd.Series, label: str) -> List[List[float]]:
        out: List[List[float]] = []
        total = len(series)
        for i, text in enumerate(series.fillna("").astype(str), start=1):
            vec = get_embedding(text, model=args.model, url=args.url)
            out.append(vec)
            if i % 25 == 0 or i == total:
                print(f"  {label}: {i}/{total} done", end="\r", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
        print()  # newline after progress
        return out

    # Compute embeddings
    if do_tokens:
        print(f">>> Embedding tokenized column: {args.tokens_col} -> qwen_embeddings")
        df["qwen_embeddings"] = iter_rows(df[args.tokens_col], label="tokens")

    if do_raw:
        print(f">>> Embedding raw column: {args.raw_col} -> qwen_embeddings_raw")
        df["qwen_embeddings_raw"] = iter_rows(df[args.raw_col], label="raw")

    # Save
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_pickle(args.output)
    print(f">>> Saved dataframe with embeddings to {args.output}")

if __name__ == "__main__":
    main()

# Example of usage
# python modules/embeddings.py \
#   --input data/Systematic_review.xlsx \
#   --output data/items_df.pkl

