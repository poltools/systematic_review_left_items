#!/usr/bin/env python3
"""
Batch classification of scale items with an instruction-tuned LLM.

Sends each item (column "Item content") through the topic prompt to any
OpenAI-compatible chat-completions endpoint and stores one
`response_<index>.joblib` file per item under data/<batch-id>/.

Usage (from the repository root):
    export LLM_BASE_URL="<openai-compatible base url, ending in /v1>"
    export LLM_API_KEY="<key, if your endpoint needs one>"
    python -m modules.classifier --topic feminism --batch-id llama_classified/<batch_id>_feminism
"""
import os
import argparse
import joblib
import pandas as pd
import requests

from modules.prompts.prompt_env import prompt_env
from modules.prompts.prompt_fem import prompt_fem

# --------- Config ----------
CSV_BY_TOPIC = {
    "environmentalism": "data/environmentalism__clustered_v1.csv",
    "feminism":         "data/feminism__clustered_v1.csv",
}
PROMPT_BY_TOPIC = {
    "environmentalism": prompt_env,
    "feminism":         prompt_fem,
}
TEXT_COL = "Item content"

DEFAULT_MODEL = "meta-llama-3.1-70b-instruct"   # Llama-3.1-70B-Instruct
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 89


# -------------------- Backend --------------------

def generate_response(base_url: str, model: str, prompt: str, temperature: float,
                      api_key: str | None = None, seed: int | None = DEFAULT_SEED) -> str:
    """POST {base_url}/chat/completions with the prompt as a single system message."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(url, headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected response: {data}")


# -------------------- I/O helpers --------------------

def save_to_joblib(response: str, folder: str, index: int, prefix: str = "response"):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(response, f"{folder}/{prefix}_{index}.joblib")


def load_responses_to_dataframe(folder: str, num_files: int) -> pd.DataFrame:
    items = [joblib.load(f"{folder}/response_{i}.joblib") for i in range(num_files)]
    return pd.DataFrame(items, columns=["Response"])


# -------------------- Batch processing --------------------

def process_batch(df, start_index, batch_size, folder, prompt_template, temperature,
                  base_url, model, api_key, seed):
    end_index = min(start_index + batch_size, len(df))
    for idx, row in df.iloc[start_index:end_index].iterrows():
        prompt = prompt_template.replace("<SEED>", f"{row[TEXT_COL]}")
        resp = generate_response(base_url, model, prompt, temperature, api_key=api_key, seed=seed)
        preview = resp.replace("\n", " ")
        print(f"[{idx}] {row[TEXT_COL][:80]} -> {preview[:160]}{'...' if len(preview) > 160 else ''}")
        save_to_joblib(resp, folder, idx, "response")


# -------------------- CLI --------------------

def main():
    p = argparse.ArgumentParser(description="Batch-classify items with Llama-3.1-70B-Instruct via an OpenAI-compatible endpoint.")
    p.add_argument("--topic", choices=list(CSV_BY_TOPIC), required=True)
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"),
                   help="OpenAI-compatible base URL ending in /v1 (default: env LLM_BASE_URL).")
    p.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"),
                   help="API key, if required (default: env LLM_API_KEY).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Model name as registered on your endpoint.")
    p.add_argument("--batch-id", default=None,
                   help="Output folder under data/ (default: llama_classified/<topic>_llama_v1).")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Passed where the endpoint supports it.")
    args = p.parse_args()

    if not args.base_url:
        p.error("Set --base-url or LLM_BASE_URL.")

    csv_path = CSV_BY_TOPIC[args.topic]
    prompt_template = PROMPT_BY_TOPIC[args.topic]
    df = pd.read_csv(csv_path)
    if TEXT_COL not in df.columns:
        raise KeyError(f"Expected '{TEXT_COL}' in {csv_path}")

    out_folder = os.path.join("data", args.batch_id or f"llama_classified/{args.topic}_llama_v1")
    os.makedirs(out_folder, exist_ok=True)
    existing = {int(f.split("_")[-1].split(".")[0])
                for f in os.listdir(out_folder)
                if f.startswith("response_") and f.endswith(".joblib")
                and f.split("_")[-1].split(".")[0].isdigit()}

    n_total, start = len(df), 0
    while start < n_total:
        end = min(start + args.batch_size, n_total)
        if not all(i in existing for i in range(start, end)):   # resume: skip finished windows
            process_batch(df, start, args.batch_size, out_folder, prompt_template, args.temperature,
                          args.base_url, args.model, args.api_key, args.seed)
        start = end

    print(f"Done. Saved responses in: {out_folder}")


if __name__ == "__main__":
    main()
