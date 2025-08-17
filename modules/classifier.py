#!/usr/bin/env python3
# run_batch_local_or_api.py
import os
import argparse
import importlib
import joblib
import pandas as pd
import requests

from modules.prompts.prompt_env import prompt_env
from modules.prompts.prompt_fem import prompt_fem

# --------- Config ----------
CSV_BY_TOPIC = {
    "environmentalism": "..data/revision_sistematica__environmentalism__clustered_v1.csv",
    "feminism":         "../data/revision_sistematica__feminism__clustered_v1.csv",
}
PROMPT_MODULE_BY_TOPIC = {
    "environmentalism": prompt_env,
    "feminism":         prompt_fem,
}
TEXT_COL = "Item content"

# -------------------- Backends --------------------

def _chat_ollama(base_url: str, model: str, prompt: str, temperature: float) -> str:
    """
    Ollama chat:
      POST {base_url}/api/chat
      body: {"model": "...", "messages":[{"role":"system","content": prompt}], "stream": false, "options":{"temperature": ...}}
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message", {})
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return content.strip()


def _chat_vllm(base_url: str, model: str, prompt: str, temperature: float, seed: int | None = 89) -> str:
    """
    vLLM via OpenAI-compatible route:
      POST {base_url}/v1/chat/completions
      body: {"model":"...", "messages":[{"role":"system","content": prompt}], "temperature": ..., "seed": ...}
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected vLLM response: {data}")


def _chat_openai_compatible(base_url: str, api_key: str, model: str, prompt: str, temperature: float, seed: int | None = 89) -> str:
    """
    Generic OpenAI-compatible API (KISSKI):
      POST {base_url}/v1/chat/completions
      headers: Authorization: Bearer <api_key>
      body: {"model":"...", "messages":[{"role":"system","content": prompt}], "temperature": ..., "seed": ... (optional)}
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected OpenAI-compatible response: {data}")


def generate_response(
    backend: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    api_key: str | None = None,
    seed: int | None = 89,
) -> str:
    if backend == "ollama":
        return _chat_ollama(base_url, model, prompt, temperature)
    elif backend == "vllm":
        return _chat_vllm(base_url, model, prompt, temperature, seed=seed)
    elif backend == "kisski":
        if not api_key:
            # fall back to env var if not provided via --api-key
            api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("KISSKI requires an API key. Provide --api-key or set LLM_API_KEY.")
        # Default base_url for KISSKI if missing
        if not base_url:
            base_url = os.environ.get("LLM_BASE_URL", "https://chat-ai.academiccloud.de/v1")
        return _chat_openai_compatible(base_url, api_key, model, prompt, temperature, seed=seed)
    else:
        raise ValueError("backend must be one of: ollama, vllm, kisski")


# -------------------- I/O helpers --------------------

def save_to_joblib(response: str, folder: str, index: int, prefix: str = "response"):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(response, f"{folder}/{prefix}_{index}.joblib")


def load_responses_to_dataframe(folder: str, num_files: int) -> pd.DataFrame:
    # unchanged helper for your downstream usage
    items = [joblib.load(f"{folder}/response_{i}.joblib") for i in range(num_files)]
    return pd.DataFrame(items, columns=["Response"])


# -------------------- Batch processing --------------------

def process_batch(
    df: pd.DataFrame,
    start_index: int,
    batch_size: int,
    folder: str,
    prompt_template: str,
    temperature: float,
    backend: str,
    base_url: str,
    model: str,
    api_key: str | None,
    seed: int | None,
):
    end_index = min(start_index + batch_size, len(df))
    for idx, row in df.iloc[start_index:end_index].iterrows():
        prompt = prompt_template.replace("<SEED>", f"{row[TEXT_COL]}")
        resp = generate_response(
            backend=backend,
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            api_key=api_key,
            seed=seed,
        )

        preview = resp.replace("\n", " ")
        print(f"[{idx}] {row[TEXT_COL][:80]} -> {preview[:160]}{'...' if len(preview)>160 else ''}")
        save_to_joblib(resp, folder, idx, "response")


# -------------------- CLI --------------------

def main():
    p = argparse.ArgumentParser(description="Batch classify with Llama 3.1 70B via Ollama, vLLM, or KISSKI (OpenAI-compatible).")
    p.add_argument("--topic", choices=["environmentalism", "feminism"], required=True,
                   help="Selects input CSV and prompt module.")
    p.add_argument("--backend", choices=["ollama", "vllm", "kisski"], default="ollama",
                   help="Inference backend.")
    p.add_argument("--base-url", default=None,
                   help="Backend base URL. Defaults: ollama=http://localhost:11434, vllm=http://localhost:8000, kisski=env LLM_BASE_URL or https://chat-ai.academiccloud.de/v1")
    p.add_argument("--api-key", default=None,
                   help="API key for KISSKI/OpenAI-compatible endpoints (falls back to env LLM_API_KEY).")
    p.add_argument("--model", default=None,
                   help="Model name/tag. Defaults: ollama=llama3.1:70b-instruct-q4_K_M; vllm/meta=meta-llama/Llama-3.1-70B-Instruct; kisski=meta-llama-3.1-70b-instruct (or your deployment's name)")
    p.add_argument("--batch-id", default=None, help="Output folder (default: <topic>_llama_v1)")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=89, help="Optional seed (passed where supported).")
    args = p.parse_args()

    # Defaults per backend
    if args.base_url is None:
        if args.backend == "ollama":
            args.base_url = "http://localhost:11434"
        elif args.backend == "vllm":
            args.base_url = "http://localhost:8000"
        elif args.backend == "kisski":
            # Use env or KISSKI default
            args.base_url = os.environ.get("LLM_BASE_URL", "https://chat-ai.academiccloud.de/v1")

    if args.model is None:
        if args.backend == "ollama":
            args.model = "llama3.1:70b-instruct-q4_K_M"
        elif args.backend == "vllm":
            args.model = "meta-llama/Llama-3.1-70B-Instruct"
        elif args.backend == "kisski":
            # Use the deployment’s registered name; override with --model if different
            args.model = "meta-llama-3.1-70b-instruct"

    # Resolve CSV & prompt
    csv_path = CSV_BY_TOPIC[args.topic]
    prompt_template = PROMPT_MODULE_BY_TOPIC[args.topic]

    # Load data
    df = pd.read_csv(csv_path)
    if TEXT_COL not in df.columns:
        raise KeyError(f"Expected '{TEXT_COL}' in {csv_path}")

    # Output folder & resume
    out_folder = f'data/{args.batch_id}' or f"data/{args.topic}_llama_v1"
    os.makedirs(out_folder, exist_ok=True)
    existing = {int(f.split("_")[-1].split(".")[0])
                for f in os.listdir(out_folder)
                if f.startswith("response_") and f.endswith(".joblib")
                and f.split("_")[-1].split(".")[0].isdigit()}

    n_total = len(df)
    start = 0
    while start < n_total:
        end = min(start + args.batch_size, n_total)
        # Skip window if all indices already processed
        if all(i in existing for i in range(start, end)):
            start = end
            continue

        process_batch(
            df=df,
            start_index=start,
            batch_size=args.batch_size,
            folder=out_folder,
            prompt_template=prompt_template,
            temperature=args.temperature,
            backend=args.backend,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            seed=args.seed,
        )
        start = end

    print(f"Done. Saved responses in: {out_folder}")


if __name__ == "__main__":
    main()


# Original implementation used:
# export LLM_BASE_URL="https://chat-ai.academiccloud.de/v1"   # or your team’s endpoint
# export LLM_API_KEY="<your_api_key>"
# python run_batch_local_or_api.py \
#   --backend kisski \
#   --topic feminism \
#   --batch-id 20250713_feminism_kisski \
#   --batch-size 50 \
#   --temperature 0.7 \
#   --model meta-llama-3.1-70b-instruct


# Example with ollama:
# ollama pull llama3.1:70b-instruct-q4_K_M
# python run_batch_local_or_api.py \
#   --backend ollama \
#   --topic environmentalism \
#   --batch-id 20250713_env_ollama \
#   --batch-size 50 \
#   --temperature 0.7
