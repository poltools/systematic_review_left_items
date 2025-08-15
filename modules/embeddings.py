import requests

def get_embedding(prompt: str, model: str = "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M") -> list:
    url = "http://localhost:11434/api/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("embedding", [])
    except requests.exceptions.RequestException as e:
        print("Request error:", e)
    except ValueError as e:
        print("Invalid JSON response:", e)
    except Exception as e:
        print("Unexpected error:", e)
    return []
