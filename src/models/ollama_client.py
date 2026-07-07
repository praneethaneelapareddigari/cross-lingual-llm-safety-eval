"""
Local model client via Ollama (https://ollama.com) — no API key, no billing,
runs entirely on the local machine. This is the free substitute for the
Together-hosted Llama/Gemma/Qwen client in providers.py.

Prerequisite (on the machine actually running this, not in this sandbox):
    ollama pull llama3.3        # or whichever tag you have RAM for
    ollama pull gemma2
    ollama pull qwen2.5
    ollama serve                 # usually auto-starts as a background service

This talks to Ollama's local REST API at http://localhost:11434 by default.
"""
import json
import os

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ModelClient, ModelResponse, register_provider

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


@register_provider("ollama")
class OllamaClient(ModelClient):
    provider_name = "ollama"

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def generate(self, prompt, model_name, max_tokens=512):
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.0},
                },
                timeout=180,  # local CPU inference can be slow — give it room
            )
            resp.raise_for_status()
            data = resp.json()
            return ModelResponse(model_name, prompt, data.get("response", ""), raw=data)
        except requests.exceptions.ConnectionError as e:
            return ModelResponse(
                model_name, prompt, "", raw={},
                error=f"Could not reach Ollama at {self.base_url} — is `ollama serve` "
                      f"running? ({e})"
            )
        except Exception as e:
            return ModelResponse(model_name, prompt, "", raw={}, error=str(e))


def check_ollama_available(base_url: str = OLLAMA_BASE_URL) -> tuple[bool, list]:
    """Utility: check Ollama is up and list locally pulled models. Call this
    before a real run to fail fast with a clear message instead of 500 retries."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, models
    except Exception:
        return False, []


if __name__ == "__main__":
    ok, models = check_ollama_available()
    if ok:
        print(f"[ollama_client] Ollama is reachable. Locally pulled models: {models}")
    else:
        print(f"[ollama_client] Ollama NOT reachable at {OLLAMA_BASE_URL}. "
              f"Run `ollama serve` and `ollama pull <model>` first.")
