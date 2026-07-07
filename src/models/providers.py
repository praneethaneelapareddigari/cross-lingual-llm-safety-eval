"""
Provider implementations. Each wraps one API family behind the common
ModelClient interface. Model name STRINGS must be verified against current
provider docs at run time — hardcoding them here would go stale.

Llama, Gemma, and Qwen are open-weight; we call them via Together AI as a
single hosting layer (swap for your own inference endpoint / HF Inference /
local vLLM server if preferred).
"""
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ModelClient, ModelResponse, register_provider


def _retry_decorator():
    return retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))


@register_provider("openai")
class OpenAIClient(ModelClient):
    provider_name = "openai"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @_retry_decorator()
    def generate(self, prompt, model_name, max_tokens=512):
        try:
            resp = self.client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return ModelResponse(model_name, prompt, resp.choices[0].message.content,
                                  raw=resp.model_dump())
        except Exception as e:
            return ModelResponse(model_name, prompt, "", raw={}, error=str(e))


@register_provider("anthropic")
class AnthropicClient(ModelClient):
    provider_name = "anthropic"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @_retry_decorator()
    def generate(self, prompt, model_name, max_tokens=512):
        try:
            resp = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return ModelResponse(model_name, prompt, text, raw=resp.model_dump())
        except Exception as e:
            return ModelResponse(model_name, prompt, "", raw={}, error=str(e))


@register_provider("google")
class GoogleClient(ModelClient):
    provider_name = "google"

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._genai = genai

    @_retry_decorator()
    def generate(self, prompt, model_name, max_tokens=512):
        try:
            model = self._genai.GenerativeModel(model_name)
            resp = model.generate_content(
                prompt, generation_config={"max_output_tokens": max_tokens}
            )
            return ModelResponse(model_name, prompt, resp.text, raw={"candidates": str(resp.candidates)})
        except Exception as e:
            return ModelResponse(model_name, prompt, "", raw={}, error=str(e))


@register_provider("together")
class TogetherClient(ModelClient):
    """Hosts open-weight Llama / Gemma / Qwen via Together AI's OpenAI-compatible API."""
    provider_name = "together"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )

    @_retry_decorator()
    def generate(self, prompt, model_name, max_tokens=512):
        try:
            resp = self.client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return ModelResponse(model_name, prompt, resp.choices[0].message.content,
                                  raw=resp.model_dump())
        except Exception as e:
            return ModelResponse(model_name, prompt, "", raw={}, error=str(e))
