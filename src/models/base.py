"""
Unified interface so the evaluation harness can call GPT, Gemini, Claude,
Llama, Gemma, and Qwen identically. Add/adjust provider clients as APIs and
model names change — always re-check current model strings against each
provider's docs before a run (these change frequently).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelResponse:
    model_name: str
    prompt: str
    completion: str
    raw: dict
    error: str | None = None


class ModelClient(ABC):
    provider_name: str

    @abstractmethod
    def generate(self, prompt: str, model_name: str, max_tokens: int = 512) -> ModelResponse:
        ...


PROVIDER_REGISTRY: dict[str, type] = {}


def register_provider(name):
    def wrapper(cls):
        PROVIDER_REGISTRY[name] = cls
        return cls
    return wrapper


def get_client(provider_name: str) -> ModelClient:
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider_name}'. Registered: {list(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[provider_name]()
