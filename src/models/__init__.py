"""
Importing this package registers all available model providers (both the
paid-API clients in providers.py and the free local client in
ollama_client.py) into the PROVIDER_REGISTRY in base.py.
"""
from . import ollama_client  # noqa: F401  (registers "ollama")

try:
    from . import providers  # noqa: F401  (registers openai/anthropic/google/together)
except ImportError:
    # Paid-API provider deps (openai, anthropic, google-generativeai) are
    # optional if you're running the fully-local Ollama path only.
    pass
