"""Calls to language models on free plans, with a cache and limits that are never passed."""

from norsk_lov_rag.llm.client import CacheMiss, ChatResult, MissingKey, ModelClient, ProviderError
from norsk_lov_rag.llm.quota import QuotaExhausted

__all__ = [
    "CacheMiss",
    "ChatResult",
    "MissingKey",
    "ModelClient",
    "ProviderError",
    "QuotaExhausted",
]
