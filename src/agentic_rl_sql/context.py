from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class MessageCounter(Protocol):
    def __call__(self, messages: list[dict[str, str]]) -> int: ...


@lru_cache(maxsize=4)
def load_tokenizer(model_or_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=False, use_fast=True)


def hf_message_counter(model_or_path: str) -> MessageCounter:
    tokenizer = load_tokenizer(model_or_path)

    def count(messages: list[dict[str, str]]) -> int:
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        return len(tokens)

    return count


def fit_schema_to_budget(
    build_messages, schema: str, limit: int, counter: MessageCounter | None,
) -> tuple[list[dict[str, str]], int | None, bool]:
    """Trim ONLY schema. Never silently discard user question or feedback."""
    messages = build_messages(schema)
    if counter is None:
        return messages, None, False
    if limit <= 0:
        raise ValueError("prompt token limit must be positive")
    original_tokens = counter(messages)
    if original_tokens <= limit:
        return messages, original_tokens, False

    marker = "\n[SCHEMA TRUNCATED TO FIT TOKEN BUDGET]"
    lo, hi = 0, len(schema)
    best = None
    best_count = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = build_messages(schema[:mid] + marker)
        size = counter(candidate)
        if size <= limit:
            best, best_count = candidate, size
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise ValueError("question/history exceed token budget even with empty schema")
    return best, best_count, True
