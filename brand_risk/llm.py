"""LLM client backed by any OpenAI-compatible endpoint.

Local dev  : llama-server (llama.cpp)  → LLM_BASE_URL=http://localhost:8080/v1
AMD Cloud  : vLLM on MI300X            → LLM_BASE_URL=http://localhost:8000/v1

LangSmith observability is zero-code: set LANGCHAIN_TRACING_V2=true and
LANGCHAIN_API_KEY — every invoke() call and the full LangGraph run are captured
automatically with token counts, latencies, and prompt/response content.

Why keep the manual retry loop instead of with_structured_output:
langchain-ai/langchain#28412 confirms with_structured_output breaks on nested
Pydantic models with local Llama3.1. The strip-fence → model_validate_json →
re-prompt loop is more robust for local models.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_llm_instance: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """Return the shared LLM client, building it on first call so that env vars
    loaded by load_dotenv() in app.py are visible before the client is created."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
            model=os.getenv("LLM_MODEL", "bartowski/Qwen_Qwen3-14B-GGUF:Q4_K_M"),
            api_key="none",  # vLLM / llama-server do not validate the key
            temperature=0.1,
        )
    return _llm_instance


def chat(system: str, user: str, temperature: float = 0.1) -> str:
    """Single-turn chat; returns the raw text content."""
    resp = _get_llm().bind(temperature=temperature).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return resp.content


def chat_stream(
    system: str, user: str, temperature: float = 0.7
) -> Iterator[str]:
    """Yield response tokens one-by-one for use with st.write_stream()."""
    for chunk in _get_llm().bind(temperature=temperature).stream(
        [SystemMessage(content=system), HumanMessage(content=user)]
    ):
        yield chunk.content


def chat_json(system: str, user: str, schema: Type[T], temperature: float = 0.1,
              retries: int = 2) -> T:
    """Call the model, parse JSON, validate against `schema`.

    On parse/validation failure re-prompts with the error so the model can
    self-correct. Never parses with regex — Pydantic validation only.
    """
    sys_with_schema = (
        system
        + "\n\nReturn ONLY a single valid JSON object, no markdown fences, "
          "no commentary. It must match this schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    bound = _get_llm().bind(temperature=temperature)
    last_err = ""
    for _ in range(retries + 1):
        raw = bound.invoke(
            [SystemMessage(content=sys_with_schema),
             HumanMessage(content=user + last_err)]
        ).content
        raw = (raw.strip()
               .removeprefix("```json")
               .removeprefix("```")
               .removesuffix("```")
               .strip())
        try:
            return schema.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = (
                f"\n\nYour previous reply was invalid ({e}). "
                "Return corrected JSON only."
            )
    raise RuntimeError(
        f"Model did not return valid {schema.__name__} after {retries + 1} tries"
    )
