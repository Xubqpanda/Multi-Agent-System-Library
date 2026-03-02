# src/llm/base.py
import os
from typing import Protocol, Literal, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from openai import OpenAI


# ─── Message ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


# ─── LLM Protocol ─────────────────────────────────────

class LLMCallable(Protocol):
    def __call__(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: int = 512,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> str: ...


# ─── Abstract BASE ──────────────────────────────────────────────────────────────────

class LLMBase(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: int = 512,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = 1,
    ) -> str: ...


# ─── Token consumption ─────────────────────────────────────────

_completion_tokens: int = 0
_prompt_tokens: int = 0
_total_tokens: int = 0

def get_token_usage() -> tuple[int, int, float]:
    """Return (completion_tokens, prompt_tokens, total_tokens).""" 
    return _completion_tokens, _prompt_tokens, _completion_tokens + _prompt_tokens

def reset_token_usage() -> None:
    global _completion_tokens, _prompt_tokens, _total_tokens
    _completion_tokens = 0
    _prompt_tokens = 0
    _total_tokens = 0


# ─── GPTChat ─────────────────────────────────────────

class GPTChat(LLMBase):

    def __init__(self, model_name: str, base_url: str = None, api_key: str = None):
        super().__init__(model_name)
        self.client = OpenAI(
            base_url=base_url or os.environ["OPENAI_API_BASE"],
            api_key=api_key or os.environ["OPENAI_API_KEY"],
        )

    def __call__(
        self,
        messages: List[Message],
    ) -> str:
        import time
        global _prompt_tokens, _completion_tokens, _total_tokens

        formatted = [{"role": m.role, "content": m.content} for m in messages]
        max_retries, wait_time = 5, 1

        for _ in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted,
                )
                answer = response.choices[0].message.content
                _prompt_tokens += response.usage.prompt_tokens
                _completion_tokens += response.usage.completion_tokens
                _total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
                if answer is None:
                    continue
                return answer
            except Exception as e:
                err = str(e)
                if "rate limit" in err.lower() or "429" in err:
                    time.sleep(wait_time)
                else:
                    print(f"[LLM] API error: {err}")
                    break
        return ""