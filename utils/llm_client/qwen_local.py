"""Local Qwen client compatible with vLLM / Ollama / LM Studio OpenAI-compatible servers."""

from typing import Optional
from .openai import OpenAIClient


class QwenLocalClient(OpenAIClient):
    """Client for local Qwen inference servers with telemetry tracking."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature: float = 0.7,
        base_url: Optional[str] = "http://localhost:8000/v1",
        api_key: Optional[str] = "EMPTY",
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
