from typing import Optional

from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LocalQwenClient(BaseClient):
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: Optional[str] = "local",
    ) -> None:
        super().__init__(model, temperature)

        if OpenAI is None:
            raise RuntimeError(
                "The openai package is required in the reevo-tsp environment."
            )

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _chat_completion_api(
        self,
        messages: list[dict],
        temperature: float,
        n: int = 1,
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=1500,
            stream=False,
        )

        return response.choices
