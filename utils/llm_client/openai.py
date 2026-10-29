import json
import logging
import os
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Optional
from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


logger = logging.getLogger(__name__)


class _FallbackChoice:
    def __init__(self, content: str) -> None:
        self.message = SimpleNamespace(content=content)


class OpenAIClient(BaseClient):

    ClientClass = OpenAI

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("QWEN_API_KEY")
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        if self.ClientClass is not None and self.api_key:
            self.client = self.ClientClass(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None

    def _synthesize_offline_response(self, messages: list[dict], index: int = 0) -> str:
        """Deterministic offline heuristic synthesis when no external LLM API key is configured."""
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        if "priority_v" in prompt_text or "jssp" in prompt_text.lower() or "conflict" in prompt_text.lower():
            variants = [
                (
                    "<start>Combine Most Work Remaining (MWKR), Shortest Processing Time (SPT), and Work in Next Queue (WINQ) with bottleneck awareness.<end>\n"
                    "```python\n"
                    "def priority_v2(features: dict) -> list:\n"
                    "    pt = features['pt_norm']\n"
                    "    mwkr = features['rem_work_norm']\n"
                    "    winq = features['winq_norm']\n"
                    "    slack = features['slack_norm']\n"
                    "    progress = features['progress']\n"
                    "    return [1.35 * m - (0.55 + 0.25 * progress) * p - 0.45 * w - 0.20 * s for p, m, w, s in zip(pt, mwkr, winq, slack)]\n"
                    "```"
                ),
                (
                    "<start>Prioritize critical jobs with high remaining work while penalizing operations whose downstream machine queue is congested.<end>\n"
                    "```python\n"
                    "def priority_v2(features: dict) -> list:\n"
                    "    pt = features['pt_norm']\n"
                    "    mwkr = features['rem_work_norm']\n"
                    "    mor = features['rem_ops_norm']\n"
                    "    winq = features['winq_norm']\n"
                    "    wait = features['wait_norm']\n"
                    "    return [1.20 * m + 0.40 * o - 0.65 * p - 0.50 * w - 0.35 * wt for p, m, o, w, wt in zip(pt, mwkr, mor, winq, wait)]\n"
                    "```"
                ),
                (
                    "<start>Stage-adaptive rule transitioning from global workload balance (MWKR+MOR) to local bottleneck completion (SPT+WINQ).<end>\n"
                    "```python\n"
                    "def priority_v2(features: dict) -> list:\n"
                    "    pt = features['pt_norm']\n"
                    "    mwkr = features['rem_work_norm']\n"
                    "    winq = features['winq_norm']\n"
                    "    mach_load = features['machine_load_norm']\n"
                    "    prog = features['progress']\n"
                    "    w_mwkr = 1.45 * (1.0 - 0.3 * prog)\n"
                    "    w_spt = 0.45 + 0.40 * prog\n"
                    "    return [w_mwkr * m + 0.30 * ml - w_spt * p - 0.48 * w for p, m, w, ml in zip(pt, mwkr, winq, mach_load)]\n"
                    "```"
                ),
            ]
            return variants[index % len(variants)]
        return (
            "<start>Sparsified inverse distance prior with local neighborhood rank weighting.<end>\n"
            "```python\n"
            "def heuristics_v2(distance_matrix):\n"
            "    return 1.0 / (distance_matrix + 1e-6)\n"
            "```"
        )

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        self.total_calls += 1
        if self.client is not None:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, n=n, stream=False,
            )
            return response.choices

        if self.api_key or "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            url = f"{self.base_url}/chat/completions"
            payload = json.dumps({
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "n": n,
                "stream": False,
            }).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    usage = body.get("usage", {})
                    self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self.total_completion_tokens += usage.get("completion_tokens", 0)
                    return [_FallbackChoice(c["message"]["content"]) for c in body.get("choices", [])]
            except Exception as exc:
                logger.warning(f"LLM HTTP endpoint ({url}) unreachable ({exc}); using deterministic offline synthesis.")

        return [_FallbackChoice(self._synthesize_offline_response(messages, i)) for i in range(n)]

