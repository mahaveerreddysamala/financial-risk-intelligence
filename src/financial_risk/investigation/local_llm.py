"""Local Ollama generation for the RAG JSON payload; no hosted fallback."""
from __future__ import annotations

import json
import re

import httpx

from financial_risk.investigation.rag import SYSTEM


class OllamaTextGenerator:
    def __init__(self, model="qwen2.5:3b", *, transport=None):
        if not model.strip() or "cloud" in model.lower() or "/" in model:
            raise ValueError("Select a local Ollama model tag")
        self.model = model
        self.transport = transport
        self.last_usage = None
        self.last_response = None

    def generate(self, prompt):
        self.last_usage = None
        self.last_response = None
        payload = json.loads(prompt)
        ids = [item["id"] for group in ("sources", "evidence")
               for item in payload.get(group, [])
               if isinstance(item.get("id"), str) and re.fullmatch(r"[SE][1-9]\d*", item["id"])]
        allowed = " ".join(f"[{source_id}]" for source_id in ids)
        instruction = SYSTEM + (
            " Answer in at most 150 words. End each factual sentence with a citation "
            "in square brackets, using the exact id of an item in sources or evidence. "
            "For example, use [S1] only when S1 exists in the supplied JSON. "
            "Do not cite document filenames or invent IDs. Do not repeat the question. "
            "If a requested conclusion is unsupported, explicitly say so and explain "
            "what the cited evidence does support."
            f" The ONLY allowed citation tokens for this answer are: {allowed}. "
            "Copy these citation tokens exactly. Do not use field names as citations.")
        with httpx.Client(base_url="http://127.0.0.1:11434", timeout=180,
                          trust_env=False, follow_redirects=False,
                          transport=self.transport) as client:
            response = client.post("/api/chat", json={
                "model": self.model, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 42, "num_predict": 600, "num_ctx": 4096},
                "messages": [{"role": "system", "content": instruction},
                             {"role": "user", "content": prompt}],
            })
            response.raise_for_status()
            body = response.json()
            result = body["message"]["content"]
            self.last_response = result if isinstance(result, str) else None
            counts = [body.get("prompt_eval_count"), body.get("eval_count")]
            if all(type(n) is int and n >= 0 for n in counts):
                self.last_usage = {"prompt_tokens": counts[0], "completion_tokens": counts[1],
                                   "total_tokens": sum(counts)}
            if not isinstance(result, str) or not result.strip() or not body.get("done"):
                raise ValueError("Local model returned no completed answer")
            if body.get("done_reason") == "length":
                raise ValueError("Local answer exceeded output limit")
            return result
