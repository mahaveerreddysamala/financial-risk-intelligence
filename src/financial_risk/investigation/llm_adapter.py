"""Provider-neutral interface for grounded investigation responses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from financial_risk.investigation.copilot import CopilotContext, build_grounded_prompt


class TextGenerator(Protocol):
    """Minimal interface implemented by an external LLM client."""

    def generate(self, prompt: str) -> str:
        """Generate a text response from a prompt."""


@dataclass(frozen=True)
class CopilotResponse:
    case_id: str
    prompt: str
    response: str
    grounded: bool = True


def run_grounded_copilot(context: CopilotContext, generator: TextGenerator) -> CopilotResponse:
    """Generate a response using only the grounded prompt contract."""
    prompt = build_grounded_prompt(context)
    response = generator.generate(prompt)
    if not isinstance(response, str) or not response.strip():
        raise ValueError("generator must return a non-empty string")
    return CopilotResponse(
        case_id=context.case_id,
        prompt=prompt,
        response=response.strip(),
    )


def build_evidence_only_fallback(context: CopilotContext) -> str:
    """Provide a deterministic fallback when no external LLM is configured."""
    evidence = "; ".join(
        f"{item.get('field')}={item.get('value')}"
        for item in context.evidence
    )
    references = ", ".join(doc.document_id for doc in context.retrieved_documents) or "none"
    return (
        f"Case {context.case_id}: evidence summary only. "
        f"Observed evidence: {evidence or 'none supplied'}. "
        f"Retrieved references: {references}. "
        "No unsupported conclusions are generated."
    )
