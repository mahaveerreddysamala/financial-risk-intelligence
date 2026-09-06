"""Narrow English request rules against declared evidence capabilities.

This is neither a general answerability model nor a security/PII filter.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class EvidenceScope:
    available_facts: frozenset[str] = frozenset()


# The built-in reference corpus and minimized case signals contain none of these facts.
REFERENCE_SCOPE = EvidenceScope()
FACT_LABELS = {
    "transaction_loss": "verified transaction-loss records",
    "customer_identity": "verified customer-identity records",
    "organization_forecast": "organization-specific forecasts",
}


def required_fact(question: str) -> str | None:
    text = " ".join(unicodedata.normalize("NFKC", question).lower().split())
    text = re.sub(r"^please\s+", "", text)
    request = re.match(r"^(what\b|how much\b|how many\b|tell\b|give\b|show\b|"
                       r"calculate\b|estimate\b|report\b)", text)
    if (request and re.search(r"\b(loss|losses|lost|stolen)\b", text)
            and re.search(r"\b(this|that|selected|our|my|actual|exact|case|transaction|customer)\b",
                          text)):
        return "transaction_loss"
    if (re.match(r"^(who (?:is|owns|holds)\b|identify (?:the )?(?:owner|customer|cardholder)\b|"
                 r"what is (?:the )?(?:name|identity)\b)", text)
            and re.search(r"\b(account|customer|cardholder|owner)\b", text)):
        return "customer_identity"
    if (re.match(r"^(predict\b|forecast\b|what will\b)", text)
            and re.search(r"\b(our|my|company|bank|organization)\b", text)
            and re.search(r"\b(fraud|loss|losses|rate)\b", text)):
        return "organization_forecast"
    return None


def unavailable_evidence(question: str, scope: EvidenceScope) -> str | None:
    fact = required_fact(question)
    if fact is not None and fact not in scope.available_facts:
        return (f"The configured evidence does not include {FACT_LABELS[fact]}. "
                "I cannot answer that request from these sources. "
                "You can ask about general investigation guidance instead.")
    return None
