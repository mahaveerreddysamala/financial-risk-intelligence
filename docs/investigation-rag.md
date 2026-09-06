# Investigation RAG: first release

The dashboard adds an offline reference copilot. It chunks an explicitly approved synthetic
guidance document, retrieves TF-IDF passages, and shows source IDs and exact excerpts.
This is a lexical baseline, not neural embeddings or an LLM in offline mode.
Queries without a similarity match of at least 0.12 abstain; that heuristic is not calibrated.

## Optional hosted synthesis (developer opt-in only)

The OpenAI adapter follows the documented Chat Completions request contract:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/

Configure OPENAI_API_KEY and OPENAI_MODEL in your secret manager, never source control.
The public dashboard deliberately does not expose paid calls. For an authorized local run:

```python
import os
from pathlib import Path
from financial_risk.investigation.rag import (
    OpenAITextGenerator, answer_question, load_chunks,
)
generator = OpenAITextGenerator(os.environ['OPENAI_API_KEY'], os.environ['OPENAI_MODEL'])
answer = answer_question('How should shared devices be reviewed?', load_chunks(Path('.')), generator)
print(answer.text)
```

This sends the question and synthetic reference excerpts to OpenAI and incurs API charges.
No live calls were used during development. HTTP transport tests use a fake provider.
The adapter bounds completion output and timeout, disables storage, and does not retry.
Do not pass sensitive questions; automated PII redaction is NOT implemented.

## Evaluation and limits

Run `pytest -q tests/test_rag.py`. Four fixed retrieval queries check expected passage inclusion
among the top three results. Further tests cover abstention, question limits, unknown citations,
provider errors, and instruction-role separation. These fixtures are small regression checks,
not a held-out benchmark or proof of resistance to prompt injection.

Hosted responses require at least one known citation ID; otherwise they are withheld.
Valid IDs do NOT verify entailment, factual accuracy, or all claims. Responses remain
`llm-unverified` and require human review. No safety certification or hallucination-rate claim
is made. There are no tools for financial actions and no arbitrary document upload or URL fetch.

## Selected-case context

Choose a transaction under **Case context** or leave **Reference only** selected.
The copilot presents observed numeric signals as `[E1]` citations alongside `[S1]` references.
Only fraud probability, anomaly score, network risk, velocity risk and shared-device count
are allowed. Identifiers, amounts, location and free-text fields are excluded. Invalid numeric
values and duplicate fields are dropped. This is data minimization, not general-purpose PII
redaction: user questions are still unfiltered and must never contain sensitive information.
Signal labels expand retrieval queries, so case mode can return general signal guidance even
when a question is unsupported. The offline result is excerpts, not a claim to answer it.

Developer callers can pass `case=build_investigation_payload(row)` to `answer_question`.
In hosted mode the question, sanitized numeric observations and references are transmitted;
the full case is not. Hosted usage remains separately opt-in and unverified.

Next evaluation milestones: semantic embedding
comparison, held-out paraphrase/adversarial questions, per-claim citation support review,
latency/token/cost measurements using a separately authorized live evaluation.
