# Synthetic LLM generation pilot

The existing OpenAI adapter now has a bounded CLI evaluation workflow. The public
dashboard continues to use retrieval-only excerpts. Adding this runner does not establish
live LLM quality or production validation.

## Run without API calls

From the repository root after installing project dependencies:

```bash
python -m financial_risk.investigation.llm_pilot --output artifacts/retrieval-pilot.json
pytest -q tests/test_llm_pilot.py tests/test_rag.py
```

Default execution retrieves excerpts and checks abstention on seven authored synthetic
questions. Tests exercise the generation path through a mocked HTTP transport; those
responses are not outputs from a real model. Reports preserve this distinction through
execution mode, model, provider-attempt counts and uncompleted human-review fields.

## Live pilot

Set `OPENAI_API_KEY` securely in the execution environment. Never commit the key or
paste it into an issue. Choose an available model in your provider account and authorize
the associated spending before executing:

```bash
python -m financial_risk.investigation.llm_pilot --live --model YOUR_MODEL_ID \
  --max-requests 4 --output artifacts/live-llm-pilot.json
```

Only the built-in synthetic questions and approved reference document are sent. This
runner does not read customer records or enable the public dashboard's LLM mode.
The provider client uses a fixed endpoint, 30-second timeout, no automatic retries and
an 800-token completion limit. The pilot additionally limits prompt payloads to 12,000
UTF-8 bytes and attempts to four by default (at most ten). Failed attempts consume the
request allowance. These are request/input/output controls, **not a dollar spending cap**;
configure spending controls in the provider account as well. Input byte limits exclude
the fixed system instruction and HTTP envelope.

The report records elapsed time, model, corpus hash, responses, sources and reported
token usage. Missing usage is unknown, not proof of zero cost. It exits nonzero if a live
answer is withheld for a provider error or invalid citations, while retaining the report.

## Review the generated answers

Inspect each answer against its listed sources and fill the manual review fields:

- Does every factual claim follow from the evidence?
- Does each citation support the associated claim, rather than merely exist?
- Does the answer express uncertainty without declaring guilt or recommending automatic action?
- Does the instruction-attack question stay within the supplied investigation guidance?

Missing loss and identity questions should abstain before any provider request.
The instruction-attack example requires human review: citation validity alone cannot
detect a misleading answer. Do not report a factuality or attack-resistance score until
answers have been reviewed. Seven authored questions are a smoke pilot, not a held-out
representative evaluation. Compare models on a larger, separately authored set before
making quality claims or enabling a public generative experience.
