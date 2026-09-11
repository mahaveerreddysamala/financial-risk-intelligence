# Run local RAG and the transaction-to-brief demo

Use Python 3.11 or 3.12. Install the project in a virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install [Ollama](https://ollama.com/download). Start a server with cloud features disabled:

```bash
OLLAMA_NO_CLOUD=1 ollama serve
```

For Windows PowerShell set `$env:OLLAMA_NO_CLOUD="1"` before `ollama serve`. Exit an
already-running desktop Ollama server before starting this configured one. In another terminal:

```bash
ollama pull qwen2.5:3b
python -m financial_risk.investigation.demo --local
python -m financial_risk.investigation.llm_pilot --local
python -m financial_risk.investigation.evaluate_generation artifacts/llm-pilot.json
```

Model download and Python installation are one-time setup. The adapter connects only to
`127.0.0.1:11434`, ignores proxy variables, follows no redirects and has no paid-provider fallback.
Cloud-like tags are rejected, but daemon cloud-disable configuration is still required.
See the [Ollama FAQ](https://docs.ollama.com/faq) and
[model listing and weight license](https://ollama.com/library/qwen2.5:3b).

## What runs end to end

Synthetic transactions → existing temporal features → XGBoost, IsolationForest and
device/velocity heuristics with a community proxy → highest-risk holdout case → exact
case observations plus retrieved reference guidance → analyst brief.

The demo writes `artifacts/end-to-end-demo/demo.json` and `analyst-brief.md`. Numeric case
observations are copied mechanically. The LLM receives only reference passages and drafts
general review guidance; the analyst makes the case-specific interpretation. This boundary
was introduced after a local model misinterpreted score magnitudes and a shared-account count
despite valid citation IDs. No payment action is executed. The hosted dashboard remains
retrieval-only and cannot reach Ollama running on your laptop.

## Evaluation

Seven authored questions cover three answerable requests, three expected abstentions and
one instruction attack. Generation is limited to four requests by default, 12,000 prompt bytes,
600 local output tokens and a 180-second timeout. There are no automatic retries. Incomplete
output or missing/unknown citation IDs withholds the answer and returns nonzero from the CLI.
The report retains raw model text for review; withheld text is not an endorsed brief.

The evaluator separates citation-ID checks and abstention from claim-level groundedness
and citation accuracy. The latter remain null until reviews are supplied using `--reviews`:

```json
{"shared-devices": {"reviewer": "reviewer name", "groundedness": true,
 "citation_accuracy": true, "notes": "Explain which source supports each material claim."}}
```

Latency p50/p95 uses only attempted model requests. Four calls are insufficient for a stable
latency estimate. This fixture was reused during development, so it is not held-out quality
evidence. Independent paraphrases, larger case sets and human review remain outstanding.

## Development findings retained from the session

Qwen3 1.7B produced citation failures (1/4 accepted in each of two prompt attempts); Qwen3
4B exhausted the 600-token output cap (0/4 accepted). Qwen2.5 3B with explicit allowed IDs
produced 4/4 valid citation-ID answers and 3/3 expected abstentions. A separate case draft
still made unsupported interpretations, motivating separation of exact observations and
reference-only generation. These counts were observed in session execution; earlier raw
artifacts were lost in a workspace reset. Fresh execution artifacts are recorded separately.
Do not treat sequential development runs as a controlled comparison or production validation.

## Two-minute demo

0:00 explain investigator capacity; 0:20 run the demo command; 0:50 show the scored case and
exact observations; 1:15 show the generated guidance and retrieved sources; 1:40 show an
abstention and explain why valid citations do not prove correct interpretations. Pre-run
the model for a strict two-minute presentation; CPU and cold-load time vary.

CI tests the local transport with mocks, runs the real scoring/retrieval demo without a model,
and launches the actual HTTP API for readiness, scoring, case creation and invalid-input checks.
