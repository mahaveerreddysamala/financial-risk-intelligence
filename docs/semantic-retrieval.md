# Semantic retrieval — local model benchmark validated

This change builds on case-evidence PR #5. It adds an interchangeable TF-IDF and
sentence-transformer retrieval layer, plus a small authored regression benchmark.
The public dashboard still defaults to TF-IDF; no paid API calls are enabled.

## Validation status

After explicit user authorization, the pinned real model ran on CPU successfully, including
a second cached-only run. Machine-readable evidence is in
[semantic-retrieval-results.json](semantic-retrieval-results.json).

| Metric | TF-IDF | MiniLM |
|---|---:|---:|
| Recall@3 (10 answerable questions) | 0.90 | 1.00 |
| MRR@3 (10 answerable questions) | 0.7333 | 1.00 |
| Out-of-domain false accepts (3 questions) | 0/3 | 0/3 |
| Median query latency, cached run | 0.503 ms | 7.795 ms |
| P95 query latency, cached run | 1.102 ms | 10.160 ms |

The fixture and thresholds were unchanged between backends. These measurements apply only
to this tiny authored fixture; they do not establish production superiority or robustness.
No hosted LLM calls were made. Twenty-six focused tests passed in the semantic environment.
Unit tests use deterministic fake embeddings to test ranking, normalization, source identity,
input rejection and RAG integration. They do not validate model quality.

## Reproduce after approving model installation/download

Install the repository dependencies and `requirements-semantic.txt` in an appropriate
environment. For a CPU deployment, install the CPU build of PyTorch first following its
official installation instructions. Then run:

```bash
python scripts/benchmark_retrieval.py
python scripts/benchmark_retrieval.py --semantic --allow-download
```

The first command requires no model downloads. The second explicitly permits fetching
`sentence-transformers/all-MiniLM-L6-v2` at the revision recorded in `retrieval.py`.
Model loading disables remote code execution and requests safetensors. After downloading,
omit `--allow-download` for cached-only use. Missing dependencies or weights fail explicitly;
there is no silent fallback that could mislabel lexical results as semantic results.

Results are written to `artifacts/retrieval-benchmark.json`. The report records corpus and
question fingerprints, model revision, package versions, startup time, query timings,
Recall@3 and MRR@3 for ten answerable questions, and false acceptance for three out-of-domain
questions. Ranking metrics are calculated before threshold filtering; false acceptance uses
each backend's threshold. These thresholds are heuristics, not calibrated confidence.

The 13 questions and six document chunks are authored regression fixtures, not an independent
held-out study. MRR is truncated at rank three. Timings include query encoding but exclude
model/index startup, which is reported separately for semantic mode. No warm-up is excluded.
Do not present this tiny benchmark as production retrieval quality or LLM factuality evidence.

## Optional local integration

```python
from financial_risk.investigation.retrieval import SemanticRetriever, SentenceEncoder

engine = SemanticRetriever(documents, SentenceEncoder())  # cached model only
answer = answer_question(question, documents, case=case, retriever=engine)
```

The supplied engine owns its indexed document snapshot. Keep it aligned with the intended
corpus. Semantic search runs on local CPU; hosted text generation remains separately opt-in.
No vector database or semantic dashboard control is included in this implementation.

## Integration and evaluation status

- Completed: real-model and cached-only execution, unchanged fixture, retained JSON evidence.
- Broaden independent evaluation before generalizing quality or adjusting thresholds.
- Prerequisite PR #5 is merged; PR #6 targets main for normal CI gates and review.
- Semantic mode remains optional and is not enabled in the public dashboard.
