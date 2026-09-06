# Semantic retrieval — draft, model validation pending

This change builds on case-evidence PR #5. It adds an interchangeable TF-IDF and
sentence-transformer retrieval layer, plus a small authored regression benchmark.
The public dashboard still defaults to TF-IDF; no paid API calls are enabled.

## Validation status

Real-model dependency installation was blocked by cancelled network approval.
No real embedding inference, semantic benchmark result or retrieval improvement is claimed.
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
No vector database or semantic dashboard control is included in this draft.

## Before marking ready

- Install optional dependencies and run the pinned real model.
- Run both backends on the unchanged fixture and retain the machine-readable results.
- Inspect failed/paraphrased and out-of-domain queries without tuning on the evaluation set.
- Merge prerequisite PR #5, retarget to main and run the normal CI gates.
