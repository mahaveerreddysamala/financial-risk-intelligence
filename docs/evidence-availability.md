# Evidence availability check

The built-in reference corpus contains general investigation guidance. Selected cases add
only allowlisted numeric risk signals; neither contains verified transaction-loss records,
customer identities or organization-specific forecasts. A narrow English rule check rejects
recognized requests for these missing facts before retrieval and optional generation.
It returns `evidence-unavailable`, an explanation and no citations. No paid calls are enabled.

The dashboard explicitly passes `REFERENCE_SCOPE` in reference and selected-case modes.
Developer callers opt in with `answer_question(..., evidence_scope=REFERENCE_SCOPE)`.
The default is `None` for custom corpora, whose capabilities are unknown. An `EvidenceScope`
may declare available fact categories; this caller-supplied declaration is not proof that a
retrieved passage contains the requested fact. Passing the check does not prove answerability.

## Fresh-question evaluation

Rules were written from earlier failure analysis, then run on 36 new AI-authored questions:
18 general guidance and 18 missing-fact requests. There is no normalized exact query overlap
with previous fixtures. The author knew the rule categories and used the same six reference
chunks. This is not independent or blind human evaluation. Rules and thresholds were not
tuned after observing these results.

| Backend | Unsupported accepted, baseline | With check | Answerable rejected, baseline / check |
|---|---:|---:|---:|
| TF-IDF | 7/18 | 2/18 | 4/18 / 4/18 |
| MiniLM | 0/18 | 0/18 | 3/18 / 3/18 |

The check added zero rejections among 18 answerable questions. MiniLM already rejected every
unsupported question here, so there is no measured semantic improvement. Results concern
offline retrieval acceptance, not generated-answer correctness or production hallucinations.

The remaining TF-IDF accepts request an account holder's full legal name and the rise in
the company's next-quarter fraud rate. Their phrasing falls outside the narrow rules. Retain
these failures; do not fix them and reuse this fixture as unseen validation. The check does
not cover arbitrary unknown facts, paraphrases, other languages, quoted requests, prompt
injection or data redaction. Mixed requests may be fully rejected. Capability declarations
are maintained manually, not inferred from document contents.

## Reproduce

```bash
python scripts/evaluate_evidence_scope.py
python scripts/evaluate_evidence_scope.py --semantic
python -m pytest tests/test_evidence_scope.py tests/test_rag.py -q
```

Semantic execution requires optional dependencies and cached pinned MiniLM weights; it
permits no downloads. `evidence-scope-results.json` records fixture/corpus/rule hashes,
model revision, package versions, scores and rejection flags. Retrieval is measured once,
then the deterministic check is applied to the same results for paired counts. Reported
latencies cover retrieval, not the gated application's end-to-end latency. Unit tests verify
that actual RAG integration skips retrieval and generation on rejection, including with
selected-case signals. Dashboard CI submits a missing-loss request and checks its response.

## Resume wording

Implemented declared-evidence checks for a reference retrieval copilot; on 36 AI-authored
questions, reduced TF-IDF unsupported acceptance from 7/18 to 2/18 with no additional
answerable rejections, documenting remaining coverage gaps and unchanged semantic results.
