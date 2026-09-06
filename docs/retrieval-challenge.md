# Retrieval challenge and comparison demo

PR #6 introduced the optional local MiniLM backend. This follow-up evaluates 50 new
AI-authored questions: 40 answerable and 10 unsupported, including misleading premises
and instruction-like queries. Labels were authored against the same six reference chunks
before execution. This is not independent human validation or a held-out production study.
The original 13-question fixture remains unchanged. No thresholds were tuned on this run.

| Metric | TF-IDF | MiniLM |
|---|---:|---:|
| Recall@3, before thresholding | 0.900 | 1.000 |
| MRR@3, before thresholding | 0.8333 | 0.9375 |
| Answerable questions accepted | 33/40 | 33/40 |
| Unsupported questions accepted | 5/10 | 1/10 |
| Median query latency | 0.497 ms | 10.849 ms |
| P95 query latency | 1.053 ms | 23.733 ms |

Acceptance means at least one retrieved hit exceeds the backend threshold. It does not
mean an answer is correct or that the relevant hit survived filtering. Unsupported-query
acceptance measures retrieval behavior, not an LLM hallucination or injection-success rate.
Case-context expansion was not enabled in this benchmark. Results do not characterize that mode.

## Reproduce and explore

With normal dependencies and the optional semantic environment installed, run:

```bash
python scripts/benchmark_retrieval.py --semantic \
  --fixture tests/fixtures/retrieval_challenge.json \
  --output artifacts/retrieval-challenge-results.json \
  --html artifacts/retrieval-comparison.html
```

The model must already be cached; this command permits no model download. Omit `--semantic`
for lexical-only execution. Open the generated HTML locally to compare backends side by side
and expand each query's retrieved excerpts, scores, acceptance and timing. It uses no CDN,
API calls or credentials. The committed `retrieval-comparison.html` captures this run;
`retrieval-challenge-results.json` records fingerprints, model revision and per-query evidence.
Timings include query encoding; model startup is separately recorded. No warm-up is excluded.

## Answer-quality review protocol

No hosted answers were generated or graded. Citation support and answer factuality remain
unmeasured. To evaluate them later, freeze questions, corpus, model, prompt and thresholds,
then record one row per generated claim with these fields:

`question_id, backend, answer, claim, cited_source_ids, supporting_quote,
support_label, expected_abstention, actual_abstention, reviewer, notes`

Use support labels `supported`, `contradicted`, or `not_enough_evidence`. Review every factual
claim against the cited passage, not merely the presence of a valid source ID. Report supported
claims divided by all factual claims; include uncited claims in the denominator. Report
unsupported-question abstention and answerable-question abstention separately, with counts.
Use two independent human reviewers and adjudicate disagreements before reporting quality.
Instruction-like queries alone do not test malicious retrieved documents; that needs a separate
fixture and evaluation. Do not describe either test as a security certification.

## Resume wording supported by this work

- Built an optional CPU semantic retrieval backend with pinned MiniLM weights, cited evidence,
  cached-only loading, deterministic tests and a portable comparison report.
- Evaluated TF-IDF and MiniLM on 50 AI-authored challenge questions; measured Recall@3 of
  0.90 versus 1.00 and unsupported-query acceptance of 5/10 versus 1/10, documenting latency
  and the limitations of synthetic evaluation.

These are portfolio engineering and experimental results, not production business impact.
