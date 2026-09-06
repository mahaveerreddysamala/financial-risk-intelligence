# Abstention failure analysis and separate-question validation

## Decision

Keep MiniLM's runtime threshold at 0.35. A candidate of 0.36 fixed one development
failure but did not improve the separate 30-question validation set. No runtime code,
dashboard settings or paid LLM behavior changed. This is a documented negative experiment.

| Set / threshold | Unsupported accepted | Answerable rejected |
|---|---:|---:|
| Development, 0.35 | 1/10 | 7/40 |
| Development, 0.36 | 0/10 | 7/40 |
| Validation, 0.35 | 1/10 | 4/20 |
| Validation, 0.36 | 1/10 | 4/20 |

Acceptance here means any of the top three similarity scores meets the threshold;
it does not establish answer correctness or support. Ranking Recall@3 is computed before
filtering and can remain perfect even while answerable questions are rejected.

## What failed on the 50-question development set

The exact-loss question for TX-999 scored 0.3568 against general velocity guidance.
The corpus contains no transaction-level loss. This illustrates topical similarity without
the requested factual evidence; source-ID validation cannot repair that distinction.

Seven answerable queries were rejected:

| Query topic | Highest similarity | Top reference |
|---|---:|---|
| Legitimate communal terminals | 0.3058 | Model signals (wrong first hit) |
| Recent-interval activity | 0.3323 | Velocity |
| Current event in historical features | 0.2303 | Human review (wrong first hit) |
| Closing an investigation without an analyst | 0.3187 | Human review |
| Recognized citation versus supported claim | 0.3490 | Human review |
| Information users should avoid submitting | 0.2616 | Human review |
| Passwords and authentic customer records | 0.2156 | Shared devices (wrong first hit) |

All seven have a relevant reference in the unfiltered top three. Four have the right
first hit but fall below the acceptance threshold; three also have a ranking error at
position one. Lowering the threshold increases unsupported acceptance, so a single cutoff
does not resolve the observed retrieval and answerability failures together.

## Experiment design and result

Treat the existing challenge set as development data after inspecting its errors. Evaluate
the fixed threshold grid 0.20, 0.25, 0.30, 0.35, 0.36, 0.40, 0.45, 0.50 there only.
Select by lowest false-accept count, then lowest false-reject count, then lowest threshold.
This selected 0.36 before running validation. The grid was informed by development errors;
it was not an independently specified research design.

The new fixture contains 20 answerable and 10 unsupported questions, with no normalized
exact query overlap with development. Both fixtures are AI-authored against the same small
corpus; semantic overlap remains and this is not independent human validation. The model,
corpus and revision are checked for consistency. Package versions and input hashes remain
in the raw reports. Case-context expansion and hosted generation are excluded.

The predeclared gate requires strictly fewer unsupported accepts and no additional
answerable rejects on validation. Both thresholds had the same counts, so the gate failed.
The remaining unsupported query asked for the company's exact next-quarter fraud rate;
its similarity was 0.3810 to model-signal guidance. Do not retune against this newly exposed
failure and then call the same questions held out.

## Reproduce

With the pinned model cached and optional dependencies installed:

```bash
python scripts/benchmark_retrieval.py --semantic \
  --fixture tests/fixtures/retrieval_validation.json \
  --output docs/retrieval-validation-results.json
python scripts/evaluate_abstention.py
python -m pytest tests/test_abstention_evaluation.py -q
```

`retrieval-validation-results.json` contains all query scores and timings;
`abstention-experiment.json` records selection, confusion counts and report hashes.
The evaluator checks normalized query overlap and rejects mismatched models or corpora.
Unit tests cover threshold equality, overlap rejection, development-only selection and
a validation regression that must fail the gate.

## Recorded demo and resume wording

`retrieval-demo.cast` is an actual timed terminal recording of cached-model execution and
the experiment report, in asciinema v2 format. It is a terminal replay, not a browser video.
With asciinema installed, run `asciinema play docs/retrieval-demo.cast` to replay it locally.
The existing HTML comparison in `retrieval-comparison.html` remains available for a visual
walkthrough of the 50-question run.

Resume bullet:

> Built a reproducible RAG retrieval evaluation workflow with separate development and
> validation fixtures, threshold-selection gates, failure analysis and recorded execution;
> rejected a threshold change that failed to improve unsupported-query acceptance on a
> 30-question AI-authored validation set.

Combine this with the engineering and measured-retrieval bullets in `retrieval-challenge.md`.
Do not claim reduced production hallucinations, independently validated quality, or a
deployed answerability improvement from this experiment.
