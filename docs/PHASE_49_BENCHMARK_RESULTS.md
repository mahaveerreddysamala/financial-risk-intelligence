# Phase 49 Benchmark Results

Phase 49 includes an explicit 1M-row scale run of the synthetic transaction benchmark. The benchmark measures bounded-memory transaction dataset generation and CSV output throughput; it does **not** represent end-to-end fraud model inference or production cloud throughput.

## Measured 1M Run

| Rows | Partitions | Runtime (s) | Rows/s | Peak Memory | Environment |
|---:|---:|---:|---:|---|---|
| 1,000,000 | 8 | 1.732874 | 577,075.9 | Not instrumented | Local Windows / Python 3.11 |

Command:

```powershell
python scripts/benchmark_scale.py --rows 1000000 --partitions 8 --output-dir artifacts/benchmarks
```

Output artifact:

```text
artifacts/benchmarks/transactions-1000000-p8.csv
```

## Interpretation

The measured run generated 1,000,000 deterministic synthetic transaction records in approximately 1.73 seconds, corresponding to approximately 577K rows/second for the benchmarked generation/output path.

This result is an engineering benchmark for the repository's scale-generation path on the stated local environment. It should not be presented as a claim that the complete financial-risk pipeline, graph enrichment, model inference, or cloud deployment processes 577K transactions/second.

## Reproducibility

The benchmark is parameterized by row count and partition count and can be rerun locally or through the Phase 49 GitHub Actions workflow. Results should be recorded with the execution environment when making performance comparisons.