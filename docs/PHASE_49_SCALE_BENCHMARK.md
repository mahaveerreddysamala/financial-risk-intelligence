# Phase 49 — Large-Scale Benchmarking

## Objective

Measure how the transaction data layer behaves as workload size increases, while keeping benchmark runs deterministic and reproducible.

## Scale profiles

| Profile | Rows | Purpose |
|---|---:|---|
| Smoke | 10K | Fast developer validation |
| Baseline | 100K | Standard benchmark comparison |
| Stress | 500K | Mid-scale throughput test |
| Production-like | 1M+ | Large-volume engineering benchmark |

Run the benchmark with:

```bash
python scripts/benchmark_scale.py --rows 100000 --partitions 4
python scripts/benchmark_scale.py --rows 1000000 --partitions 8
```

The runner writes a transaction CSV plus a JSON metadata file containing row count, configured partitions, elapsed time, throughput, output path, and process ID.

## What to compare

For each scale, record:

- elapsed wall-clock time
- rows/second throughput
- partition configuration
- output size
- peak memory when running the downstream Spark pipeline
- executor/driver behavior for distributed runs

The benchmark generator uses bounded-memory chunks so the input-generation step does not require constructing the full dataset in memory.

## Engineering interpretation

The benchmark is intentionally separate from model-quality evaluation. A faster pipeline is not automatically a better fraud model. The target is to establish a repeatable performance baseline that can be compared across partition counts, processing implementations, and infrastructure sizes.

CI should use the smoke/baseline profiles only. The 1M+ profile is intended for an explicit local or cloud benchmark run rather than every pull request.
