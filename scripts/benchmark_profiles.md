# Benchmark Profiles

- Smoke: 10,000 rows
- Baseline: 100,000 rows
- Stress: 500,000 rows
- Production-like: 1,000,000+ rows

Use `scripts/benchmark_scale.py` to generate a reproducible workload and record throughput metadata. Run the 1M+ profile explicitly rather than in pull-request CI.
