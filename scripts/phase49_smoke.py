"""Small benchmark entry point suitable for CI smoke validation."""
from __future__ import annotations

from pathlib import Path

from benchmark_scale import run_benchmark


if __name__ == "__main__":
    print(run_benchmark(10_000, 2, Path("artifacts/benchmarks")).to_dict())
