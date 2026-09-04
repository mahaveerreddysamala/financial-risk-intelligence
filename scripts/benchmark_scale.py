"""Reproducible transaction-scale benchmark runner.

The benchmark intentionally keeps orchestration separate from the transaction
scoring implementation so it can be used with progressively larger datasets.
CI can run the smallest profile; production-like runs can use the 1M+ profile.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    rows: int
    partitions: int
    elapsed_seconds: float
    rows_per_second: float
    output_path: str
    process_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_synthetic_transactions(rows: int, output_path: Path, chunk_size: int = 100_000) -> int:
    """Write deterministic CSV transaction data in bounded-memory chunks."""
    if rows < 1:
        raise ValueError("rows must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("transaction_id,customer_id,merchant_id,amount\n")
        while written < rows:
            end = min(rows, written + chunk_size)
            for index in range(written, end):
                customer_id = index % 50_000
                merchant_id = index % 5_000
                amount = 10 + (index * 37 % 10_000) / 100
                handle.write(f"TX-{index:010d},C-{customer_id:08d},M-{merchant_id:06d},{amount:.2f}\n")
            written = end
    return written


def run_benchmark(rows: int, partitions: int, output_dir: Path) -> BenchmarkResult:
    """Generate a scale-controlled dataset and record throughput metadata."""
    if partitions < 1:
        raise ValueError("partitions must be at least 1")

    output_path = output_dir / f"transactions-{rows}-p{partitions}.csv"
    started = time.perf_counter()
    written = _write_synthetic_transactions(rows, output_path)
    elapsed = max(time.perf_counter() - started, 1e-9)
    result = BenchmarkResult(
        rows=written,
        partitions=partitions,
        elapsed_seconds=round(elapsed, 6),
        rows_per_second=round(written / elapsed, 2),
        output_path=str(output_path),
        process_id=os.getpid(),
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible transaction-scale benchmark")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    args = parser.parse_args()
    result = run_benchmark(args.rows, args.partitions, args.output_dir)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
