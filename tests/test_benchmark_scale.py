from pathlib import Path

import pytest

from scripts.benchmark_scale import run_benchmark


def test_scale_benchmark_is_deterministic_and_records_metrics(tmp_path: Path):
    result = run_benchmark(rows=25, partitions=4, output_dir=tmp_path)

    assert result.rows == 25
    assert result.partitions == 4
    assert result.elapsed_seconds > 0
    assert result.rows_per_second > 0
    assert Path(result.output_path).exists()
    assert Path(result.output_path).with_suffix(".json").exists()

    lines = Path(result.output_path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "transaction_id,customer_id,merchant_id,amount"
    assert len(lines) == 26
    assert lines[1].startswith("TX-0000000000,C-00000000,M-000000")


def test_scale_benchmark_rejects_invalid_configuration(tmp_path: Path):
    with pytest.raises(ValueError, match="rows"):
        run_benchmark(rows=0, partitions=4, output_dir=tmp_path)
    with pytest.raises(ValueError, match="partitions"):
        run_benchmark(rows=10, partitions=0, output_dir=tmp_path)
