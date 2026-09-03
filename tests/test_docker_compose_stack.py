from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
WORKER_DOCKERFILE = ROOT / "Dockerfile.worker"


def test_production_compose_declares_required_services():
    content = COMPOSE.read_text(encoding="utf-8")
    assert "services:" in content
    assert "  kafka:" in content
    assert "  kafka-init:" in content
    assert "  api:" in content
    assert "  worker:" in content


def test_production_compose_uses_kafka_health_dependency():
    content = COMPOSE.read_text(encoding="utf-8")
    assert "KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092" in content
    assert "condition: service_healthy" in content
    assert "financial-risk-events" in content
    assert "financial-risk-scored" in content
    assert "financial-risk-dlq" in content


def test_worker_image_installs_streaming_dependencies_and_runs_worker():
    content = WORKER_DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY requirements.txt ." in content
    assert "COPY requirements-streaming.txt ." in content
    assert "RUN pip install --no-cache-dir -r requirements-streaming.txt" in content
    assert "COPY scripts ./scripts" in content
    assert 'CMD ["python", "scripts/run_streaming_worker.py"]' in content
