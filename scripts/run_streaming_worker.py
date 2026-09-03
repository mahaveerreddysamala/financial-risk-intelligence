"""Run the financial risk streaming worker against a Kafka broker."""
from __future__ import annotations

import logging
import os

from financial_risk.streaming.events import EventEnvelope
from financial_risk.streaming.kafka import KafkaEventConsumer, KafkaEventProducer
from financial_risk.streaming.risk_consumer import scoring_result_event
from financial_risk.streaming.runtime import DeadLetterRecord, StreamingRuntime
from financial_risk.streaming.worker import process_event


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def main() -> None:
    """Start the continuously running Kafka risk-scoring worker."""
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    input_topic = os.getenv("KAFKA_INPUT_TOPIC", "financial-risk-events")
    output_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "financial-risk-scored")
    dead_letter_topic = os.getenv("KAFKA_DLQ_TOPIC", "financial-risk-dlq")
    group_id = os.getenv("KAFKA_GROUP_ID", "financial-risk-scoring-worker")

    consumer = KafkaEventConsumer(
        bootstrap_servers,
        group_id,
        input_topic,
        auto_offset_reset="earliest",
    )
    producer = KafkaEventProducer(
        bootstrap_servers,
        client_id="financial-risk-scoring-worker",
    )

    def publish_result(result) -> None:
        producer.publish(output_topic, scoring_result_event(result))

    def publish_dead_letter(record: DeadLetterRecord) -> None:
        event = EventEnvelope(
            event_id=f"dlq-{record.event_id}",
            event_type="transaction.dead_lettered",
            schema_version=1,
            occurred_at=record.failed_at,
            payload=record.to_dict(),
        )
        producer.publish(dead_letter_topic, event)

    runtime = StreamingRuntime(
        consumer=consumer,
        processor=process_event,
        publisher=publish_result,
        dead_letter=publish_dead_letter,
    )

    try:
        runtime.run_forever()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("streaming worker stopping")
    finally:
        producer.flush(10.0)
        consumer.close()
        logging.getLogger(__name__).info("streaming worker stats=%s", runtime.stats)


if __name__ == "__main__":
    main()
