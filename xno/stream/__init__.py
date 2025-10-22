import logging

from confluent_kafka import Producer
import time
from xno import settings

__all__ = ["produce_message"]

producer = Producer({
    'bootstrap.servers': settings.kafka_bootstrap_servers,
    'queue.buffering.max.messages': 1000000,  # optional tuning
    'queue.buffering.max.kbytes': 1048576,  # 1GB total buffer if you have memory
    'queue.buffering.max.ms': 100,  # batch flush interval
})



def delivery_report(err, msg):
    if err is not None:
        logging.error(f'Message delivery failed: {err}')
    else:
        logging.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def produce_message(topic: str, key: str, value: str):
    """Produce a message with backpressure control."""
    while True:
        try:
            producer.produce(topic, key=key, value=value, callback=delivery_report)
            # Trigger callback handling and free up buffer space
            producer.poll(0)
            break
        except BufferError:
            # Buffer full — wait and retry instead of busy flushing
            logging.warning("Kafka buffer full, waiting before retry...")
            producer.poll(0.5)  # let some messages clear out
            time.sleep(0.1)
