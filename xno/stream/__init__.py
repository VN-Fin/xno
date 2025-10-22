import logging

from confluent_kafka import Producer
import time
from xno import settings

__all__ = ["produce_message", "flush_producer"]

producer = Producer({
    'bootstrap.servers': settings.kafka_bootstrap_servers,
})



def delivery_report(err, msg):
    if err is not None:
        logging.error(f'Message delivery failed: {err}')
    else:
        logging.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def produce_message(topic: str, key: str, value: str):
    producer.produce(topic, key=key, value=value, callback=delivery_report)

def flush_producer():
    producer.flush(timeout=10)