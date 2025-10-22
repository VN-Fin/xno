import logging

from confluent_kafka import Producer

from xno import settings

__all__ = ["produce_message"]

producer = Producer({
    'bootstrap.servers': settings.kafka_bootstrap_servers,
    'queue.buffering.max.messages': 1000000,  # optional tuning
    'queue.buffering.max.kbytes': 1048576,  # 1GB total buffer if you have memory
    'queue.buffering.max.ms': 100,  # batch flush interval
})



def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        logging.error(f'Message delivery failed: {err}')



def produce_message(topic: str, key: str, value: str):
    try:
        producer.produce(topic, key=key, value=value, callback=delivery_report)
    except BufferError:
        logging.warning("Kafka local buffer full, flushing...")
        producer.flush(1.0)
        producer.produce(topic, key=key, value=value, callback=delivery_report)
