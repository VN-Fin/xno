from confluent_kafka import Consumer, Producer
from xno import settings


KafkaProducer: Producer | None = None
KafkaConsumer: Consumer | None = None

def init_kafka_producer():
    global KafkaProducer
    KafkaProducer = Producer(settings.kafka_producer_config)


def init_kafka_consumer():
    global KafkaConsumer
    KafkaConsumer = Consumer(settings.kafka_consumer_config)
    KafkaConsumer.subscribe(settings.kafka_topics)
