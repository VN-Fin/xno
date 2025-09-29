from confluent_kafka import Consumer, Producer
from xno import settings


KafkaProducer: Producer | None = None
KafkaConsumer: Consumer | None = None

def init_kafka_producer():
    global KafkaProducer
    if KafkaProducer is None:
        KafkaProducer = Producer(settings.kafka_producer_config)


def init_kafka_consumer(kafka_topics):
    global KafkaConsumer
    if KafkaConsumer is None:
        KafkaConsumer = Consumer(settings.kafka_consumer_config)
        KafkaConsumer.subscribe(kafka_topics)

if __name__ == "__main__":
    init_kafka_producer()
    # init_kafka_consumer()
    KafkaProducer.produce(topic="ping.dev", key="test_key", value="test_value")
    KafkaProducer.flush()
    print("Kafka Producer and Consumer initialized.")
