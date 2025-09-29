import os
from urllib.parse import quote_plus
import logging


class AppConfig:
    # Postgresql config
    postgresql_host: str = os.environ.get('POSTGRES_HOST', 'localhost')
    postgresql_port: int = os.environ.get('POSTGRES_PORT', 5432)
    postgresql_db: str = os.environ.get('POSTGRES_DB', 'xno')
    postgresql_user: str = os.environ.get('POSTGRES_USER', 'xno')
    postgresql_password: str = os.environ.get('POSTGRES_PASSWORD', 'xno_password')
    @property
    def postgresql_url(self):
        return "postgresql://{user}:{password}@{host}:{port}/{db}".format(
            user=self.postgresql_user,
            password=quote_plus(self.postgresql_password),
            host=self.postgresql_host,
            port=self.postgresql_port,
            db=self.postgresql_db
        )

    # Redis config
    redis_host: str = os.environ.get('REDIS_HOST', 'localhost')
    redis_port: int = os.environ.get('REDIS_PORT', 6379)
    redis_db: int = os.environ.get('REDIS_DB', 0)
    semaphore_key: str = "xno_data_semaphore"
    semaphore_max_permits: int = 5  # max 5 operations
    @property
    def redis_config(self):
        return {
            'host': self.redis_host,
            'port': self.redis_port,
            'db': self.redis_db
        }

    # Kafka config
    kafka_bootstrap_servers: str = os.environ['KAFKA_SERVERS']
    kafka_topic: str = os.environ['KAFKA_TOPIC']
    kafka_default_group_id: str = 'xno-data-consumer-group'
    @property
    def kafka_consumer_config(self):
        return {
            'bootstrap.servers': self.kafka_bootstrap_servers,
            'group.id': self.kafka_default_group_id,
            'auto.offset.reset': 'earliest'
        }


settings = AppConfig()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
