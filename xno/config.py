import os
from urllib.parse import quote_plus
import logging


class AppConfig:
    # Postgresql config
    _postgresql_host: str = os.environ['POSTGRES_HOST']
    _postgresql_port: int = os.environ['POSTGRES_PORT']
    _postgresql_db: str = os.environ['POSTGRES_DB']
    _postgresql_user: str = os.environ['POSTGRES_USER']
    _postgresql_password: str = os.environ['POSTGRES_PASSWORD']
    @property
    def postgresql_url(self):
        return "postgresql://{user}:{password}@{host}:{port}/{db}".format(
            user=self._postgresql_user,
            password=quote_plus(self._postgresql_password),
            host=self._postgresql_host,
            port=self._postgresql_port,
            db=self._postgresql_db
        )

    # Redis config
    _redis_host: str = os.environ['REDIS_HOST']
    _redis_port: int = os.environ['REDIS_PORT']
    _redis_db: int = os.environ.get('REDIS_DB', 0)
    semaphore_key: str = "xno_data_semaphore"
    semaphore_max_permits: int = 5  # max 5 operations
    @property
    def redis_config(self):
        return {
            'host': self._redis_host,
            'port': self._redis_port,
            'db': self._redis_db
        }

    # Kafka config
    kafka_bootstrap_servers: str = os.environ['KAFKA_BOOTSTRAP_SERVERS']
    kafka_topic: str = os.environ['KAFKA_TOPIC']
    _kafka_default_group_id: str = 'xno-data-consumer-group'
    @property
    def kafka_consumer_config(self):
        return {
            'bootstrap.servers': self.kafka_bootstrap_servers,
            'group.id': self._kafka_default_group_id,
            'auto.offset.reset': 'earliest'
        }


settings = AppConfig()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
