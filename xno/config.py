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

    def postgresql_url(self, db_name=postgresql_db):
        return "postgresql://{user}:{password}@{host}:{port}/{db}".format(
            user=self.postgresql_user,
            password=quote_plus(self.postgresql_password),
            host=self.postgresql_host,
            port=self.postgresql_port,
            db=db_name
        )

    # Redis config
    redis_host: str = os.environ.get('REDIS_HOST', 'localhost')
    redis_port: int = os.environ.get('REDIS_PORT', 6379)
    redis_db: int = os.environ.get('REDIS_DB', 0)
    redis_user: str = os.environ.get('REDIS_USER', 'default')
    redis_password: str = os.environ.get('REDIS_PASSWORD', None)

    @property
    def redis_config(self):
        return {
            'host': self.redis_host,
            'username': self.redis_user,
            'password': self.redis_password,
            'port': self.redis_port,
            'db': self.redis_db
        }

    # Kafka config vs topics
    kafka_bootstrap_servers: str = os.environ.get('KAFKA_SERVERS', 'localhost:9092')
    kafka_market_data_topic: str = "market.data.transformed"
    kafka_state_history_topic: str = "strategy.state.history"
    kafka_state_latest_topic: str = "strategy.state.latest"
    kafka_signal_latest_topic: str = "strategy.signal.latest"
    kafka_ping_topic: str = "ping"
    # Config for execution database
    execution_db_name: str = "xno_execution"
    # Redis hash key config
    redis_signal_latest_hash: str = kafka_signal_latest_topic
    redis_state_latest_hash: str = kafka_state_latest_topic


settings = AppConfig()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


