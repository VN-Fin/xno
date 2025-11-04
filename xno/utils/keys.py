from xno import settings
from xno.models import AllowedTradeMode


def generate_latest_signal_key() -> str:
    return settings.redis_signal_latest_hash

def generate_latest_state_kafka_topic() -> str:
    return settings.kafka_state_latest_topic

def generate_latest_state_key() -> str:
    return settings.redis_state_latest_hash

def generate_latest_signal_kafka_topic() -> str:
    return settings.kafka_signal_latest_topic

def generate_backtest_overview_kafka_topic() -> str:
    return settings.kafka_backtest_overview_topic

def generate_backtest_history_kafka_topic() -> str:
    return settings.kafka_backtest_history_topic
