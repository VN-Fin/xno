from xno import settings


def generate_latest_signal_key(mode) -> str:
    return f"{settings.redis_signal_latest_hash}.{mode}"

def generate_latest_state_key(mode) -> str:
    return f"{settings.redis_state_latest_hash}.{mode}"

def generate_latest_signal_topic(mode) -> str:
    return settings.kafka_signal_latest_topic

def generate_latest_state_topic(mode) -> str:
    return settings.kafka_state_latest_topic

def generate_history_state_topic(mode) -> str:
    return settings.kafka_state_history_topic
