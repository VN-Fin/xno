from xno import settings
from xno.models import AllowedTradeMode


def generate_latest_signal_key(mode: AllowedTradeMode | int) -> str:
    # Check mode instance
    if not isinstance(mode, AllowedTradeMode):
        mode = AllowedTradeMode(mode)

    return f"{settings.redis_signal_latest_hash}.{mode}"

def generate_latest_state_key(mode: AllowedTradeMode | int) -> str:
    # Check mode instance
    if not isinstance(mode, AllowedTradeMode):
        mode = AllowedTradeMode(mode)

    return f"{settings.redis_state_latest_hash}.{mode}"

def generate_latest_signal_topic(mode: AllowedTradeMode | int) -> str:
    return settings.kafka_signal_latest_topic

def generate_latest_state_topic(mode: AllowedTradeMode | int) -> str:
    return settings.kafka_state_latest_topic

def generate_history_state_topic(mode: AllowedTradeMode | int) -> str:
    return settings.kafka_state_history_topic
