from celery import Celery
from kombu import Queue

from xno import settings

auth_part = f"{settings.redis_user}:{settings.redis_password}@" if settings.redis_password else ""
broker_url  = f"redis://{auth_part}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"  # queue
backend_url = f"redis://{auth_part}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"  # task results


class CeleryQueueNames:
    BACKTEST = "backtest_queue"
    TA_STRATEGY = "ta_strategy_queue"
    AI_STRATEGY = "ai_strategy_queue"
    QUANT_STRATEGY = "quant_strategy_queue"

class CeleryTaskGroups:
    BACKTEST = "backtest_worker"
    TA_STRATEGY = "ta_strategy_worker"
    AI_STRATEGY = "ai_strategy_worker"
    QUANT_STRATEGY = "quant_strategy_worker"

capp = Celery(
    broker=broker_url,
    backend=backend_url,
)
capp.conf.task_queues = (
    Queue(CeleryQueueNames.BACKTEST),
    Queue(CeleryQueueNames.TA_STRATEGY),
    Queue(CeleryQueueNames.AI_STRATEGY),
    Queue(CeleryQueueNames.QUANT_STRATEGY),
)
capp.conf.task_routes = {
    f"{CeleryTaskGroups.BACKTEST}.*": {"queue": CeleryQueueNames.BACKTEST},
    f"{CeleryTaskGroups.TA_STRATEGY}.*": {"queue": CeleryQueueNames.TA_STRATEGY},
    f"{CeleryTaskGroups.AI_STRATEGY}.*": {"queue": CeleryQueueNames.AI_STRATEGY},
    f"{CeleryTaskGroups.QUANT_STRATEGY}.*": {"queue": CeleryQueueNames.QUANT_STRATEGY},
}
capp.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
