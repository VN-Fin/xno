from celery import Celery
from kombu import Queue

from xno import settings

auth_part = f"{settings.redis_user}:{settings.redis_password}@" if settings.redis_password else ""
broker_url  = f"redis://{auth_part}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"  # queue
backend_url = f"redis://{auth_part}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"  # task results


class CeleryQueueNames:
    BACKTEST = "backtest_queue"
    STRATEGY = "strategy_queue"

class CeleryTaskGroups:
    RUN_BACKTEST = "backtest_worker"
    RUN_STRATEGY = "strategy_worker"

capp = Celery(
    broker=broker_url,
    backend=backend_url,
)
capp.conf.task_queues = (
    Queue(CeleryQueueNames.BACKTEST),
    Queue(CeleryQueueNames.STRATEGY),
)
capp.conf.task_routes = {
    f"{CeleryTaskGroups.RUN_BACKTEST}.*": {"queue": CeleryQueueNames.BACKTEST},
    f"{CeleryTaskGroups.RUN_STRATEGY}.*": {"queue": CeleryQueueNames.STRATEGY},
}
capp.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
