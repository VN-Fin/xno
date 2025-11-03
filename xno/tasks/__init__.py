from celery import Celery
from xno import settings

auth_part = f"{settings.redis_user}:{settings.redis_password}@" if settings.redis_password else ""
broker_url  = f"redis://{auth_part}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"  # queue
backend_url = f"redis://{auth_part}{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"  # task results

capp = Celery(
    broker=broker_url,
    backend=backend_url,
)
capp.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
