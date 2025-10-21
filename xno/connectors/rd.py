import redis
import logging
from xno import settings


RedisClient = redis.StrictRedis(
    **settings.redis_config
)
# Test connection
try:
    RedisClient.ping()
    logging.info("Connected to Redis successfully.")
except redis.ConnectionError as e:
    logging.error(f"Failed to connect to Redis: {e}")
    raise RuntimeError("Failed to connect to Redis.")
