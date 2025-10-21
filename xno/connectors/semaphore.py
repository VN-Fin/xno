import logging
import time
import uuid
from xno.connectors.rd import RedisClient


class DistributedSemaphore:
    def __init__(
            self,
            ttl: int = 30,
            retry_interval: float = 0.1,
            timeout: int = 60,
            lock_key: str = "xno_data_semaphore",
            max_permits: int = 5,
    ):
        """
        Redis-based distributed semaphore (multiple concurrent holders).
        :param ttl: Time-to-live for each semaphore slot in seconds.
        :param retry_interval: Time in seconds to wait before retrying to acquire the semaphore.
        :param timeout: Maximum time in seconds to wait to acquire the semaphore.
        :param lock_key: Redis key to use for the semaphore.
        :param max_permits: Maximum number of concurrent holders of the semaphore.
        """
        self.key = lock_key
        self.max_leases = max_permits
        self.ttl = ttl
        self.retry_interval = retry_interval
        self.timeout = timeout
        self.value = str(uuid.uuid4())

    def acquire(self) -> bool:
        """
        Try to acquire one of the semaphore slots.
        """
        logging.debug(f'Acquiring semaphore {self.key}')
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            now = int(time.time() * 1000)
            pipeline = RedisClient.pipeline()

            # remove expired holders
            pipeline.zremrangebyscore(self.key, 0, now - self.ttl * 1000)
            # add self
            pipeline.zadd(self.key, {self.value: now})
            # get rank
            pipeline.zrank(self.key, self.value)
            # set expiry on the whole key (in case no one cleans)
            pipeline.expire(self.key, self.ttl)
            _, _, rank, _ = pipeline.execute()

            if rank is not None and rank < self.max_leases:
                return True

            # if failed, cleanup our token
            RedisClient.zrem(self.key, self.value)
            logging.warning(f"Semaphore full, retrying acquire for key {self.key} after {self.retry_interval}s")
            time.sleep(self.retry_interval)

        return False

    def release(self):
        """Release semaphore slot."""
        logging.debug(f"Releasing semaphore {self.key}")
        RedisClient.zrem(self.key, self.value)

    def __enter__(self):
        if not self.acquire():
            logging.error(f"Semaphore {self.key} not acquired.")
            raise TimeoutError(f"Could not acquire semaphore for key {self.key}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# USAGE
if __name__ == "__main__":
    def query_postgres():
        print("Running query...")
        time.sleep(2)

    # with DistributedSemaphore():
    #     query_postgres()
    #     print("Done, released slot")

    with RedisClient as session:
        logging.info("Setting a test value in Redis")
        session.set('test', 'value')

    with RedisClient as session:
        logging.info(session.get('test'))
