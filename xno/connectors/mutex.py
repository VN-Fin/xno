import logging
import redis
import time
import uuid

from xno.connectors.rd import RedisClient


class DistributedMutex:
    """
    Redis-based distributed mutex (single holder lock).
    """

    _UNLOCK_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """

    def __init__(
        self,
        key: str,
        ttl: int = 30,
        retry_interval: float = 0.1,
        timeout: int = 60,
    ):
        self.key = f"mutex:{key}"
        self.ttl = ttl
        self.retry_interval = retry_interval
        self.timeout = timeout
        self.value = str(uuid.uuid4())
        self._unlock_script_sha = None

    def lock(self) -> bool:
        """
        Attempt to acquire the lock within timeout period.
        Returns True if acquired, False otherwise.
        """
        logging.debug(f"Attempting to acquire lock {self.key}")
        start = time.time()
        ttl_ms = int(self.ttl * 1000)

        while time.time() - start < self.timeout:
            # SET NX PX => Only set if not exist, with expiration
            if RedisClient.set(self.key, self.value, nx=True, px=ttl_ms):
                logging.info(f"Lock acquired: {self.key}")
                return True

            logging.debug(f"Lock {self.key} is held, retrying...")
            time.sleep(self.retry_interval)

        logging.warning(f"Timeout acquiring lock {self.key}")
        return False

    def unlock(self):
        """Safely release the lock only if we are the holder."""
        if self._unlock_script_sha is None:
            self._unlock_script_sha = RedisClient.script_load(self._UNLOCK_SCRIPT)
        try:
            result = RedisClient.evalsha(
                self._unlock_script_sha,
                1,
                self.key,
                self.value,
            )
            if result == 1:
                logging.info(f"Lock released: {self.key}")
            else:
                logging.warning(f"Lock {self.key} not released (ownership mismatch)")
        except redis.exceptions.NoScriptError:
            # In case Redis restarted or script cache flushed
            RedisClient.eval(self._UNLOCK_SCRIPT, 1, self.key, self.value)

    def __enter__(self):
        if not self.lock():
            raise TimeoutError(f"Could not acquire lock {self.key}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlock()


# USAGE EXAMPLE
if __name__ == "__main__":
    def do_critical_work():
        print("Performing critical section...")
        time.sleep(3)

    lock = DistributedMutex(key="critical_section", ttl=10)
    with lock:
        do_critical_work()
        print("Lock released automatically.")

    is_locked = DistributedMutex(key="critical_section", ttl=10).lock()
