from xno.connectors.mutex import DistributedMutex
from xno.connectors.rd import RedisClient
from xno.connectors.semaphore import DistributedSemaphore
import logging


logging.warning("Deprecated module xno.connectors.mem imported. Please use `xno.connectors.mutex`, `xno.connectors.semaphore` or `xno.connectors.lock` instead.")


if __name__ == "__main__":
    import time
    def query_postgres():
        print("Running query...")
        time.sleep(2)

    with DistributedSemaphore():
        query_postgres()
        print("Done, released slot")

    if DistributedMutex("my_mutex").lock():
        try:
            print("Mutex acquired, doing work...")
            time.sleep(2)
        finally:
            DistributedMutex("my_mutex").unlock()
            print("Mutex released.")