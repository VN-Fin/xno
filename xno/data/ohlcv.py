import logging
import threading
import time

import pandas as pd
from tqdm import tqdm

from xno.connectors import DistributedSemaphore
from xno.connectors.postgresql import SqlSession

# Template DataFrame
_ohlcv_data_template = pd.DataFrame({
    "time": pd.Series([], dtype="datetime64[ns]"),
    "open": pd.Series([], dtype="float32"),
    "high": pd.Series([], dtype="float32"),
    "low": pd.Series([], dtype="float32"),
    "close": pd.Series([], dtype="float32"),
    "volume": pd.Series([], dtype="float32"),
}).set_index("time")

consume_buffer_interval = 20 # seconds
load_chunk_size = 1000  # rows
load_data_query = """
        SELECT time, open, high, low, close, volume
        FROM trading.stock_ohlcv_history
        WHERE symbol = :symbol
          AND resolution = :resolution
          AND time >= :from_time
          AND time <= :to_time
        """


class OhlcvData:
    def __init__(self, resolution: str, symbol: str):
        self.resolution = resolution
        self.symbol = symbol
        self.data = _ohlcv_data_template.copy()
        self.buffer = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

        # Start background flusher thread
        self._threads = [
            threading.Thread(target=self._commit_buffer, daemon=True),
            threading.Thread(target=self._consume_data, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def append_data(self, data: dict):
        """Append incoming row (dict) to buffer safely."""
        with self.lock:
            self.buffer.append(data)

    def load_data(self, from_time: str, to_time: str):
        params = {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "from_time": from_time,
            "to_time": to_time,
        }
        with DistributedSemaphore(ttl=600):
            with SqlSession() as session:
                chunks = pd.read_sql_query(
                    load_data_query,
                    session.bind,
                    params=params,
                    chunksize=load_chunk_size,
                    dtype=_ohlcv_data_template.dtype,
                )
                for chunk_df in tqdm(chunks):
                    logging.info("Yielding chunk of size %d", len(chunk_df))
                    if not chunk_df.empty:
                        chunk_df.set_index("time", inplace=True)
                        with self.lock:
                            self.data = pd.concat([self.data, chunk_df])
                            # Drop duplicates by time
                            self.data = self.data[~self.data.index.duplicated(keep="last")]
                            self.data.sort_index(inplace=True)

    def get_data(self) -> pd.DataFrame:
        """Return a copy of the current DataFrame."""
        with self.lock:
            return self.data.copy()

    def _consume_data(self):
        raise NotImplementedError()

    def _commit_buffer(self):
        """Background thread that flushes buffer every 1s."""
        while not self._stop_event.is_set():
            time.sleep(consume_buffer_interval)
            with self.lock:
                if self.buffer:
                    df = pd.DataFrame(self.buffer)
                    self.buffer = []
                    if not df.empty:
                        df.set_index("time", inplace=True)
                        # Merge new data
                        self.data = pd.concat([self.data, df])
                        # Drop duplicates by time
                        self.data = self.data[~self.data.index.duplicated(keep="last")]
                        self.data.sort_index(inplace=True)

    def stop(self):
        """Stop background thread gracefully."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2)


class OhlcvDataManager:
    _instances = {}
    _lock = threading.Lock()
    @classmethod
    def get_instance(cls, resolution: str, symbol: str) -> OhlcvData:
        key = (resolution, symbol)
        with cls._lock:
            if key not in cls._instances:
                logging.info(f"Creating new OhlcvData instance for {symbol} at {resolution}")
                cls._instances[key] = OhlcvData(resolution, symbol)
            return cls._instances[resolution][symbol]

# --- Example usage ---
if __name__ == "__main__":
    import datetime, random

    acb_m = OhlcvData("1m", "ACB")  # same instance reused if called again

    for i in range(5):
        acb_m.append_data({
            "time": datetime.datetime.now(),
            "open": random.random() * 100,
            "high": random.random() * 100,
            "low": random.random() * 100,
            "close": random.random() * 100,
            "volume": random.randint(100, 1000),
        })
        time.sleep(0.2)

    time.sleep(2)  # wait for background flush
    print(acb_m.get_data().tail())
    acb_m.stop()
