import json
import logging
import threading
import time
import confluent_kafka
import pandas as pd
from confluent_kafka import Consumer
from tqdm import tqdm
from xno import settings
from xno.connectors.mem import DistributedSemaphore
from xno.connectors.sql import SqlSession
from typing import Literal, Dict
import queue
from readerwriterlock import rwlock


_accepted_resolutions = {"MIN", "HOUR1", "DAY"}
_accepted_data_type = "OH"  # Open-High-Low-Close-Volume
_accepted_data_source = "dnse"  # Data sources
_ohlcv_db = "xno_ai_data"
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
        self.buffer = queue.Queue()
        self.lock = rwlock.RWLockFair()
        self._stop_event = threading.Event()

        # Start background flusher thread
        self._threads = [
            threading.Thread(target=self._commit_buffer, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def append_data(self, data: dict):
        """Append incoming row (dict) to buffer safely."""
        with self.lock:
            self.buffer.put(data)

    def load_data(self, from_time: str, to_time: str):
        params = {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "from_time": from_time,
            "to_time": to_time,
        }
        with DistributedSemaphore():
            with SqlSession(_ohlcv_db) as session:
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
        with self.lock.gen_rlock():
            datas = self.data.copy()
        # Drop duplicate indices, keep the last
        datas = datas[~datas.index.duplicated(keep="last")]
        return datas

    def consume_buffer(self) -> bool:
        rows = []
        try:
            while True:
                rows.append(self.buffer.get_nowait())
        except queue.Empty:
            pass

        if not rows:
            return False
        df = pd.DataFrame(rows).dropna(axis=1, how="all")
        if df.empty:
            return False
        df.set_index("time", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        with self.lock.gen_wlock():
            self.data = pd.concat([self.data, df])
        return True

    def _commit_buffer(self):
        """Background thread that flushes buffer every 1s."""
        while not self._stop_event.is_set():
            time.sleep(consume_buffer_interval)
            self.consume_buffer()

    def stop(self):
        """Stop background thread gracefully."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2)


class OhlcvDataManager:
    _instances: Dict[tuple, OhlcvData] = {}
    _lock = rwlock.RWLockFair()

    @classmethod
    def stats(cls):
        with cls._lock.gen_rlock():
            return {k: v.get_data().shape[0] for k, v in cls._instances.items()}

    @classmethod
    def add(cls, resolution: str, symbol: str, payload: dict):
        key = (resolution, symbol)
        with cls._lock.gen_wlock():
            if key not in cls._instances:
                logging.info("Creating new OhlcvData instance for %s at %s", symbol, resolution)
                cls._instances[key] = OhlcvData(resolution, symbol)
            cls._instances[key].append_data(payload)

    @classmethod
    def get(cls, resolution: str, symbol: str) -> OhlcvData:
        key = (resolution, symbol)
        with cls._lock.gen_rlock():
            if key not in cls._instances:
                logging.info(f"Creating new OhlcvData instance for {symbol} at {resolution}")
                cls._instances[key] = OhlcvData(resolution, symbol)
            return cls._instances[key]

    @classmethod
    def _consume_realtime(cls):
        def latest_assign(consumer, partitions):
            for p in partitions:
                p.offset = confluent_kafka.OFFSET_END
            consumer.assign(partitions)

        consumer = Consumer(**{
            'bootstrap.servers': settings.kafka_bootstrap_servers,
            'enable.auto.commit': True,
            'group.id': 'xno-data-consumer-group',
            'auto.offset.reset': 'latest',
        })
        consumer.subscribe([settings.kafka_market_data_topic], on_assign=latest_assign)

        while True:
            m = consumer.poll(1)
            if m is None:
                continue
            if m.error():
                logging.error(f"Kafka error: {m.error()}")
                continue
            payload = json.loads(m.value())

            # Check data type and source
            data_type = payload.get('data_type')
            data_source = payload.get('source')
            if data_type != _accepted_data_type or data_source != _accepted_data_source:
                continue

            # Check resolution
            resolution = payload.get('resolution')
            if resolution not in _accepted_resolutions:
                continue

            # Add to corresponding OhlcvData instance
            cls.add(resolution, payload['symbol'], payload)
            logging.info(f'Received message [{datetime.datetime.fromtimestamp(payload['updated'])}]: {payload}')

    @classmethod
    def consume_realtime(cls):
        """Start consuming real-time data in background."""
        t = threading.Thread(target=cls._consume_realtime, daemon=True)
        t.start()
        return t


# --- Example usage ---
if __name__ == "__main__":
    import datetime, random

    OhlcvDataManager.consume_realtime()

    while True:
        time.sleep(10)
        print(OhlcvDataManager.stats())
