import json
import logging
import threading
import time
import confluent_kafka
import numpy
import pandas as pd
from confluent_kafka import Consumer
from tqdm import tqdm
from xno import settings
from xno.connectors.mem import DistributedSemaphore
from xno.connectors.sql import SqlSession
from typing import Literal, Dict
import queue
from readerwriterlock import rwlock
from sqlalchemy import text
import datetime, random

_accepted_resolutions = {"h", "D", "m"}
# Data type, "OH" stands for Open-High-Low-Close-Volume
_accepted_data_type = "OH"
_accepted_data_source = "dnse"
# Database name for OHLCV data
_ohlcv_db = "xno_ai_data"

# Template DataFrame with predefined data types for consistency
_ohlcv_data_template = pd.DataFrame({
    "time": pd.Series([], dtype="datetime64[ns]"),
    "open": pd.Series([], dtype="float64"),
    "high": pd.Series([], dtype="float64"),
    "low": pd.Series([], dtype="float64"),
    "close": pd.Series([], dtype="float64"),
    "volume": pd.Series([], dtype="float64"),
}).set_index("time")

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
    "Single ticker load and process for ohclv data"
    def __init__(self, resolution: str, symbol: str):
        self.resolution = resolution
        self.symbol = symbol
        self.data = _ohlcv_data_template.copy()
        self.buffer = queue.Queue()
        self.lock = rwlock.RWLockFair()
        self._stop_event = threading.Event()
        self.consume_buffer_interval = random.randint(10, 300)  # seconds

        # Start background flusher thread
        self._threads = [
            threading.Thread(target=self._commit_buffer, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def append_data(self, data: dict):
        """Append incoming row (dict) to buffer safely."""
        with self.lock.gen_wlock():
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
                    text(load_data_query),
                    session.bind,
                    params=params,
                    chunksize=load_chunk_size,
                    # dtype=_ohlcv_data_template.dtypes,
                )
                # Iterate over each chunk and append it to the main DataFrame
                for chunk_df in tqdm(chunks):
                    logging.info("Yielding chunk of size %d", len(chunk_df))
                    if not chunk_df.empty:
                        chunk_df.set_index("time", inplace=True)
                        with self.lock.gen_wlock():
                            # Concatenate the new chunk with existing data
                            self.data = pd.concat([self.data, chunk_df])
                            # Remove duplicates by time and keep the last one (most recent)
                            self.data = self.data[~self.data.index.duplicated(keep="last")]
                            # Sort the data by index (time)
                            self.data.sort_index(inplace=True)

    def datas(self, resolution, from_time, to_time) -> pd.DataFrame:
        """
        Retrieves and prepares OHLCV data for the specified time range and resolution.
        It loads historical data only for missing time periods to optimize performance.
        """
        self.consume_buffer()

        # Convert time strings to datetime objects
        from_ts = pd.to_datetime(from_time)
        to_ts = pd.to_datetime(to_time)

        # Get the min/max index from the currently loaded data
        with self.lock.gen_rlock():
            if not self.data.empty:
                min_index = self.data.index.min()
                max_index = self.data.index.max()
            else:
                # If no data is loaded, set min/max index to None
                min_index = None
                max_index = None

        # Check and load data for missing gaps
        # Case 1: No data is loaded yet, so load the entire requested range
        if min_index is None:
            logging.info(f"Loading initial historical data for {self.symbol} from {from_time} to {to_time}")
            self.load_data(from_time, to_time)
        else:
            # Case 2: Load historical data for a past gap
            if from_ts < min_index:
                logging.info(f"Loading historical data for {self.symbol} from {from_time} to {min_index}")
                self.load_data(from_time, min_index.strftime('%Y-%m-%d %H:%M:%S'))

            # Case 3: Load historical data for a future gap
            if to_ts > max_index:
                logging.info(f"Loading historical data for {self.symbol} from {max_index} to {to_time}")
                self.load_data(max_index.strftime('%Y-%m-%d %H:%M:%S'), to_time)

        # After loading, get a read lock and filter the data
        with self.lock.gen_rlock():
            datas = self.data.copy()

        # Filter by the requested time range
        datas = datas[(datas.index >= from_ts) & (datas.index <= to_ts)]

        # Resample data if the requested resolution is different from the instance's resolution
        if resolution != self.resolution:
            logging.info(f"Resampling data for {self.symbol} from {self.resolution} to {resolution}")
            datas = datas.resample(resolution).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna()

        return datas

    def consume_buffer(self) -> bool:
        """
        Flushes real-time data from the buffer queue to the main DataFrame.
        """
        rows = []
        try:
            while True:
                rows.append(self.buffer.get_nowait())
        except queue.Empty:
            pass

        if not rows:
            return False
        df = pd.DataFrame(
            rows,
        ).dropna(
            axis=1,
            how="all"
        ).astype({
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        if df.empty:
            return False
        df.set_index("time", inplace=True)
        # Convert index to datetime.timestamp
        df.index = (
            pd.to_datetime(df.index, unit="s")
            .tz_localize("UTC")
            .tz_convert("Asia/Ho_Chi_Minh")
            .tz_localize(None)
        )
        df = df[~df.index.duplicated(keep="last")]
        with self.lock.gen_wlock():
            self.data = pd.concat([self.data, df])
        return True

    def _commit_buffer(self):
        """Background thread that flushes buffer every 1s."""
        while not self._stop_event.is_set():
            time.sleep(self.consume_buffer_interval)
            self.consume_buffer()

    def stop(self):
        """Stop background thread gracefully."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2)


class OhlcvDataManager:
    """
    Manages and provides OhlcvData instances for different symbols and resolutions.
    """
    _allowed_symbols = set()  # If empty, allow all symbols
    _instances: Dict[tuple, OhlcvData] = {}
    _lock = rwlock.RWLockFair()

    @classmethod
    def add_symbol(cls, symbol: str):
        cls._allowed_symbols.add(symbol)
        return cls

    @classmethod
    def stats(cls):
        with cls._lock.gen_rlock():
            return {
                "total_instances": len(cls._instances),
                # "total_records": sum(len(instance.datas()) for instance in cls._instances.values())
            }

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
        """
        Background worker to consume real-time data from Kafka.
        """

        def latest_assign(consumer, partitions):
            for p in partitions:
                p.offset = confluent_kafka.OFFSET_END
            consumer.assign(partitions)

        consumer = Consumer(**{
            'bootstrap.servers': settings.kafka_bootstrap_servers,
            'enable.auto.commit': False,
            'group.id': 'xno-data-consumer-group',
            'auto.offset.reset': 'latest',
        })
        consumer.subscribe([settings.kafka_market_data_topic], on_assign=latest_assign)
        logging.info("Started Kafka consumer for real-time OHLCV data.")
        while True:
            m = consumer.poll(1)
            if m is None:
                continue
            if m.error():
                logging.error(f"Kafka error: {m.error()}")
                continue
            payload = json.loads(m.value())

            symbol = payload.get('symbol')
            if cls._allowed_symbols and symbol not in cls._allowed_symbols:
                continue

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
            new_payload = {
                "time": payload.get('time'),
                "open": payload.get('open'),
                "high": payload.get('high'),
                "low": payload.get('low'),
                "close": payload.get('close'),
                "volume": payload.get('volume'),
            }
            cls.add(resolution, payload['symbol'], new_payload)
            logging.debug(f'Received message [{datetime.datetime.fromtimestamp(payload['updated'])}]: {payload}')

    @classmethod
    def consume_realtime(cls):
        """Start consuming real-time data in background."""
        t = threading.Thread(target=cls._consume_realtime, daemon=True)
        t.start()
        return t


# --- Example usage ---
if __name__ == "__main__":

    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )
    # OhlcvDataManager.add_symbol("HPG")  # .add_symbol("SSI").add_symbol("VND")
    OhlcvDataManager.consume_realtime()

    while True:
        time.sleep(10)
        print(OhlcvDataManager.stats())
        datas = OhlcvDataManager.get("D", "HPG").datas(resolution="D", from_time="2020-09-15", to_time="2026-10-10")
        print(datas)
        print(datas.dtypes)