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
import requests

_accepted_resolutions = {"h", "D", "m"}
_accepted_data_type = "OH"
_accepted_data_source = "dnse"
_ohlcv_db = "xno_ai_data"

# API endpoints
BALANCE_SHEET_API = "https://api.xno.vn/v2/balancesheet?symbol={symbol}&yearly=0"
INCOME_STATEMENT_API = "https://api.xno.vn/v2/incomestatement?symbol={symbol}&yearly=0"
CASHFLOW_API = "https://api.xno.vn/v2/cashflow?symbol={symbol}&yearly=0"

# Template DataFrame
_bctc_data_template = pd.DataFrame({
    "time": pd.Series([], dtype="datetime64[ns]"),
    "open": pd.Series([], dtype="float64"),
    "high": pd.Series([], dtype="float64"),
    "low": pd.Series([], dtype="float64"),
    "close": pd.Series([], dtype="float64"),
    "volume": pd.Series([], dtype="float64"),

    "shortAsset": pd.Series([], dtype="float64"),
    "cash": pd.Series([], dtype="float64"),
    "shortInvest": pd.Series([], dtype="float64"),
    "shortReceivable": pd.Series([], dtype="float64"),
    "inventory": pd.Series([], dtype="float64"),
    "longAsset": pd.Series([], dtype="float64"),
    "fixedAsset": pd.Series([], dtype="float64"),
    "asset": pd.Series([], dtype="float64"),
    "debt": pd.Series([], dtype="float64"),
    "shortDebt": pd.Series([], dtype="float64"),
    "longDebt": pd.Series([], dtype="float64"),
    "equity": pd.Series([], dtype="float64"),
    "capital": pd.Series([], dtype="float64"),
    "otherDebt": pd.Series([], dtype="float64"),
    "minorShareHolderProfit": pd.Series([], dtype="float64"),
    "payable": pd.Series([], dtype="float64"),

    "revenue": pd.Series([], dtype="float64"),
    "yearRevenueGrowth": pd.Series([], dtype="float64"),
    "quarterRevenueGrowth": pd.Series([], dtype="float64"),
    "costOfGoodSold": pd.Series([], dtype="float64"),
    "grossProfit": pd.Series([], dtype="float64"),
    "operationExpense": pd.Series([], dtype="float64"),
    "operationProfit": pd.Series([], dtype="float64"),
    "interestExpense": pd.Series([], dtype="float64"),
    "preTaxProfit": pd.Series([], dtype="float64"),
    "postTaxProfit": pd.Series([], dtype="float64"),
    "shareHolderIncome": pd.Series([], dtype="float64"),
    "ebitda": pd.Series([], dtype="float64"),

    "investCost": pd.Series([], dtype="float64"),
    "fromInvest": pd.Series([], dtype="float64"),
    "fromFinancial": pd.Series([], dtype="float64"),
    "fromSale": pd.Series([], dtype="float64"),
    "freeCashFlow": pd.Series([], dtype="float64"),
}).set_index("time")

load_chunk_size = 1000
load_data_query = """
        SELECT time, open, high, low, close, volume
        FROM trading.stock_ohlcv_history
        WHERE symbol = :symbol
          AND resolution = :resolution
          AND time >= :from_time
          AND time <= :to_time
"""

class BCTCData:
    def __init__(self, resolution: str, symbol: str):
        self.resolution = resolution
        self.symbol = symbol
        self.data = _bctc_data_template.copy()
        self.buffer = queue.Queue()
        self.lock = rwlock.RWLockFair()
        self._stop_event = threading.Event()
        self.consume_buffer_interval = random.randint(10, 300)

        self._threads = [
            threading.Thread(target=self._commit_buffer, daemon=True),
            threading.Thread(target=self._fetch_financial_data, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def get_max_data_time(self):
        query = """
        SELECT MAX(time) as max_time
        FROM trading.stock_ohlcv_history
        WHERE symbol = :symbol
            AND resolution = :resolution
        """
        params = {
            "symbol": self.symbol,
            "resolution": self.resolution,
        }
        with DistributedSemaphore():
            with SqlSession(_ohlcv_db) as session:
                result = session.execute(text(query), params)
                max_time = result.scalar()
                return max_time

    def append_data(self, data: dict):
        with self.lock.gen_wlock():
            self.buffer.put(data)

    def _fetch_financial_data(self):
        """Fetch financial data from APIs periodically."""
        while not self._stop_event.is_set():
            try:

                balance_response = requests.get(BALANCE_SHEET_API.format(symbol=self.symbol))
                balance_data = balance_response.json().get('data', []) if balance_response.status_code == 200 else []

                income_response = requests.get(INCOME_STATEMENT_API.format(symbol=self.symbol))
                income_data = income_response.json().get('data', []) if income_response.status_code == 200 else []

                cashflow_response = requests.get(CASHFLOW_API.format(symbol=self.symbol))
                cashflow_data = cashflow_response.json().get('data', []) if cashflow_response.status_code == 200 else []

                financial_rows = []
                for b, i, c in zip(balance_data, income_data, cashflow_data):
                    if b['quarter'] == i['quarter'] == c['quarter'] and b['year'] == i['year'] == c['year']:

                        quarter_end = pd.to_datetime(f"{b['year']}-{(b['quarter'] * 3)}-01").to_period('Q').to_timestamp('Q')
                        row = {
                            "time": quarter_end,
                            **{k: v for k, v in b.items() if k in _bctc_data_template.columns and k != 'time'},
                            **{k: v for k, v in i.items() if k in _bctc_data_template.columns and k != 'time'},
                            **{k: v for k, v in c.items() if k in _bctc_data_template.columns and k != 'time'},
                        }
                        financial_rows.append(row)

                if financial_rows:
                    with self.lock.gen_wlock():
                        financial_df = pd.DataFrame(financial_rows).set_index("time")
                        financial_df = financial_df[~financial_df.index.duplicated(keep="last")]
                        self.data = pd.concat([self.data, financial_df]).sort_index()

                        self.data = self.data.ffill()
            except Exception as e:
                logging.error(f"Error fetching financial data: {e}")
            time.sleep(3600)

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
                )
                for chunk_df in tqdm(chunks):
                    logging.debug("Yielding chunk of size %d", len(chunk_df))
                    if not chunk_df.empty:
                        chunk_df.set_index("time", inplace=True)
                        with self.lock.gen_wlock():
                            self.data = pd.concat([self.data, chunk_df])
                            self.data = self.data[~self.data.index.duplicated(keep="last")]
                            self.data.sort_index(inplace=True)
                            # Forward fill financial data after loading OHLCV
                            self.data = self.data.ffill()

    def datas(self, from_time, to_time) -> pd.DataFrame:
        self.consume_buffer()
        with self.lock.gen_rlock():
            if not self.data.empty:
                min_index = self.data.index.min()
                max_index = self.data.index.max()
            else:
                min_index = None
                max_index = None

        if min_index is None:
            logging.debug(f"Loading initial historical data for {self.symbol} from {from_time} to {to_time}")
            self.load_data(from_time, to_time)
        else:
            if from_time < min_index:
                logging.debug(f"Loading historical data for {self.symbol} from {from_time} to {min_index}")
                self.load_data(from_time, min_index.strftime('%Y-%m-%d %H:%M:%S'))
            if to_time > max_index:
                logging.debug(f"Loading historical data for {self.symbol} from {max_index} to {to_time}")
                self.load_data(max_index.strftime('%Y-%m-%d %H:%M:%S'), to_time)

        with self.lock.gen_rlock():
            datas = self.data.copy()

        datas = datas[(datas.index >= from_time) & (datas.index <= to_time)]
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
        df = pd.DataFrame(rows).dropna(axis=1, how="all").astype({
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        if df.empty:
            return False
        df.set_index("time", inplace=True)
        df.index = (
            pd.to_datetime(df.index, unit="s")
            .tz_localize("UTC")
            .tz_convert("Asia/Ho_Chi_Minh")
            .tz_localize(None)
        )
        df = df[~df.index.duplicated(keep="last")]
        with self.lock.gen_wlock():
            self.data = pd.concat([self.data, df])
            self.data = self.data.ffill()
        return True

    def _commit_buffer(self):
        while not self._stop_event.is_set():
            time.sleep(self.consume_buffer_interval)
            self.consume_buffer()

    def stop(self):
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2)


class BCTCDataManager:
    _allowed_symbols = set()
    _instances: Dict[tuple, BCTCData] = {}
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
            }

    @classmethod
    def add(cls, resolution: str, symbol: str, payload: dict):
        key = (resolution, symbol)
        with cls._lock.gen_wlock():
            if key not in cls._instances:
                logging.debug("Creating new BCTCData instance for %s at %s", symbol, resolution)
                cls._instances[key] = BCTCData(resolution, symbol)
            cls._instances[key].append_data(payload)

    @classmethod
    def get(cls, resolution: str, symbol: str, from_time=None, to_time=None, factor=1) -> pd.DataFrame:
        resolution = resolution.lower()
        if "d" in resolution:
            query_res = "D"
            days = 30
        elif "h" in resolution:
            query_res = "h"
            days = 7
        elif "min" in resolution:
            query_res = "m"
            days = 1
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")

        key = (query_res, symbol)
        with cls._lock.gen_rlock():
            if key not in cls._instances:
                logging.debug(f"Creating new BCTCData instance for {symbol} at {resolution}")
                cls._instances[key] = BCTCData(query_res, symbol)
            instance = cls._instances[key]
        if to_time is None:
            to_time = instance.get_max_data_time()
        to_time = pd.to_datetime(to_time)

        if from_time is None:
            from_time = to_time - datetime.timedelta(days=days)
        from_time = pd.to_datetime(from_time)
        df = instance.datas(from_time, to_time)
        # Resample to requested resolution
        df = df.resample(resolution).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'shortAsset': 'last',
            'cash': 'last',
            'shortInvest': 'last',
            'shortReceivable': 'last',
            'inventory': 'last',
            'longAsset': 'last',
            'fixedAsset': 'last',
            'asset': 'last',
            'debt': 'last',
            'shortDebt': 'last',
            'longDebt': 'last',
            'equity': 'last',
            'capital': 'last',
            'otherDebt': 'last',
            'minorShareHolderProfit': 'last',
            'payable': 'last',
            'revenue': 'last',
            'yearRevenueGrowth': 'last',
            'quarterRevenueGrowth': 'last',
            'costOfGoodSold': 'last',
            'grossProfit': 'last',
            'operationExpense': 'last',
            'operationProfit': 'last',
            'interestExpense': 'last',
            'preTaxProfit': 'last',
            'postTaxProfit': 'last',
            'shareHolderIncome': 'last',
            'ebitda': 'last',
            'investCost': 'last',
            'fromInvest': 'last',
            'fromFinancial': 'last',
            'fromSale': 'last',
            'freeCashFlow': 'last',
        }).dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        df = df.ffill()
        if factor != 1:
            df['open'] = df['open'] * factor
            df['high'] = df['high'] * factor
            df['low'] = df['low'] * factor
            df['close'] = df['close'] * factor
        df = df.rename(
            columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            },
        )
        return df

    @classmethod
    def _consume_realtime(cls):
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

            data_type = payload.get('data_type')
            data_source = payload.get('source')
            if data_type != _accepted_data_type or data_source != _accepted_data_source:
                continue

            resolution = payload.get('resolution')
            if resolution not in _accepted_resolutions:
                continue
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
        t = threading.Thread(target=cls._consume_realtime, daemon=True)
        t.start()
        return t

# Test
if __name__ == "__main__":
    BCTCDataManager.consume_realtime()
    while True:
        time.sleep(10)
        print(BCTCDataManager.stats())
        datas = BCTCDataManager.get("d", "HPG", factor=1000)
        print(datas)
        print("----------------------\n")
        print(datas.iloc[-1])
        print("----------------------\n")

        print(datas.dtypes)