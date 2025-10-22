from abc import abstractmethod, ABC
from typing import List, Dict

from confluent_kafka import Producer

from xno import settings
from xno.connectors.rd import RedisClient
from xno.trade import (
    AllowedTradeMode,
    StrategyState,
    StrategySignal,
    AllowedAction,
    StrategyConfigLoader,
    FieldInfo,
)
import pandas as pd
import logging
import xno.utils.keys as ukeys
import numpy as np

from xno.utils.datas import load_data
from xno.utils.stream import delivery_report
import threading


_local = threading.local()

def get_producer():
    if not hasattr(_local, "producer"):
        _local.producer = Producer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
        })
    return _local.producer


class StrategyRunner(ABC):
    """
    The base class for running a trading strategy.
    """
    def __init__(
            self,
            strategy_id: str,
            mode: AllowedTradeMode,
            re_run: bool = False,
    ):
        self.producer = get_producer()
        self.redis_latest_signal_key = ukeys.generate_latest_signal_key(mode)
        self.redis_latest_state_key = ukeys.generate_latest_state_key(mode)
        self.kafka_latest_signal_topic = ukeys.generate_latest_signal_topic(mode)
        self.kafka_latest_state_topic = ukeys.generate_latest_state_topic(mode)
        self.kafka_history_state_topic = ukeys.generate_history_state_topic(mode)
        self.checkpoint_idx = 0
        self.strategy_id = strategy_id
        self.mode = mode
        self.re_run = re_run
        self.symbol = ""
        self.symbol_type = ""
        self.timeframe = ""
        self.init_cash = 0.0
        self.run_engine: str = ""
        self.run_from: pd.Timestamp = pd.Timestamp.now()
        self.run_to: pd.Timestamp = pd.Timestamp.now()
        self.datas: pd.DataFrame = pd.DataFrame()
        self.cfg = StrategyConfigLoader.get_config(self.strategy_id, self.mode)
        if self.cfg is None:
            raise RuntimeError(f"Strategy config not found for strategy_id={self.strategy_id} and mode={self.mode}")
        else:
            self.run_to = pd.Timestamp(self.cfg.run_to)
            self.run_from = pd.Timestamp(self.cfg.run_from)
            self.symbol = self.cfg.symbol
            self.timeframe = self.cfg.timeframe
            self.init_cash = self.cfg.init_cash
            self.run_engine = self.cfg.engine
            self.symbol_type = self.cfg.symbol_type

        self.current_state: StrategyState | None = None
        self.trading_states: List[StrategyState] | None = None
        self.pending_sell_pos = 0.0
        self.current_time = None
        self.signals: List[float] | None = None
        self.ht_prices: List[float] | None = None
        self.ht_times: List[pd.Timestamp] | None = None
        self.data_fields: Dict[str, FieldInfo] = {}

    def add_field(self, field_id: str, field_name: str, ticker: str | None = None):
        """
        Add a data field to be loaded.
        :param field_id:
        :param ticker:
        :param field_name:
        :return:
        """
        if field_id == "Close":
            logging.debug("Field 'Close' is always included; skip adding again.")
            return self
        # Add field info if not exists or override existing
        self.data_fields[field_id] = FieldInfo(
            field_id=field_id,
            field_name=field_name,
            ticker=ticker or self.symbol,
        )
        return self

    def __setup__(self):
        default_field = FieldInfo(field_id="Close", field_name="Close", ticker=self.symbol)
        self.data_fields["Close"] = default_field

    def __load_data__(self):
        # TODO: Load additional fields
        self.datas = load_data(
            resolution=self.timeframe,
            symbol=self.symbol,
            start=self.run_from,
            factor=1000,
        )

    @abstractmethod
    def __generate_signal__(self) -> List[float]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def __step__(self, time_idx: int):
        raise NotImplementedError("Subclasses should implement this method.")

    def __send_signal__(self):
        """
        Send the latest strategy signal to Kafka and Redis
        only if the signal has changed.
        """
        state = self.current_state
        redis_key = self.redis_latest_signal_key
        strategy_id = self.strategy_id
        # Retrieve previous signal JSON (bytes or str)
        prev_raw = RedisClient.hget(name=redis_key, key=strategy_id)
        # Build current signal model
        current_signal = StrategySignal(
            strategy_id=state.strategy_id,
            symbol=state.symbol,
            symbol_type=state.symbol_type,
            candle=state.candle,
            current_price=state.current_price,
            current_weight=state.current_weight,
            current_action=state.current_action,
            bt_mode=state.bt_mode,
            engine=state.engine,
        )
        # If previous exists, decode; otherwise force send
        if prev_raw:
            try:
                prev_signal = StrategySignal.model_validate_json(prev_raw)
            except Exception as e:
                logging.warning(f"Invalid prev signal for {strategy_id}: {e}, resending current.")
                prev_signal = None
        else:
            prev_signal = None
        # Compare relevant fields (weight + action + price)
        if (
                prev_signal
                and current_signal.current_weight == prev_signal.current_weight
                and current_signal.current_action == prev_signal.current_action
                and abs(current_signal.current_price - prev_signal.current_price) < 1e-9
        ):
            # Skip logging each time; use debug for quiet mode
            logging.debug(f"No signal change for {strategy_id}, skip sending.")
            return

        # Serialize once
        signal_json = current_signal.to_json_str()
        # Send to Kafka
        self.producer.produce(
            self.kafka_latest_signal_topic,
            key=strategy_id,
            value=signal_json,
            callback=delivery_report
        )
        # Cache to Redis
        RedisClient.hset(
            name=redis_key,
            key=strategy_id,
            value=signal_json,
        )
        logging.info(f"Signal sent for {strategy_id}: {signal_json}")

    def __send_state__(self):
        """
        [ALWAYS] Send the latest strategy state to Kafka and Redis.
        :return:
        """
        current_state_str = self.current_state.to_json_str()
        logging.debug(f"Sending latest state {current_state_str}")
        self.producer.produce(
            self.kafka_latest_state_topic,
            key=self.strategy_id,
            value=current_state_str,
            callback=delivery_report
        )
        # Set to redis
        RedisClient.hset(
            name=self.redis_latest_state_key,
            key=self.strategy_id,
            value=current_state_str,
        )

    def __done__(self):
        send_from_cp = self.checkpoint_idx
        if self.re_run:
            logging.info(f"Re-run mode, send all existing records for strategy_id={self.strategy_id}")
            send_from_cp = 0
        else:
            logging.info(f"Insert new records for strategy_id={self.strategy_id}")
        send_records = self.trading_states[send_from_cp:]
        if len(send_records) == 0:
            logging.info(f"No new records to send for strategy_id={self.strategy_id}")
            return
        # Send signal [Optional]
        if self.current_state.bt_mode == AllowedTradeMode.LiveTrade:
            self.__send_signal__()

        # And send to Kafka
        send_states = self.trading_states[send_from_cp:]
        if len(send_states) == 0:
            logging.warning(f"No new records to send for strategy_id={self.strategy_id}")
            return
        logging.info(f"Sending {len(send_states)} trading state records for strategy_id={self.strategy_id}")
        for record in self.trading_states[send_from_cp:]:
            record_str = record.to_json_str()
            self.producer.produce(
                self.kafka_history_state_topic,
                key=self.strategy_id,
                value=record_str,
                callback=delivery_report
            )
        # Send latest state (current state)
        self.__send_state__()
        self.producer.flush()

    def run(self):
        # Setup fields
        self.__setup__()
        # Initial strategy run ping
        self.producer.produce(
            "ping",
            key="run_strategy",
            value=f"Run strategy {self.strategy_id}",
            callback=delivery_report
        )
        self.producer.flush() # Ensure ping is sent before proceeding
        # Load data
        self.__load_data__()
        if len(self.datas) == 0:
            raise RuntimeError(f"No data loaded for symbol={self.symbol} from {self.run_from}")

        self.ht_prices = self.datas['Close'].tolist()
        self.ht_times = self.datas.index.tolist()
        # init the current state
        self.current_state = StrategyState(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            symbol_type=self.symbol_type,
            candle=self.ht_times[0],
            run_from=self.run_from,
            run_to=self.run_to,
            current_price=0.0,
            current_position=0.0,
            current_weight=0.0,
            current_action=AllowedAction.Hold,
            trade_size=0.0,
            bt_mode=self.mode,
            t0_size=0.0,
            t1_size=0.0,
            t2_size=0.0,
            sell_size=0.0,
            pending_sell_weight=0.0,
            re_run=self.re_run,
            engine=self.run_engine,
            book_size=self.init_cash,
        )  # Init the start state
        self.trading_states = []  # And init the historical
        # Check if has run before
        logging.info(f"Loaded {len(self.datas)} rows of data for symbol={self.symbol}")
        # Execute the expression to get signals
        self.signals = self.__generate_signal__()
        if isinstance(self.signals, (np.ndarray, pd.Series)):
            self.signals = self.signals.tolist()
        # Check length
        if len(self.signals) != len(self.ht_prices):
            raise RuntimeError(f"Signal length {len(self.signals)} != price length {len(self.ht_prices)}")

        # Step through each signal (buy/sell/hold) and simulate trading
        for time_idx in range(len(self.signals)):
            self.__step__(time_idx)
        # Done, send to Kafka or save to DB
        self.__done__()

    def stats(self):
        return {
            "total_trades": len([s for s in self.trading_states if s.current_action != AllowedAction.Hold]),
            "final_position": self.current_state.current_position,
            "final_weight": self.current_state.current_weight,
            "final_price": self.current_state.current_price,
            "final_time": self.current_state.candle,
        }

    def continue_run(self):
        """
        Continue running the strategy from the last checkpoint or state.
        :return:
        """
        raise NotImplementedError("Subclasses should implement this method.")