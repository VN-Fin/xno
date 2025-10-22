import abc
from abc import abstractmethod
from typing import List, Set
from xno.connectors.rd import RedisClient
from xno.stream import produce_message
from xno.trade import (
    AllowedTradeMode,
    StrategyState,
    StrategySignal,
    AllowedAction,
    StrategyConfigLoader,
)
import pandas as pd
import logging

import xno.utils.keys as ukeys

class StrategyRunner(abc.ABC):
    def __init__(
            self,
            strategy_id: str,
            mode: AllowedTradeMode,
            re_run: bool = False,
            send_signal: bool = False,
    ):
        self.redis_latest_signal_key = ukeys.generate_latest_signal_key(mode)
        self.redis_latest_state_key = ukeys.generate_latest_state_key(mode)
        self.kafka_latest_signal_topic = ukeys.generate_latest_signal_topic(mode)
        self.kafka_latest_state_topic = ukeys.generate_latest_state_topic(mode)
        self.kafka_history_state_topic = ukeys.generate_history_state_topic(mode)
        self.checkpoint_idx = 0
        self.send_signal = send_signal
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
        self.cfg = StrategyConfigLoader.get_config(self.strategy_id, self.mode, "")
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
        self.data_fields: Set[str] = set()
        if self.send_signal:
            produce_message(
                "ping",
                "run_strategy",
                f"Run strategy {self.strategy_id}",
            )

    def add_field(self, field: str):
        self.data_fields.add(field)
        return self

    @abstractmethod
    def __load_data__(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def __generate_signal__(self) -> List[float]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def __step__(self, time_idx: int):
        raise NotImplementedError("Subclasses should implement this method.")

    def __send_signal__(self):
        """
        [IF HAS CHANGED] Send the latest strategy signal to Kafka and Redis.
        :return:
        """
        prev_signal = RedisClient.hget(
            name=self.redis_latest_signal_key,
            key=self.strategy_id,
        )
        # Send signal
        current_signal = StrategySignal(
            strategy_id=self.current_state.strategy_id,
            symbol=self.current_state.symbol,
            symbol_type=self.current_state.symbol_type,
            candle=self.current_state.candle,
            current_price=self.current_state.current_price,
            current_weight=self.current_state.current_weight,
            current_action=self.current_state.current_action,
            bt_mode=self.current_state.bt_mode,
            engine=self.current_state.engine,
        )

        if prev_signal is not None:
            prev_signal = StrategySignal.model_validate_json(prev_signal)

        # Hash weight unchanged, skip sending
        if current_signal.current_weight == prev_signal.current_weight:
            logging.info(f"No change in signal for strategy_id={self.strategy_id}, skip sending.")
            return

        current_signal = current_signal.to_json_str()
        logging.debug(f"Sending signal {current_signal}")
        produce_message(
            self.kafka_latest_signal_topic,
            key=self.strategy_id,
            value=current_signal,
        )
        # Set to redis
        RedisClient.hset(
            name=self.redis_latest_signal_key,
            key=self.strategy_id,
            value=current_signal,
        )

    def __send_state__(self):
        """
        [ALWAYS] Send the latest strategy state to Kafka and Redis.
        :return:
        """
        current_state_str = self.current_state.to_json_str()
        logging.debug(f"Sending latest state {current_state_str}")
        produce_message(
            self.kafka_latest_state_topic,
            key=self.strategy_id,
            value=current_state_str,
        )
        # Set to redis
        RedisClient.hset(
            name=self.redis_latest_state_key,
            key=self.strategy_id,
            value=current_state_str,
        )

    def __done__(self):
        if not self.send_signal:
            logging.info(f"send_signal is False, skip sending records for strategy_id={self.strategy_id}")
            return # Skip sending
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
            produce_message(
                self.kafka_history_state_topic,
                key=self.strategy_id,
                value=record_str,
            )
        # Send latest state (current state)
        self.__send_state__()

    def run(self):
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

