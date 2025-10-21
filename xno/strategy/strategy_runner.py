import abc
from abc import abstractmethod
from typing import List
from xno.connectors.rd import RedisClient
from xno.trade import (
    AllowedTradeMode,
    StrategyState,
    StrategySignal,
    AllowedAction,
    StrategyConfigLoader,
)
import pandas as pd
import logging


class StrategyRunner(abc.ABC):
    def __init__(
            self,
            strategy_id: str,
            mode: AllowedTradeMode,
            re_run: bool = False,
            send_signal: bool = False,
    ):
        self.latest_signal_key = "strategy_latest_signal:" + strategy_id
        self.latest_state_key = "strategy_latest_state:" + strategy_id
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
        if self.send_signal:
            produce_message(
                "ping",
                "run_strategy",
                f"Run strategy {self.strategy_id}",
            )

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
        prev_signal = RedisClient.get(self.latest_signal_key)
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

        if current_signal == prev_signal:
            logging.info(f"No change in signal for strategy_id={self.strategy_id}, skip sending.")
            return

        current_signal = current_signal.to_json_str()
        if current_signal != "":
            logging.debug(f"Sending signal {current_signal}")
            produce_message(
                TAExpressionConfig.kafka_signal_latest_topic,
                key=self.strategy_id,
                value=current_signal,
            )
            # Set to redis
            RedisClient.set(self.latest_signal_key, current_signal)
        else:
            logging.warning(f"Invalid signal for strategy_id={self.strategy_id}, skip sending.")

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
                TAExpressionConfig.kafka_state_history_topic,
                key=self.strategy_id,
                value=record_str,
            )
        # Send latest state (current state)
        produce_message(
            TAExpressionConfig.kafka_state_latest_topic,
            key=self.strategy_id,
            value=self.current_state.to_json_str()
        )

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

