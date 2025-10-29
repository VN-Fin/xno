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

from xno.utils.stream import delivery_report
from xno.data.all_data_final import AllData
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
            mode: AllowedTradeMode | int,
            re_run: bool,
            send_data: bool,
    ):
        """
        Initialize the StrategyRunner.
        :param strategy_id:
        :param mode:
        :param re_run: re-run or continue from last checkpoint. If re-run, all data will be sent again.
        :param send_data: whether to send data to Kafka/Redis
        """
        self.send_data = send_data
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
        # Add field info if not exists or override existing
        self.data_fields[field_id] = FieldInfo(
            field_id=field_id,
            field_name=field_name,
            ticker=ticker or self.symbol,
        )
        return self

    def __setup__(self):
        # Add default Close field for the main symbol with ticker suffix
        self.data_fields["Open"] = FieldInfo(field_id="Open", field_name="Open", ticker=self.symbol)
        self.data_fields["High"] = FieldInfo(field_id="High", field_name="High", ticker=self.symbol)
        self.data_fields["Low"] = FieldInfo(field_id="Low", field_name="Low", ticker=self.symbol)
        self.data_fields["Close"] = FieldInfo(field_id="Close", field_name="Close", ticker=self.symbol)
        self.data_fields["Volume"] = FieldInfo(field_id="Volume", field_name="Volume", ticker=self.symbol)

    def __load_data__(self):
        """
        Load data for all added fields using AllData.
        Groups fields by ticker and loads data for each ticker separately.
        Then merges all data into a single DataFrame with renamed columns.
        """
        # Group fields by ticker
        ticker_fields: Dict[str, List[FieldInfo]] = {}
        for field_id, field_info in self.data_fields.items():
            ticker = field_info.ticker
            if ticker not in ticker_fields:
                ticker_fields[ticker] = []
            ticker_fields[ticker].append(field_info)

        # Load data for each ticker
        all_dataframes = []

        for ticker, fields in ticker_fields.items():
            # Create AllData instance and add all fields for this ticker
            all_data = AllData()

            for field_info in fields:
                all_data.add_field(field_info.field_name)

            # Load data for this ticker
            try:
                ticker_df = all_data.get(
                    resolution=self.timeframe,
                    symbol=ticker,
                    period='quarter'
                )

                # Rename columns to include field_id
                # For each field, find the corresponding field_id and rename the column
                rename_map = {}
                for field_info in fields:
                    # The column name in ticker_df is field_name (e.g., "Close" or "income_statement_Lợi nhuận thuần")
                    if field_info.field_name in ticker_df.columns:
                        rename_map[field_info.field_name] = field_info.field_id

                ticker_df = ticker_df.rename(columns=rename_map)
                all_dataframes.append(ticker_df)

                logging.info(f"Loaded {len(ticker_df)} rows for ticker={ticker} with fields: {list(rename_map.values())}")
            except Exception as e:
                logging.error(f"Failed to load data for ticker={ticker}: {e}")
                raise

        # Merge all dataframes on index (time)
        if len(all_dataframes) == 0:
            raise RuntimeError(f"No data loaded for any ticker")

        # Start with the first dataframe
        self.datas = all_dataframes[0]

        # Join with remaining dataframes
        for df in all_dataframes[1:]:
            self.datas = self.datas.join(df, how='outer', rsuffix='_dup')

        # Filter data by run_from if specified
        if self.run_from:
            self.datas = self.datas[self.datas.index >= self.run_from]

        logging.info(f"Total loaded data shape: {self.datas.shape}, columns: {list(self.datas.columns)}")

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
        if self.send_data:
            if self.current_state.bt_mode == AllowedTradeMode.LiveTrade:
                self.__send_signal__()
        else:
            logging.info(f"send_data is False, skipping sending latest signal for strategy_id={self.strategy_id}")
        # And send to Kafka
        send_states = self.trading_states[send_from_cp:]
        if len(send_states) == 0:
            logging.warning(f"No new records to send for strategy_id={self.strategy_id}")
            return
        if self.send_data:
            logging.info(f"Sending {len(send_states)} trading state records for strategy_id={self.strategy_id}. CP = {send_from_cp}")
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
        else:
            logging.info(f"send_data is False, skipping sending trading states for strategy_id={self.strategy_id}. CP = {send_from_cp}")

    def run(self):
        # Setup fields
        self.__setup__()
        # Initial strategy run ping
        self.producer.produce(
            "ping",
            key="run_strategy",
            value=f"Run strategy {self.strategy_id}. Re-run={self.re_run}",
            callback=delivery_report
        )
        self.producer.flush() # Ensure ping is sent before proceeding
        # Load data
        self.__load_data__()
        if len(self.datas) == 0:
            raise RuntimeError(f"No data loaded for symbol={self.symbol} from {self.run_from}")

        self.ht_prices = self.datas["Close"].tolist()
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

        logging.debug(f"Finalizing strategy run and sending data for strategy_id={self.strategy_id}")
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


if __name__ == "__main__":
    from xno.data import Fields
    from xno.trade import AllowedTradeMode


    # Test class for demonstrating add_field and load_data functionality
    class TestStrategyRunner(StrategyRunner):
        """
        Test implementation of StrategyRunner to test add_field and load_data
        """

        def __generate_signal__(self) -> List[float]:
            # Simple test signal: return zeros (Hold)
            return [0.0] * len(self.datas)

        def __step__(self, time_idx: int):
            # Simple test step: do nothing
            pass


    # Create a mock config class for testing
    class MockConfig:
        def __init__(self):
            self.run_from = "2015-01-01"
            self.run_to = "2025-12-31"
            self.symbol = "SSI"
            self.timeframe = "D"
            self.init_cash = 1000000000
            self.engine = "TA-Bot"
            self.symbol_type = "S"

    # Mock the config loader
    original_get_config = StrategyConfigLoader.get_config
    StrategyConfigLoader.get_config = lambda strategy_id, mode: MockConfig()

    try:
        runner = TestStrategyRunner(
            strategy_id="test_strategy",
            mode=AllowedTradeMode.BackTrade,
            re_run=False,
            send_data=False,
        )

        runner.add_field(
            field_id="income_statement_Lợi nhuận thuần_SSI",
            field_name=Fields.IncomeStatement.LOI_NHUAN_THUAN,
            ticker="SSI"
        )

        runner.add_field(
            field_id="ratio_P/E_SSI",
            field_name=Fields.Ratio.P_E,
            ticker="SSI"
        )

        runner.add_field(
            field_id="ratio_ROE_SSI",
            field_name=Fields.Ratio.ROE_PERCENT,
            ticker="SSI"
        )

        runner.add_field(
            field_id="income_statement_Lợi nhuận thuần_ACB",
            field_name=Fields.IncomeStatement.LOI_NHUAN_THUAN,
            ticker="ACB"
        )

        runner.add_field(
            field_id="ratio_P/E_ACB",
            field_name=Fields.Ratio.P_E,
            ticker="ACB"
        )

        runner.add_field(
            field_id="Close",
            field_name="Close",
            ticker="ACB"
        )
        runner.__setup__()
        runner.__load_data__()
        print(runner.datas.tail(3).to_string())

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
