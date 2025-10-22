"""Migrate to new runner structure."""
from abc import abstractmethod
from datetime import timedelta
from typing import List

from xno.strategy.strategy_runner import StrategyRunner
from xno.trade import (
    AllowedAction, AllowedTradeMode,
)
import logging

from xno.utils.stock import round_to_lot


class StockRunner(StrategyRunner):
    _hold_days = timedelta(days=3)
    _lot_size = 100

    @abstractmethod
    def __generate_signal__(self) -> List[float]:
        raise NotImplementedError("Implement in subclass to generate signals.")

    @abstractmethod
    def __load_data__(self):
        raise NotImplementedError("Implement in subclass to load data.")

    def __step__(self, time_idx: int):
        """
        Run the trading algorithm state, which includes setting up the algorithm, generating signals, and verifying the trading signal.
        Rules enforced:
        1. Cannot sell more than the current open position.
        2. Cannot sell before any buying has occurred.
        3. Sell is only allowed after holding for at least 3 days (T+3 logic).
        4. Shares are rounded down to the nearest stock lot size.
        5. Shares are computed based on the initial cash, current price, and lot size.
        6. If conditions for selling are not met, the signal is ignored until eligible.
        """
        self.current_state.current_action = AllowedAction.Hold
        current_price = self.ht_prices[time_idx]  # Get the current price from the history prices]
        current_time = self.ht_times[time_idx]   # Current day from the timestamp
        if current_time < self.run_to:
            self.checkpoint_idx = time_idx
        # The signal unverified, which is the current signal at the current time index
        sig: float = self.signals[time_idx]
        # Calculate the benchmark shares based on initial cash and current price
        current_max_shares = round_to_lot(self.init_cash // current_price, self._lot_size)

        # Calculate day difference AND update T0, T1, T2 positions based on the previous day
        prev_time = self.ht_times[time_idx - 1]  if time_idx > 0 else current_time  # Previous day for the first iteration
        day_diff = (current_time - prev_time).days
        if day_diff > 0:
            logging.debug(
                f"Update T0, T1, T2 for {current_time}, "
                f"T0: {self.current_state.t0_size}, "
                f"T1: {self.current_state.t1_size}, "
                f"T2: {self.current_state.t2_size}, "
                f"Sell Position: {self.current_state.sell_size}"
            )
            # Consecutive prev_day days
            self.current_state.sell_size += self.current_state.t2_size
            self.current_state.t2_size = self.current_state.t1_size
            self.current_state.t1_size = self.current_state.t0_size
            self.current_state.t0_size = 0

        # Calculate the current action based on the signal
        if sig > 0:
            updated_weight = min(sig - self.current_state.current_weight, 1 - self.current_state.current_weight)
        elif sig < 0:
            if self.current_state.current_weight > 0:
                updated_weight = max(sig - self.current_state.current_weight, -self.current_state.current_weight)  # Can reduce or reverse only what we own
            else:
                updated_weight = 0.0  # Can't sell if we have no position
        else:
            updated_weight = 0.0

        current_trade_size = 0.0
        # Handle sell logic
        if updated_weight == 0:
            # Skip if no change in position
            pass
        elif updated_weight < 0 or self.current_state.pending_sell_weight > 0:
            logging.debug(f"Entering sell logic at {current_time} with weight {sig}")
            if self.current_state.sell_size == 0:
                logging.debug(f"Sell position is 0, but trying to sell {sig} at {current_time}. This will be ignored, please waiting for the next timestamp to sell.")
                self.current_state.pending_sell_weight += abs(sig)  # Track pending sell position
            else:
                can_sell_weight = max(self.current_state.pending_sell_weight, abs(updated_weight))
                current_trade_size = min(
                    self.current_state.sell_size,
                    round_to_lot(can_sell_weight * self.current_state.current_position, self._lot_size)
                )  # Ensure we don't sell more than we have
                # Update state
                self.current_state.sell_size -= current_trade_size
                self.current_state.current_position -= current_trade_size  # Update total shares held
                self.current_state.current_weight -= can_sell_weight
                self.current_state.pending_sell_weight = max(self.current_state.pending_sell_weight - can_sell_weight, 0)  # Reduce pending sell position
                self.current_state.current_action = AllowedAction.Sell  # Set action to sell
        else: # Handle buy logic
            logging.debug(f"Entering buy logic at {current_time} with weight {sig}")
            self.current_state.current_weight += updated_weight  # Update current weight
            current_trade_size = round_to_lot(self.current_state.current_weight * current_max_shares, self._lot_size)
            self.current_state.t0_size += current_trade_size  # Update T0 position
            self.current_state.current_position += current_trade_size  # Update total shares held
            self.current_state.current_action = AllowedAction.Buy  # Set action to buy
        # Update the current state with price and time
        self.current_state.current_price = current_price
        self.current_state.candle = current_time
        self.current_state.trade_size = current_trade_size
        self.current_time_idx = time_idx
        self.trading_states.append(self.current_state.model_copy(deep=True))


if __name__ == "__main__":
    class CustomStockRunner(StockRunner):
        def __generate_signal__(self) -> List[float]:
            # Example: Generate random signals for demonstration
            import numpy as np
            return np.random.uniform(-1, 1, size=len(self.ht_prices)).tolist()

        def __load_data__(self):
            # Example: Load dummy data for demonstration
            import pandas as pd
            import numpy as np
            dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
            prices = pd.Series(100 + np.random.randn(100).cumsum(), index=dates)
            self.datas = pd.DataFrame({'Close': prices})

    # Example usage
    runner = CustomStockRunner(
        strategy_id="fad40f3b-52a7-44d1-99cb-8d4b5aa257c5",
        send_signal=False,
        mode=AllowedTradeMode.LiveTrade,
        re_run=False,
    )
    runner.add_field(
        "Volume", "Volume"
    ).add_field(
        "VN30Volume", "Volume", ticker="VN30"
    )
    runner.run()
    print(runner.stats())