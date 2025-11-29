"""Derivative trading runner"""
from abc import abstractmethod
from typing import List

from xno.strategy.strategy_runner import StrategyRunner
from xno.models import (
    AllowedAction, AllowedTradeMode,
    AdvancedConfig, AllowedEngine,
    StrategyConfig, AllowedSymbolType,
)
import logging
from xno.utils.stock import round_to_lot

class DerivativeRunner(StrategyRunner):
    _lot_size = 1  # Derivatives typically trade in units of 1 contract
    def __init__(
            self,
            config: StrategyConfig,
            re_run: bool,
            send_data: bool,
    ):
        super().__init__(config, re_run, send_data)
        self.max_contracts = self.init_cash // 25_000_000  # Fixed for derivatives

    @abstractmethod
    def __generate_signal__(self) -> List[float]:
        raise NotImplementedError("Implement in subclass to generate signals.")

    def __step__(self, time_idx: int):
        """
        Run one step of derivative trading algorithm.

        Rules enforced:
        1. Supports long and short positions.
        2. Can buy or sell at any time (intraday).
        3. Position changes directly based on signal delta.
        4. Position size = current_weight * max_contracts
        """
        self.current_state.current_action = AllowedAction.Hold
        current_price = self.prices[time_idx]
        current_time = self.times[time_idx]
        if current_time < self.run_to:
            self.checkpoint_idx = time_idx

        # Raw signal at this timestep
        sig: float = self.signals[time_idx]

        # Compute how much to adjust current position weight
        updated_weight = sig - self.current_state.current_weight
        current_trade_size = 0.0

        if abs(updated_weight) < 1e-6:
            # No significant change in signal
            pass
        elif updated_weight > 0:
            # Buy / Increase long position (or reduce short)
            logging.debug(f"[{current_time}] Buy logic triggered: signal={sig:.2f}")
            self.current_state.current_weight += updated_weight
            current_trade_size = round_to_lot(updated_weight * self.max_contracts, self._lot_size)
            self.current_state.current_position += current_trade_size
            self.current_state.current_action = AllowedAction.Buy
        elif updated_weight < 0:
            # Sell / Increase short position (or reduce long)
            logging.debug(f"[{current_time}] Sell logic triggered: signal={sig:.2f}")
            self.current_state.current_weight += updated_weight
            current_trade_size = round_to_lot(updated_weight * self.max_contracts, self._lot_size)  # Negative = short
            self.current_state.current_position += current_trade_size
            self.current_state.current_action = AllowedAction.Sell

        # Update the current trading state
        self.current_state.current_price = current_price
        self.current_state.candle = current_time
        self.current_state.trade_size = current_trade_size
        self.current_time_idx = time_idx

        # Log summary for debugging
        logging.debug(
            f"[{current_time}] Action={self.current_state.current_action}, "
            f"TradeSize={current_trade_size:.2f}, "
            f"Position={self.current_state.current_position:.2f}, "
            f"Weight={self.current_state.current_weight:.2f}"
        )



if __name__ == "__main__":
    class CustomDerivativeRunner(DerivativeRunner):
        def __generate_signal__(self) -> List[float]:
            # Example: Generate random signals for demonstration
            import numpy as np
            return np.random.uniform(-1, 1, size=len(self.prices)).tolist()

        def __load_data__(self):
            # Override to load OHLCV data for testing
            from xno.data.ohlcv import OhlcvDataManager

            self.datas = OhlcvDataManager.get(
                resolution=self.timeframe,
                symbol=self.symbol,
                from_time=self.run_from,
                to_time=self.run_to,
                factor=1,
            )
            return self.datas



    # Example usage
    config = StrategyConfig(
        strategy_id="fad40f3b-52a7-44d1-99cb-8d4b5aa257c5",
        symbol="VN30F1M",
        symbol_type=AllowedSymbolType.Derivative,
        timeframe="5min",
        init_cash=500_000_000,
        run_from="2025-01-01 09:00:00",
        run_to="2025-08-31 15:00:00",
        mode=AllowedTradeMode.BackTrade,
        advanced_config=AdvancedConfig(),
        engine=AllowedEngine.XQuant
    )
    runner = CustomDerivativeRunner(
        config=config,
        re_run=False,
        send_data=True,
    )
    runner.run()
    logging.info(f"Run stats: {runner.stats()}")
    logging.info(f"Backtest: {runner.backtest()}")
    runner.visualize()
