import copy
from dataclasses import dataclass
from xno.basic_type import DateTimeType, NumericType
from xno.models.tp import (
    TypeAction,
    TypeTradeMode,
    TypeSymbolType,
    TypeEngine,
)
import numpy as np
import pandas as pd
from xno.utils.struct import DefaultStruct


@dataclass
class StrategySignal(DefaultStruct):
    strategy_id : str # Strategy identifier, use UUID format
    symbol: str    # Trading symbol, e.g., "AAPL", "BTC-USD"
    symbol_type: TypeSymbolType
    candle: DateTimeType # Current candle data
    current_price: NumericType # Current price of the asset
    current_weight: NumericType # -1 to 1
    current_action: TypeAction # B/S/H (see AllowedAction)
    bt_mode: TypeTradeMode
    engine: TypeEngine # Trading engine being used (see AllowedEngine)

    def __eq__(self, other):
        if not isinstance(other, StrategySignal):
            return False

        if self.strategy_id != other.strategy_id:
            return False

        # compare datetime
        if pd.Timestamp(self.candle) != pd.Timestamp(other.candle):
            return False

        # current_action = self.current_action.value if isinstance(self.current_action, AllowedAction) else self.current_action
        # prev_action = other.current_action.value if isinstance(other.current_action, AllowedAction) else other.current_action
        # Compare action and weight
        if self.current_action != other.current_action:
            return False

        # if similar weight
        if not np.isclose(self.current_weight, other.current_weight, rtol=1e-8, atol=1e-12):
            return False

        # if similar price
        if not np.isclose(self.current_price, other.current_price, rtol=1e-8, atol=1e-12):
            return False

        return True


if __name__ == "__main__":
    sig_str = """{"strategy_id":"123e4567-e89b-12d3-a456-426614174000","symbol":"AAPL","market":"S","contract":"S","candle":"2024-01-01 10:00:00","current_price":150.0,"current_weight":0.5,"current_action":"B","bt_mode":3,"engine":"TA-Bot"}"""
    signal1 = StrategySignal.from_str(sig_str)

    signal2 = StrategySignal(
        strategy_id="123e4567-e89b-12d3-a456-426614174000",
        symbol="AAPL",
        symbol_type=TypeSymbolType.UsStock,
        candle="2024-01-01 10:00:00",  # different here
        current_price=150.0,
        current_weight=0.5,
        current_action=TypeAction.Buy,
        bt_mode=TypeTradeMode.Live,
        engine=TypeEngine.TABot,
    )
    signal3 = copy.deepcopy(signal1)
    signal3.current_price = 150.1
    signal2.current_price = 150.0 + 1e-10
    # Test
    print(f"signal1 == signal2: {signal1 == signal2}")  # True
    print(f"signal1 == signal3: {signal1 == signal3}")  # False
    print(f"signal2 == signal1 (after update price): {signal2 == signal1}")  # True
