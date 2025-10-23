from pydantic import field_serializer, BaseModel

from xno.basic_type import DateTimeType, NumericType
from xno.trade.tp import (
    SymbolType,
    ActionType,
    EngineType,
    AllowedAction,
    AllowedTradeMode,
    AllowedEngine,
    TradeModeType,
    AllowedSymbolType
)
import numpy as np
import pandas as pd


class StrategySignal(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }
    strategy_id : str # Strategy identifier, use UUID format
    symbol: str    # Trading symbol, e.g., "AAPL", "BTC-USD"
    symbol_type: SymbolType # Type of the symbol (see AllowedSymbolType)
    candle: DateTimeType # Current candle data
    current_price: NumericType # Current price of the asset
    current_weight: NumericType # -1 to 1
    current_action: ActionType # B/S/H (see AllowedAction)
    bt_mode: TradeModeType
    engine: EngineType # Trading engine being used (see AllowedEngine)

    @field_serializer("candle")
    def serialize_datetime(self, dt):
        if isinstance(dt, (str, bytes)):
            # load from string
            dt = pd.Timestamp(dt).to_pydatetime()
        # Convert numpy.datetime64 or pandas.Timestamp to ISO string
        elif isinstance(dt, (np.datetime64, pd.Timestamp)):
            dt = pd.Timestamp(dt).to_pydatetime()
        return dt.isoformat()

    def to_json_str(self) -> str:
        return self.model_dump_json()

    def __eq__(self, other):
        if not isinstance(other, StrategySignal):
            return False

        a, b = self.model_dump(), other.model_dump()
        for key in a.keys():
            if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
                if not np.isclose(a[key], b[key], rtol=1e-8, atol=1e-12):  # compare float
                    return False
            elif a[key] != b[key]:
                return False
        return True


if __name__ == "__main__":
    sig_str = """{
      "bt_mode": 2,
      "candle": "2024-01-01 10:00:00",
      "current_action": "B",
      "current_price": 150.0,
      "current_weight": 0.5,
      "engine": "TA-Bot",
      "strategy_id": "123e4567-e89b-12d3-a456-426614174000",
      "symbol": "AAPL",
      "symbol_type": "S"
    }"""
    signal_str = StrategySignal.model_validate_json(sig_str)

    signal = StrategySignal(
        strategy_id="123e4567-e89b-12d3-a456-426614174000",
        symbol="AAPL",
        symbol_type=AllowedSymbolType.Stock,
        candle=pd.Timestamp("2024-01-01 10:00:00"),
        current_price=150.0,
        current_weight=0.5,
        current_action=AllowedAction.Buy,
        bt_mode=AllowedTradeMode.LiveTrade,
        engine=AllowedEngine.TABot,
    )
    signal2 = signal.model_copy(deep=True)
    signal3 = signal.model_copy(deep=True)
    signal3.current_price = 150.1
    signal2.current_price = 150.0 + 1e-10
    print(signal.to_json_str())
    # Test
    print(f"signal == signal2: {signal == signal2}")  # True
    print(f"signal == signal3: {signal == signal3}")  # False
    print(f"signa_str == signal: {signal_str == signal}")  # True
