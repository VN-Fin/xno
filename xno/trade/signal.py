import logging

from pydantic import field_serializer, BaseModel

from xno.trade.tp import (
    SymbolType,
    DateTimeType,
    NumericType,
    ActionType,
    EngineType,
    TradeModeType
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
        # Convert numpy.datetime64 or pandas.Timestamp to ISO string
        if isinstance(dt, (np.datetime64, pd.Timestamp)):
            dt = pd.Timestamp(dt).to_pydatetime()
        return dt.isoformat()

    def to_json_str(self) -> str:
        if self.bt_mode != AllowedTradeMode.LiveTrade:
            logging.warning("StrategySignal can only be serialized to JSON in LiveTrade mode.")
            return ""
        return self.model_dump_json()


if __name__ == "__main__":
    from xno.trade.tp import AllowedAction, AllowedTradeMode, AllowedEngine, TradeModeType, AllowedSymbolType

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
    print(signal.to_json_str())