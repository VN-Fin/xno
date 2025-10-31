import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_serializer

from xno.basic_type import NumericType, DateTimeType, BooleanType
from xno.models.tp import (
    ActionType,
    TradeModeType,
    EngineType,
    SymbolType,
    AllowedSymbolType,
    AllowedAction,
    AllowedTradeMode,
    AllowedEngine
)


class StrategyState(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }
    strategy_id : str # Strategy identifier, use UUID format
    book_size: NumericType # Total cash available for trading
    symbol: str    # Trading symbol, e.g., "AAPL", "BTC-USD"
    symbol_type: SymbolType # Type of the symbol (see AllowedSymbolType)
    candle: DateTimeType # Current candle data
    run_from: DateTimeType # Optional: Define the time range for the strategy run
    run_to: DateTimeType # Optional: Define the time range for the strategy run
    current_price: NumericType # Current price of the asset
    current_position: NumericType # Number of stocks/contracts held (num of holding shares/contracts)
    current_weight: NumericType # -1 to 1
    current_action: ActionType # B/S/H (see AllowedAction)
    trade_size: NumericType # Size of the trade to be executed (target_amount to buy/sell)
    bt_mode: TradeModeType # BackTrade/PaperTrade/LiveTrade (see AllowedTradeMode)
    current_time_idx: NumericType = 0  # To track the index of the current time in back/live/paper trade
    t0_size: NumericType = 0.0 # Applied for stock T+3, set to 0 if not applicable
    t1_size: NumericType = 0.0 # Applied for stock T+3, set to 0 if not applicable
    t2_size: NumericType = 0.0 # Applied for stock T+3, set to 0 if not applicable
    sell_size: NumericType = 0.0        # Total size to be sold
    pending_sell_weight: NumericType = 0.0    # Weight of the pending sell order
    re_run: BooleanType                # Flag to indicate if the strategy should be re-run
    engine: EngineType # Trading engine being used (see AllowedEngine)

    @field_serializer("candle", "run_from", "run_to")
    def serialize_datetime(self, dt):
        if isinstance(dt, (str, bytes)):
            # load from string
            dt = pd.Timestamp(dt).to_pydatetime()
        # Convert numpy.datetime64 or pandas.Timestamp to ISO string
        elif isinstance(dt, (np.datetime64, pd.Timestamp)):
            dt = pd.Timestamp(dt).to_pydatetime()
        return dt.isoformat()

    @field_serializer('*')
    def serialize_numpy(self, value):
        if isinstance(value, (np.generic,)):
            return value.item()  # convert numpy.int64 → int, numpy.float64 → float
        return value

    def to_json_str(self):
        return self.model_dump_json()


TradingState = StrategyState

if __name__ == "__main__":
    state = TradingState(
        strategy_id="123e4567-e89b-12d3-a456-426614174000",
        symbol="HSG",
        symbol_type=AllowedSymbolType.Stock,
        candle=datetime.datetime.now(),
        run_from=datetime.datetime.now(),
        run_to=np.datetime64(datetime.datetime.now() - datetime.timedelta(days=5)),
        current_price=150.0,
        current_position=100,
        current_weight=np.float64(0.5),
        current_action=AllowedAction.Buy,
        trade_size=50,
        bt_mode=AllowedTradeMode.BackTrade,
        re_run=False,
        engine=AllowedEngine.TABot,
        book_size=500e6
    )
    datas = state.to_json_str()
    print(datas)