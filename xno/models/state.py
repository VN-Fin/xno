from dataclasses import dataclass
import datetime
import numpy as np
from xno.basic_type import NumericType, DateTimeType, BooleanType
from xno.models.tp import (
    TypeEngine,
    TypeAction,
    TypeTradeMode,
    TypeSymbolType,
)
from xno.utils.struct import DefaultStruct


@dataclass
class StrategyState(DefaultStruct):
    strategy_id : str # Strategy identifier, use UUID format
    book_size: NumericType # Total cash available for trading
    symbol: str    # Trading symbol, e.g., "AAPL", "BTC-USD"
    symbol_type: TypeSymbolType
    candle: DateTimeType # Current candle data
    run_from: DateTimeType # Optional: Define the time range for the strategy run
    run_to: DateTimeType # Optional: Define the time range for the strategy run
    current_price: NumericType # Current price of the asset
    current_position: NumericType # Number of stocks/contracts held (num of holding shares/contracts)
    current_weight: NumericType # -1 to 1
    current_action: TypeAction # B/S/H (see AllowedAction)
    trade_size: NumericType # Size of the trade to be executed (target_amount to buy/sell)
    bt_mode: TypeTradeMode # BackTrade/PaperTrade/LiveTrade (see AllowedTradeMode)
    re_run: BooleanType                # Flag to indicate if the strategy should be re-run
    engine: TypeEngine # Trading engine being used (see AllowedEngine)
    current_time_idx: NumericType = 0  # To track the index of the current time in back/live/paper trade
    t0_size: NumericType = 0.0 # Applied for stock T+3, set to 0 if not applicable
    t1_size: NumericType = 0.0 # Applied for stock T+3, set to 0 if not applicable
    t2_size: NumericType = 0.0 # Applied for stock T+3, set to 0 if not applicable
    sell_size: NumericType = 0.0        # Total size to be sold
    pending_sell_weight: NumericType = 0.0    # Weight of the pending sell order


TradingState = StrategyState

if __name__ == "__main__":
    state = TradingState(
        strategy_id="123e4567-e89b-12d3-a456-426614174000",
        symbol="HSG",
        symbol_type=TypeSymbolType.VnStock,
        candle=datetime.datetime.now(),
        run_from=datetime.datetime.now(),
        run_to=np.datetime64(datetime.datetime.now() - datetime.timedelta(days=5)),
        current_price=150.0,
        current_position=100,
        current_weight=np.float64(0.5),
        current_action=TypeAction.Buy,
        trade_size=50,
        bt_mode=TypeTradeMode.Train,
        re_run=False,
        engine=TypeEngine.TABot,
        book_size=500e6
    )
    datas = state.to_json()
    print(datas)