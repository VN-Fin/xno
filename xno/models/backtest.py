from dataclasses import dataclass
import numpy as np
from typing import List, Any, Type
from xno.models.tp import (
    TypeSymbolType,
    TypeTradeMode,
    TypeAction,
)

__all__ = ["BacktestInput", "BacktestOverview"]

from xno.utils.struct import DefaultStruct


@dataclass
class BacktestInput(DefaultStruct):
    bot_id: str
    timeframe: str
    bt_mode: TypeTradeMode
    bt_cls: Any
    symbol : str
    symbol_type: TypeSymbolType
    re_run: bool
    book_size: float
    actions: List[TypeAction]
    times: np.ndarray
    prices: np.ndarray
    positions: np.ndarray
    trade_sizes: np.ndarray


@dataclass
class BacktestOverview(DefaultStruct):
    id: str
    time: Any
    from_time: Any
    to_time: Any
    cash: float
    bt_mode: TypeTradeMode
    status: str
    task_id: str
    message: str
    success: bool



if __name__ == "__main__":
    from xno.backtest.vn_stocks import BacktestVnStocks

    st = BacktestInput(
        bt_mode=TypeTradeMode.Train,
        bot_id="strategy_id",
        timeframe="1d",
        bt_cls=BacktestVnStocks,
        re_run=False,
        book_size=0.1,
        symbol="BTC",
        symbol_type=TypeSymbolType.CryptoFuture,
        actions=[TypeAction.Buy],
        times=np.linspace(0, 100, 100),
        prices=np.linspace(0, 100, 100),
        positions=np.linspace(0, 100, 100),
        trade_sizes=np.linspace(0, 100, 100),
    )

    print(st.to_json())