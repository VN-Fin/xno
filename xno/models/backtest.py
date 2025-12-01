from dataclasses import dataclass
import numpy as np
from typing import List, Any, Type
from xno.models.tp import (
    TypeMarket,
    TypeContract,
    TypeTradeMode,
    TypeAction,
)

__all__ = ["BacktestInput"]

from xno.utils.struct import DefaultStruct


@dataclass
class BacktestInput(DefaultStruct):
    strategy_id: str
    bt_mode: TypeTradeMode
    bt_cls: Any
    symbol : str
    market: TypeMarket
    contract: TypeContract
    re_run: bool
    book_size: float
    actions: List[TypeAction]
    times: np.ndarray
    prices: np.ndarray
    positions: np.ndarray
    trade_sizes: np.ndarray



if __name__ == "__main__":
    from xno.backtest.vn_stocks import BacktestVnStocks

    st = BacktestInput(
        bt_mode=TypeTradeMode.Train,
        strategy_id="strategy_id",
        bt_cls=BacktestVnStocks,
        re_run=False,
        book_size=0.1,
        symbol="BTC",
        market=TypeMarket.Crypto,
        contract=TypeContract.Future,
        actions=[TypeAction.Buy],
        times=np.linspace(0, 100, 100),
        prices=np.linspace(0, 100, 100),
        positions=np.linspace(0, 100, 100),
        trade_sizes=np.linspace(0, 100, 100),
    )

    print(st.to_json())