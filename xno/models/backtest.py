from dataclasses import dataclass
import numpy as np
from typing import List
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
    st = BacktestInput(
        bt_mode=TypeTradeMode.Train,
        strategy_id="strategy_id",
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