from dataclasses import dataclass
import numpy as np
from typing import List, AnyStr

from xno.models import AllowedTradeMode


@dataclass
class BacktestInput:
    bt_mode: AllowedTradeMode | int
    strategy_id: str
    re_run: bool
    book_size: float
    symbol_type: str
    actions: List[str]
    times: np.ndarray
    prices: np.ndarray
    positions: np.ndarray
    trade_sizes: np.ndarray
    run_id: str = None
