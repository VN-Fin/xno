from dataclasses import dataclass
import numpy as np


@dataclass
class BacktestInput:
    strategy_id: str
    re_run: bool
    book_size: float
    symbol_type: str
    times: np.ndarray
    prices: np.ndarray
    positions: np.ndarray
    trade_sizes: np.ndarray

