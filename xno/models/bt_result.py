from dataclasses import dataclass
import numpy as np
import pandas as pd

from xno.utils.struct import DefaultStruct


@dataclass
class BacktestResult(DefaultStruct):
    times: np.ndarray
    prices: np.ndarray
    positions: np.ndarray
    trade_sizes: np.ndarray
    returns: np.ndarray
    cumret: np.ndarray
    pnl: np.ndarray
    fees: np.ndarray
    equity_curve: np.ndarray
    bm_equities: np.ndarray
    bm_returns: np.ndarray
    bm_cumret: np.ndarray
    bm_pnl: np.ndarray
