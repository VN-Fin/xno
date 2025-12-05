from dataclasses import dataclass

from xno.utils.struct import DefaultStruct
import numpy as np

__all__ = ["BotBacktestResult", "BotBacktestResultSummary"]


@dataclass
class BotBacktestResultSummary(DefaultStruct):
    train: dict
    test: dict
    simulate: dict
    live: dict

@dataclass
class BotBacktestResult(DefaultStruct):
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
