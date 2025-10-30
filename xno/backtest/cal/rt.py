import dataclasses
import numpy as np
import pandas as pd


@dataclasses.dataclass
class BacktestResult:
    times: np.ndarray
    prices: np.ndarray
    positions: np.ndarray
    trade_sizes: np.ndarray
    returns: pd.Series
    cumret: np.ndarray
    pnl: np.ndarray
    fees: np.ndarray
    equity_curve: np.ndarray
    bm_equities: np.ndarray
    bm_returns: np.ndarray
    bm_cumret: np.ndarray
    bm_pnl: np.ndarray


def _compound_returns(returns: np.ndarray) -> np.ndarray:
    """Cumulative return theo công thức lãi kép."""
    return np.cumprod(1 + returns) - 1


def get_returns_stock(
        init_cash,
        times,
        prices,
        positions,
        trade_sizes,
        fee_rate=0.0015  # 0.15%
) -> BacktestResult:

    if not (len(times) == len(prices) == len(positions) == len(trade_sizes)):
        raise ValueError("times, prices, positions, and trade_sizes must have the same length.")

    prices = np.asarray(prices, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    trade_sizes = np.asarray(trade_sizes, dtype=np.float64)
    positions = positions - trade_sizes

    fees = np.abs(trade_sizes) * prices * fee_rate
    price_diff = np.diff(prices, prepend=prices[0])
    pnl = positions * price_diff

    pnl_cum = np.cumsum(pnl)
    fees_cum = np.cumsum(fees)
    equity = init_cash + pnl_cum - fees_cum

    returns = np.zeros_like(equity)
    returns[1:] = (equity[1:] - equity[:-1]) / equity[:-1]

    cumret = _compound_returns(returns)

    # returns = pd.Series(returns, index=pd.to_datetime(times))

    bm_equity = (init_cash / prices[0]) * prices
    bm_returns = np.zeros_like(bm_equity)
    bm_returns[1:] = (bm_equity[1:] - bm_equity[:-1]) / bm_equity[:-1]
    bm_cumret = _compound_returns(bm_returns)
    bm_pnl = bm_equity - init_cash

    return BacktestResult(
        times=times,
        prices=prices,
        positions=positions,
        trade_sizes=trade_sizes,
        returns=returns,
        cumret=cumret,
        pnl=pnl,
        fees=fees,
        equity_curve=equity,
        bm_equities=bm_equity,
        bm_returns=bm_returns,
        bm_cumret=bm_cumret,
        bm_pnl=bm_pnl
    )


def get_returns_derivative(
        init_cash,
        times,
        prices,
        positions,
        trade_sizes,
        fee_rate=20_000
) -> BacktestResult:

    if not (len(times) == len(prices) == len(positions) == len(trade_sizes)):
        raise ValueError("times, prices, positions, and trade_sizes must have the same length.")

    prices = np.asarray(prices, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    trade_sizes = np.asarray(trade_sizes, dtype=np.float64)
    positions = positions - trade_sizes

    fees = np.abs(trade_sizes) * fee_rate
    price_diff = np.diff(prices, prepend=prices[0])
    pnl = positions * price_diff * 100_000

    pnl_cum = np.cumsum(pnl)
    fees_cum = np.cumsum(fees)
    equity = init_cash + pnl_cum - fees_cum

    returns = np.zeros_like(equity)
    returns[1:] = (equity[1:] - equity[:-1]) / equity[:-1]

    cumret = _compound_returns(returns)

    # returns = pd.Series(returns, index=pd.to_datetime(times))

    bm_pnl = np.cumsum(price_diff * 100_000)
    bm_equity = init_cash + bm_pnl
    bm_returns = np.zeros_like(bm_equity)
    bm_returns[1:] = (bm_equity[1:] - bm_equity[:-1]) / bm_equity[:-1]
    bm_cumret = _compound_returns(bm_returns)

    return BacktestResult(
        times=times,
        prices=prices,
        positions=positions,
        trade_sizes=trade_sizes,
        returns=returns,
        cumret=cumret,
        pnl=pnl,
        fees=fees,
        equity_curve=equity,
        bm_equities=bm_equity,
        bm_returns=bm_returns,
        bm_cumret=bm_cumret,
        bm_pnl=bm_pnl
    )
