import numpy as np
import pandas as pd


def get_returns_stock(
        init_cash,
        times,
        prices,
        positions,
        trade_sizes,
        fee_rate=0.0015  # 0.15%
) -> pd.Series:

    # --- Validate input lengths ---
    if not (len(times) == len(prices) == len(positions) == len(trade_sizes)):
        raise ValueError("times, prices, positions, and trade_sizes must have the same length.")

    # --- Convert to numpy arrays ---
    prices = np.asarray(prices, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    trade_sizes = np.asarray(trade_sizes, dtype=np.float64)
    positions = positions - trade_sizes

    # --- Compute trading fees (only when trade occurs) ---
    fees = np.abs(trade_sizes) * prices * fee_rate

    # --- Compute daily price changes ---
    price_diff = np.diff(prices, prepend=prices[0])

    # --- Daily PnL: position * price change ---
    pnl = positions * price_diff

    # --- Cumulative values ---
    pnl_cum = np.cumsum(pnl)
    fees_cum = np.cumsum(fees)

    # --- Compute equity: init cash + total pnl - total fees ---
    equity = init_cash + pnl_cum - fees_cum

    # --- Step returns ---
    returns = np.zeros_like(equity)
    returns[1:] = (equity[1:] - equity[:-1]) / equity[:-1]

    # --- Create result Series with metadata ---
    result = pd.Series(returns, index=pd.Index(times, name="time"), name="return")
    result.attrs['pnl'] = pnl
    result.attrs['fees'] = fees
    result.attrs['equity_curve'] = equity

    return result


def get_returns_derivative(
        init_cash,
        times,
        prices,
        positions,
        trade_sizes,
        fee_rate=20_000 
) -> pd.Series:
    # --- Validate input lengths ---
    if not (len(times) == len(prices) == len(positions) == len(trade_sizes)):
        raise ValueError("times, prices, positions, and trade_sizes must have the same length.")

    # --- Convert to numpy arrays ---
    prices = np.asarray(prices, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    trade_sizes = np.asarray(trade_sizes, dtype=np.float64)
    positions = positions - trade_sizes

    # --- Compute trading fees (20 000 VND mỗi hợp đồng) ---
    fees = np.abs(trade_sizes) * fee_rate

    # --- Compute daily price changes ---
    price_diff = np.diff(prices, prepend=prices[0])

    # --- Daily PnL: mỗi điểm = 100 000 VND ---
    pnl = positions * price_diff * 100_000

    # --- Cumulative values ---
    pnl_cum = np.cumsum(pnl)
    fees_cum = np.cumsum(fees)

    # --- Compute equity: init cash + total pnl - total fees ---
    equity = init_cash + pnl_cum - fees_cum

    # --- Step returns ---
    returns = np.zeros_like(equity)
    returns[1:] = (equity[1:] - equity[:-1]) / equity[:-1]

    # --- Create result Series with metadata ---
    result = pd.Series(returns, index=pd.Index(times, name="time"), name="return")
    result.attrs['pnl'] = pnl
    result.attrs['fees'] = fees
    result.attrs['equity_curve'] = equity

    return result
