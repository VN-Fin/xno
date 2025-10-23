from xno.basic_type import NumericType
import numpy as np
import pandas as pd


def get_returns(
        init_cash: NumericType,
        times,
        prices,
        target_sizes
) -> pd.Series:
    """
    Calculate step returns based on target sizes and prices.
    init cash: 500,
    09:00 - 500m  1  1950 ->                   = 0%
    09:01 - 505m  0  1955 -> (505 - 500) / 500 = 1%
    09:02 - 510m  0  1960 -> (510 - 505) / 505 = 0.990099%
    09:03 - 500m  0  1950 -> (500 - 510) / 510 = -1.960784%
    """
    # Validation
    if len(prices) != len(target_sizes):
        raise ValueError("Prices and target sizes must have the same length.")
    if len(prices) != len(times):
        raise ValueError("Times, prices, and target_sizes must have the same length.")

    # Convert to numpy arrays
    prices = np.asarray(prices, dtype=np.float64)
    target_sizes = np.asarray(target_sizes, dtype=np.float64)

    # Compute portfolio value each step
    # Equity = cash + position_value
    # We assume full allocation to target position each step
    position_value = target_sizes * prices
    equity_curve = init_cash * (position_value / position_value[0])

    # Compute step returns (percentage change)
    returns = np.zeros_like(prices, dtype=np.float64)
    equity_prev = np.maximum(equity_curve[:-1], 1e-12)  # avoid div/0
    returns[1:] = (equity_curve[1:] - equity_curve[:-1]) / equity_prev

    # Return as Pandas Series
    return pd.Series(returns, index=times, name="returns")
