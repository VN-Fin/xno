from xno.basic_type import NumericType
import numpy as np
import pandas as pd


def get_returns(
        init_cash: NumericType,
        times,
        prices,
        target_sizes,
        asset_type: str = "stock",  # "stock" or "derivative"
        fee_rate: NumericType = None
) -> pd.Series:
    """
    Calculate step returns based on target sizes and prices.
    Supports different fee structures:
    - Stock: 0.15% fee rate
    - Derivative: 20k VND per contract
    
    """
    # Validation
    if len(prices) != len(target_sizes):
        raise ValueError("Prices and target sizes must have the same length.")
    if len(prices) != len(times):
        raise ValueError("Times, prices, and target_sizes must have the same length.")

    # Convert to numpy arrays for vectorized operations
    prices = np.asarray(prices, dtype=np.float64)
    target_sizes = np.asarray(target_sizes, dtype=np.float64)
    
    # Set default fee rates based on asset type
    if fee_rate is None:
        if asset_type == "stock":
            fee_rate = 0.0015  # 0.15%
        elif asset_type == "derivative":
            fee_rate = 20_000  # 20k VND per contract
        else:
            raise ValueError("asset_type must be 'stock' or 'derivative'")

    # Calculate fees based on asset type 
    if asset_type == "stock":
        # Stock fee: percentage of trade value
        fees = np.where(target_sizes != 0, np.abs(target_sizes) * prices * fee_rate, 0.0)
    else:  # derivative
        # Derivative fee: fixed amount per contract
        fees = np.where(target_sizes != 0, np.abs(target_sizes) * fee_rate, 0.0)

    # Cumulative position 
    position = np.cumsum(target_sizes)

    # Cash after each trade 
    cash = init_cash - np.cumsum(target_sizes * prices + fees)

    # Equity = cash + position_value 
    equity = cash + position * prices

    # Returns calculation 
    returns = np.zeros_like(equity)
    equity_prev = np.where(equity[:-1] != 0, equity[:-1], 1e-12)
    returns[1:] = (equity[1:] - equity[:-1]) / equity_prev


    return pd.Series(returns, index=times, name="returns")

