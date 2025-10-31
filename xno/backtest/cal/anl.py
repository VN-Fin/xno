from xno.models.analysis import TradeAnalysis
import numpy as np


def get_trade_analysis_metrics(
    equity_curve: np.ndarray,
    trade_returns: np.ndarray,
    fees: float,
    pnl: np.ndarray,
    trade_sizes: np.ndarray,
) -> TradeAnalysis:
    
    # === 1. Portfolio-level metrics ===
    start_value = float(equity_curve[0])
    end_value = float(equity_curve[-1])
    total_return = (end_value - start_value) / start_value

    total_fee = float(np.sum(fees)) if isinstance(fees, (list, np.ndarray)) else float(fees)

    # === 2. Trade-level statistics ===
    total_trades = len(trade_sizes[trade_sizes != 0])
    total_closed_trades = len(trade_sizes[trade_sizes < 0])
    total_open_trades = len(trade_sizes[trade_sizes > 0])

    # === 3. Open trade unrealized PnL ===
    open_trade_pnl = float(pnl[-1])  


    # === 4. Trade performance metrics ===
    if trade_returns is not None and len(trade_returns) > 0:
        best_trade = float(np.max(trade_returns))
        worst_trade = float(np.min(trade_returns))
        avg_win_trade = float(np.mean(trade_returns[trade_returns > 0])) if np.any(trade_returns > 0) else 0.0
        avg_loss_trade = float(np.mean(trade_returns[trade_returns < 0])) if np.any(trade_returns < 0) else 0.0
    else:
        best_trade = worst_trade = avg_win_trade = avg_loss_trade = 0.0

    # === 5. Return structured results ===
    return TradeAnalysis(
        start_value=start_value,
        end_value=end_value,
        total_return=total_return,
        benchmark_return=None,
        total_fee=total_fee,
        total_trades=total_trades,
        total_closed_trades=total_closed_trades,
        total_open_trades=total_open_trades,
        open_trade_pnl=open_trade_pnl,
        best_trade=best_trade,
        worst_trade=worst_trade,
        avg_win_trade=avg_win_trade,
        avg_loss_trade=avg_loss_trade,
        avg_win_trade_duration=None, # chưa tính
        avg_loss_trade_duration=None, # chưa tính
    )
