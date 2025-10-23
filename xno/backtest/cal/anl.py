from xno.backtest.analysis import TradeAnalysis
import pandas as pd


def get_trade_analysis_metrics(
        equity_curve: pd.Series,
        trade_returns: list[float] | np.ndarray | None = None,
        trade_durations: list[int] | np.ndarray | None = None,
        fees: float = 0.0,
        benchmark_returns: pd.Series | None = None,
) -> TradeAnalysis:
    stats = pf.stats().to_dict()
    stats = TradeAnalysis(
        start_value=stats['Start Value'],
        end_value=stats['End Value'],
        total_return=stats['Total Return [%]'] / 100,
        benchmark_return=stats['Benchmark Return [%]'] / 100,
        total_fee=stats['Total Fees Paid'],
        total_trades=stats['Total Trades'],
        total_closed_trades=stats['Total Closed Trades'],
        total_open_trades=stats['Total Open Trades'],
        open_trade_pnl=stats['Open Trade PnL'],
        best_trade=stats['Best Trade [%]'] / 100,
        worst_trade=stats['Worst Trade [%]'] / 100,
        avg_win_trade=stats['Avg Winning Trade [%]'] / 100,
        avg_loss_trade=stats['Avg Losing Trade [%]'] / 100,
        avg_win_trade_duration=stats['Avg Winning Trade Duration'],
        avg_loss_trade_duration=stats['Avg Losing Trade Duration'],
    )
    return stats