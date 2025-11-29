import abc
from abc import abstractmethod
from typing import Optional, Dict, List, Any

import numpy as np

from xno.models import TradeAnalysis, TradePerformance, BacktestInput, StrategyTradeSummary, TypeTradeMode
from xno.models.bt_result import BacktestResult
import quantstats as qs
import pandas as pd


def compound_returns(returns: np.ndarray) -> np.ndarray:
    """Cumulative return theo công thức lãi kép."""
    return np.cumprod(1 + returns) - 1


def safe_divide(numer, denom, eps=1e-12):
    """Chia an toàn: nếu denom = 0 thì thay bằng eps."""
    denom_safe = np.where(denom == 0, eps, denom)
    return numer / denom_safe


class BaseBacktest(abc.ABC):
    fee_rate = None

    def __init__(
        self,
            inp: BacktestInput
    ):
        self.bt_mode = TypeTradeMode(inp.bt_mode)
        self.strategy_id = inp.strategy_id
        self.init_cash = inp.book_size
        self.times = inp.times
        self.prices = inp.prices
        self.positions = inp.positions
        self.trade_sizes = inp.trade_sizes
        # Calculated from code
        self.returns: np.ndarray | None = None
        self.cum_rets: np.ndarray | None = None
        self.fees: np.ndarray | None = None
        self.pnls: np.ndarray | None = None
        self.equities: np.ndarray | None = None
        # tracking
        self.trade_analysis: Optional[TradeAnalysis] = None
        self.performance: Optional[TradePerformance] = None
        self.state_history: Optional[Dict[str, List[Any]]] = None
        self.bt_result = self.__build__()

    @abstractmethod
    def __build__(self) -> BacktestResult:
        raise NotImplementedError()

    def get_analysis(self) -> TradeAnalysis:
        if self.trade_analysis is not None:
            return self.trade_analysis
        # === 1. Portfolio-level metrics ===
        start_value = float(self.equities[0])
        end_value = float(self.equities[-1])
        total_return = (end_value - start_value) / start_value

        total_fee = np.sum(self.fees)

        # === 2. Trade-level statistics ===
        total_trades = int(np.count_nonzero(self.trade_sizes))
        total_closed_trades = int(np.count_nonzero(self.trade_sizes < 0))
        total_open_trades = int(np.count_nonzero(self.trade_sizes > 0))

        # === 3. Open trade unrealized PnL ===
        open_trade_pnl = float(self.pnls[-1])

        # === 4. Trade performance metrics ===
        if self.returns is not None and self.returns.size > 0:
            best_trade = float(np.max(self.returns))
            worst_trade = float(np.min(self.returns))
            avg_win_trade = float(np.mean(self.returns[self.returns > 0])) if np.any(self.returns > 0) else 0.0
            avg_loss_trade = float(np.mean(self.returns[self.returns < 0])) if np.any(self.returns < 0) else 0.0
        else:
            best_trade = worst_trade = avg_win_trade = avg_loss_trade = 0.0

        # === 5. Return structured results ===
        self.trade_analysis = TradeAnalysis(
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
            avg_win_trade_duration=None,  # chưa tính
            avg_loss_trade_duration=None,  # chưa tính
        )
        return self.trade_analysis

    def get_performance(self) -> TradePerformance:
        if self.performance is not None:
            return self.performance
        # Begin
        rets = pd.Series(
            self.returns,
            index=pd.to_datetime(self.times)
        )

        # Infer frequency
        freq = pd.infer_freq(rets.index)
        is_daily_or_higher = freq in ['D', 'B', 'W', 'M']
        rs = TradePerformance(
            avg_return=qs.stats.avg_return(rets),
            cumulative_return=qs.stats.comp(rets),
            cvar=qs.stats.cvar(rets),
            gain_to_pain_ratio=qs.stats.gain_to_pain_ratio(rets),
            kelly_criterion=qs.stats.kelly_criterion(rets),
            max_drawdown=qs.stats.max_drawdown(rets),
            omega=qs.stats.omega(rets),
            profit_factor=qs.stats.profit_factor(rets),
            recovery_factor=qs.stats.recovery_factor(rets),
            sharpe=qs.stats.sharpe(rets),
            sortino=qs.stats.sortino(rets),
            tail_ratio=qs.stats.tail_ratio(rets),
            ulcer_index=qs.stats.ulcer_index(rets),
            var=qs.stats.value_at_risk(rets),
            volatility=qs.stats.volatility(rets),
            win_loss_ratio=qs.stats.win_loss_ratio(rets),
            win_rate=qs.stats.win_rate(rets),
            annual_return=qs.stats.cagr(rets),
            calmar=qs.stats.calmar(rets),
        )

        # Only compute annualized metrics if daily or higher
        if is_daily_or_higher:
            rs.annual_return = qs.stats.cagr(rets)
            rs.calmar = qs.stats.calmar(rets)

        self.performance = rs
        return self.performance

    def summarize(self) -> StrategyTradeSummary:
        return StrategyTradeSummary(
            strategy_id=self.strategy_id,
            init_cash=self.init_cash,
            from_time=self.times[0],
            to_time=self.times[-1],
            analysis=self.get_analysis(),
            performance=self.get_performance(),
            state_history=self.state_history,
            bt_mode=self.bt_mode,
        )

