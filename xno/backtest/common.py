import abc
from abc import abstractmethod
from typing import Optional, List

import numpy as np

from xno.models import (
    TradeAnalysis,
    TradePerformance,
    BacktestInput,
    BotTradeSummary,
    TypeTradeMode,
    BotStateHistory,
    StateSeries,
    SeriesMetric,
    BotBacktestResultSummary
)
import quantstats as qs
import pandas as pd


def compound_returns(returns: np.ndarray) -> np.ndarray:
    """Cumulative return theo công thức lãi kép."""
    return np.cumprod(1 + returns) - 1


def safe_divide(numer, denom, eps=1e-12):
    """Chia an toàn: nếu denom = 0 thì thay bằng eps."""
    denom_safe = np.where(denom == 0, eps, denom)
    return numer / denom_safe

minute_bar_per_day = 5.5 * 60 # 5.5 hours

def get_minutes(tf):
    tf = tf.lower()
    if tf in ("1d", "d", "day"):
        return minute_bar_per_day  # 6.5 trading hours

    if tf in ("1w", "w", "week"):
        return minute_bar_per_day * 5

    if tf in ("1m", "1mo", "month"):
        return minute_bar_per_day * 21

    # intraday (1min, 3min, 5min, 15min, etc)
    if "min" in tf:
        return float(tf.replace("min", "").strip())

    if "m" in tf:
        return float(tf.replace("m", "").strip())

    if "h" in tf:
        hours = float(tf.replace("h", "").strip())
        return hours * 60

    raise ValueError(f"Unsupported timeframe: {tf}")

def auto_window(timeframe: str) -> int:
    """
    Returns recommended rolling window size based on timeframe.
    Window is measured in *number of bars*.
    """

    timeframe = timeframe.lower().strip()

    # ---- Daily or higher ----
    if timeframe in ("1d", "d", "day"):
        return 126  # 6 months
    if timeframe in ("1w", "w", "week"):
        return 26           # 6 months of weekly bars
    if timeframe in ("1m", "m", "month"):
        return 12           # 1 year of monthly bars

    # ---- Intraday ----
    # VN market approx 285 minutes/day
    minutes_per_day = 285
    trading_days = 21  # 1 month

    # Parse intraday like "1m", "3m", "5m", "15m", "30m", "1h"
    # Convert timeframe into minutes per bar
    if timeframe.endswith("m"):
        try:
            bar_minutes = int(timeframe.replace("m", ""))
        except:
            bar_minutes = 5
    elif timeframe.endswith("h"):
        try:
            bar_minutes = int(timeframe.replace("h", "")) * 60
        except:
            bar_minutes = 60
    else:
        # Unknown format → fallback safe window
        return 100

    # Bars per day
    bars_per_day = int(minutes_per_day / bar_minutes)

    # 1-month rolling window for intraday
    window = bars_per_day * trading_days

    # ensure minimum and max sanity bounds
    return max(20, min(window, 5000))

class BaseBacktest(abc.ABC):
    fee_rate = None

    def __init__(
        self,
            inp: BacktestInput
    ):
        self.timeframe = inp.timeframe
        self.bt_mode = TypeTradeMode(inp.bt_mode)
        self.actions = inp.actions
        self.bot_id = inp.bot_id
        self.init_cash = inp.book_size
        self.times = inp.times
        self.prices = inp.prices
        self.positions = inp.positions
        self.trade_sizes = inp.trade_sizes
        self.periods = int(minute_bar_per_day / get_minutes(self.timeframe) * 250)
        # Calculated from code
        self.returns: np.ndarray | None = None
        # Build return series
        self.return_series: pd.Series | None = None
        # bench series
        self.cum_rets: np.ndarray | None = None
        self.fees: np.ndarray | None = None
        self.pnls: np.ndarray | None = None
        self.equities: np.ndarray | None = None
        self.bm_returns: np.ndarray | None = None
        self.bm_pnls: np.ndarray | None = None
        self.bm_cumrets: np.ndarray | None = None
        self.bm_equities: np.ndarray | None = None
        self.__build__()
        self.return_series = self.build_returns()  # Build pandas series
        # tracking
        self.trade_analysis: Optional[TradeAnalysis] = None
        self.performance: Optional[TradePerformance] = None
        self.state_history: BotStateHistory | None = None
        # rolling defines
        self.rolling_sharpe: np.ndarray | None = None
        self.rolling_vol: np.ndarray | None = None
        self.rolling_drawdown: np.ndarray | None = None

    def build_returns(self):
        if self.return_series is None:
            self.return_series = pd.Series(
                self.returns,
                index=pd.to_datetime(self.times)
            )
        return self.return_series

    # def rolling_metrics(self):
    #     rolling_mean = self.return_series.rolling(self.periods).mean()
    #     rolling_std  = self.return_series.rolling(self.periods).std()
    #
    #     rolling_sharpe =
    #     return {
    #         "rolling_sharpe": rolling_sharpe,
    #         "rolling_vol": rolling_vol,
    #         "rolling_drawdown": rolling_dd,
    #         "rolling_max_drawdown": rolling_max_dd,
    #         "rolling_beta": rolling_beta,
    #         "rolling_corr": rolling_corr
    #     }

    def state_history_series(self) -> List[StateSeries]:
        if self.state_history is not None:
            return self.state_history.get_series()
        else:
            raise Exception("state_history is None")

    @abstractmethod
    def __build__(self) -> BotBacktestResultSummary:
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
        rs = TradePerformance(
            avg_return=qs.stats.avg_return(self.return_series),
            cumulative_return=qs.stats.comp(self.return_series),
            cvar=qs.stats.cvar(self.return_series),
            gain_to_pain_ratio=qs.stats.gain_to_pain_ratio(self.return_series),
            kelly_criterion=qs.stats.kelly_criterion(self.return_series),
            max_drawdown=qs.stats.max_drawdown(self.return_series),
            omega=qs.stats.omega(self.return_series),
            profit_factor=qs.stats.profit_factor(self.return_series),
            recovery_factor=qs.stats.recovery_factor(self.return_series),
            sharpe=qs.stats.sharpe(self.return_series, periods=self.periods),
            sortino=qs.stats.sortino(self.return_series, periods=self.periods),
            tail_ratio=qs.stats.tail_ratio(self.return_series),
            ulcer_index=qs.stats.ulcer_index(self.return_series),
            var=qs.stats.value_at_risk(self.return_series),
            volatility=qs.stats.volatility(self.return_series, periods=self.periods),
            win_loss_ratio=qs.stats.win_loss_ratio(self.return_series),
            win_rate=qs.stats.win_rate(self.return_series),
            annual_return=qs.stats.cagr(self.return_series, periods=self.periods),
            calmar=qs.stats.calmar(self.return_series, periods=self.periods)
        )

        self.performance = rs
        return self.performance

    def summarize(self) -> BotTradeSummary:
        list_times = self.times.tolist()
        series = {
            "actions": SeriesMetric("actions", times=list_times, values=self.actions),
            "prices": SeriesMetric("prices", times=list_times, values=self.prices),
            "returns": SeriesMetric("returns", times=list_times, values=self.returns.tolist()),
            "cumrets": SeriesMetric("cumrets", times=list_times, values=self.cum_rets.tolist()),
            "fees": SeriesMetric("fees", times=list_times, values=self.fees.tolist()),
            "pnls": SeriesMetric("pnls", times=list_times, values=self.pnls.tolist()),
            "trade_sizes": SeriesMetric("trade_sizes", times=list_times, values=self.trade_sizes.tolist()),
            "equities": SeriesMetric("equities", times=list_times, values=self.equities.tolist()),
            "bm_returns": SeriesMetric("bm_returns", times=list_times, values=self.bm_returns.tolist()),
            "bm_pnls": SeriesMetric("bm_pnls", times=list_times, values=self.bm_pnls.tolist()),
            "bm_cumrets": SeriesMetric("bm_cumrets", times=list_times, values=self.bm_cumrets.tolist()),
            "bm_equities": SeriesMetric("bm_equities", times=list_times, values=self.bm_equities.tolist()),
        }
        return BotTradeSummary(
            total_candles=len(self.times),
            bot_id=self.bot_id,
            init_cash=self.init_cash,
            from_time=self.times[0],
            to_time=self.times[-1],
            analysis=self.get_analysis(),
            bt_mode=self.bt_mode,
            performance=self.get_performance(),
            series=series,
            candles=self.times.tolist(),
        )

