from datetime import datetime
import numpy as np

from xno.models import TypeTradeMode
from xno.models.analysis import TradeAnalysis
from xno.models.pf import TradePerformance
from xno.basic_type import DateTimeType, NumericType
from dataclasses import dataclass

__all__ = ["StrategyTradeSummary"]

from xno.models.state_history import StateHistory
from xno.utils.struct import DefaultStruct


@dataclass
class StrategyTradeSummary(DefaultStruct):
    strategy_id: str
    init_cash: NumericType
    from_time: DateTimeType
    to_time: DateTimeType
    analysis: TradeAnalysis
    performance: TradePerformance
    state_history: StateHistory
    bt_mode: TypeTradeMode

    def __repr__(self):
        return (f"StrategyTradeSummary(strategy_id={self.strategy_id}, init_cash={self.init_cash}, "
                f"from_time={self.from_time}, to_time={self.to_time}, "
                f"analysis={self.analysis}, performance={self.performance}), "
                f"state_history(item/length)={len(self.state_history.candles)}/{len(self.state_history.candles)}, bt_mode={self.bt_mode})")

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    st = StrategyTradeSummary(
        strategy_id='random',
        init_cash=100,
        from_time=datetime(2020, 1, 1),
        to_time=datetime(2020, 1, 2),
        analysis=TradeAnalysis(
            start_value=np.nan,
            end_value=np.float64(10),
            total_return=None,
            benchmark_return=None,
            total_fee=None,
            total_trades=None,
            total_closed_trades=None,
            total_open_trades=None,
            open_trade_pnl=None,
            best_trade=None,
            worst_trade=None,
            avg_win_trade=None,
            avg_loss_trade=None,
            avg_win_trade_duration=None,
            avg_loss_trade_duration=None,
        ),
        performance=TradePerformance(
            avg_return=None,
            cumulative_return=None,
            cvar=None,
            gain_to_pain_ratio=None,
            kelly_criterion=None,
            max_drawdown=None,
            omega=None,
            profit_factor=None,
            recovery_factor=None,
            sharpe=None,
            sortino=None,
            tail_ratio=None,
            ulcer_index=None,
            var=None,
            volatility=None,
            win_loss_ratio=None,
            win_rate=None,
            annual_return=None,
            calmar=None,
        ),
        state_history=StateHistory(
            candles=[],
            prices=[],
            actions=[],
            positions=[],
            trade_sizes=[],
            returns=[],
            pnls=[],
            cumrets=[],
            balances=[],
            fees=[],
            bm_returns=[],
            bm_pnls=[],
            bm_cumrets=[],
            bm_balances=[],
        ),
        bt_mode=TypeTradeMode.Simulate
    )
    print(st.to_json())
