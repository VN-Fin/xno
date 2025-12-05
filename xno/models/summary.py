from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from xno.models import TypeTradeMode
from xno.models.analysis import TradeAnalysis
from xno.models.pf import TradePerformance
from xno.basic_type import DateTimeType, NumericType
from dataclasses import dataclass
from xno.utils.struct import DefaultStruct

__all__ = ["BotTradeSummary", "SeriesMetric"]



@dataclass
class SeriesMetric(DefaultStruct):
    name: str
    times: List[float] | np.ndarray
    values: List[float] | np.ndarray | Any

@dataclass
class BotTradeSummary(DefaultStruct):
    bot_id: str
    total_candles: int
    candles: List[float]
    init_cash: NumericType
    from_time: DateTimeType
    to_time: DateTimeType
    analysis: TradeAnalysis
    performance: TradePerformance
    series: Dict[str, SeriesMetric]
    bt_mode: TypeTradeMode

    def __repr__(self):
        return (f"StrategyTradeSummary(strategy_id={self.bot_id}, init_cash={self.init_cash}, "
                f"from_time={self.from_time}, to_time={self.to_time}, "
                f"analysis={self.analysis}, performance={self.performance}), "
                f"total_candles={self.total_candles}/, bt_mode={self.bt_mode}, rolling_metrics={len(self.series)})")

    def __str__(self):
        return self.__repr__()

    def get_dataframe(self):
        df_data = dict()
        df_data['candles'] = self.candles
        for field, value in self.series.items():
            name = field
            df_data[name] = value.values

        df = pd.DataFrame(df_data)
        df.set_index('candles', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

if __name__ == "__main__":
    st = BotTradeSummary(
        bot_id='random',
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
        # state_history=None,
        total_candles=10,
        series={
            "return": SeriesMetric(
                "return",
                times=[1222222],
                values=[0.345224]
            )
        },
        bt_mode=TypeTradeMode.Simulate,
        candles=[1222222]
    )
    print(st.to_json())
