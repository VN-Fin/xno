from pydantic import BaseModel, Field

from xno.backtest.analysis import TradeAnalysis
from xno.backtest.pf import TradePerformance
from xno.basic_type import DateTimeType, NumericType


class StrategyTradeSummary(BaseModel):
    init_cash: NumericType = Field(..., description="Initial cash for the trade")
    from_time: DateTimeType = Field(..., description="Start time for the trade analysis")
    to_time: DateTimeType = Field(..., description="End time for the trade analysis")
    analysis: TradeAnalysis | dict = Field(..., description="Trade Analysis")
    performance: TradePerformance | dict = Field(..., description="Trade Performance")
