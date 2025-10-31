from typing import Dict, List, Any

from pydantic import BaseModel, Field

from xno.models.analysis import TradeAnalysis
from xno.models.pf import TradePerformance
from xno.basic_type import DateTimeType, NumericType


class StrategyTradeSummary(BaseModel):
    init_cash: NumericType = Field(..., description="Initial cash for the trade")
    from_time: DateTimeType = Field(..., description="Start time for the trade analysis")
    to_time: DateTimeType = Field(..., description="End time for the trade analysis")
    analysis: TradeAnalysis | dict = Field(..., description="Trade Analysis")
    performance: TradePerformance | dict = Field(..., description="Trade Performance")
    state_history: Dict[str, List[Any]] = Field(..., description="History of backtest states")
    bt_mode: str = Field(..., description="Backtest mode used")

    def __repr__(self):
        return (f"StrategyTradeSummary(init_cash={self.init_cash}, "
                f"from_time={self.from_time}, to_time={self.to_time}, "
                f"analysis={self.analysis}, performance={self.performance}), "
                f"state_history(item/length)={len(self.state_history)}/{len(self.state_history['candle'])}, bt_mode={self.bt_mode})")

    def __str__(self):
        return self.__repr__()