from typing import List

from pydantic import BaseModel, Field

from xno.backtest.analysis import TradeAnalysis
from xno.backtest.pf import TradePerformance
from xno.basic_type import DateTimeType, NumericType


class BacktestState(BaseModel):
    candle: DateTimeType = Field(..., description="Candle time")
    target_price: NumericType = Field(..., description="Target price at this state")
    signal: str = Field(..., description="Signal generated at this state, from 'B'/'S'/'H'")
    position: NumericType = Field(..., description="Position at this state")
    amount: NumericType = Field(..., description="Amount at this state")
    ret: NumericType = Field(..., description="Return at this state")
    pnl: NumericType = Field(..., description="Pnl at this state")
    cumret: NumericType = Field(..., description="Cumulative return at this state")
    balance: NumericType = Field(..., description="Balance/Equity at this state")
    fee: NumericType = Field(..., description="Fee incurred at this state")
    bm_ret: NumericType = Field(..., description="Benchmark return at this state")
    bm_pnl: NumericType = Field(..., description="Benchmark PnL at this state")
    bm_cumret: NumericType = Field(..., description="Benchmark Cumulative Return at this state")
    bm_balance: NumericType = Field(..., description="Benchmark Balance/Equity at this state")


class StrategyTradeSummary(BaseModel):
    init_cash: NumericType = Field(..., description="Initial cash for the trade")
    from_time: DateTimeType = Field(..., description="Start time for the trade analysis")
    to_time: DateTimeType = Field(..., description="End time for the trade analysis")
    analysis: TradeAnalysis | dict = Field(..., description="Trade Analysis")
    performance: TradePerformance | dict = Field(..., description="Trade Performance")
    state_history: List[BacktestState] = Field(..., description="History of backtest states")
    bt_mode: str = Field(..., description="Backtest mode used")