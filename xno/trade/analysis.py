from pydantic import BaseModel, Field


class TradeAnalysis(BaseModel):
    start_value: float | None = Field(None, description="Start Portfolio Value")
    end_value: float | None = Field(None, description="End Portfolio Value")
    total_return: float | None = Field(None, description="Total Return")
    benchmark_return: float | None = Field(None, description="Benchmark Return")
    total_fee: float | None = Field(None, description="Total Fees Paid")
    total_trades: int | None = Field(None, description="Total Number of Trades")
    total_closed_trades: int | None = Field(None, description="Total Number of Closed Trades")
    total_open_trades: int | None = Field(None, description="Total Number of Open Trades")
    open_trade_pnl: float | None = Field(None, description="Open Trade PnL")
    best_trade: float | None = Field(None, description="Best Trade Return")
    worst_trade: float | None = Field(None, description="Worst Trade Return")
    avg_win_trade: float | None = Field(None, description="Average Winning Trade Return")
    avg_loss_trade: float | None = Field(None, description="Average Losing Trade Return")
    avg_win_trade_duration: float | None = Field(None, description="Average Winning Trade Duration")
    avg_loss_trade_duration: float | None = Field(None, description="Average Losing Trade Duration")
