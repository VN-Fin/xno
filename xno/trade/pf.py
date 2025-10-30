from pydantic import BaseModel, Field


class TradePerformance(BaseModel):
    avg_return: float | None = Field(None, description="Average return per period")
    cumulative_return: float | None = Field(None, description="Cumulative return over the period")
    cvar: float | None = Field(None, description="Conditional Value at Risk")
    gain_to_pain_ratio: float | None = Field(None, description="Gain to Pain Ratio")
    kelly_criterion: float | None = Field(None, description="Kelly Criterion")
    max_drawdown: float | None = Field(None, description="Maximum Drawdown")
    omega: float | None = Field(None, description="Omega Ratio")
    profit_factor: float | None = Field(None, description="Profit Factor")
    recovery_factor: float | None = Field(None, description="Recovery Factor")
    sharpe: float | None = Field(None, description="Sharpe Ratio")
    sortino: float | None = Field(None, description="Sortino Ratio")
    tail_ratio: float | None = Field(None, description="Tail Ratio")
    ulcer_index: float | None = Field(None, description="Ulcer Index")
    var: float | None = Field(None, description="Value at Risk")
    volatility: float | None = Field(None, description="Volatility")
    win_loss_ratio: float | None = Field(None, description="Win/Loss Ratio")
    win_rate: float | None = Field(None, description="Win Rate")
    annual_return: float | None = Field(None, description="Annualized Return")
    calmar: float | None = Field(None, description="Calmar Ratio")

