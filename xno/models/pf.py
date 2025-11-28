from dataclasses import dataclass
import orjson

__all__ = ["TradePerformance"]

from xno.utils.struct import DefaultStruct


@dataclass
class TradePerformance(DefaultStruct):
    avg_return: float | None
    cumulative_return: float | None
    cvar: float | None
    gain_to_pain_ratio: float | None
    kelly_criterion: float | None
    max_drawdown: float | None
    omega: float | None
    profit_factor: float | None
    recovery_factor: float | None
    sharpe: float | None
    sortino: float | None
    tail_ratio: float | None
    ulcer_index: float | None
    var: float | None
    volatility: float | None
    win_loss_ratio: float | None
    win_rate: float | None
    annual_return: float | None
    calmar: float | None


if __name__ == "__main__":
    import numpy as np
    st = TradePerformance(
        avg_return=np.float64(3.0),
        cumulative_return=np.nan,
        cvar=3.0,
        gain_to_pain_ratio=3.0,
        kelly_criterion=3.0,
        max_drawdown=3.0,
        omega=3.0,
        profit_factor=3.0,
        recovery_factor=3.0,
        sharpe=3.0,
        sortino=3.0,
        tail_ratio=3.0,
        ulcer_index=3.0,
        var=3.0,
        volatility=3.0,
        win_loss_ratio=3.0,
        win_rate=3.0,
        annual_return=3.0,
        calmar=3.0,
    )
    print(st.to_json())