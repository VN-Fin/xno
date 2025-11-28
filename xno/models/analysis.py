from dataclasses import dataclass
import orjson

__all__ = ["TradeAnalysis"]

from xno.utils.struct import DefaultStruct


@dataclass
class TradeAnalysis(DefaultStruct):
    start_value: float | None
    end_value: float | None
    total_return: float | None
    benchmark_return: float | None
    total_fee: float | None
    total_trades: int | None
    total_closed_trades: int | None
    total_open_trades: int | None
    open_trade_pnl: float | None
    best_trade: float | None
    worst_trade: float | None
    avg_win_trade: float | None
    avg_loss_trade: float | None
    avg_win_trade_duration: float | None
    avg_loss_trade_duration: float | None


if __name__ == "__main__":
    import numpy as np

    st = TradeAnalysis(
        start_value=np.float64(0.1),
        end_value=np.nan,
        total_return=0.1,
        benchmark_return=0.1,
        total_fee=0.1,
        total_trades=1,
        total_closed_trades=1,
        total_open_trades=1,
        open_trade_pnl=0.1,
        best_trade=0.1,
        worst_trade=0.1,
        avg_win_trade=0.1,
        avg_loss_trade=0.1,
        avg_win_trade_duration=0.1,
        avg_loss_trade_duration=0.1,
    )

    print(st.to_json())
