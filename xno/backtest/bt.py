import pandas as pd

from xno import settings
from xno.models import (
    StrategyTradeSummary,
    TypeTradeMode
)
from xno.basic_type import NumericType
import numpy as np
from xno.backtest.cal.anl import get_trade_analysis_metrics
from xno.backtest.cal.pf import get_performance_metrics
from xno.backtest.cal.rt import get_returns_stock, get_returns_derivative
from typing import Optional

from xno.models import BacktestInput
from xno.models.tp import AllowedSymbolType

from xno.models.pf import TradePerformance
from xno.models.analysis import TradeAnalysis
from typing import Dict, List, Any

class BacktestCalculator:
    def __init__(self, inp: BacktestInput):
        self.strategy_id = inp.strategy_id
        self.init_cash = inp.book_size
        self.symbol_type = inp.symbol_type
        # Determine fee rate based on symbol type
        self.fee_rate: NumericType = 0.0
        if self.symbol_type == AllowedSymbolType.Stock:
            self.fee_rate = settings.trading_fee.percent_stock_fee
        elif self.symbol_type == AllowedSymbolType.Derivative:
            self.fee_rate = settings.trading_fee.fixed_derivative_fee
        else:
            raise ValueError(f"Unknown symbol type: {self.symbol_type}, do not support backtest calculator")

        # Extracted data
        self.times: np.ndarray = inp.times
        self.prices: np.ndarray = inp.prices
        self.positions: np.ndarray = inp.positions
        self.trade_sizes: np.ndarray = inp.trade_sizes
        self.actions: List[str] = inp.actions
        self.bt_mode = inp.bt_mode

        self.returns: Optional[np.ndarray] = None
        self.equity_curve: Optional[np.ndarray] = None
        self.pnl: Optional[np.ndarray] = None

        self.trade_analysis: Optional[TradeAnalysis] = None
        self.performance: Optional[TradePerformance] = None
        self.state_history: Optional[Dict[str, List[Any]]] = None

    def calculate_returns(self) -> None:
        
        if self.symbol_type == AllowedSymbolType.Stock:
            results = get_returns_stock(
                self.init_cash,
                self.times,
                self.prices,
                self.positions,
                self.trade_sizes,
                self.fee_rate
            )
        elif self.symbol_type == AllowedSymbolType.Derivative:
            results = get_returns_derivative(
                self.init_cash,
                self.times,
                self.prices,
                self.positions,
                self.trade_sizes,
                self.fee_rate
            )
        else:
            raise RuntimeError(f"Unknown symbol type: {self.symbol_type}, do not support return calculation")
        
        self.returns = results.returns # using for performance calculation
        # using for trade analysis
        self.equity_curve = results.equity_curve
        self.pnl = results.pnl
        self.fees = results.fees
        # Build state_history as dict-of-lists
        # using for visualization
        self.state_history = {
            'candles': self.times,
            'prices': self.prices,
            'actions': self.actions,
            'positions': self.positions,
            'trade_sizes': self.trade_sizes,
            'returns': results.returns,
            'pnls': results.pnl,
            'cumrets': results.cumret,
            'balances': results.equity_curve,
            'fees': results.fees,
            'bm_returns': results.bm_returns,
            'bm_pnls': results.bm_pnl,
            'bm_cumrets': results.bm_cumret,
            'bm_balances': results.bm_equities,
        }

    def calculate_performance_metrics(self) -> TradePerformance:
        if self.returns is None:
            self.calculate_returns()
        if self.performance is not None:
            return self.performance
        self.performance = get_performance_metrics(pd.Series(self.returns, index=pd.to_datetime(self.times)))
        return self.performance

    def calculate_trade_analysis(self) -> TradeAnalysis:
        if self.returns is None:
            self.calculate_returns()
        if self.trade_analysis is not None:
            return self.trade_analysis
        self.trade_analysis = get_trade_analysis_metrics(
            equity_curve=self.equity_curve,
            trade_returns=self.returns,
            fees=self.fees,
            trade_sizes=self.trade_sizes,
            pnl=self.pnl,
        )
        return self.trade_analysis

    def summarize(self) -> StrategyTradeSummary:
        if self.returns is None:
            self.calculate_returns()  # Make sure data is ready
        return StrategyTradeSummary(
            strategy_id=self.strategy_id,
            init_cash=self.init_cash,
            from_time=self.times[0].__str__(),
            to_time=self.times[-1].__str__(),
            analysis=self.calculate_trade_analysis(),
            performance=self.calculate_performance_metrics(),
            state_history=self.state_history,
            bt_mode=self.bt_mode if isinstance(self.bt_mode, AllowedTradeMode) else AllowedTradeMode(self.bt_mode)
        )
    
    