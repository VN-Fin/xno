from xno import settings
from xno.backtest import StrategyTradeSummary
from xno.basic_type import NumericType
import numpy as np
import pandas as pd
from xno.backtest.cal.anl import get_trade_analysis_metrics
from xno.backtest.cal.pf import get_performance_metrics
from xno.backtest.cal.rt import get_returns_stock, get_returns_derivative
from typing import Optional, List

from xno.trade import BacktestInput
from xno.trade.tp import AllowedSymbolType

from xno.backtest.pf import TradePerformance
from xno.backtest.analysis import TradeAnalysis


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

        # Calculated metrics
        self.pnl: Optional[np.ndarray] = None
        self.equity_curve: np.ndarray = np.array([]) 
        self.fees: np.ndarray = np.array([]) # not cumsum
        self.returns: np.ndarray = np.array([]) # not cumsum

        # Analysis results
        self.performance: Optional[TradePerformance] = None
        self.trade_analysis: Optional[TradeAnalysis] = None
        self.states_df = pd.DataFrame()

    def calculate_returns(self) -> pd.Series:
        """
        Calculate returns based on symbol type (stock vs derivative).
        
        Returns:
            pandas Series with returns indexed by time
        """
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
        
        # Store calculated metrics from the functions
        self.pnl = results.pnl
        self.fees = results.fees
        self.returns = results.returns
        self.equity_curve = results.equity_curve
        return pd.Series(results.returns, index=pd.to_datetime(self.times))

    def calculate_performance_metrics(self) -> TradePerformance:
        """
        Calculate performance metrics from returns.
        
        Returns:
            TradePerformance object with calculated metrics
        """
        if self.performance is not None:
            return self.performance

        return_series = self.calculate_returns()
        self.performance = get_performance_metrics(return_series)
        return self.performance

    def calculate_trade_analysis(self) -> TradeAnalysis:
        """
        Calculate trade analysis metrics.
        
        Returns:
            TradeAnalysis object with trade statistics
        """
        if self.trade_analysis is not None:
            return self.trade_analysis

        self.calculate_returns()
        self.trade_analysis = get_trade_analysis_metrics(
            equity_curve=self.equity_curve,
            trade_returns=self.returns,
            fees=self.fee_rate,
            trade_sizes=self.trade_sizes,
            pnl=self.pnl,
        )
        return self.trade_analysis

    def summarize(self) -> StrategyTradeSummary:
        """
        Summarize the backtest results including performance and trade analysis.

        Returns:
            Dictionary with summary of performance and trade analysis
        """
        return StrategyTradeSummary(
            init_cash=self.init_cash,
            from_time=self.times[0].__str__(),
            to_time=self.times[-1].__str__(),
            performance=self.calculate_performance_metrics(),
            analysis=self.calculate_trade_analysis(),
            bt_mode=self.bt_mode,
            state_history=None
        )
    
    def visualize(self):
        """
        Visualize the backtest results using StrategyVisualizer.
        """
        from xno.backtest.vs import StrategyVisualizer
        
        visualizer = StrategyVisualizer( name=self.strategy_id)
        visualizer.visualize()
    