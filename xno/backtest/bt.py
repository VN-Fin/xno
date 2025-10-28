from xno import settings
from xno.basic_type import NumericType
import numpy as np
import pandas as pd
from xno.backtest.cal.anl import get_trade_analysis_metrics
from xno.backtest.cal.pf import get_performance_metrics
from xno.backtest.cal.rt import get_returns_stock, get_returns_derivative
from typing import List, Optional
from xno.trade.state import StrategyState
from xno.trade.tp import AllowedSymbolType

from xno.backtest.pf import TradePerformance
from xno.backtest.analysis import TradeAnalysis


class BacktestCalculator:
    
    def __init__(self, trading_states: List[StrategyState], fee_rate: Optional[NumericType] = None):
        if not trading_states:
            raise ValueError("trading_states cannot be empty")
            
        self.trading_states = trading_states
        self.init_cash = float(trading_states[0].book_size)
        self.symbol_type = trading_states[0].symbol_type
        
        # Set default fee rate based on symbol type
        if fee_rate is None:
            self.fee_rate = settings.trading_fee.percent_stock_fee if self.symbol_type == AllowedSymbolType.Stock else settings.trading_fee.fixed_derivative_fee
        else:
            self.fee_rate = fee_rate
            
        # Extracted data
        self.times: np.ndarray = np.array([])
        self.prices: np.ndarray = np.array([])
        self.positions: np.ndarray = np.array([])
        self.trade_sizes: np.ndarray = np.array([])
        
        # Calculated metrics
        self.pnl: np.ndarray = np.array([]) # not cumsum
        self.equity_curve: np.ndarray = np.array([]) 
        self.fees: np.ndarray = np.array([]) # not cumsum
        self.returns: np.ndarray = np.array([]) # not cumsum
        
        # Analysis results
        self.performance: Optional[TradePerformance] = None
        self.trade_analysis: Optional[TradeAnalysis] = None
        
        # Extract data from trading states
        self._extract_data()
        
    def _extract_data(self):
        """Extract relevant data from trading states."""
        # Use pandas to extract data without any loops
        states_df = pd.DataFrame([s.model_dump() if hasattr(s, "model_dump") else s.__dict__ for s in self.trading_states])
        self.states_df = states_df
        # Extract data using vectorized operations and convert to numpy arrays
        self.times = states_df['candle'].values
        self.prices = states_df['current_price'].values
        self.positions = states_df['current_position'].values
        self.trade_sizes = states_df['trade_size'].values
    
    def calculate_returns(self) -> pd.Series:
        """
        Calculate returns based on symbol type (stock vs derivative).
        
        Returns:
            pandas Series with returns indexed by time
        """
        if self.symbol_type == AllowedSymbolType.Stock:
            returns = get_returns_stock(
                self.init_cash,
                self.times,
                self.prices,
                self.positions,
                self.trade_sizes,
                self.fee_rate
            )
        else:
            returns = get_returns_derivative(
                self.init_cash,
                self.times,
                self.prices,
                self.positions,
                self.trade_sizes,
                self.fee_rate
            )
        
        # Store calculated metrics from the functions
        self.pnl = returns.attrs.get('pnl', np.array([]))
        self.fees = returns.attrs.get('fees', np.array([]))
        self.returns = returns.values
        self.equity_curve = returns.attrs.get('equity_curve', np.array([]))
        
        return returns
    
    def calculate_performance_metrics(self) -> TradePerformance:
        """
        Calculate performance metrics from returns.
        
        Returns:
            TradePerformance object with calculated metrics
        """
        if self.performance is None:
            returns = self.calculate_returns()
            self.performance = get_performance_metrics(returns)
        return self.performance
    
    def calculate_trade_analysis(self) -> TradeAnalysis:
        """
        Calculate trade analysis metrics.
        
        Returns:
            TradeAnalysis object with trade statistics
        """
        if self.trade_analysis is None:
            returns = self.calculate_returns()
            self.trade_analysis = get_trade_analysis_metrics(
                equity_curve=self.equity_curve,
                trade_returns=self.returns,
                fees=self.fees,
                trading_states=self.states_df,
                pnl=self.pnl,
            )
        return self.trade_analysis
    
    def visualize(self, name: str = None):
        """
        Visualize the backtest results using StrategyVisualizer.
        
        Args:
            name: Optional name for the strategy visualization
        """
        from xno.backtest.vs import StrategyVisualizer
        
        visualizer = StrategyVisualizer(backtest_calculator=self, name=name)
        visualizer.visualize()
    