from xno.basic_type import NumericType
import numpy as np
import pandas as pd
from xno.backtest.cal.anl import get_trade_analysis_metrics
from xno.backtest.cal.pf import get_performance_metrics
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
            self.fee_rate = 0.0015 if self.symbol_type == AllowedSymbolType.Stock else 20_000
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
            return self.get_returns_stock(
                self.init_cash,
                self.times,
                self.prices,
                self.positions,
                self.trade_sizes,
                self.fee_rate
            )
        else:
            return self.get_returns_derivative(
                self.init_cash,
                self.times,
                self.prices,
                self.positions,
                self.trade_sizes,
                self.fee_rate
            )
    
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
    

    def get_returns_stock(
            self,
            init_cash,
            times,
            prices,
            positions,
            trade_sizes,
            fee_rate=0.0015  # 0.15%
    ) -> pd.Series:

        # --- Validate input lengths ---
        if not (len(times) == len(prices) == len(positions) == len(trade_sizes)):
            raise ValueError("times, prices, positions, and trade_sizes must have the same length.")

        # --- Convert to numpy arrays ---
        prices = np.asarray(prices, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        trade_sizes = np.asarray(trade_sizes, dtype=np.float64)
        positions = positions - trade_sizes

        # --- Compute trading fees (only when trade occurs) ---
        fees = np.abs(trade_sizes) * prices * fee_rate

        # --- Compute daily price changes ---
        price_diff = np.diff(prices, prepend=prices[0])

        # --- Daily PnL: position * price change ---
        pnl = positions * price_diff

        # --- Cumulative values ---
        pnl_cum = np.cumsum(pnl)
        fees_cum = np.cumsum(fees)

        # --- Compute equity: init cash + total pnl - total fees ---
        equity = init_cash + pnl_cum - fees_cum

        # --- Step returns ---
        returns = np.zeros_like(equity)
        returns[1:] = (equity[1:] - equity[:-1]) / equity[:-1]

        # --- Assign to instance variables (not cumsum) ---
        self.pnl = pnl
        self.fees = fees
        self.returns = returns
        self.equity_curve = equity

        return pd.Series(returns, index=pd.Index(times, name="time"), name="return")

    def get_returns_derivative(
            self,
            init_cash,
            times,
            prices,
            positions,
            trade_sizes,
            fee_rate=20_000 
    ) -> pd.Series:
        # --- Validate input lengths ---
        if not (len(times) == len(prices) == len(positions) == len(trade_sizes)):
            raise ValueError("times, prices, positions, and trade_sizes must have the same length.")

        # --- Convert to numpy arrays ---
        prices = np.asarray(prices, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        trade_sizes = np.asarray(trade_sizes, dtype=np.float64)
        positions = positions - trade_sizes

        # --- Compute trading fees (20 000 VND mỗi hợp đồng) ---
        fees = np.abs(trade_sizes) * fee_rate

        # --- Compute daily price changes ---
        price_diff = np.diff(prices, prepend=prices[0])

        # --- Daily PnL: mỗi điểm = 100 000 VND ---
        pnl = positions * price_diff * 100_000

        # --- Cumulative values ---
        pnl_cum = np.cumsum(pnl)
        fees_cum = np.cumsum(fees)

        # --- Compute equity: init cash + total pnl - total fees ---
        equity = init_cash + pnl_cum - fees_cum

        # --- Step returns ---
        returns = np.zeros_like(equity)
        returns[1:] = (equity[1:] - equity[:-1]) / equity[:-1]

        # --- Assign to instance variables (not cumsum) ---
        self.pnl = pnl
        self.fees = fees
        self.returns = returns
        self.equity_curve = equity

        return pd.Series(returns, index=pd.Index(times, name="time"), name="return")



