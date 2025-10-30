from typing import Dict, List
import logging
import pandas as pd
import numpy as np

from xno.backtest.summary import BacktestState


class StrategyVisualizer:
    def __init__(self, state_history: List[BacktestState], name: str = None):
        """
        Initialize StrategyVisualizer with a BacktestCalculator instance.
        
        Args:
            name: Optional name for the strategy
        """
        self.name = name
        self._bt_df = None
    
    def get_visualization_data(self) -> pd.DataFrame:
        """
        Build dataframe for visualization with all required columns.
        
        Returns:
            DataFrame with columns for backtest visualization
        """
        # Ensure returns are calculated
        returns = self._calculator.calculate_returns()
        
        # Build visualization dataframe from states_df
        df = self._calculator.states_df.copy()
        
        # Get basic data
        equity_curve = self._calculator.equity_curve
        prices = self._calculator.prices
        trade_sizes = self._calculator.trade_sizes
        fees = self._calculator.fees
        init_cash = self._calculator.init_cash
        
        # === Calculate step_ret (periodic return) ===
        # step_ret[t] = (equity[t] - equity[t-1]) / equity[t-1]
        step_ret = np.zeros_like(equity_curve, dtype=np.float64)
        step_ret[1:] = (equity_curve[1:] - equity_curve[:-1]) / equity_curve[:-1]
        df['step_ret'] = step_ret
        
        # === Calculate cum_ret (cumulative return - compound) ===
        # cum_ret[t] = ∏(1 + step_ret[i]) - 1
        cum_ret = np.cumprod(1 + step_ret) - 1
        df['cum_ret'] = cum_ret
        
        # === Calculate bm_equity (benchmark equity) ===
        # bm_equity = (init_cash / init_price) × current_price
        first_price = float(prices[0])
        bm_equity = (init_cash / first_price) * prices
        df['bm_equity'] = bm_equity
        
        # === Calculate bm_step_ret ===
        # bm_step_ret[t] = (bm_equity[t] - bm_equity[t-1]) / bm_equity[t-1]
        bm_step_ret = np.zeros_like(bm_equity, dtype=np.float64)
        bm_step_ret[1:] = (bm_equity[1:] - bm_equity[:-1]) / bm_equity[:-1]
        df['bm_step_ret'] = bm_step_ret
        
        # === Calculate bm_cum_ret (cumulative return - compound) ===
        # bm_cum_ret[t] = ∏(1 + bm_step_ret[i]) - 1
        bm_cum_ret = np.cumprod(1 + bm_step_ret) - 1
        df['bm_cum_ret'] = bm_cum_ret
        
        # Price column
        df['price'] = prices
        
        # Action column - Extract value from AllowedAction enum (B/S/H)
        df['action'] = df['current_action'].apply(lambda x: x.value if hasattr(x, 'value') else str(x) if x is not None else 'H')
        
        # Equity column from equity curve
        df['equity'] = equity_curve
        
        # Amount column from trade_sizes
        df['amount'] = trade_sizes
        
        # Fee column from fees array
        df['fee'] = fees
        
        # Value column: amount × price
        df['value'] = trade_sizes * prices
        
        # Signal column: Use current_weight as signal
        df['signal'] = df['current_weight']
        
        # Set index to candle time
        if 'candle' in df.columns:
            df.set_index('candle', inplace=True)
        return df
        
    def performance_summary(self) -> Dict:
        """
        Get performance metrics summary.
        
        Returns:
            Dictionary with performance metrics
        """
        performance = self._calculator.calculate_performance_metrics()
        
        # Extract metrics from TradePerformance object
        summary = {
            'Avg Return': performance.avg_return,
            'Cumulative Return': performance.cumulative_return,
            'Annual Return': performance.annual_return,
            'Sharpe Ratio': performance.sharpe,
            'Sortino Ratio': performance.sortino,
            'Max Drawdown': performance.max_drawdown,
            'Volatility': performance.volatility,
            'Win Rate': performance.win_rate,
            'Profit Factor': performance.profit_factor,
            'Calmar Ratio': performance.calmar,
            'CVaR': performance.cvar,
            'Var': performance.var,
            'Ulcer Index': performance.ulcer_index,
            'Tail Ratio': performance.tail_ratio,
            'Gain to Pain Ratio': performance.gain_to_pain_ratio,
            'Recovery Factor': performance.recovery_factor,
            'Omega': performance.omega,
            'Kelly Criterion': performance.kelly_criterion,
            'Win/Loss Ratio': performance.win_loss_ratio,
        }
        
        return summary
    
    def visualize(self):
        """
        Visualize the backtest results including:
        - Cumulative returns vs benchmark
        - Price with Buy/Sell signals
        - Metrics summary table
        """
        # Load dataframe for visualization
        if self._bt_df is None:
            self._bt_df = self.get_visualization_data()
        
        if self._bt_df is None or self._bt_df.empty:
            logging.warning("No backtest data available to visualize.")
            return

        df = self._bt_df.sort_index()
        performance = self.performance_summary()

        metric_names = list(sorted(performance.keys()))
        metric_values = [f"{performance[m]:.4f}" if isinstance(performance[m], float) else "-" for m in metric_names]

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.01,
            horizontal_spacing=0.01,
            column_widths=[0.8, 0.2],
            row_heights=[0.4, 0.6],
            subplot_titles=("Strategy vs Benchmark", "", "Price vs Signals", ""),
            specs=[[{"type": "xy"}, {"type": "table"}],
                   [{"type": "xy"}, None]]
        )

        # === Row 1: Strategy vs Benchmark ===
        fig.add_trace(go.Scatter(
            x=df.index, y=df['cum_ret'], mode='lines', name='Strategy',
            line=dict(color='blue')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['bm_cum_ret'], mode='lines', name='Benchmark',
            line=dict(color='gray', dash='dot')
        ), row=1, col=1)

        # === Row 2: Price with Buy/Sell signals ===
        fig.add_trace(go.Scatter(
            x=df.index, y=df['price'], mode='lines', name='Price',
            line=dict(color='black')
        ), row=2, col=1)

        # Buy markers
        buy_df = df[df['action'] == 'B'].copy()
        buy_df['date_str'] = buy_df.index.strftime('%Y-%m-%d')
        buy_df['time_str'] = buy_df.index.strftime('%H:%M:%S')
        # Format to string, 100000000 -> 100,000,000
        buy_df['balance_str'] = buy_df['equity'].apply(lambda x: f"{x:,.2f}")
        buy_df['amount_str'] = buy_df['amount'].apply(lambda x: f"{x:.2f}")
        buy_df['fee_str'] = buy_df['fee'].apply(lambda x: f"{x:.2f}")
        buy_df['price_str'] = buy_df['price'].apply(lambda x: f"{x:.2f}")
        fig.add_trace(go.Scatter(
            x=buy_df.index, y=buy_df['price'], mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', color='green', size=10),
            hovertemplate="Buy [%{customdata[0]}]<br>"
                          "Time: %{customdata[1]}<br>"
                          "Price: %{customdata[2]}<br>"
                          "Amount: %{customdata[3]}<br>"
                          "Fee: %{customdata[4]}<br>"
                          "Balance: %{customdata[5]}"
                          "<extra></extra>",
            customdata=buy_df[['date_str', 'time_str', 'price_str', 'amount_str', 'fee_str', 'balance_str']].values
        ), row=2, col=1)

        # Sell markers
        sell_df = df[df['action'] == 'S'].copy()
        sell_df['date_str'] = sell_df.index.strftime('%Y-%m-%d')
        sell_df['time_str'] = sell_df.index.strftime('%H:%M:%S')
        # Format to string, 100000000 -> 100,000,000
        sell_df['balance_str'] = sell_df['equity'].apply(lambda x: f"{x:,.2f}")
        sell_df['amount_str'] = sell_df['amount'].apply(lambda x: f"{x:.2f}")
        sell_df['fee_str'] = sell_df['fee'].apply(lambda x: f"{x:.2f}")
        sell_df['price_str'] = sell_df['price'].apply(lambda x: f"{x:.2f}")
        fig.add_trace(go.Scatter(
            x=sell_df.index, y=sell_df['price'], mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', color='red', size=10),
            hovertemplate="Sell [%{customdata[0]}]<br>"
                          "Time: %{customdata[1]}<br>"
                          "Price: %{customdata[2]}<br>"
                          "Amount: %{customdata[3]}<br>"
                          "Fee: %{customdata[4]}<br>"
                          "Balance: %{customdata[5]}"
                          "<extra></extra>",
            customdata=sell_df[['date_str', 'time_str', 'price_str', 'amount_str', 'fee_str', 'balance_str']].values
        ), row=2, col=1)

        # Latest annotation
        last_row = df.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_row.index], y=[last_row['cum_ret']],
            mode='text', text=[f" - {last_row['cum_ret']:.2f}"],
            textposition='middle right',
            textfont=dict(color='blue', size=12), showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[last_row.index], y=[last_row['bm_cum_ret']],
            mode='text', text=[f" - {last_row['bm_cum_ret']:.2f}"],
            textposition='middle right',
            textfont=dict(color='gray', size=12), showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[last_row.index], y=[last_row['price']],
            mode='text', text=[f" - {last_row['price']:.2f}"],
            textposition='middle right',
            textfont=dict(color='black', size=12), showlegend=False
        ), row=2, col=1)

        # === Table ===
        fig.add_trace(go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color="lightgray", align="left",
                font=dict(size=12)
            ),
            cells=dict(
                values=[metric_names, metric_values],
                fill_color="white", align="left",
                font=dict(size=12)
            )
        ), row=1, col=2)

        # === Layout ===
        fig.update_layout(
            title=f"Trading Strategy Performance: `{self.name}`",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis2_title="Time",
            yaxis1_title="Cumulative Return",
            yaxis2_title="Price",
            template='plotly_white',
            hovermode='closest'
        )
        # Show the figure
        import IPython
        if IPython.get_ipython():
            fig.show()
        else:
            from IPython.utils import io
            with io.capture_output() as captured:
                fig.show()