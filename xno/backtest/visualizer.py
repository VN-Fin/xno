from typing import Dict
import logging
import pandas as pd


class StrategyVisualizer:
    def __init__(self, runner, name: str = None):
        """
        Visualize from a StrategyRunner instance (not summary directly).
        Args:
          runner: StrategyRunner (must have bt_summary, or .backtest() method)
          name: Optional name (defaults to runner.strategy_id)
        """
        if not hasattr(runner, 'bt_summary') or runner.bt_summary is None:
            # Trigger backtest summary if missing
            if hasattr(runner, 'backtest'):
                summary = runner.get_backtest()
            else:
                raise ValueError("Runner must have .bt_summary or a .backtest() method!")
        else:
            summary = runner.bt_summary
        # Extract info
        self.state_history = summary.state_history  # Now a dict of lists
        self.performance = getattr(summary, 'performance', None)
        self.analysis = getattr(summary, 'analysis', None)
        self.name = name or getattr(runner, 'strategy_id', None) or runner.__class__.__name__

    def performance_summary(self) -> Dict:
        return {
            'Avg Return': self.performance.avg_return,
            'Cumulative Return': self.performance.cumulative_return,
            'Annual Return': self.performance.annual_return,
            'Sharpe Ratio': self.performance.sharpe,
            'Sortino Ratio': self.performance.sortino,
            'Max Drawdown': self.performance.max_drawdown,
            'Volatility': self.performance.volatility,
            'Win Rate': self.performance.win_rate,
            'Profit Factor': self.performance.profit_factor,
            'Calmar Ratio': self.performance.calmar,
            'CVaR': self.performance.cvar,
            'Var': self.performance.var,
            'Ulcer Index': self.performance.ulcer_index,
            'Tail Ratio': self.performance.tail_ratio,
            'Gain to Pain Ratio': self.performance.gain_to_pain_ratio,
            'Recovery Factor': self.performance.recovery_factor,
            'Omega': self.performance.omega,
            'Kelly Criterion': self.performance.kelly_criterion,
            'Win/Loss Ratio': self.performance.win_loss_ratio,
        }

    def visualize(self):
        if not self.state_history or len(next(iter(self.state_history.values()))) == 0:
            logging.warning("No backtest data available to visualize.")
            return
        df = pd.DataFrame(self.state_history)
        df.set_index('candles', inplace=True)
        df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        performance = self.performance_summary()
        metric_names = list(sorted(performance.keys()))
        metric_values = [f"{performance[m]:.4f}" if isinstance(performance[m], float) else str(performance[m]) for m in metric_names]

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
            x=df.index, y=df['cumrets'], mode='lines', name='Strategy',
            line=dict(color='blue')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['bm_cumrets'], mode='lines', name='Benchmark',
            line=dict(color='gray', dash='dot')
        ), row=1, col=1)

        # === Row 2: Price with Buy/Sell signals ===
        fig.add_trace(go.Scatter(
            x=df.index, y=df['prices'], mode='lines', name='Price',
            line=dict(color='black')
        ), row=2, col=1)

        # Buy markers
        buy_df = df[df['actions'] == 'B'].copy()
        buy_df['date_str'] = buy_df.index.strftime('%Y-%m-%d')
        buy_df['time_str'] = buy_df.index.strftime('%H:%M:%S')
        buy_df['balance_str'] = buy_df['balances'].astype(float).apply(lambda x: f"{x:,.2f}")
        buy_df['amount_str'] = buy_df['trade_sizes'].astype(float).apply(lambda x: f"{x:.2f}")
        buy_df['fee_str'] = buy_df['fees'].astype(float).apply(lambda x: f"{x:.2f}")
        buy_df['price_str'] = buy_df['prices'].astype(float).apply(lambda x: f"{x:.2f}")
        fig.add_trace(go.Scatter(
            x=buy_df.index, y=buy_df['prices'], mode='markers', name='Buy',
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
        sell_df = df[df['actions'] == 'S'].copy()
        sell_df['date_str'] = sell_df.index.strftime('%Y-%m-%d')
        sell_df['time_str'] = sell_df.index.strftime('%H:%M:%S')
        sell_df['balance_str'] = sell_df['balances'].astype(float).apply(lambda x: f"{x:,.2f}")
        sell_df['amount_str'] = sell_df['trade_sizes'].astype(float).apply(lambda x: f"{x:.2f}")
        sell_df['fee_str'] = sell_df['fees'].astype(float).apply(lambda x: f"{x:.2f}")
        sell_df['price_str'] = sell_df['prices'].astype(float).apply(lambda x: f"{x:.2f}")
        fig.add_trace(go.Scatter(
            x=sell_df.index, y=sell_df['prices'], mode='markers', name='Sell',
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
        last_row = df.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_row.name], y=[last_row['cumrets']],
            mode='text', text=[f" -> {last_row['cumrets']:.2f}"],
            textposition='middle right',
            textfont=dict(color='blue', size=12), showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[last_row.name], y=[last_row['bm_cumrets']],
            mode='text', text=[f" -> {last_row['bm_cumrets']:.2f}"],
            textposition='middle right',
            textfont=dict(color='gray', size=12), showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[last_row.name], y=[last_row['prices']],
            mode='text', text=[f" -> {last_row['prices']:.2f}"],
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