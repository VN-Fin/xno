from xno.backtest.common import BaseBacktest, safe_divide, compound_returns
from xno.models.bt_result import BacktestResult
import numpy as np


class BacktestVnStocks(BaseBacktest):
    fee_rate = 0.0015  # 0.15

    def __build__(self) -> BacktestResult:
        if not (len(self.times) == len(self.prices) == len(self.positions) == len(self.trade_sizes)):
            raise ValueError("times, prices, positions, and trade_sizes must have the same length.")

        self.prices = np.asarray(self.prices, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.trade_sizes = np.asarray(self.trade_sizes, dtype=np.float64)
        # positions = positions - trade_sizes  # Do not update positions here
        positions_prev = np.roll(self.positions, 1)  # Use previous positions for PnL calculation
        positions_prev[0] = 0  # assume no position before first bar

        self.fees = np.abs(self.trade_sizes) * self.prices * self.fee_rate
        price_diff = np.diff(self.prices, prepend=self.prices[0])
        self.pnls = positions_prev * price_diff - self.fees

        pnl_cum = np.cumsum(self.pnls)
        # fees_cum = np.cumsum(fees)
        self.equities = self.init_cash + pnl_cum

        self.returns = np.zeros_like(self.equities)
        self.returns[1:] = safe_divide(self.equities[1:] - self.equities[:-1], self.equities[:-1])

        self.cum_rets = compound_returns(self.returns)

        # returns = pd.Series(returns, index=pd.to_datetime(times))

        # bm_equity = (init_cash / prices[0]) * prices
        # bm_returns = np.zeros_like(bm_equity)
        # bm_returns[1:] = _safe_divide(bm_equity[1:] - bm_equity[:-1], bm_equity[:-1])
        # bm_cumret = _compound_returns(bm_returns)
        # bm_pnl = bm_equity - init_cash

        bm_shares = self.init_cash / self.prices[0]
        initial_fee = bm_shares * self.prices[0] * self.fee_rate
        bm_equity = bm_shares * self.prices - initial_fee
        bm_returns = np.zeros_like(bm_equity)
        bm_returns[1:] = safe_divide(bm_equity[1:] - bm_equity[:-1], bm_equity[:-1])
        bm_cumret = compound_returns(bm_returns)
        bm_pnl = bm_equity - self.init_cash

        return BacktestResult(
            times=self.times,
            prices=self.prices,
            positions=self.positions,
            trade_sizes=self.trade_sizes,
            returns=self.returns,
            cumret=self.cum_rets,
            pnl=self.pnls,
            fees=self.fees,
            equity_curve=self.equities,
            bm_equities=bm_equity,
            bm_returns=bm_returns,
            bm_cumret=bm_cumret,
            bm_pnl=bm_pnl
        )
