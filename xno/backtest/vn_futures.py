from xno.backtest.common import BaseBacktest, safe_divide, compound_returns
from xno.models.bt_result import BacktestResult
import numpy as np

from xno.utils.stock import round_to_lot



class BacktestVnFutures(BaseBacktest):
    # Const
    cash_per_contract = 100_000
    price_per_contract = 25_000_000
    fee_rate = 20_000

    def __build__(self) -> BacktestResult:
        if not (len(self.times) == len(self.prices) == len(self.positions) == len(self.trade_sizes)):
            raise ValueError("times, prices, positions, and trade_sizes must have the same length.")

        self.prices = np.asarray(self.prices, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.trade_sizes = np.asarray(self.trade_sizes, dtype=np.float64)

        # positions = positions - trade_sizes
        positions_prev = np.roll(self.positions, 1)  # Use previous positions for PnL calculation
        positions_prev[0] = 0  # assume no position before first bar

        self.fees = np.abs(self.trade_sizes) * self.fee_rate
        price_diff = np.diff(self.prices, prepend=self.prices[0])
        self.pnls = positions_prev * price_diff * self.cash_per_contract - self.fees

        pnl_cum = np.cumsum(self.pnls)
        # fees_cum = np.cumsum(fees)
        self.equities = self.init_cash + pnl_cum

        self.returns = np.zeros_like(self.equities)
        self.returns[1:] = safe_divide(self.equities[1:] - self.equities[:-1], self.equities[:-1])

        self.cum_rets = compound_returns(self.returns)
        # returns = pd.Series(returns, index=pd.to_datetime(times))

        # max_contracts = round_to_lot(init_cash / 25_000_000, 1)
        # bm_pnl = price_diff * max_contracts * 100_000
        # bm_equity = init_cash + np.cumsum(bm_pnl)
        # bm_returns = np.zeros_like(bm_equity)
        # bm_returns[1:] = _safe_divide(bm_equity[1:] - bm_equity[:-1], bm_equity[:-1])
        # bm_cumret = _compound_returns(bm_returns)
        # Benchmark: mua và giữ, trừ phí giao dịch ban đầu

        max_contracts = round_to_lot(self.init_cash / self.price_per_contract, 1)
        initial_fee = max_contracts * self.fee_rate  # Phí mua ban đầu
        self.bm_pnls = price_diff * max_contracts * self.cash_per_contract
        self.bm_equities = self.init_cash + np.cumsum(self.bm_pnls) - initial_fee  # Trừ phí ban đầu
        self.bm_returns = np.zeros_like(self.bm_equities)
        self.bm_returns[1:] = safe_divide(self.bm_equities[1:] - self.bm_equities[:-1], self.bm_equities[:-1])
        self.bm_cumrets = compound_returns(self.bm_returns)

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
            bm_equities=self.bm_equities,
            bm_returns=self.bm_returns,
            bm_cumret=self.bm_cumrets,
            bm_pnl=self.bm_pnls
        )
