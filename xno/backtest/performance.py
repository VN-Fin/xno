import pandas as pd

from xno.models.pf import TradePerformance
import quantstats as qs

def get_performance_metrics(s_returns: pd.Series) -> TradePerformance:
    rets = s_returns.dropna()

    # Infer frequency
    freq = pd.infer_freq(rets.index)
    is_daily_or_higher = freq in ['D', 'B', 'W', 'M']
    rs = TradePerformance(
        avg_return=qs.stats.avg_return(rets),
        cumulative_return=qs.stats.comp(rets),
        cvar=qs.stats.cvar(rets),
        gain_to_pain_ratio=qs.stats.gain_to_pain_ratio(rets),
        kelly_criterion=qs.stats.kelly_criterion(rets),
        max_drawdown=qs.stats.max_drawdown(rets),
        omega=qs.stats.omega(rets),
        profit_factor=qs.stats.profit_factor(rets),
        recovery_factor=qs.stats.recovery_factor(rets),
        sharpe=qs.stats.sharpe(rets),
        sortino=qs.stats.sortino(rets),
        tail_ratio=qs.stats.tail_ratio(rets),
        ulcer_index=qs.stats.ulcer_index(rets),
        var=qs.stats.value_at_risk(rets),
        volatility=qs.stats.volatility(rets),
        win_loss_ratio=qs.stats.win_loss_ratio(rets),
        win_rate=qs.stats.win_rate(rets),
        annual_return=qs.stats.cagr(rets),
        calmar=qs.stats.calmar(rets),
    )

    # Only compute annualized metrics if daily or higher
    if is_daily_or_higher:
        rs.annual_return = qs.stats.cagr(rets)
        rs.calmar = qs.stats.calmar(rets)

    return rs
