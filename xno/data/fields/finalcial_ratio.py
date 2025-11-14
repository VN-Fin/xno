from dataclasses import dataclass


@dataclass(frozen=True)
class _FinancialRatioField:
    TrailingEPS = "TrailingEPS"
    BookValuePerShare = "BookValuePerShare"
    PriceToBookRatio = "PriceToBookRatio"
    PriceToEarningsRatio = "PriceToEarningsRatio"
    PriceToSalesRatio = "PriceToSalesRatio"
    Beta = "Beta"
    DividendYield = "DividendYield"
    ReturnOnAssets = "ReturnOnAssets"
    ReturnOnEquity = "ReturnOnEquity"
    DebtToEquityRatio = "DebtToEquityRatio"
    CurrentRatio = "CurrentRatio"


class FinancialRatioAnnual(_FinancialRatioField):
    pass

class FinancialRatioQuarterly(_FinancialRatioField):
    pass

__all__ = ['FinancialRatioAnnual', 'FinancialRatioQuarterly']