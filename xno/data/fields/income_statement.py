from dataclasses import dataclass

@dataclass(frozen=True)
class _IncomeStatementField:
    Revenue = "Revenue"
    CostOfGoodsSold = "CostOfGoodsSold"
    GrossProfit = "GrossProfit"
    OperatingExpenses = "OperatingExpenses"
    OperatingIncome = "OperatingIncome"
    NetIncome = "NetIncome"
    EarningsPerShare = "EarningsPerShare"
    EBITDA = "EBITDA"
    InterestExpense = "InterestExpense"
    TaxExpense = "TaxExpense"
    DepreciationAndAmortization = "DepreciationAndAmortization"

class IncomeStatementAnnual(_IncomeStatementField):
    pass

class IncomeStatementQuarterly(_IncomeStatementField):
    pass

__all__ = ['IncomeStatementAnnual', 'IncomeStatementQuarterly']
