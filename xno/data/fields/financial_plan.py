from dataclasses import dataclass

@dataclass(frozen=True)
class _FinancialPlanField:
    TotalRevenue = "TotalRevenue"
    NetRevenue = "NetRevenue"
    TotalProfitBeforeTax = "TotalProfitBeforeTax"
    ProfitAfterTax = "ProfitAfterTax"
    OperatingExpenses = "OperatingExpenses"


class FinancialPlanAnnual(_FinancialPlanField):
    pass

class FinancialPlanQuarterly(_FinancialPlanField):
    pass

__all__ = ['FinancialPlanAnnual', 'FinancialPlanQuarterly']
