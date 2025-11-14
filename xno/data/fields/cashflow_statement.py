from dataclasses import dataclass

@dataclass(frozen=True)
class _CashFlowStatementField:
    OperatingCashFlow = "OperatingCashFlow"
    InvestingCashFlow = "InvestingCashFlow"
    FinancingCashFlow = "FinancingCashFlow"
    FreeCashFlow = "FreeCashFlow"
    NetChangeInCash = "NetChangeInCash"
    CashAtBeginningOfPeriod = "CashAtBeginningOfPeriod"
    CashAtEndOfPeriod = "CashAtEndOfPeriod"


class CashFlowStatementAnnual(_CashFlowStatementField):
    pass


class CashFlowStatementQuarterly(_CashFlowStatementField):
    pass

__all__ = ['CashFlowStatementAnnual', 'CashFlowStatementQuarterly']
