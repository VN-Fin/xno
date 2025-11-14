from dataclasses import dataclass

@dataclass(frozen=True)
class _BalanceSheetField:
    ShortTermAssets = "ShortTermAssets"
    LongTermAssets = "LongTermAssets"
    TotalAssets = "TotalAssets"
    CashAndCashEquivalents = "CashAndCashEquivalents"
    ShortTermLiabilities = "ShortTermLiabilities"
    LongTermLiabilities = "LongTermLiabilities"
    TotalLiabilities = "TotalLiabilities"
    ShareholdersEquity = "ShareholdersEquity"
    TotalLiabilitiesAndEquity = "TotalLiabilitiesAndEquity"



class BalanceSheetAnnual(_BalanceSheetField):
    pass


class BalanceSheetQuarterly(_BalanceSheetField):
    pass

__all__ = ['BalanceSheetAnnual', 'BalanceSheetQuarterly']


# field_map = {
#     "ShortTermAssets": "fieldFromAPI",
# }
