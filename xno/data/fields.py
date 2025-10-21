import enum


class PV(str, enum.Enum):
    Open = "open"
    High = "high"
    Low = "low"
    Close = "close"
    Volume = "volume"

class BalanceSheet(str, enum.Enum):
    TotalAssets = "TotalAssets"
    TotalLiabilities = "TotalLiabilities"
    ShareholdersEquity = "ShareholdersEquity"

# class Fields(str, enum.Enum):
#     PV = "pv"
#     BalanceSheet = "balance_sheet"
#
