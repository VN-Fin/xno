from dataclasses import dataclass


OhlcvFields = [
    "Open", "High", "Low", "Close", "Volume"
]

@dataclass(frozen=True)
class Ohlcv:
    Open = "Open"
    High = "High"
    Low = "Low"
    Close = "Close"
    Volume = "Volume"
