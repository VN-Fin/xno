from dataclasses import dataclass


@dataclass(frozen=True)
class Ohlcv:
    Open = "Open"
    High = "High"
    Low = "Low"
    Close = "Close"
    Volume = "Volume"
