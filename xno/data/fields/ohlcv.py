from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Ohlcv:
    Open = "Open"
    High = "High"
    Low = "Low"
    Close = "Close"
    Volume = "Volume"


# class OhlcvDataModel(BaseModel):
#     Open: str = Field(..., description="Open price")
#     High: str = Field(..., description="High price")
#     Low: str = Field(..., description="Low price")
#     Close: str = Field(..., description="Close price")
#     Volume: str = Field(..., description="Volume")
