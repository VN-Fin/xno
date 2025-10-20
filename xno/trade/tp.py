from enum import Enum
import datetime

import numpy as np
import pandas as pd


class AllowedTradeMode(int, Enum):
    BackTrade = 0
    PaperTrade = 1
    LiveTrade = 2


class AllowedSymbolType(str, Enum):
    Stock = "S"
    Derivative = "D"
    Crypto = "C"


class AllowedEngine(str, Enum):
    TABot = "TA-Bot"
    AIBot = "AI-Bot"
    XQuant = "X-Quant"

class AllowedAction(str, Enum):
    Buy = "B"
    Sell = "S"
    Hold = "H"


ActionType = AllowedAction | str
TradeModeType = AllowedTradeMode | int
EngineType = AllowedEngine | str
SymbolType = AllowedSymbolType | str
# Basic Types
DateTimeType = str | datetime.datetime | pd.Timestamp | np.datetime64
NumericType = np.number | float | int | np.float64 | np.int64
BooleanType = bool | np.bool