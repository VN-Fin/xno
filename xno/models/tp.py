from enum import Enum
from typing import List


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

    @classmethod
    def all_engines(cls) -> List[str]:
        return [e.value for e in AllowedEngine]

class AllowedAction(str, Enum):
    Buy = "B"
    Sell = "S"
    Hold = "H"


ActionType = AllowedAction | str
TradeModeType = AllowedTradeMode | int
EngineType = AllowedEngine | str
SymbolType = AllowedSymbolType | str
