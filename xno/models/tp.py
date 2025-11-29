import enum

class TypeTradeMode(enum.IntEnum):
    Train = 0
    Test = 1
    Simulate = 2
    Live = 3


class TypeMarket(enum.StrEnum):
    Default = "D"
    Stock = "S"
    Crypto = "C"
    Forex = "F"
    Index = "I"

class TypeContract(enum.StrEnum):
    Default = "D"
    Spot = "S"
    Future = "F"
    Option = "O"


class TypeEngine(enum.StrEnum):
    TABot = "TA-Bot"
    AIBot = "AI-Bot"
    XQuant = "X-Quant"


class TypeAction(enum.StrEnum):
    Buy = "B"
    Sell = "S"
    Hold = "H"
