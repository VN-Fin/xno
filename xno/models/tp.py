import enum

class TypeTradeMode(enum.StrEnum):
    Train = "train"
    Test = "test"
    Simulate = "simulate"
    Live = "live"

class TypeStage(enum.StrEnum):
    Init = "init"
    Train = "train"
    Test = "test"
    Simulate = "simulate"
    Live = "live"

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


class TypeAction(enum.IntEnum):
    Buy = 1
    Sell = -1
    Hold = 0
