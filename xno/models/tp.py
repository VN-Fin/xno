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

class TypeSymbolType(enum.StrEnum):
    Default = "default"
    UsStock = "UsStock"
    VnStock = "VnStock"
    VnFuture = "VnFuture"
    VnIndex = "index"
    CryptoSpot = "CryptoSpot"
    CryptoFuture = "CryptoFuture"
    Forex = "forex"


class TypeEngine(enum.StrEnum):
    TABot = "TA-Bot"
    AIBot = "AI-Bot"
    XQuant = "X-Quant"
    Default = "Default"


class TypeAction(enum.IntEnum):
    Buy = 1
    Sell = -1
    Hold = 0
