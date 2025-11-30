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
    Default = "default"
    Stock = "stock"
    Crypto = "crypto"
    Forex = "forex"
    Index = "index"

class TypeContract(enum.StrEnum):
    Default = "default"
    Spot = "spot"
    Future = "future"
    Option = "option"


class TypeEngine(enum.StrEnum):
    TABot = "TA-Bot"
    AIBot = "AI-Bot"
    XQuant = "X-Quant"
    Default = "Default"


class TypeAction(enum.IntEnum):
    Buy = 1
    Sell = -1
    Hold = 0
