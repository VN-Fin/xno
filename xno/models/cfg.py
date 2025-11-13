from typing import Any
from pydantic import BaseModel

from xno.models.tp import AllowedSymbolType
from xno.models.tp import AllowedTradeMode


class AdvancedConfig(BaseModel):
    expression: str = ""
    code: str = ""
    info: Any = None
    val_to: Any = None
    train_to: Any = None
    val_from: Any = None
    algorithm: Any = None
    train_from: Any = None
    action_list: Any = None
    alpha_funcs: Any = None
    train_epoch: Any = None
    window_size: Any = None


class StrategyConfig(BaseModel):
    strategy_id: str
    symbol: str
    symbol_type: AllowedSymbolType
    timeframe: str
    init_cash: float
    run_from: Any
    run_to: Any
    mode: AllowedTradeMode
    advanced_config: AdvancedConfig
    engine: str

