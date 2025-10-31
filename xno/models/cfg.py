from typing import Any
from pydantic import BaseModel

from xno.models.tp import AllowedSymbolType
from xno.models.tp import AllowedTradeMode


class AdvancedConfig(BaseModel):
    expression: str = ""
    close_on_end_day: bool = False
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0

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

