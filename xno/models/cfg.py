from typing import Any
from dataclasses import dataclass
import datetime
import pandas as pd
import numpy as np

from xno.basic_type import DateTimeType
from xno.models.tp import (
    TypeSymbolType,
    TypeTradeMode,
    TypeEngine
)
from xno.utils.struct import DefaultStruct

__all__ = ["AdvancedConfig", "BotConfig"]


@dataclass
class AdvancedConfig(DefaultStruct):
    expression: str = ""
    code: str = ""
    info: Any = None
    val_to: datetime.datetime | np.datetime64 | pd.Timestamp | None = None
    train_to: datetime.datetime | np.datetime64 | pd.Timestamp | None = None
    val_from: datetime.datetime | np.datetime64 | pd.Timestamp | None = None
    algorithm: Any = None
    train_from: datetime.datetime | np.datetime64 | pd.Timestamp | None = None
    action_list: Any = None
    alpha_funcs: Any = None
    train_epoch: Any = None
    window_size: Any = None


@dataclass
class BotConfig(DefaultStruct):
    id: str
    symbol: str    # Trading symbol, e.g., "AAPL", "BTC-USD"
    symbol_type: TypeSymbolType
    timeframe: str
    init_cash: float
    run_from: DateTimeType
    run_to: DateTimeType
    mode: TypeTradeMode
    advanced_config: AdvancedConfig
    engine: TypeEngine


if __name__ == "__main__":
    st = BotConfig(
        id="bot_id",
        symbol="APPL",
        symbol_type=TypeSymbolType.UsStock,
        timeframe="timeframe",
        init_cash=1000.0,
        run_from=datetime.datetime.now(),
        run_to=np.datetime64("2024-01-01T00:00:00"),
        mode=TypeTradeMode.Live,
        advanced_config=AdvancedConfig(
            expression="expression",
            code="code",
            info="info",
            val_to=datetime.datetime.now(),
            train_to=pd.Timestamp(datetime.datetime.now()),
        ),
        engine=TypeEngine.XQuant,
    )

    print(st.to_json())

    val = b'{"strategy_id":"strategy_id","symbol":"APPL","market":"S","contract":"S","timeframe":"timeframe","init_cash":1000.0,"run_from":"2025-11-29T01:01:16.915736","run_to":"2024-01-01T00:00:00","mode":3,"advanced_config":{"expression":"expression","code":"code","info":"info","val_to":"2025-11-29T01:01:16.915742","train_to":"2025-11-29 01:01:16.915743","val_from":null,"algorithm":null,"train_from":null,"action_list":null,"alpha_funcs":null,"train_epoch":null,"window_size":null},"engine":"X-Quant"}'

    st2 = BotConfig.from_str(val)
    print(st2.to_json())