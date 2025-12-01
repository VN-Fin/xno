from dataclasses import dataclass, fields
from typing import List, Any

import pandas as pd
from dotenv.parser import Position

from xno.models import TypeAction
from xno.utils.struct import DefaultStruct


@dataclass
class StateSeries(DefaultStruct):
    name: str
    times: List[str]
    values: List[Any]


@dataclass
class StateHistory(DefaultStruct):
    candles: List[float]
    prices: List[float]
    actions: List[TypeAction]
    positions: List[Position]
    trade_sizes: List[float]
    returns: List[float]
    pnls: List[float]
    cumrets: List[float]
    balances: List[float]
    fees: List[float]
    bm_returns: List[float]
    bm_pnls: List[float]
    bm_cumrets: List[float]
    bm_balances: List[float]

    def get_series(self) -> List[StateSeries]:
        series_list = []
        time_axis = self.candles  # all metrics share this time axis

        for f in fields(self):
            name = f.name
            if name == "candles":
                continue  # skip the time list

            values = getattr(self, name)
            series_list.append(
                StateSeries(
                    name=name,
                    times=time_axis,
                    values=values,
                )
            )

        return series_list

    def get_dataframe(self) -> pd.DataFrame:
        df_data = dict()
        for f in fields(self):
            name = f.name
            values = getattr(self, name)
            df_data[name] = values

        df = pd.DataFrame(df_data)
        df.set_index('candles', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
