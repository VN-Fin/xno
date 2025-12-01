from dataclasses import dataclass

from xno.utils.struct import DefaultStruct


@dataclass
class StrategyBacktestResult(DefaultStruct):
    train: dict
    test: dict
    simulate: dict
    live: dict
