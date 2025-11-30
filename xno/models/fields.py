from dataclasses import dataclass

__all__ = ["FieldInfo"]

from xno.utils.struct import DefaultStruct


@dataclass
class FieldInfo(DefaultStruct):
    field_id: str
    field_name: str
    ticker: str
