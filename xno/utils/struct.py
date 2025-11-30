import logging
from dataclasses import fields, MISSING
from enum import Enum
from dataclasses import dataclass
import orjson
import numpy as np
import pandas as pd
import datetime


def parse_field(value, typ):
    if value is None:
        return None
    if isinstance(typ, type) and issubclass(typ, Enum):
        return typ(value)
    if typ in (float, np.float64):
        return float(value)
    if typ in (int, np.int64):
        return int(value)
    if typ in (datetime.datetime, pd.Timestamp, np.datetime64):
        return pd.Timestamp(value)  # converts str, datetime, np.datetime64
    if typ is np.ndarray:
        return np.array(value)
    return value  # fallback


class DefaultStruct:
    def to_json(self, default=str, option=orjson.OPT_SERIALIZE_NUMPY) -> bytes:
        return orjson.dumps(self, option=option, default=default)

    @classmethod
    def from_str(cls, data: str | bytes):
        raw = orjson.loads(data)
        parsed = {}
        try:
            for f in fields(cls):
                parsed[f.name] = parse_field(raw.get(f.name, MISSING), f.type)
        except Exception as e:
            logging.exception(f"Error parsing {data}. Result default None")
            return None
        return cls(**parsed)

    def __eq__(self, other):
        if not isinstance(other, DefaultStruct):
            return False
        return self.to_json(option=orjson.OPT_SORT_KEYS) == other.to_json(option=orjson.OPT_SORT_KEYS)

if __name__ == "__main__":

    class E(Enum):
        A = 1
        B = 2

    @dataclass
    class TestStruct(DefaultStruct):
        field_value: str
        field_value2: str
        field_value3: int | np.int64
        field_value4: datetime.datetime
        enum_field: E

    st1 = TestStruct(
        field_value="test",
        field_value2="test2",
        field_value3=np.int64(100),
        field_value4=datetime.datetime(2020, 1, 1),
        enum_field=E.A
    )
    print(st1.to_json())
    val = b'{"field_value":"test","field_value2":"test2","field_value3":100,"field_value4":"2020-01-01T00:00:00","enum_field":1}'

    st2 = TestStruct.from_str(val)
    print(st2.to_json())

    print(st1 == st2)
