from pydantic import BaseModel


class FieldInfo(BaseModel):
    field_id: str
    field_name: str
    ticker: str
