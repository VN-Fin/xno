import datetime
import pandas as pd
import numpy as np
from typing import Optional


# Basic Types
DateTimeType = Optional[str | datetime.datetime | np.datetime64 | pd.Timestamp]
NumericType = float | int
BooleanType = bool 
