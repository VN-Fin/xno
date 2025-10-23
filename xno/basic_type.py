import datetime

import numpy as np
import pandas as pd


# Basic Types
DateTimeType = str | datetime.datetime | pd.Timestamp | np.datetime64
NumericType = np.number | float | int | np.float64 | np.int64
BooleanType = bool | np.bool
