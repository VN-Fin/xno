import datetime

import numpy as np
import pandas as pd


def parse_dt(value):
    """Convert datetime-like strings into datetime, numpy, or pandas objects."""
    if value is None:
        return None
    if isinstance(value, (datetime.datetime, np.datetime64, pd.Timestamp)):
        return value
    try:
        # try numpy datetime64
        return np.datetime64(value)
    except Exception:
        pass
    try:
        # fallback to datetime
        return datetime.datetime.fromisoformat(value)
    except Exception:
        pass
    try:
        # fallback to pandas timestamp
        return pd.Timestamp(value)
    except Exception:
        pass
    return value  # leave untouched
