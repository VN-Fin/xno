"""
The utility functions for stock trading operations.
"""

def round_to_lot(value, lot_size):
    """Round value to the nearest lot size."""
    remainder = value % lot_size
    if remainder < lot_size / 2:
        return int(value - remainder)
    else:
        return int(value + (lot_size - remainder))
