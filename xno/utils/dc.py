import time
import functools
import logging

def timing(func):
    """Decorator to measure and print the total execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logging.info(f">>>>> Function '{func.__name__}' executed in {elapsed:.6f} seconds")
        return result
    return wrapper
