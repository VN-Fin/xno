import datetime

from xno.data import OhlcvDataManager


def load_data(symbol, resolution, start, factor=1):
    datas = OhlcvDataManager.get(
        resolution,
        symbol,
        from_time=start,
        to_time=datetime.datetime.now(),
        factor=factor
    )
    return datas
