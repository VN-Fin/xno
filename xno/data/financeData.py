import logging

import pandas as pd
from vnstock import Finance
from cachetools import TTLCache, cached
from typing import Literal

from xno.utils.dc import timing

# Cache for 10 minutes (600 seconds)
finance_cache = TTLCache(maxsize=128, ttl=600)

@cached(cache=finance_cache)
@timing
def get_financial_data(
    symbol: str,
    period: Literal['quarter', 'year'] = 'quarter'
) -> pd.DataFrame:
    finance = Finance(symbol=symbol, source='VCI', period=period, get_all=True)
    incomStatement_df = finance.income_statement(period=period, lang='vi')
    balanSheet_df = finance.balance_sheet(period=period, lang='vi')
    cashFlow_df = finance.cash_flow(period=period, lang='vi')
    ratio_df = finance.ratio(period=period, lang='vi')

    if ratio_df.columns.nlevels > 1:
        ratio_df.columns = ratio_df.columns.get_level_values(-1)

    def quarter_to_date(row, year_col='Năm', quarter_col='Kỳ'):
        if year_col not in row.index or quarter_col not in row.index:
            return pd.NaT
        year = row[year_col]
        quarter = row[quarter_col]

        if quarter == 1:
            return pd.Timestamp(f'{year}-03-31')
        elif quarter == 2:
            return pd.Timestamp(f'{year}-06-30')
        elif quarter == 3:
            return pd.Timestamp(f'{year}-09-30')
        elif quarter == 4 or quarter == 5:
            return pd.Timestamp(f'{year}-12-31')
        else:
            return pd.NaT

    def year_to_date(row, year_col='Năm'):
        if year_col not in row.index:
            return pd.NaT
        year = row[year_col]
        return pd.Timestamp(f'{year}-12-31')

    financial_dfs = {
        'income_statement': incomStatement_df,
        'balance_sheet': balanSheet_df,
        'cash_flow': cashFlow_df,
        'ratio': ratio_df
    }

    processed_dfs = {}
    for key, df in financial_dfs.items():
        if 'Năm' not in df.columns:
            logging.warning(f"Warning: {key} does not have 'Năm' column. Skipping.")
            continue

        if 'Kỳ' in df.columns:
            df['time'] = df.apply(lambda row: quarter_to_date(row, 'Năm', 'Kỳ'), axis=1)
            df = df.dropna(subset=['time'])
            df.set_index('time', inplace=True)
            df.drop(columns=['Năm', 'Kỳ', 'CP'], inplace=True, errors='ignore')
        else:
            df['time'] = df.apply(lambda row: year_to_date(row, 'Năm'), axis=1)
            df = df.dropna(subset=['time'])
            df.set_index('time', inplace=True)
            df.drop(columns=['Năm', 'CP'], inplace=True, errors='ignore')

        df.columns = [f"{key}_{col}" for col in df.columns]
        processed_dfs[key] = df.sort_index(ascending=True)


    if processed_dfs:
        financial_df = pd.concat(processed_dfs.values(), axis=1)
    else:
        financial_df = pd.DataFrame()
        logging.warning("Warning: No financial data processed.")

    return financial_df
