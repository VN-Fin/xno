import time
import pandas as pd
from ohlcv import OhlcvDataManager
from fetchData import DataFetcher
import fields
from vnstock import Finance


class AllData:
    def __init__(self):
        self.fields = set()

    def add_field(self, field_name: str) -> 'AllData':
        self.fields.add(field_name)
        return self

    def get(self, resolution: str, symbol: str) -> pd.DataFrame:
        OhlcvDataManager.stats()
        ohlcv_df = OhlcvDataManager.get(resolution, symbol, factor=1000)
        if ohlcv_df.empty:
            raise ValueError("No OHLCV data found.")

        ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
        ohlcv_df = ohlcv_df.sort_index(ascending=True)

        # Lấy dữ liệu tài chính
        finance = Finance(symbol=symbol, source='VCI', period="quarter", get_all=True)
        incomStatement_df = finance.income_statement(period='quarter', lang='vi')
        balanSheet_df = finance.balance_sheet(period='quarter', lang='vi')
        cashFlow_df = finance.cash_flow(period='quarter', lang='vi')
        ratio_df = finance.ratio(period='quarter', lang='vi')

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
            elif quarter == 4:
                return pd.Timestamp(f'{year}-12-31')
            else:
                return pd.NaT

        financial_dfs = {
            'income_statement': incomStatement_df,
            'balance_sheet': balanSheet_df,
            'cash_flow': cashFlow_df,
            'ratio': ratio_df
        }

        processed_dfs = {}
        for key, df in financial_dfs.items():
            if 'Năm' in df.columns and 'Kỳ' in df.columns:
                df['time'] = df.apply(lambda row: quarter_to_date(row, 'Năm', 'Kỳ'), axis=1)
                df = df.dropna(subset=['time'])
                df.set_index('time', inplace=True)
                df.drop(columns=['Năm', 'Kỳ', 'CP'], inplace=True, errors='ignore')
                df.columns = [f"{key}_{col}" for col in df.columns]
                processed_dfs[key] = df.sort_index(ascending=True)
            else:
                print(f"Warning: {key} does not have 'Năm' or 'Kỳ' columns. Skipping.")

        if processed_dfs:
            financial_df = pd.concat(processed_dfs.values(), axis=1)
        else:
            financial_df = pd.DataFrame(index=ohlcv_df.index)
            print("Warning: No financial data processed.")

        selected_columns = [col for col in self.fields if col in financial_df.columns]
        selected_financial = financial_df[selected_columns]

        selected_financial = selected_financial.reindex(ohlcv_df.index, method='ffill')

        result_df = ohlcv_df.join(selected_financial, how='left')
        result_df = result_df.infer_objects(copy=False)

        return result_df

if __name__ == "__main__":
    OhlcvDataManager.consume_realtime()
    all_data = AllData() \
        .add_field(fields.IncomeStatement.CHI_PHI_TAI_CHINH) \
        .add_field(fields.BalanceSheet.DAU_TU_DAI_HAN_DONG) \
        .add_field(fields.CashFlow.TANG_GIAM_CAC_KHOAN_PHAI_THU) \
        .add_field(fields.Ratio.P_E) \

    re = all_data.get(resolution="D", symbol="SSI")
    print(re[-1:].to_string(index=True, header=True, justify='left'))
    print(f"Result dtypes:\n{re.dtypes}")