import pandas as pd
from xno.data.ohlcv import OhlcvDataManager
from xno.data import Fields
from xno.data.financeData import get_financial_data
from typing import Literal

from xno.utils.dc import timing


class AllData:
    def __init__(self):
        self.fields = set()
        self.ohlcv_fields = set()
        self.finance_fields = set()
        self.always_include_close = True

    def add_field(self, field_name: str) -> 'AllData':
        self.fields.add(field_name)

        ohlcv_values = [f.value for f in Fields.OHLCV]
        if field_name in ohlcv_values:
            self.ohlcv_fields.add(field_name)
        else:
            self.finance_fields.add(field_name)

        return self

    @timing
    def get(self, resolution: str, symbol: str, period: Literal['quarter', 'year'] = 'quarter', from_time=None) -> pd.DataFrame:
        OhlcvDataManager.stats()
        ohlcv_df = OhlcvDataManager.get(resolution, symbol, factor=1000, from_time=from_time)
        if ohlcv_df.empty:
            raise ValueError("No OHLCV data found.")

        ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
        ohlcv_df = ohlcv_df.sort_index(ascending=True)

        ohlcv_df.columns = [col.capitalize() for col in ohlcv_df.columns]

        # Always include Close
        selected_ohlcv_cols = ['Close']

        # Add any additional OHLCV fields requested
        for col in self.ohlcv_fields:
            if col in ohlcv_df.columns and col != 'Close':
                selected_ohlcv_cols.append(col)

        result_df = ohlcv_df[selected_ohlcv_cols].copy()

        # Add financial fields if requested
        if len(self.finance_fields) > 0:
            # Lấy dữ liệu tài chính từ financeData module
            financial_df = get_financial_data(symbol=symbol, period=period)

            if financial_df.empty:
                print("Warning: No financial data available.")
            else:
                selected_financial_cols = [col for col in self.finance_fields if col in financial_df.columns]

                if selected_financial_cols:
                    selected_financial = financial_df[selected_financial_cols]
                    selected_financial = selected_financial.reindex(result_df.index, method='ffill')
                    result_df = result_df.join(selected_financial, how='left')

        result_df = result_df.infer_objects(copy=False)
        
        return result_df

if __name__ == "__main__":
    import logging
    # OhlcvDataManager.consume_realtime()

    # Test with OHLCV + Financial fields
    logging.info("=== Test 1: OHLCV + Financial fields ===")
    all_data = AllData() \
        .add_field(Fields.OHLCV.OPEN) \
        .add_field(Fields.OHLCV.HIGH) \
        .add_field(Fields.OHLCV.LOW) \
        .add_field(Fields.OHLCV.VOLUME) \
        .add_field(Fields.IncomeStatement.CHI_PHI_TAI_CHINH) \
        .add_field(Fields.BalanceSheet.DAU_TU_DAI_HAN_DONG) \
        .add_field(Fields.Ratio.P_E)

    re = all_data.get(resolution="D", symbol="SSI", period="quarter", from_time="2015-01-01")
    print(re.to_string(index=True, header=True, justify='left'))
    print(f"\nColumns: {list(re.columns)}")

    # Test default behavior (no fields added - should return Close only)
    logging.info("\n=== Test 2: Default (Close always included) ===")
    all_data_default = AllData()
    re_default = all_data_default.get(resolution="D", symbol="SSI", period="quarter")
    print(re_default.to_string(index=True, header=True, justify='left'))
    print(f"\nColumns: {list(re_default.columns)}")

    # Test only financial fields (should include Close + financial fields)
    logging.info("\n=== Test 3: Only Financial fields (Close always included) ===")
    all_data_finance = AllData() \
        .add_field(Fields.IncomeStatement.CHI_PHI_TAI_CHINH) \
        .add_field(Fields.Ratio.P_E)
    re_finance = all_data_finance.get(resolution="D", symbol="SSI", period="quarter")
    print(re_finance.to_string(index=True, header=True, justify='left'))
    print(f"\nColumns: {list(re_finance.columns)}")
