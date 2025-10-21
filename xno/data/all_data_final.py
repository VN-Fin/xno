
import time
import pandas as pd
from ohlcv import OhlcvDataManager
from fetchData import DataFetcher
import fields

API_KEY = "demo"
class AllData:
    def __init__(self):
        self.fields = set()
        # Khởi tạo các DataFetcher cho từng API
        self.fetchers = {
            'general_balance_sheet': DataFetcher(
                url="https://wifeed.vn/api/tai-chinh-doanh-nghiep/bctc/can-doi-ke-toan", apikey=API_KEY,
                type="quarter"),
            'general_income_statement': DataFetcher(
                url="https://wifeed.vn/api/tai-chinh-doanh-nghiep/bctc/ket-qua-kinh-doanh", apikey=API_KEY,
                type="quarter"),
            'general_cash_flow': DataFetcher(url="https://wifeed.vn/api/tai-chinh-doanh-nghiep/bctc/luu-chuyen-tien-te",
                                             apikey=API_KEY, type="quarter"),
            'bank_balance_sheet': DataFetcher(url="https://wifeed.vn/api/demo/bctc/can-doi-ke-toan", apikey=API_KEY,
                                              type="quarter"),
            'bank_income_statement': DataFetcher(url="https://wifeed.vn/api/demo/bctc/ket-qua-kinh-doanh",
                                                 apikey=API_KEY, type="quarter"),
            'bank_cash_flow': DataFetcher(url="https://wifeed.vn/api/demo/bctc/luu-chuyen-tien-te", apikey=API_KEY,
                                          type="quarter"),
            'securities_balance_sheet': DataFetcher(url="https://wifeed.vn/api/demo/bctc/can-doi-ke-toan",
                                                    apikey=API_KEY, type="quarter"),
            'securities_income_statement': DataFetcher(url="https://wifeed.vn/api/demo/bctc/ket-qua-kinh-doanh",
                                                       apikey=API_KEY, type="quarter"),
            'securities_cash_flow': DataFetcher(url="https://wifeed.vn/api/demo/bctc/luu-chuyen-tien-te",
                                                apikey=API_KEY, type="quarter"),
            'insurance_balance_sheet': DataFetcher(url="https://wifeed.vn/api/demo/bctc/can-doi-ke-toan",
                                                   apikey=API_KEY, type="quarter"),
            'insurance_income_statement': DataFetcher(url="https://wifeed.vn/api/demo/bctc/ket-qua-kinh-doanh",
                                                      apikey=API_KEY, type="quarter"),
            'insurance_cash_flow': DataFetcher(url="https://wifeed.vn/api/demo/bctc/luu-chuyen-tien-te", apikey=API_KEY,
                                               type="quarter")
        }

    def add_field(self, field_name: str, suffix: str = "") -> 'AllData':
        self.fields.add(field_name)
        return self

    def get(self, resolution: str, symbol: str) -> pd.DataFrame:
        # Lấy OHLCV từ OhlcvDataManager
        OhlcvDataManager.stats()
        ohlcv_df = OhlcvDataManager.get(resolution, symbol, factor=1000)
        if ohlcv_df.empty:
            raise ValueError("No OHLCV data found.")

        # Xác định from_time từ min index của OHLCV
        from_time = ohlcv_df.index.min().to_pydatetime() if not ohlcv_df.empty else None

        # Fetch dữ liệu từ các API
        financial_dfs = {}
        for key, fetcher in self.fetchers.items():
            if key in ['general_balance_sheet', 'general_income_statement', 'general_cash_flow']:
                financial_dfs[key] = fetcher.fetch_data(code=symbol, from_time=from_time)
            else:
                financial_dfs[key] = fetcher.fetch_data(code=symbol, from_time=None)
        if financial_dfs:
            financial_df = pd.concat(financial_dfs.values(), axis=1)
        else:
            financial_df = pd.DataFrame()

        # Chọn chỉ các fields đã add (từ self.fields, là các value như "taisannganhan")
        selected_columns = [col for col in self.fields if col in financial_df.columns]
        selected_financial = financial_df[selected_columns]

        # Reindex để khớp với OHLCV index và forward fill
        selected_financial = selected_financial.reindex(ohlcv_df.index, method='ffill')

        # Join OHLCV với selected financial
        result_df = ohlcv_df.join(selected_financial, how='left')

        return result_df

if __name__ == "__main__":
    OhlcvDataManager.consume_realtime()
    all_data = AllData() \
        .add_field(fields.BalanceSheetManufacturing.BSM_ShortTermAssets.value) \
        .add_field(fields.BalanceSheetManufacturing.BSM_TotalAssets.value) \

    # Ví dụ gọi get
    while True:
        time.sleep(10)
        df = all_data.get(resolution="D", symbol="SSI")
        # In hàng cuối cùng của DataFrame dưới dạng ngang
        print("Final DataFrame:")
        print(df[-1:].to_string(index=True, header=True, justify='left'))