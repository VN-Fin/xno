import pandas as pd
import requests
from datetime import datetime

class DataFetcher:
    def __init__(self, url: str, apikey: str = "demo", type: str = "quarter"):
        self.url = url
        self.apikey = apikey
        self.type = type

    def fetch_data(self, code: str, from_time: datetime = None) -> pd.DataFrame:
        if from_time is None:
            params = {
                "code": code,
                "type": self.type,
            }
            response = requests.get(self.url, params=params)
            print(f"[CALL] {response.url}")  # 👈 In ra URL thật sự được gọi
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    # Nếu có nam, quy, tạo quarter_end_date làm index
                    if 'nam' in df.columns and 'quy' in df.columns:
                        quarter_end_month = {1: 3, 2: 6, 3: 9, 4: 12}
                        df['quarter_end_date'] = df.apply(
                            lambda row: datetime(row['nam'], quarter_end_month[row['quy']],
                                                31 if quarter_end_month[row['quy']] in [3, 12] else 30), axis=1)
                        df.set_index('quarter_end_date', inplace=True)
                    return df
                return pd.DataFrame()
            else:
                response.raise_for_status()
                return pd.DataFrame()
        else:
            to_time = datetime.now()
            quarters = self._get_quarters(from_time, to_time)
            dfs = []
            for year, quarter in quarters:
                params = {
                    "code": code,
                    "type": self.type,
                    "nam": year,
                    "quy": quarter,
                    "apikey": self.apikey
                }
                response = requests.get(self.url, params=params)
                print(f"[CALL] {response.url}")  # 👈 In ra URL thật sự được gọi
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        df = pd.DataFrame(data)
                        quarter_end_month = {1: 3, 2: 6, 3: 9, 4: 12}
                        df['quarter_end_date'] = df.apply(
                            lambda row: datetime(row['nam'], quarter_end_month[row['quy']],
                                                31 if quarter_end_month[row['quy']] in [3, 12] else 30), axis=1)
                        df.set_index('quarter_end_date', inplace=True)
                        dfs.append(df)
                else:
                    response.raise_for_status()
            if dfs:
                return pd.concat(dfs).sort_index()
            return pd.DataFrame()

    def _get_quarters(self, from_date: datetime, to_date: datetime) -> list:
        quarters = []
        current = from_date
        while current <= to_date:
            year = current.year
            quarter = ((current.month - 1) // 3) + 1
            quarters.append((year, quarter))
            current += pd.DateOffset(months=3)
            current = current.replace(day=1)
        return list(set(quarters))

if __name__ == "__main__":
    fetcher = DataFetcher(
        url="https://wifeed.vn/api/tai-chinh-doanh-nghiep/bctc/can-doi-ke-toan",
        apikey="demo",
        type="quarter"
    )
    df = fetcher.fetch_data(code="HSG", from_time=datetime(2022, 1, 1))
    print(df)

    fetcher = DataFetcher(
        url="https://wifeed.vn/api/demo/bctc/can-doi-ke-toan",
        apikey="demo",
        type="quarter"
    )
    df = fetcher.fetch_data(code="HSG")
    print(df)