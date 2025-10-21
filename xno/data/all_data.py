


class AllData:
    def __init__(self):
        self.fields = set()

    def add_field(self, field_name: str):
        self.fields.add(field_name)
        return self

    def get(self, resolution: str, symbol: str):
        # GEt OHLCV data for given resolution and symbol

        # GET ......

        # Merge

        raise NotImplementedError("Subclasses must implement this method.")


if __name__ == "__main__":
    all_data = AllData()
