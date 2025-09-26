from xno import OhlcvDataManager

if __name__ == "__main__":
    OhlcvDataManager.get_instance("HPG", "m").load_data(
        from_time="2023-01-01",
        to_time="2023-10-01",
    )
    data = OhlcvDataManager.get_instance("HPG", "m").get_data()
    print(data)
