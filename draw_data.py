import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path, n_rows=None):
    data = pd.read_csv(file_path, nrows=n_rows)

    data.columns = ["localtime", 
                    "InstrumentID", 
                    "TradingDay", 
                    "ActionDay", 
                    "UpdateTime",
                    "UpdateMilliseconds", 
                    "LastPrice", 
                    "Volume", 
                    "HighestPrice", 
                    "LowestPrice",
                    "OpenPrice", 
                    "ClosePrice", 
                    "AveragePrice", 
                    "AskPrice1",
                    "AskVolume1", 
                    "BidPrice1", 
                    "BidVolume1", 
                    "UpperLimitPrice",
                    "LowerLimitPrice", 
                    "OpenInterest", 
                    "Turnover", 
                    "PreClosePrice",
                    "PreOpenInterest",
                    "PreSettlementPrice"]

    pure_alkali_data = data[data['InstrumentID'].str.startswith('sa')]
    pure_alkali_data = data[data['InstrumentID'] == "pp2008"]

    plt.figure(figsize=(10,6))
    plt.plot(pd.to_datetime(pure_alkali_data['localtime'], format='%Y%m%d%H%M%S'), pure_alkali_data['LastPrice'])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Price Curve of Pure Alkali Contract')
    plt.savefig("price.png")


if __name__ == "__main__":
    data_path = ""
    load_data(data_path)
