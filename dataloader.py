import warnings
import os
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

np.set_printoptions(precision=3, threshold=np.inf)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class FakeStockDataset(Dataset):
    def __init__(self, data_size, sequence_length, feature_size):
        self.data_size = data_size
        self.sequence_length = sequence_length
        self.feature_size = feature_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        stock_data = torch.rand(self.sequence_length,
                                self.feature_size, device="cuda")
        label = torch.randint(0, 2, (4,), device="cuda")
        return stock_data, label


def fake_stock_dataloader(batch_size, data_size, sequence_length, feature_size):
    dataset = FakeStockDataset(data_size, sequence_length, feature_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class CommonDataLoader(Dataset):
    def __init__(self, data_path, label_time, sequence_length, num_samples=None, device="cpu"):
        self.sequence_length = sequence_length
        self.label_time = label_time
        self.num_samples = num_samples
        self.device = device
        self.feature_size = len(self.columns_interest()) - 1
        self.load_data_from_csv(data_path)

    def load_data_from_csv(self, csv_dir):
        self.csv_paths = glob(os.path.join(csv_dir, "**/*.csv"))

    def __len__(self):
        return self.num_samples if self.num_samples else len(self.csv_paths)

    def __getitem__(self, idx):
        csv_file = self.csv_paths[idx]

        data_df = pd.read_csv(csv_file, nrows=None)
        data_df.columns = self.static_columns()

        interval = 302
        n_tick = len(data_df)
        if n_tick < interval:
            # print("Invalid sample")
            return self.empty_fill()
        start = random.randint(0, max(n_tick - interval, 1))
        end = random.randint(min(start + interval, n_tick-2), n_tick - 1)

        seq_data = data_df.loc[start: end, self.columns_interest()]
        try:
            data_arr = seq_data.applymap(pd.to_numeric).values
        except:
            return self.empty_fill()
        data_arr[:, 1] -= data_arr[:, 0]
        data_arr[:, 3] -= data_arr[:, 0]
        data_arr[:, 4] -= data_arr[:, 0]
        data_arr[:, 5] -= data_arr[:, 0]
        data_arr[:, 7] -= data_arr[:, 0]
        data_arr = data_arr[:, 1:]
        # label = [data_arr[time_before - 301, 0] >= data_arr[-300, 0]
        #         for time_before in self.label_time]
        label = list()
        for time_before in self.label_time:
            if data_arr[time_before - 301, 0] > data_arr[-300, 0]:
                label.append(1)
            elif data_arr[time_before - 301, 0] == data_arr[-300, 0]:
                label.append(0)
            else:
                label.append(2)

        # label1 = [data_arr[time_before - 301, 0] for time_before in self.label_time]
        # print(data_arr[-300, 0], label1)
        # print(label)
        data_arr = data_arr[:-300]
        if len(data_arr) < self.sequence_length:
            pad_arr = np.zeros(
                (self.sequence_length - len(data_arr), self.feature_size))
            data_arr = np.concatenate(
                (pad_arr, data_arr), dtype=np.float32)
        else:
            data_arr = data_arr[-self.sequence_length:]

        data = torch.tensor(data_arr, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.long, device=self.device)
        if data.max() > 1e6:
            # print("overflow")
            return self.empty_fill()
        return data, label

    @staticmethod
    def check_numeric(value):
        try:
            arr = pd.to_numeric(value)
            return True
        except (TypeError, ValueError, RuntimeWarning):
            return False

    def empty_fill(self):
        data = torch.zeros(
            (self.sequence_length, self.feature_size), dtype=torch.float32, device=self.device)
        label = torch.zeros((2,), dtype=torch.long, device=self.device)
        return data, label

    @staticmethod
    def format_datatime(data):
        df['datetime'] = pd.to_datetime(
            df['ActionDay'].astype(str) + ' ' + df['UpdateTime'].astype(str))
        df['datetime'] += pd.to_timedelta(df['UpdateMilliseconds'], unit='ms')

        start_time = pd.to_timedelta('09:00:00')
        end_time = pd.to_timedelta('15:00:00')

        df = df.set_index('datetime').between_time(
            start_time, end_time).reset_index()

        df['relative_time'] = (
            df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() * 1000
        return df

    @staticmethod
    def columns_interest():
        return ["OpenPrice", "LastPrice", "Volume",
                "HighestPrice", "LowestPrice", "AskPrice1",
                "AskVolume1", "BidPrice1", "BidVolume1"]

    @staticmethod
    def static_columns():
        return ["localtime",  # 本机写入tick时间
                "InstrumentID",  # 合约
                "TradingDay",  # 交易日
                "ActionDay",  # 业务日期
                "UpdateTime",  # 时间
                "UpdateMilliseconds",  # 时间毫秒
                "LastPrice",  # 最新价
                "Volume",  # 成交量
                "HighestPrice",  # 最高价
                "LowestPrice",  # 最低价
                "OpenPrice",  # 开盘价
                "ClosePrice",  # 收盘价
                "AveragePrice",  # 均价
                "AskPrice1",  # 申卖一价
                "AskVolume1",  # 申卖一量
                "BidPrice1",  # 申买一价
                "BidVolume1",  # 申买一量
                "UpperLimitPrice",  # 涨停板价
                "LowerLimitPrice",  # 跌停板价
                "OpenInterest",  # 持仓量
                "Turnover",  # 成交额
                "PreClosePrice",  # 昨收盘
                "PreOpenInterest",  # 昨持仓
                "PreSettlementPrice",  # 上次结算价
                ]


def common_dataloader(data_path, sequence_length, batch_size, num_samples=None):
    label_time = [60, 300]
    dataset = CommonDataLoader(
        data_path, label_time, sequence_length, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


trian_data = "train_data"
test_data = "test_data"
