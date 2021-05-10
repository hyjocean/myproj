'''
生成数据
'''
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from utils import load_data, Normalize

class CityDataset(data.Dataset):
    def __init__(self, city, typo, seq_len, pre_len, num_days, if_norm, if_lack):
        '''
        city:       city name 'bj or sh'
        typo:       'speed' or 'adj'
        seq_len:    the length for train to predict 'int'
        pre_len:    the length needed to predict 'int'
        '''
        
        data = load_data(city, typo)
        if city == 'sh':
            data = data[-(288*num_days):, 0:400]
        self.mean, self.max, self.min, self.std, self.median = data.mean(), data.max(), data.min(), data.std(), np.median(data)
        data = Normalize(data, self.mean, self.std, self.max, self.min, if_norm)
        city_data = data
        CityX, CityY =[], []
        for i in range(len(city_data) - seq_len - pre_len):
            a = city_data[i: i + seq_len + pre_len]
            CityX.append(a[0:seq_len])
            CityY.append(a[seq_len:seq_len+pre_len])

        self.data, self.pre = np.asarray(CityX, dtype=np.float32), np.asarray(CityY, dtype=np.float32)

        if city == 'sh' and if_lack > 0:
            item_num = np.random.randint(self.data.size,size=int(self.data.size*if_lack))
            self.data = self.data.flatten()
            self.data[item_num] = 0
            self.data = self.data.reshape(-1, seq_len, 400)
        
        
    def __len__(self):
        return len(self.data)

    def max(self):
        return self.max

    def min(self):
        return self.min

    def mean(self):
        return self.mean

    def std(self):
        return self.std

    def median(self):
        return self.median
    def __getitem__(self, idx):
        CityX = self.data[idx]
        CityY = self.pre[idx]
        return CityX, CityY


class MissDataset(data.Dataset):
    def __init__(self, city, typo, seq_len, pre_len, num_days, if_norm, if_lack):
        '''
        city:       city name 'bj or sh'
        typo:       'speed' or 'adj'
        seq_len:    the length for train to predict 'int'
        pre_len:    the length needed to predict 'int'
        '''
        
        data = load_data(city, typo)
        if city == 'sh':
            data = data[-(288*num_days):, 0:400]
        item_num = np.random.randint(data.size,size=int(data.size*if_lack))
        data = data.flatten()
        data[item_num] = 0
        data = data.reshape(-1, 400)

        self.mean, self.max, self.min, self.std, self.median = data.mean(), data.max(), data.min(), data.std(), np.median(data)
        data = Normalize(data, self.mean, self.std, self.max, self.min, if_norm)
        city_data = data
        CityX, CityY =[], []
        for i in range(len(city_data) - seq_len - pre_len):
            a = city_data[i: i + seq_len + pre_len]
            CityX.append(a[0:seq_len])
            CityY.append(a[seq_len:seq_len+pre_len])
        self.data, self.pre = np.asarray(CityX, dtype=np.float32), np.asarray(CityY, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def max(self):
        return self.max

    def min(self):
        return self.min

    def mean(self):
        return self.mean

    def std(self):
        return self.std

    def median(self):
        return self.median
    def __getitem__(self, idx):
        CityX = self.data[idx]
        CityY = self.pre[idx]
        return CityX, CityY


# class TrainDataset(data.Dataset):
    # def __init__(self, city, typo, seq_len, pre_len, rate):
        # '''
        # city:       city name 'bj or sh'
        # seq_len:    the length for train to predict 'int'
        # pre_len:    the length needed to predict 'int'
        # '''
        # data = load_data(city, typo)
        # train_size = int(len(data)*rate)
        # train_data = data[0:train_size]
        # 
        # trainX, trainY =[], []
        # for i in range(len(train_data) - seq_len - pre_len):
            # a = train_data[i: i + seq_len + pre_len]
            # trainX.append(a[0:seq_len])
            # trainY.append(a[seq_len:seq_len+pre_len])
        # self.data, self.pre = np.asarray(trainX, dtype=np.float64), np.asarray(trainY, dtype=np.float64)
# 
    # def __len__(self):
        # return len(self.data)
# 
    # def __getitem__(self, idx):
        # trainX = self.data[idx]
        # trainY = self.pre[idx]
        # return trainX, trainY
# 
# 
# class ValDataset(data.Dataset):
    # def __init__(self, city, typo, seq_len, pre_len, rate):
        # '''
        # city:       city name 'bj or sh'
        # seq_len:    the length for train to predict 'int'
        # pre_len:    the length needed to predict 'int'
        # '''
        # data = load_data(city, typo)
        # train_size = int(len(data)*rate)
        # val_data = data[train_size:len(data)]
# 
        # valX, valY =[], []
        # for i in range(len(val_data) - seq_len - pre_len):
            # a = val_data[i: i + seq_len + pre_len]
            # valX.append(a[0:seq_len])
            # valY.append(a[seq_len:seq_len+pre_len])
        # self.data, self.pre = np.asarray(valX, dtype=np.float64), np.asarray(valY, dtype=np.float64)
# 
    # def __len__(self):
        # return len(self.data)
# 
    # def __getitem__(self, idx):
        # valX = self.data[idx]
        # valY = self.pre[idx]
        # return valX, valY
        # 
# class TestDataset(data.Dataset):
    # def __init__(self, data_pth):
        # pass