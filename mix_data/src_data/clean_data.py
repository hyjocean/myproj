import numpy as np 
import pandas as pd
import os


def Normalize(data, mean, std, type):
    mean = data.mean()
    std = data.std()
    maxnum = data.max()
    minnum = data.min()
    if type == 'z-score':
        data = (data - mean) / std
    elif type == 'inv_z-score':
        data = data*std + mean
    elif type == 'norm':
        data = (data - mean) / (maxnum - minnum)
    elif type == "None":
        data = data
    return np.asarray(data, dtype=np.float64)

def load_data(city,typo, if_norm):
    file_name = os.path.abspath('..')
    file_name = file_name+'/src_data/'+city+'_'+typo+'.csv'
    data = pd.read_csv(file_name, header=None, sep=',').values
    if typo == 'speed':
        mean = data.mean()
        std = data.std()
        data = Normalize(data, mean, std, if_norm)
    return data
# 
# bj_speed = load_data('bj', 'speed', 'None')
# sh_speed = load_data('sh', 'speed', 'None')
# bj_adj = load_data('bj', 'adj', 'None')
# sh_adj = load_data('sh', 'adj', 'None')
# 
def dt_ck(data):
    pro_id = []
    for node in np.arange(0,data.shape[1]):
        theold = 72
        cnt = 0
        for lenth in np.arange(1, data.shape[0]):
            if data[lenth, node] == data[lenth-1, node]:
                cnt+=1
                if cnt >= theold:
                    pro_id.append(node)
                    break
            elif data[lenth, node] != data[lenth-1, node]:
                cnt = 0
        # print(pro_id)
    return pro_id

# 
# sh_pro_id = dt_ck(sh_speed)
# bj_pro_id = dt_ck(bj_speed)