import numpy as np
import pandas as pd 
import os

def load_data(city,typo, if_norm='None'):
    file_name = os.path.abspath('..')
    file_name = file_name+'/graph_part/'+city+'_'+typo+'.csv'
    data = pd.read_csv(file_name, header=None, sep=',').values
    if typo == 'speed':
        mean = data.mean()
        std = data.std()
        data = Normalize(data, mean, std, if_norm)
    return data