import numpy as np 
import pandas as pd 
import pymetis
# from utils import load_data
import sys
sys.path.append('..')
from mix_data.utils import load_data

def graph_part(cityname, city_adj, num_parts):
    '''
    cityname:   'bj' or 'sh'
    num_parts:   分成的图大小
    output:     membership, list, num_parts = max(membership)
    '''

    tbl = []
    # num_parts = ((city_speed.shape[0]/sub_size))
    for i in np.arange(city_adj.shape[0]):
        lst1 = np.where(city_adj[i,:]==1)
        lst2 = np.where(city_adj[:,i]==1)
        lst = np.unique(np.append(lst1,lst2))
        tbl.append(lst)
    _, membership = pymetis.part_graph(num_parts, tbl)
    nodes_part = []
    min_num = len(membership) 
    for i in range(num_parts):
        nodes_part.append(np.argwhere(np.array(membership) == i).ravel())
        if len(nodes_part[-1]) < min_num:
            min_num = len(nodes_part[-1])
    np.save('./graph_part/'+str(cityname)+'_nodespart1', nodes_part)
    return nodes_part, min_num
    # nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel()
    # nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel()
    # nodes_part_2 = np.argwhere(np.array(membership) == 2).ravel()
    # nodes_part_3 = np.argwhere(np.array(membership) == 3).ravel()
    # nodes_part_4 = np.argwhere(np.array(membership) == 4).ravel()


def get_part_data(idx, nodes_part, adj, speed, tgt_pre):
    '''
    idx:    编号
    membership: 划分的图
    adj:    总图的临街矩阵
    speed:  总图的速度矩阵
    '''
    nodes_part = nodes_part[idx]
    adj_part = adj[nodes_part]
    adj_part = adj_part[:,nodes_part]
    speed_part = speed[:, :, nodes_part]
    pre_part = tgt_pre[:, :, nodes_part]
    return adj_part, speed_part, pre_part



