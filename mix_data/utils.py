import torch
import numpy as np
import pandas as pd 
import os
import scipy.sparse as sp
# print(os.getcwd())
# print(os.path.abspath('.'))
# print(os.path.abspath('..'))
# /home/yunjie/projects/myproj/mix_data
# /home/yunjie/projects/myproj/mix_data
# /home/yunjie/projects/myproj



def Normalize(data, mean, std, maxnum, minnum, type):
    if type == 'zscore':
        data = (data - mean) / std
    elif type == 'inv_zscore':
        data = data*std + mean
    elif type == 'minmax':
        data = (data-minnum) / (maxnum-minnum)
    elif type == 'inv_minmax':
        data = (maxnum-minnum)*data+minnum
    elif type == 'meanvalue':
        data = (data - mean) / (maxnum - minnum)
    elif type == 'inv_meanvalue':
        data = data*(maxnum - minnum) + mean
    elif type == 'sig':
        data = (data - mean) / std
        data = 1/(1+np.exp(-data))
    elif type == 'inv_sig':
        data = data*(maxnum - minnum) + mean
    elif type == "None":
        data = data
    return np.asarray(data, dtype=np.float32)

def load_data(city,typo):
    file_name = os.path.abspath('.')
    file_name = file_name+'/src_data/'+city+'_'+typo+'.csv'
    data = pd.read_csv(file_name, header=None, sep=',').values
    return data

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def normalized_adj(adj):
    ###### Degree matrix
    
    adj = adj + np.eye(adj.shape[0])
    rowsum = adj.sum(1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)]=0.  # 将溢出值变为0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt) # 变为度矩阵
    # print(d_mat_inv_sqrt.shape)
    adj = np.matmul(d_mat_inv_sqrt, adj)# D*A
    adj = np.matmul(adj, d_mat_inv_sqrt)# D*A*D
    return torch.tensor(adj, dtype=torch.float32)


def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col))#.transpose()
    L = torch.sparse_coo_tensor(torch.tensor(coords),
                                torch.tensor(mx.data),
                                mx.shape)
    return L

def calculate_laplacian(adj, lambda_max=1):
    adj = normalized_adj_torch(adj + np.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / np.array(input_dim + output_dim))
    initial = -init_range + 2*init_range*np.random.random([input_dim, output_dim])
    initial = torch.tensor(initial, dtype=torch.float32,requires_grad =True)
    return initial

# def ADJ_DTW(adj, )


