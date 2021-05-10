'''
cuda: 3
'''

import os
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
from visdom import Visdom
import numpy as np
import pandas as pd
import math
import torch
import time 
import json
import matplotlib.pyplot as plt
import torch.nn as nn
from gen_data_dl import TrainDataset, ValDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from networks import TGCN
from utils import normalized_adj, Normalize
import numpy.linalg as la

def tgcn_loss(y_pred, y_true):
    lambda_loss = 0.0015
    Lreg = 0 # 正则化项
    for para in net.parameters():
        Lreg += torch.sum(para ** 2) / 2
    Lreg = lambda_loss * Lreg

    regress_loss = torch.sum((y_pred-y_true) ** 2) / 2
    return regress_loss + Lreg


def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a,b)
    mape = mean_absolute_percentage_error(a,b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1 - ((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, mape, F_norm, r2, var

os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device('cuda')
viz = Visdom()

# para set
seq_len = 12
pre_len = 12
if_norm = 'sig'
inv_if_norm = 'inv_' + if_norm
src_city = 'bj'
tgt_city = 'sh'
load_batch_size = 32
epochs = 500
gru_units = 32
lr = 0.001
val_num_days = 1
yitaloss=0.8
# environ=str(pre_len*5)+'min_'+'TGCNdirTL_'+'FT:'+str(yitaloss)
environ='TGCNdirTL_'+'FT:'+str(yitaloss)+'_'+str(pre_len*5)+'min'
# load data
train_speed = TrainDataset(city='bj', typo='speed', seq_len=seq_len, pre_len=pre_len, num_days=val_num_days, if_norm=if_norm, if_lack=yitaloss)
train_loader = DataLoader(train_speed, batch_size=load_batch_size, pin_memory=True, num_workers=0, shuffle=True)
val_speed = ValDataset(city='sh', typo='speed', seq_len=seq_len, pre_len=pre_len, num_days=val_num_days, if_norm=if_norm, if_lack=yitaloss)
val_loader = DataLoader(val_speed, batch_size=val_speed.__len__(), pin_memory=True, num_workers=0, shuffle=None)
num_nodes = train_speed.numnodes
train_mean, train_std, train_max, train_min = train_speed.mean, train_speed.std, train_speed.max, train_speed.min
val_mean, val_std, val_max, val_min = val_speed.mean, val_speed.std, val_speed.max, val_speed.min


if __name__ == '__main__':
    torch.manual_seed(3)
    src_adj = pd.read_csv('./src_data/'+src_city+'_adj.csv').values
    src_adj = src_adj[:400, :400]
    src_adj = normalized_adj(src_adj)
    src_adj = src_adj.to(device)
    
    net = TGCN(src_adj.shape[0], 1, gru_units, seq_len, pre_len).to(device=device)
#     net = net.double()
    net.train()
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss_criterion = nn.MSELoss()
    
    training_losses = []
    training_rmses = []
    validation_losses = []
    validation_maes = []
    validation_mapes = []
    validation_pred = []
    validation_acc = []
    validation_r2 = []
    validation_var = []
    validation_rmses = []
    viz.line([0.],[0.],env=environ,win='train_loss',opts=dict(title='train loss'))
    viz.line([0.],[0.],env=environ,win='val_loss',opts=dict(title='val loss'))
    viz.line([0.],[0.],env=environ,win='rmses',opts=dict(title='rmses'))
    viz.line([0.],[0.],env=environ,win='maes',opts=dict(title='maes'))
    viz.line([0.],[0.],env=environ,win='mapes',opts=dict(title='mapes'))
    
    
    for epoch in range(epochs):
        epoch_training_loss = []
        epoch_training_rmses = []
        
        
        for i, (seq_speed, pre_speed) in enumerate(train_loader):
            
            # [32,12,400], [32,3,400]
            optimizer.zero_grad()
            X_batch = seq_speed
            y_batch = pre_speed
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            X_batch = X_batch.permute(1,0,2)
            h0 = torch.zeros(X_batch.size(1),num_nodes,gru_units).to(device=device)
            
            out = net(src_adj, X_batch, h0)
            loss = tgcn_loss(out, y_batch)
            
            loss.backward()
            optimizer.step()
            
            batch_rmse = mean_squared_error(out.detach().cpu().numpy().reshape(-1, num_nodes),
                                            y_batch.detach().cpu().numpy().reshape(-1, num_nodes))
            batch_rmse = math.sqrt(batch_rmse)
            epoch_training_rmses.append(batch_rmse)
            epoch_training_loss.append(loss.detach().cpu().numpy())
            
        loss = sum(epoch_training_loss) / len(epoch_training_loss)
        rmse = sum(epoch_training_rmses) / len(epoch_training_rmses)
        
        training_losses.append(loss)
        rmse = Normalize(rmse, train_mean, train_std, train_max, train_min, inv_if_norm)
        training_rmses.append(rmse)
        viz.line([training_losses[-1]],[epoch],env=environ,win='train_loss', update='append')
        
        with torch.no_grad():
            net.eval().to(device)
            tgt_adj = pd.read_csv('./src_data/'+tgt_city+'_adj.csv').values
            tgt_adj = tgt_adj[:400, :400]
            tgt_adj = normalized_adj(tgt_adj)
            tgt_adj = tgt_adj.to(device)
            for i, (val_seq, val_pre) in enumerate(val_loader):
                val_input = val_seq.to(device=device)
                val_target = val_pre.to(device=device)
                
                val_input = val_input.permute(1,0,2)
                h0 = torch.zeros(val_input.size(1),num_nodes,gru_units).to(device=device)
                
                pred = net(tgt_adj, val_input, h0)
                val_loss = tgcn_loss(pred, val_target)
                
                validation_losses.append(np.asscalar(val_loss.detach().cpu().numpy()))
                
                pred_cpu = pred.detach().cpu().numpy().reshape(-1, num_nodes)
                target_cpu = val_target.detach().cpu().numpy().reshape(-1, num_nodes)
                
                pred_cpu = Normalize(pred_cpu, val_mean, val_std, val_max, val_min, inv_if_norm)
                target_cpu = Normalize(target_cpu, val_mean, val_std, val_max, val_min, inv_if_norm)
                
                rmse, mae, mape, acc, r2_score, var_score = evaluation(target_cpu, pred_cpu)
                
                validation_rmses.append(rmse)
                validation_maes.append(mae)
                validation_mapes.append(mape)
                validation_acc.append(var_score)
                validation_r2.append(r2_score)
                validation_var.append(var_score)
                
                
                viz.line([validation_losses[-1]],[epoch],env=environ,win='val_loss', update='append')
                viz.line([rmse],[epoch],env=environ,win='rmses', update='append')
                viz.line([mae],[epoch],env=environ,win='maes', update='append')
                viz.line([mape],[epoch],env=environ,win='mapes', update='append')
#                 fig, axes = plt.subplots()
#                 axes.plot(np.arange(epoch+1), validation_rmses)
#                 axes.plot(np.arange(epoch+1), validation_maes)
#                 axes.legend(['rmses', 'maes','mapes'])
#                 plt.show()
                
                plt.plot(np.arange(epoch+1), validation_mapes)
            
                validation_label = target_cpu
                
                validation_pred.append(pred_cpu)
                
                out = None
                val_input = val_input
                val_target = val_target
                
        print('epoch: '+str(epoch))
        print("Training loss: "+str(training_losses[-1]))
        print("Training rmse: "+str(training_rmses[-1]))
        print("Validation loss: "+str(validation_losses[-1]))
        print("Validation rmse: "+str(validation_rmses[-1]))
        print("Validation mae: "+str(validation_maes[-1]))
        print("Validation mape: "+str(validation_mapes[-1]))
        print("Validation acc: "+str(validation_acc[-1]))
        if (epoch%1000==0):
            if os.path.exists('./epoch')==False:
                os.makedirs('./epoch')
            torch.save(net, './epoch/model_epoch'+str(epoch)+'.pk1')
    
    index = validation_rmses.index(np.min(validation_rmses))
    test_result = validation_pred[index]
    var = pd.DataFrame(test_result)
    var.to_csv('./epoch/test_epoch'+str(index)+'.csv',index=False, header=False)
    np.savez('sh_L_loss', Train_loss=training_losses, Train_rmse=training_rmses, Val_loss=validation_losses, Val_rmse=validation_rmses, Val_mae=validation_maes, Val_acc=validation_acc)
    print(validation_label.shape)
    print(test_result.shape)
    