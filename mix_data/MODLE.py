'''
cuda: 0
'''
import os
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import math
from visdom import Visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import numpy.linalg as la
import pandas as pd
from utils import *
from gen_data_nomix import *
from networks import GetFeature, D, TGCN2
from graph_part.metis_1 import graph_part, get_part_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a,b)
    mape = mean_absolute_percentage_error(a,b)
    F_norm = 1-(np.abs((a-b)/a).sum()/a.size)
    r2 = 1 - ((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, mape, F_norm, r2, var

def tgcn_loss(y_pred, y_true):
    lambda_loss = 0.0015
    Lreg = 0 # 正则化项
    for para in net.parameters():
        Lreg += torch.sum(para ** 2) / 2
    Lreg = lambda_loss * Lreg

    regress_loss = torch.sum((y_pred-y_true) ** 2) / 2
    return regress_loss + Lreg

# os.environ['CUDA_VISIBLE_DEVICES']='5'
device = torch.device('cuda:2')
cwd = os.getcwd()
viz = Visdom()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(1)
#===== para set: =====#
epochs = 500
load_batch_size = 64
seq_len = 12
pre_len = 6
gru_units = 32
src_city = 'bj'
tgt_city = 'sh'
sub_graph_num = 16
min_mape = 100
if_norm = 'sig'
inv_if_norm = 'inv_'+if_norm
yitaloss = 0
# environ='FT:'+str(yitaloss)+'_'+str(pre_len*5)+'min_'+'MODLE'
environ='MODLE_'+'FT:'+str(yitaloss)+'_'+str(pre_len*5)+'min_'+'numcluster:'+str(sub_graph_num)
# environ = 'test1'
#===== data load: =====# 
src_speed = CityDataset(city=src_city, typo='speed', seq_len=seq_len, pre_len=pre_len, num_days=1, if_norm=if_norm, if_lack = yitaloss)
src_loader = DataLoader(src_speed, batch_size=load_batch_size, pin_memory=True, num_workers=0, shuffle=True)
tgt_speed = CityDataset(city=tgt_city, typo='speed', seq_len=seq_len, pre_len=pre_len, num_days=1, if_norm=if_norm, if_lack = yitaloss)
tgt_loader = DataLoader(tgt_speed, batch_size=load_batch_size, pin_memory=True, num_workers=0, shuffle=True)
src_num_nodes, tgt_num_nodes = src_speed.data.shape[2], tgt_speed.data.shape[2]
src_adj = pd.read_csv('./src_data/'+src_city+'_adj.csv', header=None).values
tgt_adj = pd.read_csv('./src_data/'+tgt_city+'_adj.csv', header=None).values
tgt_adj = tgt_adj[:400, :400]
tgt_nodes_part, tgt_min_num = graph_part('sh',tgt_adj, sub_graph_num)
src_graph_num = src_num_nodes//tgt_min_num
src_nodes_part, src_min_num = graph_part('bj',src_adj, src_graph_num)
while src_min_num < tgt_min_num:
    src_graph_num -= 1
    src_nodes_part, src_min_num = graph_part('bj',src_adj, src_graph_num)
#===== model init: =====#
net = TGCN2(tgt_min_num, 1, gru_units, seq_len, pre_len).to(device=device)
net.train()
set_requires_grad(net, requires_grad=True)

if os.path.exists(cwd+'/pkl/'+environ+'.pkl'):
    checkpoint = torch.load(cwd+'/pkl/'+environ+'.pkl')
    # TranFeatureNet.load_state_dict(checkpoint['TF_model'])
    # discriminator.load_state_dict(checkpoint['D'])
    net.load_state_dict(checkpoint['net'])
    min_mape = checkpoint['min_mape']
    print('加载模型成功 min_mape =', min_mape)

#===== optim: =====# 
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

Train_loss = []
Val_loss = []
validation_maes = []
validation_pred = []
validation_mapes = []
validation_acc = []
validation_r2 = []
validation_var = []
validation_rmses = []

viz.line([0.],[0.],env=environ, win='train_loss',opts=dict(title='train loss'))
viz.line([0.],[0.],env=environ, win='val_loss',opts=dict(title='val loss'))
viz.line([0.],[0.],env=environ, win='rmses',opts=dict(title='rmses'))
viz.line([0.],[0.],env=environ, win='maes',opts=dict(title='maes'))
viz.line([0.],[0.],env=environ, win='mapes',opts=dict(title='mapes'))


for epoch in range(epochs):
    torch.manual_seed(3)
#     batches = zip(src_loader)
    total_batch_loss = 0
    n_batches = len(src_loader)
    total_domain_loss = total_label_accuracy = 0
    total_similar_loss = 0
    tgcn_tlt_loss = []
    print('==========EPOCH %03d==========='%(epoch))
    # print('==========EPOCH {0:0>3}==========='.format(epoch))

##########################
# mix_data混合数据，输出tgt_data_gen。
##########################
    for src_data, src_pre in tqdm(src_loader, leave=False, total=n_batches):
        # ===== 数据转移到GPU
        # src_data = src_data.to(device)
        if src_data.shape[0] != load_batch_size:
            continue
        
    ### Training
        optimizer.zero_grad()
        tgcnloss = 0
        
        train_out_item = torch.zeros(src_data.size(0), pre_len, 0)
        train_y_item = torch.zeros(src_data.size(0), pre_len, 0)
        for i in range(src_graph_num):
            adj, speed, pre = get_part_data(i, src_nodes_part, src_adj, src_data, src_pre)
            adj = normalized_adj(adj)
            adj = adj[:tgt_min_num, :tgt_min_num]
            adj = adj.to(device=device,dtype=torch.float32)

            
            X_batch = speed[:,:,:tgt_min_num].permute(1,0,2)
            if yitaloss > 0:
                item_num = np.random.randint(X_batch.numel(),size = int(X_batch.numel()*yitaloss))
                X_batch = X_batch.flatten()
                X_batch[item_num] = 0
                X_batch = X_batch.reshape(seq_len, load_batch_size, -1)

            y_batch = pre[:,:,:tgt_min_num]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)

            h0 = torch.zeros(X_batch.size(1), tgt_min_num, gru_units, dtype=torch.float32).to(device=device)
            out = net(adj, X_batch, h0)


            train_out_item = torch.cat((train_out_item,out.detach().cpu()),2)
            train_y_item = torch.cat((train_y_item,y_batch.detach().cpu()),2)

            loss = tgcn_loss(out, y_batch)
            tgcnloss = tgcnloss + loss

        tgcnloss1 = tgcnloss / load_batch_size
        tgcnloss.backward()
        optimizer.step()

        tgcn_tlt_loss.append(tgcnloss1.item())

                
        train_out_cpu = train_out_item.detach().cpu().numpy().reshape(-1,train_out_item.size(2))
        train_y_cpu = train_y_item.detach().cpu().numpy().reshape(-1,train_out_item.size(2))

        train_out_cpu = Normalize(train_out_cpu, src_speed.mean, src_speed.std, src_speed.max, src_speed.min, inv_if_norm)
        train_y_cpu = Normalize(train_y_cpu, src_speed.mean, src_speed.std, src_speed.max, src_speed.min, inv_if_norm)

        train_rmse, train_mae, train_mape, train_acc, train_r2_score, train_var_score = evaluation(train_y_cpu,train_out_cpu)
        
        # tqdm.write('TRAIN: rmse={0:.4f}, mae={1:.4f}, acc={2:.4f}, r2={3:.4f}, var={4:.4f}'.format(train_rmse, train_mae, train_acc, train_r2_score, train_var_score))
        
        
        
        # total_batch_loss += batch_loss.item()
        # total_domain_loss += domain_loss.item()
        # total_similar_loss += similar_loss

    Train_loss.append(sum(tgcn_tlt_loss) / n_batches)
    viz.line([Train_loss[-1]],[epoch],env=environ, win='train_loss', update='append')
    # mean_domain_loss.append(total_domain_loss / n_batches)
    # mean_similar_loss.append(total_similar_loss / n_batches)
    # mean_total_batch_loss.append(total_batch_loss / n_batches)
    print('Trainning loss={0:.4f}'.format(Train_loss[-1]))

    with torch.no_grad():
        net.eval()
        tgt_val_speed = CityDataset(city=tgt_city, typo='speed', seq_len=seq_len, pre_len=pre_len, num_days=1, if_norm=if_norm, if_lack=yitaloss)
        tgt_val_loader = DataLoader(tgt_val_speed, batch_size=tgt_val_speed.__len__(), pin_memory=True, num_workers=0, shuffle=True)
        for i ,(speed, pre) in enumerate(tgt_val_loader):
            tgt_dt = speed.to(device)
            tgt_pre = pre.to(device)
        
        out_item = torch.zeros(tgt_dt.size(0), pre_len, 0)
        y_item = torch.zeros(tgt_dt.size(0), pre_len, 0)
        
        for i in range(sub_graph_num):
            adj, speed, pre = get_part_data(i, tgt_nodes_part, tgt_adj, tgt_dt, tgt_pre)
            adj = normalized_adj(adj)
            adj = adj[:tgt_min_num, :tgt_min_num]
            adj = adj.to(device=device,dtype=torch.float32)

            X_batch = speed[:,:,:tgt_min_num].permute(1,0,2)
            y_batch = pre[:,:,:tgt_min_num]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)        

            h0 = torch.zeros(X_batch.size(1), tgt_min_num, gru_units, dtype=torch.float32).to(device=device)
            out = net(adj, X_batch, h0)

            out_item = torch.cat((out_item,out.detach().cpu()),2)
            y_item = torch.cat((y_item,y_batch.detach().cpu()),2)

            loss = tgcn_loss(out, y_batch)
            tgcnloss = tgcnloss + loss
            # optimizer.step()

        tgcnloss = tgcnloss / (tgt_dt.size(0))
        Val_loss.append(tgcnloss)
        out_cpu = out_item.detach().cpu().numpy().reshape(-1,out_item.size(2))
        y_cpu = y_item.detach().cpu().numpy().reshape(-1,out_item.size(2))

        out_cpu = Normalize(out_cpu, tgt_speed.mean, tgt_speed.std, tgt_speed.max, tgt_speed.min, inv_if_norm)
        y_cpu = Normalize(y_cpu, tgt_speed.mean, tgt_speed.std, tgt_speed.max, tgt_speed.min, inv_if_norm)

        rmse, mae, mape, acc, r2_score, var_score = evaluation(y_cpu,out_cpu)


        # validation_pred = []
        validation_maes.append(mae)
        validation_acc.append(acc)
        validation_mapes.append(mape)
        validation_r2.append(r2_score)
        validation_var.append(var_score)
        validation_rmses.append(rmse)
        
        viz.line([Val_loss[-1].cpu().numpy()],[epoch],env=environ, win='val_loss', update='append')
        viz.line([rmse],[epoch],env=environ, win='rmses', update='append')
        viz.line([mae],[epoch],env=environ, win='maes', update='append')
        viz.line([mape],[epoch],env=environ, win='mapes', update='append')
        
        print('VAL: rmse={0:.4f}, mae={1:.4f}, mape={2:.4f}, acc={3:.4f}, r2_score={4:.4f}, var_score={5:.4f}'.format(rmse, mae, mape, acc, r2_score, var_score))
        print('VAL loss={0:.4f}'.format(Val_loss[-1]))


    if not os.path.exists(cwd+'/pkl'):
        os.makedirs(cwd+'/pkl')
    if validation_mapes[-1] < min_mape:
        state = {'net':net.state_dict(), 'min_mape':validation_mapes[-1]}
        torch.save(state, cwd+'/pkl/'+environ+'.pkl')
        min_mape = validation_mapes[-1]

    # tqdm.write(f'EPOCH {epoch:03d}: mean_tgcn_tlt_loss={mean_tgcn_tlt_loss[-1]:.4f}, '
            #    f'domain_loss={mean_domain_loss[-1]:.4f}, similar_loss={mean_similar_loss[-1]:.4f}, '
            #    f'mean_total_batch_loss={mean_total_batch_loss[-1]:.4f}')   

    # tqdm.write(f'EPOCH {epoch:03d}: mean_tgcn_tlt_loss={mean_tgcn_tlt_loss[-1]:.4f}, '
            #    f'rmse={rmse:.4f}, mae={mae:.4f}, acc={acc:.4f}, r2={r2_score:.4f}, var={var_score:.4f}')

np.savez('losses_400',Train_loss=Train_loss, Val_loss=Val_loss, Val_maes=validation_maes, Val_mape=validation_mapes, Val_acc=validation_acc, \
                Val_rmse=validation_rmses, Val_r2=validation_r2, Val_var=validation_var)