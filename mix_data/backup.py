import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import numpy.linalg as la
import pandas as pd
from utils import *
from gen_data import *
from networks import GetFeature, D, TGCN
from graph_part.metis import graph_part, get_part_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a,b)
    mape = mean_absolute_percentage_error(a,b)
    F_norm = 1-mae/a.mean()
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

cwd = os.getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(1)
#===== para set: =====#
epochs = 500
load_batch_size = 32
seq_len = 12
pre_len = 3
gru_units = 64
src_city = 'bj'
tgt_city = 'sh'
dt_similar_lambda = 1
sub_graph_num = 4
min_mape = 100
if_norm = 'zscore'
inv_if_norm = 'inv_zscore'

#===== data load: =====# 
src_speed = CityDataset(city=src_city, typo='speed', seq_len=seq_len, pre_len=pre_len, if_norm=if_norm)
src_loader = DataLoader(src_speed, batch_size=load_batch_size, pin_memory=True, num_workers=0, shuffle=True)
tgt_speed = CityDataset(city=tgt_city, typo='speed', seq_len=seq_len, pre_len=pre_len, if_norm=if_norm)
tgt_loader = DataLoader(tgt_speed, batch_size=load_batch_size, pin_memory=True, num_workers=0, shuffle=True)
src_num_nodes, tgt_num_nodes = src_speed.data.shape[2], tgt_speed.data.shape[2]
src_adj = pd.read_csv('./src_data/'+src_city+'_adj.csv', header=None).values
tgt_adj = pd.read_csv('./src_data/'+tgt_city+'_adj.csv', header=None).values
tgt_nodes_part, tgt_min_num = graph_part(tgt_city, sub_graph_num)
src_nodes_part, src_min_num = graph_part(src_city, src_num_nodes//tgt_min_num)

#===== model init: =====#
TranFeatureNet = GetFeature(tgt_num_nodes, src_num_nodes).to(device)  # 1159, 1335
TranFeatureNet = TranFeatureNet.double()
TranFeatureNet.train()
set_requires_grad(TranFeatureNet, requires_grad=True)

discriminator = D(input_size=src_num_nodes, seq_len=seq_len, gru_units=gru_units).to(device)
discriminator = discriminator.double()
discriminator.train()
set_requires_grad(discriminator, requires_grad=True)

net = TGCN(tgt_min_num, 1, gru_units, seq_len, pre_len).to(device=device)
net = net.double()
net.train()
set_requires_grad(net, requires_grad=True)

if os.path.exists(cwd+'/pkl/NETS_params.pkl'):
    checkpoint = torch.load(cwd+'/pkl/NETS_params.pkl')
    TranFeatureNet.load_state_dict(checkpoint['TF_model'])
    discriminator.load_state_dict(checkpoint['D'])
    net.load_state_dict(checkpoint['net'])
    min_mape = checkpoint['min_mape']
    print('加载模型成功')

#===== optim: =====# 
# optim = torch.optim.Adam(list(TranFeatureNet.parameters())+list(discriminator.parameters()), lr=1e-6)
# optimizer = torch.optim.Adam(list(TranFeatureNet.parameters())+list(discriminator.parameters())+list(net.parameters()))
optimizer = torch.optim.Adam(net.parameters())

# mean_domain_loss = []
# mean_similar_loss = []
# mean_total_batch_loss = []
Train_loss = []
Val_loss = []

validation_maes = []
validation_pred = []
validation_mape = []
validation_acc = []
validation_r2 = []
validation_var = []
validation_rmse = []

for epoch in range(epochs):
    batches = zip(src_loader, src_loader, src_loader, tgt_loader)
    total_batch_loss = 0
    n_batches = min(len(src_loader), len(tgt_loader))
    total_domain_loss = total_label_accuracy = 0
    total_similar_loss = 0
    tgcn_tlt_loss = []
    print('==========EPOCH %03d==========='%(epoch))
    # print('==========EPOCH {0:0>3}==========='.format(epoch))

##########################
# mix_data混合数据，输出tgt_data_gen。
##########################
    for (src_data1,src_pre1), (src_data2,src_pre2),(src_data3,src_pre3), (tgt_data,tgt_pre) in tqdm(batches, leave=False, total=n_batches):
        # ===== 数据转移到GPU
        src_data1, src_data2, src_data3, tgt_data = src_data1.to(device), src_data2.to(device), src_data3.to(device), tgt_data.to(device)
        if tgt_data.shape[0] != load_batch_size:
            continue
        '''
        # ===== 目标域数据转成源域数据大小
        tgt_data2src_data = TranFeatureNet(tgt_data, 'tgt2src') # 目标域维度变源域维度
        # ===== 正负样本结对
        pos_smp, neg_smp = torch.cat((src_data1, src_data2),1), torch.cat((src_data3, tgt_data2src_data),1)
        pos_smp, neg_smp = pos_smp.to(device), neg_smp.to(device)
        domainX = torch.cat((pos_smp, neg_smp),0)               # domainX: [64, 24, 1159]
        domain_lbl = torch.cat((torch.ones(pos_smp.shape[0]), torch.zeros(neg_smp.shape[0]))).to(device)

        # ===== 预测域标签+计算domain_loss
        h0 = torch.ones((domainX.shape[0], 1, gru_units),dtype=torch.float64).to(device)
        domain_preds = discriminator(domainX, h0)
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_lbl.unsqueeze(1))
        domain_loss = torch.abs(domain_loss - 0.5)
        
        # ===== 处理过的目标域数据经过逆变换生成与目标域数据形状一致的目标域生成数据tgt_data_gen
        tgt_data_gen = TranFeatureNet(tgt_data2src_data, 'src2tgt')
        # TODO: tgtdt_loss (36) 与 domain_loss (0.7) 的值相差过大怎么办？ ---> 加lambda，在量级上变得一致
        # TODO: similarloss 需要重新更改。
        # ===== 计算生成的目标域数据与真正的目标与数据差值loss
        similar_loss = torch.abs(tgt_data_gen - tgt_data).sum().item() / tgt_data.numel()
        batch_loss = domain_loss + dt_similar_lambda * similar_loss
'''

        src_data = torch.cat((src_data1, src_data2, src_data3),0)
        src_pre = torch.cat((src_pre1, src_pre2, src_pre3), 0)


    ### Training
        optimizer.zero_grad()
        tgcnloss = 0
                
        train_out_item = torch.zeros(src_data.size(0), pre_len, 0)
        train_y_item = torch.zeros(src_data.size(0), pre_len, 0)
        for i in range(src_num_nodes//tgt_min_num):
            adj, speed, pre = get_part_data(i, src_nodes_part, src_adj, src_data, src_pre)
            adj = normalized_adj(adj)
            adj = adj[:tgt_min_num, :tgt_min_num]
            adj = adj.to(device=device,dtype=torch.float64)

            X_batch = speed[:,:,:tgt_min_num].permute(1,0,2)
            y_batch = pre[:,:,:tgt_min_num]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)

            h0 = torch.zeros(X_batch.size(1), tgt_min_num, gru_units, dtype=torch.float64).to(device=device)
            out = net(adj, X_batch, h0)


            train_out_item = torch.cat((train_out_item,out.detach().cpu()),2)
            train_y_item = torch.cat((train_y_item,y_batch.detach().cpu()),2)

            loss = tgcn_loss(out, y_batch)
            tgcnloss = tgcnloss + loss

        tgcnloss = tgcnloss / (load_batch_size*3)
        tgcn_tlt_loss.append(tgcnloss.item())

                
        train_out_cpu = train_out_item.detach().cpu().numpy().reshape(-1,train_out_item.size(2))
        train_y_cpu = train_y_item.detach().cpu().numpy().reshape(-1,train_out_item.size(2))

        train_out_cpu = Normalize(train_out_cpu, src_speed.mean, src_speed.std, src_speed.max, src_speed.min, inv_if_norm)
        train_y_cpu = Normalize(train_y_cpu, src_speed.mean, src_speed.std, src_speed.max, src_speed.min, inv_if_norm)

        train_rmse, train_mae, train_mape, train_acc, train_r2_score, train_var_score = evaluation(train_y_cpu,train_out_cpu)
        
        # tqdm.write('TRAIN: rmse={0:.4f}, mae={1:.4f}, acc={2:.4f}, r2={3:.4f}, var={4:.4f}'.format(train_rmse, train_mae, train_acc, train_r2_score, train_var_score))
        tgcnloss.backward()
        optimizer.step()
        
        # total_batch_loss += batch_loss.item()
        # total_domain_loss += domain_loss.item()
        # total_similar_loss += similar_loss

    Train_loss.append(sum(tgcn_tlt_loss) / n_batches)
    # mean_domain_loss.append(total_domain_loss / n_batches)
    # mean_similar_loss.append(total_similar_loss / n_batches)
    # mean_total_batch_loss.append(total_batch_loss / n_batches)
    print('Trainning loss={0:.4f}'.format(Train_loss[-1]))

    with torch.no_grad():
        net.eval()
        tgt_val_speed = CityDataset(city=tgt_city, typo='speed', seq_len=seq_len, pre_len=pre_len, if_norm=if_norm)
        tgt_val_loader = DataLoader(tgt_val_speed, batch_size=tgt_speed.__len__(), pin_memory=True, num_workers=0, shuffle=True)
        for i ,(speed, pre) in enumerate(tgt_val_loader):
            tgt_dt = speed.to(device)
            tgt_pre = pre.to(device)
        
        out_item = torch.zeros(tgt_dt.size(0), pre_len, 0)
        y_item = torch.zeros(tgt_dt.size(0), pre_len, 0)
        
        for i in range(sub_graph_num):
            adj, speed, pre = get_part_data(i, tgt_nodes_part, tgt_adj, tgt_dt, tgt_pre)
            adj = normalized_adj(adj)
            adj = adj[:tgt_min_num, :tgt_min_num]
            adj = adj.to(device=device,dtype=torch.float64)

            X_batch = speed[:,:,:tgt_min_num].permute(1,0,2)
            y_batch = pre[:,:,:tgt_min_num]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)        

            h0 = torch.zeros(X_batch.size(1), tgt_min_num, gru_units, dtype=torch.float64).to(device=device)
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
        validation_mape.append(mape)
        validation_r2.append(r2_score)
        validation_var.append(var_score)
        validation_rmse.append(rmse)

        print('VAL: rmse={0:.4f}, mae={1:.4f}, mape={2:.4f}, acc={3:.4f}, r2_score={4:.4f}, var_score={5:.4f}'.format(rmse, mae, mape, acc, r2_score, var_score))
        print('VAL loss={0:.4f}'.format(Val_loss[-1]))


    if not os.path.exists(cwd+'/pkl'):
        os.makedirs(cwd+'/pkl')
    if validation_mape[-1] < min_mape:
        state = {'TF_model':TranFeatureNet.state_dict(),'D':discriminator.state_dict(), 'net':net.state_dict(), 'min_mape':validation_mape[-1]}
        torch.save(state, cwd+'/pkl/NETS_params.pkl')
        min_mape = validation_mape[-1]

    # tqdm.write(f'EPOCH {epoch:03d}: mean_tgcn_tlt_loss={mean_tgcn_tlt_loss[-1]:.4f}, '
            #    f'domain_loss={mean_domain_loss[-1]:.4f}, similar_loss={mean_similar_loss[-1]:.4f}, '
            #    f'mean_total_batch_loss={mean_total_batch_loss[-1]:.4f}')   

    # tqdm.write(f'EPOCH {epoch:03d}: mean_tgcn_tlt_loss={mean_tgcn_tlt_loss[-1]:.4f}, '
            #    f'rmse={rmse:.4f}, mae={mae:.4f}, acc={acc:.4f}, r2={r2_score:.4f}, var={var_score:.4f}')

np.savez('losses',Train_loss=Train_loss, Val_loss=Val_loss, Val_maes=validation_maes, Val_mape=validatin_mape, Val_acc=validation_acc, \
                Val_rmse=validation_rmse, Val_r2=validation_r2, Val_var=validation_var)