import torch
import torch.nn as nn
from utils import calculate_laplacian
import math
import torch.nn.functional as F
class GetFeature(nn.Module):
    def __init__(self, tgt_num_nodes, src_num_nodes):
        super(GetFeature, self).__init__()
        # TODO: 提取的网络结构搭建，怎样才是合理的
        self.feature_extractor1 = nn.Sequential(
            nn.Linear(in_features=tgt_num_nodes,out_features=src_num_nodes, bias=True),
        )

        self.feature_extractor2 = nn.Sequential(
            nn.Linear(in_features=src_num_nodes,out_features=tgt_num_nodes, bias=True),
        )

    def forward(self, data, typo):
        '''
        data1:  [batch, seq_len, num_nodes1]
        output: [batch, seq_len, num_nodes2]
        '''
        # out_data1 = self.Linear(data1)
        if typo == 'tgt2src':
            out_data = self.feature_extractor1(data)
        elif typo == 'src2tgt':
            out_data = self.feature_extractor2(data)
        return out_data


class D(nn.Module):
    def __init__(self, input_size, seq_len, gru_units):
        # input_size/num_nodes = 1155
        # gru_units = 64
        # seq_len = 12
        super().__init__()
        self.GRU = nn.GRU(input_size=input_size, hidden_size=gru_units, batch_first=True)
        self.clf = nn.Sequential(
                                nn.Linear(in_features=seq_len*gru_units*2, out_features=516),
                                nn.Sigmoid(),
                                nn.Dropout(),
                                nn.Linear(in_features=516, out_features=1),
                                nn.Sigmoid(),
                                )

    def forward(self, x, state):
        # x: [64, 24, 1159]
        # state: [64, 1, 64] --> [batchsize, numnodes, grunits]
        state = state.permute(1,0,2)
        x,_ = self.GRU(x,state)
        x = x.reshape(x.shape[0], -1)
        # [64, 24*1159]
        lbl = self.clf(x)
        return lbl


class TGCN(nn.Module):

    def __init__(self, num_nodes, num_features, gru_units, seq_len, pre_len):

        super(TGCN, self).__init__()
        self._nodes = num_nodes
        self._units = gru_units
        self._features = num_features # 1,只有1个速度预测？
        self._len = pre_len
        self.grucell1 = nn.GRUCell(input_size=num_features, hidden_size=gru_units, bias=True)
        self.linear = nn.Linear(in_features=gru_units, out_features=pre_len)
        self.BN = nn.BatchNorm1d(pre_len)
        self.DP = nn.Dropout(0.2)

    def forward(self, A_hat, X, state):
        # A_hat = adj+I [num_nodes,num_nodes];
        # X = X_batch[seq_len, batch_size, num_nodes];
        # state = torch.zeros([batch_size, num_nodes, gru_units])
        # print(A_hat.shape, X.shape)
        for i in range(X.size(0)):
            Ax = torch.einsum("ij,jk->ki", A_hat, X[i].permute(1,0))
            # Ax [num_nodes, batch_size]
            state = torch.einsum("ij,jkl->kil", A_hat, state.permute(1,0,2))
            # state [batch_size, num_nodes, gru_units]
            Ax = Ax.reshape(-1, self._features) # 将Ax打平(根据feature判断，分为几个feature类)
            
            state = state.reshape(-1, self._units)
            state = self.grucell1(Ax, state)# [num_nodes*batch_size, gru_units]
            state = state.reshape(-1, self._nodes, self._units) # [batch_size, num_nodes, gru_units]


        output = state.reshape(-1, self._units)
        # output [batch_size*num_nodes, gru_units]
        output = self.linear(output)
        output = self.BN(output)
        # output = self.DP(output)
        # output [batch_size*num_nodes, pre_len]
        output = output.reshape(-1, self._nodes, self._len)
        # output [batch_size, num_nodes, pre_len]
        output = output.permute(0,2,1)
        # output [batch_size, pre_len, num_nodes]
        return output


class GCNBlock(nn.Module):

    def __init__(self, num_nodes, gru_units,output_size):

        super(GCNBlock, self).__init__()
        self._nodes = num_nodes
        self._outputsize = output_size
        self.linear = nn.Linear(in_features=gru_units+1,
                                out_features=output_size,bias=True)



    def forward(self,A_hat, X, state):
        """  
        :param X: Input data of shape (batch_size, num_nodes).
        :param state: state of shape (batch_size, num_nodes, gru_units).
        :param A_hat: Normalized adjacency matrix(num_nodes, num_nodes).
        """
        X = X.unsqueeze(dim=2)
        ## inputs:(batch,num_nodes,1)

        x_s = torch.cat((X, state), axis=2)
        ## x_s:(batch,num_nodes,gru_units+1)

        input_size = x_s.shape[2]
        # input_size == gru_units+1

        x0 = x_s.permute(1, 2, 0)
        ## x0:(num_nodes,input_size,-1)
        ## x0:(num_nodes,gru_units+1,batch)

        x0 = x0.reshape(self._nodes, -1)
        ## x0:(num_nodes,input_size*batch)
        ## x0:(num_nodes,(gru_units+1)*batch)

        x1 = torch.matmul(A_hat, x0)
        ## x1:(num_nodes,input_size*batch)

        x = x1.reshape(self._nodes, input_size, -1)
        ## x:(num_nodes,gru_units+1,batch)

        x = x.permute(2,0,1)
        ## x:(batch,num_nodes,gru_units+1)

        x = x.reshape(-1, input_size)
        ## x:(batch * num_nodes,gru_units+1)

        x = self.linear(x)

        x = x.reshape(-1, self._nodes, self._outputsize)
        x = x.reshape(-1, self._nodes * self._outputsize)
        ## x:(batch, num_nodes * output_size)

        return x


class TGCN2(nn.Module):

    def __init__(self, num_nodes, num_features, gru_units, seq_len,
                 pre_len):

        super(TGCN2, self).__init__()

        self._nodes = num_nodes  # 图的节点数
        self._units = gru_units  # gru个数
        self._features = num_features
        self._len = pre_len
        self.gc1 = GCNBlock(num_nodes=num_nodes,gru_units=gru_units,
                            output_size=2*gru_units)
        self.gc2 = GCNBlock(num_nodes=num_nodes, gru_units=gru_units,
                            output_size=gru_units)


        self.linear = nn.Linear(in_features=gru_units,
                                out_features=pre_len)


    def forward(self, A_hat, X, state):
        """
        :param X: Input data of shape (seq_len, batch_size, num_nodes).
        :param state: state of shape (batch_size, num_nodes, gru_units).
        :param A_hat: Normalized adjacency matrix(num_nodes, num_nodes).
        """

        ##hx = torch.zeros(X.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        for i in range(X.size(0)):
            value = torch.sigmoid(self.gc1(A_hat, X[i], state))
            value = value.chunk(2,dim = 1)
            r = value[0]
            u = value[1]
            # r, u: (batch,num_nodes * gru_units)
            r_state = r * state.reshape(len(state),-1)
            c = torch.tanh(self.gc2(A_hat, X[i], r_state.reshape(-1,self._nodes,self._units)))
            state = u * state.reshape(len(state),-1) + (1 - u) * c
            state = state.reshape(-1,self._nodes,self._units)

        ##state: state of shape (batch_size, num_nodes, gru_units).

        output = state.reshape(-1, self._units)
        ## output:(batch * num_nodes,gru_units)

        output = self.linear(output)

        ## output:(batch * num_nodes,pre_len)
        output = output.reshape(-1, self._nodes, self._len)
        ## output:(batch,num_nodes,pre_len)
        output = output.permute(0,2,1)

        return output


def train_epoch(training_input: "tensor(trainX)", training_target: "tensor(trainY)", batch_size: int):
    # permut为，0--(n-1)打乱的随机序列
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_loss = []
    epoch_training_rmses = []
    # bs为间隔
    for i in range(0, training_input.shape[0], batch_size):
        # training_input.shape[0] [数据长度]
        # print(batch_size)
        ## 反向传播前梯度清零，以免梯度叠加
        optimizer.zero_grad()
        ## 打乱的序号，随机选择数据
        indicies = permutation[i:i+batch_size] #每bs一跳
        X_batch, y_batch = training_input[indicies], training_target[indicies]
        ## 一批次的训练数据X，y
        # print(X_batch.shape) # [batch_size, seq_len, num_nodes]
        # print(y_batch.shape) # [batch_size, pre_len, num_nodes]

        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)

        X_batch = X_batch.permute(1,0,2)
        # X_batch -> [seq_len, batch_size, num_nodes]
        h0 = torch.zeros(X_batch.size(1), num_nodes, gru_units).to(device=device)
        # X_batch.size[1] = batch_size
        # gru_units = 64


        ## 前向传播
        out = net(adj, X_batch, h0) # [batch_size, pre_len, num_nodes]


        ### 计算loss
        loss = tgcn_loss(out, y_batch)


        ## 反向传播
        loss.backward()
        optimizer.step()

        ## 论文模型指标
        batch_rmse = mean_squared_error(out.detach().cpu().numpy().reshape(-1, num_nodes),
                                        y_batch.detach().cpu().numpy().reshape(-1, num_nodes))
        batch_rmse = math.sqrt(batch_rmse)
        epoch_training_rmses.append(batch_rmse)
        epoch_training_loss.append(loss.detach().cpu().numpy())
    
    return sum(epoch_training_loss) / len(epoch_training_loss), \
           sum(epoch_training_rmses) / len(epoch_training_rmses)

# class tgcnCell(nn.RNNCell):
# 
    # def call(self, inputs, **kwwargs):
        # pass
# 
    # def __init__(self, num_units, adj, num_nodes, input_size=None,
                # act=nn.Tanh, reuse=None):
        # super(tgcnCell, self).__init__(_reuse=reuse)
        # self._act = act
        # self._nodes = num_nodes
        # self._units = num_units
        # self._adj = []
        # self._adj.append(calculate_laplacian(adj))
# 
    # @property
    # def state_size(self):
        # return self._nodes * self._units
    # 
    # @property
    # def output_size(self):
        # return self._units
    # 
    # def __call__(self, inputs, state, scope=None):
        # value = nn.Sigmoid(self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
        # r, u = torch.split(value, 2,1)
        # r_state = r*state
        # c = self._act(self._gc(inputs, r_state, self.units, scope=scope))
        # new_h = u * state + (1 - u) * c
        # return new_h, new_h
# 
    # def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        # inputs = inputs.unsqueeze(2)
        # state = torch.reshape(state, (-1, self._nodes, self._units))
        # x_s = torch.cat((inputs, state), dim=2)
        # input_size = x_s.shape[2]
        # x0 = x_s.permute(1,2,0)
        # x0 = torch.reshape(x0, shape=[self._nodes, -1])
# 
        # for m in self._adj:
            # x1 = torch.sparse.mm(m, x0)
        # x = torch.reshape(x1, shape=[self._nodes, input_size, -1])
        # x = x.permute(2,0,1)
        # x = torch.reshape(x, shape=[-1, input_size])
        # weights = torch.empty([input_size, output_size])
        # nn.init.xavier_normal(weights)
        # x = torch.mm(x, weights)
        # x = nn.init.constant(x, bias)
        # biases = torch.tensor(biases, dtype=torch.float32)
        # x = torch.reshape(x, shape=[-1, self._nodes, output_size])
        # x = torch.reshape(x, shape=[-1, self._nodes * output_size])
        # return x




# class Net(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.feature_extractor = nn.Sequential(
            # nn.Conv2d(3, 10, kernel_size=5),
            # nn.MaxPool2d(2),
            # nn.ReLU(),
            # nn.Conv2d(10, 20, kernel_size=5),
            # nn.MaxPool2d(2),
            # nn.Dropout2d(),
        # )
        # 
        # self.classifier = nn.Sequential(
            # nn.Linear(320, 50),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(50, 10),
        # )
# 
    # def forward(self, x):
        # features = self.feature_extractor(x)
        # features = features.view(x.shape[0], -1)
        # logits = self.classifier(features)
        # return logits